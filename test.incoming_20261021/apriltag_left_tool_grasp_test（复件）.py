#!/usr/bin/python3
# 用途：通过 AprilTag 执行左手抓取、松爪与回位流程。
# 运行：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && python3 apriltag_left_tool_grasp_test.py --help
"""AprilTag 左手抓放：放置后上移 50 mm，复位底盘后再回 home。"""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import rclpy
import yaml
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray
from std_srvs.srv import Trigger
from tf2_ros import TransformException

import apriltag_left_tool_accuracy_test as tag_accuracy
import left_tool_accuracy_test as accuracy


GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0
GRIPPER_START_HOLD_SEC = 0.5
GRIPPER_END_HOLD_SEC = 2.0
DEFAULT_RETURN_POSE_FILE = Path(
    '/home/liuqian/copy/test/left_tool_joint_snapshot_20261010.yaml')



def topic_name(namespace, name):
    if name.startswith('/'):
        return name
    prefix = namespace.strip('/')
    return '/' + '/'.join(part for part in (prefix, name) if part)


def angular_error_deg(actual, target):
    norm_a = math.sqrt(sum(value * value for value in actual))
    norm_b = math.sqrt(sum(value * value for value in target))
    if norm_a < 1e-12 or norm_b < 1e-12:
        raise ValueError('四元数无效')
    dot = min(1.0, abs(sum(a*b for a, b in zip(actual, target))) / (norm_a * norm_b))
    return math.degrees(2.0 * math.acos(dot))

def option_present(arguments, name):
    return any(value == name or value.startswith(name + '=')
               for value in arguments)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            'AprilTag 和运动参数与 apriltag_left_tool_accuracy_test.py 相同。'
            '夹爪脚本只用连续观察窗口内的 tool 抖动判定稳定；'
            '--position-tolerance-mm/--angle-tolerance-deg 仅保留给闭合期间的安全保护。'))
    parser.add_argument(
        '--gripper-duration', type=float, default=8.0,
        help='夹爪命令从 0.0 线性增加到 1.0 的时间，默认 8 s')
    parser.add_argument(
        '--gripper-rate-hz', type=float, default=10.0,
        help='夹爪命令发布频率，默认 10 Hz，允许范围 1..20 Hz')
    parser.add_argument(
        '--gripper-topic', default='/tj/control/gripperValueL',
        help='左夹爪命令话题')
    parser.add_argument(
        '--gripper-feedback-topic', default='/info/gripper_feedback_L',
        help='左夹爪反馈话题')
    parser.add_argument(
        '--gripper-status-topic', default='/info/gripper_feedback_L_err',
        help='左夹爪状态码话题')
    parser.add_argument(
        '--stability-window-sec', type=float, default=1.5,
        help='连续位姿稳定判定窗口，单位 s，默认 1.5 s')
    parser.add_argument(
        '--stability-position-jitter-mm', type=float, default=0.5,
        help='窗口内 tool 位置最大抖动，单位 mm，默认 0.5 mm')
    parser.add_argument(
        '--stability-angle-jitter-deg', type=float, default=0.3,
        help='窗口内 tool 姿态最大抖动，单位 deg，默认 0.3 deg')
    parser.add_argument(
        '--return-speed-mm-s', type=float, default=20.0,
        help='夹爪闭合后返回 YAML 记录位姿的线速度，默认 20 mm/s')
    parser.add_argument(
        '--approach-height-mm', type=float, default=30.0,
        help='抓取前先到目标上方的高度，单位 mm，默认 30 mm；设为 0 关闭')
    parser.add_argument(
        '--return-pose-file', default=str(DEFAULT_RETURN_POSE_FILE),
        help='返回位姿快照 YAML，默认 left_tool_joint_snapshot_20261010.yaml')
    parser.add_argument(
        '--base-right-distance-mm', type=float, default=350.0,
        help='手臂返回后底盘向右横移的名义距离，默认 350 mm')
    parser.add_argument(
        '--base-lateral-speed-m-s', type=float, default=0.06,
        help='底盘横移速度绝对值，默认 0.06 m/s，范围 0.06..0.10 m/s')
    parser.add_argument(
        '--vlahost-url', default='http://127.0.0.1:8011',
        help='VLAHost HTTP 地址，默认 http://127.0.0.1:8011')

    gripper, remaining = parser.parse_known_args()
    defaults = (
        ('--position-tolerance-mm', '7.0'),
        ('--angle-tolerance-deg', '2.5'),
        ('--settle-time', '1.0'),
    )
    for name, value in defaults:
        if not option_present(remaining, name):
            remaining.extend((name, value))

    original = sys.argv
    try:
        sys.argv = [original[0]] + remaining
        tag_args, motion = tag_accuracy.parse_args()
    finally:
        sys.argv = original

    if not math.isfinite(gripper.gripper_duration) or gripper.gripper_duration <= 0.0:
        parser.error('--gripper-duration 必须为有限正数')
    if (not math.isfinite(gripper.gripper_rate_hz) or
            not 1.0 <= gripper.gripper_rate_hz <= 20.0):
        parser.error('--gripper-rate-hz 必须在 1..20 Hz 内')
    for name in (
            'stability_window_sec', 'stability_position_jitter_mm',
            'stability_angle_jitter_deg', 'return_speed_mm_s',
            'base_right_distance_mm', 'base_lateral_speed_m_s'):
        value = getattr(gripper, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f'--{name.replace("_", "-")} 必须为有限正数')
    if not 0.06 <= gripper.base_lateral_speed_m_s <= 0.10:
        parser.error('--base-lateral-speed-m-s 必须在 0.06..0.10 m/s 内')
    if (not math.isfinite(gripper.approach_height_mm) or
            gripper.approach_height_mm < 0.0):
        parser.error('--approach-height-mm 必须为大于等于 0 的有限数值')
    return tag_args, motion, gripper


def load_return_pose(path):
    """读取并校验快照中 base_link -> left_tool 的目标位姿。"""
    snapshot_path = Path(path).expanduser()
    try:
        with snapshot_path.open(encoding='utf-8') as stream:
            snapshot = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f'无法读取返回位姿文件 {snapshot_path}：{error}') from error

    try:
        pose = snapshot['tool_pose']
        position = tuple(float(value) for value in pose['xyz_m'])
        quaternion = tuple(float(value) for value in pose['quaternion_xyzw'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'返回位姿文件字段无效：{error}') from error
    if pose.get('base_frame') != 'base_link' or pose.get('tool_frame') != 'left_tool':
        raise ValueError('返回位姿文件必须记录 base_link -> left_tool')
    if (len(position) != 3 or len(quaternion) != 4 or
            not all(math.isfinite(value) for value in position + quaternion)):
        raise ValueError('返回位姿必须包含有限的 xyz_m[3] 和 quaternion_xyzw[4]')
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12 or abs(norm - 1.0) > 1e-3:
        raise ValueError(f'返回位姿四元数模长无效：{norm:.6f}')
    quaternion = tuple(value / norm for value in quaternion)
    return snapshot_path, position, quaternion


def request_json(base_url, path, body, timeout=3.0):
    """调用本机 VLAHost，并将 HTTP/JSON 错误转换为可读异常。"""
    data = json.dumps(body).encode('utf-8')
    request = Request(
        base_url.rstrip('/') + path, data=data,
        headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        raise ValueError(f'VLAHost HTTP {error.code}: {detail}') from error
    except (URLError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f'无法访问 VLAHost：{error}') from error


def get_json(base_url, path, timeout=3.0):
    """读取 VLAHost 状态接口。"""
    request = Request(base_url.rstrip('/') + path, method='GET')
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f'无法读取 VLAHost 状态：{error}') from error


class TagGrasp(tag_accuracy.TagAccuracy):
    """执行抓取、返回快照位姿，以及完成后的底盘右移。"""

    def __init__(self, motion, goal, gripper, return_position,
                 return_quaternion, return_pose_file):
        self.gripper_args = gripper
        self.phase = 'approach'
        self.close_started_at = None
        self.return_started_at = None
        self.return_start_position = None
        self.return_goal_position = None
        self.return_start_quaternion = None
        self.return_goal_quaternion = None
        self.return_duration = 0.0
        self.base_move_started_at = None
        self.base_move_duration = 0.0
        self.base_move_sequence = None
        self.last_base_move_poll_at = float('-inf')
        self.open_started_at = None
        self.recorded_return_position = tuple(return_position)
        self.recorded_return_quaternion = tuple(return_quaternion)
        self.return_pose_file = str(return_pose_file)
        self.last_gripper_publish_at = float('-inf')
        self.last_gripper_command = None
        self.last_grasp_log_at = float('-inf')
        self.gripper_feedback = None
        self.gripper_feedback_at = None
        self.gripper_status = None
        self.gripper_status_at = None
        self.stability_samples = deque()
        self.last_stability_log_at = float('-inf')
        self.go_home_client = None
        self.go_home_future = None
        self.go_home_started_at = None
        self.lift_started_at = None
        self.lift_start_position = None
        self.lift_goal_position = None
        self.lift_start_quaternion = None
        self.lift_duration = 0.0
        self.approach_waypoint_active = False
        self.approach_final_position = None
        super().__init__(motion, goal)

        # 直接调用 Apex QP 控制器提供的 ROS 服务，不经过 VLAHost HTTP 接口。
        self.go_home_client = self.create_client(
            Trigger,
            topic_name(motion.namespace, 'control/go_home'))

        self.gripper_publisher = self.create_publisher(
            Float32, gripper.gripper_topic, 10)
        self.create_subscription(
            Float32MultiArray, gripper.gripper_feedback_topic,
            self.on_gripper_feedback, 20)
        self.create_subscription(
            Int32MultiArray, gripper.gripper_status_topic,
            self.on_gripper_status, 20)
        self.get_logger().info(
            '夹爪动作：手臂稳定后 0.0 → 1.0，闭合时间 %.1f s，频率 %.1f Hz' %
            (gripper.gripper_duration, gripper.gripper_rate_hz))
        self.get_logger().info(
            '路径：直接移动到 AprilTag 转换得到的最终 left_tool 目标，'
            '保持姿态到位后再闭合夹爪')
        self.get_logger().info(
            '稳定平台判定：连续 %.1fs 窗口内 tool 抖动≤%.2fmm/%.2f°；'
            '不使用到目标的位置/姿态绝对误差作为触发条件' %
            (gripper.stability_window_sec,
             gripper.stability_position_jitter_mm,
             gripper.stability_angle_jitter_deg))
        self.get_logger().info(
            '返回目标来自 %s：base_link -> left_tool xyz=[%.6f, %.6f, %.6f]' %
            (self.return_pose_file, *self.recorded_return_position))
        self.get_logger().info(
            '松爪后直接调用 ROS %s 执行真实机器人 home 服务' %
            topic_name(motion.namespace, 'control/go_home'))
        self.get_logger().info(
            '返回完成后底盘向右横移 %.1f mm：vy=-%.3f m/s，VLAHost=%s' %
            (gripper.base_right_distance_mm,
             gripper.base_lateral_speed_m_s, gripper.vlahost_url))

    def on_gripper_feedback(self, message):
        if len(message.data) < 3:
            return
        values = tuple(float(value) for value in message.data[:3])
        if all(math.isfinite(value) for value in values):
            self.gripper_feedback = values
            self.gripper_feedback_at = time.monotonic()

    def on_gripper_status(self, message):
        if message.data:
            self.gripper_status = tuple(int(value) for value in message.data)
            self.gripper_status_at = time.monotonic()

    def use_absolute_error_settle(self):
        """稳定触发只看连续窗口内的 tool 抖动，不看绝对误差。"""
        return False

    def make_target(self, actual, quaternion, right_actual):
        """先到最终抓取点上方，再沿 base_link z 轴下降到目标点。"""
        super().make_target(actual, quaternion, right_actual)
        height_m = self.gripper_args.approach_height_mm / 1000.0
        if height_m <= 0.0:
            return
        self.approach_final_position = tuple(self.target)
        self.target = (
            self.approach_final_position[0],
            self.approach_final_position[1],
            self.approach_final_position[2] + height_m)
        self.approach_waypoint_active = True
        self.get_logger().warning(
            '抓取路径：先到 Tag 目标上方 %.1f mm，再下降到最终目标点；姿态保持标定目标' %
            self.gripper_args.approach_height_mm)

    @staticmethod
    def interpolate_quaternion(first, second, progress):
        """用最短弧线性插值四元数，避免返回时姿态突然跳变。"""
        dot = sum(a * b for a, b in zip(first, second))
        if dot < 0.0:
            second = tuple(-value for value in second)
            dot = -dot
        values = tuple(
            a + progress * (b - a) for a, b in zip(first, second))
        norm = math.sqrt(sum(value * value for value in values))
        if norm < 1e-12:
            return tuple(second)
        return tuple(value / norm for value in values)

    def other_gripper_publishers(self):
        result = []
        for endpoint in self.get_publishers_info_by_topic(
                self.gripper_args.gripper_topic):
            if (endpoint.node_name == self.get_name() and
                    endpoint.node_namespace == self.get_namespace()):
                continue
            full_name = '/'.join(
                part.strip('/') for part in
                (endpoint.node_namespace, endpoint.node_name) if part.strip('/'))
            result.append('/' + full_name)
        return sorted(set(result))

    def check_gripper(self, now):
        if self.gripper_publisher.get_subscription_count() == 0:
            return f'{self.gripper_args.gripper_topic} 没有订阅者'
        publishers = self.other_gripper_publishers()
        # if publishers:
        #     return ('夹爪命令话题存在其他发布者 %s；请先停止这些发布节点' %
        #             ', '.join(publishers))
        timeout = self.args.feedback_timeout
        if self.gripper_feedback_at is None or now - self.gripper_feedback_at > timeout:
            return '没有收到新鲜的左夹爪反馈'
        if self.gripper_status_at is None or now - self.gripper_status_at > timeout:
            return '没有收到新鲜的左夹爪状态码'
        # 达妙驱动中 1 表示 On/正常，0 表示 Off，8..14 表示各类故障。
        if any(value != 1 for value in self.gripper_status):
            return f'左夹爪状态码不是正常值 1：{list(self.gripper_status)}'
        return None

    def check_mux(self):
        problem = super().check_mux()
        if problem:
            return problem
        if self.args.execute:
            return self.check_gripper(time.monotonic())
        return None

    def finish(self, code, reason):
        if code == 0 and reason.startswith('PASS:'):
            if self.phase == 'approach':
                if self.approach_waypoint_active:
                    self.target = self.approach_final_position
                    self.approach_waypoint_active = False
                    self.reset_stability_window()
                    self.get_logger().warning(
                        '已到达 Tag 上方 %.1f mm，开始垂直下降到最终 Tag 目标点' %
                        self.gripper_args.approach_height_mm)
                    return
                self.stability_samples.clear()
                self.phase = 'closing'
                self.close_started_at = time.monotonic()
                self.settled_since = None
                self.get_logger().warning(
                    f'最终抓取点已稳定：{reason}；开始将左夹爪从 0.0 缓慢关闭到 1.0')
                return
            if self.phase == 'returning':
                self.start_base_right_move(reason)
                return
        if code != 0 and self.phase in ('base_moving', 'base_returning'):
            self.stop_base_motion()
        super().finish(code, reason)

    def start_base_right_move(self, arm_reason):
        """手臂返回完成后，通过 VLAHost 启动闭环世界坐标右移。"""
        distance_m = self.gripper_args.base_right_distance_mm / 1000.0
        body = {
            'offset_xy_m': [0.0, -distance_m],
            'speed_mps': self.gripper_args.base_lateral_speed_m_s,
        }
        try:
            result = request_json(
                self.gripper_args.vlahost_url, '/base_move_world', body)
        except ValueError as error:
            super().finish(3, f'手臂已返回，但底盘右移启动失败：{error}')
            return
        move = result.get('move', {})
        if not result.get('success') or move.get('status') != 'active':
            super().finish(
                3, '手臂已返回，但 VLAHost 未主动执行底盘动作：'
                f'{result}')
            return
        self.phase = 'base_moving'
        self.base_move_started_at = time.monotonic()
        self.base_move_duration = self.gripper_args.base_right_distance_mm / self.gripper_args.base_lateral_speed_m_s
        self.base_move_sequence = move
        self.get_logger().warning(
            '%s；底盘闭环向右横移 %.1f mm，offset_xy_m=[0.0, %.3f]，'
            'speed=%.3f m/s，VLAHost=%s' %
            (arm_reason, self.gripper_args.base_right_distance_mm,
             -distance_m, self.gripper_args.base_lateral_speed_m_s,
             self.gripper_args.vlahost_url))

    def start_base_return_move(self, arm_reason):
        """底盘右移和松爪后，沿相反方向闭环返回原位置。"""
        distance_m = self.gripper_args.base_right_distance_mm / 1000.0
        body = {'offset_xy_m': [0.0, distance_m],
                'speed_mps': self.gripper_args.base_lateral_speed_m_s}
        try:
            result = request_json(
                self.gripper_args.vlahost_url, '/base_move_world', body)
        except ValueError as error:
            super().finish(3, f'手臂回 home 后底盘反向返回失败：{error}')
            return
        move = result.get('move', {})
        if not result.get('success') or move.get('status') != 'active':
            super().finish(3, f'底盘反向返回未启动：{result}')
            return
        self.phase = 'base_returning'
        self.base_move_started_at = time.monotonic()
        self.last_base_move_poll_at = float('-inf')
        self.get_logger().warning(
            '%s；底盘沿反方向返回原位置 %.1f mm，speed=%.3f m/s' %
            (arm_reason, self.gripper_args.base_right_distance_mm,
             self.gripper_args.base_lateral_speed_m_s))

    def stop_base_motion(self):
        """用零速度块覆盖尚未结束的底盘动作。"""
        if self.base_move_started_at is None:
            return
        try:
            request_json(
                self.gripper_args.vlahost_url, '/base_move_world/cancel', {})
        except ValueError as error:
            self.get_logger().error(f'发送底盘零速度失败：{error}')
        self.base_move_started_at = None

    def start_go_home(self):
        """底盘反向返回完成后，直接调用 ROS go_home 服务。"""
        self.phase = 'go_home'
        service_name = self.go_home_client.srv_name
        self.get_logger().warning(
            f'夹爪已松开；直接调用 ROS {service_name} 返回左臂 home')
        if not self.go_home_client.wait_for_service(timeout_sec=2.0):
            super().finish(3, f'ROS go home 服务不可用：{service_name}')
            return
        self.go_home_started_at = time.monotonic()
        self.go_home_future = self.go_home_client.call_async(Trigger.Request())

    def start_lift_after_open(self):
        """松爪后先沿 base_link 的 z 轴向上移动 50 mm。"""
        try:
            (position, quaternion), _ = self.get_actuals()
        except (TransformException, ValueError) as error:
            super().finish(3, f'松爪后无法读取 left_tool 位姿，不能上移：{error}')
            return
        self.lift_start_position = tuple(position)
        self.lift_goal_position = (
            position[0], position[1], position[2] + 0.050)
        self.lift_start_quaternion = tuple(quaternion)
        self.lift_duration = 50.0 / self.gripper_args.return_speed_mm_s
        self.lift_started_at = time.monotonic()
        self.phase = 'lifting'
        self.target = self.lift_start_position
        self.target_quaternion = self.lift_start_quaternion
        self.get_logger().warning(
            '夹爪已松开；先以 %.1f mm/s 向上移动 50 mm，预计 %.1f s，'
            '之后底盘反向返回，最后执行 ROS go_home' %
            (self.gripper_args.return_speed_mm_s, self.lift_duration))

    def lift_tick(self):
        """执行松爪后的 50 mm 上移动作。"""
        now = time.monotonic()
        problem = self.check_mux()
        if problem:
            super().finish(3, '松爪后上移期间安全检查失败：' + problem)
            return
        self.publish_gripper(GRIPPER_OPEN, now)
        progress = min(
            1.0, max(0.0, (now - self.lift_started_at) / self.lift_duration))
        self.target = tuple(
            start + progress * (goal - start)
            for start, goal in zip(self.lift_start_position, self.lift_goal_position))
        self.target_quaternion = self.lift_start_quaternion
        self.publish_target()
        if progress >= 1.0:
            self.start_base_return_move('松爪后已向上移动 50 mm')

    def go_home_tick(self):
        """等待 ROS Trigger 返回，避免阻塞夹爪/底盘状态机。"""
        if self.go_home_future is None:
            return
        if (time.monotonic() - self.go_home_started_at) > 35.0:
            self.go_home_future = None
            super().finish(3, 'ROS go home 服务调用超时（35 s）')
            return
        if not self.go_home_future.done():
            return
        future = self.go_home_future
        self.go_home_future = None
        try:
            response = future.result()
        except Exception as error:
            super().finish(3, f'ROS go home 服务异常：{error}')
            return
        if not response.success:
            super().finish(3, f'ROS go home 返回失败：{response.message}')
            return
        super().finish(
            0, f'PASS: 底盘已返回原位置；ROS go_home 已启动：'
            f'{response.message or "success"}')

    def publish_gripper(self, value, now):
        period = 1.0 / self.gripper_args.gripper_rate_hz
        if now - self.last_gripper_publish_at + 1e-9 < period:
            return
        value = min(GRIPPER_CLOSED, max(GRIPPER_OPEN, float(value)))
        self.gripper_publisher.publish(Float32(data=value))
        self.last_gripper_command = value
        self.last_gripper_publish_at = now

    def fail_closing(self, reason):
        suffix = '' if self.last_gripper_command is None else (
            f'；停止增加闭合量，最后命令为 {self.last_gripper_command:.3f}')
        super().finish(3, reason + suffix)

    def closing_tick(self):
        now = time.monotonic()
        try:
            (actual, quaternion), _ = self.get_actuals()
        except (TransformException, ValueError) as error:
            if (self.last_valid_tf is None or
                    now - self.last_valid_tf > self.args.feedback_timeout):
                self.fail_closing(f'闭合期间 TF 失效：{error}')
            return
        self.last_valid_tf = now

        problem = self.check_mux()
        if problem:
            self.fail_closing('闭合期间安全检查失败：' + problem)
            return
        self.publish_target()

        position_mm = math.dist(actual, self.target) * 1000.0
        angle_deg = angular_error_deg(
            quaternion, self.target_quaternion)
        # 到位阈值外再留少量闭合扰动余量，超出即停止继续夹紧。
        position_limit = max(20.0, self.args.position_tolerance_mm)
        angle_limit = max(8.0, self.args.angle_tolerance_deg)
        if position_mm > position_limit or angle_deg > angle_limit:
            self.fail_closing(
                '闭合期间手臂偏离目标：%.3f mm / %.3f deg，限制 %.3f mm / %.3f deg' %
                (position_mm, angle_deg, position_limit, angle_limit))
            return

        elapsed = now - self.close_started_at
        if elapsed < GRIPPER_START_HOLD_SEC:
            command = GRIPPER_OPEN
            state = '起点保持'
        else:
            ramp_elapsed = elapsed - GRIPPER_START_HOLD_SEC
            progress = min(1.0, ramp_elapsed / self.gripper_args.gripper_duration)
            command = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * progress
            state = '缓慢闭合' if progress < 1.0 else '闭合保持'

        self.publish_gripper(command, now)
        if now - self.last_grasp_log_at >= 1.0:
            self.last_grasp_log_at = now
            position, velocity, torque = self.gripper_feedback
            self.get_logger().info(
                'GRASP %s command=%.3f feedback=[pos=%.5f vel=%.5f torque=%.5f] '
                'arm_error=%.3f mm / %.3f deg' %
                (state, command, position, velocity, torque,
                 position_mm, angle_deg))

        finish_time = (GRIPPER_START_HOLD_SEC +
                       self.gripper_args.gripper_duration +
                       GRIPPER_END_HOLD_SEC)
        if elapsed >= finish_time:
            self.publish_gripper(GRIPPER_CLOSED, now)
            self.phase = 'returning'
            self.return_started_at = now
            self.motion_start = now
            self.return_start_position = tuple(actual)
            self.return_goal_position = self.recorded_return_position
            self.return_start_quaternion = tuple(quaternion)
            self.return_goal_quaternion = self.recorded_return_quaternion
            return_distance_mm = math.dist(
                self.return_start_position, self.return_goal_position) * 1000.0
            if return_distance_mm > self.args.max_displacement_mm:
                self.fail_closing(
                    'YAML 坐标距抓取位置 %.3f mm，超过 '
                    '--max-displacement-mm=%.3f' %
                    (return_distance_mm, self.args.max_displacement_mm))
                return
            self.return_duration = (
                return_distance_mm / self.gripper_args.return_speed_mm_s)
            self.target = self.return_start_position
            self.target_quaternion = self.interpolate_quaternion(
                quaternion, self.return_goal_quaternion, 0.0)
            self.reset_stability_window()
            self.get_logger().warning(
                '夹爪已完全闭合并保持 %.1f s；以 %.1f mm/s 慢速移动到 YAML 坐标，'
                '预计 %.1f s' %
                (GRIPPER_END_HOLD_SEC, self.gripper_args.return_speed_mm_s,
                 self.return_duration))

    def opening_tick(self):
        """底盘右移完成后，将夹爪从 1.0 缓慢打开到 0.0。"""
        now = time.monotonic()
        problem = self.check_mux()
        if problem:
            super().finish(3, '松爪期间安全检查失败：' + problem)
            return
        self.publish_target()
        elapsed = now - self.open_started_at
        progress = min(1.0, max(0.0, elapsed / self.gripper_args.gripper_duration))
        self.publish_gripper(GRIPPER_CLOSED * (1.0 - progress), now)
        if progress >= 1.0:
            self.publish_gripper(GRIPPER_OPEN, now)
            self.start_lift_after_open()

    def reset_stability_window(self):
        self.stability_samples.clear()

    def maybe_accept_stable_plateau(self):
        """只根据连续窗口内末端实际位姿的抖动判断是否稳定。"""
        if (self.exit_code is not None or self.target is None or
                self.phase not in ('approach', 'returning')):
            self.reset_stability_window()
            return

        try:
            (position, quaternion), _ = self.get_actuals()
        except (TransformException, ValueError):
            self.reset_stability_window()
            return

        now = time.monotonic()
        self.stability_samples.append(
            (now, tuple(position), tuple(quaternion)))
        cutoff = now - self.gripper_args.stability_window_sec
        # 保留恰好跨过窗口起点的前一个样本，否则离散采样会使实际覆盖时长
        # 永远略小于配置窗口，导致稳定判定无法触发。
        while (len(self.stability_samples) > 1 and
               self.stability_samples[1][0] <= cutoff):
            self.stability_samples.popleft()
        if not self.stability_samples:
            return

        covered = now - self.stability_samples[0][0]
        if covered + 1e-9 < self.gripper_args.stability_window_sec:
            return

        samples = list(self.stability_samples)
        position_jitter_mm = max(
            math.dist(first[1], second[1]) * 1000.0
            for index, first in enumerate(samples)
            for second in samples[index:])
        angle_jitter_deg = max(
            angular_error_deg(first[2], second[2])
            for index, first in enumerate(samples)
            for second in samples[index:])
        stable = (
            position_jitter_mm <=
            self.gripper_args.stability_position_jitter_mm and
            angle_jitter_deg <=
            self.gripper_args.stability_angle_jitter_deg)
        return_position_error_mm = None
        return_angle_error_deg = None
        if self.phase == 'returning':
            return_position_error_mm = math.dist(position, self.target) * 1000.0
            return_angle_error_deg = angular_error_deg(
                quaternion, self.target_quaternion)
            stable = stable and (
                return_position_error_mm <= self.args.position_tolerance_mm and
                return_angle_error_deg <= self.args.angle_tolerance_deg)

        if now - self.last_stability_log_at >= 1.0:
            self.last_stability_log_at = now
            target_text = '' if return_position_error_mm is None else (
                ' return_error=%.3fmm/%.3f°' %
                (return_position_error_mm, return_angle_error_deg))
            self.get_logger().info(
                'STABILITY phase=%s window=%.2fs jitter=%.3fmm/%.3f° '
                'limit=%.3fmm/%.3f°%s stable=%s' %
                (self.phase, covered, position_jitter_mm, angle_jitter_deg,
                 self.gripper_args.stability_position_jitter_mm,
                 self.gripper_args.stability_angle_jitter_deg,
                 target_text, str(stable).lower()))
        if stable:
            if self.phase == 'returning':
                self.finish(
                    0, 'PASS: 已返回 YAML 记录的 base_link -> left_tool 位姿；'
                    f'误差 {return_position_error_mm:.3f} mm / '
                    f'{return_angle_error_deg:.3f} deg')
                return
            self.finish(
                0, 'PASS: STABILITY left_tool 进入连续位姿稳定窗口；'
                f'窗口抖动 {position_jitter_mm:.3f} mm / '
                f'{angle_jitter_deg:.3f} deg')

    def tick(self):
        if self.exit_code is not None:
            return
        if self.phase == 'approach':
            super().tick()
            if (self.phase == 'approach' and self.args.execute and
                    self.target is not None and self.exit_code is None):
                self.publish_gripper(GRIPPER_OPEN, time.monotonic())
                self.maybe_accept_stable_plateau()
        elif self.phase == 'returning':
            self.publish_gripper(GRIPPER_CLOSED, time.monotonic())
            now = time.monotonic()
            if self.return_duration > 1e-9:
                progress = min(
                    1.0, max(0.0, (now - self.return_started_at) /
                             self.return_duration))
            else:
                progress = 1.0
            self.target = tuple(
                start + progress * (goal - start)
                for start, goal in zip(
                    self.return_start_position, self.return_goal_position))
            self.target_quaternion = self.interpolate_quaternion(
                self.return_start_quaternion, self.return_goal_quaternion,
                progress)
            if progress < 1.0:
                self.reset_stability_window()
            super().tick()
            if (progress >= 1.0 and self.phase == 'returning' and
                    self.exit_code is None):
                self.maybe_accept_stable_plateau()
        elif self.phase in ('base_moving', 'base_returning'):
            now = time.monotonic()
            if now - self.last_base_move_poll_at < 0.2:
                return
            self.last_base_move_poll_at = now
            try:
                state = get_json(
                    self.gripper_args.vlahost_url, '/base_move_world')
            except ValueError as error:
                self.get_logger().warning(f'等待底盘闭环状态失败：{error}')
                return
            move = state.get('move', {})
            status = move.get('status')
            if status == 'reached':
                self.base_move_started_at = None
                if self.phase == 'base_moving':
                    self.phase = 'opening'
                    self.open_started_at = now
                    self.get_logger().warning(
                        '底盘右移完成；开始将左夹爪从 1.0 缓慢松开到 0.0')
                else:
                    self.start_go_home()
            elif status in ('failed', 'cancelled'):
                self.stop_base_motion()
                super().finish(
                    3, '底盘闭环%s未完成：%s；%s' %
                    ('右移' if self.phase == 'base_moving' else '反向返回',
                     status, move.get('reason', '未知原因')))
        elif self.phase == 'opening':
            self.opening_tick()
        elif self.phase == 'go_home':
            self.go_home_tick()
        elif self.phase == 'lifting':
            self.lift_tick()
        else:
            self.closing_tick()


def main():
    tag_args, motion, gripper = parse_args()
    rclpy.init(args=[])
    node = None
    try:
        return_pose_file, return_position, return_quaternion = load_return_pose(
            gripper.return_pose_file)
        node = tag_accuracy.TagCapture(tag_args)
        start = time.monotonic()
        while rclpy.ok() and node.goal is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - start > tag_args.capture_timeout:
                raise ValueError(
                    f'采集超时：未得到 ID {tag_args.tag_id} 的稳定观测')
        if node.goal is None:
            return 1
        goal = node.goal.copy()
        node.destroy_node()
        node = None
        motion.xyz = tuple(goal[:3, 3])
        node = TagGrasp(
            motion, goal, gripper, return_position, return_quaternion,
            return_pose_file)
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.exit_code if node.exit_code is not None else 1
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'apriltag_left_tool_grasp_test: {error}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            if hasattr(node, 'stop_base_motion'):
                node.stop_base_motion()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
