#!/usr/bin/python3
# 用途：通过 AprilTag 执行左手抓取、松爪与回位流程。
# 运行：source /home/xlb/data/code/code_new/Luna/Apex_Deploy/apex_ros_env.sh && python3 apriltag_left_tool_grasp_test.py --help
"""AprilTag 左手抓取；可运送到椅1/2，或执行原有抓放回位流程。"""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import re
import signal
import select
import subprocess
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
DEFAULT_RETURN_POSE_FILE = Path(__file__).with_name(
    'left_tool_joint_snapshot_20261010.yaml')
DEFAULT_BASE_RETURN_WORLD_Y_M = 0.22712
DEFAULT_BASE_RETURN_WORLD_X_M = 0.04681
DEFAULT_BASE_RETURN_WORLD_Z_M = 0.00603
DEFAULT_BASE_RETURN_WORLD_YAW_RAD = 0.05024
RESULT_PREFIX = 'RPENT_RESULT='
VISUAL_GATE_PREFIX = 'RPENT_GATE=await_visual_verdict'


def emit_result(phase, error_code, message):
    """Publish one compact outcome for the Harness; keep telemetry in the log."""
    print(RESULT_PREFIX + json.dumps({
        'phase': phase,
        'error_code': error_code,
        'message': message,
    }, ensure_ascii=False), flush=True)



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
            '夹爪脚本同时检查连续窗口内的 tool 抖动和目标误差；'
            '--position-tolerance-mm/--angle-tolerance-deg 用于到位判定。'))
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
        '--base-return-world-x-m', type=float,
        default=DEFAULT_BASE_RETURN_WORLD_X_M,
        help='最后反向移动的底盘世界坐标 x，默认 0.04681 m')
    parser.add_argument(
        '--base-return-world-y-m', type=float,
        default=DEFAULT_BASE_RETURN_WORLD_Y_M,
        help='最后反向移动的底盘世界坐标 y，默认 0.22712 m')
    parser.add_argument(
        '--base-return-world-z-m', type=float,
        default=DEFAULT_BASE_RETURN_WORLD_Z_M,
        help='最后反向移动的底盘世界坐标 z，默认 0.00603 m；仅检查，底盘接口不主动调 z')
    parser.add_argument(
        '--base-return-world-yaw-rad', type=float,
        default=DEFAULT_BASE_RETURN_WORLD_YAW_RAD,
        help='最后反向移动的底盘世界 yaw，默认 0.05024 rad（2.88 deg）')
    parser.add_argument(
        '--base-return-yaw-speed-rad-s', type=float, default=0.10,
        help='最后反向阶段底盘 yaw 闭环速度，默认 0.10 rad/s')
    parser.add_argument(
        '--base-return-pose-tolerance-m', type=float, default=0.03,
        help='底盘世界坐标位置误差容限，默认 0.03 m')
    parser.add_argument(
        '--base-return-yaw-tolerance-rad', type=float, default=0.04,
        help='底盘世界 yaw 误差容限，默认 0.04 rad')
    parser.add_argument(
        '--vlahost-url', default='http://127.0.0.1:8011',
        help='VLAHost HTTP 地址，默认 http://127.0.0.1:8011')
    parser.add_argument(
        '--destination-pose', type=int, choices=(1, 2),
        help='抓取并收回后移动到椅1或椅2并结束；设置后不右移、不松爪、不返回原点')
    parser.add_argument(
        '--cable-reference-yaw-rad', type=float,
        help='桌3导航前的电缆零位 yaw，供椅位导航沿用')
    parser.add_argument(
        '--require-visual-verdict', action='store_true',
        help='收臂后保持左臂和夹爪，只有 stdin 收到 held 才开始去椅子')
    parser.add_argument(
        '--home-on-visual-reject', action='store_true',
        help='视觉明确未抓到时先调用 ROS go_home，等待回位后交由 Harness 重试')
    parser.add_argument(
        '--visual-verdict-timeout-s', type=float, default=120.0,
        help='等待视觉判定的最长时间，默认 120 s')

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
            'base_right_distance_mm', 'base_lateral_speed_m_s',
            'base_return_world_x_m', 'base_return_world_y_m',
            'base_return_world_z_m',
            'base_return_world_yaw_rad', 'base_return_yaw_speed_rad_s',
            'base_return_pose_tolerance_m', 'base_return_yaw_tolerance_rad'):
        value = getattr(gripper, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f'--{name.replace("_", "-")} 必须为有限正数')
    if not 0.06 <= gripper.base_lateral_speed_m_s <= 0.10:
        parser.error('--base-lateral-speed-m-s 必须在 0.06..0.10 m/s 内')
    if gripper.base_return_yaw_speed_rad_s > 0.20:
        parser.error('--base-return-yaw-speed-rad-s 不能超过底盘 0.20 rad/s 限制')
    if (not math.isfinite(gripper.approach_height_mm) or
            gripper.approach_height_mm < 0.0):
        parser.error('--approach-height-mm 必须为大于等于 0 的有限数值')
    if gripper.require_visual_verdict and gripper.destination_pose is None:
        parser.error('--require-visual-verdict 需要 --destination-pose')
    if (gripper.destination_pose is not None and
            (gripper.cable_reference_yaw_rad is None or
             not math.isfinite(gripper.cable_reference_yaw_rad))):
        parser.error('--destination-pose 需要有限的 --cable-reference-yaw-rad')
    if (not math.isfinite(gripper.visual_verdict_timeout_s) or
            gripper.visual_verdict_timeout_s <= 0.0):
        parser.error('--visual-verdict-timeout-s 必须为有限正数')
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
        self.base_yaw_started_at = None
        self.base_yaw_direction_sign = 1.0
        self.base_yaw_direction_locked = False
        self.base_yaw_last_yaw = None
        self.base_yaw_last_command = 0.0
        self.base_yaw_last_motion_at = None
        self.base_return_world_pose = None
        self.destination_process = None
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
        self.go_home_wait_until = None
        self.retry_home_requested = False
        self.lift_started_at = None
        self.lift_start_position = None
        self.lift_goal_position = None
        self.lift_start_quaternion = None
        self.lift_duration = 0.0
        self.approach_waypoint_active = False
        self.approach_final_position = None
        self.visual_wait_started_at = None
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
            '先到目标上方，再下降到抓取点；保持姿态到位后再闭合夹爪')
        self.get_logger().info(
            '稳定平台判定：连续 %.1fs 窗口内 tool 抖动≤%.2fmm/%.2f°；'
            '同时要求到目标误差≤%.2fmm/%.2f°' %
            (gripper.stability_window_sec,
             gripper.stability_position_jitter_mm,
             gripper.stability_angle_jitter_deg,
             motion.position_tolerance_mm, motion.angle_tolerance_deg))
        self.get_logger().info(
            '返回目标来自 %s：base_link -> left_tool xyz=[%.6f, %.6f, %.6f]' %
            (self.return_pose_file, *self.recorded_return_position))
        self.get_logger().info(
            '松爪后直接调用 ROS %s 执行真实机器人 home 服务' %
            topic_name(motion.namespace, 'control/go_home'))
        if gripper.destination_pose is not None:
            self.get_logger().info(
                '任务模式：抓取并收回后保持夹爪闭合，移动到椅%d后结束；'
                '跳过右移、松爪和返回原点' % gripper.destination_pose)
        else:
            self.get_logger().info(
                '返回完成后底盘向右横移 %.1f mm：vy=-%.3f m/s，VLAHost=%s' %
                (gripper.base_right_distance_mm,
                 gripper.base_lateral_speed_m_s, gripper.vlahost_url))
            self.get_logger().info(
                '最后反向返回底盘世界坐标：x=%.5f m，y=%.5f m，z=%.5f m（仅检查），'
                'yaw=%.5f rad（%.2f°）；'
                '通过 /base_world_pose 闭环' %
                (gripper.base_return_world_x_m,
                 gripper.base_return_world_y_m,
                 gripper.base_return_world_z_m,
                 gripper.base_return_world_yaw_rad,
                 math.degrees(gripper.base_return_world_yaw_rad)))

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
        """由窗口判定统一检查抖动和到位误差，避免父类提前切换阶段。"""
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
                if self.gripper_args.destination_pose is not None:
                    if self.gripper_args.require_visual_verdict:
                        self.phase = 'await_visual_verdict'
                        self.visual_wait_started_at = time.monotonic()
                        self.get_logger().warning(
                            '左臂已收回；保持左臂和夹爪，等待 Harness 视觉判定')
                        print(VISUAL_GATE_PREFIX, flush=True)
                        return
                    self.start_destination_move(reason)
                    return
                self.start_base_right_move(reason)
                return
        if code != 0 and self.phase in (
                'base_moving', 'base_returning', 'base_yaw_returning'):
            self.stop_base_motion()
        super().finish(code, reason)
        error_code = None if code == 0 else (
            'precheck_failed' if reason.startswith('预检失败') else
            'destination_navigation_failed' if self.phase == 'destination_moving' else
            f'{self.phase}_failed')
        emit_result(self.phase, error_code, reason)

    def start_destination_move(self, arm_reason):
        """复用现有底盘脚本移动到椅1/椅2，运输期间保持夹爪闭合。"""
        destination = self.gripper_args.destination_pose
        self.phase = 'destination_moving'
        script = Path(__file__).with_name('base_move_world_pose_list.py')
        command = [
            sys.executable, str(script), f'[{destination}]',
            '--vlahost-url', self.gripper_args.vlahost_url,
            '--speed-m-s', str(self.gripper_args.base_lateral_speed_m_s),
            '--odometry-retries', '1',
            '--yaw-reference-rad', str(self.gripper_args.cable_reference_yaw_rad),
        ]
        try:
            self.destination_process = subprocess.Popen(command)
        except OSError as error:
            self.finish(3, f'启动椅{destination}导航失败：{error}')
            return
        self.get_logger().warning(
            f'{arm_reason}；保持左夹爪闭合并移动到椅{destination}')

    def destination_move_tick(self):
        """导航期间保持运输位姿和夹爪闭合，完成后直接结束任务。"""
        now = time.monotonic()
        self.publish_gripper(GRIPPER_CLOSED, now)
        self.publish_target()
        return_code = self.destination_process.poll()
        if return_code is None:
            return
        destination = self.gripper_args.destination_pose
        self.destination_process = None
        if return_code == 0:
            self.finish(
                0, f'PASS: 已抓取光模块并移动到椅{destination}；'
                '左夹爪保持闭合，未执行放置')
        else:
            self.finish(
                3, f'抓取已完成，但移动到椅{destination}失败，'
                f'底盘脚本退出码={return_code}')

    def visual_wait_tick(self):
        """Hold the grasp while the Harness VLM judges the camera mosaic."""
        now = time.monotonic()
        if now - self.visual_wait_started_at > self.gripper_args.visual_verdict_timeout_s:
            self.finish(3, '等待视觉判定超时；未启动椅位导航')
            return
        try:
            (actual, quaternion), _ = self.get_actuals()
        except (TransformException, ValueError) as error:
            if (self.last_valid_tf is None or
                    now - self.last_valid_tf > self.args.feedback_timeout):
                self.finish(3, f'视觉判定期间左臂 TF 失效：{error}')
            return
        self.last_valid_tf = now
        problem = self.check_mux()
        if problem:
            self.finish(3, '视觉判定期间控制检查失败：' + problem)
            return
        position_error_mm = math.dist(actual, self.target) * 1000.0
        angle_error = angular_error_deg(quaternion, self.target_quaternion)
        if (position_error_mm > self.args.position_tolerance_mm or
                angle_error > self.args.angle_tolerance_deg):
            self.finish(3, '视觉判定期间左臂偏离收回位姿：'
                        f'{position_error_mm:.1f} mm / {angle_error:.1f} deg')
            return
        self.publish_target()
        self.publish_gripper(GRIPPER_CLOSED, now)
        if not select.select([sys.stdin], [], [], 0)[0]:
            return
        verdict = sys.stdin.readline().strip()
        if verdict == 'held':
            self.start_destination_move('Harness VLM 已确认左夹爪持有光模块')
        elif verdict == 'retry' and self.gripper_args.home_on_visual_reject:
            self.retry_home_requested = True
            self.start_go_home()
        else:
            self.finish(3, '视觉判定未确认抓取；未启动椅位导航')

    def stop_destination_move(self):
        """父进程退出时通知底盘子进程走现有 Ctrl-C 停止路径。"""
        process = self.destination_process
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.get_logger().error('底盘导航子进程未在 5 s 内响应停止请求')

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
        """底盘反向阶段闭环回到指定世界 x/y；完成后再闭环修正 yaw。"""
        try:
            pose = get_json(self.gripper_args.vlahost_url, '/base_world_pose').get('pose')
            self.validate_base_world_pose(pose)
        except ValueError as error:
            super().finish(3, f'底盘反向返回前无法读取世界坐标：{error}')
            return

        current_x = float(pose['position']['x'])
        current_y = float(pose['position']['y'])
        offset_x = self.gripper_args.base_return_world_x_m - current_x
        offset_y = self.gripper_args.base_return_world_y_m - current_y
        self.base_return_world_pose = pose
        offset_distance = math.hypot(offset_x, offset_y)
        # /base_move_world 要求单次位移至少 50 mm；小误差直接跳过平移，随后修正 yaw。
        if offset_distance <= 0.03:
            self.get_logger().info(
                '底盘 x/y 已在目标容限内：current=[%.5f, %.5f] m '
                'target=[%.5f, %.5f] m；跳过平移' %
                (current_x, current_y,
                 self.gripper_args.base_return_world_x_m,
                 self.gripper_args.base_return_world_y_m))
            self.start_base_yaw_return(arm_reason)
            return
        if offset_distance < 0.05:
            super().finish(
                3, '底盘世界 x/y 误差 %.3f m 位于 0.03..0.05 m，'
                '/base_move_world 最小单次位移为 0.05 m，未强行越过目标' %
                offset_distance)
            return
        if offset_distance > 0.50:
            super().finish(
                3, '底盘世界 x/y 需要移动 %.3f m，超过 /base_move_world 单次 0.50 m 限制' %
                offset_distance)
            return
        body = {
            'offset_xy_m': [offset_x, offset_y],
            'speed_mps': self.gripper_args.base_lateral_speed_m_s,
        }
        try:
            result = request_json(
                self.gripper_args.vlahost_url, '/base_move_world', body)
        except ValueError as error:
            super().finish(3, f'手臂回 home 后底盘世界 x/y 返回失败：{error}')
            return
        move = result.get('move', {})
        if not result.get('success') or move.get('status') != 'active':
            super().finish(3, f'底盘世界 x/y 返回未启动：{result}')
            return
        self.phase = 'base_returning'
        self.base_move_started_at = time.monotonic()
        self.last_base_move_poll_at = float('-inf')
        self.get_logger().warning(
            '%s；底盘闭环返回世界 x/y：current=[%.5f, %.5f] m，'
            'target=[%.5f, %.5f] m，offset=[%.3f, %.3f] m，speed=%.3f m/s' %
            (arm_reason, current_x, current_y,
             self.gripper_args.base_return_world_x_m,
             self.gripper_args.base_return_world_y_m,
             offset_x, offset_y, self.gripper_args.base_lateral_speed_m_s))

    @staticmethod
    def validate_base_world_pose(pose):
        if not isinstance(pose, dict) or not pose.get('valid'):
            raise ValueError('世界底盘位姿无效')
        if pose.get('frame_id') != 'camera_init' or pose.get('child_frame_id') != 'body':
            raise ValueError(
                '世界底盘位姿坐标系异常，需要 camera_init -> body，实际为 '
                f'{pose.get("frame_id")} -> {pose.get("child_frame_id")}')
        if pose.get('received_age_s') is not None and pose['received_age_s'] > 0.5:
            raise ValueError(f'世界底盘位姿过期：{pose["received_age_s"]:.3f} s')
        position = pose.get('position')
        if not isinstance(position, dict) or not all(
                math.isfinite(float(position.get(axis))) for axis in ('x', 'y', 'z')):
            raise ValueError('世界底盘位置包含非有限值')
        if not math.isfinite(float(pose.get('yaw_rad'))):
            raise ValueError('世界底盘 yaw 无效')

    @staticmethod
    def wrapped_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def start_base_yaw_return(self, arm_reason):
        """平移完成后，使用 VLAHost base_velocity 闭环修正世界 yaw。"""
        try:
            pose = get_json(self.gripper_args.vlahost_url, '/base_world_pose').get('pose')
            self.validate_base_world_pose(pose)
        except ValueError as error:
            super().finish(3, f'底盘 yaw 返回前无法读取世界坐标：{error}')
            return
        yaw = float(pose['yaw_rad'])
        target = self.gripper_args.base_return_world_yaw_rad
        error = self.wrapped_angle(target - yaw)
        self.base_return_world_pose = pose
        if abs(error) <= self.gripper_args.base_return_yaw_tolerance_rad:
            self.finish_base_return(arm_reason, pose)
            return
        self.phase = 'base_yaw_returning'
        self.base_yaw_started_at = time.monotonic()
        self.base_yaw_direction_sign = 1.0
        self.base_yaw_direction_locked = False
        self.base_yaw_last_yaw = yaw
        self.base_yaw_last_command = 0.0
        self.base_yaw_last_motion_at = self.base_yaw_started_at
        self.last_base_move_poll_at = float('-inf')
        self.get_logger().warning(
            '%s；底盘开始闭环修正世界 yaw：current=%.5f rad target=%.5f rad error=%.5f rad' %
            (arm_reason, yaw, target, error))

    def finish_base_return(self, arm_reason, pose):
        self.validate_base_world_pose(pose)
        x = float(pose['position']['x'])
        y = float(pose['position']['y'])
        z = float(pose['position']['z'])
        yaw = float(pose['yaw_rad'])
        x_error = x - self.gripper_args.base_return_world_x_m
        y_error = y - self.gripper_args.base_return_world_y_m
        z_error = z - self.gripper_args.base_return_world_z_m
        yaw_error = self.wrapped_angle(
            yaw - self.gripper_args.base_return_world_yaw_rad)
        self.get_logger().info(
            '底盘世界目标完成：actual x=%.5f y=%.5f z=%.5f yaw=%.5f rad；'
            'error x=%.4f y=%.4f z=%.4f yaw=%.5f rad' %
            (x, y, z, yaw, x_error, y_error, z_error, yaw_error))
        if math.hypot(x_error, y_error) > self.gripper_args.base_return_pose_tolerance_m:
            super().finish(
                3, '底盘返回后 x/y 误差过大：%.4f m（x=%.4f, y=%.4f）' %
                (math.hypot(x_error, y_error), x_error, y_error))
            return
        if abs(z_error) > self.gripper_args.base_return_pose_tolerance_m:
            super().finish(
                3, '底盘返回后 z 与目标不一致（底盘接口不能主动调 z）：'
                f'actual={z:.5f} target={self.gripper_args.base_return_world_z_m:.5f}')
            return
        if abs(yaw_error) > self.gripper_args.base_return_yaw_tolerance_rad:
            super().finish(3, f'底盘返回后 yaw 误差过大：{yaw_error:.5f} rad')
            return
        self.start_go_home()

    def stop_base_motion(self):
        """用零速度块覆盖尚未结束的底盘动作。"""
        if self.phase == 'base_yaw_returning':
            try:
                self.publish_base_yaw_velocity(0.0)
            except ValueError as error:
                self.get_logger().error(f'发送底盘 yaw 零速度失败：{error}')
        if self.base_move_started_at is None:
            return
        try:
            request_json(
                self.gripper_args.vlahost_url, '/base_move_world/cancel', {})
        except ValueError as error:
            self.get_logger().error(f'发送底盘零速度失败：{error}')
        self.base_move_started_at = None

    def publish_base_yaw_velocity(self, velocity):
        """通过 VLAHost 单步发送底盘 yaw 速度，发送失败由调用方处理。"""
        return request_json(
            self.gripper_args.vlahost_url, '/action',
            {'base_velocity': [0.0, 0.0, float(velocity)]})

    def base_yaw_return_tick(self):
        """闭环修正底盘世界 yaw，达到目标后停止并校验 x/y/z/yaw。"""
        now = time.monotonic()
        try:
            pose = get_json(self.gripper_args.vlahost_url, '/base_world_pose').get('pose')
            self.validate_base_world_pose(pose)
        except ValueError as error:
            super().finish(3, f'底盘 yaw 返回期间读取世界坐标失败：{error}')
            return
        yaw = float(pose['yaw_rad'])
        yaw_error = self.wrapped_angle(
            self.gripper_args.base_return_world_yaw_rad - yaw)
        if self.base_yaw_last_yaw is not None:
            yaw_delta = self.wrapped_angle(yaw - self.base_yaw_last_yaw)
            if abs(yaw_delta) >= 0.003:
                self.base_yaw_last_motion_at = now
                if (not self.base_yaw_direction_locked and
                        abs(self.base_yaw_last_command) > 1e-9):
                    if yaw_delta * self.base_yaw_last_command < 0.0:
                        self.base_yaw_direction_sign = -1.0
                    self.base_yaw_direction_locked = True
                    self.get_logger().warning(
                        '已根据第一次有效 yaw 反馈锁定底盘命令方向：'
                        'sign=%+.0f，后续不再重复切换' %
                        self.base_yaw_direction_sign)
            self.base_yaw_last_yaw = yaw
        if abs(yaw_error) <= self.gripper_args.base_return_yaw_tolerance_rad:
            try:
                self.publish_base_yaw_velocity(0.0)
            except ValueError:
                pass
            self.finish_base_return('底盘 x/y 返回完成', pose)
            return
        if now - self.base_yaw_started_at > 10.0:
            try:
                self.publish_base_yaw_velocity(0.0)
            except ValueError:
                pass
            super().finish(3, f'底盘 yaw 返回超时，剩余误差 {yaw_error:.5f} rad')
            return
        if now - self.base_yaw_last_motion_at > 3.0:
            try:
                self.publish_base_yaw_velocity(0.0)
            except ValueError:
                pass
            super().finish(
                3, '底盘 yaw 命令发送后 3 s 内世界 yaw 没有有效变化；'
                f'剩余误差 {yaw_error:.5f} rad')
            return
        if now - self.last_base_move_poll_at < 0.1:
            return
        self.last_base_move_poll_at = now
        speed = math.copysign(
            min(self.gripper_args.base_return_yaw_speed_rad_s,
                max(0.06, abs(yaw_error) / 0.1)), yaw_error)
        speed *= self.base_yaw_direction_sign
        try:
            result = self.publish_base_yaw_velocity(speed)
        except ValueError as error:
            super().finish(3, f'底盘 yaw 速度命令失败：{error}')
            return
        if not result.get('success'):
            super().finish(3, f'底盘 yaw 速度命令未执行：{result}')
            return
        self.base_yaw_last_command = speed

    def start_go_home(self):
        """Call ROS go_home after release or an explicitly failed visual grasp."""
        self.phase = 'go_home'
        service_name = self.go_home_client.srv_name
        self.get_logger().warning(
            f'视觉未抓到，保持夹爪闭合并调用 ROS {service_name} 回 home'
            if self.retry_home_requested else
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
        if self.retry_home_requested:
            self.publish_gripper(GRIPPER_CLOSED, time.monotonic())
        if self.go_home_wait_until is not None:
            if time.monotonic() < self.go_home_wait_until:
                return
            try:
                self.get_actuals()
            except (TransformException, ValueError) as error:
                self.finish(3, f'go_home 后左臂 TF 无效，禁止重试：{error}')
                return
            problem = self.check_gripper(time.monotonic())
            if problem:
                self.finish(3, f'go_home 后夹爪状态异常，禁止重试：{problem}')
                return
            self.phase = 'retry_ready'
            self.finish(0, '视觉明确未抓到；go_home 计划时长已结束，允许 Harness 重新抓取')
            return
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
        if self.retry_home_requested:
            match = re.search(r'duration=([0-9.]+)s', response.message)
            duration = float(match.group(1)) if match else float('nan')
            if not math.isfinite(duration) or not 0.0 < duration <= 30.0:
                self.finish(3, f'无法确认 go_home 计划时长，禁止重试：{response.message}')
                return
            self.go_home_wait_until = self.go_home_started_at + duration + 1.0
            self.get_logger().warning(
                'ROS go_home 已启动；等待计划时长 %.1f s 再检查反馈并允许重试' % duration)
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
        self.finish(3, reason + suffix)

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
        position_limit = max(40.0, self.args.position_tolerance_mm)
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
        """末端同时稳定且到位才切换阶段，避免把停滞误判为到达。"""
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
        position_error_mm = math.dist(position, self.target) * 1000.0
        angle_error_deg = angular_error_deg(quaternion, self.target_quaternion)
        stable = stable and (
            position_error_mm <= self.args.position_tolerance_mm and
            angle_error_deg <= self.args.angle_tolerance_deg)

        if now - self.last_stability_log_at >= 1.0:
            self.last_stability_log_at = now
            target_text = ' target_error=%.3fmm/%.3f°' % (
                position_error_mm, angle_error_deg)
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
                    f'误差 {position_error_mm:.3f} mm / '
                    f'{angle_error_deg:.3f} deg')
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
                    self.start_base_yaw_return('底盘世界 x/y 返回完成')
            elif status in ('failed', 'cancelled'):
                self.stop_base_motion()
                super().finish(
                    3, '底盘闭环%s未完成：%s；%s' %
                    ('右移' if self.phase == 'base_moving' else '反向返回',
                     status, move.get('reason', '未知原因')))
        elif self.phase == 'base_yaw_returning':
            self.base_yaw_return_tick()
        elif self.phase == 'destination_moving':
            self.destination_move_tick()
        elif self.phase == 'await_visual_verdict':
            self.visual_wait_tick()
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
            emit_result('tag_capture', 'tag_capture_failed',
                        'AprilTag capture ended without a stable target')
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
        if node.exit_code is None:
            emit_result(node.phase, 'workflow_interrupted',
                        'ROS stopped before the workflow completed')
            return 1
        return node.exit_code
    except KeyboardInterrupt:
        emit_result(getattr(node, 'phase', 'tag_capture'), 'interrupted',
                    'Workflow interrupted by operator')
        return 130
    except Exception as error:
        print(f'apriltag_left_tool_grasp_test: {error}', file=sys.stderr)
        phase = getattr(node, 'phase', 'tag_capture')
        emit_result(phase, 'tag_capture_failed' if phase == 'tag_capture'
                    else f'{phase}_exception', str(error))
        return 1
    finally:
        if node is not None:
            if hasattr(node, 'stop_destination_move'):
                node.stop_destination_move()
            if hasattr(node, 'stop_base_motion'):
                node.stop_base_motion()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
