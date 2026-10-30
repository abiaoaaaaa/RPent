#!/usr/bin/python3
# 用途：交互式调整 base_link 下指定工具（left_tool/right_tool）的位置和姿态。
# 运行：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && python3 interactive_left_tool_pose.py --help
"""交互式调整 base_link 下指定工具的位姿；默认控制 left_tool。"""

import argparse
import math
import os
import shlex
import sys
import threading
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from marvin_msgs.msg import IKRequest
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from std_msgs.msg import Int16MultiArray, Int32
from tf2_ros import Buffer, TransformException, TransformListener



def topic_name(namespace, name):
    """将相对话题解析到指定命名空间。"""
    if name.startswith('/'):
        return name
    prefix = namespace.strip('/')
    return '/' + '/'.join(part for part in (prefix, name) if part)


def rotate_vector(vector, quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError('TF 四元数无效')
    x, y, z, w = (value / norm for value in quaternion)
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx)


def multiply_quaternions(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return (w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2)


SIDE_CONFIG = {
    "left": {
        "arm_link": "Arm_L7_Link",
        "flange": "flange_L",
        "tool": "left_tool",
        "pose_field": "left_hand_pose",
        "hand_index": 0,
    },
    "right": {
        "arm_link": "Arm_R7_Link",
        "flange": "flange_R",
        "tool": "right_tool",
        "pose_field": "right_hand_pose",
        "hand_index": 1,
    },
}


def read_tool_mount(model_xml, side="left"):
    """Read the selected wrist-link to tool fixed transform from the MJCF."""
    config = SIDE_CONFIG[side]
    import xml.etree.ElementTree as ET
    root = ET.parse(model_xml).getroot()
    arm = root.find(f".//body[@name='{config['arm_link']}']")
    flange = None if arm is None else arm.find(
        f"body[@name='{config['flange']}']"
    )
    tool = None if flange is None else flange.find(
        f"body[@name='{config['tool']}']"
    )
    if tool is None:
        raise ValueError(
            f"MJCF 中找不到 {config['arm_link']}/{config['flange']}/{config['tool']}"
        )
    def placement(body):
        position = tuple(float(v) for v in body.get('pos', '0 0 0').split())
        wxyz = tuple(float(v) for v in body.get('quat', '1 0 0 0').split())
        if len(position) != 3 or len(wxyz) != 4:
            raise ValueError('MJCF 工具固定变换格式错误')
        return position, (wxyz[1], wxyz[2], wxyz[3], wxyz[0])
    flange_position, flange_quaternion = placement(flange)
    tool_position, tool_quaternion = placement(tool)
    rotated = rotate_vector(tool_position, flange_quaternion)
    position = tuple(a + b for a, b in zip(flange_position, rotated))
    return position, multiply_quaternions(flange_quaternion, tool_quaternion)


def transform_pose(stamped):
    p, q = stamped.transform.translation, stamped.transform.rotation
    position, quaternion = (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
    if not all(math.isfinite(value) for value in position + quaternion):
        raise ValueError('TF 包含非有限数值')
    return position, quaternion


def angular_error_deg(actual, target):
    norm_a = math.sqrt(sum(value * value for value in actual))
    norm_b = math.sqrt(sum(value * value for value in target))
    if norm_a < 1e-12 or norm_b < 1e-12:
        raise ValueError('TF 四元数无效')
    dot = min(1.0, abs(sum(a*b for a, b in zip(actual, target))) / (norm_a * norm_b))
    return math.degrees(2.0 * math.acos(dot))

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            '启动后输入 help 查看命令。姿态角单位为 deg；位置增量单位为 mm。'
            '不加 --execute 时只预览，不发送机器人命令。'))
    parser.add_argument(
        '--execute', action='store_true',
        help='允许发布 IK 目标；不加此选项时仅预览')
    parser.add_argument('--namespace', default='/tj',
                        help='控制话题命名空间，默认 /tj')
    parser.add_argument('--side', choices=('left', 'right'), default='left',
                        help='要控制的工具侧；默认 left')
    parser.add_argument('--ik-topic', default='control/ik_request/vla',
                        help='IK 请求话题')
    parser.add_argument('--model-xml',
                        help='Luna MJCF 路径；默认读取已安装的 marvin_description')
    parser.add_argument('--rate-hz', type=float, default=20.0,
                        help='目标发布频率，默认 20 Hz')
    parser.add_argument('--tf-timeout', type=float, default=5.0,
                        help='等待初始 left_tool TF 的时间，默认 5 s')
    parser.add_argument('--max-position-offset-mm', type=float, default=50.0,
                        help='相对初始位置的交互调整上限，默认 50 mm')
    parser.add_argument('--max-angle-offset-deg', type=float, default=45.0,
                        help='相对初始姿态的交互调整角度上限，默认 45 deg')
    parser.add_argument('--max-position-step-mm', type=float, default=10.0,
                        help='单条命令允许的位置增量上限，默认 10 mm')
    parser.add_argument('--max-angle-step-deg', type=float, default=10.0,
                        help='单条命令允许的姿态旋转上限，默认 10 deg')
    parser.add_argument('--feedback-timeout', type=float, default=0.5,
                        help='TF 和机器人状态的新鲜度上限，默认 0.5 s')
    parser.add_argument('--tf-future-skew-s', type=float, default=1.0,
                        help='允许远端 ROS 时钟领先本机的上限，默认 1.0 s')
    parser.add_argument('--disarm-time', type=float, default=0.5,
                        help='start 前发送 unavailable 握手的时间，默认 0.5 s')
    args = parser.parse_args()
    for name in ('rate_hz', 'tf_timeout', 'max_position_offset_mm',
                 'max_angle_offset_deg', 'max_position_step_mm',
                 'max_angle_step_deg', 'feedback_timeout', 'tf_future_skew_s',
                 'disarm_time'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f'--{name.replace("_", "-")} 必须为有限正数')
    if args.rate_hz > 100.0:
        parser.error('--rate-hz 不能大于 100 Hz')
    return args


def format_xyz(values, digits=4):
    return '[' + ', '.join(f'{value:.{digits}f}' for value in values) + ']'


def format_angles(values, digits=2):
    return '[' + ', '.join(f'{value:.{digits}f}' for value in values) + '] deg'


class InteractivePose(Node):
    """维护交互目标，并在需要时持续发布给 IK mux。"""

    def __init__(self, args):
        self.side = args.side
        self.side_config = SIDE_CONFIG[self.side]
        super().__init__(f'interactive_{self.side}_tool_pose')
        self.args = args
        self.ik_topic = topic_name(args.namespace, args.ik_topic)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        model_xml = args.model_xml
        if model_xml is None:
            model_xml = os.path.join(
                get_package_share_directory('marvin_description'),
                'mjcf', 'Gento_Luna', 'Gento_Luna.xml')
        self.tool_mount_position, self.tool_mount_quaternion = read_tool_mount(
            model_xml, self.side)
        self.publisher = self.create_publisher(IKRequest, self.ik_topic, 10)
        state_qos = QoSProfile(depth=1)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Int32,
            topic_name(args.namespace, 'info/ik_request_mux/active_source'),
            self.on_ik_source, state_qos)
        self.create_subscription(
            Int32,
            topic_name(
                args.namespace, 'info/joint_cmd_mux/active_source'),
            self.on_joint_source, state_qos)
        self.create_subscription(
            Int16MultiArray,
            topic_name(args.namespace, 'info/robot_state'),
            self.on_robot_state, 10)
        self.ik_source = None
        self.joint_source = None
        self.robot_ready = False
        self.robot_state_at = None
        self.target_position = None
        self.target_quaternion = None
        self.initial_position = None
        self.initial_rotation = None
        self.publishing = False
        self.last_problem = None
        self.lock = threading.RLock()
        self.create_timer(1.0 / args.rate_hz, self.publish_tick)

    def on_ik_source(self, message):
        self.ik_source = message.data

    def on_joint_source(self, message):
        self.joint_source = message.data

    def on_robot_state(self, message):
        self.robot_ready = (
            len(message.data) >= 4 and
            all(message.data[index] in (1, 2) for index in (0, 1, 3)))
        self.robot_state_at = time.monotonic()

    def get_actual(self):
        return self.get_tool_actual(
            self.side_config['arm_link'], self.tool_mount_position,
            self.tool_mount_quaternion)

    def get_tool_actual(self, link_name, mount_position, mount_quaternion):
        stamped = self.buffer.lookup_transform(
            'base_link', link_name, Time(),
            timeout=Duration(seconds=0.02))
        stamp = (stamped.header.stamp.sec +
                 stamped.header.stamp.nanosec * 1e-9)
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if (stamp <= 0.0 or
                not -self.args.tf_future_skew_s <= age <= self.args.feedback_timeout):
            raise ValueError(f'{link_name} TF 时间戳无效或过旧：{age:.3f} s')
        arm_position, arm_quaternion = transform_pose(stamped)
        rotated = rotate_vector(mount_position, arm_quaternion)
        position = tuple(a + b for a, b in zip(arm_position, rotated))
        quaternion = multiply_quaternions(
            arm_quaternion, mount_quaternion)
        return position, quaternion

    @staticmethod
    def normalize_quaternion(quaternion):
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm < 1e-12 or not math.isfinite(norm):
            raise ValueError('目标四元数无效')
        return tuple(value / norm for value in quaternion)

    def set_initial(self, position, quaternion):
        rotation = Rotation.from_quat(quaternion)
        with self.lock:
            self.initial_position = tuple(position)
            self.initial_rotation = rotation
            self.target_position = tuple(position)
            self.target_quaternion = self.normalize_quaternion(quaternion)

    def target_rotation(self):
        with self.lock:
            return Rotation.from_quat(self.target_quaternion)

    def target_rpy(self):
        # 小写 xyz 表示绕固定 base_link X/Y/Z 轴的外禀旋转。
        return self.target_rotation().as_euler('xyz', degrees=True)

    def target_summary(self):
        with self.lock:
            position = self.target_position
            quaternion = self.target_quaternion
            publishing = self.publishing
        return (f'target xyz={format_xyz(position)} m, '
                f'XYZ={format_angles(self.target_rpy())}, '
                f'quat_xyzw={format_xyz(quaternion, 6)}, '
                f'publishing={publishing}')

    def actual_summary(self):
        position, quaternion = self.get_actual()
        angles = Rotation.from_quat(quaternion).as_euler(
            'xyz', degrees=True)
        return (f'actual xyz={format_xyz(position)} m, '
                f'XYZ={format_angles(angles)}, '
                f'quat_xyzw={format_xyz(quaternion, 6)}')

    def apply_rotation_delta(self, degrees):
        if len(degrees) != 3 or not all(math.isfinite(value) for value in degrees):
            raise ValueError('需要 3 个有限角度值')
        with self.lock:
            delta = Rotation.from_euler('xyz', degrees, degrees=True)
            delta_angle = math.degrees(delta.magnitude())
            if delta_angle > self.args.max_angle_step_deg + 1e-9:
                raise ValueError(
                    f'本次旋转 {delta_angle:.2f} deg，超过单次上限 '
                    f'{self.args.max_angle_step_deg:.2f} deg')
            rotation = delta * Rotation.from_quat(self.target_quaternion)
            candidate = self.normalize_quaternion(rotation.as_quat())
            self.check_limits(self.target_position, candidate)
            self.target_quaternion = candidate

    def set_rotation(self, degrees):
        if len(degrees) != 3 or not all(math.isfinite(value) for value in degrees):
            raise ValueError('需要 3 个有限角度值')
        with self.lock:
            candidate = self.normalize_quaternion(
                Rotation.from_euler('xyz', degrees, degrees=True).as_quat())
            step = angular_error_deg(
                candidate, self.target_quaternion)
            if step > self.args.max_angle_step_deg + 1e-9:
                raise ValueError(
                    f'本次旋转 {step:.2f} deg，超过单次上限 '
                    f'{self.args.max_angle_step_deg:.2f} deg')
            self.check_limits(self.target_position, candidate)
            self.target_quaternion = candidate

    def apply_position_delta(self, millimeters):
        if len(millimeters) != 3 or not all(
                math.isfinite(value) for value in millimeters):
            raise ValueError('需要 3 个有限位置增量值')
        step = math.sqrt(sum(value * value for value in millimeters))
        if step > self.args.max_position_step_mm + 1e-9:
            raise ValueError(
                f'本次位移 {step:.2f} mm，超过单次上限 '
                f'{self.args.max_position_step_mm:.2f} mm')
        with self.lock:
            candidate = tuple(
                position + value / 1000.0
                for position, value in zip(self.target_position, millimeters))
            self.check_limits(candidate, self.target_quaternion)
            self.target_position = candidate

    def reset_to_actual(self):
        position, quaternion = self.get_actual()
        with self.lock:
            candidate_position = tuple(position)
            candidate_quaternion = self.normalize_quaternion(quaternion)
            self.check_limits(candidate_position, candidate_quaternion)
            self.target_position = candidate_position
            self.target_quaternion = candidate_quaternion

    def check_limits(self, position, quaternion):
        position_offset = math.dist(position, self.initial_position) * 1000.0
        angle_offset = angular_error_deg(
            quaternion, self.initial_rotation.as_quat())
        if position_offset > self.args.max_position_offset_mm + 1e-9:
            raise ValueError(
                f'相对初始位置偏移 {position_offset:.2f} mm，超过上限 '
                f'{self.args.max_position_offset_mm:.2f} mm')
        if angle_offset > self.args.max_angle_offset_deg + 1e-9:
            raise ValueError(
                f'相对初始姿态旋转 {angle_offset:.2f} deg，超过上限 '
                f'{self.args.max_angle_offset_deg:.2f} deg')

    def check_mux(self):
        if self.ik_source is None:
            return '尚未收到 IK mux 当前源'
        if self.ik_source != 2:
            return f'IK mux 当前源为 {self.ik_source}，需要源 2'
        if self.joint_source is None:
            return '尚未收到 joint command mux 当前源'
        if self.joint_source != 0:
            return f'joint command mux 当前源为 {self.joint_source}，需要源 0'
        now = time.monotonic()
        if (self.robot_state_at is None or
                now - self.robot_state_at > self.args.feedback_timeout):
            return '尚未收到新鲜的机器人可运动状态'
        if not self.robot_ready:
            return '左右臂或躯干当前不可运动'
        if self.publisher.get_subscription_count() == 0:
            return f'{self.ik_topic} 没有订阅者'
        if self.count_publishers(self.ik_topic) > 1:
            return f'{self.ik_topic} 存在其他发布者，请先停止其他 IK 源'
        return None

    def make_message(self, available):
        with self.lock:
            position = self.target_position
            quaternion = self.target_quaternion
        message = IKRequest()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'base_link'
        message.available = available
        if available[self.side_config['hand_index']]:
            pose = getattr(message, self.side_config['pose_field'])
            pose.position.x, pose.position.y, pose.position.z = position
            (pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w) = quaternion
        return message

    def publish_tick(self):
        with self.lock:
            publishing = self.publishing
        if not publishing:
            return
        problem = self.check_mux()
        if problem is None:
            try:
                self.get_actual()
            except (TransformException, ValueError) as error:
                problem = f"{self.side_config['tool']} TF 不可用：{error}"
        if problem:
            if problem != self.last_problem:
                self.get_logger().error(problem + '；已自动暂停发布')
                self.last_problem = problem
            with self.lock:
                self.publishing = False
            self.publisher.publish(self.make_message([False] * 11))
            return
        self.last_problem = None
        available = [False] * 11
        available[self.side_config['hand_index']] = True
        self.publisher.publish(self.make_message(available))

    def start_publishing(self):
        if not self.args.execute:
            raise ValueError('当前为预览模式，请使用 --execute 启动后才能发布')
        problem = self.check_mux()
        if problem:
            raise ValueError(problem)
        self.get_logger().info('发送 unavailable 请求，准备 teleop 源重新使能')
        deadline = time.monotonic() + self.args.disarm_time
        period = 1.0 / self.args.rate_hz
        while time.monotonic() < deadline:
            self.publisher.publish(self.make_message([False] * 11))
            time.sleep(period)
        with self.lock:
            self.publishing = True
        self.get_logger().warning(
            f"开始持续发布 {self.side_config['tool']} 目标；请保持急停可用"
        )

    def stop_publishing(self):
        with self.lock:
            self.publishing = False
        if self.args.execute:
            for _ in range(5):
                self.publisher.publish(self.make_message([False] * 11))
                time.sleep(0.02)


def print_help():
    print('''命令：
  show                         显示当前 TF 和目标位姿
  rpy DR DP DY                相对 base_link X/Y/Z 轴旋转（deg）
  setrpy R P Y                设置目标 XYZ 欧拉角（deg）
  pos DX DY DZ                相对 base_link 移动（mm）
  reset                        将目标重置为当前实际位姿
  start                        开始持续发布目标（需 --execute）
  pause                        暂停发布目标
  help                         显示帮助
  quit                         停止发布并退出

说明：XYZ 使用固定 base_link 轴；例如 `rpy 0 10 0` 表示目标绕
base_link 的 Y 轴增加 10 度。每次调整后都会检查相对初始位姿的安全上限。''')


def wait_for_initial_tf(node, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            return node.get_actual()
        except (TransformException, ValueError) as error:
            last_error = error
    raise RuntimeError(f"等待 {node.side_config['tool']} TF 超时：{last_error}")


def run_interactive(node):
    print_help()
    while rclpy.ok():
        try:
            line = input('pose> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        try:
            words = shlex.split(line)
            command = words[0].lower()
            values = [float(value) for value in words[1:]]
            if command in ('quit', 'exit', 'q'):
                break
            if command in ('help', 'h', '?'):
                print_help()
            elif command in ('show', 's'):
                print(node.actual_summary())
                print(node.target_summary())
            elif command == 'rpy' and len(values) == 3:
                node.apply_rotation_delta(values)
                print(node.target_summary())
            elif command in ('setrpy', 'rpy_abs') and len(values) == 3:
                node.set_rotation(values)
                print(node.target_summary())
            elif command in ('pos', 'xyz') and len(values) == 3:
                node.apply_position_delta(values)
                print(node.target_summary())
            elif command in ('reset', 'hold') and not values:
                node.reset_to_actual()
                print('目标已重置为当前实际位姿。')
                print(node.target_summary())
            elif command in ('start', 'send') and not values:
                node.start_publishing()
                print('已开始发布。')
            elif command in ('pause', 'stop') and not values:
                node.stop_publishing()
                print('已暂停发布。')
            else:
                print('命令格式不正确，请输入 help。')
        except (ValueError, TransformException, RuntimeError) as error:
            print(f'错误：{error}')


def main():
    args = parse_args()
    rclpy.init(args=[])
    node = None
    spin_thread = None
    try:
        node = InteractivePose(args)
        position, quaternion = wait_for_initial_tf(node, args.tf_timeout)
        node.set_initial(position, quaternion)
        print(f"已读取当前 {node.side_config['tool']}：")
        print(node.actual_summary())
        print(node.target_summary())
        spin_thread = threading.Thread(
            target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        run_interactive(node)
        node.stop_publishing()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'interactive_left_tool_pose: {error}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    sys.exit(main())
