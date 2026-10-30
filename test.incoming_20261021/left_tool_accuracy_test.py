#!/usr/bin/python3
# 用途：测试 QP 控制器将 left_tool 移到指定目标位姿的精度。
# 运行：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && python3 left_tool_accuracy_test.py --help
"""测试 QP 控制器能否将 left_tool 移到 base_link 下指定位置。"""

import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseArray
from marvin_msgs.msg import IKRequest, JointcmdArm, Jointfeedback
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Int32
from tf2_ros import Buffer, TransformException, TransformListener


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--xyz', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
        help='base_link 下的绝对目标位置，单位 m；保留当前 left_tool 方向')
    group.add_argument(
        '--offset', nargs=3, type=float, metavar=('DX', 'DY', 'DZ'),
        default=(0.005, 0.0, 0.0),
        help='相对当前位置的位移，单位 m，默认沿 base_link +X 走 5 mm')
    parser.add_argument(
        '--offset-frame', choices=('base_link', 'left_tool'),
        default='base_link', help='--offset 使用的坐标轴')
    parser.add_argument(
        '--execute', action='store_true', help='通过 IK mux 发布目标')
    parser.add_argument(
        '--timeout', type=float, default=180.0,
        help='到达最终目标的总超时，单位 s')
    parser.add_argument('--position-tolerance-mm', type=float, default=1.0)
    parser.add_argument('--angle-tolerance-deg', type=float, default=0.5)
    parser.add_argument(
        '--settle-time', type=float, default=0.5,
        help='连续到位时间，单位 s')
    parser.add_argument(
        '--max-displacement-mm', type=float, default=500.0,
        help='从起点到最终目标的总位移上限，默认 500 mm')
    parser.add_argument('--feedback-timeout', type=float, default=0.5)
    parser.add_argument('--preflight-timeout', type=float, default=5.0)
    parser.add_argument(
        '--disarm-time', type=float, default=0.5,
        help='执行前持续发送空请求的时间，满足 teleop 源重新使能握手，单位 s')
    parser.add_argument(
        '--namespace', default='/tj',
        help='QP 与 mux 的 ROS 命名空间，默认 /tj')
    parser.add_argument(
        '--ik-topic', default='control/teleop/ik_request',
        help='IK 请求输入话题，默认使用 mux 源 0 的 teleop 话题')
    parser.add_argument(
        '--qp-pose-topic', default='info/upper_body_ik_result',
        help='QP 内部上半身位姿话题，默认 info/upper_body_ik_result')
    parser.add_argument(
        '--qp-left-hand-index', type=int, default=5,
        help='PoseArray 中 left_hand 的索引，当前 QP 默认是 5')
    parser.add_argument(
        '--qp-joint-topic', default='control/qp_controller/joint_cmd_A',
        help='QP 左臂关节命令话题')
    parser.add_argument(
        '--mux-joint-topic', default='control/joint_cmd_A',
        help='joint_cmd_mux 输出的左臂关节命令话题')
    parser.add_argument(
        '--joint-feedback-topic', default='info/joint_feedback',
        help='实机关节反馈话题')
    parser.add_argument(
        '--model-xml',
        help='Luna MJCF 文件路径，默认读取已安装的 marvin_description')
    parser.add_argument('--ik-source', type=int, default=0)
    parser.add_argument('--joint-source', type=int, default=0)
    args = parser.parse_args()
    values = args.xyz if args.xyz is not None else args.offset
    if not all(math.isfinite(value) for value in values):
        parser.error('目标坐标必须是有限数值')
    if args.qp_left_hand_index < 0:
        parser.error('qp_left_hand_index 必须大于等于 0')
    for name in ('timeout', 'position_tolerance_mm',
                 'angle_tolerance_deg', 'settle_time', 'max_displacement_mm',
                 'feedback_timeout', 'preflight_timeout', 'disarm_time'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f'{name} 必须是有限正数')
    return args


def topic_name(namespace, name):
    """将控制话题解析到指定命名空间；绝对话题保持原样。"""
    if name.startswith('/'):
        return name
    prefix = namespace.strip('/')
    return '/' + '/'.join(part for part in (prefix, name) if part)


def rotate_vector(vector, quaternion):
    """将工具坐标系向量旋转到 base_link 坐标系。"""
    x, y, z, w = quaternion
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError('TF 四元数无效')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    vx, vy, vz = vector
    # 使用 q * v * q^-1；不修改原始 TF 姿态。
    tx = 2.0 * (y*vz - z*vy)
    ty = 2.0 * (z*vx - x*vz)
    tz = 2.0 * (x*vy - y*vx)
    return (vx + w*tx + y*tz - z*ty,
            vy + w*ty + z*tx - x*tz,
            vz + w*tz + x*ty - y*tx)


def multiply_quaternions(first, second):
    """按 ROS xyzw 顺序计算 first * second。"""
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return (w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2)


def read_tool_mount(model_xml, side):
    """从 QP 所用 Luna MJCF 读取 Arm_L7/R7_Link 到对应 tool 的固定变换。"""
    if side not in ('L', 'R'):
        raise ValueError(f'无效手臂侧别：{side}')
    tool_name = 'left_tool' if side == 'L' else 'right_tool'
    arm_name = f'Arm_{side}7_Link'
    flange_name = f'flange_{side}'
    root = ET.parse(model_xml).getroot()
    arm = root.find(f".//body[@name='{arm_name}']")
    flange = None if arm is None else arm.find(
        f"body[@name='{flange_name}']")
    tool = None if flange is None else flange.find(
        f"body[@name='{tool_name}']")
    if tool is None:
        raise ValueError(
            f'MJCF 中找不到 {arm_name}/{flange_name}/{tool_name}')

    def placement(body):
        position = tuple(float(v) for v in body.get('pos', '0 0 0').split())
        wxyz = tuple(float(v) for v in body.get('quat', '1 0 0 0').split())
        if (len(position) != 3 or len(wxyz) != 4 or
                not all(math.isfinite(v) for v in position + wxyz)):
            raise ValueError('MJCF 工具固定变换无效')
        return position, (wxyz[1], wxyz[2], wxyz[3], wxyz[0])

    flange_position, flange_quaternion = placement(flange)
    tool_position, tool_quaternion = placement(tool)
    rotated = rotate_vector(tool_position, flange_quaternion)
    position = tuple(a+b for a, b in zip(flange_position, rotated))
    quaternion = multiply_quaternions(flange_quaternion, tool_quaternion)
    return position, quaternion


def angular_error_deg(actual, target):
    dot = abs(sum(a*b for a, b in zip(actual, target)))
    norm_a = math.sqrt(sum(value*value for value in actual))
    norm_b = math.sqrt(sum(value*value for value in target))
    if norm_a < 1e-12 or norm_b < 1e-12:
        raise ValueError('TF 四元数无效')
    return math.degrees(2.0 * math.acos(min(1.0, dot/(norm_a*norm_b))))


def transform_pose(stamped):
    p = stamped.transform.translation
    q = stamped.transform.rotation
    position = (p.x, p.y, p.z)
    quaternion = (q.x, q.y, q.z, q.w)
    if not all(math.isfinite(value) for value in position + quaternion):
        raise ValueError('TF 包含非有限数值')
    return position, quaternion


class AccuracyTest(Node):
    def __init__(self, args):
        super().__init__('left_tool_accuracy_test')
        self.args = args
        self.ik_topic = topic_name(args.namespace, args.ik_topic)
        self.qp_pose_topic = topic_name(
            args.namespace, args.qp_pose_topic)
        self.qp_joint_topic = topic_name(
            args.namespace, args.qp_joint_topic)
        self.mux_joint_topic = topic_name(
            args.namespace, args.mux_joint_topic)
        self.joint_feedback_topic = topic_name(
            args.namespace, args.joint_feedback_topic)
        model_xml = args.model_xml
        if model_xml is None:
            model_xml = os.path.join(
                get_package_share_directory('marvin_description'),
                'mjcf', 'Gento_Luna', 'Gento_Luna.xml')
        self.left_mount_position, self.left_mount_quaternion = (
            read_tool_mount(model_xml, 'L'))
        self.right_mount_position, self.right_mount_quaternion = (
            read_tool_mount(model_xml, 'R'))
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.ik_source = None
        self.joint_source = None
        self.start = time.monotonic()
        self.motion_start = None
        self.disarm_start = None
        self.disarm_done = False
        self.settled_since = None
        self.last_log = 0.0
        self.target = None
        self.target_quaternion = None
        self.start_left_position = None
        self.start_left_quaternion = None
        self.start_right_position = None
        self.last_error = None
        self.last_valid_tf = None
        self.qp_left_pose = None
        self.qp_pose_received_at = None
        self.qp_pose_problem = '尚未收到 QP 内部位姿'
        self.qp_pose_problem_logged = False
        self.qp_joint_cmd = None
        self.qp_joint_cmd_received_at = None
        self.mux_joint_cmd = None
        self.mux_joint_cmd_received_at = None
        self.left_joint_feedback = None
        self.left_joint_feedback_received_at = None
        self.exit_code = None

        state_qos = QoSProfile(depth=1)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Int32,
            topic_name(args.namespace, 'info/ik_request_mux/active_source'),
            self.on_ik_source, state_qos)
        self.create_subscription(
            Int32,
            topic_name(args.namespace, 'info/joint_cmd_mux/active_source'),
            self.on_joint_source, state_qos)
        request_qos = QoSProfile(depth=10)
        request_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            PoseArray, self.qp_pose_topic,
            self.on_qp_pose, request_qos)
        self.create_subscription(
            JointcmdArm, self.qp_joint_topic,
            self.on_qp_joint_cmd, request_qos)
        self.create_subscription(
            JointcmdArm, self.mux_joint_topic,
            self.on_mux_joint_cmd, request_qos)
        self.create_subscription(
            Jointfeedback, self.joint_feedback_topic,
            self.on_joint_feedback, request_qos)
        self.publisher = self.create_publisher(
            IKRequest, self.ik_topic, request_qos)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            '等待真实机器人 base_link -> Arm_L7_Link/Arm_R7_Link TF；工具偏移来自 '
            + model_xml)
        self.get_logger().info(f'IK 请求话题：{self.ik_topic}')
        self.get_logger().info(
            f'QP 内部 left_hand：{self.qp_pose_topic} '
            f'poses[{self.args.qp_left_hand_index}]')
        self.get_logger().info(
            '左臂关节诊断：QP=%s, mux=%s, feedback=%s' %
            (self.qp_joint_topic, self.mux_joint_topic,
             self.joint_feedback_topic))

    def on_ik_source(self, message):
        self.ik_source = message.data

    def on_joint_source(self, message):
        self.joint_source = message.data

    def on_qp_pose(self, message):
        """缓存 QP 内部模型根据命令状态计算出的 left_hand 位姿。"""
        index = self.args.qp_left_hand_index
        if len(message.poses) <= index:
            self.qp_pose_problem = (
                f'PoseArray 只有 {len(message.poses)} 项，无法读取索引 {index}')
            if not self.qp_pose_problem_logged:
                self.get_logger().warning(self.qp_pose_problem)
                self.qp_pose_problem_logged = True
            return
        pose = message.poses[index]
        position = (pose.position.x, pose.position.y, pose.position.z)
        quaternion = (pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w)
        if not all(math.isfinite(value)
                   for value in position + quaternion):
            self.qp_pose_problem = 'QP 内部 left_hand 位姿包含非有限数值'
            if not self.qp_pose_problem_logged:
                self.get_logger().warning(self.qp_pose_problem)
                self.qp_pose_problem_logged = True
            return
        quaternion_norm = math.sqrt(sum(value*value for value in quaternion))
        if quaternion_norm < 1e-12:
            self.qp_pose_problem = 'QP 内部 left_hand 四元数无效'
            if not self.qp_pose_problem_logged:
                self.get_logger().warning(self.qp_pose_problem)
                self.qp_pose_problem_logged = True
            return
        self.qp_left_pose = (
            position, quaternion, message.header.frame_id.strip('/'))
        self.qp_pose_received_at = time.monotonic()
        self.qp_pose_problem = None
        self.qp_pose_problem_logged = False

    @staticmethod
    def valid_joint_values(values):
        return len(values) == 7 and all(
            math.isfinite(value) for value in values)

    def on_qp_joint_cmd(self, message):
        values = tuple(float(value) for value in message.positions)
        if self.valid_joint_values(values):
            self.qp_joint_cmd = values
            self.qp_joint_cmd_received_at = time.monotonic()

    def on_mux_joint_cmd(self, message):
        values = tuple(float(value) for value in message.positions)
        if self.valid_joint_values(values):
            self.mux_joint_cmd = values
            self.mux_joint_cmd_received_at = time.monotonic()

    def on_joint_feedback(self, message):
        values = tuple(float(value) for value in message.arm_positions[:7])
        if self.valid_joint_values(values):
            self.left_joint_feedback = values
            self.left_joint_feedback_received_at = time.monotonic()

    def get_qp_left_in_base(self, now):
        """将最新 QP 内部 left_hand 位姿转换到 base_link。"""
        if self.qp_left_pose is None or self.qp_pose_received_at is None:
            raise ValueError(self.qp_pose_problem or '尚未收到 QP 内部位姿')
        age = now - self.qp_pose_received_at
        if age > self.args.feedback_timeout:
            raise ValueError(f'QP 内部位姿已过期：{age:.3f} s')
        position, quaternion, frame = self.qp_left_pose
        if not frame:
            raise ValueError('QP 内部位姿 frame_id 为空')
        if frame == 'base_link':
            return position, quaternion

        stamped = self.buffer.lookup_transform(
            'base_link', frame, Time(),
            timeout=Duration(seconds=0.02))
        frame_position, frame_quaternion = transform_pose(stamped)
        rotated = rotate_vector(position, frame_quaternion)
        base_position = tuple(
            origin+offset for origin, offset in zip(frame_position, rotated))
        base_quaternion = multiply_quaternions(
            frame_quaternion, quaternion)
        return base_position, base_quaternion

    def qp_diagnostic_text(self, actual, quaternion, now):
        """生成 QP 内部模型与实机 TF 的同帧误差诊断。"""
        try:
            qp_position, qp_quaternion = self.get_qp_left_in_base(now)
        except (TransformException, ValueError) as error:
            return f'qp_internal=unavailable({error})'
        qp_target_mm = math.dist(qp_position, self.target) * 1000.0
        qp_target_angle_deg = angular_error_deg(
            qp_quaternion, self.target_quaternion)
        qp_minus_tf_mm = tuple(
            (qp-robot) * 1000.0
            for qp, robot in zip(qp_position, actual))
        qp_tf_distance_mm = math.dist(qp_position, actual) * 1000.0
        qp_tf_angle_deg = angular_error_deg(qp_quaternion, quaternion)
        return (
            'qp_internal_xyz=[%.6f, %.6f, %.6f] '
            'qp_target_error=%.3f mm qp_target_angle=%.3f deg '
            'qp_minus_tf_mm=[%.3f, %.3f, %.3f] '
            'qp_tf_distance=%.3f mm qp_tf_angle=%.3f deg' %
            (*qp_position, qp_target_mm, qp_target_angle_deg,
             *qp_minus_tf_mm, qp_tf_distance_mm, qp_tf_angle_deg))

    def fresh_joint_sample(self, values, received_at, name, now):
        if values is None or received_at is None:
            raise ValueError(f'尚未收到 {name}')
        age = now - received_at
        if age > self.args.feedback_timeout:
            raise ValueError(f'{name} 已过期：{age:.3f} s')
        return values, age

    @staticmethod
    def joint_difference_deg(first, second):
        differences = tuple(
            math.degrees(a-b) for a, b in zip(first, second))
        rms = math.sqrt(
            sum(value*value for value in differences) / len(differences))
        worst_index = max(
            range(len(differences)), key=lambda index: abs(differences[index]))
        return differences, rms, worst_index

    @staticmethod
    def format_joint_vector(values, digits=5):
        return '[' + ', '.join(f'{value:.{digits}f}' for value in values) + ']'

    def joint_diagnostic_lines(self, now):
        """比较 QP输出、mux最终输出和左臂实机关节反馈。"""
        try:
            qp, qp_age = self.fresh_joint_sample(
                self.qp_joint_cmd, self.qp_joint_cmd_received_at,
                'QP左臂命令', now)
            mux, mux_age = self.fresh_joint_sample(
                self.mux_joint_cmd, self.mux_joint_cmd_received_at,
                'mux左臂命令', now)
            feedback, feedback_age = self.fresh_joint_sample(
                self.left_joint_feedback,
                self.left_joint_feedback_received_at,
                '左臂关节反馈', now)
        except ValueError as error:
            return [f'JOINT_L unavailable({error})']

        qp_mux, qp_mux_rms, qp_mux_worst = self.joint_difference_deg(
            qp, mux)
        mux_feedback, mux_feedback_rms, mux_feedback_worst = (
            self.joint_difference_deg(mux, feedback))
        qp_feedback, qp_feedback_rms, qp_feedback_worst = (
            self.joint_difference_deg(qp, feedback))
        values_line = (
            'JOINT_L rad qp=%s mux=%s feedback=%s age_ms=[%.1f, %.1f, %.1f]' %
            (self.format_joint_vector(qp), self.format_joint_vector(mux),
             self.format_joint_vector(feedback), qp_age*1000.0,
             mux_age*1000.0, feedback_age*1000.0))
        difference_line = (
            'JOINT_L diff_deg qp-mux=%s rms=%.3f max=J%d:%.3f | '
            'mux-feedback=%s rms=%.3f max=J%d:%.3f | '
            'qp-feedback=%s rms=%.3f max=J%d:%.3f' %
            (self.format_joint_vector(qp_mux, 3), qp_mux_rms,
             qp_mux_worst+1, qp_mux[qp_mux_worst],
             self.format_joint_vector(mux_feedback, 3), mux_feedback_rms,
             mux_feedback_worst+1, mux_feedback[mux_feedback_worst],
             self.format_joint_vector(qp_feedback, 3), qp_feedback_rms,
             qp_feedback_worst+1, qp_feedback[qp_feedback_worst]))
        return [values_line, difference_line]

    def get_tool_actual(self, link_name, mount_position, mount_quaternion):
        stamped = self.buffer.lookup_transform(
            'base_link', link_name, Time(),
            timeout=Duration(seconds=0.02))
        stamp = stamped.header.stamp.sec + stamped.header.stamp.nanosec * 1e-9
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if stamp <= 0.0 or not -0.05 <= age <= self.args.feedback_timeout:
            raise ValueError(
                f'{link_name} TF 时间戳无效或过旧：{age:.3f} s')
        arm_position, arm_quaternion = transform_pose(stamped)
        rotated = rotate_vector(mount_position, arm_quaternion)
        position = tuple(a+b for a, b in zip(arm_position, rotated))
        quaternion = multiply_quaternions(
            arm_quaternion, mount_quaternion)
        return position, quaternion

    def get_actuals(self):
        left = self.get_tool_actual(
            'Arm_L7_Link', self.left_mount_position,
            self.left_mount_quaternion)
        right = self.get_tool_actual(
            'Arm_R7_Link', self.right_mount_position,
            self.right_mount_quaternion)
        return left, right

    def check_mux(self):
        if self.ik_source != self.args.ik_source:
            return f'IK mux 源为 {self.ik_source}，需要 {self.args.ik_source}'
        if self.joint_source != self.args.joint_source:
            return (f'joint command mux 源为 {self.joint_source}，'
                    f'需要 {self.args.joint_source}')
        if self.publisher.get_subscription_count() == 0:
            return f'{self.ik_topic} 没有订阅者'
        if self.count_publishers(self.ik_topic) > 1:
            return (f'{self.ik_topic} 存在其他发布者；停止 teleop 发布节点后再测试，'
                    '避免两个目标源竞争')
        return None

    def make_target(self, actual, quaternion, right_actual):
        if self.args.xyz is not None:
            goal = tuple(self.args.xyz)
        else:
            offset = tuple(self.args.offset)
            if self.args.offset_frame == 'left_tool':
                offset = rotate_vector(offset, quaternion)
            goal = tuple(a+b for a, b in zip(actual, offset))
        displacement = math.dist(actual, goal)
        if displacement * 1000.0 > self.args.max_displacement_mm + 1e-9:
            raise ValueError(
                f'目标距当前位置 {displacement*1000.0:.3f} mm，超过 '
                f'{self.args.max_displacement_mm:.3f} mm')
        self.target = goal
        self.target_quaternion = quaternion
        self.start_left_position = actual
        self.start_left_quaternion = quaternion
        self.start_right_position = right_actual
        self.get_logger().info(
            'CURRENT base_link->left_tool xyz=[%.6f, %.6f, %.6f]' % actual)
        self.get_logger().info(
            'CURRENT base_link->right_tool xyz=[%.6f, %.6f, %.6f]' %
            right_actual)
        self.get_logger().info(
            'TARGET  base_link->left_tool xyz=[%.6f, %.6f, %.6f], '
            'displacement=%.3f mm' % (*goal, displacement*1000.0))
        if displacement > 0.05:
            self.get_logger().warning(
                '单次目标超过 50 mm；未验证沿途碰撞，执行前须检查整条路径')

    def publish_target(self):
        message = IKRequest()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'base_link'
        message.available = [True] + [False] * 10
        position = message.left_hand_pose.position
        orientation = message.left_hand_pose.orientation
        position.x, position.y, position.z = self.target
        orientation.x, orientation.y, orientation.z, orientation.w = (
            self.target_quaternion)
        self.publisher.publish(message)

    def publish_disarm(self):
        """发送全 unavailable 请求，让 teleop 源完成 go-home 后的重新使能握手。"""
        message = IKRequest()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'base_link'
        message.available = [False] * 11
        self.publisher.publish(message)

    def finish(self, code, reason):
        self.exit_code = code
        if code == 0:
            self.get_logger().info(reason)
        else:
            self.get_logger().error(reason)

    def use_absolute_error_settle(self):
        """是否允许用目标绝对位置/姿态误差结束运动。

        普通精度测试需要该判定；夹爪脚本改用连续窗口内的实际位姿抖动，
        因此会覆盖此方法关闭旧的绝对误差触发路径。
        """
        return True

    def tick(self):
        if self.exit_code is not None:
            return
        now = time.monotonic()
        try:
            (actual, quaternion), (right_actual, _) = self.get_actuals()
        except (TransformException, ValueError) as error:
            if now - self.last_log >= 1.0:
                self.last_log = now
                self.get_logger().warning(f'等待有效 TF：{error}')
            if (self.motion_start is None and
                    now - self.start > self.args.preflight_timeout):
                self.finish(
                    1, '预检超时：未收到有效 Arm_L7_Link/Arm_R7_Link TF')
            elif (self.motion_start is not None and
                  self.last_valid_tf is not None and
                  now - self.last_valid_tf > self.args.feedback_timeout):
                self.finish(1, '运动期间 TF 失效，停止发布目标')
            return
        self.last_valid_tf = now

        if self.target is None:
            if self.args.execute:
                problem = self.check_mux()
                if problem:
                    if now - self.start > self.args.preflight_timeout:
                        self.finish(1, '预检失败：' + problem)
                    return
                if not self.disarm_done:
                    if self.disarm_start is None:
                        self.disarm_start = now
                        self.get_logger().info('发送空请求，准备 teleop 源重新使能')
                    self.publish_disarm()
                    if now - self.disarm_start < self.args.disarm_time:
                        return
                    self.disarm_done = True
            try:
                self.make_target(actual, quaternion, right_actual)
            except ValueError as error:
                self.finish(1, str(error))
                return
            if not self.args.execute:
                self.finish(0, '预览完成；未发布运动指令')
                return
            self.motion_start = now
            self.get_logger().warning('开始持续发送最终目标；请保持急停可用')

        problem = self.check_mux()
        if problem:
            self.finish(1, '控制源变化，停止发布目标：' + problem)
            return
        if now - self.motion_start >= self.args.timeout:
            self.finish(2, 'FAIL: 超时；距目标 %.3f mm' %
                        (math.dist(actual, self.target)*1000.0))
            return
        position_mm = math.dist(actual, self.target) * 1000.0
        self.publish_target()
        angle_deg = angular_error_deg(quaternion, self.target_quaternion)
        left_delta_mm = math.dist(
            actual, self.start_left_position) * 1000.0
        right_delta_mm = math.dist(
            right_actual, self.start_right_position) * 1000.0
        self.last_error = (position_mm, angle_deg)
        if now - self.last_log >= 1.0:
            self.last_log = now
            qp_diagnostic = self.qp_diagnostic_text(
                actual, quaternion, now)
            self.get_logger().info(
                'left_tool_xyz=[%.6f, %.6f, %.6f] '
                'left_delta=%.3f mm target_error=%.3f mm '
                'angle_error=%.3f deg | '
                'right_tool_xyz=[%.6f, %.6f, %.6f] '
                'right_delta=%.3f mm | %s' %
                (*actual, left_delta_mm, position_mm, angle_deg,
                 *right_actual, right_delta_mm, qp_diagnostic))
            for diagnostic_line in self.joint_diagnostic_lines(now):
                self.get_logger().info(diagnostic_line)
        if not self.use_absolute_error_settle():
            return

        within = (position_mm <= self.args.position_tolerance_mm and
                  angle_deg <= self.args.angle_tolerance_deg)
        if within:
            if self.settled_since is None:
                self.settled_since = now
            elif now - self.settled_since >= self.args.settle_time:
                self.finish(0, 'PASS: left_tool 连续到达目标；最终误差 '
                            f'{position_mm:.3f} mm / {angle_deg:.3f} deg')
                return
        else:
            self.settled_since = None


def main():
    args = parse_args()
    rclpy.init()
    node = None
    try:
        node = AccuracyTest(args)
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.exit_code if node.exit_code is not None else 1
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as error:
        print(f'left_tool_accuracy_test: {error}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
