#!/usr/bin/env python3
# 用途：只读记录 Luna tool 位姿及同一时刻的关节反馈。
# 运行：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && python3 read_tool_pose.py --help
"""只读记录 Luna tool 位姿和同一时刻的机器人关节反馈。

程序读取具名 ``JointState``，使用 Apex_Deploy 中 QP 控制器所用 MJCF 做正运动学，
在同一份关节测量上计算 tool 位姿。可将 tool 位姿、关节位置/速度/effort 保存为
YAML。程序不会发布任何运动或控制消息。
"""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation

from luna_handeye_calibration.kinematics import Kinematics


SOURCE_WORKSPACE = Path('/home/liuqian/WBC/Apex_Deploy')
ARM_JOINT_NAMES = {
    'left': [f'Arm_L{index}_Joint' for index in range(1, 8)],
    'right': [f'Arm_R{index}_Joint' for index in range(1, 8)],
}


def default_model_path():
    """优先使用当前 ROS 环境中的模型，未安装时回退到源码工作区。"""
    try:
        share = Path(get_package_share_directory('marvin_description'))
        installed = share / 'mjcf' / 'Gento_Luna' / 'Gento_Luna.xml'
        if installed.is_file():
            return str(installed)
    except PackageNotFoundError:
        pass
    return str(
        SOURCE_WORKSPACE / 'robot_node' / 'marvin_description' / 'mjcf' /
        'Gento_Luna' / 'Gento_Luna.xml')


def validate_positive(parser, args, names):
    for name in names:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f'{name.replace("_", "-")} 必须为有限正数')


def topic_name(namespace, name):
    """将相对话题解析到命名空间，绝对话题保持不变。"""
    if name.startswith('/'):
        return name
    prefix = namespace.strip('/')
    return '/' + '/'.join(part for part in (prefix, name) if part)


def message_stamp_seconds(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def finite_sequence(values, expected_size, label):
    result = [float(value) for value in values]
    if len(result) != expected_size:
        raise ValueError(f'{label}长度为 {len(result)}，需要 {expected_size}')
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f'{label}包含非有限数值')
    return result


def write_snapshot(path, payload):
    """使用独占创建避免覆盖已有的机器人姿态记录。"""
    with path.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-frame', default='base_link')
    parser.add_argument(
        '--namespace', default='/tj',
        help='机器人 ROS 命名空间，默认 /tj')
    parser.add_argument(
        '--joint-state-topic', default='joint_states',
        help='具名 JointState 话题；相对名称会放入 --namespace')
    parser.add_argument(
        '--side', choices=('left', 'right'), default='left',
        help='读取左/右 tool，默认 left')
    parser.add_argument(
        '--model-xml', default=default_model_path(),
        help='QP 控制器使用的 Luna MJCF；默认从 ROS 安装空间或 Apex_Deploy 源码读取')
    parser.add_argument(
        '--rate-hz', type=float, default=0.0,
        help='输出频率；默认 0 表示成功读取一次后退出')
    parser.add_argument(
        '--wait-timeout', type=float, default=5.0,
        help='等待首个有效关节状态的最长时间，单位 s')
    parser.add_argument(
        '--max-age', type=float, default=0.5,
        help='允许的关节状态最大时延，单位 s')
    parser.add_argument(
        '--matrix', action='store_true',
        help='额外输出 4x4 的 T_base_tool 矩阵')
    parser.add_argument(
        '--output',
        help='将单次 tool 和关节快照保存为 YAML；拒绝覆盖已有文件')
    args = parser.parse_args(argv)

    validate_positive(parser, args, ('wait_timeout', 'max_age'))
    if not math.isfinite(args.rate_hz) or args.rate_hz < 0.0:
        parser.error('rate-hz 必须是有限非负数')
    if not args.base_frame.strip('/'):
        parser.error('base-frame 不能为空')
    if not args.joint_state_topic.strip('/'):
        parser.error('joint-state-topic 不能为空')
    if args.output and args.rate_hz > 0.0:
        parser.error('--output 只用于单次快照，不能与正数 --rate-hz 同时使用')
    model = Path(args.model_xml).expanduser().resolve()
    if not model.is_file():
        parser.error(f'MJCF 文件不存在：{model}')
    args.model_xml = str(model)
    args.base_frame = args.base_frame.strip('/')
    args.joint_state_topic = topic_name(
        args.namespace, args.joint_state_topic)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            parser.error(f'输出文件已存在，拒绝覆盖：{output}')
        if not output.parent.is_dir():
            parser.error(f'输出目录不存在：{output.parent}')
        args.output = output
    return args


class ToolPoseReader(Node):
    def __init__(self, args):
        super().__init__('read_tool_pose')
        self.args = args
        if args.side == 'left':
            self.mount_frame = 'Arm_L7_Link'
            self.tool_frame = 'left_tool'
        else:
            self.mount_frame = 'Arm_R7_Link'
            self.tool_frame = 'right_tool'

        self.kinematics = Kinematics(
            args.model_xml, args.base_frame, self.mount_frame)
        self.model_sha256 = self.kinematics.sha256
        self.mount_tool = self.kinematics.fixed_relative_pose(
            self.mount_frame, self.tool_frame)
        self.started_at = time.monotonic()
        self.last_warning_at = 0.0
        self.exit_code = None
        self.joint_state = None

        # 查询计时器保持较快响应；连续模式的实际输出频率单独限流。
        self.output_period = None if args.rate_hz == 0.0 else 1.0 / args.rate_hz
        self.last_output_at = None
        self.create_subscription(
            JointState, args.joint_state_topic,
            self.on_joint_state, qos_profile_sensor_data)
        self.create_timer(0.02, self.tick)
        self.get_logger().info(
            f'等待 {args.joint_state_topic}；'
            f'将输出 {args.base_frame} -> {self.tool_frame}')
        self.get_logger().info(f'工具固定变换读取自：{args.model_xml}')

    def on_joint_state(self, message):
        self.joint_state = message

    def warn_throttled(self, message):
        now = time.monotonic()
        if now - self.last_warning_at >= 1.0:
            self.get_logger().warning(message)
            self.last_warning_at = now

    def tick(self):
        if self.exit_code is not None:
            return
        now = time.monotonic()
        if now - self.started_at > self.args.wait_timeout and self.last_output_at is None:
            self.get_logger().error(
                f'等待有效关节状态超时：{self.args.joint_state_topic}')
            self.exit_code = 1
            return
        if (self.output_period is not None and self.last_output_at is not None and
                now - self.last_output_at < self.output_period):
            return

        try:
            if self.joint_state is None:
                raise ValueError(f'尚未收到 {self.args.joint_state_topic}')
            joint_stamp = message_stamp_seconds(self.joint_state)
            ros_now = self.get_clock().now().nanoseconds * 1e-9
            joint_age = ros_now - joint_stamp
            if joint_stamp <= 0.0 or not -0.05 <= joint_age <= self.args.max_age:
                raise ValueError(f'关节状态时间戳无效或过旧：age={joint_age:.3f}s')
            joints = self.validate_joint_state(self.joint_state)
            base_mount = self.kinematics.pose(
                dict(zip(joints['names'], joints['positions'])))
            base_tool = base_mount @ self.mount_tool
            snapshot = self.make_snapshot(base_tool, joint_age, joints)
            self.output_snapshot(snapshot)
            if self.args.output:
                write_snapshot(self.args.output, snapshot)
                self.get_logger().info(f'快照已保存：{self.args.output}')
            self.last_output_at = now
            if self.output_period is None:
                self.exit_code = 0
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.warn_throttled(str(error))

    @staticmethod
    def validate_joint_state(message):
        names = list(message.name)
        if not names or len(set(names)) != len(names):
            raise ValueError('JointState 关节名称为空或重复')
        positions = finite_sequence(message.position, len(names), 'JointState.position')
        velocities = finite_sequence(message.velocity, len(names), 'JointState.velocity')
        efforts = finite_sequence(message.effort, len(names), 'JointState.effort')
        return {
            'names': names,
            'positions': positions,
            'velocities': velocities,
            'efforts': efforts,
        }

    def make_snapshot(self, transform, joint_age, joints):
        message = self.joint_state
        index_by_name = {
            name: index for index, name in enumerate(joints['names'])}
        missing = [name for name in ARM_JOINT_NAMES[self.args.side]
                   if name not in index_by_name]
        if missing:
            raise ValueError(f'JointState 缺少所选手臂关节：{missing}')
        side_indices = [index_by_name[name]
                        for name in ARM_JOINT_NAMES[self.args.side]]
        arm_positions = [joints['positions'][index] for index in side_indices]
        arm_velocities = [joints['velocities'][index] for index in side_indices]
        arm_efforts = [joints['efforts'][index] for index in side_indices]
        xyz = transform[:3, 3]
        rotation = Rotation.from_matrix(transform[:3, :3])
        quaternion = rotation.as_quat()
        rpy_deg = rotation.as_euler('xyz', degrees=True)
        stamp = message.header.stamp
        stamp_seconds = message_stamp_seconds(message)
        recorded_at = datetime.fromtimestamp(
            stamp_seconds, tz=timezone.utc).astimezone().isoformat()
        return {
            'schema_version': 1,
            'record_type': 'luna_tool_joint_snapshot',
            'recorded_at': recorded_at,
            'ros_time': {
                'sec': int(stamp.sec),
                'nanosec': int(stamp.nanosec),
            },
            'source': {
                'joint_state_topic': self.args.joint_state_topic,
                'model_xml': self.args.model_xml,
                'model_sha256': self.model_sha256,
            },
            'tool_pose': {
                'base_frame': self.args.base_frame,
                'tool_frame': self.tool_frame,
                'xyz_m': xyz.tolist(),
                'quaternion_xyzw': quaternion.tolist(),
                'rpy_deg': rpy_deg.tolist(),
                'T_base_tool': transform.tolist(),
            },
            'selected_arm': {
                'side': self.args.side,
                'joint_names': ARM_JOINT_NAMES[self.args.side],
                'position_rad': arm_positions,
                'position_deg': np.degrees(arm_positions).tolist(),
                'velocity_rad_s': arm_velocities,
                'velocity_deg_s': np.degrees(arm_velocities).tolist(),
                'effort': arm_efforts,
            },
            # 保存完整反馈，因为躯干姿态也会影响 tool 在 base_link 下的位姿。
            'full_joint_state': {
                'names': joints['names'],
                'position_si': joints['positions'],
                'velocity_si': joints['velocities'],
                'effort': joints['efforts'],
            },
            'freshness': {
                'joint_state_age_s': float(joint_age),
            },
        }

    def output_snapshot(self, snapshot):
        tool = snapshot['tool_pose']
        joints = snapshot['selected_arm']
        xyz = tool['xyz_m']
        quaternion = tool['quaternion_xyzw']
        rpy_deg = tool['rpy_deg']
        self.get_logger().info(
            f'{self.args.base_frame} -> {self.tool_frame}\n'
            f'  xyz_m: [{xyz[0]:.9f}, {xyz[1]:.9f}, {xyz[2]:.9f}]\n'
            f'  quaternion_xyzw: [{quaternion[0]:.9f}, {quaternion[1]:.9f}, '
            f'{quaternion[2]:.9f}, {quaternion[3]:.9f}]\n'
            f'  rpy_deg: [{rpy_deg[0]:.6f}, {rpy_deg[1]:.6f}, {rpy_deg[2]:.6f}]\n'
            f'  joint_state_age_s: '
            f'{snapshot["freshness"]["joint_state_age_s"]:.6f}')
        joint_lines = [
            f'  {name}: {rad:.9f} rad ({deg:.6f} deg), '
            f'velocity={velocity:.9f} rad/s, effort={effort:.9f}'
            for name, rad, deg, velocity, effort in zip(
                joints['joint_names'], joints['position_rad'],
                joints['position_deg'], joints['velocity_rad_s'],
                joints['effort'])
        ]
        self.get_logger().info(
            f'{self.args.side} arm joints\n' + '\n'.join(joint_lines))
        if self.args.matrix:
            rows = ['    [' + ', '.join(f'{value:.9f}' for value in row) + ']'
                    for row in tool['T_base_tool']]
            self.get_logger().info('T_base_tool:\n' + '\n'.join(rows))


def main(argv=None):
    args = parse_args(argv)
    rclpy.init(args=[])
    node = None
    try:
        node = ToolPoseReader(args)
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.exit_code if node.exit_code is not None else 1
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'read_tool_pose: {error}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
