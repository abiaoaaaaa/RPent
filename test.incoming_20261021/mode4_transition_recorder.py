#!/usr/bin/python3
# 用途：只读记录双臂 mode 3→4 切换前后的实测状态、QP/最终目标、SDK 指令反馈和模式状态。
# 运行：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && \
#       /usr/bin/python3 /home/liuqian/copy/test/mode4_transition_recorder.py --duration 60
# 先启动记录，再由现有操作界面切换模式；本脚本不会调用服务或发送机器人指令。
"""将双臂模式切换相关 ROS 话题记录为带时间戳的 JSONL。"""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from marvin_msgs.msg import JointcmdArm, Jointfeedback
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import JointState
from std_msgs.msg import Int16MultiArray, String


def finite_number(value):
    """保留非有限数值的含义，同时保证每行都是合法 JSON。"""
    number = float(value)
    if math.isnan(number):
        return 'NaN'
    if math.isinf(number):
        return 'Infinity' if number > 0 else '-Infinity'
    return number


def numbers(values):
    return [finite_number(value) for value in values]


def source_stamp_ns(message):
    header = getattr(message, 'header', None)
    if header is None:
        return None
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def topic_name(namespace, suffix):
    return '/' + '/'.join(
        part for part in (namespace.strip('/'), suffix) if part)


class TransitionRecorder(Node):
    def __init__(self, namespace, output_path):
        super().__init__(f'mode4_transition_recorder_{os.getpid()}')
        self.output_path = output_path
        self.output = output_path.open(
            'x', encoding='utf-8', buffering=1024 * 1024)
        self.counts = Counter()
        self._topic_subscriptions = []
        self.last_flush = time.monotonic()
        self.record_namespace = namespace.strip('/')
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        sources = (
            ('info/joint_feedback', Jointfeedback, self.arm_feedback),
            ('joint_states', JointState, self.joint_states),
            ('control/qp_controller/joint_cmd_A', JointcmdArm,
             self.arm_command),
            ('control/qp_controller/joint_cmd_B', JointcmdArm,
             self.arm_command),
            ('control/joint_cmd_A', JointcmdArm, self.arm_command),
            ('control/joint_cmd_B', JointcmdArm, self.arm_command),
            ('info/cmd_pos_A', JointcmdArm, self.arm_command),
            ('info/cmd_pos_B', JointcmdArm, self.arm_command),
            ('info/robot_state', Int16MultiArray, self.mode_state),
            ('infor/robot_state', Int16MultiArray, self.mode_state),
            ('info/eef_left', PoseStamped, self.eef_pose),
            ('info/eef_right', PoseStamped, self.eef_pose),
            ('info/mode4_transition_diag', String, self.mode4_diagnostic),
        )
        for suffix, message_type, converter in sources:
            name = topic_name(self.record_namespace, suffix)
            callback = self.make_callback(name, converter)
            self._topic_subscriptions.append(
                self.create_subscription(message_type, name, callback, qos))
            self.counts[name] = 0

        self.write({
            'kind': 'metadata',
            'wall_time_ns': time.time_ns(),
            'namespace': self.record_namespace,
            'topics': [
                topic_name(self.record_namespace, item[0]) for item in sources],
            'units': 'joint rad/rad_s, end-effector position m',
            'note': ('info/cmd_pos_A/B 需启用 publish_cmd_pos_debug；'
                     'mode4_transition_diag 含 SDK 关节指令反馈和 SG 设置反馈 K/D，'
                     '不包含 SDK 未公开的笛卡尔平衡点。'),
        })

    @staticmethod
    def arm_feedback(message):
        return {
            'left_position': numbers(message.arm_positions[:7]),
            'right_position': numbers(message.arm_positions[7:14]),
            'left_velocity': numbers(message.arm_velocities[:7]),
            'right_velocity': numbers(message.arm_velocities[7:14]),
            'left_effort': numbers(message.arm_efforts[:7]),
            'right_effort': numbers(message.arm_efforts[7:14]),
        }

    @staticmethod
    def joint_states(message):
        return {
            'name': list(message.name[:14]),
            'position': numbers(message.position[:14]),
            'velocity': numbers(message.velocity[:14]),
        }

    @staticmethod
    def arm_command(message):
        return {'position': numbers(message.positions)}

    @staticmethod
    def mode_state(message):
        return {'data': [int(value) for value in message.data]}

    @staticmethod
    def eef_pose(message):
        point = message.pose.position
        orientation = message.pose.orientation
        return {
            'frame_id': message.header.frame_id,
            'position': numbers((point.x, point.y, point.z)),
            'orientation_xyzw': numbers((
                orientation.x, orientation.y, orientation.z, orientation.w)),
        }

    @staticmethod
    def mode4_diagnostic(message):
        """保留节点发出的 SDK 调用事件和内部反馈快照。"""
        return json.loads(message.data)

    def make_callback(self, name, converter):
        def callback(message):
            self.write({
                'kind': 'sample',
                'topic': name,
                'wall_time_ns': time.time_ns(),
                'monotonic_ns': time.monotonic_ns(),
                'source_stamp_ns': source_stamp_ns(message),
                'data': converter(message),
            })
            self.counts[name] += 1
            now = time.monotonic()
            if now - self.last_flush >= 0.5:
                self.output.flush()
                self.last_flush = now
        return callback

    def write(self, record):
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.output.write(line + '\n')

    def close(self):
        self.output.flush()
        self.output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--namespace', default=os.environ.get('APEX_ROS_NAMESPACE', 'tj'))
    parser.add_argument('--duration', type=float, default=0.0,
                        help='记录秒数；0 表示运行到 Ctrl-C')
    parser.add_argument('--output', type=Path,
                        help='输出 JSONL 路径；默认保存在脚本所在目录')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('--duration 必须是非负有限秒数')
    output_path = args.output or (
        Path(__file__).resolve().parent /
        f"mode4_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    output_path = output_path.expanduser().resolve()
    if not output_path.parent.is_dir():
        parser.error(f'输出目录不存在：{output_path.parent}')

    rclpy.init()
    recorder = None
    try:
        recorder = TransitionRecorder(args.namespace, output_path)
        print(f'正在记录：{output_path}', flush=True)
        print('请先保持 mode 3，随后用现有界面切换到 mode 4；按 Ctrl-C 结束。', flush=True)
        deadline = time.monotonic() + args.duration if args.duration else None
        while rclpy.ok() and (deadline is None or time.monotonic() < deadline):
            rclpy.spin_once(recorder, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.close()
            print(f'已保存：{recorder.output_path}', flush=True)
            for name, count in recorder.counts.items():
                print(f'{name}: {count}', flush=True)
            recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
