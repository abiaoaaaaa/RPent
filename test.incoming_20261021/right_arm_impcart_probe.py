#!/usr/bin/python3
# 用途：给 ImpCart 右臂一个相对当前工具位姿的小幅位置目标，并逐级采集 ROS/SDK/实际反馈。
# 预览：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && /usr/bin/python3 right_arm_impcart_probe.py
# 实机：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && /usr/bin/python3 right_arm_impcart_probe.py --execute --dz-mm 2
"""右臂笛卡尔阻抗的单次小位移诊断。"""

import argparse
import math
import os
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from marvin_msgs.msg import IKRequest, JointcmdArm, Jointfeedback
from rclpy.qos import QoSProfile, ReliabilityPolicy

from right_arm_teleop import ArmTeleop, topic_name


class Probe(ArmTeleop):
    def __init__(self, args):
        super().__init__(args)
        self.samples = {key: [] for key in (
            'ik', 'qp', 'final', 'sdk_cmd', 'feedback', 'eef')}
        self._probe_subscriptions = []
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        items = (
            ('ik', 'control/ik_request', IKRequest, self.ik_value),
            ('qp', 'control/qp_controller/joint_cmd_B', JointcmdArm,
             lambda msg: list(msg.positions)),
            ('final', 'control/joint_cmd_B', JointcmdArm,
             lambda msg: list(msg.positions)),
            ('sdk_cmd', 'info/cmd_pos_B', JointcmdArm,
             lambda msg: list(msg.positions)),
            ('feedback', 'info/joint_feedback', Jointfeedback,
             lambda msg: list(msg.arm_positions[7:14])),
            ('eef', 'info/eef_right', PoseStamped,
             lambda msg: [msg.pose.position.x, msg.pose.position.y,
                          msg.pose.position.z]),
        )
        for key, suffix, msg_type, extractor in items:
            callback = self.make_collector(key, extractor)
            self._probe_subscriptions.append(self.create_subscription(
                msg_type, topic_name(args.namespace, suffix), callback, qos))

    @staticmethod
    def ik_value(msg):
        p = msg.right_hand_pose.position
        return [p.x, p.y, p.z, float(msg.available[1])]

    def make_collector(self, key, extractor):
        def callback(msg):
            value = extractor(msg)
            with self.lock:
                self.samples[key].append((time.monotonic(), value))
        return callback

    def capture(self, label):
        with self.lock:
            target = self.target_position
            actual = self.actual_position
            active = self.active
            problem = self.last_problem
            state = self.robot_states
            values = {key: rows[-1][1] if rows else None
                      for key, rows in self.samples.items()}
            counts = {key: len(rows) for key, rows in self.samples.items()}
        print(f'\n[{label}] state={state}, active={active}, problem={problem}')
        print(f'  target_xyz={fmt(target)}, tf_actual_xyz={fmt(actual)}')
        if target is not None and actual is not None:
            print(f'  target_minus_actual_mm='
                  f'{fmt([(a-b)*1000 for a, b in zip(target, actual)])}')
        for key in ('ik', 'qp', 'final', 'sdk_cmd', 'feedback', 'eef'):
            print(f'  {key}: count={counts[key]}, last={fmt(values[key])}')
        if values['final'] is not None and values['feedback'] is not None:
            error_deg = [math.degrees(a-b) for a, b in zip(
                values['final'], values['feedback'])]
            print('  final_minus_feedback_deg='
                  f'{fmt(error_deg)}')


def fmt(values):
    if values is None:
        return '无数据'
    return '[' + ', '.join(f'{v:.5f}' for v in values) + ']'


def wait_for(predicate, seconds, reason):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise RuntimeError(reason)


def safe_hold(node, duration, baseline, max_motion_mm):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        with node.lock:
            actual = node.actual_position
            active = node.active
            reason = node.last_problem
            states = node.robot_states
        if not active or len(states) < 2 or states[1] != 3:
            raise RuntimeError(f'控制状态变化：active={active}, state={states}, {reason}')
        if actual is None or math.dist(actual, baseline)*1000 > max_motion_mm:
            raise RuntimeError('实测末端偏离基准超过安全阈值，停止测试')
        time.sleep(0.05)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='启用实机发送；默认只预检')
    parser.add_argument('--dx-mm', type=float, default=0.0)
    parser.add_argument('--dy-mm', type=float, default=0.0)
    parser.add_argument('--dz-mm', type=float, default=2.0)
    parser.add_argument('--hold-s', type=float, default=3.0)
    parser.add_argument('--namespace', default=os.environ.get('APEX_ROS_NAMESPACE', 'tj'))
    args = parser.parse_args()
    offset = (args.dx_mm, args.dy_mm, args.dz_mm)
    if any(not math.isfinite(v) for v in offset) or math.dist(offset, (0, 0, 0)) > 3.0:
        parser.error('总位移必须不超过 3 mm')
    if math.dist(offset, (0, 0, 0)) < 0.1:
        parser.error('总位移至少 0.1 mm')
    if not math.isfinite(args.hold_s) or not 1.0 <= args.hold_s <= 5.0:
        parser.error('--hold-s 必须在 1～5 秒之间')
    # 复用现有 CLI 的就绪检查、TF 工具位姿换算、mux 选择和释放逻辑。
    args.side = 'right'
    args.model_xml = None
    args.head_joint_count = 2
    args.gripper_topic = None
    args.feedback_timeout = 0.5
    args.rate_hz = 30.0
    args.joint_source = 0
    args.ik_source = 1
    args.max_position_offset_mm = 3.0
    args.max_angle_offset_deg = 0.0
    return args


def main():
    args = parse_args()
    rclpy.init(args=[])
    node = Probe(args)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        wait_for(lambda: node.readiness_problem() is None, 6.0,
                 '预检超时：' + str(node.readiness_problem()))
        with node.lock:
            state = node.robot_states
        if state[1] != 3:
            raise RuntimeError(f'右臂需要先进入 ImpCart 状态 3，当前状态={state}')
        node.reset_target()
        with node.lock:
            baseline = node.actual_position
        node.capture('预检')
        print(f'相对目标 mm={fmt([args.dx_mm, args.dy_mm, args.dz_mm])}')
        if not args.execute:
            print('预览结束；加 --execute 才会发送目标。')
            return 0
        # 确保没有另一个交互式 CLI 同时使用 replay IK 源。
        peers = [e for e in node.get_publishers_info_by_topic(node.request_topic)
                 if e.node_name == 'right_arm_cartesian_teleop']
        if len(peers) > 1:
            raise RuntimeError('检测到另一个 right_arm_cartesian_teleop，请先退出该 CLI')
        node.enable()
        wait_for(lambda: node.active, 3.0, 'IK mux 启用超时')
        safe_hold(node, 0.5, baseline, 10.0)
        with node.lock:
            node.target_position = tuple(
                p + d/1000.0 for p, d in zip(
                    baseline, (args.dx_mm, args.dy_mm, args.dz_mm)))
        safe_hold(node, args.hold_s, baseline, 10.0)
        node.capture('偏移目标')
        node.reset_target()
        with node.lock:
            node.target_position = baseline
        safe_hold(node, 2.0, baseline, 10.0)
        node.capture('返回基准')
        if not node.samples['sdk_cmd']:
            print('SDK 目标回读缺失：需在驱动启动参数中启用 publish_cmd_pos_debug。')
        return 0
    except (KeyboardInterrupt, RuntimeError, ValueError) as error:
        print(f'测试中止：{error}')
        return 1
    finally:
        if node.active or node.switch_pending:
            node.disable()
            time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    raise SystemExit(main())
