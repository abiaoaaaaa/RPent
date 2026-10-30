#!/usr/bin/python3
# 用途：通过 AprilTag 生成并验证 left_tool 目标位姿。
# 运行：source /home/xlb/data/code/code_new/Luna/Apex_Deploy/apex_ros_env.sh && python3 apriltag_left_tool_accuracy_test.py --help
"""识别 ID 3，冻结标定后的 left_tool 目标，并复用 QP 精度测试。"""
import argparse
from collections import deque
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
from tf2_ros import Buffer, TransformListener, TransformException
from luna_handeye_calibration.apriltag import AprilTagDetector
from luna_handeye_calibration.se3 import transform_from_dict
from luna_handeye_calibration.track_left_apriltag import (
    StableTagCapture, tag_eef_target, transform_message_to_matrix,
)

# 与原测试放在同一目录，保留它的左右手、QP 和关节误差诊断。
import left_tool_accuracy_test as accuracy

WORKSPACE = Path(os.environ.get(
    'LUNA_APEX_WORKSPACE',
    Path(__file__).resolve().parents[3] / 'Luna' / 'Apex_Deploy'))
DEFAULT_OFFSET_CONFIG = Path(__file__).with_name('tag5_left_tool_offset.yaml')



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

def read_yaml(path):
    with Path(path).expanduser().open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


class TagCapture(Node):
    def __init__(self, args):
        super().__init__('apriltag_accuracy_capture')
        self.args = args
        calibration = read_yaml(args.calibration_file)
        if calibration.get('validation_passed') is not True or calibration.get('schema_version') != 1:
            raise ValueError('外参未通过验证')
        if calibration['base_frame'] != 'base_link' or calibration['mount_frame'] != 'Arm_L7_Link':
            raise ValueError('需要 base_link / Arm_L7_Link 的左腕标定')
        self.mount_camera = transform_from_dict(calibration['T_mount_camera'])
        config = read_yaml(args.offset_config)['/**']['ros__parameters']
        if config.get('tool_frame') != 'left_tool':
            raise ValueError('偏移配置必须针对 left_tool')
        if config.get('tag_id') != args.tag_id:
            raise ValueError(
                f'偏移配置 tag_id={config.get("tag_id")} 与检测 ID={args.tag_id} 不一致')
        self.offset = config['tag_eef_offset_m']
        self.rpy = config['tag_eef_rpy_deg']
        for value in (self.offset, self.rpy):
            if np.asarray(value).shape != (3,) or not np.isfinite(value).all():
                raise ValueError('Tag→left_tool 偏移和欧拉角必须是三个有限数值')
        tag_eef_target(np.eye(4), self.offset, self.rpy)
        intr = read_yaml(args.intrinsics_file)
        self.size = (intr['image_width'], intr['image_height'])
        self.frame = intr['optical_frame']
        self.K = np.asarray(intr['camera_matrix']['data']).reshape(3, 3)
        self.D = np.asarray(intr['distortion_coefficients']['data'])
        if not np.isfinite(self.K).all() or not np.isfinite(self.D).all():
            raise ValueError('内参包含非有限数值')
        self.distortion = intr['distortion_model']
        self.detector = AprilTagDetector(args.tag_id, args.tag_size, 1.5, 1.5)
        self.capture = StableTagCapture(10, 0.003, 2.0)
        self.bridge = CvBridge()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.images = deque(maxlen=30)
        self.processed = None
        self.goal = None
        self.create_subscription(Image, args.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'检测 ID={args.tag_id}，边长={args.tag_size}m；10帧稳定后冻结目标')
        if args.tag_position_only:
            self.get_logger().info('位置测试模式：目标直接使用 Tag 中心，left_tool 保持当前姿态')
        else:
            self.get_logger().info(f'Tag→left_tool 偏移(m)={self.offset}，rpy(deg)={self.rpy}')

    def on_image(self, msg):
        self.images.append((time.monotonic(), msg))

    def tick(self):
        if self.goal is not None:
            return
        # 单线程回调中阻塞等待 TF 会妨碍 TF 自身接收；延迟处理图像让 TF 先到齐。
        ready = [msg for arrival, msg in self.images if time.monotonic() - arrival >= 0.1]
        if not ready:
            return
        msg = ready[-1]
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self.processed:
            return
        try:
            age = self.get_clock().now().nanoseconds / 1e9 - (stamp[0] + stamp[1] / 1e9)
            if stamp == (0, 0) or not -0.05 <= age <= 0.5:
                raise ValueError('图像时间戳过旧或无效')
            if (msg.width, msg.height) != self.size or msg.header.frame_id != self.frame:
                raise ValueError('图像分辨率或 optical frame 与内参不匹配')
            # 使用曝光时刻的实机安装连杆 TF，避免移动相机的新旧位姿混用。
            mount = self.tf.lookup_transform('base_link', 'Arm_L7_Link', Time.from_msg(msg.header.stamp),
                                             timeout=Duration(seconds=0.02))
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            camera_tag, reprojection, _ = self.detector.detect(gray, self.K, self.D, self.distortion)
            self.processed = stamp
            base_tag = transform_message_to_matrix(mount.transform) @ self.mount_camera @ camera_tag
            frozen = self.capture.add(base_tag, stamp)
            self.get_logger().info(f'稳定采样 {len(self.capture.samples)}/10，重投影误差={reprojection:.3f}px',
                                   throttle_duration_sec=1.0)
            if frozen is not None:
                self.goal = (frozen.copy() if self.args.tag_position_only else
                             tag_eef_target(frozen, self.offset, self.rpy))
                self.get_logger().info(f'ID {self.args.tag_id} base_link xyz={frozen[:3, 3].tolist()}')
                if self.args.tag_position_only:
                    self.get_logger().info(
                        f'冻结 left_tool 位置={self.goal[:3, 3].tolist()}；姿态将在执行前取当前值')
                else:
                    self.get_logger().info(f'冻结 left_tool xyz={self.goal[:3, 3].tolist()}')
        except (ValueError, TransformException) as exc:
            self.capture.reset()
            self.get_logger().warning(str(exc), throttle_duration_sec=1.0)


class TagAccuracy(accuracy.AccuracyTest):
    def __init__(self, args, goal):
        self.position_only = args.tag_position_only
        self.goal_quaternion = (None if self.position_only else
                                tuple(Rotation.from_matrix(goal[:3, :3]).as_quat()))
        self.max_angle = args.max_target_angle_deg
        self.robot_ready = False
        self.state_time = 0.0
        self.setpoint_validated = False
        super().__init__(args)
        self.create_subscription(Int16MultiArray, topic_name(args.namespace, 'info/robot_state'),
                                 self.on_state, 10)

    def on_state(self, msg):
        self.robot_ready = len(msg.data) >= 4 and all(msg.data[i] in (1, 2) for i in (0, 1, 3))
        self.state_time = time.monotonic()

    def check_mux(self):
        problem = super().check_mux()
        if problem:
            return problem
        if not self.robot_ready or time.monotonic() - self.state_time > self.args.feedback_timeout:
            return '等待新鲜的左右臂和躯干可运动状态'
        # 在原测试发送重新使能空请求之前，检查冻结目标距离和转角。
        if not self.setpoint_validated:
            try:
                (p, q), _ = self.get_actuals()
                self.validate_goal(p, q)
                self.setpoint_validated = True
            except (ValueError, TransformException) as exc:
                return str(exc)
        return None

    def validate_goal(self, position, quaternion):
        angle = (0.0 if self.position_only else
                 angular_error_deg(quaternion, self.goal_quaternion))
        distance = math.dist(position, self.args.xyz) * 1000
        if distance > self.args.max_displacement_mm:
            raise ValueError(f'冻结目标距离 {distance:.2f}mm / 转角 {angle:.2f}°，'
                             f'超过 --max-displacement-mm={self.args.max_displacement_mm}')
        if angle > self.max_angle:
            raise ValueError(f'目标转角 {angle:.2f}° 超过 --max-target-angle-deg={self.max_angle}')

    def make_target(self, actual, quaternion, right_actual):
        if self.position_only:
            self.goal_quaternion = quaternion
        self.validate_goal(actual, quaternion)
        super().make_target(actual, quaternion, right_actual)
        self.target_quaternion = self.goal_quaternion
        source = '保持当前姿态' if self.position_only else '使用标定姿态'
        self.get_logger().info(f'TARGET quaternion xyzw={self.target_quaternion}；{source}')
        if self.use_absolute_error_settle():
            self.get_logger().info(
                f'PASS 判定：位置误差≤{self.args.position_tolerance_mm:.3f}mm，'
                f'姿态误差≤{self.args.angle_tolerance_deg:.3f}°，'
                f'连续满足 {self.args.settle_time:.3f}s')
        else:
            self.get_logger().info(
                'PASS 判定：由上层脚本根据连续观察窗口内的 tool 位姿抖动决定')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, epilog='其余选项与 left_tool_accuracy_test.py 相同，例如 --execute --timeout 30 --max-displacement-mm 50')
    parser.add_argument('--tag-id', type=int, default=5)
    parser.add_argument('--tag-size', type=float, default=0.016)
    parser.add_argument('--image-topic', default='/wrist_left_camera/camera/color/image_raw')
    parser.add_argument('--calibration-file', default=str(WORKSPACE / 'calibration_data/luna_left_extrinsic02_home_20261009_verified.yaml'))
    parser.add_argument('--intrinsics-file', default=str(WORKSPACE / 'calibration_data/left_d405_intrinsics.yaml'))
    parser.add_argument('--offset-config', default=str(DEFAULT_OFFSET_CONFIG))
    parser.add_argument('--capture-timeout', type=float, default=30.0)
    parser.add_argument('--max-target-angle-deg', type=float, default=70.0)
    parser.add_argument(
        '--position-tolerance-mm', type=float, default=1.0,
        help='到位位置阈值，单位 mm，默认恢复为 1.0；只影响 PASS 判定')
    parser.add_argument(
        '--angle-tolerance-deg', type=float, default=0.5,
        help='到位姿态阈值，单位 deg，默认恢复为 0.5；只影响 PASS 判定')
    parser.add_argument(
        '--tag-position-only', action='store_true',
        help='直接使用 Tag 中心位置，不使用 Tag→left_tool 偏移，并保持当前姿态')
    args, rest = parser.parse_known_args()
    for key in (
            'tag_size', 'capture_timeout', 'max_target_angle_deg',
            'position_tolerance_mm', 'angle_tolerance_deg'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} 必须为有限正数')
    if any(x.split('=')[0] in ('--xyz', '--offset', '--offset-frame') for x in rest):
        parser.error('目标由 AprilTag 和标定偏移生成，不能指定 --xyz/--offset')
    original = sys.argv
    try:
        sys.argv = [original[0]] + rest
        motion = accuracy.parse_args()
    finally:
        sys.argv = original
    motion.max_target_angle_deg = args.max_target_angle_deg
    motion.tag_position_only = args.tag_position_only
    motion.position_tolerance_mm = args.position_tolerance_mm
    motion.angle_tolerance_deg = args.angle_tolerance_deg
    return args, motion


def main():
    args, motion = parse_args()
    rclpy.init(args=[])
    node = None
    try:
        node = TagCapture(args)
        start = time.monotonic()
        while rclpy.ok() and node.goal is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - start > args.capture_timeout:
                raise ValueError('采集超时：未得到 ID 3 的稳定观测')
        if node.goal is None:
            return 1
        goal = node.goal.copy()
        node.destroy_node()
        node = None
        motion.xyz = tuple(goal[:3, 3])
        node = TagAccuracy(motion, goal)
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.exit_code if node.exit_code is not None else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'apriltag_left_tool_accuracy_test: {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
