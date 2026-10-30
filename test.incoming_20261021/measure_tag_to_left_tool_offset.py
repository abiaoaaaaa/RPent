#!/usr/bin/env python3
# 用途：测量 AprilTag 到 Luna left_tool 的固定六自由度变换。
# 运行：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && python3 measure_tag_to_left_tool_offset.py --tag-id 5 --tag-size 0.016
"""测量 AprilTag 到 Luna left_tool 的固定六自由度变换。"""

import argparse
from collections import deque
import math
from pathlib import Path
import sys
import time

import cv2
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
from tf2_ros import Buffer, TransformException, TransformListener

from luna_handeye_calibration.apriltag import AprilTagDetector
from luna_handeye_calibration.kinematics import Kinematics
from luna_handeye_calibration.se3 import (
    atomic_yaml,
    inverse,
    make_transform,
    transform_from_dict,
    validate_transform,
)
from luna_handeye_calibration.track_left_apriltag import transform_message_to_matrix


WORKSPACE = Path('/home/liuqian/WBC/Apex_Deploy')
DEFAULT_OUTPUT = Path('/home/liuqian/copy/test/tag5_left_tool_offset.yaml')


def rotation_distance_deg(first, second):
    """返回两个旋转矩阵之间的最短夹角。"""
    relative = np.asarray(first).T @ np.asarray(second)
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def average_transforms(transforms):
    """分别平均平移与 SO(3) 旋转，避免直接平均四元数符号。"""
    transforms = [validate_transform(value) for value in transforms]
    return make_transform(
        Rotation.from_matrix(np.stack([value[:3, :3] for value in transforms])).mean().as_matrix(),
        np.mean([value[:3, 3] for value in transforms], axis=0),
    )


def read_yaml(path, label):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f'{label}文件不存在：{resolved}')
    with resolved.open(encoding='utf-8') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'{label}文件格式错误：{resolved}')
    return value, resolved


class OffsetMeasurement(Node):
    def __init__(self, args):
        super().__init__('measure_tag_to_left_tool_offset')
        self.args = args
        calibration, self.calibration_path = read_yaml(args.calibration_file, '外参')
        if calibration.get('schema_version') != 1 or calibration.get('validation_passed') is not True:
            raise ValueError('只允许使用已通过验证的当前格式外参')
        if calibration.get('base_frame') != 'base_link' or calibration.get('mount_frame') != 'Arm_L7_Link':
            raise ValueError('外参必须是 base_link / Arm_L7_Link 左腕相机标定')

        self.base_frame = calibration['base_frame']
        self.mount_frame = calibration['mount_frame']
        self.mount_camera = transform_from_dict(calibration['T_mount_camera'])
        self.kinematics = Kinematics(
            calibration['model_path'], self.base_frame, self.mount_frame,
            calibration.get('joint_aliases', {}))
        # if self.kinematics.sha256 != calibration['model_sha256']:
        #     raise ValueError('当前模型与外参验证时的模型哈希不一致')
        self.mount_tool = self.kinematics.fixed_relative_pose(self.mount_frame, args.tool_frame)

        intrinsics, self.intrinsics_path = read_yaml(args.intrinsics_file, '内参')
        self.image_size = (int(intrinsics['image_width']), int(intrinsics['image_height']))
        self.image_frame = intrinsics['optical_frame']
        self.distortion_model = intrinsics['distortion_model']
        self.K = np.asarray(intrinsics['camera_matrix']['data'], dtype=float).reshape(3, 3)
        self.D = np.asarray(intrinsics['distortion_coefficients']['data'], dtype=float)
        if not np.isfinite(self.K).all() or not np.isfinite(self.D).all():
            raise ValueError('相机内参包含非法数值')

        self.detector = AprilTagDetector(
            args.tag_id, args.tag_size, args.max_reprojection_px, args.ambiguity_ratio)
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.images = deque(maxlen=30)
        self.processed_stamp = None
        self.samples = []
        self.reprojection_errors = []
        self.last_base_tag = None
        self.last_base_tool = None
        self.result = None
        self.exit_code = None
        self.start_time = time.monotonic()
        self.last_warning = ''

        self.create_subscription(Image, args.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            f'只读测量：AprilTag ID={args.tag_id}，边长={args.tag_size * 1000:.1f}mm，'
            f'目标 frame={args.tool_frame}')
        self.get_logger().info(
            '请先用遥操作把工具保持在理想夹持姿态，并确保 Tag 在左腕相机中清晰可见；'
            '本程序不会发布任何运动指令')

    def on_image(self, message):
        self.images.append((time.monotonic(), message))

    def warn(self, text):
        self.last_warning = text
        self.get_logger().warning(text, throttle_duration_sec=1.0)

    def reset_window(self, transform, reprojection, base_tag, base_tool, reason):
        self.samples = [transform]
        self.reprojection_errors = [reprojection]
        self.last_base_tag = base_tag
        self.last_base_tool = base_tool
        self.warn(f'{reason}；重新开始稳定采样 1/{self.args.samples}')

    def tick(self):
        if self.exit_code is not None:
            return
        if time.monotonic() - self.start_time > self.args.timeout:
            self.exit_code = 1
            self.get_logger().error(
                f'测量超时：稳定样本 {len(self.samples)}/{self.args.samples}；'
                f'最后问题：{self.last_warning or "未收到有效图像/TF"}')
            return

        # 单线程执行器中给 TF 回调留出时间，再处理稍早的最新图像。
        ready = [message for arrival, message in self.images
                 if time.monotonic() - arrival >= 0.1]
        if not ready:
            return
        message = ready[-1]
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if stamp == self.processed_stamp:
            return
        self.processed_stamp = stamp

        try:
            ros_stamp = stamp[0] + stamp[1] * 1e-9
            age = self.get_clock().now().nanoseconds * 1e-9 - ros_stamp
            if stamp == (0, 0) or not -0.05 <= age <= self.args.image_timeout:
                raise ValueError(f'图像时间戳无效或过旧：{age:.3f}s')
            if (message.width, message.height) != self.image_size:
                raise ValueError(
                    f'图像分辨率 {(message.width, message.height)} 与内参 {self.image_size} 不一致')
            if message.header.frame_id != self.image_frame:
                raise ValueError(
                    f'图像 frame={message.header.frame_id!r}，需要 {self.image_frame!r}')

            mount_message = self.tf_buffer.lookup_transform(
                self.base_frame, self.mount_frame, Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=self.args.tf_timeout))
            base_mount = transform_message_to_matrix(mount_message.transform)
            gray = self.bridge.imgmsg_to_cv2(message, desired_encoding='mono8')
            camera_tag, reprojection, _ = self.detector.detect(
                gray, self.K, self.D, self.distortion_model)

            # 两条路径使用同一曝光时刻。公共的 base→mount 在相对变换中抵消，
            # 但仍显式计算，便于输出基座系位置供人工复核。
            base_tag = base_mount @ self.mount_camera @ camera_tag
            base_tool = base_mount @ self.mount_tool
            tag_tool = inverse(base_tag) @ base_tool

            if self.samples:
                mean = average_transforms(self.samples)
                position_error_mm = float(np.linalg.norm(
                    tag_tool[:3, 3] - mean[:3, 3]) * 1000.0)
                angle_error_deg = rotation_distance_deg(mean[:3, :3], tag_tool[:3, :3])
                if (position_error_mm > self.args.position_tolerance_mm or
                        angle_error_deg > self.args.angle_tolerance_deg):
                    self.reset_window(
                        tag_tool, reprojection, base_tag, base_tool,
                        f'Tag→tool 观测跳变 {position_error_mm:.2f}mm/{angle_error_deg:.2f}°')
                    return

            self.samples.append(tag_tool)
            self.reprojection_errors.append(reprojection)
            self.last_base_tag = base_tag
            self.last_base_tool = base_tool
            self.get_logger().info(
                f'稳定采样 {len(self.samples)}/{self.args.samples}，'
                f'重投影误差={reprojection:.3f}px', throttle_duration_sec=1.0)
            if len(self.samples) >= self.args.samples:
                self.finish()
        except (ValueError, TransformException, cv2.error) as error:
            self.samples = []
            self.reprojection_errors = []
            self.warn(str(error))

    def finish(self):
        result = average_transforms(self.samples)
        position_errors = np.asarray([
            np.linalg.norm(value[:3, 3] - result[:3, 3]) * 1000.0
            for value in self.samples])
        angle_errors = np.asarray([
            rotation_distance_deg(result[:3, :3], value[:3, :3])
            for value in self.samples])
        rpy_deg = Rotation.from_matrix(result[:3, :3]).as_euler('xyz', degrees=True)
        quaternion = Rotation.from_matrix(result[:3, :3]).as_quat()

        # 与消费端相同的欧拉角构造方式做一次回代检查。
        reconstructed = make_transform(
            Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix(), result[:3, 3])
        if (np.linalg.norm(reconstructed[:3, 3] - result[:3, 3]) > 1e-12 or
                rotation_distance_deg(reconstructed[:3, :3], result[:3, :3]) > 1e-8):
            raise ValueError('RPY 回代检查失败')

        quality = {
            'position_rms_mm': float(np.sqrt(np.mean(position_errors ** 2))),
            'position_max_mm': float(np.max(position_errors)),
            'angle_rms_deg': float(np.sqrt(np.mean(angle_errors ** 2))),
            'angle_max_deg': float(np.max(angle_errors)),
            'reprojection_mean_px': float(np.mean(self.reprojection_errors)),
            'reprojection_max_px': float(np.max(self.reprojection_errors)),
        }
        parameters = {
            'tool_frame': self.args.tool_frame,
            'tag_id': self.args.tag_id,
            'tag_size': self.args.tag_size,
            'tag_eef_offset_m': result[:3, 3].tolist(),
            'tag_eef_rpy_deg': rpy_deg.tolist(),
        }
        payload = {
            'schema_version': 1,
            'measurement': 'T_tag_left_tool',
            'calibration_file': str(self.calibration_path),
            'intrinsics_file': str(self.intrinsics_path),
            'samples': len(self.samples),
            'quality': quality,
            # 消费端当前使用 RPY；四元数单独保存，避免作为未知 ROS 参数注入节点。
            'tag_eef_quaternion_xyzw': quaternion.tolist(),
            'measurement_snapshot': {
                'base_tag_xyz_m': self.last_base_tag[:3, 3].tolist(),
                'base_left_tool_xyz_m': self.last_base_tool[:3, 3].tolist(),
            },
            '/**': {'ros__parameters': parameters},
        }
        atomic_yaml(self.args.output, payload)
        self.result = result
        self.exit_code = 0

        self.get_logger().info(
            'MEASURED base_link→Tag xyz=[%.6f, %.6f, %.6f]' %
            tuple(self.last_base_tag[:3, 3]))
        self.get_logger().info(
            'MEASURED base_link→left_tool xyz=[%.6f, %.6f, %.6f]' %
            tuple(self.last_base_tool[:3, 3]))
        self.get_logger().info(
            'RESULT Tag→left_tool offset(m)=[%.9f, %.9f, %.9f]' %
            tuple(result[:3, 3]))
        self.get_logger().info(
            'RESULT Tag→left_tool rpy(deg)=[%.6f, %.6f, %.6f]' % tuple(rpy_deg))
        self.get_logger().info(
            'RESULT Tag→left_tool quaternion(xyzw)=[%.9f, %.9f, %.9f, %.9f]' %
            tuple(quaternion))
        self.get_logger().info(
            'QUALITY position rms/max=%.3f/%.3fmm, angle rms/max=%.3f/%.3f°，'
            'reprojection mean/max=%.3f/%.3fpx' %
            (quality['position_rms_mm'], quality['position_max_mm'],
             quality['angle_rms_deg'], quality['angle_max_deg'],
             quality['reprojection_mean_px'], quality['reprojection_max_px']))
        self.get_logger().info(f'已保存可直接使用的偏移配置：{self.args.output}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag-id', type=int, default=5)
    parser.add_argument('--tag-size', type=float, default=0.016,
                        help='Tag 最外侧黑色方框边长，单位 m；默认 0.016')
    parser.add_argument('--tool-frame', default='left_tool')
    parser.add_argument('--image-topic', default='/wrist_left_camera/camera/color/image_raw')
    parser.add_argument(
        '--calibration-file',
        default=str(WORKSPACE / 'calibration_data/luna_left_extrinsic02_home_20261009_verified.yaml'))
    parser.add_argument(
        '--intrinsics-file',
        default=str(WORKSPACE / 'calibration_data/left_d405_intrinsics.yaml'))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--position-tolerance-mm', type=float, default=3.0)
    parser.add_argument('--angle-tolerance-deg', type=float, default=3.0)
    parser.add_argument('--max-reprojection-px', type=float, default=1.5)
    parser.add_argument('--ambiguity-ratio', type=float, default=1.5)
    parser.add_argument('--image-timeout', type=float, default=0.5)
    parser.add_argument('--tf-timeout', type=float, default=0.05)
    args = parser.parse_args()

    if args.tag_id < 0:
        parser.error('tag-id 必须非负')
    if not 2 <= args.samples <= 300:
        parser.error('samples 必须在 2 到 300 之间')
    for name in (
            'tag_size', 'timeout', 'position_tolerance_mm', 'angle_tolerance_deg',
            'max_reprojection_px', 'ambiguity_ratio', 'image_timeout', 'tf_timeout'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name} 必须为有限正数')
    if args.ambiguity_ratio <= 1.0:
        parser.error('ambiguity-ratio 必须大于 1')
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f'输出文件已存在，拒绝覆盖：{output}；请改用新的 --output 文件名')
    args.output = str(output)
    return args


def main():
    args = parse_args()
    rclpy.init(args=[])
    node = None
    try:
        node = OffsetMeasurement(args)
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.exit_code if node.exit_code is not None else 1
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'measure_tag_to_left_tool_offset: {error}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
