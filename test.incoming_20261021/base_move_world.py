#!/usr/bin/python3
# 用途：校验连续世界位姿反馈后，通过 VLAHost 安全移动底盘 x/y/yaw。
# 运行：source /home/xlb/data/code/code_new/Luna/Apex_Deploy/apex_ros_env.sh
#   python3 base_move_world.py --help
"""独立执行移动底盘世界坐标移动。

程序从 VLAHost 读取 ``camera_init -> body`` 位姿，确认里程计持续更新后，
通过 ``/base_move_world`` 分段调整 x/y，再用短动作块闭环修正 yaw。
定位中断时先停止，显式允许时等待新鲜位姿恢复后有限重试；
yaw 反馈背离命令时立即停止。z 只能检查，底盘不能主动调整。
"""

import argparse
import json
import math
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_VLAHOST_URL = 'http://127.0.0.1:8011'
DEFAULT_TARGET_X_M = 0.04681
DEFAULT_TARGET_Y_M = 0.22712
DEFAULT_TARGET_Z_M = 0.00603
DEFAULT_TARGET_YAW_RAD = 0.05024
MIN_LATERAL_SPEED_M_S = 0.06
MAX_LATERAL_SPEED_M_S = 0.10
MAX_YAW_SPEED_RAD_S = 0.20
MIN_WORLD_MOVE_M = 0.05
# 仅用于兼容当前 VLAHost 服务端的固定校验，不作为客户端单次请求限制。
SERVER_MAX_WORLD_MOVE_M = 0.50
# 分段请求避开服务端 0.50 m 浮点校验边界。
SEGMENT_MAX_WORLD_MOVE_M = SERVER_MAX_WORLD_MOVE_M - 0.01
POSE_MAX_AGE_S = 0.5
YAW_OPPOSITE_STEP_RAD = 0.015
YAW_DIVERGENCE_RAD = 0.05
YAW_MAX_FEEDBACK_STEP_RAD = 0.3
DEFAULT_YAW_MAX_TRAVEL_RAD = math.pi + 0.35


class PoseNotReadyError(ValueError):
    """位姿源尚未提供可用于移动的实时数据。"""


class OdometryInterruptedError(RuntimeError):
    """服务端已因里程计不可用而结束本次平移。"""


def request_json(base_url, path, body=None, timeout=3.0):
    """调用 VLAHost HTTP/JSON 接口，并将通信错误转换为可读异常。"""
    if body is None:
        request = Request(base_url.rstrip('/') + path, method='GET')
    else:
        data = json.dumps(body).encode('utf-8')
        request = Request(
            base_url.rstrip('/') + path,
            data=data,
            headers={'Content-Type': 'application/json'},
        )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        raise ValueError(f'VLAHost HTTP {error.code}: {detail}') from error
    except (URLError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f'无法访问 VLAHost：{error}') from error


def wrapped_angle(angle):
    """将角度误差归一化到 [-pi, pi]。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_control_goal(current_yaw, target_yaw, reference_yaw=None):
    """返回连续角空间中的当前角和目标角。"""
    if reference_yaw is None:
        return current_yaw, current_yaw + wrapped_angle(target_yaw - current_yaw)
    return (
        reference_yaw + wrapped_angle(current_yaw - reference_yaw),
        reference_yaw + wrapped_angle(target_yaw - reference_yaw),
    )


def pose_values_to_target(values):
    """将 xyz 和四元数的七个数值转换为底盘可执行的 xyz/yaw 目标。"""
    if len(values) != 7:
        raise ValueError('位姿需要 7 个数值：x y z qx qy qz qw')
    if not all(math.isfinite(value) for value in values):
        raise ValueError('目标位姿不能包含非有限数值')
    x, y, z, qx, qy, qz, qw = values
    norm = math.hypot(qx, qy, qz, qw)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('目标四元数模长无效')
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return x, y, z, yaw


def validate_base_world_pose(pose):
    """校验 VLAHost 返回的世界底盘位姿，并返回浮点数值。"""
    if not isinstance(pose, dict):
        raise ValueError('世界底盘位姿响应结构无效')
    if not pose.get('valid'):
        raise PoseNotReadyError('世界底盘位姿无效')
    if pose.get('frame_id') != 'camera_init' or pose.get('child_frame_id') != 'body':
        raise ValueError(
            '世界底盘位姿坐标系异常，需要 camera_init -> body，实际为 '
            f'{pose.get("frame_id")} -> {pose.get("child_frame_id")}')
    try:
        received_age = float(pose['received_age_s'])
        stamp = float(pose['stamp_s'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('世界底盘位姿时间戳无效') from error
    if (not math.isfinite(stamp) or not math.isfinite(received_age) or
            received_age < 0.0):
        raise ValueError('世界底盘位姿时间戳无效')
    if received_age > POSE_MAX_AGE_S:
        raise PoseNotReadyError(f'世界底盘位姿过期：{received_age:.3f} s')
    position = pose.get('position')
    if not isinstance(position, dict):
        raise ValueError('世界底盘位置字段无效')
    try:
        x, y, z = (float(position[axis]) for axis in ('x', 'y', 'z'))
        yaw = float(pose['yaw_rad'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('世界底盘位置或 yaw 字段无效') from error
    if not all(math.isfinite(value) for value in (x, y, z, yaw)):
        raise ValueError('世界底盘位置或 yaw 包含非有限值')
    return x, y, z, yaw


def parse_args():
    """解析并校验命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose_values', nargs='*', type=float,
                        metavar='pose_value',
                        help='目标位姿的 7 个数值：x y z qx qy qz qw')
    parser.add_argument('--check-only', action='store_true',
                        help='只检查位姿源是否持续更新，不发送底盘命令')
    parser.add_argument('--vlahost-url', default=DEFAULT_VLAHOST_URL,
                        help=f'VLAHost HTTP 地址，默认 {DEFAULT_VLAHOST_URL}')
    parser.add_argument('--target-x-m', type=float, default=DEFAULT_TARGET_X_M,
                        help=f'目标世界坐标 x，默认 {DEFAULT_TARGET_X_M}')
    parser.add_argument('--target-y-m', type=float, default=DEFAULT_TARGET_Y_M,
                        help=f'目标世界坐标 y，默认 {DEFAULT_TARGET_Y_M}')
    parser.add_argument('--target-z-m', type=float, default=DEFAULT_TARGET_Z_M,
                        help=f'目标世界坐标 z（只检查），默认 {DEFAULT_TARGET_Z_M}')
    parser.add_argument('--target-yaw-rad', type=float,
                        default=DEFAULT_TARGET_YAW_RAD,
                        help=f'目标世界 yaw，默认 {DEFAULT_TARGET_YAW_RAD} rad')
    parser.add_argument('--speed-m-s', type=float, default=MIN_LATERAL_SPEED_M_S,
                        help='x/y 平移速度，范围 0.06..0.10 m/s')
    parser.add_argument('--yaw-speed-rad-s', type=float, default=0.20,
                        help='yaw 闭环速度，范围 0..0.20 rad/s，默认 0.20')
    parser.add_argument('--yaw-command-hz', type=float, default=10.0,
                        help='yaw 动作块发布频率，范围 10..200 Hz，默认 10 Hz')
    parser.add_argument('--yaw-command-duration-s', type=float, default=0.2,
                        help='每次 yaw 动作块持续时间，默认 0.2 s')
    parser.add_argument('--position-tolerance-m', type=float, default=0.05,
                        help='x/y 及 z 位置容限，默认 0.05 m')
    parser.add_argument('--yaw-tolerance-rad', type=float, default=0.10,
                        help='yaw 容限，默认 0.10 rad（约 5.7°）')
    parser.add_argument('--move-timeout-s', type=float, default=75.0,
                        help='x/y 平移最长等待时间，默认 75 s')
    parser.add_argument('--yaw-timeout-s', type=float, default=30.0,
                        help='yaw 修正最长等待时间，默认 30 s')
    parser.add_argument('--yaw-reference-rad', type=float,
                        help='电缆零位 yaw；设置后目标朝向优先回卷到该零位附近')
    parser.add_argument('--yaw-max-travel-rad', type=float,
                        default=DEFAULT_YAW_MAX_TRAVEL_RAD,
                        help='单个目标允许的最大实际转角，默认约 200°')
    parser.add_argument('--http-timeout-s', type=float, default=3.0,
                        help='单次 HTTP 请求超时，默认 3 s')
    parser.add_argument('--pose-ready-timeout-s', type=float, default=10.0,
                        help='等待实时世界位姿的最长时间，默认 10 s')
    parser.add_argument('--pose-stable-s', type=float, default=2.0,
                        help='移动前要求里程计连续更新的时间，默认 2 s')
    parser.add_argument('--odometry-retries', type=int, default=0,
                        help='平移里程计中断后的最大自动恢复次数，默认 0')
    args = parser.parse_args()
    if args.pose_values:
        try:
            (args.target_x_m, args.target_y_m, args.target_z_m,
             args.target_yaw_rad) = pose_values_to_target(args.pose_values)
        except ValueError as error:
            parser.error(str(error))

    finite_positive = (
        'target_x_m', 'target_y_m', 'target_z_m', 'target_yaw_rad',
        'position_tolerance_m', 'yaw_tolerance_rad', 'move_timeout_s',
        'yaw_timeout_s', 'http_timeout_s', 'yaw_command_hz',
        'yaw_command_duration_s', 'yaw_max_travel_rad',
        'pose_ready_timeout_s', 'pose_stable_s')
    for name in finite_positive:
        value = getattr(args, name)
        if not math.isfinite(value):
            parser.error(f'--{name.replace("_", "-")} 必须为有限数值')
    if not MIN_LATERAL_SPEED_M_S <= args.speed_m_s <= MAX_LATERAL_SPEED_M_S:
        parser.error('--speed-m-s 必须在 0.06..0.10 m/s 内')
    if not math.isfinite(args.speed_m_s):
        parser.error('--speed-m-s 必须为有限数值')
    if not 0.0 < args.yaw_speed_rad_s <= MAX_YAW_SPEED_RAD_S:
        parser.error('--yaw-speed-rad-s 必须在 0..0.20 rad/s 内')
    if (args.yaw_reference_rad is not None and
            not math.isfinite(args.yaw_reference_rad)):
        parser.error('--yaw-reference-rad 必须为有限数值')
    if not 0.0 < args.yaw_max_travel_rad <= DEFAULT_YAW_MAX_TRAVEL_RAD:
        parser.error('--yaw-max-travel-rad 必须在 0..3.49 rad 内')
    if not 10.0 <= args.yaw_command_hz <= 200.0:
        parser.error('--yaw-command-hz 必须在 10..200 Hz 内')
    if args.position_tolerance_m <= 0.0 or args.yaw_tolerance_rad <= 0.0:
        parser.error('位置和 yaw 容限必须为正数')
    if (args.move_timeout_s <= 0.0 or args.yaw_timeout_s <= 0.0 or
            args.http_timeout_s <= 0.0 or
            not 0.1 <= args.yaw_command_duration_s <= 0.3):
        parser.error('超时时间必须为正数，yaw 动作持续时间必须在 0.1..0.3 s 内')
    if args.pose_ready_timeout_s < 0.0:
        parser.error('--pose-ready-timeout-s 必须大于等于 0')
    if args.pose_stable_s <= 0.0 or args.pose_stable_s >= args.pose_ready_timeout_s:
        parser.error('--pose-stable-s 必须大于 0 且小于 --pose-ready-timeout-s')
    if args.odometry_retries < 0:
        parser.error('--odometry-retries 必须大于等于 0')
    return args


class BaseWorldMover:
    """执行底盘世界坐标 x/y 平移和 yaw 闭环。"""

    def __init__(self, args):
        self.args = args
        self.command_attempted = False

    def get_pose(self):
        response = request_json(
            self.args.vlahost_url, '/base_world_pose',
            timeout=self.args.http_timeout_s)
        if not isinstance(response, dict):
            raise ValueError('VLAHost 世界位姿响应不是 JSON 对象')
        pose = response.get('pose')
        validate_base_world_pose(pose)
        return pose

    def check_only(self):
        """只读确认里程计连续更新，并显示当前世界位姿。"""
        pose = self.wait_for_fresh_pose()
        x, y, z, yaw = validate_base_world_pose(pose)
        print(
            '世界底盘位姿持续更新：'
            f'x={x:.5f} y={y:.5f} z={z:.5f} yaw={yaw:.5f} rad，'
            f'源时间戳={float(pose["stamp_s"]):.3f}')

    def wait_for_fresh_pose(self, timeout_s=None, stable_s=None):
        """等待连续的新里程计时间戳，避免只凭一帧恢复就启动底盘。"""
        if timeout_s is None:
            timeout_s = self.args.pose_ready_timeout_s
        if stable_s is None:
            stable_s = self.args.pose_stable_s
        deadline = time.monotonic() + timeout_s
        warned = False
        stable_since = None
        last_stamp = None
        updates = 0
        last_status = '尚未收到实时世界底盘位姿'
        while True:
            try:
                pose = self.get_pose()
                now = time.monotonic()
                stamp = float(pose['stamp_s'])
                if stable_since is None:
                    stable_since = now
                    last_stamp = stamp
                    updates = 1
                elif stamp < last_stamp:
                    raise RuntimeError('世界底盘位姿源时间戳倒退，停止移动')
                elif stamp > last_stamp:
                    last_stamp = stamp
                    updates += 1
                last_status = f'位姿新鲜，最近源时间戳 {last_stamp:.3f}'
                if (now - stable_since >= stable_s and
                        updates >= max(2, math.ceil(stable_s * 2.0))):
                    return pose
                remaining = deadline - now
                if remaining <= 0.0:
                    raise RuntimeError(
                        f'等待世界底盘位姿持续更新超时（收到 '
                        f'{updates} 个不同时间戳；最后状态：{last_status}）')
                time.sleep(min(0.1, remaining))
            except PoseNotReadyError as error:
                stable_since = None
                last_stamp = None
                updates = 0
                last_status = str(error)
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        f'等待实时世界底盘位姿恢复超时（最后状态：'
                        f'{error}）；请检查 camera_init -> body 位姿输入') from error
                if not warned:
                    print(f'等待世界底盘位姿恢复：{error}')
                    warned = True
                time.sleep(min(0.1, remaining))

    def cancel_translation(self):
        """取消仍在执行的 /base_move_world 动作。"""
        try:
            request_json(
                self.args.vlahost_url, '/base_move_world/cancel', {},
                timeout=self.args.http_timeout_s)
        except ValueError as error:
            print(f'发送底盘平移停止命令失败：{error}', file=sys.stderr)

    def send_yaw_velocity(self, velocity):
        """通过动作块持续发布 yaw 速度，避免单帧命令立即归零。"""
        if abs(velocity) > 1e-12:
            self.command_attempted = True
        action_count = max(
            1, math.floor(self.args.yaw_command_duration_s *
                          self.args.yaw_command_hz))
        action = {'base_velocity': [0.0, 0.0, float(velocity)]}
        return request_json(
            self.args.vlahost_url, '/action_chunk',
            {'actions': [action] * action_count,
             'source_hz': self.args.yaw_command_hz},
            timeout=self.args.http_timeout_s)

    def stop_yaw(self):
        try:
            result = self.send_yaw_velocity(0.0)
        except ValueError as error:
            print(f'发送底盘 yaw 零速度失败：{error}', file=sys.stderr)
            return False
        if (not isinstance(result, dict) or not result.get('success') or
                result.get('execution') not in (None, 'active')):
            print(f'底盘 yaw 零速度命令未执行：{result}', file=sys.stderr)
            return False
        return True

    @staticmethod
    def is_server_world_length_error(error):
        """判断是否为 VLAHost 的单次世界位移长度校验错误。"""
        return 'world offset length must be ' in str(error)

    def send_and_wait_xy(self, offset_x, offset_y, current_x, current_y,
                         label=''):
        """发送一次世界 x/y 位移并等待服务端报告完成。"""
        distance = math.hypot(offset_x, offset_y)
        self.command_attempted = True
        result = request_json(
            self.args.vlahost_url, '/base_move_world',
            {'offset_xy_m': [offset_x, offset_y],
             'speed_mps': self.args.speed_m_s},
            timeout=self.args.http_timeout_s)
        if not isinstance(result, dict):
            raise RuntimeError('底盘世界 x/y 平移响应不是 JSON 对象')
        move = result.get('move', {})
        if not result.get('success') or move.get('status') != 'active':
            message = result.get('message', '')
            if self.is_server_world_length_error(message):
                raise ValueError(message)
            raise RuntimeError(f'底盘世界 x/y 平移未启动：{result}')
        suffix = f'（{label}）' if label else ''
        print(
            f'开始底盘世界 x/y 平移{suffix}：'
            f'current=[{current_x:.5f}, {current_y:.5f}] m，'
            f'offset=[{offset_x:.3f}, {offset_y:.3f}] m，'
            f'length={distance:.3f} m，speed={self.args.speed_m_s:.3f} m/s')

        deadline = time.monotonic() + self.args.move_timeout_s
        while time.monotonic() < deadline:
            state = request_json(
                self.args.vlahost_url, '/base_move_world',
                timeout=self.args.http_timeout_s)
            if not isinstance(state, dict):
                raise RuntimeError('底盘世界 x/y 状态响应不是 JSON 对象')
            move = state.get('move', {})
            status = move.get('status')
            if status == 'reached':
                print(f'底盘世界 x/y 平移{suffix}完成。')
                return
            if status in ('failed', 'cancelled'):
                reason = move.get('reason', '未知原因')
                if (status == 'failed' and
                        reason == 'odometry missing, stale or invalid'):
                    raise OdometryInterruptedError(reason)
                raise RuntimeError(
                    f'底盘世界 x/y 平移{suffix}{status}：'
                    f'{reason}')
            time.sleep(0.2)
        raise TimeoutError(f'等待底盘世界 x/y 平移{suffix}完成超时')

    def move_xy_segmented(self):
        """仅在服务端明确拒绝大位移时按服务端上限分段。"""
        print(
            'VLAHost 当前强制要求单次世界位移不超过 0.50 m，'
            '自动切换为分段平移。')
        segment = 0
        previous_distance = None
        while True:
            pose = self.wait_for_fresh_pose()
            current_x, current_y, _, _ = validate_base_world_pose(pose)
            remaining_x = self.args.target_x_m - current_x
            remaining_y = self.args.target_y_m - current_y
            distance = math.hypot(remaining_x, remaining_y)
            if distance <= self.args.position_tolerance_m:
                print('底盘 x/y 分段平移完成。')
                return
            if distance < MIN_WORLD_MOVE_M:
                raise RuntimeError(
                    f'底盘世界 x/y 剩余误差 {distance:.3f} m 小于接口最小单次位移 '
                    f'{MIN_WORLD_MOVE_M:.2f} m，但超出位置容限')
            if (segment >= 20 or
                    (previous_distance is not None and
                     distance >= previous_distance - 0.01)):
                raise RuntimeError('底盘 x/y 分段平移没有持续接近目标，已停止移动')
            scale = min(1.0, SEGMENT_MAX_WORLD_MOVE_M / distance)
            segment += 1
            previous_distance = distance
            self.send_and_wait_xy(
                remaining_x * scale, remaining_y * scale,
                current_x, current_y, f'第 {segment} 段')

    def move_xy(self):
        """仅在显式允许重试时，等待定位稳定后重新计算绝对目标。"""
        for attempt in range(self.args.odometry_retries + 1):
            try:
                return self.move_xy_once()
            except OdometryInterruptedError:
                if attempt >= self.args.odometry_retries:
                    raise RuntimeError(
                        '平移因里程计中断而停止；未自动继续，请检查定位数据源')
                request_json(
                    self.args.vlahost_url, '/base_move_world/cancel', {},
                    timeout=self.args.http_timeout_s)
                state = request_json(
                    self.args.vlahost_url, '/base_move_world',
                    timeout=self.args.http_timeout_s)
                move = state.get('move') if isinstance(state, dict) else None
                if (not isinstance(move, dict) or move.get('status') not in
                        ('failed', 'cancelled', 'reached')):
                    raise RuntimeError('平移停止状态未确认，无法恢复移动')
                if not self.stop_yaw():
                    raise RuntimeError('底盘零速度命令未确认，无法恢复移动')
                print(
                    f'平移里程计中断，已停止；等待定位持续更新后恢复 '
                    f'{attempt + 1}/{self.args.odometry_retries}。')
                self.wait_for_fresh_pose()

    def move_xy_once(self):
        """一次调用接口移动到底盘世界目标 x/y。"""
        pose = self.wait_for_fresh_pose()
        current_x, current_y, _, _ = validate_base_world_pose(pose)
        offset_x = self.args.target_x_m - current_x
        offset_y = self.args.target_y_m - current_y
        distance = math.hypot(offset_x, offset_y)
        if distance <= self.args.position_tolerance_m:
            print('底盘 x/y 已在目标容限内，跳过平移。')
            return
        if distance < MIN_WORLD_MOVE_M:
            raise RuntimeError(
                f'底盘世界 x/y 误差 {distance:.3f} m 小于接口最小单次位移 '
                f'{MIN_WORLD_MOVE_M:.2f} m，但超出位置容限')
        try:
            self.send_and_wait_xy(offset_x, offset_y, current_x, current_y)
        except (ValueError, RuntimeError) as error:
            if not self.is_server_world_length_error(error):
                raise
            self.move_xy_segmented()

    def wait_for_next_pose(self, previous_stamp, deadline):
        """每次 yaw 命令后必须看到新的里程计帧。"""
        while time.monotonic() < deadline:
            pose = self.get_pose()
            stamp = float(pose['stamp_s'])
            if stamp < previous_stamp:
                raise RuntimeError('yaw 期间世界底盘位姿时间戳倒退')
            if stamp > previous_stamp:
                return pose
            time.sleep(0.05)
        raise OdometryInterruptedError('yaw 命令后 0.5 s 内没有新的世界底盘位姿')

    def move_yaw(self):
        """用短动作块和逐帧反馈修正 yaw；反馈背离目标时停止。"""
        pose = self.wait_for_fresh_pose()
        _, _, _, yaw = validate_base_world_pose(pose)
        yaw_reference = getattr(self.args, 'yaw_reference_rad', None)
        yaw_unwrapped, target_unwrapped = yaw_control_goal(
            yaw, self.args.target_yaw_rad,
            yaw_reference)
        yaw_error = target_unwrapped - yaw_unwrapped
        if abs(yaw_error) <= self.args.yaw_tolerance_rad:
            print('底盘 yaw 已在目标容限内，跳过修正。')
            return

        deadline = time.monotonic() + self.args.yaw_timeout_s
        last_progress_at = time.monotonic()
        yaw_travel = 0.0
        recoveries = 0
        print(
            f'开始底盘世界 yaw 闭环：current={yaw:.5f} rad，'
            f'target={self.args.target_yaw_rad:.5f} rad，'
            f'error={yaw_error:.5f} rad')
        while time.monotonic() < deadline:
            speed = math.copysign(
                min(self.args.yaw_speed_rad_s, max(0.06, abs(yaw_error))),
                yaw_error)
            previous_yaw = yaw
            previous_error = yaw_error
            previous_stamp = float(pose['stamp_s'])
            result = self.send_yaw_velocity(speed)
            if not isinstance(result, dict):
                raise RuntimeError('底盘 yaw 速度响应不是 JSON 对象')
            if result.get('execution') == 'shadow':
                raise RuntimeError(
                    '底盘 yaw 命令被 VLAHost 置为 shadow，未实际发布：'
                    f'{result.get("reason", result)}')
            if (not result.get('success') or
                    result.get('execution') not in (None, 'active')):
                raise RuntimeError(f'底盘 yaw 速度命令未执行：{result}')
            print(f'已发送底盘 yaw 速度：wz={speed:.3f} rad/s')
            time.sleep(self.args.yaw_command_duration_s)
            try:
                pose = self.wait_for_next_pose(
                    previous_stamp,
                    min(deadline, time.monotonic() + POSE_MAX_AGE_S))
            except (PoseNotReadyError, OdometryInterruptedError) as error:
                if not self.stop_yaw():
                    raise RuntimeError('yaw 位姿中断后零速度未确认，禁止恢复') from error
                if recoveries >= getattr(self.args, 'odometry_retries', 0):
                    raise RuntimeError('yaw 位姿中断；已停止，未自动继续') from error
                recoveries += 1
                paused_at = time.monotonic()
                print(f'yaw 位姿中断，底盘已停；等待最多 '
                      f'{self.args.pose_ready_timeout_s:.1f} s 持续更新后恢复 '
                      f'{recoveries}/{self.args.odometry_retries}。')
                pose = self.wait_for_fresh_pose()
                deadline += time.monotonic() - paused_at
                last_progress_at = time.monotonic()
            _, _, _, yaw = validate_base_world_pose(pose)
            yaw_delta = wrapped_angle(yaw - previous_yaw)
            yaw_unwrapped += yaw_delta
            yaw_travel += abs(yaw_delta)
            yaw_error = target_unwrapped - yaw_unwrapped
            print(f'世界 yaw 反馈：current={yaw:.5f} rad，error={yaw_error:.5f} rad')
            if abs(yaw_delta) > YAW_MAX_FEEDBACK_STEP_RAD:
                raise RuntimeError('yaw 位姿单步跳变过大，已停止移动')
            if (yaw_reference is None and yaw_travel > getattr(
                    self.args, 'yaw_max_travel_rad', DEFAULT_YAW_MAX_TRAVEL_RAD)):
                raise RuntimeError('yaw 累计转角超过安全上限，已停止以防电缆缠绕')
            if (yaw_reference is not None and
                    abs(yaw_unwrapped - yaw_reference) > math.pi):
                raise RuntimeError('yaw 偏离电缆零位超过半圈，已停止以防电缆缠绕')
            if yaw_delta * speed < 0.0 and abs(yaw_delta) >= YAW_OPPOSITE_STEP_RAD:
                raise RuntimeError('yaw 反馈方向与速度命令相反，已停止移动')
            if abs(yaw_error) > abs(previous_error) + YAW_DIVERGENCE_RAD:
                raise RuntimeError('yaw 误差持续增大，已停止移动')
            if abs(yaw_error) <= self.args.yaw_tolerance_rad:
                if not self.stop_yaw():
                    raise RuntimeError('底盘 yaw 已到目标，但停止命令未确认')
                print('底盘世界 yaw 修正完成。')
                return
            if abs(yaw_error) < abs(previous_error) - 0.003:
                last_progress_at = time.monotonic()
            elif time.monotonic() - last_progress_at > 1.0:
                raise RuntimeError('底盘 yaw 命令发送后 1 s 内没有接近目标')
        raise TimeoutError(f'底盘 yaw 返回超时，剩余误差 {yaw_error:.5f} rad')

    def run(self):
        pose = self.wait_for_fresh_pose()
        _, _, z, _ = validate_base_world_pose(pose)
        if abs(z - self.args.target_z_m) > self.args.position_tolerance_m:
            raise RuntimeError(
                '底盘接口不能主动调整 z，当前高度与目标相差超过位置容限：'
                f'actual={z:.5f} target={self.args.target_z_m:.5f}')
        self.move_xy()
        self.move_yaw()
        pose = self.wait_for_fresh_pose()
        x, y, z, yaw = validate_base_world_pose(pose)
        x_error = x - self.args.target_x_m
        y_error = y - self.args.target_y_m
        z_error = z - self.args.target_z_m
        yaw_error = wrapped_angle(yaw - self.args.target_yaw_rad)
        print(
            '底盘世界目标完成：'
            f'actual x={x:.5f} y={y:.5f} z={z:.5f} yaw={yaw:.5f} rad；'
            f'error x={x_error:.4f} y={y_error:.4f} '
            f'z={z_error:.4f} yaw={yaw_error:.5f} rad')
        if math.hypot(x_error, y_error) > self.args.position_tolerance_m:
            raise RuntimeError('底盘返回后 x/y 误差超过容限')
        if abs(z_error) > self.args.position_tolerance_m:
            raise RuntimeError(
                '底盘返回后 z 与目标不一致（底盘接口不能主动调 z）：'
                f'actual={z:.5f} target={self.args.target_z_m:.5f}')
        if abs(yaw_error) > self.args.yaw_tolerance_rad:
            raise RuntimeError('底盘返回后 yaw 误差超过容限')


def main():
    args = parse_args()
    mover = BaseWorldMover(args)
    try:
        if args.check_only:
            mover.check_only()
        else:
            mover.run()
    except KeyboardInterrupt:
        if mover.command_attempted:
            print('收到 Ctrl-C，正在停止底盘动作。', file=sys.stderr)
            mover.cancel_translation()
            mover.stop_yaw()
        else:
            print('收到 Ctrl-C，底盘动作尚未启动。', file=sys.stderr)
        return 130
    except (RuntimeError, TimeoutError, ValueError) as error:
        if mover.command_attempted:
            mover.cancel_translation()
            mover.stop_yaw()
        operation = '世界底盘位姿检查' if args.check_only else '底盘世界坐标移动'
        print(f'{operation}失败：{error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
