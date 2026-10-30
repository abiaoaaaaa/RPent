#!/usr/bin/python3
# 用途：按位姿编号或场景名称调用 base_move_world.py，依次移动底盘。
# 运行：source /home/xlb/data/code/code_new/Luna/Apex_Deploy/apex_ros_env.sh
#   python3 base_move_world_pose_list.py '["table_3", "chair_1"]'
"""按编号或场景名称依次移动底盘到默认世界位姿。

编号/场景对应关系：1=chair_1、2=chair_2、3=table_3。
每个默认位姿包含 position xyz 和 orientation quaternion；程序会从四元数计算
底盘需要的世界 yaw，再复用 ``base_move_world.py`` 的 x/y/yaw 闭环移动逻辑。
"""

import argparse
import ast
import math
import sys
import time
from pathlib import Path

import base_move_world
import yaml

POSE_NAMES = ('chair_1', 'chair_2', 'table_3')
POSE_INDEX_BY_NAME = {
    name: index for index, name in enumerate(POSE_NAMES, start=1)
}
POSE_FILE = Path(__file__).with_name('pos.yaml')


def load_default_poses(path=POSE_FILE):
    """从 pos.yaml 读取椅 1、椅 2 和桌 3 的唯一位姿配置。"""
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f'无法读取位姿文件 {path}：{error}') from error
    if not isinstance(payload, dict):
        raise ValueError(f'位姿文件 {path} 必须是 YAML 映射')
    poses = []
    for index, station in enumerate(POSE_NAMES, start=1):
        entry = payload.get(f'pose{index}')
        if not isinstance(entry, dict):
            raise ValueError(f'{path} 缺少 pose{index}')
        configured_station = entry.get('station')
        if configured_station not in (None, station):
            raise ValueError(
                f'{path} 中 pose{index} 应为 {station}，实际为 '
                f'{configured_station}')
        poses.append(entry)
    return tuple(poses)


DEFAULT_POSES = load_default_poses()


def parse_move_list(raw_value):
    """安全解析位姿编号或场景名称列表。"""
    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            'move list 必须是形如 [3, 1] 或 ["table_3", "chair_1"] 的列表') from error
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('move list 必须是非空列表')
    result = []
    for item in value:
        if isinstance(item, str):
            try:
                item = POSE_INDEX_BY_NAME[item]
            except KeyError as error:
                raise ValueError(
                    f'未知场景 {item!r}，可选值为 {", ".join(POSE_NAMES)}') from error
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError('move list 只能包含编号 1、2、3 或已命名场景')
        if not 1 <= item <= len(DEFAULT_POSES):
            raise ValueError(
                f'位姿编号 {item} 无效，编号范围为 1..{len(DEFAULT_POSES)}')
        result.append(item)
    return tuple(result)


def pose_to_target(pose, index):
    """从一个 position/orientation 位姿计算 x、y、z、yaw。"""
    if pose is None:
        raise ValueError(
            f'默认位姿 {index} 尚未填写，请在 DEFAULT_POSES 中补充 '
            'position 和 orientation')
    # 同时兼容直接粘贴 YAML 的 pose.pose 结构。
    data = pose
    while isinstance(data, dict) and 'position' not in data and 'pose' in data:
        data = data['pose']
    if not isinstance(data, dict):
        raise ValueError(f'默认位姿 {index} 结构无效')
    position = data.get('position')
    orientation = data.get('orientation')
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise ValueError(f'默认位姿 {index} 必须包含 position 和 orientation')
    try:
        x, y, z = (float(position[axis]) for axis in ('x', 'y', 'z'))
        qx, qy, qz, qw = (float(orientation[axis])
                          for axis in ('x', 'y', 'z', 'w'))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f'默认位姿 {index} 的 position/orientation 字段无效') from error
    values = (x, y, z, qx, qy, qz, qw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'默认位姿 {index} 包含非有限数值')
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError(f'默认位姿 {index} 的四元数模长无效')
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return x, y, z, yaw


def print_help():
    """打印列表入口说明。"""
    print('用法：')
    print('  python3 base_move_world_pose_list.py "[1, 2, 3]" [base_move_world 参数]')
    print()
    print('示例：')
    print('  python3 base_move_world_pose_list.py "[1]"')
    print("  python3 base_move_world_pose_list.py '[\"table_3\", \"chair_1\"]'")
    print('  python3 base_move_world_pose_list.py "[1, 2, 3]" --speed-m-s 0.06')
    print('  python3 base_move_world_pose_list.py "[3, 2, 1]"')
    print()
    print('对应关系：1=chair_1、2=chair_2、3=table_3；按列表顺序执行。')
    print('其余参数会原样传给 base_move_world.py。')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('move_list', nargs='?',
                        help='位姿编号或场景名称列表，例如 [3, 1]')
    parser.add_argument('--pause-s', type=float, default=0.5,
                        help='两个位姿之间的等待时间，默认 0.5 s')
    parser.add_argument('--vlahost-url', default=base_move_world.DEFAULT_VLAHOST_URL,
                        help='VLAHost HTTP 地址')
    parser.add_argument('--yaw-reference-rad', type=float,
                        help='延续前一段导航的电缆零位 yaw；默认使用本次起点')
    parser.add_argument('-h', '--help', action='store_true')
    known, remaining = parser.parse_known_args()
    if known.help:
        print_help()
        return 0
    if known.move_list is None:
        print_help()
        return 2
    if not math.isfinite(known.pause_s) or known.pause_s < 0.0:
        parser.error('--pause-s 必须是大于等于 0 的有限数值')
    if (known.yaw_reference_rad is not None and
            not math.isfinite(known.yaw_reference_rad)):
        parser.error('--yaw-reference-rad 必须是有限数值')
    try:
        move_list = parse_move_list(known.move_list)
        targets = [pose_to_target(DEFAULT_POSES[index - 1], index)
                   for index in move_list]
    except ValueError as error:
        parser.error(str(error))

    try:
        response = base_move_world.request_json(
            known.vlahost_url, '/base_world_pose')
        pose = response.get('pose') if isinstance(response, dict) else None
        _, _, _, current_yaw = base_move_world.validate_base_world_pose(pose)
    except ValueError as error:
        print(f'读取电缆零位失败：{error}', file=sys.stderr)
        return 1
    yaw_reference = (current_yaw if known.yaw_reference_rad is None
                     else known.yaw_reference_rad)
    print(f'已锁定本次列表的电缆零位 yaw={yaw_reference:.5f} rad。')

    # 每个目标都调用一次 base_move_world.py，确保复用同一套停止和误差检查逻辑。
    original_argv = sys.argv
    try:
        for step, (index, target) in enumerate(zip(move_list, targets), start=1):
            x, y, z, yaw = target
            station = POSE_NAMES[index - 1]
            print(
                f'开始执行第 {step}/{len(move_list)} 个目标：'
                f'{station}（位姿 {index}），'
                f'xyz=[{x:.6f}, {y:.6f}, {z:.6f}] m，yaw={yaw:.6f} rad')
            sys.argv = [
                original_argv[0],
                *remaining,
                '--vlahost-url', known.vlahost_url,
                '--yaw-reference-rad', str(yaw_reference),
                '--target-x-m', str(x),
                '--target-y-m', str(y),
                '--target-z-m', str(z),
                '--target-yaw-rad', str(yaw),
            ]
            result = base_move_world.main()
            if result != 0:
                return result
            if step < len(move_list) and known.pause_s > 0.0:
                time.sleep(known.pause_s)
    finally:
        sys.argv = original_argv
    print('move list 执行完成。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
