# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline regression tests for Luna base-world motion."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_DIR = Path(__file__).parents[4] / "test.incoming_20261021"
SCRIPT = SCRIPT_DIR / "base_move_world.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("luna_base_move_world", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pose(*, yaw=0.0, stamp=1.0, age=0.01):
    return {
        "valid": True,
        "frame_id": "camera_init",
        "child_frame_id": "body",
        "received_age_s": age,
        "stamp_s": stamp,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "yaw_rad": yaw,
    }


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def test_stable_pose_requires_repeated_source_stamps(monkeypatch):
    module = _load_script()
    clock = _Clock()
    mover = module.BaseWorldMover(
        SimpleNamespace(pose_ready_timeout_s=3.0, pose_stable_s=2.0)
    )
    mover.get_pose = lambda: _pose(
        stamp=1.0 + math.floor(clock.now * 10.0) / 10.0
    )
    monkeypatch.setattr(module, "time", clock)

    result = mover.wait_for_fresh_pose()

    assert clock.now >= 2.0
    assert result["stamp_s"] > 1.0
    assert not mover.command_attempted


def test_segment_stays_below_server_limit():
    module = _load_script()
    mover = module.BaseWorldMover(
        SimpleNamespace(target_x_m=1.0, target_y_m=0.0, position_tolerance_m=0.05)
    )
    mover.wait_for_fresh_pose = lambda: _pose()
    lengths = []

    class SegmentCaptured(Exception):
        pass

    def capture(dx, dy, _x, _y, _label):
        lengths.append(math.hypot(dx, dy))
        raise SegmentCaptured

    mover.send_and_wait_xy = capture
    with pytest.raises(SegmentCaptured):
        mover.move_xy_segmented()

    assert lengths[0] == pytest.approx(0.49)
    assert lengths[0] < 0.50


def test_translation_interrupt_rechecks_fresh_pose_before_one_retry(monkeypatch):
    module = _load_script()
    mover = module.BaseWorldMover(SimpleNamespace(
        odometry_retries=1, vlahost_url="http://unused", http_timeout_s=1.0,
    ))
    calls = []
    attempts = iter((module.OdometryInterruptedError(), None))

    def move_once():
        calls.append("move")
        error = next(attempts)
        if error:
            raise error

    mover.move_xy_once = move_once
    mover.stop_yaw = lambda: calls.append("stop") or True
    mover.wait_for_fresh_pose = lambda: calls.append("fresh")

    def request(_url, path, body=None, **_kwargs):
        calls.append(path)
        return {"move": {"status": "cancelled"}}

    monkeypatch.setattr(module, "request_json", request)
    mover.move_xy()

    assert calls == ["move", "/base_move_world/cancel", "/base_move_world",
                     "stop", "fresh", "move"]


def test_world_length_error_accepts_new_server_limit_message():
    module = _load_script()
    assert module.BaseWorldMover.is_server_world_length_error(
        ValueError("world offset length must be 0.05..2.00 m")
    )


def test_pose_rejects_nonfinite_quaternion_norm():
    module = _load_script()
    with pytest.raises(ValueError, match="四元数模长无效"):
        module.pose_values_to_target((0, 0, 0, 1e308, 1e308, 1e308, 1e308))


def test_yaw_reaches_target_after_fresh_feedback(monkeypatch):
    module = _load_script()
    clock = _Clock()
    args = SimpleNamespace(
        target_yaw_rad=0.12,
        yaw_tolerance_rad=0.1,
        yaw_timeout_s=5.0,
        yaw_speed_rad_s=0.2,
        yaw_command_duration_s=0.2,
    )
    mover = module.BaseWorldMover(args)
    mover.wait_for_fresh_pose = lambda: _pose(yaw=0.0, stamp=1.0)
    speeds = []
    mover.send_yaw_velocity = lambda speed: (
        speeds.append(speed) or {"success": True, "execution": "active"}
    )
    mover.wait_for_next_pose = lambda _stamp, _deadline: _pose(yaw=0.04, stamp=2.0)
    mover.stop_yaw = lambda: True
    monkeypatch.setattr(module, "time", clock)

    mover.move_yaw()

    assert len(speeds) == 1
    assert speeds[0] > 0.0


def test_yaw_feedback_opposite_command_aborts(monkeypatch):
    module = _load_script()
    clock = _Clock()
    mover = module.BaseWorldMover(
        SimpleNamespace(
            target_yaw_rad=0.5,
            yaw_tolerance_rad=0.1,
            yaw_timeout_s=5.0,
            yaw_speed_rad_s=0.2,
            yaw_command_duration_s=0.2,
        )
    )
    mover.wait_for_fresh_pose = lambda: _pose(yaw=0.0, stamp=1.0)
    mover.send_yaw_velocity = lambda _speed: {
        "success": True,
        "execution": "active",
    }
    mover.wait_for_next_pose = lambda _stamp, _deadline: _pose(
        yaw=-0.04, stamp=2.0
    )
    monkeypatch.setattr(module, "time", clock)

    with pytest.raises(RuntimeError, match="方向与速度命令相反"):
        mover.move_yaw()


def test_yaw_pose_gap_stops_then_waits_for_fresh_feedback(monkeypatch):
    module = _load_script()
    clock = _Clock()
    mover = module.BaseWorldMover(SimpleNamespace(
        target_yaw_rad=0.12, yaw_tolerance_rad=0.1, yaw_timeout_s=5.0,
        yaw_speed_rad_s=0.2, yaw_command_duration_s=0.2,
        pose_ready_timeout_s=10.0, odometry_retries=1,
    ))
    poses = iter((_pose(yaw=0.0, stamp=1.0), _pose(yaw=0.04, stamp=2.0)))
    mover.wait_for_fresh_pose = lambda: next(poses)
    mover.send_yaw_velocity = lambda speed: {"success": True, "execution": "active"}
    stops = []
    mover.stop_yaw = lambda: stops.append(clock.now) or True
    mover.wait_for_next_pose = lambda _stamp, _deadline: (_ for _ in ()).throw(
        module.PoseNotReadyError("world pose stale")
    )
    monkeypatch.setattr(module, "time", clock)

    mover.move_yaw()

    assert len(stops) == 2  # Before recovery and after target arrival.


def test_yaw_pose_gap_does_not_use_stale_pose(monkeypatch):
    module = _load_script()
    clock = _Clock()
    mover = module.BaseWorldMover(SimpleNamespace(
        target_yaw_rad=0.12, yaw_tolerance_rad=0.1, yaw_timeout_s=5.0,
        yaw_speed_rad_s=0.2, yaw_command_duration_s=0.2,
        pose_ready_timeout_s=10.0, odometry_retries=1,
    ))
    mover.wait_for_fresh_pose = lambda: _pose(yaw=0.0, stamp=1.0)
    mover.send_yaw_velocity = lambda speed: {"success": True, "execution": "active"}
    mover.stop_yaw = lambda: True
    mover.wait_for_next_pose = lambda _stamp, _deadline: (_ for _ in ()).throw(
        module.PoseNotReadyError("world pose stale")
    )
    monkeypatch.setattr(module, "time", clock)

    with pytest.raises(RuntimeError, match="yaw 位姿中断；已停止"):
        mover.move_yaw()


def test_cable_reference_reverses_direction_before_full_wrap():
    module = _load_script()
    reference = math.radians(-85.0)
    current, target = module.yaw_control_goal(
        math.radians(-180.0), math.radians(18.0), reference
    )

    assert target - current > math.pi
    assert abs(current - reference) < math.pi
    assert abs(target - reference) < math.pi


def test_cable_reference_allows_long_unwinding_leg():
    module = _load_script()
    reference = math.radians(-175.0)
    current, target = module.yaw_control_goal(
        math.radians(18.0), math.radians(-85.0), reference
    )

    assert target - current > module.DEFAULT_YAW_MAX_TRAVEL_RAD
    assert abs(current - reference) < math.pi
    assert abs(target - reference) < math.pi


def test_pose_list_loads_named_stations_from_yaml(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "luna_base_move_world_pose_list", SCRIPT_DIR / "base_move_world_pose_list.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.parse_move_list('["table_3", "chair_1"]') == (3, 1)
    position = module.DEFAULT_POSES[2]["pose"]["position"]
    assert position["x"] == pytest.approx(-0.47073965883987257)
    assert position["y"] == pytest.approx(-0.3286442175793726)


def test_pose_list_preserves_cable_reference_between_processes(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "luna_base_move_world_pose_list", SCRIPT_DIR / "base_move_world_pose_list.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.base_move_world, "request_json", lambda *_: {"pose": _pose()})
    received = []
    monkeypatch.setattr(module.base_move_world, "main",
                        lambda: received.append(tuple(sys.argv)) or 0)
    monkeypatch.setattr(sys, "argv", ["base_move_world_pose_list.py", '["chair_1"]',
                                  "--yaw-reference-rad", "-0.4"])

    assert module.main() == 0

    assert len(received) == 1
    args = received[0]
    assert args[args.index("--yaw-reference-rad") + 1] == "-0.4"
