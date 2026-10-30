# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import base64

from rpent.vlahost_monitor.server import MAX_HISTORY, MonitorStore, snapshot_to_sample


def _snapshot() -> dict:
    return {
        "connected": True,
        "timestamp": 123.0,
        "joint_states": {
            "positions": list(range(14)),
            "velocities": [x / 10 for x in range(14)],
            "body_positions": [20, 21, 22, 23, 24, 25],
            "body_velocities": [0, 1, 2, 3, 4, 5],
        },
        "gripper_left": {"position": 0.1, "velocity": 0.2},
        "gripper_right": {"position": 0.3, "velocity": 0.4},
        "jpeg_base64": base64.b64encode(b"\xff\xd8test").decode(),
    }


def test_snapshot_to_sample_contains_22_joint_rows():
    sample = snapshot_to_sample(_snapshot(), b"\xff\xd8test", 7, 1.0)
    assert sample["sequence"] == 7
    assert len(sample["joint_rows"]) == 22
    assert sample["joint_rows"][0]["name"] == "left_arm_joint_1"
    assert sample["joint_rows"][-1]["name"] == "gripper_right"
    assert "_jpeg" in sample


def test_store_history_is_bounded_without_control_methods():
    store = MonitorStore("http://example.invalid")
    for sequence in range(MAX_HISTORY + 4):
        sample = snapshot_to_sample(_snapshot(), b"\xff\xd8test", sequence + 1, 1.0)
        with store._lock:
            store._samples.append(sample)
    assert store.status()["history_size"] == MAX_HISTORY
    assert store.frame("latest") == b"\xff\xd8test"
