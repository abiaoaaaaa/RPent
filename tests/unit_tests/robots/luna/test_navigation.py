# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Offline contracts for Luna single-destination navigation."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from robots.luna.navigation import LunaNavigationRunner, load_locations
from rpent.session import EnvState
from rpent.tools import ToolResult


def _pose(x: float, y: float, yaw: float) -> dict[str, object]:
    return {
        "valid": True,
        "frame_id": "camera_init",
        "child_frame_id": "body",
        "received_age_s": 0.02,
        "position": {"x": x, "y": y, "z": 0.0},
        "yaw_rad": yaw,
    }


def _runner(tmp_path: Path, client: SimpleNamespace) -> LunaNavigationRunner:
    locations = {
        "pose1": {
            "station": "chair_1",
            "pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
        "pose2": {
            "station": "chair_2",
            "label": "沙发椅",
            "pose": {
                "position": {"x": 0.5, "y": 0.5, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
        "pose3": {
            "station": "table_3",
            "pose": {
                "position": {"x": -0.5, "y": -0.3, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0},
            },
        },
    }
    (tmp_path / "pos.yaml").write_text(yaml.safe_dump(locations))
    (tmp_path / "native_nav_test.py").write_text("# offline fixture\n")
    return LunaNavigationRunner(
        script_dir=tmp_path,
        python_executable="/usr/bin/python3",
        vlahost_url="http://robot:8011",
        output_dir=tmp_path / "logs",
        client=client,
        check_cancelled=lambda: None,
        timeout_s=1.0,
        execute=True,
    )


def test_load_and_resolve_named_locations(tmp_path: Path):
    current = _pose(0.0, 0.0, 0.25)
    client = SimpleNamespace(get=lambda _path: {"pose": current})
    runner = _runner(tmp_path, client)

    assert set(load_locations(tmp_path / "pos.yaml")) == {
        "chair_1",
        "chair_2",
        "table_3",
    }
    assert runner.resolve("3号桌子") == "table_3"
    assert runner.resolve("椅1") == "chair_1"
    assert runner.resolve("沙发椅") == "chair_1"
    assert runner.resolve("sofa") == "chair_1"

    runner.execute = False
    disabled = runner.navigate("table_3")
    assert disabled.data["error_code"] == "motion_disabled"


def test_navigation_verification_includes_camera_evidence(tmp_path: Path):
    current = _pose(1.0, 2.0, 0.0)
    frame = io.BytesIO()
    Image.new("RGB", (640, 480), (30, 90, 160)).save(frame, format="JPEG")
    client = SimpleNamespace(
        get=lambda _path: {"pose": current},
        camera_jpeg=lambda: frame.getvalue(),
    )
    runner = _runner(tmp_path, client)

    result = runner.verify("chair_1")

    assert not result.is_error
    assert result.data["dashboard_image_kind"] == "luna_navigate_chair_1_verification"
    assert result.images and result.images[0].startswith(b"\x89PNG")
    source_path = Path(result.data["visual_image_path"])
    assert source_path.is_file()
    assert source_path.stem.endswith("_source")
    state = EnvState(tmp_path / "state")
    with Image.open(io.BytesIO(result.images[0])) as image:
        pixels = np.asarray(image.convert("RGB"))
    with state.record_step(state={}) as step:
        state.save(
            f"{result.data['dashboard_image_kind']}.png",
            pixels,
            step=step,
        )
    assert state.exists(
        f"{result.data['dashboard_image_kind']}.png",
        step=step,
    )


def test_each_navigation_call_contains_one_station_and_reuses_cable_zero(
    tmp_path: Path,
):
    current = _pose(0.0, 0.0, 0.25)
    client = SimpleNamespace(get=lambda _path: {"pose": current})
    runner = _runner(tmp_path, client)
    commands: list[list[str]] = []

    def run(command: list[str], station: str, attempt: int) -> ToolResult:
        commands.append(command)
        target = runner.locations[station]["pose"]
        position = target["position"]
        current["position"] = dict(position)
        current["yaw_rad"] = 3.141592653589793 if station == "table_3" else 0.0
        return ToolResult(
            data={"success": True, "log_path": f"{station}-{attempt}.log"}
        )

    runner._run = run  # type: ignore[method-assign]
    first = runner.navigate("table_3")
    second = runner.navigate("chair_1")

    assert not first.is_error and not second.is_error
    for command in commands:
        assert command[1].endswith("native_nav_test.py")
        assert command[command.index("--relative") + 1]
        assert command[command.index("--execute")] == "--execute"
    assert first.data["control_interface"] == "/move/MoveRelativeCmd"
    assert second.data["control_interface"] == "/move/MoveRelativeCmd"


def test_failed_first_leg_keeps_reference_and_allows_one_agent_retry(tmp_path: Path):
    current = _pose(0.0, 0.0, -0.4)
    client = SimpleNamespace(get=lambda _path: {"pose": current})
    runner = _runner(tmp_path, client)
    commands: list[list[str]] = []

    def fail(command: list[str], station: str, attempt: int) -> ToolResult:
        commands.append(command)
        current["yaw_rad"] = 1.2
        return ToolResult(
            data={
                "error": "offline failure",
                "location": station,
                "attempts": attempt,
            }
        )

    runner._run = fail  # type: ignore[method-assign]
    first = runner.navigate("table_3")
    second = runner.navigate("table_3")
    exhausted = runner.navigate("table_3")

    assert first.data["retryable"] is True
    assert second.data["retryable"] is False
    assert exhausted.data["error_code"] == "navigation_retry_exhausted"
    assert len(commands) == 2
    assert runner.cable_reference_yaw_rad == -0.4
    assert all("--relative" in command for command in commands)
