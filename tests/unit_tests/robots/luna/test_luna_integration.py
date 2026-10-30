# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline contracts for the Luna Harness integration."""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from robots.luna.client import LunaVLAHostClient, LunaWorkflowRunner
from robots.luna.robot_spec import _parse_config, get_robot_spec
from robots.luna.toolkit import LunaToolkit
from robots.luna.tools import LunaPrimitives
from robots.luna.vision import comparison_png, verify_grasp
from rpent.memory import MemoryManager
from rpent.robots import enumerate_robots
from rpent.tools import ToolResult
from rpent.utils.logging import init_output_dir


def _runner(tmp_path: Path, *, destination: int = 1) -> LunaWorkflowRunner:
    (tmp_path / "base_move_world_pose_list.py").write_text("print('arrived')\n")
    (tmp_path / "apriltag_left_tool_grasp_test.py").write_text(
        "import json, sys\n"
        "print('RPENT_GATE=await_visual_verdict', flush=True)\n"
        "verdict = sys.stdin.readline().strip()\n"
        "success = verdict == 'held'\n"
        "print('RPENT_RESULT=' + json.dumps(dict(phase='destination_moving' "
        "if success else 'await_visual_verdict', error_code=None if success "
        "else 'visual_rejected', message='motion ended' if success else "
        "'transport skipped')), flush=True)\n"
        "sys.exit(0 if success else 3)\n"
    )
    image = io.BytesIO()
    Image.new("RGB", (16, 30), "white").save(image, format="JPEG")
    runner = LunaWorkflowRunner(
        script_dir=tmp_path,
        python_executable=sys.executable,
        vlahost_url="http://6.6.7.101:8011",
        destination_pose=destination,
        tag_id=5,
        output_dir=tmp_path / "output",
        check_cancelled=lambda: None,
        navigation_timeout_s=5.0,
        grasp_timeout_s=5.0,
        image_client=SimpleNamespace(
            camera_jpeg=lambda: image.getvalue(),
            get=lambda path: {"pose": {
                "valid": True, "frame_id": "camera_init", "child_frame_id": "body",
                "received_age_s": 0.02, "yaw_rad": -0.4,
            }},
        ),
        visual_verifier=lambda image: {"verdict": "held", "evidence": "visible in gripper"},
    )
    runner.cable_reference_yaw_rad = -0.4  # Simulate completed table navigation.
    return runner


def test_luna_is_discoverable_and_tasks_bind_destination(tmp_path: Path):
    assert "luna" in enumerate_robots()
    spec = get_robot_spec()
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(["--task-id", "2"])
    args.output_dir = str(tmp_path)

    config = _parse_config(args)

    assert config.recipe_tag == "luna_t2"
    assert args.luna_vlahost_url == "http://6.6.7.101:8011"
    assert Path(args.luna_ros_setup).is_file()
    assert config.prompt_vars["destination"] == "chair_2"
    assert config.task_desc == {
        "task_id": 2,
        "destination": "chair_2",
        "tag_id": 5,
    }
    assert "chair_2" in spec.prompts.render("user", variables=config.prompt_vars)
    system_prompt = spec.prompts.render("system", variables=config.prompt_vars)
    assert "top-level ok is false" in system_prompt
    assert "do not infer faults from nested raw telemetry fields" in system_prompt


def test_runner_reuses_validated_scripts_and_locks_chair(tmp_path: Path):
    runner = _runner(tmp_path, destination=2)
    commands = []

    def capture(command, label, timeout_s, **kwargs):
        commands.append((command, label, timeout_s, kwargs))
        return ToolResult(data={"success": True, "workflow": label})

    runner._run = capture

    assert not runner.move_to_table_3().is_error
    assert not runner.pick_apriltag_module().is_error
    assert not runner.completed  # A clean child exit is not proof of physical success.
    assert '["table_3"]' in commands[0][0]
    assert commands[1][0][commands[1][0].index("--destination-pose") + 1] == "2"
    assert "--execute" in commands[1][0]
    assert "--require-visual-verdict" in commands[1][0]
    assert commands[0][0][commands[0][0].index("--yaw-reference-rad") + 1] == "-0.4"
    assert commands[1][0][commands[1][0].index("--cable-reference-yaw-rad") + 1] == "-0.4"
    assert commands[1][3]["visual_gate"] is True
    assert commands[1][0][commands[1][0].index("--tag-id") + 1] == "5"
    assert commands[1][0][commands[1][0].index("--position-tolerance-mm") + 1] == "10.0"
    assert commands[1][0][commands[1][0].index("--max-target-angle-deg") + 1] == "90.0"


def test_grasp_requires_reference_from_table_navigation(tmp_path: Path):
    runner = _runner(tmp_path)
    runner.cable_reference_yaw_rad = None

    result = runner.pick_apriltag_module()

    assert result.is_error
    assert "cable reference" in result.data["error"]
    assert not (runner.output_dir / "luna_pick_to_chair_1.log").exists()


def test_stale_cable_reference_blocks_table_motion(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("robots.luna.client.POSE_RECOVERY_TIMEOUT_S", 0)
    runner = _runner(tmp_path)
    runner.image_client.get = lambda path: {"pose": {
        "valid": True, "frame_id": "camera_init", "child_frame_id": "body",
        "received_age_s": 1.0, "yaw_rad": 0.0,
    }}
    runner._run = lambda *_: pytest.fail("motion started with a stale cable reference")

    result = runner.move_to_table_3()

    assert result.is_error
    assert runner.cable_reference_yaw_rad is None


def test_table_navigation_waits_for_fresh_cable_reference(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("robots.luna.client.time.sleep", lambda _: None)
    runner = _runner(tmp_path)
    ages = iter((5.0, 1.0, 0.02))
    runner.image_client.get = lambda path: {
        "pose": {
            "valid": True,
            "frame_id": "camera_init",
            "child_frame_id": "body",
            "received_age_s": next(ages),
            "yaw_rad": -0.4,
        }
    }
    commands = []

    def capture(command, label, timeout_s):
        commands.append(command)
        return ToolResult(data={"success": True})

    runner._run = capture

    result = runner.move_to_table_3()

    assert not result.is_error
    assert len(commands) == 1
    assert runner.cable_reference_yaw_rad == -0.4


def test_runner_executes_owned_children_and_writes_logs(tmp_path: Path):
    runner = _runner(tmp_path)

    move = runner.move_to_table_3()
    pick = runner.pick_apriltag_module()

    assert move.data["success"] is True
    assert pick.data["success"] is True
    assert pick.data["payload_confirmed"] is True
    assert pick.data["verdict"] == "held"
    assert pick.data["phase"] == "destination_moving"
    assert Path(move.data["log_path"]).is_file()
    assert Path(pick.data["log_path"]).is_file()


def test_grasp_failure_reports_structured_phase(tmp_path: Path):
    runner = _runner(tmp_path)
    (tmp_path / "apriltag_left_tool_grasp_test.py").write_text(
        "import json, sys\n"
        "print('RPENT_RESULT=' + json.dumps(dict(phase='approach', "
        "error_code='precheck_failed', message='目标转角过大')))\n"
        "sys.exit(1)\n"
    )

    result = runner.pick_apriltag_module()

    assert result.is_error
    assert result.data["phase"] == "approach"
    assert result.data["error_code"] == "precheck_failed"
    assert result.data["message"] == "目标转角过大"


def test_visual_rejection_prevents_chair_transport(tmp_path: Path):
    runner = _runner(tmp_path)
    runner.visual_verifier = lambda image: (_ for _ in ()).throw(
        RuntimeError("vision unavailable")
    )

    result = runner.pick_apriltag_module()

    assert result.is_error
    assert result.data["verdict"] == "error"
    assert result.data["payload_confirmed"] is False
    assert result.data["phase"] == "await_visual_verdict"
    assert "transport skipped" in Path(result.data["log_path"]).read_text()


def test_clear_visual_miss_goes_home_then_retries_once(tmp_path: Path):
    runner = _runner(tmp_path)
    (tmp_path / "apriltag_left_tool_grasp_test.py").write_text(
        "import json, os, sys\n"
        "print('RPENT_GATE=await_visual_verdict', flush=True)\n"
        "verdict = sys.stdin.readline().strip()\n"
        "phase = 'retry_ready' if verdict == 'retry' else 'destination_moving'\n"
        "print('RPENT_RESULT=' + json.dumps(dict(phase=phase, error_code=None, "
        "message='home complete' if phase == 'retry_ready' else 'arrived')), flush=True)\n"
        "sys.exit(0)\n"
    )
    verdicts = iter((
        {"verdict": "not_held", "evidence": "module remains on the table"},
        {"verdict": "held", "evidence": "module lifted in left jaws"},
    ))
    runner.visual_verifier = lambda image: next(verdicts)

    result = runner.pick_apriltag_module()

    assert result.data["attempts"] == 2
    assert result.data["verdict"] == "held"
    assert result.data["payload_confirmed"] is True
    assert Path(result.data["first_attempt_log_path"]).is_file()
    assert Path(result.data["log_path"]).is_file()


def test_two_clear_visual_misses_stop_after_home(tmp_path: Path):
    runner = _runner(tmp_path)
    (tmp_path / "apriltag_left_tool_grasp_test.py").write_text(
        "import json, sys\n"
        "print('RPENT_GATE=await_visual_verdict', flush=True)\n"
        "assert sys.stdin.readline().strip() == 'retry'\n"
        "print('RPENT_RESULT=' + json.dumps(dict(phase='retry_ready', "
        "error_code=None, message='home complete')), flush=True)\n"
    )
    runner.visual_verifier = lambda image: {
        "verdict": "not_held", "evidence": "module remains on the table"
    }

    result = runner.pick_apriltag_module()

    assert result.is_error
    assert result.data["attempts"] == 2
    assert result.data["error_code"] == "visual_retry_exhausted"


def test_invalid_visual_result_does_not_retry(tmp_path: Path):
    runner = _runner(tmp_path)
    runner.visual_verifier = lambda image: {
        "verdict": "uncertain", "evidence": "table contact hidden"
    }

    result = runner.pick_apriltag_module()

    assert result.is_error
    assert result.data["verdict"] == "error"
    assert result.data["attempts"] == 1


def test_missing_camera_stops_before_grasp_child(tmp_path: Path):
    runner = _runner(tmp_path)
    runner.image_client = SimpleNamespace(camera_jpeg=lambda: (_ for _ in ()).throw(
        RuntimeError("camera stale")
    ))

    result = runner.pick_apriltag_module()

    assert result.is_error
    assert result.data["error_code"] == "camera_unavailable"
    assert not Path(result.data["log_path"]).exists()


def test_vlahost_camera_requires_fresh_three_view_jpeg():
    image = io.BytesIO()
    Image.new("RGB", (16, 30), "white").save(image, format="JPEG")
    client = LunaVLAHostClient("http://unused")
    snapshot = {
        "jpeg_base64": base64.b64encode(image.getvalue()).decode(),
        "freshness": {"jpeg": {"age_s": 0.02, "source_age_s": 0.03}},
    }
    client.get = lambda path: snapshot
    assert client.camera_jpeg() == image.getvalue()
    snapshot["freshness"]["jpeg"]["age_s"] = 1.0
    try:
        client.camera_jpeg()
    except RuntimeError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale camera image was accepted")


def test_visual_verifier_sends_comparison_image_and_requires_evidence(monkeypatch):
    import rpent.planner.codex as codex

    image = io.BytesIO()
    Image.new("RGB", (16, 30), "white").save(image, format="JPEG")
    comparison = comparison_png(image.getvalue(), image.getvalue())
    seen = {}
    monkeypatch.setattr(codex, "build_probe_config", lambda: object())

    def answer(config, **kwargs):
        seen.update(kwargs)
        return '{"verdict":"held","evidence":"module moves with left gripper"}'

    monkeypatch.setattr(codex, "run_probe_turn", answer)
    verdict = verify_grasp(comparison)
    assert verdict["verdict"] == "held"
    assert seen["image_bytes"] == comparison
    assert seen["output_schema"]["properties"]["verdict"]["enum"] == [
        "held",
        "not_held",
    ]
    assert "ONLY the LEFT wrist camera views" in seen["prompt"]
    assert "Do not require a visible gap" in seen["prompt"]
    verify_grasp(comparison, wrist_side="right")
    assert "ONLY the RIGHT wrist camera views" in seen["prompt"]
    assert "right gripper" in seen["prompt"]
    monkeypatch.setattr(codex, "run_probe_turn", lambda *args, **kwargs: "{}")
    with pytest.raises(ValueError, match="invalid grasp verdict"):
        verify_grasp(comparison)


def test_visual_verifier_uses_run_api_model(monkeypatch):
    from pydantic_ai.models.test import TestModel

    import rpent.planner.base as base

    image = io.BytesIO()
    Image.new("RGB", (16, 30), "white").save(image, format="JPEG")
    comparison = comparison_png(image.getvalue(), image.getvalue())
    seen = {}

    def build(model, base_url):
        seen.update(model=model, base_url=base_url)
        return TestModel(
            custom_output_args={"verdict": "held", "evidence": "block between jaws"}
        )

    monkeypatch.setattr(base, "build_api_model", build)
    verdict = verify_grasp(
        comparison,
        planner="api",
        model="openai-chat:Qwen3.5-9B",
        base_url="http://qwen.test/v1",
    )

    assert verdict == {"verdict": "held", "evidence": "block between jaws"}
    assert seen == {
        "model": "openai-chat:Qwen3.5-9B",
        "base_url": "http://qwen.test/v1",
    }


def test_visual_comparison_contains_only_left_wrist_views():
    before = Image.new("RGB", (16, 30), "red")
    after = Image.new("RGB", (16, 30), "blue")
    before.paste("green", (0, 12, 8, 18))
    after.paste("yellow", (0, 12, 8, 18))
    frames = []
    for frame in (before, after):
        output = io.BytesIO()
        frame.save(output, format="PNG")
        frames.append(output.getvalue())

    with Image.open(io.BytesIO(comparison_png(*frames))) as comparison:
        assert comparison.size == (16, 38)
        assert comparison.getpixel((3, 35)) == (0, 128, 0)
        assert comparison.getpixel((11, 35)) == (255, 255, 0)


def test_visual_comparison_can_select_right_wrist_views():
    before = Image.new("RGB", (16, 30), "red")
    after = Image.new("RGB", (16, 30), "blue")
    before.paste("purple", (8, 12, 16, 18))
    after.paste("orange", (8, 12, 16, 18))
    frames = []
    for frame in (before, after):
        output = io.BytesIO()
        frame.save(output, format="PNG")
        frames.append(output.getvalue())

    with Image.open(
        io.BytesIO(comparison_png(*frames, wrist_side="right"))
    ) as comparison:
        assert comparison.size == (16, 38)
        assert comparison.getpixel((3, 35)) == (128, 0, 128)
        assert comparison.getpixel((11, 35)) == (255, 165, 0)


def test_delivery_checks_pose_gripper_and_visual_payload(tmp_path: Path):
    runner = _runner(tmp_path)
    (tmp_path / "pos.yaml").write_text(
        "pose1:\n  pose:\n    position: {x: 1.0, y: 2.0, z: 0.0}\n"
        "    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}\n"
    )
    client = SimpleNamespace(
        snapshot=lambda: {
            "state": {
                "connected": True,
                "gripper_left": {"error": 1},
                "gripper_right": {"error": 1},
            },
            "base_world_pose": {
                "pose": {
                    "valid": True,
                    "frame_id": "camera_init",
                    "child_frame_id": "body",
                    "received_age_s": 0.02,
                    "position": {"x": 1.01, "y": 2.0, "z": 0.0},
                    "yaw_rad": 0.01,
                }
            },
        }
    )

    result = LunaPrimitives(runner=runner, client=client).pick_apriltag_module()

    assert not result.is_error
    assert result.data["arrival_ok"] is True
    assert result.data["left_gripper_ok"] is True
    assert result.data["payload_confirmed"] is True
    assert result.data["workflow_success"] is True
    assert result.data["error_code"] is None
    assert runner.completed

    snapshot = client.snapshot()
    client.snapshot = lambda: snapshot
    snapshot["base_world_pose"]["pose"]["position"]["x"] = 1.2
    off_target = LunaPrimitives(runner=runner, client=client).pick_apriltag_module()
    assert off_target.data["arrival_ok"] is False
    assert off_target.data["error_code"] == "destination_pose_unverified"
    assert not runner.completed

    snapshot["base_world_pose"]["pose"]["position"]["x"] = 1.0
    snapshot["state"]["gripper_left"]["error"] = 0
    gripper_off = LunaPrimitives(runner=runner, client=client).pick_apriltag_module()
    assert gripper_off.data["arrival_ok"] is True
    assert gripper_off.data["left_gripper_ok"] is False
    assert gripper_off.data["error_code"] == "left_gripper_not_ready"
    assert not runner.completed


def test_toolkit_exposes_bounded_workflow_and_records_state(tmp_path: Path):
    init_output_dir(tmp_path / "logs")
    runner = SimpleNamespace(
        completed=False,
        check_cancelled=lambda: None,
        move_to_table_3=lambda: ToolResult(
            data={"success": True, "workflow": "move_table_3", "log_path": "move.log"}
        ),
        pick_apriltag_module=lambda: ToolResult(
            data={
                "success": True,
                "workflow": "pick_to_chair_1",
                "log_path": "pick.log",
            }
        ),
    )
    client = SimpleNamespace(
        snapshot=lambda: {
            "state": {
                "connected": True,
                "gripper_left": {"error": 1},
                "gripper_right": {"error": 1},
            },
            "base_world_pose": {
                "pose": {
                    "valid": True,
                    "frame_id": "camera_init",
                    "child_frame_id": "body",
                    "received_age_s": 0.01,
                }
            },
        }
    )
    toolkit = LunaToolkit(
        runtime_kwargs={"runner": runner, "client": client},
        dashboard_events=SimpleNamespace(emit=lambda event: None),
        memory=MemoryManager(root=tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )

    names = {item.name for item in toolkit.list_tools()}
    assert {"inspect_luna", "move_to_table_3", "pick_apriltag_module"} <= names
    result = toolkit.execute_tool("move_to_table_3", {})

    assert not result.is_error
    assert result.data["inspection"]["ok"] is True
    assert "vlahost" not in result.data
    assert toolkit.state.latest_record().command == {"action": "move_to_table_3"}
    assert runner.check_cancelled == toolkit.raise_if_cancelled


def test_inspect_luna_rejects_stale_world_pose(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("robots.luna.tools.POSE_RECOVERY_TIMEOUT_S", 0)
    runner = SimpleNamespace(completed=False, check_cancelled=lambda: None)
    client = SimpleNamespace(
        snapshot=lambda: {
            "state": {
                "connected": True,
                "gripper_left": {"error": 1},
                "gripper_right": {"error": 1},
            },
            "base_world_pose": {
                "pose": {
                    "valid": True,
                    "frame_id": "camera_init",
                    "child_frame_id": "body",
                    "received_age_s": 1.0,
                }
            },
        }
    )
    toolkit = LunaToolkit(
        runtime_kwargs={"runner": runner, "client": client},
        dashboard_events=SimpleNamespace(emit=lambda event: None),
        memory=MemoryManager(root=tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )

    result = toolkit.execute_tool("inspect_luna", {})

    assert result.is_error
    assert result.data["ok"] is False
    assert result.data["pose_ok"] is False
    assert result.data["error"] == "Luna world-base pose is stale: 1.0s"


def test_inspect_luna_waits_for_pose_recovery(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("robots.luna.tools.time.sleep", lambda _: None)
    ages = iter((5.0, 1.0, 0.02))
    client = SimpleNamespace(
        snapshot=lambda: {
            "state": {
                "connected": True,
                "gripper_left": {"error": 1},
                "gripper_right": {"error": 1},
            },
            "base_world_pose": {
                "pose": {
                    "valid": True,
                    "frame_id": "camera_init",
                    "child_frame_id": "body",
                    "received_age_s": next(ages),
                }
            },
        }
    )
    runner = SimpleNamespace(check_cancelled=lambda: None)

    result = LunaPrimitives(runner=runner, client=client).inspect_luna()

    assert not result.is_error
    assert result.data["pose_age_s"] == 0.02


def test_inspect_luna_normalizes_dm_gripper_status(tmp_path: Path):
    runner = SimpleNamespace(completed=False, check_cancelled=lambda: None)
    client = SimpleNamespace(
        snapshot=lambda: {
            "state": {
                "connected": True,
                "joint_states": {"large_raw_state": "not returned"},
                "gripper_left": {"error": 1, "raw_error": [1]},
                "gripper_right": {"error": 1, "raw_error": [1]},
            },
            "base_world_pose": {
                "pose": {
                    "valid": True,
                    "frame_id": "camera_init",
                    "child_frame_id": "body",
                    "received_age_s": 0.04,
                }
            },
        }
    )
    toolkit = LunaToolkit(
        runtime_kwargs={"runner": runner, "client": client},
        dashboard_events=SimpleNamespace(emit=lambda event: None),
        memory=MemoryManager(root=tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )

    result = toolkit.execute_tool("inspect_luna", {})

    assert not result.is_error
    assert result.data == {
        "ok": True,
        "connected": True,
        "pose_ok": True,
        "pose_age_s": 0.04,
        "left_gripper": {"status_code": 1, "status": "on", "ok": True},
        "right_gripper": {"status_code": 1, "status": "on", "ok": True},
    }
