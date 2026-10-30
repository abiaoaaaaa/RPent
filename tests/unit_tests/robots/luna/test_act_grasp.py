# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Offline ACT color routing and guarded execution contracts."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from robots.luna.act import (
    PI05_PROMPTS,
    LunaActPrimitives,
    LunaActRunner,
    LunaClothesPoseRunner,
    LunaOpenPrimitives,
    instruction_declares_held_payload,
    instruction_destination_after_target,
    requested_color,
    requested_target,
    selected_act_model,
    selected_pi05_model,
    task_mode,
    validate_act_deployment,
    validate_pi05_deployment,
)
from robots.luna.robot_spec import (
    _init_runtime,
    _parse_config,
    _validate_args,
    get_robot_spec,
)
from robots.luna.toolkit import LunaToolkit
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolResult
from rpent.utils.logging import init_output_dir


def _configured(tmp_path: Path, *, action_dim: int = 22) -> tuple[Path, Path, Path]:
    names = (
        [f"Arm_L{i}_Joint" for i in range(1, 8)]
        + [f"Arm_R{i}_Joint" for i in range(1, 8)]
        + [f"Leg_J{i}_Joint" for i in range(1, 7)]
        + ["gripper_L", "gripper_R"]
    )
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    action_names = names if action_dim == 22 else names[:14] + names[20:]
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "action": {"names": action_names},
                    "observation.state": {"names": names},
                }
            }
        )
    )
    (dataset / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "action": {
                    "min": [-2.0] * (action_dim - 2) + [0.0, 0.0],
                    "max": [2.0] * (action_dim - 2) + [1.0, 1.0],
                }
            }
        )
    )
    model = tmp_path / "red_model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    "observation.state": {"shape": [22]},
                    "observation.velocity": {"shape": [22]},
                    **{
                        f"observation.images.{name}": {"shape": [3, 480, 640]}
                        for name in ("top", "wrist_L", "wrist_R")
                    },
                },
                "output_features": {"action": {"shape": [action_dim]}},
            }
        )
    )
    config = tmp_path / "red_deploy.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "policy": {"path": str(model)},
                "dataset": {"root": str(dataset)},
                "robot": {
                    "type": "luna_http",
                    "http_base_url": "http://robot:8011",
                    "joint_names": names,
                    "safety_stats_path": str(dataset),
                    "cameras": {
                        name: {"height": 480, "width": 640}
                        for name in ("top", "wrist_L", "wrist_R")
                    },
                },
                "inference": {
                    "strategy": "base",
                    "return_to_initial_position": False,
                    "include_velocity": True,
                },
            }
        )
    )
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "red": {"checkpoint": str(model), "deployment_config": str(config)},
                "blue": {"checkpoint": None, "deployment_config": None},
            }
        )
    )
    return manifest, model, config


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (1280, 2400), "white").save(output, format="JPEG")
    return output.getvalue()


def _configured_pi05(tmp_path: Path) -> tuple[Path, Path]:
    names = (
        [f"Arm_L{i}_Joint" for i in range(1, 8)]
        + [f"Arm_R{i}_Joint" for i in range(1, 8)]
        + [f"Leg_J{i}_Joint" for i in range(1, 7)]
        + ["gripper_L", "gripper_R"]
    )
    dataset = tmp_path / "rgb_dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(
        json.dumps({"features": {"action": {"names": names}}})
    )
    (dataset / "meta" / "stats.json").write_text("{}")
    model = tmp_path / "pi05_model"
    (model / "params").mkdir(parents=True)
    asset_id = "stack_blocks_rgb_v4_joint22_tplus2"
    (model / "assets" / asset_id).mkdir(parents=True)
    (model / "assets" / asset_id / "norm_stats.json").write_text("{}")
    config = tmp_path / "pi05.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "policy": {"path": str(model)},
                "openpi": {
                    "config_name": "pi05_luna_stack_blocks_rgb_v4_joint22_tplus2",
                    "asset_id": asset_id,
                    "action_mode": "joint",
                    "physical_action_dim": 22,
                    "observation": {
                        "image": "top",
                        "left_wrist_image": "wrist_L",
                        "right_wrist_image": "wrist_R",
                    },
                },
                "dataset": {"root": str(dataset)},
                "robot": {
                    "type": "luna_http",
                    "http_base_url": "http://robot:8011",
                    "joint_names": names,
                    "safety_stats_path": str(dataset),
                },
                "recording": {"enabled": False},
                "inference": {
                    "type": "chunk",
                    "strategy": "base",
                    "return_to_initial_position": False,
                },
            }
        )
    )
    return model, config


@pytest.mark.parametrize(
    "instruction,color",
    [
        ("抓取红色方块并抬起收回", "red"),
        ("抓取蓝色方块并抬起收回", "blue"),
    ],
)
def test_act_task_selects_color_from_language_and_exposes_staging(
    tmp_path: Path,
    instruction: str,
    color: str,
):
    spec = get_robot_spec()
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(["--task-id", "4", "--luna-instruction", instruction])
    args.output_dir = str(tmp_path)
    config = _parse_config(args)
    assert config.prompt_vars["mode"] == "act"
    assert config.prompt_vars["instruction"] == instruction
    assert config.task_desc["instruction"] == instruction
    assert requested_color(instruction) == color
    prompt = spec.prompts.render("system", variables=config.prompt_vars)
    assert "observe_scene" in prompt and "vla_grasp" in prompt
    assert "move_delta" in prompt and "rotate_delta" in prompt
    assert "at most 200 mm vector norm" in prompt
    assert "VISIBLE DECISION LOG" in prompt
    assert "决策：阶段=" in prompt
    assert "观察=" in prompt and "判断=" in prompt and "验证=" in prompt
    assert "After each tool result" in prompt
    assert "阶段验证成功：<阶段>" in prompt
    assert "Use Simplified Chinese for every operator-visible answer" in prompt
    assert "blocking grasp call" in prompt
    assert "call prepare_grasp_retry immediately" in prompt
    assert "Task 4 is a colored-block transport task" in prompt
    assert "canonical destination `chair_1`" in prompt
    assert "Red uses the left gripper and EEF; blue uses the right gripper and EEF" in prompt
    assert "Post-grasp EEF retract belongs only" in prompt

    client = SimpleNamespace(
        base_url="http://robot:8011",
        get=lambda path: {"connected": True},
        camera_jpeg=_jpeg,
    )
    runner = SimpleNamespace(
        execute=False,
        completed=False,
        attempted=False,
        check_cancelled=lambda: None,
        stop=lambda: None,
        run=lambda _color: pytest.fail("policy started"),
    )
    navigation = SimpleNamespace(
        check_cancelled=lambda: None,
        available_locations=lambda: ["chair_1", "chair_2", "table_3"],
        navigate=lambda _location: pytest.fail("navigation started"),
        verify=lambda _location: pytest.fail("navigation verification started"),
        stop=lambda: None,
    )
    session = SimpleNamespace(
        check_cancelled=lambda: None,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        execute=False,
        call=lambda action, **_kwargs: (
            {"ok": True, "xyz_m": [0.3, 0.2, 1.0]}
            if action == "state"
            else pytest.fail("motion started")
        ),
        close=lambda: None,
    )
    toolkit = LunaToolkit(
        runtime_kwargs={
            "mode": "act",
            "client": client,
            "runner": runner,
            "session": session,
            "navigation": navigation,
            "instruction": instruction,
        },
        dashboard_events=SimpleNamespace(emit=lambda _event: None),
        memory=MemoryManager(root=tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )
    init_output_dir(tmp_path / "output")
    names = {tool.name for tool in toolkit.list_tools()}
    assert {
        "observe_scene",
        "move_delta",
        "rotate_delta",
        "open_gripper",
        "close_gripper",
        "vla_grasp",
        "list_locations",
        "navigate_to",
        "verify_location",
        "prepare_grasp_retry",
        "verify_goal",
        "verify_delivery",
    } <= names
    assert "move_to_table_3" not in names
    result = toolkit.execute_tool("observe_scene", {})
    assert not result.is_error and result.data["target_instruction"] == instruction
    assert result.data["motion_enabled"] is False
    assert result.data["act_enabled"] is False
    toolkit.close()


def test_placeholder_cannot_enable_motion(tmp_path: Path):
    manifest, model, config = _configured(tmp_path)
    assert selected_act_model(manifest, "blue", required=False) is None
    with pytest.raises(ValueError, match="not configured"):
        selected_act_model(manifest, "blue", required=True)
    assert selected_act_model(manifest, "red", required=True) == (model, config)
    validate_act_deployment(model, config, "http://robot:8011")
    deployment = yaml.safe_load(config.read_text())
    deployment["inference"]["return_to_initial_position"] = True
    config.write_text(yaml.safe_dump(deployment))
    with pytest.raises(ValueError, match="automatic Home"):
        validate_act_deployment(model, config, "http://robot:8011")

    entries = yaml.safe_load(manifest.read_text())
    entries["blue"] = entries["red"]
    manifest.write_text(yaml.safe_dump(entries))
    with pytest.raises(ValueError, match="cannot share a checkpoint"):
        selected_act_model(manifest, "blue", required=True)
    entries["blue"] = {"checkpoint": None, "deployment_config": None}
    manifest.write_text(yaml.safe_dump(entries))

    parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(parser, False)
    args = parser.parse_args(
        [
            "--task-id",
            "4",
            "--luna-instruction",
            "抓取蓝色方块并抬起收回",
            "--luna-act-models",
            str(manifest),
            "--luna-act-execute",
            "--luna-eef-execute",
        ]
    )
    with pytest.raises(ValueError, match="blue model is not configured"):
        _validate_args(args)


def test_arm16_act_deployment_matches_22d_observation_contract(tmp_path: Path):
    _, model, config = _configured(tmp_path, action_dim=16)
    validate_act_deployment(model, config, "http://robot:8011")

    deployment = yaml.safe_load(config.read_text())
    dataset = Path(deployment["dataset"]["root"])
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["names"] = info["features"]["observation.state"]["names"]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="action/state order"):
        validate_act_deployment(model, config, "http://robot:8011")


def test_yellow_is_pi05_only_task_four_target(tmp_path: Path):
    manifest, _, _ = _configured(tmp_path)
    assert requested_color("抓取黄色方块并抬起收回") == "yellow"
    assert requested_color("抓取红色和黄色方块") is None
    with pytest.raises(ValueError, match="unsupported ACT target color"):
        selected_act_model(manifest, "yellow", required=True)
    parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(parser, False)
    args = parser.parse_args(
        ["--task-id", "4", "--luna-instruction", "抓取黄色方块并抬起收回"]
    )
    with pytest.raises(ValueError, match="unsupported ACT target color"):
        _validate_args(args)


def test_pi05_three_colors_share_one_checkpoint_and_exact_prompt(tmp_path: Path):
    model, config = _configured_pi05(tmp_path)
    assert PI05_PROMPTS == {
        "red": "Pick up the red block.",
        "yellow": "Pick up the yellow block.",
        "blue": "Pick up the blue block.",
    }
    assert selected_pi05_model(config, required=True) == (model, config)
    validate_pi05_deployment(model, config, "http://robot:8011")
    script = tmp_path / "deploy.py"
    script.write_text(
        "import sys\n"
        "assert sys.argv[sys.argv.index('--model') + 1] == 'openpi'\n"
        "assert sys.argv[sys.argv.index('--task') + 1] == 'Pick up the yellow block.'\n"
    )
    runner = LunaActRunner(
        color="yellow",
        manifest=tmp_path / "unused_act_manifest.yaml",
        client=SimpleNamespace(
            base_url="http://robot:8011",
            get=lambda _path: {"connected": True},
            camera_jpeg=_jpeg,
        ),
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=tmp_path / "output",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
        backend="pi05",
        pi05_config=config,
    )
    result = runner.run(verify=False)
    assert not result.is_error
    assert result.data["policy_completed"] is True


def test_22d_act_cannot_use_marvin_http_driver(tmp_path: Path):
    _, model, config = _configured(tmp_path)
    deployment = yaml.safe_load(config.read_text())
    deployment["robot"]["type"] = "marvain_m6_http"
    config.write_text(yaml.safe_dump(deployment))
    with pytest.raises(ValueError, match="robot type"):
        validate_act_deployment(model, config, "http://robot:8011")


def test_act_rollout_is_bounded_one_shot_and_verdict_gated(tmp_path: Path):
    manifest, _, _ = _configured(tmp_path)
    script = tmp_path / "deploy.py"
    script.write_text(
        "import sys\nassert '--model' in sys.argv and 'act' in sys.argv\nprint('offline fake rollout')\n"
    )
    client = SimpleNamespace(
        base_url="http://robot:8011",
        get=lambda _path: {"connected": True},
        camera_jpeg=_jpeg,
    )
    runner = LunaActRunner(
        color="red",
        manifest=manifest,
        client=client,
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=tmp_path / "output",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
        visual_verifier=lambda _image, **_kwargs: {
            "verdict": "not_held",
            "evidence": "empty jaws",
        },
    )
    result = runner.run()
    assert result.is_error
    assert result.data["error_code"] == "not_held"
    assert result.data["payload_confirmed"] is False
    assert Path(result.data["log_path"]).is_file()
    assert runner.run().data["error_code"] == "already_attempted"
    runner.stop()

    held = LunaActRunner(
        color="red",
        manifest=manifest,
        client=client,
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=tmp_path / "held",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
        visual_verifier=lambda _image, **_kwargs: {
            "verdict": "held",
            "evidence": "red block between jaws",
        },
    )
    success = held.run()
    assert not success.is_error
    assert success.data["workflow_success"] is True
    assert held.completed is True


def test_language_grasp_handoffs_before_verdict_and_skips_retract(tmp_path: Path):
    manifest, _, _ = _configured(tmp_path)
    script = tmp_path / "deploy.py"
    script.write_text("print('offline ACT completed')\n")
    events: list[str] = []
    client = SimpleNamespace(
        base_url="http://robot:8011",
        get=lambda path: {"gripper_left": {"error": 1}} if path == "/state" else {},
        camera_jpeg=_jpeg,
    )
    session = SimpleNamespace(
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **_kwargs: (
            events.append(action)
            or (
                {"ok": True, "xyz_m": session.xyz}
                if action == "state"
                else {"ok": True}
            )
        ),
        close=lambda: events.append("close_session"),
    )
    session.xyz = [0.3, 0.2, 1.0]
    runner = LunaActRunner(
        color=None,
        manifest=manifest,
        client=client,
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=tmp_path / "output",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
        visual_verifier=lambda _image, **_kwargs: (
            events.append("verdict")
            or {"verdict": "held", "evidence": "red block between jaws"}
        ),
    )
    navigation = SimpleNamespace(
        navigate=lambda location: ToolResult(
            data={"success": True, "arrival_ok": True, "location": location}
        ),
        verify=lambda location: ToolResult(
            data={"success": True, "arrival_ok": True, "location": location}
        ),
    )
    primitives = LunaActPrimitives(
        runner=runner,
        client=client,
        session=session,
        navigation=navigation,
        instruction="抓取红色方块并抬起收回",
    )
    assert not primitives.observe_scene().is_error
    assert primitives.vla_grasp("blue").is_error
    assert not runner.attempted
    attempt = primitives.vla_grasp("red")
    assert not attempt.is_error
    assert attempt.data["workflow_success"] is False
    assert (
        events.index("close_session")
        < events.index("claim_eef_control")
        < events.index("hold_closed")
        < events.index("verdict")
    )
    assert primitives.open_gripper().is_error
    assert attempt.data["requires_retract"] is False
    assert attempt.data["return_xyz_m"] is None
    assert primitives.grasp_retracted is True
    assert not primitives.completed
    assert not primitives.navigate_to("chair_1").is_error
    delivered = primitives.verify_delivery("chair_1")
    assert delivered.data["workflow_success"] is True
    assert primitives.completed


def test_blue_uses_right_gripper_and_skips_right_eef_retract(tmp_path: Path):
    manifest, red_model, red_config = _configured(tmp_path)
    blue_model = tmp_path / "blue_model"
    blue_model.mkdir()
    (blue_model / "config.json").write_text((red_model / "config.json").read_text())
    blue_config = tmp_path / "blue_deploy.yaml"
    deployment = yaml.safe_load(red_config.read_text())
    deployment["policy"]["path"] = str(blue_model)
    blue_config.write_text(yaml.safe_dump(deployment))
    entries = yaml.safe_load(manifest.read_text())
    entries["blue"] = {
        "checkpoint": str(blue_model),
        "deployment_config": str(blue_config),
    }
    manifest.write_text(yaml.safe_dump(entries))
    script = tmp_path / "deploy.py"
    script.write_text("print('offline ACT completed')\n")
    events: list[tuple[str, str | None]] = []
    session = SimpleNamespace(
        side="right",
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **kwargs: (
            events.append((action, kwargs.get("side")))
            or (
                {"ok": True, "xyz_m": [0.3, 0.2, 1.0]}
                if action == "state"
                else {"ok": True}
            )
        ),
        close=lambda: events.append(("close_session", None)),
    )
    client = SimpleNamespace(
        base_url="http://robot:8011",
        get=lambda path: (
            {"gripper_right": {"error": 1}} if path == "/state" else {}
        ),
        camera_jpeg=_jpeg,
    )
    runner = LunaActRunner(
        color=None,
        manifest=manifest,
        client=client,
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=tmp_path / "output",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
        visual_verifier=lambda _image, **kwargs: {
            "verdict": "held",
            "evidence": f"block held in {kwargs['wrist_side']} gripper",
        },
    )
    navigation = SimpleNamespace(
        navigate=lambda location: ToolResult(
            data={"success": True, "arrival_ok": True, "location": location}
        ),
        verify=lambda location: ToolResult(
            data={"success": True, "arrival_ok": True, "location": location}
        ),
    )
    primitives = LunaActPrimitives(
        runner=runner,
        client=client,
        session=session,
        navigation=navigation,
        instruction="去3号桌子拿蓝色方块去2号椅子",
    )

    assert not primitives.observe_scene().is_error
    opened = primitives.open_gripper()
    assert opened.data["gripper_side"] == "right"
    assert events[-1] == ("open", "right")
    grasped = primitives.vla_grasp("blue")
    assert not grasped.is_error
    assert grasped.data["gripper_side"] == "right"
    assert grasped.data["requires_retract"] is False
    assert grasped.data["return_xyz_m"] is None
    assert primitives.grasp_retracted is True
    assert ("hold_closed", "right") in events
    assert not primitives.navigate_to("chair_2").is_error
    delivered = primitives.verify_delivery("chair_2")
    assert delivered.data["right_gripper_ok"] is True
    assert delivered.data["gripper_released"] is True
    assert delivered.data["workflow_success"] is True
    assert events.count(("open", "right")) == 2


def test_clothes_exits_policy_for_base_transport_after_rollout(tmp_path: Path):
    manifest, model, config = _configured(tmp_path, action_dim=22)
    entries = yaml.safe_load(manifest.read_text())
    clothes_model = tmp_path / "clothes_model"
    clothes_model.mkdir()
    (clothes_model / "config.json").write_text((model / "config.json").read_text())
    clothes_config = tmp_path / "clothes_deploy.yaml"
    clothes_deployment = yaml.safe_load(config.read_text())
    clothes_deployment["policy"]["path"] = str(clothes_model)
    clothes_config.write_text(yaml.safe_dump(clothes_deployment))
    entries["clothes"] = {
        "checkpoint": str(clothes_model),
        "deployment_config": str(clothes_config),
    }
    manifest.write_text(yaml.safe_dump(entries))
    script = tmp_path / "deploy.py"
    script.write_text("print('offline clothes ACT completed')\n")
    events: list[str] = []
    session = SimpleNamespace(
        side="right",
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **_kwargs: (
            events.append(action)
            or ({"ok": True, "xyz_m": [0.3, 0.2, 1.0]} if action == "state" else {"ok": True})
        ),
        close=lambda: events.append("close_session"),
    )
    client = SimpleNamespace(
        base_url="http://robot:8011",
        get=lambda _path: {"connected": True},
        camera_jpeg=_jpeg,
    )
    runner = LunaActRunner(
        color=None,
        manifest=manifest,
        client=client,
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=tmp_path / "output",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
        visual_verifier=lambda _image, **_kwargs: {
            "verdict": "held",
            "evidence": "cloth held by the right gripper",
        },
    )
    primitives = LunaActPrimitives(
        runner=runner,
        client=client,
        session=session,
        navigation=SimpleNamespace(),
        instruction="去3号桌子抓衣服",
    )

    assert not primitives.observe_scene().is_error
    result = primitives.vla_grasp("clothes")

    assert not result.is_error
    assert result.data["workflow_success"] is True
    assert result.data["control_handoff"]["mode"] == "policy_exit_base_transport"
    assert "claim_eef_control" not in events
    assert "hold_closed" not in events
    assert primitives.completed is True
    verified = primitives.verify_goal()
    assert verified.data["workflow_success"] is True


def test_clothes_uses_direct_joint_pose_sequence_instead_of_act(tmp_path: Path):
    args_path = tmp_path / "args.txt"
    script = tmp_path / "clothes_pose_sequence_test.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"Path({str(args_path)!r}).write_text(' '.join(sys.argv[1:]))\n"
        "print('PASS: clothes pose sequence completed; gripper remains closed.')\n"
    )
    ros_setup = tmp_path / "ros_setup.sh"
    ros_setup.write_text("#!/usr/bin/env bash\n")
    runner = LunaClothesPoseRunner(
        script=script,
        python_executable=sys.executable,
        ros_setup_script=ros_setup,
        output_dir=tmp_path / "output",
        execute=True,
        timeout_s=5.0,
        gripper_close_duration_s=2.0,
        check_cancelled=lambda: None,
    )

    result = runner.run("clothes")

    assert not result.is_error
    assert result.data["script_success"] is True
    assert result.data["workflow_success"] is False
    argv = args_path.read_text()
    assert "--execute --home-mode direct" in argv
    assert "--gripper-side right --gripper-close-duration-s 2.0" in argv
    assert runner.run("clothes").data["error_code"] == "already_attempted"


def test_clothes_vla_grasp_routes_to_direct_pose_runner(tmp_path: Path):
    script = tmp_path / "clothes_pose_sequence_test.py"
    script.write_text("print('PASS')\n")
    ros_setup = tmp_path / "ros_setup.sh"
    ros_setup.write_text("#!/usr/bin/env bash\n")
    runner = LunaClothesPoseRunner(
        script=script,
        python_executable=sys.executable,
        ros_setup_script=ros_setup,
        output_dir=tmp_path / "output",
        execute=True,
        timeout_s=5.0,
        gripper_close_duration_s=2.0,
        check_cancelled=lambda: None,
        visual_verifier=lambda _image, **_kwargs: {
            "verdict": "held",
            "evidence": "cloth held by the right gripper",
        },
    )
    session = SimpleNamespace(
        side="right",
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **_kwargs: (
            {"ok": True, "xyz_m": [0.3, 0.2, 1.0]}
            if action == "state"
            else {"ok": True}
        ),
        close=lambda: None,
    )
    primitives = LunaActPrimitives(
        runner=runner,
        client=SimpleNamespace(
            base_url="http://robot:8011",
            get=lambda _path: {"connected": True},
            camera_jpeg=_jpeg,
        ),
        session=session,
        navigation=SimpleNamespace(),
        instruction="去3号桌子抓衣服",
    )

    assert not primitives.observe_scene().is_error
    result = primitives.vla_grasp("clothes")

    assert not result.is_error
    assert result.data["workflow_success"] is True
    assert result.data["control_handoff"]["mode"] == "direct_joint_home_hold"
    assert runner.attempt_count == 1


def test_luna_state_capture_outage_does_not_rewrite_completed_tool(tmp_path: Path):
    class UnavailableClient:
        def get(self, _path):
            raise RuntimeError("cannot read VLAHost /state: timed out")

    toolkit = object.__new__(LunaToolkit)
    toolkit._act_mode = True
    toolkit._eef_mode = False
    toolkit._open_mode = False
    toolkit._primitives = SimpleNamespace(
        client=UnavailableClient(),
        completed=True,
    )
    toolkit._state = EnvState(tmp_path / "state")

    result = toolkit.get_env_state(
        command={"action": "navigate_to"},
        result={
            "success": True,
            "arrival_ok": True,
            "location": "table_3",
        },
        elapsed_s=1.0,
    )

    assert not result.is_error
    assert result.data["arrival_ok"] is True
    assert "timed out" in result.data["state_capture_error"]


def test_clothes_prompt_requires_direct_joint_vla_grasp():
    spec = get_robot_spec()
    prompt = spec.prompts.render(
        "system",
        variables={
            "mode": "act",
            "instruction": "去3号桌子抓衣服",
            "destination": "tabletop",
            "task_id": 4,
        },
    )
    assert "Clothes does not use ACT" in prompt
    assert "control mode 3 (joint impedance)" in prompt
    assert "must not request position mode 1" in prompt
    assert "opens both grippers before prepare" in prompt
    assert "closes the right gripper for the configured short duration" in prompt
    assert "vla_grasp(color=clothes)" in prompt
    assert "not retryable" in prompt


def test_language_grasp_stops_when_eef_source_handoff_fails(tmp_path: Path):
    manifest, _, _ = _configured(tmp_path)
    script = tmp_path / "deploy.py"
    script.write_text("print('offline ACT completed')\n")
    events: list[str] = []
    session = SimpleNamespace(
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **_kwargs: (
            events.append(action)
            or (
                {"ok": True, "xyz_m": [0.3, 0.2, 1.0]}
                if action == "state"
                else {"ok": False, "error": "mux stayed on source 2"}
                if action == "claim_eef_control"
                else {"ok": True}
            )
        ),
        close=lambda: events.append("close_session"),
    )
    runner = LunaActRunner(
        color=None,
        manifest=manifest,
        client=SimpleNamespace(
            base_url="http://robot:8011",
            get=lambda _path: {"connected": True},
            camera_jpeg=_jpeg,
        ),
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=tmp_path / "output",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
    )
    primitives = LunaActPrimitives(
        runner=runner,
        client=runner.client,
        session=session,
        navigation=SimpleNamespace(),
        instruction="去3号桌子拿红色方块去1号椅子",
    )

    assert not primitives.observe_scene().is_error
    result = primitives.vla_grasp("red")

    assert result.data["error_code"] == "control_handoff_failed"
    assert result.data["retryable"] is False
    assert events[-2:] == ["close_session", "claim_eef_control"]


def test_not_held_allows_one_explicit_high_pose_retry(tmp_path: Path):
    manifest, _, _ = _configured(tmp_path)
    script = tmp_path / "deploy.py"
    script.write_text("print('offline ACT completed')\n")
    verdicts = iter(
        (
            {"verdict": "not_held", "evidence": "empty jaws"},
            {"verdict": "held", "evidence": "red block between jaws"},
        )
    )
    client = SimpleNamespace(
        base_url="http://robot:8011",
        get=lambda _path: {"connected": True},
        camera_jpeg=_jpeg,
    )
    session = SimpleNamespace(
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **_kwargs: (
            {"ok": True, "xyz_m": [0.3, 0.2, 1.0]}
            if action == "state"
            else {"ok": True}
        ),
        close=lambda: None,
    )
    runner = LunaActRunner(
        color=None,
        manifest=manifest,
        client=client,
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=tmp_path / "output",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
        visual_verifier=lambda _image, **_kwargs: next(verdicts),
    )
    primitives = LunaActPrimitives(
        runner=runner,
        client=client,
        session=session,
        navigation=SimpleNamespace(),
        instruction="去3号桌子拿红色方块去1号椅子",
    )

    assert not primitives.observe_scene().is_error
    first = primitives.vla_grasp("red")
    assert first.data["error_code"] == "not_held"
    assert first.data["retryable"] is True
    assert first.data["next_action"] == "prepare_grasp_retry"
    prepared = primitives.prepare_grasp_retry()
    assert not prepared.is_error and prepared.data["remaining_attempts"] == 1
    second = primitives.vla_grasp("red")
    assert not second.is_error
    assert second.data["attempts"] == 2
    assert runner.can_attempt() is False


def test_task_four_verify_goal_does_not_create_a_second_retract_verdict(tmp_path: Path):
    manifest, _, _ = _configured(tmp_path)
    script = tmp_path / "deploy.py"
    script.write_text("print('offline ACT completed')\n")
    verdicts = iter(
        (
            {"verdict": "held", "evidence": "block appears between jaws"},
            {"verdict": "not_held", "evidence": "block remains on table"},
            {"verdict": "held", "evidence": "block remains between jaws"},
            {"verdict": "not_held", "evidence": "block dropped after retry"},
        )
    )
    client = SimpleNamespace(
        base_url="http://robot:8011",
        get=lambda _path: {"connected": True},
        camera_jpeg=_jpeg,
    )
    session = SimpleNamespace(
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **_kwargs: (
            {"ok": True, "xyz_m": [0.3, 0.2, 1.0]}
            if action == "state"
            else {"ok": True}
        ),
        close=lambda: None,
    )
    runner = LunaActRunner(
        color=None,
        manifest=manifest,
        client=client,
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=tmp_path / "output",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
        visual_verifier=lambda _image, **_kwargs: next(verdicts),
    )
    primitives = LunaActPrimitives(
        runner=runner,
        client=client,
        session=session,
        navigation=SimpleNamespace(),
        instruction="去3号桌子拿红色方块去1号椅子",
    )

    assert not primitives.observe_scene().is_error
    first = primitives.vla_grasp("red")
    assert not first.is_error and runner.completed is True

    verified = primitives.verify_goal()
    assert verified.data["grasp_retracted"] is True
    assert verified.data["requires_retract"] is False
    assert runner.completed is True
    assert primitives.last_verdict == "held"


def test_ambiguous_or_missing_color_is_rejected():
    assert requested_color("抓取红色和蓝色方块") is None
    assert requested_color("抓取桌上的方块") is None


def test_clothes_target_routes_to_act_without_color_inference():
    instruction = "去3号桌子抓衣服"
    assert requested_color(instruction) is None
    assert requested_target(instruction) == "clothes"
    assert task_mode(instruction) == "pick_and_transport"
    assert not instruction_destination_after_target(instruction, "chair_2")
    assert instruction_destination_after_target("去3号桌子抓衣服放到2号椅子上", "chair_2")
    assert instruction_destination_after_target(
        "去3号桌子抓衣服放到沙发椅上", "chair_1"
    )
    assert not instruction_destination_after_target(
        "去3号桌子抓衣服放到沙发椅上", "chair_2"
    )


def test_clothes_transport_releases_right_gripper_on_final_navigation():
    events: list[tuple[str, str | None]] = []
    session = SimpleNamespace(
        side="right",
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **kwargs: events.append((action, kwargs.get("side")))
        or {"ok": True},
        close=lambda: None,
    )
    runner = SimpleNamespace(completed=True)
    navigation = SimpleNamespace(
        navigate=lambda location: ToolResult(
            data={"success": True, "arrival_ok": True, "location": location}
        )
    )
    primitives = LunaActPrimitives(
        runner=runner,
        client=SimpleNamespace(camera_jpeg=_jpeg),
        session=session,
        navigation=navigation,
        instruction="去3号桌子抓衣服放到2号椅子上",
    )

    result = primitives.navigate_to("chair_2")

    assert not result.is_error
    assert result.data["workflow_success"] is True
    assert result.data["gripper_released"] is True
    assert events == [("claim_eef_control", None), ("open", "right")]
    assert primitives.completed is True


def test_clothes_transport_does_not_release_at_source():
    events: list[str] = []
    session = SimpleNamespace(
        side="right",
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **_kwargs: events.append(action) or {"ok": True},
        close=lambda: None,
    )
    runner = SimpleNamespace(completed=True)
    navigation = SimpleNamespace(
        navigate=lambda location: ToolResult(
            data={"success": True, "arrival_ok": True, "location": location}
        )
    )
    primitives = LunaActPrimitives(
        runner=runner,
        client=SimpleNamespace(camera_jpeg=_jpeg),
        session=session,
        navigation=navigation,
        instruction="去3号桌子抓衣服放到2号椅子上",
    )

    result = primitives.navigate_to("table_3")

    assert not result.is_error
    assert "gripper_released" not in result.data
    assert events == []


def test_clothes_manifest_can_be_declared_before_checkpoint_is_ready(tmp_path: Path):
    spec = get_robot_spec()
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(
        ["--task-id", "4", "--luna-instruction", "去3号桌子抓衣服"]
    )
    # The local training job creates the checkpoint later. Read-only startup
    # must remain usable, while execute=True still fails closed in preflight.
    args.output_dir = str(tmp_path)
    _validate_args(args)
    config = _parse_config(args)
    assert config.prompt_vars["mode"] == "act"


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("去3号桌子拿蓝色方块去1号椅子", "pick_and_transport"),
        ("已经抓到红色方块，先去2号椅子再去1号椅子", "transport_held_object"),
        ("当前持有黄色方块，去3号桌子", "transport_held_object"),
        ("右夹爪里有蓝色方块，去1号椅子", "transport_held_object"),
        ("把右臂沿x正方向移动5厘米", "open_ended"),
    ],
)
def test_task_mode_classifies_initial_payload_state_not_route(instruction: str, expected: str):
    assert task_mode(instruction) == expected


def test_declared_held_transport_forbids_regrasp_and_releases_at_final_station(tmp_path: Path):
    manifest, _, _ = _configured(tmp_path)
    script = tmp_path / "deploy.py"
    script.write_text("print('policy must not start')\n")
    events: list[tuple[str, str | None]] = []
    session = SimpleNamespace(
        side="right",
        execute=True,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        call=lambda action, **kwargs: events.append((action, kwargs.get("side")))
        or ({"ok": True, "xyz_m": [0.3, -0.2, 1.0]} if action == "state" else {"ok": True}),
        close=lambda: None,
    )
    client = SimpleNamespace(
        base_url="http://robot:8011",
        get=lambda path: {"gripper_right": {"error": 1}} if path == "/state" else {},
        camera_jpeg=_jpeg,
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    runner = LunaActRunner(
        color=None,
        manifest=manifest,
        client=client,
        deploy_script=script,
        python_executable=sys.executable,
        output_dir=output_dir,
        execute=False,
        duration_s=0.2,
        check_cancelled=lambda: None,
        visual_verifier=lambda _image, **_kwargs: {"verdict": "held", "evidence": "blue block between jaws"},
    )
    navigation = SimpleNamespace(
        available_locations=lambda: ["chair_2", "table_3"],
        navigate=lambda location: ToolResult(data={"success": True, "arrival_ok": True, "location": location}),
        verify=lambda location: ToolResult(data={"success": True, "arrival_ok": True, "location": location}),
    )
    instruction = "你已经拿到了蓝色方块现在位置是2号椅子，直接去3号桌子"
    assert instruction_declares_held_payload(instruction)
    primitives = LunaActPrimitives(
        runner=runner,
        client=client,
        session=session,
        navigation=navigation,
        instruction=instruction,
        initially_holding=True,
    )

    assert primitives.list_locations().data["task_mode"] == "transport_held_object"
    assert primitives.open_gripper().is_error
    assert primitives.vla_grasp("blue").is_error
    assert primitives.move_delta(10, 0, 0).is_error
    assert not primitives.observe_scene().is_error
    assert not primitives.navigate_to("table_3").is_error
    delivered = primitives.verify_delivery("table_3")
    assert delivered.data["workflow_success"] is True
    assert delivered.data["gripper_released"] is True
    assert events == [("state", None), ("open", "right")]


def test_task_four_colorless_instruction_uses_open_tool_prompt(tmp_path: Path):
    spec = get_robot_spec()
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    instruction = "把右臂的eef沿x正方向移动5厘米"
    args = parser.parse_args(["--task-id", "4", "--luna-instruction", instruction])
    args.output_dir = str(tmp_path)
    config = _parse_config(args)

    assert config.prompt_vars["mode"] == "open"
    prompt = spec.prompts.render("system", variables=config.prompt_vars)
    assert "open-ended Luna instruction" in prompt
    assert "select_eef_side(left|right)" in prompt
    assert "never substitute the opposite arm" in prompt


def test_open_mode_selects_right_eef_then_locks_after_motion():
    calls = []
    session = SimpleNamespace(
        side="left",
        table_z_m=0.8,
        tool_clearance_m=0.06,
        execute=True,
        close=lambda: calls.append(("close_session", {})),
        call=lambda action, **kwargs: calls.append((action, kwargs))
        or ({"ok": True, "xyz_m": [0.3, -0.2, 1.0]} if action == "state" else {"ok": True}),
    )
    primitives = LunaOpenPrimitives(
        client=SimpleNamespace(camera_jpeg=_jpeg),
        session=session,
        navigation=SimpleNamespace(),
        instruction="把右臂的eef沿base_link的x正方向移动5厘米",
    )

    selected = primitives.select_eef_side("right")
    assert selected.data == {"success": True, "eef_side": "right", "changed": True}
    assert session.side == "right"
    observed = primitives.observe_scene()
    assert observed.data["eef_side"] == "right"
    moved = primitives.move_eef_delta(50, 0, 0)
    assert not moved.is_error
    assert ("move", {"delta_mm": [50, 0, 0]}) in calls
    locked = primitives.select_eef_side("left")
    assert locked.is_error
    assert "locked" in locked.data["error"]


def test_open_mode_side_selection_does_not_require_fresh_base_world_pose(
    tmp_path: Path,
):
    client = SimpleNamespace(
        get=lambda path: {"connected": True} if path == "/state" else pytest.fail(path),
        snapshot=lambda: pytest.fail("open EEF tools must not capture base pose"),
        camera_jpeg=_jpeg,
    )
    session = SimpleNamespace(
        side="left",
        table_z_m=0.8,
        tool_clearance_m=0.06,
        execute=True,
        check_cancelled=lambda: None,
        close=lambda: None,
    )
    navigation = SimpleNamespace(check_cancelled=lambda: None, stop=lambda: None)
    toolkit = LunaToolkit(
        runtime_kwargs={
            "mode": "open",
            "client": client,
            "session": session,
            "navigation": navigation,
            "instruction": "把左臂EEF沿X正方向移动5厘米",
        },
        dashboard_events=SimpleNamespace(emit=lambda _event: None),
        memory=MemoryManager(root=tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )
    init_output_dir(tmp_path / "output")
    result = toolkit.execute_tool("select_eef_side", {"side": "left"})
    assert result.data["success"] is True
    assert "inspection" not in result.data
    toolkit.close()


def test_task_four_preserves_virtualenv_python_symlink(tmp_path: Path):
    launcher = tmp_path / "policy-venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    (script_dir / "pos.yaml").write_text(
        "pose3:\n"
        "  station: table_3\n"
        "  pose:\n"
        "    position: {x: 0.0, y: 0.0, z: 0.0}\n"
        "    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}\n"
    )
    parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(parser, False)
    args = parser.parse_args(
        [
            "--task-id",
            "4",
            "--luna-instruction",
            "去3号桌子拿红色方块去1号椅子",
            "--luna-script-dir",
            str(script_dir),
            "--luna-act-python",
            str(launcher),
        ]
    )

    _, runtime = _init_runtime(
        args, tmp_path / "output", SimpleNamespace(), components=None
    )

    assert runtime["runner"].python_executable == str(launcher.absolute())
    assert runtime["runner"].python_executable != str(launcher.resolve())


def test_policy_child_uses_matching_rollout_and_local_lerobot_source(tmp_path: Path):
    manifest, _, _ = _configured(tmp_path)
    policy_bin = tmp_path / "policy-env" / "bin"
    policy_bin.mkdir(parents=True)
    policy_python = policy_bin / "python"
    policy_python.symlink_to(sys.executable)
    rollout = policy_bin / "lerobot-rollout"
    rollout.write_text("#!/bin/sh\nexit 0\n")
    rollout.chmod(0o755)
    deploy_repo = tmp_path / "deploy-repo"
    (deploy_repo / "src").mkdir(parents=True)
    deploy_script = deploy_repo / "workflows" / "robot_interaction" / "deploy.py"
    deploy_script.parent.mkdir(parents=True)
    deploy_script.write_text(
        "import os, pathlib\n"
        f"assert os.environ['LEROBOT_ROLLOUT'] == {str(rollout)!r}\n"
        f"assert os.environ['PYTHONPATH'].split(os.pathsep)[0] == {str(deploy_repo / 'src')!r}\n"
    )
    runner = LunaActRunner(
        color="red",
        manifest=manifest,
        client=SimpleNamespace(
            base_url="http://robot:8011",
            get=lambda _path: {"connected": True},
            camera_jpeg=_jpeg,
        ),
        deploy_script=deploy_script,
        python_executable=str(policy_python),
        output_dir=tmp_path / "output",
        execute=True,
        duration_s=0.2,
        check_cancelled=lambda: None,
    )

    result = runner.run(verify=False)

    assert not result.is_error
