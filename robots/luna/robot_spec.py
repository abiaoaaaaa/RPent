# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Luna RobotSpec and runtime wiring."""

from __future__ import annotations

import argparse
import functools
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from robots.luna.act import requested_target, task_mode
from robots.luna.client import LunaVLAHostClient, LunaWorkflowRunner
from robots.luna.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.utils.config import get_memory_dir, get_repo_root


@dataclass(frozen=True)
class LunaTask:
    task_id: int
    destination_pose: int | None

    @property
    def destination(self) -> str:
        return f"chair_{self.destination_pose}" if self.destination_pose else "tabletop"

    @property
    def instruction(self) -> str:
        if self.task_id == 4:
            return "Follow the operator's natural-language source, learned-policy target, and optional destinations using composable navigation and grasp tools."
        if self.destination_pose is None:
            return (
                "Grasp the visible Rubik's cube or block already on the tabletop "
                "using only Luna's left EEF and left gripper. Do not move the base. "
                "Stop after lifting and visually checking whether it is held."
            )
        return (
            "Navigate to table_3, grasp the optical module marked with AprilTag ID 5, "
            "retract the left arm without shifting right, and transport it to "
            f"{self.destination}. Stop there with the gripper closed."
        )


LUNA_TASKS = {task_id: LunaTask(task_id, task_id) for task_id in (1, 2)}
LUNA_TASKS[3] = LunaTask(3, None)
LUNA_TASKS[4] = LunaTask(4, None)

LUNA_DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task 4 <自然语言指令>",
        "fields": (
            {"name": "task_id", "kind": "integer", "minimum": 1},
            {"name": "luna_instruction", "kind": "text"},
        ),
        "display": "Luna task {task_id}: {luna_instruction}",
        "output_slug": "luna_t{task_id}",
    },
    "runtime_components": (
        {"name": "control", "label": "LUNA ROS + VLAHost", "scope": "unique"},
    ),
    "primitives": ("move_to_table_3", "pick_apriltag_module"),
    "external_env": True,
    "requires_env_endpoint": False,
}


def get_robot_spec() -> RobotSpec:
    """Return the Luna robot extension descriptor."""
    return RobotSpec(
        name="luna",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=LUNA_DASHBOARD_SPEC,
        is_real_robot=True,
        supports_exploration=False,
        validate_args=_validate_args,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    state_output_dir: Path | str | None = None,
    **_: Any,
):
    """Build the Luna Toolkit for one Harness run."""
    from robots.luna.toolkit import LunaToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("luna")
    )
    return LunaToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        state_output_dir=state_output_dir,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.add_argument(
        "--task-id",
        type=int,
        choices=sorted(LUNA_TASKS),
        default=None if use_dashboard else 1,
        help="1/2 legacy transport; 3 explores EEF cube grasp; 4 is Agent-orchestrated colored-block transport.",
    )
    parser.add_argument(
        "--luna-vlahost-url",
        default="http://6.6.7.101:8011",
        help="HTTP base URL of the externally managed Luna VLAHost.",
    )
    parser.add_argument(
        "--luna-script-dir",
        default=str(get_repo_root() / "test.incoming_20261021"),
        help="Directory containing the validated Luna navigation and grasp scripts.",
    )
    parser.add_argument(
        "--luna-python",
        default="/usr/bin/python3"
        if Path("/usr/bin/python3").exists()
        else sys.executable,
        help="ROS-enabled Python executable used for the validated scripts.",
    )
    parser.add_argument(
        "--luna-ros-setup",
        default=str(Path(__file__).with_name("config") / "remote_ros_env.sh"),
        help="Shell setup sourced before each Luna workflow process.",
    )
    parser.add_argument("--luna-tag-id", type=int, default=5)
    parser.add_argument("--luna-http-timeout-s", type=float, default=3.0)
    parser.add_argument("--luna-navigation-timeout-s", type=float, default=180.0)
    parser.add_argument("--luna-grasp-timeout-s", type=float, default=360.0)
    parser.add_argument("--luna-calibration-file", default=None)
    parser.add_argument("--luna-intrinsics-file", default=None)
    parser.add_argument(
        "--luna-eef-execute",
        action="store_true",
        help="Enable physical EEF/gripper commands for task 3 or task 4; otherwise observe only.",
    )
    parser.add_argument(
        "--luna-table-z-m",
        type=float,
        default=0.8,
        help="Measured tabletop Z in base_link for Luna EEF tasks.",
    )
    parser.add_argument(
        "--luna-tool-clearance-m",
        type=float,
        default=0.06,
        help="Minimum left_tool Z above the table; experimental, not collision proof.",
    )
    parser.add_argument(
        "--luna-act-models",
        default=str(Path(__file__).with_name("config") / "act_models.yaml"),
        help="ACT checkpoint and deployment manifest for the selected color in task 4.",
    )
    parser.add_argument(
        "--luna-policy-backend",
        choices=("act", "pi05"),
        default="act",
        help="Task 4 policy backend. ACT keeps per-color checkpoints; pi05 uses one language-conditioned RGB checkpoint.",
    )
    parser.add_argument(
        "--luna-pi05-config",
        default=str(Path(__file__).with_name("config") / "pi05_rgb.yaml"),
        help="Three-color Luna pi0.5 deployment config for task 4.",
    )
    parser.add_argument(
        "--luna-instruction",
        default=None,
        help=("Natural-language task 4 instruction. A single red/yellow/blue block selects "
              "the policy transport workflow; any other instruction enters open-tool mode."),
    )
    parser.add_argument(
        "--luna-act-execute",
        action="store_true",
        help="Enable the selected ACT policy to command the robot for task 4.",
    )
    parser.add_argument(
        "--luna-pi05-execute",
        action="store_true",
        help="Enable the Luna pi0.5 policy to command the robot for task 4.",
    )
    parser.add_argument(
        "--luna-act-duration-s",
        type=float,
        default=15.0,
        help="Maximum selected-policy rollout duration for task 4 (seconds).",
    )
    parser.add_argument(
        "--luna-clothes-gripper-close-s",
        type=float,
        default=2.0,
        help="Right-gripper close duration for the direct-joint clothes grasp.",
    )
    parser.add_argument(
        "--luna-act-deploy-script",
        default=str(
            get_repo_root().parent.parent
            / "lerobot_vlahost"
            / "workflows/robot_interaction/deploy.py"
        ),
        help="Existing VLAHost ACT/OpenPI deployment entry point.",
    )
    parser.add_argument(
        "--luna-act-python",
        default=str(Path.home() / "miniconda3/envs/lerobot_xlb/bin/python"),
        help="Python 3.12 executable with the VLAHost policy runtime.",
    )


def _task(task_id: int | None) -> LunaTask:
    try:
        return LUNA_TASKS[int(task_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "--task-id must be 1/2 (transport), 3 (EEF cube), or 4 (language-directed ACT grasp)"
        ) from exc


def _validate_args(args: argparse.Namespace) -> None:
    """Validate local workflow resources without contacting the robot."""
    # A Dashboard session has no task until its composer receives
    # `/rpent-task ...`; validate that claimed task in the session runner.
    if getattr(args, "dashboard", False) and args.task_id is None:
        return
    _task(args.task_id)
    script_dir = Path(args.luna_script_dir).expanduser()
    required = (
        (
            "base_move_world.py",
            "base_move_world_pose_list.py",
            "pos.yaml",
            "apriltag_left_tool_accuracy_test.py",
            "apriltag_left_tool_grasp_test.py",
            "left_tool_accuracy_test.py",
            "left_tool_joint_snapshot_20261010.yaml",
            f"tag{args.luna_tag_id}_left_tool_offset.yaml",
        )
        if args.task_id in (1, 2)
        else (
            (
                "interactive_left_tool_pose.py",
                "luna_eef_session.py",
                "native_nav_test.py",
                *(
                    ("pos.yaml", "clothes_pose_sequence_test.py")
                    if args.task_id == 4
                    else ()
                ),
            )
            if args.task_id in (3, 4)
            else ()
        )
    )
    missing = [name for name in required if not (script_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"Luna workflow files are missing from {script_dir}: {', '.join(missing)}"
        )
    executable = args.luna_python
    if not (
        Path(executable).is_file()
        if Path(executable).is_absolute()
        else shutil.which(executable)
    ):
        raise ValueError(f"Luna Python executable not found: {executable}")
    if args.luna_tag_id <= 0:
        raise ValueError("--luna-tag-id must be positive")
    if (
        not math.isfinite(args.luna_table_z_m)
        or not math.isfinite(args.luna_tool_clearance_m)
        or args.luna_tool_clearance_m < 0.05
    ):
        raise ValueError(
            "Luna table Z must be finite and tool clearance at least 0.05 m"
        )
    if args.task_id == 4:
        from robots.luna.act import (
            selected_act_model,
            selected_pi05_model,
            validate_act_deployment,
            validate_pi05_deployment,
        )

        instruction = (args.luna_instruction or "").strip()
        if not instruction:
            raise ValueError("task 4 requires a nonempty --luna-instruction")
        target = requested_target(instruction)
        intent = task_mode(instruction)
        held_transport = intent == "transport_held_object"
        if held_transport and target is None:
            raise ValueError(
                "an already-held transport instruction must name a supported target so Luna can preserve the correct gripper"
            )
        open_tool_mode = target is None
        backend = args.luna_policy_backend
        execute = args.luna_act_execute if backend == "act" else args.luna_pi05_execute
        if (backend == "act" and args.luna_pi05_execute) or (
            backend == "pi05" and args.luna_act_execute
        ):
            raise ValueError("the execution flag must match --luna-policy-backend")
        if execute and not args.luna_eef_execute:
            raise ValueError(
                "task 4 policy execution also requires --luna-eef-execute for the post-policy gripper hold"
            )

        # A colorless instruction has no policy/model contract.  Retain the
        # execution flags as harmless CLI compatibility switches, but never
        # validate or start a learned policy for an open-tool task.
        if not open_tool_mode and not held_transport:
            if target == "clothes":
                if backend != "act":
                    raise ValueError("clothes direct-joint grasp requires --luna-policy-backend act")
                if (
                    not math.isfinite(args.luna_clothes_gripper_close_s)
                    or args.luna_clothes_gripper_close_s <= 0
                ):
                    raise ValueError(
                        "--luna-clothes-gripper-close-s must be positive"
                    )
            else:
                if (
                    not math.isfinite(args.luna_act_duration_s)
                    or not 0 < args.luna_act_duration_s <= 120
                ):
                    raise ValueError(
                        "--luna-act-duration-s (shared policy duration) must be in (0, 120]"
                    )
                if backend == "act":
                    manifest = Path(args.luna_act_models).expanduser().resolve()
                    selected = selected_act_model(manifest, target, required=execute)
                else:
                    selected = selected_pi05_model(
                        Path(args.luna_pi05_config).expanduser().resolve(),
                        required=execute,
                    )
                if execute:
                    assert selected is not None
                    if backend == "act":
                        validate_act_deployment(*selected, args.luna_vlahost_url)
                    else:
                        validate_pi05_deployment(*selected, args.luna_vlahost_url)
                    if not Path(args.luna_act_deploy_script).expanduser().is_file():
                        raise ValueError("--luna-act-deploy-script does not exist")
                    if not Path(args.luna_act_python).expanduser().is_file():
                        raise ValueError("--luna-act-python does not exist")
    for name in (
        "luna_http_timeout_s",
        "luna_navigation_timeout_s",
        "luna_grasp_timeout_s",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("luna_calibration_file", "luna_intrinsics_file"):
        value = getattr(args, name)
        if value and not Path(value).expanduser().is_file():
            raise ValueError(f"--{name.replace('_', '-')} does not exist: {value}")
    if (
        args.task_id in (1, 2, 3, 4)
        and args.luna_ros_setup
        and not Path(args.luna_ros_setup).expanduser().is_file()
    ):
        raise ValueError(f"--luna-ros-setup does not exist: {args.luna_ros_setup}")


def _parse_config(args: argparse.Namespace) -> RunConfig:
    task = _task(args.task_id)
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    output_dir = Path(
        args.output_dir
        or get_repo_root() / "logs" / f"{timestamp}_luna_t{task.task_id}"
    )
    return RunConfig(
        recipe_tag=f"luna_t{task.task_id}",
        output_dir=output_dir,
        prompt_vars={
            "instruction": args.luna_instruction
            if task.task_id == 4
            else task.instruction,
            "destination": task.destination,
            "mode": (
                "transport_held"
                if task.task_id == 4
                and task_mode(args.luna_instruction or "") == "transport_held_object"
                else "act"
                if task.task_id == 4
                and task_mode(args.luna_instruction or "") == "pick_and_transport"
                else "open"
                if task.task_id == 4
                else "eef"
                if task.task_id == 3
                else "transport"
            ),
        },
        task_desc={
            "task_id": task.task_id,
            "destination": task.destination,
            "tag_id": args.luna_tag_id,
        }
        if task.task_id in (1, 2)
        else (
            {"task_id": 3, "target": "cube_or_block", "table_z_m": args.luna_table_z_m}
            if task.task_id == 3
            else {
                "task_id": task.task_id,
                "instruction": args.luna_instruction,
                "policy_backend": args.luna_policy_backend,
                "clothes_gripper_close_s": args.luna_clothes_gripper_close_s,
                "act_manifest": str(Path(args.luna_act_models).expanduser().resolve()),
                "pi05_config": str(Path(args.luna_pi05_config).expanduser().resolve()),
            }
        ),
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[Any], dict[str, Any]]:
    del dashboard_events
    available = {"control"}
    selected = available if components is None else components
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown Luna runtime components: {sorted(unknown)}")
    if "control" not in selected:
        return [], {}

    task = _task(args.task_id)
    from robots.luna.vision import verify_grasp

    visual_verifier = functools.partial(
        verify_grasp,
        planner=getattr(args, "planner", "codex"),
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
    )
    client = LunaVLAHostClient(
        args.luna_vlahost_url,
        timeout_s=args.luna_http_timeout_s,
    )
    if task.task_id == 4:
        from robots.luna.act import (
            LunaActRunner,
            LunaClothesPoseRunner,
            grasp_side,
        )
        from robots.luna.eef import LunaEefSession
        from robots.luna.navigation import LunaNavigationRunner

        ros_setup = (
            Path(args.luna_ros_setup).expanduser().resolve()
            if args.luna_ros_setup
            else None
        )
        policy_execute = (
            args.luna_act_execute
            if args.luna_policy_backend == "act"
            else args.luna_pi05_execute
        )
        intent = task_mode(args.luna_instruction or "")
        held_transport = intent == "transport_held_object"
        policy_selected = intent == "pick_and_transport"
        target = requested_target(args.luna_instruction or "")
        eef_side = grasp_side(target) if target else "left"
        navigation = LunaNavigationRunner(
            script_dir=Path(args.luna_script_dir).expanduser().resolve(),
            python_executable=args.luna_python,
            vlahost_url=args.luna_vlahost_url,
            output_dir=output_dir,
            client=client,
            check_cancelled=lambda: None,
            timeout_s=args.luna_navigation_timeout_s,
            execute=(args.luna_eef_execute and policy_execute)
            if policy_selected
            else args.luna_eef_execute,
            ros_setup_script=ros_setup,
        )
        runner_execute = policy_execute if policy_selected else False
        if target == "clothes":
            if ros_setup is None:
                raise ValueError("clothes direct-joint grasp requires --luna-ros-setup")
            runner = LunaClothesPoseRunner(
                script=Path(args.luna_script_dir).expanduser().resolve()
                / "clothes_pose_sequence_test.py",
                python_executable=args.luna_python,
                ros_setup_script=ros_setup,
                output_dir=output_dir,
                execute=runner_execute,
                timeout_s=args.luna_grasp_timeout_s,
                gripper_close_duration_s=args.luna_clothes_gripper_close_s,
                check_cancelled=lambda: None,
                visual_verifier=visual_verifier,
            )
        else:
            runner = LunaActRunner(
                color=None,
                manifest=Path(args.luna_act_models).expanduser().resolve(),
                client=client,
                deploy_script=Path(args.luna_act_deploy_script).expanduser().resolve(),
                # Preserve the virtualenv launcher path: resolving its symlink
                # runs the base interpreter without the venv's site-packages.
                python_executable=str(
                    Path(args.luna_act_python).expanduser().absolute()
                ),
                output_dir=output_dir,
                execute=runner_execute,
                duration_s=args.luna_act_duration_s,
                check_cancelled=lambda: None,
                backend=args.luna_policy_backend,
                pi05_config=Path(args.luna_pi05_config).expanduser().resolve(),
                visual_verifier=visual_verifier,
            )

        return [], {
            "runner": runner,
            "session": LunaEefSession(
                script_dir=Path(args.luna_script_dir).expanduser().resolve(),
                python_executable=args.luna_python,
                ros_setup_script=ros_setup,
                output_dir=output_dir,
                check_cancelled=lambda: None,
                execute=args.luna_eef_execute,
                table_z_m=args.luna_table_z_m,
                tool_clearance_m=args.luna_tool_clearance_m,
                side=eef_side,
            ),
            "client": client,
            "navigation": navigation,
            "instruction": args.luna_instruction,
            "mode": "act" if target else "open",
            "initially_holding": held_transport,
        }
    if task.task_id == 3:
        from robots.luna.eef import LunaEefSession

        return [], {
            "session": LunaEefSession(
                script_dir=Path(args.luna_script_dir).expanduser().resolve(),
                python_executable=args.luna_python,
                ros_setup_script=Path(args.luna_ros_setup).expanduser().resolve()
                if args.luna_ros_setup
                else None,
                output_dir=output_dir,
                check_cancelled=lambda: None,
                execute=args.luna_eef_execute,
                table_z_m=args.luna_table_z_m,
                tool_clearance_m=args.luna_tool_clearance_m,
            ),
            "client": client,
            "visual_verifier": visual_verifier,
            "mode": "eef",
        }
    runner = LunaWorkflowRunner(
        script_dir=Path(args.luna_script_dir).expanduser().resolve(),
        python_executable=args.luna_python,
        vlahost_url=args.luna_vlahost_url,
        destination_pose=task.destination_pose,
        tag_id=args.luna_tag_id,
        output_dir=output_dir,
        check_cancelled=lambda: None,
        navigation_timeout_s=args.luna_navigation_timeout_s,
        grasp_timeout_s=args.luna_grasp_timeout_s,
        ros_setup_script=Path(args.luna_ros_setup).expanduser().resolve()
        if args.luna_ros_setup
        else None,
        calibration_file=args.luna_calibration_file,
        intrinsics_file=args.luna_intrinsics_file,
        image_client=client,
        visual_verifier=visual_verifier,
    )
    return [], {"runner": runner, "client": client}
