# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Color-bound ACT or Pi0.5 grasp through the existing VLAHost runner."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import yaml

from robots.luna.client import LunaVLAHostClient
from robots.luna.eef import LunaEefPrimitives, LunaEefSession
from robots.luna.navigation import SOFA_CHAIR_ALIASES, SOFA_CHAIR_STATION
from robots.luna.vision import comparison_png, verify_grasp
from rpent.tools import ToolResult
from rpent.tools.base import tool

if TYPE_CHECKING:
    from robots.luna.navigation import LunaNavigationRunner

COLORS = ("red", "yellow", "blue")
ACT_COLORS = ("red", "blue")
ACT_TARGETS = ("red", "blue", "clothes")
POLICY_TARGETS = COLORS + ("clothes",)
TaskMode = Literal["pick_and_transport", "transport_held_object", "open_ended"]
PI05_PROMPTS = {
    "red": "Pick up the red block.",
    "yellow": "Pick up the yellow block.",
    "blue": "Pick up the blue block.",
}


class LunaOpenPrimitives:
    """Open-ended composition of the currently implemented Luna capabilities.

    One explicitly selected EEF owns Cartesian and gripper commands for a run.
    The Agent cannot proxy one arm through the other or start a learned policy.
    """

    def __init__(
        self,
        *,
        client: LunaVLAHostClient,
        session: LunaEefSession,
        navigation: "LunaNavigationRunner",
        instruction: str,
        **_: Any,
    ) -> None:
        self.client = client
        self.session = session
        self.navigation = navigation
        self.instruction = instruction
        self.eef = LunaEefPrimitives(session=session, client=client)
        self._eef_motion_started = False
        self.completed = False

    @tool
    def select_eef_side(self, side: Literal["left", "right"]) -> ToolResult:
        """Select LEFT or RIGHT EEF before observing or commanding that arm."""
        if side not in ("left", "right"):
            return ToolResult(data={"error": "EEF side must be left or right"})
        if self._eef_motion_started:
            return ToolResult(
                data={"error": "EEF side is locked after EEF or gripper motion; start a new run to switch arms"}
            )
        if self.session.side == side:
            return ToolResult(data={"success": True, "eef_side": side, "changed": False})
        # Observation starts a child publisher too.  Close it before choosing
        # the other arm so only one IK hand field can be active at a time.
        self.session.close()
        self.session.side = side
        self.eef = LunaEefPrimitives(session=self.session, client=self.client)
        return ToolResult(data={"success": True, "eef_side": side, "changed": True})

    @tool(readonly=True)
    def list_locations(self) -> ToolResult:
        """List named stations available to the base-navigation tool."""
        return ToolResult(
            data={
                "locations": self.navigation.available_locations(),
                "instruction": self.instruction,
            }
        )

    @tool
    def navigate_to(self, location: str) -> ToolResult:
        """Move the base to one configured named station and verify arrival."""
        return self.navigation.navigate(location)

    @tool(readonly=True)
    def verify_location(self, location: str) -> ToolResult:
        """Check the current base pose against one configured named station."""
        return self.navigation.verify(location)

    @tool
    def observe_scene(self) -> ToolResult:
        """Read fresh three-view RGB and the currently selected EEF pose."""
        result = self.eef.observe_cube()
        if not result.is_error:
            result.data["instruction"] = self.instruction
            result.data["eef_side"] = self.session.side
            result.data["available_eef_sides"] = ["left", "right"]
            result.data["available_gripper_sides"] = ["left", "right"]
            result.data["navigation_enabled"] = self.session.execute
        return result

    @tool
    def move_eef_delta(
        self, dx_mm: float, dy_mm: float, dz_mm: float
    ) -> ToolResult:
        """Translate the selected EEF in base_link axes by a bounded nonzero delta."""
        result = self.eef.move_eef_delta(dx_mm, dy_mm, dz_mm)
        if not result.is_error:
            self._eef_motion_started = True
        return result

    @tool
    def rotate_eef_delta(
        self, droll_deg: float, dpitch_deg: float, dyaw_deg: float
    ) -> ToolResult:
        """Rotate the selected EEF by a bounded nonzero base_link-axis delta."""
        result = self.eef.rotate_eef_delta(droll_deg, dpitch_deg, dyaw_deg)
        if not result.is_error:
            self._eef_motion_started = True
        return result

    @tool
    def open_gripper(self) -> ToolResult:
        """Open the gripper belonging to the selected EEF after fresh observation."""
        result = self.eef.open_gripper()
        if not result.is_error:
            self._eef_motion_started = True
        return result

    @tool
    def close_gripper(self) -> ToolResult:
        """Close the gripper belonging to the selected EEF after fresh observation."""
        result = self.eef.close_gripper()
        if not result.is_error:
            self._eef_motion_started = True
        return result


def requested_color(instruction: str) -> str | None:
    """Recognize one explicit color in the operator's task; never infer from task ID."""
    normalized = instruction.lower()
    names = {
        "red": ("红色", "红方块", "red"),
        "yellow": ("黄色", "黄方块", "yellow"),
        "blue": ("蓝色", "蓝方块", "blue"),
    }
    found = [
        color
        for color, words in names.items()
        if any(word in normalized for word in words)
    ]
    return found[0] if len(found) == 1 else None


def requested_target(instruction: str) -> str | None:
    """Recognize one learned-policy target from the operator instruction.

    Color remains the public routing vocabulary for the block policies.  The
    clothes policy is a separate ACT contract (22-D action, right-hand grasp)
    and must not be silently treated as a color or as open-tool exploration.
    """
    color = requested_color(instruction)
    if color is not None:
        return color
    normalized = instruction.lower().replace(" ", "")
    clothes_words = ("抓衣服", "拿衣服", "衣服", "衣物", "布料", "clothes", "cloth", "fabric")
    if any(word in normalized for word in clothes_words):
        return "clothes"
    return None


def instruction_declares_held_payload(instruction: str) -> bool:
    """Return whether the operator explicitly says the colored block is held.

    This is deliberately narrow: a normal request such as ``拿蓝色方块`` must
    still enter the learned grasp workflow.  The declaration is an operator
    supplied run-state precondition, not visual grasp evidence.
    """
    normalized = instruction.lower().replace(" ", "")
    return any(
        phrase in normalized
        for phrase in (
            "已经拿到",
            "已拿到",
            "已经抓到",
            "已抓到",
            "已经夹住",
            "已夹住",
            "正在拿着",
            "正在持有",
            "当前持有",
            "已经持有",
            "已持有",
            "手上有",
            "手里有",
            "夹爪里",
            "夹爪夹着",
            "正夹着",
            "正拿着",
            "alreadyholding",
            "alreadyheld",
        )
    )


def task_mode(instruction: str) -> TaskMode:
    """Classify only the run's initial state, never its route or skill plan."""
    if instruction_declares_held_payload(instruction):
        return "transport_held_object"
    return "pick_and_transport" if requested_target(instruction) else "open_ended"


def _target_end(instruction: str, target: str) -> int | None:
    """Return the end of the last target phrase in a normalized instruction."""
    normalized = instruction.lower().replace(" ", "")
    words = {
        "clothes": ("抓衣服", "拿衣服", "衣服", "衣物", "布料", "clothes", "cloth", "fabric"),
        "red": ("红色方块", "红方块", "红色", "red"),
        "yellow": ("黄色方块", "黄方块", "黄色", "yellow"),
        "blue": ("蓝色方块", "蓝方块", "蓝色", "blue"),
    }.get(target, ())
    positions = [normalized.rfind(word) + len(word) for word in words if normalized.rfind(word) >= 0]
    return max(positions) if positions else None


def instruction_destination_after_target(instruction: str, location: str) -> bool:
    """Match a named destination occurring after the learned target phrase."""
    target = requested_target(instruction)
    if target is None:
        return False
    end = _target_end(instruction, target)
    if end is None:
        return False
    suffix = instruction.lower().replace(" ", "")[end:]
    raw = location.lower().replace(" ", "")
    chinese_numbers = {"1": "一", "2": "二", "3": "三", "4": "四", "5": "五"}
    kind = number = ""
    for candidate_kind, aliases in (("table", ("table", "桌", "桌子")), ("chair", ("chair", "椅", "椅子"))):
        for candidate_number, chinese in chinese_numbers.items():
            candidate_aliases = (
                f"{candidate_kind}_{candidate_number}",
                f"{candidate_kind}{candidate_number}",
                f"{candidate_number}号{'桌子' if candidate_kind == 'table' else '椅子'}",
                f"{chinese}号{'桌子' if candidate_kind == 'table' else '椅子'}",
                f"{aliases[1]}{candidate_number}",
            )
            if raw in candidate_aliases:
                kind, number = candidate_kind, candidate_number
                break
            if (
                candidate_kind == "chair"
                and f"chair_{candidate_number}" == SOFA_CHAIR_STATION
                and raw in SOFA_CHAIR_ALIASES
            ):
                kind, number = "chair", candidate_number
                break
        if kind:
            break
    if not kind:
        return False
    chinese = chinese_numbers[number]
    aliases = (
        f"{kind}{number}",
        f"{number}号{'桌子' if kind == 'table' else '椅子'}",
        f"{chinese}号{'桌子' if kind == 'table' else '椅子'}",
    )
    destination_aliases = aliases + (
        SOFA_CHAIR_ALIASES
        if kind == "chair" and f"chair_{number}" == SOFA_CHAIR_STATION
        else ()
    )
    return any(alias and alias in suffix for alias in destination_aliases)


def instruction_requests_transport(instruction: str) -> bool:
    """Return whether a named station follows the learned target."""
    target = requested_target(instruction)
    if target is None:
        return False
    end = _target_end(instruction, target)
    if end is None:
        return False
    suffix = instruction.lower().replace(" ", "")[end:]
    return bool(re.search(r"椅子|桌子|沙发|chair|table|sofa", suffix))


def grasp_side(color: str) -> Literal["left", "right"]:
    """Map a learned-policy target to the physical gripper used for holding."""
    return "right" if color in ("blue", "clothes") else "left"


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _deploy_config(path: Path) -> dict[str, Any]:
    """Read VLAHost's YAML inheritance without importing its robot runtime."""
    value = _mapping(path)
    base = value.pop("base_config", None)
    if base is None:
        return value
    parent = Path(base).expanduser()
    if not parent.is_absolute():
        parent = path.parent / parent
    inherited = _deploy_config(parent.resolve())

    def merge(base_value: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base_value)
        for key, item in overlay.items():
            merged[key] = (
                merge(merged[key], item)
                if isinstance(item, dict) and isinstance(merged.get(key), dict)
                else item
            )
        return merged

    return merge(inherited, value)


def selected_act_model(
    manifest: Path, color: str, *, required: bool
) -> tuple[Path, Path] | None:
    """Resolve only the task-bound model; placeholders never enable motion."""
    if color not in ACT_TARGETS:
        raise ValueError(f"unsupported ACT target color: {color}")
    entries = _mapping(manifest)
    entry = entries.get(color)
    if not isinstance(entry, dict):
        raise ValueError(f"ACT model manifest has no {color} mapping")
    checkpoint, deployment = entry.get("checkpoint"), entry.get("deployment_config")
    if not checkpoint or not deployment:
        if required:
            raise ValueError(f"ACT {color} model is not configured in {manifest}")
        return None
    paths = []
    for raw in (checkpoint, deployment):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"ACT {color} paths must be nonempty strings")
        path = Path(raw).expanduser()
        paths.append(
            (manifest.parent / path if not path.is_absolute() else path).resolve()
        )
    model_path, config_path = paths
    for other in ACT_COLORS:
        if other == color or not isinstance(entries.get(other), dict):
            continue
        other_checkpoint = entries[other].get("checkpoint")
        if isinstance(other_checkpoint, str) and other_checkpoint.strip():
            other_path = Path(other_checkpoint).expanduser()
            other_path = (
                manifest.parent / other_path
                if not other_path.is_absolute()
                else other_path
            ).resolve()
            if other_path == model_path:
                raise ValueError(f"ACT {color} and {other} cannot share a checkpoint")
    if not model_path.is_dir() or not config_path.is_file():
        if required:
            raise ValueError(f"ACT {color} checkpoint or deployment config is missing")
        return None
    model = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if model.get("type") != "act":
        raise ValueError(f"ACT {color} checkpoint is not an ACT policy")
    return model_path, config_path


def validate_act_deployment(
    model_path: Path, config_path: Path, vlahost_url: str
) -> None:
    """Reject mismatched or autonomously homing rollout configs before motion."""
    config = _deploy_config(config_path)
    policy = config.get("policy", {})
    robot = config.get("robot", {})
    inference = config.get("inference", {})
    saved_path = Path(policy.get("path", "")).expanduser()
    if not saved_path.is_absolute():
        raise ValueError("ACT deployment policy.path must be absolute")
    if saved_path.resolve() != model_path.resolve():
        raise ValueError(
            "ACT deployment policy.path does not match the color-bound checkpoint"
        )
    if robot.get("type") != "luna_http" or robot.get("http_base_url", "").rstrip(
        "/"
    ) != vlahost_url.rstrip("/"):
        raise ValueError("ACT deployment robot type or VLAHost URL does not match Luna")
    if (
        inference.get("strategy") != "base"
        or inference.get("return_to_initial_position") is not False
    ):
        raise ValueError(
            "ACT deployment must use base strategy and disable automatic Home"
        )
    if config.get("recording", {}).get("enabled"):
        raise ValueError("ACT grasp deployment must not start a recorder")
    model = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    action_shape = model.get("output_features", {}).get("action", {}).get("shape")
    joint_names = robot.get("joint_names")
    if not isinstance(joint_names, list) or len(joint_names) != 22:
        raise ValueError("Luna ACT needs the canonical 22-D observation joint order")
    action_names = (
        joint_names if action_shape == [22] else joint_names[:14] + joint_names[20:]
    )
    if action_shape not in ([16], [22]):
        raise ValueError("Luna ACT action must be 16-D or legacy 22-D")
    dataset = Path(config.get("dataset", {}).get("root", ""))
    info_file = dataset / "meta" / "info.json"
    if not info_file.is_file():
        raise ValueError("Luna ACT deployment needs the training dataset metadata")
    info = json.loads(info_file.read_text(encoding="utf-8"))
    expected_names = info.get("features", {}).get("action", {}).get("names")
    state_names = info.get("features", {}).get("observation.state", {}).get("names")
    if action_names != expected_names or joint_names != state_names:
        raise ValueError("Luna ACT action/state order must match the training dataset")
    if inference.get("include_velocity") is not True:
        raise ValueError("Luna ACT needs measured observation.velocity")
    safety_path = Path(robot.get("safety_stats_path", ""))
    if (
        safety_path.resolve() != dataset.resolve()
        or not (safety_path / "meta" / "stats.json").is_file()
    ):
        raise ValueError("Luna ACT safety stats must match its training dataset")
    stats = json.loads((safety_path / "meta" / "stats.json").read_text(encoding="utf-8"))
    action_stats = stats.get("action", {})
    if any(len(action_stats.get(key, [])) != len(action_names) for key in ("min", "max")):
        raise ValueError("Luna ACT safety bounds must match the policy action dimension")
    for key in ("observation.state", "observation.velocity"):
        feature = model.get("input_features", {}).get(key)
        if feature is not None and feature.get("shape") != [len(joint_names)]:
            raise ValueError(
                f"ACT {key} dimension differs from the VLAHost joint contract"
            )
    cameras = robot.get("cameras", {})
    for key, feature in model.get("input_features", {}).items():
        if key.startswith("observation.images."):
            name = key.removeprefix("observation.images.")
            camera = cameras.get(name)
            shape = feature.get("shape")
            if not isinstance(camera, dict) or shape != [
                3,
                camera.get("height"),
                camera.get("width"),
            ]:
                raise ValueError(f"ACT camera contract mismatch: {name}")


def selected_pi05_model(
    config_path: Path, *, required: bool
) -> tuple[Path, Path] | None:
    """Resolve the single three-color pi0.5 checkpoint and deployment config."""
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        if required:
            raise ValueError(f"pi0.5 deployment config is missing: {config_path}")
        return None
    config = _deploy_config(config_path)
    checkpoint = Path(config.get("policy", {}).get("path", "")).expanduser()
    if not checkpoint.is_absolute() or not checkpoint.is_dir():
        if required:
            raise ValueError("pi0.5 checkpoint is not configured or missing")
        return None
    return checkpoint.resolve(), config_path


def validate_pi05_deployment(
    model_path: Path, config_path: Path, vlahost_url: str
) -> None:
    """Validate the Luna 22-D native OpenPI deployment contract."""
    config = _deploy_config(config_path)
    openpi = config.get("openpi", {})
    robot = config.get("robot", {})
    inference = config.get("inference", {})
    dataset = Path(config.get("dataset", {}).get("root", "")).expanduser()
    saved_path = Path(config.get("policy", {}).get("path", "")).expanduser()
    if not saved_path.is_absolute() or saved_path.resolve() != model_path.resolve():
        raise ValueError("pi0.5 deployment policy.path does not match the checkpoint")
    asset_id = str(openpi.get("asset_id", ""))
    if (
        openpi.get("config_name")
        != "pi05_luna_stack_blocks_rgb_v4_joint22_tplus2"
        or openpi.get("action_mode", "joint") != "joint"
        or int(openpi.get("physical_action_dim", 0)) != 22
    ):
        raise ValueError("pi0.5 deployment does not use the Luna 22-D config")
    if (
        not (model_path / "params").is_dir()
        or not (model_path / "assets" / asset_id / "norm_stats.json").is_file()
    ):
        raise ValueError("pi0.5 checkpoint params or normalization assets are missing")
    if robot.get("type") != "luna_http" or robot.get("http_base_url", "").rstrip(
        "/"
    ) != vlahost_url.rstrip("/"):
        raise ValueError(
            "pi0.5 deployment robot type or VLAHost URL does not match Luna"
        )
    if (
        inference.get("type") != "chunk"
        or inference.get("strategy") != "base"
        or inference.get("return_to_initial_position") is not False
    ):
        raise ValueError("Luna pi0.5 must use chunk/base and disable automatic Home")
    if config.get("recording", {}).get("enabled"):
        raise ValueError("Luna pi0.5 grasp must not start a recorder")
    info_file = dataset / "meta" / "info.json"
    stats_file = dataset / "meta" / "stats.json"
    if not info_file.is_file() or not stats_file.is_file():
        raise ValueError(
            "Luna pi0.5 deployment needs the merged training dataset metadata"
        )
    info = json.loads(info_file.read_text(encoding="utf-8"))
    joint_names = robot.get("joint_names")
    expected_names = info.get("features", {}).get("action", {}).get("names")
    if (
        not isinstance(joint_names, list)
        or len(joint_names) != 22
        or joint_names != expected_names
    ):
        raise ValueError("Luna pi0.5 joint order must match the 22-D training dataset")
    if Path(robot.get("safety_stats_path", "")).resolve() != dataset.resolve():
        raise ValueError(
            "Luna pi0.5 safety stats must match the merged training dataset"
        )
    observation = openpi.get("observation", {})
    if observation != {
        "image": "top",
        "left_wrist_image": "wrist_L",
        "right_wrist_image": "wrist_R",
    }:
        raise ValueError("Luna pi0.5 camera mapping must use top and both wrist views")


class LunaActRunner:
    """Own one finite ACT or pi0.5 rollout; never share its motion publisher."""

    def __init__(
        self,
        *,
        color: str | None,
        manifest: Path,
        client: LunaVLAHostClient,
        deploy_script: Path,
        python_executable: str,
        output_dir: Path,
        execute: bool,
        duration_s: float,
        check_cancelled: Callable[[], None],
        visual_verifier: Callable[..., dict[str, str]] = verify_grasp,
        backend: Literal["act", "pi05"] = "act",
        pi05_config: Path | None = None,
    ) -> None:
        self.color = color
        self.manifest = manifest
        self.client = client
        self.deploy_script = deploy_script
        self.python_executable = python_executable
        self.output_dir = output_dir
        self.execute = execute
        self.duration_s = duration_s
        self.check_cancelled = check_cancelled
        self.visual_verifier = visual_verifier
        self.backend = backend
        self.pi05_config = pi05_config
        self.completed = False
        self.attempted = False
        self.attempt_count = 0
        self.retry_prepared = False
        self.process: subprocess.Popen[bytes] | None = None

    def can_attempt(self) -> bool:
        """Return whether the first attempt or one explicitly prepared retry may run."""
        return self.attempt_count == 0 or (
            self.retry_prepared and self.attempt_count < 2
        )

    def prepare_retry(self) -> bool:
        """Arm one final policy attempt after the caller restores a safe pose."""
        if self.attempt_count != 1 or self.completed:
            return False
        self.retry_prepared = True
        return True

    def validate(self, color: str) -> tuple[Path, Path]:
        """Resolve and validate the selected backend without starting it."""
        if self.backend == "act":
            selected = selected_act_model(self.manifest, color, required=True)
            assert selected is not None
            validate_act_deployment(*selected, self.client.base_url)
            return selected
        if self.backend != "pi05" or self.pi05_config is None:
            raise ValueError(
                f"unsupported or incomplete Luna policy backend: {self.backend}"
            )
        if color not in PI05_PROMPTS:
            raise ValueError(
                "Pi0.5 currently has only the red/yellow/blue block prompts; "
                "use --luna-policy-backend act for clothes"
            )
        selected = selected_pi05_model(self.pi05_config, required=True)
        assert selected is not None
        validate_pi05_deployment(*selected, self.client.base_url)
        return selected

    def run(self, color: str | None = None, *, verify: bool = True) -> ToolResult:
        selected_color = color or self.color
        if selected_color not in POLICY_TARGETS:
            return ToolResult(
                data={
                    "error": "select exactly one supported target (red, yellow, blue, or clothes) from the task instruction",
                    "phase": "preflight",
                }
            )
        log_path = self.output_dir / f"luna_{self.backend}_{selected_color}.log"
        if not self.execute:
            return ToolResult(
                data={
                    "error": f"{self.backend} motion is disabled",
                    "phase": "preflight",
                    "error_code": "motion_disabled",
                    "log_path": str(log_path),
                }
            )
        if not self.can_attempt():
            return ToolResult(
                data={
                    "error": f"{self.backend} grasp retry is not prepared or exhausted",
                    "phase": "preflight",
                    "error_code": "already_attempted",
                    "attempts": self.attempt_count,
                    "log_path": str(log_path),
                }
            )
        self.attempted = True
        self.attempt_count += 1
        self.retry_prepared = False
        try:
            _, config_path = self.validate(selected_color)
            if (
                not self.deploy_script.is_file()
                or not Path(self.python_executable).is_file()
            ):
                raise ValueError(
                    "policy deployment script or Python executable is missing"
                )
            before = self.client.camera_jpeg()
            self.client.get("/state")
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            return ToolResult(
                data={
                    "error": str(exc),
                    "phase": "preflight",
                    "error_code": f"{self.backend}_precheck_failed",
                    "log_path": str(log_path),
                }
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_executable,
            str(self.deploy_script),
            "--model",
            "openpi" if self.backend == "pi05" else "act",
            "--config",
            str(config_path),
            "--duration",
            str(self.duration_s),
        ]
        if self.backend == "pi05":
            command.extend(["--task", PI05_PROMPTS[selected_color]])
        child_env = os.environ.copy()
        for name in (
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
        ):
            child_env.pop(name, None)
        rollout = Path(self.python_executable).with_name("lerobot-rollout")
        local_src = self.deploy_script.parents[2] / "src"
        if rollout.is_file() and os.access(rollout, os.X_OK):
            child_env["LEROBOT_ROLLOUT"] = str(rollout)
        if local_src.is_dir():
            existing = child_env.get("PYTHONPATH")
            child_env["PYTHONPATH"] = (
                f"{local_src}{os.pathsep}{existing}" if existing else str(local_src)
            )
        with log_path.open("wb") as log:
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=self.deploy_script.parent,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                return ToolResult(
                    data={
                        "error": str(exc),
                        "phase": "policy",
                        "error_code": "policy_start_failed",
                        "log_path": str(log_path),
                    }
                )
            try:
                startup_allowance = (
                    float(
                        _deploy_config(config_path)
                        .get("openpi", {})
                        .get("startup_timeout_s", 300.0)
                    )
                    if self.backend == "pi05"
                    else 15.0
                )
                deadline = time.monotonic() + self.duration_s + startup_allowance + 30.0
                while self.process.poll() is None:
                    self.check_cancelled()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"{self.backend} rollout exceeded its startup, duration, and shutdown allowance"
                        )
                    time.sleep(0.1)
                returncode = self.process.returncode
            except TimeoutError as exc:
                self.stop()
                return ToolResult(
                    data={
                        "error": str(exc),
                        "phase": "policy",
                        "error_code": "policy_timeout",
                        "log_path": str(log_path),
                    }
                )
            except BaseException:
                self.stop()
                raise
            finally:
                self.process = None
        if returncode != 0:
            return ToolResult(
                data={
                    "error": f"{self.backend} rollout exited with {returncode}",
                    "phase": "policy",
                    "error_code": "policy_exit",
                    "returncode": returncode,
                    "log_path": str(log_path),
                }
            )
        if not verify:
            return ToolResult(
                data={
                    "phase": "policy",
                    "policy_completed": True,
                    "workflow_success": False,
                    "attempts": self.attempt_count,
                    "log_path": str(log_path),
                }
            )
        try:
            after = self.client.camera_jpeg()
            label = self.backend.upper()
            side = grasp_side(selected_color)
            comparison = comparison_png(
                before,
                after,
                after_label=f"AFTER {label}",
                wrist_side=side,
            )
            image_path = (
                self.output_dir / f"luna_{self.backend}_{selected_color}_comparison.png"
            )
            image_path.write_bytes(comparison)
            verdict = self.visual_verifier(
                comparison,
                target=f"{selected_color} block",
                after_stage=label,
                wrist_side=side,
            )
            self.completed = verdict["verdict"] == "held"
            result = {
                "phase": "verify",
                "verdict": verdict["verdict"],
                "visual_evidence": verdict["evidence"],
                "payload_confirmed": self.completed,
                "workflow_success": self.completed,
                "attempts": self.attempt_count,
                "retryable": not self.completed and self.attempt_count < 2,
                "log_path": str(log_path),
                "visual_image_path": str(image_path),
            }
            if not self.completed:
                result.update(
                    error="Target grasp was not visually confirmed",
                    error_code="not_held",
                )
            return ToolResult(data=result, images=[comparison])
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            return ToolResult(
                data={
                    "error": str(exc),
                    "phase": "verify",
                    "error_code": "visual_error",
                    "payload_confirmed": False,
                    "log_path": str(log_path),
                }
            )

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=7.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)


class LunaClothesPoseRunner:
    """Run the clothes grasp as a bounded direct-joint pose sequence."""

    def __init__(
        self,
        *,
        script: Path,
        python_executable: str,
        ros_setup_script: Path,
        output_dir: Path,
        execute: bool,
        timeout_s: float,
        gripper_close_duration_s: float,
        check_cancelled: Callable[[], None],
        visual_verifier: Callable[..., dict[str, str]] = verify_grasp,
    ) -> None:
        self.backend = "direct_joint"
        self.script = script
        self.python_executable = python_executable
        self.ros_setup_script = ros_setup_script
        self.output_dir = output_dir
        self.execute = execute
        self.timeout_s = timeout_s
        self.gripper_close_duration_s = gripper_close_duration_s
        self.check_cancelled = check_cancelled
        self.visual_verifier = visual_verifier
        self.completed = False
        self.attempted = False
        self.attempt_count = 0
        self.process: subprocess.Popen[bytes] | None = None

    def can_attempt(self) -> bool:
        return self.attempt_count == 0

    def prepare_retry(self) -> bool:
        return False

    def validate(self, color: str) -> tuple[Path, Path]:
        if color != "clothes":
            raise ValueError("direct-joint clothes sequence only supports clothes")
        for path in (
            self.script,
            Path(self.python_executable),
            self.ros_setup_script,
        ):
            if not path.is_file():
                raise ValueError(f"clothes direct-joint dependency is missing: {path}")
        if (
            not math.isfinite(self.gripper_close_duration_s)
            or self.gripper_close_duration_s <= 0
        ):
            raise ValueError("clothes gripper close duration must be positive")
        return self.script, self.script

    def run(self, color: str | None = None, *, verify: bool = True) -> ToolResult:
        del verify
        selected = color or "clothes"
        self.validate(selected)
        log_path = self.output_dir / "luna_clothes_direct_pose.log"
        if not self.execute:
            return ToolResult(
                data={
                    "error": "clothes direct-joint motion is disabled",
                    "phase": "preflight",
                    "error_code": "motion_disabled",
                    "log_path": str(log_path),
                }
            )
        if not self.can_attempt():
            return ToolResult(
                data={
                    "error": "direct-joint clothes grasp already attempted",
                    "phase": "preflight",
                    "error_code": "already_attempted",
                    "attempts": self.attempt_count,
                    "log_path": str(log_path),
                }
            )
        self.attempted = True
        self.attempt_count = 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_executable,
            str(self.script),
            "--execute",
            "--home-mode",
            "direct",
            "--gripper-side",
            "right",
            "--gripper-close-duration-s",
            str(self.gripper_close_duration_s),
        ]
        with log_path.open("wb") as log:
            try:
                self.process = subprocess.Popen(
                    [
                        "/usr/bin/bash",
                        "-c",
                        'source "$1" && shift && exec "$@"',
                        "rpent-clothes",
                        str(self.ros_setup_script),
                        *command,
                    ],
                    cwd=self.script.parent,
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                return ToolResult(
                    data={
                        "error": str(exc),
                        "phase": "direct_pose",
                        "error_code": "direct_pose_start_failed",
                        "log_path": str(log_path),
                    }
                )
            try:
                deadline = time.monotonic() + self.timeout_s
                while self.process.poll() is None:
                    self.check_cancelled()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"clothes direct-joint sequence exceeded {self.timeout_s:.1f}s"
                        )
                    time.sleep(0.1)
                returncode = self.process.returncode
            except TimeoutError as exc:
                self.stop()
                return ToolResult(
                    data={
                        "error": str(exc),
                        "phase": "direct_pose",
                        "error_code": "direct_pose_timeout",
                        "log_path": str(log_path),
                    }
                )
            except BaseException:
                self.stop()
                raise
            finally:
                self.process = None
        if returncode != 0:
            return ToolResult(
                data={
                    "error": f"clothes direct-joint sequence exited with {returncode}",
                    "phase": "direct_pose",
                    "error_code": "direct_pose_exit",
                    "returncode": returncode,
                    "log_path": str(log_path),
                }
            )
        return ToolResult(
            data={
                "phase": "direct_pose",
                "control_mode": 3,
                "policy_completed": True,
                "script_success": True,
                "workflow_success": False,
                "attempts": self.attempt_count,
                "log_path": str(log_path),
            }
        )

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=7.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)


class LunaActPrimitives:
    """One language-directed grasp with bounded staging and an owned policy handoff."""

    def __init__(
        self,
        *,
        runner: LunaActRunner | LunaClothesPoseRunner,
        client: LunaVLAHostClient,
        session: LunaEefSession,
        navigation: LunaNavigationRunner,
        instruction: str,
        initially_holding: bool = False,
    ) -> None:
        self.runner = runner
        self.client = client
        self.session = session
        self.navigation = navigation
        self.eef = LunaEefPrimitives(session=session, client=client)
        self.instruction = instruction
        self.initial_xyz: list[float] | None = None
        self.return_xyz: list[float] | None = None
        self.before_jpeg: bytes | None = None
        self.completed = False
        self.handoff_ok = False
        self.grasp_retracted = False
        self.last_verdict: str | None = None
        self.initially_holding = initially_holding

    def _requested_target(self) -> str | None:
        return requested_target(self.instruction)

    def _requested_color(self) -> str | None:
        """Backward-compatible alias for callers that still use color wording."""
        return self._requested_target()

    def _gripper_side(self) -> Literal["left", "right"]:
        target = self._requested_target()
        return grasp_side(target) if target else "left"

    def _command_gripper(self, action: Literal["open", "close"]) -> ToolResult:
        side = self._gripper_side()
        # Base navigation may leave Apex's joint-command mux on another source.
        # Claim the EEF/QP handoff before publishing either gripper value so the
        # operation cannot be silently rejected after a successful arrival.
        claim = self.session.call("claim_eef_control")
        if not claim.get("ok"):
            return ToolResult(
                data={
                    "success": False,
                    "gripper_side": side,
                    "error": claim.get(
                        "error", "could not switch to gripper control"
                    ),
                    "error_code": "control_handoff_failed",
                    "log_path": claim.get("log_path"),
                }
            )
        result = self.session.call(action, side=side)
        return ToolResult(
            data={
                "success": result.get("ok", False),
                "gripper_side": side,
                **({"error": result.get("error")} if not result.get("ok") else {}),
                **result,
            }
        )

    def _release_clothes(self) -> ToolResult:
        """Claim the right gripper and release clothes after final arrival."""
        if self._requested_target() != "clothes" or not self.runner.completed:
            return ToolResult(
                data={
                    "error": "clothes payload is not visually confirmed",
                    "error_code": "not_held",
                }
            )
        claim = self.session.call("claim_eef_control")
        if not claim.get("ok"):
            return ToolResult(
                data={
                    "error": claim.get("error", "could not switch to gripper control"),
                    "error_code": "control_handoff_failed",
                    "success": False,
                    "workflow_success": False,
                    "gripper_side": "right",
                    "gripper_released": False,
                }
            )
        release = self.session.call("open", side="right")
        if not release.get("ok"):
            return ToolResult(
                data={
                    "error": release.get("error", "right gripper release failed"),
                    "error_code": "release_failed",
                    "success": False,
                    "workflow_success": False,
                    "gripper_side": "right",
                    "gripper_released": False,
                }
            )
        self.completed = True
        return ToolResult(
            data={
                "success": True,
                "workflow_success": True,
                "payload_confirmed": True,
                "gripper_side": "right",
                "gripper_released": True,
                "message": "arrived at the requested destination and opened the right gripper",
            }
        )

    @tool(readonly=True)
    def list_locations(self) -> ToolResult:
        """List canonical world-frame locations accepted by navigate_to."""
        target = self._requested_target()
        policy_ready: bool | None = None
        policy_error: str | None = None
        if target and not self.initially_holding and self.runner.execute:
            try:
                self.runner.validate(target)
                policy_ready = True
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                # Read-only preflight: expose the concrete model/config error
                # before the Agent moves the base to the source station.
                policy_ready = False
                policy_error = str(exc)
        return ToolResult(
            data={
                "locations": self.navigation.available_locations(),
                "instruction": self.instruction,
                "task_mode": task_mode(self.instruction),
                "target": target,
                "payload_state": (
                    "operator_declared_held"
                    if self.initially_holding
                    else "empty_or_unknown"
                ),
                "gripper_side": self._gripper_side(),
                "policy_ready": policy_ready,
                **({"policy_error": policy_error} if policy_error else {}),
            }
        )

    @tool
    def navigate_to(self, location: str) -> ToolResult:
        """Move the base to one named location and verify that single arrival."""
        if self.completed:
            return ToolResult(data={"error": "task already completed"})
        result = self.navigation.navigate(location)
        if result.is_error or result.data.get("arrival_ok") is not True:
            return result
        # Clothes transport intentionally has no post-grasp body/EEF motion.
        # Once the requested final station is reached, reclaim only the right
        # gripper command source and open it; the operator asked for release,
        # not a second visual placement gate.
        if (
            self._requested_target() == "clothes"
            and self.runner.completed
            and instruction_destination_after_target(self.instruction, location)
        ):
            release = self._release_clothes()
            result.data.update(release.data)
        return result

    @tool(readonly=True)
    def verify_location(self, location: str) -> ToolResult:
        """Check the current base pose against one configured location."""
        return self.navigation.verify(location)

    @tool
    def observe_scene(self) -> ToolResult:
        """Read the selected color-side EEF pose and three-view image."""
        observed = self.eef.observe_cube()
        if observed.is_error:
            return observed
        if self.initial_xyz is None:
            self.initial_xyz = list(observed.data["eef_xyz_m"])
            self.before_jpeg = self.client.camera_jpeg()
        observed.data["target_instruction"] = self.instruction
        observed.data["act_enabled"] = (
            self.runner.execute and self.runner.backend != "direct_joint"
        )
        observed.data["policy_enabled"] = self.runner.execute
        observed.data["policy_backend"] = getattr(self.runner, "backend", "act")
        observed.data["clothes_pose_sequence_enabled"] = (
            self.runner.execute
            and self.runner.backend == "direct_joint"
            and self._requested_target() == "clothes"
        )
        observed.data["eef_side"] = self.eef.side
        observed.data["task_mode"] = task_mode(self.instruction)
        observed.data["payload_state"] = (
            "operator_declared_held" if self.initially_holding else "empty_or_unknown"
        )
        return observed

    @tool
    def move_delta(self, dx_mm: float, dy_mm: float, dz_mm: float) -> ToolResult:
        """Move the selected color-side EEF at most 200 mm; no base movement."""
        if self.completed:
            return ToolResult(data={"error": "task already completed"})
        if self.initially_holding:
            return ToolResult(
                data={"error": "held-object transport does not permit EEF motion"}
            )
        if self.runner.completed:
            return ToolResult(
                data={"error": "task 4 does not permit post-grasp EEF retract motion"}
            )
        if self.runner.attempted and not self.handoff_ok:
            return ToolResult(
                data={"error": "policy control handoff failed; do not move"}
            )
        return self.eef.move_eef_delta(dx_mm, dy_mm, dz_mm)

    @tool
    def rotate_delta(
        self, droll_deg: float, dpitch_deg: float, dyaw_deg: float
    ) -> ToolResult:
        """Rotate the selected color-side EEF at most 10 degrees about base_link axes."""
        if self.completed:
            return ToolResult(data={"error": "task already completed"})
        if self.initially_holding:
            return ToolResult(
                data={"error": "held-object transport does not permit EEF motion"}
            )
        if self.runner.completed:
            return ToolResult(
                data={"error": "task 4 does not permit post-grasp EEF retract motion"}
            )
        if self.runner.attempted and not self.handoff_ok:
            return ToolResult(
                data={"error": "policy control handoff failed; do not rotate"}
            )
        return self.eef.rotate_eef_delta(droll_deg, dpitch_deg, dyaw_deg)

    @tool
    def open_gripper(self) -> ToolResult:
        """Open the trained policy's gripper before staging."""
        if self.initially_holding:
            return ToolResult(
                data={"error": "operator declared payload held; do not open the gripper"}
            )
        if self.runner.attempted:
            return ToolResult(
                data={
                    "error": "the policy has attempted a grasp; do not release the gripper"
                }
            )
        return self._command_gripper("open")

    @tool
    def close_gripper(self) -> ToolResult:
        """Close the trained policy's gripper; closure is not grasp proof."""
        if self.initially_holding:
            return ToolResult(
                data={"error": "held-object transport preserves the gripper state"}
            )
        if self.runner.attempted:
            return ToolResult(
                data={"error": "the policy owns the grasp; do not close again"}
            )
        return self._command_gripper("close")

    @tool
    def vla_grasp(
        self, color: Literal["red", "yellow", "blue", "clothes"]
    ) -> ToolResult:
        """Run the requested learned policy once after stopping the EEF publisher."""
        if self.initially_holding:
            return ToolResult(
                data={
                    "error": "operator declared payload already held; policy grasp is forbidden",
                    "phase": "preflight",
                }
            )
        expected = self._requested_target()
        if expected is None or color != expected:
            return ToolResult(
                data={
                    "error": "target must uniquely match the natural-language instruction",
                    "phase": "preflight",
                }
            )
        if not self.runner.execute or not self.session.execute:
            return ToolResult(
                data={
                    "error": "EEF and selected policy motion must both be enabled",
                    "phase": "preflight",
                }
            )
        if self.initial_xyz is None:
            return ToolResult(
                data={
                    "error": "call observe_scene before vla_grasp",
                    "phase": "preflight",
                }
            )
        if not self.runner.can_attempt():
            return ToolResult(
                data={
                    "error": "policy grasp retry is not prepared or exhausted",
                    "phase": "preflight",
                    "attempts": self.runner.attempt_count,
                }
            )
        try:
            self.runner.validate(color)
            side = grasp_side(color)
            if self.eef.side != side:
                return ToolResult(
                    data={
                        "error": f"{side} policy requires a {side} EEF session",
                        "phase": "preflight",
                    }
                )
            # Task 4 policy runs do not perform an automatic EEF retract.  A
            # clothes policy may either finish after grasp confirmation or
            # continue into the requested destination transport.
            self.return_xyz = None
            self.before_jpeg = self.client.camera_jpeg()
            self.session.close()
            result = self.runner.run(color, verify=False)
            if result.is_error:
                return result
            if color == "clothes":
                self.handoff_ok = True
                if self.runner.backend == "direct_joint":
                    control_handoff = {
                        "mode": "direct_joint_home_hold",
                        "eef_handoff": False,
                        "gripper_hold": "right",
                    }
                else:
                    # Legacy clothes ACT leaves control on its final source.
                    control_handoff = {
                        "mode": "policy_exit_base_transport",
                        "eef_handoff": False,
                        "gripper_hold": "unchanged",
                    }
            else:
                claim = self.session.call("claim_eef_control")
                if not claim.get("ok"):
                    return ToolResult(
                        data={
                            "error": claim.get(
                                "error", "could not switch policy control back to EEF"
                            ),
                            "error_code": "control_handoff_failed",
                            "phase": "handoff",
                            "retryable": False,
                            "log_path": claim.get("log_path"),
                        }
                    )
                hold = self.session.call("hold_closed", side=side)
                if not hold.get("ok"):
                    return ToolResult(
                        data={
                            "error": hold.get("error", "could not resume gripper hold"),
                            "error_code": "gripper_hold_failed",
                            "phase": "handoff",
                            "retryable": False,
                            "log_path": hold.get("log_path"),
                        }
                    )
                self.handoff_ok = True
                control_handoff = {
                    "mode": "eef_gripper_hold",
                    "eef_handoff": True,
                    "gripper_hold": side,
                }
            comparison = comparison_png(
                self.before_jpeg,
                self.client.camera_jpeg(),
                after_label=f"AFTER {self.runner.backend.upper()}",
                wrist_side=side,
            )
            verdict = self.runner.visual_verifier(
                comparison,
            target="clothes" if color == "clothes" else f"{color} block",
                after_stage=self.runner.backend.upper(),
                wrist_side=side,
            )
            self.runner.completed = verdict["verdict"] == "held"
            self.last_verdict = verdict["verdict"]
            self.grasp_retracted = self.runner.completed
            image_path = (
                self.runner.output_dir
                / f"luna_{self.runner.backend}_{color}_comparison.png"
            )
            image_path.write_bytes(comparison)
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            return ToolResult(data={"error": str(exc), "phase": "handoff"})
        result.data.update(
            payload_confirmed=self.runner.completed,
            attempts=self.runner.attempt_count,
            retryable=(
                self.runner.backend != "direct_joint"
                and not self.runner.completed
                and self.runner.attempt_count < 2
            ),
            verdict=verdict["verdict"],
            visual_evidence=verdict["evidence"],
            visual_image_path=str(image_path),
            return_xyz_m=self.return_xyz,
            gripper_side=side,
            requires_retract=False,
            control_handoff=control_handoff,
        )
        result.images.append(comparison)
        if not self.runner.completed:
            result.data.update(
                error="target grasp was not visually confirmed",
                error_code="not_held",
            )
            if self.runner.backend != "direct_joint":
                result.data["next_action"] = "prepare_grasp_retry"
            return result
        if color == "clothes" and self.runner.completed:
            transport = instruction_requests_transport(self.instruction)
            self.completed = not transport
            result.data["workflow_success"] = self.completed
            result.data["message"] = (
                "clothes visually held; navigate to the requested destination and it will be released on arrival"
                if transport
                else "clothes visually held; no destination was requested"
            )
        else:
            result.data["workflow_success"] = False
            result.data["message"] = "payload visually held; navigate directly without EEF retract"
        return result

    @tool
    def prepare_grasp_retry(self) -> ToolResult:
        """At a high return pose, open the gripper and arm one final policy retry."""
        if self.initially_holding:
            return ToolResult(
                data={"error": "held-object transport cannot prepare a grasp retry"}
            )
        if self.last_verdict != "not_held" or self.runner.attempt_count != 1:
            return ToolResult(
                data={
                    "error": "retry is allowed only after the first not_held verdict",
                    "retryable": False,
                }
            )
        side = self._gripper_side()
        if self._requested_target() == "clothes":
            # The first clothes rollout intentionally leaves the full-joint
            # policy source active.  Reclaim source 0 only for the recovery
            # open command, then the next rollout takes ownership again.
            claim = self.session.call("claim_eef_control")
            if not claim.get("ok"):
                return ToolResult(
                    data={
                        "error": claim.get("error", "could not switch policy control back to EEF"),
                        "error_code": "control_handoff_failed",
                        "retryable": False,
                    }
                )
        opened = self.session.call("open", side=side)
        if not opened.get("ok"):
            return ToolResult(
                data={"error": opened.get("error", "gripper open failed")}
            )
        if not self.runner.prepare_retry():
            return ToolResult(data={"error": "policy retry could not be armed"})
        self.handoff_ok = False
        self.before_jpeg = self.client.camera_jpeg()
        return ToolResult(
            data={
                "retry_prepared": True,
                "gripper_side": side,
                "attempts": self.runner.attempt_count,
                "remaining_attempts": 1,
            }
        )

    @tool
    def verify_goal(self) -> ToolResult:
        """Confirm held object and selected EEF return near its high pose."""
        if self.initially_holding:
            return ToolResult(
                data={"error": "held-object transport has no grasp/retract verification phase"}
            )
        side = self._gripper_side()
        if self._requested_target() == "clothes" and self.runner.completed:
            if instruction_requests_transport(self.instruction):
                return ToolResult(
                    data={
                        "workflow_success": False,
                        "grasp_retracted": True,
                        "payload_confirmed": True,
                        "verdict": self.last_verdict,
                        "gripper_side": side,
                        "requires_retract": False,
                        "message": "clothes held; navigate to the requested destination for automatic release",
                    }
                )
            self.completed = True
            self.grasp_retracted = True
            return ToolResult(
                data={
                    "workflow_success": True,
                    "grasp_retracted": True,
                    "payload_confirmed": True,
                    "verdict": self.last_verdict,
                    "gripper_side": side,
                    "requires_retract": False,
                    "message": "clothes visually held; no destination was requested",
                }
            )
        if self.runner.completed and self.handoff_ok:
            self.grasp_retracted = True
            return ToolResult(
                data={
                    "workflow_success": False,
                    "grasp_retracted": True,
                    "payload_confirmed": True,
                    "verdict": self.last_verdict,
                    "gripper_side": side,
                    "requires_retract": False,
                    "message": "task 4 has no post-grasp EEF retract; navigate to the destination",
                }
            )
        if (
            not self.runner.completed
            or not self.handoff_ok
            or self.return_xyz is None
            or self.before_jpeg is None
        ):
            return ToolResult(data={"error": "no confirmed policy grasp to verify"})
        state = self.session.call("state")
        if not state.get("ok"):
            return ToolResult(
                data={"error": state.get("error", "EEF state unavailable")}
            )
        xyz = state["xyz_m"]
        if (
            xyz[2] < self.session.table_z_m + 0.15
            or math.dist(xyz, self.return_xyz) > 0.06
        ):
            return ToolResult(
                data={
                    "error": "EEF has not returned near the high-clearance pose",
                    "eef_xyz_m": xyz,
                    "return_xyz_m": self.return_xyz,
                    "return_error_m": math.dist(xyz, self.return_xyz),
                }
            )
        comparison = comparison_png(
            self.before_jpeg, self.client.camera_jpeg(), wrist_side=side
        )
        verdict = self.runner.visual_verifier(
            comparison,
            target=(
                "clothes"
                if self._requested_target() == "clothes"
                else f"{self._requested_target()} block"
            ),
            after_stage="RETRACT",
            wrist_side=side,
        )
        self.grasp_retracted = verdict["verdict"] == "held"
        self.runner.completed = self.grasp_retracted
        self.last_verdict = verdict["verdict"]
        retryable = not self.grasp_retracted and self.runner.attempt_count < 2
        image_path = (
            self.runner.output_dir / f"luna_{self.runner.backend}_goal_comparison.png"
        )
        image_path.write_bytes(comparison)
        return ToolResult(
            data={
                "workflow_success": False,
                "grasp_retracted": self.grasp_retracted,
                "payload_confirmed": self.grasp_retracted,
                "verdict": verdict["verdict"],
                "gripper_side": side,
                "eef_side": self.eef.side,
                "visual_evidence": verdict["evidence"],
                "retryable": retryable,
                "eef_xyz_m": xyz,
                "return_xyz_m": self.return_xyz,
                "visual_image_path": str(image_path),
                **(
                    {
                        "error": "object not held after retract",
                        "error_code": "not_held",
                        "next_action": "prepare_grasp_retry",
                    }
                    if not self.grasp_retracted
                    else {}
                ),
            },
            images=[comparison],
        )

    @tool
    def verify_delivery(self, location: str) -> ToolResult:
        """Confirm delivery, then release the selected gripper at the final station."""
        if (
            self._requested_target() == "clothes"
            and instruction_requests_transport(self.instruction)
        ):
            return ToolResult(
                data={
                    "error": "clothes transport releases automatically in final navigate_to",
                    "error_code": "automatic_release_required",
                }
            )
        if (
            not (self.initially_holding or self.grasp_retracted)
            or self.before_jpeg is None
        ):
            return ToolResult(data={"error": "no verified retracted grasp to deliver"})
        arrival = self.navigation.verify(location)
        if arrival.is_error:
            return arrival
        try:
            side = self._gripper_side()
            comparison = comparison_png(
                self.before_jpeg,
                self.client.camera_jpeg(),
                after_label="AT DESTINATION",
                wrist_side=side,
            )
            verdict = self.runner.visual_verifier(
                comparison,
                target=(
                    "clothes"
                    if self._requested_target() == "clothes"
                    else f"{self._requested_target()} block"
                ),
                after_stage="DESTINATION",
                wrist_side=side,
            )
            snapshot = self.client.get("/state")
            gripper = snapshot.get(f"gripper_{side}", {})
            gripper_ok = isinstance(gripper, dict) and gripper.get("error") == 1
            held = verdict["verdict"] == "held"
            release = (
                self.session.call("open", side=side) if held and gripper_ok else {}
            )
            released = bool(release.get("ok"))
            self.completed = held and gripper_ok and released
            image_path = (
                self.runner.output_dir
                / f"luna_{self.runner.backend}_delivery_comparison.png"
            )
            image_path.write_bytes(comparison)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            return ToolResult(data={"error": str(exc), "phase": "delivery_verify"})
        data = {
            **arrival.data,
            "workflow_success": self.completed,
            "payload_confirmed": held,
            "gripper_side": side,
            "gripper_ok": gripper_ok,
            f"{side}_gripper_ok": gripper_ok,
            "gripper_released": released,
            "verdict": verdict["verdict"],
            "visual_evidence": verdict["evidence"],
            "visual_image_path": str(image_path),
        }
        if not self.completed:
            data.update(
                error=(
                    "delivery could not be confirmed"
                    if not (held and gripper_ok)
                    else release.get("error", "gripper release failed")
                ),
                error_code=(
                    "not_held"
                    if not held
                    else (f"{side}_gripper_not_ready" if not gripper_ok else "release_failed")
                ),
            )
        return ToolResult(data=data, images=[comparison])
