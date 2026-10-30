# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Planner-facing Luna workflow tools."""

from __future__ import annotations

import math
import time

import yaml

from robots.luna.client import (
    POSE_RECOVERY_POLL_S,
    POSE_RECOVERY_TIMEOUT_S,
    LunaVLAHostClient,
    LunaWorkflowRunner,
)
from rpent.tools import ToolResult
from rpent.tools.base import tool


class LunaPrimitives:
    """Bind task-safe Luna actions to one run."""

    def __init__(
        self, *, runner: LunaWorkflowRunner, client: LunaVLAHostClient
    ) -> None:
        self.runner = runner
        self.client = client

    @tool(readonly=True)
    def inspect_luna(self) -> ToolResult:
        """Return a compact machine verdict without commanding motion."""
        deadline = time.monotonic() + POSE_RECOVERY_TIMEOUT_S
        while True:
            self.runner.check_cancelled()
            result = summarize_luna_snapshot(self.client.snapshot())
            age = result["pose_age_s"]
            if not (
                str(result.get("error", "")).startswith("Luna world-base pose is stale")
                and result["left_gripper"]["ok"]
                and result["right_gripper"]["ok"]
                and isinstance(age, (int, float))
                and age > 0.5
            ):
                return ToolResult(data=result)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ToolResult(data=result)
            time.sleep(min(POSE_RECOVERY_POLL_S, remaining))

    @tool
    def move_to_table_3(self) -> ToolResult:
        """Navigate to the calibrated table_3 pose. Call once before the grasp tool."""
        return self.runner.move_to_table_3()

    @tool
    def pick_apriltag_module(self) -> ToolResult:
        """Pick the AprilTag-marked optical module from table 3, retract the left arm without a right shift, then move to the chair selected by the task. The gripper remains closed and the workflow ends without placing the object."""
        result = self.runner.pick_apriltag_module()
        if result.is_error:
            return result
        if result.data.get("payload_confirmed") is not True:
            result.data.update(
                error="Luna payload was not visually confirmed",
                error_code="grasp_not_visually_confirmed",
                workflow_success=False,
            )
            return result
        try:
            snapshot = self.client.snapshot()
            with (self.runner.script_dir / "pos.yaml").open(encoding="utf-8") as file:
                target = yaml.safe_load(file)[f"pose{self.runner.destination_pose}"][
                    "pose"
                ]
            actual = snapshot["base_world_pose"]["pose"]
            expected = target["position"]
            quaternion = target["orientation"]
            target_yaw = math.atan2(
                2
                * (
                    quaternion["w"] * quaternion["z"]
                    + quaternion["x"] * quaternion["y"]
                ),
                1 - 2 * (quaternion["y"] ** 2 + quaternion["z"] ** 2),
            )
            position = actual["position"]
            xy_error = math.hypot(
                position["x"] - expected["x"], position["y"] - expected["y"]
            )
            z_error = abs(position["z"] - expected["z"])
            yaw_error = abs(
                math.atan2(
                    math.sin(actual["yaw_rad"] - target_yaw),
                    math.cos(actual["yaw_rad"] - target_yaw),
                )
            )
            if not all(
                math.isfinite(value) for value in (xy_error, z_error, yaw_error)
            ):
                raise ValueError("non-finite destination pose error")
            inspection = summarize_luna_snapshot(snapshot)
            arrival_ok = bool(
                inspection["connected"]
                and inspection["pose_ok"]
                and xy_error <= 0.05
                and z_error <= 0.05
                and yaw_error <= 0.10
            )
            gripper_ok = _gripper_status(snapshot["state"]["gripper_left"])["ok"]
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            result.data.update(
                script_success=True,
                success=False,
                workflow_success=False,
                phase="delivery_verification",
                error_code="verification_unavailable",
                error=f"Cannot verify Luna delivery: {exc}",
            )
            result.data["message"] = result.data["error"]
            return result
        result.data.update(
            script_success=True,
            success=arrival_ok and gripper_ok,
            workflow_success=arrival_ok and gripper_ok,
            phase="delivery_verification",
            arrival_ok=arrival_ok,
            left_gripper_ok=gripper_ok,
            payload_confirmed=True,
            xy_error_m=xy_error,
            z_error_m=z_error,
            yaw_error_rad=yaw_error,
        )
        if not arrival_ok:
            result.data.update(
                error_code="destination_pose_unverified",
                error="Luna is not verified at the destination pose",
            )
        elif not gripper_ok:
            result.data.update(
                error_code="left_gripper_not_ready",
                error="Luna left gripper is not ready after transport",
            )
        else:
            result.data.update(
                error_code=None,
                message="Luna arrived at the destination with left gripper ready",
            )
            self.runner.completed = True
        if result.is_error:
            result.data["message"] = result.data["error"]
        return result


def summarize_luna_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    """Return the planner-facing safety verdict for one raw VLAHost snapshot."""
    state = snapshot.get("state", {})
    state = state if isinstance(state, dict) else {}
    base_world_pose = snapshot.get("base_world_pose", {})
    base_world_pose = base_world_pose if isinstance(base_world_pose, dict) else {}
    pose = base_world_pose.get("pose", {})
    pose = pose if isinstance(pose, dict) else {}
    age = pose.get("received_age_s")
    finite_age = (
        not isinstance(age, bool)
        and isinstance(age, (int, float))
        and math.isfinite(age)
    )
    result: dict[str, object] = {
        "ok": False,
        "connected": bool(state.get("connected")),
        "pose_ok": bool(
            pose.get("valid")
            and (pose.get("frame_id"), pose.get("child_frame_id"))
            == ("camera_init", "body")
            and finite_age
            and age <= 0.5
        ),
        "pose_age_s": age if finite_age else None,
        "left_gripper": _gripper_status(state.get("gripper_left")),
        "right_gripper": _gripper_status(state.get("gripper_right")),
    }
    if not state.get("connected"):
        result["error"] = "Luna VLAHost reports disconnected robot feedback"
    elif not pose.get("valid"):
        result["error"] = "Luna world-base pose is invalid"
    elif (pose.get("frame_id"), pose.get("child_frame_id")) != (
        "camera_init",
        "body",
    ):
        result["error"] = "Luna world-base pose must use camera_init -> body"
    elif not finite_age or age > 0.5:
        result["error"] = f"Luna world-base pose is stale: {age!r}s"
    elif not result["left_gripper"]["ok"]:
        result["error"] = "Luna left gripper is not ready"
    elif not result["right_gripper"]["ok"]:
        result["error"] = "Luna right gripper is not ready"
    else:
        result["ok"] = True
    return result


def _gripper_status(raw: object) -> dict[str, object]:
    """Normalize the DM gripper status code for planner consumption."""
    code = raw.get("error") if isinstance(raw, dict) else None
    if isinstance(code, bool) or not isinstance(code, (int, float)):
        code = None
    elif not math.isfinite(code) or not float(code).is_integer():
        code = None
    else:
        code = int(code)
    status = (
        "on"
        if code == 1
        else "off"
        if code == 0
        else "fault"
        if code in range(8, 15)
        else "unknown"
    )
    return {"status_code": code, "status": status, "ok": code == 1}
