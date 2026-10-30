# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Thin adapters for the validated Luna ROS and VLAHost workflow."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from rpent.tools import ToolResult

RESULT_PREFIX = "RPENT_RESULT="
GATE_PREFIX = "RPENT_GATE=await_visual_verdict"
POSE_RECOVERY_TIMEOUT_S = 10.0
POSE_RECOVERY_POLL_S = 0.2


class LunaVLAHostClient:
    """Read the externally managed VLAHost HTTP service."""

    def __init__(self, base_url: str, *, timeout_s: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._opener = build_opener(ProxyHandler({}))

    def get(self, path: str) -> dict[str, Any]:
        """Return one JSON object from VLAHost."""
        request = Request(self.base_url + path, method="GET")
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                payload = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLAHost HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read VLAHost {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"VLAHost {path} did not return a JSON object")
        return payload

    def snapshot(self) -> dict[str, Any]:
        """Read controller state and the current world-frame base pose."""
        return {
            "vlahost_url": self.base_url,
            "state": self.get("/state"),
            "base_world_pose": self.get("/base_world_pose"),
        }

    def camera_jpeg(self) -> bytes:
        """Read one fresh VLAHost head/left-wrist/right-wrist mosaic."""
        snapshot = self.get("/hil/snapshot")
        freshness = snapshot.get("freshness", {}).get("jpeg", {})
        age = freshness.get("age_s")
        source_age = freshness.get("source_age_s")
        if not all(
            isinstance(value, (int, float)) and 0 <= value <= 0.5
            for value in (age, source_age)
        ):
            raise RuntimeError("VLAHost camera frame is stale or has no freshness data")
        encoded = snapshot.get("jpeg_base64")
        if not isinstance(encoded, str) or not encoded:
            raise RuntimeError("VLAHost camera frame is unavailable")
        try:
            frame = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError("VLAHost camera frame is not valid base64") from exc
        if not frame.startswith(b"\xff\xd8") or len(frame) > 10_000_000:
            raise RuntimeError("VLAHost camera frame is not a bounded JPEG")
        from PIL import Image

        try:
            with Image.open(io.BytesIO(frame)) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            raise RuntimeError("VLAHost camera JPEG cannot be decoded") from exc
        if width % 2 or height % 3 or height * 8 != width * 15:
            raise RuntimeError(
                f"VLAHost camera is not the expected three-view mosaic: {width}x{height}"
            )
        return frame


class LunaWorkflowRunner:
    """Run the Luna scripts as cancellable Harness actions."""

    def __init__(
        self,
        *,
        script_dir: Path,
        python_executable: str,
        vlahost_url: str,
        destination_pose: int,
        tag_id: int,
        output_dir: Path,
        check_cancelled: Callable[[], None],
        navigation_timeout_s: float,
        grasp_timeout_s: float,
        ros_setup_script: Path | None = None,
        calibration_file: str | None = None,
        intrinsics_file: str | None = None,
        image_client: LunaVLAHostClient | None = None,
        visual_verifier: Callable[[bytes], dict[str, str]] | None = None,
    ) -> None:
        self.script_dir = script_dir
        self.python_executable = python_executable
        self.vlahost_url = vlahost_url.rstrip("/")
        self.destination_pose = destination_pose
        self.tag_id = tag_id
        self.output_dir = output_dir
        self.check_cancelled = check_cancelled
        self.navigation_timeout_s = navigation_timeout_s
        self.grasp_timeout_s = grasp_timeout_s
        self.ros_setup_script = ros_setup_script
        self.calibration_file = calibration_file
        self.intrinsics_file = intrinsics_file
        self.completed = False
        self.cable_reference_yaw_rad: float | None = None
        self.image_client = image_client or LunaVLAHostClient(self.vlahost_url)
        self.visual_verifier = visual_verifier

    def move_to_table_3(self) -> ToolResult:
        """Move the base to the configured table 3 world pose."""
        self.cable_reference_yaw_rad = None
        deadline = time.monotonic() + POSE_RECOVERY_TIMEOUT_S
        try:
            while True:
                self.check_cancelled()
                pose = self.image_client.get("/base_world_pose")["pose"]
                yaw = float(pose["yaw_rad"])
                age = float(pose["received_age_s"])
                if (
                    pose.get("valid") is not True
                    or pose.get("frame_id") != "camera_init"
                    or pose.get("child_frame_id") != "body"
                    or not math.isfinite(yaw)
                    or not math.isfinite(age)
                    or age < 0
                ):
                    raise ValueError("world-base pose is not fresh and valid")
                if age <= 0.5:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError(
                        "world-base pose stayed stale for "
                        f"{POSE_RECOVERY_TIMEOUT_S:g}s (age={age:.3f}s)"
                    )
                time.sleep(min(POSE_RECOVERY_POLL_S, remaining))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return ToolResult(data={"error": f"Cannot lock cable reference: {exc}"})
        command = [
            self.python_executable,
            str(self.script_dir / "base_move_world_pose_list.py"),
            '["table_3"]',
            "--vlahost-url",
            self.vlahost_url,
            "--odometry-retries",
            "1",
            "--yaw-reference-rad",
            str(yaw),
        ]
        result = self._run(
            self._with_ros(command), "move_table_3", self.navigation_timeout_s
        )
        if not result.is_error:
            self.cable_reference_yaw_rad = yaw
        return result

    def pick_apriltag_module(self) -> ToolResult:
        """Pick tag 5, retract, and transport it to the task-bound chair."""
        self.completed = False
        if self.cable_reference_yaw_rad is None:
            return ToolResult(
                data={"error": "Table navigation has not locked a cable reference"}
            )
        command = [
            self.python_executable,
            str(self.script_dir / "apriltag_left_tool_grasp_test.py"),
            "--execute",
            "--tag-id",
            str(self.tag_id),
            "--position-tolerance-mm",
            "10.0",
            "--max-target-angle-deg",
            "90.0",
            "--destination-pose",
            str(self.destination_pose),
            "--require-visual-verdict",
            "--home-on-visual-reject",
            "--cable-reference-yaw-rad",
            str(self.cable_reference_yaw_rad),
            "--vlahost-url",
            self.vlahost_url,
            "--offset-config",
            str(self.script_dir / f"tag{self.tag_id}_left_tool_offset.yaml"),
            "--return-pose-file",
            str(self.script_dir / "left_tool_joint_snapshot_20261010.yaml"),
        ]
        if self.calibration_file:
            command.extend(["--calibration-file", self.calibration_file])
        if self.intrinsics_file:
            command.extend(["--intrinsics-file", self.intrinsics_file])
        first_log_path = None
        for attempt in (1, 2):
            label = f"pick_to_chair_{self.destination_pose}"
            if attempt == 2:
                label += "_attempt_2"
            result = self._run(
                self._with_ros(command), label, self.grasp_timeout_s,
                visual_gate=True,
            )
            result.data["attempts"] = attempt
            if first_log_path is not None:
                result.data["first_attempt_log_path"] = first_log_path
            if result.data.get("phase") != "retry_ready":
                return result
            if result.data.get("verdict") != "not_held" or result.data.get("returncode") != 0:
                result.data.update(
                    error="Invalid Luna visual retry handoff",
                    error_code="invalid_retry_handoff",
                )
                return result
            if attempt == 2:
                result.data.update(
                    error="Luna grasp not held after two attempts; arm returned home",
                    error_code="visual_retry_exhausted",
                    workflow_success=False,
                )
                return result
            first_log_path = result.data["log_path"]
        raise AssertionError("unreachable Luna grasp attempt")

    def _run(
        self,
        command: list[str],
        label: str,
        timeout_s: float,
        *,
        visual_gate: bool = False,
    ) -> ToolResult:
        """Run one owned child process and preserve its output in the run directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / f"luna_{label}.log"
        before_jpeg = None
        if visual_gate:
            try:
                before_jpeg = self.image_client.camera_jpeg()
            except RuntimeError as exc:
                return ToolResult(
                    data={
                        "error": str(exc),
                        "phase": "visual_precheck",
                        "error_code": "camera_unavailable",
                        "message": str(exc),
                        "workflow": label,
                        "log_path": str(log_path),
                    }
                )
        started = time.monotonic()
        vision: dict[str, Any] = {}
        with log_path.open("w", encoding="utf-8") as log:
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
            process = subprocess.Popen(
                command,
                cwd=self.script_dir,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if visual_gate else None,
                text=True,
            )
            try:
                with log_path.open(encoding="utf-8", errors="replace") as monitor:
                    while process.poll() is None:
                        self.check_cancelled()
                        if visual_gate and not vision:
                            for line in monitor:
                                if line.startswith(GATE_PREFIX):
                                    vision = self._verify_at_gate(before_jpeg, label)
                                    assert process.stdin is not None
                                    try:
                                        process.stdin.write(
                                            {
                                                "held": "held\n",
                                                "not_held": "retry\n",
                                                "error": "reject\n",
                                            }[vision["verdict"]]
                                        )
                                        process.stdin.flush()
                                    except BrokenPipeError:
                                        # The child may time out while the VLM runs.
                                        pass
                                    break
                        if time.monotonic() - started > timeout_s:
                            self._interrupt(process)
                            message = f"Luna workflow timed out after {timeout_s:.1f}s"
                            return ToolResult(
                                data={
                                    "error": message,
                                    "phase": "unknown",
                                    "error_code": "workflow_timeout",
                                    "message": message,
                                    "workflow": label,
                                    "log_path": str(log_path),
                                }
                            )
                        time.sleep(0.1)
            except BaseException:
                self._interrupt(process)
                raise
        returncode = process.returncode
        payload = {
            "workflow": label,
            "returncode": returncode,
            "log_path": str(log_path),
        }
        payload.update(vision)
        if label.startswith("pick_to_chair_"):
            with log_path.open(encoding="utf-8", errors="replace") as log:
                for line in log:
                    if line.startswith(RESULT_PREFIX):
                        try:
                            outcome = json.loads(line[len(RESULT_PREFIX) :])
                        except json.JSONDecodeError:
                            continue
                        if isinstance(outcome, dict):
                            payload.update(
                                (key, outcome[key])
                                for key in ("phase", "error_code", "message")
                                if key in outcome
                            )
        if returncode != 0:
            payload.setdefault("phase", "unknown")
            payload.setdefault("error_code", "workflow_exit")
            payload.setdefault("message", f"Workflow exited with code {returncode}")
            payload["error"] = payload["message"]
        elif visual_gate and not vision:
            payload.update(
                error_code="missing_visual_gate",
                error="Luna grasp script did not request visual verification",
            )
        elif visual_gate and vision["verdict"] != "held":
            payload.update(
                error_code="grasp_not_visually_confirmed",
                error="Visual grasp verification did not confirm payload retention",
            )
        elif label.startswith("pick_to_chair_") and "phase" not in payload:
            payload.update(
                error_code="missing_structured_result",
                error="Luna grasp workflow did not report a structured outcome",
            )
        elif label.startswith("pick_to_chair_") and payload.get("error_code"):
            payload["error"] = payload.get("message", "Luna grasp workflow failed")
        else:
            payload["success"] = True
        return ToolResult(data=payload)

    def _verify_at_gate(self, before_jpeg: bytes | None, label: str) -> dict[str, Any]:
        """Call the VLM once while the ROS child keeps publishing hold commands."""
        from robots.luna.vision import comparison_png, verify_grasp

        try:
            if before_jpeg is None:
                raise RuntimeError("pre-grasp image is missing")
            after_jpeg = self.image_client.camera_jpeg()
            comparison = comparison_png(before_jpeg, after_jpeg)
            image_path = self.output_dir / f"luna_{label}_visual_comparison.png"
            image_path.write_bytes(comparison)
            verdict = (self.visual_verifier or verify_grasp)(comparison)
            if verdict.get("verdict") not in {"held", "not_held"}:
                raise ValueError("VLM returned an invalid grasp verdict")
            if (
                not isinstance(verdict.get("evidence"), str)
                or not verdict["evidence"].strip()
            ):
                raise ValueError("VLM returned no visual evidence")
            return {
                "verdict": verdict["verdict"],
                "visual_evidence": verdict["evidence"],
                "visual_image_path": str(image_path),
                "payload_confirmed": verdict["verdict"] == "held",
            }
        except Exception as exc:
            return {
                "verdict": "error",
                "visual_evidence": str(exc),
                "payload_confirmed": False,
            }

    def _with_ros(self, command: list[str]) -> list[str]:
        """Source the Luna ROS overlay before executing a workflow command."""
        if self.ros_setup_script is None:
            return command
        return [
            "/usr/bin/bash",
            "-c",
            'source "$1" && shift && exec "$@"',
            "rpent-luna",
            str(self.ros_setup_script),
            *command,
        ]

    @staticmethod
    def _interrupt(process: subprocess.Popen[str]) -> None:
        """Stop only the child process owned by this runner."""
        if process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
