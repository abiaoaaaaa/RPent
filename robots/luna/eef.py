# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Bounded Luna EEF exploration; the ROS child owns arm and gripper hold."""

from __future__ import annotations

import io
import json
import math
import os
import select
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Literal

from robots.luna.client import LunaVLAHostClient
from robots.luna.vision import comparison_png, verify_grasp
from rpent.tools import ToolResult
from rpent.tools.base import tool

REPLY_PREFIX = "RPENT_EEF="


class LunaEefSession:
    """Keep one checked ROS publisher alive across planner tool calls."""

    def __init__(
        self,
        *,
        script_dir: Path,
        python_executable: str,
        ros_setup_script: Path | None,
        output_dir: Path,
        check_cancelled: Callable[[], None],
        execute: bool,
        table_z_m: float,
        tool_clearance_m: float,
        side: Literal["left", "right"] = "left",
    ) -> None:
        self.script_dir = script_dir
        self.python_executable = python_executable
        self.ros_setup_script = ros_setup_script
        self.output_dir = output_dir
        self.check_cancelled = check_cancelled
        self.execute = execute
        self.table_z_m = table_z_m
        self.tool_clearance_m = tool_clearance_m
        if side not in ("left", "right"):
            raise ValueError("EEF side must be left or right")
        self.side = side
        self.process: subprocess.Popen[bytes] | None = None
        self.buffer = bytearray()
        self.serial = 0
        self.log_path = output_dir / "luna_eef_session.log"

    def _next_reply(self, timeout_s: float) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.check_cancelled()
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                with self.log_path.open("ab") as log:
                    log.write(line + b"\n")
                if line.startswith(REPLY_PREFIX.encode()):
                    return json.loads(line[len(REPLY_PREFIX) :])
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"EEF session exited with {self.process.returncode}; see {self.log_path}"
                )
            ready, _, _ = select.select(
                [self.process.stdout],
                [],
                [],
                max(0.0, min(0.1, deadline - time.monotonic())),
            )
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if chunk:
                    self.buffer.extend(chunk)
        raise TimeoutError(f"EEF session response timed out; see {self.log_path}")

    def start(self) -> None:
        if self.process is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_executable,
            str(self.script_dir / "luna_eef_session.py"),
            "--side",
            self.side,
            "--table-z-m",
            str(self.table_z_m),
            "--tool-clearance-m",
            str(self.tool_clearance_m),
        ]
        if self.execute:
            command.append("--execute")
        if self.ros_setup_script:
            command = [
                "/usr/bin/bash",
                "-c",
                'source "$1" && shift && exec "$@"',
                "rpent-luna",
                str(self.ros_setup_script),
                *command,
            ]
        env = os.environ.copy()
        for key in (
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
        ):
            env.pop(key, None)
        self.process = subprocess.Popen(
            command,
            cwd=self.script_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        try:
            ready = self._next_reply(12.0)
            if ready.get("ready") is not True:
                raise RuntimeError("EEF session did not become ready")
        except BaseException:
            self.close()
            raise

    def call(self, action: str, **arguments: Any) -> dict[str, Any]:
        self.start()
        assert self.process is not None and self.process.stdin is not None
        self.serial += 1
        request = {"id": self.serial, "action": action, **arguments}
        try:
            self.process.stdin.write((json.dumps(request) + "\n").encode())
            self.process.stdin.flush()
            result = self._next_reply(35.0 if action in {"open", "close"} else 25.0)
        except BaseException:
            self.close()
            raise
        if result.get("id") != self.serial:
            self.close()
            raise RuntimeError("EEF session returned a mismatched reply")
        return {**result, "log_path": str(self.log_path)}

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
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
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
        self.buffer.clear()


class LunaEefPrimitives:
    """Camera plus one session-owned EEF and its gripper; no base tools."""

    def __init__(
        self,
        *,
        session: LunaEefSession,
        client: LunaVLAHostClient,
        visual_verifier: Callable[..., dict[str, str]] = verify_grasp,
    ) -> None:
        self.session = session
        self.client = client
        self.side = getattr(session, "side", "left")
        self.before_jpeg: bytes | None = None
        self.closed_z: float | None = None
        self.lifted_m = 0.0
        self.completed = False
        self.visual_verifier = visual_verifier

    @tool
    def observe_cube(self) -> ToolResult:
        """Read a fresh three-view image and the selected EEF pose."""
        from PIL import Image

        state = self.session.call("state")
        if not state.get("ok"):
            return ToolResult(data={"error": state.get("error"), **state})
        jpeg = self.client.camera_jpeg()
        if self.closed_z is None:
            self.before_jpeg = jpeg
        with Image.open(io.BytesIO(jpeg)) as image:
            output = io.BytesIO()
            image.save(output, format="PNG")
        return ToolResult(
            data={
                "success": True,
                "ok": True,
                "eef_xyz_m": state["xyz_m"],
                "eef_side": self.side,
                "table_z_m": self.session.table_z_m,
                "min_tool_z_m": self.session.table_z_m + self.session.tool_clearance_m,
                "motion_enabled": self.session.execute,
            },
            images=[output.getvalue()],
        )

    def move_eef_delta(
        self, dx_mm: float, dy_mm: float, dz_mm: float
    ) -> ToolResult:
        """Move the selected tool in base_link axes by up to 200 mm."""
        if self.completed:
            return ToolResult(data={"error": "cube grasp was confirmed; stop the task"})
        values = (dx_mm, dy_mm, dz_mm)
        if (
            any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in values
            )
            or math.dist(values, (0, 0, 0)) > 200
            or not any(values)
        ):
            return ToolResult(
                data={
                    "error": "EEF delta must be a finite nonzero vector of at most 200 mm"
                }
            )
        result = self.session.call("move", delta_mm=list(values))
        if not result.get("ok"):
            return ToolResult(data={"error": result.get("error"), **result})
        if self.closed_z is not None and dz_mm > 0:
            self.lifted_m = max(
                self.lifted_m, result["actual_xyz_m"][2] - self.closed_z
            )
        return ToolResult(data={"success": True, **result})

    @tool
    def move_left_eef_delta(
        self, dx_mm: float, dy_mm: float, dz_mm: float
    ) -> ToolResult:
        """Move left_tool only when the session owns the left EEF."""
        if self.side != "left":
            return ToolResult(data={"error": "left EEF is not selected for this session"})
        return self.move_eef_delta(dx_mm, dy_mm, dz_mm)

    def move_right_eef_delta(
        self, dx_mm: float, dy_mm: float, dz_mm: float
    ) -> ToolResult:
        """Move right_tool only when the session owns the right EEF."""
        if self.side != "right":
            return ToolResult(data={"error": "right EEF is not selected for this session"})
        return self.move_eef_delta(dx_mm, dy_mm, dz_mm)

    def rotate_eef_delta(
        self, droll_deg: float, dpitch_deg: float, dyaw_deg: float
    ) -> ToolResult:
        """Rotate the selected tool about base_link XYZ axes by at most 10 degrees."""
        if self.completed:
            return ToolResult(data={"error": "grasp was confirmed; stop the task"})
        values = (droll_deg, dpitch_deg, dyaw_deg)
        if (
            any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in values
            )
            or math.dist(values, (0, 0, 0)) > 10
            or not any(values)
        ):
            return ToolResult(
                data={
                    "error": "EEF rotation must be a finite nonzero vector of at most 10 degrees"
                }
            )
        result = self.session.call("rotate", delta_deg=list(values))
        return ToolResult(
            data={"success": True, **result}
            if result.get("ok")
            else {"error": result.get("error"), **result}
        )

    def rotate_left_eef_delta(
        self, droll_deg: float, dpitch_deg: float, dyaw_deg: float
    ) -> ToolResult:
        """Rotate left_tool only when the session owns the left EEF."""
        if self.side != "left":
            return ToolResult(data={"error": "left EEF is not selected for this session"})
        return self.rotate_eef_delta(droll_deg, dpitch_deg, dyaw_deg)

    def rotate_right_eef_delta(
        self, droll_deg: float, dpitch_deg: float, dyaw_deg: float
    ) -> ToolResult:
        """Rotate right_tool only when the session owns the right EEF."""
        if self.side != "right":
            return ToolResult(data={"error": "right EEF is not selected for this session"})
        return self.rotate_eef_delta(droll_deg, dpitch_deg, dyaw_deg)

    def open_gripper(self) -> ToolResult:
        """Open the gripper that belongs to the selected EEF session."""
        if self.completed:
            return ToolResult(
                data={"error": "cube grasp was confirmed; do not release it"}
            )
        result = self.session.call("open", side=self.side)
        if result.get("ok"):
            self.before_jpeg = None
            self.closed_z = None
            self.lifted_m = 0.0
            self.completed = False
        return ToolResult(
            data={
                "success": result.get("ok", False),
                "gripper_side": self.side,
                **({"error": result.get("error")} if not result.get("ok") else {}),
                **result,
            }
        )

    def close_gripper(self) -> ToolResult:
        """Close the gripper that belongs to the selected EEF session."""
        if self.closed_z is not None:
            return ToolResult(data={"error": f"{self.side} gripper is already closed"})
        if self.before_jpeg is None:
            return ToolResult(
                data={"error": "observe the cube immediately before closing"}
            )
        result = self.session.call("close", side=self.side)
        if result.get("ok"):
            pose = self.session.call("state")
            if not pose.get("ok"):
                return ToolResult(
                    data={
                        "error": "gripper closed but EEF pose is unavailable",
                        "gripper_result": result,
                        "pose_result": pose,
                    }
                )
            self.closed_z = pose["xyz_m"][2]
            self.lifted_m = 0.0
        return ToolResult(
            data={
                "success": result.get("ok", False),
                "gripper_side": self.side,
                **({"error": result.get("error")} if not result.get("ok") else {}),
                **result,
            }
        )

    @tool
    def open_left_gripper(self) -> ToolResult:
        """Open the left gripper; never use while a payload might be held."""
        if self.side != "left":
            return ToolResult(data={"error": "left EEF is not selected for this session"})
        return self.open_gripper()

    @tool
    def close_left_gripper(self) -> ToolResult:
        """Close the left gripper gradually at the current EEF pose; holding is not proof of grasp."""
        if self.side != "left":
            return ToolResult(data={"error": "left EEF is not selected for this session"})
        return self.close_gripper()

    @tool
    def verify_cube_grasp(self) -> ToolResult:
        """After a lift, compare fresh left-wrist images and ask the VLM if the cube is held."""
        if self.before_jpeg is None or self.closed_z is None or self.lifted_m < 0.03:
            return ToolResult(
                data={"error": "observe before grasp, close, then lift at least 30 mm"}
            )
        comparison = comparison_png(self.before_jpeg, self.client.camera_jpeg())
        path = self.session.output_dir / "luna_cube_grasp_comparison.png"
        path.write_bytes(comparison)
        verdict = self.visual_verifier(comparison, target="cube or block")
        self.completed = verdict["verdict"] == "held"
        return ToolResult(
            data={
                "verdict": verdict["verdict"],
                "visual_evidence": verdict["evidence"],
                "payload_confirmed": self.completed,
                "workflow_success": self.completed,
                "visual_image_path": str(path),
            },
            images=[comparison],
        )
