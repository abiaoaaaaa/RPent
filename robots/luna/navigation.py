# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Single-destination Luna navigation through Apex's relative-motion action."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml

from robots.luna.client import (
    POSE_RECOVERY_POLL_S,
    POSE_RECOVERY_TIMEOUT_S,
    LunaVLAHostClient,
)
from rpent.tools import ToolResult

SOFA_CHAIR_ALIASES = ("沙发椅", "沙发椅子", "沙发", "sofa", "sofachair")
SOFA_CHAIR_STATION = "chair_1"


def load_locations(path: Path) -> dict[str, dict[str, Any]]:
    """Load canonical station names and poses from ``pos.yaml``."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"location file must be a YAML mapping: {path}")
    locations: dict[str, dict[str, Any]] = {}
    for key, entry in payload.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("station"), str):
            continue
        station = entry["station"].strip()
        pose = entry.get("pose")
        if not station or not isinstance(pose, dict):
            raise ValueError(f"invalid location entry {key!r} in {path}")
        if station in locations:
            raise ValueError(f"duplicate station {station!r} in {path}")
        locations[station] = {"pose_key": str(key), "pose": pose}
    if not locations:
        raise ValueError(f"no named locations in {path}")
    return locations


def _yaw_from_quaternion(value: dict[str, Any]) -> float:
    try:
        x, y, z, w = (float(value[name]) for name in ("x", "y", "z", "w"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("location orientation must be a quaternion") from exc
    if not all(math.isfinite(item) for item in (x, y, z, w)):
        raise ValueError("location quaternion contains a non-finite value")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("location quaternion has zero norm")
    x, y, z, w = (item / norm for item in (x, y, z, w))
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _nearest_yaw(target_yaw: float, reference_yaw: float) -> float:
    """Return the target-yaw equivalent nearest the cable-safe reference."""
    return target_yaw + 2.0 * math.pi * round(
        (reference_yaw - target_yaw) / (2.0 * math.pi)
    )


class LunaNavigationRunner:
    """Execute one named destination with Apex ``MoveRelative`` action."""

    def __init__(
        self,
        *,
        script_dir: Path,
        python_executable: str,
        vlahost_url: str,
        output_dir: Path,
        client: LunaVLAHostClient,
        check_cancelled: Callable[[], None],
        timeout_s: float,
        execute: bool,
        ros_setup_script: Path | None = None,
        max_attempts_per_location: int = 2,
    ) -> None:
        self.script_dir = script_dir
        self.python_executable = python_executable
        self.vlahost_url = vlahost_url.rstrip("/")
        self.output_dir = output_dir
        self.client = client
        self.check_cancelled = check_cancelled
        self.timeout_s = timeout_s
        self.execute = execute
        self.ros_setup_script = ros_setup_script
        self.max_attempts_per_location = max_attempts_per_location
        self.locations = load_locations(script_dir / "pos.yaml")
        self.cable_reference_yaw_rad: float | None = None
        self.attempts: dict[str, int] = {}
        self.process: subprocess.Popen[str] | None = None

    def available_locations(self) -> list[str]:
        """Return canonical station names in configuration order."""
        return list(self.locations)

    def resolve(self, location: str) -> str:
        """Resolve a conservative set of operator aliases to a station name."""
        raw = location.strip().lower().replace(" ", "")
        aliases: dict[str, str] = {}
        for station, entry in self.locations.items():
            key = entry["pose_key"].lower()
            aliases[station.lower()] = station
            aliases[key] = station
            suffix = station.rsplit("_", 1)[-1]
            if station.startswith("table_"):
                aliases.update(
                    {
                        suffix: station,
                        f"桌{suffix}": station,
                        f"{suffix}号桌子": station,
                    }
                )
            elif station.startswith("chair_"):
                aliases.update({f"椅{suffix}": station, f"{suffix}号椅子": station})
                if station == SOFA_CHAIR_STATION:
                    aliases.update(
                        (alias, station) for alias in SOFA_CHAIR_ALIASES
                    )
        try:
            return aliases[raw]
        except KeyError as exc:
            raise ValueError(
                f"unknown location {location!r}; choose one of {', '.join(self.locations)}"
            ) from exc

    def navigate(self, location: str) -> ToolResult:
        """Navigate to exactly one configured station and verify arrival."""
        try:
            station = self.resolve(location)
        except ValueError as exc:
            return ToolResult(data={"error": str(exc), "retryable": False})
        if not self.execute:
            return ToolResult(
                data={
                    "error": "navigation motion is disabled",
                    "error_code": "motion_disabled",
                    "location": station,
                    "retryable": False,
                }
            )
        attempt = self.attempts.get(station, 0) + 1
        if attempt > self.max_attempts_per_location:
            return ToolResult(
                data={
                    "error": f"navigation retry limit reached for {station}",
                    "error_code": "navigation_retry_exhausted",
                    "location": station,
                    "attempts": self.attempts.get(station, 0),
                    "retryable": False,
                }
            )
        self.attempts[station] = attempt
        try:
            pose = self._fresh_pose()
            if self.cable_reference_yaw_rad is None:
                # Lock before motion. A failed/partial first leg must not silently
                # redefine the cable zero on an Agent-requested retry.
                self.cable_reference_yaw_rad = float(pose["yaw_rad"])
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return ToolResult(
                data={
                    "error": f"cannot prepare navigation: {exc}",
                    "error_code": "pose_unavailable",
                    "location": station,
                    "attempts": attempt,
                    "retryable": attempt < self.max_attempts_per_location,
                }
            )
        try:
            command, motion = self._native_relative_command(station, pose)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return ToolResult(
                data={
                    "error": f"cannot compute native navigation target: {exc}",
                    "error_code": "navigation_target_invalid",
                    "location": station,
                    "attempts": attempt,
                    "retryable": attempt < self.max_attempts_per_location,
                }
            )
        result = self._run(command, station, attempt)
        result.data.update(motion)
        if result.is_error:
            result.data["retryable"] = attempt < self.max_attempts_per_location
            return result
        verified = self.verify(station)
        verified.data.update(
            motion,
            log_path=result.data["log_path"],
            returncode=0,
            attempts=attempt,
            retryable=(
                not verified.data.get("arrival_ok", False)
                and attempt < self.max_attempts_per_location
            ),
        )
        return verified

    def _native_relative_command(
        self, station: str, current: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        """Convert an absolute station pose to one body-frame native action."""
        target = self.locations[station]["pose"]
        expected = target["position"]
        current_position = current["position"]
        current_x = float(current_position["x"])
        current_y = float(current_position["y"])
        current_yaw = float(current["yaw_rad"])
        target_x = float(expected["x"])
        target_y = float(expected["y"])
        target_yaw = _yaw_from_quaternion(target["orientation"])
        reference = self.cable_reference_yaw_rad
        if reference is None:
            raise ValueError("cable yaw reference is not locked")
        target_yaw = _nearest_yaw(target_yaw, reference)
        world_dx = target_x - current_x
        world_dy = target_y - current_y
        # MoveRelative uses the robot/body frame for x/y.
        body_dx = math.cos(current_yaw) * world_dx + math.sin(current_yaw) * world_dy
        body_dy = -math.sin(current_yaw) * world_dx + math.cos(current_yaw) * world_dy
        relative_yaw = _wrap_angle(target_yaw - current_yaw)
        if abs(relative_yaw) > math.pi + 0.35:
            raise ValueError("native relative yaw exceeds the cable half-turn bound")
        command = [
            self.python_executable,
            str(self.script_dir / "native_nav_test.py"),
            "--relative",
            f"{body_dx:.9g}",
            f"{body_dy:.9g}",
            f"{math.degrees(relative_yaw):.9g}",
            "--wait-timeout",
            f"{self.timeout_s:.9g}",
            "--execute",
        ]
        return command, {
            "control_interface": "/move/MoveRelativeCmd",
            "target_world_xy_m": [target_x, target_y],
            "relative_body_xy_m": [body_dx, body_dy],
            "relative_yaw_deg": math.degrees(relative_yaw),
        }

    def verify(self, location: str) -> ToolResult:
        """Read back the world pose and compare it with one configured station."""
        try:
            station = self.resolve(location)
            pose = self._fresh_pose()
            target = self.locations[station]["pose"]
            expected = target["position"]
            position = pose["position"]
            xy_error = math.hypot(
                float(position["x"]) - float(expected["x"]),
                float(position["y"]) - float(expected["y"]),
            )
            z_error = abs(float(position["z"]) - float(expected["z"]))
            target_yaw = _yaw_from_quaternion(target["orientation"])
            yaw_error = abs(
                math.atan2(
                    math.sin(float(pose["yaw_rad"]) - target_yaw),
                    math.cos(float(pose["yaw_rad"]) - target_yaw),
                )
            )
            arrival_ok = xy_error <= 0.05 and z_error <= 0.05 and yaw_error <= 0.10
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return ToolResult(
                data={
                    "error": f"cannot verify location: {exc}",
                    "error_code": "location_verification_unavailable",
                    "location": location,
                    "arrival_ok": False,
                }
            )
        data = {
            "location": station,
            "arrival_ok": arrival_ok,
            "xy_error_m": xy_error,
            "z_error_m": z_error,
            "yaw_error_rad": yaw_error,
            "pose_age_s": float(pose["received_age_s"]),
        }
        if not arrival_ok:
            data.update(
                error="robot is not within the configured destination tolerance",
                error_code="destination_pose_unverified",
            )
        else:
            data.update(success=True, message=f"arrived at {station}")
        result = ToolResult(data=data)
        try:
            jpeg = self.client.camera_jpeg()
            from PIL import Image

            dashboard_kind = f"luna_navigate_{station}_verification"
            image_path = self.output_dir / f"{dashboard_kind}_source.png"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(BytesIO(jpeg)) as image:
                image.save(image_path, format="PNG")
            result.images.append(image_path.read_bytes())
            data.update(
                visual_image_path=str(image_path),
                dashboard_image_kind=dashboard_kind,
            )
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            data["visual_image_error"] = str(exc)
        return result

    def _fresh_pose(self) -> dict[str, Any]:
        deadline = time.monotonic() + POSE_RECOVERY_TIMEOUT_S
        last_error = "world pose unavailable"
        while True:
            self.check_cancelled()
            try:
                pose = self.client.get("/base_world_pose")["pose"]
                age = float(pose["received_age_s"])
                yaw = float(pose["yaw_rad"])
                if (
                    pose.get("valid") is not True
                    or (pose.get("frame_id"), pose.get("child_frame_id"))
                    != ("camera_init", "body")
                    or not math.isfinite(age)
                    or not math.isfinite(yaw)
                    or age < 0
                ):
                    raise ValueError("world pose is invalid")
                if age <= 0.5:
                    return pose
                last_error = f"world pose is stale: {age:.3f}s"
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                last_error = str(exc)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError(last_error)
            time.sleep(min(POSE_RECOVERY_POLL_S, remaining))

    def _run(self, command: list[str], station: str, attempt: int) -> ToolResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / f"luna_navigate_{station}_attempt_{attempt}.log"
        wrapped = self._with_ros(command)
        started = time.monotonic()
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
        with log_path.open("w", encoding="utf-8") as log:
            try:
                self.process = subprocess.Popen(
                    wrapped,
                    cwd=self.script_dir,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except OSError as exc:
                return ToolResult(
                    data={
                        "error": str(exc),
                        "error_code": "navigation_start_failed",
                        "location": station,
                        "attempts": attempt,
                        "log_path": str(log_path),
                    }
                )
            try:
                while self.process.poll() is None:
                    self.check_cancelled()
                    if time.monotonic() - started > self.timeout_s:
                        self.stop()
                        return ToolResult(
                            data={
                                "error": f"navigation timed out after {self.timeout_s:.1f}s",
                                "error_code": "navigation_timeout",
                                "location": station,
                                "attempts": attempt,
                                "log_path": str(log_path),
                            }
                        )
                    time.sleep(0.1)
                returncode = self.process.returncode
            except BaseException:
                self.stop()
                raise
            finally:
                self.process = None
        if returncode != 0:
            return ToolResult(
                data={
                    "error": f"navigation exited with code {returncode}",
                    "error_code": "navigation_exit",
                    "returncode": returncode,
                    "location": station,
                    "attempts": attempt,
                    "log_path": str(log_path),
                }
            )
        return ToolResult(
            data={
                "success": True,
                "returncode": 0,
                "location": station,
                "attempts": attempt,
                "log_path": str(log_path),
            }
        )

    def _with_ros(self, command: list[str]) -> list[str]:
        if self.ros_setup_script is None:
            return command
        return [
            "/usr/bin/bash",
            "-c",
            'source "$1" && shift && exec "$@"',
            "rpent-luna-navigation",
            str(self.ros_setup_script),
            *command,
        ]

    def stop(self) -> None:
        """Stop only the navigation child owned by this runner."""
        process = self.process
        if process is None or process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=7.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)
