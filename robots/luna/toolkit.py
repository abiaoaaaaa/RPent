# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Luna Toolkit integration and run-state recording."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.luna.tools import LunaPrimitives, summarize_luna_snapshot
from robots.luna.vision import three_view_arrays
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class LunaToolkit(Toolkit):
    """Common RPent tools plus the validated Luna transport workflow."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str | None = None,
    ) -> None:
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        runtime_kwargs = dict(runtime_kwargs)
        mode = runtime_kwargs.pop("mode", None)
        self._eef_mode = mode == "eef"
        self._act_mode = mode == "act"
        self._open_mode = mode == "open"
        if self._eef_mode:
            from robots.luna.eef import LunaEefPrimitives

            self._primitives = LunaEefPrimitives(**runtime_kwargs)
            self._primitives.session.check_cancelled = self.raise_if_cancelled
        elif self._act_mode:
            from robots.luna.act import LunaActPrimitives

            self._primitives = LunaActPrimitives(**runtime_kwargs)
            self._primitives.runner.check_cancelled = self.raise_if_cancelled
            self._primitives.session.check_cancelled = self.raise_if_cancelled
            self._primitives.navigation.check_cancelled = self.raise_if_cancelled
        elif self._open_mode:
            from robots.luna.act import LunaOpenPrimitives

            self._primitives = LunaOpenPrimitives(**runtime_kwargs)
            self._primitives.session.check_cancelled = self.raise_if_cancelled
            self._primitives.navigation.check_cancelled = self.raise_if_cancelled
        else:
            self._primitives = LunaPrimitives(**runtime_kwargs)
            self._primitives.runner.check_cancelled = self.raise_if_cancelled
        self.add_tools(iter_tools(self._primitives))

    def solved(self) -> bool:
        """Return whether transport and destination checks completed."""
        return (
            self._primitives.completed
            if self._act_mode
            else self._primitives.completed
            if self._eef_mode
            else self._primitives.completed
            if self._open_mode
            else self._primitives.runner.completed
        )

    def close(self) -> None:
        """Drain tools and stop the owned EEF publisher before leaving the run."""
        super().close()
        if self._eef_mode:
            self._primitives.session.close()
        elif self._act_mode or self._open_mode:
            self._primitives.navigation.stop()
            if self._act_mode:
                self._primitives.runner.stop()
            self._primitives.session.close()

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        """Record the post-action VLAHost snapshot and child-process result."""
        # Open EEF tools have their own selected-tool TF/mux gate.  Do not
        # replace a successful arm-side result with an unrelated stale base
        # world-pose snapshot; navigate_to performs that base verification.
        if self._eef_mode or self._act_mode or self._open_mode:
            result = dict(result)
            try:
                state = {"state": self._primitives.client.get("/state")}
            except (OSError, RuntimeError, ValueError) as exc:
                # The action result is authoritative for whether the tool
                # completed. A post-action telemetry outage is recorded but
                # must not rewrite a completed navigation/grasp into failure.
                result["state_capture_error"] = str(exc)
                state = {"state_unavailable": str(exc)}
            with self._state.record_step(
                state=state,
                terminated=self.solved(),
                command=command,
                result=result,
                elapsed_s=elapsed_s,
            ) as step:
                self._save_dashboard_camera_views(step)
                if image_path := result.get("visual_image_path"):
                    import numpy as np
                    from PIL import Image

                    kind = Path(
                        str(result.get("dashboard_image_kind") or "luna_visual_comparison")
                    ).name
                    with Image.open(image_path) as image:
                        self._state.save(
                            f"{kind}.png",
                            np.asarray(image.convert("RGB")),
                            step=step,
                        )
            return ToolResult(data=result)
        snapshot = self._primitives.client.snapshot()
        with self._state.record_step(
            state=snapshot,
            terminated=self.solved(),
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ) as step:
            if image_path := result.get("visual_image_path"):
                import numpy as np
                from PIL import Image

                with Image.open(image_path) as image:
                    self._state.save(
                        "luna_visual_comparison.png",
                        np.asarray(image.convert("RGB")),
                        step=step,
                    )
        return ToolResult(
            data={
                "success": result.get("success", False),
                "workflow": result.get("workflow"),
                "log_path": result.get("log_path"),
                **{
                    key: result[key]
                    for key in (
                        "script_success",
                        "workflow_success",
                        "phase",
                        "error_code",
                        "message",
                        "arrival_ok",
                        "left_gripper_ok",
                        "payload_confirmed",
                        "verdict",
                        "visual_evidence",
                        "visual_image_path",
                        "xy_error_m",
                        "z_error_m",
                        "yaw_error_rad",
                    )
                    if key in result
                },
                "inspection": summarize_luna_snapshot(snapshot),
            }
        )

    def _save_dashboard_camera_views(self, step: int) -> None:
        """Best-effort read-only camera projection for the live Dashboard."""
        try:
            client = self._primitives.client
            for name, image in three_view_arrays(client.camera_jpeg()).items():
                self._state.save(f"{name}.png", image, step=step)
        except (AttributeError, OSError, RuntimeError, ValueError):
            # A display refresh must not turn an otherwise valid robot tool
            # result into an error.
            return
