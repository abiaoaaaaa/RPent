# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""One-shot selected-wrist grasp verification from the VLAHost camera mosaic."""

from __future__ import annotations

import io
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


class GraspVerdict(BaseModel):
    """Structured result shared by Codex and OpenAI-compatible VLMs."""

    model_config = ConfigDict(extra="forbid")
    verdict: Literal["held", "not_held"]
    evidence: str


def three_view_arrays(mosaic_jpeg: bytes) -> dict[str, object]:
    """Decode the VLAHost mosaic into the head and two wrist dashboard views."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(mosaic_jpeg)) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width % 2 or height * 8 != width * 15:
            raise ValueError(f"unexpected Luna three-view mosaic size: {width}x{height}")
        return {
            "luna_camera_top": np.asarray(image.crop((0, 0, width, height * 2 // 5))),
            "luna_camera_left_wrist": np.asarray(
                image.crop((0, height * 2 // 5, width // 2, height * 3 // 5))
            ),
            "luna_camera_right_wrist": np.asarray(
                image.crop((width // 2, height * 2 // 5, width, height * 3 // 5))
            ),
        }


def comparison_png(
    before_jpeg: bytes,
    after_jpeg: bytes,
    *,
    after_label: str = "AFTER RETRACT",
    wrist_side: str = "left",
) -> bytes:
    """Place one wrist's before/after views side by side for one VLM call."""
    from PIL import Image, ImageDraw

    with Image.open(io.BytesIO(before_jpeg)) as before_image:
        before = before_image.convert("RGB")
    with Image.open(io.BytesIO(after_jpeg)) as after_image:
        after = after_image.convert("RGB")
    if before.size != after.size:
        raise ValueError("VLAHost camera layout changed between grasp images")
    width, height = before.size
    if width % 2 or height % 3 or height * 8 != width * 15:
        raise ValueError(f"unexpected Luna three-view mosaic size: {width}x{height}")

    if wrist_side not in {"left", "right"}:
        raise ValueError(f"unsupported wrist side: {wrist_side}")
    x0 = 0 if wrist_side == "left" else width // 2
    wrist_box = (x0, height * 2 // 5, x0 + width // 2, height * 3 // 5)
    before = before.crop(wrist_box)
    after = after.crop(wrist_box)
    comparison = Image.new("RGB", (width, before.height + 32), "white")
    comparison.paste(before, (0, 32))
    comparison.paste(after, (width // 2, 32))
    draw = ImageDraw.Draw(comparison)
    draw.text((8, 8), "BEFORE GRASP", fill="black")
    draw.text((width // 2 + 8, 8), after_label, fill="black")
    output = io.BytesIO()
    comparison.save(output, format="PNG")
    return output.getvalue()


def verify_grasp(
    comparison: bytes,
    *,
    target: str = "AprilTag-marked optical module",
    after_stage: str = "RETRACT",
    wrist_side: str = "left",
    planner: str = "codex",
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, str]:
    """Ask the run's configured base model for a bounded grasp verdict."""
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["held", "not_held"]},
            "evidence": {"type": "string"},
        },
        "required": ["verdict", "evidence"],
        "additionalProperties": False,
    }
    if wrist_side not in {"left", "right"}:
        raise ValueError(f"unsupported wrist side: {wrist_side}")
    side = wrist_side.upper()
    prompt = (
        f"Using ONLY the {side} wrist camera views, decide whether Luna's "
        f"{wrist_side} gripper caught the {target}. The left image is "
        f"BEFORE GRASP and the right image is AFTER {after_stage}. Return held if the "
        f"AFTER image shows the target object gripped between the {wrist_side} jaws; compare "
        "its position relative to the jaws before and after. Do not require a "
        "visible gap beneath the object or judge whether its far end cleared "
        "the table. Return not_held if the object is not visibly gripped; a "
        "closed gripper or an object merely near the jaws is insufficient. Choose "
        f"one of these two verdicts from the visible {wrist_side}-wrist evidence only. "
        "Do not infer a grasp from the commanded robot action. "
        "Return only the requested JSON verdict and evidence."
    )
    if planner == "codex":
        from rpent.planner.codex import build_probe_config, run_probe_turn

        response = run_probe_turn(
            build_probe_config(),
            prompt=prompt,
            model=model,
            timeout_s=45,
            image_bytes=comparison,
            output_schema=schema,
        )
        result = json.loads(response)
    elif planner == "api":
        if not model:
            raise ValueError("API vision requires the run's configured model")
        from pydantic_ai import Agent, BinaryContent
        from pydantic_ai.usage import UsageLimits

        from rpent.planner.base import build_api_model

        verdict = Agent(
            build_api_model(model, base_url),
            output_type=GraspVerdict,
            retries=0,
            model_settings={"timeout": 45, "max_tokens": 512},
        ).run_sync(
            [prompt, BinaryContent(data=comparison, media_type="image/png")],
            usage_limits=UsageLimits(request_limit=1),
        ).output
        result = verdict.model_dump()
    else:
        raise ValueError(
            f"Luna visual verification requires planner codex or api, got {planner}"
        )
    if not isinstance(result, dict) or result.get("verdict") not in {
        "held",
        "not_held",
    }:
        raise ValueError("VLM returned an invalid grasp verdict")
    evidence = result.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("VLM returned no visual evidence")
    return {"verdict": result["verdict"], "evidence": evidence.strip()[:500]}
