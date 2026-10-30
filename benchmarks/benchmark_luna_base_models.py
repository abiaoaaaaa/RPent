#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

"""Compare RPent Codex and API-model text/image request latency without robot I/O."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from robots.luna.vision import verify_grasp
from rpent.planner.base import build_api_model
from rpent.planner.codex import build_probe_config, run_probe_turn


class TextReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: Literal["ok"]


TEXT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["ok"]}},
    "required": ["answer"],
    "additionalProperties": False,
}
TEXT_PROMPT = 'Return only JSON with exactly this value: {"answer":"ok"}'
PROXY_NAMES = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


@contextmanager
def without_proxies():
    saved = {name: os.environ.get(name) for name in PROXY_NAMES}
    for name in PROXY_NAMES:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def call_text(backend: str, args: argparse.Namespace) -> str:
    if backend == "codex":
        return run_probe_turn(
            build_probe_config(),
            prompt=TEXT_PROMPT,
            model=args.codex_model,
            timeout_s=args.codex_timeout_s,
            output_schema=TEXT_SCHEMA,
        )
    with without_proxies():
        result = Agent(
            build_api_model(args.qwen_model, args.qwen_base_url),
            output_type=TextReply,
            retries=0,
            model_settings={"timeout": args.qwen_timeout_s, "max_tokens": 64},
        ).run_sync(TEXT_PROMPT, usage_limits=UsageLimits(request_limit=1))
    return result.output.model_dump_json()


def call_image(backend: str, image: bytes, args: argparse.Namespace) -> str:
    kwargs = {
        "target": "blue block",
        "after_stage": "POLICY",
        "wrist_side": "right",
    }
    if backend == "codex":
        verdict = verify_grasp(
            image,
            planner="codex",
            model=args.codex_model,
            **kwargs,
        )
    else:
        with without_proxies():
            verdict = verify_grasp(
                image,
                planner="api",
                model=args.qwen_model,
                base_url=args.qwen_base_url,
                **kwargs,
            )
    return json.dumps(verdict, ensure_ascii=False)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(samples: list[dict]) -> dict:
    measured = [item for item in samples if not item["warmup"]]
    successful = [item["seconds"] for item in measured if item["ok"]]
    wall = sum(item["seconds"] for item in measured)
    return {
        "attempts": len(measured),
        "successes": len(successful),
        "success_rate": len(successful) / len(measured) if measured else 0.0,
        "wall_s": round(wall, 3),
        "successful_requests_per_s": round(len(successful) / wall, 4) if wall else 0.0,
        "latency_s": None
        if not successful
        else {
            "mean": round(statistics.mean(successful), 3),
            "median": round(statistics.median(successful), 3),
            "p95": round(percentile(successful, 0.95), 3),
            "min": round(min(successful), 3),
            "max": round(max(successful), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--codex-model", default="gpt-5.6-sol")
    parser.add_argument("--codex-timeout-s", type=int, default=30)
    parser.add_argument("--qwen-model", default="openai-chat:Qwen3.5-9B")
    parser.add_argument("--qwen-base-url", default="http://10.121.40.31:8000/v1")
    parser.add_argument("--qwen-timeout-s", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups nonnegative")
    image = args.image.expanduser().read_bytes()
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

    cases = [
        (backend, kind, index < args.warmups)
        for index in range(args.warmups + args.runs)
        for kind in ("text", "image")
        for backend in ("codex", "qwen")
    ]
    random.Random(args.seed).shuffle(cases)
    samples = []
    for backend, kind, warmup in cases:
        started = time.perf_counter()
        try:
            answer = (
                call_text(backend, args)
                if kind == "text"
                else call_image(backend, image, args)
            )
            sample = {"ok": True, "answer": answer[:500]}
        except Exception as exc:
            sample = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sample.update(
            backend=backend,
            kind=kind,
            warmup=warmup,
            seconds=round(time.perf_counter() - started, 6),
        )
        samples.append(sample)
        print(json.dumps(sample, ensure_ascii=False), flush=True)

    summary = {
        f"{backend}_{kind}": summarize(
            [item for item in samples if item["backend"] == backend and item["kind"] == kind]
        )
        for backend in ("codex", "qwen")
        for kind in ("text", "image")
    }
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "image": str(args.image.expanduser().resolve()),
        "runs": args.runs,
        "warmups": args.warmups,
        "seed": args.seed,
        "models": {"codex": args.codex_model, "qwen": args.qwen_model},
        "summary": summary,
        "samples": samples,
    }
    output = args.output or Path("logs") / f"base_model_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
