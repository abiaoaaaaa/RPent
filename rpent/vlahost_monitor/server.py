# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""A small, read-only VLAHost monitor.

The monitor deliberately reads only ``/hil/snapshot``.  It never calls an
action, mux, ROS service, or control endpoint.  The browser keeps one selected
JPEG while this process retains the latest 30 snapshots for inspection.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

MAX_HISTORY = 30
MAX_JPEG_BYTES = 10_000_000
STATIC_DIR = Path(__file__).with_name("static")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _read_jpeg(snapshot: dict[str, Any]) -> bytes:
    encoded = snapshot.get("jpeg_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("VLAHost did not provide jpeg_base64")
    try:
        image = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("VLAHost returned invalid base64 JPEG") from exc
    if not image.startswith(b"\xff\xd8") or len(image) > MAX_JPEG_BYTES:
        raise ValueError("VLAHost JPEG is missing or exceeds the 10 MB limit")
    return image


def _joint_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    joints = snapshot.get("joint_states") or {}
    positions = joints.get("positions") or []
    velocities = joints.get("velocities") or []
    body_positions = joints.get("body_positions") or []
    body_velocities = joints.get("body_velocities") or []
    names = [f"left_arm_joint_{i}" for i in range(1, 8)]
    names += [f"right_arm_joint_{i}" for i in range(1, 8)]
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "name": name,
                "position": positions[index] if index < len(positions) else None,
                "velocity": velocities[index] if index < len(velocities) else None,
            }
        )
    for index, name in enumerate(("body_joint_1", "body_joint_2", "body_joint_3", "body_joint_4", "body_joint_5", "body_joint_6")):
        rows.append(
            {
                "name": name,
                "position": body_positions[index] if index < len(body_positions) else None,
                "velocity": body_velocities[index] if index < len(body_velocities) else None,
            }
        )
    for side in ("left", "right"):
        gripper = snapshot.get(f"gripper_{side}") or {}
        rows.append(
            {
                "name": f"gripper_{side}",
                "position": gripper.get("position"),
                "velocity": gripper.get("velocity"),
            }
        )
    return rows


def snapshot_to_sample(snapshot: dict[str, Any], jpeg: bytes, sequence: int, captured_at: float) -> dict[str, Any]:
    """Convert one VLAHost response into a bounded browser-facing sample."""
    freshness = snapshot.get("freshness") or {}
    return {
        "sequence": sequence,
        "captured_at": captured_at,
        "timestamp": snapshot.get("timestamp"),
        "freshness": freshness,
        "connected": bool(snapshot.get("connected")),
        "control_mode": snapshot.get("control_mode"),
        "intervention": snapshot.get("intervention"),
        "joint_rows": _joint_rows(snapshot),
        "eef_left": snapshot.get("eef_left"),
        "eef_right": snapshot.get("eef_right"),
        "base_world_pose": snapshot.get("base_world_pose"),
        "gripper_left": snapshot.get("gripper_left"),
        "gripper_right": snapshot.get("gripper_right"),
        "jpeg_size": len(jpeg),
        "_jpeg": jpeg,
    }


class MonitorStore:
    """Thread-safe 30-frame ring buffer backed by one VLAHost URL."""

    def __init__(self, base_url: str, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._opener = build_opener(ProxyHandler({}))
        self._samples: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self._sequence = 0
        self._lock = threading.Lock()
        self.last_error: str | None = None

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(self.base_url + path, headers={"Accept": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                value = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"VLAHost HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"VLAHost {path} returned a non-object")
        return value

    def poll(self) -> dict[str, Any]:
        captured_at = time.time()
        snapshot = self._get_json("/hil/snapshot")
        jpeg = _read_jpeg(snapshot)
        with self._lock:
            self._sequence += 1
            sample = snapshot_to_sample(snapshot, jpeg, self._sequence, captured_at)
            self._samples.append(sample)
            self.last_error = None
            return dict(sample)

    @staticmethod
    def public_sample(sample: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in sample.items() if key != "_jpeg"}

    def status(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            error = self.last_error
        return {
            "ok": bool(samples) and error is None,
            "base_url": self.base_url,
            "history_size": len(samples),
            "history_limit": MAX_HISTORY,
            "latest": self.public_sample(samples[-1]) if samples else None,
            "history": [self.public_sample(sample) for sample in samples],
            "error": error,
        }

    def frame(self, sequence: str) -> bytes | None:
        with self._lock:
            if sequence == "latest":
                sample = self._samples[-1] if self._samples else None
            else:
                try:
                    requested = int(sequence)
                except ValueError:
                    return None
                sample = next((item for item in self._samples if item["sequence"] == requested), None)
            return None if sample is None else sample["_jpeg"]


class _Handler(BaseHTTPRequestHandler):
    store: MonitorStore

    def log_message(self, message: str, *args: object) -> None:
        print("[vlahost-monitor] " + message % args, flush=True)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            name = path.removeprefix("/static/")
            if name not in {"monitor.css", "monitor.js"} or "/" in name:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            media = "text/css; charset=utf-8" if name.endswith(".css") else "text/javascript; charset=utf-8"
            self._send(200, (STATIC_DIR / name).read_bytes(), media)
            return
        if path == "/api/status":
            try:
                self.store.poll()
            except Exception as exc:  # read-only monitor should keep serving old samples
                self.store.last_error = str(exc)
            self._send(200, _json_bytes(self.store.status()), "application/json; charset=utf-8")
            return
        if path.startswith("/api/frame/") and path.endswith(".jpg"):
            sequence = path.removeprefix("/api/frame/")[:-4]
            frame = self.store.frame(sequence)
            self._send(200, frame, "image/jpeg") if frame else self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only 30-frame Luna VLAHost monitor")
    parser.add_argument("--vlahost-url", default="http://6.6.7.100:8011")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free port")
    parser.add_argument("--timeout-s", type=float, default=2.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    store = MonitorStore(args.vlahost_url, args.timeout_s)
    handler = type("VLAHostMonitorHandler", (_Handler,), {"store": store})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    host, port = server.server_address
    advertised_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"VLAHost monitor: http://{advertised_host}:{port}", flush=True)
    print(f"source (read-only): {store.base_url}/hil/snapshot", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
