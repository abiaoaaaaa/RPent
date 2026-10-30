#!/usr/bin/python3
# 用途：监测本机和远端头部相机 ROS 图像流的帧间隔。
# 运行：python3 head_camera_stall_monitor.py --help
"""检测本机与远端头部相机 ROS 图像流是否出现长时间帧间隔。

本机运行：
  python3 head_camera_stall_monitor.py --remote-user nvidia \
    --remote-host 6.6.7.100 --seconds 90

远端 worker 由本程序通过 SSH 自动启动。程序不保存 SSH 密码，建议使用
SSH key 或 ssh-agent；也可以通过 SSH 自带的交互式认证完成登录。
"""

from __future__ import annotations

import argparse
import base64
import collections
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


LOCAL_TOPICS = (
    ("local_color", "/camera/camera/color/image_raw", Image),
    ("local_depth", "/camera/camera/depth/image_rect_raw", Image),
    ("local_mosaic", "/quad_tile/compressed", CompressedImage),
)
REMOTE_TOPICS = (
    ("remote_color", "/camera/camera/color/image_raw", Image),
    ("remote_depth", "/camera/camera/depth/image_rect_raw", Image),
)


@dataclass
class Stats:
    count: int = 0
    first: float | None = None
    last: float | None = None
    max_gap: float = 0.0
    gaps: list[float] = field(default_factory=list)
    events: list[tuple[float, float]] = field(default_factory=list)
    backward: int = 0

    def add(self, arrival: float, stamp: float, threshold: float) -> None:
        if self.first is None:
            self.first = arrival
        if self.last is not None:
            gap = arrival - self.last
            self.gaps.append(gap)
            self.max_gap = max(self.max_gap, gap)
            if gap >= threshold:
                self.events.append((arrival, gap))
        self.last = arrival
        self.count += 1


class CameraMonitor(Node):
    def __init__(self, topics, threshold: float, prefix: str):
        super().__init__(f"{prefix}_head_camera_stall_monitor")
        self.started = time.monotonic()
        self.threshold = threshold
        self.stats = {name: Stats() for name, _, _ in topics}
        self._image_subscriptions = []
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        for name, topic, message_type in topics:
            self._image_subscriptions.append(
                self.create_subscription(
                    message_type,
                    topic,
                    lambda message, stream=name: self._on_message(stream, message),
                    qos,
                )
            )

    def _on_message(self, stream: str, message) -> None:
        now = time.monotonic()
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        previous_stamp = getattr(self, f"_stamp_{stream}", None)
        if previous_stamp is not None and stamp < previous_stamp:
            self.stats[stream].backward += 1
        setattr(self, f"_stamp_{stream}", stamp)
        before = self.stats[stream].last
        self.stats[stream].add(now, stamp, self.threshold)
        if before is not None and now - before >= self.threshold:
            print(
                f"GAP stream={stream} arrival_gap={now - before:.3f}s",
                flush=True,
            )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _summary(node: CameraMonitor, duration: float) -> None:
    print(f"SUMMARY duration={duration:.3f}s", flush=True)
    for stream, stats in node.stats.items():
        active = (stats.last - stats.first) if stats.first is not None and stats.last else 0.0
        rate = (stats.count - 1) / active if active > 0 and stats.count > 1 else 0.0
        print(
            f"RESULT stream={stream} frames={stats.count} rate={rate:.2f}Hz "
            f"p99_gap={_percentile(stats.gaps, 0.99):.3f}s "
            f"max_gap={stats.max_gap:.3f}s stalls={len(stats.events)} "
            f"backward_stamps={stats.backward}",
            flush=True,
        )


def run_worker(seconds: float, threshold: float, remote: bool) -> int:
    topics = REMOTE_TOPICS if remote else LOCAL_TOPICS
    prefix = "remote" if remote else "local"
    rclpy.init()
    node = CameraMonitor(topics, threshold, prefix)
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        _summary(node, time.monotonic() - node.started)
        try:
            node.destroy_node()
        except ValueError:
            # rclpy 在 spin_once 与 shutdown 交界处可能重复移除 waitable。
            pass
        rclpy.shutdown()
    return 0


def _remote_worker_command(script_path: str, seconds: float, threshold: float) -> list[str]:
    with open(script_path, "rb") as source:
        encoded = base64.b64encode(source.read()).decode("ascii")
    code = (
        "import base64,sys; payload=sys.argv.pop(1); "
        "exec(compile(base64.b64decode(payload), '<remote-monitor>', 'exec'))"
    )
    return [
        "python3",
        "-c",
        code,
        encoded,
        "--worker",
        "--remote",
        "--seconds",
        str(seconds),
        "--stall-threshold",
        str(threshold),
    ]


def run_combined(args: argparse.Namespace) -> int:
    local_cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--seconds",
        str(args.seconds),
        "--stall-threshold",
        str(args.stall_threshold),
    ]
    ssh_target = f"{args.remote_user}@{args.remote_host}"
    remote_cmd = _remote_worker_command(
        os.path.abspath(__file__), args.seconds, args.stall_threshold
    )
    # ssh 将目标主机后的参数交给远端 shell；整体引用可避免 python -c
    # 的分号、括号和 base64 内容被远端 shell 误解析。
    remote_command = (
        "source /opt/ros/humble/setup.bash && "
        + " ".join(shlex.quote(part) for part in remote_cmd)
    )
    ssh_cmd = ["ssh", "-o", "BatchMode=no", ssh_target, remote_command]
    print("监测启动：本机 + 远端；SSH 使用系统认证，不保存密码。", flush=True)
    local = subprocess.Popen(local_cmd, stdout=subprocess.PIPE, text=True)
    remote = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def forward(label: str, pipe) -> None:
        assert pipe is not None
        for line in pipe:
            print(f"[{label}] {line.rstrip()}", flush=True)

    threads = [
        threading.Thread(target=forward, args=("local", local.stdout), daemon=True),
        threading.Thread(target=forward, args=("remote", remote.stdout), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        local_code = local.wait()
        remote_code = remote.wait()
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在停止本机和远端监测……", flush=True)
        for process in (local, remote):
            if process.poll() is None:
                process.terminate()
        try:
            local_code = local.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            local.kill()
            local_code = local.wait()
        try:
            remote_code = remote.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            remote.kill()
            remote_code = remote.wait()
    for thread in threads:
        thread.join(timeout=2.0)
    print(f"EXIT local={local_code} remote={remote_code}", flush=True)
    return 0 if local_code == 0 and remote_code == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-user", default="nvidia")
    parser.add_argument("--remote-host", default="6.6.7.100")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument(
        "--stall-threshold",
        type=float,
        default=0.2,
        help="帧间隔超过此值即记录卡顿，单位秒（默认 0.2）",
    )
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.seconds, args.stall_threshold, args.remote)
    return run_combined(args)


if __name__ == "__main__":
    raise SystemExit(main())
