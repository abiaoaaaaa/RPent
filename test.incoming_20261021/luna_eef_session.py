#!/usr/bin/python3
# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

# 用途：RPent 拥有的单侧 EEF/双侧夹爪持续控制子进程；不控制底盘。
# 推荐：从仓库根目录运行 `rpent --robot luna --task-id 3 --planner codex`
# （默认只观察；实机运动须额外显式添加 --luna-eef-execute）。

"""Owned selected-EEF and selected-gripper session for tabletop trials.

The process keeps publishing the arm target and gripper hold between JSONL
commands. It deliberately has no base-control publisher.
"""

import argparse
import json
import math
import sys
import threading
import time

import rclpy
from interactive_left_tool_pose import (
    InteractivePose,
    angular_error_deg,
    topic_name,
    wait_for_initial_tf,
)
from marvin_msgs.srv import Int
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray

PREFIX = "RPENT_EEF="
def bounded_delta(values, *, maximum_mm=200.0):
    if (
        not isinstance(values, list)
        or len(values) != 3
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values
        )
    ):
        raise ValueError("delta_mm must contain three finite numbers")
    if math.dist(values, (0.0, 0.0, 0.0)) > maximum_mm + 1e-9:
        raise ValueError(f"EEF step exceeds {maximum_mm:g} mm")
    if not any(values):
        raise ValueError("EEF step must be nonzero")
    return values


def bounded_rotation(values, *, maximum_deg=10.0):
    if (
        not isinstance(values, list)
        or len(values) != 3
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values
        )
        or math.dist(values, (0.0, 0.0, 0.0)) > maximum_deg + 1e-9
        or not any(values)
    ):
        raise ValueError(
            f"rotation must be a finite nonzero XYZ vector of at most {maximum_deg:g} degrees"
        )
    return values


def wait_for_mux_ready(node, timeout_s=3.0, *, allow_source_transition=False):
    """Allow ROS transient-local mux state to arrive after the spinner starts."""
    deadline = time.monotonic() + timeout_s
    while True:
        problem = node.check_mux()
        if not problem:
            return
        initial_state_pending = (
            problem.startswith("尚未收到 IK mux 当前源")
            or problem.startswith("尚未收到 joint command mux 当前源")
            or problem.startswith("尚未收到新鲜的机器人可运动状态")
            or problem.endswith("没有订阅者")
            or (
                allow_source_transition
                and (
                    problem.startswith("IK mux 当前源为")
                    or problem.startswith("joint command mux 当前源为")
                )
            )
        )
        if not initial_state_pending or time.monotonic() >= deadline:
            raise ValueError(problem)
        time.sleep(0.05)


def call_int_service(node, service_name, value, timeout_s=3.0):
    """Call one Luna mux service and require its positive acknowledgement."""
    client = node.create_client(Int, service_name)
    if not client.wait_for_service(timeout_sec=timeout_s):
        raise ValueError(f"mux service unavailable: {service_name}")
    request = Int.Request()
    request.data = int(value)
    future = client.call_async(request)
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not future.done():
        raise ValueError(f"mux service timed out: {service_name}")
    response = future.result()
    if response is None or not response.success:
        message = "" if response is None else response.message
        raise ValueError(f"mux service rejected {value}: {service_name}: {message}")


def claim_eef_control(node, timeout_s=5.0):
    """Switch policy control back to QP/EEF and confirm both active sources."""
    namespace = node.args.namespace
    call_int_service(
        node, topic_name(namespace, "control/ik_request_mux/select"), 2, timeout_s
    )
    call_int_service(node, topic_name(namespace, "control/set_input"), 1, timeout_s)
    wait_for_mux_ready(node, timeout_s=timeout_s, allow_source_transition=True)


class GripperHold:
    def __init__(self, node, side="left"):
        if side not in ("left", "right"):
            raise ValueError(f"unsupported gripper side: {side}")
        self.node = node
        self.side = side
        suffix = "L" if side == "left" else "R"
        self.publisher = node.create_publisher(
            Float32, f"/tj/control/gripperValue{suffix}", 10
        )
        self.feedback_at = None
        self.status_at = None
        self.status = None
        self.value = None
        node.create_subscription(
            Float32MultiArray,
            f"/tj/info/gripper_feedback_{suffix}",
            self.on_feedback,
            20,
        )
        node.create_subscription(
            Int32MultiArray,
            f"/tj/info/gripper_feedback_{suffix}_err",
            self.on_status,
            20,
        )
        node.create_timer(0.1, self.tick)

    def on_feedback(self, message):
        if len(message.data) >= 3 and all(math.isfinite(v) for v in message.data[:3]):
            self.feedback_at = time.monotonic()

    def on_status(self, message):
        if message.data:
            self.status = tuple(message.data)
            self.status_at = time.monotonic()

    def check(self):
        now = time.monotonic()
        if self.publisher.get_subscription_count() == 0:
            raise ValueError(f"{self.side} gripper command has no subscriber")
        if (
            self.feedback_at is None
            or now - self.feedback_at > 0.5
            or self.status_at is None
            or now - self.status_at > 0.5
            or any(code != 1 for code in self.status)
        ):
            raise ValueError(
                f"{self.side} gripper feedback/status is stale or not ready"
            )

    def tick(self):
        if self.value is not None:
            self.publisher.publish(Float32(data=float(self.value)))

    def command(self, action):
        if action not in ("open", "close"):
            raise ValueError("gripper action must be open or close")
        self.check()
        start = 0.0 if action == "open" else (self.value or 0.0)
        duration = 2.0 if action == "open" else 8.0
        begin = time.monotonic()
        while time.monotonic() - begin < duration:
            self.check()
            progress = min(1.0, (time.monotonic() - begin) / duration)
            self.value = (
                start * (1.0 - progress) + progress
                if action == "close"
                else start * (1.0 - progress)
            )
            time.sleep(0.1)
        self.value = 1.0 if action == "close" else 0.0
        self.check()


def serve(node, grippers, *, table_z_m, clearance_m, execute):
    floor = table_z_m + clearance_m
    faulted = False
    print(
        PREFIX
        + json.dumps(
            {
                "ready": True,
                "eef_side": node.side,
                "table_z_m": table_z_m,
                "min_tool_z_m": floor,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            action = request.get("action")
            if faulted and action not in ("state", "stop"):
                raise ValueError(
                    "EEF session faulted; stop this run before another motion"
                )
            if action == "state":
                position, quaternion = node.get_actual()
                data = {
                    "xyz_m": position,
                    "quaternion_xyzw": quaternion,
                    "eef_side": node.side,
                    "publishing": node.publishing,
                }
            elif action == "claim_eef_control":
                if not execute:
                    raise ValueError("EEF motion disabled; start with --execute")
                claim_eef_control(node)
                data = {
                    "control_source": "eef",
                    "ik_source": node.ik_source,
                    "joint_source": node.joint_source,
                }
            elif action == "move":
                if not execute:
                    raise ValueError("EEF motion disabled; start with --execute")
                delta = bounded_delta(request.get("delta_mm"))
                wait_for_mux_ready(node)
                node.get_actual()  # Requires a fresh actual TF before commanding.
                candidate_z = node.target_position[2] + delta[2] / 1000.0
                if candidate_z < floor:
                    raise ValueError(
                        f"{node.side_config['tool']} target Z {candidate_z:.3f} m is below floor {floor:.3f} m"
                    )
                node.apply_position_delta(delta)
                if not node.publishing:
                    node.start_publishing()
                deadline = time.monotonic() + 20.0
                settled = None
                while time.monotonic() < deadline:
                    if not node.publishing or node.check_mux():
                        raise ValueError("EEF control stopped or mux changed")
                    actual, _ = node.get_actual()
                    if math.dist(actual, node.target_position) <= 0.007:
                        settled = settled or time.monotonic()
                        if time.monotonic() - settled >= 0.5:
                            break
                    else:
                        settled = None
                    time.sleep(0.05)
                else:
                    raise ValueError("EEF did not reach target within 20 s")
                data = {
                    "target_xyz_m": node.target_position,
                    "actual_xyz_m": actual,
                    "error_m": math.dist(actual, node.target_position),
                }
            elif action == "rotate":
                if not execute:
                    raise ValueError("EEF motion disabled; start with --execute")
                delta = bounded_rotation(request.get("delta_deg"))
                wait_for_mux_ready(node)
                node.get_actual()
                node.apply_rotation_delta(delta)
                if not node.publishing:
                    node.start_publishing()
                deadline = time.monotonic() + 20.0
                settled = None
                while time.monotonic() < deadline:
                    if not node.publishing or node.check_mux():
                        raise ValueError("EEF control stopped or mux changed")
                    actual_xyz, actual_quat = node.get_actual()
                    error_deg = angular_error_deg(actual_quat, node.target_quaternion)
                    if (
                        error_deg <= 2.5
                        and math.dist(actual_xyz, node.target_position) <= 0.007
                    ):
                        settled = settled or time.monotonic()
                        if time.monotonic() - settled >= 0.5:
                            break
                    else:
                        settled = None
                    time.sleep(0.05)
                else:
                    raise ValueError("EEF rotation did not reach target within 20 s")
                data = {
                    "target_quaternion_xyzw": node.target_quaternion,
                    "actual_quaternion_xyzw": actual_quat,
                    "error_deg": error_deg,
                }
            elif action in ("open", "close", "hold_closed"):
                if not execute:
                    raise ValueError("gripper motion disabled; start with --execute")
                wait_for_mux_ready(node)
                if not node.publishing:
                    node.start_publishing()
                side = request.get("side", "left")
                if side not in grippers:
                    raise ValueError(f"unsupported gripper side: {side}")
                gripper = grippers[side]
                if action == "hold_closed":
                    gripper.check()
                    gripper.value = 1.0
                else:
                    gripper.command(action)
                data = {
                    "gripper_command": action,
                    "gripper_side": side,
                    "status": "commanded",
                }
            elif action == "stop":
                node.stop_publishing()
                print(
                    PREFIX
                    + json.dumps(
                        {"id": request.get("id"), "ok": True, "stopped": True}
                    ),
                    flush=True,
                )
                return
            else:
                raise ValueError("unknown EEF session action")
            response = {"id": request.get("id"), "ok": True, **data}
        except Exception as error:
            node.stop_publishing()
            for gripper in grippers.values():
                gripper.value = None
            faulted = True
            response = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "ok": False,
                "error": str(error),
            }
        print(PREFIX + json.dumps(response), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--table-z-m", type=float, default=0.8)
    parser.add_argument("--tool-clearance-m", type=float, default=0.06)
    args = parser.parse_args()
    if (
        not math.isfinite(args.table_z_m)
        or not math.isfinite(args.tool_clearance_m)
        or args.tool_clearance_m < 0.05
    ):
        parser.error("table Z must be finite and tool clearance at least 0.05 m")
    # Reuse the validated Luna TF, mux and continuous-target controller.
    from interactive_left_tool_pose import parse_args

    original = sys.argv
    sys.argv = [
        original[0],
        "--side",
        args.side,
        "--max-position-step-mm",
        "200",
        "--max-position-offset-mm",
        "200",
        "--tf-future-skew-s",
        "1.0",
    ] + (["--execute"] if args.execute else [])
    try:
        pose_args = parse_args()
    finally:
        sys.argv = original
    rclpy.init(args=[])
    node = None
    spinner = None
    try:
        node = InteractivePose(pose_args)
        position, quaternion = wait_for_initial_tf(node, pose_args.tf_timeout)
        node.set_initial(position, quaternion)
        grippers = {
            "left": GripperHold(node, "left"),
            "right": GripperHold(node, "right"),
        }
        spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spinner.start()
        serve(
            node,
            grippers,
            table_z_m=args.table_z_m,
            clearance_m=args.tool_clearance_m,
            execute=args.execute,
        )
        return 0
    finally:
        if node is not None:
            node.stop_publishing()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spinner is not None:
            spinner.join(timeout=1.0)


if __name__ == "__main__":
    sys.exit(main())
