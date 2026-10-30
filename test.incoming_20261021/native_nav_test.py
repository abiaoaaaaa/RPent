# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

#!/usr/bin/env python3
"""Apex native absolute and relative base motion test.

Dry-run is the default.  ``--pose`` creates a global navigation task through
``/server/CreateTaskByPose``.  ``--relative`` uses ``/move/MoveRelativeCmd``.
Neither path uses VLAHost ``/base_move_world``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Pose
from move.action import MoveRelative
from rclpy.action import ActionClient
from rclpy.node import Node
from server.msg import State, TargetInfoWithPose
from server.srv import CancelTask, CreateTaskByPose


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class NativeNavTest(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("rpent_native_nav_test")
        self.args = args
        self.create_client_ = self.create_client(
            CreateTaskByPose, "/server/CreateTaskByPose"
        )
        self.cancel_client = self.create_client(CancelTask, "/server/CancelTask")
        self.relative_client = ActionClient(self, MoveRelative, "/move/MoveRelativeCmd")
        self.last_state: State | None = None
        self.create_subscription(State, "/server/State", self._state_cb, 10)

    def _state_cb(self, msg: State) -> None:
        self.last_state = msg

    def target(self) -> TargetInfoWithPose:
        if self.args.pose is None:
            raise ValueError("--pose X Y Z YAW is required for --execute")
        x, y, z, yaw = self.args.pose
        qx, qy, qz, qw = yaw_quaternion(yaw)
        target = TargetInfoWithPose()
        target.target_pose = Pose()
        target.target_pose.position.x = x
        target.target_pose.position.y = y
        target.target_pose.position.z = z
        target.target_pose.orientation.x = qx
        target.target_pose.orientation.y = qy
        target.target_pose.orientation.z = qz
        target.target_pose.orientation.w = qw
        # 0 is the normal navigation action in the current interface.
        target.action_type = self.args.action_type
        target.action_param = self.args.action_param
        return target

    def execute(self) -> int:
        if not self.create_client_.wait_for_service(timeout_sec=self.args.service_timeout):
            print("ERROR: /server/CreateTaskByPose service is unavailable", file=sys.stderr)
            return 2

        request = CreateTaskByPose.Request()
        request.max_speed = self.args.max_speed
        request.max_loop_times = self.args.max_loop_times
        request.target_list = [self.target()]
        future = self.create_client_.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.service_timeout)
        if not future.done() or future.result() is None:
            print("ERROR: CreateTaskByPose timed out", file=sys.stderr)
            return 3
        response = future.result()
        print(
            f"create_task task_id={response.task_id} "
            f"error_code={response.error_code} message={response.error_msg}"
        )
        if response.error_code != 0:
            return 4
        if not self.args.wait:
            return 0

        target_x, target_y, _, _ = self.args.pose
        deadline = time.monotonic() + self.args.wait_timeout
        matching_task_seen = False
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            state = self.last_state
            if state is None:
                continue
            if state.task_id == response.task_id:
                matching_task_seen = True
                print(
                    f"state task_id={state.task_id} busy={state.is_busy} "
                    f"arrived={state.has_arrived_target} normal={state.is_normal} "
                    f"scram={state.is_scram}"
                )
                if state.has_arrived_target:
                    return 0
                if state.is_scram or not state.is_normal:
                    self.cancel(response.task_id)
                    return 5
            elif matching_task_seen and not state.has_task:
                break

        state = self.last_state
        if state is not None:
            error = math.hypot(
                state.global_pose.position.x - target_x,
                state.global_pose.position.y - target_y,
            )
            print(f"final_position_error_m={error:.4f}")
            if error <= self.args.arrival_tolerance:
                return 0
        print("ERROR: navigation task ended or timed out", file=sys.stderr)
        self.cancel(response.task_id)
        return 6

    def cancel(self, task_id: int) -> int:
        if not self.cancel_client.wait_for_service(timeout_sec=self.args.service_timeout):
            print("ERROR: /server/CancelTask service is unavailable", file=sys.stderr)
            return 2
        request = CancelTask.Request()
        request.task_id = task_id
        future = self.cancel_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.service_timeout)
        if not future.done() or future.result() is None:
            print("ERROR: CancelTask timed out", file=sys.stderr)
            return 3
        response = future.result()
        print(f"cancel task_id={task_id} error_code={response.error_code} message={response.error_msg}")
        return 0 if response.error_code == 0 else 4

    def relative_move(self, dx: float, dy: float, yaw_deg: float) -> int:
        if not self.relative_client.wait_for_server(timeout_sec=self.args.service_timeout):
            print("ERROR: /move/MoveRelativeCmd action is unavailable", file=sys.stderr)
            return 2
        goal = MoveRelative.Goal()
        goal.relative_pose.position.x = dx
        goal.relative_pose.position.y = dy
        # Lingho's MoveRelative interface uses orientation.w directly as the
        # relative angle in radians, despite the field being a Quaternion.
        goal.relative_pose.orientation.w = math.radians(yaw_deg)
        send_future = self.relative_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=self.args.service_timeout
        )
        if not send_future.done() or send_future.result() is None:
            print("ERROR: MoveRelative goal send timed out", file=sys.stderr)
            return 3
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            print("ERROR: MoveRelative goal rejected", file=sys.stderr)
            return 4
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=self.args.wait_timeout
        )
        if not result_future.done() or result_future.result() is None:
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(
                self, cancel_future, timeout_sec=self.args.service_timeout
            )
            print("ERROR: MoveRelative result timed out", file=sys.stderr)
            return 5
        result = result_future.result().result
        print(
            f"relative_move error_code={result.error_code} "
            f"message={result.error_msg}"
        )
        return 0 if result.error_code == 0 else 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW"),
        required=False, default=None,
        help="target pose in the native navigation frame: x y z yaw(rad)",
    )
    parser.add_argument(
        "--relative", nargs=3, type=float, metavar=("DX", "DY", "YAW_DEG"),
        help="relative base movement in metres/degrees; uses /move/MoveRelativeCmd",
    )
    parser.add_argument("--max-speed", type=float, default=0.2)
    parser.add_argument("--max-loop-times", type=int, default=10)
    parser.add_argument("--action-type", type=int, default=0)
    parser.add_argument("--action-param", type=float, default=0.0)
    parser.add_argument("--service-timeout", type=float, default=5.0)
    parser.add_argument("--wait", action="store_true", help="also monitor /server/State")
    parser.add_argument("--wait-timeout", type=float, default=120.0)
    parser.add_argument("--arrival-tolerance", type=float, default=0.2)
    parser.add_argument("--cancel-task-id", type=int)
    parser.add_argument(
        "--execute", action="store_true",
        help="actually create a native navigation task; without this flag only print the request",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = NativeNavTest(args)
    try:
        if args.cancel_task_id is not None:
            return node.cancel(args.cancel_task_id) if args.execute else print(
                f"DRY-RUN cancel task_id={args.cancel_task_id}"
            ) or 0
        if args.relative is not None:
            dx, dy, yaw_deg = args.relative
            if not args.execute:
                print(
                    f"DRY-RUN MoveRelativeCmd dx={dx} m dy={dy} m "
                    f"yaw={yaw_deg} deg"
                )
                return 0
            return node.relative_move(dx, dy, yaw_deg)
        if not args.execute:
            print(
                "DRY-RUN CreateTaskByPose "
                f"pose={list(args.pose) if args.pose is not None else '<not set>'} "
                f"max_speed={args.max_speed} "
                f"max_loop_times={args.max_loop_times} action_type={args.action_type}"
            )
            print("Pass --execute to send the real ROS navigation request.")
            return 0
        return node.execute()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
