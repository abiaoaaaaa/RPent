#!/usr/bin/python3
"""Run a supervised clothes pose sequence on Luna.

Sequence:
  1. go to the configured direct-joint home pose
  2. open both grippers
  3. move smoothly to the recorded clothes prepare pose
  4. move smoothly to the recorded final grasp pose
  5. close the selected gripper and hold it closed
  6. return to the configured home while keeping the gripper closed

Motion is disabled by default. Pass ``--execute`` only with clear workspace and
an available emergency stop.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import rclpy
import yaml
from marvin_msgs.msg import JointcmdArm, JointcmdBody, JointcmdHead, Jointfeedback
from marvin_msgs.srv import Int as IntSrv
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import (
    Float32,
    Float32MultiArray,
    Int16MultiArray,
    Int32,
    Int32MultiArray,
)
from std_srvs.srv import Trigger


ARM_NAMES = [
    "Arm_L1_Joint",
    "Arm_L2_Joint",
    "Arm_L3_Joint",
    "Arm_L4_Joint",
    "Arm_L5_Joint",
    "Arm_L6_Joint",
    "Arm_L7_Joint",
    "Arm_R1_Joint",
    "Arm_R2_Joint",
    "Arm_R3_Joint",
    "Arm_R4_Joint",
    "Arm_R5_Joint",
    "Arm_R6_Joint",
    "Arm_R7_Joint",
]
BODY_NAMES = [f"Leg_J{i}_Joint" for i in range(1, 7)]
HEAD_NAMES = ["Head_J1_joint", "Head_J2_joint", "Head_J3_joint"]
JOINT_NAMES_22 = ARM_NAMES + BODY_NAMES + ["gripper_L", "gripper_R"]


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def load_pose(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    names = document.get("joint_names")
    if names != JOINT_NAMES_22:
        raise ValueError(f"{path} joint order does not match the Luna 22-D contract")
    _finite_vector(document.get("position_22d"), 22, f"{path}: position_22d")
    arm = _finite_vector(
        document.get("arm_positions_rad"), 14, f"{path}: arm_positions_rad"
    )
    body = _finite_vector(
        document.get("body_positions_rad"), 6, f"{path}: body_positions_rad"
    )
    raw_head = document.get("head_positions_rad")
    if not isinstance(raw_head, list) or len(raw_head) not in (2, 3):
        raise ValueError(f"{path}: head_positions_rad must contain 2 or 3 values")
    head = [float(item) for item in raw_head]
    if len(head) == 2:
        head.append(0.0)
    if not all(math.isfinite(item) for item in head):
        raise ValueError(f"{path}: head_positions_rad contains a non-finite value")
    return {"arm": arm, "body": body, "head": head}


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def interpolate(start: list[float], end: list[float], alpha: float) -> list[float]:
    return [
        left + (right - left) * alpha
        for left, right in zip(start, end, strict=True)
    ]


class ClothesPoseSequence(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("clothes_pose_sequence_test")
        self.args = args
        self.feedback: Jointfeedback | None = None
        self.feedback_at: float | None = None
        self.robot_ready = False
        self.robot_state_at: float | None = None
        self.joint_source: int | None = None
        self.gripper_feedback: dict[str, list[float] | None] = {
            "left": None,
            "right": None,
        }
        self.gripper_feedback_at: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self.gripper_status: dict[str, list[int] | None] = {
            "left": None,
            "right": None,
        }
        self.gripper_status_at: dict[str, float | None] = {
            "left": None,
            "right": None,
        }

        self.arm_a_pub = self.create_publisher(
            JointcmdArm, "/tj/control/user/joint_cmd_A", 10
        )
        self.arm_b_pub = self.create_publisher(
            JointcmdArm, "/tj/control/user/joint_cmd_B", 10
        )
        self.body_pub = self.create_publisher(
            JointcmdBody, "/tj/control/user/joint_cmd_body", 10
        )
        self.head_pub = self.create_publisher(
            JointcmdHead, "/tj/control/user/joint_cmd_head", 10
        )
        self.gripper_pubs = {
            side: self.create_publisher(
                Float32,
                f"/tj/control/gripperValue{'L' if side == 'left' else 'R'}",
                10,
            )
            for side in ("left", "right")
        }
        self.gripper_pub = self.gripper_pubs[args.gripper_side]

        self.create_subscription(
            Jointfeedback,
            "/tj/info/joint_feedback",
            self._on_feedback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Int16MultiArray, "/tj/info/robot_state", self._on_robot_state, 10
        )
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Int32,
            "/tj/info/joint_cmd_mux/active_source",
            self._on_joint_source,
            latched_qos,
        )
        for side, suffix in (("left", "L"), ("right", "R")):
            self.create_subscription(
                Float32MultiArray,
                f"/tj/info/gripper_feedback_{suffix}",
                lambda message, gripper_side=side: self._on_gripper_feedback(
                    gripper_side, message
                ),
                20,
            )
            self.create_subscription(
                Int32MultiArray,
                f"/tj/info/gripper_feedback_{suffix}_err",
                lambda message, gripper_side=side: self._on_gripper_status(
                    gripper_side, message
                ),
                20,
            )

        self.set_input_client = self.create_client(
            IntSrv, "/tj/control/set_input"
        )
        self.set_ready_client = self.create_client(
            Trigger, "/tj/control/set_ready"
        )
        self.set_mode_client = self.create_client(IntSrv, "/tj/control/set_mode")
        self.go_home_client = self.create_client(Trigger, "/tj/control/go_home")

    def _on_feedback(self, message: Jointfeedback) -> None:
        self.feedback = message
        self.feedback_at = time.monotonic()

    def _on_robot_state(self, message: Int16MultiArray) -> None:
        self.robot_ready = (
            len(message.data) >= 4
            and all(message.data[index] in (1, 2) for index in (0, 1, 3))
        )
        self.robot_state_at = time.monotonic()

    def _on_joint_source(self, message: Int32) -> None:
        self.joint_source = int(message.data)

    def _on_gripper_feedback(
        self, side: str, message: Float32MultiArray
    ) -> None:
        self.gripper_feedback[side] = [float(value) for value in message.data]
        self.gripper_feedback_at[side] = time.monotonic()

    def _on_gripper_status(
        self, side: str, message: Int32MultiArray
    ) -> None:
        self.gripper_status[side] = [int(value) for value in message.data]
        self.gripper_status_at[side] = time.monotonic()

    def spin_until(self, predicate: Callable[[], bool], timeout_s: float, label: str) -> None:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return
        raise TimeoutError(f"timed out waiting for {label}")

    def wait_feedback(self) -> Jointfeedback:
        self.spin_until(
            lambda: (
                self.feedback is not None
                and self.feedback_at is not None
                and time.monotonic() - self.feedback_at <= 0.5
            ),
            self.args.feedback_timeout_s,
            "fresh joint feedback",
        )
        assert self.feedback is not None
        return self.feedback

    def wait_ready(self) -> None:
        self.spin_until(
            lambda: (
                self.robot_ready
                and self.robot_state_at is not None
                and time.monotonic() - self.robot_state_at <= 0.5
            ),
            self.args.feedback_timeout_s,
            "movable robot state",
        )

    def _gripper_is_ready(self, side: str, now: float) -> bool:
        feedback = self.gripper_feedback[side]
        status = self.gripper_status[side]
        feedback_at = self.gripper_feedback_at[side]
        status_at = self.gripper_status_at[side]
        return (
            feedback is not None
            and status is not None
            and feedback_at is not None
            and status_at is not None
            and now - feedback_at <= 0.5
            and now - status_at <= 0.5
            and bool(status)
            and all(code == 1 for code in status)
        )

    def wait_gripper_ready(
        self, sides: tuple[str, ...] | None = None
    ) -> None:
        checked_sides = sides or (self.args.gripper_side,)
        self.spin_until(
            lambda: all(
                self._gripper_is_ready(side, time.monotonic())
                for side in checked_sides
            ),
            self.args.feedback_timeout_s,
            f"{'+'.join(checked_sides)} gripper feedback/status",
        )

    def spin_for(self, duration_s: float, *, publish_gripper: bool) -> None:
        deadline = time.monotonic() + duration_s
        period = 1.0 / self.args.rate_hz
        while rclpy.ok() and time.monotonic() < deadline:
            if publish_gripper:
                self.gripper_pub.publish(Float32(data=1.0))
            rclpy.spin_once(self, timeout_sec=period)

    def call_int(self, client: Any, value: int, label: str) -> None:
        if not client.wait_for_service(timeout_sec=self.args.service_timeout_s):
            raise RuntimeError(f"{label} service is unavailable")
        request = IntSrv.Request()
        request.data = int(value)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self.args.service_timeout_s
        )
        if not future.done() or future.result() is None:
            raise RuntimeError(f"{label} service timed out")
        response = future.result()
        if not response.success:
            raise RuntimeError(
                f"{label} failed: error_code={response.error_code} "
                f"message={response.message}"
            )

    def call_trigger(
        self,
        client: Any,
        label: str,
        *,
        retry_not_movable_s: float = 0.0,
    ) -> Trigger.Response:
        if not client.wait_for_service(timeout_sec=self.args.service_timeout_s):
            raise RuntimeError(f"{label} service is unavailable")
        deadline = time.monotonic() + retry_not_movable_s
        while True:
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(
                self, future, timeout_sec=self.args.service_timeout_s
            )
            if not future.done() or future.result() is None:
                raise RuntimeError(f"{label} service timed out")
            response = future.result()
            if response.success:
                return response
            if (
                label == "go_home"
                and response.message == "Robot not movable"
                and time.monotonic() < deadline
            ):
                self.get_logger().warning(
                    "go_home not movable yet; waiting for QP mode-state refresh"
                )
                self.spin_for(0.2, publish_gripper=False)
                continue
            raise RuntimeError(f"{label} rejected: {response.message}")

    def ensure_control_mode(self, mode: int) -> None:
        """Set the requested control mode, accepting an idempotent reply."""
        if mode not in (1, 3):
            raise ValueError(f"unsupported clothes control mode: {mode}")
        mode_name = {1: "position", 3: "joint impedance"}[mode]
        client = self.set_mode_client
        if not client.wait_for_service(timeout_sec=self.args.service_timeout_s):
            raise RuntimeError("set_mode service is unavailable")
        request = IntSrv.Request()
        request.data = mode
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self.args.service_timeout_s
        )
        if not future.done() or future.result() is None:
            raise RuntimeError("set_mode service timed out")
        response = future.result()
        if response.success:
            return
        # The Luna controller returns success=false with error_code=-6 when the
        # requested mode is already active, even though current == commanded.
        match = re.search(
            r"current=(\[[^\]]*\]),\s*commanded=(\[[^\]]*\])",
            response.message,
        )
        if match and match.group(1) == match.group(2):
            self.get_logger().info(
                f"{mode_name} mode already active; continuing without a mode change"
            )
            return
        raise RuntimeError(
            f"set_mode({mode_name}) failed: error_code={response.error_code} "
            f"message={response.message}"
        )

    def switch_joint_source(self, compat_input: int, expected_source: int) -> None:
        self.call_int(
            self.set_input_client,
            compat_input,
            f"set_input({compat_input})",
        )
        self.spin_until(
            lambda: self.joint_source == expected_source,
            self.args.service_timeout_s,
            f"joint mux source {expected_source}",
        )
        self.get_logger().info(
            f"joint command source switched to {expected_source}"
        )

    @staticmethod
    def feedback_arrays(
        feedback: Jointfeedback,
    ) -> tuple[list[float], list[float], list[float]]:
        arm = [float(value) for value in feedback.arm_positions]
        body = [float(value) for value in feedback.body_positions]
        head = [float(value) for value in feedback.head_positions]
        if len(arm) != 14 or len(body) != 6 or len(head) < 2:
            raise ValueError("joint feedback dimensions are invalid")
        head = (head + [0.0, 0.0, 0.0])[:3]
        return arm, body, head

    def publish_joint_target(
        self,
        arm: list[float],
        body: list[float],
        head: list[float],
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        arm_a = JointcmdArm()
        arm_a.header.stamp = stamp
        arm_a.positions = arm[:7]
        self.arm_a_pub.publish(arm_a)

        arm_b = JointcmdArm()
        arm_b.header.stamp = stamp
        arm_b.positions = arm[7:]
        self.arm_b_pub.publish(arm_b)

        body_message = JointcmdBody()
        body_message.header.stamp = stamp
        body_message.positions = body
        self.body_pub.publish(body_message)

        head_message = JointcmdHead()
        head_message.header.stamp = stamp
        head_message.positions = head
        self.head_pub.publish(head_message)

    def move_to_pose(self, pose: dict[str, Any], label: str) -> None:
        self.move_to_pose_with_gripper(
            pose, label, hold_gripper_closed=False
        )

    def move_to_pose_with_gripper(
        self,
        pose: dict[str, Any],
        label: str,
        *,
        hold_gripper_closed: bool,
    ) -> None:
        feedback = self.wait_feedback()
        start_arm, start_body, start_head = self.feedback_arrays(feedback)
        target_arm = pose["arm"]
        target_body = pose["body"]
        target_head = pose["head"]
        max_delta = max(
            abs(right - left)
            for left, right in zip(
                start_arm + start_body + start_head,
                target_arm + target_body + target_head,
                strict=True,
            )
        )
        safe_duration = max_delta / self.args.max_joint_speed_rad_s
        duration = max(self.args.move_duration_s, safe_duration)
        if duration > self.args.move_duration_s:
            self.get_logger().warning(
                f"{label}: extending transition to {duration:.2f}s "
                f"to respect {self.args.max_joint_speed_rad_s:.3f} rad/s"
            )
        self.get_logger().info(
            f"{label}: max joint delta={max_delta:.4f} rad, "
            f"duration={duration:.2f}s"
        )

        period = 1.0 / self.args.rate_hz
        started = time.monotonic()
        while rclpy.ok():
            elapsed = time.monotonic() - started
            alpha = smoothstep(elapsed / duration)
            if self.joint_source != 2:
                raise RuntimeError(
                    f"{label}: joint mux left user source; current={self.joint_source}"
                )
            if not self.robot_ready:
                raise RuntimeError(f"{label}: robot is no longer movable")
            self.publish_joint_target(
                interpolate(start_arm, target_arm, alpha),
                interpolate(start_body, target_body, alpha),
                interpolate(start_head, target_head, alpha),
            )
            if hold_gripper_closed:
                self.gripper_pub.publish(Float32(data=1.0))
            if elapsed >= duration:
                break
            rclpy.spin_once(self, timeout_sec=period)

        hold_deadline = time.monotonic() + self.args.hold_duration_s
        while rclpy.ok() and time.monotonic() < hold_deadline:
            if self.joint_source != 2:
                raise RuntimeError(f"{label}: joint mux changed during hold")
            if not self.robot_ready:
                raise RuntimeError(f"{label}: robot is no longer movable")
            self.publish_joint_target(target_arm, target_body, target_head)
            if hold_gripper_closed:
                self.gripper_pub.publish(Float32(data=1.0))
            rclpy.spin_once(self, timeout_sec=period)

        feedback = self.wait_feedback()
        actual_arm, actual_body, actual_head = self.feedback_arrays(feedback)
        error = max(
            abs(actual - target)
            for actual, target in zip(
                actual_arm + actual_body + actual_head,
                target_arm + target_body + target_head,
                strict=True,
            )
        )
        self.get_logger().info(f"{label}: max feedback error={error:.4f} rad")
        if error > self.args.joint_tolerance_rad:
            raise RuntimeError(
                f"{label}: target error {error:.4f} rad exceeds "
                f"{self.args.joint_tolerance_rad:.4f} rad"
            )

    def close_gripper(self) -> None:
        self.wait_gripper_ready()
        self.get_logger().warning(
            f"closing {self.args.gripper_side} gripper for "
            f"{self.args.gripper_close_duration_s:.1f}s"
        )
        deadline = time.monotonic() + self.args.gripper_close_duration_s
        period = 1.0 / self.args.rate_hz
        while rclpy.ok() and time.monotonic() < deadline:
            if not self._gripper_is_ready(
                self.args.gripper_side, time.monotonic()
            ):
                raise RuntimeError(
                    f"{self.args.gripper_side} gripper feedback/status became invalid"
                )
            self.gripper_pub.publish(Float32(data=1.0))
            rclpy.spin_once(self, timeout_sec=period)
        self.gripper_pub.publish(Float32(data=1.0))

    def open_grippers(self) -> None:
        """Open both grippers before the arm leaves its recorded home pose."""
        self.wait_gripper_ready(("left", "right"))
        self.get_logger().warning(
            f"opening both grippers for {self.args.gripper_open_duration_s:.1f}s"
        )
        deadline = time.monotonic() + self.args.gripper_open_duration_s
        period = 1.0 / self.args.rate_hz
        while rclpy.ok() and time.monotonic() < deadline:
            for side in ("left", "right"):
                if not self._gripper_is_ready(side, time.monotonic()):
                    raise RuntimeError(
                        f"{side} gripper feedback/status became invalid"
                    )
                self.gripper_pubs[side].publish(Float32(data=0.0))
            rclpy.spin_once(self, timeout_sec=period)
        for gripper_pub in self.gripper_pubs.values():
            gripper_pub.publish(Float32(data=0.0))

    def go_home(self, *, keep_gripper_closed: bool) -> None:
        # QP owns joint mux source 0 while homing.
        self.switch_joint_source(compat_input=1, expected_source=0)
        self.spin_for(self.args.mode_settle_s, publish_gripper=False)
        self.wait_ready()
        response = self.call_trigger(
            self.go_home_client,
            "go_home",
            retry_not_movable_s=self.args.gohome_not_movable_retry_s,
        )
        match = re.search(r"duration=([0-9]+(?:\.[0-9]+)?)s", response.message)
        duration = (
            float(match.group(1))
            if match
            else self.args.gohome_fallback_duration_s
        )
        self.get_logger().info(
            f"go_home accepted: {response.message}; waiting {duration:.2f}s"
        )
        deadline = time.monotonic() + duration + self.args.gohome_settle_s
        period = 1.0 / self.args.rate_hz
        while rclpy.ok() and time.monotonic() < deadline:
            if keep_gripper_closed:
                self.gripper_pub.publish(Float32(data=1.0))
            rclpy.spin_once(self, timeout_sec=period)
        self.wait_feedback()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    config = root / "robots/luna/config"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prepare-pose",
        type=Path,
        default=config / "clothes_prepare_pose.yaml",
    )
    parser.add_argument(
        "--grasp-pose",
        type=Path,
        default=config / "clothes_grasp_pose.yaml",
    )
    parser.add_argument(
        "--home-pose",
        type=Path,
        default=config / "clothes_home_pose.yaml",
    )
    parser.add_argument(
        "--gripper-side", choices=("left", "right"), default="right"
    )
    parser.add_argument("--move-duration-s", type=float, default=4.0)
    parser.add_argument("--max-joint-speed-rad-s", type=float, default=0.3)
    parser.add_argument("--hold-duration-s", type=float, default=0.5)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.08)
    parser.add_argument("--gripper-open-duration-s", type=float, default=2.0)
    parser.add_argument("--gripper-close-duration-s", type=float, default=2.0)
    parser.add_argument(
        "--home-mode",
        choices=("service", "direct"),
        default="direct",
        help=(
            "service calls /tj/control/go_home; direct records the current "
            "arm/body/head pose and sends it through the user joint source"
        ),
    )
    parser.add_argument("--mode-settle-s", type=float, default=1.0)
    parser.add_argument("--gohome-not-movable-retry-s", type=float, default=3.0)
    parser.add_argument("--gohome-settle-s", type=float, default=1.0)
    parser.add_argument("--gohome-fallback-duration-s", type=float, default=10.0)
    parser.add_argument("--service-timeout-s", type=float, default=5.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually command the robot. Without this flag only validate and print the plan.",
    )
    args = parser.parse_args()
    positive = (
        "move_duration_s",
        "max_joint_speed_rad_s",
        "hold_duration_s",
        "joint_tolerance_rad",
        "gripper_open_duration_s",
        "gripper_close_duration_s",
        "mode_settle_s",
        "gohome_not_movable_retry_s",
        "gohome_fallback_duration_s",
        "service_timeout_s",
        "feedback_timeout_s",
        "rate_hz",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.gohome_settle_s:
        parser.error("--gohome-settle-s must be non-negative")
    if args.rate_hz > 100.0:
        parser.error("--rate-hz must not exceed 100")
    return args


def main() -> int:
    args = parse_args()
    try:
        prepare = load_pose(args.prepare_pose.resolve())
        grasp = load_pose(args.grasp_pose.resolve())
        home = (
            load_pose(args.home_pose.resolve())
            if args.home_mode == "direct"
            else None
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Luna clothes pose sequence")
    if args.home_mode == "direct":
        print(f"  1. clothes home pose: {args.home_pose}")
    else:
        print("  1. go_home service")
    print(f"  2. open both grippers for {args.gripper_open_duration_s:.1f}s")
    print(f"  3. prepare pose: {args.prepare_pose}")
    print(f"  4. final grasp pose: {args.grasp_pose}")
    print(f"  5. close {args.gripper_side} gripper for {args.gripper_close_duration_s:.1f}s")
    if args.home_mode == "direct":
        print("  6. return to configured clothes home with gripper held closed")
    else:
        print("  6. go_home service with gripper held closed")
    if not args.execute:
        print("DRY-RUN: no ROS commands or motion were sent.")
        print("Pass --execute to run the supervised hardware sequence.")
        return 0

    rclpy.init()
    node = ClothesPoseSequence(args)
    try:
        node.wait_feedback()
        node.wait_ready()
        node.wait_gripper_ready()
        node.call_trigger(node.set_ready_client, "set_ready")
        node.ensure_control_mode(3 if args.home_mode == "direct" else 1)
        node.spin_for(args.mode_settle_s, publish_gripper=False)
        node.wait_ready()

        direct_home_pose: dict[str, Any] | None = None
        if args.home_mode == "direct":
            assert home is not None
            direct_home_pose = home
            node.switch_joint_source(compat_input=3, expected_source=2)
            node.move_to_pose(direct_home_pose, "configured clothes home")
        else:
            node.go_home(keep_gripper_closed=False)
            node.switch_joint_source(compat_input=3, expected_source=2)
        node.open_grippers()
        node.move_to_pose(prepare, "prepare pose")
        node.move_to_pose(grasp, "final grasp pose")
        node.close_gripper()

        try:
            if args.home_mode == "direct":
                assert direct_home_pose is not None
                node.move_to_pose_with_gripper(
                    direct_home_pose,
                    "configured clothes home return",
                    hold_gripper_closed=True,
                )
            else:
                node.go_home(keep_gripper_closed=True)
        except BaseException:
            # Keep sending the last closed-gripper command while preserving the
            # original home failure for the caller.
            node.gripper_pub.publish(Float32(data=1.0))
            raise
        print("PASS: clothes pose sequence completed; gripper remains closed.")
        return 0
    except (KeyboardInterrupt, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
