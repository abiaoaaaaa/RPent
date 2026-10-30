# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Offline grasp-gate regression without importing ROS hardware dependencies."""

from __future__ import annotations

import ast
import io
import math
import re
from collections import deque
from pathlib import Path
from types import SimpleNamespace

SCRIPT = (
    Path(__file__).parents[4]
    / "test.incoming_20261021"
    / "apriltag_left_tool_grasp_test.py"
)


def test_stable_but_off_target_tool_does_not_start_grasp():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    grasp = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TagGrasp"
    )
    method = next(
        node
        for node in grasp.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "maybe_accept_stable_plateau"
    )
    namespace = {
        "math": math,
        "time": SimpleNamespace(monotonic=lambda: 2.0),
        "angular_error_deg": lambda _actual, _target: 0.0,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(SCRIPT),
            "exec",
        ),
        namespace,
    )

    quaternion = (0.0, 0.0, 0.0, 1.0)
    target = (1.0, 0.0, 0.0)
    node = SimpleNamespace(
        phase="approach",
        exit_code=None,
        target=target,
        target_quaternion=quaternion,
        gripper_args=SimpleNamespace(
            stability_window_sec=1.5,
            stability_position_jitter_mm=0.5,
            stability_angle_jitter_deg=0.3,
        ),
        args=SimpleNamespace(position_tolerance_mm=1.0, angle_tolerance_deg=0.5),
        last_stability_log_at=2.0,
        stability_samples=deque(),
    )
    outcomes = []
    node.finish = lambda code, reason: outcomes.append((code, reason))
    node.reset_stability_window = node.stability_samples.clear
    position = (0.0, 0.0, 0.0)
    node.get_actuals = lambda: ((position, quaternion), None)
    node.stability_samples.extend(
        ((0.0, position, quaternion), (1.0, position, quaternion))
    )

    namespace["maybe_accept_stable_plateau"](node)

    assert outcomes == []

    node.stability_samples.clear()
    node.stability_samples.extend(
        ((0.0, target, quaternion), (1.0, target, quaternion))
    )
    node.get_actuals = lambda: ((target, quaternion), None)
    namespace["maybe_accept_stable_plateau"](node)
    assert outcomes and outcomes[0][0] == 0


def test_grasp_publishes_full_apriltag_waypoint_without_local_step_limit():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    grasp = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "TagGrasp")
    methods = [node for node in grasp.body if isinstance(node, ast.FunctionDef)
               and node.name in {"get_actuals", "make_target", "publish_target"}]

    class Parent:
        def make_target(self, *_):
            self.target = (0.1, 0.2, 0.3)

        def get_actuals(self):
            return (((0.0, 0.0, 0.0), None), None)

        def publish_target(self):
            self.published.append(tuple(self.target))

    probe = ast.ClassDef(
        name="GraspProbe", bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[],
    )
    namespace = {"Parent": Parent, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])),
                 str(SCRIPT), "exec"), namespace)
    node = namespace["GraspProbe"]()
    node.gripper_args = SimpleNamespace(approach_height_mm=30.0)
    node.get_logger = lambda: SimpleNamespace(warning=lambda *_: None)
    node.phase = "approach"
    node.published = []

    node.make_target(None, None, None)
    node.get_actuals()
    node.publish_target()

    assert len(node.published) == 1
    assert all(math.isclose(actual, expected, abs_tol=1e-9)
               for actual, expected in zip(node.published[0], (0.1, 0.2, 0.33)))
    assert node.approach_final_position == (0.1, 0.2, 0.3)


def test_visual_gate_holds_arm_and_rejects_nonheld_verdict():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    grasp = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "TagGrasp")
    method = next(node for node in grasp.body if isinstance(node, ast.FunctionDef)
                  and node.name == "visual_wait_tick")
    incoming = io.StringIO("reject\n")
    ready = False
    namespace = {
        "time": SimpleNamespace(monotonic=lambda: 10.0),
        "math": math,
        "angular_error_deg": lambda actual, target: 0.0,
        "select": SimpleNamespace(select=lambda *_: ([incoming], [], []) if ready
                                  else ([], [], [])),
        "sys": SimpleNamespace(stdin=incoming),
        "TransformException": ValueError,
        "GRIPPER_CLOSED": 1.0,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(SCRIPT), "exec"), namespace)
    calls = []
    node = SimpleNamespace(
        visual_wait_started_at=9.0,
        gripper_args=SimpleNamespace(visual_verdict_timeout_s=30.0,
                                     home_on_visual_reject=True),
        args=SimpleNamespace(feedback_timeout=1.0, position_tolerance_mm=7.0,
                             angle_tolerance_deg=2.5),
        last_valid_tf=10.0,
        target=(0.0, 0.0, 0.0),
        target_quaternion=(0.0, 0.0, 0.0, 1.0),
        get_actuals=lambda: (((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), None),
        check_mux=lambda: None,
        publish_target=lambda: calls.append("hold_arm"),
        publish_gripper=lambda value, now: calls.append("hold_gripper"),
        start_destination_move=lambda reason: calls.append("transport"),
        start_go_home=lambda: calls.append("go_home"),
        finish=lambda code, reason: calls.append("reject"),
    )

    namespace["visual_wait_tick"](node)
    assert calls == ["hold_arm", "hold_gripper"]
    ready = True
    namespace["visual_wait_tick"](node)
    assert calls[-1] == "reject"
    assert "transport" not in calls

    incoming = io.StringIO("held\n")
    namespace["sys"].stdin = incoming
    namespace["visual_wait_tick"](node)
    assert calls[-1] == "transport"

    incoming = io.StringIO("retry\n")
    namespace["sys"].stdin = incoming
    namespace["visual_wait_tick"](node)
    assert calls[-1] == "go_home"
    assert node.retry_home_requested is True


def test_retry_waits_for_go_home_plan_and_live_feedback():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    grasp = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "TagGrasp")
    method = next(node for node in grasp.body if isinstance(node, ast.FunctionDef)
                  and node.name == "go_home_tick")

    class Parent:
        def finish(self, code, reason):
            raise AssertionError("ordinary home completion was used")

    clock = [100.0]
    namespace = {
        "math": math, "re": re,
        "time": SimpleNamespace(monotonic=lambda: clock[0]),
        "TransformException": ValueError, "GRIPPER_CLOSED": 1.0,
        "super": lambda: Parent(),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(SCRIPT), "exec"), namespace)
    completed = []
    node = SimpleNamespace(
        go_home_future=SimpleNamespace(
            done=lambda: True,
            result=lambda: SimpleNamespace(success=True,
                                           message="Go home started, duration=2.000000s, nq=22"),
        ),
        go_home_started_at=100.0,
        go_home_wait_until=None,
        retry_home_requested=True,
        get_logger=lambda: SimpleNamespace(warning=lambda _: None),
        get_actuals=lambda: ((0, 0), None),
        check_gripper=lambda now: None,
        publish_gripper=lambda value, now: None,
        finish=lambda code, reason: completed.append((code, reason)),
    )
    namespace["go_home_tick"](node)
    assert node.go_home_wait_until == 103.0
    clock[0] = 102.0
    namespace["go_home_tick"](node)
    assert completed == []
    clock[0] = 103.0
    namespace["go_home_tick"](node)
    assert node.phase == "retry_ready"
    assert completed[0][0] == 0
