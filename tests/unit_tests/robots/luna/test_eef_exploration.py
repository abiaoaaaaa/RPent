# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline contracts for the separate Luna EEF cube task."""

from __future__ import annotations

import argparse
import ast
import io
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from robots.luna.eef import LunaEefPrimitives, LunaEefSession
from robots.luna.robot_spec import _parse_config, get_robot_spec
from robots.luna.toolkit import LunaToolkit
from rpent.memory import MemoryManager
from rpent.utils.logging import init_output_dir


def _mosaic() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (1280, 2400), "white").save(output, format="JPEG")
    return output.getvalue()


def test_task_three_has_no_base_tools_and_starts_observation_only(tmp_path: Path):
    spec = get_robot_spec()
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(["--task-id", "3"])
    args.output_dir = str(tmp_path)
    config = _parse_config(args)
    prompt = spec.prompts.render("system", variables=config.prompt_vars)
    assert config.prompt_vars["mode"] == "eef"
    assert "Do not move the base" in config.prompt_vars["instruction"]
    assert "move_left_eef_delta" in prompt
    assert "up to 200 mm vector norm" in prompt
    assert "at most three exploratory move_left_eef_delta calls" in prompt
    assert "5-10 mm, dz_mm=0" in prompt
    assert "do not descend toward the table" in prompt.lower()
    assert "Call finish with status failure or stuck" in prompt
    assert args.luna_eef_execute is False

    calls = []
    session = SimpleNamespace(
        check_cancelled=lambda: None,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        execute=False,
        call=lambda action, **kwargs: (
            calls.append((action, kwargs))
            or {
                "ok": True,
                "xyz_m": [0.3, 0.2, 1.0],
            }
        ),
        close=lambda: None,
    )
    client = SimpleNamespace(camera_jpeg=_mosaic, get=lambda path: {"connected": True})
    toolkit = LunaToolkit(
        runtime_kwargs={"mode": "eef", "session": session, "client": client},
        dashboard_events=SimpleNamespace(emit=lambda event: None),
        memory=MemoryManager(root=tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )
    init_output_dir(tmp_path / "output")
    names = {item.name for item in toolkit.list_tools()}
    assert {
        "observe_cube",
        "move_left_eef_delta",
        "open_left_gripper",
        "close_left_gripper",
        "verify_cube_grasp",
    } <= names
    assert "move_to_table_3" not in names
    assert "pick_apriltag_module" not in names
    observed = toolkit.execute_tool("observe_cube", {})
    assert not observed.is_error
    assert observed.data["success"] is True
    assert len(observed.images) == 1
    assert observed.data["min_tool_z_m"] == pytest.approx(0.86)
    assert calls == [("state", {})]
    toolkit.close()


def test_eef_delta_rejects_nonfinite_and_oversize_without_session_call():
    session = SimpleNamespace(
        call=lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("motion started"))
    )
    primitives = LunaEefPrimitives(session=session, client=SimpleNamespace())
    assert primitives.move_left_eef_delta(200.01, 0, 0).is_error
    assert primitives.move_left_eef_delta(150, 150, 0).is_error
    assert primitives.move_left_eef_delta(float("nan"), 0, 0).is_error
    assert primitives.move_left_eef_delta(0, 0, 0).is_error


def test_eef_delta_accepts_up_to_200_mm_vector_norm():
    calls = []
    session = SimpleNamespace(
        call=lambda action, **kwargs: calls.append((action, kwargs)) or {"ok": True}
    )
    primitives = LunaEefPrimitives(session=session, client=SimpleNamespace())
    assert not primitives.move_left_eef_delta(120, 160, 0).is_error
    assert calls == [("move", {"delta_mm": [120, 160, 0]})]


def test_ros_child_uses_200_mm_vector_norm_without_ros_import():
    path = (
        Path(__file__).resolve().parents[4]
        / "test.incoming_20261021/luna_eef_session.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "bounded_delta"
    )
    namespace = {"math": math}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    assert namespace["bounded_delta"]([120, 160, 0]) == [120, 160, 0]
    with pytest.raises(ValueError, match="exceeds 200 mm"):
        namespace["bounded_delta"]([150, 150, 0])


def test_eef_session_allows_bounded_remote_clock_skew_without_relaxing_stale_tf():
    pose_script = (
        Path(__file__).resolve().parents[4]
        / "test.incoming_20261021/interactive_left_tool_pose.py"
    ).read_text()
    session_script = (
        Path(__file__).resolve().parents[4]
        / "test.incoming_20261021/luna_eef_session.py"
    ).read_text()
    assert "--tf-future-skew-s', type=float, default=1.0" in pose_script
    assert "-self.args.tf_future_skew_s <= age <= self.args.feedback_timeout" in pose_script
    assert '"--tf-future-skew-s",\n        "1.0"' in session_script


def test_eef_rotation_is_bounded_and_uses_ros_session():
    calls = []
    session = SimpleNamespace(
        call=lambda action, **kwargs: calls.append((action, kwargs)) or {"ok": True}
    )
    primitives = LunaEefPrimitives(session=session, client=SimpleNamespace())
    assert primitives.rotate_left_eef_delta(11, 0, 0).is_error
    assert primitives.rotate_left_eef_delta(0, 0, 0).is_error
    assert not primitives.rotate_left_eef_delta(0, 0, 10).is_error
    assert calls == [("rotate", {"delta_deg": [0, 0, 10]})]


def test_ros_child_rejects_oversized_rotation_without_ros_import():
    path = (
        Path(__file__).resolve().parents[4]
        / "test.incoming_20261021/luna_eef_session.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "bounded_rotation"
    )
    namespace = {"math": math}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    assert namespace["bounded_rotation"]([0, 0, 10]) == [0, 0, 10]
    with pytest.raises(ValueError, match="at most 10"):
        namespace["bounded_rotation"]([0, 0, 10.1])


def test_ros_child_claims_qp_sources_and_waits_for_mux_confirmation():
    path = (
        Path(__file__).resolve().parents[4]
        / "test.incoming_20261021/luna_eef_session.py"
    )
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "claim_eef_control"
    )
    calls = []
    namespace = {
        "topic_name": lambda namespace, name: f"{namespace.rstrip('/')}/{name}",
        "call_int_service": lambda _node, service, value, timeout: calls.append(
            (service, value, timeout)
        ),
        "wait_for_mux_ready": lambda _node, **kwargs: calls.append(("wait", kwargs)),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )

    namespace["claim_eef_control"](
        SimpleNamespace(args=SimpleNamespace(namespace="/tj")), timeout_s=5.0
    )

    assert calls == [
        ("/tj/control/ik_request_mux/select", 0, 5.0),
        ("/tj/control/set_input", 1, 5.0),
        ("wait", {"timeout_s": 5.0, "allow_source_transition": True}),
    ]


def test_owned_eef_process_protocol_and_cleanup(tmp_path: Path):
    script = tmp_path / "luna_eef_session.py"
    script.write_text(
        "import json, sys\n"
        "print('RPENT_EEF=' + json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "  req = json.loads(line)\n"
        "  print('RPENT_EEF=' + json.dumps({'id': req['id'], 'ok': True}), flush=True)\n"
    )
    session = LunaEefSession(
        script_dir=tmp_path,
        python_executable=sys.executable,
        ros_setup_script=None,
        output_dir=tmp_path / "output",
        check_cancelled=lambda: None,
        execute=False,
        table_z_m=0.8,
        tool_clearance_m=0.06,
    )
    result = session.call("state")
    assert result["ok"] is True
    assert Path(result["log_path"]).is_file()
    session.close()
    assert session.process is None


def test_owned_right_eef_process_receives_right_side_argument(tmp_path: Path):
    script = tmp_path / "luna_eef_session.py"
    script.write_text(
        "import json, sys\n"
        "print('RPENT_EEF=' + json.dumps({'ready': True, 'argv': sys.argv}), flush=True)\n"
        "for line in sys.stdin:\n"
        "  req = json.loads(line)\n"
        "  print('RPENT_EEF=' + json.dumps({'id': req['id'], 'ok': True}), flush=True)\n"
    )
    session = LunaEefSession(
        script_dir=tmp_path,
        python_executable=sys.executable,
        ros_setup_script=None,
        output_dir=tmp_path / "output",
        check_cancelled=lambda: None,
        execute=False,
        table_z_m=0.8,
        tool_clearance_m=0.06,
        side="right",
    )
    session.start()
    assert session.process is not None
    assert session.side == "right"
    assert '"right"' in session.log_path.read_text()
    session.close()


def test_gripper_check_allows_registered_publishers_but_requires_fresh_feedback():
    # The ROS child cannot be imported in the offline unit-test environment.
    path = (
        Path(__file__).resolve().parents[4]
        / "test.incoming_20261021/luna_eef_session.py"
    )
    module = ast.parse(path.read_text())
    gripper_class = next(
        item
        for item in module.body
        if isinstance(item, ast.ClassDef) and item.name == "GripperHold"
    )
    method = next(
        item
        for item in gripper_class.body
        if isinstance(item, ast.FunctionDef) and item.name == "check"
    )
    namespace = {"time": time, "GRIPPER_TOPIC": "/tj/control/gripperValueL"}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    now = time.monotonic()
    gripper = SimpleNamespace(
        side="left",
        publisher=SimpleNamespace(get_subscription_count=lambda: 1),
        node=SimpleNamespace(
            get_publishers_info_by_topic=lambda _topic: [
                SimpleNamespace(node_name="VR_reader", node_namespace="/tj")
            ],
            get_name=lambda: "eef",
            get_namespace=lambda: "/",
        ),
        feedback_at=now,
        status_at=now,
        status=(1,),
    )
    namespace["check"](gripper)
    gripper.feedback_at = now - 1
    with pytest.raises(ValueError, match="stale or not ready"):
        namespace["check"](gripper)
