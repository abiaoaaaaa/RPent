# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dashboard startup memory requirements before shared robot services."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.cli import dashboard
from rpent.cli.main import _build_argparser
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import DashboardState
from rpent.memory import MemoryManager


def test_luna_camera_layout_places_wrists_in_a_vertical_stack():
    css_path = (
        Path(dashboard.__file__).resolve().parents[1]
        / "dashboard/static/dashboard.css"
    )
    css = css_path.read_text(encoding="utf-8")
    assert "grid-template-columns: minmax(0, 2fr) minmax(150px, 1fr)" in css
    assert "grid-row: 1 / span 2" in css
    assert "--camera-color: #0f9f76" in css
    assert "--camera-color: #d97706" in css
    assert "body { font-size: 17px; }" in css
    assert ".composer textarea { font-size: 16px; }" in css
    assert ".tool-evidence" in css
    assert ".ev.text.stage-success" in css
    assert ".ev.tool_result.tool-success" in css
    root = Path(dashboard.__file__).resolve().parents[1] / "dashboard"
    js = (root / "static/dashboard.js").read_text(encoding="utf-8")
    html = (root / "index.html").read_text(encoding="utf-8")
    assert "/api/live-camera-frame" in js
    assert "window.setTimeout(tick, delay)" in js
    assert "Math.min(\n      3000," in js
    assert "details.open = true" in js
    assert "中文推理摘要" in js
    assert "dashboard_image_kind" in js
    assert "阶段验证成功：" in js
    assert "tool-success" in js
    assert 'id="lunaCameraTop"' in html
    assert '<canvas id="lunaCameraTop"' in html


@pytest.mark.parametrize(
    ("robot", "profile", "sync_before_shared", "explicit_repo"),
    [
        ("robocasa", "hf", True, False),
        ("robocasa", "hf", True, True),
        ("robotwin", "hf", True, False),
        ("robotwin", "hf", True, True),
        ("robocasa", "local", False, False),
        ("libero", "hf", False, False),
        ("libero", "local", False, False),
        ("dual_franka", "local", False, False),
    ],
)
@pytest.mark.parametrize("planner", ["api", "claude_code", "codex"])
def test_shared_runtime_gets_memory_before_startup(
    robot, profile, sync_before_shared, planner, tmp_path, monkeypatch, explicit_repo
):
    root = tmp_path / "memory" / robot
    events = []

    def sync(manager, *, remote_repo):
        assert manager.root == root
        assert remote_repo == "test/memory"
        global_file = root / "global" / "GLOBAL_MEMORY.md"
        global_file.parent.mkdir(parents=True)
        global_file.write_text("Live task observations take precedence.")
        events.append("sync")
        return root

    def init_runtime(args, output_dir, state, components):
        assert components == {"vla"}
        if sync_before_shared:
            assert (root / "global" / "GLOBAL_MEMORY.md").is_file()
        events.append("shared-runtime")
        return [], {}

    def start(server):
        # Exercise actual session orchestration, then exit after service startup.
        server._state.request_shutdown()
        return "http://127.0.0.1:12345"

    def fail_session(state, error):
        pytest.fail(f"Shared service startup failed: {error}")

    monkeypatch.setattr(dashboard, "get_memory_dir", lambda name: root)
    monkeypatch.setattr(MemoryManager, "sync", sync)
    if explicit_repo:
        from rpent.memory import source

        def prepare(source_url, name):
            assert name == robot and source_url == "team/memory@release"
            sync(MemoryManager(root), remote_repo="test/memory")
            return root

        monkeypatch.setattr(source, "prepare_explicit_source", prepare)
    monkeypatch.setattr(DashboardServer, "start", start)
    monkeypatch.setattr(DashboardState, "fail_session", fail_session)
    spec = SimpleNamespace(
        name=robot,
        is_real_robot=robot == "dual_franka",
        supports_human_interactive_exploration=robot == "dual_franka",
        memory_repo_id="test/memory",
        prepare_memory=(lambda args, config: None) if robot == "libero" else None,
        dashboard={
            "task": {
                "command": "/rpent-task",
                "usage": "/rpent-task <seed>",
                "fields": ({"name": "seed", "kind": "integer", "minimum": 0},),
                "display": "seed {seed}",
                "output_slug": "s{seed}",
            },
            "runtime_components": ({"name": "vla", "scope": "shared"},),
            "primitives": (),
        },
        init_runtime=init_runtime,
    )
    args = _build_argparser().parse_args(
        [
            "--robot",
            robot,
            "--dashboard",
            "--planner",
            planner,
            "--model",
            "test",
            "--memory-profile",
            profile,
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    if explicit_repo:
        args.memory_repo = "team/memory@release"
    assert (
        dashboard.run_dashboard_session(args, spec, parser=argparse.ArgumentParser())
        == 0
    )
    assert events == (["sync"] if sync_before_shared else []) + ["shared-runtime"]
    if explicit_repo:
        assert args.memory_dir == str(root)
