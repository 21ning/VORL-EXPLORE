"""Ensure success-only rendering cannot relabel an incomplete trajectory."""
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("render_rollout", Path(__file__).resolve().parents[1] / "scripts/render_rollout.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def incomplete_fixture(tmp_path, *, claimed_success=False, corrupt_hash=False):
    run = tmp_path / "run"
    run.mkdir()
    known = np.zeros((2, 5, 5), dtype=np.uint8)
    known[:, 0, 0] = 255
    np.savez_compressed(run / "trajectory.npz", known=known)
    digest = hashlib.sha256((run / "trajectory.npz").read_bytes()).hexdigest()
    summary = {"phase": "method_evaluation", "success_no_frontiers": claimed_success,
               "trajectory_sha256": "0" * 64 if corrupt_hash else digest}
    (run / "summary.json").write_text(json.dumps(summary))
    return run


@pytest.mark.parametrize("claimed_success,corrupt_hash,message", [
    (False, False, "requires no remaining frontiers"),
    (True, False, "Completion flag"),
    (False, True, "Trajectory hash"),
])
def test_unverified_success_never_creates_gif(tmp_path, claimed_success, corrupt_hash, message):
    run = incomplete_fixture(tmp_path, claimed_success=claimed_success, corrupt_hash=corrupt_hash)
    output = tmp_path / "render"
    with pytest.raises(ValueError, match=message):
        renderer.render(run, output, require_success=True)
    assert not output.exists()


def recording_fixture(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    known = np.full((3, 5, 5), 255, dtype=np.uint8)
    known[0, :4, :4] = 0
    known[1, :4, :] = 0
    known[2] = 0
    known[2, 4, 3:] = 1
    static = np.zeros((5, 5), dtype=np.uint8)
    static[4, 4] = 1
    trajectory = run / "trajectory.npz"
    data = dict(known=known, static_map=static,
                positions=np.array([[[0, 0]], [[0, 1]], [[1, 1]]]),
                dynamic=np.array([[[4, 3]], [[4, 3]], [[4, 3]]]),
                goals=np.array([[[0, 1]], [[1, 1]]]))
    np.savez_compressed(trajectory, **data)
    config = {"sensing_radius": 3}
    (run / "config.json").write_text(json.dumps(config))
    summary = {"phase": "method_evaluation", "success_no_frontiers": True,
               "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
               "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "size": 5, "robots": 1, "dynamic_obstacles": 1, "online_adaptation": False,
               "seed": 99, "profile": "synthetic rendering test only"}
    (run / "summary.json").write_text(json.dumps(summary))
    return run, data, summary


def test_verified_terminal_map_renders_success_gif(tmp_path):
    pytest.importorskip("pogema")
    pytest.importorskip("cairosvg")
    run, _, _ = recording_fixture(tmp_path)
    output = tmp_path / "render"
    renderer.render(run, output, require_success=True, width=200)
    report = json.loads((output / "render-check.json").read_text())
    assert report["success_no_frontiers"] and report["final_frontier_count"] == 0
    assert report["rendered_state_indices"] == [0, 1, 2]
    assert report["decoded_gif_frames"] == 4  # Intro plus three recorded states.
    assert report["backend"] == "pogema.animation.AnimationMonitor"
    assert report["native_history_matches_recording"] and report["shared_map_replay_matches_recording"]
    assert report["gif_duration_ms"] == renderer.INTRO_MS + 2 * renderer.STEP_MS + renderer.FINAL_HOLD_MS
    assert report["gif_sha256"] == hashlib.sha256((output / "vorl-explore.gif").read_bytes()).hexdigest()
    assert report["svg_sha256"] == hashlib.sha256((output / "vorl-explore.svg").read_bytes()).hexdigest()


def test_team_sensing_and_current_dynamic_visibility(tmp_path):
    _, data, summary = recording_fixture(tmp_path)
    visible = renderer.validate_recording(data, summary, 3)
    assert visible[:, 0].tolist() == [False, False, True]
    data["known"][0, 4, 4] = 1  # A ground-truth leak before this cell is sensed.
    with pytest.raises(ValueError, match="Shared map differs"):
        renderer.validate_recording(data, summary, 3)


def test_team_map_is_union_not_one_robots_view():
    known = np.full((1, 5, 5), 255, dtype=np.uint8)
    known[0, :2, :2] = known[0, 3:, 3:] = 0
    data = dict(known=known, static_map=np.zeros((5, 5), dtype=np.uint8),
                positions=np.array([[[0, 0], [4, 4]]]), dynamic=np.zeros((1, 0, 2), dtype=int),
                goals=np.zeros((0, 2, 2), dtype=int))
    summary = {"size": 5, "robots": 2, "dynamic_obstacles": 0}
    renderer.validate_recording(data, summary, 1)
    data["known"][0, 3:, 3:] = 255
    with pytest.raises(ValueError, match="Shared map differs"):
        renderer.validate_recording(data, summary, 1)


@pytest.mark.parametrize("change,message", [("jump", "more than one"), ("overlap", "overlap"),
                                           ("outside", "outside"), ("obstacle", "static obstacle")])
def test_invalid_playback_geometry(tmp_path, change, message):
    _, data, summary = recording_fixture(tmp_path)
    if change == "jump":
        data["positions"][1, 0] = [0, 3]
    elif change == "overlap":
        data["dynamic"][0, 0] = data["positions"][0, 0]
    elif change == "outside":
        data["positions"][0, 0] = [-1, 0]
    else:
        data["positions"][0, 0] = [4, 4]
    with pytest.raises(ValueError, match=message):
        renderer.validate_recording(data, summary, 3)


def test_native_frames_start_unknown_and_reveal_only_recorded_cells(tmp_path):
    pytest.importorskip("pogema")
    _, data, summary = recording_fixture(tmp_path)
    monitor = renderer.make_monitor(data, summary, renderer.validate_recording(data, summary, 3))
    offset = monitor.recorded_offset
    assert np.array_equal(np.asarray(monitor.agents_xy_history)[offset:offset + 3, :1], data["positions"])
    assert np.array_equal(monitor.shared_history[offset:offset + 3], data["known"])
    assert np.all(monitor.shared_history[:offset] == 255)
    ns = {"s": "http://www.w3.org/2000/svg"}
    for index, unknown_cells, visible_moving in [(0, 25, False), (offset, 9, False), (offset + 2, 0, True)]:
        root = ET.fromstring(renderer.native_frame_svg(monitor, index))
        fog = [r for r in root.findall("s:rect", ns) if r.get("fill") == renderer.UNKNOWN_COLOR]
        assert sum(r.get("visibility") == "visible" for r in fog) == unknown_cells
        moving = [c for c in root.findall("s:circle", ns) if c.get("fill") == renderer.DYNAMIC_COLOR]
        assert len(moving) == 1
        assert (moving[0].get("visibility") == "visible") == visible_moving
        targets = [c for c in root.findall("s:circle", ns) if c.get("fill") == "none"]
        assert len(targets) == 1  # No goal ring for the moving obstacle.
        if index in (0, offset + 2):
            assert targets[0].get("visibility") == "hidden"


def test_svg_only_does_not_need_cairo(tmp_path, monkeypatch):
    pytest.importorskip("pogema")
    import sys
    monkeypatch.setitem(sys.modules, "cairosvg", None)
    monkeypatch.setitem(sys.modules, "PIL", None)
    run, _, _ = recording_fixture(tmp_path)
    output = tmp_path / "render"
    renderer.render(run, output, require_success=True, svg_only=True)
    assert (output / "vorl-explore.svg").exists()
    assert not (output / "vorl-explore.gif").exists()
