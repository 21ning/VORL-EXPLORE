"""Ensure success-only rendering cannot relabel an incomplete trajectory."""
import hashlib
import importlib.util
import json
from pathlib import Path

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


def test_verified_terminal_map_renders_success_gif(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    known = np.full((2, 40, 40), 255, dtype=np.uint8)
    known[0, :5, :5] = 0
    known[1] = 0
    trajectory = run / "trajectory.npz"
    np.savez_compressed(trajectory, known=known, static_map=np.zeros((40, 40), dtype=np.uint8),
                        positions=np.array([[[2, 2]], [[2, 3]]]), dynamic=np.zeros((2, 0, 2), dtype=int),
                        goals=np.array([[[2, 3]]]), fidelity=np.array([[0.8]]), modes=np.array([[0]]))
    summary = {"phase": "method_evaluation", "success_no_frontiers": True,
               "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
               "size": 40, "robots": 1, "dynamic_obstacles": 0, "online_adaptation": False,
               "seed": 99, "profile": "synthetic rendering test only"}
    (run / "summary.json").write_text(json.dumps(summary))
    output = tmp_path / "render"
    renderer.render(run, output, require_success=True)
    report = json.loads((output / "render-check.json").read_text())
    assert report["success_no_frontiers"] and report["final_frontier_count"] == 0
    assert report["rendered_state_indices"] == [0, 1]
    assert report["decoded_gif_frames"] == 2
    assert report["gif_sha256"] == hashlib.sha256((output / "vorl-explore.gif").read_bytes()).hexdigest()
