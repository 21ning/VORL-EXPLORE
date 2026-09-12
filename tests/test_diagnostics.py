import importlib.util
from pathlib import Path

import numpy as np


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load_script("reference_smoke")


def test_vertex_conflict_is_blocked():
    truth = np.zeros((5, 5), dtype=np.uint8)
    old = [(1, 1), (1, 3)]
    assert smoke.safe_step(truth, old, [(1, 2), (1, 2)]) == old


def test_swap_and_static_collision_are_blocked():
    truth = np.zeros((5, 5), dtype=np.uint8)
    old = [(1, 1), (1, 2)]
    assert smoke.safe_step(truth, old, list(reversed(old))) == old
    truth[2, 1] = 1
    assert smoke.safe_step(truth, old, [(2, 1), (1, 2)]) == old


def test_nonlocal_and_out_of_bounds_moves_are_blocked():
    truth = np.zeros((5, 5), dtype=np.uint8)
    old = [(0, 0), (1, 2)]
    assert smoke.safe_step(truth, old, [(-1, 0), (4, 2)]) == old


def test_observation_does_not_reveal_outside_sensor():
    truth = np.zeros((12, 12), dtype=np.uint8)
    known = np.full_like(truth, 255)
    smoke.observe(truth, known, [(5, 5)], radius=1)
    assert np.count_nonzero(known != 255) == 9
    assert known[3, 5] == 255


def test_render_labels_diagnostic(tmp_path):
    truth = smoke.make_map()
    known = np.full_like(truth, 255)
    frame = smoke.render_frame(truth, known, [(2, 2)], [None], [[(2, 2)]], 0)
    assert frame.size == (432, 542)
    frame.save(tmp_path / "frame.png")
