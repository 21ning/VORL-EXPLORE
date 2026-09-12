import json
from pathlib import Path

import numpy as np

from vorl.controller import Controller
from vorl.grid import DistanceOracle
from vorl.world import GridWorld


class FixturePolicy:
    """Unit-test observation spy; never selected by any runtime entry point."""
    radius = 7
    def __init__(self):
        self.calls = 0
    def act(self, grids, coordinates, goals):
        self.calls += 1
        assert grids.shape[1:] == (3, 15, 15)
        assert coordinates.shape == goals.shape == (len(grids), 2)
        return np.zeros(len(grids), dtype=int), np.zeros((len(grids), 5))


def test_controller_calls_policy_and_keeps_frozen_gate_fixed():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/reconstruction.json").read_text())
    world = GridWorld(size=16, robots=4, dynamic_obstacles=3, density=0.3, seed=71)
    sensor, policy = world.observe(), FixturePolicy()
    controller = Controller(sensor, policy, config, {"weights": [0.0] * 8, "bias": 0.0}, seed=71)
    before = controller.model.weights.copy()
    for step in range(6):
        decision = controller.decide(sensor, DistanceOracle(sensor.shared_map), step)
        assert decision["features"].shape == (4, 8)
        assert len(decision["actions"]) == 4
        sensor, _ = world.step(decision["actions"])
    assert policy.calls == 6
    assert np.array_equal(before, controller.model.weights)
    assert controller.counts["reassignments"] > 0


def test_runtime_gate_guard_rejects_unfitted_checkpoint(tmp_path):
    from vorl.runner import run_episode
    import pytest
    with pytest.raises(ValueError, match="fitted"):
        run_episode(config={}, policy_dir=tmp_path / "missing", gate={"weights": [0] * 8},
                    seed=1, size=40, robots=4, dynamic_obstacles=8, density=0.3,
                    horizon=320, output=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
