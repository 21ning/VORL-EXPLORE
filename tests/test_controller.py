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


def test_infeasible_planner_does_not_create_phantom_oscillations():
    """Count final consecutive-step gate states, not transient failed attempts."""
    from vorl.world import SensorState
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/reconstruction.json").read_text())
    known = np.zeros((9, 9), dtype=np.uint8)
    sensor = SensorState(known, [(4, 4)], [np.zeros((7, 7), dtype=np.uint8)], np.zeros(1))
    controller = Controller(sensor, FixturePolicy(), config, {"weights": [0.0] * 8, "bias": 20.0}, seed=71)
    for step in range(6):
        controller.decide(sensor, DistanceOracle(known), step)
        assert not controller.switches[0].planner_selected
    # One true initial A* -> RL transition. Each later high-fidelity attempt is
    # rejected by feasibility before execution and must not add another switch.
    assert controller.counts["mode_switches"] == 1
    assert sum(controller.switch_history[0]) == 1


def test_recovery_overrides_planner_with_real_reactive_action():
    from vorl.world import SensorState
    class UpPolicy(FixturePolicy):
        def act(self, grids, coordinates, goals):
            return np.ones(len(grids), dtype=int), np.zeros((len(grids), 5))
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/reconstruction.json").read_text())
    known = np.zeros((9, 9), dtype=np.uint8)
    known[4, 8] = 255
    sensor = SensorState(known, [(4, 4)], [np.zeros((7, 7), dtype=np.uint8)], np.zeros(1))
    controller = Controller(sensor, UpPolicy(), config, {"weights": [0.0] * 8, "bias": 20.0}, seed=71)
    controller.goals = [(4, 7)]
    controller.recovery_remaining[0] = 1
    decision = controller.decide(sensor, DistanceOracle(known), 1)
    assert decision["planner_selected"] == [True]
    assert decision["planner_feasible"] == [True]
    assert decision["modes"] == [2]
    assert decision["actions"] == [1]  # Real RL proposes up; A* would propose right.
