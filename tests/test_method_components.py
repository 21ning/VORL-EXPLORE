import numpy as np
import pytest

from vorl.assignment import AssignmentParameters, assign_frontiers, coupled_scores, repulsion_scores
from vorl.fidelity import FidelityModel, HysteresisGate, quality_score, sigmoid
from vorl.grid import bfs_distances, frontiers, minmax


# Values below are mathematical test fixtures, not learned or paper hyperparameters.
PARAMETERS = AssignmentParameters(0.1, 2.0, 0.1, 2.0, 3, 1, 1, 0.5)


def test_frontier_requires_adjacent_unknown():
    shared = np.array([[0, 0, 255], [0, 1, 1]], dtype=np.uint8)
    assert frontiers(shared) == [(0, 1)]


def test_bfs_blocks_unknown_and_does_not_use_manhattan_through_walls():
    shared = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    assert bfs_distances(shared, (0, 0))[0, 2] == 6
    shared[2, 1] = 255
    assert np.isinf(bfs_distances(shared, (0, 0))[0, 2])


def test_normalization_handles_constant_empty_and_invalid():
    assert np.array_equal(minmax([4, 4]), [0, 0])
    assert minmax([]).size == 0
    with pytest.raises(ValueError):
        minmax([np.inf])


def test_low_fidelity_increases_distance_and_repulsion_penalties():
    high = coupled_scores([1, 2], [1, 3], [0, 1], 1, PARAMETERS)
    low = coupled_scores([1, 2], [1, 3], [0, 1], 0, PARAMETERS)
    assert np.argmax(high) == 1
    assert np.argmax(low) == 0


def test_repulsion_is_bfs_based_radius_limited_and_missing_goal_zero():
    shared = np.zeros((1, 8), dtype=np.uint8)
    positions = [bfs_distances(shared, (0, 0)), bfs_distances(shared, (0, 3))]
    goals = [bfs_distances(shared, None), bfs_distances(shared, None)]
    result = repulsion_scores([(0, 2), (0, 7)], 0, positions, goals, PARAMETERS)
    assert result[0] == pytest.approx(np.exp(-1))
    assert result[1] == 0
    assert repulsion_scores([(0, 2)], 0, positions[:1], goals[:1], PARAMETERS)[0] == 0


def test_partition_unique_targets_unreachable_excluded_and_snapshot_not_mutated():
    shared = np.array([[0, 0, 0, 1, 0]], dtype=np.uint8)
    previous = [None, None]
    targets, audit = assign_frontiers(shared, [(0, 0), (0, 2)], previous,
                                     [(0, 1), (0, 2), (0, 4)], [1, 1, 1], [0.5, 0.5], PARAMETERS)
    assert targets == [(0, 1), (0, 2)]
    assert previous == [None, None]
    assert all((0, 4) not in row["candidates"] for row in audit)


def test_hysteresis_requires_consecutive_threshold_satisfaction():
    gate = HysteresisGate(0.3, 0.7, 2, False)
    assert not gate.update(0.8)
    assert not gate.update(0.5)
    assert not gate.update(0.8)
    assert gate.update(0.8)
    assert gate.update(0.5)
    assert gate.update(0.2)
    assert not gate.update(0.2)


def test_infeasible_planner_forces_reactive_branch():
    gate = HysteresisGate(0.3, 0.7, 1, True)
    assert not gate.update(1.0, planner_feasible=False)


def test_sigmoid_is_finite_at_extreme_logits():
    values = sigmoid([-1e6, 0, 1e6])
    assert np.array_equal(values, [0, 0.5, 1])


def test_frozen_gate_rejects_updates_without_mutating_weights():
    model = FidelityModel(np.ones((2, 8)), [0, 0])
    before = model.weights.copy()
    with pytest.raises(RuntimeError):
        model.update(np.zeros((2, 8)), [1, 1], learning_rate=0.1, l2=0.01, margin=0.1)
    assert np.array_equal(before, model.weights)


def test_update_matches_paper_gradient_and_margin():
    model = FidelityModel(np.zeros((2, 8)), [0, 0], frozen=False)
    selected = model.update(np.ones((2, 8)), [1, -0.05], learning_rate=0.2, l2=0, margin=0.1)
    assert np.array_equal(selected, [True, False])
    assert np.allclose(model.weights[0], 0.1)
    assert np.allclose(model.weights[1], 0)
    assert np.allclose(model.bias, [0.1, 0])
    assert model.update_count == 1


def test_quality_uses_all_four_terms():
    assert quality_score(3, 2, 1, 1, weights=[1, 2, 3, 4]) == 0


def test_exactly_eight_features_are_required():
    with pytest.raises(ValueError):
        FidelityModel(np.zeros((2, 7)), [0, 0])
