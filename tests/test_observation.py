import numpy as np

from vorl.grid import DistanceOracle, astar_path, bfs_distances
from vorl.observation import LocalMemory, feasible_actions, feature_vectors, window


def test_accelerated_distances_equal_reference_bfs():
    for seed in range(5):
        shared = (np.random.default_rng(seed).random((12, 12)) < 0.3).astype(np.uint8)
        starts = [(0, 0), (4, 5), None]
        oracle = DistanceOracle(shared)
        for pos, actual in zip(starts, oracle.batch(starts)):
            assert np.array_equal(actual, bfs_distances(shared, pos))


def test_astar_uses_known_space_and_shortest_detour():
    shared = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    assert len(astar_path(shared, (0, 0), (0, 2))) == 7
    shared[2, 1] = 255
    assert astar_path(shared, (0, 0), (0, 2)) is None


def test_window_padding_is_explicit_and_sensor_feasible_mask_is_local():
    shared = np.zeros((5, 5), dtype=np.uint8)
    assert window(shared, (0, 0), 1, outside=1).sum() == 5
    mask = feasible_actions(shared, (0, 0), [(0, 1)])
    assert np.array_equal(mask, [True, False, True, False, False])


def test_private_memory_does_not_use_other_robot_sensing():
    memory = LocalMemory((20, 20), [(5, 5), (15, 15)])
    sensor = np.zeros((7, 7), dtype=np.uint8)
    sensor[3, 4] = 1
    memory.observe(0, (5, 5), sensor, 3)
    assert memory.memories[0, 5, 6] == 1
    assert memory.memories[1, 5, 6] == 0


def test_feature_dimensions_and_missing_goal_are_finite():
    shared = np.zeros((7, 7), dtype=np.uint8)
    positions = [(3, 3)]
    features = feature_vectors(shared, positions, [None], [bfs_distances(shared, positions[0])],
                               [np.ones(5, dtype=bool)], [False], [[(3, 3)]],
                               sensing_radius=3, interaction_radius=3, stuck_window=3)
    assert features.shape == (1, 8)
    assert np.isfinite(features).all()
    assert features[0, 2] == 1 and features[0, 5] == 0
