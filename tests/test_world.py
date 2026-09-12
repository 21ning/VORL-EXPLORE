import numpy as np

from vorl.world import GridWorld, resolve_motion


def test_collision_cancellation_propagates_and_prevents_swaps():
    world = np.zeros((5, 5), dtype=np.uint8)
    positions = [(1, 1), (1, 2), (1, 3)]
    after, _ = resolve_motion(world, positions, [4, 4, 0])
    assert after == positions
    after, _ = resolve_motion(world, positions[:2], [4, 3])
    assert after == positions[:2]


def test_seeded_map_and_dynamic_obstacle_speed():
    kwargs = dict(size=12, robots=3, dynamic_obstacles=4, density=0.2, seed=17)
    first, repeated = GridWorld(**kwargs), GridWorld(**kwargs)
    assert np.array_equal(first.static_map, repeated.static_map)
    assert first.positions == repeated.positions
    for step in range(10):
        old = first.dynamic_positions.copy()
        first.step([0, 0, 0])
        repeated.step([0, 0, 0])
        assert first.dynamic_positions == repeated.dynamic_positions
        displacement = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(old, first.dynamic_positions)]
        assert max(displacement) <= (0 if step % 2 == 0 else 1)


def test_sensor_state_has_no_ground_truth_field_and_unknown_is_preserved():
    world = GridWorld(size=40, robots=1, dynamic_obstacles=4, density=0.3, seed=7)
    state = world.observe()
    assert set(vars(state)) == {"shared_map", "positions", "patches", "newly_observed"}
    assert np.count_nonzero(state.shared_map != 255) <= 49
    assert state.patches[0].shape == (7, 7)
