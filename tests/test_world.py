import numpy as np

from vorl.world import GridWorld, resolve_motion


def test_collision_cancellation_propagates_and_prevents_swaps():
    world = np.zeros((5, 5), dtype=np.uint8)
    positions = [(1, 1), (1, 2), (1, 3)]
    after, _ = resolve_motion(world, positions, [4, 4, 0])
    assert after == positions
    after, _ = resolve_motion(world, positions[:2], [4, 3])
    assert after == positions[:2]


def test_boundary_static_and_vertex_conflicts_are_cancelled():
    obstacles = np.zeros((5, 5), dtype=np.uint8)
    obstacles[2, 1] = 1
    positions = [(0, 0), (1, 1)]
    after, events = resolve_motion(obstacles, positions, [1, 2])
    assert after == positions
    assert np.array_equal(events, [1, 1])
    positions = [(1, 1), (1, 3)]
    after, _ = resolve_motion(obstacles, positions, [4, 3])
    assert after == positions


def test_seeded_map_and_dynamic_obstacle_speed():
    kwargs = dict(size=12, robots=3, dynamic_obstacles=4, density=0.2, seed=17)
    first, repeated = GridWorld(**kwargs), GridWorld(**kwargs)
    assert np.array_equal(first.static_map, repeated.static_map)
    assert first.positions == repeated.positions
    for step in range(10):
        old = first.dynamic_positions.copy()
        old_active = first.dynamic_active.copy()
        first.step([0, 0, 0])
        repeated.step([0, 0, 0])
        assert first.dynamic_positions == repeated.dynamic_positions
        displacement = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(old, first.dynamic_positions)]
        for distance, was_active, is_active in zip(displacement, old_active, first.dynamic_active):
            role_replaced = not was_active and is_active
            assert distance <= (0 if step % 2 == 0 else 1) or role_replaced


def test_dynamic_trips_are_short_and_inactive_on_arrival():
    world = GridWorld(size=12, robots=1, dynamic_obstacles=1, density=0.1, seed=3)
    for _ in range(40):
        world.step([0])
        path = world.dynamic_paths[0]
        assert len(path) <= 6
        if not world.dynamic_active[0]:
            assert len(path) <= 1


def test_arrival_replaces_the_dynamic_entity_with_an_ordinary_obstacle():
    world = GridWorld(size=12, robots=1, dynamic_obstacles=1, density=0.2, seed=3)
    previous = None
    replaced = False
    for _ in range(100):
        world.step([0])
        if world.dynamic_replacement_pending[0]:
            previous = world.dynamic_positions[0]
            assert world.static_map[previous] == 1
            continue
        if previous is not None and world.dynamic_active[0]:
            assert world.dynamic_positions[0] != previous
            assert world.static_map[world.dynamic_positions[0]] == 0
            replaced = True
            break
    assert previous is not None and replaced


def test_sensor_state_has_no_ground_truth_field_and_unknown_is_preserved():
    world = GridWorld(size=40, robots=1, dynamic_obstacles=4, density=0.3, seed=7)
    state = world.observe()
    assert set(vars(state)) == {"shared_map", "positions", "patches", "newly_observed"}
    assert np.count_nonzero(state.shared_map != 255) <= 49
    assert state.patches[0].shape == (7, 7)
