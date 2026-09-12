"""Explicit observation and feature conventions for the method reconstruction."""
import numpy as np

from .grid import FREE, UNKNOWN, MOVES, neighbors


def window(array, center, radius, *, outside):
    width = radius * 2 + 1
    result = np.full((width, width), outside, dtype=array.dtype)
    r, c = center
    r0, r1 = max(0, r - radius), min(array.shape[0], r + radius + 1)
    c0, c1 = max(0, c - radius), min(array.shape[1], c + radius + 1)
    result[r0 - r + radius:r1 - r + radius, c0 - c + radius:c1 - c + radius] = array[r0:r1, c0:c1]
    return result


def feasible_actions(shared_map, position, other_positions):
    """Five-action mask including no-op; no motion into occupied-at-start cells."""
    occupied = set(map(tuple, other_positions))
    result = []
    for dr, dc in MOVES:
        target = position[0] + dr, position[1] + dc
        in_bounds = 0 <= target[0] < shared_map.shape[0] and 0 <= target[1] < shared_map.shape[1]
        result.append(in_bounds and shared_map[target] == FREE and target not in occupied)
    result[0] = True
    return np.asarray(result, dtype=bool)


def unknown_ratio(shared_map, center, radius):
    if center is None:
        return 0.0
    # Out-of-map cells are known boundary constraints, not unknown information.
    return float(np.mean(window(shared_map, center, radius, outside=1) == UNKNOWN))


def feature_vectors(shared_map, positions, goals, distances, masks, planner_ok, history,
                    *, sensing_radius, interaction_radius, stuck_window):
    """Exactly eight statistics in the order declared in fidelity.FEATURE_NAMES.

Reconstruction conventions not numerically specified by the paper: distance is
clipped after division by rows+columns-2; no/unreachable goal has distance one;
stuck means no position change across the configured history window; sensing
neighborhoods are squares (matching the reference), crowding uses BFS radius.
"""
    features = []
    distance_scale = max(1, sum(shared_map.shape) - 2)
    for i, position in enumerate(positions):
        crowd = sum(j != i and distances[i][tuple(other)] <= interaction_radius for j, other in enumerate(positions))
        stuck = len(history[i]) >= stuck_window and len(set(history[i][-stuck_window:])) == 1
        distance = distances[i][goals[i]] if goals[i] is not None else np.inf
        distance = min(1.0, distance / distance_scale) if np.isfinite(distance) else 1.0
        free_neighbors = sum(shared_map[p] == FREE for p in neighbors(position, shared_map.shape))
        features.append([crowd, float(stuck), distance, float(np.mean(masks[i])),
                         unknown_ratio(shared_map, position, sensing_radius),
                         unknown_ratio(shared_map, goals[i], sensing_radius),
                         1 - free_neighbors / 4, float(planner_ok[i])])
    return np.asarray(features, dtype=np.float64)


class LocalMemory:
    """Private EPOM memory, updated only from that robot's local sensor patch."""
    def __init__(self, shape, starts):
        self.starts = np.asarray(starts, dtype=np.int32)
        self.memories = np.zeros((len(starts), *shape), dtype=np.uint8)

    def observe(self, robot_id, position, sensed_obstacles, radius):
        r, c = position
        shape = self.memories.shape[1:]
        r0, r1 = max(0, r - radius), min(shape[0], r + radius + 1)
        c0, c1 = max(0, c - radius), min(shape[1], c + radius + 1)
        self.memories[robot_id, r0:r1, c0:c1] = sensed_obstacles[
            r0 - r + radius:r1 - r + radius, c0 - c + radius:c1 - c + radius]

    def policy_inputs(self, positions, goals, *, memory_radius, sensing_radius):
        width = memory_radius * 2 + 1
        grids = np.zeros((len(positions), 3, width, width), dtype=np.float32)
        positions = np.asarray(positions, dtype=np.int32)
        relative = positions - self.starts
        targets = []
        for i, position in enumerate(positions):
            grids[i, 0] = window(self.memories[i], position, memory_radius, outside=1)
            for j, other in enumerate(positions):
                delta = other - position
                if max(abs(delta)) <= min(sensing_radius, memory_radius):
                    grids[i, 1, memory_radius + delta[0], memory_radius + delta[1]] = 1
            goal = np.asarray(goals[i] if goals[i] is not None else position, dtype=np.int32)
            targets.append(goal - self.starts[i])
            delta = np.clip(goal - position, -memory_radius, memory_radius)
            grids[i, 2, memory_radius + delta[0], memory_radius + delta[1]] = 1
        return grids, relative.astype(np.float32), np.asarray(targets, dtype=np.float32)
