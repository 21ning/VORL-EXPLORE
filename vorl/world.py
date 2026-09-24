"""Seeded dynamic grid simulator for method-level reconstruction.

This is an explicit NumPy simulator, not a claim of bitwise Pogema parity.
All policy access goes through SensorState; ground truth remains here and in
the offline trajectory recorder. Obstacle motion occurs once per two ticks;
each assigned trip is capped at five grid moves.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from .grid import MOVES, UNKNOWN, astar_path
from .observation import window


@dataclass
class SensorState:
    shared_map: np.ndarray
    positions: list
    patches: list
    newly_observed: np.ndarray


def resolve_motion(obstacles, positions, actions):
    positions = list(map(tuple, positions))
    if len(actions) != len(positions) or any(int(a) != a or not 0 <= a < 5 for a in actions):
        raise ValueError("One discrete action in [0, 4] is required per robot")
    proposed = []
    interventions = np.zeros(len(positions), dtype=np.int32)
    for i, (position, action) in enumerate(zip(positions, actions)):
        dr, dc = MOVES[int(action)]
        target = position[0] + dr, position[1] + dc
        valid = (0 <= target[0] < obstacles.shape[0] and 0 <= target[1] < obstacles.shape[1]
                 and obstacles[target] == 0)
        proposed.append(target if valid else position)
        interventions[i] += int(not valid and action != 0)
    # Cancel conflicting moves to a fixed point, including conflicts caused by
    # a previously cancelled move. This also handles chains into stationary cells.
    while True:
        cancelled = set()
        for i in range(len(positions)):
            for j in range(i):
                same = proposed[i] == proposed[j]
                swap = proposed[i] == positions[j] and proposed[j] == positions[i]
                if same or swap:
                    cancelled.update((i, j))
        moving_cancelled = [i for i in cancelled if proposed[i] != positions[i]]
        if not moving_cancelled:
            break
        for i in moving_cancelled:
            proposed[i] = positions[i]
            interventions[i] += 1
    if len(set(proposed)) != len(proposed):
        raise AssertionError("Collision resolver left overlapping robots")
    return proposed, interventions


class GridWorld:
    def __init__(self, *, size, robots, dynamic_obstacles, density, seed, sensing_radius=3):
        if size < 5 or robots < 1 or dynamic_obstacles < 0 or not 0 <= density < 1:
            raise ValueError("Invalid grid parameters")
        self.rng = np.random.default_rng(seed)
        self.seed, self.radius, self.time = seed, sensing_radius, 0
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        for attempt in range(100):
            self.static_map = (self.rng.random((size, size)) < density).astype(np.uint8)
            components, _ = label(self.static_map == 0, structure=structure)
            counts = np.bincount(components.ravel())
            counts[0] = 0
            largest = int(np.argmax(counts))
            self.free_component = np.argwhere(components == largest)
            if len(self.free_component) >= robots + dynamic_obstacles + 1:
                break
        else:
            raise ValueError("Cannot sample a component large enough for this team and traffic")
        chosen = self.rng.choice(len(self.free_component), robots + dynamic_obstacles, replace=False)
        self.positions = list(map(tuple, self.free_component[chosen[:robots]]))
        self.dynamic_positions = list(map(tuple, self.free_component[chosen[robots:]]))
        self.dynamic_paths = [[] for _ in self.dynamic_positions]
        self.dynamic_active = [False] * dynamic_obstacles
        self.dynamic_replacement_pending = [False] * dynamic_obstacles
        self.dynamic_waits = [0] * dynamic_obstacles
        self.shared_map = np.full((size, size), UNKNOWN, dtype=np.uint8)
        self.seen = np.zeros((robots, size, size), dtype=bool)
        self.visited = np.zeros_like(self.seen)
        for i, position in enumerate(self.positions):
            self.visited[i][position] = True
        self.map_attempts = attempt + 1
        self.environment_interventions = 0
        self.total_moved_steps = 0

    def obstacles(self):
        result = self.static_map.copy()
        for position in self.dynamic_positions:
            result[position] = 1
        return result

    def observe(self):
        obstacles = self.obstacles()
        patches = []
        new_cells = []
        for i, (r, c) in enumerate(self.positions):
            rows = slice(max(0, r - self.radius), min(obstacles.shape[0], r + self.radius + 1))
            cols = slice(max(0, c - self.radius), min(obstacles.shape[1], c + self.radius + 1))
            self.shared_map[rows, cols] = obstacles[rows, cols]
            new_cells.append(int(np.count_nonzero(~self.seen[i, rows, cols])))
            self.seen[i, rows, cols] = True
            patches.append(window(obstacles, (r, c), self.radius, outside=1))
        return SensorState(self.shared_map.copy(), self.positions.copy(), patches, np.asarray(new_cells))

    def _move_obstacles(self):
        occupied = set(self.dynamic_positions) | set(self.positions)
        for i, position in enumerate(self.dynamic_positions):
            if not self.dynamic_active[i]:
                if self.dynamic_replacement_pending[i]:
                    self._replace_arrived_obstacle(i, occupied)
                else:
                    self.dynamic_paths[i] = self._sample_short_path(position, occupied)
                    self.dynamic_active[i] = len(self.dynamic_paths[i]) > 1
                self.dynamic_waits[i] = 0
                position = self.dynamic_positions[i]
            if not self.dynamic_active[i]:
                continue
            if self.dynamic_waits[i] >= 10:
                self.dynamic_paths[i] = self._sample_short_path(position, occupied)
                self.dynamic_active[i] = len(self.dynamic_paths[i]) > 1
                self.dynamic_waits[i] = 0
            path = self.dynamic_paths[i]
            nxt = path[1] if len(path) > 1 else position
            if nxt != position and nxt not in occupied:
                occupied.remove(position)
                occupied.add(nxt)
                self.dynamic_positions[i] = nxt
                self.dynamic_paths[i] = path[1:]
                self.dynamic_waits[i] = 0
                if len(self.dynamic_paths[i]) <= 1:
                    self.dynamic_active[i] = False
                    self.dynamic_replacement_pending[i] = True
                    self.static_map[nxt] = 1
            else:
                self.dynamic_waits[i] += 1

    def _sample_short_path(self, position, occupied):
        """Choose a random unoccupied reachable target within five moves."""
        free_cells = np.argwhere(self.static_map == 0)
        for index in self.rng.permutation(len(free_cells)):
            goal = tuple(free_cells[index])
            if goal == position or goal in occupied:
                continue
            path = astar_path(self.static_map, position, goal)
            if path is not None and 1 < len(path) <= 6:
                return path
        return [position]

    def _replace_arrived_obstacle(self, index, occupied):
        """Turn a random eligible ordinary obstacle into the next mover."""
        arrived = self.dynamic_positions[index]
        occupied.remove(arrived)
        candidates = np.argwhere(self.static_map == 1)
        for candidate_index in self.rng.permutation(len(candidates)):
            source = tuple(candidates[candidate_index])
            if source == arrived or source in occupied:
                continue
            self.static_map[source] = 0
            path = self._sample_short_path(source, occupied)
            if len(path) > 1:
                self.dynamic_positions[index] = source
                self.dynamic_paths[index] = path
                self.dynamic_active[index] = True
                self.dynamic_replacement_pending[index] = False
                self.dynamic_waits[index] = 0
                occupied.add(source)
                return
            self.static_map[source] = 1
        occupied.add(arrived)

    def step(self, actions):
        previous = self.positions.copy()
        self.positions, interventions = resolve_motion(self.obstacles(), self.positions, actions)
        self.environment_interventions += int(interventions.sum())
        self.total_moved_steps += sum(a != b for a, b in zip(previous, self.positions))
        self.time += 1
        if self.time % 2 == 0:
            self._move_obstacles()
        for i, position in enumerate(self.positions):
            self.visited[i][position] = True
        assert len(set(self.positions + self.dynamic_positions)) == len(self.positions) + len(self.dynamic_positions)
        assert all(self.static_map[p] == 0 for p in self.positions)
        assert all(self.static_map[p] == 0 for p, active in zip(self.dynamic_positions, self.dynamic_active) if active)
        assert all(self.static_map[p] == 1 for p, active, pending in zip(
            self.dynamic_positions, self.dynamic_active, self.dynamic_replacement_pending
        ) if not active and pending)
        return self.observe(), interventions

    def metrics(self):
        multiplicity = self.visited.sum(axis=0)
        return {
            "steps": self.time, "total_moved_steps": self.total_moved_steps,
            "observed_cell_fraction": float(np.mean(self.shared_map != UNKNOWN)),
            "visited_overlap": float(np.count_nonzero(multiplicity >= 2) / max(1, np.count_nonzero(multiplicity >= 1))),
            "environment_safety_interventions": self.environment_interventions,
            "sampled_static_density": float(self.static_map.mean()), "map_sampling_attempts": self.map_attempts,
        }
