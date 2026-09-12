"""Shared-map geometry. No function here has access to unobserved ground truth."""
from collections import deque
import heapq

import numpy as np

FREE, OCCUPIED, UNKNOWN = 0, 1, 255
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def neighbors(cell, shape):
    r, c = cell
    for dr, dc in MOVES[1:]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            yield nr, nc


def frontiers(shared_map):
    """Free cells adjacent to unknown space in the four-neighbor graph (III-A)."""
    return [tuple(p) for p in np.argwhere(shared_map == FREE)
            if any(shared_map[q] == UNKNOWN for q in neighbors(p, shared_map.shape))]


def bfs_distances(shared_map, start):
    """Four-connected shortest-path distances on known free cells only."""
    result = np.full(shared_map.shape, np.inf, dtype=np.float64)
    if start is None:
        return result
    start = tuple(start)
    if not (0 <= start[0] < shared_map.shape[0] and 0 <= start[1] < shared_map.shape[1]):
        return result
    if shared_map[start] != FREE:
        return result
    result[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current, shared_map.shape):
            if shared_map[nxt] == FREE and np.isinf(result[nxt]):
                result[nxt] = result[current] + 1
                queue.append(nxt)
    return result


def minmax(values):
    """Normalize a reassignment candidate vector; a constant vector contributes zero."""
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Normalize only reachable, finite candidate scores")
    if values.size == 0:
        return values.copy()
    extent = np.ptp(values)
    return np.zeros_like(values) if extent == 0 else (values - values.min()) / extent


def astar_path(shared_map, start, goal):
    """A* on known free cells, without the reference's fixed expansion cutoff."""
    if goal is None:
        return None
    start, goal = tuple(start), tuple(goal)
    if any(not (0 <= p[0] < shared_map.shape[0] and 0 <= p[1] < shared_map.shape[1]) for p in (start, goal)):
        return None
    if shared_map[start] != FREE or shared_map[goal] != FREE:
        return None
    cost, parent = {start: 0}, {start: None}
    queue = [(0, 0, start)]
    while queue:
        _, distance, cell = heapq.heappop(queue)
        if distance != cost[cell]:
            continue
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return path[::-1]
        for nxt in neighbors(cell, shared_map.shape):
            if shared_map[nxt] == FREE and distance + 1 < cost.get(nxt, np.inf):
                cost[nxt], parent[nxt] = distance + 1, cell
                heuristic = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                heapq.heappush(queue, (distance + 1 + heuristic, distance + 1, nxt))
    return None


class DistanceOracle:
    """C-backed batched BFS-equivalent distances; only the shared map is used."""
    def __init__(self, shared_map):
        from scipy.sparse import csr_matrix
        self.shared_map = np.asarray(shared_map).copy()
        self.shape = self.shared_map.shape
        ids = np.arange(self.shared_map.size).reshape(self.shape)
        free = self.shared_map == FREE
        horizontal = free[:, :-1] & free[:, 1:]
        vertical = free[:-1, :] & free[1:, :]
        a = np.concatenate([ids[:, :-1][horizontal], ids[:-1, :][vertical]])
        b = np.concatenate([ids[:, 1:][horizontal], ids[1:, :][vertical]])
        self.graph = csr_matrix((np.ones(2 * len(a)), (np.concatenate([a, b]), np.concatenate([b, a]))),
                                shape=(self.shared_map.size, self.shared_map.size))
        self.cache = {}
        self.unreachable = np.full(self.shape, np.inf)

    def batch(self, starts):
        from scipy.sparse.csgraph import shortest_path
        starts = [tuple(p) if p is not None else None for p in starts]
        valid = lambda p: p is not None and 0 <= p[0] < self.shape[0] and 0 <= p[1] < self.shape[1] and self.shared_map[p] == FREE
        pending = list(dict.fromkeys(p for p in starts if valid(p) and p not in self.cache))
        if pending:
            indices = [np.ravel_multi_index(p, self.shape) for p in pending]
            distances = shortest_path(self.graph, method="D", directed=False, unweighted=True, indices=indices)
            for p, distance in zip(pending, distances):
                self.cache[p] = distance.reshape(self.shape)
        return [self.cache[p] if valid(p) else self.unreachable for p in starts]
