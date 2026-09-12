"""Fidelity-coupled Voronoi objective, paper equations (3)--(6).

Numerical settings are deliberately required: the paper does not specify them.
The caller supplies utility counts using its declared sensing neighborhood.
This module follows the robot-specific partition in IV-A rather than the
unfiltered candidate set appearing in line 7 of the paper pseudocode.
"""
from dataclasses import dataclass

import numpy as np

from .grid import bfs_distances, minmax


@dataclass(frozen=True)
class AssignmentParameters:
    lambda0: float
    lambda1: float
    rho0: float
    rho1: float
    interaction_radius: float
    sigma_position: float
    sigma_goal: float
    beta: float

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(np.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Assignment settings must be finite and nonnegative")
        if self.sigma_position <= 0 or self.sigma_goal <= 0:
            raise ValueError("Repulsion scales must be positive")


def coupled_scores(utility, distance, repulsion, fidelity, parameters):
    if not np.isfinite(fidelity) or not 0 <= fidelity <= 1:
        raise ValueError("Fidelity must be in [0, 1]")
    if not (len(utility) == len(distance) == len(repulsion)):
        raise ValueError("Score vectors must refer to the same candidate set")
    distance_weight = parameters.lambda0 + parameters.lambda1 * (1 - fidelity)
    repulsion_weight = parameters.rho0 + parameters.rho1 * (1 - fidelity)
    return minmax(utility) - distance_weight * minmax(distance) - repulsion_weight * minmax(repulsion)


def repulsion_scores(candidates, robot_id, position_distances, goal_distances, parameters):
    n = len(position_distances)
    if len(goal_distances) != n or not 0 <= robot_id < n:
        raise ValueError("Require one position and prior-goal distance map per robot")
    result = np.zeros(len(candidates), dtype=np.float64)
    if n == 1:
        return result
    for j in range(n):
        if j == robot_id:
            continue
        for k, candidate in enumerate(candidates):
            dx = position_distances[j][candidate]
            dg = goal_distances[j][candidate]
            if dx <= parameters.interaction_radius:
                result[k] += np.exp(-dx / parameters.sigma_position) / (n - 1)
            if dg <= parameters.interaction_radius:
                result[k] += parameters.beta * np.exp(-dg / parameters.sigma_goal) / (n - 1)
    return result


def assign_frontiers(shared_map, positions, previous_goals, candidates, utility, fidelities, parameters, *, oracle=None):
    """Return targets and auditable per-robot score data for one reassignment round.

Ties in partition ownership and final scores are resolved by input order. All
robots use the same previous-goal snapshot; no sequential goal leakage occurs.
Robots without owned candidates receive None, not a fabricated frontier.
"""
    n = len(positions)
    if not n or len(previous_goals) != n or len(fidelities) != n:
        raise ValueError("Positions, previous goals and fidelity lengths must agree")
    candidates = [tuple(p) for p in candidates]
    if len(set(candidates)) != len(candidates) or len(utility) != len(candidates):
        raise ValueError("Provide unique frontier candidates and matching utilities")
    if any(shared_map[p] != 0 for p in candidates):
        raise ValueError("Candidates must be known free cells")
    position_distances = oracle.batch(positions) if oracle is not None else [bfs_distances(shared_map, p) for p in positions]
    goal_distances = oracle.batch(previous_goals) if oracle is not None else [bfs_distances(shared_map, g) for g in previous_goals]
    owned = [[] for _ in positions]
    for k, candidate in enumerate(candidates):
        distances = [d[candidate] for d in position_distances]
        owner = int(np.argmin(distances))
        if np.isfinite(distances[owner]):
            owned[owner].append(k)
    targets, audit = [], []
    for i in range(n):
        indices = owned[i]
        local = [candidates[k] for k in indices]
        distances = [position_distances[i][f] for f in local]
        repulsion = repulsion_scores(local, i, position_distances, goal_distances, parameters)
        scores = coupled_scores([utility[k] for k in indices], distances, repulsion, fidelities[i], parameters)
        targets.append(local[int(np.argmax(scores))] if local else None)
        audit.append({"candidates": local, "utility": [int(utility[k]) for k in indices], "distance": distances, "repulsion": repulsion.tolist(),
                      "score": scores.tolist(), "fidelity": float(fidelities[i])})
    return targets, audit
