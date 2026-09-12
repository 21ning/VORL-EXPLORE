"""Eight-feature logistic model, hysteresis and adaptation equations (7)--(16).

This does not ship invented learned parameters or pretend to be a fitted gate.
Training/evaluation orchestration must provide its own traceable checkpoint.
"""
from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = (
    "nearby_teammates", "stuck", "normalized_goal_distance", "feasible_action_ratio",
    "unknown_ratio_robot", "unknown_ratio_goal", "four_neighbor_blockage", "planner_feasible",
)


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -value))


class FidelityModel:
    def __init__(self, weights, bias, *, frozen=True):
        self.weights = np.array(weights, dtype=np.float64, copy=True)
        self.bias = np.array(bias, dtype=np.float64, copy=True)
        if self.weights.ndim != 2 or self.weights.shape[1] != len(FEATURE_NAMES):
            raise ValueError("Expected one eight-feature weight vector per robot")
        if self.bias.shape != (len(self.weights),):
            raise ValueError("Expected one bias per robot")
        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self.bias)):
            raise ValueError("Gate checkpoint contains nonfinite parameters")
        self.frozen = bool(frozen)
        self.update_count = 0

    def predict(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.shape != self.weights.shape or not np.all(np.isfinite(features)):
            raise ValueError("Expected a finite eight-feature vector for every robot")
        return sigmoid(np.einsum("ij,ij->i", self.weights, features) + self.bias)

    def update(self, features, quality, *, learning_rate, l2, margin):
        if self.frozen:
            raise RuntimeError("Evaluation gate is frozen; online adaptation is disabled")
        if not all(np.isfinite(v) for v in (learning_rate, l2, margin)) or learning_rate <= 0 or l2 < 0 or margin <= 0:
            raise ValueError("Require positive learning rate/margin and nonnegative L2")
        probabilities = self.predict(features)
        features = np.asarray(features, dtype=np.float64)
        quality = np.asarray(quality, dtype=np.float64)
        if quality.shape != self.bias.shape or not np.all(np.isfinite(quality)):
            raise ValueError("Expected a finite quality score for every robot")
        selected = np.abs(quality) >= margin
        residual = probabilities - (quality >= 0).astype(np.float64)
        self.weights[selected] -= learning_rate * (residual[selected, None] * features[selected] + l2 * self.weights[selected])
        self.bias[selected] -= learning_rate * residual[selected]
        self.update_count += int(selected.sum())
        return selected


@dataclass
class HysteresisGate:
    low_threshold: float
    high_threshold: float
    dwell: int
    planner_selected: bool
    consecutive_high: int = 0
    consecutive_low: int = 0

    def __post_init__(self):
        if not 0 <= self.low_threshold < self.high_threshold <= 1:
            raise ValueError("Require 0 <= low < high <= 1")
        if isinstance(self.dwell, bool) or not isinstance(self.dwell, int) or self.dwell < 1:
            raise ValueError("Dwell must be a positive integer")

    def update(self, fidelity, *, planner_feasible=True):
        if not np.isfinite(fidelity) or not 0 <= fidelity <= 1:
            raise ValueError("Fidelity must be in [0, 1]")
        self.consecutive_high = self.consecutive_high + 1 if fidelity >= self.high_threshold else 0
        self.consecutive_low = self.consecutive_low + 1 if fidelity <= self.low_threshold else 0
        if not self.planner_selected and self.consecutive_high >= self.dwell:
            self.planner_selected = True
        elif self.planner_selected and self.consecutive_low >= self.dwell:
            self.planner_selected = False
        if not planner_feasible:
            self.planner_selected = False
        return self.planner_selected


def quality_score(coverage_gain, bfs_distance_gain, risk, stall, *, weights):
    """Caller computes all window statistics from actual execution, not proposals.

The distance gain must compare positions against the same goal on the same
current shared map; reassignment must not be mistaken for physical progress.
"""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (4,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Require four positive quality coefficients")
    components = np.asarray([coverage_gain, bfs_distance_gain, -risk, -stall], dtype=np.float64)
    if not np.all(np.isfinite(components)):
        raise ValueError("Quality components must be finite")
    return np.tensordot(weights, components, axes=(0, 0))
