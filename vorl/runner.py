"""Authentic dynamic rollouts and delayed self-supervised gate data."""
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .controller import Controller
from .fidelity import FEATURE_NAMES, quality_score
from .grid import DistanceOracle, frontiers
from .policy import EpomPolicy
from .world import GridWorld


def canonical_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hash():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_episode(*, config, policy_dir, gate, seed, size, robots, dynamic_obstacles, density,
                horizon, output, threads=4, collect=False, adapt=False):
    output = Path(output)
    if not collect and gate.get("status") != "fitted_from_self_supervised_rollouts":
        raise ValueError("Evaluation requires a fitted, traceable gate checkpoint")
    if list(gate.get("feature_names", FEATURE_NAMES)) != list(FEATURE_NAMES):
        raise ValueError("Gate feature order mismatch")
    if not collect and gate.get("config_sha256") != canonical_hash(config):
        raise ValueError("Gate training/evaluation configuration differs")
    if not collect and seed in gate.get("training_seeds", []):
        raise ValueError("Evaluation seed overlaps warm-start training seeds")
    output.mkdir(parents=True, exist_ok=False)
    running_source_hash = source_hash()
    policy_dir = Path(policy_dir)
    checkpoints = list(policy_dir.glob("checkpoint_*.pth"))
    if len(checkpoints) != 1:
        raise ValueError("Require exactly one explicit EPOM policy checkpoint")
    policy = EpomPolicy(policy_dir / "cfg.json", checkpoints[0], seed=seed, threads=threads)
    world = GridWorld(size=size, robots=robots, dynamic_obstacles=dynamic_obstacles, density=density,
                      seed=seed, sensing_radius=config["sensing_radius"])
    sensor = world.observe()
    controller = Controller(sensor, policy, config, gate, seed=seed, adapt=adapt)
    initial_gate_weights, initial_gate_bias = controller.model.weights.copy(), controller.model.bias.copy()
    records = {name: [] for name in ("static", "known", "positions", "dynamic", "dynamic_active", "goals", "fidelity", "modes", "actions", "features", "planner_selected", "planner_feasible")}
    delayed = deque(maxlen=config["history_window"])
    training_features, training_quality = [], []
    start = time.monotonic()
    outcome = "horizon"
    oracle = DistanceOracle(sensor.shared_map)
    with (output / "progress.jsonl").open("w", encoding="utf-8") as progress, (output / "allocation.jsonl").open("w", encoding="utf-8") as allocation:
        for step in range(horizon + 1):
            # Record the state reached by previous actions before deciding another.
            records["static"].append(world.static_map.copy())
            records["known"].append(sensor.shared_map.copy())
            records["positions"].append(sensor.positions.copy())
            records["dynamic"].append(world.dynamic_positions.copy())
            records["dynamic_active"].append(world.dynamic_active.copy())
            if not frontiers(sensor.shared_map):
                outcome = "no_frontiers"
                break
            if step == horizon:
                break
            decision = controller.decide(sensor, oracle, step)
            if decision["assignment"] is not None and not collect:
                row = {"step": step, "previous_goals": decision["previous_goals"], "refreshed": decision["refreshed"],
                       "audit": decision["assignment"]}
                allocation.write(json.dumps(row, default=lambda value: value.item()) + "\n")
            for name in ("fidelity", "modes", "actions", "features", "planner_selected", "planner_feasible"):
                records[name].append(decision[name])
            records["goals"].append([goal if goal is not None else (-1, -1) for goal in decision["goals"]])
            next_sensor, env_interventions = world.step(decision["actions"])
            next_oracle = DistanceOracle(next_sensor.shared_map)
            goal_distances = next_oracle.batch(decision["goals"])
            distance_gain = np.zeros(robots)
            for i, (old, new, distances) in enumerate(zip(sensor.positions, next_sensor.positions, goal_distances)):
                if np.isfinite(distances[old]) and np.isfinite(distances[new]):
                    distance_gain[i] = distances[old] - distances[new]
            stalled = np.asarray([a == b for a, b in zip(sensor.positions, next_sensor.positions)], dtype=float)
            delayed.append({"features": decision["features"].copy(), "coverage": next_sensor.newly_observed,
                            "distance": distance_gain, "risk": env_interventions + decision["filter_events"], "stall": stalled})
            if len(delayed) == config["history_window"]:
                quality = quality_score(sum(row["coverage"] for row in delayed), sum(row["distance"] for row in delayed),
                                        sum(row["risk"] for row in delayed), np.mean([row["stall"] for row in delayed], axis=0),
                                        weights=config["quality_weights"])
                delayed_features = delayed[0]["features"]
                if collect:
                    training_features.append(delayed_features.copy())
                    training_quality.append(quality.copy())
                if adapt and (step + 1) % config["update_interval"] == 0:
                    controller.model.update(delayed_features, quality, learning_rate=config["gate_learning_rate"],
                                            l2=config["gate_l2"], margin=config["quality_margin"])
            sensor, oracle = next_sensor, next_oracle
            if (step + 1) % 32 == 0:
                row = {"step": step + 1, "wall_seconds": round(time.monotonic() - start, 2), **world.metrics(), **controller.counts}
                progress.write(json.dumps(row) + "\n")
                progress.flush()
    # Decision fields are indexed by transitions, state fields by states: T vs T+1.
    arrays = {
        "static_map": records["static"][0],
        "static": np.asarray(records["static"], dtype=np.uint8),
        "known": np.asarray(records["known"], dtype=np.uint8),
        "positions": np.asarray(records["positions"], dtype=np.int16),
        "dynamic": np.asarray(records["dynamic"], dtype=np.int16).reshape(len(records["dynamic"]), dynamic_obstacles, 2),
        "dynamic_active": np.asarray(records["dynamic_active"], dtype=bool).reshape(len(records["dynamic_active"]), dynamic_obstacles),
        "goals": np.asarray(records["goals"], dtype=np.int16).reshape(-1, robots, 2),
        "fidelity": np.asarray(records["fidelity"], dtype=np.float32).reshape(-1, robots),
        "modes": np.asarray(records["modes"], dtype=np.uint8).reshape(-1, robots),
        "actions": np.asarray(records["actions"], dtype=np.uint8).reshape(-1, robots),
        "features": np.asarray(records["features"], dtype=np.float32).reshape(-1, robots, 8),
        "planner_selected": np.asarray(records["planner_selected"], dtype=bool).reshape(-1, robots),
        "planner_feasible": np.asarray(records["planner_feasible"], dtype=bool).reshape(-1, robots),
    }
    np.savez_compressed(output / "trajectory.npz", **arrays)
    if collect:
        np.savez_compressed(output / "gate-data.npz", features=np.asarray(training_features).reshape(-1, 8),
                            quality=np.asarray(training_quality).reshape(-1), seed=np.asarray(seed))
    gate_unchanged = np.array_equal(initial_gate_weights, controller.model.weights) and np.array_equal(initial_gate_bias, controller.model.bias)
    if not adapt and not gate_unchanged:
        raise AssertionError("Frozen evaluation gate changed")
    summary = {
        "profile": config["profile"], "phase": "warmstart_collection" if collect else "method_evaluation",
        "paper_original_weights_reproduced": False, "online_adaptation": adapt,
        "gate_unchanged": gate_unchanged, "gate_updates": controller.model.update_count,
        "seed": seed, "size": size, "robots": robots, "dynamic_obstacles": dynamic_obstacles,
        "obstacle_speed_ratio": 0.5, "dynamic_trip_max_moves": 5, "horizon": horizon, "stop_reason": outcome,
        "success_no_frontiers": outcome == "no_frontiers", "wall_seconds": time.monotonic() - start,
        **world.metrics(), **controller.counts, "policy": policy.provenance,
        "gate_checkpoint_sha256": canonical_hash(gate), "config_sha256": canonical_hash(config),
        "method_source_sha256": running_source_hash,
        "trajectory_sha256": hashlib.sha256((output / "trajectory.npz").read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return summary
