"""Verify state/action consistency, gate values and recorded coupled objectives."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vorl.assignment import AssignmentParameters, coupled_scores, repulsion_scores
from vorl.fidelity import HysteresisGate, sigmoid
from vorl.grid import MOVES, DistanceOracle, astar_path
from vorl.observation import feasible_actions, window
from _common import DEFAULT_GATE


def verify(run, gate_path):
    from vorl.runner import canonical_hash, source_hash
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    gate = json.loads(Path(gate_path).read_text())
    assert summary["phase"] == "method_evaluation"
    assert not summary["online_adaptation"] and summary["gate_unchanged"] and summary["gate_updates"] == 0
    assert canonical_hash(config) == summary["config_sha256"] == gate["config_sha256"]
    assert canonical_hash(gate) == summary["gate_checkpoint_sha256"]
    assert summary["method_source_sha256"] == gate["method_source_sha256"] == source_hash()
    assert summary["seed"] not in gate["training_seeds"]
    assert hashlib.sha256((run / "trajectory.npz").read_bytes()).hexdigest() == summary["trajectory_sha256"]
    with np.load(run / "trajectory.npz", allow_pickle=False) as data:
        positions, dynamic, known, actions = data["positions"], data["dynamic"], data["known"], data["actions"]
        assert len(positions) == len(actions) + 1 == summary["steps"] + 1
        assert np.all(np.abs(np.diff(positions, axis=0)).sum(axis=-1) <= 1)
        assert np.all(np.diff(np.count_nonzero(known != 255, axis=(1, 2))) >= 0)
        for t in range(len(positions)):
            all_positions = np.concatenate([positions[t], dynamic[t]], axis=0)
            assert len(set(map(tuple, all_positions))) == len(all_positions)
            assert all(data["static_map"][tuple(p)] == 0 for p in all_positions)
            if t > 0:
                dynamic_motion = np.abs(dynamic[t] - dynamic[t - 1]).sum(axis=-1)
                assert np.all(dynamic_motion <= (1 if t % 2 == 0 else 0))
                for i, action in enumerate(actions[t - 1]):
                    displacement = tuple(positions[t, i] - positions[t - 1, i])
                    assert displacement in ((0, 0), MOVES[int(action)])
            truth = data["static_map"].copy()
            for p in dynamic[t]:
                truth[tuple(p)] = 1
            radius = config["sensing_radius"]
            for r, c in positions[t]:
                rows, cols = slice(max(0, r - radius), r + radius + 1), slice(max(0, c - radius), c + radius + 1)
                assert np.array_equal(known[t, rows, cols], truth[rows, cols])
        expected_fidelity = sigmoid(data["features"] @ np.asarray(gate["weights"]) + gate["bias"])
        assert np.allclose(expected_fidelity, data["fidelity"], atol=1e-6)
        gate_config = config["gate"]
        switches = [HysteresisGate(gate_config["low"], gate_config["high"], gate_config["dwell"], gate_config["initial_planner"])
                    for _ in range(summary["robots"])]
        state_changes = 0
        for t in range(len(actions)):
            for i, switch in enumerate(switches):
                goal = tuple(data["goals"][t, i])
                goal = None if goal == (-1, -1) else goal
                position = tuple(positions[t, i])
                path = astar_path(known[t], position, goal)
                mask = feasible_actions(known[t], position, np.delete(positions[t], i, axis=0))
                delta = (path[1][0] - position[0], path[1][1] - position[1]) if path and len(path) > 1 else (0, 0)
                feasible = bool(path) and bool(mask[MOVES.index(delta)])
                assert feasible == data["planner_feasible"][t, i]
                previous = switch.planner_selected
                selected = switch.update(float(expected_fidelity[t, i]), planner_feasible=feasible)
                assert selected == data["planner_selected"][t, i]
                state_changes += int(previous != selected)
                if data["modes"][t, i] != 2:
                    assert (data["modes"][t, i] == 0) == selected
        assert state_changes == summary["mode_switches"]
        assert summary["policy"]["strict_state_dict_loaded"]
        executed_modes = {name: int(np.count_nonzero(data["modes"] == i)) for i, name in enumerate(("astar", "rl", "recovery"))}
        recorded_goals = data["goals"].copy()
    rounds = 0
    parameters = AssignmentParameters(**config["assignment"])
    for line in (run / "allocation.jsonl").read_text().splitlines():
        row = json.loads(line)
        step = row["step"]
        oracle = DistanceOracle(known[step])
        pose_distances = oracle.batch(list(map(tuple, positions[step])))
        goal_distances = oracle.batch(row["previous_goals"])
        for i, agent in enumerate(row["audit"]):
            candidates = list(map(tuple, agent["candidates"]))
            expected_utility = [np.count_nonzero(window(known[step], f, config["sensing_radius"], outside=1) == 255) for f in candidates]
            assert np.array_equal(expected_utility, agent["utility"])
            assert np.allclose([pose_distances[i][f] for f in candidates], agent["distance"])
            assert np.allclose(repulsion_scores(candidates, i, pose_distances, goal_distances, parameters), agent["repulsion"])
            expected = coupled_scores(agent["utility"], agent["distance"], agent["repulsion"], agent["fidelity"], parameters)
            assert np.allclose(expected, agent["score"])
            if row["refreshed"][i]:
                target = candidates[int(np.argmax(expected))] if candidates else (-1, -1)
                assert tuple(recorded_goals[step, i]) == target
        rounds += 1
    assert rounds > 0
    report = {"verified": True, "seed": summary["seed"], "state_count": summary["steps"] + 1,
              "coupled_assignment_rounds_checked": rounds, "executed_modes": executed_modes,
              "frozen_gate_formula_matches": True, "sensor_state_matches_local_truth": True,
              "final_switch_states_replayed": True, "final_switch_state_changes": state_changes,
              "planner_feasibility_recomputed": True,
              "state_action_consistency": True, "dynamic_half_speed": True,
              "profile": summary["profile"], "paper_original_weights_reproduced": False}
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gate", default=DEFAULT_GATE, type=Path, help="Fitted gate used for this run")
    args = parser.parse_args()
    verify(args.run, args.gate)
