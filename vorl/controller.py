"""Coupled allocation, genuine EPOM actions, hysteresis and bounded recovery."""
from collections import deque

import numpy as np

from .assignment import AssignmentParameters, assign_frontiers
from .fidelity import FidelityModel, HysteresisGate
from .grid import MOVES, astar_path, frontiers
from .observation import LocalMemory, feasible_actions, feature_vectors, window


class Controller:
    def __init__(self, sensor, policy, config, warm_gate, *, seed, adapt=False):
        self.policy, self.config = policy, config
        self.n = len(sensor.positions)
        self.goals = [None] * self.n
        self.history = [[] for _ in range(self.n)]
        self.memory = LocalMemory(sensor.shared_map.shape, sensor.positions)
        self.model = FidelityModel(np.tile(warm_gate["weights"], (self.n, 1)),
                                   np.full(self.n, warm_gate["bias"]), frozen=not adapt)
        gate = config["gate"]
        self.switches = [HysteresisGate(gate["low"], gate["high"], gate["dwell"], gate["initial_planner"])
                         for _ in range(self.n)]
        self.switch_history = [deque(maxlen=config["history_window"]) for _ in range(self.n)]
        self.recovery_remaining = np.zeros(self.n, dtype=int)
        self.last_recovery = np.full(self.n, -config["recovery_cooldown"], dtype=int)
        self.assignment_parameters = AssignmentParameters(**config["assignment"])
        self.rng = np.random.default_rng(seed)
        self.counts = {"astar_selected": 0, "rl_selected": 0, "recovery_selected": 0,
                       "safety_filtered": 0, "reassignments": 0, "mode_switches": 0}

    def decide(self, sensor, oracle, step):
        shared_map, positions = sensor.shared_map, sensor.positions
        for i, pos in enumerate(positions):
            self.history[i].append(pos)
            self.memory.observe(i, pos, sensor.patches[i], self.config["sensing_radius"])
        masks = [feasible_actions(shared_map, pos, positions[:i] + positions[i + 1:]) for i, pos in enumerate(positions)]
        distances = oracle.batch(positions)
        old_plans = [astar_path(shared_map, p, g) for p, g in zip(positions, self.goals)]
        def plan_action(path, pos):
            if not path:
                return 0
            delta = (path[1][0] - pos[0], path[1][1] - pos[1]) if len(path) > 1 else (0, 0)
            return MOVES.index(delta)
        old_ok = [bool(path) and masks[i][plan_action(path, positions[i])] for i, path in enumerate(old_plans)]
        features = feature_vectors(shared_map, positions, self.goals, distances, masks,
                                   old_ok, self.history,
                                   sensing_radius=self.config["sensing_radius"],
                                   interaction_radius=self.config["interaction_radius"],
                                   stuck_window=self.config["history_window"])
        fidelity = self.model.predict(features)
        previous_switch_states = [gate.planner_selected for gate in self.switches]
        for i, gate in enumerate(self.switches):
            gate.update(fidelity[i])
        candidates = frontiers(shared_map)
        previous_goals = self.goals.copy()
        candidate_set = set(candidates)
        refresh = [step % self.config["assignment_interval"] == 0 or goal is None or pos == goal
                   or goal not in candidate_set or self.recovery_remaining[i] > 0
                   for i, (pos, goal) in enumerate(zip(positions, self.goals))]
        if any(refresh):
            utilities = [int(np.count_nonzero(window(shared_map, f, self.config["sensing_radius"], outside=1) == 255))
                         for f in candidates]
            proposed, assignment_audit = assign_frontiers(shared_map, positions, self.goals.copy(), candidates,
                                                          utilities, fidelity, self.assignment_parameters, oracle=oracle)
            for i in range(self.n):
                if refresh[i]:
                    self.goals[i] = proposed[i]
                    self.counts["reassignments"] += 1
        else:
            assignment_audit = None
        plans = [astar_path(shared_map, p, g) for p, g in zip(positions, self.goals)]
        grids, coordinates, targets = self.memory.policy_inputs(positions, self.goals,
                                                                memory_radius=self.policy.radius,
                                                                sensing_radius=self.config["sensing_radius"])
        reactive, _ = self.policy.act(grids, coordinates, targets)
        actions, modes, planner_feasible = [], [], []
        filter_events = np.zeros(self.n, dtype=np.int32)
        for i, (pos, path, gate) in enumerate(zip(positions, plans, self.switches)):
            planner_action = plan_action(path, pos)
            valid_plan = bool(path) and masks[i][planner_action]
            planner_feasible.append(bool(valid_plan))
            if not valid_plan:
                gate.planner_selected = False
            # Feasibility is part of the final gate state at this decision step.
            # A rejected A* attempt must not create a fictitious RL -> A* switch.
            changed = previous_switch_states[i] != gate.planner_selected
            self.switch_history[i].append(int(changed))
            self.counts["mode_switches"] += int(changed)
            if gate.planner_selected:
                action, mode = planner_action, 0
            else:
                action, mode = int(reactive[i]), 1
            self.counts["astar_selected" if mode == 0 else "rl_selected"] += 1
            oscillating = sum(self.switch_history[i]) >= self.config["oscillation_switches"]
            stalled = bool(features[i, 1])
            can_recover = step - self.last_recovery[i] >= self.config["recovery_cooldown"]
            if can_recover and (not valid_plan or stalled or oscillating):
                self.recovery_remaining[i] = self.config["recovery_steps"]
                self.last_recovery[i] = step
            if self.recovery_remaining[i] > 0:
                # Use the real reactive proposal where feasible; on a stall or
                # invalid proposal choose a seeded feasible symmetry-breaking move.
                action = int(reactive[i])
                moving = np.flatnonzero(masks[i][1:]) + 1
                if stalled or not masks[i][action] or action == 0:
                    action = int(self.rng.choice(moving)) if len(moving) else 0
                self.recovery_remaining[i] -= 1
                mode = 2
                self.counts["recovery_selected"] += 1
            if not masks[i][action]:
                action = 0
                filter_events[i] = 1
                self.counts["safety_filtered"] += 1
            actions.append(action)
            modes.append(mode)
        return {"actions": actions, "modes": modes, "fidelity": fidelity, "features": features,
                "planner_selected": [gate.planner_selected for gate in self.switches],
                "planner_feasible": planner_feasible,
                "goals": self.goals.copy(), "filter_events": filter_events,
                "frontier_count": len(candidates), "assignment": assignment_audit,
                "previous_goals": previous_goals, "refreshed": refresh}
