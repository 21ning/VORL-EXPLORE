# Recovery and gate-state correction (2026-09-13)

The completion audit found that implementation `b3e7073` counted threshold
transitions before the planner-feasibility override. A high-fidelity robot with
an infeasible planner could therefore be counted as repeatedly switching to A*,
even though its final execution gate remained RL. Those intermediate attempts
could also falsely trigger the oscillation recovery condition.

The regression fixture holds an infeasible planner for six decisions. The old
implementation counted five switches; the corrected implementation counts the
one actual initial A* to RL transition. The history and reported switch count
now use the final per-step state, after feasibility enforcement. This follows
the temporal switch-state meaning in [the method](https://arxiv.org/html/2603.07973v1#S4.SS3).

Recovery also now explicitly starts from the real reactive proposal. Previously,
an active recovery could retain a feasible A* action while labeling it recovery.
An additional regression fixture gives A* a rightward action and RL an upward
action; recovery must execute the latter. Stuck, invalid and no-op proposals
still use the existing seeded feasible symmetry-breaking move.

Trajectories now record final planner-selection and planner-feasibility flags.
The verifier recomputes planning feasibility, replays the hysteresis sequence,
and checks both final states and their total switch count. It also requires
matching method-source hashes between the run, trained gate and verifier checkout.

No hyperparameters or held-out seeds are changed as part of this correction.
The gate is refitted from new training rollouts and both dynamic examples are
rerun. Earlier artifacts remain recoverable from commit `0948249` and the
original isolated run directories; they are not relabeled as corrected results.
