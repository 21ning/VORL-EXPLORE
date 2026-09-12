# Scope and explicit reconstruction choices

This project targets a runnable **method-level reconstruction**, not bitwise
recovery of the unpublished final implementation or reproduction of its result
tables. It combines the reference project's genuine upstream EPOM checkpoint
with new implementations of the [paper's coupled method](https://arxiv.org/html/2603.07973v1#S4).

## What is newly implemented

- The same eight-feature fidelity score feeds both the frontier objective and
  A*/EPOM arbitration; it is not the reference's crowd-count threshold switch.
- The allocator uses four-neighbor frontiers, reachable free-space BFS distances,
  normalized utility/distance/repulsion, and simultaneous previous-goal snapshots.
- Gate dwell counters, feasibility fallback, bounded recovery, safe actions and
  optional delayed self-supervised updates are implemented explicitly.
- Main evaluation requires a fitted checkpoint and disables online updates.
  A runtime guard rejects an unfitted or debug gate and overlapping training seeds.

## Differences and unspecified details

The downloaded EPOM was trained by the upstream When to Switch project. Its
configuration is not the paper's procedural policy-training configuration. We do
not rename it as an original VORL policy or claim to retrain that large policy.
The logistic warm-start gate is fitted here from new execution traces, not copied
from the paper. All unreported numerical settings are declared in
`configs/reconstruction.json` before evaluation and are not tuned on test runs.

The NumPy simulator is a new explicit implementation, not bitwise Pogema parity.
Static cells are independent Bernoulli draws. Robots and moving obstacles start
in the largest free component; maps without enough capacity are resampled.
Square sensing has no occlusion, as in the supplied reference. Four-connected
moving obstacles select seeded goals, follow static-map A* routes and move only
on alternate ticks; they wait for occupied cells and choose a new goal after ten
blocked obstacle ticks. Robots act first and obstacle motion follows. These
timing and traffic conventions are reconstruction choices, not recovered settings.

The controller receives only local sensor patches, shared poses and the fused
map. The renderer separately receives world truth. Local EPOM obstacle memories
are private and receive no unseen teammate sensing. Unknown cells are optimistic
zeros inside EPOM memory (reference convention), but are blocked for A* planning.

Feature conventions are explicit in `vorl/observation.py`: square unknown-area
ratios, BFS crowding radius, distance divided by rows+columns-2 and clipped at one,
unreachable/missing goal distance one, missing-goal unknown ratio zero, five-action
feasibility including no-op, and a constant-position stuck flag. Boundary cells
outside the map count as known blocked space. Partition and score ties use robot
and candidate order; this follows the robot-specific partition in the method text.

Recovery lasts two ticks by default. It retains a feasible real EPOM proposal,
or takes a seeded feasible move when stuck, blocked or proposing no-op, then
returns to the gate with a cooldown. The final action mask blocks cells occupied
at the start of a tick; the simulator resolves remaining simultaneous conflicts.
Reported safety interventions are **not** claims of actual collisions.

## Gate training and evaluation separation

Warm-start collection uses four declared seeds, 64-by-64 maps, 64 robots, 32 moving
obstacles, density drawn from Uniform(0.15, 0.45), and a 512-step maximum horizon.
An episode may terminate earlier when no frontiers remain. The initial collection
gate has zero weights, explicitly labeled untrained; evaluation cannot use it.

Labels come from actual subsequent coverage, same-goal BFS progress, safety
interventions and physical stalls. Each step's distance change compares its
before/after positions to the same assigned goal on the new shared map; changing
goals never earns distance credit. Window labels are attached to the oldest
feature vector and only sufficiently clear labels enter logistic fitting.

One shared warm-start vector is fitted by regularized BCE and copied to each
evaluation robot. This enables different team sizes; adaptation, when explicitly
enabled, updates local copies. Main runs keep them fixed. Training loss only
checks that fitting happened; it is not a held-out exploration performance claim.

The delivered small evaluation set is for executability, integration and GIF
verification. It is not the full multi-baseline, 100-trial paper benchmark.
Completion is defined strictly as no frontiers remaining; horizon-limited runs
are reported as such, without changing the criterion or hiding unsuccessful runs.
