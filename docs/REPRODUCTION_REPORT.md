# Method-level execution report

Run date: 2026-09-12 to 2026-09-13 (Asia/Shanghai).

## Proven scope

The supplied reference policy was recovered, strictly loaded and run through
the reconstructed coupled allocator, logistic gate, hysteresis and recovery.
A new warm-start gate was fitted from real dynamic rollouts. Main evaluation
kept its coefficients fixed. This is not a claim that the unpublished original
VORL policy training or the published result tables have been reproduced.

The work used an independent source checkout, Python 3.10.20 environment,
model directory, cache and run directory. No CARE source or environment was
modified or imported. The upstream release and input archive remain preserved.

## Learning and dependencies

- EPOM: 9,703,046 model parameters; strict state-dictionary loading, CPU inference,
  deterministic reset, and changing recurrent logits verified.
- Checkpoint SHA-256: `549feac19e21593af072677305945d7c22bd7f66cb07927a92eb59f2f8a3cce9`.
- Gate: seeds 0–3; 64×64 grids, 64 robots, 32 moving obstacles; 512-step maximum
  horizon, with early termination for seeds 2 and 3 at 16 and 20 steps.
- Clear-margin training examples: 46,860, including 14,746 positive and 32,114 negative.
- Regularized BCE: 0.693147 before fitting, 0.432006 after fitting. This is a
  training optimization check, not evidence of held-out task performance.
- All 28 unit tests passed in the isolated runtime; `pip check` passed.
- The tested CPU package versions are recorded in `requirements-cpu.lock`.

## Held-out dynamic execution

| Item | 40×40 / seed 1000 | 80×80 / seed 1001 |
| --- | ---: | ---: |
| Exploring robots | 4 | 16 |
| Moving obstacles | 8 | 32 |
| Decision steps | 320 | 480 |
| Observed-cell fraction | 0.996875 | 0.99796875 |
| Actual visited-cell overlap | 0.193676 | 0.310423 |
| Executed A* decisions, after recovery override | 642 | 2858 |
| Executed RL decisions, after recovery override | 470 | 3591 |
| Recovery decisions | 168 | 1231 |
| Recorded hysteresis state changes | 35 | 228 |
| Coupled allocation rounds checked | 284 | 480 |
| Gate updates at evaluation time | 0 | 0 |
| Stop reason | Horizon | Horizon |
| Complete exploration achieved | No | No |

Recorded wall times were approximately 30.6 and 199.1 seconds on the shared
server with CPU inference. These are observations, not controlled speed benchmarks.
Do not interpret two runs as a success-rate estimate or compare their numbers
directly with the paper's different implementation and trained policy.

## What the verification checks

`scripts/verify_rollout.py` checks every recorded transition for adjacency,
action compatibility, unique occupancy, static-obstacle exclusion, half-speed
dynamic movement and sensor-patch agreement. It recomputes frozen gate scores
from recorded features and checks checkpoint/configuration provenance.

For every recorded allocation round it recomputes utilities, BFS distances,
pose/previous-goal repulsion and the fidelity-coupled objective, and checks the
selected target. The GIF renderer consumes the recorded state arrays and reports
which state indices it rendered. Every GIF frame is decoded after saving.

## Remaining limitations

Both held-out examples terminate at their predeclared horizons. The 40×40 run
leaves five unknown corner cells and five remaining frontiers, none reachable
in the final shared map. The 80×80 run leaves thirteen unknown cells. The strict
no-frontiers criterion is unchanged; incomplete runs are not relabeled as successes.

The simulator, gate fitting conventions and unreported numerical settings are
explicit reconstruction choices. Exact original-code, original-policy-training
or published-number reproduction remains outside the evidence supplied here;
see [RECONSTRUCTION_SCOPE.md](RECONSTRUCTION_SCOPE.md).
