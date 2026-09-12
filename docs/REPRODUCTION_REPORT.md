# Method-level execution report

Latest corrected runs: 2026-09-13 (Asia/Shanghai). Earlier results are retained
in Git history at `0948249`; see [RECOVERY_FIX.md](RECOVERY_FIX.md).

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
- All 30 unit tests passed in the isolated runtime; `pip check` passed.
- The tested CPU package versions are recorded in `requirements-cpu.lock`.

## Held-out dynamic execution

| Item | 40×40 / seed 1000 | 80×80 / seed 1001 |
| --- | ---: | ---: |
| Exploring robots | 4 | 16 |
| Moving obstacles | 8 | 32 |
| Decision steps | 320 | 480 |
| Observed-cell fraction | 0.996875 | 0.9984375 |
| Actual visited-cell overlap | 0.213018 | 0.299736 |
| Executed A* decisions, after recovery override | 642 | 2860 |
| Executed RL decisions, after recovery override | 470 | 3590 |
| Recovery decisions | 168 | 1230 |
| Final per-step gate state changes | 14 | 46 |
| Coupled allocation rounds checked | 284 | 480 |
| Gate updates at evaluation time | 0 | 0 |
| Stop reason | Horizon | Horizon |
| Complete exploration achieved | No | No |

Recorded wall times were approximately 32.7 and 201.3 seconds on the shared
server with CPU inference. These are observations, not controlled speed benchmarks.
Do not interpret two runs as a success-rate estimate or compare their numbers
directly with the paper's different implementation and trained policy.

## What the verification checks

`scripts/verify_rollout.py` checks every recorded transition for adjacency,
action compatibility, unique occupancy, static-obstacle exclusion, half-speed
dynamic movement and sensor-patch agreement. It recomputes frozen gate scores
from recorded features and checks checkpoint/configuration provenance.
It additionally recomputes planner feasibility and replays all final switch
states, including the infeasibility fallback, to verify the switch count.
Run and training-gate method-source hashes must match the verifier checkout.

For every recorded allocation round it recomputes utilities, BFS distances,
pose/previous-goal repulsion and the fidelity-coupled objective, and checks the
selected target. The GIF renderer consumes the recorded state arrays and reports
which state indices it rendered. Every GIF frame is decoded after saving.
The first, middle and last frames of both GIFs were also visually inspected.

## Clean-checkout repeatability

The previous implementation's clean-checkout proof is preserved separately in
[`examples/clean-checkout-pre-recovery.json`](../examples/clean-checkout-pre-recovery.json).
It applies to commit `b3e7073`, not to the corrected implementation. The corrected
version has passed the full runtime checks above; a new clean-checkout repeat
will be recorded separately after publishing its checkpoint and artifacts.

## Remaining limitations

Both held-out examples terminate at their predeclared horizons. The 40×40 run
leaves five unknown corner cells and five remaining frontiers, none reachable
in the final shared map. The 80×80 run leaves ten unknown cells. The strict
no-frontiers criterion is unchanged; incomplete runs are not relabeled as successes.

The simulator, gate fitting conventions and unreported numerical settings are
explicit reconstruction choices. Exact original-code, original-policy-training
or published-number reproduction remains outside the evidence supplied here;
see [RECONSTRUCTION_SCOPE.md](RECONSTRUCTION_SCOPE.md).
