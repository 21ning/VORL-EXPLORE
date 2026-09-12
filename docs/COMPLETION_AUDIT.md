# Requested deliverables and evidence

The user's scope is to run the VORL-EXPLORE main method, produce GIFs, keep it
independent of CARE, and publish to `21ning/VORL-EXPLORE`. The earlier clarification
requested an executability check, not a full statistical benchmark.

| Requirement | Implementation and direct evidence |
| --- | --- |
| Independent VORL source/environment/data | Dedicated VORL checkout, environment, models, cache and run directories; no CARE runtime imports. CARE's tracked working-tree diff is compared before and after this revision in `examples/isolation-check.json`. |
| Coupled main-method assignment | `vorl/assignment.py` implements BFS Voronoi ownership, utility, BFS repulsion and fidelity-dependent penalties. `allocation.jsonl` records every refreshed candidate score; the verifier recomputes every round. |
| Genuine learned reactive branch and fidelity gate | Strictly loaded upstream EPOM actor and recurrent core, not a random substitute; fitted eight-feature gate with traceable training seeds and data hashes. Main evaluations freeze the gate. |
| Hysteresis, feasibility fallback and recovery | Regression tests cover the final-state counting correction and genuine reactive recovery. Per-step final gate and feasibility flags are recorded; the verifier recomputes feasibility and replays gate states and switch counts. |
| Actual dynamic executions | Complete state/action arrays for the fixed 40×40 and 80×80 cases. Each transition is checked for movement constraints, occupancy, sensor agreement and half-speed dynamic obstacles. |
| GIFs from those executions | `media/` is rendered from the verified recorded states; `render-check.json` binds each GIF to its source trajectory hash and rendered state indices. All frames are decoded and first/middle/last frames visually inspected. |
| Published runnable repository | Git `main` contains source, locked dependencies, model retrieval, fitted gate, commands, evidence and GIFs. A new clean checkout from the fetched GitHub `main` is rerun and compared to committed arrays and allocation records. |

Thirty tests include deterministic component checks and regression fixtures;
they are not treated as a substitute for the real policy rollouts and full-trace
verification. The clean-checkout report records its tested commit, and distinguishes
reuse of the isolated dependency environment from a fresh installation.

## Boundaries that remain explicit

The supplied archive is legacy MIX reference code, not the unpublished final
paper implementation. This executable reconstruction follows the coupled method
and uses its reference project's genuine EPOM checkpoint. It does **not** establish
that EPOM was retrained under the paper's exact procedure, recover unpublished
hyperparameters, or reproduce the paper's numeric benchmark results. These
limitations are retained in the README, run metadata and GIF footer.

No-frontiers completion is not replaced by a coverage threshold. Horizon-limited
executions remain failures under that criterion, even when most cells have been
observed. The correction and previous-version provenance are documented in
[`RECOVERY_FIX.md`](RECOVERY_FIX.md).
