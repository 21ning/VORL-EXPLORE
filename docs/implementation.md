# Configuration and implementation

## Method

`vorl/assignment.py` partitions four-connected frontiers by free-space BFS
distance. Within each robot's partition, utility, distance and teammate repulsion
are min–max normalized. A shared fidelity score controls the distance and
repulsion weights.

`vorl/fidelity.py` implements the eight-feature logistic gate, hysteresis and
margin-filtered self-supervised updates. `vorl/controller.py` combines allocation,
A*/EPOM actions, feasibility filtering and bounded recovery. Recovery uses a
feasible EPOM proposal or a seeded feasible move; it is not a deadlock guarantee.

`vorl/policy.py` strictly loads the recurrent EPOM architecture. It never falls
back to random model weights. The policy receives private local observation
memory; allocation and A* use the synchronized shared map.

## Settings

The default file is `configs/reconstruction.json`. Its filename is retained for
checkpoint compatibility. The main groups are:

| Setting | Purpose |
| --- | --- |
| `sensing_radius`, `interaction_radius` | Observation and teammate neighborhoods |
| `assignment_interval`, `assignment` | Refresh interval, distance and repulsion weights |
| `gate` | Low/high thresholds, dwell time and initial branch |
| `history_window`, `recovery_*`, `oscillation_switches` | Progress history and recovery |
| `quality_weights`, `quality_margin` | Self-supervised quality labels |
| `update_interval`, `gate_learning_rate`, `gate_l2` | Online gate updates |
| `gate_fit_epochs`, `gate_fit_batch_size` | Warm-start fitting |

Run a custom configuration with `--config path/to/config.json`. Fit a matching
gate with the same file before evaluation; configuration hashes are checked.
The supplied gate uses fitting seeds 0–3. Default evaluation seeds start at 1000.
Unspecified paper hyperparameters and simulator details are explicit settings
of this implementation, not recovered original values.

## Environment and outputs

The NumPy simulator samples independent static obstacles and places agents in
the largest free component. Square sensing has no occlusion. Dynamic obstacles
follow seeded grid routes at half robot speed. Communication is synchronized;
delay, packet loss and perception noise are not modeled.

Every run writes:

- `config.json`: the settings used for the run.
- `summary.json`: termination, counters and model/config/source hashes.
- `trajectory.npz`: recorded states, actions, goals and fidelity values.
- `allocation.jsonl`: frontier scores at reassignment rounds.
- `progress.jsonl`: periodic runtime progress.

State arrays have `T+1` entries; action arrays have `T`. `success_no_frontiers`
uses the strict no-frontiers completion criterion. Observed-cell fraction is
not success rate. Safety counters record rejected or cancelled proposals, not
physical collisions. `astar_selected` and `rl_selected` count gate choices
before recovery; final executed modes are in the trajectory's `modes` array.

`verify_rollout.py` checks frozen-gate runs against the same method source and
checkpoint. It does not validate online-adaptation runs or performance claims.
The `vorl/` source and supplied gate are kept together; fitting a gate after
method changes records a new source hash. CLI and documentation changes do not
change the method hash.

## Models

The default paths are `models/epom` and `checkpoints/fidelity-gate.json`. Paths
for bundled config/gate/model defaults are resolved relative to the checkout;
explicit paths and `--output` are relative to the current working directory.

The EPOM release comes from the upstream When to Switch project. Its training
distribution differs from the paper's described from-scratch PPO protocol. The
small gate checkpoint is fitted for this implementation and retains its original
provenance metadata. Gate fitting is not EPOM/PPO training.

For offline setup, transfer the official upstream `weights.zip` archive and run:

```bash
python scripts/fetch_epom.py --archive /path/to/weights.zip --output models/epom
```

The same hash checks apply. Existing output directories are never replaced.

## Successful demo

The README animation is a selected successful run, not an aggregate benchmark.
Its seed and provenance hashes are recorded in `media/successful-exploration.json`.
The renderer's `--require-success` option checks the trajectory hash and refuses
to render an incomplete run as a successful demo. It also checks the final map
against the recorded completion flag; reaching the horizon is not success.
