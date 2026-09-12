# Reproduction source audit

Historical audit of the supplied reference archive. The subsequent executable
method reconstruction is documented in [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md).
It does not retroactively make the archive the unpublished final paper source.

The target repository initially contains only its README (base commit
`9ce81d4611ebd3d26331fb73e5ed97171fe2d584`). The user supplied
`Multiexloration-github-20260709.zip` as reference code; its SHA-256 is
`5cd77f7181f0ab4e2299732b1396278ef85804ed580af2e4182d62d9dc83dde9`.
It contains 253 files, including legacy source, historical results and two
small model state dictionaries. The archive is preserved outside the source
checkout, and its documentation is reference material, not new user requests.

## Main-method fidelity checks

Primary method specification: [VORL-EXPLORE, arXiv:2603.07973v1,
Section IV and V-A](https://arxiv.org/html/2603.07973v1).

| Component | Reference archive evidence | Required before claiming reproduction |
| --- | --- | --- |
| Frontier allocation | `targetalgo/Voronoi.py`; fixed scoring parameters in `CoExMIX3.py` | Fidelity-coupled scoring and its settings |
| A*/RL arbitration | `CoExMIX3.py` switches on a nearby-agent count threshold of four and a two-position stuck check | Eight-feature logistic gate, hysteresis and recovery implementation |
| Reactive policy | `appo/model.py` loads `weights/epom/checkpoint_p0`; matching named checkpoint recovered from current upstream release | Strict loading and provenance verification; no random-policy substitution |
| Gate initialization | No logistic-gate checkpoint identified | Warm-start gate parameters and frozen evaluation configuration |
| Online recalibration | No corresponding update implementation identified | Separate optional adaptation mode; main comparisons keep the gate frozen |
| GIF | `animation.py` produces SVG | Render and inspect a GIF from a newly recorded, authentic main-method trajectory |

The default large policy checkpoint is explicitly excluded in the reference
README and is absent from the archive. A non-executing inspection of both
`weights/pe-epom/epom.pth` and `weights/pe-replan/replan.pth` finds encoder and
`value_head` parameter names, not actor/action-distribution or recurrent-core
parameters. These files must not be silently substituted for the EPOM policy.
The final source/checkpoints have been requested from the user.

The referenced upstream [When to Switch repository](https://github.com/AIRI-Institute/when-to-switch)
links to `releases/download/v0/weights.zip`. On 2026-09-12, that URL returned
HTTP 404 and the old repository's public releases API returned an empty list.
Subsequent inspection of the existing clone's Git remote identified the current
`Cognitive-AI-Systems` repository. Its release was recovered and hash-checked;
see [UPSTREAM_ARTIFACTS.md](UPSTREAM_ARTIFACTS.md). Upstream checkpoint provenance
must not be confused with the VORL paper's independently trained checkpoint.

Other implementation differences found in the legacy modules include
radius-based frontier detection instead of four-neighbor adjacency, a hard
500-expansion A* limit, and an allocator that computes utility but excludes
it from its final score. These are evidence that the archive should not be
relabeled as the final paper implementation.

The reference requirements are a broad environment snapshot, not a compatible
minimal lockfile: for example, the pinned NumPy 1.23.1 conflicts with the
listed SciPy 1.14.1 requirement. Dependency setup must use its own environment
and record any compatibility adjustment without changing CARE.

No paper result is claimed reproduced, no legacy CSV is relabeled as a fresh
run, and no unrelated CARE files are copied or changed. This audit does not
grant permission to invent missing learned parameters or paper results.

## Bounded diagnostics

`scripts/source_check.py` parses Python source and checkpoint opcode strings
without executing pickle. Exit code 2 means readiness checks found missing
artifacts or syntax errors; it is not an unexpected process crash.

`scripts/reference_smoke.py` imports the unmodified reference A* and Voronoi
functions into a separate static 24-by-24, four-robot harness. It records every
actual position, goal and observed map, renders those records as a GIF, and
labels both the metadata and animation as a non-VORL diagnostic. This does not
run `CoExMIX3.py` and does not establish that the full reference experiment runs.
No RL policy or invented gate parameters are substituted. Source hashes are
recorded to tie the diagnostic back to the supplied archive.

Commands, after creating a separate Python 3.10 environment:

```bash
python -m pip install -r requirements-diagnostic.txt
python -m pytest -q tests/test_diagnostics.py
python scripts/source_check.py --reference /path/to/extracted-reference --output runs/source-check.json
python scripts/reference_smoke.py --reference /path/to/extracted-reference --output runs/reference-smoke
```

The output directory for a smoke run must not already exist. Keep all source,
environments, caches and runs within a dedicated VORL workspace; do not reuse
or mutate another project's environment. Exact main-method execution remains
pending the missing final source and learned artifacts.

Verified on 2026-09-12 in an isolated Python 3.10.20 environment:

- All 49 reference Python files parsed without syntax errors.
- Five diagnostic tests passed; `pip check` reported no broken requirements.
- Static four-agent diagnostic: 40 steps, 160 total moved steps, 41 GIF frames.
- A second run produced a byte-identical trajectory (SHA-256
  `81f148081149689d26d812629ec20e2ee415e4505ca990f96b82c344155f2321`).
- All GIF frames decoded successfully; the final frame was visually inspected.

These historical checks establish only the explicitly described diagnostic.
The later method-level implementation and trained gate are separate; the original
reference-only GIF is never relabeled as VORL-EXPLORE.
