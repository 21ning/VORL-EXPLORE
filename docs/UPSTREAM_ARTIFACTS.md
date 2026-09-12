# Recovered upstream EPOM artifacts

The old `AIRI-Institute/when-to-switch` repository now redirects readers to
[`Cognitive-AI-Systems/when-to-switch`](https://github.com/Cognitive-AI-Systems/when-to-switch).
Its [v0 release](https://github.com/Cognitive-AI-Systems/when-to-switch/releases/tag/v0)
contains the missing policy file named by the supplied reference archive.

Verified on 2026-09-12:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `weights.zip` | 143985246 | `cbbf44bce9dc8b59ce4f6059c565c76567d024845d0378286fc862ead6f8b727` |
| `checkpoint_000311682_1000002674.pth` | 116465689 | `549feac19e21593af072677305945d7c22bd7f66cb07927a92eb59f2f8a3cce9` |

The upstream `epom/cfg.json`, `pe-epom/epom.pth`, and `pe-replan/replan.pth`
are byte-identical to their counterparts in the user-provided reference archive.
This strongly supports compatibility with that reference project. It does not
establish that this checkpoint was trained under the VORL paper protocol.

The recorded upstream configuration uses memory radius 7, observation radius 5
and named MAPF training maps. The VORL paper describes a procedural 64-by-64
training distribution and sensing radius 3. Do not silently erase this distinction.

`scripts/fetch_epom.py` retrieves only the reference policy and its configuration
after checking the complete release archive hash. It refuses to overwrite an
existing destination. The 116 MB policy is not committed to Git.

`vorl/policy.py` uses the original Sample Factory core and a topology-matched
EPOM encoder, requiring strict state-dictionary loading. It performs inference
without invoking the legacy experiment launcher. It requires PyTorch >= 2.6
because [older weights-only loaders have a published security vulnerability](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6).

The restored policy passed strict loading and inference checks and is now used
with a newly fitted gate in the method-level dynamic examples. This does not
change its upstream training provenance. See the execution scope and residual
limitations in [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md).
