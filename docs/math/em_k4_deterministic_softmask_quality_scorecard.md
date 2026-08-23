# K=4 deterministic soft-mask quality A/B scorecard

This fixed-denominator panel records the predeclared full-trajectory
quality and provenance gates for the deterministic CUDA soft-mask reduction.
The published K=4 score remains unchanged.

Quality acceptance: **7 / 7**.

| Checked | Gate | Result | Observation |
| --- | --- | --- | --- |
| [x] | `direct-pass-count-not-lower` | pass | 41/60 control -> 41/60 treatment |
| [x] | `all-class-iteration-count-not-lower` | pass | 9/15 control -> 9/15 treatment |
| [x] | `gt-delta-pass-count-not-lower` | pass | 60/60 control -> 60/60 treatment |
| [x] | `class-agreement-pass-count-not-lower` | pass | 15/15 control -> 15/15 treatment |
| [x] | `direct-nondegradation-60-of-60` | pass | 60/60; minimum -0.0007612990230934091 |
| [x] | `gt-nondegradation-60-of-60` | pass | 60/60; minimum -0.000109608469138045 |
| [x] | `cohort-and-provenance-4-of-4` | pass | 4/4 provenance gates; exact 15-iteration topology |

| Fixed quality panel | Control | Treatment | Change |
| --- | ---: | ---: | ---: |
| direct per-class FSC-AUC | 41/60 | 41/60 | 0 |
| all-class iterations | 9/15 | 9/15 | 0 |
| GT-delta panels | 60/60 | 60/60 | 0 |
| class-agreement iterations | 15/15 | 15/15 | 0 |

All treatment-minus-control nondegradation panels pass: direct **60/60** (minimum `-0.000761299023093`), GT **60/60** (minimum `-0.000109608469138`).

Observed whole-arm wall time is `27659 s -> 27684 s` (`+0.0904%`); this was recorded but was not one of the seven formal acceptance gates.

Classification: `quality_and_topology_preserved__production_integration_accepted`.

Both independent FSC/topology analyses, shellwise arrays, and pair
reports reproduce exactly. Correlation is not computed.

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k4_deterministic_softmask_quality_scorecard.py --check
```
