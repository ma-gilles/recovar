# K=1 restart particle-order restoration scorecard

This fixed-denominator same-A100 diagnostic is non-scoring.
Map acceptance uses FSC/FSC-AUC; correlation is forbidden.

Stock restart: **17 / 20**.
Restored repeats: **40 / 40**.
Paired gain: **+3 / 20 per restored repeat**.

| Checked | Fixed gate | Stock | Restored A | Restored B |
| --- | --- | ---: | ---: | ---: |
| [x] | `score-stack-0035` | pass | pass | pass |
| [x] | `score-stack-0252` | pass | pass | pass |
| [x] | `score-stack-0348` | pass | pass | pass |
| [x] | `score-stack-0591` | pass | pass | pass |
| [x] | `score-stack-0683` | pass | pass | pass |
| [x] | `score-stack-1100` | pass | pass | pass |
| [x] | `score-stack-1522` | pass | pass | pass |
| [x] | `score-stack-1640` | pass | pass | pass |
| [x] | `score-stack-1767` | pass | pass | pass |
| [x] | `score-stack-2124` | pass | pass | pass |
| [x] | `score-stack-2322` | pass | pass | pass |
| [x] | `score-stack-2330` | pass | pass | pass |
| [x] | `score-stack-2846` | pass | pass | pass |
| [x] | `score-stack-2994` | pass | pass | pass |
| [x] | `parity-fsc-auc-half1` | pass | pass | pass |
| [x] | `parity-fsc-auc-half2` | pass | pass | pass |
| [x] | `parity-fsc-auc-merged` | pass | pass | pass |
| [x] | `gt-fsc-auc-nondegradation-half1` | fail | pass | pass |
| [x] | `gt-fsc-auc-nondegradation-half2` | fail | pass | pass |
| [x] | `gt-fsc-auc-nondegradation-merged` | fail | pass | pass |

Classification: `serialized_iteration1_state_closes_case22_score_and_map_gates_when_iteration1_particle_order_is_restored`.

Dispatch controls: **4 / 4**; each dispatch contains the exact
3,000-particle permutation. Restored A and B use the fresh
iteration-1 order; stock does not.

Restored A/B FSC-AUC repeatability:

- half 1: `0.9999999999872374`
- half 2: `0.9999999999871069`
- merged: `0.9999999999932139`

The intervention is diagnostic and is not a production RELION patch.
It causally identifies stock restart reordering as the source of its
three GT FSC-AUC failures.

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k1_restart_particle_order_scorecard.py --check
```
