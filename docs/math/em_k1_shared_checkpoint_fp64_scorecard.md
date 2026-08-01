# K=1 shared-checkpoint binary64-reference scorecard

This fixed-denominator control-floor diagnostic is non-scoring.
Map acceptance uses FSC/FSC-AUC; correlation is forbidden.

Accepted gates: **16 / 34**.

| Checked | Fixed gate | Result |
| --- | --- | ---: |
| [x] | `snapshot-half1-float32-roundtrip` | pass |
| [x] | `snapshot-half2-float32-roundtrip` | pass |
| [x] | `component-structural-normal_a` | pass |
| [x] | `component-structural-normal_b` | pass |
| [x] | `component-structural-fp64_reference` | pass |
| [x] | `geometry-raw_input` | pass |
| [x] | `geometry-rotation_keys` | pass |
| [x] | `geometry-local_rotation_indices` | pass |
| [x] | `geometry-euler_matrices` | pass |
| [x] | `geometry-translation_values` | pass |
| [x] | `preprocess-norm-correction` | pass |
| [x] | `preprocess-masked-real` | pass |
| [x] | `preprocess-masked-fourier-pre-optics` | pass |
| [x] | `preprocess-masked-fourier-post-optics` | pass |
| [ ] | `score-beyond-floor-stack-0035` | fail |
| [ ] | `score-beyond-floor-stack-0252` | fail |
| [ ] | `score-beyond-floor-stack-0348` | fail |
| [ ] | `score-beyond-floor-stack-0591` | fail |
| [ ] | `score-beyond-floor-stack-0683` | fail |
| [ ] | `score-beyond-floor-stack-1100` | fail |
| [ ] | `score-beyond-floor-stack-1522` | fail |
| [ ] | `score-beyond-floor-stack-1640` | fail |
| [ ] | `score-beyond-floor-stack-1767` | fail |
| [ ] | `score-beyond-floor-stack-2124` | fail |
| [ ] | `score-beyond-floor-stack-2322` | fail |
| [ ] | `score-beyond-floor-stack-2330` | fail |
| [ ] | `score-beyond-floor-stack-2846` | fail |
| [ ] | `score-beyond-floor-stack-2994` | fail |
| [ ] | `map-parity-beyond-floor-half1` | fail |
| [x] | `map-gt-nondegradation-half1` | pass |
| [ ] | `map-parity-beyond-floor-half2` | fail |
| [ ] | `map-gt-nondegradation-half2` | fail |
| [ ] | `map-parity-beyond-floor-merged` | fail |
| [x] | `map-gt-nondegradation-merged` | pass |

Classification: `shared_checkpoint_fp64_reference_rejected`.

Live-reference-dominated score cases: **0/14**.
Parity FSC-AUC gains beyond the control floor: **0/3**.
GT FSC-AUC nondegradation: **2/3**.

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k1_shared_checkpoint_fp64_scorecard.py --check
```
