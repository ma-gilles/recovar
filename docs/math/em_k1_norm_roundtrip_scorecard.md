# K=1 continuation normalization-roundtrip scorecard

This fixed-denominator causal diagnostic is non-scoring.

| Fixed paired panel | Sampling-only baseline | Normalization treatment | Gain |
| --- | ---: | ---: | ---: |
| Preprocessing exactness | 0/2 | 2/2 | +2 |
| Geometry identity | 5/5 | 5/5 | +0 |
| Score/map gates | 17/21 | 17/21 | +0 |

| Checked | Preprocessing case | Baseline | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [x] | `norm-roundtrip-preprocess-correction` | fail | pass | improved | sampling-only 0/14 exact, max_abs=2.98023223877e-6; normalization treatment 14/14 exact, max_abs=0 |
| [x] | `norm-roundtrip-preprocess-normalized-shifted-real` | fail | pass | improved | sampling-only 0/14 exact, max_abs=1.38282775879e-5; normalization treatment 14/14 exact over 229376 elements, max_abs=0 |

| Checked | Geometry case | Baseline | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [x] | `norm-roundtrip-geometry-raw-input` | pass | pass | retained | sampling-only 14/14 exact; normalization treatment 14/14 exact |
| [x] | `norm-roundtrip-geometry-rotation-keys` | pass | pass | retained | sampling-only 14/14 exact; normalization treatment 14/14 exact |
| [x] | `norm-roundtrip-geometry-local-rotation-indices` | pass | pass | retained | sampling-only 14/14 exact; normalization treatment 14/14 exact |
| [x] | `norm-roundtrip-geometry-euler-matrices` | pass | pass | retained | sampling-only 14/14 exact; normalization treatment 14/14 exact |
| [x] | `norm-roundtrip-geometry-translations` | pass | pass | retained | sampling-only 14/14 exact; normalization treatment 14/14 exact |

| Checked | Score/map case | Baseline | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [x] | `norm-roundtrip-score-stack-0035` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-0252` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-0348` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-0591` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-0683` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-1100` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-1522` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-1640` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-1767` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-2124` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-2322` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-2330` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-2846` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-score-stack-2994` | pass | pass | retained | sampling-only pass; normalization treatment pass |
| [x] | `norm-roundtrip-map-parity-half1` | pass | pass | retained | sampling-only delta=+2.46680715965e-7; normalization treatment delta=+2.46516179692e-7 |
| [x] | `norm-roundtrip-map-parity-half2` | pass | pass | retained | sampling-only delta=+6.20228254311e-8; normalization treatment delta=+6.07684376153e-8 |
| [x] | `norm-roundtrip-map-parity-merged` | pass | pass | retained | sampling-only delta=+1.44716856831e-7; normalization treatment delta=+1.43992731183e-7 |
| [ ] | `norm-roundtrip-map-gt-half1` | fail | fail | unchanged-fail | sampling-only delta=-2.08306407599e-5; normalization treatment delta=-2.08820403353e-5 |
| [ ] | `norm-roundtrip-map-gt-half2` | fail | fail | unchanged-fail | sampling-only delta=-1.67426148376e-5; normalization treatment delta=-1.68476195946e-5 |
| [ ] | `norm-roundtrip-map-gt-merged` | fail | fail | unchanged-fail | sampling-only delta=-1.89465661311e-5; normalization treatment delta=-1.90227873128e-5 |
| [ ] | `norm-roundtrip-overall` | fail | fail | unchanged-fail | both arms pass score and parity gates but fail all three GT non-degradation gates |

Classification: `normalization_roundtrip_closes_preprocessing_exactness_but_score_map_panel_is_retained`.
