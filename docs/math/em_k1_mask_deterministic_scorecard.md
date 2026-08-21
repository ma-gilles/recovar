# K=1 continuation deterministic-mask scorecard

This fixed-denominator causal diagnostic is non-scoring.

| Fixed paired panel | Normalization baseline | Deterministic-mask treatment | Gain |
| --- | ---: | ---: | ---: |
| Masked-preprocessing exactness | 0/3 | 3/3 | +3 |
| Geometry identity | 5/5 | 5/5 | +0 |
| Score/map gates | 17/21 | 17/21 | +0 |

| Checked | Preprocessing case | Baseline | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [x] | `mask-deterministic-preprocess-background` | fail | pass | improved | normalization-only baseline did not close background identity; deterministic reduction treatment is 14/14 bitwise exact |
| [x] | `mask-deterministic-preprocess-masked-real` | fail | pass | improved | normalization-only 4/14 exact, max_abs=1.1920928955078125e-7; deterministic reduction 14/14 exact |
| [x] | `mask-deterministic-preprocess-masked-fourier` | fail | pass | improved | normalization-only 4/14 exact before and after optics, max_abs=7.483250778162943e-9; deterministic reduction 14/14 exact at both stages |

| Checked | Geometry case | Baseline | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [x] | `mask-deterministic-geometry-raw-input` | pass | pass | retained | normalization-only 14/14 exact; deterministic reduction 14/14 exact |
| [x] | `mask-deterministic-geometry-rotation-keys` | pass | pass | retained | normalization-only 14/14 exact; deterministic reduction 14/14 exact |
| [x] | `mask-deterministic-geometry-local-rotation-indices` | pass | pass | retained | normalization-only 14/14 exact; deterministic reduction 14/14 exact |
| [x] | `mask-deterministic-geometry-euler-matrices` | pass | pass | retained | normalization-only 14/14 exact; deterministic reduction 14/14 exact |
| [x] | `mask-deterministic-geometry-translations` | pass | pass | retained | normalization-only 14/14 exact; deterministic reduction 14/14 exact |

| Checked | Score/map case | Baseline | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [x] | `mask-deterministic-score-stack-0035` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-0252` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-0348` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-0591` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-0683` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-1100` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-1522` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-1640` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-1767` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-2124` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-2322` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-2330` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-2846` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-score-stack-2994` | pass | pass | retained | normalization-only pass; deterministic reduction pass |
| [x] | `mask-deterministic-map-parity-half1` | pass | pass | retained | normalization-only delta=+2.4651617969162487e-7; deterministic reduction delta=+2.4940213116941834e-7 |
| [x] | `mask-deterministic-map-parity-half2` | pass | pass | retained | normalization-only delta=+6.076843761526618e-8; deterministic reduction delta=+6.351724901598743e-8 |
| [x] | `mask-deterministic-map-parity-merged` | pass | pass | retained | normalization-only delta=+1.4399273118304023e-7; deterministic reduction delta=+1.468204324783784e-7 |
| [ ] | `mask-deterministic-map-gt-half1` | fail | fail | unchanged-fail | normalization-only delta=-2.088204033534602e-5; deterministic reduction delta=-2.0893204545124888e-5 |
| [ ] | `mask-deterministic-map-gt-half2` | fail | fail | unchanged-fail | normalization-only delta=-1.6847619594639873e-5; deterministic reduction delta=-1.6798937257617164e-5 |
| [ ] | `mask-deterministic-map-gt-merged` | fail | fail | unchanged-fail | normalization-only delta=-1.9022787312833467e-5; deterministic reduction delta=-1.9005026456325735e-5 |
| [ ] | `mask-deterministic-overall` | fail | fail | unchanged-fail | both arms pass score and parity gates but fail all three GT non-degradation gates |

Classification: `deterministic_mask_reduction_closes_masked_preprocessing_identity_but_score_map_panel_is_retained`.
