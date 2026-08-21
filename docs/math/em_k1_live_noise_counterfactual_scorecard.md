# K=1 live RELION binary64-noise counterfactual scorecard

This fixed-denominator same-A100 diagnostic is non-scoring.
Map acceptance uses FSC/FSC-AUC; correlation is forbidden.

Accepted gates: **21 / 24**.
Science gates: **17 / 20**.
Provenance gates: **4 / 4**.

| Checked | Fixed gate | Result |
| --- | --- | ---: |
| [x] | `score-live-noise-stack-0035` | pass |
| [x] | `score-live-noise-stack-0252` | pass |
| [x] | `score-live-noise-stack-0348` | pass |
| [x] | `score-live-noise-stack-0591` | pass |
| [x] | `score-live-noise-stack-0683` | pass |
| [x] | `score-live-noise-stack-1100` | pass |
| [x] | `score-live-noise-stack-1522` | pass |
| [x] | `score-live-noise-stack-1640` | pass |
| [x] | `score-live-noise-stack-1767` | pass |
| [x] | `score-live-noise-stack-2124` | pass |
| [x] | `score-live-noise-stack-2322` | pass |
| [x] | `score-live-noise-stack-2330` | pass |
| [x] | `score-live-noise-stack-2846` | pass |
| [x] | `score-live-noise-stack-2994` | pass |
| [x] | `map-parity-beyond-floor-half1` | pass |
| [x] | `map-parity-beyond-floor-half2` | pass |
| [x] | `map-parity-beyond-floor-merged` | pass |
| [ ] | `map-gt-nondegradation-half1` | fail |
| [ ] | `map-gt-nondegradation-half2` | fail |
| [ ] | `map-gt-nondegradation-merged` | fail |
| [x] | `provenance-science-arms-zero-same-gpu` | pass |
| [x] | `provenance-analysis-job-terminal-zero` | pass |
| [x] | `provenance-independent-report-byte-identical` | pass |
| [x] | `provenance-pinned-evidence-no-correlation` | pass |

Classification: `live_noise_improves_score_and_parity_but_regresses_gt`.

Score gates: **14/14**.
Parity FSC-AUC gains beyond the control floor: **3/3**.
GT FSC-AUC nondegradation: **0/3**.

The live-noise treatment improves RECOVAR-to-RELION FSC-AUC but regresses
GT FSC-AUC for half 1, half 2, and merged maps, so it is rejected.

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k1_live_noise_counterfactual_scorecard.py --check
```
