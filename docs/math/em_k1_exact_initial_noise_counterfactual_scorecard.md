# K=1 exact RELION initial-noise counterfactual scorecard

This fixed-denominator same-A100 diagnostic is non-scoring.
Map acceptance uses FSC/FSC-AUC; correlation is forbidden.

Accepted gates: **4 / 24**.
Science gates: **0 / 20**.
Provenance gates: **4 / 4**.

| Checked | Fixed gate | Result |
| --- | --- | ---: |
| [ ] | `score-exact-noise-stack-0035` | fail |
| [ ] | `score-exact-noise-stack-0252` | fail |
| [ ] | `score-exact-noise-stack-0348` | fail |
| [ ] | `score-exact-noise-stack-0591` | fail |
| [ ] | `score-exact-noise-stack-0683` | fail |
| [ ] | `score-exact-noise-stack-1100` | fail |
| [ ] | `score-exact-noise-stack-1522` | fail |
| [ ] | `score-exact-noise-stack-1640` | fail |
| [ ] | `score-exact-noise-stack-1767` | fail |
| [ ] | `score-exact-noise-stack-2124` | fail |
| [ ] | `score-exact-noise-stack-2322` | fail |
| [ ] | `score-exact-noise-stack-2330` | fail |
| [ ] | `score-exact-noise-stack-2846` | fail |
| [ ] | `score-exact-noise-stack-2994` | fail |
| [ ] | `map-parity-beyond-floor-half1` | fail |
| [ ] | `map-parity-beyond-floor-half2` | fail |
| [ ] | `map-parity-beyond-floor-merged` | fail |
| [ ] | `map-gt-nondegradation-half1` | fail |
| [ ] | `map-gt-nondegradation-half2` | fail |
| [ ] | `map-gt-nondegradation-merged` | fail |
| [x] | `provenance-science-arms-zero-same-gpu` | pass |
| [x] | `provenance-analysis-job-terminal-zero` | pass |
| [x] | `provenance-independent-report-byte-identical` | pass |
| [x] | `provenance-pinned-evidence-no-correlation` | pass |

Classification: `exact_initial_noise_bootstrap_rejected_under_fixed_score_and_fsc_gates`.

Score gates: **0/14**.
Parity FSC-AUC gains beyond the control floor: **0/3**.
GT FSC-AUC nondegradation: **0/3**.

The exact bootstrap leaves all score captures unchanged and slightly
regresses both parity and GT FSC-AUC, so it is rejected.

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k1_exact_initial_noise_counterfactual_scorecard.py --check
```
