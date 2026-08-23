# K=4 deterministic contribution-repeatability candidate scorecard

This fixed-denominator panel records strict same-observer candidate
repeatability. It is non-scoring and remains contingent on the separate
K=4 FSC/FSC-AUC and GT-quality A/B.

Strict byte equality: **3 / 3**.
Published baseline retained: **0 / 3**.

| Checked | Case | Archive | Result | Failed arrays |
| --- | --- | --- | ---: | ---: |
| [x] | `candidate-pass2-archive-byte-equality` | `pass2` | pass | 0 |
| [x] | `candidate-contribution-archive-byte-equality` | `contribution` | pass | 0 |
| [x] | `candidate-device-signature-archive-byte-equality` | `device_signature` | pass | 0 |

Classification: `same_observer_archives_repeat_bit_for_bit`.

The implementation is not production-accepted until the quality A/B passes.
No correlation, tolerance, scale, sign, threshold, map, or FSC claim is used.

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k4_deterministic_contribution_repeatability_candidate_scorecard.py --check
```
