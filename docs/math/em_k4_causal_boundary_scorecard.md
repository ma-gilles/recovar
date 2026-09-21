# K=4 exact-device causal boundary scorecard

This is a non-scoring, fixed-denominator diagnostic. It cannot change
the frozen K=1 or K=4 FSC/FSC-AUC scorecards.

Fixed causal score: **2 / 4 passing** (4 / 4 evaluated).

| Checked | Gate | Result | Observation |
| --- | --- | --- | --- |
| [x] | `native-target-operand-replay` | pass | Translations 80 and 82 both replay native production raw diff2 bitwise. |
| [x] | `fixed-target-raw-diff2` | pass | RECOVAR and native raw diff2 are bitwise equal at fixed translations 80 and 82. |
| [ ] | `global-raw-diff2` | fail | 25877 of 109184 active raw diff2 values differ bitwise. |
| [ ] | `global-combined-score` | fail | The complete active-table combined-score path is not bitwise equal. |

Classification: `global_raw_and_score_paths_differ_but_fixed_target_closes`.

Metric policy: Fixed bitwise exact-device causal gates only; no fitted scale, sign, threshold, map metric, or correlation.

Immutable evidence:

- `completion_report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_owner_pair_raw11790517_operand11812925_9959cc8a_20260730T2042ET/provenance/ANY_OWNER_PAIR_AUDIT_COMPLETE.json` (SHA-256 `963e9b6b315368ae9a8201b73624163129f92e5626fff4089ad8fe3ce6516552`)
- `raw_report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_owner_pair_raw11790517_operand11812925_9959cc8a_20260730T2042ET/analysis/RAW_SCORE_PARITY_V2.json` (SHA-256 `f19dbd316eb654d0c38d6a334cb052ef5181e201789c576bbece4f604849e214`)
- `operand_report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_owner_pair_raw11790517_operand11812925_9959cc8a_20260730T2042ET/analysis/NATIVE_TARGET_OPERAND_AUDIT_V1.json` (SHA-256 `151194b5412949aa450453e469edc6371ff25256e883b6d641d79ee1919476dd`)
- `pair_report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_owner_pair_raw11790517_operand11812925_9959cc8a_20260730T2042ET/analysis/RAW_TARGET_PAIR_V1.json` (SHA-256 `be047c3ad90220c88834ed0995339bceab65f4cf3d7358fabdf9bac28ebd142c`)
- `raw_capture`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_rawdiff2_orig53722_ec68f651_20260730T1030ET/capture/pass2_orig053722_class001_cs038.npz` (SHA-256 `ccbdc9040da463f479784e3ad270fd76bb5817006742f43c96f9b053bf9d6eef`)
- `operand_capture`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_target_operands_rot1210_6982c77_20260730T1305ET/capture/factors/part48584_stack53723_class1.fine-operand-v1.bin` (SHA-256 `93322e2b98ca11e626f178007f39cf8d6137655fdffd5239907cd2321459270f`)

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k4_causal_boundary_scorecard.py --check
```
