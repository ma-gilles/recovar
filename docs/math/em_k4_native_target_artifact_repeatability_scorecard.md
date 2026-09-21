# K=4 native target-artifact repeatability scorecard

This fixed-denominator panel admits native classes 2--4 artifacts for
target-local analysis only. Broad all-class attribution remains prohibited.

Fixed admission gates: **32 / 32**.

| Checked | Gate | Result |
| --- | --- | --- |
| [x] | `class2_artifact_validators_passed` | pass |
| [x] | `class2_bpref_bytes_exact` | pass |
| [x] | `class2_dispatch_bytes_exact` | pass |
| [x] | `class2_dispatch_row_count_exact` | pass |
| [x] | `class2_fine_score_bytes_exact` | pass |
| [x] | `class2_hard_pose_class_shift_exact` | pass |
| [x] | `class2_map_fsc_auc_at_least_threshold` | pass |
| [x] | `class2_runtime_replay_and_no_fatal_exact` | pass |
| [x] | `class2_target_state_exact` | pass |
| [x] | `class2_topology_exact` | pass |
| [x] | `class3_artifact_validators_passed` | pass |
| [x] | `class3_bpref_bytes_exact` | pass |
| [x] | `class3_dispatch_bytes_exact` | pass |
| [x] | `class3_dispatch_row_count_exact` | pass |
| [x] | `class3_fine_score_bytes_exact` | pass |
| [x] | `class3_hard_pose_class_shift_exact` | pass |
| [x] | `class3_map_fsc_auc_at_least_threshold` | pass |
| [x] | `class3_runtime_replay_and_no_fatal_exact` | pass |
| [x] | `class3_target_state_exact` | pass |
| [x] | `class3_topology_exact` | pass |
| [x] | `class4_artifact_validators_passed` | pass |
| [x] | `class4_bpref_bytes_exact` | pass |
| [x] | `class4_dispatch_bytes_exact` | pass |
| [x] | `class4_dispatch_row_count_exact` | pass |
| [x] | `class4_fine_score_bytes_exact` | pass |
| [x] | `class4_hard_pose_class_shift_exact` | pass |
| [x] | `class4_map_fsc_auc_at_least_threshold` | pass |
| [x] | `class4_runtime_replay_and_no_fatal_exact` | pass |
| [x] | `class4_target_state_exact` | pass |
| [x] | `class4_topology_exact` | pass |
| [x] | `immutable_static_provenance_exact` | pass |
| [x] | `target_gpu_exact` | pass |

Minimum signed normalized non-DC FSC-AUC over the 12 preserved class-map comparisons: `0.9999999794616498` (threshold `0.999999`).

Classification: `accepted_target_artifact_repeatability`.

Immutable evidence:

- `repeatability_report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/analysis/NATIVE_TARGET_ARTIFACT_REPEAT_RESULT_11996846.json` (SHA-256 `da59157b92956fca4095b87d2dce850cc53d1e21e4e3321474d12bd651f3c4b8`)
- `analysis_completion`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/provenance/NATIVE_TARGET_ARTIFACT_REPEAT_ANALYSIS_COMPLETE_11996846.json` (SHA-256 `acd30c07408c1475ced6eac107ac04abf3c8ccb1feea55d78fff5c645ffbc088`)
- `launcher`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/jobs/run_native_target_artifact_repeat.sbatch` (SHA-256 `363304a62ab0077eef1aae97567650ad50a4c8313c0beb8182aace4f70664177`)
- `analyzer`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/provenance/analyze_native_target_artifact_repeat.py` (SHA-256 `fb7b65c3579c66288ad78e091a1be99618ee847aa3e775f8a553e89421947ed2`)
- `predeclaration`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/provenance/EXECUTION_PREDECLARATION.md` (SHA-256 `b8a47b6ff22c24c9df75d23200ae46579ae75e8994baf47e3c11f03b5e638bc8`)
- `submission`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/provenance/SUBMISSION_11996846.md` (SHA-256 `1274fd53e7c40904b7dd1bbd743d10f4098d5ad6a38769927e7c730bdfdbc432`)
- `science_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/provenance/science_outputs_11996846.sha256` (SHA-256 `f582f642c3ccdae0251ec21eb548e44d440bb89c6b20789765427c82fcd81be0`)
- `static_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/provenance/static_inputs_11996846.sha256` (SHA-256 `5812d510f8c158a524d826f4914ffb3223644c2a927636d5e80fbedbd5952d0c`)
- `science_stdout`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/logs/science_11996846.out` (SHA-256 `51bf46819f0505854adfe2ce2c3a2a4e10c687c2081789e8ed85f62500dc4138`)
- `science_stderr`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/logs/science_11996846.err` (SHA-256 `933fae46b1152e45cbeabdd939d7ccb7e4ed5f11cb762253aab15f50dbeb333d`)

To validate:

```bash
pixi run python scripts/summarize_em_k4_native_target_artifact_repeatability_scorecard.py --check
```
