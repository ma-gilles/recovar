# K=1 reference-roundtrip rejection scorecard

This fixed-denominator diagnostic records every predeclared control and
acceptance gate from the rejected case-22 reference-roundtrip experiment.
It cannot change the FSC/FSC-AUC quality scorecards.

Accepted gates: **2 / 9**.

| Checked | Case | Result | Observation |
| --- | --- | ---: | --- |
| [ ] | `precontrol-half1-byte-identity` | fail | 14,528 voxels differ; relative L2 1.0893439009080386e-08; FSC-AUC 0.999999999999237 |
| [ ] | `precontrol-half2-byte-identity` | fail | 14,411 voxels differ; relative L2 1.1299914878841131e-08; FSC-AUC 0.999999999999129 |
| [ ] | `baseline-component-validator` | fail | Replay-p95 gate passes 12/14 particles |
| [ ] | `roundtrip-component-validator` | fail | Replay-p95 gate passes 12/14 particles |
| [ ] | `serialized-component-validator` | fail | Replay-p95 gate passes 11/14 particles |
| [ ] | `baseline-roundtrip-normalization-identity` | fail | Norm-correction bits agree for 12/14 particles |
| [x] | `serialized-score-boundary` | pass | Absolute score and live-reference-dominance gates pass 14/14 particles |
| [x] | `serialized-map-parity` | pass | FSC-AUC strictly improves for half 1, half 2, and merged maps (3/3) |
| [ ] | `serialized-map-gt-nondegradation` | fail | 0/3 maps pass; merged GT FSC-AUC change is -1.902727348485067e-05 |

Classification: `reference_roundtrip_experiment_rejected_by_preintervention_and_gt_fsc_gates`.

Immutable evidence:

- `post_terminal_audit`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_reference_roundtrip_ab_aa033a7_20260731T1905ET/post_terminal_audit/POST_TERMINAL_AUDIT.json` (SHA-256 `ad7288ae5a5bd86cba7830b9eb6cf2d6e9b2f4d1cb2040f4e34eb75955a95daf`)
- `baseline_to_roundtrip_operand_boundary`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_reference_roundtrip_ab_aa033a7_20260731T1905ET/post_terminal_audit/baseline_to_roundtrip_OPERAND_BOUNDARY.json` (SHA-256 `484a2ed1efae5aee22d90a3a4f813aae1ac062146c410a97aa3e2af181d247b0`)
- `serialized_restart_retention_score_boundary`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_reference_roundtrip_ab_aa033a7_20260731T1905ET/post_terminal_audit/serialized_restart1_RETENTION_SCORE_BOUNDARY.json` (SHA-256 `2ec32f492ba307e860156095c2fb3f391850e06c082f71f57c3652b4c5a2dce8`)
- `serialized_restart_map_fsc`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_reference_roundtrip_ab_aa033a7_20260731T1905ET/post_terminal_audit/serialized_restart1_MAP_FSC.json` (SHA-256 `1522a1955bbc5e4a909f077d4dee42291890cf85962b66e2e395a7f75ba2a07b`)
- `baseline_component_validation`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_reference_roundtrip_ab_aa033a7_20260731T1905ET/baseline_fresh/analysis/RELION_COARSE_PASS1_COMPONENTS.json` (SHA-256 `c3f3e9bda61ed5888b6704d777be8d768e33602fc7cd9c6cc737d783ae2d5d1e`)
- `roundtrip_component_validation`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_reference_roundtrip_ab_aa033a7_20260731T1905ET/roundtrip_fresh/analysis/RELION_COARSE_PASS1_COMPONENTS.json` (SHA-256 `4ea71e4c0fcc7c5bb4211584054912bde8a5952c070e4d388cf0ddf9cab88cc9`)
- `serialized_component_validation`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_reference_roundtrip_ab_aa033a7_20260731T1905ET/serialized_restart1/analysis/RELION_COARSE_PASS1_COMPONENTS.json` (SHA-256 `4288e60c655892fec6a3cbaf2f369673b45d3e3b837b160e99d2861be8d1c999`)

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k1_reference_roundtrip_rejection_scorecard.py --check
```
