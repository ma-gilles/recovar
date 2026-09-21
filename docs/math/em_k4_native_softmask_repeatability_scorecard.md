# K=4 native soft-mask observer repeatability scorecard

This fixed-denominator same-A100 panel measures native RELION observer
repeatability after deterministic soft-mask block finalization. It is
non-scoring and cannot change the frozen cross-engine FSC-AUC scorecard.

Fixed gates: **13 / 14**.

| Checked | Fixed gate | Result |
| --- | --- | ---: |
| [ ] | `artifact_bytes_exact` | fail |
| [x] | `auxiliary_reduction_stream_markers_exact` | pass |
| [x] | `capture_headers_mpi2_openmp1_exact` | pass |
| [x] | `capture_validators_passed` | pass |
| [x] | `class_map_fsc_auc_at_least_threshold` | pass |
| [x] | `current_size_and_sampling_topology_exact` | pass |
| [x] | `dispatch_bytes_exact` | pass |
| [x] | `dispatch_row_count_exact` | pass |
| [x] | `no_fatal_runtime_pattern` | pass |
| [x] | `particle_count_exact` | pass |
| [x] | `powerclass_stream_markers_exact` | pass |
| [x] | `softmask_block_partial_markers_exact` | pass |
| [x] | `target_state_exact` | pass |
| [x] | `thread_replay_markers_exact` | pass |

Classification: `native_softmask_observer_highres_xi2_residual`.

Fine score and BPref were byte-exact. Fine operand differed by 3 bytes, solely at `op.highres_Xi2_img[img_id] / 2`: 3 float32 ULP (2.2351741790771484e-08). All 1520 per-pixel fields, lane partials, and selected production/replay raw `diff2` values were exact.

Signed normalized non-DC class-map FSC-AUC values were 0.9999999942598741, 0.9999999806617705, 0.9999999895280383, 0.9999999866926460 (threshold 0.999999).

Immutable evidence:

- `analysis_result`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_softmask_partials_pair_51a9392_20260804T0410ET/analysis/SOFTMASK_PAIR_RESULT_11990914.json` (SHA-256 `2a84d68edfb4067cfbe653f70df8a3ca3373a263e27fbc34281553699c55e724`)
- `residual_report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_softmask_partials_pair_51a9392_20260804T0410ET/analysis/SOFTMASK_RESIDUAL_11990914.json` (SHA-256 `1ae4303e5ca9713b387a465bccd02a3ea3b64aeeccb5b165348a871a3759fc15`)
- `analyzer`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_softmask_partials_pair_51a9392_20260804T0410ET/provenance/analyze_softmask_pair.py` (SHA-256 `244680137df5bd73f679c669e510881fbcd5a03274cacc090f6adc67c474a7ce`)
- `residual_analyzer`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_softmask_partials_pair_51a9392_20260804T0410ET/provenance/analyze_softmask_residual.py` (SHA-256 `576c4fdf0e66cd9d5e3e0d32ce0322d178c3ff6303b932a5cd6336f81d404ac9`)
- `science_completion`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_softmask_partials_pair_51a9392_20260804T0410ET/provenance/SCIENCE_PAIR_COMPLETE_11990914.json` (SHA-256 `aff22e261093e5226a438c9e9f9baecfd2486efe5a443b2ac222196fb1afcb2e`)
- `science_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_softmask_partials_pair_51a9392_20260804T0410ET/provenance/science_outputs_11990914.sha256` (SHA-256 `b4028eff1e26f063cc5bdbc40546c1c40a7da015db847d44e316a5d779edbe20`)
- `static_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_softmask_partials_pair_51a9392_20260804T0410ET/provenance/static_inputs_11990914.sha256` (SHA-256 `8365d276f13567490ff04a9f43595bd874a911c1fe85939654e079fe397297d5`)
- `postterminal_audit`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_softmask_partials_pair_51a9392_20260804T0410ET/provenance/POSTTERMINAL_AUDIT_11990914.md` (SHA-256 `976606b3d421010200dab9e5c5e4779ad36e68280df87f26cec076b1368f34df`)

To validate:

```bash
pixi run python scripts/summarize_em_k4_native_softmask_repeatability_scorecard.py --check
```
