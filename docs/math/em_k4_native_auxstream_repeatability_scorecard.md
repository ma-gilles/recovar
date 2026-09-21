# K=4 native aux-stream repeatability scorecard

This fixed-denominator same-A100 diagnostic tests whether native CUDA
auxiliary streams make a selected RELION observer byte-repeatable. It is
non-scoring and cannot change the frozen FSC/FSC-AUC quality scorecards.

Fixed gates: **12 / 13**.

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
| [x] | `target_state_exact` | pass |
| [x] | `thread_replay_markers_exact` | pass |

Classification: `native_auxstream_observer_not_byte_repeatable`.

The first captured unequal boundary was preprocessing followed by native fine scoring: 460/1520 real and 470/1520 imaginary values differed, then 16735/109184 raw `diff2` values differed (maximum absolute delta 0.0001220703125).

Signed normalized non-DC class-map FSC-AUC values were 0.9999999895278622, 0.9999999846443698, 0.9999999749836932, 0.9999999778593218 (threshold 0.999999).

Immutable evidence:

- `analysis_result`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_auxstream_pair_retry_a609886_20260804T0304ET/analysis/AUXSTREAM_PAIR_RESULT_11988750.json` (SHA-256 `bc68328ef3c97f996018c4992ad69a3748c47eaca98f44e5b8c062f8c0fb8a57`)
- `analyzer`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_auxstream_pair_retry_a609886_20260804T0304ET/provenance/analyze_auxstream_pair_v2.py` (SHA-256 `814b9785e068713852951a901e26d3dc07fc17ad612df743ffc0b5c1870a952a`)
- `science_completion`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_auxstream_pair_retry_a609886_20260804T0304ET/provenance/SCIENCE_PAIR_COMPLETE_11988750.json` (SHA-256 `6463dea3c1b879097d27ae9e9338b3a18159c0e286421d6e6d27ad14d2fc5d9e`)
- `analysis_completion`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_auxstream_pair_retry_a609886_20260804T0304ET/provenance/AUXSTREAM_PAIR_ANALYSIS_COMPLETE_11988750.json` (SHA-256 `2e679318eb603d75fdd2f73105096257ad5274118fdee89a7e7ae9530dcfbfcf`)
- `science_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_auxstream_pair_retry_a609886_20260804T0304ET/provenance/science_outputs_11988750.sha256` (SHA-256 `bd3625b4c3cb57e272395c15687c9c3787481d9712a1f61fdaec569d142ae09b`)
- `static_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_auxstream_pair_retry_a609886_20260804T0304ET/provenance/static_inputs_11988750.sha256` (SHA-256 `05707af0b37847264215238d7e72049a97acc996568048c2f0e93297193b86e8`)

Code references:

- `scripts/summarize_em_k4_native_auxstream_repeatability_scorecard.py`
- `scripts/report_em_parity_progress.py`

To validate:

```bash
pixi run python scripts/summarize_em_k4_native_auxstream_repeatability_scorecard.py --check
```
