# K=4 native high-resolution Xi2 observer repeatability scorecard

This fixed-denominator same-A100 panel measures native RELION observer
repeatability after deterministic high-resolution Xi2 block finalization.
It is non-scoring and cannot change the frozen cross-engine FSC-AUC scorecard.

Fixed gates: **15 / 15**.

| Checked | Fixed gate | Result |
| --- | --- | ---: |
| [x] | `artifact_bytes_exact` | pass |
| [x] | `auxiliary_reduction_stream_markers_exact` | pass |
| [x] | `capture_headers_mpi2_openmp1_exact` | pass |
| [x] | `capture_validators_passed` | pass |
| [x] | `class_map_fsc_auc_at_least_threshold` | pass |
| [x] | `current_size_and_sampling_topology_exact` | pass |
| [x] | `dispatch_bytes_exact` | pass |
| [x] | `dispatch_row_count_exact` | pass |
| [x] | `highres_xi2_block_partial_markers_exact` | pass |
| [x] | `no_fatal_runtime_pattern` | pass |
| [x] | `particle_count_exact` | pass |
| [x] | `powerclass_stream_markers_exact` | pass |
| [x] | `softmask_block_partial_markers_exact` | pass |
| [x] | `target_state_exact` | pass |
| [x] | `thread_replay_markers_exact` | pass |

Classification: `deterministic_thread_highres_xi2_partial_replay_pair_exact`.

Fine score, fine operand, and BPref artifacts were byte-exact. The selected target's hard pose, class, shift, Pmax, and support state were exact.

Across all particles, hard pose/class/shift remained exact, but 13 Pmax values and 15 support counts differed; the largest support-count delta was 1. Therefore this admits stable native operand localization, not joint posterior/BPref/map parity.

Signed normalized non-DC class-map FSC-AUC values were 0.9999999961725112, 0.9999999807100690, 0.9999999921474709, 0.9999999852511000 (threshold 0.999999).

Slurm job 11992900 is excluded because its wrapper missed the predeclared powerClass marker and completed only arm A.

Immutable evidence:

- `analysis_result`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_highres_xi2_partials_pair_retry_17a9769_20260804T0555ET/analysis/HIGHRES_XI2_PAIR_RESULT_11993105.json` (SHA-256 `ee98144916d69ac618b8696176a8ec84d97d1c9d7c6dbfa9c3b0632235ecb900`)
- `analyzer`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_highres_xi2_partials_pair_retry_17a9769_20260804T0555ET/provenance/analyze_highres_xi2_pair.py` (SHA-256 `29a9ab99d0ff57619249d03dc11554fbd1fc24d89bff0a70a09164464d9ab003`)
- `science_completion`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_highres_xi2_partials_pair_retry_17a9769_20260804T0555ET/provenance/SCIENCE_PAIR_COMPLETE_11993105.json` (SHA-256 `b78436fb5b88c493cfe077f0e1a9f3b54561bf9a5197652af7d582c276d02f79`)
- `analysis_completion`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_highres_xi2_partials_pair_retry_17a9769_20260804T0555ET/provenance/HIGHRES_XI2_ANALYSIS_COMPLETE_11993105.json` (SHA-256 `d1383debda55421ce1348f9facbc68c5c7b550d8af3cc62ac7cf7424da2c5d61`)
- `science_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_highres_xi2_partials_pair_retry_17a9769_20260804T0555ET/provenance/science_outputs_11993105.sha256` (SHA-256 `0af13e1ed0a109842b275e85976907265d89f7834d1a3a7fb158a1b87366f787`)
- `static_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_highres_xi2_partials_pair_retry_17a9769_20260804T0555ET/provenance/static_inputs_11993105.sha256` (SHA-256 `05f53af5ff3bd15b8e12c152d6a6fed222f7a333773c78c4d7e8ea157380adb5`)
- `postterminal_audit`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_highres_xi2_partials_pair_retry_17a9769_20260804T0555ET/provenance/POSTTERMINAL_AUDIT_11993105.md` (SHA-256 `27ac3362cde4ddfb324405817562ad1d567b4795f1424d3851e2cea68e966009`)
- `binary`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k4_highres_xi2_marker_829f3bc_20260804T0551ET/build/bin/relion_refine_mpi` (SHA-256 `01e5ee2bd1db2612e374a21060dd7b4b9bd72c3cccea86f9d0225102082849da`)
- `build_completion`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k4_highres_xi2_marker_829f3bc_20260804T0551ET/provenance/BUILD_COMPLETE_11993050.json` (SHA-256 `0a05f99fd5d8fabfaebace7c88307caeb37ddce36b33a886cbf34fcf824d08f3`)
- `build_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k4_highres_xi2_marker_829f3bc_20260804T0551ET/provenance/build_outputs_11993050.sha256` (SHA-256 `f968e86a12ef7b77442c8c8121840906cae2e8e34334a9ddf4303d0fa1ab629e`)

To validate:

```bash
pixi run python scripts/summarize_em_k4_native_highres_xi2_repeatability_scorecard.py --check
```
