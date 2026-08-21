# K=4 preprocessing replay scorecard

This fixed-denominator diagnostic localizes numerical repeatability.
It cannot change the frozen K=1 or K=4 FSC/FSC-AUC scorecards.

Bitwise exact: **3 / 9**. Within fixed material floor: **9 / 9**.

| Checked | Case | Stage | Exact | Relative L2 | Max abs |
| --- | --- | --- | ---: | ---: | ---: |
| [x] | `normalized-shifted-repeat-1` | `normalized_shifted_real` | yes | 0 | 0 |
| [x] | `normalized-shifted-repeat-2` | `normalized_shifted_real` | yes | 0 | 0 |
| [x] | `normalized-shifted-repeat-3` | `normalized_shifted_real` | yes | 0 | 0 |
| [x] | `masked-real-repeat-1` | `masked_real` | no | 4.25812225e-10 | 1.49011612e-08 |
| [x] | `masked-real-repeat-2` | `masked_real` | no | 7.40006914e-10 | 2.98023224e-08 |
| [x] | `masked-real-repeat-3` | `masked_real` | no | 5.96995261e-10 | 1.49011612e-08 |
| [x] | `masked-fourier-repeat-1` | `masked_fourier` | no | 4.2525044e-08 | 6.2913599e-05 |
| [x] | `masked-fourier-repeat-2` | `masked_fourier` | no | 2.33415553e-08 | 4.31583729e-05 |
| [x] | `masked-fourier-repeat-3` | `masked_fourier` | no | 4.55946243e-08 | 6.2913599e-05 |

Classification: `softmask_background_reduction_drift_within_fixed_material_floor`.

Fixed material relative-L2 threshold: `5e-07`.

Immutable evidence:

- `report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_preprocess_replay_retry2_f6b69788_20260731T1013ET/analysis/PREPROCESS_REPLAY.json` (SHA-256 `2059de0e8487e2b7dc7f13f94fffe87bdb801c17ccc377924ec297ea783de146`)
- `replay_arrays`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_preprocess_replay_retry2_f6b69788_20260731T1013ET/analysis/PREPROCESS_REPLAY_ARRAYS.npz` (SHA-256 `123c51379bd563d9b22f45d7a797dc3cc6949f93d4f71ec376c347d71429fd74`)
- `sealed_input`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_sameallocation_dynamicref_retry2_20260731T0520ET/reference_continue/capture/contribution/bpref_contribution_rows_it002_h1_call000000_dump000000_cs038.npz` (SHA-256 `98c8642d7b85645f6416aa834eef931d3561e3db651111cd5d22cbd6ff7e5c0b`)

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k4_preprocess_replay_scorecard.py --check
```
