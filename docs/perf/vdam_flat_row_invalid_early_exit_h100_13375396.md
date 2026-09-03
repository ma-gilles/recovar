# VDAM flat-row invalid-row early exit — H100 jobs 13375364/13375396

## Decision

Retain the `row_image_ids == -1` early exit in
`relion_fine_diff2_fused_translate_runtime_flat_rows_f32`. On the
GF46 iteration-48 shape it is bitwise exact for every active row and reduces
the isolated four-call kernel wall time by 69.07%, a 3.23286x speedup.

This is a small kernel win, not the source of the end-to-end performance gap:
the absolute median saving is only 0.02673 seconds for the complete four-call
suite at this trajectory point.

## Qualification

| Field | Value |
|---|---|
| Source | `4dc4a79f7c5f6f31d7a8e04bfe43339b700b48c6` |
| Successful Slurm gate | `13375396` (`COMPLETED`, exit `0:0`, elapsed 24 s) |
| Hardware | `della-h19g1`, NVIDIA H100 80GB HBM3, `GPU-75c2d200-95d1-ef57-fb52-1698386c756c` |
| Kernel | `relion_fine_diff2_fused_translate_runtime_flat_rows_f32` |
| Shape source | immutable job `13374637`, GF46 checkpoint 47 to iteration 48 |
| Workload | four calls x 4,752 rows = 19,008 physical rows; 5,584 logical rows; `T=196`; `F=2,834`; `current_size=84` |
| Exactness | active outputs bitwise identical; candidate repeat bitwise identical; all 2,631,104 invalid scalar outputs exact positive infinity |
| Worktree | clean before and after the GPU gate |

The immutable source artifact reports `T=196`; it does not confirm the
provisional `T=148` assumption.

## Warm ABBA timing

The gate collected 24 samples per arm in 12 ABBA cycles. Each sample executes
all four production-shaped calls and blocks every output.

| Arm | Median (s) | Mean (s) | p05-p95 (s) | MAD (s) |
|---|---:|---:|---:|---:|
| Padding rows retain valid image IDs | 0.038706451 | 0.038709441 | 0.038660091-0.038756405 | 0.000023307 |
| Padding rows use `-1` | 0.011972814 | 0.011982127 | 0.011960076-0.012027726 | 0.000012672 |

- Median speedup: 3.23286x.
- Median wall reduction: 69.0677%.
- Paired-cycle mean speedup: 3.23060x.
- Paired-cycle bootstrap 95% speedup interval: 3.22752x to 3.23352x.
- Absolute median saving: 0.026733637 seconds per four-call suite.

Sampled peak GPU memory was 1,591 MiB. JAX reported peak bytes in use of
555,745,792 bytes.

## Failed-launch classification

Job `13375364` failed after 15 seconds before any benchmark measurement. The
runtime GPU/import gate passed, but invoking the external benchmark by file
path omitted the repository from Python's import path and raised
`ModuleNotFoundError: No module named 'recovar'`. Job `13375396` used the same
benchmark, source artifact, and CUDA binary through `runpy` from the repository
working directory. This is an infrastructure-only launcher failure and carries
no scientific or performance result.

## Immutable provenance

- Result root:
  `/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_invalid_row_early_exit_h100_retry1_4dc4a79f7_20260903T052309Z`.
- CUDA binary SHA-256:
  `d8253dc6dcd8420bca3e1935ffe4318df301fc83a96da932cb46dc91f31aa851`.
- Source artifact SHA-256:
  `5ec34cec6f774473b9b432c3f75a915929ccc4323b0c8b82642579f773571cc9`.
- Benchmark script SHA-256:
  `060e46078e0ac3fbcfce535aa2bd1cf4609c7eb07216256c41cb0f061d6e8ce0`.
- Source manifest SHA-256:
  `c34906f5383de86983852a8f7eeacbe871c9942b14e77922d2d75abe0b75bc5d`.
- Benchmark JSON SHA-256:
  `7addff54ed325613cb542d20ceb2377f4a7c9c62256244f748441b14414f4f49`.
- Run JSON SHA-256:
  `1885ae9caf4d7f0fb08d44db52ad53c91ca6b9dd4e4f6ab131879af014d61d70`.
- Aggregate `SHA256SUMS` manifest SHA-256:
  `d89361d388017f55ec77f9948673c6de6307594ab4f55edde697b4a88b2e5481`.

The aggregate manifest verifies successfully. The benchmark is a focused
kernel microbenchmark only and does not authorize trajectory promotion by
itself.
