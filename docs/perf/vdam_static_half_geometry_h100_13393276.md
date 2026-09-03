# VDAM static half-spectrum host planning — H100 job 13393276

## Decision

Retain the shared host-planned static geometry in `e04c78bdc`.  On the
comparable optimized iteration-48 stack it removes 41 XLA compilations and
0.67 seconds of compile time without changing any of the 23 exact science
fields.  This is a focused compile-churn improvement, not a runtime-parity or
trajectory promotion.

The strict atomic-repeat diagnostic remains **14 / 21**: seven fields exceed
the two-repeat envelope, but only at the already observed nondeterministic
CUDA-reduction scale.  The largest normalized L2 delta is `1.65e-7`; the map
delta is `3.76e-9`.  This diagnostic is recorded and is not promoted to a
science pass.

## Comparable result

Both arms replay the same GF46 iteration-47 checkpoint, use the
`all_optimized_q32` contract with stable Fourier quantum 32, explicitly
disable batched posterior primitives, reuse the same native RELION capture,
and run on `della-h19g1` (H100).

| Metric | Baseline `13391819` | Candidate `13393276` | Change |
|---|---:|---:|---:|
| stderr XLA compilations | 435 | 394 | **-41 (-9.43%)** |
| stderr XLA compile time | 23.387638 s | 22.716554 s | **-0.671084 s (-2.87%)** |
| recorded compile misses | 429 | 389 | **-40 (-9.32%)** |
| recorded miss time | 22.595738 s | 21.918468 s | **-0.677270 s (-3.00%)** |
| cold profiled wall | 30.068897 s | 29.914641 s | -0.154257 s (-0.51%) |
| warm profiled wall | 5.144385 s | 4.569264 s | -0.575120 s (-11.18%) |

Cold and warm walls are diagnostic because cold compile attribution is
enabled.  The warm result is one paired replay, not a repeated timing claim.

The intended call sites account for the compile-count reduction:

| Attributed source | Baseline | Candidate | Change |
|---|---:|---:|---:|
| `helpers/half_spectrum.py` | 46 / 1.395239 s | 15 / 0.507513 s | **-31 / -0.887727 s** |
| `core/fourier_transform_utils.py` | 42 / 1.337610 s | 28 / 0.989602 s | **-14 / -0.348008 s** |
| `helpers/fourier_window.py` | 28 / 0.753335 s | 28 / 0.807977 s | live gathers unchanged |

The 45 removed static-geometry calls become a net 41-call reduction after
ordinary run-to-run attribution variation elsewhere in the process.

## Science and execution contracts

| Gate | Result |
|---|---|
| Required exact metadata | **12 / 12 exact** |
| Required exact scalar state | **10 / 10 exact** |
| Numeric data STAR content | **exact** |
| Combined exact fields | **23 / 23 exact** |
| Optimized execution contract | **cold exact; warm exact** |
| Atomic-repeat envelope | **14 / 21 inside; not promoted** |

The seven envelope misses are `class001_volume`,
`halfset_0_class_bpref_weight_sums`, `halfset_0_wsum_img_power`,
`halfset_0_wsum_sigma2_noise`, `model_star_numeric_content`, and the duplicated
global `wsum_img_power` and `wsum_sigma2_noise` summaries.  Their maximum
cross/envelope ratio is `1.4045x`; their maximum normalized L2 delta is
`1.6538e-7`.  Particle IDs, poses, translations, classes, posterior maxima,
significance counts, direction posterior sums, halfset stream, resolution,
sampling state, and the numeric data STAR are bitwise exact across all four
baseline/candidate cold/warm outputs.

The implementation also has a deterministic primitive proof: the focused
half-spectrum/Fourier unit slice passes `170 / 170`, including byte-exact
old-JAX-versus-host comparisons for even, odd, and rectangular shapes through
128.  Fresh-process instrumentation records zero JAX compilations for the
new static setup.  The runtime-stack cherry-pick passed the 13 newly added host
planning tests before submission.  No broad RECOVAR test suite was run.

## Provenance and reproduction anchor

- Candidate job: `13393276`, `COMPLETED`, `00:01:44`, `della-h19g1`.
- Candidate source: `21402b8592f9cc77ae837ef08a583b026c135fc0`, made by
  applying `e04c78bdc0592e432f829062809881951221f50c` to runtime-stack head
  `b8779efad8f527dc6e26b6ac9a6e1dd4235a9dff`.
- Candidate root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_static_geometry_runtime_stack_21402b859_20260903T1734Z`.
- Baseline job/root: `13391819`,
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_fused_half_expand_cold_71513bc86_20260903T1742Z`.
- Checkpoint optimiser SHA-256:
  `addfec8d36581f9c2e02504b29bbca979f9222132eb1a60ccd534a0ed3564e25`.
- Reused native Nsight SHA-256:
  `6d13813ddf22e6062163470978d8f2bd8e41f1fcacd23cc58c27e29bcc1b7ca4`.
- RELION binary SHA-256:
  `2d070d6456ae439c3890fcbd0f6e9c8e4e56bcc2ef79f55bfc58574e50f7a11b`.
- RELION binding SHA-256:
  `fcbb2a8356c2f7ee88e947fa92c9f5bfc41535ed0a2c6a9124a2fad781a63b83`.

From a clean worktree at candidate commit `21402b859`, the exact submission
shape is:

```bash
env \
  REPO_ROOT="$PWD" \
  VDAM_LATE_PROFILE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_static_geometry_runtime_stack_21402b859_20260903T1734Z \
  EXPECTED_REPO_HEAD=21402b8592f9cc77ae837ef08a583b026c135fc0 \
  VDAM_LATE_PROFILE_CHECKPOINT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/repeat-01/vdam-gf46/relion/run_it047_optimiser.star \
  VDAM_LATE_PROFILE_DATA_STAR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/repeat-01/vdam-gf46/relion/run_it047_data.star \
  VDAM_LATE_PROFILE_DATA_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/repeat-01/vdam-gf46/data \
  CHECKPOINT_ITERATION=47 NR_ITER_SCHEDULE=200 \
  RELION_BIND_BINARY_OVERRIDE=/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_vdam_gf10_first_boundary_precision_20260826/recovar/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so \
  EXPECTED_RELION_BIND_SHA256=fcbb2a8356c2f7ee88e947fa92c9f5bfc41535ed0a2c6a9124a2fad781a63b83 \
  EXPECTED_RELION_SHA256=2d070d6456ae439c3890fcbd0f6e9c8e4e56bcc2ef79f55bfc58574e50f7a11b \
  VDAM_LATE_PROFILE_REUSE_NATIVE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_optimized_nsys_it48_harnessfix_50a402026_20260903T1340Z \
  EXPECTED_REUSED_NATIVE_NSYS_SHA256=6d13813ddf22e6062163470978d8f2bd8e41f1fcacd23cc58c27e29bcc1b7ca4 \
  VDAM_LATE_PROFILE_CONTRACT=all_optimized_q32 \
  STABLE_FOURIER_WINDOW_SHAPES=1 \
  VDAM_LATE_PROFILE_COLD_COMPILE_ATTRIBUTION=1 \
  RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES=0 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.50 \
  sbatch scripts/run_vdam_late_iteration_profile.sbatch
```

Rebuild the compile summary with:

```bash
pixi run python -m scripts.analyze_vdam_cold_compile_attribution \
  --root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_static_geometry_runtime_stack_21402b859_20260903T1734Z \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_static_geometry_runtime_stack_21402b859_20260903T1734Z/provenance/cold_compile_attribution.json \
  --top 100
```

Job `13393079` is deliberately excluded: it ran `e04c78bdc` without the
runtime-stack lineage used by `13391819`.  Job `13393262` was a two-second
launcher failure caused by a mistyped expected commit and produced no
measurement.
