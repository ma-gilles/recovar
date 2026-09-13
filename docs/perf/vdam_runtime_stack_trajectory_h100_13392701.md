# VDAM optimized-stack trajectory — H100 job 13392701

## Decision

The stable Fourier-window and fixed flat-row ABIs are a **runtime pass and
strict trajectory hold** on the complete optimized stack through iteration 50.
They reduce median fresh-process wall time by 45.15% and expectation time by
49.26%, but one of two candidate repeats changes one pose assignment at
iterations 48 and 50.  The batch job therefore correctly exits nonzero and
writes `SCIENCE_FAILED`; this result cannot promote defaults or change the
frozen runtime/correctness scores.

All four trajectory processes and the analyzer completed.  Slurm reports the
job as `FAILED` only because the final science gate deliberately returns 1
when its strict contract fails.

## Runtime result

Every arm uses the same optimized K=1 stack, including certified compact
coarse scoring, packed/deferred local work and noise, the batched posterior
primitives, the K=1 result bypasses, and fused half-volume expansion.  The only
candidate delta is stable Fourier-window shapes at quantum 32 plus fixed
`B * R` flat-row capacity.

| Arm | Stable shapes/rows | Fresh wall | Expectation | Peak GPU memory |
|---|---:|---:|---:|---:|
| `stable_off_1` | off | 553.109 s | 523.763 s | 17,063 MiB |
| `stable_on_1` | on | 286.072 s | 244.118 s | 17,055 MiB |
| `stable_on_2` | on | 263.066 s | 234.147 s | 17,083 MiB |
| `stable_off_2` | off | 448.098 s | 418.736 s | 17,063 MiB |
| control median | off | 500.604 s | 471.249 s | 17,063 MiB |
| candidate median | on | 274.569 s | 239.133 s | 17,069 MiB |
| candidate change | | **-45.15% (1.823x)** | **-49.26% (1.970x)** | **+0.04%** |

The candidates use three physical Fourier sizes for 19 logical sizes.  Their
290 local chunks all use the fixed flat-row capacity; the controls exercise
152 strict row-capacity reductions across 202 chunks.  Runtime, expectation,
memory, and execution-contract checks pass.

## Science result

| Boundary | Result |
|---|---|
| Control repeat hard pose/class state | exact through iteration 50 |
| Candidate 1 versus both controls | exact hard pose/class state through iteration 50 |
| Candidate 2 pose assignments | one particle differs at iterations 48 and 50 |
| Best rotation IDs and class assignments | exact in all arms through iteration 50 |
| Maximum checkpoint-map normalized L2 | `1.184e-5` |
| Whole-trajectory map RMS normalized L2 | `1.35e-6` to `5.84e-6` across pairs |
| Fixed `4*float32-epsilon` numerical gate | 183 failures; **not passed** |

At iteration 48, candidate 2 changes local particle index 70 from pose
`6745870` to `6745867`; the best rotation ID is unchanged and the translation
changes from `[0.29807943, 0.48150709]` to
`[0.29807943, 0.29807943]`.  At iteration 50, local particle index 59 changes
from pose `6775616` to `6775582`, again with the same best rotation ID but a
different translation.

This is not a deterministic feature-on displacement: candidate 1 remains
hard-state identical to control 1, while the two ordinary controls already
differ in thresholded `significant_counts` at iterations 31, 39, 42, 48, and
49.  Candidate 1 shares those same support-count differences relative to
control 1.  Candidate 2 accumulates a different repeat trajectory and first
changes one support count at iteration 18 before crossing a pose boundary at
iteration 48.  The evidence is consistent with repeat-sensitive accumulated
floating-point/CUDA reduction noise, but the observed hard split means the
strict trajectory gate remains failed.

The dedicated iteration-48 batched-posterior ABBA gate is independent evidence:
it keeps hard state exact, bounds continuous state by eight float32 epsilons,
removes 2,040 kernels, and changes capture span `1.629 -> 1.410 s`.  Job
`13392701` enables that primitive in every arm, so its stable-shape comparison
is not an on/off test of batched posterior behavior.

## Provenance and reproduction anchor

- Job: `13392701`; H100 node `della-h20g1`; runtime `00:27:49`.
- Source: clean commit
  `b8779efad8f527dc6e26b6ac9a6e1dd4235a9dff` on
  `codex/vdam-runtime-stack-20260903`.
- Output root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_runtime_stack_allten_gf46_it50_b8779efad_20260903T1820Z`.
- CUDA artifact SHA-256:
  `36a11c669006df579a440f5b2e4b2bd4c8c33970e3411b4c60912d93f7799b00`.
- RELION binding SHA-256:
  `fcbb2a8356c2f7ee88e947fa92c9f5bfc41535ed0a2c6a9124a2fad781a63b83`.
- Analyzer SHA-256:
  `b870348f3d9f5265eb38466184b9613e53fc384eec5e696ffddb78a9690ecfca`.

From the clean pinned worktree at `b8779efad`, reproduce with:

```bash
sbatch --parsable --nodelist=della-h20g1 \
  --export=ALL,VDAM_STABLE_TRAJECTORY_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_runtime_stack_allten_gf46_it50_b8779efad_20260903T1820Z,VDAM_STABLE_TRAJECTORY_REPO_ROOT="$PWD",VDAM_STABLE_TRAJECTORY_EXPECTED_HEAD=b8779efad8f527dc6e26b6ac9a6e1dd4235a9dff,VDAM_STABLE_TRAJECTORY_CUDA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_batched_posterior_late_abba_77a88dac7_20260903T1648Z/runs/control_1/runtime/cuda/libcuda_backproject.so,VDAM_STABLE_TRAJECTORY_RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_batched_posterior_late_abba_77a88dac7_20260903T1648Z/runs/control_1/runtime/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so,VDAM_STABLE_TRAJECTORY_LAST_ITERATION=50,VDAM_STABLE_TRAJECTORY_STABLE_FOURIER_WINDOW_SHAPES=1,VDAM_STABLE_TRAJECTORY_STABLE_FOURIER_WINDOW_QUANTUM=32,VDAM_STABLE_TRAJECTORY_STABLE_FLAT_ROW_CAPACITY=1,RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES=1,RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR=1,RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE=200,RECOVAR_EXACT_LOCAL_FUSED_PAIR_FINE_SCORE=0,RECOVAR_INITIAL_MODEL_PACKED_FINAL_NOISE=1,RECOVAR_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE=1,RECOVAR_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS=1,RECOVAR_K1_RELION_EXACT_COMPACT_PREPROCESS=1 \
  scripts/run_vdam_stable_shape_trajectory.sbatch
```

The output root is intentionally bulky scratch data and contains a
`SAFE_TO_DELETE` marker.  No broad RECOVAR test suite was run.
