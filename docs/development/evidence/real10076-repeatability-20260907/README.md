# Real 10076: unchanged-source repeatability failure

Two autonomous PR158 runs on the same physical H100 fail the standing 0.995
direct-map FSC-AUC gate from iteration 8. Both execute 16 numbered iterations;
their eight compared controller fields agree. Final merged-map FSC-AUC is
0.964376533. This negative control prevents attribution of a single candidate
mismatch to cleanup alone. It does not waive any quality gate.

## First saved state

[The array audit](first_saved_state.json) compares 15 arrays from the first
numbered iteration (`it000`). It verifies every input against its original run
digest and rechecks all 30 files after comparison. No GPU workload was rerun.

| Field | Changed elements | Maximum absolute difference |
| --- | ---: | ---: |
| `Ft_y_0` | 4,272 | 2.9802322387695312e-8 |
| `Ft_y_1` | 4,554 | 1.501712528671799e-8 |
| `Ft_ctf_0` | 590 | 2.3283064365386963e-10 |
| `Ft_ctf_1` | 650 | 2.3283064365386963e-10 |

The remaining 11 fields are exact: `noise`, `noise_half1`, `noise_half2`,
`tau2`, `fsc`, `rotations`, `translations`, `ha_half1`, `ha_half2`,
`coarse_ha_half1`, and `coarse_ha_half2`. Exact saved grid and assignment arrays
do not establish equality of every candidate probability or scoring operand.

The controller calls `_save_iteration_intermediates` after the low-resolution
half join, regularized reconstruction, and optional unregularized
reconstruction. These files therefore locate an observed boundary; they are
not a complete capture of the inputs to the first E-step. The audit does not
identify the responsible kernel or prove a causal link to later trajectory
divergence. A separate particle-field audit first finds support/Pmax
differences in iteration 2 and two 0.5-pixel translation changes in iteration 3.

## Source and reproduction

- Source for both runs: `44d770de3f9336ab2f3f6a34203394bae8d1aeed`.
- Execution: Slurm13562724; trajectory audit: Slurm13562837; saved-array
  audit: Slurm13575086 (CPU, 1.442 seconds).
- GPU: H100, `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`.
- Original output root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/hia_pr158_real10076_control_repeats_20260907/`.
- Array audit output root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/hia_real10076_first_saved_state_audit_20260907/`.
- Exact commands, environment, fixture/source identities and audit scripts:
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/real10076_control_repeats/`.

Use that directory's `pair_inputs.json`, the two repeat manifests and original
Slurm script to recreate the run in fresh output directories. The array audit
records its script digest and all consumed file digests. This archive preserves
observations, not new expected baselines or a performance qualification.

## First-iteration boundary diagnostic

[Three shortened runs](boundary_run_record.json) at the same unchanged source
completed in Slurm13576748 (669.83 seconds including setup, execution and
identity checks). The only scientific option changed from the autonomous run
was `--max_iter 999` to `--max_iter 1`. The first run retained original capture
settings; the next two added both boundary captures below. All three used the
same physical H100, `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`, with independent
empty caches and private immutable CUDA copies.

[CPU audit13576847](first_iteration_boundaries.json) took 4.32 seconds and
verified all 49 consumed files against execution-time hashes, then rechecked
them after comparison. The two boundary-capture runs already differ in all four
pre-join buffers. Their maximum numerator differences are 2.9802322e-8 and
1.4901161e-8; maximum weight differences are 2.3283064e-10 and 1.1641532e-10.
Within each run, all four post-join buffers exactly equal the later saved arrays.

The remaining 11 fields match across all three runs. They also
[match the original autonomous run by file hash](historical_first_iteration_identity.json),
giving 33 historical file matches. The shortened runs reproduce those recorded
first-iteration states, but have a different GPU UUID from the historical pair
and do not prove complete E-step input identity or autonomous repeatability.

This result places the observed difference upstream of the half join. It does
not identify a responsible kernel or explain the later particle/trajectory
failure. Original-capture versus boundary-capture runs also differ in the four
accumulators; neither capture calibration nor a null capture result may waive
the full trajectory gate.

## Capture settings and next replay

PR158 already supports two captures around the low-resolution half join:

```bash
export RECOVAR_BPREF_PREJOIN_DUMP_DIR="$RUN_DIR/prejoin"
export RECOVAR_BPREF_ACCUM_DUMP_DIR="$RUN_DIR/postjoin"
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=1
export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID="$RUN_ID"
```

The iteration selector is one-based. The pre-join file uses schema
`recovar-bpref-prejoin-v2`; the post-join file uses `recovar-bpref-accum-v2`.
The writer preserves numerator dtype and stores the real part of the weight
arrays. Keep original dtype information when checking a stage boundary.

The next replay needs identical scoring/accumulation operands, the actual
`mstep_max_r`, original particle/rotation identities, all valid rows including
zeros, and the production fused per-particle launch boundaries. The existing
`replay_bpref_contribution_bundle.py` compares active-row order and precision
using shard partitions and separate data/weight launches; it is not an exact
replay of this production path. It also derives support from `current_size`
instead of the separately captured M-step radius, which can differ during
firstiter-CC. Preserve these limitations when selecting a replay tool.

Capture introduces host synchronization, so diagnostic repeatability does not
automatically qualify ordinary execution. These shortened runs cannot replace
the complete trajectory gate.

## Completed fixed-input follow-up

The subsequent [operand capture and production accumulation replay](../real10076-scatter-repeatability-20260907/README.md)
completed on unchanged PR158. Captured half-1 operands agree exactly; three
identical-input accumulation trials produce different native outputs, including
the warm pair. The linked archive preserves this narrower finding and its
limitations. Its relationship to the autonomous trajectory failure remains open.
