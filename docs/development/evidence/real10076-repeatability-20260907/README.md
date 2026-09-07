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

## Next diagnostic boundary

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

Compare pre-join operands between unchanged-source repeats first. If they
already differ, move into scoring and accumulation. If they agree but post-join
operands differ, replay the join with identical inputs and layout metadata.
Capture introduces host synchronization, so diagnostic repeatability does not
automatically qualify ordinary execution. Any shortened diagnostic also needs
its first-iteration scheduling checked against the autonomous run; it cannot
replace the complete trajectory gate.
