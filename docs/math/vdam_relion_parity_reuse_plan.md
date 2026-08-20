# VDAM / InitialModel strict RELION parity reuse plan

This work starts from the existing `recovar.em.initial_model` implementation on
PR #158 head `b10412ca`. It is not a rewrite. The goal is to preserve the
VDAM-specific controller and reuse the supplied-map EM parity stack at every
shared numerical boundary.

## Reuse boundary

| InitialModel stage | Existing VDAM owner | Shared EM implementation to retain |
|---|---|---|
| GUI/default schedules and subset order | `initial_model/schedules.py`, `subset.py` | RELION rounding and sampling helpers |
| Bootstrap reference and initial noise | `initial_model/bootstrap_iref.py`, `avg_unaligned.py` | RELION bindings and common image preprocessing |
| Coarse significance | `initial_model/dense_adapter.py` | `_compute_k_class_significance_batched` |
| Fine posterior and reconstruction operands | `initial_model/dense_adapter.py` | `run_local_k_class_em` and sparse-pass-2 helpers |
| Projection and score windows | adapter configuration only | dense EM projection, Fourier-window, constraint, and posterior code |
| BPref accumulation | adapter layout conversion | shared RELION x-half accumulation path |
| VDAM moment update | `initial_model/m_step.py` | RELION reconstruction bindings; VDAM moments remain VDAM-specific |
| Noise, class/direction priors, and sigma offsets | `initial_model/iteration_loop.py` | common state conventions and RELION units |
| Solvent flattening and artifacts | `initial_model/iteration_loop.py`, `driver.py` | common RELION masks and STAR/MRC conventions |

The InitialModel controller must remain separate from supplied-map auto-refine:
VDAM has one class reference with pseudo-halfsets, persistent gradient moments,
pad 1, a subset/stepsize/tau schedule, and an EM tail. Those differences are
algorithmic, not duplicated implementation.

## Frozen progress contract

The v1 suite in `vdam_relion_parity_scorecard_v1.json` reuses twelve immutable
fixtures from the PR #158 EM manifest. It freezes K=1, eight iterations, GUI
sampling defaults, and checkpoints 0/1/2/4/8. The fixed cases span baseline,
noise, anisotropy, no-CTF, contrast, image offsets, outliers, high resolution,
mid-scale, and production-scale inputs.

A case passes only when all of the following hold:

- RECOVAR and RELION run on the same physical GPU from recorded source bytes;
- schedule, subset, sampling, and artifact topology checks pass at every fixed
  checkpoint;
- signed normalized non-DC cross-engine FSC-AUC is at least 0.999;
- RECOVAR-minus-RELION GT FSC-AUC is at least -0.002;
- no correlation metric participates in map acceptance.

Historical May 2026 correlation results are useful localization evidence but
do not score under this contract.

## Test ladder

1. Pure unit parity: schedules, rounding, subset/pseudo-halfset assignment,
   layouts, priors, state updates, and scorecard immutability.
2. Shared-engine contract tests: coarse significance, local fine pass,
   projector frame, BPref frame, and accumulator conversion.
3. Bound fixture replay: bootstrap, preprocessing, posterior, BPref, and VDAM
   M-step operands against RELION dumps.
4. Short same-GPU trajectories at checkpoints 0/1/2/4/8 with shellwise FSC,
   FSC-AUC, particle-state, schedule, and topology reports.
5. Frozen twelve-case Slurm suite plus a separate non-scoring K>1 diagnostic
   trajectory. K>1 cannot change the K=1 denominator.

Thresholds, fixture identities, and the denominator require a new suite
version. Failed cases stay visible and are fixed in code rather than hidden by
tolerance or baseline changes.

## Reproducible case runs

Each case is materialized from the existing immutable EM fixture manifest,
then RELION InitialModel and RECOVAR VDAM run sequentially on the one GPU in a
Slurm allocation. The audit requires the exact frozen CLI contract, common
iteration artifacts, one recorded physical GPU UUID, and signed FSC/FSC-AUC at
all five checkpoints.

```bash
OUTPUT_ROOT=/scratch/gpfs/GILLES/mg6942/vdam_relion_parity/<run-id>
sbatch \
  --export=ALL,CASE_ID=vdam-08,OUTPUT_ROOT="$OUTPUT_ROOT" \
  scripts/run_vdam_relion_parity_case.sbatch
```

The case directory contains verified fixture symlinks, complete engine logs,
exact argv JSON, timing/provenance records, `trajectory_audit.json`, and the
shellwise curves in `trajectory_shellwise_fsc.npz`. A scientifically completed
case is allowed to exit nonzero when it misses a fixed parity gate; the report
and failed checkpoint remain the progress evidence.
