# RELION parity quick handoff - 2026-04-27

Read this first. The long status file is the audit trail, not the starting
point for the next debugging session.

## Active checkout

- Repo: `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2_rebase_20260424`
- Branch: `claude/relion-parity-local-search-fix`
- RELION source: `/scratch/gpfs/GILLES/mg6942/relion`
- RELION patched build: `/scratch/gpfs/GILLES/mg6942/relion/build_patched`

## Goal

Achieve near-perfect RECOVAR/RELION EM parity:

- Quality: match RELION poses, translations, Pmax/posteriors, maps, tau2,
  noise, FSC, and intermediate accumulators.
- Speed: close to RELION where practical, and no avoidable per-image Python or
  poor bucket reuse.
- Method: compare RELION source and dumps directly. Do not tune parameters to
  make metrics look better.

For EM parity work, do not run the full RECOVAR-wide test suite. It is not
relevant to this branch. Use targeted EM/parity tests and targeted Slurm runs.

## Current best result

Best post-fix fixed-state 5k replay:

`_agent_scratch/recovar_it008_5k_priorfix_20260427_031136`

Dataset:

`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star`

Result:

- RELION it008, 5k particles, box 128, A100 80GB.
- Wall time: `239.8s`.
- Ave Pmax: RECOVAR `0.8854237`, RELION `0.8854160`.
- Pmax abs diff: mean `1.23e-4`, median `1.87e-5`, p99 `1.04e-3`,
  max `0.02667`.
- Pose parity: mean angular error `5.7e-6 deg`, max `2.0e-5 deg`.
- Translation parity: mean `1.3e-6 px`.
- Map corr RECOVAR-vs-RELION merged: `0.999952`.
- GT merged corr: RECOVAR `0.965163`, RELION `0.965239`.
- FSC0.143 shell: RECOVAR `42`, RELION `42`.

## Fix achieved

The latest important fix is matching RELION accelerated `pdf_offset` units for
the local translation prior.

Main files:

- `recovar/em/dense_single_volume/helpers/orientation_priors.py`
- `recovar/em/dense_single_volume/iteration_loop.py`
- `tests/unit/test_refine_relion_mode.py`

This fixed the former particle-256 Pmax outlier. After the fix, particle 256
went from roughly `0.943` vs `0.349` to `0.3514` vs `0.3493`, with matching
pose/translation.

## Current open problem

Next target is particle original index `4603` in the same 5k normalized
dataset.

Artifacts:

- RECOVAR focused run:
  `_agent_scratch/focused_it008_stack4603_priorfix_20260427_031644`
- RECOVAR operand dump:
  `_agent_scratch/focused_it008_stack4603_operands_20260427_032510`
- RELION dump:
  `_agent_scratch/relion_dump_stack4604_it008_20260427_031754/dump`

Known facts:

- Pose and translation match RELION to numerical noise.
- The remaining gap is score/posterior-level.
- Full 5k Pmax for particle 4603: RECOVAR `0.606924`, RELION `0.580255`.
- RELION dump Pmax is `0.576184`, close to the RELION STAR comparison row.
- RECOVAR makes the top-vs-second candidate logit gap too sharp by about
  `0.136` log units.
- Forcing RECOVAR float32 scoring did not fix it.
- Switching RECOVAR scoring to square Fourier window worsened it.

Next debugging step:

Compare RELION `fine_ref`, `fine_shifted`, and `corr_img` against RECOVAR
`debug_proj_weighted`, `debug_shifted_score`, and `debug_ctf2_over_nv` for the
six active candidates of particle 4603. This should identify whether the
remaining score gap is interpolation, boundary/support, or weighting.

## Full-trajectory and large-run caveats

Complete 5k trajectory artifact:

`_agent_scratch/long_end2end_parity_20260426_182134`

This is useful as timing/performance baseline, but it predates the latest
translation-prior fix. Do not use it as final parity evidence.

Post-fix forced long run:

`_agent_scratch/long_5k_force13_priorfix_20260427_033007`

It emitted 7 rows, then failed trying to read missing
`run_it015_sampling.star`. Treat it as timing-only until the runner is capped
to available RELION refs.

Large background runs:

- `data_noise1_100k`: job `7385332`, launched from older snapshot, still
  running at last check.
- `data_grid192_100k`: job `7385331`, failed with CUDA illegal address during
  local search.

Do not use either large run as final post-fix parity evidence.

## Where the long audit trail lives

Detailed history and older experiments:

`docs/math/relion_parity_current_status_2026_04_25.md`

Use that file only when you need exact historical context or artifact names.
