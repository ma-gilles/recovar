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

## Current best results

Full 5k/128 replay after pre-shift/pass-2 fixes:

`_agent_scratch/long_5k_exactlocal_2e2d8301_local_20260427_093922`

Dataset:

`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star`

Result:

- Command emitted 13 RECOVAR rows matching RELION it002 through it014.
- Total wall time: `2956.8s` on one visible A100.
- Final map parity: half1/half2 corr vs RELION `0.999884/0.999883`.
- Final merged GT corr: RECOVAR `0.966522`, RELION `0.966607`.
- Final merged FSC0.143 shell: RECOVAR `43`, RELION `43`.
- Final Pmax: RECOVAR `0.323451`, RELION `0.324587`; mean abs Pmax gap
  `2.98e-2`, p99 `1.32e-1`, max `4.16e-1`, corr `0.909574`.
- Final pose parity: mean angular error `0.0865 deg`, max `1.18 deg`.

The selected-only fine-grid fix was implemented after this run to remove the
`hp=7` full-grid materialization bottleneck seen at iteration 11 (`643.7s`).
Focused HP7 replay after the fix:

`_agent_scratch/fixed_hp7_selected_codex_hp7_selected_4fbba3e2_20260427_105105`

- RELION it012 fixed-state replay, one A100.
- Iteration wall: `391.0s`; total script wall: `412.9s`.
- Pmax mean abs gap: `2.51e-4`, max `0.00255`, corr `0.999992`.
- Pose/translation parity: pose mean `0.0002 deg`, translation max `0.0 px`.
- Map corr RECOVAR-vs-RELION merged: `0.999953`.

Rerun the full 13-row trajectory after this fix to measure end-to-end speed.

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

## Fixes achieved

Important fixes now in this branch:

- RELION accelerated `pdf_offset` units for the local translation prior.
- RELION-style zero-filled real-space integer old-offset pre-shift before FFT.
  This fixed particle 4603 from Pmax gap `0.026669` to `0.000692` with exact
  pose/translation agreement in focused replay.
- Adaptive sparse pass 2 and os0 global-significant-support pass 2 route
  through exact local search, not grouped/bucketed sparse pass 2.
- Exact local search no longer materializes the full perturbed fine rotation
  grid. It selects canonical fine-grid ids first and applies RELION
  `SamplingPerturbation` only to selected rotations.

## Current open problem

Current parity standing: map and pose parity are strong through the 13-row 5k
trajectory, but late-iteration per-particle Pmax parity is still not at the
`~1e-4` arithmetic target. Next debugging step is a post-selected-only rerun of
the 5k/128 trajectory, then trace the first late-iteration particle with a large
Pmax gap using RELION-vs-RECOVAR dumps of raw scores, normalized probabilities,
candidate poses/translations, noise, CTF, tau2, and Ft_y/Ft_CTF.

## Full-trajectory and large-run caveats

Old complete 8-row 5k trajectory artifact:

`_agent_scratch/long_end2end_parity_20260426_182134`

This is now historical only. Use the 13-row replay above as the current
pre-selected-only speed baseline.

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
