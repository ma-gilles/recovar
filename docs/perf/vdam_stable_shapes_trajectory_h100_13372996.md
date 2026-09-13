# VDAM stable Fourier shapes through iteration 50 — H100 job 13372996

## Decision

Do not promote stable Fourier shapes from this gate.  They reduce compilation
artifacts and improve median expectation time by 5.25%, but median end-to-end
wall improves only 2.07%, below the predeclared 5% threshold.  The strict
trajectory science gate also fails because both control and candidate repeats
are nondeterministic across fresh processes.  The crossed pairing is
incompatible with a deterministic toggle-only displacement, but this gate
cannot exclude a candidate-specific variance effect or establish equivalence.

## Qualification

| Field | Value |
|---|---|
| Source | `47190851099ba608d60114de1df0624e9efe8e1a` |
| Slurm | `13372996` (`SCIENCE_FAILED`; all four arms completed; analyzer exit `1`) |
| Hardware | `della-h19g2`, H100, `GPU-ddb1592d-744e-ea56-d0a3-aec6e7c97d10` |
| Trajectory | GF46 iterations `0 -> 50` |
| Panel | stable-off / stable-on / stable-on / stable-off |
| Isolation | fresh process and fresh JAX cache per arm |
| Only configuration delta | `--stable-fourier-window-shapes` |

## Science result

The analyzer used a fixed `2^-21` numerical tolerance and exact discrete
identity.  It reports eight exact failures and 235 numerical failures.
Crucially, the discrete failures are not confined to stable shapes:

- the second **control** repeat first changes a rotation/pose at iteration 35;
- the first candidate repeat changes a rotation/pose at iterations 43 and 48;
- the second candidate repeat remains discrete-identical to the first control
  throughout the recorded trajectory.

This crossed pairing demonstrates fresh-process trajectory nondeterminism and
is incompatible with a deterministic stable-shape displacement.  It is still
insufficient for promotion: maximum cross-mode map L2 is `2.08766e-5`, while
the largest within-mode repeat distance is `1.31763e-5` (the control-only
repeat maximum is `1.18858e-5`).  Whole-trajectory RMS map L2 is
`5.67e-6` for the control repeat and `6.60e-6` for the candidate repeat; cross
pairs range from `1.21e-6` to `1.05e-5`.

## Runtime and compilation result

| Metric | Median control | Median candidate | Change | Gate |
|---|---:|---:|---:|---|
| End-to-end wall | 530.121 s | 519.145 s | -2.07% | **FAIL** |
| Expectation stage | 493.709 s | 467.812 s | -5.25% | pass |
| Fine pass 2 | 361.062 s | 278.684 s | -22.82% | pass |
| Peak GPU memory | 17,590 MiB | 17,587 MiB | -0.02% | pass |

Individual candidate walls were `441.134 s` and `597.157 s`, showing enough
runtime variance that the 2.07% median wall signal is not robust.  Pass 2 is
consistently faster in both candidate arms (`363.022/359.101 ->
276.580/280.788 s`), while the slower candidate arm has anomalous pass-1 and
M-step totals (`208.898 s` and `43.243 s`, versus `111.450 s` and `10.252 s`
in the other candidate arm).  Stable mapping reduces 19 logical Fourier sizes
to eight physical capacities.  Per fresh run, raw persistent-cache file count
falls from 4,775 to 3,664 (`-23.3%`), but every arm still has 87
`run_local_bucket_big_jit` cache files.  Other signature dimensions therefore
remain a likely source of compilation multiplicity, without proving their
runtime contribution in this gate.

## Next action

Keep this branch as a regression track.  Prioritize stable packed-row ABIs and
validity-aware CUDA execution, which target the still-changing big-JIT shapes
and the 76.8% padded fine-score work directly.  Revisit stable Fourier windows
only as part of a repeat-controlled full backend after the larger row-shape
gain is qualified.

## Provenance

- Analyzer report SHA-256:
  `037cdda817bb5914d6836aaf305ea2a6383539035620909fef602c586b316149`
- CUDA SHA-256:
  `fc42f2672c19085fb342bfaa674d5a665dbed2efd532a6ae27a5aa25bcd2f3de`
- Disposable root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_stable_shape_trajectory_471908510_20260903T0025Z`
