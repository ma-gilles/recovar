# VDAM packed projection-reuse gate — H100 jobs 13367167 and 13367508

## Decision

Accept exact scoring-projection reuse as the default-off implementation basis
for the combined hybrid and full-trajectory gates.  The candidate
now projects the 1,868-pixel score/reconstruction union once inside the mature
EM big JIT, returns the packed reconstruction rows, scatters them into the
original dense layout, and calls the shared EM noise reductions.  This removes
the separately recomputed projection that caused the prior `1.89e-5`
`sigma2_noise` delta.

This is a one-transition **quality pass**, not an overall component, release, or
trajectory promotion.  The same-state harness is diagnostic by construction,
and the conservative diagnostics-off panel improves whole-iteration wall time
by 4.09%, just below the predeclared 5% component runtime threshold.  The
combined coarse-hybrid plus packed-deferred path and a complete trajectory
remain required.

## Qualification

| Field | Split-diagnostic panel | Clean timing panel |
|---|---|---|
| Source | `cf9791d35e2b97cd5b64426aab33b749e3b50ce3` | same |
| Slurm | `13367167` (`COMPLETED`, exit `0:0`, `00:05:58`) | `13367508` (`COMPLETED`, exit `0:0`, `00:06:31`) |
| Hardware | `della-h19g4`, H100, `GPU-990435ac-e5fe-18d9-c741-59b8fd9c9439` | `della-h20g1`, H100, `GPU-5297e2fc-3064-625f-a65a-9db11614d705` |
| Boundary | Exact in-memory GF46 state, iteration `34 -> 35` | same |
| Panel | direct / packed deferred / packed deferred / direct | same |
| Raw A2/XA diagnostics | enabled | disabled |
| Peak monitored GPU memory | 17,579 MiB | 17,579 MiB |

The candidate remains guarded by the flat-row, packed-projection,
source-faithful RELION x-half BPref, and
`RECOVAR_INITIAL_MODEL_DEFER_PACKED_VDAM=1` switches.  All are off by default.

## Science result

- Every tracked pose, translation, class, posterior, significance, particle-
  state, sampling-state, and complete support-audit field is exactly equal in
  both ABBA panels.
- Direct and candidate both project the same 1,868-pixel union.  The candidate
  evaluates 38,016 packed rows instead of 62,208 padded rows and reconstructs
  the same 489 nonzero-posterior rows.
- In the diagnostic panel, cross-backend final `sigma2_noise` normalized L2 is
  `2.48e-8` and `3.91e-8`; direct/direct is `2.26e-8` and candidate/candidate
  is `7.70e-9`.  Maximum absolute deltas are `1.86e-10` and `2.79e-10`.
- In the independent clean panel, the two cross-backend values are `3.25e-8`
  and `3.06e-8`; direct/direct is `1.22e-8` and candidate/candidate is
  `5.78e-9`.  Maximum cross-backend absolute error is `1.86e-10`.
- Map-gradient and reconstructed-map differences remain at the within-backend
  CUDA atomic-repeat scale.  There is no candidate-specific growth at this
  one-transition boundary.

The raw split sufficient statistics confirm that the earlier failure is gone:

| Raw statistic, direct 2 vs candidate 2 | Cross normalized L2 | Cross max abs | Largest within-backend normalized L2 |
|---|---:|---:|---:|
| `wsum_noise_a2` | `7.32004e-8` | `64` | `6.86865e-8` |
| `wsum_noise_xa` | `1.36823e-7` | `192` | `1.38332e-7` |
| `wsum_sigma2_noise` | `1.24130e-7` | `256` | `1.55492e-7` |
| `wsum_img_power` | `1.24599e-7` | `1,024` | `1.90590e-7` |

These are ordinary float32 reduction/atomic variations, not the stable
`10^-4` raw-operand displacement seen when VDAM reprojected its final support.
The implementation is mathematically the same shared EM projection and dense
reduction route; long-trajectory stability is the remaining correctness test.

## Runtime result

Only second-arm warmed measurements are compared.  The diagnostics-off panel
is the conservative runtime result.

| Metric | Clean direct | Clean candidate | Change |
|---|---:|---:|---:|
| Whole iteration | 2.373650 s | 2.276479 s | **-4.09%** |
| Expectation | 2.144977 s | 1.966991 s | **-8.30%** |
| Shared local EM | 1.348661 s | 1.176209 s | **-12.79%** |
| Local big JIT | 1.155216 s | 0.872998 s | **-24.43%** |
| Pass 2 | 1.361618 s | 1.191266 s | **-12.51%** |
| Deferred local noise | 0.000000 s | 0.089304 s | +0.089304 s |

The raw-diagnostic panel showed larger same-node gains (`31.02%` whole,
`46.85%` shared EM, and `55.04%` big JIT).  Because that result did not repeat
on a second node with diagnostics disabled, it is recorded as variance rather
than used as the headline performance claim.

## Provenance

Diagnostic panel:

- Report JSON SHA-256:
  `1a02a96533fbd1b7d9c111da056aef41f35ead8cbc032e05b578d8f6e10c7313`
- Qualified CUDA checksum-file SHA-256:
  `b056bd37fa01147ddccba563b571781b0d99b687fe3ec9b78acb5cdb4f1bb221`
- Artifact manifest SHA-256:
  `b05fdc4b22696e7e254981843311e73ffac0099c234e93f0d439a82bca56ef22`
- Static-input checksum-file SHA-256:
  `9e39c7be66d989d7a3755541b794e6918f3c4a630ac1e1652a2f61819892a38b`
- Disposable root:
  `/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_packed_deferred_reuse_flat_projection_same_state_it34_cf9791d35_20260903T013909Z`

Clean timing panel:

- Report JSON SHA-256:
  `6a979206fdb062678862d378d25516093bdad3fe83d41e63af21eadc6633ad08`
- Qualified CUDA checksum-file SHA-256:
  `fdaffeb6db73511bbaca01e27cc3255e98cd17c54058b9a31ae3cdc0aa5df5a0`
- Artifact manifest SHA-256:
  `edd16fb7d6d99cc12da4d8067cdc0b8f6c258e1596eb422d88883229e23712db`
- Static-input checksum-file SHA-256:
  `5e0f2b2247ecf6d81cefc60fa4904877155ca480df708c19ab87a2d2a496a090`
- Disposable root:
  `/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_packed_deferred_reuse_flat_projection_same_state_nodiag_it34_cf9791d35_20260903T014950Z`
