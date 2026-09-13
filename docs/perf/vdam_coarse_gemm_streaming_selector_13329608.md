# VDAM GF46 all-particle coarse GEMM selector diagnostic

Status: **GO for a source-rotation-block selector redesign; NO-GO for pure
GEMM scoring; not eligible for production or runtime promotion.**

## Sealed run

| Field | Value |
|---|---|
| Slurm job | `13329608` (`COMPLETED`, `0:0`, 69 s) |
| Node / GPU | `della-h21g4` / `GPU-099c0d77-bb85-f2e9-f628-148b733c9176` |
| Integrated source | `6e4e0ae65cba3cd0f86febdfe319e747ced07d97` |
| Source tree | `8b5f75555d264ed4d36ddeac894bc1da8bbe6a15` |
| Frozen state | GF46 `run_it180`; exactly iteration 181; joint pseudo-halfset stream |
| Selected particles | 1,000/1,000 exactly once; frozen ID digest `c0199226...a90be` |
| Candidate surface | 36,864 rotations x 29 translations = 1,069,056 candidates per particle |
| Persistent diagnostic state | 18,411,272 bytes at the observed physical batch of 187 |
| Result root | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_gemm_gf46_stream_v2_6e4e0ae65_h21g4_20260901` |

The result root is immutable and carries `COMPLETED`, `IDENTITY_VALIDATED`,
`SHA256SUMS`, and a `SAFE_TO_DELETE` marker. The identity certificate SHA-256
is `3dfd92da58365c33388eea7071f1814be1561a5c7557615b52a9e263ce7e230f`;
the aggregate manifest SHA-256 is
`a78674eb7c342d16f624fe3d68b6a3c2ea868a6ca880a6c4f972d728f02a2a3c`.

Job `13329572` is a superseded fail-closed launcher check. It exited before
creating the result root because the runner named the side-branch pre-prior
commit rather than its identical integrated cherry-pick. Commit `6e4e0ae65`
corrected and tests that ancestry contract; it is not a numerical failure.

## Complete paired result

| Metric | Result |
|---|---:|
| Finite direct/GEMM pairs | 1,069,056,000 / 1,069,056,000 |
| Nonfinite pairs | 0 |
| Maximum absolute score delta | 3.3125 |
| Weighted RMS delta | 0.05107886 |
| Weighted signed mean delta | +0.00789002 |
| Winner comparisons | 1,000 / 1,000 covered |
| Winner mismatches | 1 |
| Support comparisons | 1,000 / 1,000 covered |
| Direct-support false negatives | 0 |
| Direct-support false positives | 1 |

The pre-prior and post-prior all-candidate summaries agree on the maximum
error (`3.3125`) and have complete finite coverage. Particle 1,933 is the
single winner mismatch; its direct and GEMM winner margins are only
`0.000732421875` and `0.00048828125`. Particle 636 has the single support
difference, an extra GEMM point and no missing direct point. These discrete
differences confirm that the 4.5x expanded-square GEMM cannot replace the
direct RELION scorer by itself.

## Selector result

The v2 diagnostic retained the highest 2,048 individual candidates plus a
sentinel for both the posterior and pre-prior streams. This is complete for
the raw-maximum certificate on all 1,000 particles. It is complete for the
posterior nonzero-weight surface on only 727 particles because translations
from a few rotations fill the individual-candidate table:

| Certificate | Covered particles | Source 16-rotation blocks |
|---|---:|---:|
| Pre-prior raw maximum | 1,000 / 1,000 | median 1; p95 2; maximum 3 |
| Posterior nonzero surface | 727 / 1,000 | median 6; p95 6; maximum 6 |
| Deduplicated union | 727 / 1,000 | median 6; p95 6; maximum 6 |

The raw certificate added no block beyond the posterior set in the 727
covered observations, but it remains mathematically required for datasets
where a strong negative prior hides the raw winner. The focused poor-prior
unit test exercises that case.

The failed coverage is therefore a retention-granularity defect, not evidence
that exact rescoring needs a broad fraction of the grid. The next selector
will reduce every aligned 16-rotation source block to its maximum score before
retention. GF46 has only 2,304 such blocks per class; keeping a fixed `Q+1`
block table is smaller and cannot be saturated by the 29 translations of one
rotation. The hybrid path must use an aligned outer rotation chunk (for
example 4,992 rather than 5,000), union the pre-prior and posterior block
sets, and fall back to the complete direct scorer on any sentinel, nonfinite,
tail, or floating-point-certificate failure.

## Qualification boundary

The 69-second process executed both full scorers, compiled new programs, and
wrote diagnostic artifacts. It is deliberately marked timing-ineligible and
does not update the frozen runtime score. Prior clean GF46 profiling remains
the performance evidence: the shared mature-EM GEMM coarse pass is about 4.5x
faster, while the exact selected-block CUDA primitive passed H100 job
`13328717`. The next meaningful runtime number is the default-off block-level
hybrid with no full direct companion.

Focused validation at the integrated head:

- streaming certificate plus live K=2 integration and runner guards: 22 passed;
- Bash syntax, Ruff, Python-heredoc parsing, and `git diff --check`: passed;
- no broad RECOVAR test suite was run.
