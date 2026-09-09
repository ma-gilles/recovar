# VDAM exact-CTF host-native transfer — H100 jobs 13420339/13421292

## Decision

Retain commit `f77683a88`, which leaves exact source-STAR CTF evaluation on the
host until an immediate host-padding consumer has formed its final operand.
Device-first callers keep the mature shared EM wrapper and therefore still
perform one device placement.  On the sealed iteration-48 K=1 continuation,
the change removes `102.778 ms` (`7.27%`) from the RECOVAR Nsight capture and
`332.228 ms` (`7.23%`) from warm process wall with unchanged kernel count and
effectively unchanged GPU work.

All comparable discrete science state is exact, and every cross-commit final
map comparison remains inside the observed cold/warm nondeterminism envelope.
This is a focused optimization-boundary acceptance, not a trajectory, default,
or frozen-release promotion.

## Same-H100 performance

Both runs use the same physical H100 UUID
`GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`, the same iteration-47 checkpoint,
the same ordered 200-particle subset, the same production-topology schedule,
and the `all_optimized_q32` profile-free warm contract.

| Metric | Baseline `13420339` | Host-native CTF `13421292` | Change |
|---|---:|---:|---:|
| RECOVAR capture | 1.414699823 s | 1.311921397 s | **-0.102778426 s (-7.27%)** |
| Warm process wall | 4.598198874 s | 4.265970538 s | **-0.332228336 s (-7.23%)** |
| GPU busy | 57.103262 ms | 57.176251 ms | +0.072989 ms |
| GPU idle | 1357.596561 ms | 1254.745146 ms | **-102.851415 ms** |
| Kernel launches | 11,251 | 11,251 | 0 |

The unchanged arithmetic topology and one-for-one reduction in GPU-idle time
identify removal of a host/device round trip as the causal mechanism.  Native
RELION varied between the two jobs, so its single-run wall ratio is not used to
attribute this patch's speedup.

## Science boundary

All 77 comparable non-profile/non-reduction metadata fields are exactly equal
for baseline cold versus warm, candidate cold versus warm, cold across commits,
and warm across commits.  This includes selected particles, poses, classes,
best rotations/translations, significant counts, posterior/Pmax state, noise
and offset state, and the complete sampling/controller schedule.  Particle
data-STAR payloads are byte-identical after excluding their creation timestamp.

| Map comparison | Relative L2 | FSC AUC | Minimum non-DC FSC |
|---|---:|---:|---:|
| Baseline cold/warm repeat envelope | 2.6815105e-9 | 0.9999999999999223 | 0.9999999999987415 |
| Candidate cold/warm repeat | 1.9055940e-9 | 0.9999999999999609 | 0.9999999999993720 |
| Baseline/candidate cold | 2.1217441e-9 | 0.9999999999999507 | 0.9999999999992177 |
| Baseline/candidate warm | 2.3214057e-9 | 0.9999999999999409 | 0.9999999999990548 |
| Worst of all cross-commit pairings | 2.6312027e-9 | 0.9999999999999242 | 0.9999999999987982 |

The worst cross-commit relative L2 is `98.12%` of the observed repeat
envelope.  Cross-commit RMSE, maximum absolute difference, and changed-voxel
count are also bounded by the cold/warm repeats.

Raw CUDA atomic-reduction aggregates are deliberately not called bitwise
exact.  Two cold cross-commit summaries narrowly exceed the two-repeat raw
aggregate envelope: BPref weight is `6.93e-10` relative versus `6.33e-10`, and
image power is `7.38e-8` versus `6.74e-8`.  Warm cross-commit aggregates,
sigma2 noise, all discrete state, and the final science map remain bounded.

## Shared implementation

`_relion_exact_ctf_half_from_source_star_host` owns the host cache/evaluator.
The original `_relion_exact_ctf_half_from_source_star` remains the shared
device-first wrapper and performs one binary64 placement for mature EM callers.
Only consumers that immediately pad on the host request the host-native result.
No CTF formula or numerical kernel is duplicated.

## Provenance and focused checks

- Baseline job: `13420339`, `COMPLETED`, `della-h19g1`, source `e1db9fca8`.
- Candidate job: `13421292`, `COMPLETED`, `della-h19g1`, source `f77683a88`.
- Baseline root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_late_certify_off_it48_e1db9fca8_20260904T0501Z`.
- Candidate root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_late_ctf_host_it48_f77683a88_20260904T0527Z`.
- Schedule: current size 84, healpix order 2, 36,864 rotations, 148
  translations, subset size 200, perturbation `-0.43747416138648987`.
- Ordered selected-particle SHA-256:
  `ab3cf8e6f4a74396ad96ac798d95db270fd2643cc659c412c7c381ee828a0705`.
- Focused source validation: 12 tests passed in independent review; Ruff,
  Python compilation, and `git diff --check` passed.
- Map analysis used `shell_fsc` and `normalized_fsc_auc` from
  `scripts/replay_final_bpref_dump.py` plus direct MRC relative-error metrics.

No broad RECOVAR test suite was run, per the VDAM performance workflow.
