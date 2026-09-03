# K=1 box-800 sparse M-step memory qualification

This record qualifies one memory-bound path needed by the EMPIAR-10202 set-6
I1 refinement.  It is a **64-particle box-800 memory qualification**, not a
full-dataset resolution or RECOVAR-versus-RELION quality result.  The fixture
contains 32 deposited particles per half-set, their deposited poses and CTFs,
one common I1-symmetrized reference, and two forced current-size-800
iterations.  No scientific FSC claim should be drawn from these 64 particles.

The accepted source is RECOVAR commit
`dcf42cbe7bd4fc2c967161baf971c1b0c357a80c`, tree
`e4c33e4b350ebe1285599ae7ace79730d60961d1`.  It caps a sparse pass-2 M-step
bucket before `compute_local_mstep_sums` materializes its complex numerator
and real denominator, both shaped `(images, padded rotations, reconstruction
pixels)`.  The cap reuses the existing adjoint-output byte budget and preserves
particle order.  At box 800 on an 80 GiB H100 it permits eight images for the
iteration-1 `R=16` buckets and one image for the iteration-2 `R=128` and
`R=256` buckets.

## Rejected r3 outer-planner control

Jobs `13309056` and `13309057` are retained as a negative memory comparator.
Both requested and received exactly
`cpu=4,mem=500G,node=1,billing=40,gres/gpu=1`, used one H100 without an
exclusive allocation, and produced an identical iteration-1 controller row.
Both then failed in the first half of iteration 2.

| Arm | Commit | Outer plan | State / elapsed | Sampled peak HBM | Failure |
| --- | --- | --- | --- | ---: | --- |
| Historical control, job 13309056 | `3643c2ddf57cc66ddcf1412d9c2a63a6f46fa7de` | historical full-cube | `FAILED 1:0` / 898 s | 78,545 MiB | requested 1,642,496,000 B in `compute_local_weighted_sums` |
| Compact outer candidate, job 13309057 | `b67c5ce8c44c545d2d92e263de75ef9d2fa2723f` | 12 images x 41 rotations | `FAILED 1:0` / 940 s | 78,543 MiB | same allocation and call site |

The requested complex64 numerator was exactly
`5 * 128 * 320800 * 8 = 1,642,496,000` bytes.  The unchanged failure after
compact outer planning established that the outer EM planner did not bound
the inner sparse M-step output.  These jobs are not accepted outputs.  Their
complete commands, source identities, allocation records, logs, 1 Hz HBM
traces, and rejected-harness history are sealed under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_fullsize_compact_ab_b67c5ce8c_20260901`.
The final r3 launcher SHA-256 is
`2f68692ff8832e8e87e4fa2a52c625a51dc9b701ae5027e18a13609bfaab2f9c`.

## Accepted two-iteration memory gate

Candidate-only job `13310119` used the same 64 particles, I1 symmetry,
reference, box size, refinement options, sealed custom CUDA library, and
one-H100 allocation.  The iteration-1 cap was inactive because the observed
maximum bucket contained seven images.  Iteration 2 logged
`mstep_output_image_caps=[(128, 1), (256, 1)]`; both half-sets crossed the
former OOM boundary and the refinement completed.

| Measurement | Result |
| --- | ---: |
| Job state | `COMPLETED 0:0` |
| Requested = allocated | `cpu=4,mem=500G,node=1,billing=40,gres/gpu=1` |
| Iteration 1 | 549.7 s |
| Iteration 2 | 627.6 s |
| Refinement-reported total | 1,182.0 s |
| Launcher wall time, including products and validation | 1,277 s |
| Sampled peak HBM | 76,587 MiB |
| HBM change versus r3 compact candidate | -1,956 MiB |
| Final maps | three finite `800 x 800 x 800` volumes |
| Symmetry | I1, 60 operators |

The iteration-1 convergence row is byte-for-byte equal to the r3 compact
candidate's row.  Iteration 2 completed with `frac_changed=0.9219`, resolution
diagnostic 37.08 A, `ave_Pmax=0.1093`, rotation change 13.704 degrees, and
translation change 2.240 A.  These values document execution; they are not a
resolution estimate for the deposited dataset.

The source change preserves the order of image indices while splitting the
M-step buckets.  The checked unit gates verify the box-800 byte arithmetic,
the `R=64/128/256` caps, and order preservation.  The H100 run additionally
required both iteration-2 host-result records, four solvent-flatten
lifecycles, exact source and symmetry identities, no OOM or traceback, and
finite box-800 outputs.  Every check in `twoiter_comparison.json` passed.

The accepted run is sealed under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_fullsize_mstep_b1_dcf42cbe7_20260901`
with `SAFE_TO_DELETE`.  Important identities are:

- launcher SHA-256:
  `c3e9466ed9fb204bc6a96ace87e16c13e28010645b6bd73609307c112fbb0c83`;
- custom CUDA library SHA-256:
  `791e1e66a7b01303518745dad1d480844fe0d2736531e54ad4ae29d22bbafdd3`;
- particle STAR SHA-256:
  `500dc2b76fdd5554d1dba759168184109f16d91deed0fce1fc2d9f5daf164d7d`;
- initial reference SHA-256:
  `d77516a08e5e3ccdef07d9039e36d65b174afe5d56fe20d7775eef188d2e6cc6`;
- validation JSON SHA-256:
  `a7e9ee89b7b8dd604dbbda2691fe60d4f5913aa6f2fe7953f8573daba29a1006`.

## Reproduce the bounded gate

The sealed run directory is immutable evidence and must not be reused for a
new result.  To repeat the qualification, copy its `inputs/`, `scripts/`, and
sealed CUDA library to a new empty CRYOEM run root, change every absolute run
and runtime root in the copied launcher, recompute the launcher SHA-256, and
submit from the new directory:

```bash
cd /absolute/new/empty/run/root
sbatch --parsable \
  --export=EXPECTED_LAUNCHER_SHA256=<new-launcher-sha256> \
  scripts/run_candidate.sbatch
```

Before accepting the repetition, verify that `ReqTRES` and `AllocTRES` match,
only one H100 is visible, `OverSubscribe=OK`, the source checkout is clean at
the recorded commit and tree, and `outputs/candidate/COMPLETED` plus a green
`outputs/candidate/twoiter_comparison.json` exist.

The focused source-level tests run for this change were:

```bash
pixi run ruff check \
  recovar/em/dense_single_volume/helpers/sparse_pass2_bucketed.py \
  tests/unit/test_sparse_pass2_bucketed_perf.py
pixi run pytest -vv tests/unit/test_sparse_pass2_bucketed_perf.py \
  -k 'mstep_output or bucket_pass2_inputs or translation_tile_image_cap_is_bounded'
git diff --check
```

The selected pytest gate passed three tests with 179 deselected.

## Accepted final-all-data memory gate

GPU job `13310959` extended the same 64-particle fixture through one numbered
box-800 iteration and an actual forced K=1 final-all-data reconstruction.  It
requested and received exactly
`cpu=4,mem=500G,node=1,billing=40,gres/gpu=1`, used one H100 on
`della-h19g2`, and was not exclusive.  Its scientific step `13310959.1`
completed `0:0` with `MaxRSS=330469836K`.

| Measurement | Result |
| --- | ---: |
| Numbered iteration | 558.8 s |
| Final half 1 | 280.1 s |
| Final half 2 | 261.7 s |
| Final joined FSC and tau2 join | 141.9 s |
| Final reconstruction | 144.6 s |
| Final-all-data reported total | 1,221.5 s |
| Refinement-reported total | 1,781.6 s |
| Scientific-step elapsed | 31:31.60 |
| Sampled peak HBM | 76,593 MiB |
| Final FSC | 401 finite shells |
| Final maps | three finite `800 x 800 x 800` volumes |

Both exact final projectors were built.  The full-even Nyquist validator ran
2,048 times; every record contained all 320,800 packed Fourier indices and
all 401 row-zero entries.  The log contains no OOM, `RESOURCE_EXHAUSTED`,
traceback, projector-validator error, or finalizer error.

The Slurm wrapper for job `13310959` nevertheless ended `FAILED 1:0` after
31:54 because its immutable post-run validator incorrectly pinned parent
commit `b67c5ce8c44c545d2d92e263de75ef9d2fa2723f`.  The archived source and
run were correctly at tested commit
`dcf42cbe7bd4fc2c967161baf971c1b0c357a80c`.  The original validator JSON,
SHA-256
`1f04635e7642634830f5724d5acb20d60395dadaf2be17afc92ba85288f7d6b2`,
has every substantive check true and exactly one failed check:
`source_commit_exact`.  The original GPU directory was not changed and the
GPU calculation was not rerun.

A separate fail-closed, CPU-only re-audit, job `13312135`, verified the
immutable GPU artifacts and classified this as
`VALIDATOR_ONLY_FALSE_NEGATIVE`.  It requested and received exactly
`cpu=4,mem=64G,node=1,billing=16` with no GPU, completed `0:0` on
`della-i13n20` in 3:56, and used `MaxRSS=2109616K`.  All 18 corrected checks
passed.  The corrected validation artifact is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_final_all_data_n800_cpu_reaudit_dcf42cbe7_20260901/outputs/corrected_final_probe_validation.json`,
SHA-256
`e275a04d6ab5b2af4bcf2fa7f54ef56e41b71bd7c53cf0bc322defb2e830ef26`.
The CPU launcher and re-auditor SHA-256 values are, respectively,
`be0413143a06b456569364c9d440cb16a956e098447e2d1354885b3d8e6665c0`
and `7678a390553e1aad05d61142b928b71f1bd1eea45ea7e27a52ed88bcecfaf406`.

The final map hashes recorded by that audit are:

- half 1: `be89552d3e7057195923ba20d8683a4b3fbdbc40e252a95e159bc3edd09d51c1`;
- half 2: `11463d6b94742b65886f158c85d9625f60578a282190ac9a4c32903aaf73ebbe`;
- merged: `81c59f2456a59fd042ea5a9e6d3d92bc3d88ff051da6fd7ae32a34b38451ff7a`.

The immutable GPU evidence is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_final_all_data_n800_dcf42cbe7_20260901`;
the corrected CPU audit is under the directory named above.  Both have
`SAFE_TO_DELETE` markers.  This gate establishes that the bounded sparse
M-step and full-size finalizer fit on an 80 GiB H100.  It remains a
64-particle memory qualification and does not validate full-dataset
resolution or RECOVAR-versus-RELION quality.

## Accepted fresh local compact-planner gate

Job `13363559` exercised the later compact local-search planner at box 800
from a fresh physical iteration, rather than importing a prior iteration
state.  The fixture retained the deposited set-6 particle order, poses, CTFs,
I1 symmetry, and common reference, but deliberately used only the 32
particles in half 1 and stopped after local-search scoring.  It is therefore
a planner and memory-route qualification, not a reconstruction, FSC, or
RECOVAR-versus-RELION quality result.

The run used RECOVAR commit
`8069ac01508d57bfd74a7686930ebcea66b6e328`, tree
`bc7220a4dde5c85e4e615cae3113a05b042655fc`.  At
`current_size=62`, the planner selected the compact K=1 RELION route and
estimated 4.02 GB of persistent device storage instead of the historical
81.92-GB full-cube estimate.  The observed plans were 24 images by 148
rotations for the outer local pass, 29 by 41 and 64 by 72 for the two coarse
workspaces, and a caller-qualified 24 by 128 for the fine pass.  Both
score-only bucket loops completed naturally.

| Measurement | Result |
| --- | ---: |
| Job state | `COMPLETED 0:0` |
| Requested = allocated | `cpu=4,mem=500G,node=1,billing=40,gres/gpu=1` |
| Node / GPU | `della-h19g2` / NVIDIA H100 80GB HBM3 |
| Allocation elapsed | 12:08 |
| Refinement-reported profile wall | 650.344 s |
| Active particles | 32 in half 1; half 2 intentionally empty |
| Symmetry / box / current size | I1 / 800 / 62 |
| Sampled peak HBM | 38,059 MiB |
| Scientific-step MaxRSS | 282,151,892 KiB |

The exact Slurm allocation had one H100, four CPUs, 500 GB of host memory,
and `OverSubscribe=OK`; `ReqTRES` and `AllocTRES` matched.  The run root and
its separate runtime root both carry `SAFE_TO_DELETE` markers:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_planner_fresh_ff6099641_20260902`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/box800_local_planner_fresh_ff6099641_20260902`.

The sealed replay identities are:

- launcher SHA-256:
  `f602d346163a181692cba87b1fd048a81568fe69201de7a0e2fe3b83c0958805`;
- seed-builder SHA-256:
  `280325f443650f26be848afff9a6ddc177d68d122f6b2091260fd9fbe25d4e48`;
- exact pose-seed NPZ SHA-256:
  `a8d5a9426f97f8080e9abedf74a726640693e8704dda7c6b4e6852724ea84177`;
- profile/benchmark-ledger SHA-256:
  `adb38a8f5ddc73b089adfe47f6d4f7c2cdf6030be42539accf0038dc7c83c7e2`;
- stderr SHA-256:
  `7d29e6c0f60ab404d9e288eb85fd2fa60f884db8330aeee0b5fce60879fbff6a`;
- 5-second HBM trace SHA-256:
  `44f935a5df0390e456fad239f1dc09167ec20f0e3ce2ea49148442b798eb25a6`;
- resolved command SHA-256:
  `fa37231676726986844a31a7906f6dd494525e48ef38913458907584ba2a447e`.

To repeat this exact score-only gate, create a new empty run and runtime root,
copy `inputs/build_seed.py` and `jobs/fresh_score_only.sbatch`, replace their
absolute output roots, regenerate the seed with the pinned builder, verify all
recorded input and launcher hashes, and submit the copied launcher with
`sbatch --parsable`.  A repetition is accepted only if it logs
`mode=compact_k1_relion_score_bpref_overlap`, completes both score-only
passes, has matching requested and allocated resources, and produces a
`COMPLETED` marker.  Full-dataset job `13363818` is the separate production
quality and high-resolution gate for this planner; no final-quality claim is
inferred from job `13363559`.

## Full-dataset compact-planner boundary result

Full-particle job `13363818` completed numbered iteration 11 at a logged
internal resolution of 3.46 A, matching the earlier release trajectory, and
then entered iteration 12 at `current_size=564`.  The compact plan reduced the
half-1 fine M-step from the release run's 15,258 chunks to 1,263.  It completed
1,000 of those chunks (13,000/15,258 images) before failing naturally in
`relion_projector_half_texture_f32` immediately after two one-image
`bucket_rot=512` split-route buckets.  Thus the compact planner is a real
throughput improvement, but it is not by itself sufficient for the box-800
high-resolution memory boundary.

The sampled HBM peak was 79,259 MiB.  The scientific Slurm step had
`MaxRSS=496561284K`; the observed cgroup peak was 508,585,185,280 bytes under
the 500-GiB allocation, and every cgroup memory-event counter remained zero.
The failure is therefore a CUDA allocation failure, not a host-memory kill.
The failing split route retains the eager JAX `Projector::data` slab while the
transient texture launcher creates a second full CUDA texture array.  The next
bounded fix is to reuse the already implemented host-uploaded persistent
RELION texture across the exact-local loop, with an explicit error-path
lifecycle gate; no final completion claim is made until that route crosses
this exact size-564 boundary.

Job `13363818` used source commit
`8069ac01508d57bfd74a7686930ebcea66b6e328` (tree
`bc7220a4dde5c85e4e615cae3113a05b042655fc`) and requested and received
exactly one H100, four CPUs, and 500 GiB without `--exclusive`.  It failed
`1:0` after 7:33:45.  The immutable run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_full_recovar_compact_8069ac015_20260902`
and carries `SAFE_TO_DELETE`.  Its stderr, HBM trace, command, and `scontrol`
record have SHA-256
`e0e7f5db7e375286f3b586399cfab61a8fd4bca9d1752937fad79419497ed093`,
`05391a39adfabf74207c9b150babd73ece31ac852d54ef12383f20dc66ba6c7a`,
`d312726b64b59e348a22bef505f5c67beb73b496b622710fbe0dd2f7f618a495`,
and `bf4ca493ce3ac35406a934c94f371661e4e730ace499b9906378040080e9d1e8`,
respectively.  The exact terminal block is stderr lines 26415--26472.

## Accepted persistent PPref texture lifetime gate

Commit `530eba051e8cf14d21be1563027d2b189ff40741` removes the device-side
duplication at the exact failure geometry. Exact local search now opens one
host-uploaded RELION texture owner around the complete engine call. The normal
compiled bucket path receives its handle as a dynamic scalar, so a compiled
executable cannot retain a stale handle; cache construction, packed-noise
projection, and the wide split route share the same owner. The engine waits for
every borrowed compiled result before cleanup, and cleanup closes the owner on
both success and exception paths. The fallback slab and manual-projection paths
retain their previous behavior.

H100 job `13379238` exercises the full `run_local_em_exact` engine at box 800,
`current_size=564`, 84 translations, and the actual failure-sized host PPref
shape `(1131,1131,566)` complex64 (5.394259 GiB). One 256-rotation bucket runs
through `run_local_bucket_big_jit`; one 512-rotation bucket is deliberately
forced through the wide split route. Both update the full x-half BPref
accumulators, whose packed outputs each contain 724,005,126 elements. The gate
records one normal compiled bucket and one wide split, completes in 28.913 s,
and closes the owner cleanly.

The 100-ms trace peaks at 39,033 MiB, compared with 79,259 MiB immediately
before the old full-particle failure. This is a 40,226-MiB separation at the
same projector/current-size boundary, although the two-image gate is not
reported as a matched full-trajectory performance A/B. Texture-ready HBM is
6,159 MiB. The job requested and received exactly one H100, four CPUs, and 150
GiB (`ReqTRES=AllocTRES`) without `--exclusive`; it completed `0:0` in 47 s on
`della-h19g1`, with batch `MaxRSS=22475616K`.

The persistent-texture unit suite also checks transient-versus-persistent
bitwise projection equality, dynamic-handle compile reuse, stale/zero-handle
failure, normal and split engine routing, and success/error cleanup. The
15-test local-A100 file passes in 31.19 s, 22 selected routing/refinement tests
pass in 23.54 s, and `test-em-fast-guard` passes all 16 tests in 77.35 s.

A separate same-geometry arithmetic discriminator, H100 job `13379318`, used
the full `(1131,1131,566)` complex64 PPref with 2,556,626 deterministic
nonzero voxels and seven nonidentity rotations. The dynamic-handle persistent
texture and the existing transient-texture implementation produced bitwise
identical finite `(7,159612)` complex64 projections: `max_abs_diff=0.0`, with
common output SHA-256
`dd30338fbbe1a02b10003f95cf5e19a4ac407c4195f1d03c6c54835d346593b4`.
The input PPref and rotations have SHA-256
`a80cd9f556919446de07f1ec736f1785014d7412659293e8ae1389d515ea0cb4`
and
`154e8b29e51253685d57ada794eb39c490842fdfb9c3a1130640f110ed313321`.
The nonexclusive job requested and received exactly one H100, four CPUs, and
80 GiB, completed `0:0` in 19 s, and peaked at 14,369 MiB sampled HBM.

The run and runtime roots are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_persistent_texture_size564_20260903`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_persistent_texture_size564_20260903`;
both carry `SAFE_TO_DELETE`. The complete stdout, 100-ms HBM trace, and audit
Markdown have SHA-256
`4375dc5e723a7e423c50db449a01b6ae7f0eb0436a66a6b711d93f7706d427e9`,
`cc5e9eeac48c4c880c6179bdb54475640249fb38228796b00be837cc2cb89e6e`,
and `2edf9fc84cc2dd30568f90b45c67cb69c2d1db0d9d390ddb3c2c39af7d46e177`.
This accepts the exact allocation/lifetime boundary, not final reconstruction
quality; a full advancing trajectory remains the quality gate.

The arithmetic discriminator is sealed under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_persistent_texture_arithmetic_20260903`
with `SAFE_TO_DELETE`; its stdout, 100-ms HBM trace, and audit Markdown have
SHA-256
`aa7c894ee2e1fef6120c73be98f3a26ada087beeb4943bb3921210074b52156a`,
`bd1c6f0e415ae09bf998fbc05383c9ee88db3a8977067a3d03cc1cbe6a453289`,
and `d80856f22cc08de39532c90f80672ad7c6294e0b2dfa1760f3dc4086d293c47e`.
It accepts projection arithmetic at the failure geometry, not the full
refinement trajectory.

Parent commit `737018067b3ed671d218129fe53aec58fc14e825` independently transfers
already-owned K=1 NumPy half-map snapshots into the previous-reference slots
instead of copying them. At box 800 this removes a deterministic transient of
`2 * 800^3 * sizeof(complex128)` = 16,384,000,000 bytes (15.258789 GiB).
Device arrays still offload explicitly, shared cold-start device buffers are
transferred once, and non-owning NumPy views are copied. Its five lifecycle
tests, 16-test EM fast guard, scoped Ruff, and `git diff --check` pass. This is
a host-ownership change only; it does not alter map arithmetic.

## Accepted unused local-padding removal

Commit `f91c73f2907ccea635782814d0cbde1d4f8ed4c2` (tree
`5c73c5d41b62c0b1bbeae5033ae047c39eaad609`) removes an independent box-800
memory and runtime defect from exact local search.  When an exact RELION
`Projector::data` volume is supplied, every scoring and M-step branch reads
that projector and does not read the padded native mean.  The old path
nevertheless constructed a full padding-factor-2 complex64 mean before
dispatch.  At box 800, that dead `(1600,1600,1600)` array is exactly 31.25
GiB.  The fix skips native mean padding only in the supplied-projector route;
native projection retains its previous padding behavior.

The focused source regression runs exact local search with a supplied
projector and `projection_padding_factor=2`, replaces the native padding
routine with a hard failure, and exercises both the big-JIT and split
projection branches.  The matched H100 gate then replays the same 32 deposited
particles, box, I1 reference, seed, options, and score-only work as job
`13363559`.

| Measurement | Job 13363559, before | Job 13366512, after | Change |
| --- | ---: | ---: | ---: |
| Sampled peak HBM | 38,059 MiB | 20,123 MiB | -17,936 MiB (-47.13%) |
| Exact-local-window peak HBM | 38,059 MiB | 6,809 MiB | -31,250 MiB |
| Refinement ledger wall | 650.344222 s | 124.896842 s | -525.447381 s (-80.80%; 5.21x faster) |
| Parent bucket loop | 5.9 s | 5.0 s | -0.9 s |
| Fine bucket loop | 14.5 s | 14.5 s | unchanged |

The machine audit accepts all six exact semantic comparisons: ledger run
shape, ordered batch plans, significant-support summaries, adaptive-mask
counts, bucket work, and completion shape.  In particular, both jobs retain
the 24-by-148 outer plan, 29-by-41 and 64-by-72 parent workspaces, 24-by-128
fine plan, parent support `(184,8192,1368)`, mask medians/maxima
`(31,109,1008,3488)`, and fine support `(556,1904,11057)`.  Neither candidate
log nor output contains an OOM, traceback, fatal marker, or incomplete loop.

Job `13366512` requested and received exactly
`cpu=4,mem=500G,node=1,billing=40,gres/gpu=1`, used one H100 on
`della-h19g3` with `OverSubscribe=OK`, and completed `0:0` in 3:21.  Its
scientific step completed in 3:11 with `MaxRSS=101493172K`.

The accepted run and separate runtime root, both marked `SAFE_TO_DELETE`, are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_padding_guard_f91c73f29_20260902`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/box800_local_padding_guard_f91c73f29_20260902`.

Important sealed identities are:

- launcher SHA-256:
  `2eb44b0327deb2c8d6c2c7954a29d1bf5ef2c543655321c530eac77ff59b8ff6`;
- custom CUDA library SHA-256:
  `ad3b4a38bc320ca7e6f7c6793500f03799b32daf8a549869eafbefb5cc289bf1`;
- stderr SHA-256:
  `963e6b14dc3146033a8d18db5596bb656b9dd812dd47ef2238fc4188c99e7c35`;
- HBM trace SHA-256:
  `128adb3526f83c9bbcd4a3e466f579d4d05e2271daa8ff40556f7f36d462939f`;
- benchmark ledger SHA-256:
  `dd861a3c872a5483fdc79d262796ba01389d6fc53aee5c8fc2272045bb3de6c1`;
- Slurm allocation record SHA-256:
  `19d56958500682229367a5bc94f58dac4ee5b7826fe1032b9d0cd6eeb82917c5`;
- accepted comparison JSON SHA-256:
  `cdee5733e5cbd2d344a79ccee8e45d3e20656430c8146914dee2deae129dec2d`.

The reusable auditor is `scripts/audit_em_k1_box800_local_padding_gate.py`.
To reproduce the machine decision without rerunning either GPU arm, invoke it
with the two ledgers, stderr logs, HBM CSV files, and completion markers named
in the comparison JSON, requiring at least the observed conservative gates:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902
.pixi/envs/default/bin/python scripts/audit_em_k1_box800_local_padding_gate.py \
  --baseline-ledger /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_planner_fresh_ff6099641_20260902/runs/fresh_score_only_13363559/benchmark_ledger.json \
  --candidate-ledger /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_padding_guard_f91c73f29_20260902/outputs/score_only_13366512/benchmark_ledger.json \
  --baseline-log /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_planner_fresh_ff6099641_20260902/logs/fresh-score-only-13363559.err \
  --candidate-log /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_padding_guard_f91c73f29_20260902/logs/score-only-13366512.err \
  --baseline-hbm /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_planner_fresh_ff6099641_20260902/logs/fresh-score-only-13363559-hbm.csv \
  --candidate-hbm /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_padding_guard_f91c73f29_20260902/logs/score-only-13366512-hbm.csv \
  --baseline-completed /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_planner_fresh_ff6099641_20260902/runs/fresh_score_only_13363559/COMPLETED \
  --candidate-completed /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_local_padding_guard_f91c73f29_20260902/outputs/score_only_13366512/COMPLETED \
  --minimum-hbm-reduction-mib 17000 \
  --minimum-wall-reduction-s 500 \
  --output /absolute/new/output/padding-gate.json
```

This remains a score-only memory/performance qualification.  The advancing
full-particle trajectories and their checkpoint/FSC audits separately decide
final scientific quality.

## Live full-particle checkpoint-7 three-way audit

A read-only three-way audit compares zero-based checkpoint `it006` (numbered
iteration 7) from the release, compact, and corrected no-padding trajectories.
This is deliberately not called execution-exact: corrected-versus-release
hard pose assignments agree for 97.80%/97.70% of the two halves, rotation-
matrix correlations are 0.999588/0.999493, and map relative L2 differences are
0.00884/0.00919. Nevertheless, the resolved scientific diagnostics have not
moved: all three saved half-map FSC curves cross 0.5 at shell 150 and 0.143 at
shell 183, and corrected-versus-release map correlations are
0.9999608/0.9999576. The half-FSC curve RMSE is 0.0007522.

The corrected controller reports `current_size=496`, resolution shell 149
(`4.23 A`), order 4, and 589.3 s. The comparable compact and release iterations
took 3,683.1 s and 3,541.7 s. Corrected sampled HBM peaked at 39,073 MiB versus
70,871/70,305 MiB, a 44.87%/44.42% reduction. Its host cgroup peak was
356,348,030,976 bytes under the 500-GiB allocation and every memory-event
counter remained zero.

The compact JSON and Markdown are sealed, independently of the three source
run roots, under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_checkpoint7_threeway_20260903T0435`
with `SAFE_TO_DELETE`. Their SHA-256 values are
`4dc0a318e2f3e332306fb52de28f63cabd472cbda3caaa9f7659d5655b64fc92`
and `a6427feb0c12db158b2384f4e3cb2846b57243279a82c8fa86359c498b6ded75`;
`MANIFEST.sha256` verifies the retained tool and outputs. This checkpoint is
scientifically close by map/FSC but fails strict state equivalence. Final
acceptance remains tied to the converged direct RECOVAR-versus-RELION
scorecard, not to this same-engine checkpoint.

## Full-particle checkpoint-seeded local confirmation

Job `13368258` confirms that the accepted supplied-projector path also crosses
the local-search boundary with a realistic particle count and a mature
box-800 state.  It used all 15,258 particles from half 1 of EMPIAR-10202 set 6,
the release control's zero-based iteration-4 half map, deposited particle
poses and CTFs, radialized iteration-4 noise, I1 symmetry, and
`current_size=414`.  Half 2 was intentionally empty and M-step accumulation
was disabled so that this remains a bounded local-search execution and memory
diagnostic, not a reconstruction or FSC result.

The run used source commit
`f297d7218258e1ef9e461cb0591e6a7dcdc1c436`, tree
`0f70ac2f6ff49d5197f488c78f7c6518e320404f`, which includes the unused
local-padding removal.  It requested and received exactly
`cpu=4,mem=500G,node=1,billing=40,gres/gpu=1`, ran non-exclusively on one H100
on `della-h19g3`, and completed `0:0` in 11:38.  Its scientific step used
`MaxRSS=173601468K`.

| Measurement | Result |
| --- | ---: |
| Active particles | 15,258 in half 1 |
| Symmetry / box / current size | I1 / 800 / 414 |
| Staged particle stack | 78.12 GB |
| Projector build | 104.61 s |
| Projector-ready to parent-loop start | 2.546 s |
| Parent pass | 2,245,686 local rotations; 241.0 s |
| Fine pass | 161,016 local rotations; 44.5 s |
| Refinement-reported wall | 512.723 s |
| Whole-job sampled peak HBM | 20,125 MiB |
| Exact-local-window sampled peak HBM | 15,305 MiB |

As a contemporaneous diagnostic, the older full-particle compact trajectory
at commit `8069ac015` spent 457.167 s between its iteration-6 projector-ready
record and parent-loop start, while constructing the dead padded mean, and
reached 44,793 MiB in that interval.  This is not a formal A/B: its
parent-pass Fourier size and evolving state differ from job `13368258`.
Accordingly, the accepted quantitative before/after claim remains the matched
64-particle gate above.  The full-particle result establishes the narrower
fact that the fixed path starts both local loops promptly, completes every
particle without OOM, and stays far below the H100 memory limit.

The immutable run and runtime roots both carry `SAFE_TO_DELETE` markers:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/box800_fullhalf_local_checkpoint_f297d7218_20260903`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/box800_fullhalf_local_checkpoint_f297d7218_20260903`.

Important replay identities are:

- launcher SHA-256:
  `47e39b5055ceced5491e106b227ea4479a50593199c462b02ea66456b386ddfd`;
- seed-builder SHA-256:
  `a1afcea5b4994a406d0e7aba62945ce1e6068b4dff1a960b4b2064e8f4ce52cd`;
- checkpoint pose/noise seed SHA-256:
  `445fc2d866fa6ef2e68090cd5caf2fc5b993cf3bff8b7b72575beb5aef46100c`;
- benchmark ledger SHA-256:
  `1bd969556347cdca1da1093427f65b053ed8d3e10454d3826eae9adb78021da1`;
- stderr SHA-256:
  `923327503857eed5e99d2da300d5a245b079a9d68045f1cc3bef31ae8198950a`;
- 1-second HBM trace SHA-256:
  `7cd13ee04d4874933f5f4ae41c04ff8485277858d52bb01e079c8bf65ea7d8e0`;
- resolved command SHA-256:
  `3359e1edcf2f20f89aaa855d7479a24d342878e895b53d5e844c070bcaa4a802`;
- diagnostic decision JSON SHA-256:
  `bc0ece73d6f68d6b12e48e0715fb5a180042c0c6dd5227576a3f2ee12af56c5c`.

The input manifest captured the provided CUDA binary as `ad3b4a38...`, but
RECOVAR rebuilt that optional library during execution from the unchanged CUDA
source (SHA-256 `ce86b544...`), producing `60f86edf...`.  The result therefore
pins the source and runtime log rather than claiming that the originally
provided binary remained immutable.  The accumulator-donation gate below
copies the rebuilt binary into a private read-only path and pins both hashes.

To repeat the diagnostic, create new empty run and runtime roots, copy the
sealed builder and launcher, replace their output roots, rebuild the seed from
the pinned iteration-4 inputs, and submit the copied launcher with its new
SHA-256 in `EXPECTED_LAUNCHER_SHA256`.  Accept only an exact one-H100
allocation, a clean pinned source/tree, both complete bucket-loop records, no
fatal/OOM marker, and the output `COMPLETED` marker.

## Accepted exact-local accumulator donation

Commit `4749f6ad9` routes exact-local RELION x-half row updates through the
existing donating indexed-adjoint wrapper.  All four production call sites
consume and immediately replace their `Ft_y` or `Ft_ctf` accumulator.  Native
full-layout updates remain non-donating.  This exposes the existing CUDA FFI
input/output alias to XLA and prevents preservation of a second full BPref
accumulator; no projection, score, posterior, backprojection arithmetic, or
symmetry operation changes.

The matched H100 gate uses the iteration-11-sized `current_size=498` BPref
geometry.  Its full logical reconstruction grid is `(999,999,999)`, with
499,000,500 stored x-half elements in `(999,999,500)`.  A complex64 numerator
plus float32 weight pair is exactly 5,988,006,000 bytes, or 5,710.607529 MiB.
Both arms use one indexed row and the same operands, CUDA source and read-only
binary; the only source difference is accumulator donation.

| Measurement | Job 13369477, before | Job 13369478, after | Change |
| --- | ---: | ---: | ---: |
| Job state / elapsed | `COMPLETED 0:0` / 18 s | `COMPLETED 0:0` / 18 s | matched |
| Sampled peak HBM | 11,991 MiB | 6,279 MiB | -5,712 MiB (-47.64%) |
| Numerator input deleted / pointer reused | no / no | yes / yes | donated |
| Weight input deleted / pointer reused | no / no | yes / yes | donated |
| Checked numerator maximum | 2.2360677719 | 2.2360677719 | exact |
| Checked weight maximum | 3.0 | 3.0 | exact |

The observed 5,712-MiB decrease is 1.000244 times the exact donatable pair,
which is agreement to the 1-MiB sampling resolution.  Both jobs requested and
received exactly `cpu=4,mem=64G,node=1,billing=5,gres/gpu=1`, each saw one
H100, and each recorded `OverSubscribe=OK`.  The focused CPU row-chunk gate
passes two tests, including native fail-closed routing and the complete
three-chunk accumulator chain.  `pixi run test-em-fast-guard` passes all 16
tests in 58.62 s.  A pre-existing import-order finding and two pre-existing
unused imports prevent a clean whole-file Ruff result; Ruff passes on the
changed code when exactly those three pristine-HEAD findings are excluded.

The run, runtime, and detached-source roots are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/exact_local_donation_cs498_9006957c6_20260903`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/exact_local_donation_cs498_9006957c6_20260903`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/source_exact_local_donation_cs498_20260903`.

All carry `SAFE_TO_DELETE`.  Replay identities are:

- auditor SHA-256:
  `15fbe8929d316dd8ae70f61b3ff269f6079ce3e919d608b49d667f635afc30f7`;
- launcher SHA-256:
  `3af5b42b9fdd0c37a3b2dca1df57b23fe24c95bdae17e85fce7fb82de9de746f`;
- CUDA source / binary SHA-256:
  `ce86b544f3d831d48c63f7b9dff5c39db724925a5fa78c82c159409e879417b7` /
  `60f86edf44153aad98fb97487e60d4e2a76c10d322c723a765d4b6764fabda67`;
- baseline / candidate result SHA-256:
  `a60befd8b1e92d6437796103a1f00b17ccd79756f00b0c6cf127aa907271b2ac` /
  `2ceb70fc18b68ee17fa62b9b1eaf3552c76d02e595a58a7c5de301b965469816`;
- accepted comparison JSON SHA-256:
  `3c48ca363468700db17e11770650d7950ecc6899f52f23b074319f1dbc1c37d1`.

The rejected jobs `13369318`, `13369320`, `13369410`, and `13369411` are
retained as fail-closed harness attempts.  They reached neither the test
tensors nor a memory/science decision; their two corrected harness causes are
recorded in the run provenance.  Reproduce by copying the two detached source
commits, auditor, launcher, and read-only CUDA binary into fresh marked roots,
updating all absolute paths and expected hashes, and submitting one arm for
each source.  Accept only exact resources, clean imports, both pointer-
ownership contracts, identical checked numerical values, and an HBM decrease
within sampler resolution of 5,710.607529 MiB.
