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
