# Real-data K=4 independent-half refinement

This is a runnable, bounded component of the Tier-6 harness for genuine
EMPIAR-10076 K=4 half-map evidence. It is execution infrastructure, not a
completed benchmark result or the complete Tier-6 matrix. No entry may be
added to `entries/` until the Slurm run completes, the audit passes, and the
resulting artifacts are sealed.

## Why four processes are required

RELION rejects `--split_random_halves` when `nr_classes > 1`; its source tells
users to classify first and then refine classes separately. A single RELION
K=4 process therefore cannot emit four independent gold-standard half-map
pairs. RECOVAR's numbered K-class map labels are also internal process products;
their scientific role must be proved from the saved particle memberships, not
inferred from the filename or from whether two files happen to hash equally.

The harness uses this construction instead:

1. Preserve the deposited/frozen `rlnRandomSubset` labels and source-row order.
2. Write one disjoint particle STAR for subset 1 and one for subset 2.
3. Run an independent K=4 RELION process on each STAR.
4. Run an independent K=4 RECOVAR process on each STAR.
5. In each RECOVAR process, prove from `refinement_results.npz` that one
   internal membership contains every particle in that external half and the
   other contains none, then select only the occupied membership's numbered
   maps. Treat the occupied products from the two disjoint processes as the
   genuine half-map pair. Never use the empty-membership or final all-data
   products as half maps.

All four engine processes run serially on one physical H100. The setup/build
job is separate and is not included in either engine's runtime.

## Frozen EMPIAR-10076 inputs

The source fixture is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_fixture_20260712/data`.
The four common starting maps are the numbered iteration-0 maps under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_initialmodel_realgate_3942224f5_20260901/pair/relion`.
They are supplied to both half processes and low-pass filtered to 30 A before
the first expectation. Their use is explicit in the report because they were
estimated from both particle halves.

Three profiles are frozen:

| Profile | Particles | Grid | Host-memory request | Purpose |
| --- | ---: | ---: | ---: | --- |
| `shared200-128` | 200 (93/107) | 128 | 32 GiB | cheap wiring, topology, memory, and artifact discriminator |
| `pilot10k-128` | 10,000 (5,000/5,000) | 128 | 64 GiB | Tier-6 downsampled pilot |
| `native10k-256` | 10,000 (5,000/5,000) | 256 | 256 GiB | native-grid qualification after pilots pass |

The 128-grid requests are measurement-based: the sealed shared-200 job peaked
at 6,217,244 KiB RSS, while prior 10,000-particle K=4 pairs peaked at
17,722,084 KiB RSS. The native-grid request remains deliberately larger until
that profile has its own sealed peak-RSS measurement. GPU memory is monitored
separately at one-second cadence and is not inferred from these host-memory
requests.

The particle stacks, source STAR, source indices, selection, initial maps, and
instrumented RELION executable have frozen SHA-256 values in the launcher.
Every generated `rlnImageName` contains the sealed particle stack's absolute
path; the engines therefore cannot follow an unsealed run-local symlink while
the launcher verifies a different canonical file.
The executable, RELION base commit/tree, and exact tracked instrumentation
diff are sealed separately; no build-system attestation cryptographically
binds that executable to that source, and the report states this limitation.
The exact tracked patch is copied into each run's `provenance/` directory and
included in the in-job SHA-256 verification manifest; its content hash, rather
than a presentation-dependent line count, is the identity.
Every small input is rehashed while preparing the run. The particle stack is
size-checked during preparation and fully rehashed inside the Slurm job before
either engine starts.

## First-iteration native score boundary

The first material real-data K=4 disagreement was a particle-state handoff,
not a projection or score-arithmetic error. RELION Class3D discards input
orientations for its fresh global angular search, but still rounds each
`run_it000_data.star` origin and applies that integer pre-shift before the image
FFT. RECOVAR previously treated the complete pose as one indivisible state and
discarded both orientation and translation for K greater than one.

Commit `4a91369a3` loads only the two origin coordinates through
`ReplayState.init_previous_best_translations`. It deliberately supplies
`[None, None]` for the previous Euler angles and does not replay normalization,
scale, direction-prior, or noise state. The empty second accumulator used by an
independent all-data Class3D process is normalized to shape `(0, 2)` and has a
dedicated regression test.

The discriminator uses 16 frozen EMPIAR-10076 particles spanning all four
classes, same-class pose disagreements, and every observed class-confusion
direction. Native RELION pass-0/pass-1 float32 scores are passive captures from
the control-qualified binary; RECOVAR records the entire corresponding coarse
surface and sparse fine support. H100 job `13346151` completed in 2m19s with
one GPU, 8 CPUs, 96 GiB, exact requested/allocated resources, and exit 0. Its
result is:

- 16/16 winner classes and 16/16 global coarse poses equal;
- 64/64 per-class coarse best poses equal;
- 16/16 fine parents and 16/16 fine winners equal;
- 16/16 integer pre-shifts equal RELION's half-away-from-zero rounding;
- minimum full-surface correlation `0.999999999996` over 1,069,056 candidates;
- maximum centered coarse relative L2 error `2.83e-6`; and
- maximum absolute fine-score error `7.45e-8`.

The sealed report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_firstiter_originfix_panel16_4a91369a3_20260902/analysis/firstiter_cc_boundary.json`
(SHA-256
`977d115f1593a789f4c877645791a8658a9d645e8fb94996f54f2d81686d2e1a`).
The shape-only first attempt, job `13346011`, failed before scoring because the
empty second accumulator arrived as `(0,)`; it is retained as harness audit
history and contributes no scientific result.

Reproduce the sealed comparison from the retained captures with:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_reference_frame_fix_8b5a024ce_20260902
pixi run python -m scripts.analyze_em_real_k4_firstiter_cc_boundary \
  --panel /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_firstiter_cc_panel16_framefix_1ec0835_20260902/inputs/panel16.json \
  --native-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_firstiter_cc_panel16_framefix_1ec0835_20260902/native/capture/capture \
  --recovar-coarse-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_firstiter_originfix_panel16_4a91369a3_20260902/coarse \
  --recovar-fine-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_firstiter_originfix_panel16_4a91369a3_20260902/pass2 \
  --input-data-star /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_framefix_it1_1ac24b9ef_20260902/half1/relion/run_it000_data.star \
  --voxel-size 3.275 \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_firstiter_originfix_panel16_4a91369a3_20260902/analysis/firstiter_cc_boundary.reproduced.json
```

## Post-reconstruction K-class sign boundary

The first full `pilot10k-128` controls exposed a separate deterministic
second-iteration defect after the firstiter score boundary had closed. In all
three seeds, RECOVAR logged `Aligned shared class-3 volume sign to the previous
reference` after iteration 1. The heuristic compared each centered
reconstruction to its previous class reference and multiplied the
reconstruction by -1 when their overlap was negative. This is not a valid
Class3D ambiguity: the image and CTF convention determines the density sign,
and previous-reference overlap is unreliable for a weak class.

For seed 42001, the native iteration-1 class-3 reconstruction has normalized
FSC-AUC `+0.990488698` against RELION. The heuristic changed it to
`-0.990488698`, after which iteration-2 class-3 occupancy collapsed to
`0.0011606` rather than RELION's `0.038241`. The same explicit class-3
flip and collapse occurred in seeds 42002 and 42003.

Commit `7136e5c8d` preserves the data-determined sign for every K-class
reconstruction while retaining the already-qualified legacy K=1 continuity
path. The focused unit discriminator fails on the old behavior and passes
after the fix. The two-iteration H100 causal rerun, job `13348468`, then gave:

| Iteration | Quantity | Broken control | Fixed RECOVAR | RELION |
| ---: | --- | ---: | ---: | ---: |
| 1 | class-3 direct FSC-AUC | -0.990489 | 0.990489 | reference |
| 2 | class-3 occupancy | 0.001161 | 0.036801 | 0.038241 |
| 2 | four-class occupancy | [0.3331, 0.3886, 0.0012, 0.2771] | [0.3203, 0.3766, 0.0368, 0.2663] | [0.3189, 0.3724, 0.0382, 0.2705] |
| 2 | direct per-class FSC-AUC | [0.9918, 0.9949, -0.0938, 0.9918] | [0.9933, 0.9959, 0.9801, 0.9928] | reference |
| 2 | identity-label assignment agreement | 0.9146 | 0.9420 | reference |

The causal gate is a pass, not a final trajectory admission: iteration-2
assignment agreement remains below the prospective 0.99 final gate. The job
completed in 9m22s, exit 0, with exact requested/allocated one-H100, four-CPU,
128-GiB resources and peak sampled HBM 33,465 MiB. Its full JSON is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_native_signfix_half1_seed42001_it2_7136e5c8d_20260902/analysis/k4_native_signfix_causal_report.json`
(SHA-256
`25f3a772fc31d0fb630663e2f76407f89a174a9fe9aee21dd406f7e43a7a7ba8`);
the exact argv, Slurm script, logs, HBM trace, and products are in the same
immutable run root.

The original eight-iteration controls are retained as negative evidence:
seed 42001 jobs `13346676/13346677`, seed 42002 jobs
`13346756/13346759`, and seed 42003 jobs `13346757/13346758`. Fresh
fixed-commit three-seed independent-half runs are setup/qualification jobs
`13348863/13348864`, `13348862/13348865`, and
`13348861/13348866`, respectively. Their roots are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_native_signfix_seed42001_7136e5c8d_20260902`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_native_signfix_seed42002_7136e5c8d_20260902`; and
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_native_signfix_seed42003_7136e5c8d_20260902`.

The preceding setup job `13346480` succeeded, but its dependent job
`13346481` failed before science because the source worktree was transiently
dirty while this documentation was drafted. That immutable root is retained as
a harness failure and is not a scientific attempt.

## Reproduction

The launcher is dry-run by default. Use a fresh root for every retry:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k4_halfmap_repro_harness_20260901
pixi run python -m scripts.launch_em_real_kclass_halfmaps_slurm \
  --profile shared200-128 \
  --seed 42001 \
  --output-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_shared200_seed42001_<commit>_20260901
```

Inspect `submission_manifest.json`, `inputs.sha256`, and both files in `jobs/`.
Submission is an explicit separate action:

```bash
pixi run python -m scripts.launch_em_real_kclass_halfmaps_slurm \
  --profile shared200-128 \
  --seed 42001 \
  --output-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_shared200_seed42001_<commit>_submitted_20260901 \
  --submit
```

Immediately after each `sbatch`, the launcher records `scontrol show job -o`
and requires exactly one requested GPU, a nonexclusive allocation, and exact
`ReqTRES == AllocTRES` whenever Slurm has already allocated the job. At the
start of both the setup job and the qualification job, it repeats the exact
allocation check and seals the result before any build or science command. If
either new job fails the submission-time check, both newly submitted jobs are
cancelled; a job-start mismatch fails that job before work begins.

Because roots are immutable, do not add `--submit` to an already prepared dry
root. Prepare a new root. Run seeds 42001, 42002, and 42003 for the 128 pilot;
only after those are stable, run seeds 42001 and 42002 at 256.

## First bounded execution: rejected as a parity result

The `shared200-128`, seed-42001 computation completed for all four processes at
RECOVAR source commit `abaada2d66d35916a8b8826f93467e0d6af7f8e4` in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_shared200_seed42001_abaada2d6_submitted_20260901`.
Setup job `13328373` completed in 3:41. Qualification job `13328374` used one
H100, 24 CPUs, and 192 GB with exact `ReqTRES == AllocTRES`; all four engine
runs completed before the job exited 2 at the original auditor's incorrect
numbered-map semantic guard. The corrected schema-v3 auditor at commit
`82208f49f` proves the occupied/empty internal memberships described above and
produces a complete report. It exits 3 because the predeclared science gate
fails, not because report generation failed.

The matched per-class results are:

| Class | Source IDs RELION h1/h2; RECOVAR h1/h2 | Unmasked 0.143 A, RELION / RECOVAR | Common-mask 0.143 A, RELION / RECOVAR | Frozen-band half-map FSC-AUC, RELION / RECOVAR | Merged cross-engine band FSC-AUC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `1/2; 2/4` | 13.52 / 104.80 | 13.52 / 104.80 | 0.267 / 0.152 | 0.028 |
| 2 | `2/3; 4/3` | 83.84 / 16.12 | 83.84 / 16.12 | 0.298 / 0.670 | 0.736 |
| 3 | `3/1; 3/2` | 104.80 / 69.87 | 104.80 / 69.87 | 0.104 / 0.845 | 0.844 |
| 4 | `4/4; 1/1` | 20.96 / 20.96 | 20.96 / 20.96 | 0.436 / 0.319 | 0.400 |

These values are bounded relative diagnostics, not absolute-resolution claims:
the common-mask FSC is uncorrected and each half has only 93 or 107 particles.
More importantly, hard-assignment agreement is only 0.151 and 0.131 for halves
1 and 2. The class populations are `24/62/1/6` versus `4/14/70/5` in half 1
and `70/1/33/3` versus `9/81/14/3` in half 2 (RELION versus RECOVAR after label
matching). RECOVAR-half-1's unique class permutation also has absolute
best-to-second-best objective margin 0.00218, below the frozen 0.01 minimum.
The mixture has therefore split or merged differently; apparently better FSC
for classes 2 and 3 cannot rescue the result. This run is useful as a cheap
wiring and failure-localization discriminator, but it is not evidence that K=4
parity is achieved and it is not admitted to the accepted registry.

Small-run performance is recorded only as a resource regression point, not a
throughput claim:

| External half | RELION wall / peak HBM | RECOVAR wall / peak HBM |
| ---: | ---: | ---: |
| 1 | 15.24 s / 79,549 MiB | 404.39 s / 4,771 MiB |
| 2 | 14.27 s / 79,593 MiB | 374.11 s / 4,771 MiB |

The sealed report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_shared200_seed42001_abaada2d6_submitted_20260901/audit/halfmap_audit.json`
(SHA-256 `a74ffbef6b14b5ad1026a300d98e576cd11245acb583f238d4d80313bf321c44`).
The shellwise curves are in the adjacent `halfmap_fsc_curves.npz` (SHA-256
`359c692df4c6334262c1b47095dccae5d590d4d23eb1eeba45ca7b0fa36a24f6`).

## Fail-closed audit

`scripts/audit_em_real_kclass_halfmaps.py` requires:

- exact input order, split labels validated against the immutable origin STAR
  and source-index array, disjointness, complete union, input hashes, commands,
  clean source commit, exact setup and qualification nonexclusive one-GPU
  allocations, and a single physical GPU UUID;
- absolute runtime particle-stack paths in every selected and half STAR, bound
  to the one sealed stack artifact, plus all four engine wall records bound to
  the qualification allocation's Slurm job ID;
- exactly four numbered classes from the last expected iteration in each
  independent process;
- RECOVAR saved-image count, no final-all-data pass, disjoint and complete
  internal-half membership, exactly one membership containing all external-half
  particles, hard-assignment topology, and SHA-256 identities for both selected
  occupied maps and discarded empty-membership products; plus no duplicate class
  maps within an engine/half, no extra final class IDs, and no byte-identical
  maps across independent external-half processes;
- one proper rigid transform per four-class map set, fitted to a label-invariant
  equal-weight ensemble, followed by a uniquely optimal Hungarian class match
  to RELION half 1 with best-to-second-best objective margin at least `0.01`
  absolutely and `0.0025` relatively (both values are frozen in the manifest);
- one equal-weight, engine/half/class-symmetric common soft mask made from the
  nonnegative voxelwise RMS envelope of all 16 aligned unit-RMS maps, stored
  and hashed; and
- assignments joined by `rlnImageName`, not row position, with per-class
  populations and significant-support summaries.

The manifest freezes the fit shell, alignment search orders, interpolation,
crossing rule, and every mask parameter. The hashed run script passes every
one of these values explicitly to the auditor, so rerunning the audit with a
post-hoc parameter change fails closed.

For every matched class it writes shellwise masked and unmasked within-engine
half-map FSC, cross-engine half-1/half-2/merged FSC, normalized FSC-AUC, and
0.143/0.5 crossing diagnostics. The acceptance curves use proper-rigid
registration because a shared global coordinate-frame drift is scientifically
irrelevant; a single transform is shared by all four classes in each set.
Raw frozen-frame curves are retained as unmasked diagnostics and cannot rescue
an acceptance failure. A 0.143 crossing beyond the measured band is represented
explicitly rather than converted into an apparent finite resolution; when only
RELION remains beyond the band, the resolution comparison fails closed.
All masked and unmasked RECOVAR and cross-engine FSC-AUC comparisons are
integrated over one band frozen from the corresponding RELION unmasked
half-map's resolved non-DC shells. The comparison band is never shortened at
an earlier RECOVAR crossing or changed by masking, so a masked curve cannot
hide an unmasked resolution loss.

The common-mask curves are ordinary, uncorrected masked FSC: the harness does
not perform high-resolution noise substitution or phase-randomization
correction. They are labeled as relative RECOVAR-versus-RELION diagnostics and
must not be cited as absolute-resolution claims.

The prospective real-data gate is the policy already frozen in
`k4_validation_matrix.md`: RECOVAR common-mask 0.143 resolution may not trail
RELION by more than one Fourier shell or 5% (whichever is larger), masked and
unmasked half-map FSC-AUC over the frozen RELION unmasked-resolved band may not
drop by more than 0.01, registered merged cross-engine band FSC-AUC must be at
least 0.99, same-half cross-engine band FSC-AUC at least 0.90, hard-assignment
agreement at least 0.99, and no
class may collapse. Every accepted class permutation must also satisfy both
frozen objective-margin thresholds above.

An audit failure remains a result to diagnose, but it is not admitted as an
accepted registry entry. Masked FSC cannot rescue an unmasked failure.

This bounded launcher currently reports hard-assignment agreement,
populations, significant-support summaries, maps, FSC, and resources. It does
not yet report Pmax or pose/translation agreement, and one completed seed is
not a seed-stability result. Those metrics and the predeclared three-seed
aggregate remain required before this harness can satisfy the full Tier-6
admission checklist.
