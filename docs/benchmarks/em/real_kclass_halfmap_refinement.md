# Real-data K=4 independent-half refinement

This is a runnable, bounded component of the Tier-6 harness for genuine
EMPIAR-10073, EMPIAR-10076, and EMPIAR-10345 K=4 half-map evidence. It is execution infrastructure, not a
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

## Frozen dataset inputs

### EMPIAR-10076

The source fixture is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_fixture_20260712/data`.
The four common starting maps are the numbered iteration-0 maps under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_initialmodel_realgate_3942224f5_20260901/pair/relion`.
They are supplied to both half processes and low-pass filtered to 30 A before
the first expectation. Their use is explicit in the report because they were
estimated from both particle halves.

Three profiles are frozen for EMPIAR-10076:

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

### EMPIAR-10345

The native-grid 10,000-particle fixture is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_real10345_10k_fixture_v1_20260823/data`.
It selects 10,000 rows in immutable source order from the 64,174-particle
stack and freezes 5,000 particles in each external half. The fixture labels
were generated once with seed `20260823`; they are a reproducible scientific
split, not a deposited `rlnRandomSubset` field. The native stack is
`/projects/CRYOEM/singerlab/mg6942/10345/recovar_data/particles.256.mrcs`
(22,089,827,328 bytes; SHA-256
`7909a695db68b65bfe6d0391054a1b19ae37fc4cd8da5cc4eb9595d76e4116e4`).
For this filtered dataset, `source_indices.npy` stores row indices into the
frozen 64,174-row source STAR rather than physical MRC-stack indices. The
launcher and auditor therefore reselect those source-STAR rows and require
their ordered `rlnImageName` values to reproduce the fixture exactly. The
source STAR itself is frozen at SHA-256
`8ab202046b07914c45636df73f1e6551a20c1f4476cb72797b1df9e9cd107b12`.
The older 10076 fixture keeps its original direct stack-index contract; the
two conventions are explicit in each generated manifest.

The four common 256-grid starting maps are the numbered iteration-0 maps in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10345_offset_prior_fullpair_92438c285_20260901/pair/relion`.
As for 10076, both engines receive the same maps, and each independent process
low-pass filters them to 30 A. The manifest records that these shared starting
maps were estimated using both halves. EMPIAR-10345 is currently admitted only
for `native10k-256`; there is no qualified 128-grid stack. Selecting either
128-grid profile with `--dataset 10345` fails before creating a run root.

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
Every input, including each multi-gigabyte particle stack, is fully rehashed
while preparing the run and is rehashed again inside the Slurm job before
either engine starts.

### EMPIAR-10073 prospective calibration contract

The 10073 cell is deliberately narrower than a registry-admissible Tier-6
campaign. It is a native-grid, 10,000-particle calibration run with exactly
5,000 particles in each external half, `K=4`, `C1`, eight iterations, seed
`42001`, three RELION MPI ranks (two followers), and a 250 A particle
diameter. The launcher derives the fixture from the canonical 138,899-particle
RECOVAR stack and metadata using these frozen inputs:

- `/projects/CRYOEM/singerlab/mg6942/10073/recovar_data/particles.256.mrcs`,
  SHA-256
  `d0d8a932ad76d228599fe622aa2291f613c108338007134b62226077acb6e2c9`;
- `/projects/CRYOEM/singerlab/mg6942/RECOVAR_datasets/10073/poses.pkl`,
  SHA-256
  `992d7496bd340f8c1974afd17201014788bf90492b5826770dd2dfe013e7073d`;
  and
- `/projects/CRYOEM/singerlab/mg6942/RECOVAR_datasets/10073/ctf.pkl`,
  SHA-256
  `6e20b1397669dfda6c54bede2744af64354bc9e4be894ada0a034b119f57908c`.

Selection is `sort(default_rng(20260903).choice(138899, 10000,
replace=False))`. A separate PCG64 stream, seed `20260904`, shuffles exactly
5,000 labels of each half before they are attached in selected source-row
order. `source_indices.npy` retains the physical zero-based MRC-stack indices.
Euler angles are obtained with the checked-in `R_to_relion` conversion.
Fractional RECOVAR translations are converted to Angstrom with the physical
field of view, `380 * 1.4000112`; the 256-grid pixel size is therefore derived
from the CTF metadata rather than the stale 1 A MRC header. The generated STAR,
index array, selection/half-label byte streams, conversion environment, and
all source and output hashes are sealed before submission.

Four distinct heterogeneous maps are selected at fixed positions along the
historical focused zdim-4 path: `vol000`, `vol003`, `vol006`, and `vol009`
under
`/projects/CRYOEM/singerlab/mg6942/10073/recovar_data/path0/all_volumes`.
Their respective frozen SHA-256 values are
`5add97a9df6c12d922d5d7747229968de8662a98a9fd227debefabc997aaff4c`,
`3c4a75c9a76936466696b6c56d502bd031fcd7cf4c634addbf81dc2e95ee1fb1`,
`21e97f5f9e144e68f22a936083194b3292925298c569d7ceeac02b03871e6a48`,
and
`8d33783d95befad4ea61452ad26c23fa1663a87d09f0902c156070ad97bc7fb2`.
The exact analysis command and upstream pipeline command are retained in
`path0/run.log` and `cont-indnocont-focmask/run.log`; the historical producer
commit was not recorded and is reported as unavailable, never guessed. This
is acceptable only because these maps are common hashed starting inputs, not
an outcome being compared between engines.

The raw path maps are never passed to either refiner. During clean dry-run
preparation, the checked-in launcher loads every map in the RECOVAR frame,
applies the same deterministic spherical hard low-pass with all Fourier
coefficients above `1 / 30 A` set to zero, converts the result to float32, and
writes one RECOVAR-frame and one RELION-frame representation. Intended-reader
round trips must reproduce the same internal float32 array exactly. The
derivation report pins the clean source commit, Python/NumPy/mrcfile versions,
source and derived hashes, command line, cutoff, and algorithm. Before a job
may be submitted, every derived map must satisfy both frozen spectral-leak
limits: out-of-band Fourier-energy fraction at most `1e-10`, and maximum
out-of-band coefficient magnitude divided by the maximum in-band magnitude at
most `1e-5`. The four derived maps must also remain diverse: every pair's
whole-box centered correlation must be at most `0.98`, and every pair's mean
FSC over shells 1--16 must be at most `0.97`. All class priors in the RELION
reference STAR are exactly 0.25, and the completed-run no-collapse gate still
requires at least one assigned particle in every class.

These references were estimated historically from particles spanning both
new external halves. Consequently, even a passing one-seed run is a Tier-A
relative parity diagnostic: it may establish complete execution, valid
provenance, noncollapsed K=4 topology, and candidate per-class agreement, but
it is not an independent absolute-resolution measurement and cannot enter the
accepted registry. `absolute_resolution_claim=false` and
`phase_randomization_corrected=false` remain immutable. Tier-B promotion
requires fresh runs for all three predeclared seeds 42001, 42002, and 42003,
every existing per-seed half-map/assignment/permutation-margin threshold, and
the existing validated multi-seed assignment and map-stability summaries. A
failed seed is retained; no favorable seed may be selected post hoc.

### EMPIAR-10073 fixed-eight outcome: early-trajectory rejection

The seed-42001 calibration completed all four engine/half processes in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10073_native10k_tiera_seed42001_827aedd66_submitted_20260903`.
Setup job `13381089` completed `0:0`; qualification job `13381090` ended `3:0`
only because the frozen prospective science gate rejected.  Every engine
process itself exited zero, every class remained occupied, both particle
audits completed without a threshold failure, and no OOM signature or runtime
error was present.

This is explicitly a bounded early-trajectory result, not a final K=4 quality
verdict.  RECOVAR stopped at current sizes 58/58 with 93.24% and 97.16% of
assignments still changing in the two external halves.  It did not report
convergence and did not run final all-data.  The eight-iteration endpoints are:

| Class | Unmasked merged cross-engine FSC-AUC | Common-mask merged cross-engine FSC-AUC | Unmasked / masked half-map FSC-AUC delta, RECOVAR - RELION |
| ---: | ---: | ---: | ---: |
| 1 | 0.981925 | 0.980378 | -0.000459 / +0.010877 |
| 2 | 0.962807 | 0.973140 | -0.020928 / -0.029382 |
| 3 | 0.994158 | 0.994521 | +0.000967 / +0.001525 |
| 4 | 0.980283 | 0.987818 | -0.005705 / +0.009196 |

Every same-half unmasked cross-engine FSC-AUC is at least 0.94756, and all
class-permutation margin gates pass.  Hard-label agreement is 0.9372 in half 1
and 0.9700 in half 2.  The frozen gate rejects classes 1, 2, and 4 at the 0.99
merged-map threshold, class 2 at both half-map-loss thresholds, and both
assignment thresholds.  These failures remain visible; the otherwise close
maps do not relabel the run as an acceptance.

Serial same-H100 wall times are 144.26/145.27 seconds for RELION and
1571.56/1656.84 seconds for RECOVAR.  Sampled peak HBM is 79,591/79,593 MiB
for RELION and 33,493/33,497 MiB for RECOVAR.  The four measured engine runs
consume 0.9772 aggregate H100-hours; including setup, the campaign consumes
1.1039 H100-hours and 62.2 minutes elapsed.  A follow-up must therefore retain
within-half device matching while running the two independent external halves
on isolated one-H100 jobs, then join them in a CPU audit after both converge.

The submission manifest SHA-256 is
`1dc1a26c9b830c24d58645c01135f3203f145af3031f1a986a2a2435ac98ab30`.
The authoritative audit JSON SHA-256 is
`2f9faee7dc3a28484d94b1b120c4d914e5a46bb7e0f170b3e981f1c5bcd609d2`;
its FSC archive SHA-256 is
`8dbc064324d1321513257af49fe58adc94c28966332b01bef64413af2cc37d81`,
and the common-mask SHA-256 is
`a6a9ecebc467bc5b290ca6494fecb396353b7c1a016d32e27cc1d929688586ce`.
The compact checked-in record is
`docs/benchmarks/em/diagnostics/real-k4-10073-native10k-fixed8-seed42001-827aedd66-20260903.json`.

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

Focused H100 regression job `13349285` independently exercises the retained
K=1 sign-continuity path, the new K-class native-sign path, K-class shell-prior
reconstruction, and both ordinary and adaptive K-class final-iteration routes.
All five tests passed in 74.97 s (Slurm elapsed 1m21s) with exact requested and
allocated one-H100, eight-CPU, 64-GiB resources. Its log and sealed launcher are
under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_native_signfix_unit_gate_7136e5c8d_20260902`.

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

## Three-seed 10k/128 outcome: stable sign, seed-sensitive classes

All three fixed-commit runs completed both engines, both independent halves,
all eight iterations, map alignment, assignment joining, and half-map analysis.
The qualification jobs have Slurm state `FAILED` and exit code 3 because the
auditor deliberately exits nonzero when a prospective science gate is
rejected; this is not a compute crash or an incomplete report. Every job used
one H100, 24 CPUs, and 64 GiB without `--exclusive`, with exact requested and
allocated resources.

The K-class sign fix removes the deterministic collapse. All final class
populations are nonzero, all four-class map permutations are unique, and the
best permutation is the identity for every engine and half. The strict
prospective gate nevertheless rejects all three seeds. Same-seed hard-label
agreement remains only 0.742--0.787 in half 1 and 0.757--0.768 in half 2, well
below the frozen 0.99 threshold. Seed 42001 also has a real weak-class outlier:
class 3 loses 0.1193 masked half-map FSC-AUC relative to RELION and has merged
cross-engine FSC-AUC 0.8444. That result is retained and is not averaged away.

| Seed | Assignment agreement h1 / h2 | Per-class masked half-map FSC-AUC delta, RECOVAR - RELION | Per-class merged cross-engine FSC-AUC | Prospective gate |
| ---: | ---: | --- | --- | --- |
| 42001 | 0.7840 / 0.7568 | `[-0.0268, -0.0028, -0.1193, +0.0017]` | `[0.9620, 0.9782, 0.8444, 0.9662]` | rejected |
| 42002 | 0.7422 / 0.7590 | `[-0.0086, +0.0049, +0.0069, -0.0113]` | `[0.9606, 0.9800, 0.8519, 0.9641]` | rejected |
| 42003 | 0.7870 / 0.7680 | `[+0.0151, -0.0030, +0.0135, +0.0022]` | `[0.9554, 0.9792, 0.9626, 0.9680]` | rejected |

Across all 12 class/seed rows, the paired masked half-map FSC-AUC delta has
median -0.00054, mean -0.01063, and range -0.11925 to +0.01512. The median
merged cross-engine FSC-AUC is 0.96338. Class 3 is the only strongly unstable
class: its paired delta has median +0.00694 but mean -0.03293 because of the
seed-42001 outlier.

A separate discriminator compares the same particles across seeds. For half
1, within-engine cross-seed hard-label agreement spans 0.5300--0.5866, whereas
same-seed RECOVAR-versus-RELION agreement spans 0.7422--0.7870. For half 2 the
corresponding ranges are 0.5444--0.5906 and 0.7568--0.7680. Thus, in both
halves, the weakest same-seed cross-engine agreement exceeds the strongest
within-engine cross-seed agreement. This is evidence that much of the label
difference reflects seed-sensitive K-class local optima shared by both engines,
not a deterministic RECOVAR-only assignment defect. It does not rescue any
failed map-quality threshold, prove native-grid parity, or justify choosing a
favorable seed.

A second, map-level discriminator reuses each audit's sealed proper-rigid
per-set transform, merges the genuine particle halves, then registers each
engine/seed map set to seed 42001 with one additional proper-rigid transform
shared by all four classes. It fits no reflection, density sign, or scale. The
metric is full unmasked non-DC FSC-AUC, so this is a relative map-stability
test rather than an absolute-resolution claim.

Across all 12 same-seed cross-engine class cells, FSC-AUC spans
0.83517--0.96012 (median 0.93745). The 24 within-engine cross-seed class cells
span 0.51403--0.87815 (median 0.81163). More decisively, the same-seed minimum
exceeds the within-engine cross-seed maximum separately for every class:

| Class | Same-seed RECOVAR--RELION minimum | Within-engine cross-seed maximum |
| ---: | ---: | ---: |
| 1 | 0.92532 | 0.83506 |
| 2 | 0.95272 | 0.87815 |
| 3 | 0.83517 | 0.64398 |
| 4 | 0.93779 | 0.82966 |

All best class permutations are the identity. Even weak class 3 is therefore
far closer between engines under the same seed than it is between seeds within
either engine. This supports taking the pilot result as a bounded win on the
causal question: the remaining final-map spread is dominated by shared
seed-sensitive local optima, not a deterministic RECOVAR-only weak-class bug.
The original per-seed gates and the seed-42001 -0.1193 masked half-map FSC-AUC
outlier remain recorded and rejected.

The map discriminator is reproducible by a semantic contract rather than a
byte-comparison of continuous optimizer output. A cross-CPU replay on
`della-i13n15` (job `13353671`) completed the analysis but deliberately remains
recorded with Slurm state `FAILED`: its original harness ended with `cmp`, and
the proper-rigid Powell fit moved slightly on that CPU. All frozen inputs,
decisions, row identities, and class permutations were exact; same-seed FSC-AUC
was unchanged, and the largest within-engine cross-seed FSC-AUC cell shift was
0.003631. The scientific conclusion remained separated for every class, with a
minimum per-class margin of 0.07447. The checked gate therefore permits at most
0.005 absolute FSC-AUC drift, requires at least 0.05 per-class separation, and
never requires continuous fit parameters to be byte-identical. A fresh
qualified replay, job `13354170`, passed that contract with exit 0 on exact
four-CPU/64-GiB resources.

The same-GPU serial resource measurements across the six half-runs are:

| Engine | Wall time, median (range) | Peak HBM, median (range) | Host MaxRSS, median (range) |
| --- | ---: | ---: | ---: |
| RELION | 380.76 s (376.58--399.50) | 79,588 MiB (79,573--79,589) | 1,138,936 KiB (1,121,836--1,156,104) |
| RECOVAR | 1,280.24 s (1,230.22--1,382.47) | 33,478 MiB (33,477--33,483) | 9,825,424 KiB (9,762,872--9,953,972) |

RECOVAR is 3.36 times slower in this bounded profile but uses 42.1% of
RELION's sampled peak HBM. These are formal same-hardware measurements for this
profile, not an extrapolation to native resolution.

The checked-in diagnostic is
`docs/benchmarks/em/diagnostics/real-k4-pilot10k-multiseed-stability-7136e5c8d-20260902.json`.
The complete aggregate report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_multiseed_stability_7136e5c8d_20260902/analysis/multiseed_stability.json`
(SHA-256
`80255e714145c0ab77b24b48a25b850055baa5f17a58482333fb402bb919947d`).
Reproduce it from the three immutable per-seed audits with:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902
pixi run python -m scripts.aggregate_em_real_kclass_halfmap_seeds \
  --expected-seeds 42001,42002,42003 \
  --audit /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_native_signfix_seed42001_7136e5c8d_20260902/audit/halfmap_audit.json \
  --audit /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_native_signfix_seed42002_7136e5c8d_20260902/audit/halfmap_audit.json \
  --audit /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_native_signfix_seed42003_7136e5c8d_20260902/audit/halfmap_audit.json \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_multiseed_stability_7136e5c8d_20260902/analysis/multiseed_stability.reproduced.json
```

Reproduce the map-level discriminator with:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902
pixi run python -m scripts.analyze_em_real_kclass_multiseed_map_stability \
  --expected-seeds 42001,42002,42003 \
  --audit /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_native_signfix_seed42001_7136e5c8d_20260902/audit/halfmap_audit.json \
  --audit /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_native_signfix_seed42002_7136e5c8d_20260902/audit/halfmap_audit.json \
  --audit /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_native_signfix_seed42003_7136e5c8d_20260902/audit/halfmap_audit.json \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_multiseed_stability_7136e5c8d_20260902/analysis/map_stability.semantic-reproduced.json \
  --reference-report /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_multiseed_stability_7136e5c8d_20260902/analysis/map_stability.json \
  --verification-output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_multiseed_stability_7136e5c8d_20260902/analysis/map_stability.reproduction-verification.json \
  --max-fsc-auc-abs-delta 0.005 \
  --min-per-class-separation-margin 0.05
```

The sealed map report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_multiseed_stability_7136e5c8d_20260902/analysis/map_stability.json`
(SHA-256
`1afcc15f4ca42496ba8b202ecf6a4926a4bd7a6b21d7562f69e8e5a3ff1cc27e`).
CPU analysis job `13353269` completed in 1m17s with exact requested and
allocated four-CPU, 64-GiB resources and exit 0. Semantic reproduction job
`13354170` completed in 1m23s with the same exact allocation and exit 0. Its
verification report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_pilot10k_multiseed_stability_7136e5c8d_20260902/analysis/map_stability.reproduction-verification.json`
(SHA-256
`2e59490b389e750a1f9fbbc4bd0e86e52e4b2263c7b0d6a03e4d39bc2a1b189c`).

## Native-grid EMPIAR-10345 seed-42001 outcome

The first `native10k-256` independent-half run completed all four engine
processes and the schema-v3 audit at RECOVAR commit `2f6759608`.  This is a
substantially stronger outcome than the 10076 128-grid pilot: every class has
an unambiguous identity match, no class collapses, all four unmasked
RECOVAR-minus-RELION half-map FSC-AUC differences lie between `-0.00993` and
`+0.00392`, and each class has exactly the same RELION and RECOVAR unmasked
FSC=0.5 crossing shell.

It is nevertheless retained as a rejected single-seed diagnostic.  Two
unmasked merged cross-engine AUC values narrowly miss the frozen 0.99 gate,
class 4 misses the supporting masked half-map-delta gate, and hard assignment
agreement remains below 0.99.  The Slurm `FAILED` state and exit 3 therefore
mean threshold rejection only: both RELION and RECOVAR half processes exited
zero, stderr is empty, and no OOM, traceback, or fatal error occurred.

| Class | Unmasked half-map AUC, RELION / RECOVAR / delta | Direct merged / half-1 / half-2 AUC | Masked half-map delta | Unmasked FSC=0.5 shell, RELION / RECOVAR |
| ---: | --- | --- | ---: | --- |
| 1 | 0.605533 / 0.595606 / -0.009926 | 0.990280 / 0.980715 / 0.991829 | -0.003782 | 14 / 14 |
| 2 | 0.560829 / 0.564745 / +0.003916 | 0.989566 / 0.979865 / 0.989358 | +0.002089 | 13 / 13 |
| 3 | 0.653398 / 0.644528 / -0.008871 | 0.993138 / 0.988423 / 0.991162 | +0.002256 | 15 / 15 |
| 4 | 0.548895 / 0.539679 / -0.009216 | 0.987446 / 0.973209 / 0.988749 | -0.013877 | 12 / 12 |

Every AUC above uses the frozen RELION-unmasked band through shell 126.  All
registered 0.143 crossings remain beyond that measured band for both engines,
so this bounded run makes no absolute-resolution claim.  The matching 0.5
crossings correspond to 24.59, 26.49, 22.95, and 28.69 A.  Assignment
agreement is 0.9632 in half 1 and 0.9730 in half 2; corresponding class counts
are `[1124,1023,2023,830]` versus `[1118,1014,2034,834]`, and
`[1473,948,1658,921]` versus `[1463,962,1649,926]`.

The serial same-H100 wall times are 147.48/145.06 s for RELION and
1681.26/1708.03 s for RECOVAR.  Sampled peak HBM is 79,577/79,573 MiB for
RELION and 33,789/33,795 MiB for RECOVAR.  Thus RECOVAR uses about 42.5% of
RELION's HBM in this profile but is about 11.6 times slower; performance work
remains open independently of the close reconstruction-quality result.

Setup job `13371068` completed `0:0`.  Qualification job `13371069` requested
and received exactly `cpu=24,mem=256G,node=1,billing=24,gres/gpu=1`, without
exclusive allocation, and ended `3:0` after 1:04:34 solely because the frozen
gate rejected the five conditions above.  The authoritative audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10345_native10k_seed42001_2f6759608_20260903/audit/halfmap_audit.json`
(SHA-256
`84217ddbc0f5cf20967a14d2a1d29e12f1a5006114200e31659b4490c8ae9e5a`).
The adjacent curve archive has SHA-256
`cd497fe6c7731a9d97ff8c86fb4faf55917dd2d5bc68a8fbb71541124a23a620`.
The compact checked-in record is
`docs/benchmarks/em/diagnostics/real-k4-10345-native10k-seed42001-2f6759608-20260903.json`.

Seeds 42002 and 42003 subsequently completed the same frozen native-grid
contract. Across all three seeds, seed-matched RECOVAR/RELION maps are much
closer than either engine's maps across seeds for every class: full unmasked
non-DC map FSC-AUC is 0.98530--0.99314 within a seed and
0.62790--0.82231 across seeds. Common-mask merged FSC-AUC is
0.997499--0.998972 over all 12 seed/class cells. This supports a shared
seed-sensitive local optimum, while all three prospective per-seed gates
remain rejected. See `real_k4_10345_native10k_multiseed_20260903.md` and
`diagnostics/real-k4-10345-native10k-multiseed-2f6759608-20260903.json`.

An exact same-seed RELION half-1 repeat then calibrated execution-order noise.
Although MPI follower ownership changed for 31,104/40,000 particle visits, all
class assignments and raw poses were exact and all 32 numbered maps had signed
non-DC FSC-AUC at least 0.999999995094. This rules out ordinary same-engine
dispatch/reduction-order variability as the explanation for the much larger
cross-engine endpoint difference. See
`real_k4_10345_relion_repeatability_20260903.md` for the frozen launcher,
command, resource audit, and artifact hashes.

## Reproduction

The launcher is dry-run by default. Use a fresh root for every retry:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k4_halfmap_repro_harness_20260901
pixi run python -m scripts.launch_em_real_kclass_halfmaps_slurm \
  --dataset 10076 \
  --profile shared200-128 \
  --seed 42001 \
  --output-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_shared200_seed42001_<commit>_20260901
```

Inspect `submission_manifest.json`, `inputs.sha256`, and both files in `jobs/`.
Submission is an explicit separate action:

```bash
pixi run python -m scripts.launch_em_real_kclass_halfmaps_slurm \
  --dataset 10076 \
  --profile shared200-128 \
  --seed 42001 \
  --output-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10076_shared200_seed42001_<commit>_submitted_20260901 \
  --submit
```

The corresponding dry-run for the independently frozen native 10345 input is:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902
pixi run python -m scripts.launch_em_real_kclass_halfmaps_slurm \
  --dataset 10345 \
  --profile native10k-256 \
  --seed 42001 \
  --output-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10345_native10k_seed42001_<commit>_20260903
```

This native 10345 invocation is initially a bounded diagnostic. It is not
eligible for registry admission until the run and audit finish and the required
multi-seed stability evidence is available.

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

This bounded launcher reports hard-assignment agreement, populations,
significant-support summaries, maps, FSC, and resources; its retained
particle-state audits also contain Pmax, pose, and translation trajectories.
The three-seed assignment and final-map stability analyses are complete and
checked in as rejected diagnostics, but the frozen FSC and assignment
thresholds remain unmet. A green 128-grid pilot remains required before
native-grid execution can satisfy the full Tier-6 admission checklist.
