# K=4 validation matrix

This is the execution plan needed to bring K=4 validation to approximately the
breadth already used for K=1. It is intentionally bounded: cheap causal gates
run first, medium trajectories run only after those pass, and production runs
run only after the medium tier is green. Seeds and thresholds are frozen in
the launch manifest before any result is examined.

## Gate policy

Every paired synthetic trajectory retains the existing formal gate:

- exact controller schedule/topology;
- Hungarian one-to-one class matching at every numbered iteration;
- direct unmasked cross-engine FSC-AUC at least 0.995 for every matched class
  and iteration;
- final RECOVAR-minus-RELION GT FSC-AUC delta at least -0.002 per class; and
- final matched hard-assignment agreement at least 0.99.

Formal failures remain failures. A separately frozen science-equivalence gate
may classify a completed run as scientifically equivalent when controller
topology matches, each final Hungarian-matched cross-engine FSC-AUC is at
least 0.99, every GT delta is at least -0.002, assignment agreement is at
least 0.99, and no class collapses. This classification never changes the
formal result. Every numbered RELION model STAR is audited before RECOVAR is
started: each class must have finite, strictly positive class-distribution and
orientation-distribution mass. The audit is written even on failure, and a
collapsed class stops that trajectory rather than being averaged away by a
multi-seed summary.

For real data, where GT is unavailable, freeze the following before launching:
the same particles/order, half split, poses/CTFs, initial maps, symmetry,
mask-generation command, and random seed for both engines; per-class masked
and unmasked half-map curves after one proper-rigid transform shared across
each four-class set; raw frozen-frame curves as non-rescuing diagnostics; a
common mask derived from the nonnegative RMS envelope of all aligned maps;
common-mask 0.143 resolution; registered cross-engine FSC; class populations;
and seed-to-seed stability. RECOVAR's masked 0.143 resolution must not be worse
than RELION by more than one Fourier shell or 5%, whichever is larger, and
masked/unmasked half-map FSC-AUC, integrated over a band frozen from RELION's
unmasked resolved non-DC shells, must not drop by more than 0.01. Class
matching must have a unique exact K=4 permutation with best-to-second-best
objective margin at least `0.01` absolutely and `0.0025` relatively. A missing
0.143 crossing is recorded as beyond the measured range, never as a finite
resolution; if only RELION remains beyond range, the comparison fails closed.
These real-data thresholds are prospective and must not be retrofitted to
existing runs.

Unless a run explicitly records a phase-randomization/noise-substitution
correction, its common-mask FSC and 0.143 crossing are uncorrected relative
diagnostics, not absolute-resolution claims. The current EMPIAR-10076 bounded
half-map harness records `absolute_resolution_claim=false` for this reason.

## Tier 0: fail-closed CPU controls

These run in every EM PR and should finish in minutes.

| Control | Expected result |
| --- | --- |
| Permute four output class labels | Naive identity comparison fails; Hungarian FSC/GT result is unchanged and reports the known permutation. |
| Duplicate one class map and omit another | Bijection/no-collapse check fails. |
| Corrupt one input hash, source tree, executable hash, or dispatch-schedule hash | Launcher or registry validation fails before science execution. |
| Alter one controller current-size or healpix row | Exact-topology and formal trajectory gates fail. |
| Supply only three final maps for K=4 | Completeness audit fails. |
| Add a masked FSC without a hashed common mask | Registry validation fails. |
| Mark a cross-hardware runtime ratio formal | Registry validation fails. |

The unit suite must include identity and nonidentity K=4 permutations, tied
assignments, duplicated maps, missing classes, and a negative GT association.

### Current-source executable K=4 controls, 2026-09-02

Three focused controls now cover previously implicit K=4 execution boundaries:

| Boundary | Executable evidence | Established scope |
| --- | --- | --- |
| Final all-data eligibility | Commit `1a8b3521a`: `TestRelionModeSmokeTest::test_relion_final_iteration_supports_k_class` and `TestRelionModeSmokeTest::test_relion_k4_does_not_finalize_after_max_iter_even_when_diagnostic_force_enabled` | A converged tiny K=4 refinement invokes exactly one final all-data pass and retains four class means plus both halves' final class assignments. A nonconverged K=4 refinement invokes no final pass, including when the K=1 diagnostic force-after-max environment switch is enabled. |
| Simultaneous dense image/rotation partitioning | Commit `1d9011a18`: `test_dense_k4_image_and_rotation_partition_equivalence` | A deterministic four-image, four-class dense global E/M step compares image batch 4 / rotation block 5 with image batch 3 / rotation block 2. Discrete class/pose outputs are exact; complex128 accumulator, evidence, and map reductions agree within the frozen `1024 * eps(float64)` bound. The odd five-rotation grid exercises a padded tail block. |
| Numbered global-to-local continuity | Commit `ca626914d`: `test_k4_numbered_global_to_exact_local_preserves_per_half_pose_state` | Numbered iteration 1 routes both halves through the real dense K-class orchestrator; a forced controller transition routes both halves through the real exact-local K-class orchestrator at iteration 2. Exact assertions bind each half's dense pose outputs to iteration-1 history, local rotation/translation priors and integer pre-shifts, and local pose outputs to iteration-2 history, while preserving four class means. No final all-data pass is involved. |

These are CPU unit controls, not trajectory-quality, FSC, HBM, or performance
evidence. The continuity control deliberately sets
`RECOVAR_K_CLASS_RELION_X_HALF_MSTEP=0`; it qualifies the numbered orchestration
and full-volume exact-local K-class handoff, not the GPU/CUDA x-half BPref
implementation. The dense partition control covers one simultaneous partition
pair, not the complete Tier-1 batch/block matrix.

The following focused commands were run from
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902`
on `della-mol.princeton.edu` with the checkout-bound pixi interpreter and the
qualified RELION binding:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u VIRTUAL_ENV \
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
  TMPDIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_gap_audit_authoritative/tmp \
  PIXI_HOME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_gap_audit_authoritative/pixi_home \
  RATTLER_CACHE_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_gap_audit_authoritative/rattler_cache \
  RECOVAR_RELION_BIND_BUILD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/pr158_symmetry_build_20260830/relion_bind \
  .pixi/envs/default/bin/python -m pytest \
  tests/unit/test_k4_dense_partition_equivalence.py::test_dense_k4_image_and_rotation_partition_equivalence -q
# 1 passed in 13.07s
```

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u VIRTUAL_ENV \
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
  TMPDIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_gap_audit_authoritative/tmp \
  PIXI_HOME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_gap_audit_authoritative/pixi_home \
  RATTLER_CACHE_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_gap_audit_authoritative/rattler_cache \
  RECOVAR_RELION_BIND_BUILD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/pr158_symmetry_build_20260830/relion_bind \
  .pixi/envs/default/bin/python -m pytest \
  tests/unit/test_refine_relion_mode.py::test_k4_numbered_global_to_exact_local_preserves_per_half_pose_state \
  tests/unit/test_refine_relion_mode.py::TestRelionModeSmokeTest::test_relion_k4_does_not_finalize_after_max_iter_even_when_diagnostic_force_enabled \
  tests/unit/test_refine_relion_mode.py::TestRelionModeSmokeTest::test_relion_final_iteration_supports_k_class -q
# 3 passed in 137.56s (0:02:17)
```

## Tier 1: fixed-state GPU discriminators

Use a sealed 32–256 particle panel at two current sizes (one coarse, one late
local) and all four classes. Compare score components, significant support,
posterior normalization, class/pose/translation winners, Pmax, noise/tau2,
weighted-sum accumulators, BPref, and regularized maps. Include:

- seeds 41001, 41002, and 41003;
- image batches 25, 50, and 100;
- rotation blocks 512, 2000, and 4096;
- dense/reference versus sparse/fused paths;
- CTF on/off and zero/nonzero translations;
- white and radial noise; and
- f32 production arrays plus the existing fp64 diagnostic oracle.

Batch and rotation-block variants must be bitwise identical where ordering is
fixed; otherwise the predeclared numerical tolerance and downstream FSC gate
both apply. Each discriminator records wall time and peak HBM so an exactness
fix cannot silently reintroduce rectangular K-class allocation.

The first real-data fixed-state subgate is now closed on the frozen
EMPIAR-10076 shared-200 panel. Commit `5f74755c2` matches native projected
references and Euler matrices exactly, and job `13337519` matches all
1,336/1,336 selected coarse candidates over 16 probes. Replay configuration
and 11 state arrays are exact; the minimum eight-map repeat FSC-AUC is
`0.999999997335`.

The subsequent passive native firstiter-CC capture closes the full score
surface on a 16-particle, all-class panel. RELION keeps the rounded input
origins when it resets Class3D orientations for the fresh global search;
RECOVAR had incorrectly reset both. Translation-only initialization at commit
`4a91369a3` preserves those origins without exposing the input orientations,
normalization corrections, priors, or noise. H100 job `13346151` then matches
16/16 global coarse poses, 64/64 per-class coarse poses, 16/16 fine parents and
winners, and all 16 integer pre-shifts. Across 1,069,056 coarse candidates the
minimum score correlation is `0.999999999996`; the maximum centered relative
L2 error is `2.83e-6`, and the maximum fine-score absolute error is
`7.45e-8`. The fail-closed analyzer and unit tests are commit `8cbebdecc`; the
sealed report and rerun command are documented in
`real_kclass_halfmap_refinement.md`. This covers one coarse/fine size and one
batch/block setting; the complete batch/block grid remains required.

The subsequent three-seed 10k/128 controls exposed a post-M-step defect rather
than a score-boundary regression: a previous-reference overlap heuristic
sign-inverted weak class 3 after iteration 1 in all three seeds. Commit
`7136e5c8d` removes that invalid K-class sign choice. Causal H100 job
`13348468` changes class-3 iteration-1 FSC-AUC from `-0.990489` to
`+0.990489` and restores iteration-2 occupancy from `0.0012` to `0.0368`
versus RELION's `0.0382`. Its iteration-2 per-class direct FSC-AUC values are
`[0.9933, 0.9959, 0.9801, 0.9928]`; identity-label assignment agreement
improves from `0.9146` to `0.9420`. This closes the sign boundary only.
Three full independent-half seeds, jobs `13348864`--`13348866`, are the
next trajectory gate. They subsequently completed without class collapse but
all three were rejected by the predeclared real-data gate. Across 12
class/seed rows the paired masked half-map FSC-AUC delta has median -0.00054,
but seed 42001 class 3 is a -0.11925 outlier. In both particle halves, the
weakest same-seed cross-engine hard-label agreement exceeds the strongest
within-engine cross-seed agreement, showing a substantial shared
seed-sensitive local-optimum component. This remains diagnostic rather than an
accepted result; exact jobs, FSC, assignments, performance, hashes, and the
reproduction command are in `real_kclass_halfmap_refinement.md` and
`diagnostics/real-k4-pilot10k-multiseed-stability-7136e5c8d-20260902.json`.

The final merged-map discriminator reaches the same conclusion independently:
same-seed RECOVAR--RELION full unmasked FSC-AUC is 0.8352--0.9601 across all
12 class cells, while within-engine cross-seed FSC-AUC is 0.5140--0.8782.
For each of the four classes, the same-seed minimum exceeds the within-engine
cross-seed maximum. All optimal class permutations are the identity. This
closes the weak-class causal triage at pilot scale without changing any frozen
admission decision.

## Tier 2: 10k/128 multi-seed synthetic trajectories

Run five autonomous iterations per case. The first twelve rows are K=4 and
each uses seeds 41001, 41002, and 41003. The four K-scaling sentinels use seeds
41001 and 41002.

| Family/regime | K | Noise/SNR axis | Poses/classes | Other stress |
| --- | ---: | --- | --- | --- |
| Ribosembly baseline | 4 | white 1 | uniform/uniform | CTF, zero shifts |
| Ribosembly high SNR | 4 | white 0.5 | uniform/uniform | CTF |
| Ribosembly low SNR | 4 | white 3 | uniform/uniform | CTF |
| Ribosembly radial | 4 | radial1 3 | nonuniform/linear | noise-scale std 0.2, contrast std 0.2 |
| Ribosembly pose skew | 4 | white 3 | Kent/head-heavy | CTF |
| Ribosembly imbalance | 4 | white 1 | uniform/head-heavy | minimum target class 5% |
| Ribosembly outliers | 4 | radial1 3 | nonuniform/linear | 20% outliers, shift std 0.5 |
| Ribosembly translation | 4 | white 1 | uniform/uniform | nonzero translations, CTF |
| IgG-1D | 4 | white 1 and radial1 3 | uniform and nonuniform/linear | contrast/noise scaling |
| Tomotwin-100 | 4 | white 1 and radial1 3 | uniform and Kent/head-heavy | family geometry |
| IgG-RL | 4 | white 1 and radial1 3 | uniform and nonuniform/linear | 0% and 20% outliers |
| CTF ablation | 4 | white 1 | uniform/uniform | identical images with CTF disabled |
| K scaling, Ribosembly | 2, 8, 16 | white 1 or 3 | uniform plus one head-heavy case | same grid/particles |
| Family/K interaction | 2, 8 | radial1 3 | nonuniform/linear | IgG and Tomotwin |

The runnable launcher now contains cases 1--36. The original 15 medium/scale
cases remain mandatory. Fourteen bounded 3k-particle cases add the K=1 stress
axes, exact batching controls, and independent seeds; cases 30, 35, and 36 are
successive no-CTF fixture discriminators; and cases 31--34 exercise C4, D4, O,
and explicit I1. This is substantially broader executable coverage, but a
single completed seed is still not sufficient evidence and only two rows
exercise 50k/256.

### Runnable base panel

The following exact `DEFAULT_CASES` from
`scripts/run_em_kclass_robustness_matrix_slurm.py` remain mandatory; none is
replaced by the broader table above:

| Index | Existing case name | Base seed |
| ---: | --- | ---: |
| 1 | `ribo_k2_10k_g128_white_noise1_uniform` | 2801 |
| 2 | `ribo_k4_10k_g128_white_noise1_uniform` | 2802 |
| 3 | `ribo_k4_10k_g128_radial_noise3_nonuniform_linear` | 2803 |
| 4 | `ribo_k8_10k_g128_white_noise3_kent_headheavy` | 2804 |
| 5 | `ribo_k4_50k_g256_white_noise1_uniform` | 2805 |
| 6 | `ribo_k4_50k_g256_radial_noise3_nonuniform_linear` | 2806 |
| 7 | `ribo_k16_20k_g128_white_noise3_uniform` | 2807 |
| 8 | `igg_k4_10k_g128_white_noise1_uniform` | 2808 |
| 9 | `igg_k8_10k_g128_radial_noise3_nonuniform` | 2809 |
| 10 | `ribo_k4_10k_g128_radial_noise3_nonuniform_outliers_pct20` | 2810 |
| 11 | `igg_k4_10k_g128_white_noise1_uniform_outliers_pct20` | 2811 |
| 12 | `tomotwin_k4_10k_g128_white_noise1_uniform` | 2812 |
| 13 | `tomotwin_k8_10k_g128_radial_noise3_kent_headheavy` | 2813 |
| 14 | `igg_rl_k4_10k_g128_white_noise1_uniform` | 2814 |
| 15 | `igg_rl_k4_10k_g128_radial_noise3_nonuniform_outliers_pct20` | 2815 |
| 16 | `ribo_k4_3k_g128_white_noise10_uniform` | 2816 |
| 17 | `ribo_k4_3k_g128_radial_noise3_noctf_uniform` | 2817 |
| 18 | `ribo_k4_3k_g128_white_noise1_contrast_noise_scale` | 2818 |
| 19 | `ribo_k4_3k_g128_white_noise1_image_offset` | 2819 |
| 20 | `ribo_k4_3k_g128_radial_noise5_severe_outliers_pct50` | 2820 |
| 21 | `ribo_k4_3k_g128_white_noise0p2_uniform` | 2821 |
| 22 | `ribo_k4_3k_g128_white_noise0p2_kent_headheavy` | 2822 |
| 23 | `ribo_k4_3k_g128_white_noise1_extreme_class_imbalance` | 2823 |
| 24 | `ribo_k4_3k_g256_radial_noise3_highres` | 2824 |
| 25 | `ribo_k4_3k_g128_white_noise1_batch50` | 2825 |
| 26 | `ribo_k4_3k_g128_white_noise1_batch17` | 2825 |
| 27 | `ribo_k4_3k_g128_white_noise1_rotation_block257` | 2825 |
| 28 | `ribo_k4_3k_g128_white_noise1_seed3802` | 3802 |
| 29 | `ribo_k4_3k_g128_white_noise1_seed4802` | 4802 |
| 30 | `ribo_k4_3k_g128_white_noise1_noctf_positive_control` | 2830 |
| 31 | `ribo_k4_5k_g128_white_noise1_c4_uniform` | 41001 |
| 32 | `ribo_k4_5k_g128_white_noise1_d4_uniform` | 41001 |
| 33 | `ribo_k4_5k_g128_white_noise1_o_uniform` | 41001 |
| 34 | `ribo_k4_5k_g128_white_noise1_i1_uniform` | 41001 |
| 35 | `ribo_k4_10k_g128_white_noise0p2_noctf_strong_control` | 2835 |
| 36 | `ribo_k4_10k_g128_white_noise0p2_noctf_resolved_init20` | 2836 |

Case 17 is deliberately retained as the low-SNR no-CTF collapse boundary. A
collapse is a fail-closed result, not a successful parity row. Cases 30, 35,
and 36 tested progressively more favorable no-CTF controls over all three
frozen seeds. All nine RELION trajectories assigned zero class-3 mass at
iteration 1, so the harness correctly stopped before RECOVAR. They are
completed excluded diagnostics, not parity results; see
`k4_noctf_negative_boundaries_20260901.md` and
`diagnostics/k4-noctf-collapse-cases30-35-36-h100.json`.

Run any selected panel over the frozen seed set 41001, 41002, and 41003 with
`--three-seed-suite`. The launcher expands every selected base case inside one
shared setup/run root and the dependent summary job emits
`em_kclass_multiseed_summary.json` and
`em_kclass_multiseed_summary.md`. The aggregate requires exactly those three
seeds, verifies that the scientific axes did not drift, retains per-seed
failure and class-collapse outcomes, and reports worst quality and median/max
resource summaries. Every completed row is cross-checked against and bound to
a hashed runtime `case_config.json`, including its source PDB directory, seed,
symmetry, name/index, and Slurm job. It is a suite index, not a replacement for
the individual schema-v1 trajectory records required for a formal benchmark
claim.

Cases 25--27 deliberately share every simulator/refinement seed and scientific
parameter. Only RECOVAR's image-batch or rotation-block boundary changes; the
launcher therefore generates each seed's dataset exactly once in case 25,
seals every generated file with SHA-256, makes cases 26 and 27 depend on that
producer job, and verifies the same manifest in all three jobs. Case 25 also
runs and seals one RELION initialization/model, perturbation oracle, and exact
dynamic dispatch schedule; cases 26 and 27 reuse that oracle instead of
rerunning RELION. Independently regenerating either the particles or the
RELION oracle is not an exact-input test: poses and CTFs can match while
volume, GT, particle, model, or schedule bytes differ. A consumer selected
without its producer, multiple producers, a changed generator axis, a reused
scratch root, or a hash mismatch fails before refinement.

This gate is complete for cases 25--27 at source commit
`91e8a30f4ebc9a88f834b1b9220dcfc3b34c31b7`: three seeds each for image batch
50 / rotation block 8192, image batch 17 / rotation block 8192, and image batch
50 / rotation block 257. All six exact-input comparisons retained identical
controller decisions and final particle assignments; worst numbered-map
FSC-AUC was 0.9999999733 and worst per-class GT FSC-AUC delta was -8.90e-7.
See `k4_exact_input_invariance_20260901.md` for exact jobs, HBM/wall time,
artifact hashes, the fixture-root accounting caveat, and the rerun command.

## Tier 3: grid and particle scaling

Run eight iterations after Tier 2 passes.

| Scale | Cases | Seeds | Purpose |
| --- | --- | --- | --- |
| 5k/64 | K=4 baseline and radial/outlier | 41001–41003 | compile/schedule smoke and tiny-memory behavior |
| 10k/128 | repeat baseline at batches 25/50/100 | 41001–41003 | batch invariance |
| 50k/128 | baseline and imbalanced | 41001–41002 | particle scaling without grid confounder |
| 10k/256 | baseline and radial | 41001–41002 | grid scaling without particle confounder |
| 50k/256 | baseline and radial/nonuniform | 41001–41002 | medium production gate |

Every row records full per-stage timing, wall time, sampled peak HBM, Slurm
MaxRSS, controller schedules, and class populations. Performance comparisons
must use sequential same-model GPUs or an explicitly interleaved matched pair.

## Tier 4: symmetry

Shared symmetry code first gets operator-level tests for RELION labels and
aliases across Cn, Dn, T, O, I1, I2, I3, and I4: operator count, orthogonality,
determinant +1, closure, identity ordering, stable hash, and a matrix-by-matrix
RELION oracle comparison. Production accepts bare `I` as RELION's alias for
I2, canonicalizes it to `I2`, and unit tests pin that exact alias. Explicit
`I1`, `I2`, `I3`, and `I4` remain distinct conventions; dataset-specific
launchers should record an explicit convention instead of relying on the
alias.

The runnable cases 31--34 are K=4 5k/128 five-iteration paired trajectories
for C4, D4, O, and I1. The generator symmetrizes every class and outlier GT
volume with the ordered RELION right-operator set, records the canonical
symmetry label, operator count and operator SHA-256, and passes that same label
to RELION and RECOVAR. C1 remains a bitwise-preserving no-op. Their completed
three-seed evidence is sealed in the C4, D4, O, and I1 campaign records under
`docs/benchmarks/em/campaigns/`. Finally, the EMPIAR-10202 set-6 harness stays
explicitly `I1` (never bare `I`) and its K=1 high-resolution matched refinement
remains the production symmetry gate for the shared K=1/K-class machinery; it
is not replaced by an artificial K=4 split of a homogeneous capsid.

## Tier 5: full synthetic release gates

Only run after Tiers 0–4 are green.

1. Ribosembly K=4, 100k/256, white noise 1, seed 1778628798, 15 iterations,
   matched to the existing accepted oracle.
2. An independently generated, exactly sealed Ribosembly K=4 100k/256
   fixture, seed 41002, 15 iterations. This closes the current generator-
   provenance and single-fixture gaps.
3. One 100k/256 radial/nonuniform/imbalanced K=4 case, seed 41003, 15
   iterations.

Each runs RELION then RECOVAR on the same physical GPU, captures both engines'
HBM monitors, preserves both half-map pairs, creates a common mask, and emits
one registry record. A100 and H100 may both be run for portability, but their
cross-model ratio is never a formal speedup.

## Tier 6: real-data K>1

Use at least two heterogeneous datasets rather than declaring K>1 from the
single synthetic fixture. Required initial targets are EMPIAR-10076 and
EMPIAR-10345, using the already documented native stacks/metadata;
EMPIAR-10073 is an additional calibration target. For each dataset:

1. Freeze particle order, half split, poses, CTFs, solvent mask, symmetry, and
   four shared initial maps. Run a 10k/128 downsampled pilot for seeds
   42001–42003.
2. If pilots are stable, run the native-grid full-particle matched refinement
   for seeds 42001 and 42002.
3. Match classes by Hungarian common-mask FSC, never by label. Report the full
   pairwise matrix, populations, Pmax, pose/translation agreement, and
   seed-to-seed consensus.
4. Report both engines' masked and unmasked half-map FSC/FSC-AUC and 0.143
   resolution from the same mask, plus registered cross-engine half/full-map
   FSC. Without GT, half-map quality and cross-seed stability are the primary
   scientific gates.
5. If independent seeds split or merge classes differently, report the
   instability; do not select the favorable seed post hoc.

The native full run is admitted to the registry only with immutable inputs,
commands/environment, source tree, Slurm allocation, logs, wall/HBM/RSS, mask,
shellwise curves, and a predeclared pass policy.

The bounded 10k InitialModel precursor is now runnable for EMPIAR-10076 and
EMPIAR-10345 through `scripts/launch_em_real_kclass_initialmodel_slurm.py`; see
`real_kclass_initialmodel_pairs.md`. It checks iterations 1--8, class
permutation, assignment agreement, collapse, source/input provenance, and
matched one-GPU performance. Because InitialModel does not emit independently
refined half maps, this precursor does not satisfy the native full-run
admission rule above.

The independent-half EMPIAR-10076 refinement gate is now runnable through
`scripts/launch_em_real_kclass_halfmaps_slurm.py`; see
`real_kclass_halfmap_refinement.md`. RELION does not permit K>1 and
`--split_random_halves` in one process, so the harness runs two independent
K=4 processes per engine, one on each immutable random subset. The launcher is
dry-run by default. The three-seed 128-grid sign-fixed pilot is now complete
and checked in as a rejected diagnostic: classes remain occupied, but all
seeds miss the prospective assignment gate and one class/seed misses the
half-map quality gate substantially. Native execution therefore remains
blocked by policy rather than by missing infrastructure.

That launcher is deliberately narrower than the complete Tier-6 checklist:
it reports maps, half-map/cross-engine FSC, populations, hard-assignment
agreement, significant support, and resources for one invocation. The
validated three-seed hard-assignment aggregate and final-map cross-seed
analysis are emitted separately. Particle-state audits also retain Pmax,
pose, translation, and significant-support trajectories. The completed pilot
therefore establishes the independent-half infrastructure and a shared
stochastic-instability boundary, but it cannot be described as complete Tier-6
evidence while the frozen per-seed quality and assignment gates remain red.
The final-map analysis has a separate cross-CPU reproduction contract: inputs,
discrete class mappings, and decisions must match exactly; every reported
FSC-AUC may move by at most 0.005 under the continuous proper-rigid fit; and the
minimum same-seed-versus-cross-seed per-class separation must remain at least
0.05. Job `13354170` passes this contract. Job `13353671` is retained as a
completed-analysis harness failure because its obsolete final `cmp` required
the continuous optimizer parameters to be byte-identical.

## Runnable now versus planned-only

The current launcher can run cases 1--36, including the shared-input
batch/rotation-block invariance rows, no-CTF boundary discriminators, and
C4/D4/O/explicit-I1 trajectories. It can expand any selected subset over
exactly three frozen seeds and emit a validated multi-seed suite summary. The
checked-in registry, rather than presence in the launcher, determines which
trajectories completed and what they established: the symmetry campaigns are
accepted evidence, while cases 30, 35, and 36 are excluded RELION-collapse
diagnostics. The following matrix pieces still need bounded launch/test
implementation:

- the fixed-state 32--256-particle image-batch/rotation-block discriminator
  grid beyond the admitted shared-200 coarse boundary, including matched fine
  score/posterior captures;
- the added 5k/64, 50k/128, and 10k/256 scaling rows;
- fail-closed permutation/duplicate-map/controller-corruption fixtures beyond
  the existing evaluator unit tests;
- the independent exactly generated 100k/256 release fixture; and
- an independent-half K=4 launcher for EMPIAR-10345 (and optional
  EMPIAR-10073); EMPIAR-10076 has a completed but rejected three-seed pilot,
  and the InitialModel-only diagnostic cannot replace either real-data gate.

These are deliberately documented as pending execution infrastructure, not as
completed coverage.

## Merge cadence

- Per commit: Tier 0 plus focused K-class/unit guards.
- Before an EM PR update that touches K-class scoring, support, M-step,
  reconstruction, symmetry, batching, or memory: Tiers 1 and 2.
- Before merge: Tiers 3 and 4, the existing EM-targeted long suite, one full
  Tier-5 same-GPU trajectory, and at least the real-data pilots.
- Before claiming real-data K>1 or production performance: native Tier 6 and a
  same-hardware performance record.
