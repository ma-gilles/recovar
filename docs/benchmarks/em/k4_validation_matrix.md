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
formal result.

For real data, where GT is unavailable, freeze the following before launching:
the same particles/order, half split, poses/CTFs, initial maps, symmetry,
mask-generation command, and random seed for both engines; per-class masked
and unmasked half-map curves; common-mask 0.143 resolution; registered
cross-engine FSC; class populations; and seed-to-seed stability. RECOVAR's
masked 0.143 resolution must not be worse than RELION by more than one Fourier
shell or 5%, whichever is larger, and masked/unmasked half-map FSC-AUC must not
drop by more than 0.01. These real-data thresholds are prospective and must
not be retrofitted to existing runs.

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

The runnable launcher now contains 29 cases. The original 15 medium/scale
cases remain mandatory, and 14 bounded 3k-particle cases add the K=1 stress
axes, exact batching controls, and independent seeds. This is substantially
broader executable coverage, but a single completed seed is still not
sufficient evidence and only two rows exercise 50k/256.

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

Run the entire panel with seed offsets 0, 10000, and 20000, using a distinct
scratch root for each offset. The launcher already accepts `--seed-offset`, so
these 87 executions are runnable without changing the scientific case
definitions. What is not yet implemented is a checked-in wrapper that submits
and aggregates all three repetitions as one registry-ready suite.

Cases 25--27 deliberately share every simulator/refinement seed and scientific
parameter. Only RECOVAR's image-batch or rotation-block boundary changes; the
generated input hashes must therefore match before their output comparison is
admitted as an invariance result.

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
RELION oracle comparison. Bare `I` is rejected because RELION canonicalizes it
to I2 and silently using the wrong icosahedral convention is unsafe.

Then run K=4 5k/128 five-iteration paired trajectories for C4, D4, O, and I1
with seeds 41001 and 41002. The generator must symmetrize each GT state with
the exact sealed operator set. Finally, the EMPIAR-10202 set-6 I1 K=1
high-resolution matched refinement remains the production symmetry gate for
the shared K=1/K-class machinery; it is not replaced by an artificial K=4
split of a homogeneous capsid.

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
`real_kclass_initialmodel_pairs.md`. It checks iterations 0--8, class
permutation, assignment agreement, collapse, source/input provenance, and
matched one-GPU performance. Because InitialModel does not emit independently
refined half maps, this precursor does not satisfy the native full-run
admission rule above.

## Runnable now versus planned-only

The current launcher can run the exact 29-case panel, including the three
batch/rotation-block invariance rows, and can repeat the full default panel
with `--seed-offset`. The following matrix pieces still need bounded launch/test
implementation before they are executable as a single suite:

- an aggregator that binds the three seed-offset panels into registry records;
- the fixed-state 32--256-particle image-batch/rotation-block discriminator grid;
- the added 5k/64, 50k/128, and 10k/256 scaling rows;
- C4/D4/O/I1 K=4 symmetry trajectories and their generated symmetric GT;
- fail-closed permutation/duplicate-map/controller-corruption fixtures beyond
  the existing evaluator unit tests;
- the independent exactly generated 100k/256 release fixture; and
- matched Class3D/gold-standard-half-map K=4 launchers for EMPIAR-10076 and
  EMPIAR-10345 (and optional EMPIAR-10073); the InitialModel-only diagnostic is
  runnable but cannot replace this gate.

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
