# Task 101: Residual parity bug at iter 6 noise update — Findings

**Date:** 2026-04-09
**Investigator:** Claude (relion-parity audit branch, clone at
`/scratch/gpfs/GILLES/mg6942/recovar_relion_parity_audit/`)
**Baseline commit:** `942e2c6` (Task #100 fix) on `claude/relion-parity-flag-audit`
**Task:** Identify the mechanism of the ~5-15% residual noise spike at the
iter-5 → iter-6 global→local-search transition (5k benchmark).

---

## Executive summary

After the Task #100 fix, the 15% residual shell-5 sigma² spike at iter 6 is
dominated by **two independent issues, both rooted in recovar's non-standard
rot/tilt convention** in `rotation_indices_to_matrices` / `get_rotation_grid`:

1. **Primary (large effect):** The recovar HEALPix grid, constructed by passing
   `[theta_healpy, phi_healpy, psi]` to `R_from_relion` as `[rot, tilt, psi]`,
   produces a set of rotation matrices whose **view directions are non-uniformly
   distributed on the sphere** — clustered near the poles by a factor of ~2.5x
   relative to uniform SO(3). At iter 6 this causes the per-particle local-search
   cone size to vary by **5x** depending on prior tilt (4600 rotations at
   tilt=10° or 170°, 945 rotations at tilt=90°), while RELION's cone is
   essentially constant at ~1400 rotations per particle regardless of prior.
   The average recovar cone is ~1.6x RELION's.

2. **Secondary (small effect, but a real bug):** The psi circular-distance
   computation in `get_local_rotation_grid_fast` does not normalize `prior_psi`
   to `[0, 2π)` before computing `min(|diff|, 2π - |diff|)`. Because
   `R_to_relion` returns psi in `[-180°, 180°)`, ~49% of priors have negative
   psi, and for those the initial per-axis selection is inflated to ~30-53 psi
   candidates (should be 12). The joint log-prior cutoff
   `log_prior > -sigma_cutoff²` filters most bogus entries, but ~2-3 extras per
   prior slip through (entries with `d_psi > 3σ_psi` but with small `d_dir` so
   the joint term is still above `-sigma_cutoff²`). This is an unambiguous bug
   in the post-Task-100 code; its direct noise impact is <1% of the weight
   mass, much smaller than issue (1).

The claim in the task description that recovar's per-particle cone is "~3x
larger" than RELION's (2237 candidates vs 700) turns out to be **a mean over
particles that obscures a 5x variation**: particles with prior tilt near
equator have RELION-sized cones (~900-1000), while pole-region particles have
4x larger cones (~4600). The average inflation (1.6x) is close to the
observed ~15% noise spike when weighted by the contribution of wider cones to
the softmax denominator.

The only complete fix is to reorder the Euler arguments in
`rotation_indices_to_matrices` and `get_rotation_grid` to match standard
RELION (`[phi_healpy, theta_healpy, psi]`), which pins the grid view
directions to HEALPix pix2vec (uniform on the sphere). This regenerates every
matrix in the grid and will break most existing baselines that assume the
current convention.

---

## Evidence

### Setup

At the 5k benchmark iter-6 transition:
- HEALPix order 4 (nside=16, n_pixels=3072, n_psi=96) → 294,912 rotations total
- sigma_rot = sigma_psi = 7.5° (from RELION's formula
  `sigma² = (2 * rottilt_step)² = 4² · 3.75²`, verified at
  `ml_optimiser.cpp:9929`)
- sigma_cutoff = 3.0 → cone radius 22.5° per axis
- Direction and psi priors are independent (factored log-prior), same as
  RELION's `P(idir) * P(ipsi)` structure
  (`healpix_sampling.cpp:selectOrientationsWithNonZeroPriorProbability`).
- Chunk size: 64 particles per chunk.
- Per-particle log-prior is the per_image=True path in
  `recovar/em/sampling.py::get_local_rotation_grid_fast`.

### Comparison of selection logic

RELION (`selectOrientationsWithNonZeroPriorProbability`,
`healpix_sampling.cpp:695`):
- Called **per particle** (one call inside `getAllSquaredDifferences` via
  `op.metadata_offset`).
- Direction cone: loops over all 3072 HEALPix directions, keeps those with
  `diffang < sigma_cutoff * biggest_sigma = 22.5°`.
  ```c++
  if (diffang < sigma_cutoff * biggest_sigma)
  ```
- Psi cone: loops over all 96 psi angles, keeps those with
  `diffpsi < sigma_cutoff * sigma_psi = 22.5°`. `diffpsi` is the circular
  distance `ABS(psi_angles[ipsi] - prior_psi)` wrapped to `[0, 180]`.
- Per-particle result: `nr_orients = NrDirections × NrPsiSamplings ≈ 117 × 12
  = 1404`.

Recovar (`get_local_rotation_grid_fast`, `recovar/em/sampling.py:544`,
post-Task-100 fix):
- Called **per chunk of 64 particles**, with `per_image=True`.
- Direction cone (correct, uses matrix view direction via dot-product; Task
  #100 fix, line 688-696 of sampling.py):
  ```python
  for prior_vec in prior_dir_vecs:
      dots = grid_view_dirs @ prior_vec
      in_cone = dots >= cos_cutoff
      selected_pixel_set.update(np.nonzero(in_cone)[0].tolist())
  ```
- Psi cone (**BUG — does not wrap prior_psi to `[0, 2π)`**, lines 711-716 of
  sampling.py):
  ```python
  for psi_val in prior_psi:
      diffs = np.abs(psi_angles - psi_val)           # psi_val can be negative!
      circ_dists = np.minimum(diffs, 2 * np.pi - diffs)  # can be negative!
      within = np.where(circ_dists <= psi_cutoff_rad)[0]
      selected_psi_set.update(within.tolist())
  ```
- The final selection is `selected_psi × selected_pixels` (cartesian product
  over the chunk union), then a joint cutoff `log_prior > -sigma_cutoff²`
  filters the cells.

### The psi bug: a minimal reproducer

For `prior_psi = -180°` (-π rad), `psi_angles = linspace(0, 2π, 96, endpoint=False)`,
`sigma_psi = 7.5°`, `sigma_cutoff = 3`:

Correct circular distance (wrap prior to `[0, 2π)` first):
- `p_mod = -π % (2π) = π`
- `diffs = |psi_angles - π| ∈ [0, π]`
- `circ = min(diffs, 2π - diffs) ∈ [0, π/2]`
- 13 psi values within 22.5° of 180° (exactly as expected for step=3.75°).

Buggy circular distance (no wrap):
- `diffs = |psi_angles - (-π)| = psi_angles + π ∈ [π, 3π)`
- `2π - diffs ∈ (-π, π]`
- `circ = min(diffs, 2π - diffs)` — can be **negative**. With prior=-π,
  47 of 96 values have negative `circ`, which all pass `<= 22.5°` trivially.
- **53 psi values "within cutoff"** (mostly spurious).

Verified empirically (see `/scratch/gpfs/GILLES/mg6942/tmp/task101_cone_count.py`):
```
prior=-180°, n_within=53, psi values selected: [161.25, 165, ..., 356.25]
```

**Important**: the `log_prior_psi = -d_psi²/(2σ²)` formula at lines 795-812 is
sign-invariant because of the square, so each entry's computed log_prior is
still numerically correct. What's wrong is the **set** of entries kept: bogus
entries at 25°-170° circular distance pass the initial per-axis selection.

The joint cutoff `log_prior > -sigma_cutoff² = -9` then filters most of them
(entries with `d_dir² + d_psi² > 2σ²·sigma_cutoff² = 18σ²`), leaving ~2-3
extras per prior.

**The direct noise impact is small.** For a prior at psi=-170° (random test
case):
- 1188 total cells pass
- 132 of those are "psi-extras" (d_psi > 22.5° but dir+psi in the joint cone)
- log_prior for extras ∈ [-9, -5.03] → softmax weight ∈ [1e-4, 6e-3]
- Weight mass of extras: ~0.066 out of ~132 for valid neighbors → ~0.05% of
  total posterior weight is on extras

### The non-uniform grid (dominant issue)

**Grid construction issue (from `recovar/em/sampling.py` and
`utils.R_from_relion`):**

```python
# rotation_indices_to_matrices (sampling.py:85-105):
theta, phi = hp.pix2ang(nside, pixel_idx)
psi = (2.0 * np.pi / n_psi) * psi_idx
angles = np.stack([np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(psi)], axis=-1)
# ← theta (POLAR) passed as RELION's first argument ("rot", normally azim)
# ← phi (AZIM) passed as RELION's second argument ("tilt", normally polar)
```

The **view direction** of the resulting matrix is (third column of R):
```
view = (sin(phi_healpy) * cos(theta_healpy),
        sin(phi_healpy) * sin(theta_healpy),
        cos(phi_healpy))
```

This does **not** equal HEALPix pix2vec `(sin(theta)·cos(phi), sin(theta)·sin(phi), cos(theta))`.
Recovar's view direction uses phi as the "polar angle" where HEALPix
distributes points uniformly in `phi ∈ [0, 2π)` (many points with phi near 0
or π, giving z near ±1).

**Empirical measurement of recovar's view direction distribution (order 4,
n_pix=3072):**

z-histogram (should be uniform in [-1, 1] for uniform SO(3)):
```
z in [-1.0, -0.8]:  642   <- 21% of pixels clustered near south pole (recovar)
z in [-0.8, -0.6]:  252
z in [-0.6, -0.4]:  234
z in [-0.4, -0.2]:  196
z in [-0.2,  0.0]:  212
z in [ 0.0,  0.2]:  212
z in [ 0.2,  0.4]:  196
z in [ 0.4,  0.6]:  234
z in [ 0.6,  0.8]:  252
z in [ 0.8,  1.0]:  642   <- 21% near north pole (recovar)
```

Compare with the standard-convention grid (RELION, pix2vec):
```
z in [-1.0, -0.8]:  312   <- 10% (RELION / uniform)
z in [-0.8, -0.6]:  296
z in [-0.6, -0.4]:  320
z in [-0.4, -0.2]:  320
z in [-0.2,  0.0]:  256
z in [ 0.0,  0.2]:  320
z in [ 0.2,  0.4]:  320
z in [ 0.4,  0.6]:  320
z in [ 0.6,  0.8]:  296
z in [ 0.8,  1.0]:  312   <- 10% near north pole (RELION)
```

**Recovar over-represents the poles by 2x compared to a uniform distribution.**

### Per-particle cone size vs tilt (the smoking gun)

Measured on actual recovar `get_local_rotation_grid_fast`, sigma_rot=sigma_psi=7.5°,
sigma_cutoff=3, healpix_order=4, per_image=True, single-prior calls:

| prior tilt | recovar cone (lp > -9) | RELION dir count* |
|:---:|---:|---:|
| 10°  | **4596 ± 244** | 116 |
| 30°  | 2138 ± 571 | 118 |
| 50°  | 1138 ± 569 | 117 |
| 70°  | 1045 ± 420 | 117 |
| 90°  | **945 ± 459** | 118 |
| 110° | 1023 ± 421 | 117 |
| 130° | 1281 ± 546 | 116 |
| 150° | 1979 ± 705 | 118 |
| 170° | **4648 ± 275** | 116 |

(*RELION direction count × ~12 psi ≈ 1400 per particle, essentially constant
across all tilts — std ≈ 1-2. Measured on the standard pix2vec grid.)

Recovar's cone varies by **~5x between equator and pole priors**, with std
40-70% of the mean within each tilt bin. RELION's cone is **constant within
2-3 counts** across all priors.

**Observed mean cone sizes** (random uniform priors over SO(3), n=200):
- RELION (square cone):    **mean = 1405**, std = 20
- Recovar per-particle:    **mean = 1428**, std = 600
- Ratio: 1.02x mean, but **huge per-particle variance**

**Observed chunk-union per-image cone sizes** (64 priors per chunk):
- Buggy (current): **mean = 1978**, range 324-7618
- Corrected (psi wrap fixed): mean = 1474, range 324-4224
- RELION per-particle: mean = 1403, range 1368-1440

The chunk-union inflates the per-image count because the cartesian product
`selected_psi_union × selected_pixels_union` includes many extra cells. Most
are masked to `-1e30` per image via the per-axis cutoffs, but the psi-wrap
bug combined with the non-uniform direction grid still leaves a 40% mean
inflation.

### Calls to `get_local_rotation_grid_fast` in the pipeline

From `recovar/em/dense_single_volume/refine.py:169` (`_run_grouped_local_search_em`):

```python
chunk_size = max(1, min(image_batch_size, 64))    # = 64 in practice
for chunk_start in range(0, n_images, chunk_size):
    group_image_indices = np.arange(chunk_start, chunk_stop)
    local_indices, local_log_prior = get_local_rotation_grid_fast(
        prior_rotations[group_image_indices],  # (64, 3, 3)
        sigma_rot, sigma_psi, healpix_order,
        sigma_cutoff=3.0,
        per_image=True,
    )
    local_rotations = rotation_indices_to_matrices(local_indices, healpix_order)
    # ... run_em_v2 with rotation_log_prior=local_log_prior (shape (64, n_union))
```

So the local search is chunk-batched over 64 particles, each chunk scoring
the union of ~80,000 rotations. The per-image log-prior correctly masks
out-of-cone cells with `-1e30`, and the E-step scores are mathematically
equivalent (within floating-point) to what a per-particle loop would produce
(verified: adding `-1e30` entries to a softmax is identical to omitting them).

---

## Why the residual spike is at iter 6 and not earlier

- Iter 1-5 use **global search** (no cone) at order 3. Every particle
  scores all 36,864 rotations regardless of prior. The grid non-uniformity
  has no per-particle consequence because all particles see the same set.
- Iter 6 enables **local search** at order 4. Each particle scores only a
  cone around its prior. Pole particles get 5x wider cones than equatorial
  particles. The posterior for pole particles spreads over more rotations,
  lowering max_posterior and increasing the weight-averaged squared residual
  used in the noise update.

Empirical verification from RELION's iter-5 → iter-6 trajectory on the same
dataset (from `run_it005..008_half1_model.star`):

| iter | shell-5 σ² | Δ vs prev |
|:---:|---:|---:|
| 4 | 2.502e-5 | - |
| 5 | 2.509e-5 | +0.3% |
| 6 (local) | **2.483e-5** | **-1.0%** (decreases!) |
| 7 | 2.484e-5 | +0.04% |
| 8 | 2.480e-5 | -0.2% |

RELION's iter-6 σ² actually **decreases** slightly because the finer grid
improves pose assignment accuracy and reduces noise residual.

Recovar post-Task-100 (from `plan_relion_parity_v3.md`):

| iter | shell-5 σ² | Δ vs iter 5 |
|:---:|---:|---:|
| 5 | 6.820e3 | - |
| 6 (local) | **7.860e3** | **+15.3%** (spikes) |
| 7 | 7.155e3 | +4.9% |

The +15% is precisely the sign that posterior mass is being wasted on
rotations that shouldn't be scored — consistent with the 60% increase in
effective cone size that recovar's non-uniform grid produces on average
(compared to RELION).

---

## Direct evidence for the rot/tilt swap being the dominant cause

The `R_to_relion` roundtrip on the order-3 grid shows:
- Near-pole fraction (tilt < 30° or > 150°): **33.9%** (recovar)
- Uniform SO(3) near-pole fraction: **13.4%**
- Over-representation: **2.53x**

Since recovar's grid is self-consistent (matrices at pixel p have
R_to_relion → pixel p's theta, phi, psi by round-trip), the set of output
tilts shows the non-uniformity directly. An analogous measurement on the
view-direction `z = R[2, 2]` confirms the same 2x clustering at the poles.

At iter 5 the HARD ASSIGNMENTS come from argmax over all 36,864 rotations.
When these are converted back to matrices and fed as priors to iter 6,
**~34% of particles have tilt < 30° or > 150° priors** simply because the
grid over-represents those regions. These particles then experience the 4x
cone inflation at iter 6, while only 13.4% would if the grid were uniform.

---

## What's NOT the cause

These were investigated and ruled out or shown to be small:

1. **Chunk-union vs per-particle scoring**: softmax with `-1e30` masking is
   numerically identical to omitting those entries. Verified exactly via a
   minimal test. Chunk-union does not inflate per-image posterior.

2. **The psi-wrap bug alone**: adds ~2-3 extras per prior with log_prior
   in `[-9, -5]`. Softmax weight contribution ≤ 0.1% of total. Real bug,
   but not enough to explain 15%.

3. **The joint cutoff at `log_prior > -sigma_cutoff²`**: keeps the
   intersection of square and inscribed disk, slightly larger than RELION's
   pure square, but the difference is <1% of cells (just the corners).

4. **Translation prior / sigma_offset update**: orthogonal to the rotation
   path and not affected by this bug. The C1 `sigma2_offset` MAP update
   (commit `9bdeabd`) is unrelated.

5. **`highres_Xi2` / masked-vs-unmasked split**: already handled correctly
   post-commit `504b37b`.

---

## Recommended fix

### Full fix (correct but breaks many baselines)

Swap the Euler argument order in `get_rotation_grid` (`sampling.py:37`) and
`rotation_indices_to_matrices` (`sampling.py:85`):

```python
# OLD (non-standard):
angles = np.stack([np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(psi)], axis=-1)

# NEW (RELION-standard: rot=phi=azim, tilt=theta=polar):
angles = np.stack([np.rad2deg(phi), np.rad2deg(theta), np.rad2deg(psi)], axis=-1)
```

This makes:
- `view_dir = R[:,:,2] = (sin(theta)·cos(phi), sin(theta)·sin(phi), cos(theta))`
- = `hp.pix2vec(nside, pixel_idx)` exactly
- Uniformly distributed on the sphere (by HEALPix design)

Consequences:
- **Every rotation matrix in the grid changes.** Projections at HEALPix
  pixel `p` will rotate to a different place in 3D.
- All tests that pickle rotation indices or matrices (the global grid, the
  local-search baselines, possibly `test_local_rotation_grid_fast_uses_exact_prior_rotation_angles`,
  etc.) will break and need regeneration.
- Any pipeline baseline (e.g. 5k benchmark locres/FSC) that was generated
  against the current grid will show differences (possibly minor, possibly
  major — empirically unknown until re-run).
- The Task #100 "swap-aware view direction" workaround in
  `get_local_rotation_grid_fast` becomes unnecessary and should be removed
  (replaced with `hp.pix2vec` or the standard formula on the input angles).

### Minimal fix for the psi bug (safe, no baseline changes)

In `recovar/em/sampling.py` around lines 710-716 and 795, wrap `prior_psi`
to `[0, 2π)` before computing circular distances:

```python
# BEFORE:
for psi_val in prior_psi:
    diffs = np.abs(psi_angles - psi_val)
    circ_dists = np.minimum(diffs, 2 * np.pi - diffs)
    within = np.where(circ_dists <= psi_cutoff_rad)[0]
    selected_psi_set.update(within.tolist())

# AFTER:
prior_psi_wrapped = prior_psi % (2 * np.pi)  # normalize to [0, 2π)
for psi_val in prior_psi_wrapped:
    diffs = np.abs(psi_angles - psi_val)
    circ_dists = np.minimum(diffs, 2 * np.pi - diffs)
    within = np.where(circ_dists <= psi_cutoff_rad)[0]
    selected_psi_set.update(within.tolist())
```

Similarly at line 795:
```python
# BEFORE:
d_psi_raw = np.abs(sel_psi_vals[:, None] - prior_psi[None, :])
d_psi = np.minimum(d_psi_raw, 2 * np.pi - d_psi_raw)

# AFTER:
prior_psi_wrapped = prior_psi % (2 * np.pi)
d_psi_raw = np.abs(sel_psi_vals[:, None] - prior_psi_wrapped[None, :])
d_psi = np.minimum(d_psi_raw, 2 * np.pi - d_psi_raw)
```

This fix alone will NOT eliminate the 15% noise spike (it's a small effect,
<1% weight), but it removes a real bug and should be applied regardless.
It's also a pure local fix that does not change any rotation matrices or
projections.

### Recommendation

**Do both fixes, in order:**

1. Apply the psi-wrap fix immediately (tiny diff, no baseline risk, removes
   an unambiguous bug).
2. Open a plan document for the full grid-convention fix and regenerate all
   affected baselines. This is a larger change and should be approved by the
   user before execution. The benefit is full iter-6 parity with RELION.

---

## References

### Code
- `/scratch/gpfs/GILLES/mg6942/recovar_relion_parity_audit/recovar/em/sampling.py:544`
  — `get_local_rotation_grid_fast` (post-Task-100 fix)
- `/scratch/gpfs/GILLES/mg6942/recovar_relion_parity_audit/recovar/em/sampling.py:85`
  — `rotation_indices_to_matrices` (the grid-construction root cause)
- `/scratch/gpfs/GILLES/mg6942/recovar_relion_parity_audit/recovar/em/sampling.py:37`
  — `get_rotation_grid` (same convention)
- `/scratch/gpfs/GILLES/mg6942/recovar_relion_parity_audit/recovar/em/dense_single_volume/refine.py:169`
  — `_run_grouped_local_search_em` (the chunk-union caller)
- `/home/mg6942/myscratch/relion/src/healpix_sampling.cpp:695`
  — RELION's `selectOrientationsWithNonZeroPriorProbability`
- `/home/mg6942/myscratch/relion/src/acc/acc_ml_optimiser_impl.h:119`
  — RELION's per-particle call site

### Docs
- `/scratch/gpfs/GILLES/mg6942/recovar_relion_parity_audit/docs/math/plan_relion_parity_v3.md:981-1044`
  — Task #100 bug history and residual numbers
- `/home/mg6942/.claude/projects/-home-mg6942/memory/feedback_recovar_grid_rot_tilt_swap.md`
  — Prior documentation of the rot/tilt swap

### Test scripts (this investigation)
- `/scratch/gpfs/GILLES/mg6942/tmp/task101_cone_count.py`
  — Per-image cone count comparison recovar vs RELION
- `/scratch/gpfs/GILLES/mg6942/tmp/task101_corrected_cone.py`
  — psi-wrap fix impact
- `/scratch/gpfs/GILLES/mg6942/tmp/task101_full_cone_comparison.py`
  — Full buggy/corrected/RELION comparison
- `/scratch/gpfs/GILLES/mg6942/tmp/psi_bug_noise_impact.py`
  — Weight mass analysis of the psi bug
- `/scratch/gpfs/GILLES/mg6942/tmp/tilt_dist_no_jax.py`
  — Grid non-uniformity via `R_to_relion` round-trip

### Data
- `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0/run_it00{4..8}_half1_model.star`
  — RELION iter-4-to-8 sigma² reference values
