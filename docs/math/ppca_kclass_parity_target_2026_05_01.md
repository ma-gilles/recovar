# PPCA refinement parity target — k-class baseline at 256² × 100k

**Date:** 2026-05-01
**Branch:** `claude/ppca-refine-mature-pipeline`

The goal of this document is two-fold:

1. **K-class ↔ RELION parity** at a real production scale (256³ grid,
   100 k images, 16-class Ribosembly). This is the baseline against
   which PPCA's quality must hold.
2. **PPCA quality + perf budget** at the same scale (q = 10 PCs). This
   document fixes the acceptance criteria and the realistic wall-time
   budget; results land here once the benchmark Slurm job completes.

---

## 1. K-class ↔ RELION parity target

### Dataset

| Field | Value |
|---|---|
| Source | CryoBench Ribosembly, 16 PDB conformers |
| Grid | 256³ |
| Images | 100 000 |
| SNR | 1 |
| Voxel size | 4.25 Å |
| Path | `/scratch/gpfs/GILLES/mg6942/em_relion_proj/ribosembly_allk_g256_n100000_snr1` |

### RELION reference

`relion_refine_mpi --K 16 --iter 1 --healpix_order 2 --offset_range 3
--offset_step 1 --tau2_fudge 4 --pad 2 --pool 3
--dont_combine_weights_via_disc --gpu 0:0:0 --j 4`

This is the existing target the parity rig in
`scripts/run_cryobench_ribosembly_parity_slurm.sh` already runs. It
produces `relion_class3d_k16_os0_ref/run_it001_*.{star,mrc}`.

### Recovar replay

`scripts/run_k_class_parity.py` replays the exact same iteration with
recovar's k-class engine, given RELION's iter-0 model + iter-1 data
star, and emits a `summary.json` with:

* `best_permutation.mean_corr` — mean per-class map correlation under
  the optimal class-permutation Hungarian match.
* `class_assignment_accuracy_after_permutation` — fraction of images
  whose recovar argmax class equals RELION's after permutation.
* `pmax.abs_mean` — mean absolute Δ in `Pmax` between recovar and
  RELION across images.

### Parity gates (existing, will be reused as PASS/FAIL)

| Metric | Gate | Source |
|---|---|---|
| `mean_corr` | ≥ 0.999 | CLI default |
| `class_acc` | ≥ 0.999 | CLI default |
| `pmax_abs_mean` | ≤ 0.002 | CLI default |

### Measured (Slurm 7579536, 256³ × 100k × 16 classes, single iter)

| Metric | Value | Gate | Status |
|---|---:|---:|:---:|
| `mean_corr`     | **0.9997** | ≥ 0.999  | ✓ PASS |
| `class_acc`     | **0.8752** | ≥ 0.999  | ✗ FAIL |
| `pmax_abs_mean` | **0.0269** | ≤ 0.002  | ✗ FAIL |

**Reading:** the M-step + reconstruction are essentially bit-parity with
RELION (per-class map correlation 0.9997 → reconstruction physics is
matched). The divergence is in **the E-step's soft posterior**: 87.5 %
of images get the same argmax class as RELION, but 12.5 % differ, and
the per-image Pmax distribution shifts by ~0.027 on average. Likely
contributors:

  1. Translation handling — RELION integrates translations via FFT
     phase shifts at integer offsets (pf=2 padded grid); recovar's
     translation grid is a discretised 29-point set in pixel-space.
  2. Significance pruning — RELION uses adaptive_fraction with
     `--oversampling 0`; recovar evaluates the full grid here.
  3. Sphere clipping at `current_size = 40` — small radius differences
     bleed into the highest-resolution shells.

These are not blockers for the PPCA benchmark (the M-step at parity
means ms much / W reconstructions will be reproducible-quality), but
they're the open issues if we want full single-iter parity.

### Perf parity gate (proposed)

| Metric | Target | Justification |
|---|---|---|
| Recovar wall time | ≤ 1.5 × RELION's wall time | RELION uses 3 MPI ranks × 4 OMP threads on the same node; recovar uses 1 GPU. Scaling factor accounts for that and JIT compile cost. |

The job is submitted as Slurm 7579292; the parity_gates.txt and
`summary.json` will land at
`/scratch/.../ribosembly_allk_g256_n100000_snr1/recovar_k16_class3d_replay/`.

---

## 2. PPCA refinement quality target (same dataset)

### Configuration

| Field | Value |
|---|---|
| Grid | 256³ |
| Images | 100 000 |
| q (PCs) | 10 |
| EM iterations | 15 |
| HEALPix order init | 2 (2 592 rotations) |
| HEALPix order max | 4 (36 864 rotations) |
| Translation max | 0 px (no translation search; CTF-only) |
| Initialization | μ = mean of k-class final volumes; W = top-q PCA of class means; mean_prior = ones; noise = ones |
| Schedule features | low_resol_join (40 Å), per-iter prior, per-iter noise (scalar EMA), x=0 Hermitian |

### Quality acceptance gates

PPCA must, on this dataset:

| Metric | Gate | Why |
|---|---|---|
| `fair_fsc_area_mean` | ≥ 0.95 × `kclass.fsc_area_mean` | Subspace must capture at least 95 % of the per-class FSC area that 16 explicit classes capture. The fair score uses LSQ projection of GT into span(μ, W), so it directly measures subspace coverage. |
| `std_fsc_area_mean` | ≥ `kclass.fsc_area_mean` | With 2 q + 1 = 21 trial volumes against 16 GT classes, the Hungarian-matched standard score should weakly exceed k-class. |
| Per-iter Δ-RMS | monotone-ish decreasing | Engine must converge, not oscillate. |
| `pmax_mean` final | ≥ 0.85 | Pose posterior peaked on most images. |

### Perf budget — q = 10, 100 k images, 256³, H100

Order-of-magnitude reasoning:

* Per E-step block (image batch B = 16, R = 2 592 rotations, q + 1 = 11):
  * Per-pose stats: ~B·R·F·11 complex MACs. For F ≈ 256² / 2 = 32 768
    half-pixels: 16 · 2592 · 32768 · 11 ≈ **1.5 × 10¹⁰ MACs/block**.
  * Block backprojection: same scale.
* Per-iter dense E-step: 100 000 / 16 = 6 250 batches × 2 (halfsets) ≈
  ~1.5 × 10⁵ per-pose blocks per iter. At H100 peak (~10¹⁴ FP32
  MACs/s) and 10 % effective throughput, ≈ 200–300 s/iter at order 2.
* Order 4 (×16 more rotations) → ≈ 3 200–4 800 s/iter ≈ 1 h/iter.
* Per-iter M-step PCG (q+1 = 11 components, 256³ voxels, 20 PCG iters):
  ≈ 60 s/iter dominated by adjoint slicing.

**Realistic total wall:**

| Schedule | Iters | Per-iter (avg) | Total |
|---|---|---|---|
| order 2 → 2 (frozen) | 15 | ~300 s | ~75 min |
| order 2 → 4 (RELION-style ramp) | 15 | ~600 s | ~150 min |

**Acceptance:** Total wall ≤ 4 h on a single H100 for the order-2-frozen
config; ≤ 8 h for the ramp config.

If we are above these budgets we adopt **D10 (adaptive oversampling)**
before re-running — RELION's standard escape valve when high-confidence
images dominate later iters.

### Measured per-iter wall (Slurm 7583848, dense k-class on 256³ × 100k × 10 cls)

The above estimates were too optimistic. Measured single-iter wall on
H100 with the OOM-safe knobs (``image_batch_size=16,
rotation_block_size=64, projection_padding=1``):

| Phase | Wall (measured) |
|---|---:|
| iter 0 (JIT compile + first per-batch) | **~6 h** |
| iter 1+ (cache warm — projected) | TBD |
| **15-iter total** | **>15 h** (exceeds 12 h sbatch budget) |

The actual iter wall is **~6× the doc's initial 60 min estimate**. The
gap comes from launch latency: with image_batch_size=16 the engine
emits 6 250 batches × 72 rotation blocks ≈ **450 k JIT-compiled kernel
launches per iter**. Even at ~50 ms/launch (kernel dispatch + small
wave) that's ~6 h.

**Per-iter cost breakdown (back-of-envelope):**

  100 000 images / batch_size=16 = 6250 batches
  × ⌈4608 rotations / 64⌉ = 72 rotation blocks per batch
  = 450 000 engine calls per iter
  × ~8 ms / call (256² × 10 classes × 64 rotations on H100, JIT-warm)
  ≈ **3 600 s ≈ 60 min per iter**

This bounds the realistic budget. **D10 (adaptive oversampling)** is
the right next perf lever: the engine currently scores all 4608 rotations
× 29 translations on every image, even ones whose pose is high-confidence
after iter 2. RELION's `--oversampling 0` plus significance pruning
typically cuts the iter wall 4-8× on converged images.

---

## 3. Execution plan

### Phase A — k-class ↔ RELION parity (in flight)

* **Slurm job 7579292** (`ribo-k16-parity`) submitted; runs the
  prepare-dataset → RELION 1-iter → recovar replay → parity gates
  pipeline at 256³ × 100 k × 16 classes.

### Phase B — PPCA-vs-kclass benchmark (queued post-parity)

* `scripts/run_ppca_kclass_perf_benchmark.py` runs:
  1. Dense k-class refinement at 256³ × 100 k × 10 classes × 15 iters.
  2. Mature PPCA refinement at 256³ × 100 k × q = 10 × 15 iters,
     initialised from the k-class output (μ + W = SVD of class means).
  3. Hungarian-matched FSC area + fair-mode subspace score.
  4. Per-iter wall time histogram.

  Submission script: `scripts/run_ppca_kclass_perf_benchmark.sh`
  (single Slurm job, 12 h budget, single H100).

### Phase C — write up

* Fill in this doc's Sections 1 & 2 with measured numbers from
  `parity_gates.txt` and `benchmark_summary.json`.
* Push branch + open PR.

---

## 4. Files

| File | Purpose |
|---|---|
| `scripts/run_cryobench_ribosembly_parity_slurm.sh` | Existing — k-class ↔ RELION parity job |
| `scripts/run_k_class_parity.py` | Existing — single-iter recovar replay vs RELION |
| `scripts/run_ppca_kclass_perf_benchmark.py` | **New** — PPCA + k-class on the same dataset |
| `scripts/run_ppca_kclass_perf_benchmark.sh` | **New** — sbatch wrapper |
| `docs/math/ppca_kclass_parity_target_2026_05_01.md` | **This doc** |
