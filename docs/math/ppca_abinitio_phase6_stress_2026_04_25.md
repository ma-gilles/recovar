# Phase 6 — stress sweep findings (2026-04-25)

**Sweep root:** `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408/_agent_scratch/ppca_phase6_stress_20260425_150721`

**Submitted:** 124 cells × 2 seeds. **Successfully completed: 110.** Two infrastructure failures:
- 12 Tomotwin-100 cells failed at MRC load: `mrcfile` rejects the malformed machine stamp on CryoBench's Tomotwin GT volumes (`Unrecognised machine stamp: 0x00 0x00 0x00 0x00`). Axis D is unmeasured.
- 2 IgG-RL n=64 cells failed at the discrete-cluster eval: sklearn KMeans requires `n_samples ≥ n_clusters` and IgG-RL has 100 GT states. EM completed; eval is undefined for n < n_clusters.

## Axis A — SNR sweep (Ribosembly q=8 most informative)

| σ | hun | PE |
|---|---|---|
| 0.001 | 0.621 ± 0.089 | 3.680 ± 0.020 |
| 0.01 | 0.811 ± 0.016 | 3.600 ± 0.026 |
| 0.03 | 0.917 ± 0.012 | 3.347 ± 0.069 |
| 0.10 | **0.942 ± 0.053** ← peak | 3.092 ± 0.029 |
| 0.30 | 0.725 ± 0.009 | 3.509 ± 0.009 |
| 1.00 | **0.261 ± 0.000** ← collapse | 3.840 ± 0.015 |

**Two surprising findings:**

1. **σ=1.0 is the breaking point.** Hungarian drops from 0.725 (σ=0.3) to 0.261 (σ=1.0). The SNR ceiling for v0 with default settings is around σ=0.3.

2. **σ=0.001 is paradoxically WORSE than σ=0.1 (0.621 vs 0.942).** Very-low-noise regime has narrow EM basins; the algorithm gets stuck in local minima it can't escape. The likelihood landscape gets sharper as noise drops, but the warmstart isn't "in the right basin" yet — without smoothing (annealing) it can't move there.

Ribosembly q=4 is more graceful: hun stays ~0.80 from σ=0.001 through σ=0.3, then drops to 0.368 at σ=1.0. q=4 is so misspecified (16 states → 4 dims) that the basin landscape is flat enough not to trap.

IgG-1D q=2 / IgG-RL q=2: hun barely moves with σ — the q=2 ceilings on these continuous 100-state manifolds are already at chance (~0.18-0.23). Noise doesn't change a metric that's already at the floor.

## Axis B — cold μ init kills the algorithm

All cells have `μ=zero`, varying U init. Ribosembly q=4 σ=0.01:

| U init | hun | PE |
|---|---|---|
| svd | 0.293 ± 0.007 | 2.786 |
| random | 0.269 ± 0.001 | 2.816 |
| zero | 0.075 ± 0.000 | 2.828 |

Compare to perturbed-μ + svd-U: hun=0.794 (Phase 1).

**Cold-μ costs ~0.50 hun.** The U init barely matters: svd-U and random-U give similar (mediocre) results; only fully-zero-U fully collapses to chance (0.075 ≈ random for 16 states).

H2 confirmed: the algorithm is **mu-init-bound**, not U-init-bound. The weighted SVD warmstart cannot recover from a zero μ.

## Axis C — small-N

Ribosembly q=4 (discrete) is **insensitive to n_img**:

| n_img | hun | PE |
|---|---|---|
| 64 | 0.625 ± 0.031 | 2.614 |
| 128 | 0.637 ± 0.020 | 2.600 |
| 256 | 0.680 ± 0.031 | 2.582 |
| 512 | 0.649 ± 0.030 | 2.235 |

(All within noise of each other; the Phase 1 baseline at n=1024 is hun=0.755.)

IgG-RL q=2 (continuous): the Hungarian numbers look spuriously high at small N (0.598 at n=128 vs 0.305 at n=512), but PE tells the real story: PE = 1.738 at n=128 vs 1.286 at n=512. **Subspace recovery genuinely worsens with smaller N** on the continuous manifold; the inflated Hungarian at small N is a k-means/Hungarian artifact (chance scoring rises when fewer samples are spread across 100 GT clusters).

H4 partially confirmed: small-N hurts continuous more than discrete on PE; the reverse-looking Hungarian numbers are a metric artifact.

## Axis D — Tomotwin-100 (NOT MEASURED)

All 12 cells failed at MRC load. CryoBench's Tomotwin volumes have non-standard headers that mrcfile rejects in default (strict) mode. Workaround would require `mrcfile.open(..., permissive=True)` in `recovar.utils.helpers.load_mrc`, which is shared infrastructure outside this branch's scope. Recorded as a known limitation; H3 unverified.

## Hypothesis verdicts

| Hyp | Statement | Verdict |
|---|---|---|
| H1 | v0 holds at σ ≤ 0.1, breaks at σ ≥ 0.3 on Ribosembly | **Refined.** Holds through σ=0.3 (hun=0.725 q=8). Breaks at σ=1.0 (hun=0.261). Plus surprising failure at σ=0.001 (basin narrowness). |
| H2 | zero-μ + zero-U collapses | **CONFIRMED.** zero-μ collapses regardless of U init. Algorithm is mu-bound. |
| H3 | Tomotwin-100 q=4 near chance, q=16 better | **NOT TESTED.** Loader infrastructure issue. |
| H4 | small-N hurts IgG-RL more than Ribosembly | **CONFIRMED on PE.** Hungarian artifact at small N misleads. |
| H5 | random-U survives at low σ, collapses at higher σ | **NOT CONFIRMED.** Cold μ already collapses random-U at all σ tested; no σ-dependent break. |

## What this means for v0

1. **Recommend documenting σ=0.3 as the practical SNR ceiling** for v0 with default settings. Beyond that, annealing or stronger regularization would be needed.
2. **Cold-μ recovery is an algorithmic gap.** A perturbed initial μ (output of a homogeneous reconstruction) is a hard prerequisite. Without it, the algorithm cannot bootstrap.
3. **Annealing is the natural rescue mechanism** for both the high-σ collapse and (potentially) the low-σ basin narrowness. The targeted improvement experiment (Phase 6.5, in flight) tests this.

## Phase 6.5 — Targeted improvement A/B (22/24 cells, completed 2026-04-25)

`_agent_scratch/ppca_phase6_improvement_20260425_175755` — factor-only-log1000 anneal vs none on each Phase 6 failure mode.

| Group | Failure mode | none hun | anneal hun | Δhun | none PE | anneal PE | Verdict |
|---|---|---|---|---|---|---|---|
| A1 | cold-μ Ribo q=4 σ=0.01 | 0.293 | 0.276 | -0.017 | 2.786 | 2.785 | no effect |
| A2 | cold-μ Ribo q=8 σ=0.01 | 0.217 | 0.228 | +0.011 | 3.941 | 3.929 | no effect |
| A3 | cold-μ IgG-RL q=2 σ=0.01 | 0.171 | 0.175 | +0.003 | 1.986 | 1.986 | no effect |
| B1 | high-σ Ribo q=4 σ=1.0 | 0.368 | 0.311 | -0.058 | 2.647 | 2.548 | **HURTS** |
| B2 | high-σ Ribo q=8 σ=1.0 | 0.261 | 0.225 | -0.037 | 3.840 | 3.740 | **HURTS** |
| C1 | very-low-σ Ribo q=8 σ=0.001 | 0.621 | 0.812 | **+0.191** | 3.680 | 3.387 | **RESCUES** |

(2 cells in B2/C1 seed1 still in flight at writing; the trends above are stable from 17/24 → 22/24.)

### Conclusion: factor-only-log1000 annealing is σ-conditional, not universal

- **Rescues 1 / 6 failure modes** (very-low-σ basin trap on q=8)
- **Hurts 2 / 6 failure modes** (high-σ collapse — annealing inflates already-overwhelming noise)
- **Neutral on the 3 cold-μ failure modes** (algorithm is mu-init-bound; σ²-annealing cannot restart from a zero μ)

This refines the Phase 1 finding: factor-only-anneal lifts Ribo q=8 from hun=0.774 to hun=0.946 *only when μ is warm-started*. With cold μ, the lazy-basin escape mechanism is not triggered.

### Recommendation for v0 defaults — UNCHANGED

Do NOT promote `--anneal-schedule log1000 --anneal-factor-only` to default. The current default (no anneal) is correct in 4/6 measured failure modes; making anneal-on default would actively hurt high-σ regimes.

**Document factor-only-anneal as a σ-conditional opt-in:**

> Use `--anneal-schedule log1000 --anneal-factor-only` when σ < 0.005 OR for Ribosembly-like discrete heterogeneity at q ≥ 8 with warm-started μ. Do NOT enable at σ ≥ 0.3 or with cold μ — annealing actively hurts those regimes.

### Cold-μ rescue is unsolved

The biggest remaining v0 limitation. **Annealing does not rescue cold-μ on any of three datasets** (q=4 discrete, q=8 discrete, q=2 continuous). Possible follow-up branches that might help:
1. **μ-anneal jointly with factor** (currently `--anneal-mu-too`, but Phase 1 showed it diverges on continuous data — needs adaptive schedule)
2. **Multi-restart from cold + lm-based selection** (already infra: `--n-restarts`)
3. **Bootstrap μ from a preceding homogeneous reconstruction** (the natural mode in production; requires the pipeline-bridge follow-up branch)
4. **Mean-anneal with a milder σ× schedule** (log10 instead of log1000, to avoid the IgG-RL FRE-divergence Phase 1 saw)

These are all candidate research directions, not v0 deliverables.

## Final commit log for Phase 6 + 6.5

- Phase 6 stress sweep: `_agent_scratch/ppca_phase6_stress_20260425_150721/` (110 valid cells)
- Phase 6.5 improvement A/B: `_agent_scratch/ppca_phase6_improvement_20260425_175755/` (22+ cells)
- Findings doc: this file
- No code change to v0 defaults; default unchanged at `--anneal-schedule none`.
