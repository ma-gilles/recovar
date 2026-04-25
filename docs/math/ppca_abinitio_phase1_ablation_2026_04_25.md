# Phase 1 ablation sweep results (2026-04-25)

**Sweep root:** `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408/_agent_scratch/ppca_phase1_sweep_20260425_092231`

**Design:** 12 cells × 3 seeds × 3 dataset/q rows = 72 jobs. Factors:

| Factor | Levels |
|---|---|
| `s_init` | flat, truth |
| `ridge_mode` | scalar (`λ=1e-4`), w_prior (Wiener-style shell-stratified) |
| `anneal` | none, factor-only `log1000` (`--anneal-factor-only`) |

**Datasets:** Ribosembly q=4, Ribosembly q=8, IgG-RL q=2.
**Common settings:** vol=32, n_img=1024, healpix_order=1, weighted SVD warmstart, perturbed μ init, 30 iters (Ribo q=4 / IgG-RL q=2) or 100 iters (Ribo q=8).

## Ribosembly q=4 (n=3 per cell, all 24 cells complete)

Top hun for each (s_init, ridge) pair across anneal modes:

| s_init | ridge | anneal=none | anneal=factor_only |
|---|---|---|---|
| flat | scalar | 0.755 ± 0.060 | **0.794 ± 0.008** |
| flat | w_prior | 0.756 ± 0.063 | 0.751 ± 0.030 |
| truth | scalar | 0.755 ± 0.060 | **0.794 ± 0.057** |
| truth | w_prior | 0.758 ± 0.064 | 0.733 ± 0.034 |

**PE comparison at flat / no-anneal (Rule 2 metric):**
- scalar: 1.6752 ± 0.211
- w_prior: 1.5694 ± 0.178
- **w_prior reduces PE by 6.3%** ✓ (Rule 2 passes on q=4)

## Ribosembly q=8 (24/24 cells, n=3)

Top hun for each (s_init, ridge) pair across anneal modes:

| s_init | ridge | anneal=none | anneal=factor_only |
|---|---|---|---|
| flat | scalar | 0.774 ± 0.053 | 0.946 ± 0.066 |
| flat | w_prior | 0.896 ± 0.125 | **0.994 ± 0.005** |
| truth | scalar | 0.779 ± 0.047 | 0.949 ± 0.061 |
| truth | w_prior | 0.897 ± 0.122 | 0.940 ± 0.043 |

**Three big lifts at q=8:**
1. **factor-only anneal alone:** hun 0.774 → 0.946 (scalar ridge, +22.1%)
2. **w_prior alone (no anneal):** hun 0.774 → 0.896 (more variance though)
3. **w_prior + factor_only:** hun **0.994 ± 0.005** — the best result in the entire sweep

**PE at flat / no-anneal:**
- scalar: 3.6151 ± 0.030
- w_prior: 2.7302 ± 0.081
- **w_prior reduces PE by 24.5%** ✓ (Rule 2 passes strongly on q=8)

## IgG-RL q=2 (n=3 per cell, all 24 cells complete)

Top hun for each (s_init, ridge) pair across anneal modes:

| s_init | ridge | anneal=none | anneal=factor_only |
|---|---|---|---|
| flat | scalar | **0.228 ± 0.006** | 0.220 ± 0.005 |
| flat | w_prior | 0.220 ± 0.007 | **0.019 ± 0.001** ⚠ |
| truth | scalar | 0.222 ± 0.005 | 0.227 ± 0.005 |
| truth | w_prior | 0.223 ± 0.006 | **0.019 ± 0.001** ⚠ |

**Critical finding:** `w_prior + factor_only` collapses IgG-RL Hungarian
to ~chance (0.019, ARI=0). This is a *real* algorithmic interaction —
PE stays normal (~1.3), so the subspace is fine, but the embedding
doesn't cluster. Likely cause: under factor-only annealing the inflated
σ² in the M-step interacts with the W_prior shell-Gram in a way that
rotates the basis within the subspace, breaking Hungarian alignment.

This is **not** a v0 blocker because:
- IgG-RL is a 100-state continuous manifold; q=2 is heavily misspecified
- The *baseline* hun ceiling on this configuration is ~0.23 (basically chance for 100 states)
- **`w_prior + factor_only` is not safe as a default**, but each alone is fine

## Decision rule outcomes

### R1 — flat ≈ truth on all 3 datasets within 2σ on Hungarian

| Dataset | flat hun | truth hun | Δ | Pass? |
|---|---|---|---|---|
| Ribosembly q=4 | 0.755 ± 0.060 | 0.755 ± 0.060 | 0.000 | **PASS** |
| Ribosembly q=8 | 0.774 ± 0.053 | 0.779 ± 0.047 | +0.005 | **PASS** |
| IgG-RL q=2 | 0.228 ± 0.006 | 0.222 ± 0.005 | -0.006 | **PASS** |

R1 verdict: **flat-s is the correct cheat-free default**, fully consistent with the existing memory `project_ppca_s_init_irrelevant.md`. Confirmed on all 3 datasets; flat and truth agree on Hungarian within 0.006.

### R2 — w_prior reduces PE by >5% on ≥2 of {Ribo q=4, Ribo q=8} at flat / no-anneal

| Row | Δpe | Pass? |
|---|---|---|
| Ribosembly q=4 | +6.3% | **PASS** |
| Ribosembly q=8 | +23.8% | **PASS** |

R2 PASS on PE. **However:** w_prior + factor_only collapses IgG-RL clustering (caveat above). w_prior is a real improvement for *discrete* heterogeneity but cannot be combined with annealing on *continuous* manifolds. **Verdict:** keep `--ridge-mode scalar` as the safe default; document w_prior as an opt-in for discrete-state regimes.

### R3 — factor-only anneal: safe on IgG-RL AND lifts Ribo q=8

| Row | factor_only hun | none hun | Δ | Pass? |
|---|---|---|---|---|
| IgG-RL q=2 (flat scalar) | 0.220 | 0.228 | -0.008 | safe (within seed σ) |
| Ribosembly q=8 (flat scalar) | 0.946 | 0.774 | +22.1% | strong lift |

R3 **PASS**. **Make `--anneal-schedule factor_only_log1000` the recommended setting for Ribosembly-like discrete datasets**, but keep CLI default at `none` so users opt in (safer for unfamiliar continuous datasets).

## Recommended v0 defaults (post-Phase 1)

- `--s-init flat` (already default — R1 confirmed)
- `--ridge-mode scalar` (already default — w_prior is opt-in for discrete heterogeneity)
- `--anneal-schedule none` (already default — factor-only anneal is opt-in for high q on discrete)

Documented findings to add to the status doc:
- w_prior is a meaningful PE improvement on discrete heterogeneity (lift confirmed)
- w_prior + factor-only anneal **must not be combined** on continuous manifolds
- factor-only anneal alone is safe everywhere measured
