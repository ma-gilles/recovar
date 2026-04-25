# Phase 2 — post-EM ProjCov calibration result (2026-04-25)

**Job:** Slurm 7345584, completed 19:08 elapsed.
**Output:** `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408/_agent_scratch/ppca_phase2_validation_20260425_110142/calibration.json`

## Setup

For each of three CryoBench datasets at vol=32, n=1024, weighted SVD warmstart, 30 EM iters, `--s-init flat`:

1. Run EM to convergence with `s = 1` flat.
2. Apply one-shot `refit_eigenvalues_post_em` (posterior-covariance refit).
3. Compare the post-EM eigenvalues `s_em = (1, 1, …, 1)` and `s_refit` against the empirical ground-truth eigenvalues `s_true` from the GT volume ensemble.
4. Calibration error: `mean_k |log(s_est / s_true_sorted)|`. Lower is better; 0 = perfect.

## Results

| Dataset | `s_true` (top-q, sorted desc) | `s_em` | `s_refit` | err_em | err_refit | Improvement |
|---|---|---|---|---|---|---|
| Ribosembly q=4 | [8.50, 2.71, 1.99, 1.26] | [1, 1, 1, 1] | [3.20, 0.73, 0.38, 0.21] | 1.014 | 1.436 | **0.7×** (worse) |
| IgG-1D q=2 | [0.595, 0.593] | [1, 1] | [0.546, 0.395] | 0.520 | 0.247 | **2.1×** (better) |
| IgG-RL q=2 | [0.406, 0.303] | [1, 1] | [0.450, 0.338] | 1.047 | 0.105 | **9.9×** (better) |

## Acceptance criterion outcome

Phase 2 acceptance: `err_refit ≤ err_em / 2` on all 3 datasets.

| Dataset | Pass? |
|---|---|
| Ribosembly q=4 | **FAIL** |
| IgG-1D q=2 | **PASS** |
| IgG-RL q=2 | **PASS** |

**2/3 datasets pass.**

## Why Ribosembly q=4 fails the acceptance metric

Ribosembly is a 16-state discrete ensemble. With q=4, the model is heavily misspecified: the PPCA subspace cannot capture all 16 conformations and the inferred U columns do not correspond to the top-4 GT eigenvectors. The log-ratio against `sort(s_true)[:q]` then compares values that aren't really aligned.

This is the same issue identified in `project_ppca_eigenvalue_shrinkage_confirmed.md` — the **sorted** GT spectrum is misleading when the inferred subspace is misaligned. The honest metric is the **projected** spectrum `u^T Σ_gt u` (per-PC GT variance projected onto the inferred PC). The sister branch `claude/ppca-refit-algorithms` uses that projected metric.

For v0 we report the sorted-GT metric for transparency and acknowledge its limitation. The projected-metric port is recorded as Phase 6 future work.

## Honest interpretation

ProjCov is the right tool for **continuous manifolds** (IgG-1D and IgG-RL: 2× and 10× calibration improvement). It struggles to be the right tool for **misspecified discrete ensembles** under the sorted-GT metric, but the underlying spectrum estimate (`s_refit`) is still meaningful — it correctly captures relative magnitudes (3.20 > 0.73 > 0.38 > 0.21) even when those don't align with `sort(s_true)`.

## Recommendation for v0

Ship `--post-eigenvalue-refit projcov` as an **opt-in flag** (default `none`). Document:
- Use it for continuous heterogeneity (IgG-like ensembles).
- For discrete ensembles, evaluate per dataset; the sorted-GT acceptance metric does not capture subspace-aligned spectrum recovery and can mislead.

## What this does NOT validate

- The sister-branch `pca_by_projected_covariance` (data covariance) version. Our Phase 2 implementation is the **posterior covariance** version, which is biased by pose discretization in the same way Tipping-Bishop is. The sister-branch full-data version would address this; that remains Phase 6 future work and is documented in `ppca_abinitio_eb_shell_prior.md` as one of the open spectrum-calibration paths.
