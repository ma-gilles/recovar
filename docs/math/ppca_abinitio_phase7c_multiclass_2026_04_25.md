# Phase 7c — multiclass μ-init for cold-μ rescue (2026-04-25)

**Sweep root:** `_agent_scratch/ppca_phase7c_multiclass_20260425_192701/`
**Module added:** `recovar/em/ppca_abinitio/cold_init.py`
**CLI added:** `--mu-init multiclass --multiclass-K 5 --multiclass-iters 50`

## Goal

Phase 6 established that v0 is **mu-init-bound**: zero-μ + svd-U collapses to hun ≈ 0.29 on Ribosembly q=4 (vs 0.79 with perturbed-μ). σ²-annealing (Phase 6.5) does not rescue it. Phase 7c tests whether a K-class softness-annealed EM bootstrap from random blob volumes — extracted from the prototype script `run_abinitio_multiclass.py` and the memory note `project_ppca_multiclass_abinitio_works.md` (which claimed hun=0.557 on Ribosembly from zero knowledge) — can rescue cold-μ.

## Method

`multiclass_mu_init(cfg, ds, K=5, n_iters=50)` runs:
1. K random Gaussian-blob volumes from independent seeds
2. Frequency-marched K-class EM (~50 iters): factored class+pose posterior with class softness annealed 1.0 → 0.0; per-class Wiener M-step with dead-class protection
3. Returns mean of K class volumes as μ_init

The standard `run_two_stage` then runs SVD warmstart + main EM loop on top of that μ_init. **No GT fields read** in the algorithmic path.

## Results (24 cells: 4 datasets × 2 seeds × {zero, multiclass, perturbed})

| Dataset | q | σ | zero hun | **multiclass hun** | perturbed hun (target) | % of gap closed |
|---|---|---|---|---|---|---|
| Ribosembly | 4 | 0.01 | 0.293 ± 0.007 | **0.375 ± 0.077** | 0.798 | 16% |
| Ribosembly | 8 | 0.01 | 0.217 ± 0.000 | **0.315 ± 0.061** | 0.811 | 17% |
| IgG-1D | 2 | 0.10 | 0.163 ± 0.000 | **0.170 ± 0.001** | 0.271 | 7% |
| IgG-RL | 2 | 0.10 | 0.161 ± 0.003 | **0.168 ± 0.005** | 0.230 | 10% |

PE essentially unchanged across all conditions; the small hun lift comes from k-means clustering on a slightly better μ, not from improved subspace recovery.

## Verdict — NOT MET

The predetermined Phase 7c criterion was: lift Ribo q=4 cold-μ hun from 0.29 → ≥ 0.65 (matching ~80% of the perturbed-μ reference) without regressing perturbed-μ.

**Actual lift: +0.08 hun on Ribo q=4 — only ~16% of the gap.**

The full prototype pipeline (`run_abinitio_aligned.py`) reportedly achieved hun=0.557 on Ribosembly. My clean port deliberately omitted three downstream stages that the prototype's lift evidently depends on:

1. **SO(3) alignment of all class volumes to a dominant class** (~150 lines, `align_volumes_so3` + `align_class_volumes`)
2. **Per-image best-pose extraction from aligned multiclass posterior** (`extract_poses_from_multiclass`)
3. **Known-pose factor update + reconstruction** (`reconstruct_at_known_poses`, `known_pose_factor_update`)

Without those, the clean multiclass-μ-init only gives a marginal bump.

## What this is good for

The clean multiclass μ-init is still useful as:

1. **A diagnostic baseline** — establishes that ~16-17% of the cold-μ gap can be closed with random blobs alone, without alignment.
2. **An `--mu-init multiclass` opt-in CLI flag** for users who want a marginally better starting point than zero μ without porting a 600-line aligned pipeline.
3. **Infrastructure for Phase 7d** — the K-class EM machinery is now in `recovar/em/ppca_abinitio/cold_init.py` as a reusable module, so a follow-up branch can add the alignment + known-pose stages on top.

## Honest position on cold-μ rescue

**Cold-μ remains the largest unsolved gap in v0.** Phase 7c demonstrates that the simplest version of the multiclass recipe is insufficient. Two follow-up directions:

- **Phase 7d (deferred):** port the full aligned pipeline (`cold_init_aligned.py` ~600 lines). Memory says hun=0.557 on Ribosembly. Would close ~50% of the cold-μ gap.
- **Pipeline bridge (deferred):** the natural production input is a μ from a preceding homogeneous reconstruction. v0 is designed for that bootstrap. The pipeline-bridge follow-up branch (`claude/ppca-abinitio-pipeline-bridge`) handles this directly.

For v0, the **honest documentation** is:
- `--mu-init perturbed` is the recommended setting (simulates a homogeneous-reconstruction output)
- `--mu-init multiclass` is an opt-in for fully cold-start scenarios; lifts hun by ~+0.08 on Ribosembly, ~+0.01 on continuous manifolds — far from a full rescue
- True cold-start ab-initio is **not solved** by v0 alone; it needs either a prior homogeneous μ or the deferred Phase 7d aligned pipeline

## Files added by Phase 7c

- `recovar/em/ppca_abinitio/cold_init.py` — clean multiclass μ-init module (~250 lines)
- `scripts/ppca_abinitio/submit_phase7c_multiclass_ab.sh` — A/B sweep
- `scripts/ppca_abinitio/run_cryobench.py` — `--mu-init multiclass` + `--multiclass-K` + `--multiclass-iters` flags

## Defaults — UNCHANGED

`--mu-init perturbed` remains the default. Documentation will note the +0.08 modest lift for `multiclass` as opt-in; will NOT claim cold-μ rescue.
