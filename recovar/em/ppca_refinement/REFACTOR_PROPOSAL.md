# `ppca_refinement` — Deep Evaluation & Refactor Proposal

> Audience: someone (you) opening this directory cold and wanting to understand the whole pipeline in <30 minutes of reading.
>
> **Current pain**: 4,251 lines across 11 modules, two ~990-line files (`dense_dataset.py`, `local_dataset.py`) that mostly mirror each other, the worst entry point has 41 keyword arguments, mean-regularization branching is duplicated in 3 places, image-scale resolution is duplicated in 2.
>
> **Goal**: keep the math/algorithms intact, kill the bureaucratic surface area, and impose a structure where each file answers exactly one question.

---

## 1. What the package does (the only thing you have to remember)

One PPCA EM iteration over a particle stack. Each iteration:

```
            ┌─────────────────┐
            │   sufficient    │
            │   statistics    │
            │  (rhs, lhs_tri) │
            └────────▲────────┘
                     │
       ┌─────────────┴───────────────┐
       │  E-step: pose marginalization │   ← per-image posterior over (R, t)
       │  for each image, accumulate   │     produces α_aug = E[z_aug],
       │  α_aug and G_aug into volumes │              G_aug = E[z_aug z_augᵀ]
       └─────────────▲─────────────────┘
                     │ images, μ, W
            ┌────────┴────────┐
            │   dataset +     │
            │   forward model │
            └─────────────────┘

then:

            ┌─────────────────┐
            │  augmented      │
            │  M-step solve   │   ← per-voxel (P+1)×(P+1) linear solve
            │  → μ_new, W_new │     for [μ, W] joint update
            └────────▲────────┘
                     │ rhs, lhs_tri, priors, mean precision
                     │
            ┌────────┴────────┐
            │  postprocess    │   ← optional mask + grid correction
            │  (heuristic)    │
            └─────────────────┘
```

**Two flavors of E-step exist**:
- **Dense** — every image scored against a full HEALPix×translation grid (sparse-pass2 culls candidates between passes).
- **Exact-local** — every image scored against its own short candidate list (`LocalHypothesisLayout`).

Both feed the **same** M-step + postprocess.

A **halfset wrapper** runs the iteration twice (halfset 0 + halfset 1) and FSC-combines for gold-standard scoring.

A **multi-iteration loop** (`refinement_loop.py`) stitches several iterations together with schedule decisions (current_size, HEALPix order, when to switch from single-set to halfset).

---

## 2. Proposed module structure

```
recovar/em/ppca_refinement/
├── __init__.py              # re-exports the 5 public entry points + the configs
├── README.md                # 50 lines: math + reading order (this doc, condensed)
│
├── config.py            *NEW: every dataclass config in one place
│    PostprocessConfig
│    MeanRegularizationConfig
│    SparsePass2Config
│    GeometryConfig          (current_size, q, volume_domain)
│    ScoringConfig           (score_with_masked_images, half_spectrum_scoring,
│                             square_window, relion_texture_interp,
│                             class_log_prior, image_scale_corrections)
│    ScheduleConfig          (image_batch_size, rotation_block_size, mstep_chunk_size)
│
├── engine.py            *RENAMED from dense_engine.py (it's the inner kernel,
│                         not "dense" — local also calls it)
│    fused_dense_pose_ppca_block         # E+M kernel, JIT-compiled
│    dense_pose_ppca_score_stats_blocked
│    PosteriorDiagnostics, DenseImageStats, DenseScoreStats
│    run_dense_ppca_fused_refinement_blocks  # block-list driver
│
├── em_iteration.py      *NEW: single home for the two iteration entry points
│                         (currently split arbitrarily between dense_dataset
│                          and local_dataset; both are 90% identical setup)
│    run_dense_ppca_iteration            # was run_dense_ppca_fused_em_iteration
│    run_local_ppca_iteration            # was run_local_ppca_fused_em_iteration
│    run_halfset_iteration               # ONE halfset wrapper for both flavors
│
├── dataset_blocks.py    *NEW: block iterator (currently in dense_dataset
│                         lines 330-505 + local_dataset lines 367-669, two near-
│                         identical iterators — collapse into one)
│    iter_dataset_blocks                 # dispatches dense vs local internally
│    prepare_dataset_inputs              # was prepare_dense_ppca_dataset_inputs
│
├── diagnostics.py       *NEW: collect the 30+ scattered diagnostic-dict keys
│    build_iteration_diagnostics(...)    # builds the common subset
│
├── mean_regularization.py   (existing — already clean after MeanRegConfig)
│    MeanRegularizationConfig
│    resolve_mean_precision
│    relion_style_mean_precision_from_*
│
├── postprocess.py           (existing — already clean after PostprocessConfig)
│    PostprocessConfig
│    postprocess_ppca_half_volumes
│
├── initialization.py        (existing — fine as-is, ~400 lines, focused)
│    initialize_ppca_from_gt_volumes, initialize_ppca_from_random, ...
│    loading_row_norm_variance_prior, volume_power_variance_prior
│
├── refinement_loop.py       (existing — fine as-is, ~400 lines, focused)
│    run_dense_ppca_refinement_loop      # multi-iteration driver
│    halfset gating, schedule, FSC checkpoint
│
├── schedule.py              (existing — fine as-is, 122 lines)
│    PPCARefinementScheduleState
│    HalfsetResolutionGateDecision
│
├── state.py                 (existing — fine as-is, 37 lines)
│    PoseMarginalPPCAEMState
│
└── fixture_validation.py    (existing — test scaffolding, fine as-is)
```

**Net file change**:
- Delete (merge into `em_iteration.py` + `dataset_blocks.py`):
  - `dense_dataset.py` → split into `em_iteration.py` + `dataset_blocks.py`
  - `local_dataset.py` → merge into `em_iteration.py` + `dataset_blocks.py`
- Rename:
  - `dense_engine.py` → `engine.py`
- Add: `config.py`, `diagnostics.py`, `README.md`

**Result**: ~2,000 fewer lines of bureaucracy, no math changes.

---

## 3. Config dataclass hierarchy (the "where do I tune X" map)

After this refactor, the only kwargs an EM iteration takes are:

```python
def run_dense_ppca_iteration(
    dataset, mu, W=None,
    *,
    # Required inputs
    mean_prior, W_prior, noise_variance,
    rotations, translations,
    # Configs (each bundles ~3-8 fields)
    geometry: GeometryConfig = GeometryConfig(),
    scoring: ScoringConfig = ScoringConfig(),
    schedule: ScheduleConfig = ScheduleConfig(),
    mean_reg: MeanRegularizationConfig = MeanRegularizationConfig(),
    postprocess: PostprocessConfig = PostprocessConfig(),
    sparse_pass2: SparsePass2Config = SparsePass2Config(),
    # Optional priors / masks
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    rotation_translation_mask: np.ndarray | None = None,
    image_indices: np.ndarray | None = None,
    # Behavior switches
    freeze_mean: bool = False,
    fixed_mean_half=None,
) -> DensePPCAFusedEMResult:
```

Down from **41 kwargs** to **6 configs + 5 optionals + 2 switches = 13 names**.

Each config is `@dataclass(frozen=True)`, has sensible defaults, and lives in `config.py`. Configs are kwarg-only, so partial overrides are clean: `mean_reg=MeanRegularizationConfig(tau2_fudge=2.0)`.

---

## 4. Code to delete (duplication map)

From the audit:

| What | Where today | Replace with |
|---|---|---|
| Mean-precision if/elif/else branch | dense_dataset.py:805–818, dense_engine.py:507–519, local_dataset.py:855–870 | `resolve_mean_precision(stats, prior, vshape, config)` (already added in slice 2) |
| Common diagnostics keys (pmax_mean, nsig_mean, log_likelihood, logZ_mean, best_*_idx) | dense_dataset.py:762–797, local_dataset.py:816–842, dense_engine.py:487–503 | `build_iteration_diagnostics(...)` in `diagnostics.py` |
| `_project_augmented_half_volumes` | dense_dataset.py:237–261 | Single shared helper, since `_project_local_augmented` (local_dataset.py:151–167) is nearly identical |
| `_project_local_augmented` | local_dataset.py:151–167 | Same as above; collapse into `dataset_blocks.py` |
| Image-scale resolution | dense_dataset.py:422–428, local_dataset.py:592–601 | `_resolve_image_scale_corrections(...)` in `dataset_blocks.py` |
| `iter_dense_ppca_dataset_blocks` and `iter_local_ppca_dataset_blocks` are 90% the same | dense_dataset.py:330–506, local_dataset.py:367–669 | Single `iter_dataset_blocks(dataset, candidate_set, ...)` that dispatches by candidate-set type |

Already-canonical helpers (good, do not touch):

- `preprocess_batch`, `prepare_reconstruction_batch` ← `recovar/em/dense_single_volume/helpers/preprocessing.py`
- `to_batched_half_pixel_noise`, `make_radial_noise_half` ← `recovar/reconstruction/noise.py`
- `make_fourier_window_spec`, `make_scoring_half_image_weights` ← `recovar/em/dense_single_volume/helpers/{fourier_window,half_spectrum}.py`
- `ForwardModelConfig` ← `recovar/core/configs.py`

These are **already used** by both `dense_dataset.py` and `local_dataset.py`. Don't reinvent them.

---

## 5. Reading order (for understanding the package cold)

After the refactor, this order takes you from "what does it do" to "what does each line do" in <30 minutes:

1. **`README.md`** (NEW, ~50 lines) — math sketch + module map. 5 minutes.
2. **`config.py`** (NEW, ~80 lines) — every tunable lives here. 3 minutes.
3. **`state.py`** (37 lines) — what's in `PoseMarginalPPCAEMState`. 2 minutes.
4. **`engine.py`** (~570 lines) — the JIT-compiled E+M kernel. The hard math is here:
   - `fused_dense_pose_ppca_block` is the per-block kernel
   - `_per_pose_stats_block` is the σ²-weighted projection algebra
   - `compute_ppca_pose_scores_and_moments_no_contrast` is in `recovar/ppca/pose_marginal.py` — the score & moment formulas
   - 10 minutes.
5. **`dataset_blocks.py`** (NEW, target ~400 lines) — turns a `CryoEMDataset` into the blocks `engine.py` consumes. Pure data plumbing. 5 minutes.
6. **`em_iteration.py`** (NEW, target ~300 lines) — the orchestration that wraps `dataset_blocks` + `engine.py` and produces an updated state. 5 minutes.
7. **`refinement_loop.py`** (~400 lines) — multi-iteration driver, schedule, halfset gating. 5 minutes.

Total: ~35 minutes to a complete mental model.

For comparison, today:
- `dense_dataset.py` alone is 989 lines of mixed orchestration + iteration + setup + diagnostics. Reading it cold: ~25 minutes, and you don't even know yet that `local_dataset.py` is 988 lines of mostly the same thing.

---

## 6. Migration plan (incremental, tests green at each step)

**Phase 0 — Done:**
- ✅ `PostprocessConfig` extracted (+ all 5 EM iteration entry points + 3 scripts updated). 86/86 tests pass.

**Phase 1 — In flight:**
- ⏳ `MeanRegularizationConfig` + `resolve_mean_precision` helper. Dense iteration done; dense halfset wrapper done; need: `dense_engine.py`, `local_dataset.py` x2, scripts.

**Phase 2 — Easy wins (target: 1 hour):**
- `SparsePass2Config` (2 fields, dense only — small touchpoint).
- `GeometryConfig` (current_size, q, volume_domain).
- `ScheduleConfig` (image_batch_size, rotation_block_size, mstep_chunk_size).
- `ScoringConfig` (5 booleans + class_log_prior + image_scale_corrections).

After phase 2: `run_dense_ppca_iteration` has ~13 args instead of 41.

**Phase 3 — Structural moves (target: 2 hours):**
- Create `config.py`, move all dataclasses there.
- Create `engine.py` (rename `dense_engine.py`).
- Extract `build_iteration_diagnostics()` helper into `diagnostics.py`. Replace 3 inlined diagnostics dicts with calls.
- Extract `_resolve_image_scale_corrections()` helper.

**Phase 4 — The big collapse (target: 3 hours):**
- Create `dataset_blocks.py`. Move `iter_dense_ppca_dataset_blocks` + `iter_local_ppca_dataset_blocks` + the 2 projection helpers + `prepare_dense_ppca_dataset_inputs` + `_per_image_pose_prior_block` into it. Unify the two iterators into one that dispatches by candidate-set type.
- Create `em_iteration.py`. Move both iteration entry points + both halfset wrappers + `combine_halfset_scoring_model` into it. Both halfset wrappers collapse into one `run_halfset_iteration(flavor, ...)`.
- Delete `dense_dataset.py` and `local_dataset.py`.

**Phase 5 — Documentation (target: 30 minutes):**
- Write `README.md` with the math sketch + module map.
- Update `recovar/em/ppca_refinement/CLAUDE.md` to point at it.

**Each phase ends with `pixi run python -m pytest tests/unit/ppca_refinement/` green (currently 86 tests).**

---

## 7. Open questions / non-goals

- **Performance**: refactor preserves bit-equivalent outputs (configs are pure data). No JIT-compile invalidation expected because we don't touch `engine.py`'s pytree contents.
- **Backward compat**: clean break — no old kwargs accepted. The 3 production scripts and 1 benchmark are updated in lockstep.
- **CLAUDE.md / AGENTS.md**: not in scope for the refactor itself; should be updated as Phase 5.
- **The exact Sparse2Pass thresholds & defaults**: not changing, just moving into `SparsePass2Config(log_threshold=…, enabled=True)`.

---

## 8. What I want from you before continuing

1. **Sign off on the proposed structure** — especially the `dataset_blocks.py` / `em_iteration.py` split. If you'd rather see different boundaries, easier to fix now than later.
2. **Confirm the rename `dense_engine.py → engine.py`** — it's accurate (the engine is shared by both flavors) but renames break agent muscle memory.
3. **Confirm that I should delete `dense_dataset.py` + `local_dataset.py`** rather than keep them as thin shims.
