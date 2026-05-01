# M10 follow-up — Detailed implementation plan (2026-05-01)

Status as of branch `claude/ppca-refine-pose-marginal` @ `dd0e2b2a`:

* **Math layer (M0–M2)**: complete and unit-tested.
* **Engines (M4 dense, M6 sparse)**: complete and unit-tested in
  isolation (brute-force parity, sparse-equals-dense gate).
* **Drivers (M5 dense, M7 local)**: callback-based orchestration
  complete; dataset wiring deferred.
* **Contrast (M8) + multimask + postprocessing (M9)**: complete.
* **M10 partial** (`dd0e2b2a`): halfset_combine + prior_provider +
  Ribosembly dev E2E test on M3 path; production fused engine deferred.

This document is the executable spec for finishing M10. Three phases:

  1. Phase A — Production fused engine + dataset adapter (~half a day)
  2. Phase B — CLI pipeline glue (particles → output) (~quarter day)
  3. Phase C — Multi-dataset Slurm evaluation + PR (Ribosembly + IgG-1D
     + others) (~half-day code, multi-hour Slurm)

Total estimate: 1.5 days code + a multi-hour Slurm window for Phase C.

---

## Phase A — Production fused engine + dataset adapter

The current `dense_pose_ppca_E_step_blocked` returns image-level
aggregates `alpha_aug_acc [B, P]` and `G_aug_tri_acc [B, tri(P)]`
(summed over the (R, T) hypotheses inside the block). This is too
aggregated for proper backprojection, which needs to apply the per-image
moments `γ_irt · α_irt,p` in the volume frame at the matching rotation
``r``. The fix is to interleave backprojection with the pass-2 score
normalization so we never have to materialize per-rotation moments
across all rotations at once.

### A.1 Restructure dense_engine for fused production use

**File:** `recovar/em/ppca_refinement/dense_engine.py`

Add a new fused function alongside the existing
`dense_pose_ppca_E_step_blocked`:

```python
def fused_dense_pose_ppca_block(
    Y1,                          # [B, T, F] complex64
    proj_aug,                    # [R, P, F] complex64
    ctf2_over_noise,             # [B, F] real32
    y_norm,                      # [B] real32
    rotations_block,             # [R, 3, 3] real32  (added arg)
    image_shape,                 # tuple
    volume_shape,                # tuple
    disc_type,                   # str — 'cubic' or 'linear_interp'
    rhs_volume,                  # [P, half_vol] complex64 — accumulator
    lhs_tri_volume,              # [tri(P), half_vol] real64 — accumulator
    pose_log_prior=None,         # [B, R, T] or None
    *,
    significance_threshold=1e-3,
):
    """One pass of the fused dense engine: pass 1 (logZ + best pose)
    → pass 2 (γ + per-rotation backprojection accumulation).

    Returns (rhs_volume, lhs_tri_volume, PosteriorDiagnostics).

    Memory-bounded: never materializes per-rotation γα across the
    rotation block. Per-rotation backprojection happens immediately
    inside the rotation loop.
    """
```

Implementation sketch:

```python
B, T, F = Y1.shape
R, P, _ = proj_aug.shape
q = P - 1

# Pass 1 — score + logZ + best pose (reuse _per_pose_stats_block).
yn, tm, num, g, hz, Hz = _per_pose_stats_block(...)
score, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(...)
if pose_log_prior is not None:
    score = score + jnp.swapaxes(pose_log_prior, -1, -2)
score_flat = score.reshape(B, T * R)
logZ = logsumexp(score_flat, axis=-1)              # [B]
best_flat = jnp.argmax(score_flat, axis=-1)
# … pmax, n_significant, etc.

# Pass 2 — recompute moments + per-rotation backprojection.
score2, alpha, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
    ..., return_moments=True,
)                                                  # alpha [B,T,R,P], G_tri [B,T,R,tri(P)]
if pose_log_prior is not None:
    score2 = score2 + jnp.swapaxes(pose_log_prior, -1, -2)
gamma = jnp.exp(score2 - logZ[:, None, None])      # [B, T, R]

# For each rotation r in the block, build per-image per-component
# Z_rp [B, P, F] = sum_t γ_brt α_brt,p · Y1[b, t]
# and per-image per-pair w_rs [B, tri(P)] = sum_t γ_brt G_aug_tri_brt,rs
# Then backproject through rotation r into rhs and lhs_tri accumulators.
for r_idx in range(R):
    gamma_r   = gamma[:, :, r_idx]                  # [B, T]
    alpha_r   = alpha[:, :, r_idx, :]               # [B, T, P]
    G_tri_r   = G_tri[:, :, r_idx, :]               # [B, T, tri(P)]
    rotation  = rotations_block[r_idx]              # [3, 3]

    # RHS contribution per (b, p):
    Z_rp = jnp.einsum('bt, btp, btf -> bpf', gamma_r, alpha_r, Y1)  # [B, P, F]

    # Stack P "images" per particle and adjoint-slice them in one batched call.
    # Reshape to [B*P, F] half-spectrum images, with rotations broadcast.
    rotations_per_aug = jnp.broadcast_to(
        rotation, (B * P, 3, 3),
    )
    Z_flat = Z_rp.reshape(B * P, F)
    rhs_volume = batch_adjoint_slice_volume_half(
        Z_flat,
        rotations_per_aug,
        rhs_volume,                      # [P, half_vol]
        image_shape, volume_shape, disc_type,
        half_image=True, half_volume=True,
    )

    # LHS contribution per (b, idx_pq):
    # w_rs[b, tri_idx] = sum_t γ_brt G_aug_tri_brt[tri_idx]
    w_rs = jnp.einsum('bt, btk -> bk', gamma_r, G_tri_r)            # [B, tri(P)]
    # Each tri component multiplies (C²/σ²)[b, f] and is adjoint-sliced.
    # Build [B*tri(P), F] = w_rs[b, k] * ctf2_over_noise[b, f]
    weighted_ctf2 = w_rs[..., None] * ctf2_over_noise[:, None, :]   # [B, tri(P), F]
    lhs_input = weighted_ctf2.reshape(B * tri_size, F).real         # real
    rotations_per_lhs = jnp.broadcast_to(rotation, (B * tri_size, 3, 3))
    lhs_tri_volume = batch_adjoint_slice_volume_half(
        lhs_input.astype(jnp.complex64),  # cast for the slicing kernel
        rotations_per_lhs,
        lhs_tri_volume,
        image_shape, volume_shape, disc_type,
        half_image=True, half_volume=True,
    )

return rhs_volume, lhs_tri_volume, PosteriorDiagnostics(...)
```

Note on lhs Hermitian symmetry: G_aug is complex Hermitian, but the
upper-triangle pack via `unpack_tri_to_full` produces a *symmetric*
(not Hermitian) matrix. For the lhs_tri accumulator the legacy convention
is real-valued (matching `recovar.ppca.ppca._pcg_hard_mstep`'s
`lhs_tri: float32`); we project to real here. **Add a unit test** that
verifies this projection matches the legacy convention exactly.

**Tests** (CPU, fast):
* `test_fused_engine_matches_unfused_image_level_aggregates`: with a
  synthetic dataset, run both `dense_pose_ppca_E_step_blocked` (image
  aggregates) and `fused_dense_pose_ppca_block` (volume accumulators).
  After backprojecting the unfused image aggregates by hand in NumPy,
  rhs and lhs_tri must match `atol=1e-4` (CPU64 reference).
* `test_fused_engine_q_zero_matches_run_em`: with q=0 the fused engine
  must produce the same rhs/lhs_tri as
  `recovar.em.dense_single_volume.em_engine.run_em` on a tiny problem.
* `test_fused_engine_jit_compiles_per_block_shape`: confirm
  static-arg-jit on `(B, R, T, F, P, image_shape, volume_shape, disc_type)`.

### A.2 Production block_provider + backprojector

**File:** `recovar/em/ppca_refinement/dataset_adapter.py`

Replace the placeholder `make_simple_block_provider_for_test` with:

```python
def make_production_block_provider(
    cryo,
    *,
    halfset_indices,                  # {0: [...], 1: [...]} of global image idx
    rotation_grid,                    # [R_total, 3, 3] real32 — full sphere or local
    translation_grid,                 # [T, 2] real32 — pixels
    image_batch_size: int = 32,
    rotation_block_size: int = 1024,
    current_size: int | None = None,  # Fourier-window resolution cap
    disc_type: str = "cubic",
    pose_log_prior_fn=None,           # callable (image_indices, rot_idx, trans_idx) -> log prior
):
    """Production block_provider for --pose-mode dense.

    Iterates halfsets × image batches × rotation blocks, builds
    PoseBlock instances with:
      * Y1 = (C · y · phase_t) / σ²       via pre-shifted CTF-weighted whitened images
      * proj_aug = project_half_spectrum(theta_aug, rot_block)
      * ctf2_over_noise per-image
      * y_norm per-image
      * pose_log_prior (uniform = 0 or user-supplied)
      * rotations + translations + image_indices for the backprojector
    """

def make_production_dense_backprojector(
    *,
    image_shape, volume_shape, disc_type,
    half_vol_size,
):
    """Production fused backprojector that calls fused_dense_pose_ppca_block
    per (block, rotation_block) and accumulates rhs/lhs_tri half-volumes.

    Returned function signature matches the M5 Backprojector Protocol:
        (image_stats_blocks, halfset_idx) -> AugmentedPPCAStats
    but is implemented by RE-RUNNING the fused engine with rhs/lhs_tri
    accumulators (the image_stats handed in are diagnostic only).

    Alternative architecture (recommended): bypass the M5 callback split
    and have a single ``run_pose_marginal_iteration_dense`` function in
    iterations.py that drives the fused engine end-to-end. This avoids
    re-running the engine for backprojection.
    """
```

**Recommendation:** ditch the strict M5 separation
(block_provider → engine → image_stats → backprojector → AugmentedPPCAStats)
for the production path. Keep that separation for the test paths
(M4/M6 unit tests need it). The production path is a single
function that walks blocks and emits AugmentedPPCAStats directly:

```python
# recovar/em/ppca_refinement/iterations.py
def run_pose_marginal_iteration_dense(
    state, cryo, rotation_grid, translation_grid, *,
    halfset_indices, mask, masks=None, pc_mask_assignment=None,
    image_batch_size, rotation_block_size, current_size,
    disc_type, pose_log_prior_fn, opts: IterationOpts,
) -> tuple[PoseMarginalPPCAEMState, dict]:
    """Single-iteration production driver for --pose-mode dense.

    Walks halfsets, builds blocks on the fly, runs the fused engine
    per block, accumulates AugmentedPPCAStats per halfset, calls
    solve_augmented_ppca_mstep, halfset-combines, returns the new state.
    """
```

### A.3 Sparse path (M7 local-pose)

**File:** `recovar/em/ppca_refinement/sparse_engine.py`

Add `fused_sparse_pose_ppca_block` analogous to the dense version: a
flat-layout per-hypothesis variant that interleaves segment_sum
aggregation with per-rotation backprojection. Each hypothesis carries
its own rotation; we group hypotheses by rotation for batched
backprojection.

Design note: in the sparse case, hypotheses sharing a rotation can be
batch-backprojected together. Use `LocalHypothesisLayout`-style
bucketing on rotation index to amortize the `accumulate_adjoint_pair`
calls.

### A.4 Halfset combiner upgrade

**File:** `recovar/em/ppca_refinement/halfset_combine.py`

The current `low_resol_join_halfset_combine` treats both regions
(inside and outside the 40 Å sphere) identically (= simple mean). Per
the user caveat, deliver the gold-standard FSC pattern as a separate
follow-up, but at minimum:

* Implement true Fourier-domain blending: low-freq (≤ 40 Å) =
  per-half-mean; high-freq (> 40 Å) = also per-half-mean. The current
  code already does this since both regions use the average. Keep the
  API as-is; the no-op join is documented.
* (Deferred) Add `gold_standard_fsc_combine` that returns
  TWO scoring volumes (one per halfset, using the OPPOSITE half's
  high-freq + averaged low-freq) and threads them through the engines
  via a new `theta_score_per_half` field on the state.

### A.5 Mean prior wiring

**File:** `recovar/em/ppca_refinement/iterations.py`

Wire `make_mean_prior_provider` from `prior_provider.py` into the M5
driver as the default `prior_recompute_fn` when the dataset is supplied.
At iteration `cfg.recompute_once_after_iter`, the driver calls
`compute_relion_prior` on `state.mu_half[0/1]`, computes the new
per-half-voxel `mean_prior`, and replaces `state.mean_prior`.

Test: integration test that verifies after the recompute, the
`mean_prior` is non-trivially different from the initial value.

---

## Phase B — CLI pipeline glue (particles → output)

**File:** `recovar/em/ppca_refinement/cli.py`

Replace the current `--pose-mode {dense,local}` "Python API only"
message with the production wiring:

```
1. Load particles.star → CryoEMDataset via recovar.data_io.cryoem_dataset.
2. Load --init-mean → Fourier volume.
3. Load --init-poses (optional) → per-image rotation + translation.
4. Load --init-W (optional) or initialize via --init-state restart.
5. Build initial PoseMarginalPPCAEMState (mu_half via halfset
   reconstruction from init-mean + halfset assignment;
   W_half via fixed-pose PPCA M3 driver with EM_iter=1 if --init-W
   not supplied).
6. Build rotation_grid:
     --pose-mode dense: full HEALPix sphere at given order.
     --pose-mode local: local neighborhood per image around init-poses.
7. Build translation_grid from --max-shift and --shift-step.
8. Construct mean_prior_provider via make_mean_prior_provider.
9. For it in range(opts.EM_iter):
     state, log = run_pose_marginal_iteration_dense(state, cryo, ...)
       OR run_pose_marginal_iteration_sparse(state, cryo, ...)
     write per-iter MRCs (mu_score, W_score per PC) and diagnostics
     to {out}/iter_{it:03d}/.
10. After EM:
     U, S, W_half = finalize_ppca_state(state, ...)
     write final basis MRCs, S as pickle, embeddings via postprocess
     wrapper from recovar.ppca.postprocess.
11. save_state(state, {out}/state.pkl).
```

**Tests** (integration, GPU):

* `test_cli_pose_mode_fixed_e2e` — already-working M3 path through the
  CLI on Ribosembly fixture; output dir contains the expected files.
* `test_cli_pose_mode_dense_smoke` — `--pose-mode dense --max-resolution 20`
  on the Ribosembly fixture; one iter; verify state.pkl + iter_000/
  exist.
* `test_cli_state_pkl_restart` — run 2 iters, save state, restart from
  state.pkl, run 1 more iter; the resulting state should match a
  3-iter-from-scratch run modulo CG non-determinism.

---

## Phase C — Multi-dataset evaluation

**File:** `scripts/ppca_refine_eval.py` (new)

A scripts entry mirroring `recovar/ppca/compare_covariance_vs_ppca_pipeline.py`:

```
Usage:
    pixi run python scripts/ppca_refine_eval.py \
        --datasets Ribosembly,IgG-1D,IgG-RL,Tomotwin-100 \
        --grid-size 128 --n-images 100000 \
        --zdim 6 --em-iters 20 \
        --pose-modes fixed,dense,local \
        --results-root /scratch/gpfs/GILLES/mg6942/_agent_scratch/ppca_refine_eval

For each (dataset, pose-mode) cell:
  1. Build the synthetic CryoBench dataset via recovar.simulation.synthetic_dataset.
  2. Run recovar ppca-refine via subprocess with the matching CLI.
  3. Score the result with the existing
     recovar.output.metrics machinery
     (FSC vs GT, embedding error, eigenvalue calibration).
  4. Track runtime + GPU memory via the existing pipeline-comparison
     infrastructure.
  5. Append to a CSV at {results-root}/scores.csv.

After all cells:
  6. Summarize via a markdown table (extract via
     pixi run python scripts/extract_regression_tables.py format).
```

**Slurm submission** via existing `scripts/run_tests_parallel.sh`-style
parallelization. Each (dataset, pose-mode) cell is a separate Slurm job
on `cryoem` partition with `--gres=gpu:1`, `--exclusive`, `--time=12:00:00`.
Plan ~16 jobs (4 datasets × 4 modes including baseline RECOVAR PPCA
without pose marginalization).

**Comparison tables for the PR description** (per project root
CLAUDE.md mandatory format):

```
### Quality Comparison (current vs baseline)
| Dataset | Mode | spa_pcs_relative_variance_4 | spa_locres_90pct | hun_acc | ARI | Status |
|---------|------|------------------------------|-------------------|---------|------|--------|
| Ribosembly | fixed | <baseline> | <baseline> | <baseline> | <baseline> | OK |
| Ribosembly | dense | <pose-marg score> | ... | ... | ... | ... |
| Ribosembly | local | <pose-marg score> | ... | ... | ... | ... |
| IgG-1D | fixed | ... | ... | ... | ... | ... |
| IgG-1D | dense | ... | ... | ... | ... | ... |
| ...
```

```
### Performance Comparison (current vs baseline, H100)
| Dataset | Mode | EM-iter time (s) | Peak GPU (GB) | Convergence iters | Status |
|---------|------|-------------------|---------------|-------------------|--------|
| ...
```

**Phase success criterion** (CLAUDE.md §15):

```
The refinement-first PPCA run is at least as stable as fixed-pose PPCA,
does not degrade consensus map quality vs the starting mean, and improves
or stabilizes pose likelihood compared with fixed poses.
```

If the dense or local pose-marginal modes degrade quality on any
dataset, **escalate** rather than ship — investigate via dump-driven
diagnostics (compare per-iter `pmax`, `nr_significant`, log-evidence
between modes; check for known-harmful patterns like eigenvalue update
during EM).

---

## Pre-PR checklist

```
1. git fetch origin && git rebase origin/claude/relion-parity-local-search-fix
   (PR target is relion-parity, NOT dev — this branch is based on
   relion-parity)
2. ./scripts/run_tests_parallel.sh long-test
   (full long-test on the rebased branch; expect ~2-3h)
3. pixi run python scripts/extract_regression_tables.py
4. PR title <70 chars, body summary + test plan checklist
5. Quality + performance comparison tables (see Phase C output)
6. Slurm job IDs for traceability
7. PR target: claude/relion-parity-local-search-fix (or dev once
   relion-parity merges in)
8. Closes #N for any tracked issues
```

---

## Branching for this follow-up

The current branch `claude/ppca-refine-pose-marginal` (12 commits, all
green tests) is the M0–M9 + M10-partial deliverable. Phase A should
land on a new branch `claude/ppca-refine-fused-engine` off this one;
Phases B and C land on `claude/ppca-refine-cli` and
`claude/ppca-refine-eval` respectively. Each is a clean commit chain
with tests at every step (no broken intermediate states).

After Phase C passes, merge them sequentially via PR into
`claude/ppca-refine-pose-marginal`, then open the umbrella PR into
`claude/relion-parity-local-search-fix`.

---

## Effort estimate summary

| Phase | Code effort | Slurm time |
|---|---|---|
| A — Fused engine + dataset adapter | ~4 hr focused | nil |
| B — CLI pipeline glue | ~2 hr focused | nil (smoke tests on login GPU) |
| C — Multi-dataset eval + tables | ~4 hr scripting | 12–24 hr Slurm |
| **Total** | ~10 hr | 12–24 hr Slurm |

Stretch (deferred): gold-standard FSC halfset combine
(per-half scoring volume) + tests; ~3 hr extra.
