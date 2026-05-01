# Pose-Marginalized PPCA Refinement — Agent Guide

**Project:** `recovar ppca-refine` — refinement-first, EM-style, pose-marginalized
PPCA. Joint refinement of `μ`, `W₁..W_q`, pose posterior, noise/contrast,
loading prior. Starts from existing consensus (homogeneous, k-class, or
imported RELION). Ab-initio / InitialModel PPCA is **deferred**.

**This file is your operating contract.** Re-read on every fresh session
before writing code. Engineering priority: **Correctness > Performance >
Clarity.**

---

## How to use this guide

| When you need… | Read |
|---|---|
| Scope rules, non-negotiables, milestone gates, naming, anti-patterns | **this file** |
| Math derivations (model, per-pose posterior, augmented M-step, prior formulas) | `docs/math/ppca_refine_plan_2026_05_01.md` |
| Implementation playbook (code skeletons, line-level PCG audit, full reuse table, dataclass shapes, CLI sketches, integration test list) | `docs/math/ppca_refine_implementation_2026_05_01.md` |
| **Sequential implementation steps** (start here if unsure where to begin; Phase 0 = verify rebase didn't break legacy PPCA) | **`docs/math/ppca_refine_implementation_steps_2026_05_01.md`** |
| Parent EM rules (RELION parity, env, build, tests) | `recovar/em/CLAUDE.md`, `recovar/em/AGENTS.md` |
| Repo-wide rules (testing tiers, branching, PR format) | `CLAUDE.md` (root) |

If this file disagrees with the plan/implementation docs, **this file
wins** — it captures branch-reality decisions, the corrected name table,
and audit findings that supersede the original plan.

---

## Startup checklist (do before any code change)

  1. `cd $WORKDIR` (a `recovar_wt_ppca_refine_*` worktree on
     `claude/ppca-refine-pose-marginal`, branched from
     `claude/relion-parity-local-search-fix`).
  2. Run the provenance gate (see Workflow §). Stop if it fails.
  3. Read this file in full. Skim the implementation doc for the milestone
     you're about to touch.
  4. Confirm the milestone you're starting from `git log --oneline -10`
     and any open `WIP:` commits.
  5. If the user request maps to a milestone, jump to its **playbook card**
     below. If it doesn't map cleanly, stop and ask.

---

## Non-negotiables

Numbered so they can be referenced in commit messages.

  1. **Refinement first.** Build high-resolution refinement before any
     InitialModel/VDAM work.
  2. **Latent prior is identity in v1.** `z ~ N(0, I_q)`. PC eigenvalues
     belong in `W`, not in the latent prior.
  3. **`W_prior` is a variance**, not a precision. Regularizer is
     `1 / W_prior`. Larger `W_prior` ⇒ weaker regularization.
  4. **`W_prior(ξ, k) = max(τ_floor, α_prior · d_ppca(shell(ξ)) / q_total)`**.
     Use `q_total = opts.zdim`, never `q_active`.
  5. **Don't conflate priors.** RELION `tau2`, `variance_prior`,
     `prior_total_signal`, `prior_shell_subtracted`, and the mean
     reconstruction prior are NOT the PPCA loading prior.
  6. **Contrast order.** No-contrast first; profile second; marginalized
     third. M8 only.
  7. **One score function.** Dense and sparse engines call the *same*
     `compute_ppca_pose_scores_and_moments_*` function. No drift.
  8. **Sparse pruning is support-only.** Pruning may restrict the
     hypothesis support; it must not alter per-hypothesis scores.
  9. **Augmented M-step is joint.** Cross-component `G_aug[r,s]` terms
     (μ–W and W–W blocks) must appear in LHS accumulation.
  10. **Reuse, don't fork.** `recovar/ppca/` is the trusted starting
      point — generalize in place, write parity tests, don't write a
      parallel PPCA.
  11. **`ppca-abinitio-v0` is not authority.** Naming/logging only.
  12. **Ab-initio / InitialModel PPCA is deferred** until k-class
      InitialModel bugs are resolved.

Process invariants (from the project root and `recovar/em/CLAUDE.md`):

  * Never widen tolerances or edit baselines without explicit approval.
  * Don't modify `recovar/em/heterogeneity.py` (separate owners).
  * Preserve `EMState.Ft_y` and `EMState.Ft_CTF` after `finish_up_M_step`
    (heterogeneity path reads these).
  * Branch is `claude/ppca-refine-pose-marginal`. PR target is `dev`.
    Never push to `dev` directly.

---

## Reuse rules (operative)

  * **If a function in `recovar/ppca/*.py` already does what we need, call it.**
    Do not rewrite working PPCA code.
  * **If it does *almost* what we need** (e.g., `_pcg_hard_mstep` with `q`
    components vs the `q+1` we need), generalize in place with a runtime
    arg. **Required gate:** add a backwards-compat regression test on a
    pinned tiny fixture; the legacy fixed-pose pipeline must produce
    identical outputs.
  * **If the work is structurally different** (per-pose marginalization,
    two-pass accumulation across rotation/translation), put new code in
    `recovar/ppca/{pose_marginal,augmented_mstep,pose_accumulators,pc_prior_config}.py`
    and `recovar/em/ppca_refinement/{state,dense_engine,sparse_engine,iterations,cli}.py`.
    Import from the existing PPCA — don't fork.
  * Do **not** create `recovar/em/initial_model_ppca/` or any
    InitialModel-flavored package.

---

## Code organization (target)

```
recovar/
  ppca/                          ALREADY EXISTS — extend, don't fork
    ppca.py                      legacy fixed-pose EM (kept stable; PCG generalized in place)
    contrast_posterior.py        reuse as-is for M8
    prior_estimation.py          estimate_hybrid_shell_prior_from_data
    ppca_iterative_projcov.py    reuse where safe
    ppca_iterative_refitb.py     reuse where safe (post-EM eigenvalue refit only)
    postprocess.py               final SVD / orthonormalization
    pose_marginal.py             NEW (M1)
    augmented_mstep.py           NEW (M2)
    pose_accumulators.py         NEW (M2)
    pc_prior_config.py           NEW (M0)
  em/
    dense_single_volume/         ALREADY EXISTS — reuse engines + helpers
    ppca_refinement/             NEW
      state.py                   PoseMarginalPPCAEMState (M5+)
      dense_engine.py            dense two-pass E-step (M4)
      sparse_engine.py           local/sparse two-pass E-step (M6)
      iterations.py              EM loop driver (M5+)
      cli.py                     `recovar ppca-refine` (M3)
      CLAUDE.md                  this file
      AGENTS.md                  loader pointing here (Codex)
```

---

## Naming convention (canonical, operative)

Use these names everywhere new code is written. They replace the legacy
names in `recovar/ppca/ppca.py::_e_step_half_inner`:

| Legacy | Canonical | Shape | Dtype | Math |
|---|---|---|---|---|
| `y_norm_sq` | `y_norm` | `[..., ]` | f32 | `<x,x>` |
| `t` | `t_mx` | `[..., ]` | f32 | `<x,m>` |
| `nu` | `nu_mm` | `[..., ]` | f32 | `<m,m>` |
| `g` | `g_zx` | `[..., q]` | c64 | `B* x` |
| `h` | `h_zm` | `[..., q]` | c64 | `B* m` |
| `H` | `Hzz` | `[..., q, q]` Hermitian | c64 | `B* B` |

`B = Σ_i^{-1/2} C_i P_R W`, `m = Σ_i^{-1/2} C_i P_R μ`,
`x = T_{-t} Σ_i^{-1/2} y_i`. The legacy `H` is what we now call `Hzz` —
**don't confuse with `compute_H_B` / `compute_little_H_b` in
`recovar/em/heterogeneity.py`, which are different objects.**

Augmented moments (per pose):

```
alpha_aug   [..., q+1]                    E[c · [1; z]]
G_aug_tri   [..., (q+1)(q+2)/2]           upper triangle of E[c² · [1;z][1;z]*]
theta_aug = [θ₀, θ₁, …, θ_q] = [μ, W₁, …, W_q]      r=0 is mean (uses mean prior)
```

When generalizing the PCG (M2), keep both legacy and canonical names as
aliases for one transition commit, then drop legacy.

---

## Required public signatures (don't invent — implement these)

```python
# recovar/ppca/pose_marginal.py  (M1)
def compute_ppca_pose_scores_and_moments_no_contrast(
    y_norm, t_mx, nu_mm,            # [...] real
    g_zx, h_zm,                     # [..., q] complex
    Hzz,                            # [..., q, q] complex Hermitian
    *, return_moments: bool,
) -> tuple[Array, Array | None, Array | None]:
    """Vectorized over arbitrary leading batch dims. JIT-friendly.
    Cholesky for M = I + Hzz; log-det from L; cho_solve for S_z;
    symmetrize Hzz; jitter only behind explicit debug_jitter. No pinv."""

# recovar/ppca/augmented_mstep.py  (M2)
def solve_augmented_ppca_mstep(
    stats: AugmentedPPCAStats,
    *, mean_prior, W_prior, masks, solver_opts,
    theta_init: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    """Calls into the generalized recovar.ppca.ppca._pcg_hard_mstep with
    n_components = q+1 and reg_diag stacked as [mean_reg_diag, W_reg_diag]."""

# recovar/em/ppca_refinement/dense_engine.py  (M4)
def dense_pose_ppca_E_step_and_stats(
    state, dataset, image_indices, sampler, opts,
) -> tuple[AugmentedPPCAStats, PosteriorDiagnostics]:
    """Two-pass: pass-1 logsumexp + best pose + significance;
    pass-2 recompute γ + accumulate. Never materializes [N,R,T,*]."""

# recovar/em/ppca_refinement/sparse_engine.py  (M6)
def sparse_pose_ppca_E_step_and_stats(
    state, dataset, layout: LocalHypothesisLayout, opts,
) -> tuple[AugmentedPPCAStats, SparsePPCAPosterior]:
    """Same score function as dense; significance pruning restricts
    support only — must not alter per-hypothesis scores."""
```

Sufficient stats:

```python
@dataclass
class AugmentedPPCAStats:
    rhs: Array           # [half_vol, q+1] complex64        Σ γ α_r A* x
    lhs_tri: Array       # [half_vol, (q+1)(q+2)//2] real64 Σ γ G_rs A* A
    residual_num: Array  # per-shell numerator
    residual_den: Array  # per-shell denominator
    log_likelihood: float
    n_images: int
    diagnostics: dict
```

Axis order matches `recovar/ppca/ppca.py::E_M_step_batch_half`. **Don't
re-axis the codebase.**

---

## Reuse pointers (high-frequency lookup)

| Need | File:function |
|---|---|
| Per-pose stats from existing fixed-pose PPCA | `recovar/ppca/ppca.py::_e_step_half_inner` |
| Existing PCG M-step (generalize q→q+1) | `recovar/ppca/ppca.py::_pcg_hard_mstep` (line audit in implementation doc) |
| Auto W_prior estimator | `recovar/ppca/prior_estimation.py::estimate_hybrid_shell_prior_from_data` |
| Contrast posterior moments (M8) | `recovar/ppca/contrast_posterior.py` |
| Final basis SVD / orthonormalization (M9) | `recovar/ppca/postprocess.py`, `ppca_iterative_refitb.py` |
| Dense two-pass engine | `recovar/em/dense_single_volume/em_engine.py::run_em` |
| K-class dense engine (closest analogue) | `recovar/em/dense_single_volume/dense_k_class_engine.py::run_dense_k_class_em_native` |
| Local engine | `recovar/em/dense_single_volume/local_em_engine.py::run_local_em_exact` |
| K-class local engine (closest analogue) | `recovar/em/dense_single_volume/local_k_class_engine.py::run_local_k_class_em_native` |
| Local hypothesis layout + bucketing | `recovar/em/dense_single_volume/local_layout.py` |
| Half-spectrum projection | `recovar/em/dense_single_volume/helpers/projection.py::project_half_spectrum` |
| Half-spectrum backprojection | `recovar/em/dense_single_volume/helpers/backprojection.py::adjoint_slice_volume_half`, `accumulate_adjoint_pair` |
| Half-volume accumulator + x=0 enforcement | `recovar/em/dense_single_volume/helpers/half_volume_mstep.py`, `local_backprojection.py::enforce_relion_half_volume_x0_hermitian` |
| `current_size` scheduling | `recovar/em/dense_single_volume/helpers/fourier_window.py` |
| Pose log-priors | `recovar/em/dense_single_volume/helpers/orientation_priors.py`, `helpers/translation_prior.py` |
| Significance pruning + bucketed pass-2 | `recovar/em/dense_single_volume/helpers/significance.py`, `helpers/sparse_pass2_bucketed.py`, `helpers/oversampling.py` |
| Halfset combine for scoring | `recovar/em/dense_single_volume/helpers/convergence.py::RefinementState` |

Full table with one-line purposes and signatures:
`docs/math/ppca_refine_implementation_2026_05_01.md`.

---

## Milestone playbook

Each card: **scope · files touched · gate test · failure mode**. Don't
move past a milestone until its gate passes.

### M0 — Audit + naming map + PCPriorConfig

  * **Scope:** Add `pc_prior_config.py` with `PCPriorConfig` dataclass.
    Wire prior diagnostics into the existing PPCA state-serialization path.
  * **Files:** `recovar/ppca/pc_prior_config.py` (new); minor edits to
    PPCA state save/load.
  * **Gate:** legacy PPCA pipeline still runs end-to-end with
    `PCPriorConfig(latent_prior_mode="identity")` defaults.
  * **Watch:** don't touch math; this milestone is plumbing.

### M1 — Per-pose math, no contrast

  * **Scope:** `compute_ppca_pose_scores_and_moments_no_contrast`.
    Cholesky, log-det from `L`, no `pinv`, vectorized.
  * **Files:** `recovar/ppca/pose_marginal.py` (new);
    `tests/unit/ppca_refinement/test_pose_marginal.py` (new).
  * **Gate tests (must pass):**
    * `q=0` reduction → homogeneous score + pose-independent constant.
    * `W=0` reduction → homogeneous score + pose-independent constant.
    * Random PD `Hzz` matches brute-force latent integral (tiny `q`).
    * Basis rotation invariance: `W ← W Q`, `Q` orthogonal → score
      unchanged.
  * **Watch:** floating-point — symmetrize `Hzz` defensively;
    don't carry `pinv` from legacy code.

### M2 — Augmented M-step (q → q+1 PCG generalization)

  * **Scope:** Generalize `_pcg_hard_mstep` in place. Component `r=0`
    uses `mean_reg_diag`; `r ≥ 1` use `1 / W_prior`. Single-mask only.
    Multimask via `pc_mask_assignment = [mean_mask_idx, *W_mask_assignment]`
    (gate fully on M9). Add `solve_augmented_ppca_mstep` wrapper +
    `AugmentedPPCAStats`.
  * **Files:** `recovar/ppca/ppca.py` (in-place edit; line audit in
    implementation doc); `recovar/ppca/augmented_mstep.py`,
    `pose_accumulators.py` (new); tests.
  * **Gate tests (must pass):**
    * **Backwards-compat regression:** legacy fixed-pose pipeline on a
      pinned tiny fixture produces identical outputs after the
      generalization.
    * Fixed-mean reduction: augmented PCG with `μ` fixed matches legacy
      `_pcg_hard_mstep`.
    * Free-mean toy: augmented PCG vs explicit dense normal-equations
      solve, agreement to 1e-5.
    * `W_prior` penalty: `Σ |W_k(ξ)|² / W_prior(ξ,k)` matches the
      augmented solve's prior term.
  * **Watch:** the gridding kernel (`sinc²`) is baked into the operator
    in `_pcg_hard_mstep` — keep it. Don't change axis order
    (`[half_vol, q]` for rhs, `[half_vol, tri]` for lhs_tri).

### M3 — Fixed-pose driver `recovar ppca-refine --pose-mode fixed`

  * **Scope:** Boring CLI wrapper that calls the existing
    `recovar/ppca/ppca.py` EM directly with the augmented M-step. Output
    parity with the legacy fixed-pose PPCA pipeline.
  * **Files:** `recovar/em/ppca_refinement/cli.py`,
    `recovar/em/ppca_refinement/iterations.py`,
    `recovar/em/ppca_refinement/state.py` (subset for fixed mode).
  * **Gate:** end-to-end run on a tiny fixture matches legacy PPCA
    pipeline outputs.
  * **Watch:** this milestone is "boring + reliable." Resist scope creep.

### M4 — Dense pose-marginalized E-step, no contrast

  * **Scope:** `dense_pose_ppca_E_step_and_stats`. Project augmented bank
    `[μ, W]` to `proj_aug [R, p, F]`. One half-spectrum GEMM for D
    `[B,T,R,p]`. Translation-independent `K_aug [B,R,p,p]`.
  * **Files:** `recovar/em/ppca_refinement/dense_engine.py`.
  * **Gate tests (must pass):**
    * Dense block matches brute-force enumeration (tiny image, tiny grid).
    * `q=0` dense path matches `dense_single_volume/em_engine::run_em`.
    * Fixed-pose dense limit (single hypothesis) matches M3 stats.
  * **Watch:** **never materialize `[N, R, T, *]`**. Allowed block
    tensors only. Set `RECOVAR_DEBUG_ASSERT_NO_FULL_POSTERIORS=1` in tests.

### M5 — Dense driver `--pose-mode dense --engine dense`

  * **Scope:** EM loop wiring; halfset combine via existing
    `RefinementState` path; W-init falls back to M3 fixed-pose.
  * **Gate:** dense low-resolution run on synthetic linear-heterogeneity
    fixture recovers `W` subspace up to orthogonal rotation.

### M6 — Sparse / local pose-marginalized E-step

  * **Scope:** `sparse_pose_ppca_E_step_and_stats`. Reuse
    `LocalHypothesisLayout` + bucketing. Same score function as dense.
  * **Files:** `recovar/em/ppca_refinement/sparse_engine.py`.
  * **Gate test:** **sparse with all hypotheses retained matches dense
    bit-for-bit** (within float32 fused-multiply tolerance) on a tiny
    image batch and pose grid.
  * **Watch:** pruning may restrict support; it must not alter
    per-hypothesis scores. Don't depend on `use_global_significant_support`
    (known regression: per-image Python loop).

### M7 — Main local-pose driver `--pose-mode local --engine sparse`

  * **Scope:** Mode B local refinement around current poses. Update hard
    poses from posterior maxima. Single-mask only. No contrast.
  * **Gate:** on synthetic linear-heterogeneity fixture, log evidence
    improves vs fixed-pose; subspace recovered. CryoBench ribosome run
    is at least as stable as fixed-pose PPCA.
  * **This is the main near-term deliverable.**

### M8 — Contrast (profile → marginalized)

  * **Scope:** Wrap `recovar/ppca/contrast_posterior.py` per pose. Add
    contrast renormalization that scales **both `μ` and `W`** (failure
    mode: scaling only `μ`).
  * **Gate:** contrast posterior parity vs `contrast_posterior.py` for
    profile and marginalized modes on random sufficient stats.

### M9 — Multimask + final PPCA postprocessing

  * **Scope:** Wire `pc_mask_assignment` through; reuse `postprocess.py`
    SVD/ortho; reuse `ppca_iterative_refitb.py` post-EM eigenvalue refit
    (post-EM ONLY — eigenvalue updates during EM are harmful).
    State.pkl restart support.

### M10 — CryoBench ribosome refinement-first eval

  * **Scope:** Compare fixed-pose / dense low-res / local-sparse against
    k-class high-res EM and existing RECOVAR PPCA without pose
    marginalization.
  * **Phase success:** at least as stable as fixed-pose PPCA, no
    consensus degradation, pose likelihood improved or stabilized.

---

## Anti-patterns (common agent failure modes)

  * **Writing a parallel implementation** instead of generalizing
    `recovar/ppca/ppca.py` in place. → Use the M2 reuse rule + parity test.
  * **Carrying `pinv` forward** from `_e_step_half_inner`. → Cholesky
    only in new code; flag legacy `pinv` for a separate parity-tested
    migration.
  * **Materializing `[N_images, N_rot, N_trans, *]` tensors.** → Memory
    explosion + CI failure. Block-only tensors. Memory split order:
    translation → rotation → image → contrast nodes.
  * **Updating eigenvalues during the EM loop.** → Documented harmful
    in `project_ppca_eigenval_update_during_anneal_harmful.md`. Refit
    is post-EM only (M9).
  * **Using RELION `tau2` as `W_prior`.** → Different prior. Use
    `estimate_hybrid_shell_prior_from_data` only.
  * **Putting eigenvalues into `z_prior_precision_diag` or
    `contrast_lambdas`.** → Breaks contrast renormalization. Eigenvalues
    live in `W`.
  * **Updating `W_prior` every iteration.** → Pose, contrast, noise, and
    regularization scale all drift together. Schedule:
    `prior_freeze_iters=3`, `recompute_prior_once_after_iter=5`.
  * **Replicating RELION iter-1 `--firstiter_cc`.** Hard winner-take-all
    CC, not Bayesian. Don't replicate.
  * **Skipping `enforce_relion_half_volume_x0_hermitian`** on the
    augmented path. → Hermitian-symmetry violation. Call after every
    half-volume accumulation.
  * **Inventing a new translation strategy.** → GEMM with shifted-image
    copies is the default (200× reuse, 45 ms). Stay on it.
  * **Inventing a new sparse layout.** → Reuse `LocalHypothesisLayout`
    + `bucket_local_hypothesis_layout`. Extend only if structurally
    impossible.
  * **Re-axing the codebase** for new code. → `[half_vol, q]` for `rhs`,
    `[half_vol, tri]` for `lhs_tri`. Keep it.
  * **Widening test tolerances or editing baselines.** → Hard-stop
    project rule.
  * **Modifying `recovar/em/heterogeneity.py`.** → Separate owners. Hands off.
  * **Reading `recovar/em/initial_model/*` or `claude/ppca-abinitio-v0`
    for math.** → Naming/logging only.
  * **Pushing to `dev` directly.** → Always feature branch + PR.

---

## Numerical contract

  * **Floats:** `complex64` images / projections / accumulators.
    `float64` available behind `--use-float64-scoring` (mirrors high-res
    EM).
  * **Cholesky everywhere.** No `pinv`. Symmetrize Hermitian matrices
    before factoring. Log-det from `2 · sum(log(real(diag(L))))`. Jitter
    only behind explicit `debug_jitter` flag; default zero; report
    Cholesky failures.
  * **Half-volume Hermitian:** call `enforce_relion_half_volume_x0_hermitian`
    on every half-volume accumulator before public exposure.
  * **Half-spectrum weights:** `1` at DC and Nyquist, `2` at interior
    columns for full-spectrum inner product. Scoring uses RELION
    unit-weight convention (`make_scoring_half_image_weights(...,
    relion_half_sum=True)`). The asymmetry is intentional — don't fix.
  * **FFT normalization:** `1/N` forward IFFT, `1` backward FFT (RECOVAR).
    Image rfft via numpy: forward `1`, irfft `1/N`. The legacy
    `_pcg_hard_mstep` bakes the gridding kernel `K(x) = sinc²(x/D)` into
    the operator — reuse unchanged.
  * **Volume frame:** RECOVAR `[z,y,x]`, RELION `[x,y,z]`. Conversion
    `vol_recovar = -np.transpose(vol_relion, (2,1,0))`. Use
    `recovar.utils.helpers.load_relion_volume(...)` for RELION MRCs.
  * **Parity escalation rule** (from `recovar/em/CLAUDE.md`): RELION GPU
    score parity is `~1e-4` arithmetic-level. Gaps at `1e-3`, pose flips,
    or multi-iteration drift are escalations.

---

## Workflow

### Provenance gate (run on every fresh shell)

```bash
cd $WORKDIR
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
pixi run python -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
print('ENV_OK')"
```

### Tests

  * Login node: `pixi run test-fast` and ppca_refinement unit tests for
    quick guard.
  * GPU / long: Slurm via `scripts/run_tests_parallel.sh`. Don't run the
    full long-test suite for interim parity work
    (`feedback_no_longtest_for_parity.md`); long-test is required only
    before opening the PR.

### Diagnostics (every iteration writes)

  * `W_prior` radial curve, floor fraction, raw / repaired shell totals,
    `|μ|²` fallback curve, reliable-shell mask, `median_ratio`,
    `α_prior`, `q_total`/`q_active`, `latent_prior_mode`,
    `pc_prior_mode`, mean / median `W_prior` inside mask, prior penalty
    `Σ |W|²/W_prior`, data-vs-prior ratio.
  * Posterior `pmax`, `nr_significant`, omitted-mass estimate, log
    evidence, pose-change histogram, noise spectrum.
  * Use the existing `EMProfileStats` struct for timing — don't invent
    a parallel timing channel.

### Scratch discipline

Any agent-created artifact (Slurm logs, parity dumps, debug volumes,
profiling traces) goes under `_agent_scratch/<run-name>/` or carries a
`SAFE_TO_DELETE` marker. See `~/.claude/CLAUDE.md` (root user CLAUDE.md).

### Memory hygiene

Update `MEMORY.md` only for *non-obvious, future-relevant* facts (a
hidden constraint, a load-bearing convention, a parity gotcha). Do not
record code patterns rediscoverable by reading the code.

---

## End-of-task delivery

  * Summary of what changed and which milestone(s) advanced.
  * Files modified.
  * Exact test commands run + Slurm job IDs / logs if applicable.
  * How to reproduce.
  * `git status` and `git diff --stat`.
  * If a milestone gate test passed, name it.
  * If you found a non-obvious gotcha, propose a memory entry.

---

## Pointers

  * Full math/scope plan: `docs/math/ppca_refine_plan_2026_05_01.md`.
  * Implementation playbook (full reuse table, code skeletons, line-level
    PCG audit, integration tests, CLI sketches):
    `docs/math/ppca_refine_implementation_2026_05_01.md`.
  * **Sequential implementation steps** (Phase 0 = verify rebase didn't
    break legacy PPCA; Phases 1–11 = milestones with explicit gates):
    `docs/math/ppca_refine_implementation_steps_2026_05_01.md`.
  * Parent EM agent guide: `recovar/em/CLAUDE.md`, `recovar/em/AGENTS.md`.
  * RELION-parity status: `docs/math/relion_parity_current_status_2026_04_25.md`,
    `docs/math/relion_parity_roadmap_2026_04_27.md`,
    `docs/math/relion_parity_agent_notes.md`.
  * Memory: project + feedback entries listed in
    `project_ppca_refine_pose_marginal_branch.md` (auto-memory).
  * Fixtures: `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_tiny_parity/`.
  * RELION source: `/scratch/gpfs/GILLES/mg6942/relion`.
