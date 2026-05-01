# Pose-Marginalized PPCA Refinement — Implementation Steps (2026-05-01)

Sequential, executable plan for the `recovar ppca-refine` project. Companion to
`docs/math/ppca_refine_plan_2026_05_01.md` (math/scope) and
`docs/math/ppca_refine_implementation_2026_05_01.md` (code skeletons +
reuse table). Operating contract: `recovar/em/ppca_refinement/CLAUDE.md`.

This file tells you **what to do, in what order, with what gates**. Don't
skip phases.

Branch: `claude/ppca-refine-pose-marginal` off
`origin/claude/relion-parity-local-search-fix`.
Worktree: `/scratch/gpfs/GILLES/mg6942/recovar_wt_ppca_refine_20260501_101028/`.

---

## Phase 0 — Verify the rebase didn't break legacy PPCA (DO THIS FIRST)

The new branch was branched off `claude/relion-parity-local-search-fix`,
which was itself recently rebased and now carries both
`recovar/em/dense_single_volume/` and `recovar/ppca/`. **Before any new
code, prove the existing PPCA path still works.** If it doesn't, fix the
breakage on this branch (or escalate) — don't build pose-marginal PPCA on
top of a broken base.

There is no `recovar/ppca/CLAUDE.md`. The verification ladder below is
derived from what's actually in the repo: the public surface in
`recovar/ppca/__init__.py`, the existing unit tests, and the existing
end-to-end pipeline-comparison driver
`recovar/ppca/compare_covariance_vs_ppca_pipeline.py`.

### 0.1 Provenance gate (60 s)

```bash
cd /scratch/gpfs/GILLES/mg6942/recovar_wt_ppca_refine_20260501_101028
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
pixi run python -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
print('ENV_OK')"
```

Stop on failure.

### 0.2 PPCA import + symbol surface (5 s, login node)

```bash
pixi run python -m pytest -q tests/unit/test_ppca_compatibility.py
```

Two tests: `EM`, `EM_step_half`, `output.mkdir_safe`, `linalg.batch_linear_solver`
must be importable; `regularization.batch_average_over_shells` must produce
the expected shape. **First failure mode to look for** if the rebase broke
something — function rename or moved symbol.

### 0.3 PPCA unit tests, CPU (≤ 5 min)

```bash
pixi run python -m pytest -q \
    tests/unit/test_ppca.py \
    tests/unit/test_ppca_compatibility.py \
    tests/unit/test_ppca_prior_estimation.py \
    tests/unit/test_ppca_multimask_contrast.py
```

Covers `_e_step_half_inner` shape contracts, `E_M_step_batch_half`
accumulator shapes (the axis order new code must match), Gauss–Legendre
contrast quadrature, single-mask vs multimask PCG equivalence,
`pc_mask_assignment` defaults, prior-estimation helpers. **All must pass
before Phase 1.** Any failure here is a rebase regression — fix it first.

### 0.4 PPCA multimask synthetic test, GPU (~15 min on 1 GPU)

This is the most useful smoke test for "PPCA still works end-to-end" because
it actually runs `EM(...)` on simulated Ribosembly data with multimask:

```bash
nvidia-smi  # confirm a GPU is idle on the login node, OR submit via Slurm
pixi run python -m pytest -q \
    tests/unit/test_ppca_multimask_synthetic.py \
    --run-gpu -k "test_full_em_multimask_runs or test_full_em_single_mask_baseline"
```

The two `test_full_em_*` tests run a real PPCA EM on a downsampled
Ribosembly fixture. **Pass = legacy PPCA EM is bit-functional after the
rebase.**

If the login-node GPU is busy, submit via Slurm using the template in
`recovar/em/CLAUDE.md` (single GPU, `--account=amits`, `--partition=cryoem`).

### 0.5 EM-fast guard (2 min, login node)

```bash
pixi run test-em-fast-guard
```

Confirms the EM/dense_single_volume path the new engines will reuse is
healthy after rebase (bound-imports, half-spectrum projection, dense
backprojection).

### 0.6 End-to-end PPCA pipeline run (optional, GPU, ~30 min)

This is a "bigger" smoke test — runs the existing CryoBench-style
PPCA-vs-covariance pipeline driver on a small synthetic dataset and scores
both. Useful if Phase 0.4 passes but you want to confirm the full
postprocess + embedding stack still works.

```bash
pixi run python -m recovar.ppca.compare_covariance_vs_ppca_pipeline \
    --dataset igg-1d \
    --grid-size 64 \
    --n-images 5000 \
    --zdim 4 \
    --ppca-em-iters 5 \
    --results-root /scratch/gpfs/GILLES/mg6942/_agent_scratch/ppca_phase0_smoke
```

Place the run under `_agent_scratch/` so it's safe to delete. Check
`scores.json` exists and `ppca` mode produced a non-empty result dir.
**Skip this if Phase 0.4 passes and you trust the public surface tests.**

### 0.7 Pin the verified state

If everything in 0.1–0.5 passed:

  * Make the **first commit on this branch** a tiny verification artifact:
    add a single empty file `.phase0_verified` (or a one-line note in a
    new `recovar/em/ppca_refinement/__init__.py` saying "package created
    after Phase 0 verification on 2026-05-01"). This pins the green state
    and gives you a known-good base commit to bisect against if a later
    failure is a rebase issue surfacing late.
  * Capture the test-output line counts and any flaky-skip messages in
    the commit body so a future agent can compare.

If anything failed: stop, read the actual error, decide whether it's
(a) a rebase-induced regression (fix here, then proceed) or
(b) pre-existing on the source branch (open an issue, don't paper over).
**Do not start Phase 1 with red tests.**

---

## Phase 1 — Repo audit + naming map + PCPriorConfig (M0)

Goal: zero-risk plumbing commit that other agents can rebase onto cleanly.

  1. Add `recovar/em/ppca_refinement/__init__.py` (empty if not created in
     Phase 0.7).
  2. Add `recovar/ppca/pc_prior_config.py`:
     ```python
     @dataclass
     class PCPriorConfig:
         latent_prior_mode: str = "identity"
         pc_prior_mode: str = "hybrid_shell"
         prior_scale: float = 1.0           # α_prior
         variance_floor: float = 1e-8       # τ_floor
         use_q_total_for_division: bool = True
         smooth_shell_prior: bool = True
         prior_freeze_iters: int = 3
         recompute_once_after_iter: int | None = 5
         allow_every_iter_prior_update: bool = False
     ```
     Plus a `to_dict()` for diagnostics serialization.
  3. Add a single short module docstring to `recovar/ppca/pose_marginal.py`,
     `augmented_mstep.py`, `pose_accumulators.py` — empty placeholders so
     future commits show a clean diff.
  4. Wire `PCPriorConfig.to_dict()` into the existing PPCA per-iteration
     diagnostics writer (find it via
     `grep -nE "iteration_data|diagnostics" recovar/ppca/ppca.py` — the
     `EM(...)` function returns `iteration_data` if `return_iteration_data=True`).
  5. Tests: extend `tests/unit/test_ppca.py` (or create a tiny new file)
     with a `test_pc_prior_config_defaults` checking the defaults are the
     ones documented in §7 of CLAUDE.md.

**Gate:** all Phase 0.3 tests still pass.
**Commit:** `Add PCPriorConfig and ppca_refinement package skeleton`.

---

## Phase 2 — Per-pose math, no contrast (M1)

Goal: implement and parity-test the per-pose score/moment function. Pure
JAX, no I/O, no dataset, no engine. Should be < 200 lines + tests.

  1. Implement
     `recovar/ppca/pose_marginal.py::compute_ppca_pose_scores_and_moments_no_contrast`
     per the signature in CLAUDE.md §"Required public signatures."
  2. Implement helper
     `recovar/ppca/pose_marginal.py::_pack_upper_tri` for the
     `[..., q+1, q+1] → [..., (q+1)(q+2)/2]` packing. Reuse
     `recovar/ppca/ppca.py::_tri_size` for the triangle size.
  3. Tests in `tests/unit/ppca_refinement/test_pose_marginal.py` (create
     the directory):
     * `test_q_zero_reduction`: with `q=0`, the score equals the homogeneous
       pose score from `recovar/em/dense_single_volume/helpers/scoring.py`
       up to a pose-independent additive constant.
     * `test_w_zero_reduction`: with `q≥1`, `Hzz=0`, `g_zx=0`, `h_zm=0` →
       same as `q=0` up to additive constant.
     * `test_brute_force_latent_integral_tiny_q`: for `q ∈ {1, 2}` and
       random PD `Hzz`, compute the score by Gauss–Hermite integration
       over `z` and compare to the analytic Cholesky expression. Use
       `scipy.stats.multivariate_normal` for ground truth.
     * `test_basis_rotation_invariance`: replace `W ← W Q` for orthogonal
       `Q`, recompute `Hzz`, `g_zx`, `h_zm`, score must be unchanged.
     * `test_cholesky_symmetrization`: feed deliberately non-Hermitian
       `Hzz`, verify the function symmetrizes and produces the same result
       as the symmetrized input.
     * `test_jit_compiles`: wrap with `@jax.jit` and call twice; second
       call must be cached.
     * `test_no_pinv`: `grep` the file source ensures `pinv` does not
       appear (regression guard against agents copying from
       `_e_step_half_inner`).
  4. Numerical contract in code:
     `M = jnp.eye(q) + 0.5 * (Hzz + jnp.swapaxes(Hzz.conj(), -1, -2))`,
     `L = jnp.linalg.cholesky(M)`,
     `logdetM = 2 * jnp.sum(jnp.log(jnp.real(jnp.diagonal(L, axis1=-2, axis2=-1))), axis=-1)`,
     `S_z = jax.scipy.linalg.cho_solve((L, True), jnp.eye(q))`,
     `b = g_zx - h_zm`,
     `M_inv_b = jax.scipy.linalg.cho_solve((L, True), b[..., None])[..., 0]`,
     `score = -0.5 * (rho - jnp.sum(b.conj() * M_inv_b, -1).real + logdetM)`.

**Gate:** all six new unit tests pass. `pixi run python -m pytest -q
tests/unit/ppca_refinement/`. Phase 0.3 tests still pass.
**Commit:** `M1: per-pose PPCA score/moments, no contrast`.

---

## Phase 3 — Augmented M-step, q→q+1 PCG generalization (M2)

Goal: generalize `_pcg_hard_mstep` in place with `n_components` arg, add a
thin wrapper. **The hard part is the parity test, not the code.**

  1. **First**: write the backwards-compat regression test BEFORE touching
     `ppca.py`. Pin a tiny-fixture run by importing the simplest
     end-to-end EM call from `tests/unit/test_ppca_multimask_synthetic.py`,
     reducing it to `q=2` and `EM_iter=2`, and serializing the resulting
     `(U, S, W, expected_zs, second_moment_zs)` via `pickle` to
     `tests/baselines/ppca_refinement/legacy_pcg_q2.pkl`. Commit that
     baseline. The test asserts the post-generalization run reproduces it
     to `np.allclose(rtol=1e-6, atol=1e-7)`.
  2. Re-read the line audit in
     `docs/math/ppca_refine_implementation_2026_05_01.md` ("Augmented
     M-step"). Run `grep -nE "jnp\.arange\(q\)|range\(0, q,|reshape\(q,"
     recovar/ppca/ppca.py` to verify the line numbers haven't drifted; if
     they have, update the audit doc.
  3. Refactor `_pcg_hard_mstep` to take `n_components: int = q` (default
     keeps existing call sites untouched). Then thread `n_components`
     through:
     `_mstep_AL_solve_fourier`, `_mstep_batched_rfft`,
     `_mstep_batched_irfft`, `_mstep_A_mul_fourier`, `scatter`, `gather`.
     Replace `q` with `n_components` in array shapes and
     `jnp.arange(q)` indexers. Add `assert reg_diag.shape[-1] == n_components`
     near the top.
  4. Add the mean-component branch: `reg_diag` becomes
     `reg_diag_aug = stack([mean_reg_diag, W_reg_diag], axis=-1)`. The
     existing path passes `W_reg_diag` only (size q); the new path passes
     size q+1 with component 0 being the mean prior. **Make sure both
     paths still hit the same gridding-kernel `K(x) = sinc²(x/D)` operator
     line.**
  5. Implement `recovar/ppca/augmented_mstep.py::solve_augmented_ppca_mstep`
     and `recovar/ppca/pose_accumulators.py::AugmentedPPCAStats`.
  6. Tests in `tests/unit/ppca_refinement/test_augmented_mstep.py`:
     * `test_legacy_q2_regression` (loads the pinned pickle from step 1).
     * `test_fixed_mean_reduction_matches_legacy`: build a problem where
       `mean_reg_diag = +∞` (huge `1/W_prior`) effectively pins `μ`; the
       remaining solve must match the legacy `_pcg_hard_mstep(q=q_old)`
       output to `np.allclose(rtol=1e-5)`.
     * `test_free_mean_toy_vs_dense_normal_eqs`: tiny problem (e.g.
       `D=8, q=2`), build the dense `(q+1)·N` x `(q+1)·N` normal-equations
       matrix in NumPy via `np.linalg.solve`, compare PCG output to
       `atol=1e-5`.
     * `test_W_prior_penalty_matches`: synthesize known `W`, evaluate the
       prior term `Σ |W_k(ξ)|² / W_prior(ξ,k)` directly and via the PCG
       residual; must match.
     * `test_mean_component_uses_mean_prior`: with `W_prior` huge (loadings
       free) and `mean_reg_diag` non-trivial, the recovered μ component
       must equal the homogeneous Wiener solve from
       `recovar.reconstruction.regularization`.
     * `test_multimask_pc_assignment_with_mean`: ensure
       `pc_mask_assignment = [mean_mask_idx, *W_assignment]` zeroes
       outside support per-component. Mirror the existing
       `test_multi_mask_pcg_localizes` test pattern but with `q+1`.
  7. **Numerical guardrails:** keep complex64 axes
     `[half_vol, n_components]` for `rhs` and `[half_vol, tri(n_components)]`
     for `lhs_tri`. Do NOT touch `_compute_gridding_kernel` or any
     half-volume Hermitian path. The wrapper's job is to call the
     generalized `_pcg_hard_mstep` with a stacked `reg_diag_aug` and split
     the returned `[n_components, *vs]` array into `(μ_real, W_real)`.

**Gate:** all Phase 0.3 + Phase 2 + Phase 3 tests pass. The pinned legacy
fixture matches.
**Commit:** `M2: generalize PCG to q+1 components for augmented [μ,W] M-step`
(may split into two commits: the backwards-compat refactor, then the
augmented wrapper).

---

## Phase 4 — Fixed-pose driver `recovar ppca-refine --pose-mode fixed` (M3)

Goal: the first user-visible CLI entry point. Boring. Reliable.

  1. `recovar/em/ppca_refinement/state.py`: a *minimal* state for fixed
     mode — `(mu, W_half, W_prior, masks, contrast_params)`. Don't build
     the full `PoseMarginalPPCAEMState` yet; that lands at M5.
  2. `recovar/em/ppca_refinement/iterations.py`: a single
     `run_fixed_pose_ppca_refine(dataset, init_state, opts)` that calls
     `recovar.ppca.ppca.EM(...)` directly (using the now-generalized PCG
     for `q+1` if `opts.refine_mean=True`, else falling back to legacy
     `q` PCG for parity sanity).
  3. `recovar/em/ppca_refinement/cli.py`: argparse-based CLI registered
     under the `recovar` console-script entry. Mirror existing
     `recovar/commands/*.py` style.
  4. End-to-end test
     `tests/unit/ppca_refinement/test_fixed_pose_driver.py`:
     run on the same Ribosembly fixture used in Phase 0.4 with `EM_iter=2`,
     `q=2`, fixed poses, `refine_mean=False`. Output must match a direct
     `recovar.ppca.ppca.EM(...)` call to `np.allclose(rtol=1e-6)`.
  5. Add a smoke test that exercises `refine_mean=True` (mean component
     active in the augmented PCG) on the same fixture and asserts the
     mean updates non-trivially without crashing.

**Gate:** new CLI runs. End-to-end parity vs legacy `EM(...)` holds when
`refine_mean=False`. Smoke test passes when `refine_mean=True`.
**Commit:** `M3: recovar ppca-refine --pose-mode fixed`.

---

## Phase 5 — Dense pose-marginalized E-step (M4)

Goal: dense two-pass engine that scores all (rotation, translation) pairs
exactly. Reuse k-class engine machinery; replace the score function with
the M1 augmented one.

  1. Read
     `recovar/em/dense_single_volume/dense_k_class_engine.py::run_dense_k_class_em_native`
     in full. The augmented PPCA dense engine has the same shape: replace
     "K class templates" with "p = q+1 augmented components."
  2. `recovar/em/ppca_refinement/dense_engine.py::dense_pose_ppca_E_step_and_stats`
     per the playbook in `docs/math/ppca_refine_implementation_2026_05_01.md`.
     Reuse:
     * `helpers/projection.py::project_half_spectrum` for `proj_mu`,
       `proj_W` (loop or `vmap` over the q axis).
     * Build `proj_aug = jnp.concatenate([proj_mu[:, None, :], proj_W], axis=1)`.
     * Reuse `helpers/significance.py::_compute_significance_batched` for
       pass-1 logsumexp + best-pose tracking — it's already block-aware.
     * Reuse `helpers/backprojection.py::accumulate_adjoint_pair` per
       augmented component for RHS accumulation in pass-2.
  3. Unit tests in `tests/unit/ppca_refinement/test_dense_engine.py`:
     * `test_q_zero_matches_em_engine_run_em`: with `W=zeros`, `q=0` (no
       augmented components beyond mean), the dense PPCA engine output
       (rhs, lhs_tri, log_likelihood) must match
       `recovar/em/dense_single_volume/em_engine.py::run_em` on a tiny
       fixture to `atol=1e-5`.
     * `test_dense_matches_brute_force`: tiny image (D=8), tiny grid
       (R=4 rotations, T=3 shifts), `q=2`. Enumerate all (R,T) pairs in
       Python, compute pose scores by direct `_e_step_half_inner` per
       pose, and compare logZ + γ-weighted alpha_aug, G_aug_tri to
       blockwise dense output. `atol=1e-5`.
     * `test_fixed_pose_limit_matches_m3`: when the rotation grid is a
       single hypothesis (current pose), output must equal Phase 4
       `run_fixed_pose_ppca_refine` stats.
     * `test_no_full_posterior_materialization`: set
       `RECOVAR_DEBUG_ASSERT_NO_FULL_POSTERIORS=1` (env var; check
       relion-parity branch for the assertion mechanism); run dense
       engine; assert no `[N_images, N_rot, N_trans]` allocation.
  4. **Memory/perf invariant:** the only allowed inside-block tensors are
     `score [B,T,R]`, `alpha_aug [B,T,R,p]`, `G_aug_tri [B,T,R,p(p+1)/2]`,
     `D [B,T,R,p]`, `K_aug_tri [B,R,p(p+1)/2]`. If your test suite needs
     a brute-force ground truth, build it in Python with explicit loops —
     the JAX engine must use blockwise GEMMs.

**Gate:** all four new tests pass. Phase 0–3 tests still pass.
**Commit:** `M4: dense pose-marginalized PPCA E-step (no contrast)`.

---

## Phase 6 — Dense driver `--pose-mode dense --engine dense` (M5)

Goal: full EM loop wiring with halfsets, noise updates, prior recomputation
schedule.

  1. Promote the M3 `state.py` to the full `PoseMarginalPPCAEMState`
     dataclass per CLAUDE.md.
  2. `recovar/em/ppca_refinement/iterations.py::run_pose_marginal_ppca_refine`:
     halfset combine for scoring uses
     `recovar/em/dense_single_volume/helpers/convergence.py::RefinementState`
     and the existing combine path.
  3. Wire `estimate_hybrid_shell_prior_from_data` into the EM loop with
     the §7.4 schedule (`prior_freeze_iters=3`,
     `recompute_once_after_iter=5`, `allow_every_iter_prior_update=False`).
  4. Add diagnostics dumping per CLAUDE.md §17.
  5. Integration test (slow, GPU): synthetic linear-heterogeneity dataset
     (`q=4`, mild pose perturbation), `EM_iter=10`. Assert (a) `W` subspace
     recovers GT subspace to `subspace_angle ≤ 30°`, (b) log-evidence
     monotonically increases (within numerical noise).

**Gate:** integration test passes on Slurm GPU job. Phase 0–5 unit tests
still pass.
**Commit:** `M5: dense pose-marginalized PPCA driver`.

---

## Phase 7 — Sparse / local pose-marginalized E-step (M6)

Goal: replace exhaustive enumeration with significance-pruned local search.
Reuse `LocalHypothesisLayout` + bucketing.

  1. Read
     `recovar/em/dense_single_volume/local_k_class_engine.py::run_local_k_class_em_native`
     in full. Same template-bank substitution.
  2. `recovar/em/ppca_refinement/sparse_engine.py::sparse_pose_ppca_E_step_and_stats`.
     Mode A (coarse-to-fine) and Mode B (local-around-current) share the
     same score function and stats accumulator.
  3. **Critical test:** sparse with all hypotheses retained must match
     dense bit-for-bit (within float32 fused-multiply tolerance) on a tiny
     image batch and pose grid. This is the M6 gate.
  4. Add omitted-mass diagnostic per CLAUDE.md §17.
  5. **Watch:** do NOT depend on `use_global_significant_support`
     (per `project_use_global_significant_support_path.md`).

**Gate:** unpruned-sparse-equals-dense parity test passes; pruned support
normalization matches k-class convention; omitted-mass written.
**Commit:** `M6: sparse pose-marginalized PPCA E-step`.

---

## Phase 8 — Main local-pose driver `--pose-mode local --engine sparse` (M7)

Goal: the deliverable. Local pose refinement around current poses.

  1. Wire Mode B in the EM loop. Update hard poses from posterior maxima
     each iteration.
  2. Reuse `--reuse-kclass-pose-schedule` to import the angular/shift
     schedule from k-class.
  3. Integration test (slow, GPU) on synthetic linear-heterogeneity with
     mildly perturbed initial poses: log-evidence improves vs Phase 6
     fixed-grid; subspace recovered.
  4. CryoBench ribosome run from reliable initial poses. Compare to
     fixed-pose PPCA from Phase 4. Phase success: at least as stable; no
     consensus degradation; pose likelihood improved or stabilized.

**Gate:** CryoBench run meets the phase-success criterion.
**Commit:** `M7: recovar ppca-refine --pose-mode local --engine sparse`.

---

## Phase 9 — Contrast (M8)

Goal: add profile contrast, then marginalized contrast. Wrap
`recovar/ppca/contrast_posterior.py`.

  1. `recovar/ppca/pose_marginal.py::compute_ppca_pose_scores_and_moments_with_contrast`
     dispatching to profile or marginalize.
  2. Build `alpha_aug`, `G_aug_tri` from the moments returned by
     `contrast_posterior.solve_latent_posterior` (`mean_c`, `second_c`,
     `mean_cz`, `mean_c2z`, `second_c2zz`, `marginal_ll`).
  3. Renormalization that scales **both** μ AND W (failure mode: scaling
     only μ).
  4. Tests: contrast posterior parity vs `contrast_posterior.py` directly
     for both modes on random sufficient stats.

**Gate:** parity tests pass. Phase 8 main run continues to converge with
contrast on.
**Commit:** `M8: contrast (profile + marginalized) for pose-marginal PPCA`.

---

## Phase 10 — Multimask + final postprocessing (M9)

Goal: production-ready. Multimask, post-EM SVD/ortho, post-EM eigenvalue
refit, state.pkl restart.

  1. Wire `pc_mask_assignment` end-to-end: dataset → engine → augmented
     M-step. Mean component default mask = standard solvent mask.
  2. Reuse `postprocess.py::_orthonormalize_W_to_basis_multimask`.
  3. Reuse `ppca_iterative_refitb.py` for **post-EM** eigenvalue refit
     ONLY (eigenvalues during EM are harmful).
  4. Add `state.pkl` save/load.

**Commit:** `M9: multimask + final PPCA postprocessing`.

---

## Phase 11 — CryoBench eval (M10)

  1. Run fixed-pose / dense low-res / sparse-local on CryoBench ribosome
     from reliable initial poses.
  2. Compare to k-class high-res EM and existing RECOVAR PPCA (no pose
     marginalization).
  3. Track: mean FSC, posterior pmax, nr_significant, pose changes per
     iter, log evidence, noise spectrum, contrast distribution, W
     singular values, halfset W subspace angles, runtime, GPU memory.
  4. Produce a side-by-side report table for the PR description.

**Phase success:** at least as stable as fixed-pose PPCA; no consensus
degradation; pose likelihood improved or stabilized.

---

## Pre-PR checklist (before opening PR to dev)

  1. Rebase on dev: `git fetch origin && git rebase origin/dev`. Resolve
     conflicts; re-run Phase 0.3.
  2. Run the full long-test parallel suite:
     `./scripts/run_tests_parallel.sh long-test`. Wait for all groups
     including metrics-spa, metrics-et, outliers-long.
  3. Extract regression tables: `pixi run python scripts/extract_regression_tables.py`.
  4. PR body must include the quality and performance comparison tables
     (project root `CLAUDE.md` mandates this).
  5. PR title under 70 chars; body summary + test plan checklist; Slurm
     job IDs for traceability.
  6. PR target: `dev`. Never push to `dev` directly.

---

## Open questions / decisions to escalate to the user, not auto-decide

  * **W initialization for M5+ when `W` is not supplied.** Default per
    CLAUDE.md is "run existing fixed-pose PPCA from current poses + mean."
    Confirm this is the desired default before M5.
  * **Halfset combine rule.** The high-res EM convention uses filtered
    averaging via `RefinementState`. Confirm this is the intended scoring
    volume, not "use halfset 1 only."
  * **Mean prior source for the augmented PCG.** §7.2 of CLAUDE.md says
    `mean_reg_diag` comes from "the homogeneous reconstruction prior."
    Identify the concrete provider (likely the per-shell tau2 from the
    half-volume reconstruction) and confirm before M2 lands.
  * **CryoBench dataset handle.** M7/M10 needs a concrete dataset
    selection. Confirm with user (likely Ribosembly or IgG-1D from
    `compare_covariance_vs_ppca_pipeline.py`).

Surface these in the M2/M5/M7 PR descriptions if not resolved earlier.
