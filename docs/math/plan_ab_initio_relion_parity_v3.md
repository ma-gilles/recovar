# Plan v3 — Exact RELION InitialModel / VDAM parity

*2026-04-24, branch `claude/relion-parity-flag-audit` at commit `84440d7e`.*

## Status and scope

This plan supersedes `plan_ab_initio_relion_parity.md` (v1, commit `9f7b9e18`,
Apr 16) which has since been removed from the tree. The v1 plan was
directionally right (separate InitialModel module, reject `SGDState`, pseudo-
halfsets) but structurally wrong: it assumed two physical recovar halfset
datasets, inverted the `tau2_fudge` schedule, hard-coded resolution caps that
RELION no longer applies during VDAM, and proposed to reuse
`_refine_relion_mode` — an auto-refine half-map FSC loop with pad 2 — as the
control plane. See the review notes attached to this plan for the full
findings list.

The implementation target of v3 is the **GUI-generated InitialModel command**,
byte-identical to what `pipeline_jobs.cpp::getCommandsInimodelJob` assembles.
Bare-CLI fallback parity is **not** a goal; it differs in `--iter` default,
`grad_ini_resol` / `grad_fin_resol` legacy constants, and a few other knobs
that the GUI always overrides.

No code exists yet. The `initial_model_vdam.py` / `initialmodel_bind.cpp` /
`run_ab_initio.py` files referenced in the earlier handoff were never
committed and were explicitly trimmed from this branch by
`ac4731d8 Trim parity branch to minimal workflow` (Apr 21). v3 starts from
zero code and treats the existing `docs/math/relion_parity_deep_dive.md` and
`recovar/em/dense_single_volume/` auto-refine stack as *reference material*,
not *drop-in dependencies*.

## 0. Exact GUI InitialModel command (parity target)

From `relion/src/pipeline_jobs.cpp::getCommandsInimodelJob` (lines 3428–3613):

```
relion_refine \
  --o <dir>/run \
  --iter <nr_iter> \
  --grad --denovo_3dref \
  --i <particles.star> \
  --ctf [--ctf_intact_first_peak] \
  --K <nr_classes> \
  --sym <C1 during opt; apply real sym later> \
  --flatten_solvent \
  --zero_mask \
  [--dont_combine_weights_via_disc] \
  [--no_parallel_disc_io] \
  [--preread_images | --scratch_dir <dir>] \
  --pool <nr_pool> \
  --pad 1 \
  --particle_diameter <A> \
  --oversampling 1 \
  --healpix_order 1 \
  --offset_range 6 \
  --offset_step 2 \
  --auto_sampling \
  --tau2_fudge <T> \
  --j <nr_threads> \
  [--gpu "..."]
```

Followed by `relion_align_symmetry --i <last_model.star> --o
initial_model.mrc [--sym <G> --apply_sym --select_largest_class]` which
ALIGNs the final VDAM reference to the user's symmetry group and picks the
largest class as `initial_model.mrc`.

Hard rules, byte-identical to the GUI:

- MPI is rejected at command-assembly time: `pipeline_jobs.cpp:3435` returns
  "Gradient refinement is not supported together with MPI."
- `--split_random_halves` is **never added**.
- `--pad 1` (not 2; `_refine_relion_mode` uses 2 and is therefore not reusable
  as-is).
- GUI defaults that parity must reproduce: `nr_iter=200`, `K=1`, `T=4` (i.e.
  `--tau2_fudge 4`), `sym=C1` during optimisation, mask diameter in Å from
  the GUI form, `--oversampling 1 --healpix_order 1 --offset_range 6
  --offset_step 2 --auto_sampling`.

These constraints are asserted by a snapshot test (§ G below) so any silent
drift toward auto-refine defaults fails loudly.

## 1. Why the current branch cannot be reused as-is

The dense-path refinement loop at commit `84440d7e`
(`recovar/em/dense_single_volume/iteration_loop.py`,
`em_engine.py`, `local_em_engine.py`) is engineered for RELION **auto-refine**
parity:

- two physical half-sets with independent E/M per half,
- FSC-to-join at every iteration,
- `padding_factor=2` on the M-step reconstruction,
- convergence controller that counts iterations without resolution gain under
  `MAX_NR_ITER_WO_RESOL_GAIN`,
- `updateCurrentResolution` assumes `do_split_random_halves && do_auto_refine`
  for the high-res tail check
  (`ml_optimiser.cpp:5755`).

VDAM needs none of that and actively breaks most of it: there is a single
reference per class, gradient moment arrays (`Igrad1`, `Igrad2`) that live
through the whole run, pseudo-halfset splitting that is an *internal*
bookkeeping trick (not two physical datasets), `padding_factor=1`, and a
different convergence rule (`MAX_NR_ITER_WO_RESOL_GAIN_GRAD` + an EM tail
`grad_em_iters`).

Conclusion: reuse the low-level kernels (projector, CTF-applied projection,
slice-based GEMM, backprojection primitives, half-spectrum layouts) behind a
**new** VDAM controller. Do **not** extend `_refine_relion_mode`.

## 2. Ground-truth schedules (from RELION 5.0 source)

All schedules below are read off `relion/src/ml_optimiser.cpp` at the target
checkout (`/scratch/gpfs/GILLES/mg6942/relion/src`). Each recovar function
implementing one of these will have a `relion_bind` parity test (§ F).

### 2.1 Phase fractions (`parseInitial`, 994–996)

```
grad_ini_iter       = nr_iter * grad_ini_frac        # grad_ini_frac default 0.3
grad_fin_iter       = nr_iter * grad_fin_frac        # grad_fin_frac default 0.2
grad_inbetween_iter = nr_iter - grad_ini_iter - grad_fin_iter
```

With GUI defaults (`nr_iter=200`, `grad_ini_frac=0.3`, `grad_fin_frac=0.2`):
`grad_ini_iter=60`, `grad_inbetween_iter=100`, `grad_fin_iter=40`.

Integer arithmetic is C++ `int` truncation (`RFLOAT*RFLOAT` assigned to
`int`). The recovar implementation must mimic this exactly; a snapshot test
pins the 200/0.3/0.2 case and a boundary case where `ini_frac + fin_frac >
0.9` triggers the renormalisation branch at
`ml_optimiser.cpp:416`.

### 2.2 Subset size (`updateSubsetSize`, 10212–10276)

For `gradient_refine && !do_auto_refine`:

```
if iter <  grad_ini_iter:                   subset = grad_ini_subset_size
elif iter < grad_ini_iter + grad_inbetween: subset = ini + round(frac * (fin - ini))
else:                                       subset = grad_fin_subset_size
```

where `frac = (iter - grad_ini_iter) / grad_inbetween_iter`. Then:

```
if not do_grad or nr_iter - iter < grad_em_iters or subset >= nr_particles
    or grad_has_converged:
        subset = -1   # means "all particles"
```

Auto-default resolution for `grad_ini_subset_size==-1 && is_3d_model`
(`ml_optimiser.cpp:2663`):

```
grad_ini_subset_size = clamp(round(N * 0.005), 200, 5000)   # 3D initial model
grad_fin_subset_size = clamp(round(N * 0.1),   1000, 50000)
```

so for the 500-particle 64px fixture under
`/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run/`:
`grad_ini_subset_size = clamp(2.5, 200, 5000) = 200`,
`grad_fin_subset_size = clamp(50, 1000, 50000) = 1000 > N` → fallback to -1.
This test case is also what the snapshot test pins.

### 2.3 Step size (`updateStepSize`, 10278–10325)

For 3D initial model with `_stepsize <= 0` and empty scheme:

```
_stepsize = 0.5                       # 3D initial model default
_scheme   = f"{0.9 / _stepsize}-step" # i.e. "1.8-step"
```

then `"<inflate>-step"` evaluates:

```
x      = iter
a      = grad_inbetween_iter / 2
b      = grad_ini_iter
scale  = 1. / (10 ** ((x - b - a/2) / (a/4)) + 1)    # sigmoid
stepsize = _stepsize * inflate * scale + _stepsize * (1 - scale)
         = 0.9 * scale + 0.5 * (1 - scale)           # decays 0.9 → 0.5
```

### 2.4 `tau2_fudge` schedule (`updateTau2Fudge`, 10327–10379)

For 3D initial model with `_fudge <= 0` and empty scheme:

```
_fudge  = 4                             # 3D initial model default
_scheme = f"{_fudge/1.}-step"           # i.e. "4-step"
```

then:

```
deflate = 4
scale   = same sigmoid with a = grad_inbetween_iter/4, b = grad_ini_iter
tau2_fudge = (_fudge/deflate)*scale + _fudge*(1-scale)
           = 1 * scale + 4 * (1 - scale)             # grows 1 → 4
```

**v1 had this inverted (16→4). Fix pinned by § F parity test.** At GUI
defaults with `--tau2_fudge 4`, the user-supplied `_fudge=4` takes the
`_fudge > 0` branch, so `_fudge=4` and `_scheme` is still empty — the
same "4-step" default applies, yielding the 1→4 trajectory.

### 2.5 Resolution (`updateCurrentResolution`, 5721 + `updateImageSizeAndResolutionPointers`, 5826)

**Key finding:** `grad_ini_resol` / `grad_fin_resol` are **not** applied
during VDAM E-step. The only code path that reads them is the model STAR
round-trip at `ml_optimiser.cpp:1365` (fallback 35 / 15 when the field is
absent from a continuation `_optimiser.star`). Line 10247 is a stale
comment. In an initial run the resolution cap is driven purely by
`updateCurrentResolution` the same way it is for EM:

- iter 0 or (iter 1 + `do_firstiter_cc`): `maxres = round(N * px / ini_high)`.
  `ini_high` for denovo (`do_average_unaligned`) is set at
  `ml_optimiser.cpp:2463` to `round(0.07 * ori_size)` pixels inverted to Å,
  i.e. `ini_high = ori_size * pixel_size / round(0.07 * ori_size)`
  (≈ 136 Å on the 64×8.5 Å fixture).
- iter ≥ 2: walk `data_vs_prior_class` shell-by-shell until it drops below 1,
  subtract 1 for safety, floor at `minres_map`. No `do_split_random_halves`
  high-res tail check for InitialModel.
- `updateImageSizeAndResolutionPointers` then expands maxres by 25% of
  `ori_size/2` when `ave_Pmax > 0.1 && has_high_fsc_at_limit`, else by
  `incr_size` (default 10 shells).

**v3 does not hard-code 35 Å / 15 Å and does not hard-code 60 Å ini_high.**
The only constants set explicitly are `incr_size = 10` (RELION CLI default)
and the 7% ini_high rule for denovo.

### 2.6 Pseudo-halfset activation (`parseInitial`, 1920)

```
grad_pseudo_halfsets = gradient_refine && !do_split_random_halves
```

Always true for the GUI InitialModel path.

### 2.7 EM tail (`updateSubsetSize` + `iterate()` in ml_optimiser.cpp:3462)

```
do_grad = !(has_converged || iter > nr_iter - grad_em_iters)
grad_pseudo_halfsets = do_grad
```

With `--grad_em_iters 0` (CLI default, also what the GUI passes for
InitialModel), the last iteration drops gradient mode and runs a single EM
pass. v3 must reproduce this exactly: at `iter == nr_iter` with
`grad_em_iters==0`, switch to an EM M-step with `subset=-1` and
`pseudo_halfsets=False`.

## 3. State (`InitialModelState`)

Single-dataset state; pseudo-halfset slots are internal bookkeeping, not two
recovar halfset objects.

```python
@equinox.module
class InitialModelState:
    iter:            int                   # 0-indexed pre-update; iter == nr_iter is EM tail
    nr_iter:         int                   # 200 by GUI default
    K:               int                   # nr_classes, 1 by GUI default
    Iref:            Float[K, N, N, N]     # real-space reference(s)
    Igrad1:          Complex[2*K_if_ps, Nx, Ny, Nz//2+1]  # first-moment, two slots/class when pseudo-halfsets
    Igrad2:          Complex[K, Nx, Ny, Nz//2+1]          # second-moment; init constant
    sigma2_noise:    Float[G, N//2+1]      # per-optics-group
    tau2_class:      Float[K, N//2+1]
    fsc_halves_class:Float[K, N//2+1]
    data_vs_prior_class: Float[K, N//2+1]
    pdf_class:       Float[K]
    pdf_direction:   Float[K, n_directions]
    grad_stepsize_sched:  ScheduleState
    tau2_fudge_sched:     ScheduleState
    subset_size_sched:    ScheduleState
    # Current-iter subset
    subset_part_ids: Int[subset_size]
    pseudo_halfset_ids: Int[subset_size]   # 0 or 1 for each particle
    # Resolution pointers
    current_size:    int
    current_resolution: float
    image_current_size:  Int[G]
    image_coarse_size:   Int[G]
```

`Igrad2` is initialised constant to `MOM2_INIT_CONSTANT` exactly as in
`ml_model.cpp:933`. The constant's numeric value is copied verbatim from
`relion/src/ml_model.h` (1e-6 at the time of writing; parity test reads it at
runtime).

## 4. Particle order and subset selection

RELION path for `!do_split_random_halves`:

1. `Experiment::randomiseParticlesOrder(random_seed + iter, false)` shuffles
   the full particle list. For our path `do_random_halves=false`, so the
   shuffle is a single permutation over `[0, N)`.
   (`exp_model.cpp::randomiseParticlesOrder`.)
2. Take the first `subset_size` of the shuffled list.
3. Stable-sort the selected prefix by optics group id
   (`ml_optimiser.cpp:4907`) so images from the same optics group are
   processed together.
4. For `grad_pseudo_halfsets=true`, alternate halfset ids along the
   stable-sorted list (`ml_optimiser.cpp:5139`).

v3 reproduces steps 1–4 via a `relion_bind` exposure of
`Experiment::randomiseParticlesOrder` so the first-iteration subset order is
byte-identical.

## 5. Module layout (all new)

```
recovar/em/initial_model/                   # new package, not under dense_single_volume
├── __init__.py
├── state.py                  # InitialModelState, ScheduleState
├── schedules.py              # subset_size / stepsize / tau2_fudge pure functions
├── init.py                   # denovo state seeding: Iref=0, Igrad1=0, Igrad2=const,
│                             #   sigma2 from avg-unaligned, current_size via 0.07 rule
├── subset.py                 # particle order / halfset assignment
├── e_step.py                 # VDAM E-step adapter around the low-level kernels
├── m_step.py                 # backprojector accumulation → moments → reference update
├── iteration_loop.py         # run_vdam_iterations orchestrator
└── align_symmetry.py         # post-run relion_align_symmetry equivalent

recovar/relion_bind/
└── initialmodel_bind.cpp     # new: VDAM primitives + schedules + moment arrays

scripts/
└── run_ab_initio.py          # standalone driver, GUI-command-equivalent

tests/unit/initial_model/
├── test_command_snapshot.py  # § G.1
├── test_schedules_parity.py  # § F.1
├── test_particle_order.py    # § F.2
├── test_denovo_init.py       # § F.3
└── test_moments_parity.py    # § F.4

tests/unit/test_relion_bind/
└── test_initialmodel_bind.py # binding-level tests feeding fixtures to § F
```

No touch points in `recovar/em/dense_single_volume/`. No touch points in
`recovar/commands/pipeline.py`. Exposure through the main pipeline is gated
behind § H.

## 6. E-step adapter

The dense-path E-step kernels in `em_engine.py` (half-spectrum scoring,
CTF-applied projection, GEMM cross-term, posterior evaluation) can be reused
**if and only if** their outputs can be adapted to what the VDAM M-step
expects. The adapter must emit, for each particle and each `(k, r, t)`:

- the normalised posterior weights used to accumulate moments,
- `Pmax` and the significant-sample count per particle,
- the masked / unmasked distinction RELION makes between
  `Mweight` vs `Mweight_coarse`,
- noise shells with the RELION half-complex Hermitian-weight convention
  (`w=1` for all half pixels — tracked in the existing parity-debt note),
- `Minvsigma2[0] = 0` (DC zeroing) — this was the source of a prior
  amplitude bug
  (`project_relion_pf2_dc_exclusion_bug.md`).

The adapter runs at `padding_factor=1` — this is the switch, not a
`_refine_relion_mode` fork. A narrow wrapper in
`recovar/em/initial_model/e_step.py` is responsible for this. Existing
pad-2 code paths are untouched.

## 7. M-step: VDAM gradient update

Each mini-batch accumulates gradient data in the backprojector-like
buffer (`data`, `weight`) per class. Then:

```
for k in range(K):
    # 1. pseudo-halfset averaging of the accumulator (RELION does this inline
    #    per pixel in ml_optimiser.cpp:5139; we do it at the end of the batch)
    reweightGrad(data_k, weight_k)                     # backprojector.cpp:1933
    getFristMoment(Igrad1[k_h0], data_k, lambda=mu)    # backprojector.cpp:1943
    getFristMoment(Igrad1[k_h1], data_k_other, lambda=mu)
    getSecondMoment(Igrad2[k], data_k, data_k_other, lambda=mu)
                                                       # backprojector.cpp:1975
    applyMomenta(Igrad1[k_h0], Igrad1[k_h1], Igrad2[k])
                                                       # backprojector.cpp:2000
    reconstructGrad(Iref[k], fsc_halves_class[k],
                    grad_current_stepsize, tau2_fudge_factor,
                    grad_min_resol_shell, use_fsc=False)
                                                       # backprojector.cpp:2054
```

All four moment primitives are bound from RELION C++ via
`initialmodel_bind.cpp` and tested at machine-precision tolerance on tiny
fixtures (§ F.4). `reconstructGrad` is the one piece that has enough recovar
analogue (Wiener + FSC-regularised inverse) to have a "write it in JAX"
temptation — **resist**. The first implementation calls the C++ primitive.
A pure-JAX version comes later, after primitive parity is green.

The `use_fsc=False` default for InitialModel means `reconstructGrad` derives
the FSC estimate from `mom1_noise_power` internally, then clamps to 1 with
an SNR threshold. The v1 plan also missed this path and would have produced a
subtly different FSC-driven regularisation.

## 8. Iteration loop

```python
def run_vdam_iterations(state, dataset, opts, binder):
    for iter in range(1, state.nr_iter + 1):
        state = schedule_update(state, iter)               # stepsize, tau2_fudge, subset_size
        do_grad = (state.nr_iter - iter) >= opts.grad_em_iters and not state.has_converged
        pseudo_halfsets = do_grad
        state = select_subset(state, iter, dataset, pseudo_halfsets)
        state = update_current_resolution(state)           # FSC-driven, no grad_ini_resol shortcut
        state = update_image_size_and_resolution_pointers(state)
        posteriors = e_step_vdam(state, dataset)           # adapter over dense kernels
        state = m_step_vdam(state, posteriors, do_grad, binder)
        state = maybe_apply_ini_constraints(state, iter)   # flatten_solvent, zero_mask, C1 sym
        state = write_iter_artifacts(state, iter)          # run_itNNN_{model,data,optimiser}.star
    state = align_symmetry(state, opts)                    # --apply_sym --select_largest_class
    return state
```

Ordering inside the iteration matches `ml_optimiser.cpp::iterate` lines
3458–3550. The single EM tail iteration at `iter == nr_iter` is handled by
`do_grad=False`, `pseudo_halfsets=False`, `subset_size=-1`.

## 9. Rollout

1. **Phase 1 — schedules, ordering, denovo init (pure Python, no EM yet)**
   Implement `state.py`, `schedules.py`, `init.py`, `subset.py` + matching
   tests (§ F.1–§ F.3). No E/M, no binding calls yet. This shakes out all
   the integer-truncation / indexing / seeding subtleties before any GPU
   work.

2. **Phase 2 — VDAM primitives binding**
   `initialmodel_bind.cpp` exposes `reweightGrad`, `getFristMoment`,
   `getSecondMoment`, `applyMomenta`, `reconstructGrad`. All covered by
   § F.4.

3. **Phase 3 — E-step adapter on a single particle batch**
   Show that `pad=1` scoring with `Minvsigma2[0]=0` + Hermitian-weight-parity
   reproduces RELION's `Mweight` on a 1-particle, 1-class fixture within
   `1e-10`.

4. **Phase 4 — Single VDAM iteration end-to-end**
   Feed one GUI InitialModel iteration (particles subset + moments snapshot)
   into recovar, assert all state fields match the corresponding
   `run_it001_*.star` / `run_it001_class00?.mrc` arrays at RELION-provable
   tolerances. Fail unless passes.

5. **Phase 5 — Full 200-iter run**
   Only after phases 1–4 pass: wire the loop. Compare `run_itNNN_*.star` for
   `NNN ∈ {1, 2, 5, 10, 50, 100, 200}` against the fixture at
   `/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run/`.
   Compare `initial_model.mrc` after `align_symmetry`.

6. **Phase 6 — Pipeline integration**
   Expose through `scripts/run_ab_initio.py` only. No main `pipeline.py`
   integration in this branch.

Each phase has a **hard parity gate**: the phase cannot ship unless its
tests pass. The gate for phase 4 is the most important one; that is where
"still building" becomes "actually parity-ready."

## F. Primitive parity tests (all at machine tolerance, CPU-only fixtures)

All tests use tiny deterministic input (≤ 32³, ≤ 16 particles) so RELION runs
in seconds and every comparison is elementwise.

- **F.1 — Schedules.** For 10 parameter combinations covering defaults,
  boundaries (`ini_frac+fin_frac>0.9`), and continuation
  (`iter > nr_iter - grad_em_iters`), emit `(subset_size, stepsize,
  tau2_fudge)` per-iter. Compare bit-exactly against a RELION binding that
  exposes `updateSubsetSize`, `updateStepSize`, `updateTau2Fudge`.

- **F.2 — Particle order.** Bind `Experiment::randomiseParticlesOrder`.
  Assert that for 10 `(random_seed, iter)` combinations with 500 particles
  and 2 optics groups, the full shuffle + stable-sort + halfset-alternation
  sequence matches RELION elementwise.

- **F.3 — Denovo init.** Given a STAR/stack with no reference:
  - `sigma2_noise` matches `run_it000_model.star` (already exact on the
    fixture per the handoff note).
  - `current_size`, `current_resolution`, `ini_high` at iter 0 match
    `run_it000_optimiser.star`.
  - `Iref[k]` are zeros, `Igrad1[2k + h]` are zeros, `Igrad2[k]` equal
    `MOM2_INIT_CONSTANT`.
  - `pdf_class`, `pdf_direction` match RELION's uniform init.

- **F.4 — Moment primitives.** For each of `reweightGrad`,
  `getFristMoment`, `getSecondMoment`, `applyMomenta`, `reconstructGrad`:
  feed deterministic input arrays into the bound C++ primitive and into a
  recovar JAX implementation (Phase 2 delivers the binding; the JAX
  implementation is optional in v3 but if added its output must match the
  binding to `1e-12` in relative error).

- **F.5 — Denovo average-unaligned.** Reproduce RELION's
  `calculateSumOfPowerSpectraAndAverageImage` +
  `setSigmaNoiseEstimatesAndSetAverageImage` (CPU path). Compare power-spectra
  and `sigma2_noise` against `run_it000_model.star` element-wise.

## G. Command / intermediate parity tests

- **G.1 — Command snapshot.** A fixed GUI InitialModel job dict feeds
  `recovar.commands.run_ab_initio.build_command(...)`. The output string must:
  - contain `--grad --denovo_3dref --pad 1 --auto_sampling --oversampling 1
    --healpix_order 1 --offset_range 6 --offset_step 2`,
  - contain the exact `--tau2_fudge`, `--K`, `--sym`, `--iter` values the
    user passed,
  - NOT contain `--split_random_halves`, `--auto_refine`,
    `--low_resol_join_halves`, `--norm`, `--scale`, `--firstiter_cc`,
    `--ini_high`, `--grad_ini_resol`, `--grad_fin_resol`,
  - reject MPI (`nr_mpi > 1` → assertion, matching RELION's
    `pipeline_jobs.cpp:3437` error).

- **G.2 — Per-iter parity against the fixture.** For the reference run at
  `/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run/`
  compare at iters 1, 2, 5, and the iterations covered by the
  `..._extract/` directory:
  - selected particle subset (by `_rlnImageName`) and halfset assignment,
  - `Iref[k]`, `Igrad1[2k+h]`, `Igrad2[k]` (elementwise, relative 1e-9 on the
    unaveraged raw arrays),
  - `sigma2_noise`, `tau2_class`, `fsc_halves_class`, `data_vs_prior_class`,
  - `Pmax` and significant-sample counts from the E-step adapter,
  - `pdf_class`, `pdf_direction`.

- **G.3 — Final-map parity.** After `align_symmetry` with the fixture's
  symmetry and largest-class selection, recovar's
  `initial_model.mrc` has FSC ≥ 0.999 (shell-by-shell) against RELION's
  `initial_model.mrc` using `load_relion_volume` to align conventions. FSC
  against ground truth is **only** reported, not a gate.

## H. Exit criteria for merging

- All of § F and § G pass CPU-only and under a fixed `--random_seed`.
- No modification to files under `recovar/em/dense_single_volume/`,
  `recovar/commands/`, or `recovar/relion_bind/` other than the new
  `initialmodel_bind.cpp`.
- Dense-path tests still pass (`pixi run test-fast` on this branch).
- The branch is rebased on the latest `codex/em-phase01-sparse-pass2` tip so
  the local-posterior changes from `84440d7e` remain unchanged.
- PR description lists the Slurm job ids for § G.2 and § G.3, plus the
  per-iter tolerance actually achieved (not just "pass").

Out of scope for this branch:
- main-pipeline integration,
- GPU acceleration of VDAM M-step (phase-5 proof first on CPU),
- multi-class InitialModel (K > 1) — the state supports it and the snapshot
  test covers the K=1 GUI default; K > 1 is a follow-up branch,
- tomo InitialModel (`--denovo_3dref` tomo path),
- the bare-CLI fallback defaults (the 35 / 15 Å schedule that only fires on
  continuation from a minimal `_optimiser.star` without those labels),
- `--grad_em_iters > 0` final-EM-tail case (grad-only + a single EM iter is
  what the GUI ships, and what § G pins).

## Fixture-wiring roadmap (remaining work for full parity)

As of 2026-04-24, Phases 1–6 of this plan are implemented + tested
(142 unit tests, all CPU-only, all passing in ~3 s). Per-iter parity
against the RELION fixture at
`/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run/`
requires the following additional work. Each step has an explicit hard
parity gate against the fixture.

### F5 — iter-0 `Mavg` + `sigma2_noise` parity

**Status: DONE (2026-04-24).**

`recovar/em/initial_model/avg_unaligned.py::compute_avg_unaligned_and_sigma2`
matches `run_it000_model.star` `data_model_optics_group_1` sigma2 at:

  max abs diff = 3.413e-7
  max rel diff = 4.953e-5

Test: `tests/unit/initial_model/test_avg_unaligned_fixture.py`.

### F6 — Iref seeding from `Mavg`

`setSigmaNoiseEstimatesAndSetAverageImage` also seeds each class's Iref
from the average-unaligned backprojection of a C1 volume from a
random-orientation subset of the particles (`ml_optimiser.cpp:3259-3265`
— calls `wsum_model.BPref[iclass].reconstruct(Iref[iclass], ...)` on an
accumulator built from one particle per class). This is what Phase 3's
`initialise_denovo_state` currently leaves at zero.

**Implementation:** extend `avg_unaligned.py` to optionally return the
random-orientation-backprojected Iref and wire it into
`initialise_denovo_state`. Use the existing `vdam_randomise_particles_order`
binding so the random class assignment is byte-identical to RELION.

**Parity gate:** `Iref` matches `run_it000_class001.mrc` after
`initialLowPassFilterReferences` at 0.07·ori_size with edge-width 2.
Tolerance: correlation > 0.999 and std within ±10%. (The handoff's
earlier observation of CC=0.67 with 2× std came from a different
bootstrap path — we bypass that by implementing the RELION code path
literally rather than trying to replicate via a separate "bootstrap".)

### F7 — E-step wiring at `padding_factor=1`

The Phase-3 `e_step.py` helpers (`minvsigma2_with_dc_zero`,
`hermitian_weights_relion`, `fourier_crop_half`) need to be threaded into
a real E-step that:

  - constructs a HEALPix order-1 rotation grid (48 orientations × K
    classes × 8 sub-orientations from `--oversampling 1` = 384 rotations
    at iter 1),
  - constructs a translation grid at `--offset_range 6 --offset_step 2`
    (7×7 = 49 translations),
  - CTF-multiplies each image at `padding_factor=1`,
  - scores `log p(image | rotation, translation, class)` using the
    current `Iref[class]`,
  - normalises the posterior with `exp_Mweight` clamping to the top
    `exp_significant_weight` window (RELION prunes orientations whose
    weight is below significance_threshold × max_weight),
  - emits per-(class, halfset) `VdamAccumulator`s by applying the
    posterior-weighted images into the backprojector at `pad=1`.

Most of this machinery already exists in
`recovar/em/dense_single_volume/em_engine.py` at `padding_factor=2` for
auto-refine. The adapter is ~300 lines: switch to pad=1, zero
`Minvsigma2[0]`, apply RELION's Hermitian weights, and route outputs
through the existing HEALPix sampler and translation-shift kernels.

**Parity gate:** posterior `Pmax` for 8 hand-picked particles (the
handoff's [429, 459, 248, 12, 148, 409, 188, 318] list) matches
`run_it001_data.star` `_rlnMaxValueProbDistribution` within 1% *when*
Iref is `run_it000_class001.mrc` (not recovar's own Iref). The handoff
already achieved this at the RELION-ref-injection level
(mean Pmax 0.1058 vs RELION 0.1060), so this step mainly means porting
that scoring harness into `recovar/em/initial_model/e_step.py`.

### F8 — Full single-iter parity

With `Mavg`/sigma2 (F5) + Iref seeding (F6) + E-step (F7) + Phase-4
M-step primitives, run iter 1 end-to-end and compare:

  - `Iref[0]` vs `run_it001_class001.mrc`:
      correlation > 0.995 real-space + shell-by-shell amplitude ratio
      within 5% to Nyquist.
  - `Igrad1[slot_0]`, `Igrad1[slot_1]`, `Igrad2[0]` vs
      `run_it001_1moment001.mrc` / `run_it001_2moment001.mrc` after
      RELION-frame conversion (`load_relion_volume`).
      Relative L2 diff < 1e-6 since these are accumulators, not
      reconstructions.
  - `sigma2_noise[0]` vs iter-1 sigma2 in `run_it001_model.star`:
      max abs diff < 1e-6.
  - `tau2_class`, `fsc_halves_class`, `data_vs_prior_class`:
      max abs diff < 1e-5.

If this gate passes, iter-2+ parity follows mechanically (same code
path, no additional algorithmic surface). The RELION fixture has 2 iters
of extract and full-run MRCs for iter 0, 1, 2 which lets us pin iter-2
parity at the same tolerances before declaring "iter-by-iter matches".

### F9 — Full 200-iter parity (quality gate)

`Iref[0]` after 200 iters and after `align_symmetry` matches the
fixture's `initial_model.mrc` with shell-by-shell FSC ≥ 0.999 to
Nyquist. FSC against ground truth is reported but not gated.

### F10 — Speed parity (performance gate)

RELION's reference run wall-time was recorded in `relion_stdout.log`
(~7.5 min for 500 particles × 200 iters on CPU in the fixture's `--j 4`
configuration). recovar's target:

  - CPU parity: within ±25% of RELION wall-time on the same CPU.
  - GPU target: ≤ 60 s per iter on A100-80GB for 5000 particles at box
    128 (extrapolating from the existing dense-path numbers in
    `recovar/em/CLAUDE.md` — ~29 s/iter for pad=2 auto-refine on 36k
    rotations; VDAM at pad=1 / 384 rotations should be ~6-10× faster).

The GPU path requires JIT-compiling the E-step adapter (no host-round-
trips per rotation), using the existing half-spectrum GEMM kernel, and
keeping `Iref`/`Igrad*` on-device across iterations. M-step primitives
currently route through C++ host bindings; the bindings must either (a)
become FFI XLA kernels, or (b) be replaced by pure-JAX ports once F8
parity is locked in.

**Hard gates per step:** F5, F6, F7, F8 each have their own parity
test against fixed bytes in the fixture. Nothing ships until all four
are green, then F9 provides the full-run sanity check and F10 provides
the speed sanity check. All F-steps run CPU-only for the first
implementation; GPU path is a separate branch that must not be merged
until F8 passes.

## Open questions that must be resolved before phase 4 begins

1. How to represent `Igrad1` / `Igrad2` storage: Equinox modules with
   complex-dtype leaves, or a flat `ModelStateBackprojectors` PyTree? This
   affects how the JIT boundary is drawn between the iteration loop and the
   moment primitives.
2. Whether to call the C++ `reconstructGrad` primitive through
   `initialmodel_bind` at every iteration (simplest, slowest) or mirror it in
   JAX from day one. v3 says **call the primitive**; revisit only if the
   per-iter cost is dominated by it.
3. How `auto_sampling` interacts with `updateImageSizeAndResolutionPointers`
   during the VDAM gradient phase. `ml_optimiser.cpp:2650` shows
   `do_auto_sampling=true` is set unconditionally for gradient refinement,
   but the first call to `updateAngularSampling` only happens after the
   E-step returns (line 3701). A small probe test on the fixture will pin
   the exact sequence.
