# RECOVAR / RELION InitialModel Ab-Initio Algorithm

Reviewed against this checkout on 2026-05-01. This is a developer-facing
description of the native RECOVAR implementation of RELION GUI
`InitialModel`:

```text
relion_refine --grad --denovo_3dref --pad 1 --auto_sampling \
  --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 2 \
  --tau2_fudge 4 --K <nr_classes> ...
relion_align_symmetry --apply_sym --select_largest_class ...
```

`InitialModel` is not standard dense EM from a supplied reference. It is
RELION's denovo gradient-refinement path: initialize references from randomly
oriented particles, run VDAM mini-batch E/M iterations, update gradient momenta,
and finally choose the largest class for `initial_model.mrc`.

## The Shared EM Problem

Both standard refinement and ab-initio estimate hidden orientations,
translations, and classes for each image. For image \(y_i\), class \(k\),
rotation \(r\), translation \(t\), CTF \(C_i\), projection operator \(P_r\),
and class volume \(\mu_k\):

```text
y_i = S_t C_i P_r mu_k + eps_i
eps_i ~ N_C(0, Sigma_i)
z_i = (k, r, t)
```

The E-step computes posterior weights over the active support:

```text
gamma_i(k,r,t) =
  exp(log pi_k + log p_r + log p_t + score_i(k,r,t))
  / sum_{k',r',t' in support_i}
      exp(log pi_k' + log p_r' + log p_t' + score_i(k',r',t'))
```

The score is the Gaussian data likelihood up to image-only constants:

```text
score_i(k,r,t) =
  -0.5 * || y_i - S_t C_i P_r mu_k ||^2_{Sigma_i^-1}
```

RECOVAR's dense score kernel implements the expanded form in
[`helpers/scoring.py::_score_rotation_block`](../../recovar/em/dense_single_volume/helpers/scoring.py):

```text
cross = -2 Re(conj(shifted_image) @ projected_volume)
norm  = (CTF^2 / sigma2) @ |projected_volume|^2
score = -0.5 * (cross + norm)
```

The same posterior algebra is used by:

- standard dense EM and K-class/Class3D replay, through
  [`em_engine.py::run_em`](../../recovar/em/dense_single_volume/em_engine.py),
  [`k_class.py::run_dense_k_class_em`](../../recovar/em/dense_single_volume/k_class.py),
  and [`k_class.py::run_local_k_class_em`](../../recovar/em/dense_single_volume/k_class.py);
- native InitialModel, through
  [`driver.py::_native_expectation_step`](../../recovar/em/initial_model/driver.py)
  and [`dense_adapter.py::run_dense_initial_model_estep`](../../recovar/em/initial_model/dense_adapter.py).

The debugging rule is: equal per-hypothesis scores are not enough for map
parity. The map also depends on the support being normalized over, the class
and pose priors, pseudo-halfset routing, BPref frame conversion, and the M-step.

## What Standard EM / Class3D Does With The Math

Standard EM treats the E-step statistics as sufficient statistics for a
regularized reconstruction. It accumulates, per class:

```text
Ft_y(k) =
  sum_i sum_{r,t}
    gamma_i(k,r,t) P_r^* C_i Sigma_i^-1 S_t^* y_i

Ft_ctf(k) =
  sum_i sum_{r,t}
    gamma_i(k,r,t) P_r^* C_i Sigma_i^-1 C_i P_r
```

Then the M-step solves a Wiener-like reconstruction:

```text
mu_k_new ~= Ft_y(k) / (Ft_ctf(k) + prior_precision_k)
```

In RECOVAR dense EM this path is implemented by
[`em_engine.py::run_em`](../../recovar/em/dense_single_volume/em_engine.py)
for one class and
[`k_class.py::run_dense_k_class_em`](../../recovar/em/dense_single_volume/k_class.py)
for joint class x pose normalization. RELION Class3D parity replay reads the
previous and target RELION model STAR files in
[`scripts/run_k_class_parity.py`](../../scripts/run_k_class_parity.py), uses
the RELION class distributions and per-class tau spectra, then compares the
RECOVAR reconstruction to the RELION iteration output.

For RELION-style adaptive/local support, the two-pass structure is:

```text
coarse pass:
  score coarse rotations/translations
  retain significant parent hypotheses

fine pass:
  expand retained parents to oversampled children
  normalize gamma only over retained fine hypotheses
  accumulate Ft_y/Ft_ctf only over that retained support
```

The support restriction is part of the algorithm, not an optimization detail.
The helpers are
[`helpers/significance.py::_compute_k_class_significance_batched`](../../recovar/em/dense_single_volume/helpers/significance.py)
for coarse support and
[`local_layout.py::build_pass2_hypothesis_layout`](../../recovar/em/dense_single_volume/local_layout.py)
for fine-support expansion.

For K-class EM, \(\pi_k\) and each class's tau/reference-variance curve are
per-class model state. For `K=1`, these collapse to one scalar class prior and
one volume prior. The posterior is still normalized over class x pose; with
`K=1`, the class axis is just length one.

## What VDAM / InitialModel Does With The Same Math

RELION GUI InitialModel uses the same likelihood and posterior weights, but it
does not replace the volume with the direct regularized EM solution. It runs a
VDAM gradient refinement loop over subsets of particles:

```text
for iter = 1 .. nr_iter:
    update subset size, step size, tau2_fudge, current resolution
    choose a shuffled particle subset
    split selected particles into pseudo-halfsets
    run E-step on class x pose support
    convert dense Ft_y/Ft_ctf to RELION BackProjector slabs
    update first/second gradient moments
    reconstructGrad updates Iref using momenta and tau2_fudge
    update class, direction, offset, and Pmax metadata
```

The native RECOVAR entry point is
[`driver.py::run_native_initial_model`](../../recovar/em/initial_model/driver.py).
The command-line options in
[`scripts/run_ab_initio.py::InitialModelJobOptions`](../../scripts/run_ab_initio.py)
mirror the RELION GUI command, while
[`scripts/run_ab_initio.py::build_command`](../../scripts/run_ab_initio.py)
is only a command snapshot for parity/debugging.

The loop is in
[`iteration_loop.py::run_vdam_iterations`](../../recovar/em/initial_model/iteration_loop.py).
That loop deliberately does not know about JAX arrays or projection kernels. It
selects subsets, applies RELION schedules, calls an injected expectation-step
closure, then calls
[`m_step.py::vdam_m_step`](../../recovar/em/initial_model/m_step.py).

The InitialModel state is not just a volume:

```text
Iref[k]       = current reference for class k
Igrad1[k]     = first moment for class k, pseudo-halfset 0
Igrad1[K + k] = first moment for class k, pseudo-halfset 1
Igrad2[k]     = second moment for class k
pdf_class[k]  = class prior
pdf_direction[k, r] = direction prior
```

This state is created by
[`init.py::initialise_denovo_state`](../../recovar/em/initial_model/init.py)
and bootstrapped by
[`bootstrap_iref.py::compute_bootstrap_iref_via_cpp`](../../recovar/em/initial_model/bootstrap_iref.py).
The dense E-step bridge in
[`dense_adapter.py`](../../recovar/em/initial_model/dense_adapter.py)
converts `Iref` to dense Fourier means, runs dense or sparse K-class EM, packs
selected particles by pseudo-halfset, and emits VDAM accumulators.

The critical difference from standard EM is the M-step. Instead of
`Ft_y / (Ft_ctf + prior_precision)`, InitialModel sends the BackProjector data
through RELION VDAM primitives in
[`m_step.py::vdam_m_step_single_class`](../../recovar/em/initial_model/m_step.py):

```text
vdam_reweight_grad(data, weight)
vdam_first_moment(Igrad1_h, reweighted_data_h)
vdam_second_moment(Igrad2, halfset difference)
vdam_apply_momenta(Igrad1, Igrad2)
vdam_reconstruct_grad(Iref, momentum_data, weight, tau2_fudge, step_size)
```

The dense-to-RELION BackProjector layout conversion is handled by
[`layout.py::run_em_output_to_bpref`](../../recovar/em/initial_model/layout.py).
For the default dense InitialModel path, the RELION BPref frame scales are:

```text
bp_data   *= -N^2
bp_weight *=  N^4
```

Those scales are implemented by
[`layout.py::relion_bpref_frame_scales`](../../recovar/em/initial_model/layout.py).
Applying them twice, or omitting them on a dense path, gives the common failure
mode where scores look good but BPref or maps do not.

## Current Parity Status

Do not read this document as claiming perfect parity. Current measurements on
this branch show:

- InitialModel / ab-initio is not yet RELION-parity under GUI defaults. The
  E-step and BPref tests are regression gates, not parity proof.
- K-class/Class3D is near-perfect for the `os0` diagnostic harness, but the
  GUI-style oversampling fixture still shows class-assignment and Pmax
  mismatch.
- The Ribosembly 100k-image jobs are intended to test larger K-class behavior;
  they should not be used as evidence until their parity gates complete.

## Implementation Index

Use this index only after reading the algorithm sections above.

- GUI command snapshot:
  [`scripts/run_ab_initio.py`](../../scripts/run_ab_initio.py)
- Native InitialModel driver:
  [`driver.py::run_native_initial_model`](../../recovar/em/initial_model/driver.py)
- Denovo state:
  [`state.py::InitialModelState`](../../recovar/em/initial_model/state.py),
  [`init.py::initialise_denovo_state`](../../recovar/em/initial_model/init.py)
- Bootstrap and noise:
  [`avg_unaligned.py::compute_avg_unaligned_and_sigma2`](../../recovar/em/initial_model/avg_unaligned.py),
  [`bootstrap_iref.py::compute_bootstrap_iref_via_cpp`](../../recovar/em/initial_model/bootstrap_iref.py)
- VDAM loop and schedules:
  [`iteration_loop.py::run_vdam_iterations`](../../recovar/em/initial_model/iteration_loop.py),
  [`schedules.py`](../../recovar/em/initial_model/schedules.py),
  [`subset.py::select_vdam_subset`](../../recovar/em/initial_model/subset.py)
- InitialModel dense bridge and M-step:
  [`dense_adapter.py`](../../recovar/em/initial_model/dense_adapter.py),
  [`layout.py`](../../recovar/em/initial_model/layout.py),
  [`m_step.py`](../../recovar/em/initial_model/m_step.py)
- Standard dense EM / K-class:
  [`em_engine.py`](../../recovar/em/dense_single_volume/em_engine.py),
  [`k_class.py`](../../recovar/em/dense_single_volume/k_class.py)
- K-class parity harness:
  [`scripts/run_k_class_parity.py`](../../scripts/run_k_class_parity.py)

## Detail: Denovo State and Bootstrap

The initial state is split into data-independent and data-dependent parts.

`initialise_denovo_state` constructs:

- `Iref[k] = 0` in RELION real-space volume convention.
- `Igrad1[h*K + k] = 0` for first-moment slots. With pseudo-halfsets active
  there are `2K` slots.
- `Igrad2[k] = 1 + 1j`, matching RELION's `MOM2_INIT_CONSTANT`.
- `pdf_class[k] = 1/K`.
- `pdf_direction[k, r] = 1/(K * n_directions)`.
- Empty `sigma2_noise`, `tau2_class`, `fsc_halves_class`, and
  `data_vs_prior_class`.
- `ini_high`, `current_resolution_shell`, and `current_size` from RELION's
  `0.07 * ori_size` rule.

The initial resolution helpers are:

```text
ini_shell = ROUND(0.07 * N)
ini_high_A = N * pixel_size / ini_shell
current_size = min(N, 2 * (ini_shell + 10))
```

`compute_avg_unaligned_and_sigma2` mirrors RELION's average-unaligned noise
setup. For each optics group and up to `minimum_nr_particles` particles:

```text
Mavg = mean(masked image)
particle_power_g[s] = mean_i radial_average(|FFT(masked image_i)|^2)[s]
Mavg_power[s] = radial_average(|FFT(Mavg)|^2)[s]
sigma2_noise[g, s] = particle_power_g[s] / 2 - Mavg_power[s] / 2
```

Non-positive shells are repaired by copying a positive neighbor, matching the
RELION source behavior.

`compute_bootstrap_iref_via_cpp` is the recommended bootstrap path. It calls the
local RELION binding so the following C++ operations are not reimplemented in
Python:

- per-particle RNG reset with `random_seed + part_id`,
- random Euler angle draw,
- soft real-space mask,
- RELION FFT normalization and `CenterFFTbySign`,
- CTF image construction,
- `BackProjector::set2DFourierTransform`,
- `BackProjector::reconstruct(..., do_map=false)`.

The Python wrapper currently defaults the bootstrap binding to `padding_factor=2`
because that matches the available RELION fixture better than `1`, even though
the GUI command uses `--pad 1` for the later refinement. Treat this as a known
bootstrap parity quirk, not as a license to change the VDAM E/M padding.

## Detail: VDAM Iteration Loop

`run_vdam_iterations` owns the RELION gradient-refine control flow:

```text
phase_lengths = compute_phase_lengths(nr_iter, grad_ini_frac, grad_fin_frac)

for iter = 1 .. nr_iter:
    do_grad = ((nr_iter - iter) >= grad_em_iters) and not state.has_converged
    state = default_schedule_update(...)
    state = select_subset_for_iter(..., do_grad=do_grad)
    accumulators, meta = expectation_step(state, subset_particle_ids, halfset_ids)
    state = vdam_m_step(state, accumulators, stepsize, tau2_fudge)
    state = update_probabilities_from_estep_meta(state, meta, do_grad, mu)
    iter_artifact_sink(state, iter, meta)
```

This module deliberately does not know about JAX, particle stacks, or dense
projection kernels. The E-step is an injected callback. That separation makes
it easier to debug whether a parity failure is caused by scheduling/subset
state, posterior math, BPref conversion, or RELION M-step bindings.

Schedule defaults mirror RELION GUI InitialModel:

```text
nr_iter = 200
K = 1
tau2_fudge_arg = 4
grad_ini_frac = 0.3
grad_fin_frac = 0.2
grad_em_iters = 0
mu = 0.9
base grad_current_stepsize = 0.5
```

Subset sizes come from `default_subset_sizes_for_3d_initial_model`:

```text
grad_ini_subset_size = clamp(round(0.005 * N_particles), 200, 5000)
grad_fin_subset_size = clamp(round(0.1   * N_particles), 1000, 50000)
```

`compute_subset_size` returns `-1` to mean "all particles" when gradient mode
is off, convergence flags are set, the requested subset covers the effective
particle count, suspended local searches are active, or the final multiclass
iteration requires all particles.

`select_subset_for_iter` uses the exact RELION ordering contract:

```text
shuffle all particle ids with seed random_seed + iter
take the prefix of length subset_size
stable-sort the prefix by optics group
if pseudo-halfsets active, assign 0,1,0,1,... along the stable-sorted prefix
```

This is not equivalent to sorting first or alternating halfsets in STAR order.
The halfset routing changes the M-step because `Igrad1[k]` and `Igrad1[K+k]`
receive different BackProjectors.

## Detail: E-Step Scoring

For image \(y_i\), class \(k\), rotation \(r\), translation \(t\), CTF \(C_i\),
projection \(P_r\), and reference \(\mu_k\):

```text
y_i = S_t C_i P_r mu_k + eps_i
eps_i ~ N_C(0, Sigma_i)
z_i = (k, r, t)
```

The posterior normalized over the active support is:

```text
gamma_i(k,r,t) =
  exp(log pi_k + log p_r + log p_t + score_i(k,r,t))
  / sum_{k',r',t' in support_i} exp(log pi_k' + log p_r' + log p_t' + score_i(k',r',t'))
```

The Gaussian score used by the dense kernels is, up to image-only constants:

```text
score_i(k,r,t) =
  -0.5 * || y_i - S_t C_i P_r mu_k ||^2_{Sigma_i^-1}
```

The implementation expands this in `_score_rotation_block`:

```text
cross = -2 Re(conj(shifted_image) @ projected_volume)
norm  = (CTF^2 / sigma2) @ |projected_volume|^2
score = -0.5 * (cross + norm)
```

Dense K-class EM computes class evidence and normalizes over the joint
`class x pose` hidden variable. InitialModel uses that same posterior algebra,
but its support and M-step differ from standard reconstruction EM.

## Detail: Dense Adapter, Sparse Pass 2, and BPref Conversion

`dense_adapter.py` is the bridge between native dense K-class EM and RELION
VDAM. Its responsibilities are:

- Convert `InitialModelState.Iref` to dense Fourier means with
  `reference_to_dense_means`.
- Build class log priors from `state.pdf_class` with
  `class_log_priors_from_state`.
- Split or pack selected particles into pseudo-halfset reconstruction groups.
- Run the dense or sparse-pass2 K-class E-step.
- Convert dense `Ft_y` / `Ft_ctf` into `VdamAccumulator` objects.
- Return E-step metadata used for `pdf_class`, `pdf_direction`, particle
  offsets, class assignments, and per-iteration artifacts.

The dense, non-sparse path calls `run_dense_k_class_em` once with all classes.
When pseudo-halfsets are active, both halves are packed into one dense E-step
and separated only in the returned grouped reconstruction accumulators. This
avoids duplicating projection/scoring work while still giving VDAM independent
halfset BackProjectors.

The sparse pass-2 path is closer to RELION adaptive InitialModel:

```text
pass 1:
  score all coarse class x rotation x translation hypotheses
  compute per-image significant coarse support

pass 2:
  union significant coarse parents across classes for each image
  expand parents to oversampled fine rotations/translations
  run exact local K-class EM only on that sparse fine support
  normalize posterior over retained fine support
  accumulate only retained/support-weighted hypotheses
```

This support restriction is mathematically important. A full fine-grid
implementation can match candidate scores but still disagree with RELION's
BackProjectors because it normalizes over hypotheses that RELION never
retained.

The dense-to-VDAM bridge is:

```text
dense Ft_y[k], Ft_ctf[k]
  -> layout.py::run_em_output_to_bpref
  -> optional z flip when relion_projector_frame=True
  -> optional frame scales from relion_bpref_frame_scales
  -> VdamAccumulator(data, weight, class_idx=k, halfset_idx=h)
```

For the default native InitialModel config, `relion_bpref_frame=True`, so
`relion_bpref_frame_scales(N)` applies:

```text
bp_data   *= -N^2
bp_weight *=  N^4
```

Do not mix this dense-frame conversion with a path that already emits native
RELION BPref arrays. Frame/sign mistakes usually show up as good score parity
but bad `BPref` or `Iref` parity.

`gpu_pipeline.py` used to contain the old monolithic GPU path. On this branch it
only re-exports `DenseInitialModelEstepConfig`, `run_dense_initial_model_estep`,
and the layout helpers for compatibility. New InitialModel work should not add
logic there.

## Detail: M-Step Math and VDAM Momenta

The E-step accumulates standard EM sufficient statistics per class and halfset:

```text
Ft_y_h(k) =
  sum_{i in halfset h} sum_{r,t}
    gamma_i(k,r,t) P_r^* C_i Sigma_i^-1 S_t^* y_i

Ft_ctf_h(k) =
  sum_{i in halfset h} sum_{r,t}
    gamma_i(k,r,t) P_r^* C_i Sigma_i^-1 C_i P_r
```

Standard dense EM would reconstruct a new mean approximately as:

```text
mu_k_new ~= Ft_y(k) / (Ft_ctf(k) + prior_precision(k))
```

InitialModel does not use that replacement M-step. `m_step.py` sends each
class's BPref data through RELION VDAM primitives:

1. `vdam_reweight_grad(data, weight)`: normalize raw data by
   `max(1, weight)` inside the active Fourier radius.
2. `vdam_first_moment`: update each first-moment slot:
   `m1_h <- 0.9 * m1_h + 0.1 * reweighted_data_h`.
3. `vdam_second_moment`: update the class second moment from the normalized
   halfset difference with `lambda = 0.999`.
4. `vdam_apply_momenta`: combine halfset first moments, divide by
   `sqrt(m2)`, and estimate `mom1_noise_power`.
5. `vdam_reconstruct_grad`: update `Iref[k]` using the current reference,
   post-momentum data, BPref weights, `grad_current_stepsize`,
   `tau2_fudge_factor`, and `mom1_noise_power`.

The critical state layout is:

```text
Igrad1[k]     = halfset-0 first moment for class k
Igrad1[K + k] = halfset-1 first moment for class k
Igrad2[k]     = second moment for class k
```

`vdam_m_step` expects accumulators ordered as all halfset-0 classes followed by
all halfset-1 classes when pseudo-halfsets are active. The adapter preserves
this by sorting halfset groups before appending accumulators.

## Detail: Volume Prior, Tau2, and Regularization

This section separates what is known directly from RECOVAR code in this branch
from behavior inferred from RELION's InitialModel/Class3D conventions and the
local RELION C++ bindings. The word "prior" is overloaded: RELION uses class,
direction, and translation priors in the E-step, and a Fourier-shell reference
variance prior (`tau2`, also written as reference variance/sigma2 in STAR files)
in the M-step reconstruction filter.

### RELION settings and STAR/model state

In standard RELION EM, AutoRefine, and Class3D, the previous iteration's
`run_itNNN_model.star` is part of the model state for the next iteration. The
important blocks are:

- `data_model_classes`: `_rlnReferenceImage` points at each class map and
  `_rlnClassDistribution` stores class weights.
- `data_model_class_1`, `data_model_class_2`, ...: per-shell reference
  variance is stored as `_rlnReferenceTau2` or `_rlnReferenceSigma2` depending
  on the RELION/version path; these tables also carry FSC/SSNR/data-vs-prior
  quantities in mature refinement output.
- `data_model_pdf_orient_class_1`, ...: `_rlnOrientationDistribution` stores
  learned direction priors for the next E-step.
- `data_model_general`: stores control scalars such as current image size,
  current resolution, `rlnTau2FudgeFactor`, and `rlnSigmaOffsetsAngst` when
  present in the produced model file.
- `run_itNNN_optimiser.star`: stores optimiser/control metadata, including
  convergence counters and accuracy estimates used by replay code.

RECOVAR parsing helpers for this state are concentrated in
[`recovar/em/sampling.py`](../../recovar/em/sampling.py). In particular,
`read_relion_model_metadata` reads current image size/resolution,
`read_relion_optimiser_metadata` reads optimiser counters, and
`read_relion_direction_prior` reads `model_pdf_orient_class_1`.
[`scripts/run_k_class_parity.py`](../../scripts/run_k_class_parity.py) contains
the K-class replay-specific readers: `_tau_spectrum` reads
`rlnReferenceTau2`/`rlnReferenceSigma2`, `_class_distributions` reads
`rlnClassDistribution`, and `_read_class_direction_priors` reads every
`model_pdf_orient_class_N` table.

Native RECOVAR InitialModel currently writes a much smaller model STAR in
[`driver.py::_write_model_star`](../../recovar/em/initial_model/driver.py):
`data_model_classes` contains reference map paths and class distributions, and
`data_model_optics_group_1` contains `_rlnSigma2Noise`. It does not yet write
per-class `model_class_N` tau2/FSC/data-vs-prior tables or
`model_pdf_orient_class_N` direction-prior tables. That is a current code-level
difference from mature RELION model output, not an inferred RELION rule.

### InitialModel startup vs later iterations

Native InitialModel starts cold. In
[`init.py::initialise_denovo_state`](../../recovar/em/initial_model/init.py),
`tau2_class`, `fsc_halves_class`, and `data_vs_prior_class` are allocated as
zeros for every class. The initial class prior is uniform (`pdf_class[k]=1/K`),
and the initial direction prior is uniform over class x direction. The first
data-dependent spectral state is `sigma2_noise`, estimated by
[`avg_unaligned.py::compute_avg_unaligned_and_sigma2`](../../recovar/em/initial_model/avg_unaligned.py).

The InitialModel M-step in this branch does not explicitly compute and store a
new `tau2_class` array after each native iteration. Instead,
[`m_step.py::vdam_m_step_single_class`](../../recovar/em/initial_model/m_step.py)
passes the current reference, BPref data, BPref weights, `fsc_halves_class[k]`,
`grad_current_stepsize`, the scheduled `tau2_fudge_factor`, and
`mom1_noise_power` to the RELION binding `vdam_reconstruct_grad`. From RECOVAR
code, the known contract is that `reconstructGrad` receives `use_fsc=False` and
`do_grad=True`, so RELION's gradient reconstruction derives its effective FSC /
noise weighting internally from the momentum-noise spectrum. The exact
per-shell tau2 update inside that binding is RELION behavior inferred from the
C++ primitive, not reimplemented Python code in this branch.

Later native InitialModel iterations therefore reuse the same Python-carried
state fields but update the reference through RELION's VDAM binding. The
regularization strength seen by that binding changes through
[`schedules.py::compute_tau2_fudge`](../../recovar/em/initial_model/schedules.py),
which mirrors RELION `updateTau2Fudge`: for GUI InitialModel defaults
`--tau2_fudge 4` and an empty scheme, the scheduled value grows from near `1`
toward `4` over the gradient phases. This is different from standard dense EM,
where the Python reconstruction path receives an explicit tau spectrum.

### Standard EM/Class3D tau2 path

The dense single-volume RELION-parity loop in
[`recovar/em/dense_single_volume/iteration_loop.py`](../../recovar/em/dense_single_volume/iteration_loop.py)
does use explicit tau2-like arrays. `_reconstruct_volume_eager` calls
[`relion_functions.post_process_from_filter_v2`](../../recovar/reconstruction/relion_functions.py)
with `tau=mean_variance`, `tau2_fudge`, `use_spherical_mask=True`,
`grid_correct=True`, and `minres_map=RELION_MINRES_MAP` unless a diagnostic
variant overrides those defaults. `RELION_MINRES_MAP` is `5` in this branch.

The lower-level regularization formula is in
[`relion_functions.adjust_regularization_relion_style`](../../recovar/reconstruction/relion_functions.py):

```text
regularized_weight = Ft_ctf + inv_tau
inv_tau = 1 / (padding_factor^3 * tau2_fudge * tau2[shell])
```

`minres_map` gates the prior term: shells below `minres_map` do not receive the
tau2 prior penalty. If tau is effectively zero where there is data, RECOVAR
matches RELION's floor by using `1 / (0.001 * shell-averaged weight)` rather
than an infinite penalty. After division by the regularized filter,
`post_process_from_filter_v2` inverse-FFTs, crops, optionally applies RELION's
soft spherical mask, optionally applies a supplied mask, optionally applies
gridding correction, and returns the Fourier map.

For RELION-style tau2 updates from half-map weights, RECOVAR implements
[`regularization.compute_relion_tau2_from_weights`](../../recovar/reconstruction/regularization.py):

```text
SSNR[s] = FSC[s] / (1 - FSC[s]) * tau2_fudge
sigma2[s] = 1 / (padding_factor^3 * avg_weight[s])
tau2[s] = SSNR[s] * sigma2[s]
```

[`regularization.compute_data_vs_prior`](../../recovar/reconstruction/regularization.py)
computes RELION's resolution-control ratio:

```text
data_vs_prior[s] = avg_weight[s] * tau2_fudge * tau2[s] * padding_factor^3
```

The dense refinement loop records these diagnostics in
`tau2_radial_trajectory`, `tau2_sigma2_trajectory`,
`tau2_avg_weight_trajectory`, `tau2_fsc_used_trajectory`,
`tau2_ssnr_trajectory`, and `data_vs_prior_trajectory`. In contrast, native
InitialModel currently carries `tau2_class`/`data_vs_prior_class` fields but
does not populate equivalent per-iteration trajectories in Python.

### 1-class vs K-class behavior

For `K=1`, the class prior is a scalar and the volume prior is a single tau2
spectrum/reference variance curve. For K-class RELION Class3D, each class has
its own reference image, class distribution, direction-prior table, and
per-class reference-variance table. The K-class E-step normalizes over the
joint class x pose hidden variable, but the M-step reconstructs each class with
that class's own `Ft_y`, `Ft_ctf`, class weight/normalisation, and tau spectrum.

RECOVAR's current dense K-class replay mirrors that layout at the E-step level:
[`k_class.py::run_dense_k_class_em`](../../recovar/em/dense_single_volume/k_class.py)
and `run_local_k_class_em` accept one mean and one `mean_variance` per class,
then normalize evidence over class x pose. The parity harness
[`scripts/run_k_class_parity.py`](../../scripts/run_k_class_parity.py) reads
per-class tau spectra from target or previous RELION model files, reconstructs
diagnostic variants through `_reconstruct_volume_eager`, and compares class
maps after the best permutation.

Native InitialModel is also K-aware in state layout and in
[`dense_adapter.py::class_log_priors_from_state`](../../recovar/em/initial_model/dense_adapter.py):
`pdf_class` becomes class log priors, sparse pass 2 unions significant support
across classes per image, and `vdam_m_step` loops per class. However, because
native InitialModel delegates the gradient reconstruction to `vdam_reconstruct_grad`
and does not write full RELION tau2/model blocks, K-class InitialModel parity is
limited by RELION support selection, bootstrap class assignment, VDAM state, and
the opaque RELION binding's per-class regularization behavior.

### Solvent flattening, spherical masks, grid correction, and BPref weights

RELION GUI InitialModel command construction in
[`scripts/run_ab_initio.py::build_command`](../../scripts/run_ab_initio.py)
adds `--flatten_solvent` by default when `InitialModelJobOptions.do_solvent` is
true, and also adds `--zero_mask`. In native RECOVAR InitialModel, image masking
is configured in `driver.py::_configure_relion_image_mask`, but the final
native `initial_model.mrc` is written directly from `state.Iref[best_class]`;
there is no separate native `relion_align_symmetry` postprocess or documented
Python-side solvent-flattening pass for the final selected InitialModel map in
this branch.

In dense EM/Class3D replay, solvent/mask effects are explicit and easier to
audit. The dense loop reconstructs with `_reconstruct_volume_eager`, which uses
`post_process_from_filter_v2` with the RELION-style spherical mask and gridding
correction enabled by default. The loop also applies RELION-style
`solventFlatten` to real-space maps before the next E-step in the relevant
branches. `scripts/run_k_class_parity.py` deliberately tests variants with and
without spherical mask, solvent mask, grid correction, and `minres_map`, which
is useful because those operations can change map parity even when the same
`Ft_y`/`Ft_ctf` and tau2 are used.

The data-vs-weight part of regularization enters through BPref / reconstruction
weights. In standard reconstruction, `Ft_ctf` is the data weight and tau2 adds
`inv_tau` to its denominator. In native InitialModel, dense E-step accumulators
are converted to RELION BackProjector slabs by
[`layout.py::run_em_output_to_bpref`](../../recovar/em/initial_model/layout.py)
and then handed to `vdam_reconstruct_grad`; that binding sees the BPref weight
array, the current reference, `tau2_fudge_factor`, and `mom1_noise_power`. The
Python-visible dense-to-BPref frame conversion is documented above: dense
InitialModel uses `bp_data *= -N^2` and `bp_weight *= N^4` when
`relion_bpref_frame=True`. Adaptive/replay paths that already emit RELION-frame
BPref rows must not receive that extra conversion.

### Current limitations and known differences

- Native InitialModel does not currently write full RELION `model_class_N`
  tau2/FSC/data-vs-prior tables or `model_pdf_orient_class_N` prior maps in
  `_write_model_star`.
- Native InitialModel's effective tau2 update is inside the RELION
  `vdam_reconstruct_grad` binding; Python code schedules `tau2_fudge` and
  supplies BPref weights/noise power but does not expose the resulting tau2
  spectrum as a first-class per-iteration artifact.
- Dense K-class replay reads per-class tau2/reference variance from RELION
  model files and reconstructs with explicit Python RELION-style regularization;
  native InitialModel does not yet have the same replay-from-model-star startup
  surface.
- Native InitialModel defaults to RELION GUI `--pad 1` for refinement, while
  dense refinement/Class3D replay commonly uses projection/reconstruction
  padding factors from the dense loop and parity harness.
- `minres_map`, spherical masking, solvent flattening, and gridding correction
  are explicit switches in dense replay diagnostics; in native InitialModel,
  parts of the equivalent behavior are either in the RELION binding or not yet
  implemented as separate Python-visible artifact steps.
- The current final native InitialModel output selects the largest `pdf_class`
  and writes it directly; it does not run native `relion_align_symmetry
  --select_largest_class`.

## Detail: Posterior Priors and Metadata Updates

After each M-step, `update_probabilities_from_estep_meta` updates class and
direction priors from dense E-step metadata:

```text
if do_grad and subset_size != -1:
    my_mu = mu        # default 0.9
else:
    my_mu = 0

pdf_class_new =
  my_mu * pdf_class_old
  + (1 - my_mu) * class_posterior_sums / sum_weight
```

If `class_direction_posterior_sums` is present, `pdf_direction` is updated by
the same momentum rule. InitialModel sparse pass 2 uses fine rotation ids for
direction posterior updates via `_initial_model_pass2_layout`; standard local
refinement may instead track parent coarse rotations.

`driver.py::_update_particle_state_from_estep_meta` also updates:

- STAR origin offsets from best translation assignments,
- `_rlnClassNumber` from best class assignments,
- `_rlnMaxValueProbDistribution` from per-image max posterior.

Those values are written into per-iteration data STAR files when iteration
artifacts are enabled.

## Detail: RELION Parity Assumptions

The native path intentionally encodes these RELION GUI InitialModel assumptions:

- `--grad` and `--denovo_3dref` are required.
- MPI is rejected for gradient refinement in `build_command`.
- Refinement padding is `--pad 1`; native execution rejects other padding.
- Default sampling is `healpix_order=1`, `oversampling=1`,
  `offset_range=6`, and `offset_step=2`.
- Default `tau2_fudge` is `4`, with the RELION sigmoid schedule.
- Default `K` is `1`, but the current dense adapter and M-step are K-aware.
- The driver runs in C1 by default and writes the best class as
  `initial_model.mrc`; external `relion_align_symmetry` execution is not wired.
- Native bootstrap currently supports one optics group.
- Native execution supports SPA particle STAR files, not tilt-series.
- The current scoring path uses masked images for scores but unmasked images
  for reconstruction accumulation, matching the dense-engine contract.
- The dense scoring noise uses `sigma2_noise * N^4` via
  `driver.py::_noise_variance_from_sigma2`.
- `e_step.py::minvsigma2_with_dc_zero` documents the RELION DC-exclusion
  convention; use it when debugging direct InitialModel half-spectrum scoring.
- RELION/RECOVAR volume frames differ. Use the helpers documented in
  [`recovar/CLAUDE.md`](../../recovar/CLAUDE.md), not raw MRC loading, when
  comparing RELION and RECOVAR volumes.

## Differences from Standard Class3D / Dense EM

| Topic | Standard dense EM / Class3D | InitialModel / ab initio |
|---|---|---|
| Entry point | Existing map/model refinement | `--grad --denovo_3dref` |
| Initial references | Supplied maps/model state | Random-orientation bootstrap references |
| State | Dense EM means and regularization state | `InitialModelState`: `Iref`, `Igrad1`, `Igrad2`, priors, schedules |
| Dataset split | Gold-standard halves or explicit image groups | VDAM pseudo-halfsets during gradient mode |
| Particle usage | Usually all particles or refinement-specific local subsets | RELION subset schedule from 0.5% to 10% of particles |
| E-step hidden variable | Pose, or class x pose for K-class | Class x pose over dense or adaptive sparse support |
| E-step support | Dense grid, local search, or refinement adaptive support | Coarse significance plus oversampled sparse pass 2 when enabled |
| M-step | Regularized reconstruction / Wiener-style solve | RELION VDAM momentum update and `reconstructGrad` |
| Step size | Not a VDAM gradient step | `grad_current_stepsize` sigmoid schedule, base 0.5 |
| Tau2 | Refinement/model update path | InitialModel `tau2_fudge=4` sigmoid schedule |
| Output | Refined maps/halfmaps/model STAR | Largest class written as `initial_model.mrc` |

Mathematically both paths estimate posterior weights
\(\gamma_i(k,r,t)\). The debugging mistake to avoid is assuming that equal
scores imply equal maps. InitialModel parity also requires equal support,
normalization domain, pseudo-halfset routing, BPref layout/scales, VDAM moment
state, and RELION volume frame.

## Current Limitations

These are code-level limitations in the current branch:

- `run_native_initial_model` rejects `padding_factor != 1`.
- `run_native_initial_model` rejects `run_relion_align_symmetry=True`; the
  command can be built, but native execution does not spawn RELION's symmetry
  tool.
- `run_native_initial_model` rejects tilt-series datasets.
- `_initial_state_from_particles` rejects multiple optics groups because
  `compute_bootstrap_iref_via_cpp` currently takes scalar optics parameters.
- `compute_bootstrap_iref_via_cpp` carries a documented bootstrap
  `padding_factor=2` fixture quirk.
- Sparse pass 2 currently rejects empty pseudo-halfsets.
- The dense adapter is K-aware, but multiclass quality parity still depends on
  RELION support, bootstrap class assignment, and VDAM state parity; do not
  claim K>1 parity from type support alone.
- `gpu_pipeline.py` is a compatibility shim. Any reference to
  `run_iter_gpu_vdam` is stale for this branch.
- The final native `initial_model.mrc` is selected by largest `pdf_class`; it is
  not postprocessed by native `relion_align_symmetry`.

## Debugging Checklist

When RECOVAR diverges from RELION InitialModel, isolate the first mismatch:

1. Confirm the RELION command flags produced by
   `scripts/run_ab_initio.py::build_command`.
2. Confirm the input order: micrograph stable sort, bootstrap first-N selection,
   VDAM shuffle seed `random_seed + iter`, optics stable sort, and pseudo-halfset
   alternation are separate ordering rules.
3. Confirm denovo state values: `Igrad2 = 1 + 1j`, uniform `pdf_class`, uniform
   `pdf_direction`, `ini_high`, and `current_size`.
4. Confirm `Mavg` and `sigma2_noise` before scoring.
5. Confirm bootstrap settings, especially CTF metadata and the current
   `padding_factor=2` bootstrap quirk.
6. Confirm the E-step mode: full dense and sparse pass 2 normalize over
   different supports.
7. Confirm class priors and translation priors are shaped in selected-image
   order after subset packing.
8. Confirm dense-to-BPref conversion: active `r_max`, half-complex slab layout,
   optional z flip, and `-N^2` / `N^4` scales.
9. If score parity is good but BPref parity is bad, inspect support membership,
   posterior normalization, reconstruction group ids, and frame scales before
   changing projection math.
10. If BPref parity is good but map parity is bad, inspect `Igrad1`, `Igrad2`,
    `mom1_noise_power`, `tau2_fudge_factor`, and `grad_current_stepsize`.
11. Compare RELION-produced MRCs only after converting volume conventions with
    the documented RELION helpers.
