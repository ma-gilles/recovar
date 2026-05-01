# RECOVAR / RELION Ab Initio InitialModel Algorithm

Reviewed against the current checkout on 2026-05-01. This note documents the
RELION `InitialModel` / ab-initio gradient-refine path, RECOVAR's corresponding
code, and how it differs from the standard RELION EM / Class3D path.

This is a developer-facing parity map. It intentionally focuses on the math,
state transitions, and code paths that matter when debugging RELION parity.

## Scope and Code Map

Primary RECOVAR code paths:

- `scripts/run_ab_initio.py::build_command`, lines 68-143: RELION GUI-equivalent
  InitialModel command assembly.
- `scripts/run_ab_initio.py::main`, lines 216-229: current real-execution caveat;
  the script assembles commands but does not yet run the full RECOVAR pipeline.
- `recovar/em/initial_model/init.py::initialise_denovo_state`, lines 97-191:
  denovo state initialization.
- `recovar/em/initial_model/avg_unaligned.py::compute_avg_unaligned_and_sigma2`,
  lines 171-240: initial average image and noise-spectrum estimate.
- `recovar/em/initial_model/bootstrap_iref.py::compute_bootstrap_iref_via_cpp`,
  lines 187-262: RELION-style random-orientation bootstrap reference.
- `recovar/em/initial_model/iteration_loop.py::run_vdam_iterations`, lines
  140-205: pure VDAM iteration orchestrator.
- `recovar/em/initial_model/schedules.py`, lines 98-170 and 178-369: RELION
  subset, step-size, and tau2-fudge schedules.
- `recovar/em/initial_model/subset.py::select_vdam_subset`, lines 141-171:
  shuffle-prefix, optics-group stable sort, and pseudo-halfset assignment.
- `recovar/em/initial_model/gpu_pipeline.py::run_iter_gpu_vdam`, lines 560-1040:
  current GPU E-step plus RELION-bound VDAM M-step path.
- `recovar/em/initial_model/m_step.py::vdam_m_step`, lines 232-273, and
  `vdam_m_step_single_class`, lines 79-229: generic Python wrapper for RELION
  VDAM M-step primitives.
- `recovar/em/dense_single_volume/em_engine.py::run_em`, lines 592-688:
  standard dense EM E/M engine contract.
- `recovar/em/dense_single_volume/helpers/scoring.py::_score_rotation_block`,
  lines 11-64: shared Gaussian and normalized-CC score block.
- `recovar/em/dense_single_volume/iteration_loop.py`, lines 1597-1604,
  2411-2450, and 3694-3832: standard RELION EM / Class3D loop conventions.
- `recovar/em/dense_single_volume/k_class.py::run_dense_k_class_em`, lines
  315-413: K-class posterior normalization over class x pose.

Useful existing notes:

- `docs/math/relion_initial_model_em_parity_conventions.md`
- `docs/math/relion_refinement_algorithm.md`
- `docs/math/dense_single_volume_em.md`
- `docs/math/relion_updateSSNR_algorithm_2026_04_25.md`

RELION source entry points mirrored by the RECOVAR comments and tests:

- `pipeline_jobs.cpp::initialiseInimodelJob` / `getCommandsInimodelJob`
- `ml_model.cpp::initialiseFromImages`
- `ml_optimiser.cpp::calculateSumOfPowerSpectraAndAverageImage`
- `ml_optimiser.cpp::iterate`
- `ml_optimiser.cpp::updateSubsetSize`, `updateStepSize`, `updateTau2Fudge`
- `ml_optimiser.cpp::storeWeightedSums`
- `backprojector.cpp::reweightGrad`, `getFristMoment`, `getSecondMoment`,
  `applyMomenta`, `reconstructGrad`

## High-Level Algorithm

RELION InitialModel is not "standard EM from a supplied map." It is a denovo
gradient-refine procedure:

1. Build a zero/reference state and VDAM moment buffers.
2. Estimate initial noise spectra and an average image from masked, unaligned
   particles.
3. Bootstrap initial 3D references by assigning random orientations to particles
   and reconstructing low-resolution maps.
4. Run many VDAM mini-batch iterations. Each iteration selects a subset of
   particles, performs an E-step over class/orientation/translation hypotheses,
   accumulates RELION `BackProjector` sufficient statistics, updates first and
   second gradient moments, and applies a momentum-normalized reference update.
5. At the end of the RELION GUI job, run `relion_align_symmetry` to select the
   largest class, align to the requested symmetry, and write `initial_model.mrc`.

The GUI-equivalent RELION command is encoded in
`scripts/run_ab_initio.py::build_command`:

```text
relion_refine --grad --denovo_3dref --pad 1 --auto_sampling \
  --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 2 \
  --tau2_fudge 4 --K <nr_classes> ...
relion_align_symmetry --apply_sym --select_largest_class ...
```

Those flags come from RELION's InitialModel GUI setup, not from the standard
AutoRefine/Class3D defaults. In particular, InitialModel uses `--grad` and
`--denovo_3dref`, rejects MPI in the GUI path, defaults to 200 VDAM mini-batches,
defaults to `K=1`, and runs in C1 before optional symmetry alignment.

## Forward Model and Objective

For particle image \(y_i\), class \(k\), rotation \(r\), translation \(t\), CTF
\(C_i\), projection \(P_r\), and volume \(\mu_k\):

```text
y_i = S_t C_i P_r mu_k + eps_i,
eps_i ~ complex Gaussian(0, Sigma_i).
```

The hidden variable is \(z=(k,r,t)\). The standard EM posterior is

```text
gamma_i(k,r,t) =
  p(k,r,t) exp(score_i(k,r,t)) /
  sum_{k',r',t'} p(k',r',t') exp(score_i(k',r',t')).
```

The Gaussian score used by the standard dense engine is

```text
score_i(k,r,t) =
  -0.5 * || y_i - S_t C_i P_r mu_k ||^2_{Sigma_i^-1}
```

up to image-only constants. `helpers/scoring.py::_score_rotation_block` expands
this into a cross term and a projection norm term:

```text
cross = -2 Re(conj(shifted_image) @ projected_volume)
norm  = (CTF^2 / sigma2) @ |projected_volume|^2
score = -0.5 * (cross + norm)
```

The M-step sufficient statistics for a class are

```text
Ft_y(k) =
  sum_i,r,t gamma_i(k,r,t)
    P_r^* C_i Sigma_i^-1 S_t^* y_i

Ft_ctf(k) =
  sum_i,r,t gamma_i(k,r,t)
    P_r^* C_i Sigma_i^-1 C_i P_r.
```

Standard EM reconstructs by a regularized filtered backprojection / Wiener solve:

```text
mu_k_new ~= Ft_y(k) / (Ft_ctf(k) + prior_precision(k)).
```

InitialModel shares the same likelihood and posterior logic, but its M-step does
not simply replace `mu` with a Wiener map. It routes the accumulated
`BackProjector` data through RELION's VDAM gradient-momentum update.

## Denovo Initialization

`recovar/em/initial_model/init.py::initialise_denovo_state` mirrors RELION's
`fn_ref == "None"` initialization:

- `Iref[k]` starts as a zero real-space reference.
- `Igrad1` first-moment buffers start at zero; there are `2K` slots when
  pseudo-halfsets are active.
- `Igrad2[k]` starts at `Complex(1, 1)`, matching RELION's
  `MOM2_INIT_CONSTANT` behavior.
- `pdf_class` is uniform over classes.
- `pdf_direction` is uniform over class x direction.
- The initial resolution uses RELION's `0.07 * ori_size` rule; for example, the
  code computes `current_size = 2 * (ROUND(0.07 * ori_size) + 10)` before
  clamping to the box.

`avg_unaligned.py::compute_avg_unaligned_and_sigma2` implements the first
data-dependent preprocessing step:

```text
Mavg = average of masked images
sigma2_noise[group, shell] =
  average_particle_power[group, shell] / 2
  - average_image_power[shell] / 2
```

with RELION's correction for non-positive shells by copying the nearest positive
neighbor.

`bootstrap_iref.py::compute_bootstrap_iref_via_cpp` is the most RELION-faithful
reference bootstrap path. For each selected particle, RELION resets the RNG with
`random_seed + part_id`, draws random Euler angles, masks the image, FFTs it in
RELION's normalized convention, multiplies by CTF, and inserts it into
`BPref[iclass]`. Then `BPref.reconstruct(..., do_map=false)` reconstructs a
low-resolution reference, followed by the InitialModel low-pass filter.

## VDAM Iteration Flow

`iteration_loop.py::run_vdam_iterations` is the pure orchestration layer:

1. Compute phase lengths from `grad_ini_frac`, `grad_fin_frac`, and `nr_iter`.
2. For each iteration, update the subset-size, step-size, and tau2-fudge
   schedules.
3. Shuffle particles with seed `random_seed + iter`.
4. Select the current subset and pseudo-halfset ids.
5. Run the caller-provided E-step adapter to produce `VdamAccumulator` objects.
6. Run `vdam_m_step`.
7. Write iteration artifacts.

The schedule code mirrors RELION scalar behavior:

- `default_subset_sizes_for_3d_initial_model`: `grad_ini_subset_size` is
  clamped `round(0.005 N)` in `[200, 5000]`; `grad_fin_subset_size` is clamped
  `round(0.1 N)` in `[1000, 50000]`.
- `compute_subset_size`: the subset grows across the in-between phase and
  becomes `-1` for all particles when gradient mode stops, convergence triggers,
  the requested subset exceeds the available particles, or on the last
  multi-class iteration.
- `compute_stepsize`: 3D InitialModel defaults to a base step size of `0.5`
  with a `0.9 / stepsize` sigmoid scheme.
- `compute_tau2_fudge`: 3D InitialModel defaults to `tau2_fudge=4` with a
  sigmoid transition from about `1` toward `4`.

Particle subset selection is not just "take random N particles." In
`subset.py::select_vdam_subset`, RECOVAR mirrors RELION's sequence:

```text
shuffle full particle order with seed random_seed + iter
take prefix of length subset_size
stable-sort that prefix by optics group
assign pseudo-halfsets by alternating 0, 1, 0, 1, ...
```

This ordering matters because `storeWeightedSums` uses particle parity to route
backprojections into pseudo-halfsets when gradient refinement is active.

## E-Step Details

The InitialModel parity E-step currently lives in
`gpu_pipeline.py::run_iter_gpu_vdam`.

There are two E-step modes:

- `estep_mode="dense"`: diagnostic mode that calls `run_em` over the explicit
  dense rotation x translation grid. This is useful for algebra and scaling
  checks, but it is not RELION InitialModel parity mode.
- `estep_mode="relion_adaptive"`: parity mode. It scores a coarse RELION grid,
  retains image-specific significant support, then scores only oversampled child
  hypotheses in sparse pass 2.

The adaptive path uses the same conceptual posterior as standard EM, but it
changes the support:

```text
coarse pass: score all coarse (r,t), find significant parents
fine pass:   score children of significant parents only
posterior:   normalize over the retained fine support
M-step:      accumulate only retained/support-weighted hypotheses
```

This is the dominant difference from a naive full fine-grid implementation. A
full fine grid can match RELION scores for individual candidates and still fail
BPref parity because it assigns posterior mass to candidates RELION never
backprojects.

Important InitialModel-specific E-step conventions in `run_iter_gpu_vdam`:

- `padding_factor=1` by default for GUI InitialModel scoring.
- RELION-adaptive mode consumes `sigma2_noise` in RELION's `Minvsigma2` frame
  (`noise_scale = 1.0`), while dense RECOVAR mode uses the historical `N^4`
  noise conversion.
- RELION-adaptive mode builds `Projector::data` / PPref via the RELION binding
  (`compute_fourier_transform_map`) rather than scoring from RECOVAR's centered
  full Fourier grid.
- The returned projector data is multiplied by `ori_size` to match RELION dumped
  PPref/Fref amplitudes.
- The adaptive image correction is `-1 / N^2`; in that mode the final BPref
  data and weights are already in RELION's frame.
- The translation prior path intentionally follows the current RELION replay
  convention:

```text
log p(t_px) = -0.5 * ||t_px||^2 * pixel_size^4 / sigma_offset_A^2.
```

This `pixel_size^4` factor is a source-level parity convention, not the clean
statistical expression one would choose from first principles.

## VDAM M-Step Details

`m_step.py::vdam_m_step_single_class` wraps RELION's gradient primitives:

1. `reweightGrad`: divide raw `data` by `max(1, weight)` inside the active
   Fourier radius.
2. `getFristMoment`: update first moment
   `m1 <- lambda1 * m1 + (1 - lambda1) * reweighted_data`, except RELION copies
   into an all-zero moment buffer on first use.
3. `getSecondMoment`: update second moment from the normalized halfset
   difference, using `lambda2`.
4. `applyMomenta`: average pseudo-halfset first moments, divide by
   `sqrt(m2)`, and estimate per-shell noise from halfset moment differences.
5. `reconstructGrad`: compute the current reference's RELION projector, estimate
   FSC/noise if requested, and apply a gradient update controlled by
   `grad_current_stepsize` and `tau2_fudge_factor`.

RECOVAR uses RELION's defaults:

```text
lambda1 = 0.9
lambda2 = 0.999
```

The active GPU VDAM path converts `Ft_y` / `Ft_ctf` into RELION BPref layout via
`run_em_output_to_bpref`. In adaptive InitialModel mode, no extra frame scale is
applied:

```text
bp_data_scale   = 1
bp_weight_scale = 1
```

In dense diagnostic mode, the historical conversion remains:

```text
bp_data_scale   = -N^2
bp_weight_scale =  N^4
```

This distinction is critical. A score-parity fix in one Fourier/noise frame can
produce a wrong M-step if the accumulator is converted with the other frame's
scale/sign convention.

## Standard EM / Class3D Path

The standard RELION-style EM loop in `dense_single_volume/iteration_loop.py`
starts from an existing map/model state and performs gold-standard refinement:

- two independent half-set references,
- projection and reconstruction padding factor `2`,
- current-size growth from FSC/data-vs-prior logic,
- adaptive oversampling or exact local search as the refinement progresses,
- low-resolution half-map joining before reconstruction,
- per-iteration FSC, tau2, noise, sigma-offset, direction-prior, and
  convergence updates.

The dense engine (`em_engine.py::run_em`) streams over image batches and rotation
blocks with two passes:

```text
pass 1: compute scores and logsumexp normalizers
pass 2: recompute scores, normalize posterior weights, accumulate Ft_y/Ft_ctf
```

Class3D adds a class axis to the hidden state. `k_class.py::run_dense_k_class_em`
first computes per-class log evidence, then normalizes each class against the
global class x pose evidence before accumulating class-specific `Ft_y` and
`Ft_ctf`.

Mathematically, standard Class3D and InitialModel both estimate
\(\gamma_i(k,r,t)\). The implementation contracts differ:

- Standard Class3D updates class maps by the regularized EM reconstruction path.
- InitialModel updates references through VDAM momenta and `reconstructGrad`.
- Standard Class3D assumes mature model metadata and half-set state.
- InitialModel creates denovo references, noise spectra, gradient moment buffers,
  pseudo-halfsets, and increasing particle subsets.

## Key Differences from Standard EM / Class3D

| Topic | Standard EM / Class3D | InitialModel / ab initio |
|---|---|---|
| Entry point | Existing map/model refinement | `--grad --denovo_3dref` |
| Initialization | Supplied initial reference and model metadata | Zero refs, initial noise estimate, random-orientation bootstrap refs |
| Padding | RELION replay uses projection/reconstruction pad 2 | GUI InitialModel command uses `--pad 1`; bootstrap has additional BPref quirks |
| Dataset split | Gold-standard halfsets | VDAM pseudo-halfsets while `do_grad` is active |
| Particle usage | Usually all particles per iteration, subject to halfsets | Iteration-dependent mini-batch subset schedule |
| E-step support | Dense, adaptive, or local depending on refinement state | RELION-adaptive coarse support plus oversampled sparse pass for parity |
| M-step | Wiener/regularized reconstruction from `Ft_y`, `Ft_ctf` | `reweightGrad -> first moment -> second moment -> applyMomenta -> reconstructGrad` |
| Step size | Not a VDAM gradient step in the same sense | `grad_current_stepsize` schedule, default base 0.5 |
| Tau2 fudge | Auto-refine default is usually 1 unless overridden | GUI InitialModel default is 4 with a sigmoid schedule |
| Class behavior | K-class posterior is implemented in dense/local class wrappers | Generic state/M-step are K-aware, but active GPU VDAM iteration is K=1-only today |
| Final map | Current refined half/full maps | `relion_align_symmetry --select_largest_class` output |

## Current RECOVAR Parity Caveats

The following caveats are confirmed in current code and should be treated as
debugging constraints:

- The public `scripts/run_ab_initio.py` execution path is not wired to run a full
  RECOVAR InitialModel job. It builds the RELION-equivalent command and returns a
  "real-execution path not yet wired" message in `main`.
- `InitialModelState` and `m_step.py::vdam_m_step` are written for generic `K`,
  including `2K` pseudo-halfset first-moment slots.
- The active GPU VDAM iteration path in
  `gpu_pipeline.py::run_iter_gpu_vdam` hardcodes `K = 1` around line 955, creates
  one-class `Igrad1` / `Igrad2` buffers, and returns one reference update.
- `scripts/run_cryobench_ribosembly_parity_slurm.sh`, lines 215-228, records
  multi-class InitialModel parity as unsupported for this reason.
- Dense full-grid InitialModel mode is diagnostic. RELION parity should use
  `estep_mode="relion_adaptive"` unless a fixture proves RELION used a different
  path.
- Adaptive InitialModel and dense RECOVAR mode use different Fourier/noise frames;
  do not mix their BPref scale/sign conversions.
- RELION volume/frame conversions remain a high-risk source of false parity
  failures. Use the existing helpers documented in `recovar/CLAUDE.md`; do not
  raw-load RELION MRCs for comparisons.

## Developer Debugging Checklist

When RECOVAR InitialModel diverges from RELION, check these in order:

1. Confirm the exact RELION command flags: `--grad`, `--denovo_3dref`, `--pad 1`,
   `--auto_sampling`, `--oversampling 1`, `--healpix_order 1`, `--offset_range 6`,
   `--offset_step 2`, and `--tau2_fudge 4` for GUI defaults.
2. Confirm particle order. RELION sorted order, STAR row order, stack index order,
   optics-group stable sorting, and pseudo-halfset alternation are distinct.
3. Confirm denovo initialization: `Igrad2` must be `1+1j`, not `1+0j`.
4. Confirm `current_size`, `r_max`, and pass-1/pass-2 Fourier windows from the
   RELION model/sampling STARs.
5. Confirm the E-step path is RELION-adaptive, not dense full-grid, when comparing
   to RELION InitialModel output.
6. Confirm projector frame: adaptive mode should score against RELION PPref
   (`compute_fourier_transform_map`), not a RECOVAR full Fourier grid.
7. Confirm noise frame: adaptive InitialModel uses `noise_scale=1`; dense mode
   uses `N^4`.
8. Confirm translation-prior sigma comes from the correct previous model metadata
   and that the current replay path uses the documented pixel-size behavior.
9. Confirm BPref conversion: adaptive mode should not apply the dense `-N^2` /
   `N^4` conversion.
10. If score parity is good but BPref parity is bad, suspect support membership,
    posterior normalization, particle ordering, or frame conversion before
    suspecting the projection operator.
