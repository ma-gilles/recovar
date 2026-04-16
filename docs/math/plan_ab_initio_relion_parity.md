# Plan: RELION-parity ab-initio 3D reconstruction

Status: DRAFT — awaiting user review before implementation.
Last updated: 2026-04-07.

## 1. Research summary

### 1.1 What exists in recovar today

**Abbreviated answer: recovar does NOT currently have a working ab-initio
path.** There is a dead stub called `SGDState` but nothing consumes it.

Detailed inventory of SGD/ab-initio-adjacent code that was found:

- `recovar/em/states.py::SGDState` (lines 58–140).  A Python class with
  `E_step` / `M_step` that implements a plain momentum-GD update (Polyak
  momentum, `mu = 0.9`, step `= 1 / max(|Ft_CTF|)`, clips to `±10 σ`).  It is
  a loose port of a much older recovar experiment (`~/recovar/recovar/em/states.py`
  at revision pre-refactor) and has several issues:
  - No second-moment / Adam-style normalisation.
  - No FSC-gated update.
  - No split half-maps.
  - Hard-coded `mu = 0.9` and `step * 0.1` ("`There is a necessary 0.1 that
    shouldn't be there`" — a leftover debug comment in the old version).
  - No "random init volume" generation routine.
- `recovar/em/states.py::get_default_sgd_options()` returns a dict with
  `minibatch_size=30`, `steps_size="hess"`, `mu=0.9`.  The function is
  exported from `recovar.em.__init__` but **nothing calls it**, and in the
  old version at `~/recovar/recovar/em/states.py` it has the bug `return`
  with no value.
- `recovar/em/iterations.py::E_M_batches_2` has a minor branch at lines
  39–43: when `state_obj.name == "SGD"`, it uses `state_obj.sgd_batchsize`
  as the image batch size.  This is the ONLY runtime connection to the
  SGD state.
- `recovar/em/iterations.py::split_E_M_v2` at line 79 also has
  `use_fsc_prior = state_objs[0].name != "SGD"` and falls back to a simple
  power-spectrum-based prior for SGD, but the function was never actually
  exercised with an `SGDState` in any test or script.  The unit tests in
  `tests/unit/test_em_iteration.py` only check `hasattr(em, "SGDState")`
  and `tests/unit/test_em_states.py` only tests the constructor — no
  end-to-end SGD execution.

**Other things that were searched for and NOT found:**

- No `ab_initio`, `ab-initio`, `denovo`, `de_novo`, `random_init`,
  `init_random_volume`, `from_scratch`, `subset_em` anywhere in
  `recovar/`, `scripts/`, or `recovar/commands/`.
- `recovar/em/dense_single_volume/` (the clean RELION-parity extraction)
  does NOT reference `SGDState` or any SGD code at all.  Its
  `_refine_relion_mode` is strictly "polish-from-init" and requires a
  low-pass filtered initial volume.
- `recovar/em/heterogeneity.py` has no ab-initio code either (it is
  purely the low-rank heterogeneity path on top of a fixed mean).
- `scripts/run_full_refinement.py` has no `--ab_initio` flag; it loads
  `reference_init.mrc` unconditionally.

**Conclusion: SGDState is effectively dead code.  A working ab-initio
path would have to be built from scratch.  It does NOT share algorithm
or schedule with RELION.**

### 1.2 What RELION 5 actually does for ab-initio

RELION 5 dropped the "SGD" naming internally — the code now calls it
`gradient_refine` / `grad_*`, but the ab-initio flow is still the same
basic Adam-like optimiser described in Scheres (2020).  The CLI entry
point is `relion_refine --denovo_3dref --grad` (Class3D with a single
class, starting from `fn_ref = "None"`).

Key source files:

- `relion/src/ml_optimiser.h` lines 354–444 declare the gradient-refine
  fields: `gradient_refine`, `do_grad`, `grad_ini_iter`, `grad_fin_iter`,
  `grad_inbetween_iter`, `grad_ini_subset_size`, `grad_fin_subset_size`,
  `grad_ini_resol`, `grad_fin_resol`, `mu`, `grad_stepsize`,
  `grad_stepsize_scheme`, `tau2_fudge_arg`, etc.
- `relion/src/ml_optimiser.cpp` lines 833–876 parse the CLI flags.
- Lines 1223–1226 set the defaults `grad_ini_resol = 35.0 A`,
  `grad_fin_resol = 15.0 A`.
- Line 1983 sets `nr_iter = 200` as the default for `gradient_refine`.
- Lines 839–854 convert the `*_frac` defaults into per-phase iteration
  counts: `grad_ini_frac = 0.3`, `grad_fin_frac = 0.2`, so the 200 iters
  split as `60 ini / 100 inbetween / 40 fin`.
- Lines 2521–2563 set default subset sizes for ab-initio (`is_3d_model`):
  - ini: `max(min(N * 0.005, 5000), 200)`
  - fin: `max(min(N * 0.1, 50000), 1000)`
- Line 863 sets momentum `mu = 0.9` (default).
- Lines 10032–10079 `updateStepSize()`: default step size for
  `is_3d_model` is `0.5`, with a sigmoid schedule that starts at
  `inflate * 0.5` (inflate = `0.9 / 0.5 = 1.8`) and decays to `0.5`
  across the inbetween phase.
- Lines 10081–10133 `updateTau2Fudge()`: default `tau2_fudge = 4` for
  ab-initio, with sigmoid decay from `4 * deflate = 16` down to `4`.
- Line 871 `do_init_blobs`: in 2D classification, initialise with random
  Gaussian blobs.  For 3D initial models this is skipped; the init volume
  is built by random-orientation backprojection (see §1.3 below).

Key algorithmic pieces in RELION:

- `relion/src/ml_model.cpp::initialiseFromImages` (lines 851–1148):
  when `fn_ref == "None"` and `_is_3d_model == true`, initialise `Iref`
  to **all zeros** (line 1100), initialise `Igrad1` (first-moment
  accumulator, complex) to zeros, `Igrad2` (second-moment, real packed
  as complex) to `MOM2_INIT_CONSTANT = 1`.
- `relion/src/ml_optimiser.cpp::calculateSumOfPowerSpectraAndAverageImage`
  (lines 2749–3098): loops over the first ~`max(5*nr_classes,
  minimum_nr_particles_sigma2_noise * nr_optics_groups)` particles, draws
  a random (rot, tilt, psi) uniformly per particle (lines 2993–3004),
  applies CTF², and backprojects into `wsum_model.BPref[iclass]`.
- `relion/src/ml_optimiser.cpp::setSigmaNoiseEstimatesAndSetAverageImage`
  (lines 3101–3192): solves the initial random-orientation
  backprojection with `BPref.reconstruct()` to get the initial reference
  volume.  This is a Wiener-filtered gridding reconstruction — NOT
  gaussian noise, NOT a sphere.
- `relion/src/backprojector.cpp::reconstructGrad` (lines 1995–2134): the
  per-iteration gradient step.  The math is:
  ```
  reweightGrad():  data = Fweighted_backproj / max(1, weight)     # = Wiener grad
  mom1  = mu * mom1  + (1 - mu) * data                            # first moment (EMA)
  mom2  = mu * mom2  + (1 - mu) * |half1 - half2|^2 / |avg|^2     # second moment
  g     = mom1 / (sqrt(mom2) + 1e-12)                             # Adam-style normalise
  Fgrad = fsc * g - (1 - fsc) / tau2_fudge * PPref.data           # RL-like update
  PPref.data += stepsize * Fgrad
  Iref = iFFT(PPref.data)                                         # → real space
  ```
  This is a hybrid Adam + Richardson-Lucy-style scheme.  The second-moment
  uses pseudo-halfset disagreement rather than `g²` — a RELION-specific
  design choice.
- `relion/src/backprojector.cpp::getFristMoment` (lines 1884–1914):
  `mom1 = lambda * mom1 + (1 - lambda) * data`; on the first call
  (when `mom1.sum() == 0`), `mom1 = data` directly.  `lambda = mu = 0.9`
  by default (passed in from `ml_optimiser.cpp` line 5385 implicitly).
- `relion/src/exp_model.cpp::randomiseParticlesOrder` (lines 406–461):
  re-shuffles particles each iteration with `seed + iter`, then the
  first `subset_size` particles are used.

### 1.3 The subtle init-volume detail

RELION's initial volume for `--denovo_3dref` is **not** a sphere or
gaussian noise — it is the Wiener-filtered reconstruction from
backprojecting ~1000 particles at randomly-sampled orientations.  The
resulting volume is extremely blurry (essentially the radial average of
all particles, smeared around the origin), but it has the right power
spectrum.  Then the low-pass filter is applied at
`0.07 * ori_size` digital-frequency = ~9 px for a 128-box.  For a
4.25 Å/px box, that is ~60 Å resolution.

This "random backprojection" init is reported to work much better than
random noise or a sphere in practice because:

1. It has the correct low-frequency content.
2. It is rotationally averaged → spherical symmetry breaks gracefully
   during the first few EM iterations.
3. The E-step sees a meaningful signal at low frequencies from iteration 1.

### 1.4 Side-by-side parameter comparison

| Parameter            | RELION default     | recovar SGDState | Notes |
|----------------------|--------------------|-------------------|-------|
| Total iters          | `nr_iter=200`      | (N/A — stub)      | RELION splits into ini/inbetween/fin |
| ini iters            | `60`  (30 %)       | —                 | `grad_ini_frac=0.3` |
| inbetween iters      | `100` (50 %)       | —                 | — |
| fin iters            | `40`  (20 %)       | —                 | `grad_fin_frac=0.2` |
| ini subset size      | `max(min(N·0.005,5000),200)` | `sgd_batchsize=100` | RELION scales with dataset; recovar hard-codes |
| fin subset size      | `max(min(N·0.1,50000),1000)` | `sgd_batchsize=100` | RELION ramps linearly during inbetween phase |
| `mu` (momentum)      | `0.9`              | `0.9`             | Match |
| step size            | `0.5` (3D init)    | `1 / max|Ft_CTF|` (hardcoded + ×0.1) | Different |
| step size scheme     | `inflate-step` sigmoid, starts at `0.9`, ends at `0.5` | plain | Different |
| `tau2_fudge`         | `4`                | — (uses `mean_variance`) | Different |
| `tau2_fudge` scheme  | sigmoid from `16` → `4` | constant        | Different |
| ini resolution       | `35.0 Å`           | —                 | RELION uses this for the low-pass on init |
| fin resolution       | `15.0 Å`           | —                 | RELION's final window limit in inbetween phase |
| second-moment (Adam) | pseudo-halfset variance | — (none)    | Different |
| init volume          | backproject random-orientation subset | — (external MRC) | Different |
| FSC gating           | `data * fsc - (1 - fsc) / tau2 * prev` | — (none) | Different |
| Halfset split        | pseudo-halfsets for mom2 only (single reference shared) | — (SGD was single-volume) | Different |
| Re-shuffle subset    | every iteration with `seed + iter` | — (unused) | — |

**Conclusion: recovar's `SGDState` is structurally very different from
RELION's gradient refine, so there is no minimal-patch path.  We need a
new routine.**

## 2. Decision: Path **(C)** — write a new module

Three paths were considered:

- **(A) Parameter alignment only.**  Not viable.  `SGDState` lacks all of:
  second-moment accumulator, FSC gating, pseudo-halfsets, random
  per-iteration subsets, volume backprojection init, resolution schedule.
- **(B) Extend `SGDState`.**  Possible but would bolt many new fields
  onto a class in `recovar/em/states.py` that is consumed by
  `E_M_batches_2` / `split_E_M_v2` (the OLD path), not by
  `dense_single_volume/`.  Mixing those code paths again would defeat
  the point of the dense_single_volume refactor.  Also risks breaking
  the existing (untested, stub) SGD state.
- **(C) New module under `dense_single_volume/`.**  Clean: reuses the
  JIT-compiled engine (`run_em_v2`), the adaptive pass (`adaptive.py`),
  and the shared types.  Stays in the new architecture.  The polish-
  from-init path in `refine.py::_refine_relion_mode` is **not touched**.

**Chosen path: (C).  New files live under
`recovar/em/dense_single_volume/ab_initio.py` and
`recovar/em/dense_single_volume/gradient_update.py`.**

## 3. Architecture

### 3.1 New files

```
recovar/em/dense_single_volume/
├── ab_initio.py           # Top-level ab-initio entry point (~400 lines)
├── gradient_update.py     # Adam-like Fourier gradient update (~180 lines)
└── random_init.py         # Random-orientation backprojection init (~120 lines)
```

Existing files that are **read but not modified**:

- `engine_v2.py::run_em_v2` — reused verbatim for E+M on each subset.
- `adaptive.py` — reused for adaptive oversampling support (optional).
- `refine.py::_refine_relion_mode` — not touched.
- `fourier_window.py::quantize_current_size` — reused.
- `convergence.py` — `RefinementState` and friends are NOT reused
  (ab-initio has its own iteration/schedule logic, not convergence-based).
- `types.py::MeanStats` — reused for `(Ft_y, Ft_ctf)` accumulation.
- `recovar/reconstruction/relion_functions.py::post_process_from_filter_v2`
  — reused for the Wiener solve inside the Adam update.
- `recovar/em/sampling.py::get_rotation_grid` — reused for the fixed
  HEALPix grid at each resolution level.
- `recovar/reconstruction/noise.py::estimate_initial_noise_spectrum_from_unaligned_images`
  — reused to produce the initial `sigma2_noise`.

### 3.2 New function signatures

#### `random_init.py`

```python
def random_orientation_init_volume(
    experiment_dataset,
    *,
    n_particles: int = 1000,
    seed: int = 42,
    ini_high_angstrom: float = 60.0,
    disc_type: str = "linear_interp",
) -> jax.Array:
    """Generate RELION-style ab-initio init volume.

    Mirrors relion/src/ml_optimiser.cpp::calculateSumOfPowerSpectraAndAverageImage
    + setSigmaNoiseEstimatesAndSetAverageImage: backproject the first
    ``n_particles`` images at uniformly-random (rot, tilt, psi), then
    Wiener-solve.

    Returns a flat centered-FT volume (matching the convention in
    ~/CLAUDE.md and recovar/em/CLAUDE.md).

    Internally uses `sum_up_images_fixed_rots_eqx` (same kernel as
    ``run_em_v2``'s M-step) with a one-hot probability matrix where each
    image has weight 1 at exactly one (randomly-chosen) rotation.
    """


def low_pass_ft_volume(
    volume_ft: jax.Array,
    volume_shape: tuple,
    voxel_size: float,
    resolution_angstrom: float,
    edge_width_pixels: int = 5,
) -> jax.Array:
    """Apply a RELION-style raised-cosine low-pass filter to a centered-FT
    volume.  Matches ``initialLowPassFilterReferences`` in
    ``relion/src/ml_optimiser.cpp`` (lines 3194+).
    """
```

#### `gradient_update.py`

```python
from typing import NamedTuple

import jax
import jax.numpy as jnp


class GradientState(NamedTuple):
    """Adam-like optimiser state for RELION ab-initio.

    All arrays are flat centered-FT (complex64) of length volume_size.
    Mirrors relion/src/ml_model.cpp::Iref + Igrad1 + Igrad2, but we
    collapse the two pseudo-halfset Iref's into the two halfset means
    tracked by split-EM.

    Attributes:
        iref_halves: tuple of 2 flat centered-FT arrays (pseudo-halfsets).
        mom1_halves: first-moment EMAs, one per halfset (same layout as iref).
        mom2_shared: second-moment EMA (shared across halfsets because it
            accumulates halfset disagreement).  Real values stored as complex
            for alignment with RELION's MultidimArray<Complex>.
        step: step count (for scheduling).
    """

    iref_halves: tuple
    mom1_halves: tuple
    mom2_shared: jax.Array
    step: int


def gradient_update_step(
    state: GradientState,
    Ft_y_halves: tuple,
    Ft_ctf_halves: tuple,
    volume_shape: tuple,
    *,
    step_size: float,
    tau2_fudge: float,
    mu: float,
    mean_variance: jax.Array,   # RELION's tau2
    fsc: jax.Array,             # half-map FSC (1D radial)
    volume_size_radial_map: jax.Array,  # int32 shell-index per voxel
    min_resol_shell: int,
    kernel: str = "triangular",
) -> GradientState:
    """Single gradient step for ab-initio.

    Performs the full RELION update in flat-FT space:

        1. For each halfset k:
           a. data_k = reweight(Ft_y[k], Ft_ctf[k]) via a Wiener solve
              (uses post_process_from_filter_v2 with tau2_fudge applied).
           b. mom1[k] = mu * mom1[k] + (1 - mu) * data_k    (EMA)
        2. diff = mom1[0] - mom1[1]
           avg  = (mom1[0] + mom1[1]) / 2
           mom2 = mu * mom2 + (1 - mu) * |diff|^2 / (|avg|^2 + 1e-12)
        3. For each halfset k:
           g_k = mom1[k] / (sqrt(mom2.real) + 1e-12)
           Fgrad_k = fsc[r] * g_k - (1 - fsc[r]) / tau2_fudge * iref_halves[k]
           iref_new[k] = iref_halves[k] + step_size * Fgrad_k
           (with fsc[r] = fsc[radial_index_of_voxel_r], and voxels beyond
           min_resol_shell use fsc = 1)

    This is a single JAX-jitted kernel and runs on GPU.
    """


def initial_gradient_state(
    init_volume_ft: jax.Array,
    volume_size: int,
    dtype=jnp.complex64,
) -> GradientState:
    """Zero-initialise `mom1_halves` and set `mom2_shared = 1 + 1j` to match
    ``MOM2_INIT_CONSTANT`` from ``relion/src/ml_model.cpp::23``.
    Both halfsets start from the same init volume.
    """
```

#### `ab_initio.py`

```python
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from .gradient_update import GradientState, gradient_update_step, initial_gradient_state
from .random_init import random_orientation_init_volume, low_pass_ft_volume


def ab_initio_single_volume(
    experiment_datasets: Sequence,     # list of two halfset datasets
    rotations: np.ndarray,             # HEALPix grid at init_healpix_order
    translations: jnp.ndarray,
    *,
    # --- Algorithm controls ---
    init_volume: jnp.ndarray = None,   # if None, use random-orientation init
    disc_type: str = "linear_interp",
    seed: int = 42,

    # --- RELION schedule defaults (all match ml_optimiser.cpp) ---
    n_total_iter: int = 200,
    ini_frac: float = 0.3,
    fin_frac: float = 0.2,
    ini_subset_size: int = None,       # auto: max(min(N*0.005, 5000), 200)
    fin_subset_size: int = None,       # auto: max(min(N*0.1, 50000), 1000)
    ini_resol_angstrom: float = 35.0,
    fin_resol_angstrom: float = 15.0,
    step_size: float = 0.5,
    step_size_inflate: float = 1.8,    # = 0.9 / 0.5
    mu: float = 0.9,
    tau2_fudge: float = 4.0,
    tau2_fudge_inflate: float = 4.0,
    grad_min_resol_angstrom: float = 20.0,
    init_high_angstrom: float = 60.0,  # low-pass filter on init

    # --- Engine controls (reused from refine_single_volume) ---
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    healpix_order: int = 2,

    # --- Output ---
    save_intermediates_dir: str = None,
) -> dict:
    """RELION-parity ab-initio 3D reconstruction.

    Pseudocode:

        1. Derive auto defaults for ini/fin subset sizes from dataset size.
        2. Derive iteration counts:
             ini_iter = round(ini_frac * n_total_iter)
             fin_iter = round(fin_frac * n_total_iter)
             inbetween_iter = n_total_iter - ini_iter - fin_iter
        3. Generate initial state:
             if init_volume is None:
                 init_volume = random_orientation_init_volume(
                     ds, n_particles=max(5, 1000),
                     ini_high_angstrom=init_high_angstrom,
                 )
             else:
                 init_volume = low_pass_ft_volume(
                     init_volume, ds.volume_shape, ds.voxel_size,
                     init_high_angstrom,
                 )
             state = initial_gradient_state(init_volume, ds.volume_size)
             noise_variance = estimate_initial_noise_spectrum_from_unaligned_images(...)
             mean_variance = from_init_volume_power_spectrum(init_volume)
        4. For iter in 1..n_total_iter:
             # Schedule
             subset_size = _subset_size_schedule(iter, ini_iter, inbetween_iter,
                                                 ini_subset_size, fin_subset_size)
             current_size = _current_size_schedule(iter, ini_iter, inbetween_iter,
                                                   ini_resol_angstrom, fin_resol_angstrom,
                                                   ds.voxel_size, ds.grid_size)
             curr_tau2 = _tau2_schedule(iter, ini_iter, inbetween_iter,
                                        tau2_fudge, tau2_fudge_inflate)
             curr_step = _step_schedule(iter, ini_iter, inbetween_iter,
                                        step_size, step_size_inflate)

             # Re-shuffle + pick subset
             rng = np.random.RandomState(seed + iter)
             subset_indices = [rng.permutation(ds.n_units)[:subset_size]
                               for ds in experiment_datasets]

             # Run E+M on each halfset
             Ft_y_halves, Ft_ctf_halves = [], []
             for k in (0, 1):
                 result = run_em_v2(
                     experiment_datasets[k],
                     state.iref_halves[k],
                     mean_variance,
                     noise_variance,
                     rotations,
                     translations,
                     disc_type=disc_type,
                     image_batch_size=image_batch_size,
                     rotation_block_size=rotation_block_size,
                     current_size=current_size,
                     image_indices=subset_indices[k],
                 )
                 Ft_y_halves.append(result.mean_stats.Ft_y)
                 Ft_ctf_halves.append(result.mean_stats.Ft_ctf)

             # FSC from previous iteration's volumes
             fsc = get_fsc_gpu(state.iref_halves[0], state.iref_halves[1],
                               ds.volume_shape)

             # Adam-like gradient update → new iref_halves
             state = gradient_update_step(
                 state,
                 Ft_y_halves, Ft_ctf_halves,
                 ds.volume_shape,
                 step_size=curr_step, tau2_fudge=curr_tau2, mu=mu,
                 mean_variance=mean_variance,
                 fsc=fsc, min_resol_shell=...,
             )

             # Update noise from residuals (reuse run_em_v2's noise accumulation)
             noise_variance = ...

             # Optional: save intermediate MRCs via save_volume helper
        5. Return dict matching refine_single_volume's output shape (for
           downstream compatibility with run_full_refinement.py and test
           scripts).

    Returns:
        dict with keys:
            mean : jnp.ndarray -- averaged final volume (flat centered FT)
            means : list of 2 jnp.ndarray -- per-halfset volumes
            fsc : jnp.ndarray -- final FSC curve
            iteration_history : list of dicts per iter with
                {subset_size, current_size, tau2_fudge, step_size,
                 fsc_at_cs_limit, mean_signal_energy, wall_time}
    """
```

### 3.3 Helper schedules (private in `ab_initio.py`)

```python
def _subset_size_schedule(iter, ini_iter, inbetween_iter,
                          ini_subset, fin_subset):
    if iter < ini_iter:
        return ini_subset
    if iter < ini_iter + inbetween_iter:
        frac = (iter - ini_iter) / inbetween_iter
        return int(round(ini_subset + frac * (fin_subset - ini_subset)))
    return fin_subset


def _current_size_schedule(iter, ini_iter, inbetween_iter,
                           ini_res, fin_res, voxel_size, grid_size):
    """Convert a resolution in Å to current_size in pixels, then interpolate.

    current_size = 2 * round(grid_size * voxel_size / resolution).
    """
    if iter < ini_iter:
        res = ini_res
    elif iter < ini_iter + inbetween_iter:
        frac = (iter - ini_iter) / inbetween_iter
        res = ini_res + frac * (fin_res - ini_res)
    else:
        res = fin_res
    cs = 2 * round(grid_size * voxel_size / res)
    return max(16, min(cs, grid_size))


def _tau2_schedule(iter, ini_iter, inbetween_iter, tau2_base, inflate):
    """RELION sigmoid schedule for tau2_fudge (ml_optimiser.cpp::updateTau2Fudge)."""
    x = iter
    a = max(inbetween_iter / 4, 1)     # sigmoid length
    b = ini_iter                        # sigmoid start
    scale = 1.0 / (10.0 ** ((x - b - a / 2.0) / (a / 4.0)) + 1.0)
    return (tau2_base * inflate) * scale + tau2_base * (1 - scale)


def _step_schedule(iter, ini_iter, inbetween_iter, step_base, inflate):
    """Same sigmoid shape, different inflate factor."""
    x = iter
    a = max(inbetween_iter / 2, 1)     # sigmoid length (step uses /2, not /4!)
    b = ini_iter
    scale = 1.0 / (10.0 ** ((x - b - a / 2.0) / (a / 4.0)) + 1.0)
    return (step_base * inflate) * scale + step_base * (1 - scale)
```

Note: I double-checked the RELION sigmoid — `updateTau2Fudge` uses
`a = grad_inbetween_iter / 4`, but `updateStepSize` uses
`a = grad_inbetween_iter / 2`.  Both use `b = grad_ini_iter`.

## 4. Parameter table — recovar defaults that MUST match RELION

| Parameter              | recovar default (proposed) | RELION default  |
|------------------------|----------------------------|-----------------|
| `n_total_iter`         | `200`                      | `200`           |
| `ini_frac`             | `0.3`                      | `0.3`           |
| `fin_frac`             | `0.2`                      | `0.2`           |
| `ini_subset_size`      | auto: `clip(N*0.005, 200, 5000)` | `clip(N*0.005, 200, 5000)` |
| `fin_subset_size`      | auto: `clip(N*0.1, 1000, 50000)`  | `clip(N*0.1, 1000, 50000)` |
| `mu`                   | `0.9`                      | `0.9`           |
| `step_size`            | `0.5`                      | `0.5` (3D init) |
| `step_size_inflate`    | `1.8`  (= `0.9 / 0.5`)     | `0.9 / 0.5`     |
| `tau2_fudge`           | `4.0`                      | `4.0`           |
| `tau2_fudge_inflate`   | `4.0`                      | `4.0`           |
| `ini_resol_angstrom`   | `35.0`                     | `35.0`          |
| `fin_resol_angstrom`   | `15.0`                     | `15.0`          |
| `grad_min_resol_angstrom` | `20.0`                  | `20.0`          |
| `init_high_angstrom`   | `ori_size * voxel_size / round(0.07 * ori_size)` | same formula |
| init volume generation | random-orientation backprojection | random-orientation backprojection |

## 5. Test plan

A fast synthetic test at `tests/integration/test_ab_initio_single_volume.py`.

Requirements per `CLAUDE.md`:
- Baselines stored under `tests/baselines/` and left untouched (the baseline
  for ab-initio will be a small dataset of ~500 128px particles simulated
  from a known ground truth with seed=42, noise ≤ 0.1).
- Runs <5 minutes on CPU, <60 s on a single login-node GPU.
- Asserts FSC(final, ground_truth) ≥ 0.5 at 15 Å (half the Nyquist of a
  128px box at 4.25 Å/px).
- Prints the per-iteration schedule via `logger.info` so the user can
  eyeball the ini/inbetween/fin transitions.

The test will also assert:
- `iter_history[0]["subset_size"] == ini_subset_size`
- `iter_history[-1]["subset_size"] == fin_subset_size`
- Monotone (linear) ramp during the inbetween phase.
- Tau2 schedule decays from inflate*base to base.

Plus unit tests under `tests/unit/test_ab_initio_schedules.py`:
- `_subset_size_schedule` → correct endpoints, correct linear interp.
- `_current_size_schedule` → quantises to even integers, bounded.
- `_tau2_schedule`, `_step_schedule` → monotone sigmoids, correct limits.

## 6. CLI wiring

Add a new script `scripts/run_ab_initio.py` (analogous to
`scripts/run_full_refinement.py`) that:

1. Loads the dataset.
2. Optionally takes `--init_volume path.mrc` (for continuation); otherwise
   calls `random_orientation_init_volume`.
3. Calls `ab_initio_single_volume(...)`.
4. Saves per-iteration volumes via `recovar.output.output.save_volume`
   (to correctly handle the recovar↔cryosparc axis transpose).
5. Exposes all the schedule knobs from §4 as CLI flags.

**`scripts/run_full_refinement.py` is NOT modified** — it stays the
polish-from-init path.  We keep a clean separation: one script for
ab-initio, one for refinement-from-init.

## 7. Estimated diff size

| File                  | Lines added | Kind |
|-----------------------|-------------|------|
| `recovar/em/dense_single_volume/ab_initio.py` | ~400 | new |
| `recovar/em/dense_single_volume/gradient_update.py` | ~180 | new |
| `recovar/em/dense_single_volume/random_init.py` | ~120 | new |
| `recovar/em/dense_single_volume/__init__.py` | ~10   | edit (export new API) |
| `scripts/run_ab_initio.py` | ~200 | new |
| `tests/unit/test_ab_initio_schedules.py` | ~120 | new |
| `tests/integration/test_ab_initio_single_volume.py` | ~150 | new |
| **Total** | **~1180** | — |

Plus a small Slurm sbatch wrapper script (~50 lines) under
`scripts/run_ab_initio_slurm.sh` for GPU testing.

## 8. Implementation phases

To avoid a 1200-line single PR, I propose three incremental PRs:

1. **PR-1**: `random_init.py` + `gradient_update.py` + their unit tests
   (~500 lines).  Uses `sum_up_images_fixed_rots_eqx` and
   `post_process_from_filter_v2` directly.  Does NOT depend on the
   full ab-initio loop yet — can be tested in isolation.
2. **PR-2**: `ab_initio.py` schedule helpers + the orchestration loop
   + the synthetic integration test (~600 lines).  Requires PR-1 merged.
3. **PR-3**: `scripts/run_ab_initio.py` + the Slurm wrapper + a
   benchmark comparing recovar ab-initio vs RELION ab-initio on the
   same synthetic dataset (~250 lines).

Each PR is independently reviewable and testable.

## 9. Open questions for the user (please answer before implementation)

1. **Is the RELION hybrid Adam/RL update really what we want, or do we
   want a simpler momentum-GD first?**  The RELION update is complex
   (two moments, FSC gating, pseudo-halfsets).  A simpler momentum-GD
   would be ~30 % less code and easier to debug.  RELION's choice is
   empirically very good, so I lean toward matching it.
2. **Pseudo-halfsets vs real halfsets.**  RELION uses `do_split_random_halves=true`
   for the E-step and M-step, but `grad_pseudo_halfsets` internally
   splits each halfset into two for the second-moment calculation.
   Should we follow this exactly, or use the two halfsets already in
   `experiment_datasets` as the pseudo-halfsets?  The latter is simpler
   and still gives a valid halfset-disagreement second-moment, but it is
   NOT byte-identical to RELION.
3. **Where should the ini init volume go?**  RELION stores the random
   backprojection init in memory.  In recovar's pipeline, should we
   also save it to `{save_intermediates_dir}/init_volume.mrc` for
   debugging?  (My default: yes.)
4. **`n_total_iter = 200` for a synthetic test is expensive.**  The test
   plan above uses a smaller subset (maybe `n_total_iter=30, ini_iter=9,
   fin_iter=6`) to stay <5 min on CPU.  Is this OK?
5. **Single class only** — `nr_classes = 1` (matching Refine3D behaviour,
   not Class3D).  Multi-class ab-initio (`--K > 1`) is a strictly larger
   project and not in scope here.  OK?
6. **Adaptive oversampling.**  Should the ab-initio loop support the
   adaptive pass-1/pass-2 split?  RELION does run adaptive oversampling
   during gradient refine (via `auto_ignore_angle_changes = true` and the
   resolution-based angle schedule).  My default: **no** for PR-1 and
   PR-2, **optional flag** in PR-3.  Justification: HEALPix order 2
   (768 rotations) at low resolution is already cheap enough to do dense
   without oversampling.
