"""VDAM InitialModel schedules (subset size, step size, tau2_fudge).

Each function below mirrors the corresponding RELION 5.0
`ml_optimiser.cpp` routine line-for-line so that recovar and RELION produce
byte-identical schedule trajectories given the same inputs. See
`docs/math/plan_ab_initio_relion_parity_v3.md` section 2 for the RELION
source citations.

All functions are pure scalar Python operating on native `int` / `float`.
C++ integer truncation is reproduced exactly by using Python integer
arithmetic where RELION uses `int`, and Python floats where RELION uses
`RFLOAT` (double).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# GUI InitialModel defaults (pipeline_jobs.cpp::initialiseInimodelJob)
# ---------------------------------------------------------------------------

# From parseInitial (ml_optimiser.cpp:982-983): grad_ini_frac=0.3, grad_fin_frac=0.2
DEFAULT_GRAD_INI_FRAC: float = 0.3
DEFAULT_GRAD_FIN_FRAC: float = 0.2

# From parseInitial (ml_optimiser.cpp:978): grad_em_iters = 0 when --grad_em_iters unset
DEFAULT_GRAD_EM_ITERS: int = 0

# From parseInitial (ml_optimiser.cpp:1098): --mu default for gradient refinement
DEFAULT_GRAD_MU: float = 0.9

# From pipeline_jobs.cpp:3376-3385: GUI defaults for the InitialModel job
GUI_DEFAULT_NR_ITER: int = 200
GUI_DEFAULT_NR_CLASSES: int = 1
GUI_DEFAULT_TAU2_FUDGE: float = 4.0

# From updateStepSize (ml_optimiser.cpp:10287-10288): 3D initial model stepsize default
DEFAULT_STEPSIZE_3D_INITIAL_MODEL: float = 0.5

# From updateTau2Fudge (ml_optimiser.cpp:10340-10341): 3D initial model fudge default
DEFAULT_TAU2_FUDGE_3D_INITIAL_MODEL: float = 4.0


@dataclass(frozen=True)
class GuiInitialModelDefaults:
    """Knobs that a GUI-InitialModel recovar driver must reproduce verbatim."""

    nr_iter: int = GUI_DEFAULT_NR_ITER
    nr_classes: int = GUI_DEFAULT_NR_CLASSES
    tau2_fudge: float = GUI_DEFAULT_TAU2_FUDGE
    grad_ini_frac: float = DEFAULT_GRAD_INI_FRAC
    grad_fin_frac: float = DEFAULT_GRAD_FIN_FRAC
    grad_em_iters: int = DEFAULT_GRAD_EM_ITERS
    stepsize: float = DEFAULT_STEPSIZE_3D_INITIAL_MODEL
    mu: float = DEFAULT_GRAD_MU


@dataclass(frozen=True)
class VdamPhaseLengths:
    """Iteration counts for the three VDAM phases.

    Matches ml_optimiser.cpp:994-998. `grad_ini_iter + grad_inbetween_iter +
    grad_fin_iter == nr_iter` exactly after the renormalisation branch at
    ml_optimiser.cpp:416-420.
    """

    grad_ini_iter: int
    grad_inbetween_iter: int
    grad_fin_iter: int


# ---------------------------------------------------------------------------
# Phase-length computation (ml_optimiser.cpp:411-998)
# ---------------------------------------------------------------------------


def compute_phase_lengths(
    nr_iter: int,
    grad_ini_frac: float = DEFAULT_GRAD_INI_FRAC,
    grad_fin_frac: float = DEFAULT_GRAD_FIN_FRAC,
) -> VdamPhaseLengths:
    """Reproduce `grad_*_iter` assignment from parseInitial.

    RELION source: ml_optimiser.cpp:411-420 + 994-998.

    The renormalisation branch at line 416 triggers when
    `grad_ini_frac + grad_fin_frac > 0.9` and scales both fracs by
    `sum / (sum + 0.1)` so the inbetween phase is at least 10% of nr_iter.
    `grad_*_iter` are then computed as `int(nr_iter * frac)` (C++
    `int(RFLOAT)` truncation).
    """
    if grad_ini_frac <= 0.0 or grad_ini_frac >= 1.0:
        raise ValueError("Invalid value for grad_ini_frac (must be in (0, 1))")
    if grad_fin_frac <= 0.0 or grad_fin_frac >= 1.0:
        raise ValueError("Invalid value for grad_fin_frac (must be in (0, 1))")

    if grad_ini_frac + grad_fin_frac > 0.9:
        s = grad_ini_frac + grad_fin_frac + 0.1
        grad_ini_frac = grad_ini_frac / s
        grad_fin_frac = grad_fin_frac / s

    # C++ `int grad_ini_iter = nr_iter * grad_ini_frac;` truncates toward zero
    grad_ini_iter = int(nr_iter * grad_ini_frac)
    grad_fin_iter = int(nr_iter * grad_fin_frac)
    grad_inbetween_iter = nr_iter - grad_ini_iter - grad_fin_iter
    if grad_inbetween_iter < 0:
        grad_inbetween_iter = 0

    return VdamPhaseLengths(
        grad_ini_iter=grad_ini_iter,
        grad_inbetween_iter=grad_inbetween_iter,
        grad_fin_iter=grad_fin_iter,
    )


# ---------------------------------------------------------------------------
# Default subset sizes for 3D initial model
# (ml_optimiser.cpp:2663-2705, is_3d_model branch)
# ---------------------------------------------------------------------------


def default_subset_sizes_for_3d_initial_model(
    dataset_size: int,
) -> tuple[int, int]:
    """Return `(grad_ini_subset_size, grad_fin_subset_size)` for 3D initial model.

    RELION source: ml_optimiser.cpp:2672-2696 (`is_3d_model` branch).

    grad_ini_subset_size = clamp(round(N * 0.005), 200, 5000)
    grad_fin_subset_size = clamp(round(N * 0.1),   1000, 50000)
    """
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")

    # XMIPP_MAX(XMIPP_MIN(dataset_size * 0.005, 5000), 200)
    ini = max(min(int(round(dataset_size * 0.005)), 5000), 200)
    # XMIPP_MAX(XMIPP_MIN(dataset_size * 0.1, 50000), 1000)
    fin = max(min(int(round(dataset_size * 0.1)), 50000), 1000)
    return ini, fin


def default_step_size_for_3d_initial_model() -> float:
    """3D initial model default stepsize (ml_optimiser.cpp:10287-10288)."""
    return DEFAULT_STEPSIZE_3D_INITIAL_MODEL


def default_tau2_fudge_for_3d_initial_model() -> float:
    """3D initial model default tau2_fudge (ml_optimiser.cpp:10340-10341)."""
    return DEFAULT_TAU2_FUDGE_3D_INITIAL_MODEL


# ---------------------------------------------------------------------------
# updateSubsetSize (ml_optimiser.cpp:10212-10276) — gradient_refine branch
# ---------------------------------------------------------------------------


def compute_subset_size(
    iter: int,
    phase_lengths: VdamPhaseLengths,
    grad_ini_subset_size: int,
    grad_fin_subset_size: int,
    nr_particles: int,
    nr_iter: int,
    grad_em_iters: int = DEFAULT_GRAD_EM_ITERS,
    do_grad: bool = True,
    has_converged: bool = False,
    grad_has_converged: bool = False,
    nr_classes: int = 1,
    do_split_random_halves: bool = False,
    grad_suspended_local_searches_iter: int = 0,
) -> int:
    """Reproduce `updateSubsetSize` for `gradient_refine && !do_auto_refine`.

    Returns the subset size for the given iter. A value of `-1` means
    "all particles".

    RELION source: ml_optimiser.cpp:10238-10271.
    """
    grad_ini_iter = phase_lengths.grad_ini_iter
    grad_inbetween_iter = phase_lengths.grad_inbetween_iter

    if iter < grad_ini_iter:
        subset_size = grad_ini_subset_size
    elif iter < grad_ini_iter + grad_inbetween_iter:
        # ROUND((RFLOAT(iter - grad_ini_iter) / RFLOAT(grad_inbetween_iter)) * (fin - ini))
        if grad_inbetween_iter <= 0:
            frac = 0.0
        else:
            frac = (iter - grad_ini_iter) / grad_inbetween_iter
        increment = _relion_round(frac * (grad_fin_subset_size - grad_ini_subset_size))
        subset_size = grad_ini_subset_size + increment
    else:
        subset_size = grad_fin_subset_size

    # Halfset-adjusted denominator (ml_optimiser.cpp:10261-10263)
    effective_nr_particles = nr_particles
    if do_split_random_halves:
        effective_nr_particles = nr_particles // 2  # RELION: floor(nr_particles/2)

    # "Switch to all particles" conditions (ml_optimiser.cpp:10265-10271)
    final_em_tail = (nr_iter - iter) < grad_em_iters
    last_iter_with_single_class = nr_iter == iter and nr_classes > 1  # force skip-all
    if (
        not do_grad
        or final_em_tail
        or last_iter_with_single_class
        or subset_size >= effective_nr_particles
        or grad_suspended_local_searches_iter > 0
        or has_converged
        or grad_has_converged
    ):
        subset_size = -1

    return subset_size


# ---------------------------------------------------------------------------
# updateStepSize (ml_optimiser.cpp:10278-10325)
# ---------------------------------------------------------------------------


def compute_stepsize(
    iter: int,
    phase_lengths: VdamPhaseLengths,
    is_3d_model: bool,
    ref_dim: int,
    grad_stepsize: Optional[float] = None,
    grad_stepsize_scheme: Optional[str] = None,
) -> float:
    """Reproduce `updateStepSize` for 3D initial model / 3D class / 2D class.

    RELION source: ml_optimiser.cpp:10278-10325.

    `grad_stepsize=None` or `<=0` triggers the RELION default resolution:
      - 3D initial model: 0.5
      - 3D classification: 0.3
      - 2D classification: 0.3
    `grad_stepsize_scheme=None` or `""` triggers the default scheme:
      - 3D initial model / 2D: "<0.9/_stepsize>-step"  (e.g. "1.8-step")
      - 3D classification:     "plain"
    """
    if ref_dim not in (2, 3):
        raise ValueError(f"ref_dim must be 2 or 3, got {ref_dim}")

    _stepsize = grad_stepsize if (grad_stepsize is not None and grad_stepsize > 0) else -1.0
    _scheme = grad_stepsize_scheme if grad_stepsize_scheme is not None else ""

    if _stepsize <= 0:
        if ref_dim == 3 and not is_3d_model:
            _stepsize = 0.3
        elif ref_dim == 3 and is_3d_model:
            _stepsize = 0.5
        else:
            _stepsize = 0.3

    if _scheme == "":
        if ref_dim == 3 and not is_3d_model:
            _scheme = "plain"
        elif ref_dim == 3 and is_3d_model:
            # std::to_string uses %f (6-decimal); matches C++ behaviour
            _scheme = f"{0.9 / _stepsize:f}-step"
        else:
            _scheme = f"{0.9 / _stepsize:f}-step"

    if _scheme == "plain":
        return _stepsize

    if "-step" in _scheme:
        inflate_str = _scheme[: _scheme.find("-step")]
        # textToFloat returns C++ float (32-bit). Match RELION's precision.
        inflate = float(np.float32(inflate_str))
        if inflate <= 0.0:
            raise ValueError("Invalid inflate value for --grad_stepsize_scheme <inflate>-step (inflate > 1)")
        return _step_sigmoid_value(
            iter=iter,
            grad_ini_iter=phase_lengths.grad_ini_iter,
            grad_inbetween_iter=phase_lengths.grad_inbetween_iter,
            base=_stepsize,
            inflated=_stepsize * inflate,
            sigmoid_length=phase_lengths.grad_inbetween_iter / 2.0,
        )

    raise ValueError("Invalid value for --grad_stepsize_scheme")


# ---------------------------------------------------------------------------
# updateTau2Fudge (ml_optimiser.cpp:10327-10379)
# ---------------------------------------------------------------------------


def compute_tau2_fudge(
    iter: int,
    phase_lengths: VdamPhaseLengths,
    is_3d_model: bool,
    ref_dim: int,
    do_auto_refine: bool = False,
    tau2_fudge_arg: Optional[float] = None,
    tau2_fudge_scheme: Optional[str] = None,
) -> float:
    """Reproduce `updateTau2Fudge` for auto-refine / 3D init / 3D class / 2D class.

    RELION source: ml_optimiser.cpp:10327-10379.

    For GUI InitialModel (`is_3d_model=True`, `ref_dim=3`, not auto-refine,
    `tau2_fudge_arg=4`, empty scheme) the trajectory is
    `tau2_fudge = scale + 4*(1-scale)` which grows from ~1 at iter 0 to
    ~4 at iter >> grad_ini_iter + grad_inbetween_iter/2.
    """
    if ref_dim not in (2, 3):
        raise ValueError(f"ref_dim must be 2 or 3, got {ref_dim}")

    _fudge = tau2_fudge_arg if (tau2_fudge_arg is not None and tau2_fudge_arg > 0) else -1.0
    _scheme = tau2_fudge_scheme if tau2_fudge_scheme is not None else ""

    if _fudge <= 0:
        if do_auto_refine:
            _fudge = 1.0
        else:
            # 3D classification, 3D initial model, 2D classification — all 4
            _fudge = 4.0

    if _scheme == "":
        if ref_dim == 3 and not is_3d_model:
            _scheme = "plain"
        elif ref_dim == 3 and is_3d_model:
            _scheme = f"{_fudge / 1.0:f}-step"
        else:
            _scheme = f"{_fudge / 1.0:f}-step"

    if _scheme == "plain":
        return _fudge

    if "-step" in _scheme:
        deflate_str = _scheme[: _scheme.find("-step")]
        # textToFloat returns C++ float (32-bit). Match RELION's precision.
        deflate = float(np.float32(deflate_str))
        if deflate <= 0.0:
            raise ValueError("Invalid deflate value for --tau2_fudge_scheme <deflate>-step (deflate > 1)")
        return _step_sigmoid_value(
            iter=iter,
            grad_ini_iter=phase_lengths.grad_ini_iter,
            grad_inbetween_iter=phase_lengths.grad_inbetween_iter,
            base=_fudge,
            inflated=_fudge / deflate,
            sigmoid_length=phase_lengths.grad_inbetween_iter / 4.0,
        )

    raise ValueError("Invalid value for --tau2_fudge_scheme")


# ---------------------------------------------------------------------------
# Shared sigmoid
# ---------------------------------------------------------------------------


def _step_sigmoid_value(
    iter: int,
    grad_ini_iter: int,
    grad_inbetween_iter: int,
    base: float,
    inflated: float,
    sigmoid_length: float,
) -> float:
    """Shared sigmoid used by updateStepSize (a=inbetween/2) and
    updateTau2Fudge (a=inbetween/4).

    RELION source: ml_optimiser.cpp:10316-10320 and 10370-10374.

    scale = 1 / (10**((x - b - a/2) / (a/4)) + 1)
    value = inflated * scale + base * (1 - scale)
    """
    x = float(iter)
    a = float(sigmoid_length)
    b = float(grad_ini_iter)
    if a <= 0.0:
        # Degenerate schedule: inbetween phase is zero. Sigmoid is undefined.
        # Fall back to RELION's behaviour at x >> b: scale -> 0, value -> base.
        # (Matches C++ divide-by-zero in rare boundary configs by short-
        # circuiting to the asymptote.)
        return base
    exponent = (x - b - a / 2.0) / (a / 4.0)
    # Cap the exponent to avoid math.pow overflow; RELION relies on IEEE-754
    # saturating to +inf which makes scale -> 0.
    if exponent > 308.0:
        scale = 0.0
    elif exponent < -308.0:
        scale = 1.0
    else:
        scale = 1.0 / (math.pow(10.0, exponent) + 1.0)
    return inflated * scale + base * (1.0 - scale)


# ---------------------------------------------------------------------------
# RELION integer rounding
# ---------------------------------------------------------------------------


def _relion_round(x: float) -> int:
    """Integer round that matches RELION's `ROUND` macro.

    RELION uses `ROUND(x) == (int)floor(x + 0.5)` for positive values. Python's
    built-in `round` uses banker's rounding which diverges at the 0.5
    boundary. We replicate the RELION behaviour explicitly.
    """
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    else:
        # Symmetric around zero (ROUND(-0.5) == 0 in RELION)
        return -int(math.floor(-x + 0.5))
