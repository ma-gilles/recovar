"""VDAM InitialModel schedules: subset size, step size, tau2_fudge.

Mirrors RELION 5.0 ml_optimiser.cpp routines line-for-line so trajectories
match byte-for-byte. C++ ``int`` truncation is reproduced with Python ``int``
arithmetic; ``RFLOAT`` uses Python ``float`` (double).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# GUI InitialModel defaults (pipeline_jobs.cpp::initialiseInimodelJob,
# parseInitial ml_optimiser.cpp:978-1098, updateStepSize :10287, updateTau2Fudge :10340).
DEFAULT_GRAD_INI_FRAC: float = 0.3
DEFAULT_GRAD_FIN_FRAC: float = 0.2
DEFAULT_GRAD_EM_ITERS: int = 0
DEFAULT_GRAD_MU: float = 0.9
GUI_DEFAULT_NR_ITER: int = 200
GUI_DEFAULT_NR_CLASSES: int = 1
GUI_DEFAULT_TAU2_FUDGE: float = 4.0
DEFAULT_STEPSIZE_3D_INITIAL_MODEL: float = 0.5
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
    """Per-phase iter counts. Sums to nr_iter (ml_optimiser.cpp:994-998)."""

    grad_ini_iter: int
    grad_inbetween_iter: int
    grad_fin_iter: int


def compute_phase_lengths(
    nr_iter: int,
    grad_ini_frac: float = DEFAULT_GRAD_INI_FRAC,
    grad_fin_frac: float = DEFAULT_GRAD_FIN_FRAC,
) -> VdamPhaseLengths:
    """Reproduce ``grad_*_iter`` assignment (ml_optimiser.cpp:411-420 + 994-998).

    Renormalises so the inbetween phase is at least 10% of ``nr_iter`` when
    ``grad_ini_frac + grad_fin_frac > 0.9``; phase iters are ``int(nr_iter * frac)``.
    """
    if grad_ini_frac <= 0.0 or grad_ini_frac >= 1.0:
        raise ValueError("Invalid value for grad_ini_frac (must be in (0, 1))")
    if grad_fin_frac <= 0.0 or grad_fin_frac >= 1.0:
        raise ValueError("Invalid value for grad_fin_frac (must be in (0, 1))")

    if grad_ini_frac + grad_fin_frac > 0.9:
        s = grad_ini_frac + grad_fin_frac + 0.1
        grad_ini_frac /= s
        grad_fin_frac /= s

    grad_ini_iter = int(nr_iter * grad_ini_frac)
    grad_fin_iter = int(nr_iter * grad_fin_frac)
    grad_inbetween_iter = max(0, nr_iter - grad_ini_iter - grad_fin_iter)
    return VdamPhaseLengths(grad_ini_iter, grad_inbetween_iter, grad_fin_iter)


def default_subset_sizes_for_3d_initial_model(dataset_size: int) -> tuple[int, int]:
    """Return ``(grad_ini, grad_fin)`` subset sizes (ml_optimiser.cpp:2672-2696)."""
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    ini = max(min(int(round(dataset_size * 0.005)), 5000), 200)
    fin = max(min(int(round(dataset_size * 0.1)), 50000), 1000)
    return ini, fin


def default_step_size_for_3d_initial_model() -> float:
    return DEFAULT_STEPSIZE_3D_INITIAL_MODEL


def default_tau2_fudge_for_3d_initial_model() -> float:
    return DEFAULT_TAU2_FUDGE_3D_INITIAL_MODEL


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
    """``updateSubsetSize`` for ``gradient_refine && !do_auto_refine``
    (ml_optimiser.cpp:10238-10271). ``-1`` means all particles."""
    grad_ini_iter = phase_lengths.grad_ini_iter
    grad_inbetween_iter = phase_lengths.grad_inbetween_iter

    if iter < grad_ini_iter:
        subset_size = grad_ini_subset_size
    elif iter < grad_ini_iter + grad_inbetween_iter:
        frac = (iter - grad_ini_iter) / grad_inbetween_iter if grad_inbetween_iter > 0 else 0.0
        subset_size = grad_ini_subset_size + _relion_round(frac * (grad_fin_subset_size - grad_ini_subset_size))
    else:
        subset_size = grad_fin_subset_size

    effective_nr_particles = nr_particles // 2 if do_split_random_halves else nr_particles
    if (
        not do_grad
        or (nr_iter - iter) < grad_em_iters
        or (nr_iter == iter and nr_classes > 1)
        or subset_size >= effective_nr_particles
        or grad_suspended_local_searches_iter > 0
        or has_converged
        or grad_has_converged
    ):
        subset_size = -1

    return subset_size


def compute_stepsize(
    iter: int,
    phase_lengths: VdamPhaseLengths,
    is_3d_model: bool,
    ref_dim: int,
    grad_stepsize: Optional[float] = None,
    grad_stepsize_scheme: Optional[str] = None,
) -> float:
    """``updateStepSize`` (ml_optimiser.cpp:10278-10325).

    Default stepsize is 0.5 (3D init) else 0.3; default scheme is ``"plain"``
    for 3D class else ``"<0.9/stepsize>-step"``.
    """
    if ref_dim not in (2, 3):
        raise ValueError(f"ref_dim must be 2 or 3, got {ref_dim}")

    _stepsize = grad_stepsize if (grad_stepsize is not None and grad_stepsize > 0) else -1.0
    _scheme = grad_stepsize_scheme if grad_stepsize_scheme is not None else ""

    if _stepsize <= 0:
        _stepsize = 0.5 if (ref_dim == 3 and is_3d_model) else 0.3
    if _scheme == "":
        _scheme = "plain" if (ref_dim == 3 and not is_3d_model) else f"{0.9 / _stepsize:f}-step"

    if _scheme == "plain":
        return _stepsize
    if "-step" in _scheme:
        inflate = float(np.float32(_scheme[: _scheme.find("-step")]))
        if inflate <= 0.0:
            raise ValueError("Invalid inflate value for --grad_stepsize_scheme <inflate>-step (inflate > 1)")
        # RELION assigns `float a = grad_inbetween_iter/2`, so the division
        # is integer division before conversion to float.
        sigmoid_len = float(max(int(phase_lengths.grad_inbetween_iter), 0) // 2)
        return _step_sigmoid_value(
            iter=iter,
            grad_ini_iter=phase_lengths.grad_ini_iter,
            grad_inbetween_iter=phase_lengths.grad_inbetween_iter,
            base=_stepsize,
            inflated=_stepsize * inflate,
            sigmoid_length=sigmoid_len,
        )
    raise ValueError("Invalid value for --grad_stepsize_scheme")


def compute_tau2_fudge(
    iter: int,
    phase_lengths: VdamPhaseLengths,
    is_3d_model: bool,
    ref_dim: int,
    do_auto_refine: bool = False,
    tau2_fudge_arg: Optional[float] = None,
    tau2_fudge_scheme: Optional[str] = None,
) -> float:
    """``updateTau2Fudge`` (ml_optimiser.cpp:10327-10379).

    Default fudge is 1.0 (auto-refine) else 4.0; default scheme is ``"plain"``
    for 3D class else ``"<fudge>-step"``.
    """
    if ref_dim not in (2, 3):
        raise ValueError(f"ref_dim must be 2 or 3, got {ref_dim}")

    _fudge = tau2_fudge_arg if (tau2_fudge_arg is not None and tau2_fudge_arg > 0) else -1.0
    _scheme = tau2_fudge_scheme if tau2_fudge_scheme is not None else ""

    if _fudge <= 0:
        _fudge = 1.0 if do_auto_refine else 4.0
    if _scheme == "":
        _scheme = "plain" if (ref_dim == 3 and not is_3d_model) else f"{_fudge / 1.0:f}-step"

    if _scheme == "plain":
        return _fudge
    if "-step" in _scheme:
        deflate = float(np.float32(_scheme[: _scheme.find("-step")]))
        if deflate <= 0.0:
            raise ValueError("Invalid deflate value for --tau2_fudge_scheme <deflate>-step (deflate > 1)")
        # RELION assigns `float a = grad_inbetween_iter/4`, so short runs can
        # hit a=0. The resulting 0/0 at iter==grad_ini_iter intentionally
        # produces NaN and is written to InitialModel model.star.
        sigmoid_len = float(max(int(phase_lengths.grad_inbetween_iter), 0) // 4)
        return _step_sigmoid_value(
            iter=iter,
            grad_ini_iter=phase_lengths.grad_ini_iter,
            grad_inbetween_iter=phase_lengths.grad_inbetween_iter,
            base=_fudge,
            inflated=_fudge / deflate,
            sigmoid_length=sigmoid_len,
        )
    raise ValueError("Invalid value for --tau2_fudge_scheme")


def _step_sigmoid_value(
    iter: int,
    grad_ini_iter: int,
    grad_inbetween_iter: int,
    base: float,
    inflated: float,
    sigmoid_length: float,
) -> float:
    """Shared sigmoid (ml_optimiser.cpp:10316-10320 and 10370-10374).

    ``value = inflated * scale + base * (1 - scale)`` with
    ``scale = 1 / (10**((iter - grad_ini - len/2) / (len/4)) + 1)``.
    """
    x = float(iter)
    a = float(sigmoid_length)
    b = float(grad_ini_iter)
    if a <= 0.0:
        offset = x - b
        if offset == 0.0:
            return math.nan
        scale = 0.0 if offset > 0.0 else 1.0
        return inflated * scale + base * (1.0 - scale)
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


def _relion_round(x: float) -> int:
    """``ROUND(x) == (int)floor(x + 0.5)``; symmetric around zero (RELION macro)."""
    return int(math.floor(x + 0.5)) if x >= 0.0 else -int(math.floor(-x + 0.5))
