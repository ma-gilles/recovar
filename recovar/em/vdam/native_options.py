"""Shared run defaults and native InitialModel options.

Sampling and continuation read these records without importing the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from recovar.em.vdam.schedules import (
    DEFAULT_GRAD_EM_ITERS,
    DEFAULT_GRAD_FIN_FRAC,
    DEFAULT_GRAD_INI_FRAC,
    DEFAULT_GRAD_MU,
    DEFAULT_STEPSIZE_3D_INITIAL_MODEL,
    GUI_DEFAULT_NR_CLASSES,
    GUI_DEFAULT_NR_ITER,
    GUI_DEFAULT_TAU2_FUDGE,
)


@dataclass(frozen=True, kw_only=True)
class InitialModelDefaults:
    """Run settings shared by the native driver and the command/GUI defaults."""

    nr_iter: int = GUI_DEFAULT_NR_ITER
    grad_write_iter: int = 10
    nr_classes: int = GUI_DEFAULT_NR_CLASSES
    tau2_fudge: float = GUI_DEFAULT_TAU2_FUDGE
    sym_name: str = "C1"
    do_run_C1: bool = True
    particle_diameter: float = 200.0
    do_solvent: bool = True
    do_zero_mask: bool = True
    do_ctf_correction: bool = True
    random_seed: int = 0
    healpix_order: int = 1
    oversampling: int = 1
    offset_range_px: float = 6.0
    offset_step_px: float = 2.0
    perturbation_factor: float = 0.5
    image_batch_size: int = 500
    rotation_block_size: int = 5000
    pass2_engine: str = "auto"
    relion_wavg_sequential_cuda: bool = True
    exact_local_bucket_radix: int = 4
    exact_local_physical_order_chunk_size: int = 0
    stable_fourier_window_shapes: bool = False
    bootstrap_min_particles: int = 1000
    sigma2_min_particles: int = 1000
    padding_factor: int = 1
    lazy: bool = True
    write_iter_artifacts: bool = True
    deterministic_cuda: bool = False
    random_perturbation: float | None = None
    translation_sigma_angstrom: float | None = None
    grad_ini_frac: float = DEFAULT_GRAD_INI_FRAC
    grad_fin_frac: float = DEFAULT_GRAD_FIN_FRAC
    grad_em_iters: int = DEFAULT_GRAD_EM_ITERS
    stepsize: float = DEFAULT_STEPSIZE_3D_INITIAL_MODEL
    mu: float = DEFAULT_GRAD_MU


@dataclass(frozen=True, kw_only=True)
class NativeInitialModelOptions(InitialModelDefaults):
    """Options for a native InitialModel run; defaults mirror the GUI command."""

    fn_img: str
    outputname: str = "ab_initio/run"
    width_mask_edge_px: float = 5.0
    image_fourier_backend: str = "host_numpy"
    projector_setup_backend: Literal["native", "jax"] = "native"
    mstep_backend: Literal["native", "jax"] = "native"
    mstep_compute_dtype: Literal["float32", "float64"] = "float64"
    datadir: str | None = None
    strip_prefix: str | None = None
    # Diagnostic-only, one-next-iteration restart from a native RELION VDAM
    # optimiser.  This is deliberately not a general production continuation
    # surface: the caller must also stop at checkpoint_iteration + 1.
    diagnostic_continue_optimiser: str | None = None
    diagnostic_stop_after_iteration: int | None = None
