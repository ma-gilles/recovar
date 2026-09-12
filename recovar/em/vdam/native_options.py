"""Options of the native InitialModel driver.

``NativeInitialModelOptions`` is the RELION-command-shaped option record every
native InitialModel stage reads; it lives apart from the driver so the sampling
and continuation owners can import it without importing the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from recovar.em.vdam.schedules import GuiInitialModelDefaults

INITIAL_MODEL_GUI_DEFAULTS = GuiInitialModelDefaults()


DEFAULT_WIDTH_MASK_EDGE_PX = 5.0


DEFAULT_HEALPIX_ORDER = INITIAL_MODEL_GUI_DEFAULTS.healpix_order


DEFAULT_OFFSET_RANGE_PX = INITIAL_MODEL_GUI_DEFAULTS.offset_range_px


DEFAULT_OFFSET_STEP_PX = INITIAL_MODEL_GUI_DEFAULTS.offset_step_px


DEFAULT_RANDOM_SEED = INITIAL_MODEL_GUI_DEFAULTS.random_seed


DEFAULT_OVERSAMPLING = INITIAL_MODEL_GUI_DEFAULTS.oversampling


DEFAULT_PERTURBATION_FACTOR = INITIAL_MODEL_GUI_DEFAULTS.perturbation_factor


@dataclass(frozen=True, kw_only=True)
class NativeInitialModelOptions:
    """Options for a native InitialModel run; defaults mirror the GUI command."""

    fn_img: str
    outputname: str = "ab_initio/run"
    nr_iter: int = INITIAL_MODEL_GUI_DEFAULTS.nr_iter
    nr_classes: int = INITIAL_MODEL_GUI_DEFAULTS.nr_classes
    tau2_fudge: float = INITIAL_MODEL_GUI_DEFAULTS.tau2_fudge
    grad_ini_frac: float = INITIAL_MODEL_GUI_DEFAULTS.grad_ini_frac
    grad_fin_frac: float = INITIAL_MODEL_GUI_DEFAULTS.grad_fin_frac
    grad_em_iters: int = INITIAL_MODEL_GUI_DEFAULTS.grad_em_iters
    stepsize: float = INITIAL_MODEL_GUI_DEFAULTS.stepsize
    mu: float = INITIAL_MODEL_GUI_DEFAULTS.mu
    sym_name: str = INITIAL_MODEL_GUI_DEFAULTS.sym_name
    do_run_C1: bool = INITIAL_MODEL_GUI_DEFAULTS.do_run_C1
    particle_diameter: float = INITIAL_MODEL_GUI_DEFAULTS.particle_diameter
    do_solvent: bool = INITIAL_MODEL_GUI_DEFAULTS.do_solvent
    do_zero_mask: bool = INITIAL_MODEL_GUI_DEFAULTS.do_zero_mask
    do_ctf_correction: bool = INITIAL_MODEL_GUI_DEFAULTS.do_ctf_correction
    random_seed: int = DEFAULT_RANDOM_SEED
    width_mask_edge_px: float = DEFAULT_WIDTH_MASK_EDGE_PX
    healpix_order: int = DEFAULT_HEALPIX_ORDER
    oversampling: int = DEFAULT_OVERSAMPLING
    perturbation_factor: float = DEFAULT_PERTURBATION_FACTOR
    random_perturbation: float | None = INITIAL_MODEL_GUI_DEFAULTS.random_perturbation
    offset_range_px: float = DEFAULT_OFFSET_RANGE_PX
    offset_step_px: float = DEFAULT_OFFSET_STEP_PX
    image_batch_size: int = INITIAL_MODEL_GUI_DEFAULTS.image_batch_size
    rotation_block_size: int = INITIAL_MODEL_GUI_DEFAULTS.rotation_block_size
    pass2_engine: str = INITIAL_MODEL_GUI_DEFAULTS.pass2_engine
    relion_wavg_sequential_cuda: bool = INITIAL_MODEL_GUI_DEFAULTS.relion_wavg_sequential_cuda
    exact_local_bucket_radix: int = INITIAL_MODEL_GUI_DEFAULTS.exact_local_bucket_radix
    exact_local_physical_order_chunk_size: int = (
        INITIAL_MODEL_GUI_DEFAULTS.exact_local_physical_order_chunk_size
    )
    stable_fourier_window_shapes: bool = (
        INITIAL_MODEL_GUI_DEFAULTS.stable_fourier_window_shapes
    )
    bootstrap_min_particles: int = INITIAL_MODEL_GUI_DEFAULTS.bootstrap_min_particles
    sigma2_min_particles: int = INITIAL_MODEL_GUI_DEFAULTS.sigma2_min_particles
    padding_factor: int = INITIAL_MODEL_GUI_DEFAULTS.padding_factor
    image_fourier_backend: str = "host_numpy"
    projector_setup_backend: Literal["native", "jax"] = "native"
    mstep_backend: Literal["native", "jax"] = "native"
    mstep_compute_dtype: Literal["float32", "float64"] = "float64"
    deterministic_cuda: bool = INITIAL_MODEL_GUI_DEFAULTS.deterministic_cuda
    lazy: bool = INITIAL_MODEL_GUI_DEFAULTS.lazy
    datadir: str | None = None
    strip_prefix: str | None = None
    translation_sigma_angstrom: float | None = INITIAL_MODEL_GUI_DEFAULTS.translation_sigma_angstrom
    write_iter_artifacts: bool = INITIAL_MODEL_GUI_DEFAULTS.write_iter_artifacts
    grad_write_iter: int = INITIAL_MODEL_GUI_DEFAULTS.grad_write_iter
    run_relion_align_symmetry: bool = False
    # Diagnostic-only, one-next-iteration restart from a native RELION VDAM
    # optimiser.  This is deliberately not a general production continuation
    # surface: the caller must also stop at checkpoint_iteration + 1.
    diagnostic_continue_optimiser: str | None = None
    diagnostic_stop_after_iteration: int | None = None
