"""RELION image-size scheduling and resolution helpers.

Owns current-size bootstrapping, coarse image sizes, FSC/data-vs-prior
boundaries, and first-iteration resolution and tau2-reporting rules. The
iteration controller selects when to apply these rules.
"""

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.convergence import healpix_angular_step

# Re-import so callers can get it from this module.
from recovar.em.dense_single_volume.helpers.fourier_window import quantize_current_size
from recovar.reconstruction.regularization import compute_current_size_relion, fsc_to_relion_ssnr


def relion_local_pass1_current_size(
    *,
    pre_update_healpix_order: int,
    pixel_size: float,
    ori_size: int,
    particle_diameter: float | None,
    current_size: int | None,
) -> int | None:
    """Return RELION's local pass-1 size for the current expectation.

    RELION computes ``image_coarse_size`` before ``updateAngularSampling``.
    When an expectation advances the sampling order, its parent hypotheses
    therefore use the incoming order for Fourier sizing even though the
    updated order controls the parent grid and fine-child expansion.
    """

    coarse_size = compute_coarse_image_size(
        healpix_angular_step(int(pre_update_healpix_order)),
        pixel_size,
        ori_size,
        particle_diameter=particle_diameter,
    )
    coarse_size = clamp_relion_coarse_image_size(
        coarse_size,
        current_size,
        ori_size,
    )
    return coarse_size if coarse_size < int(ori_size) else None

def relion_expectation_coarse_size_order(
    *,
    state_healpix_order: int,
    replay_saved_healpix_order: int | None,
) -> int:
    """Choose the pre-update sampling order at an expectation boundary.

    In strict replay, RECOVAR's live refinement state may have advanced after
    the preceding M-step.  RELION instead enters the next expectation from
    the preceding numbered sampling STAR and only then updates angular
    sampling, so the saved replay order is authoritative for Fourier sizing.
    """

    if replay_saved_healpix_order is not None:
        return int(replay_saved_healpix_order)
    return int(state_healpix_order)


def shell_index_to_resolution_angstrom(shell_index, ori_size, voxel_size):
    """Convert a Fourier shell index into a real-space resolution in Angstrom."""
    if voxel_size <= 0:
        return float(shell_index)
    shell_index = float(shell_index)
    if shell_index <= 0:
        return float("inf")
    return float(ori_size) * float(voxel_size) / shell_index


def relion_optics_image_current_sizes(
    model_current_size,
    *,
    model_ori_size,
    model_pixel_size,
    optics_image_sizes,
    optics_pixel_sizes,
):
    """Return RELION's per-optics particle-image Fourier window sizes.

    RELION keeps ``mymodel.current_size`` in reference-map coordinates, but
    remaps the particle window separately for each optics group::

        remap = (image_pixel_size * image_size) /
                (model_pixel_size * model_ori_size)
        image_current_size = 2 * ceil(0.5 * remap * model_current_size)

    The distinction matters even when the nominal fields print identically in
    STAR files.  For example, an MRC voxel size stored as float32 may be
    1.4166666269 while the optics STAR stores 1.416667.  At model size 56 the
    upward ``ceil`` then produces a 58-pixel particle window, while the
    Projector and BackProjector retain radius 28 from the 56-pixel model size.
    """

    model_current_size = int(model_current_size)
    model_ori_size = int(model_ori_size)
    model_pixel_size = float(model_pixel_size)
    image_sizes = np.asarray(optics_image_sizes, dtype=np.int64).reshape(-1)
    pixel_sizes = np.asarray(optics_pixel_sizes, dtype=np.float64).reshape(-1)
    if image_sizes.shape != pixel_sizes.shape or image_sizes.size == 0:
        raise ValueError(
            "optics image sizes and pixel sizes must be non-empty arrays with identical shapes",
        )
    if model_current_size <= 0 or model_ori_size <= 0 or model_pixel_size <= 0.0:
        raise ValueError("model current size, original size, and pixel size must be positive")
    if np.any(image_sizes <= 0) or np.any(pixel_sizes <= 0.0):
        raise ValueError("optics image sizes and pixel sizes must be positive")

    remap_sizes = (
        pixel_sizes * image_sizes.astype(np.float64)
    ) / (model_pixel_size * float(model_ori_size))
    current_sizes = 2 * np.ceil(0.5 * remap_sizes * float(model_current_size))
    return np.minimum(current_sizes.astype(np.int64), image_sizes)


# Default threshold for when adaptive pass 2 is skipped.
ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION = 0.5


def compute_coarse_image_size(
    angular_step_deg,
    pixel_size,
    ori_size,
    particle_diameter=None,
):
    """Compute the coarse image size for pass 1 of adaptive oversampling.

    RELION formula (expectation.cpp line 5760):
        rotated_distance = (angular_step / 360) * pi * particle_diameter
        coarse_resolution = rotated_distance / 1.2       (3D)
        image_coarse_size = 2 * ceil(pixel_size * ori_size / coarse_resolution)

    Parameters
    ----------
    angular_step_deg : float
        Effective angular step in degrees (after oversampling).
    pixel_size : float
        Pixel size in Angstrom.
    ori_size : int
        Original image box size in pixels.
    particle_diameter : float or None
        Particle diameter in Angstrom.  If None, use box_size * pixel_size.

    Returns
    -------
    coarse_size : int
        Coarse image size (diameter in pixels), clamped to [8, ori_size].
    """
    if particle_diameter is None:
        particle_diameter = ori_size * pixel_size

    rotated_distance = (angular_step_deg / 360.0) * np.pi * particle_diameter
    coarse_resolution = rotated_distance / 1.2  # keepsafe_factor for 3D

    if coarse_resolution <= 0:
        return ori_size

    coarse_size = int(2 * np.ceil(pixel_size * ori_size / coarse_resolution))
    coarse_size = max(8, min(coarse_size, ori_size))
    return coarse_size


def clamp_relion_coarse_image_size(coarse_size, current_size, ori_size):
    """Clamp pass-1 image size the way RELION does.

    RELION computes ``image_coarse_size`` from the angular step and particle
    diameter, then clamps it to ``image_current_size`` rather than forcing a
    smaller fallback. See ``ml_optimiser.cpp`` around the
    ``image_coarse_size = XMIPP_MIN(image_current_size, image_coarse_size)``
    update.
    """
    coarse_size = int(coarse_size)
    if coarse_size % 2 != 0:
        coarse_size += 1
    coarse_size = max(8, min(coarse_size, int(ori_size)))
    if current_size is None:
        return coarse_size
    return min(int(current_size), coarse_size)


def should_skip_adaptive_pass2(
    significant_counts,
    n_rotations,
    n_translations,
    *,
    threshold=ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION,
):
    """Return whether adaptive pass 2 should be skipped for this batch.

    RELION's two-pass search only helps when significance pruning is actually
    selective. If most coarse samples remain significant, the fine pass is pure
    overhead. We therefore disable pass 2 whenever the mean fraction of
    significant coarse samples is at least ``threshold``.
    """
    if threshold is None or float(threshold) < 0.0:
        return False, 0.0
    total_samples = max(int(n_rotations) * int(n_translations), 1)
    sig_counts = np.asarray(significant_counts, dtype=np.float32)
    mean_fraction = float(np.mean(sig_counts) / total_samples)
    return mean_fraction >= float(threshold), mean_fraction


def _bootstrap_current_size_relion(init_current_size: int, ori_size: int, incr_size: int = 10) -> int:
    """Match RELION's first expectation-time current_size growth step.

    RELION seeds the initial resolution from ``--ini_high`` and then immediately
    calls ``updateImageSizeAndResolutionPointers()`` before the first E-step.
    At startup ``ave_Pmax == 0`` and ``has_high_fsc_at_limit == false``, so the
    first current_size is the initial resolution shell plus ``incr_size``.
    """
    init_shell = max(0, int(np.ceil(init_current_size / 2.0)))
    raw_cs = compute_current_size_relion(
        init_shell,
        ori_size,
        ave_Pmax=0.0,
        has_high_fsc_at_limit=False,
        incr_size=incr_size,
    )
    return quantize_current_size(raw_cs, ori_size=ori_size)


def bootstrap_current_size_from_ini_high_relion(
    ori_size: int,
    voxel_size: float,
    ini_high_angstrom: float | None,
    incr_size: int = 10,
) -> int | None:
    """Bootstrap RELION's first current_size directly from ``--ini_high``."""
    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return None
    init_shell = max(1, int(np.round(float(ori_size) * float(voxel_size) / float(ini_high_angstrom))))
    return _bootstrap_current_size_relion(2 * init_shell, ori_size=ori_size, incr_size=incr_size)


def fsc_to_current_size(fsc, threshold=1.0 / 7.0, min_size=32):
    """Convert an FSC curve to a current_size (diameter in pixels).

    Parameters
    ----------
    fsc : array-like, shape (n_shells,)
        FSC curve between half-maps.
    threshold : float
        FSC threshold for resolution cutoff.  Default 1/7 ~ 0.143.
    min_size : int
        Minimum returned size (prevents collapse to 0 at first iteration).

    Returns
    -------
    int
        Raw current_size = 2 * shell_index.  Needs quantization before use.
    """
    from recovar.heterogeneity.locres import find_fsc_resol

    fsc_arr = jnp.asarray(fsc)
    pixel_res = float(find_fsc_resol(fsc_arr, threshold=threshold))

    # current_size = 2 * shell_index (Nyquist: need 2 pixels per cycle)
    raw_size = int(2 * pixel_res)
    return max(raw_size, min_size)


def _k1_data_vs_prior_for_scheduling(
    *,
    raw_fsc,
    corrected_data_vs_prior,
    current_size,
    grid_size,
    tau2_fudge,
    dtype=np.float32,
):
    """Return the K=1 DVP curve RELION uses for current-resolution updates.

    Auto-refine normally uses raw split-half FSC. If RELION's
    ``--solvent_correct_fsc`` path is enabled, the corrected FSC-derived DVP
    is passed here instead.
    """
    if corrected_data_vs_prior is not None:
        return _truncate_data_vs_prior_for_current_size(
            corrected_data_vs_prior,
            current_size=current_size,
            grid_size=grid_size,
            dtype=dtype,
        )

    runtime_dtype = dtype
    fsc_prev = np.asarray(raw_fsc, dtype=runtime_dtype).copy()
    if int(current_size) < int(grid_size):
        fsc_prev[min(len(fsc_prev), int(current_size) // 2 + 1) :] = 0.0
    return np.asarray(fsc_to_relion_ssnr(fsc_prev, tau2_fudge=tau2_fudge), dtype=runtime_dtype)


def _truncate_data_vs_prior_for_current_size(data_vs_prior, *, current_size, grid_size, dtype=np.float32):
    """Zero DVP shells beyond RELION's inclusive current-size boundary."""
    truncated = np.asarray(data_vs_prior, dtype=dtype).copy()
    if int(current_size) < int(grid_size):
        first_unavailable_shell = min(truncated.shape[-1], int(current_size) // 2 + 1)
        truncated[..., first_unavailable_shell:] = 0.0
    return truncated


def _truncate_fsc_for_current_size_growth(fsc, *, current_size, grid_size, dtype=np.float32):
    """Zero FSC shells beyond RELION's inclusive current-size boundary.

    BackProjector includes radii ``R <= current_size / 2``.  The boundary
    shell therefore remains part of RELION's full-array FSC threshold scan;
    only shells starting at ``current_size // 2 + 1`` are unavailable.
    """
    truncated = np.asarray(fsc, dtype=dtype).copy()
    if int(current_size) < int(grid_size):
        first_unavailable_shell = min(len(truncated), int(current_size) // 2 + 1)
        truncated[first_unavailable_shell:] = 0.0
    return truncated


def _firstiter_cc_ini_high_resolution_shell(grid_size, voxel_size, ini_high_angstrom):
    """RELION's firstiter_cc current-resolution shell from ``--ini_high``."""
    px = float(voxel_size if voxel_size > 0 else 1.0)
    shell = int(np.floor(int(grid_size) * px / float(ini_high_angstrom) + 0.5))
    return max(1, min(int(grid_size) // 2, shell))


def _firstiter_cc_scheduling_resolution_shell(
    resolution_shell,
    *,
    emulate_relion_firstiter_cc,
    ini_high_angstrom,
    relion_iteration,
    grid_size,
    voxel_size,
):
    """Apply RELION's iter-1 ``--firstiter_cc`` current-size rule.

    ``MlOptimiser::updateCurrentResolution`` uses the ``--ini_high`` shell at
    physical iteration 1 regardless of whether refinement has one or multiple
    classes. Keep this override independent of the K=1/K-class data-vs-prior
    calculation so both paths schedule iteration 2 identically.
    """

    if (
        emulate_relion_firstiter_cc
        and ini_high_angstrom is not None
        and int(relion_iteration) == 1
    ):
        return _firstiter_cc_ini_high_resolution_shell(
            grid_size,
            voxel_size,
            ini_high_angstrom,
        )
    return int(resolution_shell)


def _firstiter_cc_ini_high_tau2_taper(
    n_shells,
    grid_size,
    voxel_size,
    ini_high_angstrom,
    *,
    filter_edgewidth,
):
    """RELION's squared post-firstiter ``ini_high`` taper for tau2 state."""

    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return np.ones(int(n_shells), dtype=np.float64)
    edge = float(filter_edgewidth)
    radius = float(grid_size) * float(voxel_size) / float(ini_high_angstrom) - edge / 2.0
    radius_p = radius + edge
    shells = np.arange(int(n_shells), dtype=np.float64)
    taper = np.ones(int(n_shells), dtype=np.float64)
    taper[shells > radius_p] = 0.0
    transition = (shells >= radius) & (shells <= radius_p)
    taper[transition] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - shells[transition]) / edge)
    return taper * taper
