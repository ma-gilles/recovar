"""Core refinement loop for dense single-volume EM.

This file contains the three core algorithm functions:
- ``refine_single_volume`` — public entry point
- ``_run_relion_iteration_loop`` — RELION-parity iteration loop
- ``_run_local_search_iteration`` — exact local angular search

All supporting helpers live in ``helpers/``.
See ``docs/math/relion_refinement_algorithm.md`` for the full algorithm map.
"""

import logging
import os
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils
from recovar.em.core import hard_assignment_idx_to_pose
from recovar.em.dense_single_volume import parity_dump as _parity_dump
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.dense_single_volume.helpers.convergence import (
    LOCAL_SEARCH_HEALPIX_ORDER,
    RefinementState,
    calculate_expected_angular_errors,
    healpix_angular_step,
    update_refinement_state,
)
from recovar.em.dense_single_volume.helpers.fourier_window import quantize_current_size
from recovar.em.dense_single_volume.helpers.local_search import _local_search_engine_rotation_block_size
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    collapse_rotation_posterior_to_direction_prior,
    infer_direction_prior_healpix_order,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    normalize_direction_prior_per_half,
    relion_translation_prior_center,
    relion_translation_search_base,
    remap_direction_prior_to_healpix_order,
)
from recovar.em.dense_single_volume.helpers.oversampling import compute_pass2_stats
from recovar.em.dense_single_volume.helpers.resolution import (
    ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION,
    _bootstrap_current_size_relion,
    bootstrap_current_size_from_ini_high_relion,
    clamp_relion_coarse_image_size,
    compute_coarse_image_size,
    shell_index_to_resolution_angstrom,
    should_skip_adaptive_pass2,
)
from recovar.em.dense_single_volume.helpers.types import NoiseStats, RelionStats
from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact
from recovar.em.dense_single_volume.local_layout import (
    _selected_rotation_matrices,
    build_local_hypothesis_layout,
    build_pass2_hypothesis_layout,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights, make_shell_indices_half
from recovar.em.sampling import (
    advance_relion_perturbation,
    apply_relion_rotation_perturbation,
    apply_relion_rotation_perturbation_to_eulers,
    apply_relion_translation_perturbation,
    build_local_search_grid_metadata,
    get_oversampled_translation_grid,
    get_relion_rotation_grid,
    get_relion_rotation_grid_eulers,
    get_translation_grid,
    read_relion_direction_prior,
    read_relion_model_metadata,
    read_relion_optimiser_metadata,
    read_relion_sampling_metadata,
    relion_angular_sampling_deg,
    rotation_grid_size,
)
from recovar.reconstruction.regularization import (
    compute_current_size_relion,
    fsc_to_relion_ssnr,
    resolution_from_data_vs_prior,
    update_relion_growth_state_from_fsc,
)

logger = logging.getLogger(__name__)

RELION_SCORE_TENSOR_FLOAT_BUDGET = 200_000_000
RELION_MAX_FULL_GRID_ORDER = 4


def _precompute_exact_local_fine_grid_enabled(healpix_order: int) -> bool:
    """Return whether exact local search should materialize the fine grid once."""

    raw = os.environ.get("RECOVAR_RELION_EXACT_LOCAL_PRECOMPUTE_FINE_GRID", "auto").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"", "auto"}:
        max_rotations = int(os.environ.get("RECOVAR_RELION_EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS", "3000000"))
        return rotation_grid_size(int(healpix_order)) <= max_rotations
    raise ValueError(
        "RECOVAR_RELION_EXACT_LOCAL_PRECOMPUTE_FINE_GRID must be one of auto/1/0/true/false",
    )


def _parse_env_int_set(value: str | None) -> set[int] | None:
    if not value:
        return None
    parsed = set()
    for token in value.replace(",", " ").split():
        token = token.strip()
        if token:
            parsed.add(int(token))
    return parsed or None


def _relion_half_plane_shell_counts(image_shape):
    """Count RELION's non-redundant FFTW half-plane shell pixels."""

    height, width = int(image_shape[0]), int(image_shape[1])
    n_shells = height // 2 + 1
    counts = np.zeros(n_shells, dtype=np.float64)
    for iy in range(height):
        ky = iy if iy <= height // 2 else iy - height
        for ix in range(width // 2 + 1):
            # RELION excludes redundant jp==0, ip<0 FFTW half-plane entries.
            if ix == 0 and ky < 0:
                continue
            shell = int(np.rint(np.sqrt(float(ky * ky + ix * ix))))
            if shell < n_shells:
                counts[shell] += 1.0
    return counts


def _maybe_dump_noise_update_debug(
    *,
    iteration: int,
    current_size: int | None,
    image_shape,
    noise_stats_per_half,
    previous_noise_radial_per_half,
    noise_from_res_per_half,
    noise_from_res,
):
    """Write raw noise M-step terms for RELION parity debugging when requested."""

    dump_dir = os.environ.get("RECOVAR_NOISE_DEBUG_DUMP_DIR")
    if not dump_dir:
        return
    requested_iterations = _parse_env_int_set(os.environ.get("RECOVAR_NOISE_DEBUG_DUMP_ITERATION"))
    if requested_iterations is not None and int(iteration) not in requested_iterations:
        return

    os.makedirs(dump_dir, exist_ok=True)
    n_shells = int(image_shape[0]) // 2 + 1
    shell_indices_half = np.asarray(make_shell_indices_half(image_shape), dtype=np.int64)
    half_counts = np.bincount(shell_indices_half, minlength=n_shells).astype(np.float64)[:n_shells]
    half_weights = np.asarray(make_half_image_weights(image_shape), dtype=np.float64)
    half_weighted_counts = np.bincount(shell_indices_half, weights=half_weights, minlength=n_shells).astype(
        np.float64,
    )[:n_shells]

    payload = {
        "zero_based_iteration": np.array([int(iteration)], dtype=np.int32),
        "one_based_iteration": np.array([int(iteration) + 1], dtype=np.int32),
        "current_size": np.array([-1 if current_size is None else int(current_size)], dtype=np.int32),
        "image_shape": np.asarray(image_shape, dtype=np.int32),
        "shell_index_half": shell_indices_half.astype(np.int32),
        "half_shell_counts": half_counts,
        "half_weighted_shell_counts": half_weighted_counts,
        "relion_half_plane_shell_counts": _relion_half_plane_shell_counts(image_shape),
        "mean_sigma2_noise": np.asarray(noise_from_res, dtype=np.float64),
    }
    for half_id, stats_k in enumerate(noise_stats_per_half, start=1):
        prefix = f"half{half_id}"
        wsum_sigma2 = np.asarray(stats_k.wsum_sigma2_noise, dtype=np.float64)
        img_power = np.asarray(stats_k.wsum_img_power, dtype=np.float64)
        payload[f"{prefix}_wsum_sigma2_noise"] = wsum_sigma2
        payload[f"{prefix}_wsum_img_power"] = img_power
        payload[f"{prefix}_wsum_total"] = wsum_sigma2 + img_power
        payload[f"{prefix}_sumw"] = np.array([float(stats_k.sumw)], dtype=np.float64)
        payload[f"{prefix}_sigma2_noise"] = np.asarray(noise_from_res_per_half[half_id - 1], dtype=np.float64)
        payload[f"{prefix}_previous_sigma2_noise"] = np.asarray(
            previous_noise_radial_per_half[half_id - 1],
            dtype=np.float64,
        )
        if getattr(stats_k, "wsum_noise_a2", None) is not None:
            payload[f"{prefix}_wsum_noise_a2"] = np.asarray(stats_k.wsum_noise_a2, dtype=np.float64)
        if getattr(stats_k, "wsum_noise_xa", None) is not None:
            payload[f"{prefix}_wsum_noise_xa"] = np.asarray(stats_k.wsum_noise_xa, dtype=np.float64)

    path = os.path.join(dump_dir, f"recovar_noise_update_it{int(iteration) + 1:03d}.npz")
    np.savez_compressed(path, **payload)
    logger.info("Wrote RECOVAR noise update debug dump: %s", path)


# TRACKED TODOs: RELION_LOCAL_ENGINE
# TODO(RELION_LOCAL_ENGINE/T002): active RELION local path uses exact per-image local hypotheses
# TODO(RELION_LOCAL_ENGINE/T003): local path should not depend on dense shared-grid engine contracts
# TODO(RELION_LOCAL_ENGINE/T004): parity hacks should move inward, out of outer-loop control flow
# See docs/relion_local_engine_refactor.md

# RELION stores windowFourierTransform(in, out, current_size) as a rectangular
# FFTW half image, but the likelihood support is the nonzero Minvsigma2 mask:
# rounded radial shells, no DC, no redundant negative-row kx=0 entries.
RELION_FOURIER_WINDOW_SQUARE = False
# RELION's --minres_map default: do not add the Wiener prior term to the
# lowest Fourier shells during MAP reconstruction.
RELION_MINRES_MAP = 5


def _enable_relion_parity_defaults():
    """Enable source-matched RELION arithmetic unless explicitly overridden."""
    defaults = {
        # RELION's accelerated E-step computes diff2 directly rather than via
        # an algebraically equivalent <y, A x> - |A x|^2 expansion.
        "RECOVAR_RELION_DIRECT_DIFF2_SCORING": "1",
        # RELION's CUDA projector samples Fourier references through
        # cudaFilterModeLinear texture objects.
        "RECOVAR_RELION_TEXTURE_INTERP": "1",
        # RELION image preprocessing uses FFTW-style centered complex FFTs.
        "RECOVAR_RELION_NUMPY_IMAGE_FFT": "1",
        # RELION's BackProjector accumulates into a compact half-volume,
        # folds negative stored-axis coordinates before interpolation, then
        # enforces Hermitian symmetry on the x=0 plane before maximization.
        "RECOVAR_RELION_SPARSE_PASS2_HALF_VOLUME": "1",
        "RECOVAR_RELION_BACKPROJECT_FOLD_X": "1",
        "RECOVAR_RELION_SPARSE_PASS2_HALF_VOLUME_ENFORCE_X0": "1",
    }
    enabled = []
    for name, value in defaults.items():
        if name not in os.environ:
            os.environ[name] = value
            enabled.append(name)
    if enabled:
        logger.info("RELION mode parity defaults enabled: %s", ", ".join(enabled))


def _relion_use_float64_scoring() -> bool:
    """Return whether RELION-mode E-step scoring should upcast to float64.

    RELION's accelerated path stores E-step weights as XFLOAT, which is float
    unless RELION is compiled with ACC_DOUBLE_PRECISION. Keep the old double
    path available for diagnostics by setting RECOVAR_RELION_FLOAT32_SCORING=0.
    """
    return os.environ.get("RECOVAR_RELION_FLOAT32_SCORING", "1").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _relion_exact_local_image_batch_override() -> int | None:
    """Return an optional debug override for exact local-search image batches."""
    raw = os.environ.get("RECOVAR_RELION_EXACT_LOCAL_IMAGE_BATCH_SIZE")
    if raw is None or raw.strip() == "":
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("RECOVAR_RELION_EXACT_LOCAL_IMAGE_BATCH_SIZE must be positive")
    return value


def _replay_control_model_iteration(init_relion_iteration: int, loop_iteration: int) -> int:
    """Return the RELION model.star index whose control state governs this replay step."""
    return int(init_relion_iteration) + int(loop_iteration) + 1


def _normalize_noise_variance_per_half(init_noise_variance, n_halves=2):
    """Return a list of per-half flattened noise-variance arrays.

    RELION stores and updates ``sigma2_noise`` separately for each half-model.
    Legacy RECOVAR callers pass one shared image-shaped array; keep that path
    by duplicating the shared vector.
    """
    if n_halves <= 0:
        raise ValueError(f"n_halves must be positive, got {n_halves}")

    if isinstance(init_noise_variance, (list, tuple)):
        if len(init_noise_variance) != n_halves:
            raise ValueError(
                f"Expected {n_halves} per-half noise arrays, got {len(init_noise_variance)}",
            )
        per_half = [jnp.asarray(noise_k).reshape(-1) for noise_k in init_noise_variance]
    else:
        noise_arr = jnp.asarray(init_noise_variance)
        if noise_arr.ndim == 1:
            shared = noise_arr.reshape(-1)
            per_half = [jnp.array(shared) for _ in range(n_halves)]
        elif noise_arr.ndim == 2 and noise_arr.shape[0] == n_halves:
            per_half = [jnp.asarray(noise_arr[k]).reshape(-1) for k in range(n_halves)]
        else:
            raise ValueError(
                "init_noise_variance must be a flat shared array or a "
                f"({n_halves}, image_size) per-half array; got shape {tuple(noise_arr.shape)}",
            )

    sizes = [int(noise_k.size) for noise_k in per_half]
    if len(set(sizes)) != 1:
        raise ValueError(f"Per-half noise arrays must have the same size; got {sizes}")
    return per_half


def _mean_noise_variance(noise_variance_per_half):
    """Average per-half image noise for diagnostics and compatibility outputs."""
    return jnp.mean(
        jnp.stack([jnp.asarray(noise_k).reshape(-1) for noise_k in noise_variance_per_half], axis=0),
        axis=0,
    )


def _optional_float32_half_pair(values):
    """Return optional per-half arrays normalized to float32."""
    if values is None:
        return [None, None]
    return [
        np.asarray(values[0], dtype=np.float32) if values[0] is not None else None,
        np.asarray(values[1], dtype=np.float32) if values[1] is not None else None,
    ]


def _normalize_logged_float32_half_pair(values, *, label: str):
    """Normalize per-half correction arrays and log summary statistics."""
    per_half = _optional_float32_half_pair(values)
    for k, arr in enumerate(per_half):
        if arr is None:
            continue
        if arr.size:
            logger.info(
                "RELION mode: %s half-%d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f (%d images)",
                label,
                k + 1,
                arr.mean(),
                arr.std(),
                arr.min(),
                arr.max(),
                len(arr),
            )
        else:
            logger.info("RELION mode: %s half-%d: empty", label, k + 1)
    return per_half


@dataclass
class _RelionHalfInputState:
    """Mutable per-half inputs carried across replay and local-search iterations."""

    previous_best_translations: list
    previous_best_rotation_eulers: list
    image_corrections: list
    scale_corrections: list

    @classmethod
    def from_initial_values(
        cls,
        *,
        previous_best_translations,
        previous_best_rotation_eulers,
        image_corrections,
        scale_corrections,
    ):
        return cls(
            previous_best_translations=_optional_float32_half_pair(previous_best_translations),
            previous_best_rotation_eulers=_optional_float32_half_pair(previous_best_rotation_eulers),
            image_corrections=_normalize_logged_float32_half_pair(
                image_corrections,
                label="image_corrections",
            ),
            scale_corrections=_normalize_logged_float32_half_pair(
                scale_corrections,
                label="scale_corrections",
            ),
        )

def _relion_rotation_grid_float32(healpix_order: int):
    """Return RELION rotation matrices/eulers using the loop's float32 policy."""
    order = int(healpix_order)
    return (
        get_relion_rotation_grid(order).astype(np.float32),
        get_relion_rotation_grid_eulers(order).astype(np.float32),
    )


def _radial_profile_from_noise_variance(noise_variance, image_shape):
    """Average an image-shaped noise vector into integer radial shells."""
    n_shells = image_shape[0] // 2 + 1
    radial_dist = np.clip(
        fourier_transform_utils.get_grid_of_radial_distances(
            image_shape,
            scaled=False,
            frequency_shift=0,
        )
        .astype(int)
        .reshape(-1),
        0,
        n_shells - 1,
    )
    noise_np = np.asarray(noise_variance, dtype=np.float64).reshape(-1)
    radial = np.zeros(n_shells, dtype=np.float64)
    counts = np.zeros(n_shells, dtype=np.float64)
    np.add.at(radial, radial_dist[: noise_np.size], noise_np)
    np.add.at(counts, radial_dist[: noise_np.size], 1.0)
    return radial / np.maximum(counts, 1.0)


def _reconstruct_volume_eager(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    tau,
    tau2_fudge,
    projection_padding_factor,
    use_spherical_mask=True,
    grid_correct=True,
    minres_map=0,
):
    """Eager RELION-style reconstruction from full or half Fourier accumulators.

    This keeps the reconstruction boundary out of a single monolithic JIT while
    letting the local exact path keep its accumulators in packed half-volume
    layout until the final iDFT boundary.
    """
    from recovar.reconstruction import relion_functions

    return relion_functions.post_process_from_filter_v2(
        Ft_ctf,
        Ft_y,
        vol_shape,
        padding_factor,
        tau=tau,
        kernel="triangular",
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct="radial",
        kernel_width=1,
        tau2_fudge=tau2_fudge,
        gridding_padding_factor=projection_padding_factor,
        minres_map=minres_map,
    )


def _apply_relion_initial_lowpass_filter(
    volume_ft_flat, volume_shape, voxel_size, ini_high_angstrom, filter_edgewidth=5
):
    """Apply RELION's ``initialLowPassFilterReferences`` to a full Fourier volume."""
    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return volume_ft_flat
    from recovar.heterogeneity import locres

    filtered = locres.low_pass_filter_map(
        jnp.asarray(volume_ft_flat).reshape(volume_shape),
        volume_shape[0],
        float(ini_high_angstrom),
        float(voxel_size),
        int(filter_edgewidth),
        do_highpass_instead=False,
        volume_shape=volume_shape,
    )
    return filtered.reshape(-1)


def _align_fourier_volume_sign_to_reference(volume_ft_flat, reference_ft_flat, volume_shape):
    """Keep reconstructed volumes on the same real-space sign branch as the reference."""
    if reference_ft_flat is None:
        return volume_ft_flat, False
    vol_real = np.asarray(
        fourier_transform_utils.get_idft3(jnp.asarray(volume_ft_flat).reshape(volume_shape)),
        dtype=np.float64,
    ).reshape(-1)
    ref_real = np.asarray(
        fourier_transform_utils.get_idft3(jnp.asarray(reference_ft_flat).reshape(volume_shape)),
        dtype=np.float64,
    ).reshape(-1)
    vol_centered = vol_real - float(np.mean(vol_real))
    ref_centered = ref_real - float(np.mean(ref_real))
    overlap = float(np.dot(ref_centered, vol_centered))
    if overlap < 0.0:
        return -volume_ft_flat, True
    return volume_ft_flat, False


def _run_local_search_iteration(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    prior_rotations,
    rotation_grid_rotations,
    rotation_grid_eulers,
    healpix_order,
    sigma_rot,
    sigma_psi,
    translations,
    prior_translations,
    sigma_offset_angstrom,
    offset_range_pixels,
    disc_type,
    image_batch_size,
    rotation_block_size,
    current_size,
    *,
    accumulate_noise=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    use_float64_scoring=False,
    use_float64_projections=False,
    do_gridding_correction=False,
    square_window=False,
    half_spectrum_scoring=False,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    score_with_masked_images=True,
    return_profile=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    adaptive_fraction=0.999,
    max_significants=-1,
    reconstruct_significant_only=True,
    translation_prior_reference_translations=None,
    debug_iteration=None,
    pass2_layout=None,
    return_best_pose_details=False,
    normalization_log_z=None,
    translation_prior_centers=None,
    rotation_grid_random_perturbation=0.0,
    rotation_grid_angular_sampling_deg=None,
):
    """Run exact local search on the fine HEALPix grid.

    Each image carries its own exact prior orientation from the previous
    iteration. ``prior_rotations`` may be either RELION Euler angles
    ``(rot, tilt, psi)`` or rotation matrices.
    """
    return _run_local_search_iteration_exact_v1(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        accumulate_noise=accumulate_noise,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
        half_spectrum_scoring=half_spectrum_scoring,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        score_with_masked_images=score_with_masked_images,
        return_profile=return_profile,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        reconstruct_significant_only=reconstruct_significant_only,
        translation_prior_reference_translations=translation_prior_reference_translations,
        debug_iteration=debug_iteration,
        local_layout_override=pass2_layout,
        return_best_pose_details=return_best_pose_details,
        normalization_log_z=normalization_log_z,
        translation_prior_centers=translation_prior_centers,
        rotation_grid_random_perturbation=rotation_grid_random_perturbation,
        rotation_grid_angular_sampling_deg=rotation_grid_angular_sampling_deg,
    )


def _run_sparse_pass2_local_search_iteration(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    *,
    oversampling_order=1,
    current_size=None,
    translation_step=None,
    rotation_log_prior=None,
    translation_log_prior=None,
    score_with_masked_images=True,
    return_stats=True,
    accumulate_noise=True,
    half_spectrum_scoring=True,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    use_float64_scoring=False,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation=0.0,
    image_batch_size=1,
    rotation_block_size=5000,
    adaptive_fraction=0.999,
    debug_iteration=None,
    return_profile=False,
    normalization_log_z=None,
    translation_prior_centers=None,
):
    """Run RELION adaptive pass 2 through the exact local-search engine."""

    n_images = int(experiment_dataset.n_units)
    translations_np = np.asarray(translations, dtype=np.float32)
    n_coarse_rot = int(rotation_grid_size(nside_level))
    n_coarse_trans = int(translations_np.shape[0])
    pass2_layout = build_pass2_hypothesis_layout(
        significant_sample_indices,
        n_coarse_rot,
        n_coarse_trans,
        int(nside_level),
        translations_np,
        oversampling_order=int(oversampling_order),
        translation_step=translation_step,
        rotation_log_prior=rotation_log_prior,
        translation_log_prior=translation_log_prior,
        random_perturbation=random_perturbation,
    )
    dummy_prior_rotations = np.zeros((n_images, 3), dtype=np.float32)
    dummy_prior_translations = np.zeros((n_images, translations_np.shape[1]), dtype=np.float32)
    dummy_rotation_grid = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], max(n_coarse_rot, 1), axis=0)

    outputs = _run_local_search_iteration(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        dummy_prior_rotations,
        dummy_rotation_grid,
        None,
        int(nside_level),
        0.0,
        0.0,
        pass2_layout.translation_grid,
        dummy_prior_translations,
        1.0,
        None,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        accumulate_noise=accumulate_noise,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        use_float64_scoring=use_float64_scoring,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
        half_spectrum_scoring=half_spectrum_scoring,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        score_with_masked_images=score_with_masked_images,
        return_profile=return_profile,
        adaptive_fraction=adaptive_fraction,
        max_significants=-1,
        debug_iteration=debug_iteration,
        pass2_layout=pass2_layout,
        return_best_pose_details=True,
        normalization_log_z=normalization_log_z,
        translation_prior_centers=translation_prior_centers,
        # RELION re-thresholds fine pass weights only when adaptive
        # oversampling is active. With oversampling_order == 0, FPCMasks already
        # contain the coarse pass-1 significant samples and the final threshold
        # is the minimum selected weight, so all selected samples contribute.
        reconstruct_significant_only=int(oversampling_order) > 0,
    )

    outputs = list(outputs)
    outputs[2] = _decode_pass2_local_hard_assignment(
        pass2_layout,
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
    )

    profile_summary = None
    if return_profile:
        profile_summary = outputs.pop()
    if return_profile:
        if return_stats and accumulate_noise:
            return tuple(outputs + [profile_summary])
        if return_stats:
            return tuple(outputs[:-1] + [profile_summary])
        if accumulate_noise:
            return tuple(outputs[:6] + [outputs[-1], profile_summary])
        return tuple(outputs[:6] + [profile_summary])
    if return_stats and accumulate_noise:
        return tuple(outputs)
    if return_stats:
        return tuple(outputs[:-1])
    if accumulate_noise:
        return tuple(outputs[:6] + [outputs[-1]])
    return tuple(outputs[:6])


def _decode_pass2_local_hard_assignment(
    pass2_layout,
    global_hard_assignment,
    best_pose_rotations,
    best_pose_translations,
    best_pose_rotation_ids,
):
    """Decode exact-local best poses into sparse pass-2 local assignment ids.

    ``run_local_em_exact`` reports hard assignments as global/fine rotation
    ids, which is correct for ordinary local search. Sparse adaptive pass 2
    historically exposes per-image local-row assignments; keep that return
    contract so downstream diagnostics can decode against each image's
    oversampled candidate list.
    """

    n_images = int(pass2_layout.n_images)
    n_trans = int(np.asarray(pass2_layout.translation_grid).shape[0])
    translation_grid = np.asarray(pass2_layout.translation_grid, dtype=np.float32)
    rotation_offsets = np.asarray(pass2_layout.rotation_offsets, dtype=np.int64)
    rotation_ids = np.asarray(pass2_layout.rotation_ids_flat, dtype=np.int64)
    rotations = np.asarray(pass2_layout.rotations_flat, dtype=np.float32)
    sample_mask = (
        None
        if pass2_layout.sample_mask_flat is None
        else np.asarray(pass2_layout.sample_mask_flat, dtype=bool)
    )
    best_rots = np.asarray(best_pose_rotations, dtype=np.float32)
    best_trans = np.asarray(best_pose_translations, dtype=np.float32)
    best_ids = np.asarray(best_pose_rotation_ids, dtype=np.int64).reshape(-1)
    global_assignment = np.asarray(global_hard_assignment, dtype=np.int64).reshape(-1)
    hard_assignment = np.empty(n_images, dtype=np.int32)

    for image_idx in range(n_images):
        start = int(rotation_offsets[image_idx])
        end = int(rotation_offsets[image_idx + 1])
        rows = np.arange(start, end, dtype=np.int64)
        if rows.size == 0:
            raise ValueError(f"Image {image_idx} has no pass-2 local rotations")

        trans_idx = int(global_assignment[image_idx] % n_trans)
        trans_delta = float(np.max(np.abs(translation_grid[trans_idx] - best_trans[image_idx])))
        if trans_delta > 1e-5:
            raise RuntimeError(
                f"Could not decode sparse pass-2 translation for image {image_idx}: "
                f"engine hard-assignment delta={trans_delta:.3e}",
            )

        valid = rotation_ids[rows] == int(best_ids[image_idx])
        if sample_mask is not None:
            valid &= sample_mask[rows, trans_idx]
        candidate_rows = rows[valid]
        if candidate_rows.size == 0 and sample_mask is not None:
            candidate_rows = rows[sample_mask[rows, trans_idx]]
        if candidate_rows.size == 0:
            candidate_rows = rows

        rot_delta = np.max(
            np.abs(rotations[candidate_rows] - best_rots[image_idx][None, :, :]),
            axis=(1, 2),
        )
        best_delta = float(np.min(rot_delta))
        best_row = int(candidate_rows[int(np.argmin(rot_delta))])
        if best_delta > 1e-5:
            raise RuntimeError(
                f"Could not decode sparse pass-2 rotation for image {image_idx}: "
                f"nearest delta={best_delta:.3e}",
            )
        if sample_mask is not None and not bool(sample_mask[best_row, trans_idx]):
            raise RuntimeError(
                f"Decoded sparse pass-2 assignment is outside the candidate mask for image {image_idx}",
            )

        hard_assignment[image_idx] = np.int32((best_row - start) * n_trans + trans_idx)

    return hard_assignment


def _run_local_search_iteration_exact_v1(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    prior_rotations,
    rotation_grid_rotations,
    rotation_grid_eulers,
    healpix_order,
    sigma_rot,
    sigma_psi,
    translations,
    prior_translations,
    sigma_offset_angstrom,
    offset_range_pixels,
    disc_type,
    image_batch_size,
    rotation_block_size,
    current_size,
    *,
    accumulate_noise=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    use_float64_scoring=False,
    use_float64_projections=False,
    do_gridding_correction=False,
    square_window=False,
    half_spectrum_scoring=False,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    score_with_masked_images=True,
    return_profile=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    adaptive_fraction=0.999,
    max_significants=-1,
    reconstruct_significant_only=True,
    translation_prior_reference_translations=None,
    debug_iteration=None,
    local_layout_override=None,
    return_best_pose_details=False,
    normalization_log_z=None,
    translation_prior_centers=None,
    rotation_grid_random_perturbation=0.0,
    rotation_grid_angular_sampling_deg=None,
):
    """Per-image exact local engine over image-specific rotation neighborhoods."""

    rotation_block_size = _local_search_engine_rotation_block_size(rotation_block_size)
    prior_rotations = np.asarray(prior_rotations, dtype=np.float32)
    if prior_rotations.ndim == 3:
        n_prior = prior_rotations.shape[0]
    elif prior_rotations.ndim == 2 and prior_rotations.shape[1] == 3:
        n_prior = prior_rotations.shape[0]
    else:
        raise ValueError(f"prior_rotations must have shape (n,3,3) or (n,3), got {prior_rotations.shape}")
    if prior_translations is None:
        prior_translations = np.zeros(
            (n_prior, np.asarray(translations).shape[1]),
            dtype=np.float32,
        )
    else:
        prior_translations = np.asarray(prior_translations, dtype=np.float32).reshape(
            -1,
            np.asarray(translations).shape[1],
        )

    if local_layout_override is None:
        metadata_t0 = time.time()
        # RELION local priors remain factorized in canonical direction/psi index
        # space even when the scored trial rotations have been perturbed.
        local_grid_metadata = build_local_search_grid_metadata(healpix_order)
        metadata_build_time = time.time() - metadata_t0

        layout_t0 = time.time()
        local_layout = build_local_hypothesis_layout(
            prior_rotations,
            rotation_grid_rotations,
            sigma_rot,
            sigma_psi,
            healpix_order,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            # Match the grouped RELION-mode path: local translation priors use the
            # learned/model sigma, not the legacy range/3 override.
            None,
            experiment_dataset.voxel_size,
            grid_metadata=local_grid_metadata,
            translation_prior_reference_translations=translation_prior_reference_translations,
            rotation_grid_random_perturbation=rotation_grid_random_perturbation,
            rotation_grid_angular_sampling_deg=rotation_grid_angular_sampling_deg,
        )
        selector_time = time.time() - layout_t0
    else:
        local_layout = local_layout_override
        metadata_build_time = 0.0
        selector_time = 0.0

    engine_outputs = run_local_em_exact(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        disc_type,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        current_size=current_size,
        accumulate_noise=accumulate_noise,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        use_float64_scoring=use_float64_scoring,
        use_float64_normalization=use_float64_scoring,
        use_float64_projections=use_float64_projections,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        return_profile=return_profile,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        reconstruct_significant_only=reconstruct_significant_only,
        adaptive_fraction=adaptive_fraction,
        # RELION's maximum_significants cap is used to define the coarse pass-1
        # adaptive support. In pass 2, the reconstruction threshold is governed
        # by adaptive_fraction only; do not reapply the cap here.
        max_significants=-1,
        debug_iteration=debug_iteration,
        return_best_pose_details=return_best_pose_details,
        normalization_log_z=normalization_log_z,
        translation_prior_centers=translation_prior_centers,
    )

    if accumulate_noise:
        if return_profile:
            if return_best_pose_details:
                (
                    Ft_y,
                    Ft_ctf,
                    hard_assignment,
                    best_pose_rotations,
                    best_pose_translations,
                    best_pose_rotation_ids,
                    relion_stats,
                    noise_stats,
                    profile_summary,
                ) = engine_outputs
            else:
                Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats, profile_summary = engine_outputs
        else:
            if return_best_pose_details:
                (
                    Ft_y,
                    Ft_ctf,
                    hard_assignment,
                    best_pose_rotations,
                    best_pose_translations,
                    best_pose_rotation_ids,
                    relion_stats,
                    noise_stats,
                ) = engine_outputs
            else:
                Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats = engine_outputs
            profile_summary = None
    else:
        if return_profile:
            if return_best_pose_details:
                (
                    Ft_y,
                    Ft_ctf,
                    hard_assignment,
                    best_pose_rotations,
                    best_pose_translations,
                    best_pose_rotation_ids,
                    relion_stats,
                    profile_summary,
                ) = engine_outputs
            else:
                Ft_y, Ft_ctf, hard_assignment, relion_stats, profile_summary = engine_outputs
        else:
            if return_best_pose_details:
                (
                    Ft_y,
                    Ft_ctf,
                    hard_assignment,
                    best_pose_rotations,
                    best_pose_translations,
                    best_pose_rotation_ids,
                    relion_stats,
                ) = engine_outputs
            else:
                Ft_y, Ft_ctf, hard_assignment, relion_stats = engine_outputs
            profile_summary = None
            noise_stats = None

    if return_profile and profile_summary is not None:
        profile_summary = dict(profile_summary)
        profile_summary["metadata_build_time_s"] = np.float64(metadata_build_time)
        profile_summary["selector_time_s"] = np.float64(selector_time)
        profile_summary["translation_prior_time_s"] = np.float64(0.0)

    if accumulate_noise:
        if return_best_pose_details:
            if return_profile:
                return (
                    Ft_y,
                    Ft_ctf,
                    hard_assignment,
                    best_pose_rotations,
                    best_pose_translations,
                    best_pose_rotation_ids,
                    relion_stats,
                    noise_stats,
                    profile_summary,
                )
            return (
                Ft_y,
                Ft_ctf,
                hard_assignment,
                best_pose_rotations,
                best_pose_translations,
                best_pose_rotation_ids,
                relion_stats,
                noise_stats,
            )
        if return_profile:
            return Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats, profile_summary
        return Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats
    if return_best_pose_details:
        if return_profile:
            return (
                Ft_y,
                Ft_ctf,
                hard_assignment,
                best_pose_rotations,
                best_pose_translations,
                best_pose_rotation_ids,
                relion_stats,
                profile_summary,
            )
        return Ft_y, Ft_ctf, hard_assignment, best_pose_rotations, best_pose_translations, best_pose_rotation_ids, relion_stats
    if return_profile:
        return Ft_y, Ft_ctf, hard_assignment, relion_stats, profile_summary
    return Ft_y, Ft_ctf, hard_assignment, relion_stats


from recovar.em.dense_single_volume.helpers.significance import (
    _compute_significance_batched,
)

# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------


def refine_single_volume(
    experiment_datasets,
    init_volume,
    init_noise_variance,
    init_mean_variance,
    rotations,
    translations,
    disc_type="linear_interp",
    max_iter=10,
    image_batch_size=500,
    rotation_block_size=5000,
    relion_current_sizes=None,
    init_current_size=32,
    fsc_threshold=1.0 / 7.0,
    adaptive_oversampling=0,
    adaptive_fraction=0.999,
    max_significants=500,
    nside_level=None,
    translation_pixel_offset=None,
    mode="relion",
    adaptive_pass2_skip_threshold=ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION,
    # --- RELION-mode parameters ---
    init_healpix_order=2,
    max_healpix_order=7,
    init_translation_range=10.0,
    init_translation_step=2.0,
    init_translation_sigma_angstrom=10.0,
    particle_diameter_ang=None,
    save_intermediates_dir=None,
    low_resol_join_halves_angstrom=40.0,
    tau2_fudge=1.0,
    perturb_factor=0.0,
    perturb_seed=None,
    perturb_replay_relion_dir=None,
    init_fsc=None,
    init_ave_Pmax=None,
    init_has_high_fsc_at_limit=None,
    init_relion_iteration=0,
    init_image_corrections=None,
    init_scale_corrections=None,
    init_direction_prior=None,
    init_previous_best_translations=None,
    init_previous_best_rotation_eulers=None,
    replay_iteration_overrides=None,
    skip_final_iteration=False,
    local_search_profile_mode="auto",
    local_search_translation_prior_mode="coarse",
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    emulate_relion_firstiter_cc=False,
    relion_firstiter_ini_high_angstrom=None,
    first_iteration_score_mode="gaussian",
    first_iteration_reconstruction_mode="soft",
    force_max_iter_after_convergence=False,
):
    """Multi-iteration RELION-parity EM refinement.

    The legacy FSC-driven refinement loop has been removed. ``mode`` remains
    as a compatibility keyword for callers that already pass ``"relion"``,
    but ``"relion"`` is the only supported value.

    Parameters
    ----------
    experiment_datasets : list of 2 dataset objects
        Half-set datasets (same format as split_E_M_v2 expects).
    init_volume : jnp.ndarray, shape (volume_size,)
        Initial volume in Fourier space.
    init_noise_variance : jnp.ndarray, shape (image_size,)
        Initial per-pixel noise variance.
    init_mean_variance : jnp.ndarray, shape (volume_size,)
        Initial signal prior (tau^2).
    rotations : np.ndarray, shape (n_rot, 3, 3)
        Optional initial rotation grid for compatibility. RELION mode
        regenerates grids from the HEALPix refinement state.
    translations : jnp.ndarray, shape (n_trans, 2)
        Translation grid.
    disc_type : str
        Discretization type for forward/adjoint slicing.
    max_iter : int
        Maximum number of iterations.
    image_batch_size : int
        Number of images per GPU batch.
    rotation_block_size : int
        Number of rotations per block in em_engine.
    relion_current_sizes : list of int or None
        Oracle mode: if provided, use these current_sizes instead of
        computing RELION-style current sizes from the FSC/data-vs-prior
        trajectory. relion_current_sizes[i] is used at iteration i.
    init_current_size : int
        Starting current_size for the first iteration (when no FSC is
        available yet).  Ignored if relion_current_sizes is provided.
    fsc_threshold : float
        FSC threshold for resolution estimation.
    adaptive_oversampling : int
        Number of HEALPix subdivision levels for pass 2 (0=disabled,
        1=2x finer = 4 children, 2=4x finer = 16 children).
    adaptive_fraction : float
        Fraction of posterior weight to keep in significance pruning
        (default 0.999 = 99.9%, matching RELION).
    max_significants : int
        Maximum significant (rotation x translation) samples per image.
        Matches RELION's --maxsig semantics (counts SAMPLES, not just
        orientations; see C5 in plan_relion_parity.md).
    nside_level : int or None
        Compatibility keyword for older callers. RELION mode derives the
        coarse rotation grid from ``init_healpix_order``.
    translation_pixel_offset : float or None
        Step size between coarse translation grid points (pixels).
        Required when adaptive_oversampling > 0.
    mode : str
        Only ``"relion"`` is supported.
    adaptive_pass2_skip_threshold : float
        Skip adaptive pass 2 when the mean significant-sample fraction is at
        least this value. Set to a negative value to disable this shortcut and
        keep the full RELION-style two-pass adaptive search.
    init_healpix_order : int
        Starting HEALPix order for RELION mode (default 2, ~14.7 deg).
    max_healpix_order : int
        Maximum HEALPix order (finest angular sampling, default 7).
    init_translation_range : float
        Initial translation search range in pixels (RELION mode).
    init_translation_step : float
        Initial translation step size in pixels (RELION mode).
    init_translation_sigma_angstrom : float
        Initial RELION-style translation prior width in Angstrom.
    particle_diameter_ang : float or None
        RELION particle diameter in Angstrom for the adaptive coarse-image-size
        formula. When None, fall back to ``ori_size * pixel_size``.

    Returns
    -------
    dict with keys:
        mean : jnp.ndarray -- final merged mean volume
        means : list of 2 jnp.ndarray -- per-half-set means
        fsc : jnp.ndarray -- final FSC curve
        hard_assignments : list of 2 np.ndarray -- per-half-set assignments
        current_sizes : list of int -- current_size at each iteration
        fsc_history : list of jnp.ndarray -- FSC curve at each iteration
        pixel_resolutions : list of float -- pixel resolution at each iter
        wall_times : list of float -- wall time per iteration
        significant_counts : list of (jnp.ndarray or None) -- per-image
            significant sample counts at each iteration (None when
            adaptive_oversampling=0).

    RELION-specific keys:
        convergence_state : RefinementState -- final convergence state
        data_vs_prior_trajectory : list of jnp.ndarray -- per-iteration
            data_vs_prior curves
        healpix_order_trajectory : list of int -- HEALPix order per iter
        ave_Pmax_trajectory : list of float -- average Pmax per iter
    """
    if mode != "relion":
        raise ValueError(f"Unknown mode={mode!r}; expected 'relion'")
    if relion_current_sizes is not None and len(relion_current_sizes) == 0:
        raise ValueError("relion_current_sizes must be non-empty when provided")

    _enable_relion_parity_defaults()
    return _run_relion_iteration_loop(
        experiment_datasets=experiment_datasets,
        init_volume=init_volume,
        init_noise_variance=init_noise_variance,
        init_mean_variance=init_mean_variance,
        rotations=rotations,
        translations=translations,
        disc_type=disc_type,
        max_iter=max_iter,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        init_current_size=init_current_size,
        fsc_threshold=fsc_threshold,
        adaptive_oversampling=adaptive_oversampling,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        relion_current_sizes=relion_current_sizes,
        init_healpix_order=init_healpix_order,
        max_healpix_order=max_healpix_order,
        init_translation_range=init_translation_range,
        init_translation_step=init_translation_step,
        init_translation_sigma_angstrom=init_translation_sigma_angstrom,
        particle_diameter_ang=particle_diameter_ang,
        nside_level=nside_level,
        adaptive_pass2_skip_threshold=adaptive_pass2_skip_threshold,
        save_intermediates_dir=save_intermediates_dir,
        low_resol_join_halves_angstrom=low_resol_join_halves_angstrom,
        tau2_fudge=tau2_fudge,
        perturb_factor=perturb_factor,
        perturb_seed=perturb_seed,
        perturb_replay_relion_dir=perturb_replay_relion_dir,
        init_fsc=init_fsc,
        init_ave_Pmax=init_ave_Pmax,
        init_has_high_fsc_at_limit=init_has_high_fsc_at_limit,
        init_relion_iteration=init_relion_iteration,
        init_image_corrections=init_image_corrections,
        init_scale_corrections=init_scale_corrections,
        init_direction_prior=init_direction_prior,
        init_previous_best_translations=init_previous_best_translations,
        init_previous_best_rotation_eulers=init_previous_best_rotation_eulers,
        replay_iteration_overrides=replay_iteration_overrides,
        skip_final_iteration=skip_final_iteration,
        local_search_profile_mode=local_search_profile_mode,
        local_search_translation_prior_mode=local_search_translation_prior_mode,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        emulate_relion_firstiter_cc=emulate_relion_firstiter_cc,
        relion_firstiter_ini_high_angstrom=relion_firstiter_ini_high_angstrom,
        first_iteration_score_mode=first_iteration_score_mode,
        first_iteration_reconstruction_mode=first_iteration_reconstruction_mode,
        force_max_iter_after_convergence=force_max_iter_after_convergence,
    )


# ---------------------------------------------------------------------------
# RELION-parity refinement mode
# ---------------------------------------------------------------------------


def _run_relion_iteration_loop(
    experiment_datasets,
    init_volume,
    init_noise_variance,
    init_mean_variance,
    rotations,
    translations,
    disc_type,
    max_iter,
    image_batch_size,
    rotation_block_size,
    init_current_size,
    fsc_threshold,
    adaptive_oversampling,
    adaptive_fraction,
    max_significants,
    relion_current_sizes,
    init_healpix_order,
    max_healpix_order,
    init_translation_range,
    init_translation_step,
    init_translation_sigma_angstrom,
    particle_diameter_ang,
    nside_level,
    adaptive_pass2_skip_threshold,
    save_intermediates_dir=None,
    low_resol_join_halves_angstrom=40.0,
    tau2_fudge=1.0,
    perturb_factor=0.0,
    perturb_seed=None,
    perturb_replay_relion_dir=None,
    init_fsc=None,
    init_ave_Pmax=None,
    init_has_high_fsc_at_limit=None,
    init_relion_iteration=0,
    init_image_corrections=None,
    init_scale_corrections=None,
    init_direction_prior=None,
    init_previous_best_translations=None,
    init_previous_best_rotation_eulers=None,
    replay_iteration_overrides=None,
    skip_final_iteration=False,
    local_search_profile_mode="auto",
    local_search_translation_prior_mode="coarse",
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    emulate_relion_firstiter_cc=False,
    relion_firstiter_ini_high_angstrom=None,
    first_iteration_score_mode="gaussian",
    first_iteration_reconstruction_mode="soft",
    force_max_iter_after_convergence=False,
):
    """RELION-parity refinement loop with convergence detection.

    This implements the full RELION auto-refine algorithm:
    1. Convergence-driven iteration (not fixed max_iter)
    2. data_vs_prior for resolution instead of FSC < 0.143
    3. Angular step refinement (HEALPix order increments)
    4. Local angular search when HEALPix order >= 4
    5. Per-image best assignment tracking
    6. Average Pmax computation for adaptive current_size growth

    Corresponds to RELION's autoRefine iteration loop.
    See docs/relion5_auto_refine_algorithm.md.
    """
    from recovar.reconstruction import noise, regularization

    cryo = experiment_datasets[0]
    volume_shape = cryo.volume_shape
    grid_size = cryo.image_shape[0]  # ori_size in RELION terms

    # --- RELION image mask (softMaskOutsideMap on particles) ---
    # RELION masks images to particle_diameter/(2*pixel_size) with a 5-pixel
    # cosine taper before E-step scoring (ml_optimiser.cpp:6288).  The default
    # edge-taper mask (window_mask(D, 0.85, 0.99)) is too tight — it tapers
    # at 54 px vs RELION's 64 px for a 128-px box.
    RELION_WIDTH_MASK_EDGE = 5
    def _image_backend(ds):
        return getattr(getattr(ds, "image_source", None), "backend", None)

    for ds in experiment_datasets:
        backend = _image_backend(ds)
        if backend is not None and hasattr(backend, "image_mask_mode"):
            backend.image_mask_mode = "multiply"
    if particle_diameter_ang is not None and particle_diameter_ang > 0:
        from recovar.core import mask
        from recovar.core.mask import relion_soft_image_mask

        relion_mask = relion_soft_image_mask(
            image_size=grid_size,
            pixel_size=cryo.voxel_size,
            particle_diameter_ang=particle_diameter_ang,
            width_mask_edge_px=RELION_WIDTH_MASK_EDGE,
        )
        for ds in experiment_datasets:
            backend = _image_backend(ds)
            if backend is None:
                continue
            backend.image_mask = relion_mask
            if hasattr(backend, "image_mask_mode"):
                backend.image_mask_mode = "relion_background_fill"
        logger.info(
            "RELION mode: image mask radius=%.1f px (particle_diameter=%.1f A, edge=%d px)",
            particle_diameter_ang / (2.0 * cryo.voxel_size),
            particle_diameter_ang,
            RELION_WIDTH_MASK_EDGE,
        )

    # --- Initialize RefinementState ---
    # Corresponds to RELION's initialiseSamplingVectors + initialLowPassFilterReferences
    state = RefinementState(
        iteration=0,
        healpix_order=init_healpix_order,
        adaptive_oversampling=adaptive_oversampling,
        translation_range=init_translation_range,
        translation_step=init_translation_step,
        max_healpix_order=max_healpix_order,
        current_resolution=float("inf"),
        particle_diameter_angstrom=float(particle_diameter_ang or 0.0),
    )
    # RELION's convergence counters are not initialized against an infinite
    # previous resolution.  They resume from the previous optimiser/model STAR
    # in replay mode, or from the initial FSC/ini_high state in a fresh run.
    if perturb_replay_relion_dir is not None and int(init_relion_iteration) > 0:
        _init_opt_star = os.path.join(
            perturb_replay_relion_dir,
            f"run_it{int(init_relion_iteration):03d}_optimiser.star",
        )
        _init_model_star = os.path.join(
            perturb_replay_relion_dir,
            f"run_it{int(init_relion_iteration):03d}_half1_model.star",
        )
        if os.path.exists(_init_model_star):
            _init_model_meta = read_relion_model_metadata(_init_model_star)
            _init_res_angstrom = float(_init_model_meta["current_resolution"])
            if np.isfinite(_init_res_angstrom) and _init_res_angstrom > 0.0:
                state.current_resolution = _init_res_angstrom
                state.previous_resolution = _init_res_angstrom
        if os.path.exists(_init_opt_star):
            _init_opt_meta = read_relion_optimiser_metadata(_init_opt_star)
            state.nr_iter_wo_resol_gain = int(_init_opt_meta.get("number_iter_without_resolution_gain") or 0)
            _hvc = int(_init_opt_meta.get("number_iter_without_changing_assignments") or 0)
            state.nr_iter_wo_large_hidden_variable_changes = _hvc
            state.nr_iter_wo_assignment_changes = _hvc
            if _init_opt_meta.get("overall_accuracy_rotations") is not None:
                state.acc_rot = float(_init_opt_meta["overall_accuracy_rotations"])
            if _init_opt_meta.get("overall_accuracy_translations_angst") is not None:
                _px = float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0)
                state.acc_trans = float(_init_opt_meta["overall_accuracy_translations_angst"]) / _px
            if _init_opt_meta.get("smallest_changes_orientations") is not None:
                state.smallest_changes_optimal_orientations = float(_init_opt_meta["smallest_changes_orientations"])
            if _init_opt_meta.get("smallest_changes_offsets") is not None:
                state.smallest_changes_optimal_offsets_angstrom = float(_init_opt_meta["smallest_changes_offsets"])
            if _init_opt_meta.get("smallest_changes_classes") is not None:
                state.smallest_changes_optimal_classes = float(_init_opt_meta["smallest_changes_classes"])
            if _init_opt_meta.get("has_converged") is not None:
                state.has_converged = bool(int(_init_opt_meta["has_converged"]))
        logger.info(
            "Replay convergence init from RELION iter %03d: res=%.2f A, "
            "stalls=(res=%d,hvc=%d), smallest=(rot=%.3f deg, trans=%.3f A, class=%.3f)",
            int(init_relion_iteration),
            state.current_resolution,
            state.nr_iter_wo_resol_gain,
            state.nr_iter_wo_large_hidden_variable_changes,
            state.smallest_changes_optimal_orientations,
            state.smallest_changes_optimal_offsets_angstrom,
            state.smallest_changes_optimal_classes,
        )
    elif init_fsc is not None:
        _init_fsc_for_state = np.asarray(init_fsc, dtype=np.float32).copy()
        _prev_cs_for_state = int(init_current_size)
        if _prev_cs_for_state < grid_size:
            _init_fsc_for_state[min(len(_init_fsc_for_state), _prev_cs_for_state // 2) :] = 0.0
        _init_dvp = np.asarray(fsc_to_relion_ssnr(_init_fsc_for_state, tau2_fudge=tau2_fudge))
        _init_res_shell = resolution_from_data_vs_prior(_init_dvp, allow_high_res_recovery=True)
        _init_res_angstrom = shell_index_to_resolution_angstrom(
            _init_res_shell,
            grid_size,
            cryo.voxel_size,
        )
        if np.isfinite(_init_res_angstrom) and _init_res_angstrom > 0.0:
            state.current_resolution = float(_init_res_angstrom)
            state.previous_resolution = float(_init_res_angstrom)
    elif init_relion_iteration == 0 and relion_firstiter_ini_high_angstrom is not None:
        _px = float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0)
        _init_shell = int(np.floor(grid_size * _px / float(relion_firstiter_ini_high_angstrom) + 0.5))
        _init_shell = max(1, min(grid_size // 2, _init_shell))
        _init_res_angstrom = shell_index_to_resolution_angstrom(_init_shell, grid_size, _px)
        state.current_resolution = float(_init_res_angstrom)
        state.previous_resolution = float(_init_res_angstrom)

    # RELION mode owns the coarse HEALPix grid. When coarse-grid metadata is
    # provided, regenerate the matching coarse grid here instead of inheriting
    # any finer caller-supplied rotation table.
    current_healpix_order = int(init_healpix_order)
    if nside_level is not None:
        if int(nside_level) != current_healpix_order:
            logger.info(
                "RELION mode: ignoring caller nside_level=%d and regenerating initial coarse grid at healpix_order=%d",
                int(nside_level),
                current_healpix_order,
            )
        current_rotations, current_rotation_eulers = _relion_rotation_grid_float32(current_healpix_order)
        current_nside_level = current_healpix_order
    elif rotations is not None:
        current_rotations = np.asarray(rotations, dtype=np.float32)
        current_rotation_eulers = utils.R_to_relion(np.asarray(current_rotations), degrees=True).astype(np.float32)
        current_nside_level = current_healpix_order
    else:
        current_rotations, current_rotation_eulers = _relion_rotation_grid_float32(current_healpix_order)
        current_nside_level = current_healpix_order
    if translations is None:
        current_translations = jnp.asarray(
            get_translation_grid(init_translation_range, init_translation_step), dtype=jnp.float32
        )
    else:
        current_translations = jnp.asarray(translations, dtype=jnp.float32)
    # Unperturbed base grid — `current_translations` may be replaced per-iter by
    # a perturbed copy (SamplingPerturbation). Keep the base so each iter
    # perturbs a fresh copy rather than compounding prior perturbations.
    base_translations = current_translations
    if save_intermediates_dir is not None:
        os.makedirs(save_intermediates_dir, exist_ok=True)
    if local_search_profile_mode not in {"auto", "on", "off"}:
        raise ValueError(
            f"local_search_profile_mode must be one of {{'auto', 'on', 'off'}}, got {local_search_profile_mode!r}",
        )
    collect_local_search_profile = (
        save_intermediates_dir is not None if local_search_profile_mode == "auto" else local_search_profile_mode == "on"
    )

    # RELION uses pf=2 for both projection and reconstruction (--pad 2).
    # Projection: real-space zero-pad N³→(2N)³, DFT, then trilinear slice.
    # Reconstruction: backproject into (2N)³ Fourier grid, Wiener solve,
    # iDFT at (2N)³, crop real-space to N³.
    PADDING_FACTOR = 2
    PROJECTION_PADDING_FACTOR = 2
    padded_volume_shape = tuple(d * PADDING_FACTOR for d in volume_shape)

    def _safe_batch_sizes(n_rot, n_trans):
        """Reduce batch sizes for large pose grids to avoid GPU OOM.

        2026-04-08: bumped budget from 50M to 200M floats. This is the
        score-tensor size budget; the M-step GEMMs and CTF accumulators
        allocate ~10x this much in working memory, so 200M maps to ~2 GB
        peak. Verified to fit on 80 GB A100s for both 64-px tiny (1k
        particles) and 128-px 5k benchmarks. Larger budgets give faster
        per-iter times on tiny but OOM on 128-px boxes.
        """
        # Target the actual score-tensor size: n_img * n_rot_block * n_trans.
        n_trans = max(int(n_trans), 1)
        ibs = min(
            image_batch_size,
            max(1, RELION_SCORE_TENSOR_FLOAT_BUDGET // max(n_rot * n_trans, 1)),
        )
        rbs = min(
            rotation_block_size,
            max(64, RELION_SCORE_TENSOR_FLOAT_BUDGET // max(ibs * n_trans, 1)),
        )
        return ibs, rbs

    # State: two half-set volumes, noise, prior.
    # init_volume can be a single array (used for both halves) or a list/tuple
    # of 2 arrays (one per half-set, matching RELION auto-refine).
    if isinstance(init_volume, (list, tuple)) and len(init_volume) == 2:
        means = [jnp.array(init_volume[0]), jnp.array(init_volume[1])]
    else:
        means = [jnp.array(init_volume), jnp.array(init_volume)]
    noise_variance_per_half = _normalize_noise_variance_per_half(
        init_noise_variance,
        n_halves=2,
    )
    noise_variance = _mean_noise_variance(noise_variance_per_half)
    mean_variance = jnp.array(init_mean_variance)

    # History tracking. Keep these plain lists for now because they are
    # serialized directly into legacy intermediate files.
    current_sizes = []
    fsc_history = []
    pixel_resolutions = []
    wall_times = []
    hard_assignments = [None, None]
    previous_assignments = [None, None]
    previous_best_rotations = [None, None]
    relion_half_inputs = _RelionHalfInputState.from_initial_values(
        previous_best_translations=init_previous_best_translations,
        previous_best_rotation_eulers=init_previous_best_rotation_eulers,
        image_corrections=init_image_corrections,
        scale_corrections=init_scale_corrections,
    )
    max_posterior_per_half = [None, None]
    rotation_posterior_per_half = [None, None]
    significant_counts = []
    data_vs_prior_trajectory = []
    healpix_order_trajectory = []
    ave_Pmax_trajectory = []
    pmax_per_image_history = []
    # Per-iter per-shell trajectories for RELION parity diff (added for the
    # 2026-04 audit). noise_radial_trajectory[i] = sigma2_noise per shell after
    # iter i's noise update; tau2_radial_trajectory[i] = recovar's tau2 prior
    # per shell after iter i's signal-prior update.
    noise_radial_trajectory = []
    noise_radial_per_half_trajectory = []
    tau2_radial_trajectory = []
    tau2_sigma2_trajectory = []
    tau2_avg_weight_trajectory = []
    tau2_shell_sum_trajectory = []
    tau2_shell_count_trajectory = []
    tau2_fsc_used_trajectory = []
    tau2_ssnr_trajectory = []
    tau2_update_details = None
    tau2_update_details_per_half = None

    # C1 (RELION-parity): per-iter sigma2_offset update from data. Initialized
    # from `init_translation_sigma_angstrom`; updated from RELION's
    # posterior-weighted offset moment when the E-step path propagates it.
    # RELION stores and updates this quantity in Angstrom², and its default
    # lower bound is min_sigma2_offset=2 Å² (ml_optimiser.cpp).
    current_sigma_offset_angstrom = float(init_translation_sigma_angstrom)
    sigma_offset_used_trajectory = []
    sigma_offset_trajectory = []
    frac_changed_trajectory = []
    acc_rot_trajectory = []
    smallest_change_angles_trajectory = []
    smallest_change_offsets_trajectory = []
    best_rotation_eulers_history = []
    best_translations_history = []
    local_profile_history = []
    relion_incr_size = 10  # RELION default
    relion_has_high_fsc_at_limit = bool(init_has_high_fsc_at_limit) if init_has_high_fsc_at_limit is not None else False
    global_direction_prior_per_half = [None, None]
    global_direction_prior_order_per_half = [None, None]

    # --- Direction prior from snapshot ---
    # When starting from a RELION snapshot, the previous iteration's
    # pdf_orientation is a non-uniform prior over HEALPix directions.
    # RELION applies this in the next E-step.  recovar must do the same.
    if init_direction_prior is not None:
        global_direction_prior_per_half = normalize_direction_prior_per_half(init_direction_prior)
        for k in range(2):
            if global_direction_prior_per_half[k] is None:
                continue
            prior_k = np.asarray(global_direction_prior_per_half[k], dtype=np.float32)
            global_direction_prior_per_half[k] = prior_k
            global_direction_prior_order_per_half[k] = infer_direction_prior_healpix_order(prior_k)
            logger.info(
                "RELION mode: loaded init direction prior half-%d: %d directions, range=[%.6f, %.6f], %d zero-probability",
                k + 1,
                len(prior_k),
                prior_k.min(),
                prior_k.max(),
                int(np.sum(prior_k == 0)),
            )

    # Extract per-shell radial profiles from the input pixel-array noise
    # variances for diagnostic logging ("noise update per shell: old=... new=...").
    previous_noise_radial_per_half = [
        _radial_profile_from_noise_variance(noise_k, cryo.image_shape)
        for noise_k in noise_variance_per_half
    ]
    previous_noise_radial = jnp.asarray(
        np.mean(np.stack(previous_noise_radial_per_half, axis=0), axis=0),
        dtype=jnp.float32,
    )

    # --- RELION SamplingPerturbation state (healpix_sampling.cpp:167-174) ---
    # RELION applies a random rigid rotation of the entire SO(3) trial grid at
    # each iteration: A -> A @ R_perturb with R_perturb = R_from_relion([m,m,m])
    # and m = random_perturbation * angular_sampling. The random_perturbation
    # is advanced per iter via realWRAP(prev + rnd_unif(0.5*pf, pf), -pf, +pf).
    # For exact parity replay, read _rlnSamplingPerturbInstance from RELION's
    # per-iter sampling.star.
    random_perturbation = 0.0
    perturb_rng = np.random.default_rng(perturb_seed) if perturb_seed is not None else np.random.default_rng()
    iteration = 0
    while (force_max_iter_after_convergence or not state.has_converged) and iteration < max_iter:
        t0 = time.time()
        _parity_dump.start_iteration(iteration)
        iter_replay_override = None
        if replay_iteration_overrides is not None and iteration < len(replay_iteration_overrides):
            iter_replay_override = replay_iteration_overrides[iteration]
        relion_firstiter_cc_this_iter = bool(
            emulate_relion_firstiter_cc and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_normalized_cc_this_iter = bool(
            first_iteration_score_mode == "normalized_cc" and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_hard_reconstruction_this_iter = bool(
            first_iteration_reconstruction_mode == "hard" and init_relion_iteration == 0 and iteration == 0
        )

        # --- Determine current_size using RELION's FSC-derived SSNR (C4/C5) ---
        # At iteration 0, no previous half-map FSC exists yet; use the initial
        # resolution plus RELION's bootstrap image-size growth. After that,
        # mimic RELION's auto-refine update:
        # 1. zero FSC beyond the previous current_size limit
        # 2. convert FSC -> SSNR (= data_vs_prior in split-half auto-refine)
        # 3. grow current_size using ave_Pmax, FSC at the current limit, and
        #    RELION's dynamic incr_size heuristic.
        if iteration == 0:
            if init_relion_iteration == 0:
                seeded_cs = bootstrap_current_size_from_ini_high_relion(
                    grid_size,
                    float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
                    relion_firstiter_ini_high_angstrom,
                    incr_size=relion_incr_size,
                )
            else:
                seeded_cs = None
            if seeded_cs is not None:
                cs = int(seeded_cs)
                data_vs_prior_iter = None
                logger.info(
                    "RELION init bootstrap: seeding iter-1 current_size from ini_high=%.2f A -> %d",
                    float(relion_firstiter_ini_high_angstrom),
                    cs,
                )
            elif init_fsc is not None:
                fsc_prev = np.asarray(init_fsc, dtype=np.float32).copy()
                prev_cs = int(init_current_size)
                if prev_cs < grid_size:
                    fsc_prev[min(len(fsc_prev), prev_cs // 2) :] = 0.0
                data_vs_prior_iter = np.asarray(
                    fsc_to_relion_ssnr(fsc_prev, tau2_fudge=tau2_fudge),
                )
                data_vs_prior_trajectory.append(data_vs_prior_iter)
                res_shell = resolution_from_data_vs_prior(
                    data_vs_prior_iter,
                    allow_high_res_recovery=True,
                )
                relion_incr_size, relion_has_high_fsc_at_limit = update_relion_growth_state_from_fsc(
                    fsc_prev,
                    prev_cs,
                    incr_size=relion_incr_size,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                )
                _init_pmax = float(init_ave_Pmax) if init_ave_Pmax is not None else 0.0
                raw_cs = compute_current_size_relion(
                    res_shell,
                    grid_size,
                    ave_Pmax=_init_pmax,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                    incr_size=relion_incr_size,
                )
                cs = quantize_current_size(raw_cs, ori_size=grid_size)
            else:
                cs = _bootstrap_current_size_relion(init_current_size, grid_size)
                data_vs_prior_iter = None
        else:
            fsc_prev = np.asarray(fsc_history[-1], dtype=np.float32).copy()
            prev_cs = current_sizes[-1]
            if prev_cs < grid_size:
                fsc_prev[min(len(fsc_prev), prev_cs // 2) :] = 0.0

            # data_vs_prior = tau2_fudge * fsc / (1 - fsc), matching
            # RELION's updateSSNRarrays at backprojector.cpp:1117-1123
            # for the gold-standard split-half auto-refine path.
            data_vs_prior_iter = np.asarray(
                fsc_to_relion_ssnr(fsc_prev, tau2_fudge=tau2_fudge),
            )
            data_vs_prior_trajectory.append(data_vs_prior_iter)
            res_shell = resolution_from_data_vs_prior(
                data_vs_prior_iter,
                allow_high_res_recovery=True,
            )
            relion_incr_size, relion_has_high_fsc_at_limit = update_relion_growth_state_from_fsc(
                fsc_prev,
                prev_cs,
                incr_size=relion_incr_size,
                has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
            )

            raw_cs = compute_current_size_relion(
                res_shell,
                grid_size,
                ave_Pmax=state.ave_Pmax,
                has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                incr_size=relion_incr_size,
            )
            cs = quantize_current_size(raw_cs, ori_size=grid_size)

        cs = quantize_current_size(cs, ori_size=grid_size)
        if relion_current_sizes is not None:
            if iteration < len(relion_current_sizes):
                oracle_cs = int(relion_current_sizes[iteration])
            else:
                oracle_cs = int(relion_current_sizes[-1])
            if oracle_cs <= 0:
                oracle_cs = int(init_current_size)
            cs = quantize_current_size(oracle_cs, ori_size=grid_size)
            logger.info(
                "Current-size oracle: iteration %d using current_size=%d",
                iteration + 1,
                cs,
            )

        # --- Replay override: force recovar's sampling state to mirror RELION ---
        # When replaying, RELION's per-iter sampling.star dictates the actual
        # hp_order, offset_range, and offset_step used at this iteration.
        # Overriding `state.healpix_order`, `state.translation_range` and
        # `state.translation_step` here makes the downstream grid regen code
        # produce the same grid RELION did, so the perturbation applied later
        # is on the correct base grid.
        _replay_meta = None
        _replay_prior_translations = None
        if perturb_replay_relion_dir is not None:
            _star = os.path.join(
                perturb_replay_relion_dir,
                f"run_it{init_relion_iteration + iteration + 1:03d}_sampling.star",
            )
            _replay_meta = read_relion_sampling_metadata(_star)
            _relion_hp = int(_replay_meta["healpix_order"])
            _relion_psi_step_deg = float(_replay_meta.get("psi_step", healpix_angular_step(_relion_hp)))
            # RELION stores offset_{range,step} in Angstroms; convert to px.
            _px = float(cryo.voxel_size) if cryo.voxel_size > 0 else 1.0
            _relion_offset_range = float(_replay_meta["offset_range"]) / _px
            _relion_offset_step = float(_replay_meta["offset_step"]) / _px
            _replay_prior_translations = jnp.array(
                get_translation_grid(
                    _relion_offset_range,
                    _relion_offset_step,
                ).astype(np.float32)
            )
            _capped_hp = min(_relion_hp, state.max_healpix_order)
            if state.healpix_order != _capped_hp:
                if _capped_hp < _relion_hp:
                    logger.info(
                        "Replay override: healpix_order %d -> %d (RELION %d capped by max_healpix_order=%d, from %s)",
                        state.healpix_order,
                        _capped_hp,
                        _relion_hp,
                        state.max_healpix_order,
                        _star,
                    )
                else:
                    logger.info(
                        "Replay override: healpix_order %d -> %d (from %s)",
                        state.healpix_order,
                        _capped_hp,
                        _star,
                    )
                state.healpix_order = _capped_hp
            _replay_do_local = bool(state.healpix_order >= LOCAL_SEARCH_HEALPIX_ORDER)
            if state.do_local_search != _replay_do_local:
                logger.info(
                    "Replay override: local_search %s -> %s (healpix_order=%d)",
                    state.do_local_search,
                    _replay_do_local,
                    state.healpix_order,
                )
                state.do_local_search = _replay_do_local
                if _replay_do_local:
                    state.sigma_rot = 0.0
                    state.sigma_psi = 0.0
            if _replay_do_local:
                _relion_sigma_rad = np.deg2rad(2.0 * _relion_psi_step_deg)
                if (
                    abs(float(state.sigma_rot) - _relion_sigma_rad) > 1e-8
                    or abs(float(state.sigma_psi) - _relion_sigma_rad) > 1e-8
                ):
                    logger.info(
                        "Replay override: local prior sigma %.3f/%.3f deg -> %.3f deg "
                        "(2 * RELION psi_step %.3f deg)",
                        float(np.rad2deg(state.sigma_rot)),
                        float(np.rad2deg(state.sigma_psi)),
                        float(np.rad2deg(_relion_sigma_rad)),
                        _relion_psi_step_deg,
                    )
                state.sigma_rot = _relion_sigma_rad
                state.sigma_psi = _relion_sigma_rad
            if (
                abs(float(state.translation_range) - _relion_offset_range) > 1e-6
                or abs(float(state.translation_step) - _relion_offset_step) > 1e-6
            ):
                logger.info(
                    "Replay override: translation_range %.3f -> %.3f px, step %.3f -> %.3f px",
                    float(state.translation_range),
                    _relion_offset_range,
                    float(state.translation_step),
                    _relion_offset_step,
                )
                state.translation_range = _relion_offset_range
                state.translation_step = _relion_offset_step

            # Override current_size from the RELION model star that records the
            # control state for the replayed E-step. Empirically, replaying
            # RELION iter N+1 against the saved benchmark trajectory requires
            # reading run_it{N+1}_model.star, not run_it{N}_model.star:
            # the saved model star already carries the control variables
            # (current_size, sigma_offset) used by that E-step.
            _cs_iter = _replay_control_model_iteration(init_relion_iteration, iteration)
            _model_star = os.path.join(
                perturb_replay_relion_dir,
                f"run_it{_cs_iter:03d}_half1_model.star",
            )
            if os.path.exists(_model_star):
                _model_meta = read_relion_model_metadata(_model_star)
                _relion_cs = int(_model_meta["current_image_size"])
                if _relion_cs <= 0:
                    logger.info(
                        "Replay override: ignoring non-positive current_size=%d from %s",
                        _relion_cs,
                        _model_star,
                    )
                elif cs != _relion_cs:
                    logger.info(
                        "Replay override: current_size %d -> %d (from %s)",
                        cs,
                        _relion_cs,
                        _model_star,
                    )
                    cs = _relion_cs

            if iteration > 0:
                _prior_iter = init_relion_iteration + iteration
                if iter_replay_override is None or iter_replay_override.get("direction_prior") is None:
                    for _half_idx in range(2):
                        _prior_star = os.path.join(
                            perturb_replay_relion_dir,
                            f"run_it{_prior_iter:03d}_half{_half_idx + 1}_model.star",
                        )
                        if not os.path.exists(_prior_star):
                            continue
                        _relion_direction_prior = read_relion_direction_prior(_prior_star)
                        _relion_direction_prior_order = infer_direction_prior_healpix_order(_relion_direction_prior)
                        if _relion_direction_prior_order != state.healpix_order:
                            logger.info(
                                "Replay override: remapping half-%d direction prior from healpix_order=%d to %d",
                                _half_idx + 1,
                                _relion_direction_prior_order,
                                state.healpix_order,
                            )
                            _relion_direction_prior = remap_direction_prior_to_healpix_order(
                                _relion_direction_prior,
                                _relion_direction_prior_order,
                                state.healpix_order,
                            )
                            _relion_direction_prior_order = state.healpix_order
                        global_direction_prior_per_half[_half_idx] = _relion_direction_prior
                        global_direction_prior_order_per_half[_half_idx] = _relion_direction_prior_order
                        logger.info(
                            "Replay override: direction prior half-%d <- %s (%d directions, range=[%.6f, %.6f], zeros=%d)",
                            _half_idx + 1,
                            _prior_star,
                            len(_relion_direction_prior),
                            float(_relion_direction_prior.min()),
                            float(_relion_direction_prior.max()),
                            int(np.sum(_relion_direction_prior == 0)),
                        )

        if iter_replay_override is not None:
            _replay_sigma = iter_replay_override.get("translation_sigma_angstrom")
            if _replay_sigma is not None:
                current_sigma_offset_angstrom = float(_replay_sigma)
                logger.info(
                    "Replay override: sigma_offset <- %.4f A (iter=%d)",
                    current_sigma_offset_angstrom,
                    iteration + 1,
                )
            _replay_prev_trans = iter_replay_override.get("previous_best_translations")
            if _replay_prev_trans is not None:
                relion_half_inputs.previous_best_translations = _optional_float32_half_pair(_replay_prev_trans)
                logger.info(
                    "Replay override: previous_best_translations <- half1=%s half2=%s",
                    "set" if relion_half_inputs.previous_best_translations[0] is not None else "none",
                    "set" if relion_half_inputs.previous_best_translations[1] is not None else "none",
                )
            _replay_prev_rots = iter_replay_override.get("previous_best_rotations")
            if _replay_prev_rots is not None:
                previous_best_rotations = _optional_float32_half_pair(_replay_prev_rots)
                logger.info(
                    "Replay override: previous_best_rotations <- half1=%s half2=%s",
                    "set" if previous_best_rotations[0] is not None else "none",
                    "set" if previous_best_rotations[1] is not None else "none",
                )
            _replay_prev_eulers = iter_replay_override.get("previous_best_rotation_eulers")
            if _replay_prev_eulers is not None:
                relion_half_inputs.previous_best_rotation_eulers = _optional_float32_half_pair(_replay_prev_eulers)
                logger.info(
                    "Replay override: previous_best_rotation_eulers <- half1=%s half2=%s",
                    "set" if relion_half_inputs.previous_best_rotation_eulers[0] is not None else "none",
                    "set" if relion_half_inputs.previous_best_rotation_eulers[1] is not None else "none",
                )
            _replay_img_corr = iter_replay_override.get("image_corrections")
            if _replay_img_corr is not None:
                relion_half_inputs.image_corrections = _optional_float32_half_pair(_replay_img_corr)
                logger.info(
                    "Replay override: image_corrections <- half1=%s half2=%s",
                    "set" if relion_half_inputs.image_corrections[0] is not None else "none",
                    "set" if relion_half_inputs.image_corrections[1] is not None else "none",
                )
            _replay_scale_corr = iter_replay_override.get("scale_corrections")
            if _replay_scale_corr is not None:
                relion_half_inputs.scale_corrections = _optional_float32_half_pair(_replay_scale_corr)
                logger.info(
                    "Replay override: scale_corrections <- half1=%s half2=%s",
                    "set" if relion_half_inputs.scale_corrections[0] is not None else "none",
                    "set" if relion_half_inputs.scale_corrections[1] is not None else "none",
                )
            _replay_noise = iter_replay_override.get("noise_variance")
            if _replay_noise is not None:
                noise_variance_per_half = _normalize_noise_variance_per_half(_replay_noise, n_halves=2)
                noise_variance = _mean_noise_variance(noise_variance_per_half)
                previous_noise_radial_per_half = [
                    _radial_profile_from_noise_variance(noise_k, cryo.image_shape)
                    for noise_k in noise_variance_per_half
                ]
                previous_noise_radial = jnp.asarray(
                    np.mean(np.stack(previous_noise_radial_per_half, axis=0), axis=0),
                    dtype=jnp.float32,
                )
                logger.info("Replay override: sigma2_noise <- per-half model.star arrays")
            _replay_dir_prior = iter_replay_override.get("direction_prior")
            if _replay_dir_prior is not None:
                replay_priors = normalize_direction_prior_per_half(_replay_dir_prior)
                for _half_idx in range(2):
                    if replay_priors[_half_idx] is None:
                        continue
                    prior_k = np.asarray(replay_priors[_half_idx], dtype=np.float32)
                    prior_order_k = infer_direction_prior_healpix_order(prior_k)
                    if prior_order_k != state.healpix_order:
                        logger.info(
                            "Replay override: remapping provided half-%d direction prior from healpix_order=%d to %d",
                            _half_idx + 1,
                            prior_order_k,
                            state.healpix_order,
                        )
                        prior_k = remap_direction_prior_to_healpix_order(
                            prior_k,
                            prior_order_k,
                            state.healpix_order,
                        )
                        prior_order_k = state.healpix_order
                    global_direction_prior_per_half[_half_idx] = prior_k
                    global_direction_prior_order_per_half[_half_idx] = prior_order_k
                    logger.info(
                        "Replay override: direction prior half-%d <- provided override (%d directions, range=[%.6f, %.6f], zeros=%d)",
                        _half_idx + 1,
                        len(prior_k),
                        float(prior_k.min()),
                        float(prior_k.max()),
                        int(np.sum(prior_k == 0)),
                    )

        sigma_offset_used_trajectory.append(float(current_sigma_offset_angstrom))
        current_sizes.append(cs)
        healpix_order_trajectory.append(state.healpix_order)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, healpix_order=%d, local_search=%s ===",
            iteration + 1,
            max_iter,
            cs,
            state.healpix_order,
            state.do_local_search,
        )

        # --- Angular step refinement: regenerate rotation grid if needed ---
        # When update_refinement_state incremented healpix_order, we need
        # a new rotation grid at the finer level.
        # IMPORTANT: At order >= 5, the full grid has 2.4M+ rotations which
        # OOMs the GPU.  Instead, keep the order-4 grid as the "base" and
        # rely on local search + oversampling to achieve finer angular steps.
        # The order is still tracked for sigma calculation.
        if state.healpix_order != current_healpix_order:
            new_order = min(state.healpix_order, RELION_MAX_FULL_GRID_ORDER)
            if new_order != current_healpix_order:
                logger.info(
                    "Regenerating rotation grid: order %d -> %d",
                    current_healpix_order,
                    new_order,
                )
                current_rotations, current_rotation_eulers = _relion_rotation_grid_float32(new_order)
                current_healpix_order = new_order
                global_direction_prior_per_half = [None, None]
                global_direction_prior_order_per_half = [None, None]
            else:
                logger.info(
                    "Angular step refined to order %d (grid stays at order %d — local search handles finer sampling)",
                    state.healpix_order,
                    current_healpix_order,
                )
            current_nside_level = current_healpix_order

            # Regenerate translation grid based on updated parameters
            current_translations = jnp.array(
                get_translation_grid(
                    state.translation_range,
                    state.translation_step,
                ).astype(np.float32)
            )
            base_translations = current_translations
            logger.info(
                "New grid: %d rotations, %d translations (range=%.1f, step=%.1f)",
                current_rotations.shape[0],
                current_translations.shape[0],
                state.translation_range,
                state.translation_step,
            )
        elif _replay_meta is not None:
            # Translation params may have changed under replay without an
            # hp_order bump. Regenerate the translation grid to match RELION.
            _new_t = jnp.array(
                get_translation_grid(
                    state.translation_range,
                    state.translation_step,
                ).astype(np.float32)
            )
            if _new_t.shape != base_translations.shape or not jnp.allclose(_new_t, base_translations):
                current_translations = _new_t
                base_translations = _new_t
                logger.info(
                    "Replay: regenerated translation grid: %d translations (range=%.2f px, step=%.2f px)",
                    current_translations.shape[0],
                    state.translation_range,
                    state.translation_step,
                )

        # --- Local angular search bookkeeping ---
        # Once RELION enters local search, each image should search around its
        # own previous orientation on the true current HEALPix order. Use the
        # exact rotations selected in the previous iteration, not the nearest
        # snapped grid indices.
        effective_rotations = current_rotations
        effective_rotation_eulers = np.asarray(current_rotation_eulers, dtype=np.float32)
        rotation_log_prior_per_half = [None, None]
        use_local = state.do_local_search and all(
            eulers is not None for eulers in relion_half_inputs.previous_best_rotation_eulers
        )
        # --- Apply RELION SamplingPerturbation to the trial grid for this iter ---
        # healpix_sampling.cpp:1909-1934 (rotations) + 1810-1820 (translations)
        # Perturbation is a rigid rotation of SO(3): A := A @ R_perturb applied
        # AFTER oversampling. At adaptive_oversampling=0 (os0 RELION runs),
        # the coarse grid IS the trial grid so we apply directly here.
        if _replay_meta is not None:
            random_perturbation = float(_replay_meta["random_perturbation"])
            logger.info(
                "Perturbation replay: iter=%d rp=%+.5f pf=%.3f relion_hp_order=%d",
                iteration + 1,
                random_perturbation,
                float(_replay_meta["perturbation_factor"]),
                int(_replay_meta["healpix_order"]),
            )
        elif perturb_factor > 0:
            random_perturbation = advance_relion_perturbation(random_perturbation, perturb_factor, perturb_rng)
            logger.info("Perturbation advance: iter=%d rp=%+.5f", iteration + 1, random_perturbation)
        if _replay_meta is not None or perturb_factor > 0:
            # Use RELION's actual hp_order when replaying (recovar's current
            # grid order may be capped at MAX_FULL_GRID_ORDER=4 for memory).
            _angsamp_order = int(_replay_meta["healpix_order"]) if _replay_meta is not None else current_healpix_order
            angsamp_deg = relion_angular_sampling_deg(_angsamp_order, adaptive_oversampling=0)
            if effective_rotation_eulers is not None:
                effective_rotations, effective_rotation_eulers = apply_relion_rotation_perturbation_to_eulers(
                    effective_rotation_eulers,
                    random_perturbation,
                    angsamp_deg,
                )
            else:
                effective_rotations = apply_relion_rotation_perturbation(
                    np.asarray(effective_rotations),
                    random_perturbation,
                    angsamp_deg,
                ).astype(np.float32)
                effective_rotation_eulers = utils.R_to_relion(np.asarray(effective_rotations), degrees=True).astype(
                    np.float32
                )
            _perturbed_translations = apply_relion_translation_perturbation(
                np.asarray(base_translations),
                random_perturbation,
                float(state.translation_step),
            )
            current_translations = jnp.asarray(_perturbed_translations, dtype=jnp.float32)
        if relion_firstiter_cc_this_iter and current_translations.shape[0] > 1:
            center_idx = int(current_translations.shape[0] // 2)
            current_translations = current_translations[center_idx : center_idx + 1]
            logger.info(
                "RELION iter-1 CC emulation: restricting translation grid to the perturbed center shift %s",
                np.asarray(current_translations[0], dtype=np.float32),
            )
        local_search_order = None
        local_search_rotations = None
        local_search_rotation_eulers = None
        sigma_rot = state.sigma_rot
        sigma_psi = state.sigma_psi if state.sigma_psi > 0 else sigma_rot
        if use_local and sigma_rot <= 0:
            step_rad = np.deg2rad(healpix_angular_step(state.healpix_order) / (2**state.adaptive_oversampling))
            sigma_rot = np.sqrt(2.0 * 2.0) * step_rad
            sigma_psi = sigma_rot

        if use_local:
            local_search_order = state.healpix_order + state.adaptive_oversampling
            local_search_random_perturbation = 0.0
            local_search_angular_sampling_deg = None
            if effective_rotations.shape[0] != rotation_grid_size(local_search_order):
                logger.info(
                    "Using selected-only fine local-search grid: order=%d (%d rotations) from capped base order=%d",
                    local_search_order,
                    rotation_grid_size(local_search_order),
                    current_healpix_order,
                )
                local_search_angular_sampling_deg = relion_angular_sampling_deg(
                    local_search_order,
                    adaptive_oversampling=0,
                )
                if _precompute_exact_local_fine_grid_enabled(local_search_order):
                    _, local_search_rotation_eulers = _relion_rotation_grid_float32(local_search_order)
                    local_search_rotations, local_search_rotation_eulers = apply_relion_rotation_perturbation_to_eulers(
                        local_search_rotation_eulers,
                        float(random_perturbation),
                        local_search_angular_sampling_deg,
                    )
                    local_search_random_perturbation = 0.0
                else:
                    local_search_rotations = None
                    local_search_rotation_eulers = None
                    local_search_random_perturbation = float(random_perturbation)
            else:
                local_search_rotations = effective_rotations
                local_search_rotation_eulers = None
            logger.info(
                "Local search (batched exact): fine_order=%d, sigma_rot=%.4f rad (%.2f deg), sigma_psi=%.4f rad",
                local_search_order,
                sigma_rot,
                np.rad2deg(sigma_rot),
                sigma_psi,
            )
        else:
            for _half_idx in range(2):
                prior_k = global_direction_prior_per_half[_half_idx]
                prior_order_k = global_direction_prior_order_per_half[_half_idx]
                if prior_k is None or prior_order_k != current_healpix_order:
                    continue
                rotation_log_prior_per_half[_half_idx] = make_relion_direction_log_prior(
                    prior_k,
                    current_healpix_order,
                )
                logger.info(
                    "Using learned global direction prior half-%d: %d directions at healpix_order=%d",
                    _half_idx + 1,
                    prior_k.shape[0],
                    current_healpix_order,
                )

        cs_for_engine = cs if cs < cryo.image_shape[0] else None

        # --- Run E+M on each half-set ---
        # Two modes: single-pass (adaptive_oversampling=0) or two-pass
        # coarse/fine (adaptive_oversampling>=1).
        iter_sig_counts = None
        use_adaptive = state.adaptive_oversampling > 0 and not use_local and effective_rotations.shape[0] > 16
        is_initial_global_iteration = init_relion_iteration == 0 and iteration == 0 and not use_local
        use_global_significant_support = (
            state.adaptive_oversampling == 0
            and not use_local
            and effective_rotations.shape[0] > 16
            and adaptive_fraction < 1.0
            and not is_initial_global_iteration
            and not (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter)
            and not first_iter_hard_reconstruction_this_iter
        )

        # Track the rotation grids used for pose extraction.
        # When adaptive oversampling is active, ha_k indices refer to the
        # oversampled grid (from pass 2), not effective_rotations.
        pose_rotations = [None, None]  # rotations to use with ha for poses
        pose_rotation_eulers = [None, None]
        pose_translations = [
            np.asarray(current_translations, dtype=np.float32),
            np.asarray(current_translations, dtype=np.float32),
        ]
        best_pose_rotations = [None, None]
        best_pose_rotation_eulers = [None, None]
        best_pose_translations = [None, None]
        translation_search_bases = [None, None]
        # Coarse-grid assignments for local search tracking (always indexed
        # into effective_rotations, even when adaptive oversampling is used).
        coarse_ha = [None, None]
        adaptive_pass1_diag = [None, None]

        if use_adaptive:
            # --- TWO-PASS ADAPTIVE OVERSAMPLING (RELION parity) ---
            # Pass 1: coarse E-step at reduced resolution to find
            #         significant orientations.
            # Pass 2: oversampled E+M at full current_size for significant
            #         orientations only.

            # Compute coarse image size from angular step
            effective_step_deg = healpix_angular_step(current_healpix_order)
            pixel_size = cryo.voxel_size if cryo.voxel_size > 0 else 1.0
            coarse_size = compute_coarse_image_size(
                effective_step_deg,
                pixel_size,
                grid_size,
                particle_diameter=particle_diameter_ang,
            )
            coarse_size = clamp_relion_coarse_image_size(
                coarse_size,
                cs if cs_for_engine is not None else None,
                grid_size,
            )
            coarse_cs = coarse_size if coarse_size < grid_size else None

            logger.info(
                "Adaptive oversampling: pass 1 at coarse_size=%s, "
                "pass 2 at current_size=%s (oversampling=%d, particle_diameter=%s)",
                coarse_cs,
                cs_for_engine,
                state.adaptive_oversampling,
                (f"{float(particle_diameter_ang):.1f} A" if particle_diameter_ang is not None else "box_size"),
            )

        noise_stats_per_half = [None, None]

        for k in range(2):
            noise_variance_k = noise_variance_per_half[k]
            rotation_log_prior_k = rotation_log_prior_per_half[k]
            previous_translations_k = relion_half_inputs.previous_best_translations[k]
            translation_search_base = relion_translation_search_base(previous_translations_k)
            translation_search_bases[k] = translation_search_base
            current_translation_range = float(state.translation_range)
            # RELION translation prior sigma (ml_optimiser.cpp:7737-7746):
            # RELION checks `offset_range_x` (rlnOffsetRangeX in optimiser.star),
            # NOT the search-grid `offset_range` (rlnOffsetRange in sampling.star).
            # When offset_range_x > 0: sigma² = range_x²/9 (per-axis override)
            # When offset_range_x <= 0: sigma² = model.sigma2_offset (learned)
            # For this dataset, rlnOffsetRangeX = -1 → model sigma is used.
            # We always use current_sigma_offset_angstrom (from model star).
            #
            # Evaluate the translation prior on RELION's unperturbed coarse
            # sampling grid. SamplingPerturbation changes the shifts used for
            # projection, but RELION builds pdf_offset from
            # sampling.translations_x/y before perturbation. RELION stores
            # that grid in Angstrom while projection shifts use
            # getTranslationsInPixel; convert the rounded old-offset center
            # into the pixel-space search grid used below.
            trans_prior_center = relion_translation_prior_center(
                previous_translations_k,
                cryo.voxel_size,
            )
            translation_prior_translations = np.asarray(base_translations, dtype=np.float32)
            if current_translations.shape[0] != base_translations.shape[0]:
                if current_translations.shape[0] == 1 and base_translations.shape[0] > 1:
                    center_idx = int(base_translations.shape[0] // 2)
                    translation_prior_translations = np.asarray(
                        base_translations[center_idx : center_idx + 1],
                        dtype=np.float32,
                    )
                else:
                    translation_prior_translations = np.asarray(current_translations, dtype=np.float32)
            translation_log_prior = None
            if not use_local:
                translation_log_prior = make_relion_translation_log_prior(
                    translation_prior_translations,
                    cryo.voxel_size,
                    current_sigma_offset_angstrom,
                    trans_prior_center,
                    offset_range_pixels=None,
                )
            if experiment_datasets[k].n_units == 0:
                logger.info("Skipping E-step/M-step accumulation for empty half-%d dataset", k + 1)
                n_shells = int(cryo.image_shape[0] // 2 + 1)
                n_rot_for_stats = int(rotation_grid_size(local_search_order) if use_local else effective_rotations.shape[0])
                Ft_y_k = jnp.zeros(int(np.prod(padded_volume_shape)), dtype=jnp.complex128)
                Ft_ctf_k = jnp.zeros(int(np.prod(padded_volume_shape)), dtype=jnp.complex128)
                ha_k = np.zeros(0, dtype=np.int32)
                em_stats_k = RelionStats(
                    log_evidence_per_image=jnp.zeros(0, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(0, dtype=jnp.float32),
                    max_posterior_per_image=jnp.zeros(0, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.zeros(n_rot_for_stats, dtype=jnp.float32),
                )
                noise_stats_k = NoiseStats(
                    wsum_sigma2_noise=jnp.zeros(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.zeros(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=0.0,
                )
                noise_stats_per_half[k] = noise_stats_k
                hard_assignments[k] = ha_k
                coarse_ha[k] = ha_k
                max_posterior_per_half[k] = np.zeros(0, dtype=np.float32)
                rotation_posterior_per_half[k] = np.zeros(n_rot_for_stats, dtype=np.float32)
                if k == 0:
                    Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
                else:
                    Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k
                _parity_dump.collect_e_step(
                    half=k,
                    em_stats=em_stats_k,
                    hard_assignment=ha_k,
                    coarse_hard_assignment=coarse_ha[k],
                    noise_stats=noise_stats_k,
                    Ft_y=Ft_y_k,
                    Ft_ctf=Ft_ctf_k,
                    pose_rotation_eulers=pose_rotation_eulers[k],
                    best_pose_rotation_eulers=best_pose_rotation_eulers[k],
                    best_pose_translations=best_pose_translations[k],
                    translation_search_base=translation_search_bases[k],
                    original_image_indices=np.zeros(0, dtype=np.int64),
                )
                continue
            if use_local:
                # For local search the per-chunk M-step only sees the
                # cone-restricted rotation set (typically a few thousand
                # rotations per image with high overlap across the chunk)
                # rather than the full ~10⁶-rotation grid at healpix order
                # 5+. Sizing the batch by the full grid produces ibs ≈ 5
                # at order 5 → chunks of 5 images → ~500 chunks per half
                # → ~7 hours per iter on the 5k benchmark.
                #
                # Estimate the per-image cone size from
                #     fraction = (sigma_cutoff * sigma_rot / pi)^2
                # which is the spherical cap area as a fraction of the
                # full SO(3) volume (good to within ~30% for reasonable
                # cones). Use that to compute an effective rotation count
                # equal to ``chunk_size * cone_size``, with a safety
                # factor of 2x for cone-overlap inefficiency.
                _cone_radius = 3.0 * float(sigma_rot)  # sigma_cutoff=3.0
                _cone_fraction = max(
                    (_cone_radius / float(np.pi)) ** 2,
                    1.0 / float(rotation_grid_size(local_search_order)),
                )
                _est_cone_rots = int(np.ceil(rotation_grid_size(local_search_order) * _cone_fraction))
                # Per-chunk effective rotations ≈ 2 * cone_size
                # (after dedup of overlapping cones).
                _eff_n_rot = max(64, 2 * _est_cone_rots)
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    _eff_n_rot,
                    current_translations.shape[0],
                )
                exact_local_batch_override = _relion_exact_local_image_batch_override()
                if exact_local_batch_override is not None:
                    safe_ibs = min(int(exact_local_batch_override), image_batch_size)
                logger.info(
                    "Local search batch sizing: cone_radius=%.3f rad "
                    "(%.2f deg), est_cone_rots=%d, eff_n_rot=%d "
                    "→ image_batch_size=%d, rotation_block_size=%d",
                    _cone_radius,
                    np.rad2deg(_cone_radius),
                    _est_cone_rots,
                    _eff_n_rot,
                    safe_ibs,
                    safe_rbs,
                )
                translation_prior_reference_translations = np.asarray(current_translations, dtype=np.float32)
                if local_search_translation_prior_mode == "coarse":
                    if _replay_prior_translations is not None:
                        translation_prior_reference_translations = np.asarray(
                            _replay_prior_translations, dtype=np.float32
                        )
                    else:
                        translation_prior_reference_translations = np.asarray(base_translations, dtype=np.float32)
                    logger.info(
                        "RELION mode: local translation prior uses coarse base grid (n=%d) while scoring perturbed translations",
                        translation_prior_reference_translations.shape[0],
                    )
                # RELION's accelerated local-search loop still executes the
                # symbolic second pass when adaptive_oversampling == 0. In
                # that case convertAllSquaredDifferencesToWeights sets
                # significant_weight to the minimum fine-pass weight, so
                # storeWeightedSums keeps all local candidates. Do not apply
                # the 0.999 significant-support prune on this os0 path.
                local_reconstruct_significant_only = state.adaptive_oversampling > 0
                local_outputs = _run_local_search_iteration(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance_k,
                    relion_half_inputs.previous_best_rotation_eulers[k],
                    local_search_rotations,
                    local_search_rotation_eulers,
                    local_search_order,
                    sigma_rot,
                    sigma_psi,
                    current_translations,
                    trans_prior_center,
                    current_sigma_offset_angstrom,
                    current_translation_range,
                    disc_type,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=cs_for_engine,
                    accumulate_noise=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    reconstruction_padding_factor=PADDING_FACTOR,
                    use_float64_scoring=_relion_use_float64_scoring(),
                    use_float64_projections=False,
                    do_gridding_correction=True,
                    square_window=RELION_FOURIER_WINDOW_SQUARE,
                    half_spectrum_scoring=True,
                    image_corrections=relion_half_inputs.image_corrections[k],
                    scale_corrections=relion_half_inputs.scale_corrections[k],
                    image_pre_shifts=translation_search_base,
                    score_with_masked_images=True,
                    return_profile=collect_local_search_profile,
                    disable_adjoint_y=disable_adjoint_y,
                    disable_adjoint_ctf=disable_adjoint_ctf,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    reconstruct_significant_only=local_reconstruct_significant_only,
                    translation_prior_reference_translations=translation_prior_reference_translations,
                    debug_iteration=iteration + 1,
                    return_best_pose_details=True,
                    translation_prior_centers=trans_prior_center,
                    rotation_grid_random_perturbation=local_search_random_perturbation,
                    rotation_grid_angular_sampling_deg=local_search_angular_sampling_deg,
                )
                if collect_local_search_profile:
                    (
                        Ft_y_k,
                        Ft_ctf_k,
                        ha_k,
                        best_rots_k,
                        best_trans_k,
                        _best_rot_ids_k,
                        em_stats_k,
                        noise_stats_k,
                        local_profile_k,
                    ) = local_outputs
                    profile_row = dict(local_profile_k)
                    profile_row["iteration"] = np.int32(iteration)
                    profile_row["half_index"] = np.int32(k)
                    local_profile_history.append(profile_row)
                    if save_intermediates_dir is not None:
                        np.savez_compressed(
                            os.path.join(
                                save_intermediates_dir,
                                f"it{iteration:03d}_half{k + 1}_local_profile.npz",
                            ),
                            **local_profile_k,
                        )
                else:
                    (
                        Ft_y_k,
                        Ft_ctf_k,
                        ha_k,
                        best_rots_k,
                        best_trans_k,
                        _best_rot_ids_k,
                        em_stats_k,
                        noise_stats_k,
                    ) = local_outputs
                    best_pose_rotations[k] = np.asarray(best_rots_k, dtype=np.float32)
                best_pose_rotations[k] = np.asarray(best_rots_k, dtype=np.float32)
                best_pose_rotation_eulers[k] = utils.R_to_relion(
                    np.asarray(best_rots_k),
                    degrees=True,
                ).astype(np.float32)
                best_pose_translations[k] = np.asarray(best_trans_k, dtype=np.float32)
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = None
                coarse_ha[k] = ha_k

            elif use_adaptive:
                # --- PASS 1: Coarse significance pruning ---
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )

                t_pass1 = time.time()
                sig_rot_any, n_sig_batch, ha_coarse, sig_sample_indices = _compute_significance_batched(
                    experiment_datasets[k],
                    means[k],
                    noise_variance_k,
                    effective_rotations,
                    current_translations,
                    disc_type,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=coarse_cs,
                    score_with_masked_images=True,
                    return_significant_sample_indices=True,
                    rotation_log_prior=rotation_log_prior_k,
                    translation_log_prior=translation_log_prior,
                    image_corrections=relion_half_inputs.image_corrections[k],
                    scale_corrections=relion_half_inputs.scale_corrections[k],
                    image_pre_shifts=translation_search_base,
                    half_spectrum_scoring=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    do_gridding_correction=True,
                    square_window=RELION_FOURIER_WINDOW_SQUARE,
                    use_float64_scoring=_relion_use_float64_scoring(),
                )
                total_coarse_samples = int(
                    effective_rotations.shape[0] * current_translations.shape[0],
                )
                adaptive_pass1_diag[k] = {
                    "n_significant_per_image": np.asarray(n_sig_batch, dtype=np.int32),
                    "significant_rotation_union_mask": np.asarray(sig_rot_any, dtype=bool),
                    "coarse_hard_assignment": np.asarray(ha_coarse, dtype=np.int32),
                    "coarse_size": -1 if coarse_cs is None else int(coarse_cs),
                    "total_coarse_samples": total_coarse_samples,
                    "significant_rotation_union_count": int(np.sum(sig_rot_any)),
                }
                n_sig_total = int(np.sum(sig_rot_any))
                dt_pass1 = time.time() - t_pass1

                logger.info(
                    "Pass 1 (half %d): %d / %d significant coarse rotations in %.1fs (median n_sig/image=%d)",
                    k,
                    n_sig_total,
                    effective_rotations.shape[0],
                    dt_pass1,
                    int(np.median(n_sig_batch)),
                )

                skip_pass2, sig_fraction = should_skip_adaptive_pass2(
                    n_sig_batch,
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                    threshold=adaptive_pass2_skip_threshold,
                )

                if skip_pass2:
                    logger.info(
                        "Pass 2 skipped (half %d): mean significant fraction=%.3f >= %.3f; "
                        "running single-pass full-resolution E+M",
                        k,
                        sig_fraction,
                        ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION,
                    )
                    _, ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = run_em(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance_k,
                        effective_rotations,
                        current_translations,
                        disc_type,
                        image_batch_size=safe_ibs,
                        rotation_block_size=safe_rbs,
                        current_size=cs_for_engine,
                        rotation_log_prior=rotation_log_prior_k,
                        translation_log_prior=translation_log_prior,
                        score_with_masked_images=True,
                        return_stats=True,
                        accumulate_noise=True,
                        half_spectrum_scoring=True,
                        projection_padding_factor=PROJECTION_PADDING_FACTOR,
                        reconstruction_padding_factor=PADDING_FACTOR,
                        image_corrections=relion_half_inputs.image_corrections[k],
                        scale_corrections=relion_half_inputs.scale_corrections[k],
                        image_pre_shifts=translation_search_base,
                        translation_prior_centers=trans_prior_center,
                        use_float64_scoring=_relion_use_float64_scoring(),
                        use_float64_projections=False,
                        do_gridding_correction=True,
                        square_window=RELION_FOURIER_WINDOW_SQUARE,
                        sparse_pass2=False,
                        disable_adjoint_y=disable_adjoint_y,
                        disable_adjoint_ctf=disable_adjoint_ctf,
                        relion_firstiter_score_mode=(
                            "normalized_cc"
                            if (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter)
                            else "gaussian"
                        ),
                        relion_firstiter_winner_take_all=(
                            relion_firstiter_cc_this_iter or first_iter_hard_reconstruction_this_iter
                        ),
                    )
                    noise_stats_per_half[k] = noise_stats_k
                    pose_rotations[k] = effective_rotations
                    pose_rotation_eulers[k] = effective_rotation_eulers
                    pose_translations[k] = np.asarray(current_translations, dtype=np.float32)
                    coarse_ha[k] = ha_k
                elif np.all(np.asarray(n_sig_batch) == total_coarse_samples):
                    # Exact early-iteration fast path: if every coarse sample is
                    # significant for every image, sparse per-image pass 2 is
                    # equivalent to one shared dense oversampled pass.
                    t_pass2 = time.time()
                    pass2_outputs = compute_pass2_stats(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance_k,
                        effective_rotations,
                        current_translations,
                        np.ones(effective_rotations.shape[0], dtype=bool),
                        current_nside_level,
                        disc_type,
                        oversampling_order=state.adaptive_oversampling,
                        current_size=cs_for_engine,
                        translation_step=state.translation_step,
                        rotation_log_prior=rotation_log_prior_k,
                        translation_log_prior=translation_log_prior,
                        score_with_masked_images=True,
                        return_stats=True,
                        accumulate_noise=True,
                        half_spectrum_scoring=True,
                        projection_padding_factor=PROJECTION_PADDING_FACTOR,
                        reconstruction_padding_factor=PADDING_FACTOR,
                        image_corrections=relion_half_inputs.image_corrections[k],
                        scale_corrections=relion_half_inputs.scale_corrections[k],
                        image_pre_shifts=translation_search_base,
                        use_float64_scoring=_relion_use_float64_scoring(),
                        do_gridding_correction=True,
                        square_window=RELION_FOURIER_WINDOW_SQUARE,
                        random_perturbation=random_perturbation,
                    )
                    Ft_y_k, Ft_ctf_k, ha_k, oversampled_rots_k, em_stats_k, noise_stats_k = pass2_outputs
                    noise_stats_per_half[k] = noise_stats_k
                    dt_pass2 = time.time() - t_pass2
                    logger.info(
                        "Pass 2 dense exact (half %d): %.1fs using full oversampled grid",
                        k,
                        dt_pass2,
                    )
                    pose_rotations[k] = np.asarray(oversampled_rots_k, dtype=np.float32)
                    pose_rotation_eulers[k] = utils.R_to_relion(
                        np.asarray(oversampled_rots_k),
                        degrees=True,
                    ).astype(np.float32)
                    oversampled_translations, _ = get_oversampled_translation_grid(
                        np.asarray(current_translations, dtype=np.float32),
                        state.translation_step,
                        oversampling_order=state.adaptive_oversampling,
                    )
                    pose_translations[k] = np.asarray(
                        oversampled_translations,
                        dtype=np.float32,
                    )
                    coarse_ha[k] = ha_coarse
                else:
                    # --- Exact sparse pass 2 over significant coarse samples ---
                    t_pass2 = time.time()
                    sparse_pass2_profile_k = None
                    sparse_outputs = _run_sparse_pass2_local_search_iteration(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance_k,
                        current_translations,
                        sig_sample_indices,
                        current_nside_level,
                        disc_type,
                        oversampling_order=state.adaptive_oversampling,
                        current_size=cs_for_engine,
                        translation_step=state.translation_step,
                        rotation_log_prior=rotation_log_prior_k,
                        translation_log_prior=translation_log_prior,
                        score_with_masked_images=True,
                        return_stats=True,
                        accumulate_noise=True,
                        half_spectrum_scoring=True,
                        projection_padding_factor=PROJECTION_PADDING_FACTOR,
                        reconstruction_padding_factor=PADDING_FACTOR,
                        image_corrections=relion_half_inputs.image_corrections[k],
                        scale_corrections=relion_half_inputs.scale_corrections[k],
                        image_pre_shifts=translation_search_base,
                        use_float64_scoring=_relion_use_float64_scoring(),
                        do_gridding_correction=True,
                        square_window=RELION_FOURIER_WINDOW_SQUARE,
                        random_perturbation=random_perturbation,
                        image_batch_size=image_batch_size,
                        rotation_block_size=rotation_block_size,
                        adaptive_fraction=adaptive_fraction,
                        debug_iteration=iteration,
                        return_profile=collect_local_search_profile,
                    )
                    if collect_local_search_profile:
                        (
                            Ft_y_k,
                            Ft_ctf_k,
                            ha_k,
                            best_rots_k,
                            best_trans_k,
                            _best_rot_indices_k,
                            em_stats_k,
                            noise_stats_k,
                            sparse_pass2_profile_k,
                        ) = sparse_outputs
                    else:
                        (
                            Ft_y_k,
                            Ft_ctf_k,
                            ha_k,
                            best_rots_k,
                            best_trans_k,
                            _best_rot_indices_k,
                            em_stats_k,
                            noise_stats_k,
                        ) = sparse_outputs
                    noise_stats_per_half[k] = noise_stats_k
                    dt_pass2 = time.time() - t_pass2
                    logger.info(
                        "Pass 2 exact-local (half %d): %.1fs",
                        k,
                        dt_pass2,
                    )
                    if sparse_pass2_profile_k is not None:
                        profile_row = dict(sparse_pass2_profile_k)
                        profile_row["iteration"] = np.int32(iteration)
                        profile_row["half_index"] = np.int32(k)
                        profile_row["profile_context"] = np.array("adaptive_sparse_pass2")
                        local_profile_history.append(profile_row)
                        if save_intermediates_dir is not None:
                            np.savez_compressed(
                                os.path.join(
                                    save_intermediates_dir,
                                    f"it{iteration:03d}_half{k + 1}_pass2_local_profile.npz",
                                ),
                                **sparse_pass2_profile_k,
                            )
                    best_pose_rotations[k] = np.asarray(best_rots_k, dtype=np.float32)
                    best_pose_rotation_eulers[k] = utils.R_to_relion(
                        np.asarray(best_rots_k),
                        degrees=True,
                    ).astype(np.float32)
                    best_pose_translations[k] = np.asarray(best_trans_k, dtype=np.float32)
                    oversampled_translations, _ = get_oversampled_translation_grid(
                        np.asarray(current_translations, dtype=np.float32),
                        state.translation_step,
                        oversampling_order=state.adaptive_oversampling,
                    )
                    pose_translations[k] = np.asarray(
                        oversampled_translations,
                        dtype=np.float32,
                    )

                    # Store coarse-grid assignment from pass 1 for local search.
                    coarse_ha[k] = ha_coarse

                if iter_sig_counts is None:
                    iter_sig_counts = n_sig_batch
                else:
                    iter_sig_counts = np.concatenate([iter_sig_counts, n_sig_batch])

            elif use_global_significant_support:
                # --- SINGLE-PASS GLOBAL SIGNIFICANT SUPPORT (RELION os0 parity) ---
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )
                t_pass1 = time.time()
                _sig_result = _compute_significance_batched(
                    experiment_datasets[k],
                    means[k],
                    noise_variance_k,
                    effective_rotations,
                    current_translations,
                    disc_type,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=cs_for_engine,
                    score_with_masked_images=True,
                    return_significant_sample_indices=True,
                    rotation_log_prior=rotation_log_prior_k,
                    translation_log_prior=translation_log_prior,
                    image_corrections=relion_half_inputs.image_corrections[k],
                    scale_corrections=relion_half_inputs.scale_corrections[k],
                    image_pre_shifts=translation_search_base,
                    half_spectrum_scoring=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    do_gridding_correction=True,
                    square_window=RELION_FOURIER_WINDOW_SQUARE,
                    use_float64_scoring=_relion_use_float64_scoring(),
                    return_full_stats=True,
                )
                if len(_sig_result) == 5:
                    sig_rot_any, n_sig_batch, ha_coarse, sig_sample_indices, full_coarse_stats = _sig_result
                else:
                    sig_rot_any, n_sig_batch, ha_coarse, sig_sample_indices = _sig_result
                    full_coarse_stats = None
                total_samples = int(effective_rotations.shape[0] * current_translations.shape[0])
                adaptive_pass1_diag[k] = {
                    "n_significant_per_image": np.asarray(n_sig_batch, dtype=np.int32),
                    "significant_rotation_union_mask": np.asarray(sig_rot_any, dtype=bool),
                    "coarse_hard_assignment": np.asarray(ha_coarse, dtype=np.int32),
                    "coarse_size": -1 if cs_for_engine is None else int(cs_for_engine),
                    "total_coarse_samples": total_samples,
                    "significant_rotation_union_count": int(np.sum(sig_rot_any)),
                }
                dt_pass1 = time.time() - t_pass1
                logger.info(
                    "Global significant support (half %d): median n_sig/image=%d, max=%d / %d in %.1fs",
                    k,
                    int(np.median(n_sig_batch)),
                    int(np.max(n_sig_batch)),
                    total_samples,
                    dt_pass1,
                )

                t_pass2 = time.time()
                sparse_pass2_profile_k = None
                sparse_outputs = _run_sparse_pass2_local_search_iteration(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance_k,
                    current_translations,
                    sig_sample_indices,
                    current_nside_level,
                    disc_type,
                    oversampling_order=0,
                    current_size=cs_for_engine,
                    translation_step=state.translation_step,
                    rotation_log_prior=rotation_log_prior_k,
                    translation_log_prior=translation_log_prior,
                    score_with_masked_images=True,
                    return_stats=True,
                    accumulate_noise=True,
                    half_spectrum_scoring=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    reconstruction_padding_factor=PADDING_FACTOR,
                    image_corrections=relion_half_inputs.image_corrections[k],
                    scale_corrections=relion_half_inputs.scale_corrections[k],
                    image_pre_shifts=translation_search_base,
                    use_float64_scoring=_relion_use_float64_scoring(),
                    do_gridding_correction=True,
                    square_window=RELION_FOURIER_WINDOW_SQUARE,
                    random_perturbation=random_perturbation,
                    image_batch_size=image_batch_size,
                    rotation_block_size=rotation_block_size,
                    adaptive_fraction=adaptive_fraction,
                    debug_iteration=iteration,
                    return_profile=collect_local_search_profile,
                    translation_prior_centers=trans_prior_center,
                    normalization_log_z=(
                        None if full_coarse_stats is None else full_coarse_stats["normalization_log_z"]
                    ),
                )
                if collect_local_search_profile:
                    (
                        Ft_y_k,
                        Ft_ctf_k,
                        ha_k,
                        best_rots_k,
                        best_trans_k,
                        _best_rot_indices_k,
                        em_stats_k,
                        noise_stats_k,
                        sparse_pass2_profile_k,
                    ) = sparse_outputs
                else:
                    (
                        Ft_y_k,
                        Ft_ctf_k,
                        ha_k,
                        best_rots_k,
                        best_trans_k,
                        _best_rot_indices_k,
                        em_stats_k,
                        noise_stats_k,
                    ) = sparse_outputs
                noise_stats_per_half[k] = noise_stats_k
                dt_pass2 = time.time() - t_pass2
                logger.info("Global significant support exact-local pass 2 (half %d): %.1fs", k, dt_pass2)
                if sparse_pass2_profile_k is not None:
                    profile_row = dict(sparse_pass2_profile_k)
                    profile_row["iteration"] = np.int32(iteration)
                    profile_row["half_index"] = np.int32(k)
                    profile_row["profile_context"] = np.array("global_sparse_pass2")
                    local_profile_history.append(profile_row)
                    if save_intermediates_dir is not None:
                        np.savez_compressed(
                            os.path.join(
                                save_intermediates_dir,
                                f"it{iteration:03d}_half{k + 1}_global_pass2_local_profile.npz",
                            ),
                            **sparse_pass2_profile_k,
                        )

                if full_coarse_stats is not None:
                    # In this RELION os0 parity path the sparse pass-2
                    # reconstruction uses the full coarse denominator, but
                    # the public pose/Pmax history remains in the coarse grid
                    # coordinate system expected by downstream comparison
                    # scripts and local-search replay.
                    ha_k = ha_coarse
                    em_stats_k = RelionStats(
                        log_evidence_per_image=jnp.asarray(full_coarse_stats["log_evidence_per_image"]),
                        best_log_score_per_image=jnp.asarray(full_coarse_stats["best_log_score_per_image"]),
                        max_posterior_per_image=jnp.asarray(full_coarse_stats["max_posterior_per_image"]),
                        rotation_posterior_sums=em_stats_k.rotation_posterior_sums,
                    )

                pose_rotations[k] = effective_rotations
                pose_rotation_eulers[k] = effective_rotation_eulers
                pose_translations[k] = np.asarray(current_translations, dtype=np.float32)
                coarse_ha[k] = ha_coarse

                if iter_sig_counts is None:
                    iter_sig_counts = np.asarray(n_sig_batch, dtype=np.int32)
                else:
                    iter_sig_counts = np.concatenate([iter_sig_counts, np.asarray(n_sig_batch, dtype=np.int32)])

            else:
                # --- SINGLE-PASS E+M (no adaptive oversampling) ---
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )
                _, ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = run_em(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance_k,
                    effective_rotations,
                    current_translations,
                    disc_type,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=cs_for_engine,
                    rotation_log_prior=rotation_log_prior_k,
                    translation_log_prior=translation_log_prior,
                    score_with_masked_images=True,
                    return_stats=True,
                    accumulate_noise=True,
                    half_spectrum_scoring=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    reconstruction_padding_factor=PADDING_FACTOR,
                    image_corrections=relion_half_inputs.image_corrections[k],
                    scale_corrections=relion_half_inputs.scale_corrections[k],
                    image_pre_shifts=translation_search_base,
                    translation_prior_centers=trans_prior_center,
                    use_float64_scoring=_relion_use_float64_scoring(),
                    use_float64_projections=False,
                    do_gridding_correction=True,
                    square_window=RELION_FOURIER_WINDOW_SQUARE,
                    sparse_pass2=False,
                    disable_adjoint_y=disable_adjoint_y,
                    disable_adjoint_ctf=disable_adjoint_ctf,
                    relion_firstiter_score_mode=(
                        "normalized_cc"
                        if (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter)
                        else "gaussian"
                    ),
                    relion_firstiter_winner_take_all=(
                        relion_firstiter_cc_this_iter or first_iter_hard_reconstruction_this_iter
                    ),
                )
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = effective_rotations
                pose_rotation_eulers[k] = effective_rotation_eulers
                pose_translations[k] = np.asarray(current_translations, dtype=np.float32)
                coarse_ha[k] = ha_k  # same grid, no oversampling

                # --- Manifest dump for deterministic replay (Phase 0.1) ---
                if save_intermediates_dir is not None:
                    _manifest_path = os.path.join(
                        save_intermediates_dir,
                        f"manifest_iter{iteration}_half{k}.npz",
                    )
                    _manifest = {
                        "effective_rotations": np.asarray(effective_rotations, dtype=np.float32),
                        "current_translations": np.asarray(current_translations, dtype=np.float32),
                        "rotation_log_prior": np.asarray(rotation_log_prior_k, dtype=np.float64)
                        if rotation_log_prior_k is not None
                        else np.array([]),
                        "translation_log_prior": np.asarray(translation_log_prior, dtype=np.float64)
                        if translation_log_prior is not None
                        else np.array([]),
                        "image_corrections": np.asarray(relion_half_inputs.image_corrections[k], dtype=np.float64)
                        if relion_half_inputs.image_corrections[k] is not None
                        else np.array([]),
                        "scale_corrections": np.asarray(relion_half_inputs.scale_corrections[k], dtype=np.float64)
                        if relion_half_inputs.scale_corrections[k] is not None
                        else np.array([]),
                        "image_pre_shifts": np.asarray(translation_search_base, dtype=np.float32)
                        if translation_search_base is not None
                        else np.array([]),
                        "absolute_previous_translations": np.asarray(previous_translations_k, dtype=np.float32)
                        if previous_translations_k is not None
                        else np.array([]),
                        "mean_vol_ft": np.asarray(means[k]),
                        "mean_variance": np.asarray(mean_variance),
                        "noise_variance": np.asarray(noise_variance_k),
                        "current_size": np.int32(cs_for_engine) if cs_for_engine is not None else np.int32(-1),
                        "half_spectrum_scoring": np.bool_(True),
                        "use_float64_scoring": np.bool_(_relion_use_float64_scoring()),
                        "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                        "reconstruction_padding_factor": np.int32(PADDING_FACTOR),
                        "score_with_masked_images": np.bool_(True),
                        "perturbation_instance": np.float64(random_perturbation),
                        "perturbation_factor": np.float64(perturb_factor),
                        "iteration": np.int32(iteration),
                        "half_index": np.int32(k),
                        "ave_Pmax": np.float64(float(np.mean(em_stats_k.max_posterior_per_image))),
                    }
                    np.savez(_manifest_path, **_manifest)
                    logger.info("Manifest dumped: %s", _manifest_path)

            # NOTE: means[k] reconstruction is DEFERRED until after the
            # low_resol_join_halves step below — we need both halves'
            # Ft_y / Ft_ctf accumulators in hand before we can average
            # the low-frequency shells across the two halves.
            hard_assignments[k] = ha_k
            max_posterior_per_half[k] = np.asarray(
                em_stats_k.max_posterior_per_image,
                dtype=np.float32,
            )
            rotation_posterior_per_half[k] = np.asarray(
                em_stats_k.rotation_posterior_sums,
                dtype=np.float32,
            )

            if k == 0:
                Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
            else:
                Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k

            # Capture original-stack image indices for the half so dumps can be
            # matched to RELION's data.star image_name ordering.
            try:
                _half_orig_idx = np.asarray(
                    experiment_datasets[k]._index_layout.original_image_indices_for_local(
                        np.arange(experiment_datasets[k].n_images, dtype=np.int32)
                    ),
                    dtype=np.int64,
                )
            except Exception:
                _half_orig_idx = None
            _parity_dump.collect_e_step(
                half=k,
                em_stats=em_stats_k,
                hard_assignment=ha_k,
                coarse_hard_assignment=coarse_ha[k],
                noise_stats=noise_stats_per_half[k],
                Ft_y=Ft_y_k,
                Ft_ctf=Ft_ctf_k,
                pose_rotation_eulers=pose_rotation_eulers[k],
                best_pose_rotation_eulers=best_pose_rotation_eulers[k],
                best_pose_translations=best_pose_translations[k],
                translation_search_base=translation_search_bases[k] if "translation_search_bases" in dir() else None,
                original_image_indices=_half_orig_idx,
            )

        # E-step + per-half M-step accumulators are now both populated.
        _parity_dump.mark_stage(iteration, "e_step")

        # --- RELION's --low_resol_join_halves: average the low-resolution
        # shells of the per-half Fourier accumulators between the two halves
        # BEFORE the Wiener solve. This forces the two half-maps to share
        # their low-frequency content, preventing them from diverging in
        # orientation space at SNR-poor low shells. RELION mirrors this in
        # ml_optimiser_mpi.cpp::joinTwoHalvesAtLowResolution; without it
        # recovar's iter-N FSC drops gradually from shell ~2 while RELION's
        # stays at 1.0 through shell 13 (= 40 A for a 128/4.25 dataset),
        # which directly translates to a ~5-shell deficit in
        # ``first_shell_below_0.5`` and a ~10-pixel/iter deficit in
        # ``current_size`` growth (the dominant convergence-speed gap
        # observed in the 2026-04 5k normalized parity benchmark).
        #
        # Use the previous iteration's resolution to cap the join radius
        # (so we never join shells beyond the actual resolution of the
        # map). Mirrors the ``XMIPP_MAX(low_resol_join_halves,
        # 1./mymodel.current_resolution)`` in RELION's source.
        prev_res_angstrom = None
        if pixel_resolutions:
            prev_pixel_res = pixel_resolutions[-1]
            if prev_pixel_res > 0:
                prev_res_angstrom = shell_index_to_resolution_angstrom(
                    prev_pixel_res,
                    grid_size,
                    cryo.voxel_size,
                )
        Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1 = regularization.join_halves_at_low_resolution(
            Ft_y_0,
            Ft_y_1,
            Ft_ctf_0,
            Ft_ctf_1,
            padded_volume_shape,
            cryo.voxel_size,
            grid_size,
            low_resol_join_halves_angstrom,
            current_resolution_angstrom=prev_res_angstrom,
        )

        # --- RELION-exact M-step ordering (auto-refine, split-half) ---
        # RELION (ml_optimiser_mpi.cpp:4031, 4091; backprojector.cpp:1044):
        #   1. compareTwoHalves() -> CURRENT iter's FSC from BPref accumulators
        #   2. maximization() -> updateSSNRarrays(THIS_ITER_FSC) -> tau2
        #   3. reconstruct(tau2) -> regularized half-map
        #
        # Recovar previously called compute_relion_tau2_from_weights with
        # fsc_history[-1] / init_fsc (PREVIOUS iter's FSC). At cold start
        # init_fsc is essentially zeros and at iter 2 prev-iter FSC is
        # poisoned (~0.999) by leakage of the under-regularized iter-1 maps,
        # which gives ssnr ≈ 999 → tau2 amplifies 1e6× → ave_Pmax collapse.
        # Algorithm doc: docs/math/relion_updateSSNR_algorithm_2026_04_25.md
        #
        # Snapshot the previous-iter means BEFORE the unreg reconstruction so
        # sign alignment has a reference at iter 1 (where the init volumes
        # are means[*] before any reconstruction overwrites them).
        previous_means = [np.asarray(mean).copy() if mean is not None else None for mean in means]

        # Compute CURRENT iter FSC FIRST, then derive tau2 from that fresh FSC,
        # then the regularized Wiener solve. RELION computes this FSC from
        # BackProjector::getDownsampledAverage, not from reconstructed
        # unregularized maps; using the reconstructed maps underestimates the
        # joined low-resolution shells and depresses data_vs_prior.
        _t_unreg_first = time.time()
        current_iter_fsc = regularization.compute_relion_fsc_from_backprojector(
            Ft_y_0,
            Ft_y_1,
            Ft_ctf_0,
            Ft_ctf_1,
            volume_shape,
            padding_factor=PADDING_FACTOR,
            r_max=cs // 2,
        )
        logger.info(
            "Computed iter-%d FSC for tau2 (RELION backprojector path): %.1fs",
            iteration + 1,
            time.time() - _t_unreg_first,
        )

        # RELION calls BackProjector::updateSSNRarrays independently for each
        # half-map BPref.  The gold-standard FSC is shared, but sigma2/tau2
        # come from each half's own Fourier weight outside the joined shells.
        tau2_update_details_per_half = []
        mean_signal_variance_per_half = []
        for Ft_ctf_half in (Ft_ctf_0, Ft_ctf_1):
            mean_signal_variance_k, _, tau2_update_details_k = regularization.compute_relion_tau2_from_weights(
                Ft_ctf_half,
                Ft_ctf_half,
                current_iter_fsc,
                volume_shape,
                tau2_fudge=tau2_fudge,
                padding_factor=PADDING_FACTOR,
                r_max=cs // 2,
                return_details=True,
            )
            mean_signal_variance_per_half.append(mean_signal_variance_k)
            tau2_update_details_per_half.append(tau2_update_details_k)
        mean_signal_variance = 0.5 * (mean_signal_variance_per_half[0] + mean_signal_variance_per_half[1])
        # Keep the legacy single tau2 diagnostic fields aligned with RELION's
        # half1 model.star, which is what the parity diff script reports.
        tau2_update_details = tau2_update_details_per_half[0]
        logger.info(
            "tau2 update from THIS-iter FSC: old_max=%.4e new_max=%.4e half_max=(%.4e, %.4e)",
            float(jnp.max(jnp.abs(mean_variance))),
            float(jnp.max(jnp.abs(mean_signal_variance))),
            float(jnp.max(jnp.abs(mean_signal_variance_per_half[0]))),
            float(jnp.max(jnp.abs(mean_signal_variance_per_half[1]))),
        )
        mean_variance = mean_signal_variance

        # --- Free previous-iteration means to reclaim GPU memory ---
        # (previous_means already snapshotted earlier for FSC sign alignment)
        for k in range(2):
            means[k] = None

        # --- Now reconstruct the regularized per-half means from the
        # (post-join) Ft_y / Ft_ctf accumulators.  When PADDING_FACTOR > 1,
        # the engine already backprojected into a (pf*N)³ grid.
        # Use eager (non-JIT) reconstruction to avoid ~30 min XLA compile
        # overhead for the monolithic 256³ graph in post_process_from_filter_v2.
        _t_recon = time.time()
        for k in range(2):
            Ft_y_k_local = Ft_y_0 if k == 0 else Ft_y_1
            Ft_ctf_k_local = Ft_ctf_0 if k == 0 else Ft_ctf_1
            means[k] = _reconstruct_volume_eager(
                Ft_ctf_k_local,
                Ft_y_k_local,
                volume_shape,
                PADDING_FACTOR,
                tau=mean_signal_variance_per_half[k],
                tau2_fudge=tau2_fudge,
                projection_padding_factor=PROJECTION_PADDING_FACTOR,
                minres_map=RELION_MINRES_MAP,
            ).reshape(-1)

            # RELION's solventFlatten (ml_optimiser.cpp:5469): mask the
            # reconstructed reference outside particle_diameter to remove
            # solvent noise before the next E-step's projections.
            if particle_diameter_ang is not None and particle_diameter_ang > 0:
                flatten_radius = particle_diameter_ang / (2.0 * cryo.voxel_size)
                vol_real = fourier_transform_utils.get_idft3(means[k].reshape(volume_shape))
                solvent_mask = mask.raised_cosine_mask(
                    volume_shape,
                    radius=flatten_radius,
                    radius_p=flatten_radius + RELION_WIDTH_MASK_EDGE,
                    offset=jnp.zeros(3),
                )
                vol_real = vol_real * solvent_mask
                means[k] = fourier_transform_utils.get_dft3(vol_real).reshape(-1)
            if relion_firstiter_cc_this_iter:
                means[k] = _apply_relion_initial_lowpass_filter(
                    means[k],
                    volume_shape,
                    cryo.voxel_size,
                    relion_firstiter_ini_high_angstrom,
                    filter_edgewidth=RELION_WIDTH_MASK_EDGE,
                )
        if relion_firstiter_cc_this_iter and relion_firstiter_ini_high_angstrom is not None:
            logger.info(
                "RELION iter-1 CC emulation: reapplying ini_high low-pass filter at %.2f A",
                float(relion_firstiter_ini_high_angstrom),
            )
        logger.info("Regularized reconstruction (2 halves + flatten): %.1fs", time.time() - _t_recon)
        _parity_dump.mark_stage(iteration, "recon")

        significant_counts.append(iter_sig_counts)

        if (
            not use_local
            and all(rot_sum is not None for rot_sum in rotation_posterior_per_half)
            and effective_rotations.shape[0] == rotation_grid_size(current_healpix_order)
        ):
            for k in range(2):
                global_direction_prior_per_half[k] = collapse_rotation_posterior_to_direction_prior(
                    np.asarray(rotation_posterior_per_half[k], dtype=np.float64),
                    current_healpix_order,
                )
                global_direction_prior_order_per_half[k] = current_healpix_order

        # --- Combined Fourier weights for data_vs_prior at next iteration ---
        Ft_ctf_combined = Ft_ctf_0 + Ft_ctf_1

        # --- Compute unregularized half-maps only when diagnostics need them ---
        #
        # The FSC used for tau2 and convergence is already computed above
        # directly from the BackProjector accumulators (`current_iter_fsc`),
        # matching RELION's ordering. Reconstructing unregularized maps here is
        # only needed for saved intermediates/parity dumps, so skip it in normal
        # timing/production paths.
        _t_unreg = time.time()
        need_unreg_means = save_intermediates_dir is not None or _parity_dump.is_active()
        if need_unreg_means:
            unreg_means = [
                _reconstruct_volume_eager(
                    Ft_ctf_0,
                    Ft_y_0,
                    volume_shape,
                    PADDING_FACTOR,
                    tau=None,
                    tau2_fudge=tau2_fudge,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    minres_map=RELION_MINRES_MAP,
                ),
                _reconstruct_volume_eager(
                    Ft_ctf_1,
                    Ft_y_1,
                    volume_shape,
                    PADDING_FACTOR,
                    tau=None,
                    tau2_fudge=tau2_fudge,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    minres_map=RELION_MINRES_MAP,
                ),
            ]
        else:
            unreg_means = [None, None]
        for k in range(2):
            means[k], sign_flipped = _align_fourier_volume_sign_to_reference(means[k], previous_means[k], volume_shape)
            if sign_flipped and unreg_means[k] is not None:
                unreg_means[k] = -unreg_means[k]
            if sign_flipped:
                logger.info("Aligned half-%d volume sign to the previous reference", k + 1)
        logger.info(
            "Unregularized reconstruction (2 halves): %.1fs%s",
            time.time() - _t_unreg,
            "" if need_unreg_means else " (skipped; diagnostics disabled)",
        )

        # FSC was already computed above in the RELION-exact ordering block
        # (current_iter_fsc) and used to derive tau2 BEFORE the Wiener solve.
        # Reuse it here — recomputing would give the same value (same
        # underlying unreg accumulators).
        fsc = current_iter_fsc
        fsc_history.append(fsc)
        _parity_dump.mark_stage(iteration, "fsc")

        # --- Save intermediate volumes if requested ---
        if save_intermediates_dir is not None:
            from recovar.output.output import save_volume

            os.makedirs(save_intermediates_dir, exist_ok=True)
            np.save(os.path.join(save_intermediates_dir, f"it{iteration:03d}_Ft_y_0.npy"), np.asarray(Ft_y_0))
            np.save(os.path.join(save_intermediates_dir, f"it{iteration:03d}_Ft_y_1.npy"), np.asarray(Ft_y_1))
            np.save(os.path.join(save_intermediates_dir, f"it{iteration:03d}_Ft_ctf_0.npy"), np.asarray(Ft_ctf_0))
            np.save(os.path.join(save_intermediates_dir, f"it{iteration:03d}_Ft_ctf_1.npy"), np.asarray(Ft_ctf_1))
            for k_half in range(2):
                save_volume(
                    np.asarray(means[k_half]).reshape(-1),
                    os.path.join(
                        save_intermediates_dir,
                        f"it{iteration:03d}_half{k_half + 1}_reg",
                    ),
                    volume_shape=volume_shape,
                    from_ft=True,
                    voxel_size=cryo.voxel_size,
                )
                save_volume(
                    np.asarray(unreg_means[k_half]).reshape(-1),
                    os.path.join(
                        save_intermediates_dir,
                        f"it{iteration:03d}_half{k_half + 1}_unreg",
                    ),
                    volume_shape=volume_shape,
                    from_ft=True,
                    voxel_size=cryo.voxel_size,
                )
            # Save FSC and noise/tau2 per iteration
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_fsc.npy"),
                np.asarray(fsc),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_noise.npy"),
                np.asarray(noise_variance),
            )
            for k_half, noise_k in enumerate(noise_variance_per_half):
                np.save(
                    os.path.join(save_intermediates_dir, f"it{iteration:03d}_noise_half{k_half + 1}.npy"),
                    np.asarray(noise_k),
                )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_tau2.npy"),
                np.asarray(mean_variance),
            )
            # Save hard assignments for angular error analysis
            for k_half in range(2):
                if hard_assignments[k_half] is not None:
                    np.save(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_ha_half{k_half + 1}.npy",
                        ),
                        hard_assignments[k_half],
                    )
            # Save per-iteration metadata
            iter_meta = {
                "iteration": iteration,
                "current_size": int(cs),
                "n_rotations": int(
                    rotation_grid_size(local_search_order) if use_local else effective_rotations.shape[0]
                ),
                "n_translations": int(current_translations.shape[0]),
                "healpix_order": int(state.healpix_order),
                "local_search": bool(use_local),
                "sigma_rot": float(state.sigma_rot),
            }
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_meta.npy"),
                iter_meta,
            )
            # Save the effective rotation grid for angular error computation
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_rotations.npy"),
                (np.asarray(effective_rotations) if not use_local else np.empty((0, 3, 3), dtype=np.float32)),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_translations.npy"),
                np.asarray(current_translations),
            )
            for k_half in range(2):
                if coarse_ha[k_half] is not None:
                    np.save(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_coarse_ha_half{k_half + 1}.npy",
                        ),
                        np.asarray(coarse_ha[k_half], dtype=np.int32),
                    )
                pass1_diag = adaptive_pass1_diag[k_half]
                if pass1_diag is not None:
                    np.savez_compressed(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_half{k_half + 1}_pass1_diag.npz",
                        ),
                        **pass1_diag,
                    )
            logger.info(
                "Saved intermediate volumes to %s (iteration %d)",
                save_intermediates_dir,
                iteration,
            )

        # --- Compute ave_Pmax from the actual E-step maxima ---
        if any(pmax is None for pmax in max_posterior_per_half):
            raise RuntimeError(
                "RELION mode expected per-image posterior maxima from the EM engine",
            )
        combined_max_posterior = np.concatenate(
            [np.asarray(pmax, dtype=np.float32) for pmax in max_posterior_per_half],
            axis=0,
        )
        ave_pmax = float(np.mean(combined_max_posterior))
        ave_Pmax_trajectory.append(ave_pmax)
        pmax_per_image_history.append(combined_max_posterior.copy())

        # --- Track per-image best assignments for convergence detection ---
        # Combine both half-sets' assignments into a single array for
        # update_refinement_state.  Use coarse_ha (indexed into
        # effective_rotations) for consistent convergence tracking.
        current_combined_ha = np.concatenate(
            [np.asarray(ha, dtype=np.int32) for ha in coarse_ha],
            axis=0,
        )
        if all(ha is not None for ha in previous_assignments):
            previous_combined_ha = np.concatenate(
                [np.asarray(ha, dtype=np.int32) for ha in previous_assignments],
                axis=0,
            )
        else:
            previous_combined_ha = None

        # tau2 was already updated BEFORE the Wiener solve (matching RELION's
        # reconstruct() which calls updateSSNRarrays before the filter).

        # --- Resolution from updated FSC-derived SSNR (RELION auto-refine) ---
        # Matches RELION updateSSNRarrays at backprojector.cpp:1117-1123:
        # data_vs_prior[i] = tau2_fudge * fsc / (1 - fsc), with fsc clamped
        # to [0.001, 0.999] inside fsc_to_relion_ssnr.
        dvp_iter = np.asarray(fsc, dtype=np.float32).copy()
        if cs < grid_size:
            dvp_iter[min(len(dvp_iter), cs // 2) :] = 0.0
        dvp_iter = np.asarray(
            fsc_to_relion_ssnr(dvp_iter, tau2_fudge=tau2_fudge),
        )
        dvp_res_shell = resolution_from_data_vs_prior(
            dvp_iter,
            allow_high_res_recovery=True,
        )
        pixel_res = float(dvp_res_shell)
        pixel_resolutions.append(pixel_res)

        # --- Update poses and noise ---
        # Snapshot the iter K-1 best rotations / translations BEFORE the
        # loop overwrites them, so update_refinement_state below can compute
        # the RELION-exact change metrics (B3) between iter K-1 and iter K.
        prior_iter_best_rotations = [
            np.asarray(rot).copy() if rot is not None else None for rot in previous_best_rotations
        ]
        prior_iter_best_translations = [
            np.asarray(trans).copy() if trans is not None else None
            for trans in relion_half_inputs.previous_best_translations
        ]
        new_iter_best_rotations = [None, None]
        new_iter_best_rotation_eulers = [None, None]
        new_iter_best_translations = [None, None]
        for k in range(2):
            if best_pose_rotations[k] is not None:
                best_rots = np.asarray(best_pose_rotations[k], dtype=np.float32)
                best_eulers = (
                    np.asarray(best_pose_rotation_eulers[k], dtype=np.float32)
                    if best_pose_rotation_eulers[k] is not None
                    else utils.R_to_relion(best_rots, degrees=True).astype(np.float32)
                )
                best_trans = np.asarray(best_pose_translations[k], dtype=np.float32)
            elif use_local:
                rot_idx = hard_assignments[k] // current_translations.shape[0]
                trans_idx = hard_assignments[k] % current_translations.shape[0]
                if local_search_rotations is None:
                    local_grid_metadata = build_local_search_grid_metadata(local_search_order)
                    best_rots = _selected_rotation_matrices(
                        rot_idx,
                        None,
                        local_grid_metadata,
                        random_perturbation=local_search_random_perturbation,
                        angular_sampling_deg=local_search_angular_sampling_deg,
                    )
                    best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
                else:
                    best_rots = np.asarray(local_search_rotations, dtype=np.float32)[rot_idx]
                    if local_search_rotation_eulers is not None:
                        best_eulers = np.asarray(local_search_rotation_eulers, dtype=np.float32)[rot_idx]
                    else:
                        best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
                best_trans = np.asarray(current_translations)[trans_idx]
            else:
                # Global search uses the dense grid in pose_rotations[k].
                rot_idx = hard_assignments[k] // current_translations.shape[0]
                best_rots, best_trans = hard_assignment_idx_to_pose(
                    hard_assignments[k],
                    pose_rotations[k],
                    pose_translations[k],
                )
                if pose_rotation_eulers[k] is not None:
                    best_eulers = np.asarray(pose_rotation_eulers[k], dtype=np.float32)[rot_idx]
                else:
                    best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
            new_iter_best_rotations[k] = np.asarray(best_rots, dtype=np.float32)
            new_iter_best_rotation_eulers[k] = np.asarray(best_eulers, dtype=np.float32)
            # When image_pre_shifts is used, best_trans is relative to the
            # rounded pre-shift base. Store the total (absolute) translation
            # so the next iteration pre-centers by the updated offset.
            total_trans = np.asarray(best_trans, dtype=np.float32)
            if translation_search_bases[k] is not None:
                total_trans = total_trans + translation_search_bases[k]
            new_iter_best_translations[k] = total_trans
            previous_best_rotations[k] = new_iter_best_rotations[k]
            relion_half_inputs.previous_best_rotation_eulers[k] = new_iter_best_rotation_eulers[k]
            relion_half_inputs.previous_best_translations[k] = new_iter_best_translations[k]
            experiment_datasets[k].update_poses(best_rots, total_trans)

        try:
            best_rotation_eulers_history.append(
                np.concatenate(new_iter_best_rotation_eulers, axis=0).astype(np.float32)
            )
            best_translations_history.append(np.concatenate(new_iter_best_translations, axis=0).astype(np.float32))
        except (ValueError, TypeError):
            best_rotation_eulers_history.append(None)
            best_translations_history.append(None)

        if save_intermediates_dir is not None:
            for k_half in range(2):
                np.save(
                    os.path.join(
                        save_intermediates_dir,
                        f"it{iteration:03d}_best_rotation_eulers_half{k_half + 1}.npy",
                    ),
                    np.asarray(new_iter_best_rotation_eulers[k_half], dtype=np.float32),
                )
                np.save(
                    os.path.join(
                        save_intermediates_dir,
                        f"it{iteration:03d}_best_translations_half{k_half + 1}.npy",
                    ),
                    np.asarray(new_iter_best_translations[k_half], dtype=np.float32),
                )
                if prior_iter_best_rotations[k_half] is not None:
                    np.save(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_prev_rotation_matrices_half{k_half + 1}.npy",
                        ),
                        np.asarray(prior_iter_best_rotations[k_half], dtype=np.float32),
                    )
                if prior_iter_best_translations[k_half] is not None:
                    np.save(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_prev_translations_half{k_half + 1}.npy",
                        ),
                        np.asarray(prior_iter_best_translations[k_half], dtype=np.float32),
                    )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_effective_rotations.npy"),
                np.asarray(effective_rotations, dtype=np.float32),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_effective_rotation_eulers.npy"),
                np.asarray(effective_rotation_eulers, dtype=np.float32),
            )

        # --- RELION-exact change tracking inputs (B3 / B4) ---
        # Combine both half-sets in the same image order as
        # current_combined_ha. RELION's monitorHiddenVariableChanges sums
        # over all particles, so the per-half order is irrelevant for the
        # mean -- but we keep the half-0-then-half-1 convention for
        # consistency with the rest of the loop.
        try:
            current_rotation_matrices_combined = np.concatenate(
                new_iter_best_rotations,
                axis=0,
            ).astype(np.float64)
            current_translations_pixel_combined = np.concatenate(
                new_iter_best_translations,
                axis=0,
            ).astype(np.float64)
        except (ValueError, TypeError):
            current_rotation_matrices_combined = None
            current_translations_pixel_combined = None
        if all(rot is not None for rot in prior_iter_best_rotations):
            try:
                previous_rotation_matrices_combined = np.concatenate(
                    prior_iter_best_rotations,
                    axis=0,
                ).astype(np.float64)
                previous_translations_pixel_combined = np.concatenate(
                    prior_iter_best_translations,
                    axis=0,
                ).astype(np.float64)
            except (ValueError, TypeError):
                previous_rotation_matrices_combined = None
                previous_translations_pixel_combined = None
        else:
            previous_rotation_matrices_combined = None
            previous_translations_pixel_combined = None

        # RELION-style posterior-weighted noise update. Sums the wsum/img_power
        # accumulators from both half-sets and normalizes via the M-step formula.
        if noise_stats_per_half[0] is None or noise_stats_per_half[1] is None:
            raise RuntimeError(
                "RELION mode expected per-half NoiseStats from the EM engine; "
                "ensure accumulate_noise=True is plumbed through pass 2.",
            )
        if relion_firstiter_cc_this_iter:
            noise_from_res_per_half = [
                np.asarray(noise_k, dtype=np.float64) for noise_k in previous_noise_radial_per_half
            ]
            noise_from_res = np.mean(np.stack(noise_from_res_per_half, axis=0), axis=0)
            logger.info(
                "RELION iter-1 CC emulation: keeping previous sigma2_noise (skip first-iter noise update)",
            )
        else:
            noise_from_res_per_half = []
            for k_noise, stats_k in enumerate(noise_stats_per_half):
                noise_k = noise.normalize_wsum_to_sigma2_noise(
                    np.asarray(stats_k.wsum_sigma2_noise, dtype=np.float64),
                    np.asarray(stats_k.wsum_img_power, dtype=np.float64),
                    stats_k.sumw,
                    cryo.image_shape,
                )
                noise_from_res_per_half.append(np.asarray(noise_k, dtype=np.float64))
                noise_variance_per_half[k_noise] = jnp.asarray(
                    noise.make_radial_noise(noise_k, cryo.image_shape),
                ).reshape(-1)
            noise_from_res = np.mean(np.stack(noise_from_res_per_half, axis=0), axis=0)

            # Log per-shell noise comparison (first 10 shells) for convergence diagnostics
            old_noise_radial = previous_noise_radial
            n_log = min(10, len(noise_from_res), len(old_noise_radial))
            logger.info(
                "Noise update per shell (first %d): old=[%s] new=[%s]",
                n_log,
                ", ".join(f"{float(x):.3e}" for x in old_noise_radial[:n_log]),
                ", ".join(f"{float(x):.3e}" for x in noise_from_res[:n_log]),
            )
            _maybe_dump_noise_update_debug(
                iteration=iteration,
                current_size=cs,
                image_shape=cryo.image_shape,
                noise_stats_per_half=noise_stats_per_half,
                previous_noise_radial_per_half=previous_noise_radial_per_half,
                noise_from_res_per_half=noise_from_res_per_half,
                noise_from_res=noise_from_res,
            )

            previous_noise_radial_per_half = noise_from_res_per_half
            previous_noise_radial = jnp.asarray(noise_from_res, dtype=jnp.float32)
            noise_variance = _mean_noise_variance(noise_variance_per_half)
            _parity_dump.mark_stage(iteration, "noise_update")

        # Save per-iter per-shell sigma2 (after this iter's noise update) and
        # the exact shell-wise tau2 ingredients used in the Wiener update.
        noise_radial_trajectory.append(np.asarray(noise_from_res, dtype=np.float64))
        noise_radial_per_half_trajectory.append(
            np.stack([np.asarray(noise_k, dtype=np.float64) for noise_k in noise_from_res_per_half], axis=0),
        )
        if tau2_update_details is not None:
            tau2_radial_trajectory.append(np.asarray(tau2_update_details["prior_shells"], dtype=np.float64))
            tau2_sigma2_trajectory.append(np.asarray(tau2_update_details["sigma2_shells"], dtype=np.float64))
            tau2_avg_weight_trajectory.append(np.asarray(tau2_update_details["avg_weight_shells"], dtype=np.float64))
            tau2_shell_sum_trajectory.append(np.asarray(tau2_update_details["shell_sum"], dtype=np.float64))
            tau2_shell_count_trajectory.append(np.asarray(tau2_update_details["shell_count"], dtype=np.float64))
            tau2_fsc_used_trajectory.append(np.asarray(tau2_update_details["fsc_shells"], dtype=np.float64))
            tau2_ssnr_trajectory.append(np.asarray(tau2_update_details["ssnr_shells"], dtype=np.float64))
        else:
            tau2_radial_trajectory.append(None)
            tau2_sigma2_trajectory.append(None)
            tau2_avg_weight_trajectory.append(None)
            tau2_shell_sum_trajectory.append(None)
            tau2_shell_count_trajectory.append(None)
            tau2_fsc_used_trajectory.append(None)
            tau2_ssnr_trajectory.append(None)

        # --- Update convergence state ---
        # This checks assignment changes, resolution stalls, and may trigger
        # angular step refinement or convergence.
        n_rot_current = rotation_grid_size(local_search_order) if use_local else effective_rotations.shape[0]
        n_trans_current = current_translations.shape[0]

        # ``update_refinement_state`` expects ``new_resolution`` in
        # Angstroms (lower = better resolution), matching RELION's
        # ``mymodel.current_resolution``.  Convert from the shell index
        # ``pixel_res`` to Å here so the resol_gain stall detection
        # compares apples to apples (not shell-vs-shell with the wrong
        # sign).
        new_res_angstrom = shell_index_to_resolution_angstrom(
            pixel_res,
            cryo.image_shape[0],
            cryo.voxel_size,
        )

        # RELION's calculateExpectedAngularErrors (ml_optimiser.cpp:9534)
        iter_acc_rot = None
        iter_acc_trans = None
        if iter_sig_counts is not None and len(iter_sig_counts) > 0:
            iter_acc_rot, _ = calculate_expected_angular_errors(
                state.healpix_order,
                iter_sig_counts,
                n_translations=n_trans_current,
            )
            logger.info(
                "acc_rot=%.3f deg (from %d images, mean n_sig=%.1f)",
                iter_acc_rot,
                len(iter_sig_counts),
                float(np.mean(iter_sig_counts)),
            )

        if perturb_replay_relion_dir is not None:
            _optimiser_iter = int(init_relion_iteration) + iteration + 1
            _optimiser_star = os.path.join(
                perturb_replay_relion_dir,
                f"run_it{_optimiser_iter:03d}_optimiser.star",
            )
            if os.path.exists(_optimiser_star):
                try:
                    _optimiser_meta = read_relion_optimiser_metadata(_optimiser_star)
                    _relion_acc_rot = _optimiser_meta.get("overall_accuracy_rotations")
                    _relion_acc_trans_angst = _optimiser_meta.get("overall_accuracy_translations_angst")
                    if _relion_acc_rot is not None and np.isfinite(float(_relion_acc_rot)):
                        iter_acc_rot = float(_relion_acc_rot)
                    if _relion_acc_trans_angst is not None and np.isfinite(float(_relion_acc_trans_angst)):
                        _px = float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0)
                        iter_acc_trans = float(_relion_acc_trans_angst) / _px
                    logger.info(
                        "Replay override: optimiser accuracy <- %s (acc_rot=%.3f deg, acc_trans=%s px)",
                        _optimiser_star,
                        float(iter_acc_rot) if iter_acc_rot is not None else float("nan"),
                        f"{iter_acc_trans:.3f}" if iter_acc_trans is not None else "unset",
                    )
                except Exception as exc:
                    logger.warning("Replay override: failed to read optimiser metadata from %s: %s", _optimiser_star, exc)

        state = update_refinement_state(
            state,
            current_assignments=current_combined_ha,
            previous_assignments=previous_combined_ha,
            n_rotations=n_rot_current,
            n_translations=n_trans_current,
            translations=np.asarray(current_translations),
            new_resolution=new_res_angstrom,
            max_posterior_per_image=combined_max_posterior,
            acc_rot=iter_acc_rot,
            acc_trans=iter_acc_trans,
            current_rotation_matrices=current_rotation_matrices_combined,
            previous_rotation_matrices=previous_rotation_matrices_combined,
            current_translations_pixel=current_translations_pixel_combined,
            previous_translations_pixel=previous_translations_pixel_combined,
            voxel_size_angstrom=float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
        )

        # Track frac_changed for local search fallback
        from recovar.em.dense_single_volume.helpers.convergence import compute_assignment_changes

        frac_changed = compute_assignment_changes(
            current_combined_ha,
            previous_combined_ha,
            n_rot_current,
            n_trans_current,
            current_healpix_order,
        )
        state._last_frac_changed = frac_changed
        frac_changed_trajectory.append(float(frac_changed))

        # --- C1 (RELION-parity): update sigma2_offset from data ---
        # Prefer RELION's posterior-weighted sufficient statistic:
        #   sigma2_offset_new = wsum_sigma2_offset / (2 * sum_weight)
        # for 2D single-particle data. Fall back to the older hard-assignment
        # proxy only when a path does not propagate the full posterior moment.
        sigma2_offset_wsum = 0.0
        sigma2_offset_sumw = 0.0
        for stats_k in noise_stats_per_half:
            if stats_k is None:
                continue
            sigma2_offset_wsum += float(getattr(stats_k, "wsum_sigma2_offset", 0.0))
            sigma2_offset_sumw += float(getattr(stats_k, "sumw", 0.0))
        if sigma2_offset_wsum > 0.0 and sigma2_offset_sumw > 0.0:
            min_sigma2_angstrom2 = 2.0
            sigma2_offset_angstrom2 = max(
                sigma2_offset_wsum / (2.0 * sigma2_offset_sumw),
                min_sigma2_angstrom2,
            )
            current_sigma_offset_angstrom = float(np.sqrt(sigma2_offset_angstrom2))
            logger.info(
                "C1: sigma_offset updated %.3f Å from posterior variance (clamp sigma^2 >= %.3f Å^2)",
                current_sigma_offset_angstrom,
                min_sigma2_angstrom2,
            )
        else:
            new_sigma_offset_angstrom = state.current_changes_optimal_offsets_angstrom
            if np.isfinite(new_sigma_offset_angstrom) and new_sigma_offset_angstrom > 0:
                min_sigma_angstrom = float(np.sqrt(2.0))  # RELION min_sigma2_offset = 2 Å²
                current_sigma_offset_angstrom = max(
                    float(new_sigma_offset_angstrom),
                    min_sigma_angstrom,
                )
                logger.info(
                    "C1 fallback: sigma_offset updated %.3f Å from hard assignments (clamp >= %.3f Å)",
                    current_sigma_offset_angstrom,
                    min_sigma_angstrom,
                )
        sigma_offset_trajectory.append(float(current_sigma_offset_angstrom))
        acc_rot_trajectory.append(float(iter_acc_rot) if iter_acc_rot is not None else np.nan)
        smallest_change_angles_trajectory.append(float(state.current_changes_optimal_orientations))
        smallest_change_offsets_trajectory.append(float(state.current_changes_optimal_offsets_angstrom))

        if _parity_dump.is_active():
            try:
                _parity_dump.dump_iteration(
                    iteration=iteration,
                    init_relion_iteration=int(init_relion_iteration),
                    current_size=int(cs),
                    sigma_offset=float(current_sigma_offset_angstrom),
                    translation_step=float(state.translation_step),
                    translation_range=float(state.translation_range),
                    random_perturbation=float(random_perturbation) if random_perturbation is not None else 0.0,
                    random_perturbation_instance=int(state.perturbation_instance)
                    if hasattr(state, "perturbation_instance")
                    else 0,
                    tau2_fudge=float(tau2_fudge),
                    voxel_size=float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
                    grid_size=int(grid_size),
                    volume_shape=tuple(volume_shape),
                    ave_pmax=float(ave_pmax),
                    fsc=np.asarray(fsc, dtype=np.float64),
                    sigma2_noise=np.asarray(noise_variance, dtype=np.float64)
                    if "noise_variance" in dir()
                    else np.zeros(0),
                    means=means,
                    unreg_means=unreg_means,
                    new_iter_best_rotation_eulers=new_iter_best_rotation_eulers,
                    new_iter_best_translations=new_iter_best_translations,
                )
            except Exception as exc:
                logger.warning("parity_dump.dump_iteration failed at iter %d: %s", iteration, exc)

        # Save assignments for next iteration's change tracking.
        # Use coarse_ha (indexed into effective_rotations/current_rotations)
        # so that local search and convergence detection work correctly
        # regardless of whether adaptive oversampling was used.
        previous_assignments = [ha.copy() if ha is not None else None for ha in coarse_ha]
        _parity_dump.mark_stage(iteration, "convergence")

        # --- Timing ---
        elapsed = time.time() - t0
        wall_times.append(elapsed)

        res_angstrom = shell_index_to_resolution_angstrom(
            pixel_res,
            cryo.image_shape[0],
            cryo.voxel_size,
        )
        logger.info(
            "RELION Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, ave_Pmax=%.4f, healpix_order=%d, "
            "converged=%s, time=%.1fs",
            iteration + 1,
            cs,
            pixel_res,
            res_angstrom,
            ave_pmax,
            state.healpix_order,
            state.has_converged,
            elapsed,
        )

        if state.has_converged and not force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d. Final resolution: %.2f A (pixel_res=%.1f)",
                iteration + 1,
                res_angstrom,
                pixel_res,
            )
            break
        if state.has_converged and force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d, continuing because "
                "force_max_iter_after_convergence=True",
                iteration + 1,
            )

        iteration += 1

    # RELION's final all-data iteration is a real next iteration after
    # convergence flags are set at the top of the loop. Do not synthesize it
    # after plain max_iter exhaustion, and do not synthesize it when
    # convergence is first detected on the last allowed iteration.
    should_run_final_iteration = bool(
        state.has_converged
        and not force_max_iter_after_convergence
        and (iteration + 1) < max_iter
    )
    if skip_final_iteration or not should_run_final_iteration:
        if not skip_final_iteration and not should_run_final_iteration:
            logger.info(
                "Skipping RELION final all-data iteration: has_converged=%s, "
                "iteration=%d, max_iter=%d, force_max_iter_after_convergence=%s",
                state.has_converged,
                iteration,
                max_iter,
                force_max_iter_after_convergence,
            )
        merged_mean = (means[0] + means[1]) / 2
        return {
            "mean": merged_mean,
            "means": means,
            "fsc": fsc_history[-1] if fsc_history else None,
            "hard_assignments": hard_assignments,
            "current_sizes": current_sizes,
            "fsc_history": fsc_history,
            "pixel_resolutions": pixel_resolutions,
            "wall_times": wall_times,
            "significant_counts": significant_counts,
            "convergence_state": state,
            "data_vs_prior_trajectory": data_vs_prior_trajectory,
            "healpix_order_trajectory": healpix_order_trajectory,
            "ave_Pmax_trajectory": ave_Pmax_trajectory,
            "pmax_per_image_history": pmax_per_image_history,
            "noise_radial_trajectory": noise_radial_trajectory,
            "noise_radial_per_half_trajectory": noise_radial_per_half_trajectory,
            "tau2_radial_trajectory": tau2_radial_trajectory,
            "tau2_sigma2_trajectory": tau2_sigma2_trajectory,
            "tau2_avg_weight_trajectory": tau2_avg_weight_trajectory,
            "tau2_shell_sum_trajectory": tau2_shell_sum_trajectory,
            "tau2_shell_count_trajectory": tau2_shell_count_trajectory,
            "tau2_fsc_used_trajectory": tau2_fsc_used_trajectory,
            "tau2_ssnr_trajectory": tau2_ssnr_trajectory,
            "sigma_offset_used_trajectory": sigma_offset_used_trajectory,
            "sigma_offset_trajectory": sigma_offset_trajectory,
            "frac_changed_trajectory": frac_changed_trajectory,
            "acc_rot_trajectory": acc_rot_trajectory,
            "smallest_change_angles_trajectory": smallest_change_angles_trajectory,
            "smallest_change_offsets_trajectory": smallest_change_offsets_trajectory,
            "best_rotation_eulers_history": best_rotation_eulers_history,
            "best_translations_history": best_translations_history,
            "local_profile_history": local_profile_history,
        }

    # --- RELION's final iteration: do_join_random_halves + do_use_all_data ---
    # After convergence, RELION runs ONE more iter with:
    #   - current_size = ori_size (Nyquist, all shells)
    #   - joined weighted sums for reconstruction
    #   - each half still scored against its own half-map
    # See ml_optimiser.cpp:10157-10160 (sets do_join_random_halves and
    # do_use_all_data) and ml_optimiser.cpp:5707-5708 (forces current_size to
    # ori_size when do_use_all_data is true).
    #
    # Implementation: run one more E+M at full Nyquist for each half, using
    # that half's own reference map, then join the weighted sums into one final
    # reconstruction.
    final_join_means = [means[0], means[1]]
    final_iter_t0 = time.time()
    logger.info("=== RELION final all-data Nyquist iteration (do_join_random_halves=True, do_use_all_data=True) ===")
    final_cs = grid_size  # = ori_size, full Nyquist
    recon_vol_size = int(np.prod([d * PADDING_FACTOR for d in volume_shape]))
    final_ft_y = jnp.zeros(recon_vol_size, dtype=cryo.dtype)
    final_ft_ctf = jnp.zeros(recon_vol_size, dtype=cryo.dtype)
    final_noise_wsum = np.zeros_like(np.asarray(noise_radial_trajectory[-1])) if noise_radial_trajectory else None
    final_img_power = np.zeros_like(np.asarray(noise_radial_trajectory[-1])) if noise_radial_trajectory else None
    final_sumw = 0.0
    for k in range(2):
        # Pass the merged mean as input (both halves get the same projection source).
        # Run on each half-set's particles (avoids loading all particles at once),
        # then accumulate Ft_y/Ft_ctf and noise stats from BOTH halves.
        safe_ibs, safe_rbs = _safe_batch_sizes(
            current_rotations.shape[0],
            current_translations.shape[0],
        )
        _, ha_k_final, Ft_y_k_final, Ft_ctf_k_final, _, noise_stats_k_final = run_em(
            experiment_datasets[k],
            final_join_means[k],
            mean_variance,
            noise_variance_per_half[k],
            current_rotations,
            current_translations,
            disc_type,
            image_batch_size=safe_ibs,
            rotation_block_size=safe_rbs,
            current_size=final_cs,  # full Nyquist
            score_with_masked_images=True,
            return_stats=True,
            accumulate_noise=True,
            half_spectrum_scoring=True,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            reconstruction_padding_factor=PADDING_FACTOR,
            image_corrections=relion_half_inputs.image_corrections[k],
            scale_corrections=relion_half_inputs.scale_corrections[k],
            image_pre_shifts=relion_translation_search_base(relion_half_inputs.previous_best_translations[k]),
            use_float64_scoring=_relion_use_float64_scoring(),
            use_float64_projections=False,
            do_gridding_correction=True,
            square_window=RELION_FOURIER_WINDOW_SQUARE,
            sparse_pass2=False,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
        )
        # --- Manifest dump for final all-data iteration (Phase 0.1) ---
        if save_intermediates_dir is not None:
            _manifest_path = os.path.join(
                save_intermediates_dir,
                f"manifest_final_half{k}.npz",
            )
            _manifest = {
                "effective_rotations": np.asarray(current_rotations, dtype=np.float32),
                "current_translations": np.asarray(current_translations, dtype=np.float32),
                "rotation_log_prior": np.array([]),
                "translation_log_prior": np.array([]),
                "image_corrections": np.asarray(relion_half_inputs.image_corrections[k], dtype=np.float64)
                if relion_half_inputs.image_corrections[k] is not None
                else np.array([]),
                "scale_corrections": np.asarray(relion_half_inputs.scale_corrections[k], dtype=np.float64)
                if relion_half_inputs.scale_corrections[k] is not None
                else np.array([]),
                "image_pre_shifts": np.asarray(
                    relion_translation_search_base(relion_half_inputs.previous_best_translations[k]), dtype=np.float32
                )
                if relion_half_inputs.previous_best_translations[k] is not None
                else np.array([]),
                "absolute_previous_translations": np.asarray(
                    relion_half_inputs.previous_best_translations[k],
                    dtype=np.float32,
                )
                if relion_half_inputs.previous_best_translations[k] is not None
                else np.array([]),
                "mean_vol_ft": np.asarray(final_join_means[k]),
                "mean_variance": np.asarray(mean_variance),
                "noise_variance": np.asarray(noise_variance_per_half[k]),
                "current_size": np.int32(final_cs),
                "half_spectrum_scoring": np.bool_(True),
                "use_float64_scoring": np.bool_(_relion_use_float64_scoring()),
                "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                "reconstruction_padding_factor": np.int32(PADDING_FACTOR),
                "score_with_masked_images": np.bool_(True),
                "perturbation_instance": np.float64(random_perturbation),
                "perturbation_factor": np.float64(perturb_factor),
                "iteration": np.int32(-1),
                "half_index": np.int32(k),
            }
            np.savez(_manifest_path, **_manifest)
            logger.info("Final manifest dumped: %s", _manifest_path)

        final_ft_y = final_ft_y + Ft_y_k_final
        final_ft_ctf = final_ft_ctf + Ft_ctf_k_final
        if noise_stats_k_final is not None and final_noise_wsum is not None:
            final_noise_wsum += np.asarray(noise_stats_k_final.wsum_sigma2_noise, dtype=np.float64)
            final_img_power += np.asarray(noise_stats_k_final.wsum_img_power, dtype=np.float64)
            final_sumw += float(noise_stats_k_final.sumw)

    # Reconstruct the final volume from the COMBINED Ft_y/Ft_ctf accumulators
    # at the full Nyquist resolution. Skip the join_halves step (we're already
    # combining the two halves into one dataset for this final iter).
    merged_mean = _reconstruct_volume_eager(
        final_ft_ctf,
        final_ft_y,
        volume_shape,
        PADDING_FACTOR,
        tau=mean_variance,
        tau2_fudge=tau2_fudge,
        projection_padding_factor=PROJECTION_PADDING_FACTOR,
        minres_map=RELION_MINRES_MAP,
    ).reshape(-1)
    final_iter_elapsed = time.time() - final_iter_t0
    logger.info(
        "Final iter complete: current_size=%d (Nyquist), wall=%.1fs",
        final_cs,
        final_iter_elapsed,
    )
    wall_times.append(final_iter_elapsed)

    return {
        "mean": merged_mean,
        "means": means,
        "fsc": fsc_history[-1] if fsc_history else None,
        "hard_assignments": hard_assignments,
        "current_sizes": current_sizes,
        "fsc_history": fsc_history,
        "pixel_resolutions": pixel_resolutions,
        "wall_times": wall_times,
        "significant_counts": significant_counts,
        # RELION-mode specific outputs
        "convergence_state": state,
        "data_vs_prior_trajectory": data_vs_prior_trajectory,
        "healpix_order_trajectory": healpix_order_trajectory,
        "ave_Pmax_trajectory": ave_Pmax_trajectory,
        "pmax_per_image_history": pmax_per_image_history,
        "noise_radial_trajectory": noise_radial_trajectory,
        "noise_radial_per_half_trajectory": noise_radial_per_half_trajectory,
        "tau2_radial_trajectory": tau2_radial_trajectory,
        "tau2_sigma2_trajectory": tau2_sigma2_trajectory,
        "tau2_avg_weight_trajectory": tau2_avg_weight_trajectory,
        "tau2_shell_sum_trajectory": tau2_shell_sum_trajectory,
        "tau2_shell_count_trajectory": tau2_shell_count_trajectory,
        "tau2_fsc_used_trajectory": tau2_fsc_used_trajectory,
        "tau2_ssnr_trajectory": tau2_ssnr_trajectory,
        "sigma_offset_used_trajectory": sigma_offset_used_trajectory,
        "sigma_offset_trajectory": sigma_offset_trajectory,
        "frac_changed_trajectory": frac_changed_trajectory,
        "acc_rot_trajectory": acc_rot_trajectory,
        "smallest_change_angles_trajectory": smallest_change_angles_trajectory,
        "smallest_change_offsets_trajectory": smallest_change_offsets_trajectory,
        "best_rotation_eulers_history": best_rotation_eulers_history,
        "best_translations_history": best_translations_history,
        "local_profile_history": local_profile_history,
    }
