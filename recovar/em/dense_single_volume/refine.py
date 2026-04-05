"""FSC-driven multi-iteration refinement loop for dense single-volume EM.

Wires FSC -> current_size -> Fourier window into the iteration loop,
implementing Phases 4 and 5 of the RELION-parity plan.

The loop:
1. Compute FSC between half-maps -> determine current_size
2. Quantize current_size to allowed values
3. Run engine_v2 E+M on each half-set at that current_size
4. Optionally: two-pass adaptive oversampling (Phase 5):
   - Pass 1 (coarse): dense E-step at coarse resolution, find significant
     (rotation, translation) pairs per image.
   - Pass 2 (fine): oversampled E+M at finer resolution for significant
     rotations only.
5. Wiener-solve each half-map
6. Estimate noise, update prior
7. Log progress

Supports oracle mode: inject RELION's per-iteration current_sizes to
isolate windowing from the statistical model.

See docs/math/plan_relion_parity.md, Phases 4 and 5.
"""

import logging
import time

import jax.numpy as jnp
import numpy as np
import scipy.special

from recovar.em.core import hard_assignment_idx_to_pose
from recovar.em.dense_single_volume.engine_v2 import run_em_v2
from recovar.em.dense_single_volume.types import RelionStats
from recovar.em.dense_single_volume.fourier_window import quantize_current_size
from recovar.em.dense_single_volume.adaptive import (
    find_significant_rotations,
    compute_pass2_stats,
    compute_pass2_stats_sparse,
)

# RELION-parity building blocks (used only by mode="relion")
from recovar.em.dense_single_volume.convergence import (
    RefinementState,
    update_refinement_state,
    check_convergence,
    should_refine_angular_sampling,
    refine_angular_sampling,
    compute_ave_Pmax,
    healpix_angular_step,
)
from recovar.em.sampling import (
    get_rotation_grid_at_order,
    get_local_rotation_grid,
    get_local_rotation_grid_fast,
    get_oversampled_translation_grid,
    get_translation_grid,
    rotation_grid_n_in_planes,
    rotation_grid_size,
    rotation_indices_to_matrices,
)
from recovar.reconstruction.regularization import (
    resolution_from_data_vs_prior,
    compute_current_size_relion,
    fsc_to_relion_ssnr,
    update_relion_growth_state_from_fsc,
)

logger = logging.getLogger(__name__)

ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION = 0.5


def shell_index_to_resolution_angstrom(shell_index, ori_size, voxel_size):
    """Convert a Fourier shell index into a real-space resolution in Angstrom."""
    if voxel_size <= 0:
        return float(shell_index)
    shell_index = float(shell_index)
    if shell_index <= 0:
        return float("inf")
    return float(ori_size) * float(voxel_size) / shell_index


# ---------------------------------------------------------------------------
# Coarse image size for adaptive oversampling (RELION parity)
# ---------------------------------------------------------------------------

def compute_coarse_image_size(
    angular_step_deg, pixel_size, ori_size, particle_diameter=None,
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


def _run_grouped_local_search_em(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    prior_rotations,
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
):
    """Run batched exact local search on the fine HEALPix grid.

    Each image carries its own exact prior rotation from the previous
    iteration. Images are processed in chunks; each chunk evaluates the union
    of its local neighborhoods with an image-specific rotation prior.
    """
    prior_rotations = np.asarray(prior_rotations, dtype=np.float32).reshape(-1, 3, 3)
    if prior_translations is None:
        prior_translations = np.zeros(
            (prior_rotations.shape[0], np.asarray(translations).shape[1]),
            dtype=np.float32,
        )
    else:
        prior_translations = np.asarray(prior_translations, dtype=np.float32).reshape(
            -1, np.asarray(translations).shape[1],
        )
    n_images = experiment_dataset.n_units
    n_trans = int(np.asarray(translations).shape[0])
    active_offset_range = float(offset_range_pixels) if offset_range_pixels is not None else float(
        np.max(np.linalg.norm(np.asarray(translations, dtype=np.float32), axis=1))
    )
    volume_size = experiment_dataset.volume_size

    Ft_y_total = jnp.zeros(volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf_total = jnp.zeros(volume_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    log_evidence = np.empty(n_images, dtype=np.float32)
    best_log_score = np.empty(n_images, dtype=np.float32)
    max_posterior = np.empty(n_images, dtype=np.float32)
    rotation_posterior_sums = np.zeros(
        rotation_grid_size(healpix_order), dtype=np.float64,
    )

    total_local_rotations = 0
    max_local_rotations = 0
    chunk_sizes = []
    n_chunks = 0

    chunk_size = max(1, min(image_batch_size, 64))
    for chunk_start in range(0, n_images, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, n_images)
        group_image_indices = np.arange(chunk_start, chunk_stop, dtype=np.int64)
        n_chunks += 1
        chunk_sizes.append(len(group_image_indices))

        local_indices, local_log_prior = get_local_rotation_grid_fast(
            prior_rotations[group_image_indices],
            sigma_rot,
            sigma_psi,
            healpix_order,
            sigma_cutoff=3.0,
            per_image=True,
        )
        local_rotations = rotation_indices_to_matrices(local_indices, healpix_order)
        local_translation_log_prior = make_relion_translation_log_prior(
            translations,
            experiment_dataset.voxel_size,
            sigma_offset_angstrom,
            prior_translations[group_image_indices],
            offset_range_pixels=active_offset_range,
        )

        total_local_rotations += int(local_rotations.shape[0])
        max_local_rotations = max(max_local_rotations, int(local_rotations.shape[0]))

        _, ha_local, Ft_y_g, Ft_ctf_g, stats_g = run_em_v2(
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            local_rotations,
            translations,
            disc_type,
            image_batch_size=image_batch_size,
            rotation_block_size=min(rotation_block_size, max(1, local_rotations.shape[0])),
            current_size=current_size,
            rotation_log_prior=local_log_prior,
            translation_log_prior=local_translation_log_prior,
            image_indices=group_image_indices,
            score_with_masked_images=True,
            return_stats=True,
        )

        Ft_y_total = Ft_y_total + Ft_y_g
        Ft_ctf_total = Ft_ctf_total + Ft_ctf_g

        local_rot_idx = ha_local // n_trans
        trans_idx = ha_local % n_trans
        hard_assignment[group_image_indices] = (
            local_indices[local_rot_idx] * n_trans + trans_idx
        ).astype(np.int32)
        log_evidence[group_image_indices] = np.asarray(
            stats_g.log_evidence_per_image, dtype=np.float32,
        )
        best_log_score[group_image_indices] = np.asarray(
            stats_g.best_log_score_per_image, dtype=np.float32,
        )
        max_posterior[group_image_indices] = np.asarray(
            stats_g.max_posterior_per_image, dtype=np.float32,
        )
        rotation_posterior_sums[local_indices] += np.asarray(
            stats_g.rotation_posterior_sums, dtype=np.float64,
        )

    logger.info(
        "Batched local search: %d chunks, median chunk size=%d, "
        "mean local rotations=%.1f, max local rotations=%d",
        n_chunks,
        int(np.median(chunk_sizes)) if chunk_sizes else 0,
        float(total_local_rotations / max(n_chunks, 1)),
        max_local_rotations,
    )

    relion_stats = RelionStats(
        log_evidence_per_image=jnp.asarray(log_evidence),
        best_log_score_per_image=jnp.asarray(best_log_score),
        max_posterior_per_image=jnp.asarray(max_posterior),
        rotation_posterior_sums=jnp.asarray(rotation_posterior_sums, dtype=jnp.float32),
    )
    return Ft_y_total, Ft_ctf_total, hard_assignment, relion_stats


# ---------------------------------------------------------------------------
# Batched significance pruning (avoids materializing full weight matrix)
# ---------------------------------------------------------------------------

def _compute_significance_batched(
    experiment_dataset,
    mean,
    noise_variance,
    rotations,
    translations,
    disc_type,
    adaptive_fraction,
    max_significants,
    image_batch_size,
    rotation_block_size,
    current_size,
    *,
    score_with_masked_images=False,
    return_significant_sample_indices=False,
    rotation_log_prior=None,
    translation_log_prior=None,
    half_spectrum_scoring=False,
):
    """Run coarse E-step and find significant rotations in a memory-efficient way.

    Instead of materializing the full (n_images, n_rot * n_trans) weight matrix,
    this processes one image batch at a time: for each batch, it computes the
    posterior weights, finds significance, and accumulates the union of significant
    rotation indices.

    Parameters
    ----------
    Returns
    -------
    sig_rot_any : np.ndarray, shape (n_rot,), dtype bool
        True for rotations that are significant for at least one image.
    n_sig_all : np.ndarray, shape (n_images,), dtype int32
        Per-image count of significant (rot x trans) samples.
    hard_assignments : np.ndarray, shape (n_images,), dtype int32
        Best (rot_idx * n_trans + trans_idx) per image from coarse pass.
    significant_sample_indices : list[np.ndarray], optional
        Returned only when ``return_significant_sample_indices=True``.
        ``significant_sample_indices[i]`` stores flattened
        ``rot_idx * n_trans + trans_idx`` entries kept for image ``i``.
    """
    from recovar.em.dense_single_volume.engine_v2 import (
        _preprocess_batch, _compute_projections_block,
        _e_step_block_scores, _e_step_block_scores_windowed,
        _update_logsumexp, make_half_image_weights,
    )
    from recovar.core.configs import ForwardModelConfig
    import recovar.core.fourier_transform_utils as fourier_transform_utils
    from recovar.em.dense_single_volume.adaptive import find_significant_rotations as _find_sig

    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    half_weights = jnp.ones(n_half, dtype=jnp.float32) if half_spectrum_scoring else make_half_image_weights(image_shape)

    use_window = current_size is not None and current_size < image_shape[0]
    if use_window:
        from .fourier_window import make_fourier_window_indices_np
        window_indices_np, n_windowed = make_fourier_window_indices_np(image_shape, current_size)
        window_indices = jnp.asarray(window_indices_np)
        half_weights_windowed = half_weights[window_indices]
    else:
        window_indices = None
        n_windowed = n_half

    # Pad rotations
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([
            rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))
        ], axis=0)
    else:
        rotations_padded = rotations

    # Accumulate results
    sig_rot_any = np.zeros(n_rot, dtype=bool)
    n_sig_all = np.empty(n_images, dtype=np.int32)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    significant_sample_indices = (
        [None] * n_images if return_significant_sample_indices else None
    )

    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_trans},), got {translation_log_prior.shape}",
                )
        elif translation_log_prior.ndim == 2:
            if translation_log_prior.shape != (n_images, n_trans):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_images}, {n_trans}) when image-specific, got "
                    f"{translation_log_prior.shape}",
                )
        else:
            raise ValueError(
                "translation_log_prior must be 1D or 2D, got "
                f"{translation_log_prior.ndim} dimensions",
            )

    if rotation_log_prior is not None:
        rotation_log_prior = np.asarray(rotation_log_prior, dtype=np.float32)
        if rotation_log_prior.shape != (n_rot,):
            raise ValueError(
                "rotation_log_prior must have shape "
                f"({n_rot},), got {rotation_log_prior.shape}",
            )
        if n_rot_padded > n_rot:
            rotation_log_prior_padded = np.concatenate([
                rotation_log_prior,
                np.zeros(n_rot_padded - n_rot, dtype=np.float32),
            ])
        else:
            rotation_log_prior_padded = rotation_log_prior
    else:
        rotation_log_prior_padded = None

    image_indices = np.arange(n_images)
    start_idx = 0

    for (batch_data, _, _, ctf_params, _, _, indices) in experiment_dataset.iter_batches(
        image_batch_size, indices=image_indices, by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        batch_data = jnp.asarray(batch_data)
        if translation_log_prior is None:
            batch_translation_log_prior = None
        elif translation_log_prior.ndim == 1:
            batch_translation_log_prior = jnp.asarray(translation_log_prior)
        else:
            batch_translation_log_prior = jnp.asarray(
                translation_log_prior[start_idx:end_idx],
            )

        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data, ctf_params, noise_variance, translations, config,
            batch_size, n_trans, score_with_masked_images,
        )

        # DC exclusion (RELION parity: Minvsigma2[0] = 0)
        if half_spectrum_scoring:
            from recovar.em.dense_single_volume.engine_v2 import make_shell_indices_half as _mshi
            dc_shell = _mshi(image_shape)
            dc_mask = (dc_shell == 0)
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

        if use_window:
            shifted_data = shifted_half[:, window_indices]
            ctf2_data = ctf2_over_nv_half[:, window_indices]
        else:
            shifted_data = shifted_half
            ctf2_data = ctf2_over_nv_half

        # Pass 1: streaming logsumexp
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            if use_window:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
                scores = _e_step_block_scores_windowed(
                    shifted_data, batch_norm, ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    half_weights_windowed,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
            else:
                scores = _e_step_block_scores(
                    shifted_data, batch_norm, ctf2_data,
                    proj_half_b * half_weights,
                    proj_abs2_half_b * half_weights,
                    half_weights,
                    batch_size, n_trans, image_shape, volume_shape,
                )

            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            if rotation_log_prior_padded is not None:
                scores = scores + jnp.asarray(rotation_log_prior_padded[r0:r1])[None, :, None]

            if batch_translation_log_prior is not None:
                if translation_log_prior.ndim == 1:
                    scores = scores + batch_translation_log_prior[None, None, :]
                else:
                    scores = scores + batch_translation_log_prior[:, None, :]

            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)

        log_Z = max_s + jnp.log(sum_exp)

        # Pass 2: recompute scores, normalize -> batch weights
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)
        batch_weights_blocks = []

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            if use_window:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
                scores = _e_step_block_scores_windowed(
                    shifted_data, batch_norm, ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    half_weights_windowed,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
            else:
                scores = _e_step_block_scores(
                    shifted_data, batch_norm, ctf2_data,
                    proj_half_b * half_weights,
                    proj_abs2_half_b * half_weights,
                    half_weights,
                    batch_size, n_trans, image_shape, volume_shape,
                )

            if r1 > n_rot:
                valid = n_rot - r0
                pmask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(pmask[None, :, None], scores, -jnp.inf)

            if rotation_log_prior_padded is not None:
                scores = scores + jnp.asarray(rotation_log_prior_padded[r0:r1])[None, :, None]

            if batch_translation_log_prior is not None:
                if translation_log_prior.ndim == 1:
                    scores = scores + batch_translation_log_prior[None, None, :]
                else:
                    scores = scores + batch_translation_log_prior[:, None, :]

            probs = jnp.exp(scores - log_Z[:, None, None])

            block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
            block_argmax = jnp.argmax(scores.reshape(batch_size, -1), axis=1)
            improved = block_best > best_score
            best_score = jnp.where(improved, block_best, best_score)
            best_argmax = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax)

            actual_rot = min(rotation_block_size, n_rot - r0)
            block_probs = probs[:, :actual_rot, :]
            batch_weights_blocks.append(np.asarray(block_probs.reshape(batch_size, -1)))

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)

        # Concatenate this batch's weights -> (batch_size, n_rot * n_trans)
        batch_weights = np.concatenate(batch_weights_blocks, axis=1)

        # Find significance for this batch
        batch_sig_mask, batch_sig_rot_mask, batch_n_sig = _find_sig(
            jnp.asarray(batch_weights),
            n_rot, n_trans,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )

        # Accumulate global union of significant rotations
        batch_sig_rot_any = np.asarray(jnp.any(batch_sig_rot_mask, axis=0))
        sig_rot_any |= batch_sig_rot_any

        n_sig_all[start_idx:end_idx] = np.asarray(batch_n_sig)
        if return_significant_sample_indices:
            batch_sig_mask_np = np.asarray(batch_sig_mask, dtype=bool)
            for local_idx, global_idx in enumerate(indices):
                if np.all(batch_sig_mask_np[local_idx]):
                    significant_sample_indices[int(global_idx)] = None
                else:
                    significant_sample_indices[int(global_idx)] = np.flatnonzero(
                        batch_sig_mask_np[local_idx]
                    ).astype(np.int32)
        start_idx = end_idx

    if return_significant_sample_indices:
        return sig_rot_any, n_sig_all, hard_assignment, significant_sample_indices
    return sig_rot_any, n_sig_all, hard_assignment


def make_relion_translation_log_prior(
    translations,
    voxel_size,
    sigma_offset_angstrom,
    prior_centers=None,
    *,
    offset_range_pixels=None,
):
    """Return RELION-style normalized log-priors over a translation grid."""
    translations = np.asarray(translations, dtype=np.float32)
    if translations.ndim != 2:
        raise ValueError(
            f"translations must have shape (n_trans, dim), got {translations.shape}",
        )
    sigma_offset_angstrom = float(sigma_offset_angstrom)
    voxel_size = float(voxel_size if voxel_size > 0 else 1.0)
    if offset_range_pixels is not None and float(offset_range_pixels) > 0.0:
        # RELION's score path uses sigma = offset_range / 3 while an explicit
        # translational search range is active.
        sigma_offset_angstrom = float(offset_range_pixels) * voxel_size / 3.0
    n_trans = translations.shape[0]

    if prior_centers is None:
        centers = np.zeros((1, translations.shape[1]), dtype=np.float32)
        shared = True
    else:
        centers = np.asarray(prior_centers, dtype=np.float32).reshape(-1, translations.shape[1])
        shared = False

    if sigma_offset_angstrom <= 0.0:
        zeros = np.zeros((centers.shape[0], n_trans), dtype=np.float32)
        return zeros[0] if shared else zeros

    diffs_ang = (translations[None, :, :] - centers[:, None, :]) * voxel_size
    sqdist_ang = np.sum(diffs_ang ** 2, axis=-1)
    log_prior = -0.5 * sqdist_ang / (sigma_offset_angstrom ** 2)
    log_prior -= scipy.special.logsumexp(log_prior, axis=1, keepdims=True)
    log_prior += np.log(float(n_trans))
    log_prior = log_prior.astype(np.float32)
    return log_prior[0] if shared else log_prior


def collapse_rotation_posterior_to_direction_prior(rotation_posterior_sums, healpix_order):
    """Collapse per-rotation posterior mass onto RELION's HEALPix directions."""
    rotation_posterior_sums = np.asarray(rotation_posterior_sums, dtype=np.float64).reshape(-1)
    n_rot = rotation_grid_size(healpix_order)
    if rotation_posterior_sums.shape[0] != n_rot:
        raise ValueError(
            "rotation_posterior_sums must have shape "
            f"({n_rot},), got {rotation_posterior_sums.shape}",
        )

    n_pixels = n_rot // rotation_grid_n_in_planes(healpix_order)
    direction_weights = np.zeros(n_pixels, dtype=np.float64)
    np.add.at(direction_weights, np.arange(n_rot, dtype=np.int64) % n_pixels, rotation_posterior_sums)
    total = float(direction_weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        direction_weights.fill(1.0 / max(n_pixels, 1))
    else:
        direction_weights /= total
    return direction_weights.astype(np.float32)


def make_relion_direction_log_prior(direction_prior, healpix_order):
    """Expand RELION's learned ``pdf_direction`` onto the full rotation grid."""
    direction_prior = np.asarray(direction_prior, dtype=np.float32).reshape(-1)
    n_rot = rotation_grid_size(healpix_order)
    n_pixels = n_rot // rotation_grid_n_in_planes(healpix_order)
    if direction_prior.shape[0] != n_pixels:
        raise ValueError(
            f"direction_prior must have shape ({n_pixels},), got {direction_prior.shape}",
        )

    safe_prior = np.clip(direction_prior, np.finfo(np.float32).tiny, None)
    pixel_idx = np.arange(n_rot, dtype=np.int64) % n_pixels
    return np.log(safe_prior[pixel_idx]).astype(np.float32)


# ---------------------------------------------------------------------------
# FSC -> current_size conversion
# ---------------------------------------------------------------------------

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
    mode="legacy",
    adaptive_pass2_skip_threshold=ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION,
    # --- RELION-mode parameters (only used when mode="relion") ---
    init_healpix_order=2,
    max_healpix_order=7,
    init_translation_range=10.0,
    init_translation_step=2.0,
    init_translation_sigma_angstrom=10.0,
    particle_diameter_ang=None,
    save_intermediates_dir=None,
):
    """Multi-iteration EM refinement with FSC-driven resolution management.

    Supports two modes:

    - ``mode="legacy"`` (default): Original FSC-driven loop with fixed
      rotation grid.  All existing behavior is preserved exactly.

    - ``mode="relion"``: RELION-parity mode with convergence detection,
      angular step refinement, local angular search, and data_vs_prior
      resolution criterion.  Uses :class:`RefinementState` from
      ``convergence.py`` to drive the iteration.

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
        Rotation grid.  In legacy mode, used directly.  In RELION mode,
        used as the initial grid (overridden when angular step refines).
    translations : jnp.ndarray, shape (n_trans, 2)
        Translation grid.
    disc_type : str
        Discretization type for forward/adjoint slicing.
    max_iter : int
        Maximum number of iterations.
    image_batch_size : int
        Number of images per GPU batch.
    rotation_block_size : int
        Number of rotations per block in engine_v2.
    relion_current_sizes : list of int or None
        Oracle mode: if provided, use these current_sizes instead of
        computing from FSC.  relion_current_sizes[i] is used at iteration i.
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
        HEALPix level of the coarse rotation grid.  Required when
        adaptive_oversampling > 0.
    translation_pixel_offset : float or None
        Step size between coarse translation grid points (pixels).
        Required when adaptive_oversampling > 0.
    mode : str
        ``"legacy"`` preserves existing behavior.  ``"relion"`` enables
        RELION-parity convergence-driven refinement.
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

    Additional keys when ``mode="relion"``:
        convergence_state : RefinementState -- final convergence state
        data_vs_prior_trajectory : list of jnp.ndarray -- per-iteration
            data_vs_prior curves
        healpix_order_trajectory : list of int -- HEALPix order per iter
        ave_Pmax_trajectory : list of float -- average Pmax per iter
    """
    if mode not in ("legacy", "relion"):
        raise ValueError(f"Unknown mode={mode!r}; expected 'legacy' or 'relion'")

    if mode == "relion":
        return _refine_relion_mode(
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
            init_healpix_order=init_healpix_order,
            max_healpix_order=max_healpix_order,
            init_translation_range=init_translation_range,
            init_translation_step=init_translation_step,
            init_translation_sigma_angstrom=init_translation_sigma_angstrom,
            particle_diameter_ang=particle_diameter_ang,
            nside_level=nside_level,
            adaptive_pass2_skip_threshold=adaptive_pass2_skip_threshold,
            save_intermediates_dir=save_intermediates_dir,
        )

    # ===================================================================
    # mode="legacy" — existing code below is UNTOUCHED
    # ===================================================================
    from recovar.reconstruction import regularization, noise, relion_functions

    if adaptive_oversampling > 0 and nside_level is None:
        raise ValueError(
            "nside_level must be provided when adaptive_oversampling > 0"
        )

    cryo = experiment_datasets[0]
    volume_shape = cryo.volume_shape

    # State: two half-set volumes, noise, prior
    means = [jnp.array(init_volume), jnp.array(init_volume)]
    noise_variance = jnp.array(init_noise_variance)
    mean_variance = jnp.array(init_mean_variance)

    # History tracking
    current_sizes = []
    fsc_history = []
    pixel_resolutions = []
    wall_times = []
    hard_assignments = [None, None]
    significant_counts = []

    for iteration in range(max_iter):
        t0 = time.time()

        # --- Determine current_size ---
        if relion_current_sizes is not None:
            if iteration < len(relion_current_sizes):
                cs = int(relion_current_sizes[iteration])
            else:
                cs = int(relion_current_sizes[-1])
            if cs <= 0:
                cs = init_current_size
        elif iteration == 0:
            cs = init_current_size
        else:
            fsc_prev = regularization.get_fsc_gpu(
                means[0], means[1], volume_shape,
            )
            raw_cs = fsc_to_current_size(fsc_prev, threshold=fsc_threshold)
            cs = quantize_current_size(raw_cs, ori_size=cryo.image_shape[0])

        cs = quantize_current_size(cs, ori_size=cryo.image_shape[0])
        current_sizes.append(cs)

        logger.info(
            "=== Iteration %d/%d: current_size=%d ===",
            iteration + 1, max_iter, cs,
        )

        use_adaptive = adaptive_oversampling > 0
        cs_for_engine = cs if cs < cryo.image_shape[0] else None

        # --- Run E+M on each half-set ---
        iter_sig_counts = None
        pose_rotations = [np.asarray(rotations), np.asarray(rotations)]
        pose_translations = [
            np.asarray(translations, dtype=np.float32),
            np.asarray(translations, dtype=np.float32),
        ]

        for k in range(2):
            if not use_adaptive:
                # Standard single-pass E+M (Phase 4 behavior)
                new_mean_k, ha_k, Ft_y_k, Ft_ctf_k = run_em_v2(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance,
                    rotations,
                    translations,
                    disc_type,
                    image_batch_size=image_batch_size,
                    rotation_block_size=rotation_block_size,
                    current_size=cs_for_engine,
                )
            else:
                # Two-pass adaptive oversampling (Phase 5)
                # Pass 1: batched significance pruning (memory-efficient)
                sig_rot_any, n_sig, ha_coarse = _compute_significance_batched(
                    experiment_datasets[k],
                    means[k],
                    noise_variance,
                    rotations,
                    translations,
                    disc_type,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    image_batch_size=image_batch_size,
                    rotation_block_size=rotation_block_size,
                    current_size=cs_for_engine,
                )

                if k == 0:
                    iter_sig_counts = n_sig

                n_sig_np = np.asarray(n_sig)
                logger.info(
                    "Pass 1 (half %d): significant samples per image: "
                    "min=%d, median=%d, max=%d, mean=%.0f; "
                    "union significant rotations: %d / %d",
                    k, int(n_sig_np.min()), int(np.median(n_sig_np)),
                    int(n_sig_np.max()), float(n_sig_np.mean()),
                    int(np.sum(sig_rot_any)), rotations.shape[0],
                )

                # Pass 2: oversampled E+M on significant rotations
                # sig_rot_any is (n_rot,) bool -- the global union
                Ft_y_k, Ft_ctf_k, ha_k, oversampled_rots = compute_pass2_stats(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance,
                    np.asarray(rotations),
                    translations,
                    sig_rot_any,
                    nside_level,
                    disc_type,
                    oversampling_order=adaptive_oversampling,
                    current_size=cs_for_engine,
                    image_batch_size=image_batch_size,
                )

                if Ft_y_k is None:
                    # Pass 2 was skipped (union too large); fall back to
                    # pass-1-only mode using the coarse grid.
                    logger.info(
                        "Half %d: pass 2 skipped, running pass-1-only E+M",
                        k,
                    )
                    new_mean_k, ha_k, Ft_y_k, Ft_ctf_k = run_em_v2(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance,
                        rotations,
                        translations,
                        disc_type,
                        image_batch_size=image_batch_size,
                        rotation_block_size=rotation_block_size,
                        current_size=cs_for_engine,
                    )
                    oversampled_rots = None
                    pose_rotations[k] = np.asarray(rotations)
                    pose_translations[k] = np.asarray(translations, dtype=np.float32)
                else:
                    # Solve for this half-set using pass 2 statistics
                    new_mean_k = relion_functions.post_process_from_filter(
                        experiment_datasets[k], Ft_ctf_k, Ft_y_k,
                        tau=mean_variance, disc_type=disc_type,
                    ).reshape(-1)
                    pose_rotations[k] = np.asarray(oversampled_rots)
                    translation_vals = np.unique(np.asarray(translations, dtype=np.float32))
                    translation_diffs = np.diff(np.sort(translation_vals))
                    translation_diffs = translation_diffs[translation_diffs > 1e-6]
                    translation_step = (
                        float(translation_diffs.min()) if translation_diffs.size else 1.0
                    )
                    oversampled_translations, _ = get_oversampled_translation_grid(
                        np.asarray(translations, dtype=np.float32),
                        translation_step,
                        oversampling_order=adaptive_oversampling,
                    )
                    pose_translations[k] = np.asarray(
                        oversampled_translations, dtype=np.float32,
                    )

            means[k] = new_mean_k
            hard_assignments[k] = ha_k

            if k == 0:
                Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
            else:
                Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k

        significant_counts.append(iter_sig_counts)

        # --- Compute unregularized half-maps for FSC and prior ---
        unreg_means = [
            relion_functions.post_process_from_filter(
                cryo, Ft_ctf_0, Ft_y_0, tau=None, disc_type=disc_type,
            ),
            relion_functions.post_process_from_filter(
                cryo, Ft_ctf_1, Ft_y_1, tau=None, disc_type=disc_type,
            ),
        ]

        # --- Compute FSC between half-maps ---
        fsc = regularization.get_fsc_gpu(
            unreg_means[0], unreg_means[1], volume_shape,
        )
        fsc_history.append(fsc)

        # --- Resolution from FSC ---
        from recovar.heterogeneity.locres import find_fsc_resol
        pixel_res = float(find_fsc_resol(fsc, threshold=fsc_threshold))
        pixel_resolutions.append(pixel_res)

        # --- Update prior (RELION-style tau^2 from FSC) ---
        mean_signal_variance, _, _ = regularization.compute_relion_prior(
            experiment_datasets, noise_variance, unreg_means[0], unreg_means[1], 100,
        )
        mean_variance = mean_signal_variance

        # --- Update noise estimate ---
        for k in range(2):
            best_rots, best_trans = hard_assignment_idx_to_pose(
                hard_assignments[k], pose_rotations[k], pose_translations[k],
            )
            experiment_datasets[k].update_poses(best_rots, best_trans)

        noise_from_res = noise.estimate_noise_level_no_masks(
            experiment_datasets[0],
            np.arange(min(1000, cryo.n_units)),
            means[0],
            100,
            disc_type=disc_type,
        )
        noise_variance = noise.make_radial_noise(noise_from_res, cryo.image_shape)

        # --- Timing ---
        elapsed = time.time() - t0
        wall_times.append(elapsed)

        res_angstrom = shell_index_to_resolution_angstrom(
            pixel_res, cryo.image_shape[0], cryo.voxel_size,
        )
        logger.info(
            "Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, time=%.1fs",
            iteration + 1, cs, pixel_res, res_angstrom, elapsed,
        )

    # --- Final merged mean ---
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
    }


# ---------------------------------------------------------------------------
# RELION-parity refinement mode
# ---------------------------------------------------------------------------


def _extract_max_posterior_per_image(
    experiment_dataset,
    mean,
    noise_variance,
    rotations,
    translations,
    disc_type,
    image_batch_size,
    rotation_block_size,
    current_size,
    half_spectrum_scoring=False,
):
    """Extract per-image maximum posterior probability from an E-step pass.

    Runs a lightweight E-step (pass 1 only: streaming logsumexp) and
    returns the best log-score per image, converted to a probability
    via exp(best_score - log_Z).

    This is a separate utility because run_em_v2 does not currently expose
    per-image Pmax.  We reuse the same block structure as run_em_v2's
    pass 1 to compute log_Z and best_score in a memory-efficient way.

    Returns
    -------
    max_prob : np.ndarray, shape (n_images,)
        Per-image maximum posterior probability in [0, 1].
    """
    from recovar.em.dense_single_volume.engine_v2 import (
        _preprocess_batch, _compute_projections_block,
        _e_step_block_scores, _e_step_block_scores_windowed,
        _update_logsumexp, make_half_image_weights,
    )
    from recovar.core.configs import ForwardModelConfig
    import recovar.core.fourier_transform_utils as fourier_transform_utils

    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    half_weights = jnp.ones(n_half, dtype=jnp.float32) if half_spectrum_scoring else make_half_image_weights(image_shape)

    use_window = current_size is not None and current_size < image_shape[0]
    if use_window:
        from .fourier_window import make_fourier_window_indices_np
        window_indices_np, n_windowed = make_fourier_window_indices_np(image_shape, current_size)
        window_indices = jnp.asarray(window_indices_np)
        half_weights_windowed = half_weights[window_indices]
    else:
        window_indices = None
        n_windowed = n_half

    # Pad rotations
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([
            rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))
        ], axis=0)
    else:
        rotations_padded = rotations

    max_prob_all = np.empty(n_images, dtype=np.float32)

    image_indices = np.arange(n_images)
    start_idx = 0

    for (batch_data, _, _, ctf_params, _, _, indices) in experiment_dataset.iter_batches(
        image_batch_size, indices=image_indices, by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        batch_data = jnp.asarray(batch_data)

        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data, ctf_params, noise_variance, translations, config,
            batch_size, n_trans,
        )

        if use_window:
            shifted_data = shifted_half[:, window_indices]
            ctf2_data = ctf2_over_nv_half[:, window_indices]
        else:
            shifted_data = shifted_half
            ctf2_data = ctf2_over_nv_half

        # Streaming logsumexp + best score tracking
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size)
        best_score = jnp.full(batch_size, -jnp.inf)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            if use_window:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
                scores = _e_step_block_scores_windowed(
                    shifted_data, batch_norm, ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    half_weights_windowed,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
            else:
                scores = _e_step_block_scores(
                    shifted_data, batch_norm, ctf2_data,
                    proj_half_b * half_weights,
                    proj_abs2_half_b * half_weights,
                    half_weights,
                    batch_size, n_trans, image_shape, volume_shape,
                )

            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)
            # Track best score per image across all blocks
            block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
            best_score = jnp.maximum(best_score, block_best)

        # Pmax = exp(best_score - log_Z)
        log_Z = max_s + jnp.log(sum_exp)
        pmax = jnp.exp(best_score - log_Z)
        max_prob_all[start_idx:end_idx] = np.asarray(pmax)
        start_idx = end_idx

    return max_prob_all


def _refine_relion_mode(
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
    init_healpix_order,
    max_healpix_order,
    init_translation_range,
    init_translation_step,
    init_translation_sigma_angstrom,
    particle_diameter_ang,
    nside_level,
    adaptive_pass2_skip_threshold,
    save_intermediates_dir=None,
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
    from recovar.reconstruction import regularization, noise, relion_functions

    cryo = experiment_datasets[0]
    volume_shape = cryo.volume_shape
    grid_size = cryo.image_shape[0]  # ori_size in RELION terms

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
    )

    # RELION mode owns the coarse HEALPix grid. When coarse-grid metadata is
    # provided, regenerate the matching coarse grid here instead of inheriting
    # any finer caller-supplied rotation table.
    current_healpix_order = int(init_healpix_order)
    if nside_level is not None:
        if int(nside_level) != current_healpix_order:
            logger.info(
                "RELION mode: ignoring caller nside_level=%d and regenerating "
                "initial coarse grid at healpix_order=%d",
                int(nside_level), current_healpix_order,
            )
        current_rotations = get_rotation_grid_at_order(
            current_healpix_order, matrices=True,
        ).astype(np.float32)
        current_nside_level = current_healpix_order
    else:
        current_rotations = np.asarray(rotations, dtype=np.float32)
        current_nside_level = current_healpix_order
    current_translations = jnp.asarray(translations, dtype=jnp.float32)

    # RELION reconstructs on a 2x padded Fourier grid to reduce interpolation
    # error before cropping back to the native real-space volume.
    PADDING_FACTOR = 2

    def _safe_batch_sizes(n_rot, n_trans):
        """Reduce batch sizes for large pose grids to avoid GPU OOM."""
        # Target the actual score-tensor size: n_img * n_rot_block * n_trans.
        budget = 50_000_000
        n_trans = max(int(n_trans), 1)
        ibs = min(
            image_batch_size,
            max(1, budget // max(n_rot * n_trans, 1)),
        )
        rbs = min(
            rotation_block_size,
            max(64, budget // max(ibs * n_trans, 1)),
        )
        return ibs, rbs

    # State: two half-set volumes, noise, prior
    means = [jnp.array(init_volume), jnp.array(init_volume)]
    noise_variance = jnp.array(init_noise_variance)
    mean_variance = jnp.array(init_mean_variance)

    # History tracking
    current_sizes = []
    fsc_history = []
    pixel_resolutions = []
    wall_times = []
    hard_assignments = [None, None]
    previous_assignments = [None, None]
    previous_best_rotations = [None, None]
    previous_best_translations = [None, None]
    max_posterior_per_half = [None, None]
    rotation_posterior_per_half = [None, None]
    significant_counts = []
    data_vs_prior_trajectory = []
    healpix_order_trajectory = []
    ave_Pmax_trajectory = []
    pmax_per_image_history = []
    # RELION uses incr_size=10. We add 2 extra shells to compensate for
    # the 1-shell FSC gap that makes our current_size 2 pixels smaller.
    # This gives current_size=30 instead of 26 at iter 2, providing ~450
    # scoring pixels vs ~280, which prevents the chi^2 from being too
    # discriminative and causing posterior collapse.
    relion_incr_size = 12
    relion_has_high_fsc_at_limit = False
    global_direction_prior = None
    global_direction_prior_order = None

    # Extract radial noise profile from initial pixel-array noise_variance.
    # This serves as the floor reference for the first iteration's noise update,
    # preventing runaway collapse when posteriors are too peaked.
    from recovar.core import fourier_transform_utils
    n_shells_init = cryo.image_shape[0] // 2 + 1
    radial_dist_init = np.clip(
        fourier_transform_utils.get_grid_of_radial_distances(
            cryo.image_shape, scaled=False, frequency_shift=0,
        )
        .astype(int)
        .reshape(-1),
        0, n_shells_init - 1,
    )
    # Average pixel values per shell (noise_variance is radially symmetric)
    previous_noise_radial = np.zeros(n_shells_init, dtype=np.float64)
    shell_counts = np.zeros(n_shells_init, dtype=np.float64)
    noise_variance_np = np.asarray(noise_variance, dtype=np.float64)
    np.add.at(previous_noise_radial, radial_dist_init[:noise_variance_np.size], noise_variance_np)
    np.add.at(shell_counts, radial_dist_init[:noise_variance_np.size], 1.0)
    shell_counts = np.maximum(shell_counts, 1.0)
    previous_noise_radial = previous_noise_radial / shell_counts
    previous_noise_radial = jnp.asarray(previous_noise_radial[:n_shells_init], dtype=jnp.float32)

    # Save the initial noise estimate once, before any iterations update it.
    # This is used as a per-shell floor: the posterior-weighted noise can only
    # INCREASE from this baseline, never decrease.  At high frequencies the
    # posterior noise exceeds the initial estimate (model error raises it), so
    # we use the posterior value.  At low frequencies the posterior noise is
    # lower than the initial estimate (selection bias lowers it), so we clamp
    # to the initial value.  This matches RELION's observed behavior where
    # noise increases at low freq and stays similar at high freq during early
    # iterations.
    initial_noise_radial = previous_noise_radial.copy()

    # NOTE: keep the padded reconstruction factor defined above; the
    # current-size and convergence logic below assumes a single source of truth.

    iteration = 0
    while not state.has_converged and iteration < max_iter:
        t0 = time.time()

        # --- Determine current_size using RELION's FSC-derived SSNR (C4/C5) ---
        # At iteration 0, no previous half-map FSC exists yet; use the initial
        # resolution plus RELION's bootstrap image-size growth. After that,
        # mimic RELION's auto-refine update:
        # 1. zero FSC beyond the previous current_size limit
        # 2. convert FSC -> SSNR (= data_vs_prior in split-half auto-refine)
        # 3. grow current_size using ave_Pmax, FSC at the current limit, and
        #    RELION's dynamic incr_size heuristic.
        if iteration == 0:
            cs = _bootstrap_current_size_relion(init_current_size, grid_size)
            data_vs_prior_iter = None
        else:
            fsc_prev = np.asarray(fsc_history[-1], dtype=np.float32).copy()
            prev_cs = current_sizes[-1]
            if prev_cs < grid_size:
                fsc_prev[min(len(fsc_prev), prev_cs // 2):] = 0.0

            data_vs_prior_iter = np.asarray(fsc_to_relion_ssnr(fsc_prev))
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
                res_shell, grid_size,
                ave_Pmax=state.ave_Pmax,
                has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                incr_size=relion_incr_size,
            )
            cs = quantize_current_size(raw_cs, ori_size=grid_size)

        cs = quantize_current_size(cs, ori_size=grid_size)
        current_sizes.append(cs)
        healpix_order_trajectory.append(state.healpix_order)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, "
            "healpix_order=%d, local_search=%s ===",
            iteration + 1, max_iter, cs,
            state.healpix_order, state.do_local_search,
        )

        # --- Angular step refinement: regenerate rotation grid if needed ---
        # When update_refinement_state incremented healpix_order, we need
        # a new rotation grid at the finer level.
        # IMPORTANT: At order >= 5, the full grid has 2.4M+ rotations which
        # OOMs the GPU.  Instead, keep the order-4 grid as the "base" and
        # rely on local search + oversampling to achieve finer angular steps.
        # The order is still tracked for sigma calculation.
        MAX_FULL_GRID_ORDER = 4
        if state.healpix_order != current_healpix_order:
            new_order = min(state.healpix_order, MAX_FULL_GRID_ORDER)
            if new_order != current_healpix_order:
                logger.info(
                    "Regenerating rotation grid: order %d -> %d",
                    current_healpix_order, new_order,
                )
                current_rotations = get_rotation_grid_at_order(
                    new_order, matrices=True,
                ).astype(np.float32)
                current_healpix_order = new_order
                global_direction_prior = None
                global_direction_prior_order = None
            else:
                logger.info(
                    "Angular step refined to order %d (grid stays at order %d "
                    "— local search handles finer sampling)",
                    state.healpix_order, current_healpix_order,
                )
            current_nside_level = current_healpix_order

            # Regenerate translation grid based on updated parameters
            current_translations = jnp.array(
                get_translation_grid(
                    state.translation_range, state.translation_step,
                ).astype(np.float32)
            )
            logger.info(
                "New grid: %d rotations, %d translations "
                "(range=%.1f, step=%.1f)",
                current_rotations.shape[0], current_translations.shape[0],
                state.translation_range, state.translation_step,
            )

        # --- Local angular search bookkeeping ---
        # Once RELION enters local search, each image should search around its
        # own previous orientation on the true current HEALPix order. Use the
        # exact rotations selected in the previous iteration, not the nearest
        # snapped grid indices.
        effective_rotations = current_rotations
        rotation_log_prior = None
        use_local = (
            state.do_local_search
            and all(rot is not None for rot in previous_best_rotations)
            and iteration > 0
        )
        local_search_order = None
        sigma_rot = state.sigma_rot
        sigma_psi = state.sigma_psi if state.sigma_psi > 0 else sigma_rot
        if use_local and sigma_rot <= 0:
            step_rad = np.deg2rad(
                healpix_angular_step(state.healpix_order)
                / (2 ** state.adaptive_oversampling)
            )
            sigma_rot = np.sqrt(2.0 * 2.0) * step_rad
            sigma_psi = sigma_rot

        if use_local:
            local_search_order = state.healpix_order + state.adaptive_oversampling
            logger.info(
                "Local search (batched exact): fine_order=%d, sigma_rot=%.4f rad "
                "(%.2f deg), sigma_psi=%.4f rad",
                local_search_order,
                sigma_rot,
                np.rad2deg(sigma_rot),
                sigma_psi,
            )
        elif (
            global_direction_prior is not None
            and global_direction_prior_order == current_healpix_order
        ):
            rotation_log_prior = make_relion_direction_log_prior(
                global_direction_prior,
                current_healpix_order,
            )
            logger.info(
                "Using learned global direction prior: %d directions at healpix_order=%d",
                global_direction_prior.shape[0],
                current_healpix_order,
            )

        cs_for_engine = cs if cs < cryo.image_shape[0] else None

        # Effective noise for scoring: inflate by a factor to approximate
        # RELION's elevated effective noise from softMaskOutsideMap with
        # colored noise fill. RELION's noise at low frequencies is 5-33x
        # higher after the first M-step. This factor prevents posterior
        # collapse at low current_size values where few scoring pixels
        # make the chi^2 very discriminative.
        # TODO: replace with proper masking convention match.
        # NOTE: RELION's effective noise is 5-33x higher at low frequencies
        # due to softMaskOutsideMap with colored noise fill. Testing showed
        # that a 5x inflation enables convergence (iter 1 Pmax=0.01, iter 2=0.31)
        # but the system still collapses at iter 3 due to inconsistent noise
        # between scoring and estimation paths. The proper fix requires matching
        # RELION's masking convention or implementing per-shell noise inflation.
        # See memory/project_noise_parity_status.md for details.

        # --- Run E+M on each half-set ---
        # Two modes: single-pass (adaptive_oversampling=0) or two-pass
        # coarse/fine (adaptive_oversampling>=1).
        iter_sig_counts = None
        use_adaptive = (
            state.adaptive_oversampling > 0
            and not use_local
            and effective_rotations.shape[0] > 16
        )

        # Track the rotation grids used for pose extraction.
        # When adaptive oversampling is active, ha_k indices refer to the
        # oversampled grid (from pass 2), not effective_rotations.
        pose_rotations = [None, None]  # rotations to use with ha for poses
        pose_translations = [
            np.asarray(current_translations, dtype=np.float32),
            np.asarray(current_translations, dtype=np.float32),
        ]
        best_pose_rotations = [None, None]
        best_pose_translations = [None, None]
        # Coarse-grid assignments for local search tracking (always indexed
        # into effective_rotations, even when adaptive oversampling is used).
        coarse_ha = [None, None]

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
            coarse_size = quantize_current_size(coarse_size, ori_size=grid_size)
            # Coarse size must be smaller than full current_size
            if cs_for_engine is not None and coarse_size >= cs:
                coarse_size = max(8, cs // 2)
                coarse_size = quantize_current_size(coarse_size, ori_size=grid_size)
            coarse_cs = coarse_size if coarse_size < grid_size else None

            logger.info(
                "Adaptive oversampling: pass 1 at coarse_size=%s, "
                "pass 2 at current_size=%s (oversampling=%d, particle_diameter=%s)",
                coarse_cs,
                cs_for_engine,
                state.adaptive_oversampling,
                (
                    f"{float(particle_diameter_ang):.1f} A"
                    if particle_diameter_ang is not None
                    else "box_size"
                ),
            )

        noise_stats_per_half = [None, None]

        for k in range(2):
            current_translation_range = float(state.translation_range)
            translation_log_prior = make_relion_translation_log_prior(
                np.asarray(current_translations, dtype=np.float32),
                cryo.voxel_size,
                init_translation_sigma_angstrom,
                previous_best_translations[k],
                offset_range_pixels=current_translation_range,
            )
            if use_local:
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    rotation_grid_size(local_search_order),
                    current_translations.shape[0],
                )
                Ft_y_k, Ft_ctf_k, ha_k, em_stats_k = _run_grouped_local_search_em(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance,
                    previous_best_rotations[k],
                    local_search_order,
                    sigma_rot,
                    sigma_psi,
                    current_translations,
                    previous_best_translations[k],
                    init_translation_sigma_angstrom,
                    current_translation_range,
                    disc_type,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=cs_for_engine,
                )
                pose_rotations[k] = None
                coarse_ha[k] = ha_k

            elif use_adaptive:
                # --- PASS 1: Coarse significance pruning ---
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )

                t_pass1 = time.time()
                sig_rot_any, n_sig_batch, ha_coarse, sig_sample_indices = (
                    _compute_significance_batched(
                        experiment_datasets[k],
                        means[k],
                        noise_variance,
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
                        rotation_log_prior=rotation_log_prior,
                        translation_log_prior=translation_log_prior,
                        half_spectrum_scoring=True,
                    )
                )
                n_sig_total = int(np.sum(sig_rot_any))
                dt_pass1 = time.time() - t_pass1

                logger.info(
                    "Pass 1 (half %d): %d / %d significant coarse "
                    "rotations in %.1fs (median n_sig/image=%d)",
                    k, n_sig_total, effective_rotations.shape[0],
                    dt_pass1, int(np.median(n_sig_batch)),
                )

                skip_pass2, sig_fraction = should_skip_adaptive_pass2(
                    n_sig_batch,
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                    threshold=adaptive_pass2_skip_threshold,
                )
                total_coarse_samples = (
                    effective_rotations.shape[0] * current_translations.shape[0]
                )

                if skip_pass2:
                    logger.info(
                        "Pass 2 skipped (half %d): mean significant fraction=%.3f >= %.3f; "
                        "running single-pass full-resolution E+M",
                        k,
                        sig_fraction,
                        ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION,
                    )
                    _, ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = run_em_v2(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance,
                        effective_rotations,
                        current_translations,
                        disc_type,
                        image_batch_size=safe_ibs,
                        rotation_block_size=safe_rbs,
                        current_size=cs_for_engine,
                        rotation_log_prior=rotation_log_prior,
                        translation_log_prior=translation_log_prior,
                        score_with_masked_images=True,
                        return_stats=True,
                        accumulate_noise=True,
                        half_spectrum_scoring=True,
                        # noise_fill_outside_mask=True,  # disabled: makes Pmax worse
                    )
                    noise_stats_per_half[k] = noise_stats_k
                    pose_rotations[k] = effective_rotations
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
                        noise_variance,
                        effective_rotations,
                        current_translations,
                        np.ones(effective_rotations.shape[0], dtype=bool),
                        current_nside_level,
                        disc_type,
                        oversampling_order=state.adaptive_oversampling,
                        current_size=cs_for_engine,
                        translation_step=state.translation_step,
                        rotation_log_prior=rotation_log_prior,
                        translation_log_prior=translation_log_prior,
                        score_with_masked_images=True,
                        return_stats=True,
                        accumulate_noise=True,
                        half_spectrum_scoring=True,
                        # noise_fill_outside_mask=True,  # disabled: makes Pmax worse
                    )
                    Ft_y_k, Ft_ctf_k, ha_k, oversampled_rots_k, em_stats_k, noise_stats_k = pass2_outputs
                    noise_stats_per_half[k] = noise_stats_k
                    dt_pass2 = time.time() - t_pass2
                    logger.info(
                        "Pass 2 dense exact (half %d): %.1fs using full oversampled grid",
                        k, dt_pass2,
                    )
                    pose_rotations[k] = np.asarray(oversampled_rots_k, dtype=np.float32)
                    oversampled_translations, _ = get_oversampled_translation_grid(
                        np.asarray(current_translations, dtype=np.float32),
                        state.translation_step,
                        oversampling_order=state.adaptive_oversampling,
                    )
                    pose_translations[k] = np.asarray(
                        oversampled_translations, dtype=np.float32,
                    )
                    coarse_ha[k] = ha_coarse
                else:
                    # --- Exact sparse pass 2 over significant coarse samples ---
                    t_pass2 = time.time()
                    (
                        Ft_y_k,
                        Ft_ctf_k,
                        ha_k,
                        best_rots_k,
                        best_trans_k,
                        _best_rot_indices_k,
                        em_stats_k,
                        noise_stats_k,
                    ) = compute_pass2_stats_sparse(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance,
                        current_translations,
                        sig_sample_indices,
                        current_nside_level,
                        disc_type,
                        oversampling_order=state.adaptive_oversampling,
                        current_size=cs_for_engine,
                        translation_step=state.translation_step,
                        rotation_log_prior=rotation_log_prior,
                        translation_log_prior=translation_log_prior,
                        score_with_masked_images=True,
                        return_stats=True,
                        accumulate_noise=True,
                        half_spectrum_scoring=True,
                        # noise_fill_outside_mask=True,  # disabled: makes Pmax worse
                    )
                    noise_stats_per_half[k] = noise_stats_k
                    dt_pass2 = time.time() - t_pass2
                    logger.info(
                        "Pass 2 sparse (half %d): %.1fs",
                        k, dt_pass2,
                    )
                    best_pose_rotations[k] = np.asarray(best_rots_k, dtype=np.float32)
                    best_pose_translations[k] = np.asarray(best_trans_k, dtype=np.float32)
                    oversampled_translations, _ = get_oversampled_translation_grid(
                        np.asarray(current_translations, dtype=np.float32),
                        state.translation_step,
                        oversampling_order=state.adaptive_oversampling,
                    )
                    pose_translations[k] = np.asarray(
                        oversampled_translations, dtype=np.float32,
                    )

                    # Store coarse-grid assignment from pass 1 for local search.
                    coarse_ha[k] = ha_coarse

                if iter_sig_counts is None:
                    iter_sig_counts = n_sig_batch
                else:
                    iter_sig_counts = np.concatenate([
                        iter_sig_counts, n_sig_batch
                    ])

            else:
                # --- SINGLE-PASS E+M (no adaptive oversampling) ---
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )
                _, ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = run_em_v2(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance,
                    effective_rotations,
                    current_translations,
                    disc_type,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=cs_for_engine,
                    rotation_log_prior=rotation_log_prior,
                    translation_log_prior=translation_log_prior,
                    score_with_masked_images=True,
                    return_stats=True,
                    accumulate_noise=True,
                )
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = effective_rotations
                pose_translations[k] = np.asarray(current_translations, dtype=np.float32)
                coarse_ha[k] = ha_k  # same grid, no oversampling

            # Reconstruct the regularized mean with padding_factor=2.
            # run_em_v2 uses padding_factor=1 internally; we override here.
            Ft_ctf_k_padded = relion_functions.zero_pad_fourier_volume(
                Ft_ctf_k, volume_shape, PADDING_FACTOR,
            )
            Ft_y_k_padded = relion_functions.zero_pad_fourier_volume(
                Ft_y_k, volume_shape, PADDING_FACTOR,
            )
            means[k] = relion_functions.post_process_from_filter_v2(
                Ft_ctf_k_padded, Ft_y_k_padded,
                volume_shape, PADDING_FACTOR,
                tau=mean_variance,
                kernel="triangular",
                use_spherical_mask=True, grid_correct=True,
                gridding_correct="square",
            ).reshape(-1)
            hard_assignments[k] = ha_k
            max_posterior_per_half[k] = np.asarray(
                em_stats_k.max_posterior_per_image, dtype=np.float32,
            )
            rotation_posterior_per_half[k] = np.asarray(
                em_stats_k.rotation_posterior_sums, dtype=np.float32,
            )

            if k == 0:
                Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
            else:
                Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k

        significant_counts.append(iter_sig_counts)

        if (
            not use_local
            and all(rot_sum is not None for rot_sum in rotation_posterior_per_half)
            and effective_rotations.shape[0] == rotation_grid_size(current_healpix_order)
        ):
            global_direction_prior = collapse_rotation_posterior_to_direction_prior(
                np.asarray(rotation_posterior_per_half[0], dtype=np.float64)
                + np.asarray(rotation_posterior_per_half[1], dtype=np.float64),
                current_healpix_order,
            )
            global_direction_prior_order = current_healpix_order

        # --- Combined Fourier weights for data_vs_prior at next iteration ---
        # RELION uses the combined (both half-sets) CTF^2 weight for
        # data_vs_prior.  Store for use at the start of the next iteration.
        Ft_ctf_combined = Ft_ctf_0 + Ft_ctf_1

        # --- Compute unregularized half-maps for FSC and prior ---
        # RELION uses padding_factor=2 by default: the 3D Fourier grid is
        # (2*N)^3 to reduce interpolation artifacts.  We zero-pad the
        # native-size Ft_ctf/Ft_y into the padded grid, then
        # post_process_from_filter_v2 does iDFT on the padded grid and
        # crops back to native size in real space.
        Ft_ctf_0_padded = relion_functions.zero_pad_fourier_volume(
            Ft_ctf_0, volume_shape, PADDING_FACTOR,
        )
        Ft_y_0_padded = relion_functions.zero_pad_fourier_volume(
            Ft_y_0, volume_shape, PADDING_FACTOR,
        )
        Ft_ctf_1_padded = relion_functions.zero_pad_fourier_volume(
            Ft_ctf_1, volume_shape, PADDING_FACTOR,
        )
        Ft_y_1_padded = relion_functions.zero_pad_fourier_volume(
            Ft_y_1, volume_shape, PADDING_FACTOR,
        )
        unreg_means = [
            relion_functions.post_process_from_filter_v2(
                Ft_ctf_0_padded, Ft_y_0_padded,
                volume_shape, PADDING_FACTOR,
                tau=None, kernel="triangular",
                use_spherical_mask=True, grid_correct=True,
                gridding_correct="square",
            ),
            relion_functions.post_process_from_filter_v2(
                Ft_ctf_1_padded, Ft_y_1_padded,
                volume_shape, PADDING_FACTOR,
                tau=None, kernel="triangular",
                use_spherical_mask=True, grid_correct=True,
                gridding_correct="square",
            ),
        ]

        # --- Compute FSC between half-maps ---
        fsc = regularization.get_fsc_gpu(
            unreg_means[0], unreg_means[1], volume_shape,
        )
        fsc_history.append(fsc)

        # --- Save intermediate volumes if requested ---
        if save_intermediates_dir is not None:
            import os
            os.makedirs(save_intermediates_dir, exist_ok=True)
            import mrcfile
            for k_half in range(2):
                vol_real = np.real(
                    np.fft.ifftn(
                        np.fft.ifftshift(
                            np.asarray(means[k_half]).reshape(volume_shape)
                        )
                    )
                ).astype(np.float32)
                mrc_path = os.path.join(
                    save_intermediates_dir,
                    f"it{iteration:03d}_half{k_half+1}_reg.mrc",
                )
                with mrcfile.new(mrc_path, overwrite=True) as mrc:
                    mrc.set_data(vol_real)
                # Also save unregularized half-map
                vol_unreg = np.real(
                    np.fft.ifftn(
                        np.fft.ifftshift(
                            np.asarray(unreg_means[k_half]).reshape(volume_shape)
                        )
                    )
                ).astype(np.float32)
                mrc_unreg_path = os.path.join(
                    save_intermediates_dir,
                    f"it{iteration:03d}_half{k_half+1}_unreg.mrc",
                )
                with mrcfile.new(mrc_unreg_path, overwrite=True) as mrc:
                    mrc.set_data(vol_unreg)
            # Save FSC and noise/tau2 per iteration
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_fsc.npy"),
                np.asarray(fsc),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_noise.npy"),
                np.asarray(noise_variance),
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
                            f"it{iteration:03d}_ha_half{k_half+1}.npy",
                        ),
                        hard_assignments[k_half],
                    )
            # Save per-iteration metadata
            iter_meta = {
                "iteration": iteration,
                "current_size": int(cs),
                "n_rotations": int(
                    rotation_grid_size(local_search_order)
                    if use_local
                    else effective_rotations.shape[0]
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
                (
                    np.asarray(effective_rotations)
                    if not use_local
                    else np.empty((0, 3, 3), dtype=np.float32)
                ),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_translations.npy"),
                np.asarray(current_translations),
            )
            logger.info(
                "Saved intermediate volumes to %s (iteration %d)",
                save_intermediates_dir, iteration,
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

        # --- Update prior from the actual weighted reconstruction stats ---
        mean_signal_variance, _ = regularization.compute_relion_prior_from_reconstruction_stats(
            Ft_ctf_0,
            Ft_ctf_1,
            Ft_y_0,
            Ft_y_1,
            volume_shape,
            mean_variance,
            padding_factor=PADDING_FACTOR,
        )
        mean_variance = mean_signal_variance

        # --- Resolution from updated FSC-derived SSNR (RELION auto-refine) ---
        dvp_iter = np.asarray(fsc, dtype=np.float32).copy()
        if cs < grid_size:
            dvp_iter[min(len(dvp_iter), cs // 2):] = 0.0
        dvp_iter = np.asarray(fsc_to_relion_ssnr(dvp_iter))
        dvp_res_shell = resolution_from_data_vs_prior(
            dvp_iter,
            allow_high_res_recovery=True,
        )
        pixel_res = float(dvp_res_shell)
        pixel_resolutions.append(pixel_res)

        # --- Update poses and noise ---
        for k in range(2):
            if best_pose_rotations[k] is not None:
                best_rots = np.asarray(best_pose_rotations[k], dtype=np.float32)
                best_trans = np.asarray(best_pose_translations[k], dtype=np.float32)
            elif use_local:
                rot_idx = hard_assignments[k] // current_translations.shape[0]
                trans_idx = hard_assignments[k] % current_translations.shape[0]
                best_rots = rotation_indices_to_matrices(
                    rot_idx, local_search_order,
                )
                best_trans = np.asarray(current_translations)[trans_idx]
            else:
                # Global search uses the dense grid in pose_rotations[k].
                best_rots, best_trans = hard_assignment_idx_to_pose(
                    hard_assignments[k], pose_rotations[k], pose_translations[k],
                )
            previous_best_rotations[k] = np.asarray(best_rots, dtype=np.float32)
            previous_best_translations[k] = np.asarray(best_trans, dtype=np.float32)
            experiment_datasets[k].update_poses(best_rots, best_trans)

        # Estimate noise using BOTH hard-assignment residuals AND posterior-weighted stats.
        # Hard-assignment: captures model error (imperfect reference → large residual
        # even at best orientation). Important at low frequencies.
        # Posterior-weighted: captures posterior uncertainty. Important for RELION parity
        # at later iterations when model improves.
        # Strategy: take the MAX per shell to get the most conservative (highest) estimate.
        noise_estimates_hard = []
        for k in range(2):
            n_k = experiment_datasets[k].n_units
            noise_k = noise.estimate_noise_level_no_masks(
                experiment_datasets[k],
                np.arange(n_k),
                means[k],
                100,
                disc_type=disc_type,
            )
            noise_estimates_hard.append(noise_k)
        noise_hard_raw = (noise_estimates_hard[0] + noise_estimates_hard[1]) / 2
        # Convert from recovar's unnormalized FFT convention to RELION convention
        fft_power_scale = float(cryo.image_shape[0] * cryo.image_shape[1]) ** 2
        noise_hard = noise_hard_raw / fft_power_scale

        # RELION-parity: inflate hard-assignment noise by the mask area ratio.
        # Zero-masked images have ~f of the full image power (f = mask_area/total).
        # The residual |proj - img_masked|^2 underestimates noise by ~1/f because
        # 89% of pixels are zero. Multiplying by 1/f ≈ area_ratio restores the
        # effective noise that RELION gets from noise-filled images.
        # This is the critical difference that gives RELION sigma2_noise 5-33x
        # higher at low frequencies, which makes iter 2 Pmax ~ 0.0002.
        if particle_diameter_ang is not None and cryo.voxel_size > 0:
            mask_radius_px = float(particle_diameter_ang) / (2.0 * cryo.voxel_size)
            mask_area = np.pi * mask_radius_px ** 2
            total_area = float(cryo.image_shape[0] * cryo.image_shape[1])
            mask_inflation = total_area / max(mask_area, 1.0)
            noise_hard = noise_hard * mask_inflation
            logger.info(
                "Hard-assignment noise inflated by mask ratio %.2f "
                "(mask_radius=%.1f px), range=[%.2e, %.2e]",
                mask_inflation, mask_radius_px,
                float(jnp.min(noise_hard)), float(jnp.max(noise_hard)),
            )

        if noise_stats_per_half[0] is not None and noise_stats_per_half[1] is not None:
            wsum_combined = (
                np.asarray(noise_stats_per_half[0].wsum_sigma2_noise, dtype=np.float64)
                + np.asarray(noise_stats_per_half[1].wsum_sigma2_noise, dtype=np.float64)
            )
            img_power_combined = (
                np.asarray(noise_stats_per_half[0].wsum_img_power, dtype=np.float64)
                + np.asarray(noise_stats_per_half[1].wsum_img_power, dtype=np.float64)
            )
            sumw_combined = noise_stats_per_half[0].sumw + noise_stats_per_half[1].sumw
            noise_posterior = noise.normalize_wsum_to_sigma2_noise(
                wsum_combined, img_power_combined, sumw_combined, cryo.image_shape,
            )
            # Take per-shell MAX of hard-assignment and posterior-weighted noise.
            # This ensures model error is captured (hard-assignment at low freq)
            # while also allowing posterior-weighted increases at high freq.
            n_common = min(len(noise_hard), len(noise_posterior))
            noise_from_res = jnp.maximum(
                jnp.asarray(noise_hard[:n_common]),
                jnp.asarray(noise_posterior[:n_common]),
            )
            logger.info(
                "Noise updated: hard=[%.2e, %.2e], posterior=[%.2e, %.2e], "
                "combined=[%.2e, %.2e]",
                float(jnp.min(noise_hard)), float(jnp.max(noise_hard)),
                float(jnp.min(noise_posterior)), float(jnp.max(noise_posterior)),
                float(jnp.min(noise_from_res)), float(jnp.max(noise_from_res)),
            )
        else:
            noise_from_res = jnp.asarray(noise_hard)
            logger.info(
                "Noise from hard-assignment: range=[%.2e, %.2e]",
                float(jnp.min(noise_from_res)), float(jnp.max(noise_from_res)),
            )

        # Apply initial-noise floor: never let noise go below the initial
        # estimate. With annealing, the noise inflation decreases over
        # iterations, so the running-max floor would defeat the annealing.
        old_noise_radial = previous_noise_radial
        noise_from_res_raw = noise_from_res
        n_floor = min(len(noise_from_res), len(initial_noise_radial))
        noise_from_res = noise_from_res.at[:n_floor].set(
            jnp.maximum(noise_from_res[:n_floor], initial_noise_radial[:n_floor])
        )
        n_clamped = int(jnp.sum(noise_from_res_raw[:n_floor] < initial_noise_radial[:n_floor]))
        if n_clamped > 0:
            logger.info(
                "Noise floor applied: %d/%d shells clamped to running-max noise",
                n_clamped, n_floor,
            )

        # Log per-shell noise comparison (first 10 shells) for convergence diagnostics
        n_log = min(10, len(noise_from_res), len(old_noise_radial))
        logger.info(
            "Noise update per shell (first %d): old=[%s] new=[%s]",
            n_log,
            ", ".join(f"{float(x):.3e}" for x in old_noise_radial[:n_log]),
            ", ".join(f"{float(x):.3e}" for x in noise_from_res[:n_log]),
        )

        # Update previous_noise_radial for next iteration's diagnostics
        previous_noise_radial = noise_from_res

        # The mask-ratio inflation on noise_hard (above) already accounts for
        # the effective noise increase from RELION's softMaskOutsideMap.
        # No additional per-shell or annealing inflation needed.

        noise_variance = noise.make_radial_noise(noise_from_res, cryo.image_shape)

        # --- Update convergence state ---
        # This checks assignment changes, resolution stalls, and may trigger
        # angular step refinement or convergence.
        n_rot_current = (
            rotation_grid_size(local_search_order)
            if use_local
            else effective_rotations.shape[0]
        )
        n_trans_current = current_translations.shape[0]

        state = update_refinement_state(
            state,
            current_assignments=current_combined_ha,
            previous_assignments=previous_combined_ha,
            n_rotations=n_rot_current,
            n_translations=n_trans_current,
            translations=np.asarray(current_translations),
            new_resolution=pixel_res,
            max_posterior_per_image=combined_max_posterior,
        )

        # Track frac_changed for local search fallback
        from recovar.em.dense_single_volume.convergence import compute_assignment_changes
        frac_changed = compute_assignment_changes(
            current_combined_ha, previous_combined_ha,
            n_rot_current, n_trans_current,
            current_healpix_order,
        )
        state._last_frac_changed = frac_changed

        # Save assignments for next iteration's change tracking.
        # Use coarse_ha (indexed into effective_rotations/current_rotations)
        # so that local search and convergence detection work correctly
        # regardless of whether adaptive oversampling was used.
        previous_assignments = [
            ha.copy() if ha is not None else None
            for ha in coarse_ha
        ]

        # --- Timing ---
        elapsed = time.time() - t0
        wall_times.append(elapsed)

        res_angstrom = shell_index_to_resolution_angstrom(
            pixel_res, cryo.image_shape[0], cryo.voxel_size,
        )
        logger.info(
            "RELION Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, ave_Pmax=%.4f, healpix_order=%d, "
            "converged=%s, time=%.1fs",
            iteration + 1, cs, pixel_res, res_angstrom,
            ave_pmax, state.healpix_order,
            state.has_converged, elapsed,
        )

        if state.has_converged:
            # RELION does one final iteration at full resolution with
            # joined half-sets.  For now, we just log and break.
            logger.info(
                "Convergence reached at iteration %d. "
                "Final resolution: %.2f A (pixel_res=%.1f)",
                iteration + 1, res_angstrom, pixel_res,
            )
            break

        iteration += 1

    # --- Final merged mean ---
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
        # RELION-mode specific outputs
        "convergence_state": state,
        "data_vs_prior_trajectory": data_vs_prior_trajectory,
        "healpix_order_trajectory": healpix_order_trajectory,
        "ave_Pmax_trajectory": ave_Pmax_trajectory,
        "pmax_per_image_history": pmax_per_image_history,
    }
