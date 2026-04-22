"""Batched significance pruning for adaptive two-pass oversampling.

Runs a coarse E-step and identifies significant (rotation, translation)
pairs per image without materializing the full weight matrix.
Called by ``refine_single_volume`` and ``_run_relion_iteration_loop`` in ``refine.py``.
"""

import jax.numpy as jnp
import numpy as np


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
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    use_float64_scoring=False,
):
    """Run coarse E-step and find significant rotations in a memory-efficient way.

    Instead of materializing the full (n_images, n_rot * n_trans) weight matrix,
    this processes one image batch at a time: for each batch, it computes the
    posterior weights, finds significance, and accumulates the union of significant
    rotation indices.

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
    from recovar.core import fourier_transform_utils
    from recovar.core.configs import ForwardModelConfig
    from recovar.em.dense_single_volume.em_engine import (
        _compute_projections_block,
        _e_step_block_scores,
        _e_step_block_scores_windowed,
        _preprocess_batch,
        _update_logsumexp,
        make_half_image_weights,
    )
    from recovar.em.dense_single_volume.helpers.oversampling import (
        find_significant_rotations as _find_sig,
    )

    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            mean,
            experiment_dataset.volume_shape,
            projection_padding_factor,
            do_gridding_correction=False,
            current_size=current_size,
        )
    else:
        mean_for_proj = mean
        proj_volume_shape = experiment_dataset.volume_shape

    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    # TODO(RELION-parity-debt): w=1 matches RELION's incorrect half-sum.
    # See em_engine.py for full explanation. Post-parity: use make_half_image_weights.
    half_weights = (
        jnp.ones(n_half, dtype=jnp.float32) if half_spectrum_scoring else make_half_image_weights(image_shape)
    )

    use_window = current_size is not None and current_size < image_shape[0]
    if use_window:
        from recovar.em.dense_single_volume.helpers.fourier_window import (
            make_fourier_window_indices_np,
        )

        window_indices_np, n_windowed = make_fourier_window_indices_np(image_shape, current_size)
        window_indices = jnp.asarray(window_indices_np)
        half_weights_windowed = half_weights[window_indices]
    else:
        window_indices = None
        n_windowed = n_half

    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        if use_window:
            half_weights_windowed = half_weights[window_indices]

    # Pad rotations
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))], axis=0)
    else:
        rotations_padded = rotations

    # Accumulate results
    sig_rot_any = np.zeros(n_rot, dtype=bool)
    n_sig_all = np.empty(n_images, dtype=np.int32)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    significant_sample_indices = [None] * n_images if return_significant_sample_indices else None

    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}",
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
                f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions",
            )

    if rotation_log_prior is not None:
        rotation_log_prior = np.asarray(rotation_log_prior, dtype=np.float32)
        if rotation_log_prior.shape != (n_rot,):
            raise ValueError(
                f"rotation_log_prior must have shape ({n_rot},), got {rotation_log_prior.shape}",
            )
        if n_rot_padded > n_rot:
            rotation_log_prior_padded = np.concatenate(
                [
                    rotation_log_prior,
                    np.zeros(n_rot_padded - n_rot, dtype=np.float32),
                ]
            )
        else:
            rotation_log_prior_padded = rotation_log_prior
    else:
        rotation_log_prior_padded = None

    image_indices = np.arange(n_images)
    start_idx = 0

    for batch_data, _, _, ctf_params, _, _, indices in experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
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
            batch_data,
            ctf_params,
            noise_variance,
            translations,
            config,
            batch_size,
            n_trans,
            score_with_masked_images,
        )

        if image_corrections is not None:
            batch_corr = jnp.asarray(image_corrections[np.asarray(indices)])
            corr_expanded = jnp.repeat(batch_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            batch_norm = batch_norm * (batch_corr**2)[:, None]

        if scale_corrections is not None:
            batch_scale = jnp.asarray(scale_corrections[np.asarray(indices)])
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]

        if image_pre_shifts is not None:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(indices)])
            lattice_half = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
                image_shape,
                voxel_size=1,
                scaled=True,
            )
            phase_factors = jnp.exp(-2j * jnp.pi * (lattice_half @ batch_shifts.T)).T
            phase_expanded = jnp.repeat(phase_factors, n_trans, axis=0)
            shifted_half = shifted_half * phase_expanded

        # DC exclusion (RELION parity: Minvsigma2[0] = 0)
        if half_spectrum_scoring:
            from recovar.em.dense_single_volume.em_engine import make_shell_indices_half as _mshi

            dc_shell = _mshi(image_shape)
            dc_mask = dc_shell == 0
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

        if use_window:
            shifted_data = shifted_half[:, window_indices]
            ctf2_data = ctf2_over_nv_half[:, window_indices]
        else:
            shifted_data = shifted_half
            ctf2_data = ctf2_over_nv_half

        if use_float64_scoring:
            shifted_half = shifted_half.astype(jnp.complex128)
            ctf2_over_nv_half = ctf2_over_nv_half.astype(jnp.float64)
            if use_window:
                shifted_data = shifted_data.astype(jnp.complex128)
                ctf2_data = ctf2_data.astype(jnp.float64)
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
                mean_for_proj, rots_b, image_shape, proj_volume_shape, disc_type
            )

            if use_window:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
                scores = _e_step_block_scores_windowed(
                    shifted_data,
                    batch_norm,
                    ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            else:
                scores = _e_step_block_scores(
                    shifted_data,
                    batch_norm,
                    ctf2_data,
                    proj_half_b * half_weights,
                    proj_abs2_half_b * half_weights,
                    half_weights,
                    batch_size,
                    n_trans,
                    image_shape,
                    volume_shape,
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
                mean_for_proj, rots_b, image_shape, proj_volume_shape, disc_type
            )

            if use_window:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
                scores = _e_step_block_scores_windowed(
                    shifted_data,
                    batch_norm,
                    ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            else:
                scores = _e_step_block_scores(
                    shifted_data,
                    batch_norm,
                    ctf2_data,
                    proj_half_b * half_weights,
                    proj_abs2_half_b * half_weights,
                    half_weights,
                    batch_size,
                    n_trans,
                    image_shape,
                    volume_shape,
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
            n_rot,
            n_trans,
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
                    significant_sample_indices[int(global_idx)] = np.flatnonzero(batch_sig_mask_np[local_idx]).astype(
                        np.int32
                    )
        start_idx = end_idx

    if return_significant_sample_indices:
        return sig_rot_any, n_sig_all, hard_assignment, significant_sample_indices
    return sig_rot_any, n_sig_all, hard_assignment
