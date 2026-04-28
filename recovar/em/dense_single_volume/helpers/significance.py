"""Batched significance pruning for adaptive two-pass oversampling.

Runs a coarse E-step and identifies significant (rotation, translation)
pairs per image without materializing the full weight matrix.
Called by ``refine_single_volume`` and ``_run_relion_iteration_loop`` in ``refine.py``.
"""

import jax.numpy as jnp
import numpy as np


def _parse_int_set_env(name):
    value = __import__("os").environ.get(name)
    if not value:
        return None
    out = set()
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def _maybe_dump_significance_batch(
    *,
    experiment_dataset,
    indices,
    batch_weights,
    batch_sig_mask,
    batch_n_sig,
    hard_assignment_batch,
    log_z,
    best_score,
    max_posterior,
    rotations,
    translations,
    rotation_log_prior,
    batch_translation_log_prior,
    current_size,
    adaptive_fraction,
    max_significants,
    scores_pre_prior_full=None,
    scores_with_prior_full=None,
    dump_target_positions=None,
    shifted_data=None,
    ctf2_data=None,
    batch_norm=None,
    window_indices=None,
    half_weights_used=None,
):
    """Env-gated debug dump for RELION pass-1 significance parity."""
    import os

    dump_dir = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_DIR")
    if not dump_dir:
        return
    target_original_indices = _parse_int_set_env("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return
    target_current_size = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return

    local_indices = np.asarray(indices, dtype=np.int64)
    original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
    if original_indices_all is None:
        original_indices = local_indices
    else:
        original_indices = np.asarray(original_indices_all, dtype=np.int64)[local_indices]

    os.makedirs(dump_dir, exist_ok=True)
    n_trans = int(translations.shape[0])
    n_candidates = int(batch_weights.shape[1])
    flat_indices = np.arange(n_candidates, dtype=np.int32)
    rot_indices = (flat_indices // n_trans).astype(np.int32)
    trans_indices = (flat_indices % n_trans).astype(np.int32)

    for local_pos, original_idx in enumerate(original_indices):
        if int(original_idx) not in target_original_indices:
            continue
        weights = np.asarray(batch_weights[local_pos], dtype=np.float64)
        sig_mask = np.asarray(batch_sig_mask[local_pos], dtype=bool)
        trans_prior = None
        if batch_translation_log_prior is not None:
            prior_arr = np.asarray(batch_translation_log_prior)
            trans_prior = prior_arr if prior_arr.ndim == 1 else prior_arr[local_pos]
        dump_row = None
        if dump_target_positions is not None:
            matches = np.flatnonzero(np.asarray(dump_target_positions, dtype=np.int64) == int(local_pos))
            if matches.size:
                dump_row = int(matches[0])
        image_rows = slice(local_pos * n_trans, (local_pos + 1) * n_trans)
        ctf2_arr = None if ctf2_data is None else np.asarray(ctf2_data)
        if ctf2_arr is not None and ctf2_arr.shape[0] == local_indices.shape[0]:
            ctf2_target = ctf2_arr[local_pos : local_pos + 1]
        elif ctf2_arr is not None:
            ctf2_target = ctf2_arr[image_rows]
        else:
            ctf2_target = None
        out_path = os.path.join(
            dump_dir,
            f"significance_orig{int(original_idx):06d}_cs{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        np.savez_compressed(
            out_path,
            original_index=np.int64(original_idx),
            local_index=np.int64(local_indices[local_pos]),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            adaptive_fraction=np.float64(adaptive_fraction),
            max_significants=np.int64(max_significants),
            n_rot=np.int64(rotations.shape[0]),
            n_trans=np.int64(n_trans),
            weights_full=weights,
            significant_mask=sig_mask,
            significant_indices=np.flatnonzero(sig_mask).astype(np.int32),
            n_significant=np.int64(batch_n_sig[local_pos]),
            hard_assignment=np.int64(hard_assignment_batch[local_pos]),
            normalization_log_z=np.float64(log_z[local_pos]),
            best_score=np.float64(best_score[local_pos]),
            max_posterior=np.float64(max_posterior[local_pos]),
            rotations=np.asarray(rotations, dtype=np.float32),
            translations=np.asarray(translations, dtype=np.float32),
            rot_indices=rot_indices,
            trans_indices=trans_indices,
            rotation_log_prior=(
                np.asarray(rotation_log_prior, dtype=np.float64)
                if rotation_log_prior is not None
                else np.empty((0,), dtype=np.float64)
            ),
            translation_log_prior=(
                np.asarray(trans_prior, dtype=np.float64)
                if trans_prior is not None
                else np.empty((0,), dtype=np.float64)
            ),
            scores_pre_prior_full=(
                np.asarray(scores_pre_prior_full[dump_row], dtype=np.float64)
                if scores_pre_prior_full is not None and dump_row is not None
                else np.empty((0,), dtype=np.float64)
            ),
            scores_with_prior_full=(
                np.asarray(scores_with_prior_full[dump_row], dtype=np.float64)
                if scores_with_prior_full is not None and dump_row is not None
                else np.empty((0,), dtype=np.float64)
            ),
            shifted_data=(
                np.asarray(shifted_data[image_rows], dtype=np.complex128)
                if shifted_data is not None
                else np.empty((0,), dtype=np.complex128)
            ),
            ctf2_data=(
                np.asarray(ctf2_target, dtype=np.float64)
                if ctf2_target is not None
                else np.empty((0,), dtype=np.float64)
            ),
            batch_norm=(
                np.asarray(batch_norm[local_pos], dtype=np.float64)
                if batch_norm is not None
                else np.empty((0,), dtype=np.float64)
            ),
            window_indices=(
                np.asarray(window_indices, dtype=np.int32)
                if window_indices is not None
                else np.empty((0,), dtype=np.int32)
            ),
            half_weights=(
                np.asarray(half_weights_used, dtype=np.float64)
                if half_weights_used is not None
                else np.empty((0,), dtype=np.float64)
            ),
        )


def _uses_relion_background_fill(experiment_dataset) -> bool:
    image_source = getattr(experiment_dataset, "image_source", None)
    while hasattr(image_source, "parent"):
        image_source = image_source.parent
    backend = getattr(image_source, "backend", image_source)
    return getattr(backend, "image_mask_mode", None) == "relion_background_fill"


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
    do_gridding_correction=False,
    square_window=False,
    use_float64_scoring=False,
    return_full_stats=False,
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
    full_stats : dict[str, np.ndarray], optional
        Returned only when ``return_full_stats=True``.  Contains the full
        coarse-grid log normalizer and best-pose statistics before any
        significant-pose pruning.  RELION os0 uses these full-grid weights for
        Pmax / weight_norm, while ``significant_weight`` only gates
        reconstruction.
    """
    from recovar.core import fourier_transform_utils
    from recovar.core.configs import ForwardModelConfig
    from recovar import core
    from recovar.reconstruction import noise as noise_utils
    from recovar.em.dense_single_volume.em_engine import (
        _compute_projections_block,
        _e_step_block_scores,
        _e_step_block_scores_windowed,
        _preprocess_batch,
        _update_logsumexp,
    )
    from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
    from recovar.em.dense_single_volume.helpers.oversampling import (
        find_significant_rotations as _find_sig,
    )
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
    from recovar.em.dense_single_volume.helpers.image_shifts import (
        apply_relion_integer_pre_shifts,
        integer_pre_shifts_or_none,
    )

    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            mean,
            experiment_dataset.volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
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

    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=False,
    )
    use_window = window_spec.use_window
    window_indices = window_spec.score_indices
    n_windowed = window_spec.n_score
    projection_kwargs = window_spec.projection_kwargs()
    if use_window:
        half_weights_windowed = window_spec.score_values(half_weights)

    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        if use_window:
            half_weights_windowed = window_spec.score_values(half_weights)

    use_relion_numpy_preprocess = (
        _uses_relion_background_fill(experiment_dataset)
        and __import__("os").environ.get("RECOVAR_RELION_NUMPY_IMAGE_FFT") == "1"
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    norm_half_weights = make_half_image_weights(image_shape)

    def _preprocess_batch_relion_numpy(batch_data, ctf_params, batch_size):
        processed_half = experiment_dataset.process_images_half(
            np.asarray(batch_data),
            apply_image_mask=score_with_masked_images,
        )
        processed_half = jnp.asarray(processed_half)
        ctf_half = config.compute_ctf_half(ctf_params)
        ctf2_over_nv_half = ctf_half**2 / noise_variance_half
        ctf_weighted = processed_half * ctf_half / noise_variance_half
        translations_tiled = jnp.repeat(jnp.asarray(translations)[None], batch_size, axis=0).reshape(
            batch_size * n_trans,
            -1,
        )
        weighted_tiled = jnp.repeat(ctf_weighted[:, None, :], n_trans, axis=1).reshape(
            batch_size * n_trans,
            -1,
        )
        shifted_half = core.translate_images(
            weighted_tiled,
            translations_tiled,
            image_shape,
            half_image=True,
        )
        batch_norm = jnp.sum(
            (jnp.abs(processed_half) ** 2 / noise_variance_half) * norm_half_weights[None, :],
            axis=-1,
            keepdims=True,
        ).real
        return shifted_half, batch_norm, ctf2_over_nv_half

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
    normalization_log_z = np.empty(n_images, dtype=np.float64) if return_full_stats else None
    log_evidence = np.empty(n_images, dtype=np.float32) if return_full_stats else None
    best_log_score = np.empty(n_images, dtype=np.float32) if return_full_stats else None
    max_posterior = np.empty(n_images, dtype=np.float32) if return_full_stats else None

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
        integer_pre_shifts = integer_pre_shifts_or_none(image_pre_shifts, indices, batch=batch_data)
        real_space_pre_shift_applied = integer_pre_shifts is not None
        if real_space_pre_shift_applied:
            batch_data = apply_relion_integer_pre_shifts(batch_data, integer_pre_shifts)
        batch_data = jnp.asarray(batch_data)
        if translation_log_prior is None:
            batch_translation_log_prior = None
        elif translation_log_prior.ndim == 1:
            batch_translation_log_prior = jnp.asarray(translation_log_prior)
        else:
            batch_translation_log_prior = jnp.asarray(
                translation_log_prior[start_idx:end_idx],
            )

        if use_relion_numpy_preprocess:
            shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch_relion_numpy(
                batch_data,
                ctf_params,
                batch_size,
            )
        else:
            shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
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

        if image_pre_shifts is not None and not real_space_pre_shift_applied:
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
            from recovar.em.dense_single_volume.helpers.half_spectrum import make_shell_indices_half as _mshi

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
        else:
            # Diagnostic path for RELION's accelerated kernels: XFLOAT is
            # float unless RELION is compiled with ACC_DOUBLE_PRECISION.
            shifted_half = shifted_half.astype(jnp.complex64)
            ctf2_over_nv_half = ctf2_over_nv_half.astype(jnp.float32)
            if use_window:
                shifted_data = shifted_data.astype(jnp.complex64)
                ctf2_data = ctf2_data.astype(jnp.float32)
            else:
                shifted_data = shifted_half
                ctf2_data = ctf2_over_nv_half

        dump_target_positions = None
        dump_score_pre_prior_blocks = None
        dump_score_with_prior_blocks = None
        if __import__("os").environ.get("RECOVAR_SIGNIFICANCE_DUMP_DIR"):
            target_original_indices = _parse_int_set_env("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
            if target_original_indices:
                local_indices_for_dump = np.asarray(indices, dtype=np.int64)
                original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
                if original_indices_all is None:
                    original_indices_for_dump = local_indices_for_dump
                else:
                    original_indices_for_dump = np.asarray(original_indices_all, dtype=np.int64)[
                        local_indices_for_dump
                    ]
                dump_target_positions = np.flatnonzero(
                    np.isin(original_indices_for_dump, np.fromiter(target_original_indices, dtype=np.int64))
                ).astype(np.int64)
                if dump_target_positions.size:
                    dump_score_pre_prior_blocks = []
                    dump_score_with_prior_blocks = []

        # Pass 1: streaming logsumexp
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size, dtype=jnp.float64)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean_for_proj,
                rots_b,
                image_shape,
                proj_volume_shape,
                disc_type,
                **projection_kwargs,
            )

            if use_window:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
                if not use_float64_scoring:
                    proj_w = proj_w.astype(jnp.complex64)
                    proj_abs2_w = proj_abs2_w.astype(jnp.float32)
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
                if not use_float64_scoring:
                    proj_half_b = proj_half_b.astype(jnp.complex64)
                    proj_abs2_half_b = proj_abs2_half_b.astype(jnp.float32)
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
                mean_for_proj,
                rots_b,
                image_shape,
                proj_volume_shape,
                disc_type,
                **projection_kwargs,
            )

            if use_window:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
                if not use_float64_scoring:
                    proj_w = proj_w.astype(jnp.complex64)
                    proj_abs2_w = proj_abs2_w.astype(jnp.float32)
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
                if not use_float64_scoring:
                    proj_half_b = proj_half_b.astype(jnp.complex64)
                    proj_abs2_half_b = proj_abs2_half_b.astype(jnp.float32)
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

            scores_pre_prior = scores
            if rotation_log_prior_padded is not None:
                scores = scores + jnp.asarray(rotation_log_prior_padded[r0:r1])[None, :, None]

            if batch_translation_log_prior is not None:
                if translation_log_prior.ndim == 1:
                    scores = scores + batch_translation_log_prior[None, None, :]
                else:
                    scores = scores + batch_translation_log_prior[:, None, :]

            if dump_score_pre_prior_blocks is not None and dump_target_positions is not None:
                actual_rot = min(rotation_block_size, n_rot - r0)
                dump_score_pre_prior_blocks.append(
                    np.asarray(scores_pre_prior[dump_target_positions, :actual_rot, :], dtype=np.float64).reshape(
                        dump_target_positions.size,
                        -1,
                    )
                )
                dump_score_with_prior_blocks.append(
                    np.asarray(scores[dump_target_positions, :actual_rot, :], dtype=np.float64).reshape(
                        dump_target_positions.size,
                        -1,
                    )
                )

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
        if return_full_stats:
            log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
            log_z_np = np.asarray(log_Z, dtype=np.float64)
            best_score_np = np.asarray(best_score, dtype=np.float64)
            normalization_log_z[start_idx:end_idx] = log_z_np
            log_evidence[start_idx:end_idx] = (log_z_np + log_score_offset).astype(np.float32)
            best_log_score[start_idx:end_idx] = (best_score_np + log_score_offset).astype(np.float32)
            max_posterior[start_idx:end_idx] = np.exp(best_score_np - log_z_np).astype(np.float32)

        # Concatenate this batch's weights -> (batch_size, n_rot * n_trans)
        batch_weights = np.concatenate(batch_weights_blocks, axis=1)
        dump_scores_pre_prior = (
            np.concatenate(dump_score_pre_prior_blocks, axis=1)
            if dump_score_pre_prior_blocks is not None
            else None
        )
        dump_scores_with_prior = (
            np.concatenate(dump_score_with_prior_blocks, axis=1)
            if dump_score_with_prior_blocks is not None
            else None
        )

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
        if __import__("os").environ.get("RECOVAR_SIGNIFICANCE_DUMP_DIR"):
            best_score_np_for_dump = np.asarray(best_score, dtype=np.float64)
            log_z_np_for_dump = np.asarray(log_Z, dtype=np.float64)
            _maybe_dump_significance_batch(
                experiment_dataset=experiment_dataset,
                indices=indices,
                batch_weights=batch_weights,
                batch_sig_mask=np.asarray(batch_sig_mask, dtype=bool),
                batch_n_sig=np.asarray(batch_n_sig, dtype=np.int64),
                hard_assignment_batch=np.asarray(best_argmax, dtype=np.int64),
                log_z=log_z_np_for_dump,
                best_score=best_score_np_for_dump,
                max_posterior=np.exp(best_score_np_for_dump - log_z_np_for_dump),
                rotations=rotations,
                translations=translations,
                rotation_log_prior=rotation_log_prior,
                batch_translation_log_prior=batch_translation_log_prior,
                current_size=current_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                scores_pre_prior_full=dump_scores_pre_prior,
                scores_with_prior_full=dump_scores_with_prior,
                dump_target_positions=dump_target_positions,
                shifted_data=shifted_data,
                ctf2_data=ctf2_data,
                batch_norm=batch_norm,
                window_indices=window_indices,
                half_weights_used=half_weights_windowed if use_window else half_weights,
            )
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

    full_stats = None
    if return_full_stats:
        full_stats = {
            "normalization_log_z": normalization_log_z,
            "log_evidence_per_image": log_evidence,
            "best_log_score_per_image": best_log_score,
            "max_posterior_per_image": max_posterior,
        }

    if return_significant_sample_indices:
        if return_full_stats:
            return sig_rot_any, n_sig_all, hard_assignment, significant_sample_indices, full_stats
        return sig_rot_any, n_sig_all, hard_assignment, significant_sample_indices
    if return_full_stats:
        return sig_rot_any, n_sig_all, hard_assignment, full_stats
    return sig_rot_any, n_sig_all, hard_assignment
