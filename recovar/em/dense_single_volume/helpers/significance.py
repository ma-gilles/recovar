"""Batched significance pruning for adaptive two-pass oversampling.

Runs a coarse E-step and identifies significant (rotation, translation)
pairs per image without materializing the full weight matrix.
Called by ``refine_single_volume`` and ``_run_relion_iteration_loop`` in ``refine.py``.
"""

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.env_flags import parse_env_int_set
from recovar.em.dense_single_volume.helpers.projection import compute_projections_block
from recovar.em.dense_single_volume.helpers.scoring import (
    _e_step_block_scores,
    _e_step_block_scores_windowed,
    _update_logsumexp,
)
from recovar.utils.nvtx_shim import nvtx

_SIGNIFICANCE_SCORE_CACHE_ENV = "RECOVAR_SIGNIFICANCE_SCORE_CACHE"
_SIGNIFICANCE_SCORE_CACHE_MAX_GB_ENV = "RECOVAR_SIGNIFICANCE_SCORE_CACHE_MAX_GB"
_SIGNIFICANCE_SCORE_CACHE_DEFAULT_MAX_GB = 2.0
_SIGNIFICANCE_FUSED_PASS1_ENV = "RECOVAR_PASS1_FUSED"
NVTX_DOMAIN_EM = "recovar_em"


def _pass1_fused_enabled() -> bool:
    """Whether the fused-pass1 fast path is enabled.

    Off by default while the path is being validated. Set
    ``RECOVAR_PASS1_FUSED=1`` to opt in. Bit-identical to the unfused path
    when active (same ops, same order, same dtypes).
    """
    mode = os.environ.get(_SIGNIFICANCE_FUSED_PASS1_ENV, "0").strip().lower()
    return mode in {"1", "true", "yes", "on"}


@partial(
    jax.jit,
    static_argnames=(
        "image_shape",
        "proj_volume_shape",
        "volume_shape",
        "disc_type",
        "use_window",
        "use_float64_scoring",
        "rotation_block_size",
        "batch_size",
        "n_trans",
        "n_windowed",
        "max_r_static",
    ),
)
def _fused_score_priors_logsumexp_block(
    mean_for_proj,
    rots_b,
    shifted_data,
    batch_norm,
    ctf2_data,
    half_weights_for_score,
    window_indices,
    rotation_log_prior_block,
    translation_log_prior_per_image,
    class_log_prior_scalar,
    valid_count,
    class_max,
    class_sum,
    global_max,
    global_sum,
    *,
    image_shape: tuple,
    proj_volume_shape: tuple,
    volume_shape: tuple,
    disc_type: str,
    use_window: bool,
    use_float64_scoring: bool,
    rotation_block_size: int,
    batch_size: int,
    n_trans: int,
    n_windowed: int,
    max_r_static,
):
    """Fused pass-1 inner block: project + score + pad-mask + priors + 2× logsumexp.

    Replaces 4-5 separate JIT dispatches per (image_batch, class, rotation_block)
    with a single compiled boundary. JAX inlines the @jit-d leaf functions
    (compute_projections_block, _e_step_block_scores_windowed, _update_logsumexp)
    when traced from inside another @jit, so this is bit-identical to the
    unfused path while saving ~150ms of per-batch host-side dispatch at
    50k/256 K=1 (~16s/iter → ~2-4s/iter for pass1).
    """
    proj_kwargs = {}
    if use_window and max_r_static is not None:
        proj_kwargs["max_r"] = max_r_static
    proj_half_b, proj_abs2_half_b = compute_projections_block(
        mean_for_proj,
        rots_b,
        image_shape,
        proj_volume_shape,
        disc_type,
        **proj_kwargs,
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
            proj_w * half_weights_for_score,
            proj_abs2_w * half_weights_for_score,
            half_weights_for_score,
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
            proj_half_b * half_weights_for_score,
            proj_abs2_half_b * half_weights_for_score,
            half_weights_for_score,
            batch_size,
            n_trans,
            image_shape,
            volume_shape,
        )

    # Padding mask: -inf for rotations beyond valid_count (= n_rot - r0).
    pad_mask = jnp.arange(rotation_block_size)[None, :, None] < valid_count
    neg_inf = jnp.asarray(-jnp.inf, dtype=scores.dtype)
    scores = jnp.where(pad_mask, scores, neg_inf)

    # Priors: class scalar + rotation block + per-image translation.
    scores = scores + jnp.asarray(class_log_prior_scalar, dtype=scores.real.dtype)
    scores = scores + rotation_log_prior_block[None, :, None]
    scores = scores + translation_log_prior_per_image[:, None, :]

    class_max, class_sum = _update_logsumexp(class_max, class_sum, scores)
    global_max, global_sum = _update_logsumexp(global_max, global_sum, scores)
    return scores, class_max, class_sum, global_max, global_sum


def _significance_score_cache_enabled(n_images, n_classes, n_rot, n_trans, *, use_float64_scoring: bool) -> bool:
    """Whether to keep pass-1 score blocks for reuse in pass 2.

    The cache is exact: it stores the already-prior-adjusted score tensors
    computed for the streaming logsumexp pass and reuses them when forming
    posterior weights/significance masks.  If the estimated tensor footprint is
    too large, callers fall back to the previous recompute path.
    """

    mode = os.environ.get(_SIGNIFICANCE_SCORE_CACHE_ENV, "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    force = mode in {"1", "true", "yes", "on", "force", "always"}
    itemsize = 8 if use_float64_scoring else 4
    estimated_bytes = int(n_images) * int(n_classes) * int(n_rot) * int(n_trans) * itemsize
    max_gb = float(os.environ.get(_SIGNIFICANCE_SCORE_CACHE_MAX_GB_ENV, _SIGNIFICANCE_SCORE_CACHE_DEFAULT_MAX_GB))
    return force or estimated_bytes <= int(max_gb * (1024**3))


def _significance_debug_dump_enabled() -> bool:
    return bool(os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_DIR"))


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
    target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
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


def _maybe_dump_k_class_significance_batch(
    *,
    experiment_dataset,
    indices,
    n_classes: int,
    rotations,
    translations,
    class_weight_mats,
    batch_sig_mask,
    batch_n_sig,
    hard_assignment_batch,
    class_assignment_batch,
    global_log_z,
    class_log_z_values,
    best_score,
    max_posterior,
    rotation_log_prior_padded,
    batch_translation_log_prior,
    class_log_priors,
    current_size,
    adaptive_fraction,
    max_significants,
    target_local_positions=None,
    target_scores_pre_prior_per_class=None,
    target_scores_with_prior_per_class=None,
    shifted_data=None,
    ctf2_data=None,
    window_indices=None,
    half_weights_used=None,
):
    """Env-gated debug dump for the K-class significance pass.

    File naming matches the single-class dump so existing diff tooling works.
    The payload extends the K=1 schema with per-class fields and an explicit
    ``n_classes`` scalar so the user can decode the joint candidate space.
    """

    dump_dir = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_DIR")
    if not dump_dir:
        return
    target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
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
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])

    weights_per_class = np.stack(
        [np.asarray(mat, dtype=np.float64) for mat in class_weight_mats],
        axis=1,
    )
    sig_mask_full = np.asarray(batch_sig_mask, dtype=bool).reshape(
        local_indices.shape[0],
        n_classes,
        n_rot * n_trans,
    )
    class_log_z_stack = np.stack(
        [np.asarray(class_log_z, dtype=np.float64) for class_log_z in class_log_z_values],
        axis=1,
    )

    flat_indices = np.arange(n_classes * n_rot * n_trans, dtype=np.int32)
    class_indices_flat = (flat_indices // (n_rot * n_trans)).astype(np.int32)
    rot_indices_flat = ((flat_indices % (n_rot * n_trans)) // n_trans).astype(np.int32)
    trans_indices_flat = (flat_indices % n_trans).astype(np.int32)

    # Build a map from local_pos to dump-target index (row in
    # target_scores_pre_prior_per_class[c]) so we can pick the right
    # per-class raw-score slab for each saved particle.
    target_pos_to_dump_row = None
    if target_local_positions is not None:
        target_pos_to_dump_row = {int(p): row for row, p in enumerate(np.asarray(target_local_positions).tolist())}

    for local_pos, original_idx in enumerate(original_indices):
        if int(original_idx) not in target_original_indices:
            continue
        weights_full = weights_per_class[local_pos].reshape(-1)
        sig_mask = sig_mask_full[local_pos].reshape(-1)
        sig_indices = np.flatnonzero(sig_mask).astype(np.int32)
        trans_prior = None
        if batch_translation_log_prior is not None:
            prior_arr = np.asarray(batch_translation_log_prior)
            trans_prior = prior_arr if prior_arr.ndim == 1 else prior_arr[local_pos]
        rot_prior_arr = (
            np.asarray(rotation_log_prior_padded, dtype=np.float64)[:, :n_rot]
            if rotation_log_prior_padded is not None
            else None
        )

        # Per-class raw scores (pre-prior and with-prior) for this image,
        # if the engine collected them. Shape per class: (n_rot, n_trans).
        scores_pre_prior_per_class = None
        scores_with_prior_per_class = None
        if target_pos_to_dump_row is not None and target_scores_pre_prior_per_class is not None:
            dump_row = target_pos_to_dump_row.get(int(local_pos))
            if dump_row is not None:
                scores_pre_prior_per_class = np.stack(
                    [np.asarray(arr[dump_row], dtype=np.float64) for arr in target_scores_pre_prior_per_class],
                    axis=0,
                )
                scores_with_prior_per_class = np.stack(
                    [np.asarray(arr[dump_row], dtype=np.float64) for arr in target_scores_with_prior_per_class],
                    axis=0,
                )

        image_rows = slice(local_pos * n_trans, (local_pos + 1) * n_trans)
        shifted_target = None
        if shifted_data is not None:
            shifted_target = np.asarray(shifted_data[image_rows], dtype=np.complex128)
        ctf2_target = None
        if ctf2_data is not None:
            ctf2_arr = np.asarray(ctf2_data)
            ctf2_target = (
                ctf2_arr[local_pos : local_pos + 1]
                if ctf2_arr.shape[0] == local_indices.shape[0]
                else ctf2_arr[image_rows]
            )

        out_path = os.path.join(
            dump_dir,
            f"significance_orig{int(original_idx):06d}_cs{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        save_kwargs = dict(
            original_index=np.int64(original_idx),
            local_index=np.int64(local_indices[local_pos]),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            adaptive_fraction=np.float64(adaptive_fraction),
            max_significants=np.int64(max_significants),
            n_classes=np.int64(n_classes),
            n_rot=np.int64(n_rot),
            n_trans=np.int64(n_trans),
            weights_full=weights_full,
            weights_per_class=weights_per_class[local_pos],
            significant_mask=sig_mask,
            significant_indices=sig_indices,
            n_significant=np.int64(batch_n_sig[local_pos]),
            hard_assignment=np.int64(hard_assignment_batch[local_pos]),
            class_assignment=np.int64(class_assignment_batch[local_pos]),
            normalization_log_z=np.float64(global_log_z[local_pos]),
            class_log_z=class_log_z_stack[local_pos],
            best_score=np.float64(best_score[local_pos]),
            max_posterior=np.float64(max_posterior[local_pos]),
            rotations=np.asarray(rotations, dtype=np.float32),
            translations=np.asarray(translations, dtype=np.float32),
            class_indices=class_indices_flat,
            rot_indices=rot_indices_flat,
            trans_indices=trans_indices_flat,
            class_log_priors=np.asarray(class_log_priors, dtype=np.float64),
            rotation_log_prior=(rot_prior_arr if rot_prior_arr is not None else np.empty((0,), dtype=np.float64)),
            translation_log_prior=(
                np.asarray(trans_prior, dtype=np.float64)
                if trans_prior is not None
                else np.empty((0,), dtype=np.float64)
            ),
            shifted_data=(
                shifted_target
                if shifted_target is not None
                else np.empty((0,), dtype=np.complex128)
            ),
            ctf2_data=(
                np.asarray(ctf2_target, dtype=np.float64)
                if ctf2_target is not None
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
        if scores_pre_prior_per_class is not None:
            # Per-class raw recovar score (= -0.5 * residual in
            # `_e_step_block_scores`; differs from RELION's diff2 by the
            # per-image Xi2/2 constant which cancels in relative pose
            # comparisons). Shape (n_classes, n_rot, n_trans).
            save_kwargs["scores_pre_prior_per_class"] = scores_pre_prior_per_class
            save_kwargs["scores_with_prior_per_class"] = scores_with_prior_per_class
        np.savez_compressed(out_path, **save_kwargs)


def _uses_relion_background_fill(experiment_dataset) -> bool:
    image_source = getattr(experiment_dataset, "image_source", None)
    while hasattr(image_source, "parent"):
        image_source = image_source.parent
    backend = getattr(image_source, "backend", image_source)
    return getattr(backend, "image_mask_mode", None) == "relion_background_fill"


@nvtx.annotate("adaptive.pass1_significance", color="orange", domain=NVTX_DOMAIN_EM)
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
    projection_force_jax=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
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
    from recovar import core
    from recovar.core.configs import ForwardModelConfig
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
    from recovar.em.dense_single_volume.helpers.half_spectrum import (
        make_half_image_weights,
        make_scoring_half_image_weights,
    )
    from recovar.em.dense_single_volume.helpers.image_shifts import (
        apply_relion_integer_pre_shifts,
        integer_pre_shifts_or_none,
        tiled_half_image_phase_factors,
    )
    from recovar.em.dense_single_volume.helpers.oversampling import (
        find_significant_rotations as _find_sig,
    )
    from recovar.em.dense_single_volume.helpers.preprocessing import (
        preprocess_batch as _preprocess_batch,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_projections_block as _compute_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.scoring import (
        _e_step_block_scores,
        _e_step_block_scores_windowed,
        _update_logsumexp,
    )
    from recovar.reconstruction import noise as noise_utils

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

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
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
    projection_kwargs["force_jax"] = bool(projection_force_jax)
    if use_window:
        half_weights_windowed = window_spec.score_values(half_weights)

    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")

    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        if use_window:
            half_weights_windowed = window_spec.score_values(half_weights)

    use_relion_numpy_preprocess = _uses_relion_background_fill(experiment_dataset)
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

    def _score_rotation_block_for_batch(
        *,
        rots_b,
        r0,
        r1,
        shifted_data,
        batch_norm,
        ctf2_data,
        batch_size,
        batch_translation_log_prior,
    ):
        if use_relion_projector:
            proj_half_b, proj_abs2_half_b = _compute_relion_projector_projections_block(
                relion_projector_half,
                jnp.asarray(rots_b),
                image_shape,
                r_max=int(relion_projector_r_max),
                padding_factor=int(projection_padding_factor),
                centered_rows=True,
                dense_scale=True,
            )
        else:
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

        return scores, scores_pre_prior

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

        batch_scale = None
        if scale_corrections is not None:
            batch_scale = jnp.asarray(scale_corrections[np.asarray(indices)])

        if image_corrections is not None:
            batch_corr = jnp.asarray(image_corrections[np.asarray(indices)])
            corr_expanded = jnp.repeat(batch_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            # ``image_corrections`` carries ``(avg_norm/normcorr)*scale``;
            # the image-only ``|F_img|^2`` term must drop ``scale`` so it is
            # not double-counted with the reference-side ``ctf2 *= scale^2``
            # below. Matches em_engine._relion_image_correction_factors and
            # ``ml_optimiser.cpp:6240,7298,8516``.
            norm_corr = batch_corr if batch_scale is None else batch_corr / batch_scale
            batch_norm = batch_norm * (norm_corr**2)[:, None]

        if batch_scale is not None:
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]

        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(indices)])
            phase_expanded = tiled_half_image_phase_factors(image_shape, batch_shifts, n_trans)
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
        debug_dump_enabled = _significance_debug_dump_enabled()
        if debug_dump_enabled:
            target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
            if target_original_indices:
                local_indices_for_dump = np.asarray(indices, dtype=np.int64)
                original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
                if original_indices_all is None:
                    original_indices_for_dump = local_indices_for_dump
                else:
                    original_indices_for_dump = np.asarray(original_indices_all, dtype=np.int64)[local_indices_for_dump]
                dump_target_positions = np.flatnonzero(
                    np.isin(original_indices_for_dump, np.fromiter(target_original_indices, dtype=np.int64))
                ).astype(np.int64)
                if dump_target_positions.size:
                    dump_score_pre_prior_blocks = []
                    dump_score_with_prior_blocks = []

        # Pass 1: streaming logsumexp
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size, dtype=jnp.float64)
        cache_score_blocks = (
            _significance_score_cache_enabled(
                batch_size,
                1,
                n_rot_padded,
                n_trans,
                use_float64_scoring=use_float64_scoring,
            )
            and not debug_dump_enabled
        )
        cached_score_blocks = [] if cache_score_blocks else None

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            scores, _ = _score_rotation_block_for_batch(
                rots_b=rots_b,
                r0=r0,
                r1=r1,
                shifted_data=shifted_data,
                batch_norm=batch_norm,
                ctf2_data=ctf2_data,
                batch_size=batch_size,
                batch_translation_log_prior=batch_translation_log_prior,
            )
            if cached_score_blocks is not None:
                cached_score_blocks.append(scores)
            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)

        log_Z = max_s + jnp.log(sum_exp)

        # Pass 2: reuse pass-1 scores when memory allows, then normalize.
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)
        batch_weights_blocks = []

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            if cached_score_blocks is not None:
                scores = cached_score_blocks[b]
                scores_pre_prior = None
            else:
                scores, scores_pre_prior = _score_rotation_block_for_batch(
                    rots_b=rots_b,
                    r0=r0,
                    r1=r1,
                    shifted_data=shifted_data,
                    batch_norm=batch_norm,
                    ctf2_data=ctf2_data,
                    batch_size=batch_size,
                    batch_translation_log_prior=batch_translation_log_prior,
                )

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
            batch_weights_blocks.append(block_probs.reshape(batch_size, -1))

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)
        if return_full_stats:
            log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
            log_z_np = np.asarray(log_Z, dtype=np.float64)
            best_score_np = np.asarray(best_score, dtype=np.float64)
            normalization_log_z[start_idx:end_idx] = log_z_np
            log_evidence[start_idx:end_idx] = (log_z_np + log_score_offset).astype(np.float32)
            best_log_score[start_idx:end_idx] = (best_score_np + log_score_offset).astype(np.float32)
            max_posterior[start_idx:end_idx] = np.exp(best_score_np - log_z_np).astype(np.float32)

        # Concatenate this batch's weights -> (batch_size, n_rot * n_trans).
        batch_weights = jnp.concatenate(batch_weights_blocks, axis=1)
        dump_scores_pre_prior = (
            np.concatenate(dump_score_pre_prior_blocks, axis=1) if dump_score_pre_prior_blocks is not None else None
        )
        dump_scores_with_prior = (
            np.concatenate(dump_score_with_prior_blocks, axis=1) if dump_score_with_prior_blocks is not None else None
        )

        # Find significance for this batch
        batch_sig_mask, batch_sig_rot_mask, batch_n_sig = _find_sig(
            batch_weights,
            n_rot,
            n_trans,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )

        # Accumulate global union of significant rotations
        batch_sig_rot_any = np.asarray(jnp.any(batch_sig_rot_mask, axis=0))
        sig_rot_any |= batch_sig_rot_any

        n_sig_all[start_idx:end_idx] = np.asarray(batch_n_sig)
        if debug_dump_enabled:
            batch_weights_np = np.asarray(batch_weights)
            best_score_np_for_dump = np.asarray(best_score, dtype=np.float64)
            log_z_np_for_dump = np.asarray(log_Z, dtype=np.float64)
            _maybe_dump_significance_batch(
                experiment_dataset=experiment_dataset,
                indices=indices,
                batch_weights=batch_weights_np,
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


@nvtx.annotate("kclass.adaptive.pass1_significance", color="orange", domain=NVTX_DOMAIN_EM)
def _compute_k_class_significance_batched(
    experiment_dataset,
    means,
    noise_variance,
    rotations,
    translations,
    disc_type,
    *,
    class_log_priors,
    adaptive_fraction,
    max_significants,
    image_batch_size,
    rotation_block_size,
    current_size,
    score_with_masked_images=False,
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
    relion_projector_half=None,
    relion_projector_r_max=None,
    score_mode: str = "gaussian",
    collect_significance: bool = True,
    return_class_best: bool = False,
):
    """Find significant samples from one posterior over ``class x rotation x translation``."""

    from recovar import core
    from recovar.core.configs import ForwardModelConfig
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
    from recovar.em.dense_single_volume.helpers.half_spectrum import (
        make_half_image_weights,
        make_scoring_half_image_weights,
    )
    from recovar.em.dense_single_volume.helpers.image_shifts import (
        apply_relion_integer_pre_shifts,
        integer_pre_shifts_or_none,
        tiled_half_image_phase_factors,
    )
    from recovar.em.dense_single_volume.helpers.oversampling import (
        find_significant_rotations as _find_sig,
    )
    from recovar.em.dense_single_volume.helpers.preprocessing import (
        preprocess_batch as _preprocess_batch,
    )
    from recovar.em.dense_single_volume.helpers.preprocessing import (
        preprocess_batch_firstiter_cc as _preprocess_batch_firstiter_cc,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_projections_block as _compute_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.scoring import (
        _e_step_block_scores,
        _e_step_block_scores_normalized_cc,
        _e_step_block_scores_windowed,
        _e_step_block_scores_windowed_normalized_cc,
        _update_logsumexp,
    )
    from recovar.reconstruction import noise as noise_utils

    score_mode = str(score_mode)
    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {score_mode!r}")
    means_array = jnp.asarray(means)
    if means_array.ndim != 2:
        raise ValueError(f"means must have shape (n_classes, volume_size), got {means_array.shape}")
    n_classes = int(means_array.shape[0])
    class_log_priors_np = np.asarray(class_log_priors, dtype=np.float64).reshape(-1)
    if class_log_priors_np.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {class_log_priors_np.shape}")

    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    n_images = int(experiment_dataset.n_units)
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    n_half = int(image_shape[0] * (image_shape[1] // 2 + 1))

    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
    if use_relion_projector:
        relion_projector_half = jnp.asarray(relion_projector_half)
        if relion_projector_half.ndim != 4 or int(relion_projector_half.shape[0]) != n_classes:
            raise ValueError(
                "relion_projector_half must have shape "
                f"({n_classes}, z, y, x_half), got {relion_projector_half.shape}",
            )

    if projection_padding_factor > 1 and not use_relion_projector:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        means_for_proj = []
        proj_volume_shape = None
        for class_index in range(n_classes):
            mean_for_proj, proj_volume_shape = pad_volume_for_projection(
                means_array[class_index],
                experiment_dataset.volume_shape,
                projection_padding_factor,
                do_gridding_correction=do_gridding_correction,
                current_size=current_size,
            )
            means_for_proj.append(mean_for_proj)
    else:
        means_for_proj = [means_array[class_index] for class_index in range(n_classes)]
        proj_volume_shape = experiment_dataset.volume_shape

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )
    window_spec_kwargs = {}
    if score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=False,
        **window_spec_kwargs,
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

    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate(
            [rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))],
            axis=0,
        )
    else:
        rotations_padded = rotations

    rotation_log_prior_padded = None
    if rotation_log_prior is not None:
        prior = np.asarray(rotation_log_prior, dtype=np.float32)
        if prior.ndim == 1:
            if prior.shape != (n_rot,):
                raise ValueError(f"rotation_log_prior must have shape ({n_rot},), got {prior.shape}")
            prior = np.broadcast_to(prior[None, :], (n_classes, n_rot)).copy()
        elif prior.shape != (n_classes, n_rot):
            raise ValueError(
                f"rotation_log_prior must have shape ({n_rot},) or ({n_classes}, {n_rot}), got {prior.shape}",
            )
        if n_rot_padded > n_rot:
            rotation_log_prior_padded = np.pad(
                prior,
                ((0, 0), (0, n_rot_padded - n_rot)),
                mode="constant",
            )
        else:
            rotation_log_prior_padded = prior

    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}"
                )
        elif translation_log_prior.ndim == 2:
            if translation_log_prior.shape != (n_images, n_trans):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_images}, {n_trans}) when image-specific, got {translation_log_prior.shape}",
                )
        else:
            raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions")

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    norm_half_weights = make_half_image_weights(image_shape)
    use_relion_numpy_preprocess = _uses_relion_background_fill(experiment_dataset)

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

    def _score_block(class_index, mean_for_proj, rots_b, shifted_data, batch_norm, ctf2_data, batch_size):
        if use_relion_projector:
            proj_half_b, proj_abs2_half_b = _compute_relion_projector_projections_block(
                relion_projector_half[class_index],
                rots_b,
                image_shape,
                r_max=int(relion_projector_r_max),
                padding_factor=int(projection_padding_factor),
                centered_rows=True,
                dense_scale=True,
            )
        else:
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
            if score_mode == "normalized_cc":
                return _e_step_block_scores_windowed_normalized_cc(
                    shifted_data,
                    batch_norm,
                    ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            return _e_step_block_scores_windowed(
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
        if not use_float64_scoring:
            proj_half_b = proj_half_b.astype(jnp.complex64)
            proj_abs2_half_b = proj_abs2_half_b.astype(jnp.float32)
        if score_mode == "normalized_cc":
            return _e_step_block_scores_normalized_cc(
                shifted_data,
                batch_norm,
                ctf2_data,
                proj_half_b * half_weights,
                proj_abs2_half_b * half_weights,
                batch_size,
                n_trans,
                image_shape,
                volume_shape,
            )
        return _e_step_block_scores(
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

    def _add_priors(scores, class_index, r0, r1, batch_translation_log_prior):
        if score_mode == "normalized_cc":
            return scores
        scores = scores + jnp.asarray(class_log_priors_np[class_index], dtype=scores.real.dtype)
        if rotation_log_prior_padded is not None:
            scores = scores + jnp.asarray(rotation_log_prior_padded[class_index, r0:r1])[None, :, None]
        if batch_translation_log_prior is not None:
            if translation_log_prior.ndim == 1:
                scores = scores + batch_translation_log_prior[None, None, :]
            else:
                scores = scores + batch_translation_log_prior[:, None, :]
        return scores

    sig_rot_any = np.zeros((n_classes, n_rot), dtype=bool)
    n_sig_all = np.empty(n_images, dtype=np.int32)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    class_assignment = np.empty(n_images, dtype=np.int32)
    significant_sample_indices = [[None] * n_images for _ in range(n_classes)] if collect_significance else None
    normalization_log_z = np.empty(n_images, dtype=np.float64)
    normalization_log_evidence = np.empty(n_images, dtype=np.float64)
    log_evidence = np.empty(n_images, dtype=np.float32)
    best_log_score = np.empty(n_images, dtype=np.float32)
    max_posterior = np.empty(n_images, dtype=np.float32)
    class_log_evidence = np.empty((n_classes, n_images), dtype=np.float64)
    class_best_log_score = (
        np.empty((n_classes, n_images), dtype=np.float32) if return_class_best else None
    )
    class_hard_assignment = (
        np.empty((n_classes, n_images), dtype=np.int32) if return_class_best else None
    )

    start_idx = 0
    image_indices = np.arange(n_images)
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
            batch_translation_log_prior = jnp.asarray(translation_log_prior[start_idx:end_idx])

        if score_mode == "normalized_cc":
            cc_window_indices = window_indices if use_window else None
            score_complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
            score_real_dtype = jnp.float64 if use_float64_scoring else jnp.float32
            shifted_half, batch_norm, ctf2_half_score, ctf2_over_nv_half = _preprocess_batch_firstiter_cc(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
                window_indices=cc_window_indices,
                score_complex_dtype=score_complex_dtype,
                score_real_dtype=score_real_dtype,
                norm_real_dtype=jnp.float64,
            )
        elif use_relion_numpy_preprocess:
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
        batch_scale = None
        if scale_corrections is not None:
            batch_scale = jnp.asarray(scale_corrections[np.asarray(indices)])
        if image_corrections is not None:
            batch_corr = jnp.asarray(image_corrections[np.asarray(indices)])
            corr_expanded = jnp.repeat(batch_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            # ``image_corrections`` carries ``(avg_norm/normcorr) * scale``;
            # ``scale_corrections`` carries ``scale``. The image-only
            # ``|F_img|^2`` term must be weighted by ``(avg_norm/normcorr)^2``
            # alone — divide ``batch_corr`` by ``batch_scale`` to isolate it.
            # Otherwise ``batch_norm`` picks up an extra ``scale^2`` that is
            # already accounted for on the reference side via
            # ``ctf2_over_nv_half *= batch_scale^2`` below, double-counting
            # ``scale^2`` in the Wiener score offset. See
            # ``em_engine._relion_image_correction_factors`` and
            # ``ml_optimiser.cpp:6240,7298,8516``.
            norm_corr = batch_corr if batch_scale is None else batch_corr / batch_scale
            batch_norm = batch_norm * (norm_corr**2)[:, None]
        if batch_scale is not None:
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
            if score_mode == "normalized_cc":
                ctf2_half_score = ctf2_half_score * (batch_scale**2)[:, None]
        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(indices)])
            shifted_half = shifted_half * tiled_half_image_phase_factors(image_shape, batch_shifts, n_trans)
        if score_mode == "normalized_cc":
            inv_xi2 = 1.0 / jnp.maximum(batch_norm, jnp.asarray(1e-30, dtype=batch_norm.dtype))
            score_weight_half = ctf2_half_score * inv_xi2
            shifted_half = shifted_half * jnp.repeat(inv_xi2, n_trans, axis=0)
        else:
            score_weight_half = ctf2_over_nv_half
        if half_spectrum_scoring and score_mode != "normalized_cc":
            from recovar.em.dense_single_volume.helpers.half_spectrum import make_shell_indices_half as _mshi

            dc_mask = _mshi(image_shape) == 0
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            score_weight_half = jnp.where(dc_mask[None, :], 0.0, score_weight_half)
        if use_window:
            shifted_data = shifted_half[:, window_indices]
            ctf2_data = score_weight_half[:, window_indices]
        else:
            shifted_data = shifted_half
            ctf2_data = score_weight_half
        if use_float64_scoring:
            shifted_data = shifted_data.astype(jnp.complex128)
            ctf2_data = ctf2_data.astype(jnp.float64)
        else:
            shifted_data = shifted_data.astype(jnp.complex64)
            ctf2_data = ctf2_data.astype(jnp.float32)

        # Identify per-batch dump target rows so we can record raw scores
        # (pre-prior) for each target image inside the per-class block loop.
        # This enables direct diff against RELION's exp_Mweight_diff2
        # without needing the full (batch, n_classes, n_rot*n_trans) cache.
        dump_target_local_positions = None
        if _significance_debug_dump_enabled():
            _dump_targets = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
            if _dump_targets:
                _local_for_dump = np.asarray(indices, dtype=np.int64)
                _orig_all = getattr(experiment_dataset, "dataset_indices", None)
                _orig = _local_for_dump if _orig_all is None else np.asarray(_orig_all, dtype=np.int64)[_local_for_dump]
                _positions = np.flatnonzero(np.isin(_orig, np.fromiter(_dump_targets, dtype=np.int64)))
                if _positions.size:
                    dump_target_local_positions = _positions.astype(np.int64)
        # Per-class collectors for raw (pre-prior) score blocks at target rows.
        # Shape after concat per class: (n_targets, n_rot, n_trans)
        dump_target_pre_prior_blocks_per_class = (
            [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
        )
        dump_target_with_prior_blocks_per_class = (
            [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
        )

        global_max = jnp.full(batch_size, -jnp.inf)
        global_sum = jnp.zeros(batch_size, dtype=jnp.float64)
        class_max_values = []
        class_sum_values = []
        best_score_batch = jnp.full(batch_size, -jnp.inf)
        best_argmax_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        best_class_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        class_best_scores = [jnp.full(batch_size, -jnp.inf) for _ in range(n_classes)] if return_class_best else None
        class_best_argmaxes = [jnp.zeros(batch_size, dtype=jnp.int32) for _ in range(n_classes)] if return_class_best else None
        cache_score_blocks = collect_significance and _significance_score_cache_enabled(
            batch_size,
            n_classes,
            n_rot_padded,
            n_trans,
            use_float64_scoring=use_float64_scoring,
        )
        cached_class_score_blocks = [] if cache_score_blocks else None

        # ``RECOVAR_PASS1_FUSED=1`` swaps the per-block 4-5 separate JIT
        # dispatches (project/score, padding-mask, add-priors, 2× logsumexp)
        # for one fused @jit call. Bit-identical when active; disabled if any
        # debug-dump path is on so per-block pre/post-prior captures still
        # have access to intermediate scores.
        use_fused_pass1 = (
            _pass1_fused_enabled()
            and score_mode == "gaussian"
            and not use_relion_projector
            and dump_target_pre_prior_blocks_per_class is None
            and dump_target_with_prior_blocks_per_class is None
        )

        # Precompute fused-path inputs once per batch (constant across class/block).
        if use_fused_pass1:
            _fused_half_weights = half_weights_windowed if use_window else half_weights
            _fused_max_r_static = projection_kwargs.get("max_r", None) if use_window else None
            _fused_window_indices = window_indices if use_window else jnp.zeros(0, dtype=jnp.int32)
            if translation_log_prior is None:
                _fused_trans_lp_per_image = jnp.zeros((batch_size, n_trans), dtype=jnp.float32)
            elif translation_log_prior.ndim == 1:
                _fused_trans_lp_per_image = jnp.broadcast_to(
                    jnp.asarray(batch_translation_log_prior, dtype=jnp.float32),
                    (batch_size, n_trans),
                )
            else:
                _fused_trans_lp_per_image = jnp.asarray(batch_translation_log_prior, dtype=jnp.float32)

        for class_index, mean_for_proj in enumerate(means_for_proj):
            class_max = jnp.full(batch_size, -jnp.inf)
            class_sum = jnp.zeros(batch_size, dtype=jnp.float64)
            cached_score_blocks = [] if cached_class_score_blocks is not None else None
            for block_index in range(n_blocks):
                r0 = block_index * rotation_block_size
                r1 = r0 + rotation_block_size
                if use_fused_pass1:
                    valid_count = jnp.asarray(min(rotation_block_size, n_rot - r0), dtype=jnp.int32)
                    if rotation_log_prior_padded is None:
                        rot_lp_block = jnp.zeros(rotation_block_size, dtype=jnp.float32)
                    else:
                        rot_lp_block = jnp.asarray(rotation_log_prior_padded[class_index, r0:r1], dtype=jnp.float32)
                    scores, class_max, class_sum, global_max, global_sum = _fused_score_priors_logsumexp_block(
                        mean_for_proj,
                        rotations_padded[r0:r1],
                        shifted_data,
                        batch_norm,
                        ctf2_data,
                        _fused_half_weights,
                        _fused_window_indices,
                        rot_lp_block,
                        _fused_trans_lp_per_image,
                        float(class_log_priors_np[class_index]),
                        valid_count,
                        class_max,
                        class_sum,
                        global_max,
                        global_sum,
                        image_shape=image_shape,
                        proj_volume_shape=proj_volume_shape,
                        volume_shape=volume_shape,
                        disc_type=disc_type,
                        use_window=use_window,
                        use_float64_scoring=use_float64_scoring,
                        rotation_block_size=int(rotation_block_size),
                        batch_size=int(batch_size),
                        n_trans=int(n_trans),
                        n_windowed=int(n_windowed) if use_window else 0,
                        max_r_static=_fused_max_r_static,
                    )
                    if cached_score_blocks is not None:
                        cached_score_blocks.append(scores)
                else:
                    scores = _score_block(
                        class_index,
                        mean_for_proj,
                        rotations_padded[r0:r1],
                        shifted_data,
                        batch_norm,
                        ctf2_data,
                        batch_size,
                    )
                    if r1 > n_rot:
                        valid = n_rot - r0
                        scores = jnp.where(jnp.arange(rotation_block_size)[None, :, None] < valid, scores, -jnp.inf)
                    # Capture pre-prior raw scores for dump targets BEFORE _add_priors.
                    # scores shape: (batch_size, rotation_block_size, n_trans).
                    # For comparison vs RELION exp_Mweight_diff2, recovar's score is
                    # -0.5 * residual where residual = sum_pixel((proj*ctf - shifted_img)² - |img|²)
                    # / sigma² × half_weights. RELION's diff2 has the same core term
                    # plus the per-image Xi2/2 constant. Per-pose RELATIVE differences
                    # cancel the constant, so direct diff is meaningful.
                    if dump_target_pre_prior_blocks_per_class is not None:
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        dump_target_pre_prior_blocks_per_class[class_index].append(
                            np.asarray(
                                scores[dump_target_local_positions, :actual_rot, :],
                                dtype=np.float64,
                            )
                        )
                    scores = _add_priors(scores, class_index, r0, r1, batch_translation_log_prior)
                    if dump_target_with_prior_blocks_per_class is not None:
                        actual_rot = min(rotation_block_size, n_rot - r0)
                        dump_target_with_prior_blocks_per_class[class_index].append(
                            np.asarray(
                                scores[dump_target_local_positions, :actual_rot, :],
                                dtype=np.float64,
                            )
                        )
                    if cached_score_blocks is not None:
                        cached_score_blocks.append(scores)
                    class_max, class_sum = _update_logsumexp(class_max, class_sum, scores)
                    global_max, global_sum = _update_logsumexp(global_max, global_sum, scores)
                block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
                block_argmax = jnp.argmax(scores.reshape(batch_size, -1), axis=1)
                improved = block_best > best_score_batch
                best_score_batch = jnp.where(improved, block_best, best_score_batch)
                best_argmax_batch = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax_batch)
                best_class_batch = jnp.where(improved, class_index, best_class_batch)
                if return_class_best:
                    class_improved = block_best > class_best_scores[class_index]
                    class_best_scores[class_index] = jnp.where(
                        class_improved,
                        block_best,
                        class_best_scores[class_index],
                    )
                    class_best_argmaxes[class_index] = jnp.where(
                        class_improved,
                        block_argmax + r0 * n_trans,
                        class_best_argmaxes[class_index],
                    )
            if cached_class_score_blocks is not None:
                cached_class_score_blocks.append(cached_score_blocks)
            class_max_values.append(class_max)
            class_sum_values.append(class_sum)

        global_log_z = global_max + jnp.log(global_sum)
        class_log_z_values = [
            class_max + jnp.log(class_sum) for class_max, class_sum in zip(class_max_values, class_sum_values)
        ]

        class_weight_mats = []
        if collect_significance:
            for class_index, mean_for_proj in enumerate(means_for_proj):
                class_weight_blocks = []
                for block_index in range(n_blocks):
                    r0 = block_index * rotation_block_size
                    r1 = r0 + rotation_block_size
                    if cached_class_score_blocks is None:
                        scores = _score_block(
                            class_index,
                            mean_for_proj,
                            rotations_padded[r0:r1],
                            shifted_data,
                            batch_norm,
                            ctf2_data,
                            batch_size,
                        )
                        if r1 > n_rot:
                            valid = n_rot - r0
                            scores = jnp.where(jnp.arange(rotation_block_size)[None, :, None] < valid, scores, -jnp.inf)
                        scores = _add_priors(scores, class_index, r0, r1, batch_translation_log_prior)
                    else:
                        scores = cached_class_score_blocks[class_index][block_index]
                    probs = jnp.exp(scores - global_log_z[:, None, None])

                    actual_rot = min(rotation_block_size, n_rot - r0)
                    class_weight_blocks.append(probs[:, :actual_rot, :].reshape(batch_size, -1))
                class_weight_mats.append(jnp.concatenate(class_weight_blocks, axis=1))

            batch_weights = jnp.concatenate(class_weight_mats, axis=1)
            batch_sig_mask, batch_sig_rot_mask, batch_n_sig = _find_sig(
                batch_weights,
                n_classes * n_rot,
                n_trans,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
            )
            batch_sig_mask_np = np.asarray(batch_sig_mask, dtype=bool)
            sig_rot_any |= np.asarray(jnp.any(batch_sig_rot_mask, axis=0), dtype=bool).reshape(n_classes, n_rot)
            n_sig_all[start_idx:end_idx] = np.asarray(batch_n_sig, dtype=np.int32)
        else:
            batch_sig_mask_np = None
            n_sig_all[start_idx:end_idx] = 0

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax_batch, dtype=np.int32)
        class_assignment[start_idx:end_idx] = np.asarray(best_class_batch, dtype=np.int32)

        log_score_offset = -0.5 * np.asarray(jnp.squeeze(batch_norm, axis=1), dtype=np.float64)
        global_log_z_np = np.asarray(global_log_z, dtype=np.float64)
        best_score_np = np.asarray(best_score_batch, dtype=np.float64)
        normalization_log_z[start_idx:end_idx] = global_log_z_np
        normalization_log_evidence[start_idx:end_idx] = global_log_z_np + log_score_offset
        log_evidence[start_idx:end_idx] = normalization_log_evidence[start_idx:end_idx].astype(np.float32)
        best_log_score[start_idx:end_idx] = (best_score_np + log_score_offset).astype(np.float32)
        max_posterior[start_idx:end_idx] = np.exp(best_score_np - global_log_z_np).astype(np.float32)
        for class_index, class_log_z in enumerate(class_log_z_values):
            class_log_evidence[class_index, start_idx:end_idx] = (
                np.asarray(class_log_z, dtype=np.float64) + log_score_offset
            )
        if return_class_best:
            for class_index in range(n_classes):
                class_best_log_score[class_index, start_idx:end_idx] = (
                    np.asarray(class_best_scores[class_index], dtype=np.float64) + log_score_offset
                ).astype(np.float32)
                class_hard_assignment[class_index, start_idx:end_idx] = np.asarray(
                    class_best_argmaxes[class_index],
                    dtype=np.int32,
                )

        if _significance_debug_dump_enabled():
            if not collect_significance:
                raise ValueError("debug significance dumps require collect_significance=True")
            # Concatenate per-class per-block raw scores for the dump targets
            # into per-class arrays of shape (n_targets, n_rot, n_trans).
            target_scores_pre_prior_per_class = None
            target_scores_with_prior_per_class = None
            target_local_positions_for_dump = None
            if dump_target_pre_prior_blocks_per_class is not None:
                target_scores_pre_prior_per_class = [
                    np.concatenate(blocks, axis=1) if blocks else None
                    for blocks in dump_target_pre_prior_blocks_per_class
                ]
                target_scores_with_prior_per_class = [
                    np.concatenate(blocks, axis=1) if blocks else None
                    for blocks in dump_target_with_prior_blocks_per_class
                ]
                target_local_positions_for_dump = dump_target_local_positions
            _maybe_dump_k_class_significance_batch(
                experiment_dataset=experiment_dataset,
                indices=indices,
                n_classes=n_classes,
                rotations=rotations,
                translations=translations,
                class_weight_mats=[np.asarray(mat, dtype=np.float64) for mat in class_weight_mats],
                batch_sig_mask=batch_sig_mask_np,
                batch_n_sig=np.asarray(batch_n_sig, dtype=np.int64),
                hard_assignment_batch=np.asarray(best_argmax_batch, dtype=np.int64),
                class_assignment_batch=np.asarray(best_class_batch, dtype=np.int64),
                global_log_z=global_log_z_np,
                class_log_z_values=class_log_z_values,
                best_score=best_score_np,
                max_posterior=max_posterior[start_idx:end_idx],
                rotation_log_prior_padded=rotation_log_prior_padded,
                batch_translation_log_prior=batch_translation_log_prior,
                class_log_priors=class_log_priors_np,
                current_size=current_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                target_local_positions=target_local_positions_for_dump,
                target_scores_pre_prior_per_class=target_scores_pre_prior_per_class,
                target_scores_with_prior_per_class=target_scores_with_prior_per_class,
                shifted_data=shifted_data,
                ctf2_data=ctf2_data,
                window_indices=window_indices,
                half_weights_used=half_weights_windowed if use_window else half_weights,
            )

        if collect_significance:
            samples_per_class = n_rot * n_trans
            for local_idx, global_idx in enumerate(indices):
                global_idx = int(global_idx)
                for class_index in range(n_classes):
                    c0 = class_index * samples_per_class
                    c1 = c0 + samples_per_class
                    mask = batch_sig_mask_np[local_idx, c0:c1]
                    significant_sample_indices[class_index][global_idx] = (
                        None if np.all(mask) else np.flatnonzero(mask).astype(np.int32)
                    )
        start_idx = end_idx

    full_stats = {
        "normalization_log_z": normalization_log_z,
        "normalization_log_evidence": normalization_log_evidence,
        "log_evidence_per_image": log_evidence,
        "best_log_score_per_image": best_log_score,
        "max_posterior_per_image": max_posterior,
        "class_log_evidence_per_image": class_log_evidence,
        "class_assignments": class_assignment,
    }
    if return_class_best:
        full_stats["class_best_log_score_per_image"] = class_best_log_score
        full_stats["class_hard_assignments"] = class_hard_assignment
    return sig_rot_any, n_sig_all, hard_assignment, class_assignment, significant_sample_indices, full_stats
