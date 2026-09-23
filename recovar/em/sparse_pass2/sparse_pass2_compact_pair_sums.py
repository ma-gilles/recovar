"""Compact-pair weighted sums and active-row selection of the sparse bucketed pass 2.

The probability-weighted image and rotation sums of RELION's ``storeWeightedSums``
counterpart for compact hypothesis pairs (dense, pair-sparse, fused and native
variants), the active flat-row selection and grouping, and the rectangular
active weighted sums. ``sparse_pass2_bucketed`` calls them per bucket.
"""

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.deterministic_reduce import add_segment_sum
from recovar.em.helpers.env_flags import parse_env_flag
from recovar.em.local.local_backprojection import (
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_mstep_sums,
    compute_local_weighted_sums,
    flatten_bucket_rows,
    relion_x_half_sequential_translation_reduction_enabled,
)
from recovar.em.sparse_pass2.sparse_pass2_budget import _dtype_itemsize
from recovar.em.sparse_pass2.sparse_pass2_noise_blocks import (
    _compute_noise_block_and_norm_residual_from_flat_rows,
    _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms,
)
from recovar.em.sparse_pass2.sparse_pass2_policy import (
    _SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV,
    _compact_pair_pair_sparse_mstep_enabled_for_pass,
)

logger = logging.getLogger(__name__)


_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS_ENV = "RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS"


_active_noise_gather_chunk_log_keys: set[tuple[int, int, int, int]] = set()


def _bucket_row_bytes(values) -> int:
    if values is None or len(values.shape) == 0:
        return 1
    n_pixels = int(values.shape[-1]) if len(values.shape) > 1 else 1
    return max(1, n_pixels * _dtype_itemsize(values.dtype))


def _active_flat_gather_chunk_rows(values, ctf_values, flat_rotations, *, max_block_bytes: int | None) -> int:
    if max_block_bytes is None:
        return 2**62
    row_bytes = _bucket_row_bytes(values) + _bucket_row_bytes(ctf_values)
    if flat_rotations is not None and len(flat_rotations.shape) > 1:
        rotation_items = int(np.prod(tuple(int(dim) for dim in flat_rotations.shape[1:])))
        row_bytes += max(1, rotation_items * _dtype_itemsize(flat_rotations.dtype))
    return max(1, int(max_block_bytes) // max(1, row_bytes))


def _compact_pair_valid_weights_and_indices(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    *,
    n_rotation_rows,
    n_trans,
):
    finite_pair_probs = jnp.isfinite(pair_probs)
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    valid_pair = (
        pair_mask
        & finite_pair_probs
        & (safe_rotation_row >= 0)
        & (safe_rotation_row < int(n_rotation_rows))
        & (safe_translation_idx >= 0)
        & (safe_translation_idx < int(n_trans))
    )
    weights = jnp.where(valid_pair, pair_probs, 0.0)
    safe_rotation_row = jnp.where(valid_pair, safe_rotation_row, 0)
    safe_translation_idx = jnp.where(valid_pair, safe_translation_idx, 0)
    return weights, safe_rotation_row, safe_translation_idx, valid_pair


@partial(jax.jit, static_argnames=("n_rotation_rows", "n_trans"))
def _compact_pair_dense_probs(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    n_rotation_rows,
    n_trans,
):
    """Scatter compact pair probabilities into dense scalar row probabilities."""

    batch, n_pairs = pair_probs.shape
    batch_idx = jnp.broadcast_to(jnp.arange(batch, dtype=jnp.int32)[:, None], (batch, n_pairs))

    finite_pair_probs = jnp.isfinite(pair_probs)
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    valid_pair = (
        pair_mask
        & finite_pair_probs
        & (safe_rotation_row >= 0)
        & (safe_rotation_row < int(n_rotation_rows))
        & (safe_translation_idx >= 0)
        & (safe_translation_idx < int(n_trans))
    )
    weights = jnp.where(valid_pair, pair_probs, 0.0)
    scatter_rotation_row = jnp.where(valid_pair, safe_rotation_row, 0)
    scatter_translation_idx = jnp.where(valid_pair, safe_translation_idx, 0)

    dense_probs = jnp.zeros((batch, int(n_rotation_rows), int(n_trans)), dtype=weights.dtype)
    dense_probs = dense_probs.at[batch_idx, scatter_rotation_row, scatter_translation_idx].add(weights)
    return dense_probs


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_sparse_weighted_image_and_prob_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    n_rotation_rows,
):
    """Reduce compact pairs in pair order, then translations in dense order."""

    batch, n_pairs = pair_probs.shape
    n_trans = shifted_recon_split.shape[1]
    n_pixels = shifted_recon_split.shape[-1]
    weights, safe_rotation_row, safe_translation_idx, valid_pair = _compact_pair_valid_weights_and_indices(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=n_trans,
    )
    batch_idx = jnp.arange(batch, dtype=jnp.int32)
    pair_indices = jnp.arange(n_pairs, dtype=jnp.int32)

    summed_dtype = jnp.result_type(weights, shifted_recon_split)
    summed0 = jnp.zeros((batch, int(n_rotation_rows), n_pixels), dtype=summed_dtype)
    probs_sum_t0 = jnp.zeros((batch, int(n_rotation_rows)), dtype=weights.dtype)
    translation_posterior0 = jnp.zeros((batch, n_trans), dtype=weights.dtype)

    def translation_body(carry, trans_idx):
        summed, probs_sum_t, translation_posterior = carry

        def pair_body(row_probs, pair_idx):
            pair_valid = valid_pair[:, pair_idx] & (safe_translation_idx[:, pair_idx] == trans_idx)
            pair_weights = jnp.where(pair_valid, weights[:, pair_idx], 0.0)
            pair_rows = jnp.where(pair_valid, safe_rotation_row[:, pair_idx], 0)
            row_probs = row_probs.at[batch_idx, pair_rows].add(pair_weights)
            return row_probs, None

        row_probs0 = jnp.zeros((batch, int(n_rotation_rows)), dtype=weights.dtype)
        row_probs, _ = jax.lax.scan(pair_body, row_probs0, pair_indices)
        summed = summed + row_probs[:, :, None] * shifted_recon_split[:, trans_idx, :][:, None, :]
        probs_sum_t = probs_sum_t + row_probs
        translation_posterior = translation_posterior.at[:, trans_idx].set(jnp.sum(row_probs, axis=1))
        return (summed, probs_sum_t, translation_posterior), None

    (summed, probs_sum_t, translation_posterior), _ = jax.lax.scan(
        translation_body,
        (summed0, probs_sum_t0, translation_posterior0),
        jnp.arange(n_trans, dtype=jnp.int32),
    )
    return summed, probs_sum_t, translation_posterior


@partial(jax.jit, static_argnames=("n_rotation_rows", "relion_x_half"))
def _compact_pair_weighted_rotation_sums_dense(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
    *,
    relion_x_half=False,
):
    """Accumulate compact-pair M-step stats without forming dense ``(B,R,T)``.

    Returns the dense helper equivalents:
    ``summed = compute_local_weighted_sums(probs, shifted_recon_split)``,
    ``ctf_probs = compute_local_ctf_sums(probs, ctf2_over_nv_recon)``,
    plus ``probs_sum_t`` and ``translation_posterior``.
    """

    dense_probs = _compact_pair_dense_probs(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )

    # Use the same dense weighted-sum primitive as the rectangular path after
    # compacting the scalar probabilities. Scattering complex image rows directly
    # changes GPU accumulation order enough to break x-half M-step parity.
    probs_sum_t = jnp.sum(dense_probs, axis=-1)
    summed, ctf_probs = compute_local_mstep_sums(
        dense_probs,
        shifted_recon_split,
        ctf2_over_nv_recon,
        relion_x_half=bool(relion_x_half),
        default_probs_sum_t=probs_sum_t,
    )
    translation_posterior = jnp.sum(dense_probs, axis=1)
    return summed, ctf_probs, probs_sum_t, translation_posterior


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_rotation_sums_pair_sparse(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    """Experimental compact-pair M-step reduction without dense ``(B,R,T)``."""

    summed, probs_sum_t, translation_posterior = _compact_pair_sparse_weighted_image_and_prob_sums(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        n_rotation_rows=n_rotation_rows,
    )
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv_recon)
    return summed, ctf_probs, probs_sum_t, translation_posterior


def _compact_pair_weighted_rotation_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
    *,
    allow_pair_sparse=True,
    relion_x_half=False,
):
    use_sequential_relion_reduction = bool(
        relion_x_half and relion_x_half_sequential_translation_reduction_enabled()
    )
    impl = (
        _compact_pair_weighted_rotation_sums_pair_sparse
        if (
            not use_sequential_relion_reduction
            and _compact_pair_pair_sparse_mstep_enabled_for_pass(
                allow_pair_sparse=allow_pair_sparse
            )
        )
        else _compact_pair_weighted_rotation_sums_dense
    )
    kwargs = dict(
        n_rotation_rows=n_rotation_rows,
    )
    if impl is _compact_pair_weighted_rotation_sums_dense:
        kwargs["relion_x_half"] = bool(relion_x_half)
    return impl(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        ctf2_over_nv_recon,
        **kwargs,
    )


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_image_sums_dense(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    n_rotation_rows,
):
    """Accumulate compact-pair image weighted sums only.

    This is used when masked scoring makes image sums differ from the M-step
    reconstruction sums, while the CTF/probability reductions can still be
    reused from the M-step.
    """

    dense_probs = _compact_pair_dense_probs(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )
    return compute_local_weighted_sums(dense_probs, shifted_recon_split)


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_image_sums_pair_sparse(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    n_rotation_rows,
):
    summed, _probs_sum_t, _translation_posterior = _compact_pair_sparse_weighted_image_and_prob_sums(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        n_rotation_rows=n_rotation_rows,
    )
    return summed


def _compact_pair_weighted_image_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    n_rotation_rows,
    *,
    allow_pair_sparse=True,
):
    impl = (
        _compact_pair_weighted_image_sums_pair_sparse
        if _compact_pair_pair_sparse_mstep_enabled_for_pass(allow_pair_sparse=allow_pair_sparse)
        else _compact_pair_weighted_image_sums_dense
    )
    return impl(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        n_rotation_rows=n_rotation_rows,
    )


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_rotation_and_image_sums_legacy(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    """Accumulate compact-pair M-step sums plus an alternate image sum.

    The default RELION K-class path scores with masked images but reconstructs
    from unmasked images. Build the compact dense probability tensor once and
    reuse it for both image sums, while keeping the CTF/probability reductions
    identical to ``_compact_pair_weighted_rotation_sums``.
    """

    dense_probs = _compact_pair_dense_probs(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )

    summed = compute_local_weighted_sums(dense_probs, shifted_recon_split)
    summed_image = compute_local_weighted_sums(dense_probs, shifted_image_split)
    probs_sum_t = jnp.sum(dense_probs, axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv_recon)
    translation_posterior = jnp.sum(dense_probs, axis=1)
    return summed, summed_image, ctf_probs, probs_sum_t, translation_posterior


def _compact_pair_weighted_rotation_and_image_sums_pair_sparse(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    summed, ctf_probs, probs_sum_t, translation_posterior = _compact_pair_weighted_rotation_sums_pair_sparse(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        ctf2_over_nv_recon,
        n_rotation_rows=n_rotation_rows,
    )
    summed_image = _compact_pair_weighted_image_sums_pair_sparse(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_image_split,
        n_rotation_rows=n_rotation_rows,
    )
    return summed, summed_image, ctf_probs, probs_sum_t, translation_posterior


@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_rotation_and_image_sums_fused_image_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    """Accumulate compact-pair M-step and image sums in one weighted reduction."""

    dense_probs = _compact_pair_dense_probs(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )

    recon_n_pixels = shifted_recon_split.shape[-1]
    combined_shifted = jnp.concatenate((shifted_recon_split, shifted_image_split), axis=-1)
    combined_summed = compute_local_weighted_sums(dense_probs, combined_shifted)
    summed = combined_summed[..., :recon_n_pixels]
    summed_image = combined_summed[..., recon_n_pixels:]
    probs_sum_t = jnp.sum(dense_probs, axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv_recon)
    translation_posterior = jnp.sum(dense_probs, axis=1)
    return summed, summed_image, ctf_probs, probs_sum_t, translation_posterior


@partial(jax.jit, static_argnames=("n_rotation_rows", "n_trans"))
def _compact_pair_sorted_csr(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    *,
    n_rotation_rows,
    n_trans,
):
    """CSR row offsets for compact pairs that are already in source order.

    ``build_compact_pair_index_arrays`` emits each image's valid pairs as a
    rotation-major, translation-minor prefix (C order of the ``(R, T)`` mask)
    followed by ``-1`` padding, so no sort is needed: masked pairs are mapped
    to row ``n_rotation_rows`` (they sit after every real row) and the offsets
    are one ``searchsorted`` per image. Non-finite or out-of-range pairs keep
    their position with zero weight, which the kernel skips exactly like the
    dense kernel skips zero table entries. A sort over the padded ``(B, P)``
    keys was tried first and cost more than it saved at 100k/256 (job
    13806345: wide-pair class 630 -> 756 s).
    """
    if jnp.asarray(pair_probs).dtype != jnp.float32:
        raise ValueError("native pair-sparse sums require float32 probabilities")
    weights, _safe_rotation_row, safe_translation_idx, _valid_pair = _compact_pair_valid_weights_and_indices(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=n_trans,
    )
    n_rotation_rows = int(n_rotation_rows)
    rows_in_order = jnp.where(
        pair_mask,
        jnp.asarray(local_rotation_row).astype(jnp.int32),
        jnp.int32(n_rotation_rows),
    )
    row_ids = jnp.arange(n_rotation_rows + 1, dtype=jnp.int32)
    row_offsets = jax.vmap(lambda rows: jnp.searchsorted(rows, row_ids, side="left"))(rows_in_order)
    return (
        weights,
        safe_translation_idx.astype(jnp.int32),
        row_offsets.astype(jnp.int32),
    )

@partial(jax.jit, static_argnames=("n_rotation_rows",))
def _compact_pair_weighted_rotation_and_image_sums_native(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
):
    """Use one native launch boundary for two independent weighted sums."""

    from recovar.cuda_backproject import dual_weighted_sums_pairs_f32

    dense_probs = _compact_pair_dense_probs(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )
    weights, pair_translations, row_offsets = _compact_pair_sorted_csr(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=shifted_recon_split.shape[1],
    )
    summed, summed_image = dual_weighted_sums_pairs_f32(
        weights,
        pair_translations,
        row_offsets,
        shifted_recon_split,
        shifted_image_split,
    )
    probs_sum_t = jnp.sum(dense_probs, axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv_recon)
    translation_posterior = jnp.sum(dense_probs, axis=1)
    return summed, summed_image, ctf_probs, probs_sum_t, translation_posterior


@partial(
    jax.jit,
    static_argnames=("n_rotation_rows", "shell_count", "batch_size"),
)
def _compact_pair_weighted_sums_and_noise_native(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    proj_for_noise,
    proj_abs2_for_noise,
    noise_variance_half,
    shell_indices,
    *,
    n_rotation_rows: int,
    shell_count: int,
    batch_size: int,
    flat_rows=None,
    flat_scale_old_scale=None,
    flat_scale_pixel_mask=None,
):
    """Fuse compact weighted sums with dense noise/norm sufficient statistics."""

    if flat_rows is not None:
        return _compact_pair_weighted_sums_and_noise_native_flat_rows(
            pair_probs, local_rotation_row, translation_idx, pair_mask,
            shifted_recon_split, shifted_image_split, ctf2_over_nv_recon,
            proj_for_noise, proj_abs2_for_noise, noise_variance_half,
            shell_indices, flat_rows,
            n_rotation_rows=n_rotation_rows, shell_count=shell_count,
            batch_size=batch_size, scale_old_scale=flat_scale_old_scale,
            scale_pixel_mask=flat_scale_pixel_mask,
        )

    (
        summed,
        summed_image,
        ctf_probs,
        probs_sum_t,
        translation_posterior,
    ) = _compact_pair_weighted_rotation_and_image_sums_native(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        shifted_image_split,
        ctf2_over_nv_recon,
        n_rotation_rows=n_rotation_rows,
    )
    flat_image_indices = jnp.broadcast_to(
        jnp.arange(int(batch_size), dtype=jnp.int32)[:, None],
        (int(batch_size), int(n_rotation_rows)),
    ).reshape(-1)
    block_noise_shells, block_norm_residual = (
        _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms(
            proj_for_noise.reshape((-1, proj_for_noise.shape[-1])),
            proj_abs2_for_noise.reshape((-1, proj_abs2_for_noise.shape[-1])),
            summed_image.reshape((-1, summed_image.shape[-1])),
            ctf_probs.reshape((-1, ctf_probs.shape[-1])),
            noise_variance_half,
            shell_indices,
            flat_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )
    )
    return (
        summed,
        summed_image,
        ctf_probs,
        probs_sum_t,
        translation_posterior,
        block_noise_shells,
        block_norm_residual,
    )


def _compact_pair_weighted_rotation_and_image_sums(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    n_rotation_rows,
    *,
    allow_pair_sparse=True,
):
    """Accumulate compact-pair M-step sums plus an alternate image sum."""

    if _compact_pair_pair_sparse_mstep_enabled_for_pass(allow_pair_sparse=allow_pair_sparse):
        return _compact_pair_weighted_rotation_and_image_sums_pair_sparse(
            pair_probs,
            local_rotation_row,
            translation_idx,
            pair_mask,
            shifted_recon_split,
            shifted_image_split,
            ctf2_over_nv_recon,
            n_rotation_rows=n_rotation_rows,
        )

    impl = (
        _compact_pair_weighted_rotation_and_image_sums_fused_image_sums
        if parse_env_flag(_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS_ENV, default=True)
        else _compact_pair_weighted_rotation_and_image_sums_legacy
    )
    return impl(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        shifted_recon_split,
        shifted_image_split,
        ctf2_over_nv_recon,
        n_rotation_rows=n_rotation_rows,
    )


def _active_flat_row_indices_from_probs_sum_t(
    probs_sum_t,
    *,
    pad_multiple: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return stable active row indices, padded to reduce active-path JIT churn."""

    probs_sum_t_np = np.asarray(jax.device_get(probs_sum_t))
    total_rows = int(probs_sum_t_np.size)
    active_indices = np.flatnonzero(probs_sum_t_np.reshape(-1) != 0.0).astype(np.int32, copy=False)
    active_count = int(active_indices.size)
    if active_count == 0:
        return active_indices, np.zeros((0,), dtype=np.float32), 0

    pad_multiple = max(1, int(pad_multiple))
    if pad_multiple <= 1:
        return active_indices, np.ones((active_count,), dtype=np.float32), active_count

    padded_count = min(
        total_rows,
        ((active_count + pad_multiple - 1) // pad_multiple) * pad_multiple,
    )
    if padded_count <= active_count:
        return active_indices, np.ones((active_count,), dtype=np.float32), active_count

    padded_indices = np.empty((padded_count,), dtype=np.int32)
    padded_indices[:active_count] = active_indices
    padded_indices[active_count:] = active_indices[0]
    active_mask = np.zeros((padded_count,), dtype=np.float32)
    active_mask[:active_count] = 1.0
    return padded_indices, active_mask, active_count


@jax.jit
def _apply_active_row_mask_jit(values, active_mask):
    if active_mask is None:
        return values
    mask = jnp.asarray(active_mask, dtype=jnp.asarray(values).real.dtype)
    while mask.ndim < values.ndim:
        mask = mask[:, None]
    return values * mask


def _apply_active_row_mask(values, active_mask):
    if active_mask is None:
        return values
    if hasattr(values, "shape") and hasattr(values, "dtype"):
        return _apply_active_row_mask_jit(values, jnp.asarray(active_mask))
    mask = jnp.asarray(active_mask, dtype=jnp.asarray(values).real.dtype)
    while mask.ndim < values.ndim:
        mask = mask[:, None]
    return values * mask


@jax.jit
def _select_active_flat_rows_jit(values, flat_rotations, active_indices, active_mask):
    """Gather + mask the active rows and their rotations in one program.

    The gather, the mask broadcast/multiply and the rotation gather used to be
    four to six separate executables per class-chunk (perfetto trace, job
    13832290); each executable costs a launch and a dispatch.
    """
    active_values = _gather_active_flat_bucket_rows(values, active_indices)
    if active_mask is not None:
        active_values = _apply_active_row_mask_jit(active_values, active_mask)
    return active_values, flat_rotations[active_indices]


def _select_active_flat_rows(values, flat_rotations, active_indices, active_mask=None):
    """Gather active flattened rows with matching rotations."""

    if active_indices.size == 0:
        return None, None
    return _select_active_flat_rows_jit(
        values,
        flat_rotations,
        jnp.asarray(active_indices, dtype=jnp.int32),
        None if active_mask is None else jnp.asarray(active_mask),
    )


@jax.jit
def _select_active_flat_values_jit(values, active_indices, active_mask):
    active_values = _gather_active_flat_bucket_rows(values, active_indices)
    if active_mask is not None:
        active_values = _apply_active_row_mask_jit(active_values, active_mask)
    return active_values


def _select_active_flat_values(values, active_indices, active_mask=None):
    """Gather active flattened rows."""

    if active_indices.size == 0:
        return None
    return _select_active_flat_values_jit(
        values,
        jnp.asarray(active_indices, dtype=jnp.int32),
        None if active_mask is None else jnp.asarray(active_mask),
    )


@jax.jit
def _gather_active_flat_bucket_rows(values, active_indices):
    """Gather flat row indices without materializing the full flattened bucket.

    Jitted: the index split and gather used to cost one eager program each per
    bucket shape (618 gather compiles in one 100k/256 iteration, job 13807792).
    """

    values = jnp.asarray(values)
    if values.ndim >= 3:
        n_rotation_rows = int(values.shape[1])
        image_indices = active_indices // n_rotation_rows
        rotation_row_indices = active_indices - image_indices * n_rotation_rows
        return values[image_indices, rotation_row_indices]
    if values.ndim == 2:
        return values[active_indices]
    return flatten_bucket_rows(values)[active_indices]


def _active_image_indices_for_rotation_rows(active_indices, active_mask, n_rotation_rows: int):
    """Return image ids for active flattened ``(batch, rotation_row)`` rows."""

    if active_indices.size == 0:
        return None
    active_indices_jax = jnp.asarray(active_indices, dtype=jnp.int32)
    image_indices = active_indices_jax // int(n_rotation_rows)
    if active_mask is None:
        return image_indices
    active_mask_jax = jnp.asarray(active_mask, dtype=jnp.int32)
    return jnp.where(active_mask_jax != 0, image_indices, 0)


@partial(
    jax.jit,
    static_argnames=("n_rotation_rows", "shell_count", "batch_size", "use_residual_terms"),
)
def _compute_active_noise_rows_block(
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs_for_noise,
    active_indices,
    active_mask,
    noise_variance_half,
    shell_indices,
    *,
    n_rotation_rows: int,
    shell_count: int,
    batch_size: int,
    use_residual_terms: bool,
):
    """Gather one active-row chunk and accumulate its noise/norm residuals."""

    active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
    active_mask = jnp.asarray(active_mask)
    row_mask = active_mask.astype(jnp.asarray(summed_masked_noise).real.dtype)
    active_image_indices = jnp.where(active_mask.astype(jnp.int32) != 0, active_indices // int(n_rotation_rows), 0)

    def gather(values):
        flat_values = values.reshape((values.shape[0] * values.shape[1], values.shape[-1]))
        gathered = flat_values[active_indices]
        return gathered * row_mask[:, None]

    flat_proj_for_noise = gather(proj_for_noise)
    flat_proj_abs2_for_noise = gather(proj_abs2_for_noise)
    flat_summed_masked_noise = gather(summed_masked_noise)
    flat_ctf_probs_for_noise = gather(ctf_probs_for_noise)
    if use_residual_terms:
        return _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms(
            flat_proj_for_noise,
            flat_proj_abs2_for_noise,
            flat_summed_masked_noise,
            flat_ctf_probs_for_noise,
            noise_variance_half,
            shell_indices,
            active_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )
    return _compute_noise_block_and_norm_residual_from_flat_rows(
        flat_proj_for_noise,
        flat_proj_abs2_for_noise,
        flat_summed_masked_noise,
        flat_ctf_probs_for_noise,
        noise_variance_half,
        shell_indices,
        active_image_indices,
        shell_count=int(shell_count),
        batch_size=int(batch_size),
    )


def _compute_active_noise_rows_chunked(
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs_for_noise,
    active_indices,
    active_mask,
    noise_variance_half,
    shell_indices,
    *,
    n_rotation_rows: int,
    shell_count: int,
    batch_size: int,
    max_block_bytes: int | None,
):
    """Gather compact active noise rows in row chunks before accumulation."""

    n_rows = int(active_indices.size)
    accumulator_dtype = jnp.result_type(
        proj_abs2_for_noise.dtype,
        ctf_probs_for_noise.dtype,
        noise_variance_half.dtype,
    )
    if n_rows <= 0:
        return (
            jnp.zeros(int(shell_count), dtype=accumulator_dtype),
            jnp.zeros(int(batch_size), dtype=accumulator_dtype),
        )

    use_residual_terms = parse_env_flag(_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV, default=True)
    if max_block_bytes is None:
        max_rows = n_rows
    else:
        n_pixels = int(proj_for_noise.shape[-1])
        complex_bytes = max(
            _dtype_itemsize(proj_for_noise.dtype),
            _dtype_itemsize(summed_masked_noise.dtype),
        )
        real_bytes = max(
            _dtype_itemsize(proj_abs2_for_noise.dtype),
            _dtype_itemsize(ctf_probs_for_noise.dtype),
            _dtype_itemsize(noise_variance_half.dtype),
        )
        bytes_per_row = max(1, int(n_pixels) * (2 * int(complex_bytes) + 3 * int(real_bytes)))
        max_rows = max(1, int(max_block_bytes) // bytes_per_row)

    if n_rows > max_rows:
        n_pixels = int(proj_for_noise.shape[-1])
        n_chunks = (n_rows + max_rows - 1) // max_rows
        log_key = (n_rows, n_pixels, max_rows, int(max_block_bytes or 0))
        if log_key not in _active_noise_gather_chunk_log_keys:
            _active_noise_gather_chunk_log_keys.add(log_key)
            logger.info(
                "Sparse pass-2 compact active noise gather chunking: rows=%d pixels=%d max_rows=%d "
                "chunks=%d max_block_bytes=%.2f GiB",
                n_rows,
                n_pixels,
                max_rows,
                n_chunks,
                int(max_block_bytes or 0) / float(1024**3),
            )

    noise_total = jnp.zeros(int(shell_count), dtype=accumulator_dtype)
    norm_total = jnp.zeros(int(batch_size), dtype=accumulator_dtype)
    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        noise_chunk, norm_chunk = _compute_active_noise_rows_block(
            proj_for_noise,
            proj_abs2_for_noise,
            summed_masked_noise,
            ctf_probs_for_noise,
            active_indices[start:stop],
            active_mask[start:stop],
            noise_variance_half,
            shell_indices,
            n_rotation_rows=int(n_rotation_rows),
            shell_count=int(shell_count),
            batch_size=int(batch_size),
            use_residual_terms=bool(use_residual_terms),
        )
        noise_total = noise_total + noise_chunk
        norm_total = norm_total + norm_chunk
    return noise_total, norm_total


def _active_row_grouping_shape(active_indices, active_mask, n_images, n_rotation_rows):
    """Return active count, max active rows per image, and grouped dense rows."""

    active_indices = np.asarray(active_indices, dtype=np.int32)
    active_mask = np.asarray(active_mask, dtype=np.float32)
    valid = active_mask != 0.0
    active_count = int(np.count_nonzero(valid))
    if active_count == 0:
        return 0, 1, int(n_images)
    image_indices = active_indices[valid] // int(n_rotation_rows)
    counts = np.bincount(image_indices, minlength=int(n_images))
    active_row_slots = max(1, int(np.max(counts, initial=0)))
    return active_count, active_row_slots, int(n_images) * active_row_slots


def _active_row_grouping_for_canonical_matmul(active_indices, active_mask, n_images, n_rotation_rows):
    """Group flat active rows by image while preserving flat active-row order."""

    active_indices = np.asarray(active_indices, dtype=np.int32)
    active_mask = np.asarray(active_mask, dtype=np.float32)
    image_indices = (active_indices // int(n_rotation_rows)).astype(np.int32, copy=False)
    active_slots = np.zeros(active_indices.shape, dtype=np.int32)
    valid = active_mask != 0.0
    valid_positions = np.flatnonzero(valid).astype(np.int32, copy=False)
    if valid_positions.size == 0:
        return image_indices, active_slots, np.zeros((int(n_images), 1), dtype=np.int32)

    valid_image_indices = image_indices[valid_positions]
    counts = np.bincount(valid_image_indices, minlength=int(n_images))
    active_row_slots = max(1, int(np.max(counts, initial=0)))
    grouped_rotation_rows = np.zeros((int(n_images), active_row_slots), dtype=np.int32)

    order = np.lexsort((valid_positions, valid_image_indices))
    sorted_positions = valid_positions[order]
    sorted_image_indices = valid_image_indices[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_image_indices)) + 1]
    group_lengths = np.diff(np.r_[group_starts, sorted_image_indices.size])
    sorted_slots = np.arange(sorted_image_indices.size, dtype=np.int32) - np.repeat(
        group_starts.astype(np.int32, copy=False),
        group_lengths,
    )
    active_slots[sorted_positions] = sorted_slots
    grouped_rotation_rows[sorted_image_indices, sorted_slots] = (
        active_indices[sorted_positions] % int(n_rotation_rows)
    )
    return image_indices, active_slots, grouped_rotation_rows


def _rectangular_active_prematmul_is_efficient(
    active_indices,
    active_mask,
    *,
    n_images: int,
    n_rotation_rows: int,
    max_grouped_dense_ratio: float,
):
    active_count, active_slots, grouped_rows = _active_row_grouping_shape(
        active_indices,
        active_mask,
        n_images=n_images,
        n_rotation_rows=n_rotation_rows,
    )
    dense_rows = int(n_images) * int(n_rotation_rows)
    grouped_dense_ratio = float(grouped_rows) / float(dense_rows) if dense_rows > 0 else 1.0
    use_prematmul = active_count > 0 and grouped_dense_ratio <= float(max_grouped_dense_ratio)
    return use_prematmul, active_count, active_slots, grouped_rows, dense_rows, grouped_dense_ratio


@jax.jit
def _rectangular_active_weighted_image_sums_grouped(
    probs,
    shifted,
    active_image_indices,
    active_slots,
    grouped_rotation_rows,
    active_mask,
):
    """Compute active rows with the same per-image matmul shape as the dense path."""

    grouped_probs = jnp.take_along_axis(
        probs,
        jnp.asarray(grouped_rotation_rows, dtype=jnp.int32)[:, :, None],
        axis=1,
    )
    grouped_summed = compute_local_weighted_sums(grouped_probs, shifted)
    active_summed = grouped_summed[
        jnp.asarray(active_image_indices, dtype=jnp.int32),
        jnp.asarray(active_slots, dtype=jnp.int32),
    ]
    active_mask = jnp.asarray(active_mask, dtype=active_summed.real.dtype)
    return active_summed * active_mask[:, None]


@jax.jit
def _rectangular_active_weighted_sums(
    probs,
    probs_sum_t,
    shifted,
    ctf2_over_nv,
    active_indices,
    active_image_indices,
    active_slots,
    grouped_rotation_rows,
    active_mask,
):
    """Compute rectangular M-step rows after gathering active ``(image, rotation)`` rows."""

    n_rotation_rows = probs.shape[1]
    active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
    active_summed = _rectangular_active_weighted_image_sums_grouped(
        probs,
        shifted,
        active_image_indices,
        active_slots,
        grouped_rotation_rows,
        active_mask,
    )
    active_image_indices = jnp.asarray(active_image_indices, dtype=jnp.int32)
    active_ctf_probs = probs_sum_t.reshape((probs.shape[0] * n_rotation_rows,))[active_indices, None]
    active_ctf_probs = active_ctf_probs * ctf2_over_nv[active_image_indices]
    active_mask = jnp.asarray(active_mask, dtype=active_ctf_probs.real.dtype)
    active_ctf_probs = active_ctf_probs * active_mask[:, None]
    return active_summed, active_ctf_probs


@jax.jit
def _rectangular_active_weighted_image_sums(
    probs,
    shifted,
    active_image_indices,
    active_slots,
    grouped_rotation_rows,
    active_mask,
):
    """Compute active rectangular weighted image rows without recomputing CTF sums."""

    return _rectangular_active_weighted_image_sums_grouped(
        probs,
        shifted,
        active_image_indices,
        active_slots,
        grouped_rotation_rows,
        active_mask,
    )


def _rectangular_active_weighted_sums_or_none(
    probs,
    probs_sum_t,
    shifted,
    ctf2_over_nv,
    flat_rotations,
    active_indices,
    active_mask,
):
    if active_indices.size == 0:
        return None, None, None
    active_image_indices, active_slots, grouped_rotation_rows = _active_row_grouping_for_canonical_matmul(
        active_indices,
        active_mask,
        n_images=probs.shape[0],
        n_rotation_rows=probs.shape[1],
    )
    active_indices_jax = jnp.asarray(active_indices, dtype=jnp.int32)
    active_summed, active_ctf_probs = _rectangular_active_weighted_sums(
        probs,
        probs_sum_t,
        shifted,
        ctf2_over_nv,
        active_indices_jax,
        jnp.asarray(active_image_indices, dtype=jnp.int32),
        jnp.asarray(active_slots, dtype=jnp.int32),
        jnp.asarray(grouped_rotation_rows, dtype=jnp.int32),
        jnp.asarray(active_mask),
    )
    return active_summed, active_ctf_probs, flat_rotations[active_indices_jax]


def _rectangular_active_weighted_image_sums_or_none(
    probs,
    shifted,
    active_indices,
    active_mask,
):
    if active_indices.size == 0:
        return None
    active_image_indices, active_slots, grouped_rotation_rows = _active_row_grouping_for_canonical_matmul(
        active_indices,
        active_mask,
        n_images=probs.shape[0],
        n_rotation_rows=probs.shape[1],
    )
    return _rectangular_active_weighted_image_sums(
        probs,
        shifted,
        jnp.asarray(active_image_indices, dtype=jnp.int32),
        jnp.asarray(active_slots, dtype=jnp.int32),
        jnp.asarray(grouped_rotation_rows, dtype=jnp.int32),
        jnp.asarray(active_mask),
    )


@partial(jax.jit, static_argnames=("n_rotation_rows", "shell_count", "batch_size"))
def _flat_rows_noise_and_ctf_from_dense_probs(
    dense_probs,
    ctf2_over_nv_recon,
    summed_image_flat,
    proj_for_noise,
    proj_abs2_for_noise,
    noise_variance_half,
    shell_indices,
    row_batch,
    row_rotation,
    active_mask,
    *,
    n_rotation_rows: int,
    shell_count: int,
    batch_size: int,
):
    """Flat-row CTF sums, posterior reductions and noise terms (same arithmetic as the padded path)."""
    probs_sum_t = jnp.sum(dense_probs, axis=-1)  # (B, R)
    translation_posterior = jnp.sum(dense_probs, axis=1)
    mask = jnp.asarray(active_mask, dtype=ctf2_over_nv_recon.dtype)
    probs_sum_t_flat = probs_sum_t[row_batch, row_rotation] * mask  # masked rows carry zero mass
    ctf2_flat = ctf2_over_nv_recon[row_batch]  # (F, N)
    ctf_probs_flat = jnp.where(
        probs_sum_t_flat[:, None] != 0.0,
        probs_sum_t_flat[:, None] * ctf2_flat,
        0.0,
    )
    proj_flat = proj_for_noise[row_batch, row_rotation]
    proj_abs2_flat = proj_abs2_for_noise[row_batch, row_rotation]
    block_noise_shells, block_norm_residual = (
        _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms(
            proj_flat,
            proj_abs2_flat,
            summed_image_flat,
            ctf_probs_flat,
            noise_variance_half,
            shell_indices,
            row_batch,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )
    )
    return probs_sum_t, translation_posterior, ctf_probs_flat, block_noise_shells, block_norm_residual


def _compact_pair_weighted_sums_and_noise_native_flat_rows(
    pair_probs,
    local_rotation_row,
    translation_idx,
    pair_mask,
    shifted_recon_split,
    shifted_image_split,
    ctf2_over_nv_recon,
    proj_for_noise,
    proj_abs2_for_noise,
    noise_variance_half,
    shell_indices,
    flat_rows,
    *,
    n_rotation_rows: int,
    shell_count: int,
    batch_size: int,
    scale_old_scale=None,
    scale_pixel_mask=None,
):
    """Flat real-row form of :func:`_compact_pair_weighted_sums_and_noise_native`.

    Returns ``(summed_flat, summed_image_flat, ctf_probs_flat, probs_sum_t,
    translation_posterior, block_noise_shells, block_norm_residual,
    scale_xa_per_image, scale_aa_per_image)`` with the first three as
    ``[rows, pixel]`` arrays over ``flat_rows``; rows whose mask is false (shape
    padding) are exactly zero. The per-image group-scale terms are ``None`` unless
    ``scale_old_scale`` is given.
    """
    from recovar.cuda_backproject import dual_weighted_sums_pairs_rows_f32

    row_batch, row_rotation, active_mask = flat_rows
    n_trans = int(shifted_recon_split.shape[1])
    dense_probs = _compact_pair_dense_probs(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=n_trans,
    )
    sorted_probs, sorted_translations, row_offsets = _compact_pair_sorted_csr(
        pair_probs,
        local_rotation_row,
        translation_idx,
        pair_mask,
        n_rotation_rows=n_rotation_rows,
        n_trans=n_trans,
    )
    summed_flat, summed_image_flat = dual_weighted_sums_pairs_rows_f32(
        sorted_probs,
        sorted_translations,
        row_offsets,
        row_batch,
        row_rotation,
        shifted_recon_split,
        shifted_image_split,
    )
    summed_flat = _apply_active_row_mask_jit(summed_flat, active_mask)
    summed_image_flat = _apply_active_row_mask_jit(summed_image_flat, active_mask)
    (
        probs_sum_t,
        translation_posterior,
        ctf_probs_flat,
        block_noise_shells,
        block_norm_residual,
    ) = _flat_rows_noise_and_ctf_from_dense_probs(
        dense_probs,
        ctf2_over_nv_recon,
        summed_image_flat,
        proj_for_noise,
        proj_abs2_for_noise,
        noise_variance_half,
        shell_indices,
        row_batch,
        row_rotation,
        active_mask,
        n_rotation_rows=int(n_rotation_rows),
        shell_count=int(shell_count),
        batch_size=int(batch_size),
    )
    scale_xa_per_image = scale_aa_per_image = None
    if scale_old_scale is not None:
        scale_xa_per_image, scale_aa_per_image = _flat_rows_scale_correction_terms_per_image(
            proj_for_noise,
            proj_abs2_for_noise,
            summed_image_flat,
            ctf_probs_flat,
            noise_variance_half,
            scale_old_scale,
            None if scale_pixel_mask is None else jnp.asarray(scale_pixel_mask, dtype=bool),
            row_batch,
            row_rotation,
            batch_size=int(batch_size),
        )
    return (
        summed_flat,
        summed_image_flat,
        ctf_probs_flat,
        probs_sum_t,
        translation_posterior,
        block_noise_shells,
        block_norm_residual,
        scale_xa_per_image,
        scale_aa_per_image,
    )


@partial(jax.jit, static_argnames=("batch_size",))
def _flat_rows_scale_correction_terms_per_image(
    proj_for_noise,
    proj_abs2_for_noise,
    summed_flat,
    ctf_probs_flat,
    noise_variance_half,
    old_scale,
    scale_pixel_mask,
    row_batch,
    row_rotation,
    *,
    batch_size: int,
):
    """Flat-row form of ``compute_scale_correction_terms_per_image`` (same terms, rows summed per image)."""
    proj_flat = proj_for_noise[row_batch, row_rotation]
    proj_abs2_flat = proj_abs2_for_noise[row_batch, row_rotation]
    safe_scale = jnp.maximum(jnp.asarray(old_scale, dtype=proj_abs2_flat.real.dtype), 1e-30)
    ctf_has_mass = ctf_probs_flat != 0.0
    if scale_pixel_mask is not None:
        ctf_has_mass = ctf_has_mass & scale_pixel_mask.reshape(-1)[None, :]
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs_flat * noise_variance_half[None, :], 0.0)
    aa_terms = jnp.where(ctf_has_mass, proj_abs2_flat * ctf_probs_raw, 0.0)
    aa_rows = jnp.sum(aa_terms, axis=1)
    cross_has_mass = summed_flat != 0.0
    if scale_pixel_mask is not None:
        cross_has_mass = cross_has_mass & scale_pixel_mask.reshape(-1)[None, :]
    cross_terms = jnp.where(cross_has_mass, proj_flat * jnp.conj(summed_flat), 0.0)
    xa_rows = jnp.sum(noise_variance_half[None, :] * cross_terms.real, axis=1)
    aa_per_image = add_segment_sum(jnp.zeros(int(batch_size), dtype=aa_rows.dtype), row_batch, aa_rows)
    xa_per_image = add_segment_sum(jnp.zeros(int(batch_size), dtype=xa_rows.dtype), row_batch, xa_rows)
    return xa_per_image / safe_scale, aa_per_image / (safe_scale**2)


def device_active_row_indices_enabled() -> bool:
    """Build the active flat-row index vector and mask on the device.

    They are a pure function of the bucket's per-image row counts, which are small
    host metadata, but the host materialised both ``(active rows,)`` arrays and
    uploaded them for every selection call: ~17 s of an iteration at 100k/256 K=4
    (job 13838987 stack samples, ``_select_active_flat_rows`` /
    ``_select_active_flat_values``). Returning device arrays from the builder makes
    the later ``jnp.asarray`` calls no-ops, so no call site changes.
    """

    return parse_env_flag(_SPARSE_KCLASS_DEVICE_ACTIVE_ROW_INDICES_ENV, default=False)


@partial(jax.jit, static_argnames=("n_rotation_rows", "padded_count"))
def _active_flat_row_indices_device(counts, *, n_rotation_rows: int, padded_count: int):
    """``(padded_count,)`` flat row indices and float32 mask from per-image counts.

    Equals the host construction element for element: image ``b`` contributes its
    ``counts[b]`` rows ``b * n_rotation_rows + j`` in order, images with zero count
    contribute nothing (``searchsorted`` on the inclusive prefix ends skips them),
    and the padded tail is masked to zero. The padded slots' indices are clamped
    in range; their gathered values are multiplied by a zero mask exactly as the
    host path's repeated first index is.
    """

    counts = counts.astype(jnp.int32)
    ends = jnp.cumsum(counts, dtype=jnp.int32)
    starts = ends - counts
    slots = jnp.arange(padded_count, dtype=jnp.int32)
    row = jnp.minimum(jnp.searchsorted(ends, slots, side="right"), counts.shape[0] - 1)
    flat = jnp.clip(
        row * jnp.int32(n_rotation_rows) + (slots - starts[row]),
        0,
        counts.shape[0] * n_rotation_rows - 1,
    )
    return flat, (slots < ends[-1]).astype(jnp.float32)


def _real_flat_row_indices_from_actual_counts(
    actual_counts,
    n_rotation_rows: int,
    *,
    pad_multiple: int = 1,
    pad_to: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Flat ``(batch, rotation_row)`` indices of the rows an image really owns.

    Rows at or beyond ``actual_counts[b]`` exist only to pad the bucket to its
    quantized size; their posteriors are exactly zero, so the M-step adjoint can
    skip them. Unlike :func:`_active_flat_row_indices_from_probs_sum_t` this
    needs no device-to-host transfer: the counts are host metadata. Padding to
    ``pad_multiple`` repeats the first index with a zero mask so shapes repeat.
    """
    counts = np.asarray(actual_counts, dtype=np.int64).reshape(-1)
    n_rotation_rows = int(n_rotation_rows)
    counts = np.clip(counts, 0, n_rotation_rows)
    total = int(counts.sum())
    if total == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32), 0
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    real_indices = (
        np.repeat(np.arange(counts.size, dtype=np.int64) * n_rotation_rows, counts)
        + np.arange(total, dtype=np.int64)
        - np.repeat(starts, counts)
    ).astype(np.int32, copy=False)
    pad_multiple = max(1, int(pad_multiple))
    # Power-of-two ladder above the multiple: the linear ladder gave the
    # active-row selection 84 distinct compiled shapes in one iteration at
    # 100k/256 K=4 (12 s of XLA compile, census job 13837258). Padded slots
    # repeat the first index under a zero mask, so only the shape changes.
    padded_count = ((total + pad_multiple - 1) // pad_multiple) * pad_multiple
    if pad_multiple > 1:
        # Snap to a power of two only when it wastes at most an eighth of the rows.
        # Padded rows are gathered and multiplied through the M step, so an unbounded
        # snap buys ~12 s of XLA compile with up to 88 % more real work per chunk
        # (census job 13837258: 28 distinct active-row counts, mean +34 %, max +88 %,
        # +37 % rows overall).
        pow2_count = max(pad_multiple, 1 << (padded_count - 1).bit_length())
        if pow2_count <= padded_count + padded_count // 8:
            padded_count = pow2_count
    if pad_to is not None:
        # Group-static target (see ``group_static_active_rows_enabled``): never below
        # the multiple padding this chunk would get on its own.
        padded_count = max(padded_count, int(pad_to))
    padded_count = min(int(counts.size) * n_rotation_rows, padded_count)
    if device_active_row_indices_enabled():
        device_indices, device_mask = _active_flat_row_indices_device(
            jnp.asarray(counts.astype(np.int32, copy=False)),
            n_rotation_rows=n_rotation_rows,
            padded_count=int(max(padded_count, total)),
        )
        return device_indices, device_mask, total
    if padded_count <= total:
        return real_indices, np.ones((total,), dtype=np.float32), total
    padded_indices = np.empty((padded_count,), dtype=np.int32)
    padded_indices[:total] = real_indices
    padded_indices[total:] = real_indices[0]
    active_mask = np.zeros((padded_count,), dtype=np.float32)
    active_mask[:total] = 1.0
    return padded_indices, active_mask, total


_SPARSE_KCLASS_DEVICE_ACTIVE_ROW_INDICES_ENV = "RECOVAR_SPARSE_KCLASS_DEVICE_ACTIVE_ROW_INDICES"
