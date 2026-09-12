"""Noise-block accumulation of the sparse bucketed pass 2.

The chunked per-hypothesis noise block and the residual-norm variants that
feed RELION's ``sigma2_noise`` update from flat hypothesis rows.
``sparse_pass2_bucketed`` and the compact-pair sums call them per bucket.
"""

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp

from recovar.em.helpers.deterministic_reduce import add_segment_sum
from recovar.em.helpers.env_flags import parse_env_flag
from recovar.em.helpers.half_spectrum import bin_shell_values_jax
from recovar.em.helpers.projection import compute_noise_block as _compute_noise_block
from recovar.em.sparse_pass2.sparse_pass2_budget import _dtype_itemsize
from recovar.em.sparse_pass2.sparse_pass2_policy import _SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV

logger = logging.getLogger("recovar.em.dense_single_volume.helpers.sparse_pass2_noise_blocks")


_noise_block_chunk_log_keys: set[tuple[int, int, int, int]] = set()


def _compute_noise_block_chunked(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    shell_count,
    *,
    max_block_bytes: int | None,
):
    """Run ``compute_noise_block`` in row chunks when one bucket is too large."""

    n_rows = int(proj_half.shape[0])
    if n_rows <= 0:
        return _compute_noise_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            shell_count,
        )
    if max_block_bytes is None:
        return _compute_noise_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            shell_count,
        )

    n_pixels = int(proj_half.shape[1])
    complex_bytes = max(
        _dtype_itemsize(proj_half.dtype),
        _dtype_itemsize(summed_masked.dtype),
    )
    real_bytes = max(
        _dtype_itemsize(proj_abs2_half.dtype),
        _dtype_itemsize(ctf_probs.dtype),
        _dtype_itemsize(noise_variance_half.dtype),
    )
    # compute_noise_block's live temporaries include complex cross terms and
    # several real products. Keep the estimate conservative because this path
    # is only needed for pathological sparse tail buckets.
    bytes_per_row = max(1, int(n_pixels) * (2 * int(complex_bytes) + 3 * int(real_bytes)))
    max_rows = max(1, int(max_block_bytes) // bytes_per_row)
    if n_rows <= max_rows:
        return _compute_noise_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            shell_count,
        )

    n_chunks = (n_rows + max_rows - 1) // max_rows
    log_key = (n_rows, n_pixels, max_rows, int(max_block_bytes))
    if log_key not in _noise_block_chunk_log_keys:
        _noise_block_chunk_log_keys.add(log_key)
        logger.info(
            "Sparse pass-2 noise block chunking: rows=%d pixels=%d max_rows=%d chunks=%d max_block_bytes=%.2f GiB",
            n_rows,
            n_pixels,
            max_rows,
            n_chunks,
            int(max_block_bytes) / float(1024**3),
        )
    accumulator_dtype = jnp.result_type(
        proj_abs2_half.dtype,
        ctf_probs.dtype,
        noise_variance_half.dtype,
    )
    noise_total = jnp.zeros(shell_count, dtype=accumulator_dtype)
    a2_total = jnp.zeros(shell_count, dtype=accumulator_dtype)
    xa_total = jnp.zeros(shell_count, dtype=accumulator_dtype)
    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        noise_chunk, a2_chunk, xa_chunk = _compute_noise_block(
            proj_half[start:stop],
            proj_abs2_half[start:stop],
            summed_masked[start:stop],
            ctf_probs[start:stop],
            noise_variance_half,
            shell_indices,
            shell_count,
        )
        noise_total = noise_total + noise_chunk
        a2_total = a2_total + a2_chunk
        xa_total = xa_total + xa_chunk
    return noise_total, a2_total, xa_total


@partial(jax.jit, static_argnames=("shell_count", "batch_size"))
def _compute_noise_block_and_norm_residual_from_flat_rows(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    flat_image_indices,
    *,
    shell_count: int,
    batch_size: int,
):
    """Return shell-binned noise and per-image norm residuals for active rows."""

    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half[None, :], 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    a2 = jnp.sum(a2_terms, axis=0)
    a2_per_row = jnp.sum(a2_terms, axis=1)

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    cross = jnp.sum(cross_terms, axis=0)
    xa = jnp.where(cross.real != 0.0, noise_variance_half * cross.real, 0.0)
    xa_per_row = jnp.sum(noise_variance_half[None, :] * cross_terms.real, axis=1)

    block_noise = a2 - 2.0 * xa
    # No explicit dtype cast: preserve whatever real dtype the inputs
    # naturally promote to (float64 under double-precision scoring), same
    # as compute_noise_block/compute_norm_residual_per_image. The zero-init
    # container below must match residual_per_row's dtype exactly --
    # jnp .at[].add() requires an exact dtype match, unlike plain addition.
    noise_shells = bin_shell_values_jax(block_noise, shell_indices, shell_count)

    residual_per_row = a2_per_row - 2.0 * xa_per_row
    norm_residual = add_segment_sum(
        jnp.zeros(int(batch_size), dtype=residual_per_row.dtype),
        flat_image_indices,
        residual_per_row,
    )
    return noise_shells, norm_residual


@partial(jax.jit, static_argnames=("shell_count", "batch_size"))
def _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    flat_image_indices,
    *,
    shell_count: int,
    batch_size: int,
):
    """Real-valued residual term form for K-class noise/norm accumulation."""

    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half[None, :], 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    summed_has_mass = summed_masked != 0.0
    cross_real = (proj_half.real * summed_masked.real) + (proj_half.imag * summed_masked.imag)
    cross_real = jnp.where(summed_has_mass, cross_real, 0.0)
    xa_terms = noise_variance_half[None, :] * cross_real
    residual_terms = a2_terms - 2.0 * xa_terms

    block_noise = jnp.sum(residual_terms, axis=0)
    noise_shells = bin_shell_values_jax(block_noise, shell_indices, shell_count)

    residual_per_row = jnp.sum(residual_terms, axis=1)
    norm_residual = add_segment_sum(
        jnp.zeros(int(batch_size), dtype=residual_per_row.dtype),
        flat_image_indices,
        residual_per_row,
    )
    return noise_shells, norm_residual


def _compute_noise_block_and_norm_residual_chunked(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    flat_image_indices,
    *,
    shell_count: int,
    batch_size: int,
    max_block_bytes: int | None,
):
    """Run fused noise shell / norm-residual accumulation in row chunks."""

    compute_block = (
        _compute_noise_block_and_norm_residual_from_flat_rows_residual_terms
        if parse_env_flag(_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV, default=True)
        else _compute_noise_block_and_norm_residual_from_flat_rows
    )
    n_rows = int(proj_half.shape[0])
    if n_rows <= 0:
        return compute_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            flat_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )
    if max_block_bytes is None:
        return compute_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            flat_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )

    n_pixels = int(proj_half.shape[1])
    complex_bytes = max(
        _dtype_itemsize(proj_half.dtype),
        _dtype_itemsize(summed_masked.dtype),
    )
    real_bytes = max(
        _dtype_itemsize(proj_abs2_half.dtype),
        _dtype_itemsize(ctf_probs.dtype),
        _dtype_itemsize(noise_variance_half.dtype),
    )
    bytes_per_row = max(1, int(n_pixels) * (2 * int(complex_bytes) + 3 * int(real_bytes)))
    max_rows = max(1, int(max_block_bytes) // bytes_per_row)
    if n_rows <= max_rows:
        return compute_block(
            proj_half,
            proj_abs2_half,
            summed_masked,
            ctf_probs,
            noise_variance_half,
            shell_indices,
            flat_image_indices,
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )

    n_chunks = (n_rows + max_rows - 1) // max_rows
    log_key = (n_rows, n_pixels, max_rows, int(max_block_bytes))
    if log_key not in _noise_block_chunk_log_keys:
        _noise_block_chunk_log_keys.add(log_key)
        logger.info(
            "Sparse pass-2 fused noise/norm block chunking: rows=%d pixels=%d max_rows=%d "
            "chunks=%d max_block_bytes=%.2f GiB",
            n_rows,
            n_pixels,
            max_rows,
            n_chunks,
            int(max_block_bytes) / float(1024**3),
        )
    accumulator_dtype = jnp.result_type(
        proj_abs2_half.dtype,
        ctf_probs.dtype,
        noise_variance_half.dtype,
    )
    noise_total = jnp.zeros(int(shell_count), dtype=accumulator_dtype)
    norm_total = jnp.zeros(int(batch_size), dtype=accumulator_dtype)
    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        noise_chunk, norm_chunk = compute_block(
            proj_half[start:stop],
            proj_abs2_half[start:stop],
            summed_masked[start:stop],
            ctf_probs[start:stop],
            noise_variance_half,
            shell_indices,
            flat_image_indices[start:stop],
            shell_count=int(shell_count),
            batch_size=int(batch_size),
        )
        noise_total = noise_total + noise_chunk
        norm_total = norm_total + norm_chunk
    return noise_total, norm_total
