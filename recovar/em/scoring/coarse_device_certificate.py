"""Compose the existing streamed coarse certificate on device.

This opt-in building block does not publish approximate scores or alter the
certificate arithmetic. It moves image preparation, aligned rotation chunks,
and interval-state carries inside one executable. Exact rescoring still owns
published scores. The significance engine does not use this module yet.
"""

import operator
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.scoring.coarse_gemm_hybrid import (
    DEFAULT_ROTATION_BLOCK_CAPACITY,
    SOURCE_ROTATION_BLOCK_SIZE,
    initialize_coarse_gemm_hybrid_interval_state,
    validate_coarse_gemm_certificate_topology,
)
from recovar.em.scoring.scoring import (
    _prepare_relion_coarse_gaussian_gemm_f64_image_batch_jit,
    _relion_coarse_gaussian_gemm_update_certificate_state_jit,
)


@partial(jax.jit, static_argnames=("chunk_rows",))
def _coarse_certificate_state_jit(
    projections,
    shifted,
    weight,
    initial,
    actual_image_count,
    class_prior,
    rotation_prior,
    translation_prior,
    gammas,
    direct_gamma,
    full_position_count,
    *,
    chunk_rows,
):
    """Retain only compact state between chunks; never pad rotation coverage."""
    image_batch = _prepare_relion_coarse_gaussian_gemm_f64_image_batch_jit(shifted, weight, initial, actual_image_count)
    n_rotations = projections.shape[0]
    state = initialize_coarse_gemm_hybrid_interval_state(shifted.shape[0], n_rotations)
    full_chunks, tail = divmod(n_rotations, chunk_rows)

    def update(state, projection_block, prior_block, offset):
        return _relion_coarse_gaussian_gemm_update_certificate_state_jit(
            state,
            projection_block,
            image_batch,
            class_prior,
            prior_block,
            translation_prior,
            *gammas,
            direct_gamma,
            full_position_count,
            offset,
        )

    def body(index, carry):
        offset = index * jnp.int32(chunk_rows)
        projection_block = jax.lax.dynamic_slice_in_dim(projections, offset, chunk_rows, axis=0)
        prior_block = (
            None if rotation_prior is None else jax.lax.dynamic_slice_in_dim(rotation_prior, offset, chunk_rows, axis=0)
        )
        return update(carry, projection_block, prior_block, offset)

    # A static zero-trip fori_loop still traces its body. Avoid tracing an
    # oversized dynamic_slice when the whole input is one short tail.
    if full_chunks:
        state = jax.lax.fori_loop(0, full_chunks, body, state)
    if tail:
        offset = full_chunks * chunk_rows
        state = update(
            state,
            projections[offset:],
            None if rotation_prior is None else rotation_prior[offset:],
            jnp.int32(offset),
        )
    return state


def _prepare_coarse_certificate_inputs(
    projections,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    topology,
    class_log_prior,
    rotation_log_prior=None,
    translation_log_prior=None,
    chunk_rows=256,
):
    """Validate static metadata and compose the mature certificate operations.

    ``actual_image_count`` may be a device int32 scalar, so changing the active
    image prefix does not specialize the executable. Its value is validated by
    the downstream device selector. No array values are read here. The sealed
    host topology is checked at admission; only its scalar coefficients enter
    the executable, not a callable or dataset object.
    """
    validate_coarse_gemm_certificate_topology(topology)
    projections, shifted, weight, initial = map(
        jnp.asarray,
        (projections, shifted_corrected, pixel_weight, initial_diff2),
    )
    if projections.ndim != 2 or shifted.ndim != 3:
        raise ValueError("expected projections[R,F] and shifted images[B,T,F]")
    batch, translations, pixels = shifted.shape
    rotations = projections.shape[0]
    if (
        min(batch, translations, pixels, rotations) <= 0
        or rotations % SOURCE_ROTATION_BLOCK_SIZE
        or projections.shape[1] != pixels
        or weight.shape != (batch, pixels)
        or initial.shape != (batch,)
        or topology.compact_pixel_count != pixels
        or topology.translation_count != translations
        or rotations * translations > np.iinfo(np.int32).max
    ):
        raise ValueError("inconsistent coarse certificate shape or topology")
    if (
        projections.dtype != jnp.complex64
        or shifted.dtype != jnp.complex64
        or weight.dtype != jnp.float32
        or initial.dtype != jnp.float32
    ):
        raise TypeError("certificate requires stored complex64/float32 operands")
    try:
        chunk_rows = operator.index(chunk_rows)
    except TypeError as error:
        raise ValueError("chunk_rows must be a positive aligned integer") from error
    if chunk_rows <= 0 or chunk_rows % SOURCE_ROTATION_BLOCK_SIZE:
        raise ValueError("chunk_rows must be a positive aligned integer")
    if isinstance(actual_image_count, (int, np.integer)):
        if not 0 < actual_image_count <= batch:
            raise ValueError("actual_image_count must be within the physical batch")
        actual_image_count = np.int32(actual_image_count)
    actual = jnp.asarray(actual_image_count)
    if actual.shape != () or actual.dtype != jnp.int32:
        raise TypeError("actual_image_count must be a scalar int32")
    class_prior = jnp.asarray(class_log_prior, dtype=jnp.float32)
    if class_prior.shape != ():
        raise ValueError("class_log_prior must be scalar")
    rotation_prior = None if rotation_log_prior is None else jnp.asarray(rotation_log_prior, dtype=jnp.float32)
    if rotation_prior is not None and rotation_prior.shape != (rotations,):
        raise ValueError("rotation_log_prior must have one value per rotation")
    translation_prior = None if translation_log_prior is None else jnp.asarray(translation_log_prior, dtype=jnp.float32)
    if translation_prior is not None and translation_prior.shape not in ((translations,), (batch, translations)):
        raise ValueError("translation_log_prior must have shape [T] or [B,T]")
    with jax.enable_x64(True):
        return (
            projections,
            shifted,
            weight,
            initial,
            actual,
            class_prior,
            rotation_prior,
            translation_prior,
            tuple(jnp.float64(value) for value in topology.expanded_f64_gammas),
            jnp.float64(topology.direct_f32_gamma),
            jnp.int32(topology.full_position_count),
        )


def prepare_coarse_certificate_state(
    projections,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    topology,
    class_log_prior,
    rotation_log_prior=None,
    translation_log_prior=None,
    chunk_rows=256,
):
    """Build complete interval state with one device-resident streamed loop."""
    operands = _prepare_coarse_certificate_inputs(
        projections,
        shifted_corrected,
        pixel_weight,
        initial_diff2,
        actual_image_count,
        topology=topology,
        class_log_prior=class_log_prior,
        rotation_log_prior=rotation_log_prior,
        translation_log_prior=translation_log_prior,
        chunk_rows=chunk_rows,
    )
    with jax.enable_x64(True):
        return _coarse_certificate_state_jit(*operands, chunk_rows=operator.index(chunk_rows))


@partial(jax.jit, static_argnames=("chunk_rows", "block_capacity"))
def _certify_coarse_rotation_blocks_jit(*operands, chunk_rows, block_capacity):
    from recovar.em.scoring.coarse_device_selection import select_coarse_rotation_blocks_device

    state = _coarse_certificate_state_jit(*operands, chunk_rows=chunk_rows)
    return select_coarse_rotation_blocks_device(
        state,
        actual_image_count=operands[4],
        n_rotations=operands[0].shape[0],
        n_translations=operands[1].shape[1],
        certificate_valid=True,
        block_capacity=block_capacity,
    )


def certify_coarse_rotation_blocks(
    projections,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    topology,
    class_log_prior,
    rotation_log_prior=None,
    translation_log_prior=None,
    chunk_rows=256,
    block_capacity=DEFAULT_ROTATION_BLOCK_CAPACITY,
):
    """Return bounded ordered source blocks with no certificate host transfer.

    Failure and active-count validation stay in the device result. The caller
    must honor ``eligible`` before publishing selected-rescore results; this
    function does not replace the whole-batch full-direct fallback policy.
    """
    operands = _prepare_coarse_certificate_inputs(
        projections,
        shifted_corrected,
        pixel_weight,
        initial_diff2,
        actual_image_count,
        topology=topology,
        class_log_prior=class_log_prior,
        rotation_log_prior=rotation_log_prior,
        translation_log_prior=translation_log_prior,
        chunk_rows=chunk_rows,
    )
    with jax.enable_x64(True):
        return _certify_coarse_rotation_blocks_jit(
            *operands,
            chunk_rows=operator.index(chunk_rows),
            block_capacity=operator.index(block_capacity),
        )
