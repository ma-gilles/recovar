"""Projection blocks of the sparse bucketed pass 2.

The per-bucket projection of hypothesis rotations (full and windowed) with
their chunking and finalization. ``sparse_pass2_bucketed`` projects every
bucket through these owners.

Radius propagation: ``docs/math/sparse_projection_radius.md``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from recovar.em.helpers.projection import compute_projections_block as _compute_projections_block
from recovar.em.helpers.projection import (
    compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
)
from recovar.em.sparse_pass2.sparse_pass2_budget import _MAX_PROJECTED_ROTATIONS_ENV, _optional_positive_int_env


def _compute_sparse_pass2_projections_block(
    mean_for_proj,
    rotations_block,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    max_projected_rotations: int | None = None,
    output_complex_dtype=None,
    output_abs2_dtype=None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    projector_output_size: int | None = None,
    **projection_kwargs,
):
    projection_kwargs = dict(projection_kwargs)
    return_abs2 = projection_kwargs.pop("return_abs2", True)
    # The generic projector also needs this cutoff; consuming it here made
    # sparse pass 2 silently use a different radius from coarse scoring.
    projection_max_r = projection_kwargs.get("max_r", None)
    projection_relion_texture_interp = projection_kwargs.get("relion_texture_interp")
    projection_mask_current_image_disk = bool(
        projection_kwargs.pop("mask_current_image_disk", True)
    )
    if projector_output_size is None and projection_max_r is not None:
        projector_output_size = int(2 * float(projection_max_r))
    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")

    def _project(rotations):
        if use_relion_projector:
            return _compute_relion_projector_projections_block(
                relion_projector_half,
                rotations,
                image_shape,
                r_max=int(relion_projector_r_max),
                padding_factor=int(projection_padding_factor),
                return_abs2=bool(return_abs2),
                centered_rows=True,
                dense_scale=True,
                relion_texture_interp=projection_relion_texture_interp,
                projector_output_size=projector_output_size,
                mask_current_image_disk=projection_mask_current_image_disk,
            )
        return _compute_projections_block(
            mean_for_proj,
            rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            return_abs2=bool(return_abs2),
            **projection_kwargs,
        )

    if max_projected_rotations is None:
        max_projected_rotations = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)
    if max_projected_rotations is None:
        proj_half, proj_abs2 = _project(rotations_block)
        if output_complex_dtype is not None:
            proj_half = proj_half.astype(output_complex_dtype)
        if proj_abs2 is not None and output_abs2_dtype is not None:
            proj_abs2 = proj_abs2.astype(output_abs2_dtype)
        return proj_half, proj_abs2

    n_rotations = int(rotations_block.shape[0])
    max_projected_rotations = max(1, int(max_projected_rotations))
    if n_rotations <= max_projected_rotations:
        proj_half, proj_abs2 = _project(rotations_block)
        if output_complex_dtype is not None:
            proj_half = proj_half.astype(output_complex_dtype)
        if proj_abs2 is not None and output_abs2_dtype is not None:
            proj_abs2 = proj_abs2.astype(output_abs2_dtype)
        return proj_half, proj_abs2

    proj_chunks = []
    abs2_chunks = []
    for start in range(0, n_rotations, max_projected_rotations):
        stop = min(start + max_projected_rotations, n_rotations)
        proj_chunk, abs2_chunk = _project(rotations_block[start:stop])
        if output_complex_dtype is not None:
            proj_chunk = proj_chunk.astype(output_complex_dtype)
        if abs2_chunk is not None and output_abs2_dtype is not None:
            abs2_chunk = abs2_chunk.astype(output_abs2_dtype)
        proj_chunks.append(proj_chunk)
        abs2_chunks.append(abs2_chunk)

    proj_half = jnp.concatenate(proj_chunks, axis=0)
    if all(abs2_chunk is None for abs2_chunk in abs2_chunks):
        return proj_half, None
    if any(abs2_chunk is None for abs2_chunk in abs2_chunks):
        raise RuntimeError("Inconsistent projection abs2 chunks")
    return proj_half, jnp.concatenate(abs2_chunks, axis=0)


def _projection_kwargs_for_relion_score_window(
    projection_kwargs,
    *,
    use_relion_projector: bool,
    current_size: int | None,
):
    """Keep the RELION projector crop large enough for the particle image.

    ``r_max`` describes the model sphere, but it does not always describe the
    particle-image crop.  In particular, fresh first-iteration CC can score a
    size-58 particle image from a projector whose model ``r_max`` is 28.  A
    crop inferred as ``2 * r_max == 56`` drops the valid ``ky=-28`` row before
    the score window gathers it.  RELION projects into the particle-image box
    and clips samples independently to the model sphere, so preserve that
    distinction here.
    """

    kwargs = dict(projection_kwargs)
    if use_relion_projector:
        if current_size is None:
            raise ValueError("windowed RELION projection requires current_size")
        kwargs["projector_output_size"] = int(current_size)
    return kwargs


def _compute_sparse_pass2_windowed_projections_block(
    mean_for_proj,
    rotations_block,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    score_indices,
    recon_indices=None,
    max_projected_rotations: int | None = None,
    output_complex_dtype=None,
    output_abs2_dtype=None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    **projection_kwargs,
):
    """Project in capped chunks and retain only score/reconstruction windows."""

    if max_projected_rotations is None:
        max_projected_rotations = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)

    projection_kwargs = dict(projection_kwargs)
    projection_kwargs["return_abs2"] = False
    score_indices = jnp.asarray(score_indices, dtype=jnp.int32)
    recon_indices = None if recon_indices is None else jnp.asarray(recon_indices, dtype=jnp.int32)

    n_rotations = int(rotations_block.shape[0])
    if max_projected_rotations is None:
        chunk_ranges = [(0, n_rotations)]
    else:
        max_projected_rotations = max(1, int(max_projected_rotations))
        chunk_ranges = [
            (start, min(start + max_projected_rotations, n_rotations))
            for start in range(0, n_rotations, max_projected_rotations)
        ]

    score_chunks = []
    recon_chunks = []
    for start, stop in chunk_ranges:
        proj_chunk, _ = _compute_sparse_pass2_projections_block(
            mean_for_proj,
            rotations_block[start:stop],
            image_shape,
            proj_volume_shape,
            disc_type,
            max_projected_rotations=None,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            **projection_kwargs,
        )
        score_chunk, recon_chunk = _window_projection_chunk(
            proj_chunk,
            score_indices,
            recon_indices,
            output_complex_dtype=output_complex_dtype,
        )
        score_chunks.append(score_chunk)
        if recon_indices is not None:
            recon_chunks.append(recon_chunk)
        del proj_chunk

    if recon_indices is None:
        return jnp.concatenate(score_chunks, axis=0), None, None
    return _finalize_windowed_projection_chunks(
        tuple(score_chunks),
        tuple(recon_chunks),
        output_abs2_dtype=output_abs2_dtype,
    )


@partial(jax.jit, static_argnames=("output_complex_dtype",))
def _window_projection_chunk(proj_chunk, score_indices, recon_indices, *, output_complex_dtype):
    """Select the score and reconstruction windows of one projection chunk."""

    score_chunk = proj_chunk[:, score_indices]
    if output_complex_dtype is not None:
        score_chunk = score_chunk.astype(output_complex_dtype)
    if recon_indices is None:
        return score_chunk, None
    recon_chunk = proj_chunk[:, recon_indices]
    if output_complex_dtype is not None:
        recon_chunk = recon_chunk.astype(output_complex_dtype)
    return score_chunk, recon_chunk


@partial(jax.jit, static_argnames=("output_abs2_dtype",))
def _finalize_windowed_projection_chunks(score_chunks, recon_chunks, *, output_abs2_dtype):
    """Concatenate projection chunks and form |recon|^2 in one program."""

    score_proj = jnp.concatenate(score_chunks, axis=0)
    recon_proj = jnp.concatenate(recon_chunks, axis=0)
    recon_abs2 = jnp.abs(recon_proj) ** 2
    if output_abs2_dtype is not None:
        recon_abs2 = recon_abs2.astype(output_abs2_dtype)
    return score_proj, recon_proj, recon_abs2
