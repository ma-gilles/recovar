"""Device-resident scoring stage of the K=1 sparse pass-2 prototype.

Stages 1-3 of ``em_device_resident_pass2_design_20260918.md``: gather,
"project" (a gather from the per-iteration projection cache, not a projector
call) and score. The candidate rows and their capacity chunks come from
:mod:`recovar.em.sparse_pass2.resident_candidates` (T5); the per-image image /
CTF / noise operands come from
:func:`recovar.em.sparse_pass2.sparse_pass2_bucket_io._prepare_bucket_io`, run
once per half instead of once per bucket and kept on the device.

What this module changes relative to the compact engine, and what it does not
-----------------------------------------------------------------------------
Unchanged: the candidate set and its order, the per-cell CUDA diff2 arithmetic
(the same fused translate-then-score kernel body the compact engine's pairs
route calls), the RELION common-minimum score conversion including its
optimization barriers, and the float32 scoring precision.

Changed: the *layout* the kernel is called on. The compact engine dispatches
one rectangular ``(B, R, T)`` program per bucket shape; this module dispatches
one flat ``(C_R, T)`` program per capacity class, with ``row_image_ids``
carrying the row-to-image addressing. ``n_valid_rows`` / ``n_valid_images`` are
runtime int32 operands, and the RELION current size is the runtime scalar of
:func:`recovar.em.cuda.kernels.relion_fine_diff2_fused_translate_runtime_flat_rows_f32`,
so a chunk's program is keyed only on
``(row_capacity, image_capacity, n_fine_trans, n_score_pixels)``.

Padding contract
----------------
Rows at or past ``n_valid_rows`` are handed to the kernel with image id ``-1``:
the kernel writes ``+inf`` for the whole row and skips its pixel traversal, so
padding costs no arithmetic. Their candidate mask is all-false as well (T5 pads
``row_mask_mode`` with the empty mode), so the mask alone would already
invalidate them; the ``-1`` is the performance half of the same contract.
Padded image slots carry ``image_ids == -1`` and are never addressed by a valid
row.

Indexing
--------
"image" is the local position of an image inside the half's ``per_image_inputs``
list, exactly as in :mod:`resident_candidates`. Resident operand row ``i`` is
that image. Chunk-local image ids (``row_image_local``) are offsets from
``chunk.image_start``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.sparse_pass2.resident_candidates import (
    CapacityChunk,
    ResidentCandidateTables,
    expand_chunk_mask_jnp,
    materialize_chunk,
)
from recovar.em.sparse_pass2.resident_local_layout import expand_local_chunk_mask_jnp
from recovar.em.sparse_pass2.sparse_pass2_bucket_io import _prepare_bucket_io
from recovar.em.sparse_pass2.sparse_pass2_projection_blocks import (
    _compute_sparse_pass2_windowed_projections_block,
)
from recovar.em.sparse_pass2.sparse_pass2_scoring import (
    _relion_cuda_fine_pixel_weights,
    _relion_powerclass_noise_terms,
)

__all__ = [
    "ResidentChunkScores",
    "ResidentImageOperands",
    "materialize_chunk_device",
    "prepare_resident_image_operands",
    "project_resident_rows",
    "resident_operand_bytes",
    "resident_projection_block_rows",
    "resident_row_projection_bytes",
    "score_all_chunks",
    "score_resident_chunk",
    "score_resident_projected_chunk",
]


@dataclass(frozen=True)
class ResidentImageOperands:
    """Per-half, per-image scoring operands kept on the device.

    Every per-image array is indexed by the local image position used by
    :class:`~recovar.em.sparse_pass2.resident_candidates.ResidentCandidateTables`.

    Fields
    ------
    score_input
        complex64 ``[n_images, n_score_pixels]``: ``direct_score_input``, the
        untranslated image divided by its score weight factors, already gathered
        to the score window. The fused kernel applies the translation per pixel.
    corr_img_score
        float32 ``[n_images, n_score_pixels]``: ``ctf2_over_nv_score``
        (``Minvsigma2 * CTF^2 * scale^2``), window-gathered.
    highres_xi2_half
        float32 ``[n_images]``: RELION's ``powerClass`` tail already divided by
        two, added by the kernel as ``initial_diff2``. ``None`` when the pass
        does not use it (only possible outside exact-Gaussian scoring).
    batch_norm
        ``[n_images]``: the per-image normalization ``_prepare_bucket_io``
        returns; carried for the later M-step/noise stages, unused here.
    translation_prior
        float32 ``[n_images, n_fine_trans]``: the per-image fine translation
        log prior, zero when the pass has no translation prior.
    half_weights
        float32 ``[n_score_pixels]``: the scoring half-image weights of the
        score window (``direct_half_weights`` in the loop).
    translation_angles
        float32 ``[n_fine_trans, 2]``: RELION fine translation angles.
    full_to_compact
        int32 ``[current_size * (current_size // 2 + 1)]``: the RELION
        full-rectangle to compact-window lookup.
    current_size
        The logical RELION current size; passed to the kernel as a runtime
        scalar, so it does not key the program.
    """

    n_images: int
    n_score_pixels: int
    n_fine_trans: int
    current_size: int
    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    batch_norm: jax.Array
    translation_prior: jax.Array
    half_weights: jax.Array
    translation_angles: jax.Array
    full_to_compact: jax.Array

    def __post_init__(self):
        expected = (self.n_images, self.n_score_pixels)
        if self.score_input.shape != expected:
            raise ValueError(f"score_input must have shape {expected}, got {self.score_input.shape}")
        if self.corr_img_score.shape != expected:
            raise ValueError(f"corr_img_score must have shape {expected}, got {self.corr_img_score.shape}")
        if self.translation_prior.shape != (self.n_images, self.n_fine_trans):
            raise ValueError(
                "translation_prior must have shape (n_images, n_fine_trans), got "
                f"{self.translation_prior.shape}",
            )
        if self.half_weights.shape != (self.n_score_pixels,):
            raise ValueError(f"half_weights must have shape ({self.n_score_pixels},)")
        if self.translation_angles.shape != (self.n_fine_trans, 2):
            raise ValueError(f"translation_angles must have shape ({self.n_fine_trans}, 2)")
        if self.highres_xi2_half is not None and self.highres_xi2_half.shape != (self.n_images,):
            raise ValueError(f"highres_xi2_half must have shape ({self.n_images},)")

    def nbytes(self) -> dict:
        """Device bytes of each resident array, plus their total."""

        return resident_operand_bytes(
            n_images=self.n_images,
            n_score_pixels=self.n_score_pixels,
            n_fine_trans=self.n_fine_trans,
            score_complex_bytes=self.score_input.dtype.itemsize,
            real_bytes=self.corr_img_score.dtype.itemsize,
        )


def resident_operand_bytes(
    *,
    n_images: int,
    n_score_pixels: int,
    n_fine_trans: int,
    score_complex_bytes: int = 8,
    real_bytes: int = 4,
) -> dict:
    """Resident per-image operand bytes for one half at a given state.

    The two dominant terms are ``n_images * n_score_pixels`` of complex score
    input and of real ``corr_img``; the priors and per-image scalars are a
    rounding error next to them.
    """

    n_images = int(n_images)
    n_score_pixels = int(n_score_pixels)
    n_fine_trans = int(n_fine_trans)
    parts = {
        "score_input": n_images * n_score_pixels * int(score_complex_bytes),
        "corr_img_score": n_images * n_score_pixels * int(real_bytes),
        "translation_prior": n_images * n_fine_trans * int(real_bytes),
        "highres_xi2_half": n_images * int(real_bytes),
        "batch_norm": n_images * int(real_bytes),
        "half_weights": n_score_pixels * int(real_bytes),
        "translation_angles": n_fine_trans * 2 * 4,
    }
    parts["total"] = sum(parts.values())
    return parts


def prepare_resident_image_operands(
    experiment_dataset,
    image_indices,
    *,
    bucket_io_kwargs: dict,
    half_weights,
    full_to_compact,
    translation_angles,
    window_indices,
    windowed_prepare: bool,
    image_shape,
    current_size,
    n_fine_trans: int,
    use_exact_relion_gaussian: bool,
    accumulate_noise: bool,
    source_faithful_spectrum_norm: bool,
    fine_translation_prior_2d=None,
    score_real_dtype=jnp.float32,
    image_batch_size: int = 256,
) -> ResidentImageOperands:
    """Run ``_prepare_bucket_io`` once per half and keep its operands resident.

    ``image_indices`` are the dataset image indices of this half, in the order
    the pass processes them (RELION particle order); resident row ``i``
    corresponds to ``image_indices[i]``. ``bucket_io_kwargs`` must be exactly
    the keyword arguments the K=1 loop passes to
    :func:`_prepare_bucket_io`, minus the four per-batch positional operands
    (``experiment_dataset``, ``batch``, ``ctf_params``, ``image_indices``) and
    minus ``return_direct_scoring_io``, which is forced on here because the
    fused translate route needs ``direct_score_input``.

    ``image_batch_size`` only controls how many images one ``_prepare_bucket_io``
    call covers; the results are per image, so it does not change the output.
    The last batch is short rather than padded.

    The window gather of ``ctf2_over_nv_half`` and the ``powerClass`` terms
    reproduce the loop's post-prepare block for the score window.
    """

    image_indices = np.asarray(image_indices)
    if image_indices.ndim != 1:
        raise ValueError(f"image_indices must be 1-D, got {image_indices.shape}")
    n_images = int(image_indices.shape[0])
    if n_images == 0:
        raise ValueError("prepare_resident_image_operands needs at least one image")
    image_batch_size = int(image_batch_size)
    if image_batch_size <= 0:
        raise ValueError(f"image_batch_size must be positive, got {image_batch_size}")
    if "return_direct_scoring_io" in bucket_io_kwargs:
        raise ValueError("return_direct_scoring_io is forced on; remove it from bucket_io_kwargs")

    use_window = window_indices is not None
    window_indices_device = (
        None if not use_window else jnp.asarray(window_indices, dtype=jnp.int32)
    )
    # Position of each dataset image index within the requested order, so a
    # fetch that returns its own order is scattered back to that order
    # exactly (a pure gather, no arithmetic).
    position_of = {int(index): position for position, index in enumerate(image_indices.tolist())}
    if len(position_of) != n_images:
        raise ValueError("image_indices must not repeat an image")

    score_input_rows: list[np.ndarray] = [None] * n_images
    corr_img_rows: list[np.ndarray] = [None] * n_images
    xi2_rows: list[np.ndarray] = [None] * n_images
    norm_rows: list[np.ndarray] = [None] * n_images
    n_score_pixels = None
    highres_available = None

    for start in range(0, n_images, image_batch_size):
        batch_image_indices = image_indices[start : start + image_batch_size]
        batch_data, ctf_params, fetched_indices = fetch_indexed_batch(
            experiment_dataset, batch_image_indices
        )
        fetched_indices = np.asarray(fetched_indices)
        prepared = _prepare_bucket_io(
            experiment_dataset,
            jnp.asarray(batch_data),
            ctf_params,
            fetched_indices,
            return_direct_scoring_io=True,
            **bucket_io_kwargs,
        )
        (
            _shifted_score_half,
            _shifted_recon_half,
            batch_norm,
            ctf2_over_nv_half,
            _ctf2_over_nv_half_with_dc,
            _shifted_score_half_with_dc,
            processed_score_half_for_noise,
            _shifted_corrected_score_half,
            direct_score_input,
            *_rest,
        ) = prepared

        if use_window and not windowed_prepare:
            score_input = direct_score_input[:, window_indices_device]
            corr_img_score = ctf2_over_nv_half[:, window_indices_device]
        else:
            score_input = direct_score_input
            corr_img_score = ctf2_over_nv_half
        highres_xi2_half, _norm_high_shell = _relion_powerclass_noise_terms(
            processed_score_half_for_noise,
            image_shape=image_shape,
            current_size=current_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        )
        if highres_available is None:
            highres_available = highres_xi2_half is not None
        elif highres_available != (highres_xi2_half is not None):
            raise ValueError("powerClass tail availability changed between image batches")

        score_input = np.asarray(score_input)
        corr_img_score = np.asarray(corr_img_score)
        batch_norm = np.asarray(batch_norm).reshape(-1)
        highres_host = None if highres_xi2_half is None else np.asarray(highres_xi2_half).reshape(-1)
        if n_score_pixels is None:
            n_score_pixels = int(score_input.shape[1])
        elif int(score_input.shape[1]) != n_score_pixels:
            raise ValueError("score pixel count changed between image batches")

        for local, dataset_index in enumerate(fetched_indices.tolist()):
            position = position_of.get(int(dataset_index))
            if position is None:
                raise ValueError(
                    f"the dataset returned image {dataset_index}, which is not in image_indices",
                )
            score_input_rows[position] = score_input[local]
            corr_img_rows[position] = corr_img_score[local]
            norm_rows[position] = batch_norm[local]
            if highres_host is not None:
                xi2_rows[position] = highres_host[local]

    if any(row is None for row in score_input_rows):
        raise ValueError("the dataset did not return every requested image")

    score_input_device = jnp.asarray(np.stack(score_input_rows, axis=0))
    corr_img_device = jnp.asarray(np.stack(corr_img_rows, axis=0), dtype=score_real_dtype)
    batch_norm_device = jnp.asarray(np.stack(norm_rows, axis=0))
    highres_device = (
        None
        if not highres_available
        else jnp.asarray(np.stack(xi2_rows, axis=0), dtype=jnp.float32)
    )

    if fine_translation_prior_2d is None:
        translation_prior = jnp.zeros((n_images, int(n_fine_trans)), dtype=score_real_dtype)
    else:
        translation_prior = jnp.asarray(
            np.asarray(fine_translation_prior_2d)[image_indices], dtype=score_real_dtype
        )

    return ResidentImageOperands(
        n_images=n_images,
        n_score_pixels=int(n_score_pixels),
        n_fine_trans=int(n_fine_trans),
        current_size=int(current_size),
        score_input=score_input_device,
        corr_img_score=corr_img_device,
        highres_xi2_half=highres_device,
        batch_norm=batch_norm_device,
        translation_prior=translation_prior,
        half_weights=jnp.asarray(half_weights),
        translation_angles=jnp.asarray(translation_angles, dtype=jnp.float32),
        full_to_compact=jnp.asarray(full_to_compact, dtype=jnp.int32),
    )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["raw_diff2", "scores", "min_diff2"],
    meta_fields=[],
)
@dataclass(frozen=True)
class ResidentChunkScores:
    """Scoring outputs of one capacity chunk, all on the device.

    ``raw_diff2`` and ``scores`` are ``[row_capacity, n_fine_trans]``;
    ``min_diff2`` is ``[image_capacity]``. Invalid cells, and every cell of a
    padded row, are ``+inf`` in ``raw_diff2`` and ``-inf`` in ``scores``, the
    same invalidation the rectangular consumers already apply.

    Registered as a JAX pytree so the scoring program can return it directly
    from ``jax.jit`` instead of an unnamed tuple.
    """

    raw_diff2: jax.Array
    scores: jax.Array
    min_diff2: jax.Array


@partial(
    jax.jit,
    static_argnames=("row_capacity", "image_capacity", "n_fine_trans", "n_score_pixels"),
)
def score_resident_chunk(
    row_image_local,  # int32 [C_R] chunk-local image id of each row
    row_fine_rot,  # int32 [C_R] index into the iteration's fine rotation grid
    row_log_prior,  # float32 [C_R] rotation log prior of each row (nats)
    row_mask_bits,  # uint32 [C_R] per-row coarse translation bitset
    row_mask_mode,  # int8 [C_R] 0 full, 1 bitset, 2 empty
    n_valid_rows,  # int32 scalar, runtime
    image_ids,  # int32 [C_B] resident image row of each chunk image slot, -1 when padded
    projection_score_cache,  # complex64 [n_fine_rot, N] per-iteration cache
    score_input,  # complex64 [n_images, N]
    corr_img_score,  # real [n_images, N]
    highres_xi2_half,  # float32 [n_images] or None
    translation_prior,  # real [n_images, T]
    *,
    half_weights,  # real [N]
    translation_angles,  # float32 [T, 2]
    full_to_compact,  # int32 [P]
    fine_translation_parent,  # int32 [T]
    logical_current_size,  # int32 scalar, runtime
    row_capacity: int,
    image_capacity: int,
    n_fine_trans: int,
    n_score_pixels: int,
):
    """Score one capacity chunk in a single device program.

    Gathers the chunk's images and projection rows, forms the RELION pixel
    weights, calls the flat-row fused translate-and-score CUDA kernel, expands
    the candidate mask from its bitsets, reduces the per-image common minimum
    over the chunk's rows, and applies RELION's diff2-to-log-weight conversion.

    There is no host round trip and no Python loop over rows or images inside.
    ``n_valid_rows`` and ``logical_current_size`` are traced scalars, so two
    chunks of the same capacity class with different occupancy or a different
    RELION current size reuse one compiled program.

    The per-image minimum computed here is the image's *global* minimum,
    because a capacity chunk holds a contiguous image range and an image's rows
    never straddle two chunks.

    The chunk's ``n_valid_images`` is not a separate operand: padded image
    slots are exactly the slots with ``image_ids == -1``, and no valid row ever
    addresses one.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    row_fine_rot = jnp.asarray(row_fine_rot, dtype=jnp.int32)
    image_ids = jnp.asarray(image_ids, dtype=jnp.int32)
    if row_image_local.shape != (row_capacity,):
        raise ValueError(f"row_image_local must have shape ({row_capacity},)")
    if row_fine_rot.shape != (row_capacity,):
        raise ValueError(f"row_fine_rot must have shape ({row_capacity},)")
    if image_ids.shape != (image_capacity,):
        raise ValueError(f"image_ids must have shape ({image_capacity},)")
    if int(score_input.shape[1]) != n_score_pixels:
        raise ValueError(f"score_input must have {n_score_pixels} pixels")
    if int(translation_angles.shape[0]) != n_fine_trans:
        raise ValueError(f"translation_angles must have {n_fine_trans} rows")

    # --- stage 1: gather ---------------------------------------------------
    safe_image_ids = jnp.where(image_ids >= 0, image_ids, jnp.int32(0))
    chunk_image = jnp.asarray(score_input, dtype=jnp.complex64)[safe_image_ids]
    chunk_corr = corr_img_score[safe_image_ids]
    chunk_translation_prior = translation_prior[safe_image_ids]
    chunk_initial_diff2 = (
        jnp.zeros((image_capacity,), dtype=jnp.float32)
        if highres_xi2_half is None
        else jnp.asarray(highres_xi2_half, dtype=jnp.float32)[safe_image_ids]
    )

    # --- stage 2: "project" = gather the cached fine-rotation projections ---
    reference = jnp.asarray(projection_score_cache, dtype=jnp.complex64)[row_fine_rot]

    # --- stage 3: score ----------------------------------------------------
    row_is_valid = jnp.arange(row_capacity, dtype=jnp.int32) < jnp.asarray(
        n_valid_rows, dtype=jnp.int32
    )
    candidate_mask = expand_chunk_mask_jnp(
        row_mask_bits, row_mask_mode, fine_translation_parent
    )
    return _score_flat_rows(
        reference,
        row_image_local,
        row_log_prior,
        chunk_image,
        chunk_corr,
        chunk_translation_prior,
        chunk_initial_diff2,
        candidate_mask,
        row_is_valid,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        logical_current_size=logical_current_size,
        image_capacity=image_capacity,
    )


def _score_flat_rows(
    reference,  # complex64 [C_R, N] the rows' projections, cached or projected
    row_image_local,  # int32 [C_R]
    row_log_prior,  # real [C_R]
    chunk_image,  # complex64 [C_B, N]
    chunk_corr,  # real [C_B, N]
    chunk_translation_prior,  # real [C_B, T]
    chunk_initial_diff2,  # float32 [C_B]
    candidate_mask,  # bool [C_R, T] or None for "every cell is a candidate"
    row_is_valid,  # bool [C_R]
    *,
    half_weights,
    translation_angles,
    full_to_compact,
    logical_current_size,
    image_capacity: int,
):
    """RELION's flat-row diff2, common minimum and log-weight conversion.

    Shared by the cached-projection route (the global pass 2, which gathers a
    per-iteration fine-rotation cache) and the projected-row route (local
    search, whose order-5 fine grid cannot be cached). Only the source of
    ``reference`` and the shape of the candidate mask differ between them;
    every statement below is common, so the two routes cannot drift apart.
    """

    from recovar.em.cuda import kernels as em_cuda_kernels

    weights = _relion_cuda_fine_pixel_weights(
        chunk_corr, jnp.asarray(half_weights)[None, :]
    ).astype(jnp.float32)
    # Padded rows are handed to the kernel as image -1: it writes +inf for the
    # whole row and skips the pixel traversal.
    kernel_row_image_ids = jnp.where(row_is_valid, row_image_local, jnp.int32(-1))
    raw_from_kernel = em_cuda_kernels.relion_fine_diff2_fused_translate_runtime_flat_rows_f32(
        reference,
        kernel_row_image_ids,
        chunk_image,
        jnp.asarray(translation_angles, dtype=jnp.float32),
        weights,
        jnp.asarray(full_to_compact, dtype=jnp.int32),
        jnp.asarray(logical_current_size, dtype=jnp.int32),
        chunk_initial_diff2,
    )

    if candidate_mask is None:
        candidate_mask = jnp.broadcast_to(row_is_valid[:, None], raw_from_kernel.shape)
    else:
        candidate_mask = candidate_mask & row_is_valid[:, None]
    raw_diff2 = jnp.where(
        candidate_mask,
        raw_from_kernel,
        jnp.asarray(jnp.inf, dtype=raw_from_kernel.dtype),
    )

    # --- per-image common minimum over the chunk's rows --------------------
    valid = candidate_mask & jnp.isfinite(raw_diff2)
    per_row_min = jnp.min(
        jnp.where(valid, raw_diff2, jnp.asarray(jnp.inf, dtype=raw_diff2.dtype)), axis=1
    )
    segment_min = jax.ops.segment_min(
        per_row_min,
        row_image_local,
        num_segments=image_capacity,
        indices_are_sorted=True,
    )
    min_diff2 = jnp.where(
        jnp.isfinite(segment_min), segment_min, jnp.asarray(0.0, dtype=segment_min.dtype)
    )

    # --- RELION diff2 -> log weight, in the flat-row layout ----------------
    # Same operation order as ``_relion_cuda_fine_diff2_to_scores``: the two
    # optimization barriers keep ``(prior + min) - diff2`` from being
    # re-associated, which changes float32 tie-breaking at diff2 ~ 1e3.
    real_dtype = raw_diff2.dtype
    row_min = min_diff2[row_image_local].astype(real_dtype)
    valid = valid & (raw_diff2 >= row_min[:, None])
    scores = jnp.asarray(row_log_prior, dtype=real_dtype)[:, None] + jnp.asarray(
        chunk_translation_prior, dtype=real_dtype
    )[row_image_local]
    scores = jax.lax.optimization_barrier(scores)
    scores = scores + row_min[:, None]
    scores = jax.lax.optimization_barrier(scores)
    scores = scores - raw_diff2
    scores = jnp.where(valid & jnp.isfinite(scores), scores, -jnp.inf)

    return ResidentChunkScores(raw_diff2=raw_diff2, scores=scores, min_diff2=min_diff2)


@partial(
    jax.jit,
    static_argnames=("row_capacity", "image_capacity", "n_fine_trans", "n_score_pixels"),
)
def score_resident_projected_chunk(
    reference,  # complex64 [C_R, N] projections of this chunk's own rows
    row_image_local,  # int32 [C_R] chunk-local image id of each row
    row_log_prior,  # real [C_R] rotation log prior of each row (nats)
    row_mask_bits,  # uint8 [C_R, ceil(T/8)] little-endian, or None for full support
    n_valid_rows,  # int32 scalar, runtime
    image_ids,  # int32 [C_B] resident image row of each slot, -1 when padded
    score_input,  # complex64 [n_images, N]
    corr_img_score,  # real [n_images, N]
    highres_xi2_half,  # float32 [n_images] or None
    translation_prior,  # real [n_images, T]
    *,
    half_weights,  # real [N]
    translation_angles,  # float32 [T, 2]
    full_to_compact,  # int32 [P]
    logical_current_size,  # int32 scalar, runtime
    row_capacity: int,
    image_capacity: int,
    n_fine_trans: int,
    n_score_pixels: int,
):
    """Score one capacity chunk whose rows were projected, not gathered.

    The local-search twin of :func:`score_resident_chunk`: identical arithmetic
    (both call :func:`_score_flat_rows`), with two inputs replaced.
    ``reference`` arrives already projected for this chunk's own rows, because
    the local fine grid at order 5 has 2.4M rotations and is never materialized;
    and the candidate mask is the layout's per-row little-endian packing over
    the fine translation axis rather than a per-parent uint32 bitset, because
    the local pass 2 runs more than 32 translations.

    ``row_mask_bits=None`` is the layout's compact spelling of full support and
    is not expanded; padded rows are excluded by ``n_valid_rows`` alone.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    image_ids = jnp.asarray(image_ids, dtype=jnp.int32)
    if row_image_local.shape != (row_capacity,):
        raise ValueError(f"row_image_local must have shape ({row_capacity},)")
    if reference.shape != (row_capacity, n_score_pixels):
        raise ValueError(
            f"reference must have shape ({row_capacity}, {n_score_pixels}), got {reference.shape}"
        )
    if image_ids.shape != (image_capacity,):
        raise ValueError(f"image_ids must have shape ({image_capacity},)")
    if int(translation_angles.shape[0]) != n_fine_trans:
        raise ValueError(f"translation_angles must have {n_fine_trans} rows")

    safe_image_ids = jnp.where(image_ids >= 0, image_ids, jnp.int32(0))
    chunk_image = jnp.asarray(score_input, dtype=jnp.complex64)[safe_image_ids]
    chunk_corr = corr_img_score[safe_image_ids]
    chunk_translation_prior = translation_prior[safe_image_ids]
    chunk_initial_diff2 = (
        jnp.zeros((image_capacity,), dtype=jnp.float32)
        if highres_xi2_half is None
        else jnp.asarray(highres_xi2_half, dtype=jnp.float32)[safe_image_ids]
    )

    row_is_valid = jnp.arange(row_capacity, dtype=jnp.int32) < jnp.asarray(
        n_valid_rows, dtype=jnp.int32
    )
    candidate_mask = expand_local_chunk_mask_jnp(row_mask_bits, n_trans=n_fine_trans)
    return _score_flat_rows(
        jnp.asarray(reference, dtype=jnp.complex64),
        row_image_local,
        row_log_prior,
        chunk_image,
        chunk_corr,
        chunk_translation_prior,
        chunk_initial_diff2,
        candidate_mask,
        row_is_valid,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        logical_current_size=logical_current_size,
        image_capacity=image_capacity,
    )


# ---------------------------------------------------------------------------
# Projection by rows (local search: no per-iteration fine-rotation cache)
# ---------------------------------------------------------------------------


def resident_row_projection_bytes(
    *,
    n_score_pixels: int,
    n_recon_pixels: int,
    complex_bytes: int = 8,
    real_bytes: int = 4,
) -> int:
    """Device bytes one projected row holds while its chunk is being scored.

    A row carries its score-window projection (complex), its reconstruction
    window projection (complex) and that window's ``|proj|^2`` (real). This is
    the term that decides a local chunk's row capacity, because the local route
    cannot amortize projections across chunks the way the global route's
    per-iteration cache does.
    """

    return max(int(n_score_pixels), 0) * int(complex_bytes) + max(
        int(n_recon_pixels), 0
    ) * (int(complex_bytes) + int(real_bytes))


def resident_projection_block_rows(
    *,
    n_score_pixels: int,
    n_recon_pixels: int,
    max_block_bytes: int,
    complex_bytes: int = 8,
    real_bytes: int = 4,
) -> int:
    """Rows per projector call, from a byte budget on the projected rows.

    Only the transient of one projector call is bounded here; the caller bounds
    the chunk's resident projections separately with
    :func:`resident_row_projection_bytes`. The result is at least one row, so a
    single heavy row is never unprojectable.
    """

    per_row = resident_row_projection_bytes(
        n_score_pixels=n_score_pixels,
        n_recon_pixels=n_recon_pixels,
        complex_bytes=complex_bytes,
        real_bytes=real_bytes,
    )
    return max(int(max_block_bytes) // max(per_row, 1), 1)


def project_resident_rows(
    mean_for_proj,
    rotations,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    score_indices,
    recon_indices,
    max_projected_rotations: int,
    output_complex_dtype=None,
    output_abs2_dtype=None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    **projection_kwargs,
):
    """Project a chunk's own rows into the score and reconstruction windows.

    Stage 2 of the design for local search. The compact/global route gathers
    ``projection_cache["score"][row_fine_rot]``; the local fine grid has up to
    2.4M rotations at order 5 and is never cached, so a chunk's rows are
    projected here instead, in blocks of ``max_projected_rotations`` rows so the
    projector's transient stays inside its byte budget. The blocking is the
    compact engine's own
    :func:`_compute_sparse_pass2_windowed_projections_block`, so the projector
    call, its window gathers and the ``|proj|^2`` formation are the same code
    the cache build uses, only driven by row rotations instead of grid rotations.

    Returns ``(score_proj [Q, N_score], recon_proj [Q, N_recon],
    recon_abs2 [Q, N_recon])``.
    """

    return _compute_sparse_pass2_windowed_projections_block(
        mean_for_proj,
        rotations,
        image_shape,
        proj_volume_shape,
        disc_type,
        score_indices=score_indices,
        recon_indices=recon_indices,
        max_projected_rotations=int(max_projected_rotations),
        output_complex_dtype=output_complex_dtype,
        output_abs2_dtype=output_abs2_dtype,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=relion_projector_r_max,
        projection_padding_factor=projection_padding_factor,
        **projection_kwargs,
    )


def materialize_chunk_device(tables: ResidentCandidateTables, chunk: CapacityChunk) -> dict:
    """Host-gather one chunk (T5) and upload its arrays once.

    Returned dict keys match :func:`resident_candidates.materialize_chunk`; the
    array values are device arrays and ``n_valid_rows`` / ``n_valid_images``
    are device int32 scalars, so the scoring program receives them as runtime
    operands rather than as compile keys.
    """

    host = materialize_chunk(tables, chunk)
    return {
        "row_image_local": jnp.asarray(host["row_image_local"], dtype=jnp.int32),
        "row_fine_rot": jnp.asarray(host["row_fine_rot"], dtype=jnp.int32),
        "row_parent_local": jnp.asarray(host["row_parent_local"], dtype=jnp.int32),
        "row_log_prior": jnp.asarray(host["row_log_prior"], dtype=jnp.float32),
        "row_mask_bits": jnp.asarray(host["row_mask_bits"], dtype=jnp.uint32),
        "row_mask_mode": jnp.asarray(host["row_mask_mode"], dtype=jnp.int8),
        "n_valid_rows": jnp.asarray(host["n_valid_rows"], dtype=jnp.int32),
        "n_valid_images": jnp.asarray(host["n_valid_images"], dtype=jnp.int32),
        "image_ids": jnp.asarray(host["image_ids"], dtype=jnp.int32),
    }


def score_all_chunks(
    tables: ResidentCandidateTables,
    chunks,
    operands: ResidentImageOperands,
    projection_score_cache,
    fine_translation_parent,
    *,
    chunk_arrays=None,
):
    """Score every chunk of one half, one device program per chunk.

    This Python loop is the only host loop of the stage: every chunk's arrays
    are uploaded before the loop starts (or supplied by the caller through
    ``chunk_arrays``), so the loop itself only dispatches. It returns one
    :class:`ResidentChunkScores` per chunk, in chunk order.
    """

    chunks = list(chunks)
    if chunk_arrays is None:
        chunk_arrays = [materialize_chunk_device(tables, chunk) for chunk in chunks]
    chunk_arrays = list(chunk_arrays)
    if len(chunk_arrays) != len(chunks):
        raise ValueError("chunk_arrays must have one entry per chunk")

    projection_score_cache = jnp.asarray(projection_score_cache)
    fine_translation_parent = jnp.asarray(fine_translation_parent, dtype=jnp.int32)
    logical_current_size = jnp.asarray(operands.current_size, dtype=jnp.int32)

    results = []
    for chunk, arrays in zip(chunks, chunk_arrays, strict=True):
        results.append(
            score_resident_chunk(
                arrays["row_image_local"],
                arrays["row_fine_rot"],
                arrays["row_log_prior"],
                arrays["row_mask_bits"],
                arrays["row_mask_mode"],
                arrays["n_valid_rows"],
                arrays["image_ids"],
                projection_score_cache,
                operands.score_input,
                operands.corr_img_score,
                operands.highres_xi2_half,
                operands.translation_prior,
                half_weights=operands.half_weights,
                translation_angles=operands.translation_angles,
                full_to_compact=operands.full_to_compact,
                fine_translation_parent=fine_translation_parent,
                logical_current_size=logical_current_size,
                row_capacity=int(chunk.row_capacity),
                image_capacity=int(chunk.image_capacity),
                n_fine_trans=int(operands.n_fine_trans),
                n_score_pixels=int(operands.n_score_pixels),
            )
        )
    return results
