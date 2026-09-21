"""Adapter from the exact local-search layout to device-resident row tables (T12).

The device-resident pass-2 stages (T5-T9b) consume a flat, image-CSR table of
candidate rows plus fixed-capacity chunks. The global pass 2 builds that table
from
:func:`recovar.em.scoring.sparse_bucket_arrays._prepare_per_image_pass2_inputs`
(see :mod:`recovar.em.sparse_pass2.resident_candidates`). Local search instead
carries its candidate set in a
:class:`~recovar.em.local.local_layout.LocalHypothesisLayout`, whose rows
already are the flat, image-CSR objects the resident stages want, with three
differences this module resolves:

1. **Rotations are per row, not ids into a shared grid.** The local fine grid at
   order 5 has 2.4M rotations and is never materialized, so a row carries its
   own ``rotations_flat[row]`` and ``mstep_rotations_flat[row]``. The resident
   local driver therefore projects a chunk's rows on the fly
   (:func:`recovar.em.sparse_pass2.resident_scoring.project_resident_rows`)
   instead of gathering a per-iteration projection cache. Because the layout's
   flat order *is* the CSR order, a chunk's rotations are a contiguous slice,
   not a gather.
2. **Masks are per row and wider than 32 translations.** ``sample_mask_bits`` is
   already ``np.packbits(..., bitorder="little")`` over the fine translation
   axis: uint8 ``[n_rows, ceil(T/8)]``. The resident candidate table packs a
   per-*parent* uint32 bitset over at most 32 *coarse* translations, which
   cannot hold the local per-row mask (the auto-refine local pass 2 runs 84 fine
   translations). This module keeps the layout's packing and expands it on the
   device with the same little-endian bit order
   (:func:`expand_local_chunk_mask_jnp`).
   ``sample_mask_bits is None`` is the layout's compact spelling of "every
   (row, translation) is a candidate" and is preserved as such, never expanded
   into an all-ones host array.
3. **The rotation-posterior histogram bins by parent.** ``rotation_posterior_ids_flat``
   maps each fine row to the coarse/parent rotation whose posterior mass it
   contributes to, over ``n_global_rotations`` bins. When the layout does not
   carry it, the local engine falls back to the fine ``rotation_ids_flat``; this
   module reproduces that fallback (see ``local_bucket_stages.py`` lines
   ~704-709).

Everything here is host planning in plain numpy, plus one ``jax.numpy`` mask
expander used inside the jitted chunk programs. No scoring, projection or
M-step arithmetic lives in this module.

Index conventions
-----------------
* "image" is the local position ``0..n_images-1`` inside the layout, i.e. the
  index used for ``rotation_offsets``/``translation_log_priors``. It is not a
  dataset particle id; the driver keeps the dataset index alongside.
* "row" is one (image, local rotation) candidate, in the layout's own flat
  order. Rows are never reordered, so RELION particle order is preserved for the
  x-half BPref exactly as the bucketed local engine preserves it.
* A cell ``(row, t)`` is a candidate iff the row's mask bit ``t`` is set (or the
  layout has no mask at all).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recovar.em.sparse_pass2.resident_candidates import CapacityChunk, plan_capacity_chunks

__all__ = [
    "ResidentLocalTables",
    "expand_local_chunk_mask_jnp",
    "expand_local_mask_rows",
    "materialize_local_chunk",
    "plan_local_capacity_chunks",
    "tables_from_local_layout",
]

# Padded rows must never validate a cell and must never project into a real
# rotation. An all-zero mask byte string is "no candidate translation", which is
# the only property the scoring stage relies on; the identity rotation keeps the
# projector's texture fetch in range for rows whose posterior is zero anyway.
_ROW_LOG_PRIOR_PAD = np.float32(-1e30)


@dataclass(frozen=True)
class ResidentLocalTables:
    """Flat, image-CSR view of a local-search hypothesis layout.

    Field units: ``row_log_prior`` is a natural-log prior density in nats (the
    layout's ``rotation_log_priors_flat``); ``translation_log_prior`` likewise,
    per image and fine translation. Every other field is an integer id, a
    rotation matrix or a bitset.

    ``rotations``/``mstep_rotations`` are views into the layout's own arrays, not
    copies, so building the table costs no bulk memory.
    """

    n_images: int
    n_rows: int
    n_trans: int
    n_posterior_bins: int
    # CSR row ranges: image i owns rows [row_offsets[i], row_offsets[i + 1]).
    row_offsets: np.ndarray  # int64 [n_images + 1]
    row_image: np.ndarray  # int32 [n_rows]
    row_log_prior: np.ndarray  # float32 [n_rows]
    row_posterior_id: np.ndarray  # int32 [n_rows], bin of the rotation histogram
    row_rotation_id: np.ndarray  # int64 [n_rows], the layout's fine rotation id
    # uint8 [n_rows, n_mask_bytes] little-endian over the fine translation axis,
    # or None for "every cell is a candidate".
    row_mask_bits: np.ndarray | None
    n_mask_bytes: int
    rotations: np.ndarray  # float32/float64 [n_rows, 3, 3]
    mstep_rotations: np.ndarray  # float32/float64 [n_rows, 3, 3]
    source_eulers: np.ndarray | None  # float64 [n_rows, 3] or None
    translation_grid: np.ndarray  # [n_trans, n_dims]
    translation_log_prior: np.ndarray  # [n_images, n_trans]

    def __post_init__(self):
        if self.row_offsets.shape != (self.n_images + 1,):
            raise ValueError("row_offsets must have shape (n_images + 1,)")
        if int(self.row_offsets[-1]) != int(self.n_rows):
            raise ValueError("row_offsets[-1] must equal n_rows")
        for name in ("row_image", "row_log_prior", "row_posterior_id", "row_rotation_id"):
            arr = getattr(self, name)
            if arr.shape != (self.n_rows,):
                raise ValueError(f"{name} must have shape (n_rows,), got {arr.shape}")
        for name in ("rotations", "mstep_rotations"):
            arr = getattr(self, name)
            if arr.shape != (self.n_rows, 3, 3):
                raise ValueError(f"{name} must have shape (n_rows, 3, 3), got {arr.shape}")
        if self.row_mask_bits is not None and self.row_mask_bits.shape != (
            self.n_rows,
            self.n_mask_bytes,
        ):
            raise ValueError(
                "row_mask_bits must have shape (n_rows, n_mask_bytes), got "
                f"{self.row_mask_bits.shape}",
            )
        if self.translation_log_prior.shape != (self.n_images, self.n_trans):
            raise ValueError(
                "translation_log_prior must have shape (n_images, n_trans), got "
                f"{self.translation_log_prior.shape}",
            )


def tables_from_local_layout(layout, *, rotation_dtype=np.float32) -> ResidentLocalTables:
    """Flatten a :class:`LocalHypothesisLayout` into resident row tables.

    Only the fields the resident stages read are touched:
    ``rotation_offsets``, ``rotations_flat``, ``mstep_rotations_flat`` (falling
    back to ``rotations_flat`` exactly as
    ``local_layout._local_mstep_rotations`` does), ``rotation_log_priors_flat``,
    ``rotation_ids_flat``, ``rotation_posterior_ids_flat``, ``source_eulers_flat``,
    ``translation_grid``, ``translation_log_priors`` and ``sample_mask_bits``.

    ``rotation_dtype`` casts only the two rotation arrays; the local engine's
    own float32 default is the RELION accelerated-GPU precision, and a float64
    layout (the diagnostic double path) is preserved by passing ``np.float64``.
    """

    row_offsets = np.asarray(layout.rotation_offsets, dtype=np.int64)
    n_images = int(row_offsets.shape[0]) - 1
    n_rows = int(row_offsets[-1])

    rotations = np.asarray(layout.rotations_flat, dtype=rotation_dtype)
    if rotations.shape != (n_rows, 3, 3):
        raise ValueError(
            f"rotations_flat must have shape ({n_rows}, 3, 3), got {rotations.shape}"
        )
    mstep_source = layout.mstep_rotations_flat
    mstep_rotations = (
        rotations if mstep_source is None else np.asarray(mstep_source, dtype=rotation_dtype)
    )
    if mstep_rotations.shape != (n_rows, 3, 3):
        raise ValueError(
            f"mstep_rotations_flat must have shape ({n_rows}, 3, 3), got {mstep_rotations.shape}"
        )

    row_image = np.repeat(
        np.arange(n_images, dtype=np.int32), np.diff(row_offsets).astype(np.int64)
    ).astype(np.int32)
    row_log_prior = np.asarray(layout.rotation_log_priors_flat, dtype=np.float32).reshape(-1)
    row_rotation_id = np.asarray(layout.rotation_ids_flat, dtype=np.int64).reshape(-1)

    # The bucketed engine bins the rotation posterior by parent id when the
    # layout carries one and by fine rotation id otherwise
    # (local_bucket_stages.py ~704-709); the histogram width follows the same
    # choice, so a fine-id fallback needs a table wide enough for those ids.
    posterior_source = getattr(layout, "rotation_posterior_ids_flat", None)
    if posterior_source is None:
        row_posterior_id = row_rotation_id.astype(np.int64, copy=False)
        n_posterior_bins = int(row_posterior_id.max(initial=-1)) + 1
    else:
        row_posterior_id = np.asarray(posterior_source, dtype=np.int64).reshape(-1)
        n_posterior_bins = int(layout.n_global_rotations)
        if n_rows and int(row_posterior_id.max(initial=-1)) >= n_posterior_bins:
            raise ValueError(
                "rotation_posterior_ids_flat references a bin outside n_global_rotations"
            )
    if n_rows and int(row_posterior_id.min(initial=0)) < 0:
        raise ValueError("rotation posterior ids must be non-negative")
    row_posterior_id = row_posterior_id.astype(np.int32, copy=False)

    translation_grid = np.asarray(layout.translation_grid)
    n_trans = int(translation_grid.shape[0])
    translation_log_prior = np.asarray(layout.translation_log_priors)
    if translation_log_prior.ndim != 2:
        raise ValueError(
            f"translation_log_priors must be 2-D, got {translation_log_prior.shape}"
        )

    n_mask_bytes = (n_trans + 7) // 8
    row_mask_bits = layout.sample_mask_bits
    if row_mask_bits is not None:
        row_mask_bits = np.asarray(row_mask_bits, dtype=np.uint8)
        if row_mask_bits.shape != (n_rows, n_mask_bytes):
            raise ValueError(
                "sample_mask_bits must have shape (n_rows, ceil(n_trans / 8)) = "
                f"{(n_rows, n_mask_bytes)}, got {row_mask_bits.shape}",
            )

    source_eulers = getattr(layout, "source_eulers_flat", None)
    if source_eulers is not None:
        source_eulers = np.asarray(source_eulers, dtype=np.float64)
        if source_eulers.shape != (n_rows, 3):
            raise ValueError(
                f"source_eulers_flat must have shape ({n_rows}, 3), got {source_eulers.shape}"
            )

    for name, arr in (
        ("rotation_log_priors_flat", row_log_prior),
        ("rotation_ids_flat", row_rotation_id),
    ):
        if arr.shape != (n_rows,):
            raise ValueError(f"{name} must have shape ({n_rows},), got {arr.shape}")

    return ResidentLocalTables(
        n_images=n_images,
        n_rows=n_rows,
        n_trans=n_trans,
        n_posterior_bins=int(n_posterior_bins),
        row_offsets=row_offsets,
        row_image=row_image,
        row_log_prior=row_log_prior,
        row_posterior_id=row_posterior_id,
        row_rotation_id=row_rotation_id,
        row_mask_bits=row_mask_bits,
        n_mask_bytes=int(n_mask_bytes),
        rotations=rotations,
        mstep_rotations=mstep_rotations,
        source_eulers=source_eulers,
        translation_grid=translation_grid,
        translation_log_prior=translation_log_prior,
    )


def plan_local_capacity_chunks(
    tables: ResidentLocalTables,
    *,
    row_capacity_ladder,
    image_capacity_ladder,
) -> list[CapacityChunk]:
    """Chunk a local table with the shared greedy planner (T5).

    The planner reads only ``row_offsets`` and ``n_images``, so the local table
    reuses it verbatim: images stay in order, a chunk never exceeds its
    capacities, and a single image whose own rows overflow the largest class
    becomes a one-image chunk rounded up to a multiple of that class. Local
    search's supports are heavy-tailed (the auto-refine median is ~100 rows with
    a ~87k-row maximum), so one-image chunks are expected, not exceptional.
    """

    return plan_capacity_chunks(
        tables,
        row_capacity_ladder=row_capacity_ladder,
        image_capacity_ladder=image_capacity_ladder,
    )


def expand_local_mask_rows(tables: ResidentLocalTables, start: int, stop: int) -> np.ndarray:
    """Reference (numpy) dense mask for rows ``[start, stop)``.

    Exactly ``LocalHypothesisLayout.sample_mask_rows(start, stop)``, including
    the ``None`` case, which the layout means as full support and the bucketed
    engine materializes as an all-true ``local_sample_mask``.
    """

    start, stop = int(start), int(stop)
    n_rows = stop - start
    if tables.row_mask_bits is None:
        return np.ones((n_rows, tables.n_trans), dtype=bool)
    return np.unpackbits(
        tables.row_mask_bits[start:stop],
        axis=1,
        count=int(tables.n_trans),
        bitorder="little",
    ).view(np.bool_)


def expand_local_chunk_mask_jnp(row_mask_bits, *, n_trans: int):
    """Dense candidate mask of one padded chunk, from its packed per-row bytes.

    ``jax.numpy`` twin of :func:`expand_local_mask_rows` for a whole
    :func:`materialize_local_chunk` output at once, so a jitted device program
    rebuilds ``bool[row_capacity, n_trans]`` with no host array. ``row_mask_bits``
    is uint8 ``[row_capacity, n_mask_bytes]`` in the layout's little-endian bit
    order: translation ``t`` is byte ``t // 8``, bit ``t % 8``.

    ``row_mask_bits is None`` returns ``None``, the caller's signal that every
    cell is a candidate; the caller still applies its own row-validity mask, so
    padded rows are excluded there rather than here.
    """

    import jax.numpy as jnp

    if row_mask_bits is None:
        return None
    row_mask_bits = jnp.asarray(row_mask_bits, dtype=jnp.uint8)
    if row_mask_bits.ndim != 2:
        raise ValueError(f"row_mask_bits must be 2-D, got {row_mask_bits.shape}")
    n_trans = int(n_trans)
    trans = jnp.arange(n_trans, dtype=jnp.int32)
    byte_index = trans // jnp.int32(8)
    bit_index = (trans % jnp.int32(8)).astype(jnp.uint8)
    selected = row_mask_bits[:, byte_index]
    return ((selected >> bit_index[None, :]) & jnp.uint8(1)) != 0


def materialize_local_chunk(tables: ResidentLocalTables, chunk: CapacityChunk) -> dict:
    """Gather one chunk's rows into padded, capacity-shaped numpy arrays.

    Because the local table's rows are already contiguous per chunk, every
    per-row field is a slice plus a pad, never a gather.

    Padding contract: ``row_image_local`` pads to ``image_capacity - 1`` (a
    padded image slot, which no valid row addresses), ``row_log_prior`` to
    -1e30, ``row_mask_bits`` to all-zero bytes (no candidate translation),
    ``row_posterior_id`` to ``n_posterior_bins`` (dropped by the scatter),
    and both rotation arrays to the identity so a padded row's projection is
    well defined even though its posterior is exactly zero. ``image_ids`` pads
    to -1.

    ``row_mask_bits`` is ``None`` in the returned dict when the layout has full
    support; the padded rows are then excluded by ``n_valid_rows`` alone, which
    is the same contract the T6 scoring stage already applies through its
    ``row_is_valid`` mask.
    """

    row_capacity = int(chunk.row_capacity)
    image_capacity = int(chunk.image_capacity)
    n_valid_rows = int(chunk.n_valid_rows)
    n_valid_images = int(chunk.n_valid_images)
    if n_valid_rows > row_capacity:
        raise ValueError(f"chunk has {n_valid_rows} valid rows but capacity {row_capacity}")
    if n_valid_images > image_capacity:
        raise ValueError(f"chunk has {n_valid_images} valid images but capacity {image_capacity}")

    rs, re = int(chunk.row_start), int(chunk.row_stop)

    row_image_local = np.full(row_capacity, image_capacity - 1, dtype=np.int32)
    row_log_prior = np.full(row_capacity, _ROW_LOG_PRIOR_PAD, dtype=np.float32)
    row_posterior_id = np.full(row_capacity, tables.n_posterior_bins, dtype=np.int32)
    row_rotation_id = np.zeros(row_capacity, dtype=np.int64)
    rotations = np.broadcast_to(
        np.eye(3, dtype=tables.rotations.dtype), (row_capacity, 3, 3)
    ).copy()
    mstep_rotations = np.broadcast_to(
        np.eye(3, dtype=tables.mstep_rotations.dtype), (row_capacity, 3, 3)
    ).copy()
    image_ids = np.full(image_capacity, -1, dtype=np.int32)

    row_mask_bits = None
    if tables.row_mask_bits is not None:
        row_mask_bits = np.zeros((row_capacity, tables.n_mask_bytes), dtype=np.uint8)

    if n_valid_rows:
        row_image_local[:n_valid_rows] = tables.row_image[rs:re] - int(chunk.image_start)
        row_log_prior[:n_valid_rows] = tables.row_log_prior[rs:re]
        row_posterior_id[:n_valid_rows] = tables.row_posterior_id[rs:re]
        row_rotation_id[:n_valid_rows] = tables.row_rotation_id[rs:re]
        rotations[:n_valid_rows] = tables.rotations[rs:re]
        mstep_rotations[:n_valid_rows] = tables.mstep_rotations[rs:re]
        if row_mask_bits is not None:
            row_mask_bits[:n_valid_rows] = tables.row_mask_bits[rs:re]

    if n_valid_images:
        image_ids[:n_valid_images] = np.arange(
            chunk.image_start, chunk.image_stop, dtype=np.int32
        )

    return {
        "row_image_local": row_image_local,
        "row_log_prior": row_log_prior,
        "row_posterior_id": row_posterior_id,
        "row_rotation_id": row_rotation_id,
        "row_mask_bits": row_mask_bits,
        "rotations": rotations,
        "mstep_rotations": mstep_rotations,
        "n_valid_rows": np.int32(n_valid_rows),
        "n_valid_images": np.int32(n_valid_images),
        "image_ids": image_ids,
    }
