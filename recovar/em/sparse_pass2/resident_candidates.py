"""Host data model for the device-resident K=1 sparse pass-2 prototype.

This module builds a flat, row-major (CSR-by-image) table of pass-2 candidate
rows from :func:`recovar.em.scoring.sparse_bucket_arrays._prepare_per_image_pass2_inputs`
output, and chunks those rows into fixed-capacity "programs" so a future
device-resident kernel can be traced once per (row capacity, image capacity,
pixel count) triple instead of once per bucket shape.

Everything here is host planning: plain numpy (int32/float32/uint32/int8) plus
one jax.numpy reference twin (:func:`expand_mask_jnp`) used only to validate
that the same bit-unpacking arithmetic works on device arrays. No scoring,
projection or M-step arithmetic lives in this module; see
``../em_device_resident_pass2_design_20260918.md`` for the stages this table
feeds.

Index conventions
------------------
* "image" always means the *local* position (0..n_images-1) of an image
  within the ``per_image_inputs`` list passed to
  :func:`build_resident_candidate_tables`, i.e. exactly the index used to
  index every ``per_image_inputs[...][i]`` list. It is not a dataset/original
  particle id.
* "row" means one (image, fine rotation) candidate: one entry of
  ``per_image_inputs["oversampled_rot_indices"][i]``. Rows are laid out
  image-major (CSR), and never reordered relative to
  ``per_image_inputs`` — the row order for image ``i`` is exactly
  ``per_image_inputs["oversampled_rot_indices"][i]``'s order.
* "fine rotation id" (``row_fine_rot``) indexes the shared, parent-major fine
  rotation grid the compact engine already uses (see the design doc); this
  module never looks at rotation matrices, only integer ids.
* "parent" means an image-local coarse rotation: the position of a coarse
  rotation within that image's own ``unique_rot`` array, i.e. exactly
  ``per_image_inputs["parent_map"][i]``'s values. Two different images' parent
  index ``0`` generally refer to different coarse rotations.
* Coarse translation bitsets pack bit ``k`` = "coarse translation ``k`` is a
  valid candidate for this (image, parent)". Bit order is the raw coarse
  translation id (0..n_coarse_trans-1), matching
  ``SparseCandidateMask.coarse_valid``'s column order exactly.  This requires
  ``n_coarse_trans <= 32``, checked at construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recovar.em.scoring.compact_candidates import SparseCandidateMask

__all__ = [
    "CapacityChunk",
    "ResidentCandidateTables",
    "build_resident_candidate_tables",
    "expand_chunk_mask_jnp",
    "expand_mask_jnp",
    "expand_mask_rows",
    "materialize_chunk",
    "plan_capacity_chunks",
]

# Per-image mask modes (int8). Kept as module-level constants so both the
# numpy and jax.numpy expanders and the tests share one vocabulary.
_MASK_MODE_FULL = np.int8(0)
_MASK_MODE_BITSET = np.int8(1)
_MASK_MODE_EMPTY = np.int8(2)

_ROW_FINE_ROT_PAD = np.int32(0)
_ROW_PARENT_LOCAL_PAD = np.int32(0)
_ROW_LOG_PRIOR_PAD = np.float32(-1e30)
_ROW_MASK_BITS_PAD = np.uint32(0)


@dataclass(frozen=True)
class ResidentCandidateTables:
    """Flat, image-CSR table of pass-2 candidate rows plus their masks.

    All arrays are plain numpy; nothing here is device-resident yet (that is
    the job of the chunk programs this table feeds). Units: ``row_log_prior``
    is a natural-log prior density in the same units as
    ``per_image_inputs["log_prior"]`` (nats); every other field is an integer
    id or bitset.
    """

    n_images: int
    n_rows: int
    n_fine_trans: int
    n_coarse_trans: int
    # CSR row ranges: image i owns rows [row_offsets[i], row_offsets[i + 1]).
    row_offsets: np.ndarray  # int32 [n_images + 1]
    row_image: np.ndarray  # int32 [n_rows], non-decreasing, values in [0, n_images)
    row_fine_rot: np.ndarray  # int32 [n_rows]
    row_parent_local: np.ndarray  # int32 [n_rows]
    row_log_prior: np.ndarray  # float32 [n_rows]
    # Per-image mask mode and, for bitset-mode images only, a CSR table of
    # per-parent coarse-translation bitsets.
    mask_mode: np.ndarray  # int8 [n_images]
    parent_offsets: np.ndarray  # int32 [n_images + 1]
    parent_trans_bits: np.ndarray  # uint32 [parent_offsets[-1]]

    def __post_init__(self):
        if self.row_offsets.shape != (self.n_images + 1,):
            raise ValueError("row_offsets must have shape (n_images + 1,)")
        if self.parent_offsets.shape != (self.n_images + 1,):
            raise ValueError("parent_offsets must have shape (n_images + 1,)")
        if int(self.row_offsets[-1]) != int(self.n_rows):
            raise ValueError("row_offsets[-1] must equal n_rows")
        for name in ("row_image", "row_fine_rot", "row_parent_local", "row_log_prior"):
            arr = getattr(self, name)
            if arr.shape != (self.n_rows,):
                raise ValueError(f"{name} must have shape (n_rows,), got {arr.shape}")
        if self.mask_mode.shape != (self.n_images,):
            raise ValueError("mask_mode must have shape (n_images,)")


@dataclass(frozen=True)
class CapacityChunk:
    """One contiguous, capacity-padded image range.

    ``[image_start, image_stop)`` and the matching ``[row_start, row_stop)``
    (exactly ``row_offsets[image_start]`` and ``row_offsets[image_stop]``) are
    the *valid* extents; ``row_capacity``/``image_capacity`` are the padded
    program shape this chunk will run at (from the capacity ladders, or, for
    a single image whose own rows exceed the largest row class, the smallest
    multiple of that largest class that covers it).
    """

    image_start: int
    image_stop: int
    row_start: int
    row_stop: int
    row_capacity: int
    image_capacity: int

    @property
    def n_valid_images(self) -> int:
        return self.image_stop - self.image_start

    @property
    def n_valid_rows(self) -> int:
        return self.row_stop - self.row_start


def _pack_bits_rows(bool_rows: np.ndarray) -> np.ndarray:
    """Pack each row of a boolean matrix into one uint32, bit k = column k."""

    n_bits = bool_rows.shape[1]
    weights = (np.uint32(1) << np.arange(n_bits, dtype=np.uint32))
    return (bool_rows.astype(np.uint32) * weights[None, :]).sum(axis=1, dtype=np.uint32)


def _bits_for_mask(mask: SparseCandidateMask, n_coarse_trans: int) -> np.ndarray:
    """Return one uint32 bitset per image-local parent for a bitset-mode mask.

    Only valid for ``mask.mode in {"coarse", "coarse_exclude"}``. The number
    of parents returned is the number of distinct parent indices actually
    referenced by ``mask.parent_map`` (``max(parent_map) + 1``), which is
    exactly the range ``row_parent_local`` can take for this image -- there is
    no need to size the table to the full coarse rotation grid.
    """

    if mask.mode == "coarse":
        if mask.coarse_valid is None or mask.fine_translation_parent is None:
            raise ValueError("coarse candidate mask spec is missing coarse_valid")
        if mask.coarse_valid.shape[1] != n_coarse_trans:
            raise ValueError(
                "coarse candidate mask coarse_valid width does not match n_coarse_trans: "
                f"{mask.coarse_valid.shape[1]} vs {n_coarse_trans}",
            )
        return _pack_bits_rows(mask.coarse_valid)
    if mask.mode == "coarse_exclude":
        if mask.coarse_excluded is None or mask.parent_map is None:
            raise ValueError("coarse_exclude candidate mask spec is missing excluded/parent arrays")
        n_parents = int(mask.parent_map.max(initial=-1)) + 1
        full_bits = np.uint32((1 << n_coarse_trans) - 1)
        bits = np.full(n_parents, full_bits, dtype=np.uint32)
        excluded = np.unique(np.asarray(mask.coarse_excluded, dtype=np.int64).reshape(-1))
        if excluded.size:
            excluded_rot = excluded // int(n_coarse_trans)
            excluded_trans = excluded % int(n_coarse_trans)
            if int(excluded_rot.max(initial=-1)) >= n_parents:
                raise ValueError("coarse_exclude excluded rotation is outside this image's referenced parents")
            clear_bits = np.uint32(1) << excluded_trans.astype(np.uint32)
            # A parent can appear more than once in excluded_rot (several
            # excluded translations for the same rotation); fold with a
            # scatter-AND so every exclusion is applied.
            for parent, bit in zip(excluded_rot.tolist(), clear_bits.tolist()):
                bits[parent] &= np.uint32(~np.uint32(bit))
        return bits
    raise ValueError(f"_bits_for_mask does not support mode {mask.mode!r}")


def build_resident_candidate_tables(
    per_image_inputs: dict,
    *,
    n_coarse_trans: int,
    n_fine_trans: int,
    fine_translation_parent,
) -> ResidentCandidateTables:
    """Flatten ``_prepare_per_image_pass2_inputs`` output into one CSR table.

    ``per_image_inputs`` is exactly the dict returned by
    :func:`recovar.em.scoring.sparse_bucket_arrays._prepare_per_image_pass2_inputs`;
    only the ``oversampled_rot_indices``, ``parent_map``, ``log_prior`` and
    ``candidate_mask`` entries are read (rotations/mstep rotations/source
    Eulers stay on the shared fine grid and are looked up later by
    ``row_fine_rot``, not copied here).

    ``fine_translation_parent`` is accepted (rather than read off any single
    image) because it is one shared, iteration-global table; every image's
    mask expands against the same array.
    """

    n_coarse_trans = int(n_coarse_trans)
    if n_coarse_trans <= 0 or n_coarse_trans > 32:
        raise ValueError(f"n_coarse_trans must be in [1, 32] to pack into a uint32 bitset, got {n_coarse_trans}")
    n_fine_trans = int(n_fine_trans)
    fine_translation_parent = np.asarray(fine_translation_parent)
    if fine_translation_parent.shape != (n_fine_trans,):
        raise ValueError(
            f"fine_translation_parent must have shape (n_fine_trans,)={(n_fine_trans,)}, "
            f"got {fine_translation_parent.shape}",
        )

    oversampled_rot_indices = per_image_inputs["oversampled_rot_indices"]
    parent_map_list = per_image_inputs["parent_map"]
    log_prior_list = per_image_inputs["log_prior"]
    candidate_mask_list = per_image_inputs["candidate_mask"]
    n_images = len(oversampled_rot_indices)
    if not (len(parent_map_list) == len(log_prior_list) == len(candidate_mask_list) == n_images):
        raise ValueError("per_image_inputs lists disagree on image count")

    row_offsets = np.zeros(n_images + 1, dtype=np.int32)
    parent_offsets = np.zeros(n_images + 1, dtype=np.int32)
    mask_mode = np.empty(n_images, dtype=np.int8)

    row_image_parts: list[np.ndarray] = []
    row_fine_rot_parts: list[np.ndarray] = []
    row_parent_local_parts: list[np.ndarray] = []
    row_log_prior_parts: list[np.ndarray] = []
    parent_trans_bits_parts: list[np.ndarray] = []

    int32_max = np.iinfo(np.int32).max
    for i in range(n_images):
        rot_ids = np.asarray(oversampled_rot_indices[i])
        parent_map = np.asarray(parent_map_list[i], dtype=np.int32)
        log_prior = np.asarray(log_prior_list[i], dtype=np.float32)
        n_rows_i = int(rot_ids.shape[0])
        if parent_map.shape != (n_rows_i,) or log_prior.shape != (n_rows_i,):
            raise ValueError(f"image {i}: oversampled_rot_indices/parent_map/log_prior disagree on row count")
        if rot_ids.size and int(rot_ids.max()) > int32_max:
            raise ValueError(f"image {i}: a fine rotation id overflows int32")

        row_offsets[i + 1] = row_offsets[i] + n_rows_i
        row_image_parts.append(np.full(n_rows_i, i, dtype=np.int32))
        row_fine_rot_parts.append(rot_ids.astype(np.int32, copy=False))
        row_parent_local_parts.append(parent_map)
        row_log_prior_parts.append(log_prior)

        mask = candidate_mask_list[i]
        if not isinstance(mask, SparseCandidateMask):
            raise TypeError(f"image {i}: candidate_mask must be a SparseCandidateMask, got {type(mask)!r}")
        if mask.n_fine_trans != n_fine_trans:
            raise ValueError(f"image {i}: candidate mask n_fine_trans={mask.n_fine_trans} != {n_fine_trans}")
        if mask.mode == "full":
            mask_mode[i] = _MASK_MODE_FULL
            parent_offsets[i + 1] = parent_offsets[i]
        elif mask.mode == "empty":
            mask_mode[i] = _MASK_MODE_EMPTY
            parent_offsets[i + 1] = parent_offsets[i]
        elif mask.mode in ("coarse", "coarse_exclude"):
            mask_mode[i] = _MASK_MODE_BITSET
            bits_i = _bits_for_mask(mask, n_coarse_trans)
            if n_rows_i and int(parent_map.max(initial=-1)) >= bits_i.shape[0]:
                raise ValueError(f"image {i}: row_parent_local references a parent outside its bitset table")
            parent_offsets[i + 1] = parent_offsets[i] + bits_i.shape[0]
            parent_trans_bits_parts.append(bits_i)
        else:
            raise ValueError(f"image {i}: unknown candidate mask mode {mask.mode!r}")

    n_rows = int(row_offsets[-1])
    row_image = np.concatenate(row_image_parts) if row_image_parts else np.zeros(0, dtype=np.int32)
    row_fine_rot = np.concatenate(row_fine_rot_parts) if row_fine_rot_parts else np.zeros(0, dtype=np.int32)
    row_parent_local = (
        np.concatenate(row_parent_local_parts) if row_parent_local_parts else np.zeros(0, dtype=np.int32)
    )
    row_log_prior = np.concatenate(row_log_prior_parts) if row_log_prior_parts else np.zeros(0, dtype=np.float32)
    parent_trans_bits = (
        np.concatenate(parent_trans_bits_parts) if parent_trans_bits_parts else np.zeros(0, dtype=np.uint32)
    )

    return ResidentCandidateTables(
        n_images=n_images,
        n_rows=n_rows,
        n_fine_trans=n_fine_trans,
        n_coarse_trans=n_coarse_trans,
        row_offsets=row_offsets,
        row_image=row_image,
        row_fine_rot=row_fine_rot,
        row_parent_local=row_parent_local,
        row_log_prior=row_log_prior,
        mask_mode=mask_mode,
        parent_offsets=parent_offsets,
        parent_trans_bits=parent_trans_bits,
    )


def expand_mask_rows(tables: ResidentCandidateTables, image: int, fine_translation_parent) -> np.ndarray:
    """Reference (numpy) dense mask for one image; matches ``_candidate_mask_to_dense`` exactly.

    Returns a ``bool[n_rows_i, len(fine_translation_parent)]`` array. This is
    the ground truth :func:`expand_mask_jnp` and the eventual device gather
    kernel must reproduce.
    """

    image = int(image)
    start, stop = int(tables.row_offsets[image]), int(tables.row_offsets[image + 1])
    n_rows_i = stop - start
    fine_translation_parent = np.asarray(fine_translation_parent)
    n_fine_trans = int(fine_translation_parent.shape[0])
    mode = int(tables.mask_mode[image])

    if mode == _MASK_MODE_FULL:
        return np.ones((n_rows_i, n_fine_trans), dtype=bool)
    if mode == _MASK_MODE_EMPTY:
        return np.zeros((n_rows_i, n_fine_trans), dtype=bool)
    if mode != _MASK_MODE_BITSET:
        raise ValueError(f"image {image}: unknown mask_mode {mode}")

    p0, p1 = int(tables.parent_offsets[image]), int(tables.parent_offsets[image + 1])
    bits = tables.parent_trans_bits[p0:p1]
    row_parent = tables.row_parent_local[start:stop]
    row_bits = bits[row_parent].astype(np.uint32)
    shifts = fine_translation_parent.astype(np.uint32)
    return (((row_bits[:, None] >> shifts[None, :]) & np.uint32(1)) != 0)


def expand_mask_jnp(tables: ResidentCandidateTables, image: int, fine_translation_parent):
    """``jax.numpy`` twin of :func:`expand_mask_rows` for the same one image.

    Same contract and same result as :func:`expand_mask_rows` (validated by
    the unit tests bitwise), but written with ``jax.numpy`` primitives so the
    bit-unpacking arithmetic can be copied verbatim into a jitted device
    gather kernel in a later ticket. Runs on whatever platform JAX is
    configured for (CPU in this ticket); nothing here is CUDA-specific.
    """

    import jax.numpy as jnp

    image = int(image)
    start, stop = int(tables.row_offsets[image]), int(tables.row_offsets[image + 1])
    n_rows_i = stop - start
    fine_translation_parent = jnp.asarray(fine_translation_parent, dtype=jnp.uint32)
    n_fine_trans = int(fine_translation_parent.shape[0])
    mode = int(tables.mask_mode[image])

    if mode == _MASK_MODE_FULL:
        return jnp.ones((n_rows_i, n_fine_trans), dtype=bool)
    if mode == _MASK_MODE_EMPTY:
        return jnp.zeros((n_rows_i, n_fine_trans), dtype=bool)
    if mode != _MASK_MODE_BITSET:
        raise ValueError(f"image {image}: unknown mask_mode {mode}")

    p0, p1 = int(tables.parent_offsets[image]), int(tables.parent_offsets[image + 1])
    bits = jnp.asarray(tables.parent_trans_bits[p0:p1], dtype=jnp.uint32)
    row_parent = jnp.asarray(tables.row_parent_local[start:stop], dtype=jnp.int32)
    row_bits = bits[row_parent]
    return (((row_bits[:, None] >> fine_translation_parent[None, :]) & jnp.uint32(1)) != 0)


def expand_chunk_mask_jnp(row_mask_bits, row_mask_mode, fine_translation_parent):
    """Dense candidate mask of one padded chunk, from its per-row mask fields.

    ``jax.numpy`` twin of :func:`expand_mask_rows` evaluated for a whole
    :func:`materialize_chunk` output at once, so a jitted device program can
    rebuild the ``bool[row_capacity, n_fine_trans]`` mask without any host
    array. Inputs are the chunk fields ``row_mask_bits`` (uint32
    ``[row_capacity]``) and ``row_mask_mode`` (int8 ``[row_capacity]``), plus
    the iteration-global ``fine_translation_parent`` (int32
    ``[n_fine_trans]``).

    Mask modes follow the module vocabulary: ``0`` accepts every translation,
    ``1`` tests bit ``fine_translation_parent[t]`` of that row's bitset and
    ``2`` rejects every translation. Padded rows carry mode ``2``, so they are
    all-false whatever their other fields hold.
    """

    import jax.numpy as jnp

    row_mask_bits = jnp.asarray(row_mask_bits, dtype=jnp.uint32)
    row_mask_mode = jnp.asarray(row_mask_mode, dtype=jnp.int8)
    fine_translation_parent = jnp.asarray(fine_translation_parent, dtype=jnp.uint32)
    if row_mask_bits.ndim != 1 or row_mask_mode.shape != row_mask_bits.shape:
        raise ValueError(
            "row_mask_bits and row_mask_mode must be 1-D arrays of equal length, got "
            f"{row_mask_bits.shape} and {row_mask_mode.shape}",
        )
    if fine_translation_parent.ndim != 1:
        raise ValueError(
            f"fine_translation_parent must be 1-D, got {fine_translation_parent.shape}",
        )

    bitset = (
        (row_mask_bits[:, None] >> fine_translation_parent[None, :]) & jnp.uint32(1)
    ) != 0
    is_full = (row_mask_mode == _MASK_MODE_FULL)[:, None]
    is_empty = (row_mask_mode == _MASK_MODE_EMPTY)[:, None]
    return jnp.where(is_full, True, jnp.where(is_empty, False, bitset))


def _smallest_fit(ladder: tuple[int, ...], value: int) -> int | None:
    for capacity in ladder:
        if value <= capacity:
            return capacity
    return None


def plan_capacity_chunks(
    tables: ResidentCandidateTables,
    *,
    row_capacity_ladder=(8192, 32768, 131072, 524288),
    image_capacity_ladder=(32, 128, 512),
) -> list[CapacityChunk]:
    """Greedily group images (in image order) into fixed-capacity chunks.

    Images are never reordered (RELION particle order is preserved for the
    x-half BPref). The chunker grows a chunk one image at a time while both
    its row count and its image count still fit some (row_capacity,
    image_capacity) pair from the ladders, then closes it and starts the next
    chunk at the first image that no longer fits. A single image whose own
    row count exceeds the largest row class becomes a one-image chunk whose
    row capacity is rounded up to the smallest multiple of the largest row
    class that covers it (its image capacity is the smallest image class,
    i.e. the ladder's first entry, since one image always fits there).
    """

    row_capacity_ladder = tuple(int(v) for v in row_capacity_ladder)
    image_capacity_ladder = tuple(int(v) for v in image_capacity_ladder)
    if not row_capacity_ladder or not image_capacity_ladder:
        raise ValueError("capacity ladders must be non-empty")
    if list(row_capacity_ladder) != sorted(row_capacity_ladder):
        raise ValueError("row_capacity_ladder must be increasing")
    if list(image_capacity_ladder) != sorted(image_capacity_ladder):
        raise ValueError("image_capacity_ladder must be increasing")

    row_offsets = tables.row_offsets
    n_images = tables.n_images
    chunks: list[CapacityChunk] = []
    i = 0
    while i < n_images:
        row_start = int(row_offsets[i])
        best_row_cap = None
        best_img_cap = None
        best_stop = None
        j = i
        while j < n_images:
            candidate_images = j - i + 1
            candidate_rows = int(row_offsets[j + 1]) - row_start
            row_cap = _smallest_fit(row_capacity_ladder, candidate_rows)
            img_cap = _smallest_fit(image_capacity_ladder, candidate_images)
            if row_cap is None or img_cap is None:
                break
            best_row_cap, best_img_cap, best_stop = row_cap, img_cap, j + 1
            j += 1

        if best_stop is None:
            # Even the lone first image overflows the largest row class.
            row_stop = int(row_offsets[i + 1])
            candidate_rows = row_stop - row_start
            largest_row = row_capacity_ladder[-1]
            row_cap = -(-candidate_rows // largest_row) * largest_row  # ceil to a multiple
            chunks.append(
                CapacityChunk(
                    image_start=i,
                    image_stop=i + 1,
                    row_start=row_start,
                    row_stop=row_stop,
                    row_capacity=row_cap,
                    image_capacity=image_capacity_ladder[0],
                ),
            )
            i += 1
        else:
            chunks.append(
                CapacityChunk(
                    image_start=i,
                    image_stop=best_stop,
                    row_start=row_start,
                    row_stop=int(row_offsets[best_stop]),
                    row_capacity=best_row_cap,
                    image_capacity=best_img_cap,
                ),
            )
            i = best_stop

    return chunks


def materialize_chunk(tables: ResidentCandidateTables, chunk: CapacityChunk) -> dict:
    """Gather one chunk's rows into padded, capacity-shaped numpy arrays.

    Padding contract (padded rows must never validate any (row, t) cell):
    ``row_image_local`` pads to ``image_capacity - 1`` (a padded image slot,
    see ``image_ids`` below), ``row_fine_rot``/``row_parent_local`` pad to 0,
    ``row_log_prior`` pads to -1e30, ``row_mask_bits`` pads to 0 and
    ``row_mask_mode`` pads to 2 (empty) -- the mode alone is sufficient to
    invalidate every padded cell regardless of bits, so padding the other
    fields to 0 is a safety margin, not a correctness requirement.
    """

    row_capacity = int(chunk.row_capacity)
    image_capacity = int(chunk.image_capacity)
    n_valid_rows = chunk.n_valid_rows
    n_valid_images = chunk.n_valid_images
    if n_valid_rows > row_capacity:
        raise ValueError(f"chunk has {n_valid_rows} valid rows but capacity {row_capacity}")
    if n_valid_images > image_capacity:
        raise ValueError(f"chunk has {n_valid_images} valid images but capacity {image_capacity}")

    rs, re = chunk.row_start, chunk.row_stop

    row_image_local = np.full(row_capacity, image_capacity - 1, dtype=np.int32)
    row_fine_rot = np.full(row_capacity, _ROW_FINE_ROT_PAD, dtype=np.int32)
    row_parent_local = np.full(row_capacity, _ROW_PARENT_LOCAL_PAD, dtype=np.int32)
    row_log_prior = np.full(row_capacity, _ROW_LOG_PRIOR_PAD, dtype=np.float32)
    row_mask_bits = np.full(row_capacity, _ROW_MASK_BITS_PAD, dtype=np.uint32)
    row_mask_mode = np.full(row_capacity, _MASK_MODE_EMPTY, dtype=np.int8)
    image_ids = np.full(image_capacity, -1, dtype=np.int32)

    if n_valid_rows:
        row_image_global = tables.row_image[rs:re]
        row_image_local[:n_valid_rows] = row_image_global - chunk.image_start
        row_fine_rot[:n_valid_rows] = tables.row_fine_rot[rs:re]
        row_parent_local_valid = tables.row_parent_local[rs:re]
        row_parent_local[:n_valid_rows] = row_parent_local_valid
        row_log_prior[:n_valid_rows] = tables.row_log_prior[rs:re]

        row_mode_valid = tables.mask_mode[row_image_global]
        row_mask_mode[:n_valid_rows] = row_mode_valid

        bits_out = np.zeros(n_valid_rows, dtype=np.uint32)
        bitset_rows = row_mode_valid == _MASK_MODE_BITSET
        if np.any(bitset_rows):
            flat_idx = tables.parent_offsets[row_image_global[bitset_rows]] + row_parent_local_valid[bitset_rows]
            bits_out[bitset_rows] = tables.parent_trans_bits[flat_idx]
        row_mask_bits[:n_valid_rows] = bits_out

    if n_valid_images:
        image_ids[:n_valid_images] = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int32)

    return {
        "row_image_local": row_image_local,
        "row_fine_rot": row_fine_rot,
        "row_parent_local": row_parent_local,
        "row_log_prior": row_log_prior,
        "row_mask_bits": row_mask_bits,
        "row_mask_mode": row_mask_mode,
        "n_valid_rows": np.int32(n_valid_rows),
        "n_valid_images": np.int32(n_valid_images),
        "image_ids": image_ids,
    }
