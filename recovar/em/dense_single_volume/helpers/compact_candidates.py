"""Host candidate masks and source-ordered compact pair/fine-job layouts.

Numerical execution stays in the local and sparse engines. This module owns
mask materialization, index order, valid prefixes and static padding only.
"""

from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size


class SparseCandidateMask:
    """Compact host representation of one image's pass-2 candidate mask."""

    __slots__ = (
        "mode",
        "n_rows",
        "n_fine_trans",
        "parent_map",
        "coarse_valid",
        "coarse_excluded",
        "fine_translation_parent",
        "count",
    )

    def __init__(
        self,
        *,
        mode: str,
        n_rows: int,
        n_fine_trans: int,
        parent_map=None,
        coarse_valid=None,
        coarse_excluded=None,
        fine_translation_parent=None,
        count: int | None = None,
    ):
        self.mode = str(mode)
        self.n_rows = int(n_rows)
        self.n_fine_trans = int(n_fine_trans)
        self.parent_map = None if parent_map is None else np.asarray(parent_map, dtype=np.int32)
        self.coarse_valid = None if coarse_valid is None else np.asarray(coarse_valid, dtype=bool)
        self.coarse_excluded = None if coarse_excluded is None else np.asarray(coarse_excluded, dtype=np.int32)
        self.fine_translation_parent = (
            None if fine_translation_parent is None else np.asarray(fine_translation_parent, dtype=np.int32)
        )
        self.count = int(_dense_candidate_mask_from_spec(self).sum()) if count is None else int(count)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_fine_trans)

    def __array__(self, dtype=None, copy=None):
        dense = _dense_candidate_mask_from_spec(self)
        if dtype is not None:
            return dense.astype(dtype, copy=False if copy is None else bool(copy))
        if copy:
            return dense.copy()
        return dense


def _dense_candidate_mask_from_spec(mask: SparseCandidateMask) -> np.ndarray:
    if mask.mode == "full":
        return np.ones(mask.shape, dtype=bool)
    if mask.mode == "empty":
        return np.zeros(mask.shape, dtype=bool)
    if mask.mode == "coarse":
        if mask.coarse_valid is None or mask.parent_map is None or mask.fine_translation_parent is None:
            raise ValueError("coarse candidate mask spec is missing parent/coarse arrays")
        return mask.coarse_valid[:, mask.fine_translation_parent][mask.parent_map]
    if mask.mode == "coarse_exclude":
        if mask.coarse_excluded is None or mask.parent_map is None or mask.fine_translation_parent is None:
            raise ValueError("coarse_exclude candidate mask spec is missing excluded/parent arrays")
        dense = np.ones(mask.shape, dtype=bool)
        excluded = np.asarray(mask.coarse_excluded, dtype=np.int64).reshape(-1)
        if excluded.size:
            n_coarse_trans = int(mask.fine_translation_parent.max(initial=-1) + 1)
            if n_coarse_trans <= 0:
                raise ValueError("coarse_exclude candidate mask has empty translation parent map")
            excluded_rot = excluded // n_coarse_trans
            excluded_trans = excluded % n_coarse_trans
            for coarse_rot, coarse_trans in zip(excluded_rot.tolist(), excluded_trans.tolist(), strict=False):
                rows = np.flatnonzero(mask.parent_map == int(coarse_rot))
                cols = np.flatnonzero(mask.fine_translation_parent == int(coarse_trans))
                if rows.size and cols.size:
                    dense[np.ix_(rows, cols)] = False
        return dense
    raise ValueError(f"Unknown sparse candidate mask mode {mask.mode!r}")


def _candidate_mask_to_dense(candidate_mask) -> np.ndarray:
    if isinstance(candidate_mask, SparseCandidateMask):
        return _dense_candidate_mask_from_spec(candidate_mask)
    return np.asarray(candidate_mask, dtype=bool)


def _candidate_mask_count(candidate_mask) -> int:
    if isinstance(candidate_mask, SparseCandidateMask):
        return int(candidate_mask.count)
    return int(np.asarray(candidate_mask, dtype=bool).sum())


def _candidate_mask_is_full(candidate_mask) -> bool:
    if isinstance(candidate_mask, SparseCandidateMask):
        total = int(candidate_mask.n_rows) * int(candidate_mask.n_fine_trans)
        return total > 0 and int(candidate_mask.count) >= total
    dense = np.asarray(candidate_mask, dtype=bool)
    return dense.size > 0 and bool(np.all(dense))


def compact_candidate_indices_in_source_order(candidate_mask):
    """Return compact ``(rotation, translation)`` ids in dense source order.

    Compact pass 2 and the exact-local scorer must agree on one ordering:
    rotations are the major axis and translations are the minor axis, exactly
    as if the dense ``(R, T)`` mask had been flattened in C order.  Keep this
    encoder shared so selected-pair CUDA paths cannot silently invent a
    different posterior order.
    """

    if isinstance(candidate_mask, SparseCandidateMask):
        if candidate_mask.mode == "empty":
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        if candidate_mask.mode == "full":
            rows = np.repeat(np.arange(candidate_mask.n_rows, dtype=np.int64), candidate_mask.n_fine_trans)
            trans = np.tile(np.arange(candidate_mask.n_fine_trans, dtype=np.int64), candidate_mask.n_rows)
            return rows, trans
        if candidate_mask.mode == "coarse_exclude":
            dense = _dense_candidate_mask_from_spec(candidate_mask)
            return np.nonzero(dense)
    return np.nonzero(_candidate_mask_to_dense(candidate_mask))


def _batched_compact_candidate_indices(candidate_masks):
    """Source-ordered ``(rotation, translation)`` ids for a whole bucket at once.

    ``compact_candidate_indices_in_source_order`` materializes a dense ``(R, T)``
    mask and calls ``np.nonzero`` once per image. Measured 2026-09-12 on the
    100k/256 K=4 fixture, that per-image loop made bucket construction 22 % of the
    pass-2 loop. Within one bucket every image shares ``R``, ``T`` and normally the
    fine-to-coarse translation parent, so the bucket is one padded
    ``(B, cR, cT)`` table, one gather to ``(B, R, T)`` and one ``np.nonzero``.
    Coarse specs are enumerated per coarse row and repeated along the fine
    rows, which keeps the per-image rotation-major, translation-minor source
    order without materializing the padded ``(B, R, T)`` table.

    Returns ``None`` when the bucket is not uniform enough (dense NumPy masks,
    ``coarse_exclude`` specs, or differing translation parents); the caller then
    uses the per-image path unchanged.
    """

    if not candidate_masks:
        return None
    if not all(isinstance(m, SparseCandidateMask) for m in candidate_masks):
        return None
    modes = {m.mode for m in candidate_masks}
    if not modes <= {"coarse", "coarse_exclude", "full", "empty"}:
        return None
    first = candidate_masks[0]
    n_trans = int(first.n_fine_trans)
    if any(int(m.n_fine_trans) != n_trans for m in candidate_masks):
        return None
    # Images keep their own rotation-row counts (the bucket pads rows separately);
    # requiring equal n_rows made every real-size bucket fall back to the
    # per-image dense path (build stage unchanged at 141 s, jobs 13808173/13808781).
    # A coarse_exclude spec is the all-ones coarse table with the excluded
    # (coarse rotation, coarse translation) cells cleared; expressing it as a
    # coarse table lets it share the per-coarse-row expansion below instead of
    # the per-image dense nonzero (80 s per 100k/256 iteration, job 13808173:
    # the real-size masks are complement specs, so the batched path never ran).
    coarse_tables = {}
    coarse = [m for m in candidate_masks if m.mode in ("coarse", "coarse_exclude")]
    if coarse:
        ftp = np.asarray(coarse[0].fine_translation_parent)
        for i, m in enumerate(candidate_masks):
            if m.mode not in ("coarse", "coarse_exclude"):
                continue
            if m.parent_map is None or m.fine_translation_parent is None:
                return None
            if not np.array_equal(np.asarray(m.fine_translation_parent), ftp):
                return None
            if m.mode == "coarse":
                if m.coarse_valid is None:
                    return None
                coarse_tables[i] = np.asarray(m.coarse_valid, dtype=bool)
            else:
                if m.coarse_excluded is None:
                    return None
                n_coarse_trans = int(ftp.max(initial=-1) + 1)
                parents = np.asarray(m.parent_map, dtype=np.int64)
                n_coarse_rot = int(parents.max(initial=-1) + 1)
                if n_coarse_trans <= 0 or n_coarse_rot <= 0:
                    return None
                table = np.ones((n_coarse_rot, n_coarse_trans), dtype=bool)
                excluded = np.asarray(m.coarse_excluded, dtype=np.int64).reshape(-1)
                if excluded.size:
                    rot, trans = excluded // n_coarse_trans, excluded % n_coarse_trans
                    keep = (rot >= 0) & (rot < n_coarse_rot) & (trans >= 0) & (trans < n_coarse_trans)
                    table[rot[keep], trans[keep]] = False
                coarse_tables[i] = table
    batch = len(candidate_masks)
    if n_trans == 0:
        empty = np.zeros(0, dtype=np.int64)
        return tuple((empty, empty) for _ in candidate_masks)

    # A coarse spec is ``coarse_valid[:, ftp][parent_map]``: every fine row
    # ``r`` repeats the fine-translation list of its coarse row ``parent_map[r]``.
    # Enumerate that list once per coarse row on the small ``(Bc, cR, T)`` table
    # and expand it along the fine rows with repeats: O(Bc * cR * T + pairs)
    # instead of the O(B * R * T) dense ``np.nonzero``. At 100k/256 the bucket
    # rows R are padded to thousands while the median image keeps ~9k of ~1M
    # candidate slots, so the dense scan was the host "build" stage (317 s of
    # the 1 106 s iteration 2, job 13804810). C order of ``np.nonzero`` on
    # ``(Bc, cR, T)`` is coarse-row major, translation minor, so the expanded
    # order is exactly the per-image rotation-major, translation-minor order.
    out: list = [None] * batch
    full_pairs: dict = {}
    for i, m in enumerate(candidate_masks):
        if m.mode == "full":
            rows_i = int(m.n_rows)
            if rows_i not in full_pairs:
                full_pairs[rows_i] = (
                    np.repeat(np.arange(rows_i, dtype=np.int64), n_trans),
                    np.tile(np.arange(n_trans, dtype=np.int64), rows_i),
                )
            out[i] = full_pairs[rows_i]
        elif m.mode == "empty":
            out[i] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    coarse_rows = sorted(coarse_tables)
    if coarse_rows:
        # Pad every image's coarse-validity table to the bucket's largest coarse
        # rotation count. Padded rows are False and are never addressed, because
        # each image's parent_map only points into its own table.
        tables = [coarse_tables[i] for i in coarse_rows]
        c_rot = max(t.shape[0] for t in tables)
        c_trans = tables[0].shape[1]
        if any(t.shape[1] != c_trans for t in tables):
            return None
        stacked = np.zeros((len(coarse_rows), c_rot, c_trans), dtype=bool)
        for k, t in enumerate(tables):
            stacked[k, : t.shape[0]] = t
        fine_trans = stacked[:, :, ftp]  # (Bc, cR, T)
        b_idx, c_idx, t_idx = np.nonzero(fine_trans)
        coarse_counts = np.bincount(b_idx * c_rot + c_idx, minlength=len(coarse_rows) * c_rot)
        coarse_starts = np.concatenate(([0], np.cumsum(coarse_counts)[:-1]))
        t_idx = t_idx.astype(np.int64, copy=False)
        for k, i in enumerate(coarse_rows):
            parents = np.asarray(candidate_masks[i].parent_map, dtype=np.int64)
            n_rows_i = int(candidate_masks[i].n_rows)
            if parents.shape[0] != n_rows_i:
                return None
            flat_parents = k * c_rot + parents
            row_counts = coarse_counts[flat_parents]  # (R_i,)
            total = int(row_counts.sum())
            if total == 0:
                out[i] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
                continue
            rotation_row = np.repeat(np.arange(n_rows_i, dtype=np.int64), row_counts)
            row_starts = np.concatenate(([0], np.cumsum(row_counts)[:-1]))
            within = np.arange(total, dtype=np.int64) - np.repeat(row_starts, row_counts)
            translation_id = t_idx[np.repeat(coarse_starts[flat_parents], row_counts) + within]
            out[i] = (rotation_row, translation_id)
    return tuple(out)


def build_compact_pair_index_arrays(
    candidate_masks,
    *,
    pair_bucket_size: int | None = None,
    pair_block_size_for_quantization: int = 5000,
):
    """Pack candidate masks with the mature compact-pass-2 pair ABI.

    Valid pairs occupy a source-ordered prefix of each image row. Padding uses
    ``-1`` indices plus a false ``pair_mask``.  When no explicit capacity is
    supplied, use the same compile-friendly bucket quantization as compact
    pass 2 instead of creating exact-count shape families.
    """

    candidate_masks = tuple(candidate_masks)
    compact_indices = _batched_compact_candidate_indices(candidate_masks)
    if compact_indices is None:
        compact_indices = tuple(
            compact_candidate_indices_in_source_order(candidate_mask) for candidate_mask in candidate_masks
        )
    pair_counts = np.asarray(
        [rotation_rows.shape[0] for rotation_rows, _ in compact_indices],
        dtype=np.int32,
    )
    required_capacity = int(pair_counts.max(initial=0))
    if pair_bucket_size is None:
        pair_bucket_size = _exact_bucket_rotation_size(
            required_capacity,
            pair_block_size_for_quantization,
        )
    pair_bucket_size = int(pair_bucket_size)
    if pair_bucket_size <= 0:
        raise ValueError("compact pair bucket size must be positive")
    if required_capacity > pair_bucket_size:
        raise ValueError(
            "compact pair bucket is smaller than the source-ordered candidate "
            f"prefix: required={required_capacity}, capacity={pair_bucket_size}"
        )

    batch_size = len(candidate_masks)
    local_rotation_row = np.full(
        (batch_size, pair_bucket_size),
        -1,
        dtype=np.int32,
    )
    translation_idx = np.full_like(local_rotation_row, -1)
    pair_mask = np.zeros((batch_size, pair_bucket_size), dtype=bool)
    for image_row, (rotation_rows, translation_ids) in enumerate(compact_indices):
        count = int(rotation_rows.shape[0])
        if count == 0:
            continue
        local_rotation_row[image_row, :count] = rotation_rows.astype(
            np.int32,
            copy=False,
        )
        translation_idx[image_row, :count] = translation_ids.astype(
            np.int32,
            copy=False,
        )
        pair_mask[image_row, :count] = True
    return {
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": pair_counts,
        "local_rotation_row": local_rotation_row,
        "translation_idx": translation_idx,
        "pair_mask": pair_mask,
    }


def _quantize_up(value: int, quantum: int) -> int:
    value = int(value)
    quantum = int(quantum)
    return max(quantum, ((value + quantum - 1) // quantum) * quantum)


_DEVICE_INDEX_ROW_QUANTUM = 64
_DEVICE_INDEX_COARSE_ROW_QUANTUM = 64


def compact_pair_index_arrays_device(
    candidate_masks,
    *,
    pair_bucket_size: int,
    n_alloc: int | None = None,
    rows_capacity: int | None = None,
):
    """Build the compact pair index arrays on the device from the coarse tables.

    Returns ``{"pair_counts", "local_rotation_row", "translation_idx", "pair_mask",
    "pair_bucket_size"}`` with the three ``(n_alloc, pair_bucket_size)`` arrays as
    device arrays and ``pair_counts`` as a host ``int32`` vector, or ``None`` when
    the bucket cannot take this path (a dense NumPy mask, an unknown mode, or
    images that disagree on the translation count or translation parent).

    Values, order, dtypes and padding are those of ``build_compact_pair_index_arrays``
    followed by ``_rows_at_capacity``: valid pairs form a source-ordered prefix
    (rotation-major, translation-minor) of each real image row, the remainder of a
    real row is ``-1``/``False``, and rows beyond the real images are ``0``/``False``
    with count ``0``.

    Why: at 100k/256 the host allocated and filled these arrays for every chunk
    (``np.full`` + prefix copies, then a host-to-device copy of the same bytes),
    about 100 GB of memset per iteration across ~5 000 chunks; the "build" stage
    was 215 s of the 788 s iteration 2 (job 13812775). The device needs only the
    per-image coarse tables, the per-row parent map and the translation parent.

    Shapes are quantized so the jitted kernel compiles once per
    ``(n_alloc, rows, coarse rows, T, cT, pair_bucket_size)`` family: rows come from
    ``rows_capacity`` (the class bucket size) or a 64-row quantum, coarse rows from a
    64-row quantum.
    """

    candidate_masks = tuple(candidate_masks)
    if not candidate_masks:
        return None
    if not all(isinstance(m, SparseCandidateMask) for m in candidate_masks):
        return None
    if not {m.mode for m in candidate_masks} <= {"coarse", "coarse_exclude", "full", "empty"}:
        return None
    first = candidate_masks[0]
    n_trans = int(first.n_fine_trans)
    if n_trans <= 0 or any(int(m.n_fine_trans) != n_trans for m in candidate_masks):
        return None
    batch = len(candidate_masks)
    n_alloc = batch if n_alloc is None else max(batch, int(n_alloc))
    pair_bucket_size = int(pair_bucket_size)
    counts = np.zeros(n_alloc, dtype=np.int32)
    for i, m in enumerate(candidate_masks):
        counts[i] = int(m.count)
    if int(counts.max(initial=0)) > pair_bucket_size:
        raise ValueError(
            "compact pair bucket is smaller than the source-ordered candidate "
            f"prefix: required={int(counts.max())}, capacity={pair_bucket_size}"
        )

    coarse = [m for m in candidate_masks if m.mode in ("coarse", "coarse_exclude")]
    ftp = None
    c_trans = 1
    if coarse:
        ftp = np.asarray(coarse[0].fine_translation_parent, dtype=np.int32)
        if ftp is None or ftp.shape != (n_trans,):
            return None
        for m in coarse:
            if m.parent_map is None or m.fine_translation_parent is None:
                return None
            if not np.array_equal(np.asarray(m.fine_translation_parent), ftp):
                return None
        c_trans_exclude = int(ftp.max(initial=-1) + 1)
        c_trans_valid = [int(m.coarse_valid.shape[1]) for m in coarse if m.mode == "coarse" and m.coarse_valid is not None]
        if any(m.mode == "coarse" and m.coarse_valid is None for m in coarse):
            return None
        if any(m.mode == "coarse_exclude" and m.coarse_excluded is None for m in coarse):
            return None
        if c_trans_valid and any(c != c_trans_valid[0] for c in c_trans_valid):
            return None
        c_trans = c_trans_valid[0] if c_trans_valid else c_trans_exclude
        if c_trans <= 0 or c_trans_exclude > c_trans:
            return None
    else:
        ftp = np.zeros(n_trans, dtype=np.int32)

    max_rows = max(int(m.n_rows) for m in candidate_masks)
    if rows_capacity is not None and int(rows_capacity) >= max_rows:
        rows = int(rows_capacity)
    else:
        rows = _quantize_up(max_rows, _DEVICE_INDEX_ROW_QUANTUM)
    c_rot_needed = 1
    for m in candidate_masks:
        if m.mode == "coarse":
            c_rot_needed = max(c_rot_needed, int(m.coarse_valid.shape[0]))
        elif m.mode == "coarse_exclude":
            c_rot_needed = max(c_rot_needed, int(np.asarray(m.parent_map).max(initial=-1) + 1))
    c_rot = _quantize_up(c_rot_needed, _DEVICE_INDEX_COARSE_ROW_QUANTUM)

    tables = np.zeros((n_alloc, c_rot, c_trans), dtype=bool)
    parents = np.full((n_alloc, rows), -1, dtype=np.int32)
    for i, m in enumerate(candidate_masks):
        n_rows_i = int(m.n_rows)
        if m.mode == "full":
            tables[i, 0, :] = True
            parents[i, :n_rows_i] = 0
        elif m.mode == "empty":
            continue
        elif m.mode == "coarse":
            table = np.asarray(m.coarse_valid, dtype=bool)
            tables[i, : table.shape[0], :] = table
            parents[i, :n_rows_i] = np.asarray(m.parent_map, dtype=np.int32)
        else:  # coarse_exclude
            parent_map = np.asarray(m.parent_map, dtype=np.int32)
            n_coarse_rot = int(parent_map.max(initial=-1) + 1)
            n_coarse_trans = int(ftp.max(initial=-1) + 1)
            if parent_map.shape[0] != n_rows_i or n_coarse_rot <= 0 or n_coarse_trans <= 0:
                return None
            table = np.ones((n_coarse_rot, n_coarse_trans), dtype=bool)
            excluded = np.asarray(m.coarse_excluded, dtype=np.int64).reshape(-1)
            if excluded.size:
                rot, trans = excluded // n_coarse_trans, excluded % n_coarse_trans
                keep = (rot >= 0) & (rot < n_coarse_rot) & (trans >= 0) & (trans < n_coarse_trans)
                table[rot[keep], trans[keep]] = False
            tables[i, :n_coarse_rot, :n_coarse_trans] = table
            parents[i, :n_rows_i] = parent_map
    real_rows = np.zeros(n_alloc, dtype=bool)
    real_rows[:batch] = True

    import jax.numpy as jnp

    local_rotation_row, translation_idx, pair_mask = _compact_pair_index_arrays_jit(
        jnp.asarray(tables),
        jnp.asarray(parents),
        jnp.asarray(ftp),
        jnp.asarray(counts),
        jnp.asarray(real_rows),
        pair_bucket_size=pair_bucket_size,
    )
    return {
        "pair_bucket_size": pair_bucket_size,
        "pair_counts": counts,
        "local_rotation_row": local_rotation_row,
        "translation_idx": translation_idx,
        "pair_mask": pair_mask,
    }


def _compact_pair_index_arrays_impl(tables, parents, ftp, counts, real_rows, *, pair_bucket_size):
    """Source-ordered pair prefix per image from ``(B, cR, cT)`` coarse tables.

    Pair ``p`` of image ``b`` is the ``p``-th ``True`` of the dense ``(R, T)`` mask in
    C order: its row is the last row whose exclusive start is ``<= p``, its
    translation the ``(p - start)``-th valid translation of that row's coarse row.
    """
    import jax
    import jax.numpy as jnp

    n_alloc, rows = parents.shape
    n_trans = int(ftp.shape[0])
    fine = jnp.take(tables, ftp, axis=2)  # (B, cR, T)
    coarse_counts = jnp.sum(fine, axis=-1, dtype=jnp.int32)  # (B, cR)
    # k-th valid translation of each coarse row, without a sort: the rank of a valid
    # entry is its exclusive prefix count, so scatter t into that slot (invalid entries
    # go to the dropped slot T). A sort here cost ~35 CUB launches per class-chunk
    # (nsys, job 13828749); the scatter is one kernel and yields the same integers.
    ranks = jnp.cumsum(fine, axis=-1, dtype=jnp.int32) - fine.astype(jnp.int32)  # exclusive
    slots = jnp.where(fine, ranks, jnp.int32(n_trans))
    n_alloc_b, c_rot, _ = fine.shape
    b_idx = jnp.arange(n_alloc_b, dtype=jnp.int32)[:, None, None]
    c_idx = jnp.arange(c_rot, dtype=jnp.int32)[None, :, None]
    t_ids = jnp.broadcast_to(jnp.arange(n_trans, dtype=jnp.int32)[None, None, :], fine.shape)
    valid_translations = (
        jnp.full((n_alloc_b, c_rot, n_trans + 1), jnp.int32(n_trans), dtype=jnp.int32)
        .at[jnp.broadcast_to(b_idx, fine.shape), jnp.broadcast_to(c_idx, fine.shape), slots]
        .set(t_ids, mode="drop")[:, :, :n_trans]
    )  # (B, cR, T), valid ids ascending then n_trans padding
    has_parent = parents >= 0
    safe_parent = jnp.where(has_parent, parents, 0)
    row_counts = jnp.where(
        has_parent, jnp.take_along_axis(coarse_counts, safe_parent, axis=1), jnp.int32(0)
    )  # (B, R)
    row_starts = jnp.cumsum(row_counts, axis=1, dtype=jnp.int32) - row_counts  # exclusive
    slots = jnp.arange(pair_bucket_size, dtype=jnp.int32)  # (P,)
    row = jax.vmap(lambda starts: jnp.searchsorted(starts, slots, side="right").astype(jnp.int32) - 1)(
        row_starts
    )  # (B, P): last row whose start <= p
    row = jnp.clip(row, 0, rows - 1)
    within = slots[None, :] - jnp.take_along_axis(row_starts, row, axis=1)
    coarse_row = jnp.take_along_axis(safe_parent, row, axis=1)  # (B, P)
    translation = valid_translations[
        jnp.arange(n_alloc, dtype=jnp.int32)[:, None],
        coarse_row,
        jnp.clip(within, 0, n_trans - 1),
    ]
    valid = (slots[None, :] < counts.astype(jnp.int32)[:, None]) & real_rows[:, None]
    padding_value = jnp.where(real_rows, jnp.int32(-1), jnp.int32(0))[:, None]
    local_rotation_row = jnp.where(valid, row, padding_value).astype(jnp.int32)
    translation_idx = jnp.where(valid, translation, padding_value).astype(jnp.int32)
    return local_rotation_row, translation_idx, valid


def _compact_pair_index_arrays_jit(tables, parents, ftp, counts, real_rows, *, pair_bucket_size):
    import jax

    fn = _compact_pair_index_arrays_jit.__dict__.get("_compiled")
    if fn is None:
        fn = jax.jit(_compact_pair_index_arrays_impl, static_argnames=("pair_bucket_size",))
        _compact_pair_index_arrays_jit.__dict__["_compiled"] = fn
    return fn(tables, parents, ftp, counts, real_rows, pair_bucket_size=int(pair_bucket_size))


def build_compact_fine_job_plan(
    candidate_masks,
    reference_row_lookup,
    *,
    job_bucket_size: int | None = None,
    job_block_size_for_quantization: int = 5000,
):
    """Pack all selected fine hypotheses into one global source-order plan.

    The mature pair encoder quantizes a separate capacity for every image and
    therefore executes ``B * max(P_i)`` slots.  This companion ABI preserves
    the identical image-major, rotation-major, translation-major order while
    quantizing only the total selected count.  Rows encode ``(image,
    projected-reference row, dense rotation row, translation)``; an all-``-1``
    tail is inert static padding for JAX compilation reuse.
    """

    candidate_masks = tuple(np.asarray(mask, dtype=bool) for mask in candidate_masks)
    if not candidate_masks:
        raise ValueError("compact fine jobs require at least one image mask")
    first_shape = candidate_masks[0].shape
    if len(first_shape) != 2 or first_shape[0] <= 0 or first_shape[1] <= 0:
        raise ValueError("compact fine-job masks must have nonempty (rotation, translation) shape")
    if any(mask.shape != first_shape for mask in candidate_masks):
        raise ValueError("compact fine-job masks must share one dense shape")

    batch_size = len(candidate_masks)
    rotation_count, _ = first_shape
    reference_row_lookup = np.asarray(reference_row_lookup)
    if reference_row_lookup.dtype != np.int32 or reference_row_lookup.shape != (batch_size, rotation_count):
        raise ValueError(
            "compact fine-job reference lookup must be int32 with shape "
            f"{(batch_size, rotation_count)}, got "
            f"{reference_row_lookup.shape} {reference_row_lookup.dtype}"
        )

    pair_arrays = build_compact_pair_index_arrays(candidate_masks)
    return build_compact_fine_job_plan_from_pair_arrays(
        pair_arrays,
        reference_row_lookup,
        job_bucket_size=job_bucket_size,
        job_block_size_for_quantization=job_block_size_for_quantization,
    )


def build_compact_fine_job_plan_from_pair_arrays(
    pair_arrays,
    reference_row_lookup,
    *,
    job_bucket_size: int | None = None,
    job_block_size_for_quantization: int = 5000,
):
    """Collapse the mature per-image pair ABI into one global job prefix."""

    local_rotation_row = np.asarray(pair_arrays["local_rotation_row"])
    translation_idx = np.asarray(pair_arrays["translation_idx"])
    pair_mask = np.asarray(pair_arrays["pair_mask"])
    job_counts = np.asarray(pair_arrays["pair_counts"])
    if (
        local_rotation_row.dtype != np.int32
        or translation_idx.dtype != np.int32
        or pair_mask.dtype != np.bool_
        or job_counts.dtype != np.int32
        or local_rotation_row.ndim != 2
        or translation_idx.shape != local_rotation_row.shape
        or pair_mask.shape != local_rotation_row.shape
        or job_counts.shape != (local_rotation_row.shape[0],)
    ):
        raise ValueError("compact pair arrays are not aligned with the mature ABI")
    expected_mask = np.arange(local_rotation_row.shape[1])[None, :] < job_counts[:, None]
    if not np.array_equal(pair_mask, expected_mask):
        raise ValueError("compact pair validity must be a source-ordered prefix")

    batch_size = int(local_rotation_row.shape[0])
    reference_row_lookup = np.asarray(reference_row_lookup)
    if reference_row_lookup.dtype != np.int32 or reference_row_lookup.ndim != 2:
        raise ValueError("compact fine-job reference lookup must be a 2-D int32 array")
    if reference_row_lookup.shape[0] != batch_size:
        raise ValueError("compact fine-job reference lookup batch axis is misaligned")
    valid_job_count = int(np.sum(job_counts, dtype=np.int64))
    if job_bucket_size is None:
        job_bucket_size = _exact_bucket_rotation_size(
            valid_job_count,
            job_block_size_for_quantization,
        )
    job_bucket_size = int(job_bucket_size)
    if job_bucket_size <= 0:
        raise ValueError("compact fine-job bucket size must be positive")
    if valid_job_count > job_bucket_size:
        raise ValueError(
            "compact fine-job bucket is smaller than the source-ordered prefix: "
            f"required={valid_job_count}, capacity={job_bucket_size}"
        )

    job_plan = np.full((job_bucket_size, 4), -1, dtype=np.int32)
    cursor = 0
    for image_row, count_value in enumerate(job_counts):
        count = int(count_value)
        if count == 0:
            continue
        rotation_rows = local_rotation_row[image_row, :count]
        translation_ids = translation_idx[image_row, :count]
        if np.any(rotation_rows < 0) or np.any(rotation_rows >= reference_row_lookup.shape[1]):
            raise ValueError("a selected fine job has an invalid dense rotation row")
        if np.any(translation_ids < 0):
            raise ValueError("a selected fine job has an invalid translation id")
        reference_rows = reference_row_lookup[image_row, rotation_rows]
        if np.any(reference_rows < 0):
            raise ValueError("a selected fine job has no projected-reference row")
        next_cursor = cursor + count
        job_plan[cursor:next_cursor, 0] = image_row
        job_plan[cursor:next_cursor, 1] = reference_rows
        job_plan[cursor:next_cursor, 2] = rotation_rows
        job_plan[cursor:next_cursor, 3] = translation_ids
        cursor = next_cursor

    return {
        "job_bucket_size": job_bucket_size,
        "job_counts": job_counts,
        "valid_job_count": valid_job_count,
        "job_plan": job_plan,
    }


# Preserve the stored class identity; the sparse owner imports this same class.
SparseCandidateMask.__module__ = "recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed"
