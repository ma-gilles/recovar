"""Device compaction of the coarse significance mask, and its candidate tables.

Ticket T13 (``em_parity_tickets_20260918/T13_device_significance_compaction.md``).

The coarse posterior produces one boolean support mask per image batch, shaped
``[batch, n_classes * n_rot * n_trans]``.  At healpix order 3 with 21
translations that is 774144 cells per image, so a 5000-image half carries
about 3.8 GB of booleans per iteration.  The host path in
``recovar/em/scoring/significance.py`` pulls that whole mask
(``batch_sig_mask_np = np.array(batch_sig_mask, dtype=bool, copy=True)``),
reduces it per image with :func:`numpy.flatnonzero`, and
``_prepare_per_image_pass2_inputs`` then re-expands the resulting ids into
per-image candidate structures that the resident candidate tables flatten
again.

This module replaces the first two steps with one jitted program that compacts
the mask on the device into a CSR pair ``(offsets, ids)``, and the third with a
vectorized builder that produces the same
:class:`~recovar.em.sparse_pass2.resident_candidates.ResidentCandidateTables`
straight from that CSR.  Only the compact ids cross the bus: a few MB per half
instead of a few GB.

Scope and selection
-------------------
Everything here is opt-in behind ``RECOVAR_COARSE_SIGNIFICANCE_DEVICE=1`` and
K=1 only.  The host path stays the oracle: the CSR must equal its
``significant_sample_indices`` per image bitwise, and the tables built here
must equal the tables built through ``_prepare_per_image_pass2_inputs`` field
by field.  Nothing here changes a scientific default; the RELION significance
selection itself (adaptive fraction, ``max_significants``, the tie-inclusive
cutoff) happens upstream in the posterior and is only read here.

The compaction covers supports up to half of the coarse grid per image.  Above
that the host encoder switches to its sparse-complement encoding, whose rows
come from the *full* rotation grid; a compact id list would then be larger than
the mask it replaces, so this module refuses that regime with a named error
instead of guessing (see :func:`host_support_rows`).

Index conventions match :mod:`recovar.em.sparse_pass2.resident_candidates`:
"image" is a local position inside the half's dataset, "parent" is an
image-local coarse rotation, coarse cell ids are ``rot * n_coarse_trans +
trans``, and coarse-translation bitsets pack bit ``k`` for coarse translation
``k`` (so ``n_coarse_trans <= 32``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

import numpy as np

from recovar.em.helpers.env_flags import parse_env_strict_flag
from recovar.em.scoring.sparse_bucket_arrays import relion_parent_execution_key
from recovar.em.sparse_pass2.resident_candidates import (
    ResidentCandidateTables,
    build_resident_candidate_tables,
)

logger = logging.getLogger(__name__)

__all__ = [
    "COARSE_SIGNIFICANCE_DEVICE_ENV",
    "CoarseSignificanceCSR",
    "DeviceCompactedSignificantSamples",
    "build_coarse_significance_csr",
    "build_resident_candidate_tables_from_csr",
    "coarse_significance_device_requested",
    "compact_batch_significance",
    "csr_capacity_for_total",
    "host_support_rows",
    "resident_candidate_tables",
    "resident_significance_csr",
]

COARSE_SIGNIFICANCE_DEVICE_ENV = "RECOVAR_COARSE_SIGNIFICANCE_DEVICE"

# Compacted-id buffers are traced at a power-of-two capacity, so one program
# serves every batch whose support falls in the same octave instead of one
# program per exact support total.
_MIN_CSR_CAPACITY = 1 << 12

_MASK_MODE_FULL = np.int8(0)
_MASK_MODE_BITSET = np.int8(1)
_MASK_MODE_EMPTY = np.int8(2)

_compact_jitted = None


def coarse_significance_device_requested() -> bool:
    """Return whether the device significance compaction is selected."""

    return parse_env_strict_flag(COARSE_SIGNIFICANCE_DEVICE_ENV, default=False)


def csr_capacity_for_total(total: int) -> int:
    """Smallest power-of-two id capacity that holds ``total`` compact ids."""

    total = int(total)
    if total < 0:
        raise ValueError("compact id total must be non-negative")
    capacity = _MIN_CSR_CAPACITY
    while capacity < total:
        capacity <<= 1
    return capacity


# ---------------------------------------------------------------------------
# Device compaction
# ---------------------------------------------------------------------------


def _compact_program():
    """The jitted compaction, built once so JAX reuses its trace cache."""

    global _compact_jitted
    if _compact_jitted is not None:
        return _compact_jitted

    import jax
    import jax.numpy as jnp

    @partial(jax.jit, static_argnames=("capacity", "n_coarse_trans"))
    def _run(mask, image_valid, store_excluded, *, capacity, n_coarse_trans):
        # ``image_valid`` clears the padded rows of a short final batch, so
        # every batch of a given shape reuses one program.  ``store_excluded``
        # flips an image whose support is dense, so the ids compacted for it
        # are its *excluded* cells: the same choice the host encoder makes
        # when more than half the grid is significant.
        valid = jnp.asarray(image_valid, dtype=bool)[:, None]
        support = jnp.asarray(mask, dtype=bool) & valid
        rot_any = jnp.any(
            support.reshape(support.shape[0], -1, n_coarse_trans),
            axis=(0, 2),
        )
        mask = (jnp.asarray(mask, dtype=bool) ^ jnp.asarray(store_excluded, dtype=bool)[:, None]) & valid
        counts = jnp.sum(mask, axis=1, dtype=jnp.int32)
        starts = jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)[:-1]],
        )
        # Rank of each selected cell inside its image, shifted to the batch's
        # flat CSR position; unselected cells scatter out of bounds and drop.
        # The drop sentinel is ``capacity`` rather than -1: a negative scatter
        # index is not reliably dropped, and with a full buffer it lands on
        # the last slot and overwrites a real id (regression test
        # ``test_compaction_fills_an_exactly_full_capacity``).
        rank = jnp.cumsum(mask, axis=1, dtype=jnp.int32) - jnp.int32(1)
        position = jnp.where(mask, starts[:, None] + rank, jnp.int32(capacity))
        sample_ids = jnp.broadcast_to(
            jnp.arange(mask.shape[1], dtype=jnp.int32)[None, :],
            mask.shape,
        )
        ids = jnp.full((capacity,), -1, dtype=jnp.int32)
        ids = ids.at[position.reshape(-1)].set(sample_ids.reshape(-1), mode="drop")
        return ids, counts, rot_any

    _compact_jitted = _run
    return _run


@dataclass(frozen=True)
class CoarseSignificanceCSR:
    """Per-image coarse significance support in CSR form.

    ``ids`` holds every image's significant coarse cell ids
    (``rot * n_coarse_trans + trans``), ascending within each image; image
    ``i`` owns ``ids[offsets[i]:offsets[i + 1]]``.  Both arrays are int32.
    The mask they were compacted from never leaves the device.
    """

    n_images: int
    n_coarse_rot: int
    n_coarse_trans: int
    offsets: np.ndarray  # int32 [n_images + 1]
    ids: np.ndarray  # int32 [offsets[-1]]
    store_excluded: np.ndarray  # bool [n_images]
    n_significant: np.ndarray  # int32 [n_images]

    def __post_init__(self):
        if self.offsets.shape != (self.n_images + 1,):
            raise ValueError("CSR offsets must have shape (n_images + 1,)")
        if self.offsets.dtype != np.int32 or self.ids.dtype != np.int32:
            raise ValueError("CSR offsets and ids must both be int32")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != int(self.ids.shape[0]):
            raise ValueError("CSR offsets must start at 0 and end at the id count")
        if self.store_excluded.shape != (self.n_images,) or self.store_excluded.dtype != np.bool_:
            raise ValueError("store_excluded must be one boolean per image")
        if self.n_significant.shape != (self.n_images,) or self.n_significant.dtype != np.int32:
            raise ValueError("n_significant must be one int32 per image")
        stored = np.where(
            self.store_excluded,
            self.n_samples - self.n_significant.astype(np.int64),
            self.n_significant.astype(np.int64),
        )
        if not np.array_equal(stored, np.diff(self.offsets).astype(np.int64)):
            raise ValueError("CSR row lengths disagree with the stored-set sizes")

    @property
    def n_samples(self) -> int:
        return int(self.n_coarse_rot) * int(self.n_coarse_trans)

    def counts(self) -> np.ndarray:
        """Stored ids per image: included, or excluded for a dense support."""

        return np.diff(self.offsets).astype(np.int32, copy=False)

    def image_ids(self, image: int) -> np.ndarray:
        image = int(image)
        return self.ids[int(self.offsets[image]) : int(self.offsets[image + 1])]


class DeviceCompactedSignificantSamples(list):
    """The usual per-image support list, carrying its device-compacted CSR.

    Every existing host consumer keeps treating this as the list of per-image
    encodings it already expects.  The resident pass-2 driver additionally
    reads ``csr`` and builds its candidate tables from the compact ids, so the
    per-image arrays are never re-expanded into per-image structures.
    """

    def __init__(self, rows, *, csr: CoarseSignificanceCSR):
        super().__init__(rows)
        if len(self) != int(csr.n_images):
            raise ValueError(
                f"support list has {len(self)} images but the CSR covers {csr.n_images}",
            )
        self.csr = csr


def compact_batch_significance(
    batch_sig_mask,
    *,
    actual_batch_size: int,
    n_coarse_rot: int,
    n_coarse_trans: int,
    batch_n_sig,
):
    """Compact one batch's coarse significance mask on the device.

    ``batch_sig_mask`` is the posterior's ``[batch, n_rot * n_trans]`` support
    for a single class and ``batch_n_sig`` its per-image count, which the
    posterior already returns, so the compaction capacity is chosen without an
    extra device synchronization.

    Returns ``(n_significant, store_excluded, ids, rot_any)``.
    ``n_significant`` is the per-image support size, ``store_excluded`` marks
    the images whose ids are their *excluded* cells (the host encoder's
    sparse-complement choice, taken for exactly the same images), ``ids`` is
    the concatenated image-major ascending id array, and ``rot_any`` marks the
    coarse rotations carrying any significant sample.  The full mask never
    reaches the host, and the ids stored per image are always the smaller of
    the included and excluded sets.
    """

    import jax.numpy as jnp

    n_coarse_rot = int(n_coarse_rot)
    n_coarse_trans = int(n_coarse_trans)
    actual_batch_size = int(actual_batch_size)
    mask = jnp.asarray(batch_sig_mask)
    if mask.ndim != 2:
        raise ValueError(f"coarse significance mask must be rank 2, got {mask.shape}")
    batch_size, n_samples = int(mask.shape[0]), int(mask.shape[1])
    if n_samples != n_coarse_rot * n_coarse_trans:
        raise ValueError(
            "coarse significance mask width does not match one class's coarse grid: "
            f"{n_samples} vs {n_coarse_rot * n_coarse_trans}",
        )
    if not 0 <= actual_batch_size <= batch_size:
        raise ValueError("actual batch size is outside the mask's image axis")

    n_significant = np.asarray(batch_n_sig, dtype=np.int32)[:actual_batch_size].copy()
    # The host encoder keeps the included ids unless more than half the grid is
    # significant, in which case it keeps the excluded ones. Reproduce that
    # choice exactly, per image, so the two paths encode identically.
    store_excluded = (n_significant.astype(np.int64) * 2) > n_samples
    stored = np.where(
        store_excluded, n_samples - n_significant.astype(np.int64), n_significant,
    ).astype(np.int32)
    total = int(stored.sum(dtype=np.int64))
    capacity = csr_capacity_for_total(total)
    image_valid = jnp.asarray(np.arange(batch_size, dtype=np.int32) < actual_batch_size)
    padded_polarity = np.zeros(batch_size, dtype=bool)
    padded_polarity[:actual_batch_size] = store_excluded

    ids, device_counts, rot_any = _compact_program()(
        mask,
        image_valid,
        jnp.asarray(padded_polarity),
        capacity=capacity,
        n_coarse_trans=n_coarse_trans,
    )
    device_counts = np.asarray(device_counts, dtype=np.int32)[:actual_batch_size]
    if not np.array_equal(device_counts, stored):
        raise RuntimeError(
            "the device significance compaction disagrees with the posterior's "
            "significant-sample counts",
        )
    ids = np.asarray(ids, dtype=np.int32)[:total].copy()
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= n_samples):
        raise RuntimeError("a compacted significance id is outside the coarse pose grid")
    return n_significant, store_excluded, ids, np.asarray(rot_any, dtype=bool)


def build_coarse_significance_csr(
    *,
    n_images: int,
    n_coarse_rot: int,
    n_coarse_trans: int,
    n_significant_per_batch,
    store_excluded_per_batch,
    ids_per_batch,
) -> CoarseSignificanceCSR:
    """Assemble one half's CSR from the per-batch compaction results."""

    n_images = int(n_images)
    n_significant_per_batch = list(n_significant_per_batch)
    store_excluded_per_batch = list(store_excluded_per_batch)
    ids_per_batch = list(ids_per_batch)
    n_significant = (
        np.concatenate([np.asarray(c, dtype=np.int32) for c in n_significant_per_batch])
        if n_significant_per_batch
        else np.zeros(0, dtype=np.int32)
    )
    store_excluded = (
        np.concatenate([np.asarray(c, dtype=bool) for c in store_excluded_per_batch])
        if store_excluded_per_batch
        else np.zeros(0, dtype=bool)
    )
    ids = (
        np.concatenate([np.asarray(i, dtype=np.int32) for i in ids_per_batch])
        if ids_per_batch
        else np.zeros(0, dtype=np.int32)
    )
    if n_significant.shape != (n_images,) or store_excluded.shape != (n_images,):
        raise ValueError(
            f"compacted counts cover {n_significant.shape[0]} images, expected {n_images}",
        )
    n_samples = int(n_coarse_rot) * int(n_coarse_trans)
    counts = np.where(
        store_excluded, n_samples - n_significant.astype(np.int64), n_significant,
    ).astype(np.int64)
    running = np.cumsum(counts)
    if running.size and int(running[-1]) > np.iinfo(np.int32).max:
        raise OverflowError("the half's compacted significance support overflows int32")
    offsets = np.zeros(n_images + 1, dtype=np.int32)
    if running.size:
        offsets[1:] = running.astype(np.int32)
    if int(offsets[-1]) != int(ids.shape[0]):
        raise ValueError("compacted counts and ids disagree on the total support")
    return CoarseSignificanceCSR(
        n_images=n_images,
        n_coarse_rot=int(n_coarse_rot),
        n_coarse_trans=int(n_coarse_trans),
        offsets=offsets,
        ids=ids.astype(np.int32, copy=False),
        store_excluded=store_excluded,
        n_significant=n_significant,
    )


def host_support_rows(csr: CoarseSignificanceCSR) -> list:
    """Per-image host encodings equal to the host path's, taken from the CSR.

    Reproduces
    :func:`recovar.em.scoring.significant_samples.compact_significant_sample_indices_from_mask`
    exactly, for every regime: ``None`` when the whole grid is significant, a
    :class:`ComplementSignificantSampleIndices` when more than half of it is,
    and an explicit ascending int32 id array otherwise. The compaction already
    stored whichever of the two id sets that rule selects.
    """

    from recovar.em.scoring.significant_samples import ComplementSignificantSampleIndices

    n_samples = csr.n_samples
    rows: list = []
    for image in range(csr.n_images):
        if int(csr.n_significant[image]) == n_samples:
            rows.append(None)
        elif bool(csr.store_excluded[image]):
            rows.append(
                ComplementSignificantSampleIndices(
                    excluded_indices=csr.image_ids(image),
                    total_size=n_samples,
                ),
            )
        else:
            rows.append(csr.image_ids(image))
    return rows


def resident_significance_csr(
    significant_sample_indices,
    *,
    n_images: int,
    n_coarse_rot: int,
    n_coarse_trans: int,
):
    """Return the device-compacted CSR behind a support list, or ``None``.

    ``None`` means the candidate tables must be built through the host path:
    either the flag is off, or this support did not come from the coarse
    posterior at all (the local-search routes pass their own parent support).
    The caller logs which route it took, so a measured result always knows
    which one produced it.
    """

    if not coarse_significance_device_requested():
        return None
    csr = getattr(significant_sample_indices, "csr", None)
    if csr is None:
        logger.info(
            "Resident pass-2 candidate tables: %s=1 but this support carries no "
            "device-compacted CSR; building them through the host path",
            COARSE_SIGNIFICANCE_DEVICE_ENV,
        )
        return None
    if (
        int(csr.n_images) != int(n_images)
        or int(csr.n_coarse_rot) != int(n_coarse_rot)
        or int(csr.n_coarse_trans) != int(n_coarse_trans)
    ):
        raise ValueError(
            "the device-compacted significance CSR does not match this pass: "
            f"images {csr.n_images} vs {n_images}, rotations {csr.n_coarse_rot} vs "
            f"{n_coarse_rot}, translations {csr.n_coarse_trans} vs {n_coarse_trans}",
        )
    return csr


# ---------------------------------------------------------------------------
# Candidate tables straight from the CSR
# ---------------------------------------------------------------------------


def _children_by_parent(
    *,
    n_coarse_rot: int,
    nside_level: int,
    oversampling_order: int,
    random_perturbation: float,
    fine_rotation_parent_override,
):
    """Fine-rotation children of every coarse parent, in host-path row order.

    Returns ``(child_offsets, child_ids)``: parent ``p`` owns
    ``child_ids[child_offsets[p]:child_offsets[p + 1]]``.  The order inside a
    parent is the order ``_prepare_per_image_pass2_inputs`` produces for it --
    ascending fine id when the caller supplies a fine rotation/parent override
    (its rows come from :func:`numpy.flatnonzero`), and the sampling
    generator's own child order otherwise.  Both are functions of the parent
    alone, which is why one table serves every image.
    """

    n_coarse_rot = int(n_coarse_rot)
    if fine_rotation_parent_override is not None:
        parent = np.asarray(fine_rotation_parent_override, dtype=np.int64).reshape(-1)
        if parent.size and (int(parent.min()) < 0 or int(parent.max()) >= n_coarse_rot):
            raise ValueError("fine rotation parents must be in [0, n_coarse_rot)")
        order = np.argsort(parent, kind="stable")
        counts = np.bincount(parent, minlength=n_coarse_rot).astype(np.int64)
        child_offsets = np.zeros(n_coarse_rot + 1, dtype=np.int64)
        child_offsets[1:] = np.cumsum(counts)
        return child_offsets, order.astype(np.int64, copy=False)

    from recovar.em.sampling import get_oversampled_rotation_grid_from_samples

    _rotations, parent_map, child_ids = get_oversampled_rotation_grid_from_samples(
        np.arange(n_coarse_rot, dtype=np.int64),
        int(nside_level),
        oversampling_order=int(oversampling_order),
        random_perturbation=float(random_perturbation),
        return_rotation_indices=True,
    )
    parent_map = np.asarray(parent_map, dtype=np.int64)
    child_ids = np.asarray(child_ids, dtype=np.int64)
    if parent_map.size and not bool(np.all(np.diff(parent_map) >= 0)):
        raise RuntimeError("the oversampled rotation grid is not parent-major")
    counts = np.bincount(parent_map, minlength=n_coarse_rot).astype(np.int64)
    child_offsets = np.zeros(n_coarse_rot + 1, dtype=np.int64)
    child_offsets[1:] = np.cumsum(counts)
    return child_offsets, child_ids


def _ragged_gather(
    child_offsets: np.ndarray,
    child_ids: np.ndarray,
    parents: np.ndarray,
    child_counts: np.ndarray,
) -> np.ndarray:
    """Concatenate each parent's child ids, in ``parents`` order."""

    total = int(child_counts.sum(dtype=np.int64))
    if total == 0:
        return np.zeros(0, dtype=child_ids.dtype)
    group_starts = np.zeros(child_counts.size, dtype=np.int64)
    if child_counts.size:
        group_starts[1:] = np.cumsum(child_counts)[:-1]
    within = np.arange(total, dtype=np.int64) - np.repeat(group_starts, child_counts)
    return child_ids[np.repeat(child_offsets[parents], child_counts) + within]


def build_resident_candidate_tables_from_csr(
    csr: CoarseSignificanceCSR,
    *,
    nside_level: int,
    oversampling_order: int,
    n_fine_trans: int,
    fine_translation_parent,
    rotation_log_prior,
    random_perturbation: float,
    fine_rotation_parent_override=None,
    relion_parent_execution_order: bool = False,
    dtype=np.float32,
) -> ResidentCandidateTables:
    """Build the resident candidate tables directly from the compact CSR.

    Produces exactly what
    :func:`recovar.em.sparse_pass2.resident_candidates.build_resident_candidate_tables`
    produces from ``_prepare_per_image_pass2_inputs`` for the same support, and
    is validated against it field by field.  Nothing here materializes a dense
    mask, a per-image rotation-matrix block, or a per-image Python loop over
    the fine grid: every table is assembled with whole-array numpy over the
    concatenated CSR.
    """

    n_images = int(csr.n_images)
    n_coarse_rot = int(csr.n_coarse_rot)
    n_coarse_trans = int(csr.n_coarse_trans)
    if n_coarse_trans <= 0 or n_coarse_trans > 32:
        raise ValueError(
            "n_coarse_trans must be in [1, 32] to pack into a uint32 bitset, got "
            f"{n_coarse_trans}",
        )
    n_fine_trans = int(n_fine_trans)
    fine_translation_parent = np.asarray(fine_translation_parent)
    if fine_translation_parent.shape != (n_fine_trans,):
        raise ValueError(
            "fine_translation_parent must have shape (n_fine_trans,), got "
            f"{fine_translation_parent.shape}",
        )

    counts = csr.counts().astype(np.int64)
    n_samples = csr.n_samples
    n_significant = csr.n_significant.astype(np.int64)
    store_excluded = np.asarray(csr.store_excluded, dtype=bool)
    # Three regimes, matching ``_prepare_per_image_pass2_inputs`` exactly:
    # a full support takes the full rotation grid with an all-true mask, a
    # dense (complement-encoded) support takes the full rotation grid with the
    # excluded pairs cleared, and everything else takes only the rotations its
    # own support references.
    full_images = np.flatnonzero(n_significant == n_samples)
    complement_images = np.flatnonzero(store_excluded & (n_significant != n_samples))
    empty_images = np.flatnonzero(n_significant == 0)
    sparse_images = np.flatnonzero(
        ~store_excluded & (n_significant != n_samples) & (n_significant != 0),
    )

    child_offsets, child_ids = _children_by_parent(
        n_coarse_rot=n_coarse_rot,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        random_perturbation=random_perturbation,
        fine_rotation_parent_override=fine_rotation_parent_override,
    )

    ids = csr.ids.astype(np.int64, copy=False)
    all_cell_image = np.repeat(np.arange(n_images, dtype=np.int64), counts)
    all_cell_rot = ids // n_coarse_trans
    all_cell_trans = ids % n_coarse_trans
    sparse_cell = np.isin(all_cell_image, sparse_images)
    cell_image = all_cell_image[sparse_cell]
    cell_rot = all_cell_rot[sparse_cell]
    cell_trans = all_cell_trans[sparse_cell]

    # Sparse images: their parents are the runs of equal coarse rotation in
    # their ids, image-major and ascending, the same set and order as the host
    # path's ``np.unique(coarse_rot)``.
    if cell_rot.size:
        parent_start = np.ones(cell_rot.size, dtype=bool)
        parent_start[1:] = (cell_rot[1:] != cell_rot[:-1]) | (
            cell_image[1:] != cell_image[:-1]
        )
        cell_parent = np.cumsum(parent_start) - 1
        first_cell = np.flatnonzero(parent_start)
    else:
        cell_parent = np.zeros(0, dtype=np.int64)
        first_cell = np.zeros(0, dtype=np.int64)
    sparse_parent_rot = cell_rot[first_cell]
    sparse_parent_image = cell_image[first_cell]
    # One uint32 per parent: exactly ``SparseCandidateMask.coarse_valid``
    # packed bit by bit, bit k = coarse translation k.
    sparse_parent_bits = np.zeros(sparse_parent_rot.size, dtype=np.uint32)
    if cell_rot.size:
        np.bitwise_or.at(
            sparse_parent_bits,
            cell_parent,
            np.uint32(1) << cell_trans.astype(np.uint32),
        )

    # Full-support and complement-encoded images take the whole coarse
    # rotation grid as their parents, ascending, exactly as the host path's
    # full-rotation-support branch does.
    grid_images = np.sort(np.concatenate([full_images, complement_images]))
    grid_rot = np.tile(np.arange(n_coarse_rot, dtype=np.int64), grid_images.size)
    grid_image = np.repeat(grid_images, n_coarse_rot)
    all_translations = np.uint32((1 << n_coarse_trans) - 1)
    grid_bits = np.full(grid_rot.size, all_translations, dtype=np.uint32)
    if complement_images.size:
        complement_cell = np.isin(all_cell_image, complement_images)
        if bool(complement_cell.any()):
            slot = np.searchsorted(grid_images, all_cell_image[complement_cell])
            np.bitwise_and.at(
                grid_bits,
                slot * n_coarse_rot + all_cell_rot[complement_cell],
                ~(np.uint32(1) << all_cell_trans[complement_cell].astype(np.uint32)),
            )

    # An image with no significant sample still carries one parent, coarse
    # rotation 0, and an all-false candidate mask, matching
    # ``_prepare_per_image_pass2_inputs``' ``unique_rot = [0]`` branch.
    mask_mode = np.full(n_images, _MASK_MODE_BITSET, dtype=np.int8)
    mask_mode[full_images] = _MASK_MODE_FULL
    mask_mode[empty_images] = _MASK_MODE_EMPTY

    parent_image = np.concatenate([sparse_parent_image, grid_image, empty_images])
    parent_rot = np.concatenate(
        [sparse_parent_rot, grid_rot, np.zeros(empty_images.size, dtype=np.int64)],
    )
    parent_bits = np.concatenate(
        [sparse_parent_bits, grid_bits, np.zeros(empty_images.size, dtype=np.uint32)],
    )
    # A stable sort by image keeps each image's own parent order; an image
    # belongs to exactly one regime, so the blocks never interleave inside one.
    order = np.argsort(parent_image, kind="stable")
    parent_image = parent_image[order]
    parent_rot = parent_rot[order]
    parent_bits = parent_bits[order]

    parents_per_image = np.bincount(parent_image, minlength=n_images).astype(np.int64)
    parent_row_offsets = np.zeros(n_images + 1, dtype=np.int64)
    parent_row_offsets[1:] = np.cumsum(parents_per_image)
    parent_local = np.arange(parent_rot.size, dtype=np.int64) - parent_row_offsets[
        parent_image
    ]

    # The bitset table holds only bitset-mode images, in image-major order.
    is_bitset = mask_mode == _MASK_MODE_BITSET
    support_parent_bits = parent_bits[is_bitset[parent_image]]
    bits_per_image = np.where(is_bitset, parents_per_image, 0)
    parent_offsets = np.zeros(n_images + 1, dtype=np.int32)
    parent_offsets[1:] = np.cumsum(bits_per_image).astype(np.int32)

    # Row order: image-major, then RELION's parent execution key when the fine
    # posterior requires it, then the host path's within-parent child order.
    if relion_parent_execution_order:
        key = relion_parent_execution_key(
            parent_rot, n_coarse_rot=n_coarse_rot, nside_level=nside_level
        )
        parent_order = np.lexsort((key, parent_image))
    else:
        parent_order = np.arange(parent_rot.size, dtype=np.int64)

    ordered_rot = parent_rot[parent_order]
    ordered_image = parent_image[parent_order]
    ordered_local = parent_local[parent_order]
    child_counts = (child_offsets[ordered_rot + 1] - child_offsets[ordered_rot]).astype(
        np.int64,
    )

    rows_per_image = np.zeros(n_images, dtype=np.int64)
    np.add.at(rows_per_image, ordered_image, child_counts)
    row_offsets = np.zeros(n_images + 1, dtype=np.int32)
    row_offsets[1:] = np.cumsum(rows_per_image).astype(np.int32)
    n_rows = int(row_offsets[-1])

    row_fine_rot = _ragged_gather(child_offsets, child_ids, ordered_rot, child_counts)
    row_image = np.repeat(ordered_image, child_counts)
    row_parent_local = np.repeat(ordered_local, child_counts)
    row_parent_rot = np.repeat(ordered_rot, child_counts)

    if not relion_parent_execution_order and fine_rotation_parent_override is not None:
        # Without RELION's execution order the host path leaves the rows in
        # ``np.flatnonzero`` order, i.e. ascending fine id inside each image.
        order = np.lexsort((row_fine_rot, row_image))
        row_fine_rot = row_fine_rot[order]
        row_image = row_image[order]
        row_parent_local = row_parent_local[order]
        row_parent_rot = row_parent_rot[order]

    if row_fine_rot.size and int(row_fine_rot.max()) > np.iinfo(np.int32).max:
        raise ValueError("a fine rotation id overflows int32")

    if rotation_log_prior is None:
        row_log_prior = np.zeros(n_rows, dtype=dtype)
    else:
        prior = np.asarray(rotation_log_prior, dtype=dtype)
        if prior.shape != (n_coarse_rot,):
            raise ValueError(
                f"rotation_log_prior must have shape ({n_coarse_rot},), got {prior.shape}",
            )
        row_log_prior = prior[row_parent_rot].astype(dtype, copy=False)

    return ResidentCandidateTables(
        n_images=n_images,
        n_rows=n_rows,
        n_fine_trans=n_fine_trans,
        n_coarse_trans=n_coarse_trans,
        row_offsets=row_offsets,
        row_image=row_image.astype(np.int32, copy=False),
        row_fine_rot=row_fine_rot.astype(np.int32, copy=False),
        row_parent_local=row_parent_local.astype(np.int32, copy=False),
        row_log_prior=np.asarray(row_log_prior, dtype=np.float32),
        mask_mode=mask_mode,
        parent_offsets=parent_offsets,
        parent_trans_bits=support_parent_bits,
    )


def resident_candidate_tables(
    significance_csr,
    per_image_inputs,
    *,
    n_coarse_trans,
    n_fine_trans,
    fine_translation_parent,
    nside_level,
    oversampling_order,
    rotation_log_prior,
    random_perturbation,
    fine_rotation_parent_override,
    relion_parent_execution_order,
    dtype,
):
    """Candidate tables from the device-compacted CSR, or from the host path.

    ``significance_csr`` is present only when the coarse pass compacted its
    support on the device (ticket T13, ``RECOVAR_COARSE_SIGNIFICANCE_DEVICE``);
    both routes return the same ``ResidentCandidateTables``.
    """

    if significance_csr is not None:
        return build_resident_candidate_tables_from_csr(
            significance_csr,
            nside_level=nside_level,
            oversampling_order=oversampling_order,
            n_fine_trans=n_fine_trans,
            fine_translation_parent=fine_translation_parent,
            rotation_log_prior=rotation_log_prior,
            random_perturbation=random_perturbation,
            fine_rotation_parent_override=fine_rotation_parent_override,
            relion_parent_execution_order=relion_parent_execution_order,
            dtype=dtype,
        )
    return build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=n_coarse_trans,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
    )
