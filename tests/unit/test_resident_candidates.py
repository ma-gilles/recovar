"""Tests for the device-resident pass-2 host data model (T5, CPU only).

Ticket: em_parity_tickets_20260918/T5_resident_candidate_tables.md.
Design: em_device_resident_pass2_design_20260918.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.scoring.compact_candidates import _candidate_mask_to_dense
from recovar.em.scoring.significant_samples import ComplementSignificantSampleIndices
from recovar.em.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
from recovar.em.sparse_pass2.resident_candidates import (
    ResidentCandidateTables,
    build_resident_candidate_tables,
    expand_mask_jnp,
    expand_mask_rows,
    materialize_chunk,
    plan_capacity_chunks,
)

pytestmark = pytest.mark.unit


def _z_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


# --- Fixture: one image of each SparseCandidateMask mode --------------------
#
# n_coarse_rot=5, children_per_parent=2 (fine_rotations_override controls the
# shared fine rotation grid exactly, so this fixture never touches
# get_oversampled_rotation_grid_from_samples).
N_COARSE_ROT = 5
CHILDREN = 2
N_COARSE_TRANS = 3
N_FINE_TRANS = 6
# Two fine translations per coarse translation.
FINE_TRANS_PARENT = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)


def _fine_rotation_grid():
    rotations = np.stack([_z_rotation(0.1 * k) for k in range(N_COARSE_ROT * CHILDREN)]).astype(np.float32)
    parent = np.repeat(np.arange(N_COARSE_ROT, dtype=np.int32), CHILDREN)
    return rotations, parent


def _build_fixture_per_image_inputs():
    """Build one image of each of the four SparseCandidateMask modes."""

    fine_rotations, fine_parent = _fine_rotation_grid()

    # Image 0: "coarse" mode -- explicit significant (rot, trans) samples.
    image0_samples = np.sort(
        np.asarray(
            [0 * N_COARSE_TRANS + 1, 2 * N_COARSE_TRANS + 0, 2 * N_COARSE_TRANS + 2, 4 * N_COARSE_TRANS + 1],
            dtype=np.int32,
        )
    )
    # Image 1: "full" mode -- full support.
    image1_samples = None
    # Image 2: "empty" mode -- an empty significant sample list.
    image2_samples = np.asarray([], dtype=np.int32)
    # Image 3: "coarse_exclude" mode -- a sparse complement.
    total_size = N_COARSE_ROT * N_COARSE_TRANS
    image3_samples = ComplementSignificantSampleIndices(
        excluded_indices=np.asarray([1, 7, 12], dtype=np.int32),
        total_size=total_size,
    )

    significant_sample_indices = [image0_samples, image1_samples, image2_samples, image3_samples]
    rotation_log_prior = np.linspace(-1.0, 1.0, N_COARSE_ROT, dtype=np.float32)

    per_image_inputs = _prepare_per_image_pass2_inputs(
        significant_sample_indices,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=0,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        dtype=np.float32,
    )
    expected_modes = {0: "coarse", 1: "full", 2: "empty", 3: "coarse_exclude"}
    for image, mode in expected_modes.items():
        assert per_image_inputs["candidate_mask"][image].mode == mode, (
            f"fixture image {image} built mode {per_image_inputs['candidate_mask'][image].mode!r}, "
            f"expected {mode!r} -- update the fixture, not the assertion"
        )
    return per_image_inputs


@pytest.fixture(scope="module")
def fixture_inputs():
    return _build_fixture_per_image_inputs()


@pytest.fixture(scope="module")
def fixture_tables(fixture_inputs):
    return build_resident_candidate_tables(
        fixture_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
    )


# --- Mask/id/parent/prior equivalence to _prepare_per_image_pass2_inputs ----


def test_row_ids_parents_and_priors_match_per_image_inputs(fixture_inputs, fixture_tables):
    tables = fixture_tables
    n_images = len(fixture_inputs["oversampled_rot_indices"])
    assert tables.n_images == n_images
    for image in range(n_images):
        start, stop = int(tables.row_offsets[image]), int(tables.row_offsets[image + 1])
        expected_ids = np.asarray(fixture_inputs["oversampled_rot_indices"][image])
        expected_parents = np.asarray(fixture_inputs["parent_map"][image])
        expected_prior = np.asarray(fixture_inputs["log_prior"][image], dtype=np.float32)

        assert stop - start == expected_ids.shape[0]
        np.testing.assert_array_equal(tables.row_fine_rot[start:stop], expected_ids.astype(np.int32))
        np.testing.assert_array_equal(tables.row_parent_local[start:stop], expected_parents)
        np.testing.assert_allclose(tables.row_log_prior[start:stop], expected_prior)
        assert np.all(tables.row_image[start:stop] == image)


def test_expand_mask_rows_matches_candidate_mask_to_dense_for_every_mode(fixture_inputs, fixture_tables):
    tables = fixture_tables
    n_images = len(fixture_inputs["oversampled_rot_indices"])
    for image in range(n_images):
        expected_dense = _candidate_mask_to_dense(fixture_inputs["candidate_mask"][image])
        got_dense = expand_mask_rows(tables, image, FINE_TRANS_PARENT)
        np.testing.assert_array_equal(got_dense, expected_dense)


def test_expand_mask_jnp_matches_expand_mask_rows(fixture_tables):
    tables = fixture_tables
    for image in range(tables.n_images):
        rows = expand_mask_rows(tables, image, FINE_TRANS_PARENT)
        jnp_rows = np.asarray(expand_mask_jnp(tables, image, FINE_TRANS_PARENT))
        np.testing.assert_array_equal(jnp_rows, rows)


def test_n_coarse_trans_over_32_is_rejected(fixture_inputs):
    with pytest.raises(ValueError):
        build_resident_candidate_tables(
            fixture_inputs,
            n_coarse_trans=33,
            n_fine_trans=N_FINE_TRANS,
            fine_translation_parent=FINE_TRANS_PARENT,
        )


# --- Chunk planning / materialization invariants ----------------------------


def _synthetic_tables(row_counts, *, n_fine_trans=1, n_coarse_trans=1):
    """A minimal all-``full``-mode table for chunk-planning-only tests."""

    row_counts = np.asarray(row_counts, dtype=np.int64)
    n_images = row_counts.shape[0]
    row_offsets = np.zeros(n_images + 1, dtype=np.int32)
    row_offsets[1:] = np.cumsum(row_counts)
    n_rows = int(row_offsets[-1])
    return ResidentCandidateTables(
        n_images=n_images,
        n_rows=n_rows,
        n_fine_trans=n_fine_trans,
        n_coarse_trans=n_coarse_trans,
        row_offsets=row_offsets,
        row_image=np.repeat(np.arange(n_images, dtype=np.int32), row_counts),
        row_fine_rot=np.arange(n_rows, dtype=np.int32),
        row_parent_local=np.zeros(n_rows, dtype=np.int32),
        row_log_prior=np.zeros(n_rows, dtype=np.float32),
        mask_mode=np.zeros(n_images, dtype=np.int8),  # all "full"
        parent_offsets=np.zeros(n_images + 1, dtype=np.int32),
        parent_trans_bits=np.zeros(0, dtype=np.uint32),
    )


def _row_level_valid(row_mask_bits, row_mask_mode, fine_translation_parent):
    """The same mode-dispatch bit-unpack arithmetic as expand_mask_{rows,jnp},
    but taking already-materialized per-row (not per-image) arrays -- this is
    what a chunk program would evaluate per cell."""

    shifts = np.asarray(fine_translation_parent, dtype=np.uint32)
    bit_valid = ((row_mask_bits[:, None].astype(np.uint32) >> shifts[None, :]) & np.uint32(1)) != 0
    full = (row_mask_mode == 0)[:, None]
    bitset = (row_mask_mode == 1)[:, None]
    return full | (bitset & bit_valid)


ROW_CAPACITY_LADDER = (4, 8, 16)
IMAGE_CAPACITY_LADDER = (2, 4, 8)


def test_chunks_are_contiguous_ordered_and_cover_every_row_exactly_once():
    row_counts = [3, 5, 2, 7, 1, 4, 6, 2, 1, 1, 9]
    tables = _synthetic_tables(row_counts)
    chunks = plan_capacity_chunks(
        tables, row_capacity_ladder=ROW_CAPACITY_LADDER, image_capacity_ladder=IMAGE_CAPACITY_LADDER
    )

    assert chunks[0].image_start == 0
    assert chunks[-1].image_stop == tables.n_images
    covered_rows = 0
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.image_stop == cur.image_start, "chunks must be contiguous in image order"
        assert prev.row_stop == cur.row_start
    for chunk in chunks:
        assert chunk.image_start < chunk.image_stop
        assert chunk.row_start == int(tables.row_offsets[chunk.image_start])
        assert chunk.row_stop == int(tables.row_offsets[chunk.image_stop])
        assert chunk.n_valid_rows <= chunk.row_capacity, "no chunk may exceed its row capacity"
        assert chunk.n_valid_images <= chunk.image_capacity, "no chunk may exceed its image capacity"
        covered_rows += chunk.n_valid_rows
    assert covered_rows == tables.n_rows


def test_materialize_chunk_padded_rows_never_validate_any_cell():
    row_counts = [3, 5, 2, 7, 1, 4, 6, 2, 1, 1, 9]
    tables = _synthetic_tables(row_counts)
    chunks = plan_capacity_chunks(
        tables, row_capacity_ladder=ROW_CAPACITY_LADDER, image_capacity_ladder=IMAGE_CAPACITY_LADDER
    )
    probe_fine_translation_parent = np.arange(4, dtype=np.int32)

    for chunk in chunks:
        materialized = materialize_chunk(tables, chunk)
        n_valid_rows = int(materialized["n_valid_rows"])
        n_valid_images = int(materialized["n_valid_images"])
        assert n_valid_rows == chunk.n_valid_rows
        assert n_valid_images == chunk.n_valid_images

        assert materialized["row_image_local"].shape == (chunk.row_capacity,)
        assert materialized["image_ids"].shape == (chunk.image_capacity,)

        # Every valid row must address a real (in-range) image slot.
        valid_image_local = materialized["row_image_local"][:n_valid_rows]
        assert np.all(valid_image_local >= 0)
        assert np.all(valid_image_local < n_valid_images)

        # Padded image ids are -1; valid ones are exactly this chunk's images.
        np.testing.assert_array_equal(
            materialized["image_ids"][:n_valid_images],
            np.arange(chunk.image_start, chunk.image_stop, dtype=np.int32),
        )
        assert np.all(materialized["image_ids"][n_valid_images:] == -1)

        valid = _row_level_valid(
            materialized["row_mask_bits"], materialized["row_mask_mode"], probe_fine_translation_parent
        )
        assert not np.any(valid[n_valid_rows:]), "a padded row validated a cell"


def test_materialize_chunk_matches_expand_mask_rows_for_a_bitset_image(fixture_inputs, fixture_tables):
    """A chunk covering the "coarse" (bitset) image must reproduce the same
    per-cell validity as expand_mask_rows once its own row_mask_bits/mode are
    unpacked."""

    tables = fixture_tables
    image = 0  # "coarse" mode in the fixture.
    chunk = plan_capacity_chunks(
        tables,
        row_capacity_ladder=(64, 256),
        image_capacity_ladder=(1, 8),
    )
    # Isolate exactly the chunk containing image 0's rows (image 0 is first,
    # so with these generous capacities the whole table may pack into one
    # chunk; materialize_chunk still lets us slice out image 0's local rows).
    containing = next(c for c in chunk if c.image_start <= image < c.image_stop)
    materialized = materialize_chunk(tables, containing)

    local_image = image - containing.image_start
    row_mask = materialized["row_image_local"][: containing.n_valid_rows] == local_image
    chunk_valid = _row_level_valid(
        materialized["row_mask_bits"][: containing.n_valid_rows][row_mask],
        materialized["row_mask_mode"][: containing.n_valid_rows][row_mask],
        FINE_TRANS_PARENT,
    )
    expected = expand_mask_rows(tables, image, FINE_TRANS_PARENT)
    np.testing.assert_array_equal(chunk_valid, expected)


def test_heavy_tailed_image_yields_a_one_image_chunk_padded_to_largest_class():
    largest = ROW_CAPACITY_LADDER[-1]
    heavy_rows = 3 * largest + 2  # not a multiple of `largest`, exercises the ceiling.
    row_counts = [2, heavy_rows, 3]
    tables = _synthetic_tables(row_counts)
    chunks = plan_capacity_chunks(
        tables, row_capacity_ladder=ROW_CAPACITY_LADDER, image_capacity_ladder=IMAGE_CAPACITY_LADDER
    )

    heavy_chunks = [c for c in chunks if c.image_start == 1]
    assert len(heavy_chunks) == 1
    heavy = heavy_chunks[0]
    assert heavy.image_stop == 2, "the heavy image must be alone in its chunk"
    assert heavy.n_valid_rows == heavy_rows
    expected_capacity = -(-heavy_rows // largest) * largest
    assert heavy.row_capacity == expected_capacity
    assert heavy.row_capacity % largest == 0
    assert heavy.row_capacity >= heavy_rows
    assert heavy.image_capacity == IMAGE_CAPACITY_LADDER[0]

    # The neighboring images must still be covered by other chunks.
    covered_images = set()
    for c in chunks:
        covered_images.update(range(c.image_start, c.image_stop))
    assert covered_images == set(range(tables.n_images))
