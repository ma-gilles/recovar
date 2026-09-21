"""The local-search layout adapter for the device-resident stages (T12).

Ticket: ``em_parity_tickets_20260918/T12_resident_local_search.md``.

Everything here is CPU: the adapter is host planning. The reference for the
candidate mask is the layout's own ``sample_mask_rows``, which is what the
bucketed local engine expands into ``LocalBucketSpec.local_sample_mask``, so a
mismatch here would be a mismatch against the accepted engine's candidate set.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.local.local_layout import (
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
)
from recovar.em.sampling import build_local_search_grid_metadata
from recovar.em.sparse_pass2 import resident_local_layout as rll

pytestmark = pytest.mark.unit


PARENT_ORDER = 1
OVERSAMPLING = 1
N_IMAGES = 7
VOXEL_SIZE = 1.5


def _prior_eulers(n_images, seed=20260919):
    rng = np.random.default_rng(seed)
    eulers = np.zeros((n_images, 3), dtype=np.float64)
    eulers[:, 0] = rng.uniform(0.0, 360.0, size=n_images)
    eulers[:, 1] = rng.uniform(20.0, 160.0, size=n_images)
    eulers[:, 2] = rng.uniform(0.0, 360.0, size=n_images)
    return eulers


def _parent_layout(n_images=N_IMAGES, n_coarse_trans=4, seed=20260919):
    translations = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float32
    )[:n_coarse_trans]
    prior_translations = np.zeros((n_images, 2), dtype=np.float32)
    return build_local_hypothesis_layout(
        _prior_eulers(n_images, seed),
        None,
        0.35,
        0.35,
        PARENT_ORDER,
        translations,
        prior_translations,
        3.0,
        None,
        VOXEL_SIZE,
        grid_metadata=build_local_search_grid_metadata(PARENT_ORDER),
        translation_prior_reference_translations=translations,
        dtype=np.float32,
    ), translations


def _pass2_layout(significant_mode="mixed", seed=20260919):
    """A pass-2 layout with the three support shapes half_scoring can produce.

    ``mixed`` gives some images an explicit significant-sample list (so the
    layout carries a real per-row mask) and some ``None`` (full parent support).
    ``full`` gives every image ``None``, which is how the layout spells "no mask
    at all" and is the shape RELION's ``full_parent`` diagnostic produces.
    """

    parent, translations = _parent_layout(seed=seed)
    n_coarse_trans = int(translations.shape[0])
    rng = np.random.default_rng(seed + 1)
    samples = []
    for image in range(parent.n_images):
        start = int(parent.rotation_offsets[image])
        stop = int(parent.rotation_offsets[image + 1])
        parent_ids = np.asarray(parent.rotation_ids_flat[start:stop], dtype=np.int64)
        if significant_mode == "full" or (significant_mode == "mixed" and image % 3 == 0):
            samples.append(None)
            continue
        pairs = (parent_ids[:, None] * n_coarse_trans + np.arange(n_coarse_trans)).reshape(-1)
        keep = rng.choice(pairs, size=max(1, pairs.size // 3), replace=False)
        samples.append(np.sort(keep.astype(np.int64)))
    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent,
        samples,
        PARENT_ORDER,
        oversampling_order=OVERSAMPLING,
        random_perturbation=0.0,
        dtype=np.float32,
    )
    return layout


def test_tables_mirror_the_layout_row_for_row():
    layout = _pass2_layout()
    tables = rll.tables_from_local_layout(layout)

    assert tables.n_images == layout.n_images
    assert tables.n_rows == layout.total_local_rotations
    assert tables.n_trans == int(layout.translation_grid.shape[0])
    np.testing.assert_array_equal(tables.row_offsets, layout.rotation_offsets)
    np.testing.assert_array_equal(tables.rotations, layout.rotations_flat)
    np.testing.assert_array_equal(tables.mstep_rotations, layout.mstep_rotations_flat)
    np.testing.assert_array_equal(tables.row_log_prior, layout.rotation_log_priors_flat)
    np.testing.assert_array_equal(tables.row_rotation_id, layout.rotation_ids_flat)
    np.testing.assert_array_equal(tables.row_posterior_id, layout.rotation_posterior_ids_flat)
    np.testing.assert_array_equal(tables.source_eulers, layout.source_eulers_flat)
    np.testing.assert_array_equal(tables.translation_log_prior, layout.translation_log_priors)
    assert tables.n_posterior_bins == int(layout.n_global_rotations)

    # row_image is the CSR expansion of the offsets, in image order.
    for image in range(tables.n_images):
        start = int(tables.row_offsets[image])
        stop = int(tables.row_offsets[image + 1])
        assert set(tables.row_image[start:stop].tolist()) <= {image}
        assert stop - start == int(layout.rotation_counts[image])


def test_mstep_rotations_fall_back_to_the_scoring_rotations():
    """The layout's own fallback when no separate M-step grid was generated."""

    layout = _pass2_layout()
    stripped = type(layout)(
        **{
            **{
                field: getattr(layout, field)
                for field in layout.__dataclass_fields__
            },
            "mstep_rotations_flat": None,
        }
    )
    tables = rll.tables_from_local_layout(stripped)
    np.testing.assert_array_equal(tables.mstep_rotations, tables.rotations)


def test_mask_expansion_matches_the_layouts_own_unpacking():
    layout = _pass2_layout()
    tables = rll.tables_from_local_layout(layout)
    assert tables.row_mask_bits is not None, "the mixed fixture must carry a real mask"

    reference = layout.sample_mask_rows()
    expanded = rll.expand_local_mask_rows(tables, 0, tables.n_rows)
    np.testing.assert_array_equal(expanded, reference)

    device = rll.expand_local_chunk_mask_jnp(
        jnp.asarray(tables.row_mask_bits), n_trans=tables.n_trans
    )
    np.testing.assert_array_equal(np.asarray(device), reference)


def test_full_support_layout_keeps_the_compact_none_spelling():
    layout = _pass2_layout(significant_mode="full")
    tables = rll.tables_from_local_layout(layout)
    assert layout.sample_mask_bits is None
    assert tables.row_mask_bits is None
    # ``None`` means full support on both sides.
    np.testing.assert_array_equal(
        rll.expand_local_mask_rows(tables, 0, tables.n_rows),
        np.ones((tables.n_rows, tables.n_trans), dtype=bool),
    )
    assert rll.expand_local_chunk_mask_jnp(None, n_trans=tables.n_trans) is None


@pytest.mark.parametrize("n_trans", [1, 7, 8, 9, 31, 84, 129])
def test_device_mask_expansion_is_bit_exact_for_any_translation_count(n_trans):
    """The little-endian packing the layout uses, at and across byte boundaries."""

    rng = np.random.default_rng(1000 + n_trans)
    rows = 11
    dense = rng.random((rows, n_trans)) < 0.5
    bits = np.packbits(dense, axis=1, bitorder="little")
    device = rll.expand_local_chunk_mask_jnp(jnp.asarray(bits), n_trans=n_trans)
    np.testing.assert_array_equal(np.asarray(device), dense)


def test_chunks_cover_every_row_once_and_respect_capacities():
    layout = _pass2_layout()
    tables = rll.tables_from_local_layout(layout)
    row_ladder = (64, 256, 1024)
    image_ladder = (2, 4, 8)
    chunks = rll.plan_local_capacity_chunks(
        tables, row_capacity_ladder=row_ladder, image_capacity_ladder=image_ladder
    )
    assert chunks
    covered_rows = 0
    next_image = 0
    for chunk in chunks:
        assert chunk.image_start == next_image
        next_image = chunk.image_stop
        assert chunk.row_start == int(tables.row_offsets[chunk.image_start])
        assert chunk.row_stop == int(tables.row_offsets[chunk.image_stop])
        assert chunk.n_valid_rows <= chunk.row_capacity
        assert chunk.n_valid_images <= chunk.image_capacity
        covered_rows += chunk.n_valid_rows
    assert next_image == tables.n_images
    assert covered_rows == tables.n_rows


def test_materialized_chunk_is_the_slice_plus_inert_padding():
    layout = _pass2_layout()
    tables = rll.tables_from_local_layout(layout)
    chunks = rll.plan_local_capacity_chunks(
        tables, row_capacity_ladder=(64, 256, 1024), image_capacity_ladder=(2, 4, 8)
    )
    for chunk in chunks:
        host = rll.materialize_local_chunk(tables, chunk)
        n_valid = int(host["n_valid_rows"])
        rs, re = chunk.row_start, chunk.row_stop
        assert n_valid == re - rs

        np.testing.assert_array_equal(host["rotations"][:n_valid], tables.rotations[rs:re])
        np.testing.assert_array_equal(
            host["mstep_rotations"][:n_valid], tables.mstep_rotations[rs:re]
        )
        np.testing.assert_array_equal(host["row_log_prior"][:n_valid], tables.row_log_prior[rs:re])
        np.testing.assert_array_equal(
            host["row_posterior_id"][:n_valid], tables.row_posterior_id[rs:re]
        )
        np.testing.assert_array_equal(
            host["row_image_local"][:n_valid], tables.row_image[rs:re] - chunk.image_start
        )
        np.testing.assert_array_equal(
            host["image_ids"][: chunk.n_valid_images],
            np.arange(chunk.image_start, chunk.image_stop),
        )
        assert np.all(host["image_ids"][chunk.n_valid_images :] == -1)

        # Padding is inert: no candidate cell, a dropped histogram bin, a finite
        # rotation and a prior no winner can reach.
        mask = rll.expand_local_chunk_mask_jnp(
            host["row_mask_bits"], n_trans=tables.n_trans
        )
        if mask is not None:
            assert not bool(np.asarray(mask)[n_valid:].any())
        assert np.all(host["row_posterior_id"][n_valid:] == tables.n_posterior_bins)
        assert np.all(np.isfinite(host["rotations"][n_valid:]))
        assert np.all(host["row_log_prior"][n_valid:] == np.float32(-1e30))

        # The chunk's device mask agrees with the reference expansion row by row.
        if mask is not None:
            np.testing.assert_array_equal(
                np.asarray(mask)[:n_valid], rll.expand_local_mask_rows(tables, rs, re)
            )


def test_one_image_chunk_for_a_row_count_above_the_largest_class():
    layout = _pass2_layout()
    tables = rll.tables_from_local_layout(layout)
    counts = np.diff(tables.row_offsets)
    tiny_ladder = (int(counts.min()),)
    chunks = rll.plan_local_capacity_chunks(
        tables, row_capacity_ladder=tiny_ladder, image_capacity_ladder=(1, 2)
    )
    heavy = [chunk for chunk in chunks if chunk.n_valid_rows > tiny_ladder[0]]
    assert heavy, "the fixture must contain an image above the largest row class"
    for chunk in heavy:
        assert chunk.n_valid_images == 1
        assert chunk.row_capacity % tiny_ladder[0] == 0
        assert chunk.row_capacity >= chunk.n_valid_rows


def test_adapter_rejects_a_mask_of_the_wrong_width():
    layout = _pass2_layout()
    bad = type(layout)(
        **{
            **{field: getattr(layout, field) for field in layout.__dataclass_fields__},
            "sample_mask_bits": np.zeros(
                (layout.total_local_rotations, 1), dtype=np.uint8
            ),
        }
    )
    if (int(layout.translation_grid.shape[0]) + 7) // 8 == 1:
        pytest.skip("the fixture's translation count already packs into one byte")
    with pytest.raises(ValueError, match="sample_mask_bits"):
        rll.tables_from_local_layout(bad)
