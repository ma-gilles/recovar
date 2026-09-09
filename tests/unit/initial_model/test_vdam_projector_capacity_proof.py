"""Focused CPU/C++ proof for a stable-shape InitialModel projector ABI."""

import logging

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.fourier_window import (
    stable_fourier_window_current_size,
)
from scripts.prove_vdam_projector_capacity import (
    GF46_IMAGE_SIZE,
    GF46_LOGICAL_CURRENT_SIZES,
    analyze,
    center_pad_relion_projector,
    crop_relion_projection_capacity,
    crop_relion_projector_capacity,
    gf46_logical_physical_pairs,
)

pytestmark = pytest.mark.unit


def test_center_padding_preserves_relion_frequency_coordinates_bitwise():
    logical = (np.arange(7 * 7 * 4, dtype=np.float32).reshape(7, 7, 4) + np.complex64(1j)).astype(np.complex64)

    padded = center_pad_relion_projector(logical, (11, 11, 6))

    assert padded.shape == (11, 11, 6)
    assert padded.dtype == np.complex64
    np.testing.assert_array_equal(
        crop_relion_projector_capacity(padded, logical.shape),
        logical,
    )
    assert np.count_nonzero(padded) == logical.size


def test_gf46_fixture_tracks_every_observed_logical_to_physical_pair():
    pairs = gf46_logical_physical_pairs()

    assert tuple(logical for logical, _physical in pairs) == GF46_LOGICAL_CURRENT_SIZES
    assert all(physical == stable_fourier_window_current_size(logical, GF46_IMAGE_SIZE) for logical, physical in pairs)
    assert len(pairs) == 32
    assert len({physical for _logical, physical in pairs}) == 14


def test_projection_capacity_crop_preserves_fftw_signed_rows():
    physical_size = 8
    logical_size = 4
    physical = np.arange(
        physical_size * (physical_size // 2 + 1),
        dtype=np.float32,
    )[None, :]

    cropped = crop_relion_projection_capacity(
        physical,
        physical_size=physical_size,
        logical_size=logical_size,
    ).reshape(logical_size, logical_size // 2 + 1)

    physical_grid = physical.reshape(physical_size, physical_size // 2 + 1)
    np.testing.assert_array_equal(
        cropped,
        physical_grid[[0, 1, 2, 7], : logical_size // 2 + 1],
    )


@pytest.mark.slow
def test_full_gf46_projector_capacity_proof(caplog):
    pytest.importorskip("recovar.relion_bind._relion_bind_core")
    caplog.set_level(logging.INFO)

    report = analyze()
    logging.info("\n%s", report["markdown"])

    summary = report["summary"]
    assert summary["pair_count"] == len(GF46_LOGICAL_CURRENT_SIZES)
    assert summary["physical_class_count"] == 14
    assert summary["active_sphere_texels_all_bitwise"] is True
    assert summary["center_padded_logical_box_all_bitwise"] is True

    constructions = summary["projection_constructions"]
    assert constructions["center_padded_logical_cutoff"]["all_bitwise"] is True

    # A larger C++ rebuild fills a new Fourier annulus. Trilinear interpolation
    # can read that annulus at the logical boundary, so it is not an exact
    # substitute even if the logical radial cutoff is retained.
    assert constructions["physical_rebuild_logical_cutoff"]["all_bitwise"] is False

    # Capacity alone is insufficient: the logical cutoff is part of exact
    # semantics and cannot be replaced by the bucket's physical r_max.
    assert constructions["center_padded_physical_cutoff"]["all_bitwise"] is False
