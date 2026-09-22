"""Batched symmetry sampling retains scalar RELION Euler rows and row order."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("symmetry", ["C1", "D5", "O", "I1"])
@pytest.mark.parametrize("oversampling", [0, 1])
def test_batched_symmetry_rows_match_scalar_source(symmetry, oversampling):
    from recovar.relion_bind._relion_bind_core import (
        get_healpix_sampling_metadata,
        get_oversampled_orientations,
        get_oversampled_orientations_batch,
    )

    metadata = get_healpix_sampling_metadata(3, -1.0, symmetry)
    directions = np.array([len(metadata["rot"]) - 1, 0, 0], dtype=np.int64)
    psi = np.array([1, len(metadata["psi"]) - 1, 1], dtype=np.int64)
    perturbation = 0.173
    expected = np.concatenate(
        [
            get_oversampled_orientations(3, oversampling, int(d), int(p), perturbation, symmetry)
            for d, p in zip(directions, psi, strict=True)
        ]
    )
    actual = get_oversampled_orientations_batch(
        3, oversampling, directions, psi, perturbation, symmetry
    )
    assert actual.dtype == np.float64  # Preserve RFLOAT metadata before GPU casts.
    assert actual.shape == (3 * 8**oversampling, 3)
    np.testing.assert_array_equal(actual, expected)
