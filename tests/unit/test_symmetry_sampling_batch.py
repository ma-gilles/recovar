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
    actual = get_oversampled_orientations_batch(3, oversampling, directions, psi, perturbation, symmetry)
    assert actual.dtype == np.float64  # Preserve RFLOAT metadata before GPU casts.
    assert actual.shape == (3 * 8**oversampling, 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("symmetry", ["D5", "O", "I1"])
@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("oversampling", [0, 1])
def test_batched_symmetry_rows_match_scalar_source_for_random_rows(symmetry, order, oversampling):
    from recovar.relion_bind._relion_bind_core import (
        get_healpix_sampling_metadata,
        get_oversampled_orientations,
        get_oversampled_orientations_batch,
    )

    metadata = get_healpix_sampling_metadata(order, -1.0, symmetry)
    rng = np.random.default_rng(order * 10 + oversampling)
    directions = rng.integers(0, len(metadata["rot"]), size=24, dtype=np.int64)
    psi = rng.integers(0, len(metadata["psi"]), size=24, dtype=np.int64)
    perturbation = -0.152378
    expected = np.concatenate(
        [
            get_oversampled_orientations(order, oversampling, int(d), int(p), perturbation, symmetry)
            for d, p in zip(directions, psi, strict=True)
        ]
    )
    actual = get_oversampled_orientations_batch(order, oversampling, directions, psi, perturbation, symmetry)
    np.testing.assert_array_equal(actual, expected)


def test_batched_symmetry_rows_reuse_one_symmetry_reduced_sampling():
    """A local-search batch must not rebuild the point-group grid per row.

    Rebuilding it per row made 10202's I1 order-4 local-search layout take
    11 h per half-iteration. The ratio bound is machine-independent: reuse
    costs about one scalar call, rebuilding costs one call per row.
    """
    import time

    from recovar.relion_bind._relion_bind_core import (
        get_healpix_sampling_metadata,
        get_oversampled_orientations,
        get_oversampled_orientations_batch,
    )

    order, n_rows = 4, 4096
    metadata = get_healpix_sampling_metadata(order, -1.0, "I1")
    rng = np.random.default_rng(10202)
    directions = rng.integers(0, len(metadata["rot"]), size=n_rows, dtype=np.int64)
    psi = rng.integers(0, len(metadata["psi"]), size=n_rows, dtype=np.int64)

    single_seconds = []
    for _ in range(3):
        start = time.perf_counter()
        get_oversampled_orientations(order, 0, int(directions[0]), int(psi[0]), 0.1, "I1")
        single_seconds.append(time.perf_counter() - start)
    start = time.perf_counter()
    rows = get_oversampled_orientations_batch(order, 0, directions, psi, 0.1, "I1")
    batch_seconds = time.perf_counter() - start

    assert rows.shape == (n_rows, 3)
    assert batch_seconds < 64.0 * float(np.median(single_seconds))
