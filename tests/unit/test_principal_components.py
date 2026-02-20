import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.principal_components as pc
import recovar.core as core

pytestmark = pytest.mark.unit


def test_zero_boundary_mask_has_expected_structure():
    m = pc.get_zero_boundary_mask((4, 4, 4), dtype=np.float32).reshape(4, 4, 4)
    assert np.all(m[0, :, :] == 0)
    assert np.all(m[:, 0, :] == 0)
    assert np.all(m[:, :, 0] == 0)
    assert np.all(m[1:, 1:, 1:] == 1)


def test_minus_index_negates_frequencies():
    volume_shape = (4, 4, 4)
    idx = np.arange(np.prod(volume_shape))
    minus_idx = np.asarray(pc.get_minus_vec_index(idx, volume_shape))
    freqs = np.asarray(core.vec_indices_to_frequencies(idx, volume_shape))
    minus_freqs = np.asarray(core.vec_indices_to_frequencies(minus_idx, volume_shape))
    # Exclude Nyquist boundary where mapping is not strictly invertible on the wrapped grid.
    interior = np.all(np.abs(freqs) < (volume_shape[0] // 2), axis=1)
    np.testing.assert_array_equal(minus_freqs[interior], -freqs[interior])


def test_make_symmetric_columns_matches_numpy_version():
    volume_shape = (4, 4, 4)
    vol_size = int(np.prod(volume_shape))
    n_cols = 5
    rng = np.random.default_rng(0)

    columns = (rng.normal(size=(vol_size, n_cols)) + 1j * rng.normal(size=(vol_size, n_cols))).astype(np.complex64)
    picked = rng.choice(vol_size, size=n_cols, replace=False)

    cols_a, minus_a, good_a = pc.make_symmetric_columns(columns, picked, volume_shape)
    cols_b, minus_b, good_b = pc.make_symmetric_columns_np(columns, picked, volume_shape)

    np.testing.assert_allclose(np.asarray(cols_a), np.asarray(cols_b), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(minus_a), np.asarray(minus_b))
    np.testing.assert_array_equal(np.asarray(good_a), np.asarray(good_b))
