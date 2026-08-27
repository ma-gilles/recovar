"""Unit tests for the exact native/RECOVAR fine-operand GPU replay helpers."""

import numpy as np
import pytest

from scripts.replay_k1_native_fine_operand_arms_gpu import (
    _center,
    _dense_compact_rows,
    _native_packed_physical_indices,
)

pytestmark = pytest.mark.unit


def test_dense_compact_rows_scatter_preserves_leading_axes():
    compact = np.asarray(
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
        dtype=np.complex64,
    )
    lookup = np.asarray([-1, 1, 0, -1], dtype=np.int32)

    dense = _dense_compact_rows(compact, lookup)

    expected = np.asarray(
        [[0, 3 + 4j, 1 + 2j, 0], [0, 7 + 8j, 5 + 6j, 0]],
        dtype=np.complex64,
    )
    np.testing.assert_array_equal(dense, expected)


def test_dense_compact_rows_rejects_out_of_bounds_lookup():
    with pytest.raises(ValueError, match="exceeds compact row width"):
        _dense_compact_rows(np.zeros((2, 3), dtype=np.float32), np.asarray([3]))


def test_center_uses_float32_max_subtraction():
    values = np.asarray([4.0, 2.5, -1.0], dtype=np.float32)
    np.testing.assert_array_equal(
        _center(values),
        np.asarray([0.0, -1.5, -5.0], dtype=np.float32),
    )


def test_native_packed_physical_indices_preserve_even_positive_nyquist_row():
    indices = _native_packed_physical_indices(
        current_size=4,
        physical_image_size=8,
    )
    half_width = 5
    expected = np.asarray(
        [
            4 * half_width + 0,
            4 * half_width + 1,
            4 * half_width + 2,
            5 * half_width + 0,
            5 * half_width + 1,
            5 * half_width + 2,
            6 * half_width + 0,
            6 * half_width + 1,
            6 * half_width + 2,
            3 * half_width + 0,
            3 * half_width + 1,
            3 * half_width + 2,
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(indices, expected)
