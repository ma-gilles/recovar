"""Unit tests for the focused native K=1 fine-score operand analyzer."""

import numpy as np
import pytest

from scripts.analyze_em_k1_native_fine_operands import _full_to_compact, _tree_sum

pytestmark = pytest.mark.unit


def test_tree_sum_matches_relion_lane_order():
    values = np.arange(600, dtype=np.float32).reshape(2, 300) / np.float32(17.0)
    lanes = np.zeros((2, 256), dtype=np.float32)
    lanes[:, :] += values[:, :256]
    lanes[:, :44] += values[:, 256:]
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes = lanes[:, :width] + lanes[:, width : 2 * width]

    np.testing.assert_array_equal(_tree_sum(values), lanes[:, 0])


def test_full_to_compact_maps_centered_rows_to_relion_fftw_rows():
    # Full-size centered rows 2, 3, 4 correspond to ky -2, -1, 0. In a
    # current-size-4 packed FFT they become RELION rows 2, 3, 0.
    window = np.array([2 * 5, 3 * 5 + 1, 4 * 5 + 2], dtype=np.int32)

    lookup = _full_to_compact(window, full_size=8, current_size=4)

    expected = np.full(12, -1, dtype=np.int32)
    expected[[2 * 3, 3 * 3 + 1, 2]] = np.arange(3, dtype=np.int32)
    np.testing.assert_array_equal(lookup, expected)
