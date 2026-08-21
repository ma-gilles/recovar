"""Unit tests for the VDAM native translation-boundary analyzer."""

import numpy as np
import pytest

from scripts.analyze_vdam_native_translation_boundary import (
    _current_crop_to_compact,
    _metric,
    _native_crop_rows,
)

pytestmark = pytest.mark.unit


def test_native_crop_rows_map_centered_half_to_relion_fftw_crop():
    full_size = 8
    current_size = 4
    half_width = full_size // 2 + 1
    # Centered rows ky=-1, 0, 1 map to native FFTW rows 3, 0, 1.
    score_indices = np.asarray([3 * half_width + 1, 4 * half_width, 5 * half_width + 2])

    crop = _native_crop_rows(score_indices, full_size, current_size)

    np.testing.assert_array_equal(crop, np.asarray([10, 0, 5], dtype=np.int32))


def test_current_crop_to_compact_preserves_native_pixel_lanes():
    lookup = _current_crop_to_compact(np.asarray([10, 0, 5]), current_size=4)

    assert lookup.shape == (12,)
    np.testing.assert_array_equal(lookup[[0, 5, 10]], np.asarray([1, 2, 0]))
    assert np.count_nonzero(lookup < 0) == 9


def test_metric_reports_exact_complex_values_and_residual():
    reference = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64)
    candidate = reference.copy()
    candidate[1] += np.complex64(1j)

    result = _metric(reference, candidate)

    assert result["exact_count"] == 1
    assert result["value_count"] == 2
    assert result["max_abs"] == pytest.approx(1.0)
