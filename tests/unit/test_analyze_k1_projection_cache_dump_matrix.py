import numpy as np

from scripts.analyze_k1_projection_cache_dump_matrix import _array_delta


def test_array_delta_reports_exact_and_numeric_differences():
    left = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    right = np.asarray([1.0, 3.0, 4.0], dtype=np.float32)

    result = _array_delta(left, right)

    assert result["bitwise_equal"] is False
    assert result["different_count"] == 1
    assert result["max_abs"] == 1.0
    assert result["relative_l2"] > 0.0


def test_array_delta_treats_matching_nan_as_equal():
    values = np.asarray([np.nan, 1.0], dtype=np.float32)

    result = _array_delta(values, values.copy())

    assert result["bitwise_equal"] is True
    assert result["different_count"] == 0
