import numpy as np

from scripts.analyze_k1_parity_state_ab import _array_metrics, _f32


def test_f32_reports_exact_bits():
    assert _f32(1.0) == {"value": 1.0, "bits_hex": "0x3f800000"}


def test_array_metrics_reports_equality_and_delta():
    control = np.asarray([1.0, 2.0], dtype=np.float32)
    candidate = np.asarray([1.0, 3.0], dtype=np.float32)

    result = _array_metrics(control, candidate)

    assert result["shape"] == [2]
    assert result["bit_equal_fraction"] == 0.5
    assert result["max_abs_delta"] == 1.0
    assert result["relative_l2"] == np.sqrt(1.0 / 5.0)
