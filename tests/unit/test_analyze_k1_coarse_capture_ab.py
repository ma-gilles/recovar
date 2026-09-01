import numpy as np

from scripts.analyze_k1_coarse_capture_ab import (
    OPTIONAL_PROJECTION_FIELDS,
    ORDERED_FIELDS,
    _available_ordered_fields,
    _centered,
    _metrics,
)


def test_centered_removes_common_score_offset():
    lhs = np.asarray([1.0, 3.0], dtype=np.float32)
    rhs = np.asarray([5.0, 7.0], dtype=np.float32)

    assert np.array_equal(_centered(lhs), _centered(rhs))


def test_metrics_preserves_complex_residual():
    lhs = np.asarray([1.0 + 2.0j], dtype=np.complex64)
    rhs = np.asarray([1.0 + 3.0j], dtype=np.complex64)

    result = _metrics(lhs, rhs)

    assert result["bit_equal_fraction"] == 0.0
    assert result["max_abs_delta"] == 1.0
    assert result["first_bit_difference"] == {
        "flat_index": 0,
        "index": [0],
        "control": {
            "dtype": "complex64",
            "bytes_hex": lhs[0].tobytes().hex(),
            "real": 1.0,
            "imag": 2.0,
        },
        "candidate": {
            "dtype": "complex64",
            "bytes_hex": rhs[0].tobytes().hex(),
            "real": 1.0,
            "imag": 3.0,
        },
    }


def test_metrics_uses_byte_equality_for_signed_zero_and_nan():
    lhs = np.asarray([0.0, np.nan], dtype=np.float32)
    rhs = np.asarray([-0.0, np.nan], dtype=np.float32)

    result = _metrics(lhs, rhs)

    assert result["bit_equal_fraction"] == 0.5
    assert result["bit_unequal_count"] == 1
    assert result["first_bit_difference"]["index"] == [0]
    assert result["first_bit_difference"]["control"]["bytes_hex"] == "00000000"
    assert result["first_bit_difference"]["candidate"]["bytes_hex"] == "00000080"


def test_available_fields_allow_one_sided_optional_projection_capture():
    required = [field for field in ORDERED_FIELDS if field not in OPTIONAL_PROJECTION_FIELDS]
    available, missing_control, missing_candidate = _available_ordered_fields(
        required,
        list(ORDERED_FIELDS),
    )

    assert available == required
    assert missing_control == list(OPTIONAL_PROJECTION_FIELDS)
    assert missing_candidate == []
