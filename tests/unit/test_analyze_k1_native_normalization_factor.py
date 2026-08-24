from __future__ import annotations

import numpy as np

from scripts.analyze_k1_native_normalization_factor import (
    recover_normalization_factor,
    zero_padded_integer_shift,
)


def test_zero_padded_integer_shift_matches_relion_destination_convention():
    image = np.arange(1, 10, dtype=np.float32).reshape(3, 3)

    shifted = zero_padded_integer_shift(image, shift_x=1, shift_y=-1)

    np.testing.assert_array_equal(
        shifted,
        np.asarray(
            [
                [0.0, 4.0, 5.0],
                [0.0, 7.0, 8.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_recover_normalization_factor_requires_exact_float32_bytes():
    source = np.linspace(-3.0, 4.0, 257, dtype=np.float32).reshape(1, -1)
    factor = np.asarray(0x3F809AC5, dtype=np.uint32).view(np.float32)
    native = np.multiply(source, factor, dtype=np.float32)

    report = recover_normalization_factor(source, native, search_ulp=16)

    assert report["factor_float32_bits"] == "0x3f809ac5"
    assert report["bit_exact_count"] == source.size
    assert report["mismatch_count"] == 0
    assert report["relative_l2"] == 0.0
