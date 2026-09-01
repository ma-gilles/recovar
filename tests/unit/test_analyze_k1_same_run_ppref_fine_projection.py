from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_k1_same_run_ppref_fine_projection import (
    _classification,
    _native_pixel_indices,
)
from scripts.validate_relion_fine_operand_capture import PIXEL_DTYPE


@pytest.mark.unit
def test_native_pixel_indices_map_signed_y_to_centered_rows() -> None:
    pixels = np.zeros(4, dtype=PIXEL_DTYPE)
    pixels["x"] = [0, 28, 1, 7]
    pixels["y"] = [0, 28, -27, -1]
    half_width = 129
    np.testing.assert_array_equal(
        _native_pixel_indices(pixels, 256),
        np.asarray([128 * half_width, 156 * half_width + 28, 101 * half_width + 1, 127 * half_width + 7]),
    )


@pytest.mark.unit
def test_classification_requires_projection_and_score_exactness() -> None:
    exact = {"exact_equal": True, "relative_l2_over_relion": 0.0}
    unequal = {"exact_equal": False, "relative_l2_over_relion": 1e-7}
    assert (
        _classification(
            {"projected_reference": exact, "raw_diff2": exact},
            {"manual": {"projected_reference": unequal}},
        )
        == "same_run_ppref_and_texture_projection_are_bit_exact"
    )
    assert (
        _classification(
            {"projected_reference": unequal, "raw_diff2": exact},
            {"manual": {"projected_reference": unequal}},
        )
        == "same_run_texture_projection_score_is_exact_despite_pixel_rounding"
    )
