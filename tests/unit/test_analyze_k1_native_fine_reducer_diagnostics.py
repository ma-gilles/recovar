"""Focused tests for native RELION fine-score reducer diagnostics."""

import numpy as np
import pytest

from scripts.analyze_k1_native_fine_operand_ffi_replay import _ulp_distance
from scripts.analyze_k1_native_texture_fine_counterfactual import (
    _winner_from_raw_diff2,
)


pytestmark = pytest.mark.unit


def test_ulp_distance_handles_sign_boundary_and_adjacent_values() -> None:
    one = np.float32(1.0)
    adjacent = np.nextafter(one, np.float32(np.inf), dtype=np.float32)

    assert _ulp_distance(one, one) == 0
    assert _ulp_distance(one, adjacent) == 1
    assert _ulp_distance(np.float32(-0.0), np.float32(0.0)) == 1


def test_winner_uses_mask_and_stable_row_major_tie_order() -> None:
    result = _winner_from_raw_diff2(
        np.asarray([[4.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([[True, True], [True, False]], dtype=bool),
    )

    assert result["rotation_row"] == 0
    assert result["translation_row"] == 1
    assert result["raw_diff2"] == 1.0
    assert result["minimum_raw_diff2"] == 1.0


def test_winner_includes_rotation_and_translation_priors() -> None:
    result = _winner_from_raw_diff2(
        np.ones((2, 2), dtype=np.float32),
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([0.0, 3.0], dtype=np.float32),
        np.ones((2, 2), dtype=bool),
    )

    assert result["rotation_row"] == 1
    assert result["translation_row"] == 1
    assert result["score_with_prior"] == 5.0
