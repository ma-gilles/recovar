from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_k1_firstiter_cc_native_operands import (
    _normalized_cc_components,
    _rotation_map,
)


pytestmark = pytest.mark.unit


def test_normalized_cc_components_preserve_identical_candidate_score() -> None:
    reference = np.asarray([[1 + 2j, 3 - 4j, -2 + 1j]], dtype=np.complex64)
    shifted = reference.copy()
    corr_img = np.asarray([0.5, 2.0, 3.0], dtype=np.float32)
    *_, score = _normalized_cc_components(reference, shifted, corr_img)
    expected = np.sqrt(
        np.sum(np.abs(reference.astype(np.complex128)) ** 2 * corr_img, axis=1)
    )
    np.testing.assert_allclose(score, expected.astype(np.float32), rtol=2e-7)


def test_rotation_map_uses_matrix_identity_not_engine_integer_id() -> None:
    rotations = np.asarray(
        [
            np.eye(3),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    mapped, errors = _rotation_map(rotations[::-1], rotations)
    np.testing.assert_array_equal(mapped, [1, 0])
    np.testing.assert_allclose(errors, 0.0, atol=1e-12)
