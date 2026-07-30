from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_em_k1_coarse_score_components import (
    decompose_additive_score_residual,
)


def test_additive_decomposition_identifies_pure_rotation_effect() -> None:
    residual = np.asarray([-2.0, -0.5, 1.0, 1.5])[:, None] * np.ones((1, 3))
    result = decompose_additive_score_residual(residual)
    assert result["energy_fraction"]["rotation_only"] == pytest.approx(1.0)
    assert result["energy_fraction"]["translation_only"] == pytest.approx(0.0)
    assert result["energy_fraction"]["interaction"] == pytest.approx(0.0)


def test_additive_decomposition_identifies_pure_translation_effect() -> None:
    residual = np.ones((4, 1)) * np.asarray([-2.0, 0.5, 1.5])[None, :]
    result = decompose_additive_score_residual(residual)
    assert result["energy_fraction"]["rotation_only"] == pytest.approx(0.0)
    assert result["energy_fraction"]["translation_only"] == pytest.approx(1.0)
    assert result["energy_fraction"]["interaction"] == pytest.approx(0.0)
