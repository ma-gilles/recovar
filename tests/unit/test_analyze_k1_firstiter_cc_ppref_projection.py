from __future__ import annotations

import pytest

from scripts.analyze_k1_firstiter_cc_ppref_projection import classify


pytestmark = pytest.mark.unit


def test_classify_pretexture_boundary() -> None:
    assert (
        classify(frozen_texture_l2=1e-9, captured_cross_l2=1e-5)
        == "projected_reference_difference_enters_before_texture_projection"
    )


def test_classify_open_texture_boundary() -> None:
    assert (
        classify(frozen_texture_l2=2e-6, captured_cross_l2=1e-5)
        == "texture_projection_boundary_remains_open"
    )
