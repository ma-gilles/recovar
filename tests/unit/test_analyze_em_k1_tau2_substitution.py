import pytest

from scripts.analyze_em_k1_tau2_substitution import (
    _infer_cubic_shape,
    classify_substitution,
)


def test_classify_substitution_accepts_majority_explanation():
    result = classify_substitution(0.01, 0.004)

    assert result["classification"] == (
        "relion_tau2_explains_majority_of_map_residual"
    )
    assert result["relative_l2_explained_fraction"] == pytest.approx(0.6)


def test_classify_substitution_rejects_small_or_adverse_change():
    small = classify_substitution(0.01, 0.009)
    adverse = classify_substitution(0.01, 0.011)

    assert small["classification"] == (
        "relion_tau2_does_not_explain_map_residual"
    )
    assert small["relative_l2_explained_fraction"] == pytest.approx(0.1)
    assert adverse["classification"] == (
        "relion_tau2_does_not_explain_map_residual"
    )
    assert adverse["relative_l2_explained_fraction"] == pytest.approx(-0.1)


def test_infer_cubic_shape_fails_closed():
    assert _infer_cubic_shape(27) == (3, 3, 3)
    with pytest.raises(ValueError, match="not cubic"):
        _infer_cubic_shape(28)
