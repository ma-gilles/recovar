from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_em_k1_fine_top_pair_operands import (
    classify_factorial,
    score_candidate_components,
    top_pair_margin,
)


def test_score_candidate_components_matches_direct_formula() -> None:
    reference = np.asarray([1.0 + 2.0j, -0.5 + 0.25j])
    shifted = np.asarray([0.5 - 1.0j, 2.0 + 0.5j])
    correction = np.asarray([2.0, 0.75])
    half_weights = np.asarray([1.0, 2.0])
    result = score_candidate_components(reference, shifted, correction, half_weights)
    expected_norm = -0.5 * np.sum(correction * np.abs(reference) ** 2 * half_weights)
    expected_cross = np.real(
        np.sum(np.conj(shifted) * reference * correction * half_weights)
    )
    assert result["norm"] == pytest.approx(expected_norm)
    assert result["cross"] == pytest.approx(expected_cross)
    assert result["score"] == pytest.approx(expected_norm + expected_cross)


def test_top_pair_margin_is_first_minus_second() -> None:
    references = np.asarray([[1.0 + 0.0j, 0.5j], [0.25 + 0.0j, -1.0j]])
    shifted = np.asarray([[2.0 + 0.0j, 1.0j], [1.0 + 0.0j, -0.5j]])
    correction = np.asarray([1.0, 1.0])
    half_weights = np.asarray([1.0, 2.0])
    result = top_pair_margin(references, shifted, correction, half_weights)
    first, second = result["candidate_components"]
    for component in ("norm", "cross", "score"):
        assert result["pair_margin"][component] == pytest.approx(
            first[component] - second[component]
        )


def _arms(relion_projection_margin: float, recovar_projection_margin: float) -> dict:
    return {
        projected + shifted + correction: {
            "pair_margin": {
                "norm": 0.0,
                "cross": 0.0,
                "score": (
                    relion_projection_margin
                    if projected == "R"
                    else recovar_projection_margin
                ),
            }
        }
        for projected in "RC"
        for shifted in "RC"
        for correction in "RC"
    }


def test_classifies_projected_reference_determined_winner_flip() -> None:
    assert (
        classify_factorial(_arms(relion_projection_margin=0.003, recovar_projection_margin=-0.001))
        == "fine_winner_flip_is_projected_reference_determined"
    )


def test_classifies_mixed_factorial_and_rejects_incomplete_factorial() -> None:
    arms = _arms(relion_projection_margin=0.003, recovar_projection_margin=-0.001)
    arms["RRC"]["pair_margin"]["score"] = -0.01
    assert classify_factorial(arms) == "fine_winner_flip_has_mixed_operand_attribution"
    del arms["CCC"]
    with pytest.raises(ValueError, match="all eight"):
        classify_factorial(arms)
