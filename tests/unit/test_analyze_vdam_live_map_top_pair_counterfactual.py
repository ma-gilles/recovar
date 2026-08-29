import pytest

from scripts.analyze_vdam_live_map_top_pair_counterfactual import (
    summarize_pair_spacing,
)


def test_summarize_pair_spacing_separates_likelihood_and_prior_flip():
    report = summarize_pair_spacing(
        first_diff2=10.0,
        second_diff2=12.5,
        first_prior=-4.0,
        second_prior=-1.6,
        first_translation=66,
        second_translation=70,
    )

    assert report["likelihood_spacing_first_minus_second"] == pytest.approx(2.5)
    assert report["prior_spacing_first_minus_second"] == pytest.approx(-2.4)
    assert report["total_spacing_first_minus_second"] == pytest.approx(0.1)
    assert report["winner_translation_index"] == 66


def test_summarize_pair_spacing_reports_second_winner():
    report = summarize_pair_spacing(
        first_diff2=10.0,
        second_diff2=12.3,
        first_prior=-4.0,
        second_prior=-1.6,
        first_translation=66,
        second_translation=70,
    )

    assert report["total_spacing_first_minus_second"] == pytest.approx(-0.1)
    assert report["winner_translation_index"] == 70
