import pytest

from scripts.analyze_vdam_native_score_repeat_panel import (
    summarize_reports,
    summarize_score_vectors,
)


def _report(native_log_odds: float, recovar_log_odds: float = 0.25) -> dict[str, object]:
    return {
        "status": "complete",
        "comparisons": {
            "top_pair_score_boundary": {
                "native_best": {"mapped_key": [3, 7]},
                "native_second": {"mapped_key": [2, 5]},
                "native_log_odds_best_over_second": native_log_odds,
                "recovar_log_odds_same_order": recovar_log_odds,
            }
        },
    }


@pytest.mark.unit
def test_summarize_reports_classifies_candidate_inside_native_range() -> None:
    summary = summarize_reports([_report(0.2), _report(0.3), _report(0.4)])

    assert summary["native_log_odds_min"] == 0.2
    assert summary["native_log_odds_max"] == 0.4
    assert summary["recovar_inside_native_range"] is True
    assert summary["minimum_absolute_distance_to_native"] == pytest.approx(0.05)


@pytest.mark.unit
def test_summarize_reports_rejects_native_top_pair_change() -> None:
    changed = _report(0.3)
    changed["comparisons"]["top_pair_score_boundary"]["native_best"]["mapped_key"] = [4, 8]

    with pytest.raises(ValueError, match="top-pair identity"):
        summarize_reports([_report(0.2), changed])


@pytest.mark.unit
def test_summarize_score_vectors_measures_native_envelope() -> None:
    summary = summarize_score_vectors(
        native_vectors=[[10.0, 20.0, 30.0], [11.0, 19.0, 30.0]],
        candidate_vector=[0.0, 9.0, 21.0],
    )

    assert summary["candidate_inside_native_envelope_count"] == 2
    assert summary["candidate_inside_native_envelope_fraction"] == pytest.approx(2 / 3)
    assert summary["candidate_max_distance_outside_native_envelope"] == pytest.approx(1.0)
