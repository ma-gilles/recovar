import pytest

from scripts.analyze_k1_fine_score_stage_panel import (
    _parse_expected_stacks,
    summarize_reports,
)
from scripts.analyze_k1_fine_score_stages import STAGES


def _report(first_unequal: str, *, scale: float = 1.0) -> dict:
    stage_exact = {}
    still_exact = True
    for stage in STAGES:
        if stage == first_unequal:
            still_exact = False
        stage_exact[stage] = still_exact
    if first_unequal == "all_stages_exact":
        stage_exact = {stage: True for stage in STAGES}
    return {
        "stage_exact": stage_exact,
        "first_exact_unequal_boundary": first_unequal,
        "native_active_count": 160,
        "recovar_active_candidate_count": 160,
        "native_significant_count": 14,
        "recovar_significant_count": 14,
        "support_intersection_count": 14,
        "comparisons": {
            "preprior_score_centered": {
                "max_abs": 1.0e-4 * scale,
                "relative_l2_over_reference": 2.0e-6 * scale,
                "mismatch_count": int(10 * scale),
            }
        },
    }


@pytest.mark.unit
def test_expected_stacks_accepts_colons_and_rejects_duplicates() -> None:
    assert _parse_expected_stacks("8792:94084:98352") == {8792, 94084, 98352}
    with pytest.raises(ValueError, match="unique"):
        _parse_expected_stacks("8792,8792")


@pytest.mark.unit
def test_stage_panel_summary_uses_a_fixed_denominator() -> None:
    summary = summarize_reports(
        {
            8792: _report("preprior_score_centered"),
            94084: _report("all_stages_exact", scale=2.0),
        }
    )
    assert summary["particle_count"] == 2
    assert summary["stage_pass_counts"]["candidate_tuple_presence"] == {
        "passed": 2,
        "total": 2,
    }
    assert summary["stage_pass_counts"]["preprior_score_centered"] == {
        "passed": 1,
        "total": 2,
    }
    assert summary["stage_pass_counts"]["normalized_posterior"] == {
        "passed": 1,
        "total": 2,
    }
    assert summary["first_unequal_boundary_counts"] == {
        "all_stages_exact": 1,
        "preprior_score_centered": 1,
    }
    assert summary["stage_error_envelopes"]["preprior_score_centered"] == {
        "particle_count": 2,
        "worst_max_abs": 2.0e-4,
        "worst_relative_l2_over_reference": 4.0e-6,
        "total_mismatch_count": 30,
    }
