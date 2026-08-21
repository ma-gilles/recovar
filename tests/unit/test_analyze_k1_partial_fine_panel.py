import pytest

from scripts.analyze_k1_partial_fine_panel import (
    _parse_expected_stacks,
    stage_outcomes,
    summarize_reports,
)


def _report(*, preprior_exact: bool, support_exact: bool = True):
    return {
        "rotation_topology": {"native_count": 2, "recovar_count": 2, "common_count": 2},
        "active_tuple_topology": {"native_count": 8, "recovar_count": 8, "common_count": 8},
        "production_boundary": {
            "preprior_score_centered": {"exact_equal": preprior_exact},
            "orientation_log_prior": {"exact_equal": True},
            "translation_log_prior": {"exact_equal": True},
            "posterior_on_common_native_normalization": {"exact_equal": preprior_exact},
            "fine_significant_support": {"exact": support_exact},
        },
    }


@pytest.mark.unit
def test_expected_stacks_accepts_submission_safe_colons():
    assert _parse_expected_stacks("79:469:2498") == {79, 469, 2498}
    with pytest.raises(ValueError, match="unique"):
        _parse_expected_stacks("79,79")


@pytest.mark.unit
def test_stage_outcomes_and_fixed_denominator_summary():
    exact = stage_outcomes(_report(preprior_exact=True))
    assert all(exact.values())

    summary = summarize_reports(
        {
            79: _report(preprior_exact=False),
            469: _report(preprior_exact=True, support_exact=False),
        }
    )
    assert summary["particle_count"] == 2
    assert summary["stage_pass_counts"]["rotation_topology"] == {"passed": 2, "total": 2}
    assert summary["stage_pass_counts"]["preprior_score_centered"] == {"passed": 1, "total": 2}
    assert summary["first_unequal_boundary_counts"] == {
        "fine_significant_support": 1,
        "preprior_score_centered": 1,
    }
