import pytest

from scripts.analyze_k1_partial_fine_panel import (
    _expected_stacks_from_selection,
    _parse_expected_stacks,
    _selected_capture_stacks,
    stage_outcomes,
    summarize_reports,
)


def _report(*, preprior_exact: bool, support_exact: bool = True):
    return {
        "rotation_topology": {"native_count": 2, "recovar_count": 2, "common_count": 2},
        "active_tuple_topology": {"native_count": 8, "recovar_count": 8, "common_count": 8},
        "active_tuple_sequence": {"exact": True},
        "production_boundary": {
            "preprior_score_centered": {"exact_equal": preprior_exact},
            "orientation_log_prior": {"exact_equal": True},
            "translation_log_prior": {"exact_equal": True},
            "combined_log_weight_centered": {"exact_equal": preprior_exact},
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
def test_expected_stacks_from_same_process_selection(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        """{
          "schema": "recovar.em.k1_same_process_final_boundary_panel.v1",
          "targets": [
            {"stack_index_one_based": 79, "expected_mpi_rank": 1},
            {"stack_index_one_based": 469, "expected_mpi_rank": 2}
          ]
        }"""
    )
    assert _expected_stacks_from_selection(selection) == {79, 469}


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
    assert summary["stage_pass_counts"]["active_tuple_sequence"] == {"passed": 2, "total": 2}
    assert summary["stage_pass_counts"]["preprior_score_centered"] == {"passed": 1, "total": 2}
    assert summary["first_unequal_boundary_counts"] == {
        "fine_significant_support": 1,
        "preprior_score_centered": 1,
    }


@pytest.mark.unit
def test_selected_capture_stacks_accepts_explicit_recovar_subset():
    assert _selected_capture_stacks(
        factor_stacks={79, 469, 2498},
        fine_score_stacks={79, 469, 2498},
        recovar_stacks={79},
        expected_stacks={79},
    ) == {79}


@pytest.mark.unit
def test_selected_capture_stacks_remains_fail_closed():
    with pytest.raises(ValueError, match="factor/fine-score"):
        _selected_capture_stacks(
            factor_stacks={79, 469},
            fine_score_stacks={79},
            recovar_stacks={79},
            expected_stacks={79},
        )
    with pytest.raises(ValueError, match="expected RECOVAR"):
        _selected_capture_stacks(
            factor_stacks={79, 469},
            fine_score_stacks={79, 469},
            recovar_stacks={79},
            expected_stacks={469},
        )
    with pytest.raises(ValueError, match="capture stack sets"):
        _selected_capture_stacks(
            factor_stacks={79, 469},
            fine_score_stacks={79, 469},
            recovar_stacks={79},
            expected_stacks=None,
        )
