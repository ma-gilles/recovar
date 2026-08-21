from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.reclassify_em_k1_case7_bpref_support_boundary import (
    EXACT_COMPARISONS,
    classify_particle,
    reclassify,
)


def _particle(index: int, *, exact: bool = True) -> dict:
    return {
        "original_index_zero_based": index,
        "stack_index_one_based": index + 1,
        "image_identity": f"{index + 1}@particles.mrcs",
        "accepted_hypothesis_count_native": 4,
        "accepted_hypothesis_count_recovar": 4,
        "support_intersection_count": 4,
        "support_union_count": 4,
        "support_exact": True,
        "capture_self_closes": True,
        "same_posterior_operands_close": True,
        "sequential_summary_closes": True,
        "highest_summary_closes": True,
        "classification": "particle_prescatter_boundary_closes",
        "comparisons": {name: {"exact_equal": exact} for name in EXACT_COMPARISONS},
    }


def _report(*, exact: bool = True) -> dict:
    return {
        "schema": "recovar.em_k1_case7_bpref_factor_boundary.v1",
        "status": "complete",
        "production_authorized": False,
        "fixed_scorecard_changed": False,
        "relative_l2_bound": 1e-5,
        "classification": "fixed_panel_particle_prescatter_boundary_closes",
        "particles": [_particle(index, exact=exact) for index in range(10)],
    }


@pytest.mark.unit
def test_exact_support_and_comparisons_close_exactly() -> None:
    corrected = reclassify(_report())

    assert corrected["classification"] == "fixed_panel_particle_prescatter_boundary_exactly_closes"
    assert corrected["support_exact_count"] == 10
    assert corrected["next_boundary"] == "accumulator_destination_and_inter_particle_reduction"


@pytest.mark.unit
def test_relative_l2_closure_is_not_labeled_exact() -> None:
    corrected = reclassify(_report(exact=False))

    assert corrected["classification"] == "fixed_panel_particle_prescatter_boundary_closes_within_relative_l2_bound"
    assert corrected["next_boundary"].startswith("exactness_residual_then_")
    assert all(not particle["selected_comparisons_exact"] for particle in corrected["particles"])


@pytest.mark.unit
def test_support_mismatch_precedes_apparent_operand_closure() -> None:
    report = _report()
    report["particles"][3].update(
        {
            "support_exact": False,
            "support_intersection_count": 3,
            "support_union_count": 5,
        }
    )
    corrected = reclassify(report)

    assert classify_particle(report["particles"][3]) == "significant_support_identity_mismatch_before_bpref_operands"
    assert corrected["classification"] == "fixed_panel_significant_support_identity_mismatch"
    assert corrected["support_exact_count"] == 9
    assert corrected["next_boundary"] == "posterior_normalization_and_significance"


@pytest.mark.unit
def test_capture_self_failure_precedes_support_mismatch() -> None:
    report = _report()
    report["particles"][2]["support_exact"] = False
    report["particles"][7]["capture_self_closes"] = False
    corrected = reclassify(report)

    assert corrected["classification"] == "fixed_panel_relion_capture_self_mismatch"
    assert corrected["next_boundary"] == "repair_or_requalify_native_capture"


@pytest.mark.unit
def test_rejects_nonfixed_particle_count() -> None:
    report = _report()
    report["particles"].pop()

    with pytest.raises(ValueError, match="fixed ten-particle panel changed"):
        reclassify(report)


@pytest.mark.unit
def test_does_not_mutate_original_report() -> None:
    report = _report()
    before = deepcopy(report)

    reclassify(report)

    assert report == before
