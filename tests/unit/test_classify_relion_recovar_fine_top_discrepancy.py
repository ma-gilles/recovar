import copy

import pytest

from scripts import classify_relion_recovar_fine_top_discrepancy as classifier


def _inertness():
    return {
        "schema": "em_relion_iteration1_particle_state_inertness_v1",
        "status": "pass",
        "particle_count": 100_000,
        "target_original_index": 65_070,
        "fields": {
            field: {"exact": True, "mismatch_count": 0, "max_abs": 0.0}
            for field in classifier._REQUIRED_INERTNESS_FIELDS
        },
    }


def _comparison():
    return {
        "match_mode": "matrix",
        "reconstruction_only": False,
        "recovar_original_index": 65_070,
        "recovar_current_size": 56,
        "relion_candidate_count": 64,
        "recovar_candidate_count": 64,
        "common_candidate_count": 64,
        "relion_top_key": [7, 3],
        "recovar_top_key": [7, 2],
        "cross_top_candidate_details": [
            {
                "key": [7, 3],
                "relion": {"score_pre_prior": 0.25},
                "recovar": {"score_pre_prior": 0.5},
            },
            {
                "key": [7, 2],
                "relion": {"score_pre_prior": 0.25},
                "recovar": {"score_pre_prior": 0.5},
            },
        ],
    }


def _classify(comparison=None, inertness=None):
    return classifier.classify(
        _comparison() if comparison is None else comparison,
        _inertness() if inertness is None else inertness,
        expected_original_index=65_070,
        expected_current_size=56,
        expected_particle_count=100_000,
    )


def test_exact_raw_score_ties_localize_discrepancy_to_candidate_order():
    report = _classify()

    assert report["classification"] == "compact_candidate_tie_order"
    assert report["exact_raw_pre_prior_tie"] == {"relion": True, "recovar": True}
    assert report["scorecard_change_admissible"] is False


def test_non_tie_localizes_discrepancy_to_score_arithmetic():
    comparison = _comparison()
    comparison["cross_top_candidate_details"][0]["relion"]["score_pre_prior"] = 0.25000000000000006

    report = _classify(comparison=comparison)

    assert report["classification"] == "fine_score_arithmetic"
    assert report["exact_raw_pre_prior_tie"] == {"relion": False, "recovar": True}
    assert report["raw_pre_prior_score_delta_at_relion_top_minus_at_recovar_top"]["relion"] > 0.0


def test_classifier_rejects_nonexact_capture_inertness():
    inertness = _inertness()
    inertness["fields"]["rlnAnglePsi"] = {"exact": False, "mismatch_count": 1, "max_abs": 0.01}

    with pytest.raises(ValueError, match="rlnAnglePsi is not exact"):
        _classify(inertness=inertness)


def test_classifier_rejects_details_that_do_not_cover_both_winners():
    comparison = copy.deepcopy(_comparison())
    comparison["cross_top_candidate_details"][1]["key"] = [8, 2]

    with pytest.raises(ValueError, match="do not match the two engine winners"):
        _classify(comparison=comparison)
