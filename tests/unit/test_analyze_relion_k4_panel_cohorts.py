import copy

import pytest

from scripts import analyze_relion_k4_panel_cohorts as analyzer


def _reports():
    cohorts = [
        "persistent_class_mismatch",
        "corrected_by_relion_cuda",
        "introduced_by_relion_cuda",
    ]
    threeway_rows = []
    repeatability_rows = []
    for identity, cohort in enumerate(cohorts * 4):
        candidate_count = identity + 1
        host_energy = 20.0 if cohort == "corrected_by_relion_cuda" else 10.0
        candidate_energy = 18.0 if cohort == "corrected_by_relion_cuda" else 11.0
        threeway_rows.append(
            {
                "zero_based_identity_row": identity,
                "rlnImageName": f"{identity + 1}@particles.mrcs",
                "cohort": cohort,
                "active_candidate_count": candidate_count,
                "winner": {
                    "winner_defined": True,
                    "host_matches_relion": True,
                    "recovar_relion_cuda_matches_relion": True,
                },
                "data_score_residual": {
                    backend: {
                        "candidate_count": candidate_count,
                        "residual_energy": energy,
                    }
                    for backend, energy in (
                        ("host_numpy", host_energy),
                        ("relion_cuda", candidate_energy),
                    )
                },
                "combined_score_residual": {
                    backend: {
                        "candidate_count": candidate_count,
                        "residual_energy": energy,
                    }
                    for backend, energy in (
                        ("host_numpy", host_energy),
                        ("relion_cuda", candidate_energy),
                    )
                },
            }
        )
        repeatability_rows.append(
            {
                "zero_based_identity_row": identity,
                "rlnImageName": f"{identity + 1}@particles.mrcs",
                "cohort": cohort,
                "active_candidate_count": candidate_count,
                "centered_raw_diff2_repeatability": {
                    "candidate_count": candidate_count,
                    "residual_energy": 3.0,
                },
                "centered_combined_repeatability": {
                    "candidate_count": candidate_count,
                    "residual_energy": 3.0,
                },
            }
        )
    threeway = {
        "schema": analyzer.THREEWAY_SCHEMA,
        "status": "complete",
        "classification": analyzer.WITHIN_FLOOR,
        "scorecard_change_admissible": False,
        "quality_metric_policy": {"correlation_computed": False},
        "scope": {
            "physical_iteration": 10,
            "class_one_based": 2,
            "current_size": 74,
            "target_count": 12,
            "winner_evaluable_target_count": 12,
            "host_winner_matches_relion_count": 12,
            "relion_cuda_winner_matches_relion_count": 12,
        },
        "targets": threeway_rows,
    }
    repeatability = {
        "schema": analyzer.REPEATABILITY_SCHEMA,
        "status": "complete",
        "scorecard_change_admissible": False,
        "scope": {
            "physical_iteration": 10,
            "class_one_based": 2,
            "target_count": 12,
            "winners_exact_all": True,
        },
        "targets": repeatability_rows,
    }
    return threeway, repeatability


def test_cohort_effect_is_stratified_against_its_own_floor():
    threeway, repeatability = _reports()

    report = analyzer.analyze(threeway, repeatability)

    assert report["classification"] == "heterogeneous_cohort_effect_without_robust_reduction"
    assert (
        report["cohorts"]["corrected_by_relion_cuda"]["classification"]
        == analyzer.WITHIN_FLOOR
    )
    assert (
        report["cohorts"]["persistent_class_mismatch"]["classification"]
        == analyzer.NO_UNIFORM_REDUCTION
    )
    corrected = report["cohorts"]["corrected_by_relion_cuda"]["families"]["data"]
    assert corrected["improvement_energy"] == 8.0
    assert corrected["capture_repeatability_residual_energy"] == 12.0
    assert corrected["improvement_to_repeatability_energy_ratio"] == 2.0 / 3.0
    assert report["scorecard_change_admissible"] is False


def test_identity_sets_must_match():
    threeway, repeatability = _reports()
    repeatability["targets"][0]["zero_based_identity_row"] = 999

    with pytest.raises(ValueError, match="identity sets differ"):
        analyzer.analyze(threeway, repeatability)


def test_predeclared_cohort_membership_must_match_by_identity():
    threeway, repeatability = _reports()
    repeatability["targets"][0]["cohort"] = "corrected_by_relion_cuda"

    with pytest.raises(ValueError, match="cohort or image identity changed"):
        analyzer.analyze(threeway, repeatability)


def test_candidate_counts_must_match():
    threeway, repeatability = _reports()
    repeatability["targets"][0]["active_candidate_count"] += 1

    with pytest.raises(ValueError, match="active candidate count changed"):
        analyzer.analyze(threeway, repeatability)


def test_input_cannot_authorize_scorecard_change():
    threeway, repeatability = _reports()
    threeway["scorecard_change_admissible"] = True

    with pytest.raises(ValueError, match="permits a scorecard change"):
        analyzer.analyze(threeway, repeatability)


def test_winner_closure_must_hold_for_every_target():
    threeway, repeatability = _reports()
    changed = copy.deepcopy(threeway)
    changed["targets"][0]["winner"]["host_matches_relion"] = False

    with pytest.raises(ValueError, match="winner closure changed"):
        analyzer.analyze(changed, repeatability)
