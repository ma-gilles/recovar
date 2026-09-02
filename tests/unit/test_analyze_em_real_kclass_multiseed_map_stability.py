import copy

import numpy as np
import pytest

from scripts import analyze_em_real_kclass_multiseed_map_stability as analyzer


def _row(values):
    return {"matched_per_class_fsc_auc": values}


def test_summarize_separation_requires_every_class_to_clear_seed_variability() -> None:
    same = [
        _row([0.95, 0.96, 0.85, 0.94]),
        _row([0.94, 0.97, 0.84, 0.95]),
        _row([0.96, 0.95, 0.87, 0.93]),
    ]
    within = [
        _row([0.82, 0.88, 0.64, 0.81]),
        _row([0.80, 0.84, 0.54, 0.83]),
        _row([0.81, 0.86, 0.57, 0.82]),
    ]

    summary = analyzer.summarize_separation(same, within)

    assert summary["separated_for_every_class"] is True
    assert [row["class"] for row in summary["per_class"]] == [1, 2, 3, 4]
    assert summary["per_class"][2]["same_seed_cross_engine"]["min"] == pytest.approx(0.84)
    assert summary["per_class"][2]["within_engine_cross_seed"]["max"] == pytest.approx(0.64)


def test_summarize_separation_fails_if_one_class_overlaps() -> None:
    same = [_row([0.95, 0.96, 0.63, 0.94])] * 3
    within = [_row([0.82, 0.88, 0.64, 0.81])] * 3

    summary = analyzer.summarize_separation(same, within)

    assert summary["separated_for_every_class"] is False
    assert summary["per_class"][2]["same_seed_min_exceeds_within_engine_cross_seed_max"] is False


def test_canonical_order_rejects_duplicate_class_mapping() -> None:
    audit = {
        "class_matching": {
            "source_class_for_anchor": {"relion_half1": [1, 2, 2, 4]}
        }
    }

    with pytest.raises(ValueError, match="not a K-class permutation"):
        analyzer._canonical_order(audit, "relion_half1")


def test_identity_alignment_preserves_f32_values() -> None:
    maps = [np.arange(27, dtype=np.float32).reshape(3, 3, 3)]

    aligned = analyzer._alignment_transform(
        maps,
        {"identity_anchor": True},
        interpolation_order=1,
    )

    np.testing.assert_array_equal(aligned[0], maps[0])
    assert aligned[0].dtype == np.float32


def _match_row(*, seed=42001, engine=None, anchor_seed=None, source_seed=None):
    row = {
        "source_class_for_anchor": [1, 2, 3, 4],
        "matched_per_class_fsc_auc": [0.92, 0.94, 0.84, 0.93],
        "matched_mean_fsc_auc": 0.9075,
        "pairwise_fsc_auc": np.diag([0.92, 0.94, 0.84, 0.93]).tolist(),
        "objective_margin": 1.0,
        "relative_objective_margin": 0.5,
        "exact_optimum_count": 1,
    }
    if engine is None:
        row["seed"] = seed
    else:
        row.update(engine=engine, anchor_seed=anchor_seed, source_seed=source_seed)
        row["matched_per_class_fsc_auc"] = [0.80, 0.86, 0.63, 0.81]
        row["matched_mean_fsc_auc"] = 0.775
        row["pairwise_fsc_auc"] = np.diag([0.80, 0.86, 0.63, 0.81]).tolist()
    return row


def _alignment_row(seed):
    if seed == 42001:
        return {"seed": seed, "identity_anchor": True}
    return {
        "seed": seed,
        "fit_box_size": 65,
        "fit_max_shell_full_box": 32,
        "interpolation_order": 1,
        "optimizer_success": True,
        "optimizer_status": 0,
        "optimizer_message": "Optimization terminated successfully.",
        "refine_healpix_orders": [2],
        "seed_healpix_order": 1,
        "seed_rotation_matrix": np.eye(3).tolist(),
        "seed_source": "identity_augmented_RELION_HEALPix_grid",
        "phase_correlation_seed_shift_fit_zyx": [0.0, 0.1, 0.0],
        "rotation_matrix_recovar_to_relion": np.eye(3).tolist(),
        "translation_recovar_to_relion_zyx": [0.0, 0.0, 0.0],
        "optimizer_function_evaluations": 100,
    }


def _reproduction_report():
    same = [_match_row(seed=seed) for seed in (42001, 42002, 42003)]
    within = [
        _match_row(
            engine=engine,
            anchor_seed=left_seed,
            source_seed=right_seed,
        )
        for engine in ("relion", "recovar")
        for left_seed, right_seed in ((42001, 42002), (42001, 42003), (42002, 42003))
    ]
    separation = analyzer.summarize_separation(same, within)
    return {
        "schema": analyzer.SCHEMA,
        "status": "complete",
        "admission_status": "DIAGNOSTIC_ONLY_PER_SEED_GATES_RETAINED",
        "accepted_result": False,
        "metric": {"name": "full_unmasked_non_dc_fsc_auc"},
        "expected_seeds": [42001, 42002, 42003],
        "anchor_seed": 42001,
        "contract": {"dataset": "EMPIAR-10076"},
        "source_assignment_observations": {"separated": True},
        "audits": [{"seed": 42001, "sha256": "audit"}],
        "map_artifacts": [{"sha256": "map"}],
        "seed_alignment": {
            engine: [_alignment_row(seed) for seed in (42001, 42002, 42003)]
            for engine in ("relion", "recovar")
        },
        "same_seed_cross_engine": same,
        "within_engine_cross_seed": within,
        "separation": separation,
        "observations": {
            "same_seed_cross_engine_min_exceeds_cross_seed_max_for_every_class": True
        },
    }


def test_verify_reproduction_allows_small_continuous_fit_and_fsc_drift() -> None:
    reference = _reproduction_report()
    candidate = copy.deepcopy(reference)
    candidate["seed_alignment"]["relion"][1]["optimizer_function_evaluations"] = 412
    candidate["seed_alignment"]["relion"][1]["translation_recovar_to_relion_zyx"] = [
        0.01,
        -0.02,
        0.03,
    ]
    candidate["within_engine_cross_seed"][0]["matched_per_class_fsc_auc"][0] += 0.003
    candidate["within_engine_cross_seed"][0]["pairwise_fsc_auc"][0][0] += 0.003
    candidate["within_engine_cross_seed"][0]["matched_mean_fsc_auc"] += 0.00075
    candidate["separation"] = analyzer.summarize_separation(
        candidate["same_seed_cross_engine"],
        candidate["within_engine_cross_seed"],
    )

    verification = analyzer.verify_reproduction(reference, candidate)

    assert verification["status"] == "PASS"
    assert verification["violations"] == []
    assert verification["measurements"][
        "max_within_engine_cross_seed_fsc_auc_absolute_delta"
    ] == pytest.approx(0.003)
    assert verification["measurements"][
        "candidate_minimum_per_class_separation_margin"
    ] == pytest.approx(0.08)


def test_verify_reproduction_rejects_changed_class_permutation() -> None:
    reference = _reproduction_report()
    candidate = copy.deepcopy(reference)
    candidate["within_engine_cross_seed"][0]["source_class_for_anchor"] = [2, 1, 3, 4]

    verification = analyzer.verify_reproduction(reference, candidate)

    assert verification["status"] == "FAIL"
    assert verification["checks"][
        "within_engine_cross_seed_class_assignments_exact"
    ] is False


def test_verify_reproduction_rejects_fsc_drift_beyond_tolerance() -> None:
    reference = _reproduction_report()
    candidate = copy.deepcopy(reference)
    candidate["within_engine_cross_seed"][0]["matched_per_class_fsc_auc"][0] += 0.006
    candidate["within_engine_cross_seed"][0]["pairwise_fsc_auc"][0][0] += 0.006
    candidate["within_engine_cross_seed"][0]["matched_mean_fsc_auc"] += 0.0015
    candidate["separation"] = analyzer.summarize_separation(
        candidate["same_seed_cross_engine"],
        candidate["within_engine_cross_seed"],
    )

    verification = analyzer.verify_reproduction(reference, candidate)

    assert verification["status"] == "FAIL"
    assert verification["checks"][
        "within_engine_cross_seed_fsc_auc_within_tolerance"
    ] is False


def test_verify_reproduction_rejects_lost_scientific_separation() -> None:
    reference = _reproduction_report()
    candidate = copy.deepcopy(reference)
    for row in candidate["same_seed_cross_engine"]:
        row["matched_per_class_fsc_auc"][1] = 0.89
        row["pairwise_fsc_auc"][1][1] = 0.89
        row["matched_mean_fsc_auc"] = float(np.mean(row["matched_per_class_fsc_auc"]))
    candidate["separation"] = analyzer.summarize_separation(
        candidate["same_seed_cross_engine"],
        candidate["within_engine_cross_seed"],
    )

    verification = analyzer.verify_reproduction(
        reference,
        candidate,
        max_fsc_auc_abs_delta=1.0,
    )

    assert verification["status"] == "FAIL"
    assert verification["checks"]["all_classes_remain_separated"] is True
    assert verification["checks"]["minimum_per_class_separation_margin_retained"] is False
