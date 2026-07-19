from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recovar.data_io.starfile import write_star
from scripts import audit_em_particle_state_distribution as auditor


def _write_star(path: Path, rows: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_star(str(path), pd.DataFrame(rows))
    return path.resolve()


def _write_scalar_star(path: Path, values: dict[str, object]) -> Path:
    path.write_text("data_general\n\n" + "\n".join(f"_{key} {value}" for key, value in values.items()) + "\n")
    return path.resolve()


def _write_k1_model_star(
    path: Path,
    *,
    current_resolution: float,
    estimated_resolution: float,
    estimated_label: str = "_rlnEstimatedResolution",
    average_pmax: float | None = None,
) -> Path:
    average_pmax_line = "" if average_pmax is None else f"_rlnAveragePmax {average_pmax}\n"
    path.write_text(
        "data_model_general\n\n"
        "_rlnCurrentImageSize 64\n"
        f"_rlnCurrentResolution {current_resolution}\n"
        f"{average_pmax_line}\n"
        "data_model_classes\n\n"
        "loop_\n"
        "_rlnReferenceImage #1\n"
        f"{estimated_label} #2\n"
        f"class001.mrc {estimated_resolution}\n"
    )
    return path.resolve()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    n_images = 12
    names = np.asarray([f"{index + 1:06d}@particles.mrcs" for index in range(n_images)])
    halves = np.asarray([1, 2] * 6, dtype=np.int64)
    source = _write_star(
        tmp_path / "particles.star",
        {
            "_rlnImageName": names,
            "_rlnRandomSubset": halves,
            "_rlnDefocusU": np.linspace(10_000, 20_000, n_images),
            "_rlnDefocusV": np.linspace(10_200, 20_200, n_images),
        },
    )

    rel_pmax = np.asarray([0.2, 0.55, 0.75, 0.92, 0.96, 0.98, 0.991, 0.995, 0.997, 0.998, 0.999, 1.0])
    rel_support = np.asarray([100, 80, 60, 40, 30, 20, 12, 10, 8, 6, 4, 2])
    rel_eulers = np.column_stack([np.arange(n_images), np.full(n_images, 30.0), np.zeros(n_images)])
    rec_eulers = rel_eulers.copy()
    rec_eulers[0, 0] += 1.0
    rec_trans_px = np.column_stack([np.linspace(0, 1, n_images), np.linspace(1, 0, n_images)])
    rec_classes = np.asarray([0, 1, 2, 3] * 3, dtype=np.int32)
    rec_pmax = rel_pmax + np.linspace(-0.001, 0.001, n_images)
    rec_support = rel_support + np.asarray([0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1])
    half1 = np.flatnonzero(halves == 1)
    half2 = np.flatnonzero(halves == 2)
    results = tmp_path / "refinement_results.npz"
    np.savez(
        results,
        n_images=n_images,
        voxel_size=2.0,
        half1_indices=half1,
        half2_indices=half2,
        pmax_per_image_by_image_iter_000=rec_pmax,
        sig_counts_by_image_iter_000=rec_support,
        best_rotation_eulers_by_image_iter_000=rec_eulers,
        best_translations_by_image_iter_000=rec_trans_px,
        class_assignments_by_image_iter_000=rec_classes,
    )

    permutation = np.asarray([7, 0, 11, 4, 2, 9, 1, 8, 5, 3, 10, 6])
    relion = _write_star(
        tmp_path / "run_it001_data.star",
        {
            "_rlnImageName": names[permutation],
            "_rlnMaxValueProbDistribution": rel_pmax[permutation],
            "_rlnNrOfSignificantSamples": rel_support[permutation],
            "_rlnAngleRot": rel_eulers[permutation, 0],
            "_rlnAngleTilt": rel_eulers[permutation, 1],
            "_rlnAnglePsi": rel_eulers[permutation, 2],
            "_rlnOriginXAngst": (2.0 * rec_trans_px[permutation, 0]),
            "_rlnOriginYAngst": (2.0 * rec_trans_px[permutation, 1]),
            "_rlnClassNumber": (rec_classes[permutation] + 1),
        },
    )
    control = _write_star(
        tmp_path / "control_it001_data.star",
        {
            "_rlnImageName": names[::-1],
            "_rlnMaxValueProbDistribution": (rel_pmax + 1e-5)[::-1],
            "_rlnNrOfSignificantSamples": rel_support[::-1],
            "_rlnAngleRot": rel_eulers[::-1, 0],
            "_rlnAngleTilt": rel_eulers[::-1, 1],
            "_rlnAnglePsi": rel_eulers[::-1, 2],
            "_rlnOriginXAngst": (2.0 * rec_trans_px[::-1, 0]),
            "_rlnOriginYAngst": (2.0 * rec_trans_px[::-1, 1]),
            "_rlnClassNumber": (rec_classes[::-1] + 1),
        },
    )
    return results.resolve(), source, relion, control


@pytest.mark.unit
def test_audit_aligns_shuffled_rows_and_reports_full_distribution(tmp_path):
    results, source, relion, control = _fixture(tmp_path)

    report = auditor.audit(
        recovar_results=results,
        recovar_particles_star=source,
        relion_stars={1: relion},
        control_stars={1: control},
    )

    assert report["schema"] == auditor.SCHEMA
    assert report["status"] == "complete"
    assert report["n_images"] == 12
    assert len(report["iterations"]) == 1
    row = report["iterations"][0]
    assert (row["recovar_iteration"], row["relion_iteration"]) == (0, 1)
    overall = row["recovar_vs_relion"]
    assert overall["pmax"]["signed"]["mean"] == pytest.approx(0.0, abs=1e-15)
    assert overall["significant_support"]["different_count"] == 6
    assert overall["angular_error_deg"]["max"] == pytest.approx(1.0, abs=1e-8)
    assert overall["translation_error"]["units"] == "angstrom"
    assert overall["translation_error"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert overall["class_assignment"]["agreement"] == pytest.approx(1.0)
    assert set(row["subgroups"]) == {
        "half",
        "relion_pmax_bin",
        "relion_support_quantile",
        "defocus_quantile_angstrom",
    }
    assert row["subgroups"]["half"]["half1"]["n"] == 6
    control_summary = row["relion_control_vs_relion"]
    assert control_summary["pmax"]["absolute"]["mean"] == pytest.approx(1e-5)
    relative = row["recovar_vs_relion_relative_to_control"]["metrics"]
    expected_mean_ratio = overall["pmax"]["absolute"]["mean"] / 1e-5
    assert relative["pmax"]["recovar_to_control_absolute_error_ratio"]["mean"] == pytest.approx(
        expected_mean_ratio
    )
    assert relative["pmax"]["count_recovar_abs_error_gt_control_max"] == 12
    assert relative["significant_support"]["recovar_to_control_absolute_error_ratio"]["max"] is None
    assert set(relative["significant_support"]["ratio_undefined_zero_control_statistics"]) == {
        "mean",
        "p95",
        "p99",
        "max",
    }
    assert relative["significant_support"]["count_recovar_abs_error_gt_control_max"] == 6
    assert "No correlation computed" in report["quality_metric_policy"]


@pytest.mark.unit
def test_cross_iteration_tail_enrichment_uses_exact_aligned_state_and_is_diagnostic(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)
    source_table = auditor._particle_table(source)
    identities = auditor._identity_array(source_table, source=source)
    relion_state, _ = auditor._load_relion_state(relion, identities)
    with np.load(results, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}

    # At iteration 1, support differs for identity rows 1,3,5,7,9,11. Make
    # row 1 the exact top-5% Pmax exposure. At iteration 2, rows 1,5,8 form
    # the >0.1-degree pose tail.
    previous_pmax = np.asarray(relion_state["pmax"], dtype=np.float64).copy()
    previous_pmax[1] += 0.5
    data["pmax_per_image_by_image_iter_000"] = previous_pmax
    current_eulers = np.asarray(relion_state["eulers"], dtype=np.float64).copy()
    current_eulers[[1, 5, 8], 0] += 1.0
    data.update(
        pmax_per_image_by_image_iter_001=np.asarray(relion_state["pmax"]),
        sig_counts_by_image_iter_001=np.asarray(relion_state["support"]),
        best_rotation_eulers_by_image_iter_001=current_eulers,
    )
    np.savez(results, **data)
    relion_it002 = tmp_path / "run_it002_data.star"
    relion_it002.write_text(relion.read_text())
    arrays: dict[str, np.ndarray] = {}

    report = auditor.audit(
        recovar_results=results,
        recovar_particles_star=source,
        relion_stars={1: relion, 2: relion_it002},
        artifact_arrays=arrays,
    )

    section = report["cross_iteration_tail_enrichment"]
    assert section["diagnostic_only"] is True
    assert section["quality_gate"].startswith("none")
    assert section["correlation"] == "not computed"
    boundary = section["boundaries"][0]
    support = boundary["significant_support_count_mismatch_at_t"]
    assert support["contingency"] == {
        "exposure_and_next_pose_tail": 2,
        "exposure_only": 4,
        "next_pose_tail_only": 1,
        "neither": 5,
    }
    assert support["conditional_rates"]["next_pose_tail_given_exposure"] == pytest.approx(2 / 6)
    assert support["conditional_rates"]["next_pose_tail_without_exposure"] == pytest.approx(1 / 6)
    assert support["enrichment_vs_unexposed"] == pytest.approx(2.0)
    assert support["next_pose_tail_capture_fraction"] == pytest.approx(2 / 3)
    pmax = boundary["top_5pct_absolute_pmax_delta_at_t"]
    assert pmax["selection"]["selected_count"] == 1
    assert pmax["contingency"]["exposure_and_next_pose_tail"] == 1
    assert pmax["enrichment_vs_unexposed"] == pytest.approx(5.5)
    assert pmax["next_pose_tail_capture_fraction"] == pytest.approx(1 / 3)
    np.testing.assert_array_equal(
        arrays["it001_to_it002_support_mismatch_at_t"],
        np.asarray([False, True, False, True, False, True, False, True, False, True, False, True]),
    )
    assert np.flatnonzero(arrays["it001_to_it002_top5_abs_pmax_delta_at_t"]).tolist() == [1]
    assert np.flatnonzero(arrays["it001_to_it002_pose_tail_at_t_plus_1"]).tolist() == [1, 5, 8]


@pytest.mark.unit
def test_cross_iteration_tail_enrichment_names_zero_denominators():
    empty_exposure = auditor._binary_tail_enrichment(
        np.zeros(4, dtype=bool), np.zeros(4, dtype=bool)
    )

    assert empty_exposure["conditional_rates"]["next_pose_tail_given_exposure"] is None
    assert empty_exposure["conditional_rates"]["next_pose_tail_without_exposure"] == 0.0
    assert empty_exposure["enrichment_vs_unexposed"] is None
    assert empty_exposure["next_pose_tail_capture_fraction"] is None
    assert empty_exposure["undefined_zero_denominators"] == [
        "pose_tail_rate_given_exposure: exposure_count=0",
        "tail_capture_fraction: pose_tail_count=0",
        "enrichment: pose_tail_rate_without_exposure=0",
    ]

    no_complement = auditor._binary_tail_enrichment(
        np.ones(4, dtype=bool), np.asarray([True, False, False, False])
    )
    assert no_complement["conditional_rates"]["next_pose_tail_without_exposure"] is None
    assert no_complement["enrichment_vs_unexposed"] is None
    assert "pose_tail_rate_without_exposure: unexposed_count=0" in no_complement[
        "undefined_zero_denominators"
    ]


@pytest.mark.unit
def test_class_agreement_is_hungarian_matched_and_retains_raw_confusion():
    recovar_zero_based = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    relion_permuted = np.asarray([3, 3, 2, 2, 4, 4, 1, 1])

    summary = auditor._class_summary(recovar_zero_based, relion_permuted)

    assert summary["agreement"] == pytest.approx(1.0)
    assert summary["agreement_count"] == 8
    assert summary["raw_agreement"] == pytest.approx(0.25)
    assert summary["raw_agreement_count"] == 2
    assert summary["confusion_rows_recovar_cols_relion"] == [
        [0, 0, 2, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 2],
        [2, 0, 0, 0],
    ]
    assert summary["hungarian_recovar_to_relion"] == [
        {"recovar_class": 1, "relion_class": 3},
        {"recovar_class": 2, "relion_class": 2},
        {"recovar_class": 3, "relion_class": 4},
        {"recovar_class": 4, "relion_class": 1},
    ]


@pytest.mark.unit
def test_class_ids_are_converted_from_zero_based_when_class_zero_is_absent():
    recovar_zero_based = np.asarray([1, 1, 2, 2])
    relion_one_based = np.asarray([2, 2, 3, 3])

    summary = auditor._class_summary(recovar_zero_based, relion_one_based)

    assert summary["raw_agreement"] == pytest.approx(1.0)
    assert summary["labels"] == [2, 3]
    assert summary["hungarian_recovar_to_relion"] == [
        {"recovar_class": 2, "relion_class": 2},
        {"recovar_class": 3, "relion_class": 3},
    ]


@pytest.mark.unit
def test_subgroup_class_agreement_uses_whole_iteration_mapping():
    recovar_zero_based = np.asarray([0, 0, 0, 1, 1])
    relion_one_based = np.asarray([1, 1, 2, 2, 2])
    overall = auditor._class_summary(recovar_zero_based, relion_one_based)
    mapping = {
        item["recovar_class"]: item["relion_class"]
        for item in overall["hungarian_recovar_to_relion"]
    }

    subgroup = auditor._class_summary(
        recovar_zero_based[2:3],
        relion_one_based[2:3],
        class_mapping=mapping,
    )

    assert overall["agreement"] == pytest.approx(0.8)
    assert subgroup["matching_scope"] == "whole_iteration"
    assert subgroup["agreement"] == pytest.approx(0.0)


@pytest.mark.unit
def test_identical_euler_arrays_have_exact_zero_angular_error():
    eulers = np.asarray([[13.2, 51.7, -108.3], [-179.0, 90.1, 179.5]])

    errors = auditor._angular_error_deg(eulers, eulers.copy())

    np.testing.assert_array_equal(errors, np.zeros(2))


@pytest.mark.unit
def test_cli_writes_versioned_json_and_has_explicit_help(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)
    output = tmp_path / "audit.json"

    status = auditor.main(
        [
            "--recovar-results",
            str(results),
            "--recovar-particles-star",
            str(source),
            "--relion-star",
            str(relion),
            "--output-json",
            str(output),
        ]
    )

    assert status == 0
    assert json.loads(output.read_text())["schema"] == "em_particle_state_distribution_audit_v1"
    arrays_path = tmp_path / "audit_arrays.npz"
    manifest_path = tmp_path / "audit.json.sha256"
    assert arrays_path.is_file()
    assert manifest_path.is_file()
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert str(arrays["schema"]) == auditor.ARRAY_SCHEMA
        assert "it001_pmax_delta" in arrays.files
        assert "it001_rotation_view_deg" in arrays.files
    manifest = manifest_path.read_text()
    assert str(output.resolve()) in manifest
    assert str(arrays_path.resolve()) in manifest
    help_text = auditor._parser().format_help()
    assert "--relion-control-star" in help_text
    assert "image identities" in help_text


@pytest.mark.unit
def test_pose_distributions_include_geodesic_view_inplane_rmse_and_threshold_fractions(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)

    report = auditor.audit(
        recovar_results=results,
        recovar_particles_star=source,
        relion_stars={1: relion},
    )

    metrics = report["iterations"][0]["recovar_vs_relion"]
    assert metrics["angular_error_deg"]["max"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["view_direction_error_deg"]["max"] == pytest.approx(0.5, abs=0.01)
    assert metrics["inplane_error_deg"]["max"] > 0.8
    assert metrics["pmax"]["absolute"]["rmse"] > 0.0
    assert set(metrics["pmax"]["absolute"]["threshold_fractions"]) == {
        "le_1e-06",
        "le_1e-05",
        "le_0.0001",
        "le_0.001",
    }


@pytest.mark.unit
def test_scalar_schedule_convergence_and_final_state_are_reported_and_can_gate(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)
    with np.load(results, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    data.update(
        current_sizes=np.asarray([64]),
        pixel_resolutions=np.asarray([12.8]),
        volume_shape=np.asarray([64, 64, 64]),
        ave_Pmax_trajectory=np.asarray([0.4321]),
        healpix_order_trajectory=np.asarray([3]),
        acc_rot_trajectory=np.asarray([2.0]),
        acc_trans_trajectory=np.asarray([1.5]),
        smallest_change_angles_trajectory=np.asarray([4.0]),
        smallest_change_offsets_trajectory=np.asarray([1.0]),
        convergence_iteration=np.asarray(1),
        convergence_has_converged=np.asarray(True),
        final_all_data_ran=np.asarray(True),
        pmax_final_all_data_by_image=data["pmax_per_image_by_image_iter_000"],
        best_rotation_eulers_final_all_data_by_image=data["best_rotation_eulers_by_image_iter_000"],
        best_translations_final_all_data_by_image=data["best_translations_by_image_iter_000"],
    )
    np.savez(results, **data)
    _write_k1_model_star(
        tmp_path / "run_it001_half1_model.star",
        current_resolution=12.0,
        estimated_resolution=10.0,
        average_pmax=0.4321,
    )
    _write_scalar_star(tmp_path / "run_it001_sampling.star", {"rlnHealpixOrder": 3})
    _write_scalar_star(
        tmp_path / "run_it001_optimiser.star",
        {
            "rlnCurrentIteration": 1,
            "rlnOverallAccuracyRotations": 2.0,
            "rlnOverallAccuracyTranslationsAngst": 1.5,
            "rlnChangesOptimalOrientations": 4.0,
            "rlnChangesOptimalOffsets": 1.0,
            "rlnHasConverged": 0,
        },
    )
    final_star = tmp_path / "run_data.star"
    final_star.write_text(relion.read_text())
    _write_scalar_star(tmp_path / "run_optimiser.star", {"rlnCurrentIteration": -1, "rlnHasConverged": 1})

    report = auditor.audit(
        recovar_results=results,
        recovar_particles_star=source,
        relion_stars={1: relion},
        relion_final_star=final_star,
        require_exact_schedule=True,
        require_exact_convergence=True,
    )

    assert report["status"] == "pass"
    assert report["threshold_failures"] == []
    scalar = report["iterations"][0]["scalar_state"]
    assert scalar["comparison"]["current_image_size"]["exact_equal"] is True
    assert scalar["comparison"]["current_resolution_angstrom"]["exact_equal"] is True
    assert scalar["recovar"]["current_resolution_shell_index"] == pytest.approx(12.8)
    assert scalar["relion"]["fields"]["estimated_resolution_angstrom"] == 10.0
    assert scalar["relion"]["fields"]["scheduling_current_resolution_angstrom"] == 12.0
    assert scalar["comparison"]["healpix_order"]["exact_equal"] is True
    assert scalar["recovar"]["average_pmax_particles"] == pytest.approx(
        np.mean(data["pmax_per_image_by_image_iter_000"])
    )
    assert scalar["recovar"]["average_pmax_mstep"] == pytest.approx(0.4321)
    assert scalar["comparison"]["average_pmax_particles"]["recovar_minus_relion"] == pytest.approx(0.0)
    assert scalar["comparison"]["average_pmax_mstep"]["recovar_minus_relion"] == pytest.approx(0.0)
    assert scalar["relion"]["artifacts"]["optimiser"]["present"] is True
    assert report["convergence_topology"]["recovar"] == {
        "iteration": 1,
        "has_converged": True,
        "final_all_data_ran": True,
    }
    assert report["convergence_topology"]["relion"] == {
        "iteration": 1,
        "has_converged": 1,
        "final_data_star_present": True,
    }
    assert report["final_all_data"]["status"] == "measured"
    assert report["final_all_data"]["recovar_vs_relion"]["significant_support"]["status"] == "not_measured"


@pytest.mark.unit
def test_cli_serializes_unavailable_nan_trajectory_scalar_as_null(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)
    with np.load(results, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    data["acc_rot_trajectory"] = np.asarray([np.nan])
    np.savez(results, **data)
    output = tmp_path / "nan_scalar.json"

    status = auditor.main(
        [
            "--recovar-results",
            str(results),
            "--recovar-particles-star",
            str(source),
            "--relion-star",
            str(relion),
            "--output-json",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    scalar = report["iterations"][0]["scalar_state"]
    assert status == 0
    assert scalar["recovar"]["accuracy_rotations_deg"] is None
    assert scalar["comparison"]["accuracy_rotations_deg"]["status"] == "not_measured"


@pytest.mark.unit
def test_scalar_star_parser_accepts_underscored_and_legacy_bare_labels(tmp_path):
    path = tmp_path / "mixed_scalars.star"
    path.write_text("data_general\n\n_rlnCurrentIteration -1\nrlnHasConverged 1\n")

    assert auditor._star_scalar_values(path) == {
        "rlnCurrentIteration": -1,
        "rlnHasConverged": 1,
    }


@pytest.mark.unit
@pytest.mark.parametrize("estimated_label", ["_rlnEstimatedResolution", "rlnEstimatedResolution"])
def test_model_class_resolution_parser_accepts_underscored_and_legacy_bare_labels(
    tmp_path, estimated_label
):
    path = _write_k1_model_star(
        tmp_path / "model.star",
        current_resolution=30.222222,
        estimated_resolution=32.0,
        estimated_label=estimated_label,
    )

    np.testing.assert_array_equal(
        auditor._star_loop_numeric_values(path, "rlnEstimatedResolution"),
        np.asarray([32.0]),
    )


@pytest.mark.unit
def test_near_tie_pmax_distribution_is_diagnostic_by_default_and_threshold_gated_explicitly(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)
    table, _ = auditor.read_star(str(relion))
    pmax_column = next(column for column in table if column.lstrip("_") == "rlnMaxValueProbDistribution")
    relion_pmax = table[pmax_column].astype(float).to_numpy()
    identity_column = next(column for column in table if column.lstrip("_") == "rlnImageName")
    by_name = dict(zip(table[identity_column].astype(str), relion_pmax, strict=True))
    source_table, _ = auditor.read_star(str(source))
    source_identity_column = next(
        column for column in source_table if column.lstrip("_") == "rlnImageName"
    )
    aligned = np.asarray([by_name[name] for name in source_table[source_identity_column].astype(str)])
    with np.load(results, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    data["pmax_per_image_by_image_iter_000"] = aligned + np.resize(np.asarray([-5e-5, 5e-5]), aligned.size)
    np.savez(results, **data)

    diagnostic = auditor.audit(
        recovar_results=results,
        recovar_particles_star=source,
        relion_stars={1: relion},
    )
    passing = auditor.audit(
        recovar_results=results,
        recovar_particles_star=source,
        relion_stars={1: relion},
        thresholds={"max_pmax_abs_p95": 1e-4},
    )
    failing = auditor.audit(
        recovar_results=results,
        recovar_particles_star=source,
        relion_stars={1: relion},
        thresholds={"max_pmax_abs_p95": 1e-5},
    )

    assert diagnostic["status"] == "complete"
    assert diagnostic["gating"]["enabled"] is False
    assert passing["status"] == "pass"
    assert failing["status"] == "fail"
    assert failing["threshold_failures"] == ["it001 pmax.absolute.p95=5e-05 > 1e-05"]


def _add_recovar_iterations(results: Path, iterations: tuple[int, ...]) -> None:
    with np.load(results, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    for iteration in iterations:
        for stem in ("pmax_per_image_by_image", "sig_counts_by_image"):
            data[f"{stem}_iter_{iteration:03d}"] = data[f"{stem}_iter_000"]
    np.savez(results, **data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("supplied_iterations", "missing_iteration"),
    [
        ((1, 3), 2),
        ((1, 2), 3),
    ],
    ids=("omitted-middle", "omitted-trailing"),
)
def test_cli_fails_closed_when_relion_iteration_is_omitted(
    tmp_path, supplied_iterations, missing_iteration
):
    results, source, relion, _control = _fixture(tmp_path)
    _add_recovar_iterations(results, (1, 2))
    relion_stars = {}
    for iteration in (1, 2, 3):
        path = tmp_path / f"run_it{iteration:03d}_data.star"
        if iteration == 1:
            path = relion
        else:
            path.write_text(relion.read_text())
        relion_stars[iteration] = path
    output = tmp_path / f"missing_{missing_iteration}.json"
    args = [
        "--recovar-results",
        str(results),
        "--recovar-particles-star",
        str(source),
    ]
    for iteration in supplied_iterations:
        args.extend(["--relion-star", str(relion_stars[iteration])])
    args.extend(["--output-json", str(output)])

    status = auditor.main(args)

    report = json.loads(output.read_text())
    assert status == 2
    assert report["status"] == "error"
    assert report["missing_relion_iterations"] == [missing_iteration]
    assert report["earliest_failure"] == f"missing_relion_iterations=[{missing_iteration}]"
    assert "iterations" not in report


@pytest.mark.unit
def test_identity_set_mismatch_fails_closed_without_particle_dump(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)
    table, _ = auditor.read_star(str(relion))
    image_column = next(column for column in table if column.lstrip("_") == "rlnImageName")
    table.loc[0, image_column] = "999999@different.mrcs"
    write_star(str(relion), table)
    output = tmp_path / "error.json"

    status = auditor.main(
        [
            "--recovar-results",
            str(results),
            "--recovar-particles-star",
            str(source),
            "--relion-star",
            str(relion),
            "--output-json",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert status == 2
    assert report["status"] == "error"
    assert "identity set mismatch" in report["earliest_failure"]
    assert "999999@different.mrcs" in report["earliest_failure"]
    assert "iterations" not in report


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["pmax", "support"])
def test_missing_required_particle_state_fails_closed(tmp_path, missing):
    results, source, relion, _control = _fixture(tmp_path)
    with np.load(results, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    key = {
        "pmax": "pmax_per_image_by_image_iter_000",
        "support": "sig_counts_by_image_iter_000",
    }[missing]
    data.pop(key)
    np.savez(results, **data)

    with pytest.raises(
        auditor.AuditError, match=f"missing required .*{'pmax_per_image' if missing == 'pmax' else 'sig_counts'}"
    ):
        auditor.audit(
            recovar_results=results,
            recovar_particles_star=source,
            relion_stars={1: relion},
        )


@pytest.mark.unit
def test_duplicate_identities_fail_closed(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)
    table, _ = auditor.read_star(str(source))
    image_column = next(column for column in table if column.lstrip("_") == "rlnImageName")
    table.loc[1, image_column] = table.loc[0, image_column]
    write_star(str(source), table)

    with pytest.raises(auditor.AuditError, match="duplicate rlnImageName"):
        auditor.audit(
            recovar_results=results,
            recovar_particles_star=source,
            relion_stars={1: relion},
        )


@pytest.mark.unit
def test_half_order_fallback_is_reindexed_to_image_identity(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)
    with np.load(results, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    half_order = np.concatenate([data["half1_indices"], data["half2_indices"]])
    pmax_by_image = data.pop("pmax_per_image_by_image_iter_000")
    support_by_image = data.pop("sig_counts_by_image_iter_000")
    data["pmax_per_half_order_iter_000"] = pmax_by_image[half_order]
    data["sig_counts_half_order_iter_000"] = support_by_image[half_order]
    np.savez(results, **data)

    report = auditor.audit(
        recovar_results=results,
        recovar_particles_star=source,
        relion_stars={1: relion},
    )

    assert report["iterations"][0]["recovar_vs_relion"]["pmax"]["signed"]["mean"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.unit
def test_explicit_iteration_selection_fails_closed_when_boundary_is_absent(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)

    with pytest.raises(auditor.AuditError, match="requested RECOVAR iterations are absent"):
        auditor.audit(
            recovar_results=results,
            recovar_particles_star=source,
            relion_stars={1: relion},
            recovar_iterations={3},
        )


@pytest.mark.unit
def test_cli_explicit_iteration_selection_allows_complete_boundary_subset(tmp_path):
    results, source, relion, _control = _fixture(tmp_path)
    _add_recovar_iterations(results, (1, 2))
    relion_it002 = tmp_path / "run_it002_data.star"
    relion_it002.write_text(relion.read_text())
    output = tmp_path / "explicit_boundary.json"

    status = auditor.main(
        [
            "--recovar-results",
            str(results),
            "--recovar-particles-star",
            str(source),
            "--recovar-iteration",
            "1",
            "--relion-star",
            str(relion_it002),
            "--output-json",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert status == 0
    assert report["status"] == "complete"
    assert report["iteration_alignment"]["selected_recovar_iterations"] == [1]
    assert report["iteration_alignment"]["missing_relion_iterations"] == []
    assert [(row["recovar_iteration"], row["relion_iteration"]) for row in report["iterations"]] == [(1, 2)]


@pytest.mark.unit
def test_independent_relion_control_pair_sets_the_numerical_envelope(tmp_path):
    results, source, relion, control = _fixture(tmp_path)
    table, _ = auditor.read_star(str(control))
    pmax_column = next(column for column in table if column.lstrip("_") == "rlnMaxValueProbDistribution")
    table[pmax_column] = table[pmax_column].astype(float) - 5e-6
    control_reference = tmp_path / "control_reference_it001_data.star"
    write_star(str(control_reference), table)

    report = auditor.audit(
        recovar_results=results,
        recovar_particles_star=source,
        relion_stars={1: relion},
        control_stars={1: control},
        control_reference_stars={1: control_reference},
    )

    row = report["iterations"][0]
    assert row["relion_control_vs_relion"]["pmax"]["absolute"]["mean"] == pytest.approx(5e-6)
    rec_mean = row["recovar_vs_relion"]["pmax"]["absolute"]["mean"]
    ratio = row["recovar_vs_relion_relative_to_control"]["metrics"]["pmax"][
        "recovar_to_control_absolute_error_ratio"
    ]["mean"]
    assert ratio == pytest.approx(rec_mean / 5e-6)
    assert report["sources"]["relion_control_reference_stars"]["1"] == str(control_reference.resolve())
