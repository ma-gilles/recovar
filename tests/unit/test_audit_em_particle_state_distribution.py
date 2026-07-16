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
    help_text = auditor._parser().format_help()
    assert "--relion-control-star" in help_text
    assert "image identities" in help_text


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
