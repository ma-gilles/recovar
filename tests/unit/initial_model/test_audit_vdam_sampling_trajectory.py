from __future__ import annotations

import json

import pytest

from scripts.audit_vdam_sampling_trajectory import audit_sampling_trajectory

pytestmark = pytest.mark.unit


def _write_iteration(
    candidate_dir,
    relion_dir,
    iteration: int,
    *,
    candidate_range: float = 6.0,
    native_range: float = 6.0,
    candidate_translations: int = 52,
    candidate_prior_mode: int | None = 0,
    native_prior_mode: int = 0,
    candidate_current_size: int = 28,
    native_current_size: int = 28,
    candidate_current_resolution_angstrom: float = 20.0,
    native_current_resolution_angstrom: float = 20.0,
) -> None:
    tag = f"{iteration:03d}"
    candidate = {
        "oversampling": 1,
        "healpix_order": 3,
        "offset_range_angstrom": candidate_range,
        "offset_step_angstrom": 3.0,
        "random_perturbation": 0.1250001,
        "n_translations": candidate_translations,
        "sampling_acc_rot": 1.823,
        "sampling_acc_trans_angstrom": 1.717,
        "sampling_updated": False,
        "current_changes_optimal_offsets_angstrom": 1.25,
        "sampling_nr_iter_wo_resol_gain": 0,
        "current_size": candidate_current_size,
        "current_resolution": 1.0 / candidate_current_resolution_angstrom,
        "current_resolution_shell": 5,
    }
    if candidate_prior_mode is not None:
        candidate["orientational_prior_mode"] = candidate_prior_mode
        candidate["uniform_local_orientation_prior"] = candidate_prior_mode == 1
    (candidate_dir / f"run_it{tag}_recovar_meta.json").write_text(json.dumps(candidate))
    (relion_dir / f"run_it{tag}_sampling.star").write_text(
        "\n".join(
            [
                "data_sampling_general",
                "_rlnHealpixOrder 3",
                f"_rlnOffsetRange {native_range:.6f}",
                "_rlnOffsetStep 3.000000",
                "_rlnSamplingPerturbInstance 0.125000",
                "_rlnSamplingPerturbFactor 0.500000",
                "",
                "data_sampling_directions",
                "loop_",
                "_rlnAngleRot #1",
                "_rlnAngleTilt #2",
                "0.0 0.0",
                "",
            ]
        )
    )
    (relion_dir / f"run_it{tag}_model.star").write_text(
        "\n".join(
            [
                "data_model_general",
                f"_rlnCurrentResolution {native_current_resolution_angstrom:.6f}",
                f"_rlnCurrentImageSize {native_current_size}",
                f"_rlnOrientationalPriorMode {native_prior_mode}",
                "",
                "data_model_classes",
                "loop_",
                "_rlnReferenceImage #1",
                "_rlnAccuracyRotations #2",
                "_rlnAccuracyTranslationsAngst #3",
                "map.mrc 1.823000 1.717000",
                "",
            ]
        )
    )
    (relion_dir / f"run_it{tag}_optimiser.star").write_text(
        "\n".join(
            [
                "data_optimiser_general",
                "_rlnChangesOptimalOffsets 1.250000",
                "_rlnNumberOfIterWithoutResolutionGain 1",
                "",
            ]
        )
    )


def test_sampling_trajectory_reports_first_geometry_and_topology_mismatch(tmp_path):
    candidate_dir = tmp_path / "candidate"
    relion_dir = tmp_path / "relion"
    candidate_dir.mkdir()
    relion_dir.mkdir()
    _write_iteration(candidate_dir, relion_dir, 1)
    _write_iteration(
        candidate_dir,
        relion_dir,
        2,
        candidate_range=5.9,
        native_range=6.0,
        candidate_translations=20,
    )

    report = audit_sampling_trajectory(candidate_dir, relion_dir, pixel_size=2.0)

    assert report["result"] == "fail"
    assert report["first_mismatch"]["offset_range"] == 2
    assert report["first_mismatch"]["translation_topology"] == 2
    assert report["first_mismatch"]["sampling_updated"] is None
    assert report["iterations"][0]["pass"] is True


def test_sampling_trajectory_accepts_relion_serialization_rounding(tmp_path):
    candidate_dir = tmp_path / "candidate"
    relion_dir = tmp_path / "relion"
    candidate_dir.mkdir()
    relion_dir.mkdir()
    _write_iteration(candidate_dir, relion_dir, 1, candidate_range=6.0000004)

    report = audit_sampling_trajectory(candidate_dir, relion_dir, pixel_size=2.0)

    assert report["result"] == "pass"
    assert all(value is None for value in report["first_mismatch"].values())


def test_sampling_trajectory_reports_first_orientation_prior_mode_mismatch(tmp_path):
    candidate_dir = tmp_path / "candidate"
    relion_dir = tmp_path / "relion"
    candidate_dir.mkdir()
    relion_dir.mkdir()
    _write_iteration(candidate_dir, relion_dir, 89)
    _write_iteration(
        candidate_dir,
        relion_dir,
        90,
        candidate_prior_mode=0,
        native_prior_mode=1,
    )

    report = audit_sampling_trajectory(candidate_dir, relion_dir, pixel_size=2.0)

    assert report["result"] == "fail"
    assert report["first_mismatch"]["orientational_prior_mode"] == 90
    assert report["iterations"][1]["candidate"]["orientational_prior_mode"] == 0
    assert report["iterations"][1]["native"]["orientational_prior_mode"] == 1


def test_sampling_trajectory_fails_closed_when_candidate_prior_mode_is_unrecorded(tmp_path):
    candidate_dir = tmp_path / "candidate"
    relion_dir = tmp_path / "relion"
    candidate_dir.mkdir()
    relion_dir.mkdir()
    _write_iteration(candidate_dir, relion_dir, 1, candidate_prior_mode=None)

    report = audit_sampling_trajectory(candidate_dir, relion_dir, pixel_size=2.0)

    assert report["result"] == "fail"
    assert report["first_mismatch"]["orientational_prior_mode"] == 1
    assert report["iterations"][0]["candidate"]["orientational_prior_mode"] is None
    assert report["iterations"][0]["candidate"]["uniform_local_orientation_prior"] is None


def test_sampling_trajectory_reports_current_image_size_transition_lag(tmp_path):
    candidate_dir = tmp_path / "candidate"
    relion_dir = tmp_path / "relion"
    candidate_dir.mkdir()
    relion_dir.mkdir()
    _write_iteration(candidate_dir, relion_dir, 172)
    _write_iteration(
        candidate_dir,
        relion_dir,
        173,
        candidate_current_size=74,
        native_current_size=72,
    )

    report = audit_sampling_trajectory(candidate_dir, relion_dir, pixel_size=2.0)

    assert report["result"] == "fail"
    assert report["first_mismatch"]["current_size"] == 173
    assert report["first_mismatch"]["current_resolution"] is None
    assert report["iterations"][1]["candidate"]["current_size"] == 74
    assert report["iterations"][1]["native"]["current_size"] == 72


def test_sampling_trajectory_reports_current_resolution_transition_lag(tmp_path):
    candidate_dir = tmp_path / "candidate"
    relion_dir = tmp_path / "relion"
    candidate_dir.mkdir()
    relion_dir.mkdir()
    _write_iteration(candidate_dir, relion_dir, 171)
    _write_iteration(
        candidate_dir,
        relion_dir,
        172,
        candidate_current_resolution_angstrom=20.1481481481,
        native_current_resolution_angstrom=20.923077,
    )

    report = audit_sampling_trajectory(candidate_dir, relion_dir, pixel_size=2.0)

    assert report["result"] == "fail"
    assert report["first_mismatch"]["current_resolution"] == 172
    assert report["first_mismatch"]["current_size"] is None
    assert report["iterations"][1]["absolute_errors"]["current_resolution_angstrom"] > 0.7
