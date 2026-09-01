from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_k1_realdata_science_equivalence.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k1_realdata_science_equivalence",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

pytestmark = pytest.mark.unit


def _target(scorecard: dict) -> dict:
    return next(case for case in scorecard["cases"] if case["role"] == "scoring")


def _frozen_target_scorecard() -> dict:
    scorecard = copy.deepcopy(MODULE.load_and_validate_scorecard())
    target = _target(scorecard)
    contract = target["input_contract"]
    contract["prepared_star_sha256"] = "e" * 64
    contract["initial_reference"]["relion_file_sha256"] = "f" * 64
    contract["initial_reference"]["recovar_frame_file_sha256"] = "d" * 64
    contract["initial_reference"]["canonical_array_sha256"] = "9" * 64
    target["pending_contract_fields"] = []
    return scorecard


def _valid_metrics(artifacts: dict[str, str] | None = None) -> dict:
    return {
        "schema": MODULE.COLLECTOR_SCHEMA,
        "dataset": "10202",
        "scientifically_valid": True,
        "initial_reference_frame_gate": {"pass": True},
        "particle_identity_gate": {
            "pass": True,
            "prepared_count": 30_515,
            "relion_count": 30_515,
            "recovar_count": 30_515,
        },
        "convergence_and_topology": {
            "relion_converged": True,
            "recovar_converged": True,
        },
        "artifacts": {} if artifacts is None else artifacts,
    }


def _curves(cross_engine_fsc: float = 0.99) -> dict[str, np.ndarray]:
    half = np.full(260, 0.8, dtype=np.float64)
    half[0] = 1.0
    half[220:] = 0.1
    cross = np.full_like(half, cross_engine_fsc)
    return {
        "relion_final_half_fsc": half,
        "recovar_final_half_fsc": half.copy(),
        "final_cross_engine_raw": cross,
        "final_cross_engine_half1": np.full_like(half, 0.99),
        "final_cross_engine_half2": np.full_like(half, 0.99),
    }


def _reseal_execution_envelope(evidence: dict) -> None:
    envelope_path = Path(evidence["execution_binding"]["evidence_envelope_path"])
    envelope = json.loads(envelope_path.read_text())
    for field in ("collector", "analysis_commands", "analysis_artifacts"):
        if field in evidence:
            envelope[field] = evidence[field]
        else:
            envelope.pop(field, None)
    envelope_path.write_text(json.dumps(envelope))
    evidence["execution_binding"]["evidence_envelope_sha256"] = MODULE.sha256_file(envelope_path)


def _write_evidence(
    tmp_path: Path,
    scorecard: dict,
    *,
    cross_engine_fsc: float = 0.99,
    subject_commit: str | None = None,
) -> Path:
    metrics_path = tmp_path / "metrics.json"
    curves_path = tmp_path / "fsc_curves.npz"
    input_paths = {name: tmp_path / f"{name}.mrc" for name in MODULE.SCIENCE_INPUT_TO_COLLECTOR_ARTIFACT}
    for name, path in input_paths.items():
        path.write_bytes(f"{name}-map-fixture".encode())
    collector_artifacts = {
        MODULE.SCIENCE_INPUT_TO_COLLECTOR_ARTIFACT[name]: MODULE.sha256_file(path) for name, path in input_paths.items()
    }
    metrics_path.write_text(json.dumps(_valid_metrics(collector_artifacts)))
    np.savez(curves_path, **_curves(cross_engine_fsc))
    contract = _target(scorecard)["input_contract"]
    subject = {
        "commit": subject_commit or scorecard["suite"]["subject_commit"],
        "tree_clean": True,
        "diff_sha256": MODULE.EMPTY_SHA256,
    }
    launch_path = tmp_path / "launch_manifest.json"
    launch_path.write_text(
        json.dumps(
            {
                "schema": MODULE.LAUNCH_MANIFEST_SCHEMA,
                "run_root": str(tmp_path.resolve()),
                "subject": subject,
            }
        )
    )
    finalizer_path = REPO_ROOT / MODULE.FINALIZER_RELATIVE_PATH
    execution_envelope_path = tmp_path / "execution_envelope.json"
    execution_envelope_path.write_text(
        json.dumps(
            {
                "schema": MODULE.EXECUTION_BINDING_SCHEMA,
                "launch": {
                    "launch_manifest": {
                        "path": str(launch_path.resolve()),
                        "sha256": MODULE.sha256_file(launch_path),
                    }
                },
            }
        )
    )
    finalizer_argv = [
        sys.executable,
        str(finalizer_path.resolve()),
        "--launch-manifest",
        str(launch_path.resolve()),
    ]
    evidence = {
        "schema": MODULE.EVIDENCE_SCHEMA,
        "case_id": MODULE.TARGET_CASE_ID,
        "subject": subject,
        "input_contract": {
            "particle_count": contract["particle_count"],
            "box_size": contract["box_size"],
            "voxel_size_angstrom": contract["voxel_size_angstrom"],
            "prepared_star_sha256": contract["prepared_star_sha256"],
            "poses_pkl_sha256": contract["poses_pkl"]["sha256"],
            "ctf_pkl_sha256": contract["ctf_pkl"]["sha256"],
            "prepared_star_preserves_source_particle_order": True,
            "prepared_star_preserves_source_half_assignment": True,
            "prepared_star_preserves_source_pose_shift_metadata": True,
            "particle_stack_size_bytes": contract["particle_stack"]["size_bytes"],
            "particle_stack_sha256": contract["particle_stack"]["sha256"],
            "half_assignment_sha256": contract["half_assignment_sha256"],
            "half_assignment_encoding": contract["half_assignment_encoding"],
            "half_assignment_bytes": contract["half_assignment_bytes"],
            "half1_count": contract["half1_count"],
            "half2_count": contract["half2_count"],
            "initial_reference_canonical_exact": True,
            "initial_reference_relion_file_sha256": contract["initial_reference"]["relion_file_sha256"],
            "initial_reference_recovar_frame_file_sha256": contract["initial_reference"]["recovar_frame_file_sha256"],
            "initial_reference_canonical_array_sha256": contract["initial_reference"]["canonical_array_sha256"],
            "k": 1,
            "autonomous_refinement": True,
            "fixed_poses": False,
            "local_only": False,
            "forced_final_after_nonconvergence": False,
            "symmetry": {
                "family": "icosahedral",
                "requested_label": contract["symmetry"]["requested_label"],
                "relion_label": contract["symmetry"]["label"],
                "recovar_label": contract["symmetry"]["label"],
                "operator_count": contract["symmetry"]["operator_count"],
                "operator_order": contract["symmetry"]["operator_order"],
                "left_operator_policy": contract["symmetry"]["left_operator_policy"],
                "hash_encoding": contract["symmetry"]["hash_encoding"],
                "operators_sha256": contract["symmetry"]["operators_sha256"],
            },
        },
        "collector": {
            "schema": MODULE.COLLECTOR_SCHEMA,
            "collector_sha256": scorecard["collector"]["sha256"],
            "metrics_json": str(metrics_path),
            "metrics_sha256": MODULE.sha256_file(metrics_path),
            "fsc_curves_npz": str(curves_path),
            "curves_sha256": MODULE.sha256_file(curves_path),
        },
        "diagnostics": {
            "status": "pass",
            "common_mask_cross_engine_band_auc": 1.0,
            "cross_validated_rigid_alignment_band_auc": 1.0,
        },
        "execution_binding": {
            "schema": MODULE.EXECUTION_BINDING_SCHEMA,
            "launch_manifest_path": str(launch_path.resolve()),
            "launch_manifest_sha256": MODULE.sha256_file(launch_path),
            "finalizer_command_path": str(finalizer_path.resolve()),
            "finalizer_command_sha256": MODULE.sha256_file(finalizer_path),
            "finalizer_argv": finalizer_argv,
            "finalizer_argv_sha256": MODULE.sha256_json(finalizer_argv),
            "evidence_envelope_path": str(execution_envelope_path.resolve()),
            "evidence_envelope_sha256": MODULE.sha256_file(execution_envelope_path),
        },
    }
    _reseal_execution_envelope(evidence)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    return evidence_path


def _attach_science_diagnostics(
    evidence_path: Path,
    scorecard: dict,
    *,
    aligned_fsc: float = 0.99,
    masked_fsc: float = 0.99,
    rotation_matrix: np.ndarray | None = None,
    alignment_overrides: dict[str, object] | None = None,
) -> None:
    evidence = json.loads(evidence_path.read_text())
    output_dir = evidence_path.parent / "proper_so3"
    output_dir.mkdir()
    curves_path = output_dir / "science_diagnostic_curves.npz"
    mask_path = output_dir / "common_soft_mask.mrc"
    mask_path.write_bytes(b"common-mask-fixture")
    with np.load(evidence["collector"]["fsc_curves_npz"], allow_pickle=False) as primary_archive:
        n_shells = int(primary_archive[MODULE.CURVE_KEYS[0]].size)
    aligned = np.full(n_shells, aligned_fsc, dtype=np.float64)
    masked = np.full(n_shells, masked_fsc, dtype=np.float64)
    curve_payload = {key: aligned.copy() for key in MODULE.ALIGNED_CURVE_KEYS}
    curve_payload.update({key: masked.copy() for key in MODULE.MASKED_CURVE_KEYS})
    np.savez(curves_path, **curve_payload)
    rotation = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix, dtype=np.float64)
    determinant = float(np.linalg.det(rotation))
    orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    contract = _target(scorecard)["input_contract"]
    mask_sha256 = MODULE.sha256_file(mask_path)
    curves_sha256 = MODULE.sha256_file(curves_path)
    fields = sorted(curve_payload)
    diagnostics = {
        "schema": MODULE.SCIENCE_DIAGNOSTICS_SCHEMA,
        "case_id": MODULE.TARGET_CASE_ID,
        "producer": {
            "path": str(REPO_ROOT / scorecard["diagnostic_producer"]["path"]),
            "sha256": scorecard["diagnostic_producer"]["sha256"],
        },
        "inputs": {
            name: {
                "path": str(evidence_path.parent / f"{name}.mrc"),
                "sha256": MODULE.sha256_file(evidence_path.parent / f"{name}.mrc"),
            }
            for name in MODULE.SCIENCE_INPUT_TO_COLLECTOR_ARTIFACT
        },
        "box_size": contract["box_size"],
        "voxel_size_angstrom": contract["voxel_size_angstrom"],
        "diagnostics": {
            "proper_so3_alignment": {
                "fit_source": "merged_low_frequency",
                "method": "identity-augmented HEALPix proper-rotation seed plus continuous scipy rotvec Powell and subpixel translation",
                "seed_source": "identity_augmented_RELION_HEALPix_grid",
                "continuous_so3_refinement": True,
                "translation_subpixel": True,
                "fit_max_shell_full_box": MODULE.PROPER_ALIGNMENT_FIT_MAX_SHELL_FULL_BOX,
                "fit_box_size": MODULE.PROPER_ALIGNMENT_FIT_BOX_SIZE,
                "seed_healpix_order": MODULE.PROPER_ALIGNMENT_SEED_HEALPIX_ORDER,
                "refine_healpix_orders": MODULE.PROPER_ALIGNMENT_REFINE_HEALPIX_ORDERS,
                "applied_unchanged_to": ["merged", "half1", "half2"],
                "rotation_matrix_recovar_to_relion": rotation.tolist(),
                "translation_recovar_to_relion_zyx": [0.25, -0.5, 0.125],
                "no_reflection": True,
                "sign_fit": False,
                "scale_fit": False,
                "optimizer_success": True,
                "determinant": determinant,
                "orthogonality_frobenius": orthogonality,
                "symmetry_label": contract["symmetry"]["label"],
                "symmetry_operators_sha256": contract["symmetry"]["operators_sha256"],
            },
            "common_mask": {
                "path": str(mask_path),
                "sha256": mask_sha256,
                "engine_symmetric": True,
                "applied_identically_to_both_engines": True,
                "acceptance_metric": False,
            },
        },
        "artifacts": {
            "curve_archive": {
                "path": str(curves_path),
                "sha256": curves_sha256,
                "fields": fields,
            },
            "common_mask": {"path": str(mask_path), "sha256": mask_sha256},
        },
        "acceptance_policy": {
            "proper_alignment_can_replace_only_failed_raw_cross_engine_gates": True,
            "within_engine_half_map_quality_remains_mandatory": True,
            "common_mask_can_rescue": False,
        },
    }
    diagnostics["diagnostics"]["proper_so3_alignment"].update(alignment_overrides or {})
    diagnostics_path = output_dir / "science_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))
    evidence["analysis_artifacts"] = {
        "science_diagnostics": {
            "path": str(diagnostics_path),
            "sha256": MODULE.sha256_file(diagnostics_path),
            "schema": MODULE.SCIENCE_DIAGNOSTICS_SCHEMA,
        },
        "curve_archive": {
            "path": str(curves_path),
            "sha256": curves_sha256,
            "fields": fields,
        },
        "common_mask": {"path": str(mask_path), "sha256": mask_sha256},
    }
    _reseal_execution_envelope(evidence)
    evidence_path.write_text(json.dumps(evidence))


def test_fixed_scorecard_is_valid_and_markdown_is_fresh() -> None:
    scorecard = MODULE.load_and_validate_scorecard()
    report = MODULE.build_report(scorecard, scorecard_path=MODULE.DEFAULT_SCORECARD)
    target = next(row for row in report["cases"] if row["id"] == MODULE.TARGET_CASE_ID)

    assert report["aggregate"] == {
        "scoring_denominator": 1,
        "passed": 0,
        "failed": 0,
        "pending": 1,
        "invalid": 0,
    }
    assert _target(scorecard)["scope"] == "06_Final_Stack only; image set 07 is excluded"
    assert report["masked_fsc_support"]["role"] == "supporting_only"
    assert report["masked_fsc_support"]["acceptance_metric"] is False
    assert report["masked_fsc_support"]["can_rescue"] is False
    assert target["status"] == "pending"
    assert target["high_resolution_achievement"] == {
        "status": "pending",
        "pass": None,
        "each_engine_resolution_angstrom_max": 3.0,
        "relion": {"status": "complete", "resolution_angstrom": 2.511554, "pass": True},
        "recovar": {"status": "pending", "resolution_angstrom": None, "pass": None},
        "partial_result_does_not_score_case": True,
    }
    assert target["partial_engine_results"]["verification_status"] == "frozen_not_replayed"
    assert target["partial_engine_results"]["relion"]["corrected_masked"] == {
        "role": "supporting_only",
        "acceptance_metric": False,
        "can_rescue": False,
        "resolution_angstrom": 2.122559,
    }
    assert target["partial_engine_results"]["recovar"] == {"status": "pending"}
    assert MODULE.DEFAULT_MARKDOWN.read_text() == MODULE.render_markdown(report)


def test_partial_relion_result_cannot_score_pending_recovar_case() -> None:
    scorecard = MODULE.load_and_validate_scorecard()
    report = MODULE.build_report(scorecard, scorecard_path=MODULE.DEFAULT_SCORECARD)
    target = next(row for row in report["cases"] if row["id"] == MODULE.TARGET_CASE_ID)

    assert target["status"] == "pending"
    assert target["science_equivalence_pass"] is None
    assert target["primary_metrics"] is None
    assert report["aggregate"] == {
        "scoring_denominator": 1,
        "passed": 0,
        "failed": 0,
        "pending": 1,
        "invalid": 0,
    }


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("unmasked", "resolution_angstrom", 2.6, "unmasked resolution changed"),
        ("corrected_masked", "can_rescue", True, "masked result became a rescue route"),
    ],
)
def test_partial_relion_result_contract_is_fail_closed(
    section: str,
    field: str,
    value: object,
    error: str,
) -> None:
    scorecard = MODULE.load_and_validate_scorecard()
    relion = _target(scorecard)["partial_engine_results"]["relion"]
    relion[section][field] = value

    with pytest.raises(ValueError, match=error):
        MODULE._validate_target_partial_engine_results(_target(scorecard))


def test_reproduction_contract_does_not_invent_masked_submission_command() -> None:
    scorecard = MODULE.load_and_validate_scorecard()
    masked = scorecard["reproduction"]["masked"]["10073_10345"]
    masked["recorded_submission_commands"] = ["sbatch invented.sbatch"]

    with pytest.raises(ValueError, match="masked reproduction fields changed"):
        MODULE._validate_reproduction_contract(scorecard["reproduction"])


def test_joint_band_and_resolution_use_crossing_shell() -> None:
    recovar = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1])
    relion = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.1, 0.1])
    curves = {
        "recovar_final_half_fsc": recovar,
        "relion_final_half_fsc": relion,
        "final_cross_engine_raw": np.ones_like(recovar),
        "final_cross_engine_half1": np.ones_like(recovar),
        "final_cross_engine_half2": np.ones_like(recovar),
    }

    result = MODULE.score_curves(
        curves,
        box_size=800,
        voxel_size_angstrom=0.788,
        thresholds=MODULE.EXPECTED_THRESHOLDS,
    )

    band = result["jointly_resolved_band"]
    assert band["recovar_crossing_shell"] == 5
    assert band["relion_crossing_shell"] == 6
    assert band["last_shell"] == 4
    assert band["recovar_resolution_angstrom"] == pytest.approx(800 * 0.788 / 5)
    assert band["relion_resolution_angstrom"] == pytest.approx(800 * 0.788 / 6)


def test_frozen_calibrations_pass_half_map_gates_and_report_cross_engine_separately() -> None:
    scorecard = MODULE.load_and_validate_scorecard()

    for case in scorecard["cases"]:
        if case["role"] == "calibration":
            assert MODULE.apply_half_map_quality_gates(case["expected_metrics"], scorecard["thresholds"]) == []
    by_id = {case["id"]: case for case in scorecard["cases"]}
    assert by_id["empiar-10073-native-c1"]["expected_cross_engine_route"] == "raw_canonical"
    assert by_id["empiar-10345-native-c1"]["expected_cross_engine_route"] == "raw_canonical"
    case_10097 = by_id["empiar-10097-native-c1"]
    assert MODULE.apply_cross_engine_gates(case_10097["expected_metrics"], scorecard["thresholds"])
    assert case_10097["expected_cross_engine_route"] == "none_unqualified"
    assert case_10097["proper_so3_diagnostics"]["expected_can_rescue_cross_engine"] is False
    report = MODULE.build_report(scorecard, scorecard_path=MODULE.DEFAULT_SCORECARD)
    row_10097 = next(row for row in report["cases"] if row["id"] == "empiar-10097-native-c1")
    assert row_10097["status"] == "pass"
    assert row_10097["half_map_quality_pass"] is True
    assert row_10097["primary_pass"] is False
    assert row_10097["science_equivalence_pass"] is False


def test_masked_support_contract_cannot_become_a_rescue_route() -> None:
    scorecard = MODULE.load_and_validate_scorecard()
    scorecard["masked_fsc_support"]["can_rescue"] = True

    with pytest.raises(ValueError, match="masked FSC became a rescue route"):
        MODULE._validate_masked_fsc_support_contract(scorecard["masked_fsc_support"])


@pytest.mark.parametrize(
    ("metric", "value", "failure"),
    [
        ("half_resolution_ratio", 1.050001, "half_resolution_ratio"),
        ("half_curve_rmse", 0.020001, "half_curve_rmse"),
        ("half_band_auc_abs_delta", 0.020001, "half_band_auc_abs_delta"),
        ("merged_cross_engine_band_auc", 0.949999, "merged_cross_engine_band_auc"),
        ("half1_cross_engine_band_auc", 0.899999, "each_half_cross_engine_band_auc"),
    ],
)
def test_each_primary_gate_fails_independently(metric: str, value: float, failure: str) -> None:
    metrics = {
        "half_resolution_ratio": 1.0,
        "half_curve_rmse": 0.0,
        "half_band_auc_abs_delta": 0.0,
        "merged_cross_engine_band_auc": 1.0,
        "half1_cross_engine_band_auc": 1.0,
        "half2_cross_engine_band_auc": 1.0,
    }
    metrics[metric] = value

    assert MODULE.apply_primary_gates(metrics, MODULE.EXPECTED_THRESHOLDS) == [failure]


def test_legacy_or_mask_only_diagnostics_cannot_rescue_cross_engine_failure(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard, cross_engine_fsc=0.5)

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "fail"
    assert result["primary_pass"] is False
    assert result["failed_gates"] == ["cross_engine_equivalence"]
    assert result["legacy_collector_diagnostics"]["status"] == "pass"
    assert result["legacy_collector_diagnostics"]["can_rescue_cross_engine"] is False


def test_valid_proper_so3_alignment_can_rescue_only_cross_engine_gates(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard, cross_engine_fsc=0.5)
    _attach_science_diagnostics(evidence_path, scorecard, aligned_fsc=0.99, masked_fsc=1.0)

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "pass"
    assert result["primary_pass"] is True
    assert result["equivalence_route"] == "continuous_proper_so3_rigid"
    assert result["raw_cross_engine_pass"] is False
    assert result["science_diagnostics"]["proper_so3_alignment"]["can_rescue_cross_engine"] is True
    assert result["science_diagnostics"]["common_mask"]["can_rescue"] is False


def test_proper_alignment_cannot_rescue_half_map_quality_failure(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard, cross_engine_fsc=0.5)
    evidence = json.loads(evidence_path.read_text())
    curves_path = Path(evidence["collector"]["fsc_curves_npz"])
    curves = _curves(0.5)
    curves["recovar_final_half_fsc"][1:220] -= 0.1
    np.savez(curves_path, **curves)
    evidence["collector"]["curves_sha256"] = MODULE.sha256_file(curves_path)
    _reseal_execution_envelope(evidence)
    evidence_path.write_text(json.dumps(evidence))
    _attach_science_diagnostics(evidence_path, scorecard, aligned_fsc=0.99, masked_fsc=1.0)

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "fail"
    assert result["equivalence_route"] == "continuous_proper_so3_rigid"
    assert any(name.startswith("half_") for name in result["failed_gates"])
    assert "cross_engine_equivalence" not in result["failed_gates"]


def test_masked_fsc_cannot_rescue_failed_raw_and_aligned_routes(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard, cross_engine_fsc=0.5)
    _attach_science_diagnostics(evidence_path, scorecard, aligned_fsc=0.5, masked_fsc=1.0)

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "fail"
    assert result["failed_gates"] == ["cross_engine_equivalence"]
    assert result["science_diagnostics"]["common_mask"]["metrics"]["merged_cross_engine_band_auc"] == 1.0
    assert result["masked_fsc_can_rescue"] is False


def test_reflection_in_supplied_alignment_is_fail_closed(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    reflection = np.diag([-1.0, 1.0, 1.0])
    _attach_science_diagnostics(evidence_path, scorecard, rotation_matrix=reflection)

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    assert "science_diagnostics:rotation_determinant" in result["provenance_failures"]


def test_subject_commit_mismatch_is_invalid_even_when_metrics_pass(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(
        tmp_path,
        scorecard,
        subject_commit="0" * 40,
    )

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    assert result["primary_pass"] is False
    assert result["failed_gates"] == []
    assert result["provenance_failures"] == ["subject_commit", "suite_subject_commit"]


def test_target_evidence_requires_execution_binding(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    evidence = json.loads(evidence_path.read_text())
    del evidence["execution_binding"]
    evidence_path.write_text(json.dumps(evidence))

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    assert "execution_binding:missing" in result["provenance_failures"]


def test_supplied_execution_binding_is_fail_closed(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    evidence = json.loads(evidence_path.read_text())
    evidence["execution_binding"]["launch_manifest_sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(evidence))

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    assert "execution_binding:launch_manifest_sha256" in result["provenance_failures"]


def test_raw_collector_substitution_breaks_sealed_execution_envelope(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    evidence = json.loads(evidence_path.read_text())
    curves_path = Path(evidence["collector"]["fsc_curves_npz"])
    curves = _curves()
    curves["final_cross_engine_raw"][:] = 0.98
    np.savez(curves_path, **curves)
    evidence["collector"]["curves_sha256"] = MODULE.sha256_file(curves_path)
    evidence_path.write_text(json.dumps(evidence))

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    assert "execution_binding:evidence_envelope_collector" in result["provenance_failures"]


def test_science_inputs_must_match_external_collector_artifact_hashes(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    _attach_science_diagnostics(evidence_path, scorecard)
    evidence = json.loads(evidence_path.read_text())
    science_spec = evidence["analysis_artifacts"]["science_diagnostics"]
    science_path = Path(science_spec["path"])
    science = json.loads(science_path.read_text())
    substituted = tmp_path / "substituted_map.mrc"
    substituted.write_bytes(b"a different valid map artifact")
    science["inputs"]["recovar_merged"] = {
        "path": str(substituted),
        "sha256": MODULE.sha256_file(substituted),
    }
    science_path.write_text(json.dumps(science))
    science_spec["sha256"] = MODULE.sha256_file(science_path)
    evidence_path.write_text(json.dumps(evidence))

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    assert "science_diagnostics:recovar_merged_collector_artifact_sha256" in result["provenance_failures"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fit_max_shell_full_box", 31),
        ("fit_box_size", 63),
        ("seed_healpix_order", 2),
        ("refine_healpix_orders", [3]),
        ("applied_unchanged_to", ["merged", "half1"]),
    ],
)
def test_nondefault_proper_alignment_contract_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    _attach_science_diagnostics(evidence_path, scorecard, alignment_overrides={field: value})

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    expected = "transform_reuse" if field == "applied_unchanged_to" else field
    assert f"science_diagnostics:{expected}" in result["provenance_failures"]


@pytest.mark.parametrize(("recovar_crossing", "relion_crossing"), [(210, 220), (220, 210)])
def test_equivalence_and_absolute_high_resolution_are_separate_gates(
    tmp_path: Path,
    recovar_crossing: int,
    relion_crossing: int,
) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    evidence = json.loads(evidence_path.read_text())
    curves_path = Path(evidence["collector"]["fsc_curves_npz"])
    curves = _curves()
    for key, crossing in (
        ("recovar_final_half_fsc", recovar_crossing),
        ("relion_final_half_fsc", relion_crossing),
    ):
        curve = np.full(260, 0.8, dtype=np.float64)
        curve[0] = 1.0
        curve[crossing:] = 0.1
        curves[key] = curve
    np.savez(curves_path, **curves)
    evidence["collector"]["curves_sha256"] = MODULE.sha256_file(curves_path)
    _reseal_execution_envelope(evidence)
    evidence_path.write_text(json.dumps(evidence))

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "fail"
    assert result["science_equivalence_pass"] is True
    assert result["equivalence_route"] == "raw_canonical"
    assert result["high_resolution_achievement"]["pass"] is False
    assert result["failed_gates"] == ["target_high_resolution"]
    failing_engine = "recovar" if recovar_crossing == 210 else "relion"
    passing_engine = "relion" if failing_engine == "recovar" else "recovar"
    assert result["high_resolution_achievement"][failing_engine]["pass"] is False
    assert result["high_resolution_achievement"][passing_engine]["pass"] is True


def test_live_markdown_reports_route_resolutions_and_key_metrics(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    _attach_science_diagnostics(evidence_path, scorecard)

    report = MODULE.build_report(
        scorecard,
        scorecard_path=MODULE.DEFAULT_SCORECARD,
        evidence_paths=(evidence_path,),
    )
    markdown = MODULE.render_markdown(report)

    assert "| `empiar-10202-set06-k1-I1` | pass | pass | `raw_canonical` | pass |" in markdown
    assert "2.865 | 2.865" in markdown
    assert "0.990000 | 0.990000 | 0.000000" in markdown
    assert "threshold at shell 81 on" in markdown
    assert "10073 and shell 49 on 10345" in markdown
    assert "6.568 A and 8.235 A" in markdown
    assert "shells 44 and 45 (7.622 A and 7.452 A)" in markdown
    assert "`none_unqualified`" in markdown
    assert "Supporting RELION corrected-masked FSC" in markdown
    assert "cannot rescue an unmasked failure" in markdown
    assert "## Submitted EMPIAR-10202 scoring evidence" in markdown
    assert "terminal status is `pass`" in markdown
    assert "| RECOVAR | pending | -- | -- | pending | -- |" not in markdown
    assert "RECOVAR and every cross-engine acceptance metric remain pending" not in markdown
    assert MODULE.REPRODUCTION_REPLAY_COMMAND in markdown
    assert "No original `sbatch` argv was separately sealed" in markdown
    assert "3.8 A sharpened full-complex map" in markdown
    assert "EMD-9012 records 1.86 A" in markdown


@pytest.mark.parametrize("terminal_status", ("pass", "fail", "invalid"))
def test_terminal_markdown_never_reverts_to_pending_result_prose(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    _attach_science_diagnostics(evidence_path, scorecard)
    evidence = json.loads(evidence_path.read_text())

    if terminal_status == "fail":
        curves_path = Path(evidence["collector"]["fsc_curves_npz"])
        with np.load(curves_path, allow_pickle=False) as archive:
            curves = {name: np.asarray(archive[name]) for name in archive.files}
        recovar_half = np.full(260, 0.8, dtype=np.float64)
        recovar_half[0] = 1.0
        recovar_half[180:] = 0.1
        curves["recovar_final_half_fsc"] = recovar_half
        np.savez(curves_path, **curves)
        evidence["collector"]["curves_sha256"] = MODULE.sha256_file(curves_path)
        _reseal_execution_envelope(evidence)
    elif terminal_status == "invalid":
        evidence["input_contract"]["k"] = 2
    evidence_path.write_text(json.dumps(evidence))

    report = MODULE.build_report(
        scorecard,
        scorecard_path=MODULE.DEFAULT_SCORECARD,
        evidence_paths=(evidence_path,),
    )
    target = next(row for row in report["cases"] if row["id"] == MODULE.TARGET_CASE_ID)
    markdown = MODULE.render_markdown(report)

    assert target["status"] == terminal_status
    assert "RECOVAR and every cross-engine acceptance metric remain pending" not in markdown
    assert "pending until the RECOVAR full refinement" not in markdown
    assert "| RECOVAR | pending | -- | -- | pending | -- |" not in markdown
    if terminal_status == "invalid":
        assert "## Rejected EMPIAR-10202 scoring evidence" in markdown
        assert "not admitted as a two-engine scientific result" in markdown
        assert "Validation failures: `k_is_one`." in markdown
        assert "Both engines are represented by the submitted target evidence" not in markdown
    else:
        assert "## Submitted EMPIAR-10202 scoring evidence" in markdown
        assert f"terminal status is `{terminal_status}`" in markdown


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("k", 2, "k_is_one"),
        ("autonomous_refinement", False, "autonomous_refinement"),
        ("fixed_poses", True, "fixed_poses_disabled"),
        ("local_only", True, "local_only_disabled"),
        ("forced_final_after_nonconvergence", True, "forced_final_disabled"),
        ("prepared_star_sha256", "0" * 64, "prepared_star_sha256"),
        (
            "prepared_star_preserves_source_half_assignment",
            False,
            "prepared_star_half_assignment",
        ),
        ("poses_pkl_sha256", "0" * 64, "poses_pkl_sha256"),
        ("ctf_pkl_sha256", "0" * 64, "ctf_pkl_sha256"),
        ("half_assignment_sha256", "0" * 64, "half_assignment_sha256"),
        (
            "initial_reference_canonical_array_sha256",
            "0" * 64,
            "initial_reference_canonical_array_sha256",
        ),
    ],
)
def test_refinement_or_input_contract_mismatch_is_invalid(
    tmp_path: Path,
    field: str,
    value: object,
    failure: str,
) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    evidence = json.loads(evidence_path.read_text())
    evidence["input_contract"][field] = value
    evidence_path.write_text(json.dumps(evidence))

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    assert result["primary_pass"] is False
    assert failure in result["provenance_failures"]


def test_symmetry_label_and_operator_mismatch_is_invalid(tmp_path: Path) -> None:
    scorecard = _frozen_target_scorecard()
    evidence_path = _write_evidence(tmp_path, scorecard)
    evidence = json.loads(evidence_path.read_text())
    evidence["input_contract"]["symmetry"]["relion_label"] = "I2"
    evidence["input_contract"]["symmetry"]["operators_sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(evidence))

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    assert result["primary_pass"] is False
    assert "relion_symmetry_label" in result["provenance_failures"]
    assert "symmetry_operators_sha256" in result["provenance_failures"]


def test_unfrozen_target_contract_is_fail_closed(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate_scorecard()
    target = _target(scorecard)
    target["input_contract"]["prepared_star_sha256"] = None
    target["pending_contract_fields"] = ["input_contract.prepared_star_sha256"]
    evidence_path = _write_evidence(tmp_path, scorecard)

    result = MODULE.score_case_evidence(scorecard, evidence_path)

    assert result["status"] == "invalid"
    assert result["primary_pass"] is False
    assert "scorecard_contract_not_frozen" in result["provenance_failures"]


def test_scorecard_rejects_image_set_7(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate_scorecard()
    target = _target(scorecard)
    target["input_contract"]["source_star"]["path"] = target["input_contract"]["source_star"]["path"].replace(
        "/06_Final_Stack/", "/07_Final_Stack/"
    )
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="target STAR is not set 6"):
        MODULE.load_and_validate_scorecard(path)


def test_scorecard_rejects_old_star_that_rerandomized_halves(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate_scorecard()
    target = _target(scorecard)
    target["input_contract"]["prepared_star_sha256"] = MODULE.REJECTED_PREPARED_STAR_SHA256
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="explicitly rejected"):
        MODULE.load_and_validate_scorecard(path)
