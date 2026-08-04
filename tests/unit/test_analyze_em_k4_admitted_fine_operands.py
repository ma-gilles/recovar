from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.analyze_em_k4_admitted_fine_operands import (
    build_report,
    validate_admissions,
)


def _native_admission(gpu_uuid: str = "GPU-fixed") -> dict:
    return {
        "schema": "recovar.em_k4_native_class1_fine_operand_admission.v1",
        "status": "complete",
        "accepted": True,
        "gpu_uuid": gpu_uuid,
        "fixed_metric": {
            "passing": 7,
            "evaluated": 7,
            "gates": {f"native-{index}": True for index in range(7)},
        },
        "scope": {
            "iteration": 2,
            "source_row_zero_based": 53722,
            "stack_index_one_based": 53723,
            "class_one_based": 1,
            "rotation_local": 1790,
            "candidate_count": 96,
        },
        "target_class_local_operand_use_allowed": True,
        "allclass_cross_engine_attribution_allowed": False,
        "scorecard_change_admissible": False,
        "correlation_used": False,
    }


def _recovar_admission(gpu_uuid: str = "GPU-fixed") -> dict:
    return {
        "schema": "recovar.em_k4_contribution_repeatability.v1",
        "status": "complete",
        "accepted": True,
        "gpu_uuid": gpu_uuid,
        "fixed_metric": {
            "passing": 3,
            "evaluated": 3,
            "gates": {f"recovar-{index}": True for index in range(3)},
        },
        "cross_engine_attribution_allowed": True,
        "cross_engine_scope": "iteration2_half1_source53722_class1_only",
        "allclass_cross_engine_attribution_allowed": False,
        "scorecard_change_admissible": False,
        "correlation_used": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.unit
def test_accepts_only_same_gpu_target_local_admissions() -> None:
    validate_admissions(_native_admission(), _recovar_admission())


@pytest.mark.unit
def test_rejects_native_allclass_promotion() -> None:
    native = deepcopy(_native_admission())
    native["allclass_cross_engine_attribution_allowed"] = True

    with pytest.raises(ValueError, match="fixed target-local 7/7"):
        validate_admissions(native, _recovar_admission())


@pytest.mark.unit
def test_rejects_cross_device_join() -> None:
    with pytest.raises(ValueError, match="same physical GPU"):
        validate_admissions(
            _native_admission("GPU-native"),
            _recovar_admission("GPU-recovar"),
        )


@pytest.mark.unit
def test_build_report_uses_only_admitted_artifacts(tmp_path: Path) -> None:
    native_operand = tmp_path / "native.bin"
    contribution = tmp_path / "contribution.npz"
    reference = tmp_path / "reference.mrc"
    native_operand.write_bytes(b"native")
    contribution.write_bytes(b"recovar")
    reference.write_bytes(b"reference")

    native = _native_admission()
    native["artifacts"] = {
        "fine_operand": {
            "path": str(native_operand.resolve()),
            "sha256": _sha256(native_operand),
        }
    }
    recovar = _recovar_admission()
    recovar["comparisons"] = {
        "contribution": {
            "reference": {
                "path": str(contribution.resolve()),
                "sha256": _sha256(contribution),
            }
        }
    }
    native_path = tmp_path / "native.json"
    recovar_path = tmp_path / "recovar.json"
    native_path.write_text(json.dumps(native))
    recovar_path.write_text(json.dumps(recovar))

    observed = {}

    def compare_fn(capture, contribution_path, reference_path, **kwargs):
        observed.update(
            capture=capture,
            contribution=contribution_path,
            reference=reference_path,
            kwargs=kwargs,
        )
        return {
            "schema": "k4_relion_recovar_fine_operand_comparison_v8",
            "status": "complete",
            "classification": (
                "shifted_image_has_largest_centered_fine_operand_"
                "single_substitution_effect"
            ),
            "classification_basis": "centered_raw_diff2",
            "capture_validation": {
                "status": "accepted",
                "candidate_count": 96,
                "exact_production_replay_count": 96,
            },
            "scope": {
                "original_index_zero_based": 53722,
                "stack_index_one_based": 53723,
                "relion_rotation_local": 1790,
                "recovar_global_rotation": 4446,
            },
            "candidates": [{} for _ in range(96)],
        }

    report = build_report(
        native_admission_path=native_path,
        recovar_admission_path=recovar_path,
        reference_path=reference,
        particle_diameter_angstrom=380.0,
        mask_edge_pixels=5.0,
        compare_fn=compare_fn,
    )

    assert observed["capture"] == native_operand
    assert observed["contribution"] == contribution
    assert observed["reference"] == reference
    assert observed["kwargs"]["recovar_global_rotation"] == 4446
    assert report["classification_basis"] == "centered_raw_diff2"
    assert report["scope"]["candidate_count"] == 96
    assert report["scope"]["allclass_cross_engine_attribution_allowed"] is False
    assert report["scorecard_change_admissible"] is False
    assert report["correlation_used"] is False


@pytest.mark.unit
def test_rejects_artifact_hash_drift(tmp_path: Path) -> None:
    operand = tmp_path / "operand.bin"
    contribution = tmp_path / "contribution.npz"
    reference = tmp_path / "reference.mrc"
    operand.write_bytes(b"operand")
    contribution.write_bytes(b"contribution")
    reference.write_bytes(b"reference")
    native = _native_admission()
    native["artifacts"] = {
        "fine_operand": {"path": str(operand.resolve()), "sha256": "wrong"}
    }
    recovar = _recovar_admission()
    recovar["comparisons"] = {
        "contribution": {
            "reference": {
                "path": str(contribution.resolve()),
                "sha256": _sha256(contribution),
            }
        }
    }
    native_path = tmp_path / "native.json"
    recovar_path = tmp_path / "recovar.json"
    native_path.write_text(json.dumps(native))
    recovar_path.write_text(json.dumps(recovar))

    with pytest.raises(ValueError, match="native fine operand hash changed"):
        build_report(
            native_admission_path=native_path,
            recovar_admission_path=recovar_path,
            reference_path=reference,
            particle_diameter_angstrom=380.0,
            mask_edge_pixels=5.0,
            compare_fn=lambda *_args, **_kwargs: {},
        )


@pytest.mark.unit
def test_rejects_single_candidate_comparison(tmp_path: Path) -> None:
    operand = tmp_path / "operand.bin"
    contribution = tmp_path / "contribution.npz"
    reference = tmp_path / "reference.mrc"
    operand.write_bytes(b"operand")
    contribution.write_bytes(b"contribution")
    reference.write_bytes(b"reference")
    native = _native_admission()
    native["artifacts"] = {
        "fine_operand": {
            "path": str(operand.resolve()),
            "sha256": _sha256(operand),
        }
    }
    recovar = _recovar_admission()
    recovar["comparisons"] = {
        "contribution": {
            "reference": {
                "path": str(contribution.resolve()),
                "sha256": _sha256(contribution),
            }
        }
    }
    native_path = tmp_path / "native.json"
    recovar_path = tmp_path / "recovar.json"
    native_path.write_text(json.dumps(native))
    recovar_path.write_text(json.dumps(recovar))

    def compare_fn(*_args, **_kwargs):
        return {
            "schema": "k4_relion_recovar_fine_operand_comparison_v8",
            "status": "complete",
            "classification": "uninformative",
            "classification_basis": "raw_diff2",
            "capture_validation": {
                "status": "accepted",
                "candidate_count": 1,
                "exact_production_replay_count": 1,
            },
            "scope": {},
            "candidates": [{}],
        }

    with pytest.raises(ValueError, match="multi-candidate scope"):
        build_report(
            native_admission_path=native_path,
            recovar_admission_path=recovar_path,
            reference_path=reference,
            particle_diameter_angstrom=380.0,
            mask_edge_pixels=5.0,
            compare_fn=compare_fn,
        )
