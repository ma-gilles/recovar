from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.analyze_em_k4_native_target_local_boundary import (
    BOUNDARY_ORDER,
    class_stage_exact,
    validate_admissions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _target_admission() -> dict:
    gates = {f"gate-{index}": True for index in range(32)}
    return {
        "schema": "recovar-k4-native-target-artifact-repeatability-v1",
        "status": "complete",
        "accepted": True,
        "target_local_artifact_use_allowed": True,
        "allclass_cross_engine_attribution_allowed": False,
        "scorecard_change_admissible": False,
        "correlation_used": False,
        "fixed_metric": {"passing": 32, "evaluated": 32, "gates": gates},
        "classes": [
            {"class_one_based": value, "accepted": True} for value in (2, 3, 4)
        ],
    }


def _recovar_repeatability() -> dict:
    gates = {f"gate-{index}": True for index in range(9)}
    return {
        "schema": "recovar.em_k4_allclass_recovar_repeatability.v1",
        "status": "complete",
        "accepted": True,
        "first_unequal_group": "all_observed_pass2_fields_exact",
        "scorecard_change_admissible": False,
        "correlation_used": False,
        "fixed_metric": {"passing": 9, "evaluated": 9, "gates": gates},
    }


@pytest.mark.unit
def test_accepts_only_two_passed_one_sided_admissions() -> None:
    validate_admissions(_target_admission(), _recovar_repeatability())


@pytest.mark.unit
def test_direct_script_help_resolves_local_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/analyze_em_k4_native_target_local_boundary.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--target-admission" in result.stdout


@pytest.mark.unit
def test_rejects_native_allclass_promotion() -> None:
    target = deepcopy(_target_admission())
    target["allclass_cross_engine_attribution_allowed"] = True

    with pytest.raises(ValueError, match="did not pass 32/32"):
        validate_admissions(target, _recovar_repeatability())


@pytest.mark.unit
def test_rejects_incomplete_recovar_repeatability() -> None:
    recovar = deepcopy(_recovar_repeatability())
    recovar["fixed_metric"]["passing"] = 8

    with pytest.raises(ValueError, match="did not pass 9/9"):
        validate_admissions(_target_admission(), recovar)


@pytest.mark.unit
def test_class_stage_projection_preserves_causal_order() -> None:
    class_report = {
        "candidate_tuples": {"exact": True},
        "raw_diff2": {"bitwise_exact": False},
        "combined_class_rotation_prior": {"bitwise_exact": True},
        "translation_prior": {"bitwise_exact": True},
        "unnormalized_class_pose_log_weight": {"bitwise_exact": False},
        "joint_posterior_native_float32_vs_recovar_capture_cast_to_float32": {
            "bitwise_exact": False
        },
        "global_significant_support": {"exact": True},
    }

    stage_exact = class_stage_exact(class_report)

    assert tuple(stage_exact) == BOUNDARY_ORDER
    assert stage_exact == {
        "candidate_tuple_set": True,
        "raw_diff2": False,
        "combined_class_rotation_prior": True,
        "translation_prior": True,
        "unnormalized_class_pose_log_weight": False,
        "joint_class_pose_normalization": False,
        "global_significant_support": True,
    }
