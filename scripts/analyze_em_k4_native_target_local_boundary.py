#!/usr/bin/env python3
"""Join only admitted native K=4 target artifacts to stable RECOVAR captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_em_k4_allclass_native_boundary import (
    BOUNDARY_ORDER,
    EXPECTED_CURRENT_SIZE,
    EXPECTED_ORIGINAL_INDEX,
    _class_join,
    _sha256,
    classify_first_unequal_boundary,
)

SCHEMA = "recovar.em_k4_native_target_local_boundary.v1"
TARGET_ADMISSION_SCHEMA = "recovar-k4-native-target-artifact-repeatability-v1"
RECOVAR_REPEATABILITY_SCHEMA = "recovar.em_k4_allclass_recovar_repeatability.v1"
TARGET_CLASSES = (2, 3, 4)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_admissions(
    target_admission: dict[str, Any],
    recovar_repeatability: dict[str, Any],
) -> None:
    """Fail closed unless both one-sided boundaries permit target-local use."""

    _require(
        target_admission.get("schema") == TARGET_ADMISSION_SCHEMA,
        "native target admission schema changed",
    )
    target_metric = target_admission.get("fixed_metric")
    _require(
        target_admission.get("status") == "complete"
        and target_admission.get("accepted") is True
        and target_admission.get("target_local_artifact_use_allowed") is True
        and target_admission.get("allclass_cross_engine_attribution_allowed") is False
        and target_admission.get("scorecard_change_admissible") is False
        and target_admission.get("correlation_used") is False
        and isinstance(target_metric, dict)
        and target_metric.get("passing") == 32
        and target_metric.get("evaluated") == 32
        and len(target_metric.get("gates", {})) == 32
        and all(target_metric.get("gates", {}).values()),
        "native target-local admission did not pass 32/32",
    )
    target_classes = target_admission.get("classes")
    _require(
        isinstance(target_classes, list)
        and tuple(row.get("class_one_based") for row in target_classes)
        == TARGET_CLASSES
        and all(row.get("accepted") is True for row in target_classes),
        "native target class admission changed",
    )

    _require(
        recovar_repeatability.get("schema") == RECOVAR_REPEATABILITY_SCHEMA,
        "RECOVAR repeatability schema changed",
    )
    recovar_metric = recovar_repeatability.get("fixed_metric")
    _require(
        recovar_repeatability.get("status") == "complete"
        and recovar_repeatability.get("accepted") is True
        and recovar_repeatability.get("first_unequal_group")
        == "all_observed_pass2_fields_exact"
        and recovar_repeatability.get("scorecard_change_admissible") is False
        and recovar_repeatability.get("correlation_used") is False
        and isinstance(recovar_metric, dict)
        and recovar_metric.get("passing") == 9
        and recovar_metric.get("evaluated") == 9
        and len(recovar_metric.get("gates", {})) == 9
        and all(recovar_metric.get("gates", {}).values()),
        "RECOVAR repeatability did not pass 9/9",
    )


def class_stage_exact(class_report: dict[str, Any]) -> dict[str, bool]:
    """Project one target-local class join onto the fixed causal ordering."""

    return {
        "candidate_tuple_set": bool(class_report["candidate_tuples"]["exact"]),
        "raw_diff2": bool(class_report["raw_diff2"]["bitwise_exact"]),
        "combined_class_rotation_prior": bool(
            class_report["combined_class_rotation_prior"]["bitwise_exact"]
        ),
        "translation_prior": bool(
            class_report["translation_prior"]["bitwise_exact"]
        ),
        "unnormalized_class_pose_log_weight": bool(
            class_report["unnormalized_class_pose_log_weight"]["bitwise_exact"]
        ),
        "joint_class_pose_normalization": bool(
            class_report[
                "joint_posterior_native_float32_vs_recovar_capture_cast_to_float32"
            ]["bitwise_exact"]
        ),
        "global_significant_support": bool(
            class_report["global_significant_support"]["exact"]
        ),
    }


def _admitted_artifact_paths(
    class_admission: dict[str, Any],
) -> tuple[Path, Path]:
    artifacts = class_admission.get("artifacts", {}).get("repeat", {})
    fine = artifacts.get("fine_score", {})
    bpref = artifacts.get("bpref", {})
    fine_path = Path(str(fine.get("path", "")))
    bpref_path = Path(str(bpref.get("path", "")))
    _require(fine_path.is_absolute() and bpref_path.is_absolute(), "artifact path is not absolute")
    _require(
        fine_path.is_file()
        and bpref_path.is_file()
        and fine.get("sha256") == _sha256(fine_path)
        and bpref.get("sha256") == _sha256(bpref_path),
        "admitted native target artifact hash changed",
    )
    return fine_path, bpref_path


def build_report(
    *,
    target_admission_path: Path,
    recovar_repeatability_path: Path,
) -> dict[str, Any]:
    target_admission = json.loads(target_admission_path.read_text())
    recovar_repeatability = json.loads(recovar_repeatability_path.read_text())
    validate_admissions(target_admission, recovar_repeatability)

    recovar_root = Path(recovar_repeatability["inputs"]["arm_a_root"])
    _require(recovar_root.is_absolute() and recovar_root.is_dir(), "RECOVAR arm root is invalid")
    target_rows = {
        int(row["class_one_based"]): row for row in target_admission["classes"]
    }
    classes = []
    for class_one_based in TARGET_CLASSES:
        fine_path, factor_path = _admitted_artifact_paths(
            target_rows[class_one_based]
        )
        recovar_paths = sorted(
            recovar_root.glob(
                f"pass2_orig{EXPECTED_ORIGINAL_INDEX:06d}_class"
                f"{class_one_based:03d}_cs{EXPECTED_CURRENT_SIZE:03d}.npz"
            )
        )
        _require(len(recovar_paths) == 1, "RECOVAR target class input count changed")
        class_report, _ = _class_join(
            class_one_based=class_one_based,
            factor_path=factor_path,
            fine_score_path=fine_path,
            recovar_path=recovar_paths[0],
        )
        stage_exact = class_stage_exact(class_report)
        _require(tuple(stage_exact) == BOUNDARY_ORDER, "boundary identity/order changed")
        classes.append(
            {
                "class_one_based": class_one_based,
                "classification": (
                    "first_unequal_target_local_boundary__"
                    + classify_first_unequal_boundary(stage_exact)
                ),
                "first_unequal_boundary": classify_first_unequal_boundary(
                    stage_exact
                ),
                "stage_exact": stage_exact,
                "boundary": class_report,
            }
        )

    all_target_classes_stage_exact = {
        stage: all(row["stage_exact"][stage] for row in classes)
        for stage in BOUNDARY_ORDER
    }
    first_unequal = classify_first_unequal_boundary(all_target_classes_stage_exact)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": f"first_unequal_target_local_boundary__{first_unequal}",
        "first_unequal_boundary_across_target_classes": first_unequal,
        "scope": {
            "classes_one_based": list(TARGET_CLASSES),
            "target_local_only": True,
            "allclass_cross_engine_attribution_allowed": False,
            "joint_posterior_bpref_map_parity_established": False,
        },
        "all_target_classes_stage_exact": all_target_classes_stage_exact,
        "classes": classes,
        "scorecard_change_admissible": False,
        "correlation_used": False,
        "metric_policy": (
            "admitted target-local class/rotation/translation keys and original float32 "
            "bits; no cross-particle or all-class promotion; no correlation"
        ),
        "inputs": {
            "target_admission": {
                "path": str(target_admission_path.resolve()),
                "sha256": _sha256(target_admission_path),
            },
            "recovar_repeatability": {
                "path": str(recovar_repeatability_path.resolve()),
                "sha256": _sha256(recovar_repeatability_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-admission", type=Path, required=True)
    parser.add_argument("--recovar-repeatability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        target_admission_path=args.target_admission,
        recovar_repeatability_path=args.recovar_repeatability,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "first_unequal_boundary_across_target_classes": report[
                    "first_unequal_boundary_across_target_classes"
                ],
                "all_target_classes_stage_exact": report[
                    "all_target_classes_stage_exact"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
