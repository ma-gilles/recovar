#!/usr/bin/env python3
"""Compare matched case-26 default and double cross-half M-step arms.

The two arms must use the same clean source head, fixed scorecard fixture,
autonomous runtime contract, and strict FSC/topology auditor.  The only
accepted science-contract difference is
``RECOVAR_RELION_X_HALF_MSTEP_DOUBLE=0`` versus ``1``.  Map quality is
evaluated with signed shellwise FSC/FSC-AUC only; correlation is forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.summarize_em_completion_bench import (
    _load_recovar_volume,
    normalized_fsc_auc,
    shell_fsc,
)

SCHEMA = "em-k1-case26-xhalf-precision-factorial-v1"
CASE_NAME = "26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80"
FIXTURE_MANIFEST_SHA256 = "422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee"
NUMBERED_ITERATIONS = 11
CROSS_FSC_AUC_MIN = 0.995
GT_DELTA_MIN = -0.002
CLASSIFICATION = "double_xhalf_mstep_introduces_numbered_failures_and_worsens_case26_final_parity_on_matched_head"
CONTRACT_KEYS = (
    "HEAD",
    "EM_K1_MATRIX_RUN_RELION",
    "EM_K1_MATRIX_TRAJECTORY_MODE",
    "EM_K1_MATRIX_SCORECARD_MODE",
    "EM_K1_MATRIX_FIXTURE_MANIFEST_SHA256",
    "EM_K1_MATRIX_MAX_ITER",
    "K1_IMAGE_BATCH_SIZE",
    "K1_ROTATION_BLOCK_SIZE",
    "STREAMING_CHUNK_SIZE",
    "RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION",
    "RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE",
    "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT",
    "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT",
    "RECOVAR_SAVE_INTERMEDIATES_DIR",
    "RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED",
)
BOOKKEEPING_KEYS = frozenset(
    {
        "SUBMISSION_GIT_PROVENANCE_DIR",
        "SCRATCH_DIR",
        "RUNTIME_ROOT",
        "EM_K1_MATRIX_VENV",
        "PIXI_PY",
        "SETUP_JOB_ID",
        "CASE_JOB_IDS",
        "SUMMARY_JOB_ID",
        "CASE_TABLE",
        "CUDA_LIB",
    }
)
PRECISION_KEY = "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _submission_env(path: Path) -> dict[str, str]:
    _require(path.is_file(), f"missing submission environment: {path}")
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip("'\"")
    return result


def validate_matched_contract(
    control: dict[str, str],
    double: dict[str, str],
) -> dict[str, Any]:
    """Fail closed unless the fixed science contracts differ only by precision."""

    missing = [key for key in (*CONTRACT_KEYS, PRECISION_KEY) if key not in control or key not in double]
    _require(not missing, f"submission contracts are missing keys: {missing}")
    _require(
        control.keys() == double.keys(),
        "submission contracts declare different environment keys",
    )
    matched_keys = sorted(control.keys() - BOOKKEEPING_KEYS - {PRECISION_KEY})
    differences = {
        key: {"control": control[key], "double": double[key]} for key in matched_keys if control[key] != double[key]
    }
    _require(
        not differences,
        f"matched science contract differs outside precision arm: {differences}",
    )
    _require(
        control[PRECISION_KEY] == "0" and double[PRECISION_KEY] == "1",
        "precision factorial must be control=0 and double=1",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", control["HEAD"]) is not None,
        "matched source HEAD is not a full Git SHA",
    )
    _require(
        control["EM_K1_MATRIX_RUN_RELION"] == "1"
        and control["EM_K1_MATRIX_TRAJECTORY_MODE"] == "autonomous"
        and control["EM_K1_MATRIX_SCORECARD_MODE"] == "1",
        "matched arms are not autonomous scorecard-mode RELION pairs",
    )
    _require(
        control["EM_K1_MATRIX_FIXTURE_MANIFEST_SHA256"] == FIXTURE_MANIFEST_SHA256,
        "fixture manifest SHA-256 differs from the frozen contract",
    )
    _require(
        control["RECOVAR_FINAL_ALL_DATA_GRID_CORRECT"] == "0",
        "final all-data grid correction is not explicitly off",
    )
    return {
        "matched_source_head": control["HEAD"],
        "matched_keys": matched_keys,
        "only_difference": {
            "name": PRECISION_KEY,
            "control": "0",
            "double": "1",
        },
    }


def classify_precision_effect(
    *,
    control_numbered_failures: int,
    double_numbered_failures: int,
    control_final_cross_fsc_auc: float,
    double_final_cross_fsc_auc: float,
) -> str:
    """Classify fixed strict-gate counts and the signed final parity delta."""

    final_delta = double_final_cross_fsc_auc - control_final_cross_fsc_auc
    if final_delta < 0.0 and double_numbered_failures > control_numbered_failures:
        return CLASSIFICATION
    if final_delta < 0.0:
        return "double_xhalf_mstep_worsens_case26_final_parity_only"
    if final_delta > 0.0 and double_numbered_failures < control_numbered_failures:
        return "double_xhalf_mstep_removes_numbered_failures_and_improves_case26_final_parity"
    if final_delta > 0.0:
        return "double_xhalf_mstep_improves_case26_final_parity_only"
    if double_numbered_failures == control_numbered_failures:
        return "double_xhalf_mstep_is_exactly_neutral_on_fixed_metrics"
    return "double_xhalf_mstep_has_mixed_case26_precision_effect"


def _audit(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == "em_k1_fsc_trajectory_audit_v2",
        f"unexpected FSC audit schema: {path}",
    )
    _require(
        report.get("thresholds")
        == {
            "merged_cross_engine_fsc_auc_min": CROSS_FSC_AUC_MIN,
            "recovar_minus_relion_merged_gt_fsc_auc_min": GT_DELTA_MIN,
        },
        f"FSC audit thresholds differ from the fixed contract: {path}",
    )
    _require(
        report.get("gt_sign_policy", {}).get("used") == "signed",
        f"FSC audit did not use signed GT curves: {path}",
    )
    _require(
        report.get("topology_failures") == [],
        f"FSC audit contains topology failures: {path}",
    )
    numbered = report.get("numbered_iterations")
    _require(
        isinstance(numbered, list) and len(numbered) == NUMBERED_ITERATIONS,
        f"FSC audit does not contain {NUMBERED_ITERATIONS} numbered rows: {path}",
    )
    return report


def _numbered_failure_count(report: dict[str, Any]) -> int:
    failures = 0
    for row in report["numbered_iterations"]:
        cross = float(row["cross_engine"]["merged"]["fsc_auc"])
        gt_delta = float(row["merged_gt_fsc_auc_delta"])
        failures += int(cross < CROSS_FSC_AUC_MIN or gt_delta < GT_DELTA_MIN)
    return failures


def _arm_paths(case_root: Path) -> dict[str, Any]:
    case_root = case_root.resolve()
    _require(case_root.name == CASE_NAME, f"unexpected case root: {case_root}")
    _require(case_root.parent.name == "cases", f"case root is not under cases/: {case_root}")
    run_root = case_root.parent.parent
    config_path = case_root / "case_config.json"
    fixture_path = case_root / "data" / "fixture_materialization.json"
    audit_path = case_root / "trajectory_analysis" / "k1_fsc_trajectory.json"
    return {
        "case_root": case_root,
        "run_root": run_root,
        "submission_env": run_root / "submission.env",
        "case_config": config_path,
        "fixture_materialization": fixture_path,
        "audit": audit_path,
        "recovar": case_root / "recovar",
    }


def _map_metric(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    key: str,
    shellwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    _require(
        curve.size > 1 and np.count_nonzero(np.isfinite(curve[1:])) > 0,
        f"no finite non-DC FSC shells for {key}",
    )
    shellwise[key] = curve
    return {
        "fsc_auc": float(normalized_fsc_auc(curve)),
        "n_shells": int(curve.size),
        "shellwise_key": key,
    }


def _load_pair(
    control_recovar: Path,
    double_recovar: Path,
    *,
    prefix: str,
    shellwise: dict[str, np.ndarray],
    hashes: dict[str, str],
) -> dict[str, Any]:
    volumes: dict[str, dict[str, np.ndarray]] = {}
    for arm, root in (("control", control_recovar), ("double", double_recovar)):
        volumes[arm] = {}
        for label, half in (("half1", 1), ("half2", 2)):
            if prefix == "final":
                path = root / f"final_half{half}.mrc"
            else:
                path = root / "intermediates" / f"{prefix}_half{half}_reg.mrc"
            _require(path.is_file(), f"missing map: {path}")
            hashes[str(path.resolve())] = _sha256(path)
            volume = np.asarray(_load_recovar_volume(path), dtype=np.float64)
            _require(
                volume.ndim == 3 and np.all(np.isfinite(volume)),
                f"invalid map: {path}",
            )
            volumes[arm][label] = volume
        volumes[arm]["merged"] = 0.5 * (volumes[arm]["half1"] + volumes[arm]["half2"])
    return {
        label: _map_metric(
            volumes["control"][label],
            volumes["double"][label],
            key=f"{prefix}_control_vs_double_{label}",
            shellwise=shellwise,
        )
        for label in ("half1", "half2", "merged")
    }


def _selected_audit_metrics(report: dict[str, Any]) -> dict[str, Any]:
    numbered = [
        {
            "relion_iteration": int(row["relion_iteration"]),
            "merged_cross_engine_fsc_auc": float(row["cross_engine"]["merged"]["fsc_auc"]),
            "merged_gt_fsc_auc_delta": float(row["merged_gt_fsc_auc_delta"]),
        }
        for row in report["numbered_iterations"]
    ]
    return {
        "status": report["status"],
        "numbered": numbered,
        "numbered_failure_count": _numbered_failure_count(report),
        "final_merged_cross_engine_fsc_auc": float(report["final"]["cross_engine"]["merged"]["fsc_auc"]),
        "final_merged_gt_fsc_auc_delta": float(report["final"]["merged_gt_fsc_auc_delta"]),
    }


def build_report(
    *,
    control_case_root: Path,
    double_case_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the matched-head precision-factorial report."""

    paths = {
        "control": _arm_paths(control_case_root),
        "double": _arm_paths(double_case_root),
    }
    configs = {arm: json.loads(values["case_config"].read_text()) for arm, values in paths.items()}
    _require(configs["control"] == configs["double"], "case configurations differ")
    fixtures = {arm: json.loads(values["fixture_materialization"].read_text()) for arm, values in paths.items()}
    _require(
        fixtures["control"].get("manifest_sha256") == FIXTURE_MANIFEST_SHA256
        and fixtures["double"].get("manifest_sha256") == FIXTURE_MANIFEST_SHA256,
        "materialized fixtures do not bind the frozen manifest",
    )
    control_files = fixtures["control"].get("files")
    double_files = fixtures["double"].get("files")
    _require(
        isinstance(control_files, list)
        and isinstance(double_files, list)
        and [(row["name"], row["sha256"], row["size"]) for row in control_files]
        == [(row["name"], row["sha256"], row["size"]) for row in double_files],
        "materialized fixture bytes differ between arms",
    )

    environments = {arm: _submission_env(values["submission_env"]) for arm, values in paths.items()}
    contract = validate_matched_contract(
        environments["control"],
        environments["double"],
    )
    audits = {arm: _audit(values["audit"]) for arm, values in paths.items()}
    selected = {arm: _selected_audit_metrics(report) for arm, report in audits.items()}

    shellwise: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    numbered_map_identity = []
    for index in range(NUMBERED_ITERATIONS):
        numbered_map_identity.append(
            {
                "relion_iteration": index + 1,
                "control_vs_double": _load_pair(
                    paths["control"]["recovar"],
                    paths["double"]["recovar"],
                    prefix=f"it{index:03d}",
                    shellwise=shellwise,
                    hashes=hashes,
                ),
            }
        )
    final_map_identity = _load_pair(
        paths["control"]["recovar"],
        paths["double"]["recovar"],
        prefix="final",
        shellwise=shellwise,
        hashes=hashes,
    )

    control_final = selected["control"]["final_merged_cross_engine_fsc_auc"]
    double_final = selected["double"]["final_merged_cross_engine_fsc_auc"]
    classification = classify_precision_effect(
        control_numbered_failures=selected["control"]["numbered_failure_count"],
        double_numbered_failures=selected["double"]["numbered_failure_count"],
        control_final_cross_fsc_auc=control_final,
        double_final_cross_fsc_auc=double_final,
    )
    input_hashes = {
        str(path.resolve()): _sha256(path)
        for values in paths.values()
        for path in (
            values["submission_env"],
            values["case_config"],
            values["fixture_materialization"],
            values["audit"],
        )
    }
    input_hashes.update(hashes)
    return (
        {
            "schema": SCHEMA,
            "status": "complete",
            "classification_ready": True,
            "classification": classification,
            "metric_policy": (
                "fixed signed shellwise FSC and normalized non-DC FSC-AUC; "
                "numbered failure means cross <0.995 or GT delta <-0.002; "
                "strict double-minus-control final delta; no fitted "
                "tolerance; no correlation"
            ),
            "fixed_contract": {
                "numbered_iterations": NUMBERED_ITERATIONS,
                "merged_cross_engine_fsc_auc_min": CROSS_FSC_AUC_MIN,
                "recovar_minus_relion_merged_gt_fsc_auc_min": GT_DELTA_MIN,
                **contract,
            },
            "arms": selected,
            "fixed_metric": {
                "control_numbered_failures": selected["control"]["numbered_failure_count"],
                "double_numbered_failures": selected["double"]["numbered_failure_count"],
                "double_minus_control_final_cross_engine_fsc_auc": (double_final - control_final),
                "double_minus_control_final_gt_fsc_auc_delta": (
                    selected["double"]["final_merged_gt_fsc_auc_delta"]
                    - selected["control"]["final_merged_gt_fsc_auc_delta"]
                ),
            },
            "control_vs_double_map_identity": {
                "numbered": numbered_map_identity,
                "final": final_map_identity,
            },
            "paths": {
                arm: {key: str(value.resolve()) for key, value in values.items() if isinstance(value, Path)}
                for arm, values in paths.items()
            },
            "artifact_sha256": input_hashes,
        },
        shellwise,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-case-root", type=Path, required=True)
    parser.add_argument("--double-case-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shellwise-npz", type=Path, required=True)
    args = parser.parse_args()
    _require(
        not args.output_json.exists(),
        f"refusing to overwrite report: {args.output_json}",
    )
    _require(
        not args.output_shellwise_npz.exists(),
        f"refusing to overwrite shellwise output: {args.output_shellwise_npz}",
    )
    report, shellwise = build_report(
        control_case_root=args.control_case_root,
        double_case_root=args.double_case_root,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_shellwise_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    np.savez_compressed(args.output_shellwise_npz, **shellwise)
    print(json.dumps(report["fixed_metric"], indent=2, sort_keys=True))
    print(report["classification"])


if __name__ == "__main__":
    main()
