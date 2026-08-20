#!/usr/bin/env python3
"""Audit one frozen K=1 VDAM/RELION InitialModel trajectory.

The audit intentionally excludes map correlation.  Every fixed checkpoint is
gated with signed shellwise FSC and normalized non-DC FSC-AUC, while run
configuration, fixture materialization, artifact topology, and physical GPU
identity are checked exactly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from recovar.data_io.starfile import read_star

if __package__:
    from scripts.summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc
else:
    from summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc


SCHEMA = "recovar.vdam_relion_fsc_trajectory_audit.v1"
SCORECARD_SCHEMA = "recovar.vdam_relion_parity_scorecard.v1"
CHECKPOINTS = (0, 1, 2, 4, 8)
COMMON_ARTIFACT_SUFFIXES = ("class001.mrc", "model.star", "data.star")


class AuditError(RuntimeError):
    """Raised when frozen inputs or required trajectory products are invalid."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} must contain a JSON object")
    return value


def _case(scorecard: dict[str, Any], case_id: str) -> dict[str, Any]:
    if scorecard.get("schema") != SCORECARD_SCHEMA:
        raise AuditError(f"unsupported scorecard schema: {scorecard.get('schema')!r}")
    matches = [row for row in scorecard.get("cases", []) if row.get("id") == case_id]
    if len(matches) != 1:
        raise AuditError(f"expected exactly one scorecard row for {case_id}, found {len(matches)}")
    return matches[0]


def _flag_value(argv: list[str], flag: str) -> str:
    positions = [idx for idx, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise AuditError(f"RELION command must contain exactly one {flag} VALUE")
    return argv[positions[0] + 1]


def _float_equal(actual: Any, expected: Any) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def _validate_run_contract(
    definition: dict[str, Any], native_options: dict[str, Any], relion_command: dict[str, Any]
) -> dict[str, Any]:
    argv = relion_command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(token, str) for token in argv):
        raise AuditError("RELION command evidence must contain a string argv list")
    required_switches = {"--grad", "--denovo_3dref", "--flatten_solvent", "--zero_mask", "--auto_sampling"}
    missing_switches = sorted(required_switches.difference(argv))
    if missing_switches:
        raise AuditError(f"RELION command is missing switches: {missing_switches}")
    if _flag_value(argv, "--grad_write_iter") != "1":
        raise AuditError("RELION --grad_write_iter must be exactly 1 for the frozen checkpoint topology")

    pairs = (
        ("nr_classes", "nr_classes", "--K"),
        ("nr_iter", "nr_iter", "--iter"),
        ("random_seed", "random_seed", "--random_seed"),
        ("tau2_fudge", "tau2_fudge", "--tau2_fudge"),
        ("healpix_order", "healpix_order", "--healpix_order"),
        ("oversampling", "oversampling", "--oversampling"),
        ("offset_range_px", "offset_range_px", "--offset_range"),
        ("offset_step_px", "offset_step_px", "--offset_step"),
        ("padding_factor", "padding_factor", "--pad"),
    )
    checked: dict[str, Any] = {}
    for definition_key, native_key, relion_flag in pairs:
        expected = definition[definition_key]
        native_actual = native_options.get(native_key)
        relion_actual = _flag_value(argv, relion_flag)
        if not _float_equal(native_actual, expected):
            raise AuditError(
                f"native option {native_key}={native_actual!r} does not match frozen {definition_key}={expected!r}"
            )
        if not _float_equal(relion_actual, expected):
            raise AuditError(
                f"RELION {relion_flag}={relion_actual!r} does not match frozen {definition_key}={expected!r}"
            )
        checked[definition_key] = expected
    return checked


def _validate_fixture(
    materialization: dict[str, Any], scorecard: dict[str, Any], definition: dict[str, Any]
) -> dict[str, Any]:
    source = scorecard["source_fixture_manifest"]
    if materialization.get("schema") != "recovar.em_k1_fixture_materialization.v1":
        raise AuditError("unsupported fixture materialization schema")
    if materialization.get("manifest_sha256") != source["sha256"]:
        raise AuditError("fixture materialization manifest digest does not match the frozen scorecard")
    if materialization.get("case_id") != definition["source_em_case_id"]:
        raise AuditError("fixture materialization case does not match the frozen VDAM case")
    files = materialization.get("files")
    if not isinstance(files, list) or not files:
        raise AuditError("fixture materialization contains no verified files")
    return {
        "source_em_case_id": definition["source_em_case_id"],
        "manifest_sha256": source["sha256"],
        "verified_file_count": len(files),
    }


def _validate_gpu(report: dict[str, Any]) -> str:
    values = [report.get(key) for key in ("physical_gpu_uuid", "relion_gpu_uuid", "recovar_gpu_uuid")]
    if not all(isinstance(value, str) and value for value in values) or len(set(values)) != 1:
        raise AuditError(f"paired run did not record one identical physical GPU UUID: {values}")
    return values[0]


def _star_column(table, names: tuple[str, ...], *, label: str) -> str:
    column = next((name for name in names if name in table.columns), None)
    if column is None:
        raise AuditError(f"{label} has none of the required columns {names}")
    return column


def _validate_iteration_one_particle_subset(
    fixture_dir: Path, recovar_dir: Path, relion_dir: Path
) -> dict[str, Any]:
    recovar_meta = _load_json(
        recovar_dir / "run_it001_recovar_meta.json",
        label="RECOVAR iteration-1 metadata",
    )
    raw_recovar_ids = recovar_meta.get("selected_particle_ids")
    if not isinstance(raw_recovar_ids, list) or not raw_recovar_ids:
        raise AuditError("RECOVAR iteration-1 metadata contains no selected_particle_ids")
    recovar_ids = np.asarray(raw_recovar_ids, dtype=np.int64)
    if np.unique(recovar_ids).size != recovar_ids.size or np.any(recovar_ids < 0):
        raise AuditError("RECOVAR iteration-1 selected_particle_ids must be unique nonnegative rows")

    fixture_star_path = fixture_dir / "particles.star"
    try:
        fixture_table, _ = read_star(str(fixture_star_path))
    except Exception as exc:
        raise AuditError(f"cannot read frozen particle STAR at {fixture_star_path}: {exc}") from exc
    if np.any(recovar_ids >= len(fixture_table)):
        raise AuditError("RECOVAR iteration-1 selected_particle_ids exceed the frozen particle table")
    fixture_image_column = _star_column(
        fixture_table,
        ("_rlnImageName", "rlnImageName"),
        label="frozen particle STAR",
    )
    recovar_images = fixture_table.iloc[recovar_ids][fixture_image_column].astype(str).to_numpy()
    if np.unique(recovar_images).size != recovar_images.size:
        raise AuditError("RECOVAR iteration-1 selected particles do not have unique image identities")

    relion_star_path = relion_dir / "run_it001_data.star"
    try:
        relion_table, _ = read_star(str(relion_star_path))
    except Exception as exc:
        raise AuditError(f"cannot read RELION iteration-1 data STAR at {relion_star_path}: {exc}") from exc
    posterior_column = _star_column(
        relion_table,
        ("_rlnMaxValueProbDistribution", "rlnMaxValueProbDistribution"),
        label="RELION iteration-1 data STAR",
    )
    relion_image_column = _star_column(
        relion_table,
        ("_rlnImageName", "rlnImageName"),
        label="RELION iteration-1 data STAR",
    )
    posterior = relion_table[posterior_column].astype(float).to_numpy()
    relion_ids = np.flatnonzero(np.isfinite(posterior) & (posterior > 0.0)).astype(np.int64)
    if relion_ids.size == 0:
        raise AuditError("RELION iteration-1 data STAR records no visited particles")
    relion_images = relion_table.iloc[relion_ids][relion_image_column].astype(str).to_numpy()
    if np.unique(relion_images).size != relion_images.size:
        raise AuditError("RELION iteration-1 visited particles do not have unique image identities")

    recovar_sorted = np.sort(recovar_images)
    relion_sorted = np.sort(relion_images)
    if not np.array_equal(recovar_sorted, relion_sorted):
        recovar_only = np.setdiff1d(recovar_sorted, relion_sorted)
        relion_only = np.setdiff1d(relion_sorted, recovar_sorted)
        raise AuditError(
            "iteration-1 particle subsets differ: "
            f"RECOVAR count={recovar_images.size}, RELION count={relion_images.size}, "
            f"RECOVAR-only={recovar_only[:10].tolist()}, RELION-only={relion_only[:10].tolist()}"
        )
    return {
        "exact": True,
        "identity": "_rlnImageName",
        "particle_count": int(relion_images.size),
        "first_image_name": str(relion_sorted[0]),
        "last_image_name": str(relion_sorted[-1]),
    }


def _artifact_paths(directory: Path, iteration: int) -> dict[str, Path]:
    prefix = f"run_it{iteration:03d}_"
    return {suffix: directory / f"{prefix}{suffix}" for suffix in COMMON_ARTIFACT_SUFFIXES}


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _map_metric(lhs: np.ndarray, rhs: np.ndarray, *, key: str, shellwise: dict[str, np.ndarray]) -> dict[str, Any]:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    if curve.size <= 1 or np.count_nonzero(np.isfinite(curve[1:])) < 1:
        raise AuditError(f"{key} produced no finite non-DC FSC shells")
    shellwise[key] = curve
    return {
        "fsc_auc": _finite(normalized_fsc_auc(curve)),
        "n_shells": int(curve.size),
        "shellwise_key": key,
    }


def audit(
    *,
    scorecard_path: Path,
    case_id: str,
    fixture_dir: Path,
    recovar_dir: Path,
    relion_dir: Path,
    paired_gpu_report_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scorecard = _load_json(scorecard_path, label="scorecard")
    case = _case(scorecard, case_id)
    definition = case["definition"]
    acceptance = scorecard["acceptance_contract"]
    if tuple(acceptance.get("required_checkpoints", ())) != CHECKPOINTS:
        raise AuditError("scorecard checkpoint contract does not match this auditor")
    if acceptance.get("correlation_used") is not False:
        raise AuditError("scorecard must explicitly forbid correlation")

    fixture = _validate_fixture(
        _load_json(fixture_dir / "fixture_materialization.json", label="fixture materialization"),
        scorecard,
        definition,
    )
    native_options = _load_json(recovar_dir / "run_native_options.json", label="native options")
    run_contract = _validate_run_contract(
        definition,
        native_options,
        _load_json(relion_dir / "relion_command.json", label="RELION command"),
    )
    physical_gpu_uuid = _validate_gpu(_load_json(paired_gpu_report_path, label="paired GPU report"))
    iteration_one_particle_subset = _validate_iteration_one_particle_subset(fixture_dir, recovar_dir, relion_dir)

    gt_path = fixture_dir / "reference_gt_relion.mrc"
    if not gt_path.is_file():
        raise AuditError(f"missing frozen GT map: {gt_path}")
    gt = _load_relion_volume(gt_path)
    shellwise: dict[str, np.ndarray] = {}
    checkpoints: list[dict[str, Any]] = []
    cross_min = float(acceptance["cross_engine_fsc_auc_min"])
    delta_min = float(acceptance["recovar_minus_relion_gt_fsc_auc_min"])

    for iteration in CHECKPOINTS:
        rec_paths = _artifact_paths(recovar_dir, iteration)
        rel_paths = _artifact_paths(relion_dir, iteration)
        missing = [str(path) for path in (*rec_paths.values(), *rel_paths.values()) if not path.is_file()]
        if missing:
            raise AuditError(f"iteration {iteration} is missing required artifacts: {missing}")
        rec_map = _load_relion_volume(rec_paths["class001.mrc"])
        rel_map = _load_relion_volume(rel_paths["class001.mrc"])
        if rec_map.shape != rel_map.shape or rec_map.shape != gt.shape:
            raise AuditError(
                f"iteration {iteration} map shapes differ: recovar={rec_map.shape}, relion={rel_map.shape}, gt={gt.shape}"
            )
        cross = _map_metric(rec_map, rel_map, key=f"it{iteration:03d}_cross_engine", shellwise=shellwise)
        rec_gt = _map_metric(rec_map, gt, key=f"it{iteration:03d}_recovar_gt", shellwise=shellwise)
        rel_gt = _map_metric(rel_map, gt, key=f"it{iteration:03d}_relion_gt", shellwise=shellwise)
        delta = float(rec_gt["fsc_auc"] - rel_gt["fsc_auc"])
        passed = bool(cross["fsc_auc"] >= cross_min and delta >= delta_min)
        checkpoints.append(
            {
                "iteration": iteration,
                "cross_engine": cross,
                "recovar_gt": rec_gt,
                "relion_gt": rel_gt,
                "recovar_minus_relion_gt_fsc_auc": delta,
                "pass": passed,
                "artifact_topology_exact": True,
            }
        )

    report = {
        "schema": SCHEMA,
        "suite_id": scorecard["suite_id"],
        "case_id": case_id,
        "source_em_case_id": definition["source_em_case_id"],
        "result": "pass" if all(row["pass"] for row in checkpoints) else "fail",
        "metric_policy": "signed shellwise FSC and normalized non-DC FSC-AUC only; correlation is not computed",
        "correlation_used": False,
        "thresholds": {
            "cross_engine_fsc_auc_min": cross_min,
            "recovar_minus_relion_gt_fsc_auc_min": delta_min,
        },
        "fixture": fixture,
        "run_contract": run_contract,
        "artifact_topology_exact": True,
        "same_physical_gpu": True,
        "physical_gpu_uuid": physical_gpu_uuid,
        "iteration_one_particle_subset": iteration_one_particle_subset,
        "checkpoints": checkpoints,
        "minimum_cross_engine_fsc_auc": min(row["cross_engine"]["fsc_auc"] for row in checkpoints),
        "minimum_recovar_minus_relion_gt_fsc_auc": min(
            row["recovar_minus_relion_gt_fsc_auc"] for row in checkpoints
        ),
    }
    return report, shellwise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--recovar-dir", type=Path, required=True)
    parser.add_argument("--relion-dir", type=Path, required=True)
    parser.add_argument("--paired-gpu-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shells-npz", type=Path, required=True)
    args = parser.parse_args(argv)
    report, shellwise = audit(
        scorecard_path=args.scorecard,
        case_id=args.case_id,
        fixture_dir=args.fixture_dir,
        recovar_dir=args.recovar_dir,
        relion_dir=args.relion_dir,
        paired_gpu_report_path=args.paired_gpu_report,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_shells_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_shells_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
