#!/usr/bin/env python3
"""Fail-closed analysis for the crossed late-VDAM local donation A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import starfile

from scripts.run_local_mstep_donation_ab import SCHEMA as ARM_SCHEMA

SCHEMA = "recovar.local_mstep_donation_ab_analysis.v1"
EXPECTED_REPEATS_PER_ARM = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    return parser.parse_args()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode() + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def mrc_science_snapshot(path: Path) -> dict[str, Any]:
    """Return exact file bytes, voxels, and reconstruction-relevant headers."""

    with mrcfile.open(path, permissive=False) as handle:
        data = np.array(handle.data, copy=True)
        header = handle.header
        header_fields = {
            "nx_ny_nz": [int(header.nx), int(header.ny), int(header.nz)],
            "mode": int(header.mode),
            "nxstart_nystart_nzstart": [int(header.nxstart), int(header.nystart), int(header.nzstart)],
            "mx_my_mz": [int(header.mx), int(header.my), int(header.mz)],
            "cella": [float(header.cella.x), float(header.cella.y), float(header.cella.z)],
            "cellb": [float(header.cellb.alpha), float(header.cellb.beta), float(header.cellb.gamma)],
            "mapc_mapr_maps": [int(header.mapc), int(header.mapr), int(header.maps)],
            "dmin_dmax_dmean": [float(header.dmin), float(header.dmax), float(header.dmean)],
            "origin": [float(header.origin.x), float(header.origin.y), float(header.origin.z)],
            "ispg": int(header.ispg),
            "nsymbt": int(header.nsymbt),
            "rms": float(header.rms),
        }
    return {
        "file_sha256": _bytes_sha256(path.read_bytes()),
        "shape": list(data.shape),
        "dtype": data.dtype.str,
        "voxel_sha256": _array_sha256(data),
        "header": header_fields,
    }


def _normalize_output_path(value: str, output_prefix: Path) -> str:
    return value.replace(str(output_prefix), "<OUTPUT_PREFIX>")


def _series_snapshot(series: Any, output_prefix: Path) -> dict[str, Any]:
    values = series.to_numpy()
    if values.dtype.kind in "biufcMm":
        return {
            "kind": "numeric",
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "sha256": _array_sha256(values),
        }
    normalized = [
        None if value is None else _normalize_output_path(str(value), output_prefix) for value in values.tolist()
    ]
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "kind": "string",
        "shape": list(values.shape),
        "sha256": _bytes_sha256(encoded),
        "values": normalized,
    }


def _scalar_snapshot(value: Any, output_prefix: Path) -> dict[str, Any]:
    """Snapshot one scalar from a non-loop STAR block without coercion loss."""

    array = np.asarray(value)
    if array.ndim == 0 and array.dtype.kind in "biufcMm":
        return {
            "kind": "numeric_scalar",
            "dtype": array.dtype.str,
            "sha256": _array_sha256(array),
        }
    if value is None or array.ndim == 0:
        normalized = None if value is None else _normalize_output_path(str(value), output_prefix)
        encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()
        return {
            "kind": "string_scalar",
            "sha256": _bytes_sha256(encoded),
            "value": normalized,
        }
    raise TypeError(f"STAR scalar block contains a non-scalar value: {type(value)!r}")


def star_science_snapshot(path: Path, output_prefix: Path) -> dict[str, Any]:
    """Parse every STAR block/row/column, normalizing only the output prefix."""

    document = starfile.read(path, always_dict=True)
    blocks: dict[str, Any] = {}
    for block_name, frame in document.items():
        if isinstance(frame, dict):
            keys = [str(key) for key in frame]
            blocks[str(block_name)] = {
                "kind": "scalar_mapping",
                "entry_count": len(frame),
                "keys": keys,
                "values": {str(key): _scalar_snapshot(frame[key], output_prefix) for key in frame},
            }
            continue
        columns = [str(column) for column in frame.columns]
        blocks[str(block_name)] = {
            "kind": "table",
            "row_count": int(len(frame)),
            "columns": columns,
            "series": {column: _series_snapshot(frame[column], output_prefix) for column in columns},
        }
    return {"block_order": [str(name) for name in document], "blocks": blocks}


def science_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    outputs = report["warm"]["science_outputs"]
    prefix = Path(outputs["output_prefix"]["path"])
    snapshot = {
        "class_map": mrc_science_snapshot(Path(outputs["class_map"]["path"])),
        "stars": {
            name: star_science_snapshot(Path(outputs[name]["path"]), prefix) for name in ("data_star", "model_star")
        },
    }
    return snapshot


def _without_timings(value: Any) -> Any:
    """Keep declared discrete/science profile values while excluding timings."""

    if isinstance(value, dict):
        return {
            key: _without_timings(item)
            for key, item in value.items()
            if not key.endswith("_s")
            and "time" not in key
            and key not in {"meta_path", "meta_sha256", "continuation_path"}
        }
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


def declared_science_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    warm = report["warm"]
    return {
        "schedule": warm["schedule"],
        "halfset_profiles": _without_timings(warm["halfset_profiles"]),
        "sparse_pass2_profile": _without_timings(warm["sparse_pass2_profile"]),
    }


def _load_samples(path: Path, start_ns: int, end_ns: int) -> dict[str, Any]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("unix_ns"):
            continue
        timestamp, memory_mib = line.split("\t")
        timestamp_ns = int(timestamp)
        if start_ns <= timestamp_ns <= end_ns:
            rows.append((timestamp_ns, int(memory_mib)))
    if not rows:
        raise RuntimeError(f"no nvidia-smi samples overlap the warm interval: {path}")
    memory = [row[1] for row in rows]
    return {
        "sample_count": len(rows),
        "first_mib": memory[0],
        "min_mib": min(memory),
        "peak_mib": max(memory),
        "first_unix_ns": rows[0][0],
        "last_unix_ns": rows[-1][0],
    }


def _profile_metrics(report: dict[str, Any]) -> dict[str, float]:
    warm = report["warm"]
    iteration = warm["iteration_profile"]
    halfsets = warm["halfset_profiles"].values()
    return {
        "e2e_wall_s": float(warm["wall_s"]),
        "expectation_s": float(iteration["expectation_time_s"]),
        "mstep_s": float(iteration["mstep_time_s"]),
        "big_jit_bucket_s": float(sum(float(profile["big_jit_bucket_s"]) for profile in halfsets)),
        "outer_backproject_y_s": float(sum(float(profile["local_backproject_y_s"]) for profile in halfsets)),
        "final_accumulator_s": float(sum(float(profile["local_final_accumulator_s"]) for profile in halfsets)),
    }


def _jax_peak_bytes(report: dict[str, Any]) -> int:
    if report["warm"].get("jax_memory_scope") != "whole_process_cumulative_not_warm_isolated":
        raise RuntimeError("arm report does not label cumulative JAX peak scope")
    stats = report["warm"]["jax_memory_after"]
    if "peak_bytes_in_use" not in stats:
        raise RuntimeError(f"JAX memory stats lack peak_bytes_in_use: {sorted(stats)}")
    return int(stats["peak_bytes_in_use"])


def _median_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {name: float(statistics.median(value[name] for value in values)) for name in values[0]}


def _ratio(candidate: float, control: float) -> float:
    if control <= 0.0:
        raise RuntimeError(f"control metric must be positive, got {control}")
    return float(candidate / control)


def _load_runs(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(root.glob("runs/repeat-*/**/donation_arm_summary.json"))
    runs = [(path, json.loads(path.read_text())) for path in paths]
    if len(runs) != 2 * EXPECTED_REPEATS_PER_ARM:
        raise RuntimeError(f"expected {2 * EXPECTED_REPEATS_PER_ARM} arm reports, got {len(runs)}")
    return runs


def analyze(
    root: Path,
    *,
    expected_repo_head: str,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    runs = _load_runs(root)
    by_arm: dict[str, list[dict[str, Any]]] = {"control": [], "donated": []}
    run_rows = []
    reference_science = None
    reference_declared = None
    reference_contract_source = None
    program_tables: dict[str, dict[str, dict[str, Any]]] = {
        "control": {},
        "donated": {},
    }
    for path, report in runs:
        if report.get("schema") != ARM_SCHEMA or report.get("passed") is not True:
            raise RuntimeError(f"invalid arm report: {path}")
        if report.get("git_head") != expected_repo_head:
            raise RuntimeError(f"arm report source head drifted: {path}")
        if report.get("gpu_uuid") != expected_gpu_uuid:
            raise RuntimeError(f"arm report GPU drifted: {path}")
        arm = str(report["arm"])
        if arm not in by_arm:
            raise RuntimeError(f"unknown arm in {path}: {arm}")
        contract = report["compiled_local_programs"]["contract"]
        if not contract["same_wrapped_numeric_source"]:
            raise RuntimeError(f"arm does not use the shared numeric source: {path}")
        source_sha = contract["numeric_source_sha256"]
        if reference_contract_source is None:
            reference_contract_source = source_sha
        elif source_sha != reference_contract_source:
            raise RuntimeError("numeric source SHA differs across arms")

        science = science_snapshot(report)
        declared = declared_science_snapshot(report)
        if reference_science is None:
            reference_science = science
            reference_declared = declared
        else:
            if science != reference_science:
                raise RuntimeError(f"parsed science outputs are not exact: {path}")
            if declared != reference_declared:
                raise RuntimeError(f"declared discrete/science metadata is not exact: {path}")

        run_dir = path.parents[1]
        samples = _load_samples(
            run_dir / "nvidia_smi.tsv",
            int(report["warm"]["timed_start_unix_ns"]),
            int(report["warm"]["timed_end_unix_ns"]),
        )
        metrics = _profile_metrics(report)
        row = {
            "report": str(path.resolve()),
            "arm": arm,
            "metrics": metrics,
            "whole_process_jax_peak_bytes": _jax_peak_bytes(report),
            "sampled_memory": samples,
            "raw_science_sha256": {
                name: value["sha256"] for name, value in report["warm"]["science_outputs"].items() if "sha256" in value
            },
        }
        run_rows.append(row)
        by_arm[arm].append(row)
        for program in report["compiled_local_programs"]["programs"]:
            prior = program_tables[arm].setdefault(program["program_key"], program)
            if program != prior:
                raise RuntimeError(f"compiled program metadata changed within {arm}")

    for arm, values in by_arm.items():
        if len(values) != EXPECTED_REPEATS_PER_ARM:
            raise RuntimeError(f"expected {EXPECTED_REPEATS_PER_ARM} {arm} runs, got {len(values)}")
    if set(program_tables["control"]) != set(program_tables["donated"]):
        raise RuntimeError("control/donated compiled program signatures differ")

    qualifying = []
    for key in sorted(program_tables["control"]):
        control = program_tables["control"][key]
        donated = program_tables["donated"][key]
        if (
            donated["mstep_relion_x_half"]
            and donated["adjoints_enabled"]
            and int(donated["accumulator_bytes"]["total"]) > 0
        ):
            accumulator_bytes = int(donated["accumulator_bytes"]["total"])
            if control["accumulator_bytes"] != donated["accumulator_bytes"]:
                raise RuntimeError("control/donated accumulator shapes or dtypes differ")
            control_alias = int(control["memory_analysis"]["alias_size_in_bytes"])
            donated_alias = int(donated["memory_analysis"]["alias_size_in_bytes"])
            qualifying.append(
                {
                    "program_key": key,
                    "accumulator_bytes": accumulator_bytes,
                    "control_alias_bytes": control_alias,
                    "donated_alias_bytes": donated_alias,
                    "alias_gain_bytes": donated_alias - control_alias,
                }
            )
    if not qualifying:
        raise RuntimeError("gate observed no production-sized RELION x-half local program with enabled adjoints")
    alias_contract_passed = any(row["alias_gain_bytes"] >= row["accumulator_bytes"] for row in qualifying)

    arm_summary = {}
    for arm, values in by_arm.items():
        arm_summary[arm] = {
            "median_metrics": _median_metrics([value["metrics"] for value in values]),
            "median_whole_process_jax_peak_bytes": int(
                statistics.median(value["whole_process_jax_peak_bytes"] for value in values)
            ),
            "median_sampled_peak_mib": float(
                statistics.median(value["sampled_memory"]["peak_mib"] for value in values)
            ),
        }
    ratios = {
        name: _ratio(
            arm_summary["donated"]["median_metrics"][name],
            arm_summary["control"]["median_metrics"][name],
        )
        for name in arm_summary["control"]["median_metrics"]
    }
    ratios["jax_whole_process_peak"] = _ratio(
        arm_summary["donated"]["median_whole_process_jax_peak_bytes"],
        arm_summary["control"]["median_whole_process_jax_peak_bytes"],
    )
    ratios["sampled_peak"] = _ratio(
        arm_summary["donated"]["median_sampled_peak_mib"],
        arm_summary["control"]["median_sampled_peak_mib"],
    )
    material_runtime_win = bool(ratios["e2e_wall_s"] <= 0.90)
    material_stage_win = bool(ratios["expectation_s"] <= 0.90 or ratios["big_jit_bucket_s"] <= 0.90)
    material_memory_win = bool(ratios["sampled_peak"] <= 0.95)
    no_material_regression = bool(
        ratios["e2e_wall_s"] <= 1.10
        and ratios["expectation_s"] <= 1.10
        and ratios["big_jit_bucket_s"] <= 1.10
        and ratios["sampled_peak"] <= 1.10
    )
    passed = bool(alias_contract_passed and no_material_regression)
    return {
        "schema": SCHEMA,
        "classification": "diagnostic_performance_only",
        "passed": passed,
        "same_source_exact_science_outputs": True,
        "donation_specific_exact_science_contract": True,
        "same_declared_discrete_science_metadata": True,
        "arithmetic_changed": False,
        "broader_optimized_arithmetic_policy": {
            "stable_repeat_bounded_noise_allowed": True,
            "directional_bias_allowed": False,
            "iteration_amplified_drift_allowed": False,
            "exact_discrete_choices_and_trajectory_basin_required": True,
            "material_final_quality_loss_allowed": False,
            "material_end_to_end_runtime_gain_required": True,
        },
        "production_xhalf_adjoint_program_observed": True,
        "alias_contract_passed": alias_contract_passed,
        "no_material_regression": no_material_regression,
        "material_runtime_win": material_runtime_win,
        "material_stage_win": material_stage_win,
        "material_memory_win": material_memory_win,
        "speed_claim_allowed": bool(passed and material_runtime_win),
        "memory_claim_allowed": bool(passed and material_memory_win),
        "default_promotion_allowed": False,
        "repo_head": expected_repo_head,
        "gpu_uuid": expected_gpu_uuid,
        "numeric_source_sha256": reference_contract_source,
        "qualifying_programs": qualifying,
        "arm_summary": arm_summary,
        "donated_over_control_ratios": ratios,
        "runs": run_rows,
    }


def main() -> int:
    args = _parse_args()
    payload = analyze(
        args.root.resolve(strict=True),
        expected_repo_head=args.expected_repo_head,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
