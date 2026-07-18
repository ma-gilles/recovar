#!/usr/bin/env python3
"""Audit a complete K=1 RECOVAR/RELION map trajectory with FSC/FSC-AUC.

RECOVAR numbered intermediates are zero-based (``it000``), while RELION
numbered products are one-based (``run_it001``).  This tool validates that
topology, computes matched half/merged cross-engine curves, and evaluates both
engines' half/merged maps against the same ground truth.  Correlation is
intentionally not computed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.summarize_em_completion_bench import (
        _load_recovar_volume,
        _load_relion_volume,
        normalized_fsc_auc,
        shell_fsc,
    )
else:
    from summarize_em_completion_bench import (
        _load_recovar_volume,
        _load_relion_volume,
        normalized_fsc_auc,
        shell_fsc,
    )

SCHEMA = "em_k1_fsc_trajectory_audit_v1"
RECOVAR_MAP_RE = re.compile(r"^it(\d{3})_half([12])_reg\.mrc$")
RELION_MAP_RE = re.compile(r"^run_it(\d{3})_half([12])_class001\.mrc$")


class AuditError(RuntimeError):
    """Raised when the trajectory is incomplete or internally inconsistent."""


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _metric(curve: np.ndarray, *, sign_invariant: bool, shellwise_key: str) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    signed = float(normalized_fsc_auc(values))
    flipped = float(normalized_fsc_auc(-values))
    if math.isfinite(signed) and math.isfinite(flipped):
        invariant = max(signed, flipped)
        best_sign = 1 if signed >= flipped else -1
    elif math.isfinite(signed):
        invariant = signed
        best_sign = 1
    elif math.isfinite(flipped):
        invariant = flipped
        best_sign = -1
    else:
        invariant = float("nan")
        best_sign = None
    selected = invariant if sign_invariant else signed
    return {
        "fsc_auc": _finite(selected),
        "signed_fsc_auc": _finite(signed),
        "sign_flipped_fsc_auc": _finite(flipped),
        "sign_invariant_fsc_auc": _finite(invariant),
        "sign_invariant_best_sign": best_sign,
        "sign_mode_used": "sign_invariant" if sign_invariant else "signed",
        "n_shells": int(values.size),
        "shellwise_key": shellwise_key,
    }


def _map_metric(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    sign_invariant: bool,
    shellwise_key: str,
    shellwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    if curve.size <= 1 or np.count_nonzero(np.isfinite(curve[1:])) < 1:
        raise AuditError(f"{shellwise_key} produced no finite non-DC FSC shells")
    shellwise[shellwise_key] = curve
    return _metric(curve, sign_invariant=sign_invariant, shellwise_key=shellwise_key)


def _discover_maps(directory: Path, pattern: re.Pattern[str], *, engine: str) -> dict[int, dict[int, Path]]:
    grouped: dict[int, dict[int, Path]] = {}
    for path in directory.glob("*.mrc"):
        match = pattern.match(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        half = int(match.group(2))
        if engine == "RELION" and iteration == 0:
            continue
        if half in grouped.setdefault(iteration, {}):
            raise AuditError(f"duplicate {engine} iteration {iteration} half {half}")
        grouped[iteration][half] = path
    return grouped


def _validate_numbered_topology(
    recovar_maps: dict[int, dict[int, Path]],
    relion_maps: dict[int, dict[int, Path]],
    refinement_results: Path,
) -> list[tuple[int, int]]:
    if not recovar_maps:
        raise AuditError("no RECOVAR numbered regularized half maps found")
    if not relion_maps:
        raise AuditError("no RELION numbered half maps found")

    for engine, maps in (("RECOVAR", recovar_maps), ("RELION", relion_maps)):
        incomplete = {iteration: sorted(halves) for iteration, halves in maps.items() if set(halves) != {1, 2}}
        if incomplete:
            raise AuditError(f"{engine} numbered half-map pairs are incomplete: {incomplete}")

    recovar_iterations = sorted(recovar_maps)
    relion_iterations = sorted(relion_maps)
    expected_recovar = list(range(len(recovar_iterations)))
    expected_relion = list(range(1, len(relion_iterations) + 1))
    if recovar_iterations != expected_recovar:
        raise AuditError(
            f"RECOVAR iterations are not contiguous zero-based: found {recovar_iterations}, expected {expected_recovar}"
        )
    if relion_iterations != expected_relion:
        raise AuditError(
            f"RELION iterations are not contiguous one-based: found {relion_iterations}, expected {expected_relion}"
        )
    if len(recovar_iterations) != len(relion_iterations):
        raise AuditError(
            f"numbered iteration count mismatch: RECOVAR={len(recovar_iterations)} RELION={len(relion_iterations)}"
        )

    if not refinement_results.is_file():
        raise AuditError(f"missing RECOVAR refinement results: {refinement_results}")
    with np.load(refinement_results, allow_pickle=False) as payload:
        if "current_sizes" not in payload.files:
            raise AuditError(f"missing current_sizes in {refinement_results}")
        result_count = int(np.asarray(payload["current_sizes"]).size)
    if result_count != len(recovar_iterations):
        raise AuditError(
            f"RECOVAR map/result iteration count mismatch: maps={len(recovar_iterations)} current_sizes={result_count}"
        )
    return list(zip(recovar_iterations, relion_iterations, strict=True))


def _case_is_noctf(case_root: Path) -> bool:
    config_path = case_root / "case_config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"failed to parse {config_path}: {exc}") from exc
    return str(config.get("dataset_params_option", "")).strip().lower() == "noctf"


def _gt_sign_invariant(case_root: Path, mode: str) -> tuple[bool, str]:
    if mode == "signed":
        return False, "explicit_signed"
    if mode == "sign-invariant":
        return True, "explicit_sign_invariant"
    if _case_is_noctf(case_root):
        return True, "auto_noctf"
    return False, "auto_signed"


def _series_metrics(
    rec_half1: np.ndarray,
    rec_half2: np.ndarray,
    rel_half1: np.ndarray,
    rel_half2: np.ndarray,
    gt: np.ndarray,
    *,
    prefix: str,
    gt_sign_invariant: bool,
    shellwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    rec_merged = 0.5 * (rec_half1 + rec_half2)
    rel_merged = 0.5 * (rel_half1 + rel_half2)
    cross = {
        "half1": _map_metric(
            rec_half1,
            rel_half1,
            sign_invariant=False,
            shellwise_key=f"{prefix}_cross_half1",
            shellwise=shellwise,
        ),
        "half2": _map_metric(
            rec_half2,
            rel_half2,
            sign_invariant=False,
            shellwise_key=f"{prefix}_cross_half2",
            shellwise=shellwise,
        ),
        "merged": _map_metric(
            rec_merged,
            rel_merged,
            sign_invariant=False,
            shellwise_key=f"{prefix}_cross_merged",
            shellwise=shellwise,
        ),
    }
    gt_metrics: dict[str, dict[str, Any]] = {"recovar": {}, "relion": {}}
    for engine, volumes in (
        ("recovar", (rec_half1, rec_half2, rec_merged)),
        ("relion", (rel_half1, rel_half2, rel_merged)),
    ):
        for label, volume in zip(("half1", "half2", "merged"), volumes, strict=True):
            gt_metrics[engine][label] = _map_metric(
                volume,
                gt,
                sign_invariant=gt_sign_invariant,
                shellwise_key=f"{prefix}_{engine}_gt_{label}",
                shellwise=shellwise,
            )
    rec_gt_auc = gt_metrics["recovar"]["merged"]["fsc_auc"]
    rel_gt_auc = gt_metrics["relion"]["merged"]["fsc_auc"]
    gt_delta = None if rec_gt_auc is None or rel_gt_auc is None else float(rec_gt_auc - rel_gt_auc)
    return {"cross_engine": cross, "vs_gt": gt_metrics, "merged_gt_fsc_auc_delta": gt_delta}


def _load_numbered_row(
    recovar_paths: dict[int, Path],
    relion_paths: dict[int, Path],
    gt: np.ndarray,
    *,
    relion_iteration: int,
    gt_sign_invariant: bool,
    shellwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    rec_half1 = _load_recovar_volume(recovar_paths[1])
    rec_half2 = _load_recovar_volume(recovar_paths[2])
    rel_half1 = _load_relion_volume(relion_paths[1])
    rel_half2 = _load_relion_volume(relion_paths[2])
    metrics = _series_metrics(
        rec_half1,
        rec_half2,
        rel_half1,
        rel_half2,
        gt,
        prefix=f"it{relion_iteration:03d}",
        gt_sign_invariant=gt_sign_invariant,
        shellwise=shellwise,
    )
    return {
        "recovar_index": relion_iteration - 1,
        "relion_iteration": relion_iteration,
        **metrics,
    }


def _optional_final_metrics(
    recovar_dir: Path,
    relion_dir: Path,
    gt: np.ndarray,
    *,
    gt_sign_invariant: bool,
    shellwise: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    paths = {
        "rec_half1": recovar_dir / "final_half1_unfil.mrc",
        "rec_half2": recovar_dir / "final_half2_unfil.mrc",
        "rec_merged": recovar_dir / "final_merged.mrc",
        "rel_half1": relion_dir / "run_half1_class001_unfil.mrc",
        "rel_half2": relion_dir / "run_half2_class001_unfil.mrc",
        "rel_merged": relion_dir / "run_class001.mrc",
    }
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        return None
    missing = [str(paths[name]) for name, is_present in present.items() if not is_present]
    if missing:
        raise AuditError(
            "final products are partially present; a complete split-half and merged "
            f"RECOVAR/RELION comparison is required. Missing: {missing}"
        )

    out: dict[str, Any] = {
        "available_products": present,
        "cross_engine": {},
        "vs_gt": {"recovar": {}, "relion": {}},
        "merged_gt_fsc_auc_delta": None,
    }
    loaded: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        if not present[name]:
            continue
        loaded[name] = _load_relion_volume(path) if name.startswith("rel_") else _load_recovar_volume(path)

    for label in ("half1", "half2", "merged"):
        rec_key = f"rec_{label}"
        rel_key = f"rel_{label}"
        if rec_key in loaded and rel_key in loaded:
            out["cross_engine"][label] = _map_metric(
                loaded[rec_key],
                loaded[rel_key],
                sign_invariant=False,
                shellwise_key=f"final_cross_{label}",
                shellwise=shellwise,
            )
        for engine, key in (("recovar", rec_key), ("relion", rel_key)):
            if key in loaded:
                out["vs_gt"][engine][label] = _map_metric(
                    loaded[key],
                    gt,
                    sign_invariant=gt_sign_invariant,
                    shellwise_key=f"final_{engine}_gt_{label}",
                    shellwise=shellwise,
                )

    rec_gt = out["vs_gt"]["recovar"].get("merged", {}).get("fsc_auc")
    rel_gt = out["vs_gt"]["relion"].get("merged", {}).get("fsc_auc")
    if rec_gt is not None and rel_gt is not None:
        out["merged_gt_fsc_auc_delta"] = float(rec_gt - rel_gt)
    return out


def _apply_gates(
    rows: list[dict[str, Any]],
    final: dict[str, Any] | None,
    *,
    min_cross_merged: float,
    min_gt_delta: float,
) -> list[str]:
    failures: list[str] = []

    def check(label: str, values: dict[str, Any]) -> None:
        cross = values.get("cross_engine", {}).get("merged", {}).get("fsc_auc")
        delta = values.get("merged_gt_fsc_auc_delta")
        if cross is None or not math.isfinite(float(cross)):
            failures.append(f"{label} merged cross-engine FSC-AUC is missing or non-finite")
        elif float(cross) < min_cross_merged:
            failures.append(f"{label} merged cross-engine FSC-AUC {float(cross):.9f} < {min_cross_merged:.9f}")
        if delta is None or not math.isfinite(float(delta)):
            failures.append(f"{label} merged GT FSC-AUC delta is missing or non-finite")
        elif float(delta) < min_gt_delta:
            failures.append(f"{label} merged GT FSC-AUC delta {float(delta):+.9f} < {min_gt_delta:+.9f}")

    for row in rows:
        check(f"it{int(row['relion_iteration']):03d}", row)
    if final is not None and "merged" in final.get("cross_engine", {}):
        check("final", final)
    return failures


def audit_case(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case_root = args.case_root.resolve()
    recovar_dir = (args.recovar_dir or case_root / "recovar").resolve()
    relion_dir = (args.relion_dir or case_root / "relion_ref").resolve()
    gt_path = (args.gt_volume or case_root / "data" / "reference_gt.mrc").resolve()
    intermediates = recovar_dir / "intermediates"
    if not intermediates.is_dir():
        raise AuditError(f"missing RECOVAR intermediates directory: {intermediates}")
    if not relion_dir.is_dir():
        raise AuditError(f"missing RELION directory: {relion_dir}")
    if not gt_path.is_file():
        raise AuditError(f"missing GT volume: {gt_path}")

    recovar_maps = _discover_maps(intermediates, RECOVAR_MAP_RE, engine="RECOVAR")
    relion_maps = _discover_maps(relion_dir, RELION_MAP_RE, engine="RELION")
    pairs = _validate_numbered_topology(recovar_maps, relion_maps, recovar_dir / "refinement_results.npz")
    gt_sign_invariant, sign_reason = _gt_sign_invariant(case_root, args.gt_sign_mode)
    gt = _load_recovar_volume(gt_path)
    shellwise: dict[str, np.ndarray] = {}
    rows = [
        _load_numbered_row(
            recovar_maps[recovar_index],
            relion_maps[relion_iteration],
            gt,
            relion_iteration=relion_iteration,
            gt_sign_invariant=gt_sign_invariant,
            shellwise=shellwise,
        )
        for recovar_index, relion_iteration in pairs
    ]
    final = _optional_final_metrics(
        recovar_dir,
        relion_dir,
        gt,
        gt_sign_invariant=gt_sign_invariant,
        shellwise=shellwise,
    )
    failures = _apply_gates(
        rows,
        final,
        min_cross_merged=float(args.min_cross_merged_fsc_auc),
        min_gt_delta=float(args.min_merged_gt_delta),
    )
    report = {
        "schema": SCHEMA,
        "status": "pass" if not failures else "fail",
        "quality_metric_policy": "FSC curves and normalized FSC-AUC only; correlation is not computed",
        "paths": {
            "case_root": str(case_root),
            "recovar_dir": str(recovar_dir),
            "relion_dir": str(relion_dir),
            "gt_volume": str(gt_path),
        },
        "gt_sign_policy": {
            "requested": args.gt_sign_mode,
            "used": "sign_invariant" if gt_sign_invariant else "signed",
            "reason": sign_reason,
        },
        "thresholds": {
            "merged_cross_engine_fsc_auc_min": float(args.min_cross_merged_fsc_auc),
            "recovar_minus_relion_merged_gt_fsc_auc_min": float(args.min_merged_gt_delta),
        },
        "numbered_iteration_count": len(rows),
        "numbered_iterations": rows,
        "final": final,
        "failures": failures,
        "earliest_failure": failures[0] if failures else None,
    }
    return report, shellwise


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.9f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# K=1 FSC trajectory audit",
        "",
        f"Status: **{str(report['status']).upper()}**",
        "",
        "FSC curves and normalized FSC-AUC are the map-quality metrics. Correlation was not computed.",
        "",
        "| Iteration | Cross half1 | Cross half2 | Cross merged | REC GT merged | RELION GT merged | GT delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("numbered_iterations", []):
        cross = row["cross_engine"]
        gt = row["vs_gt"]
        lines.append(
            f"| {row['relion_iteration']} | {_fmt(cross['half1']['fsc_auc'])} | "
            f"{_fmt(cross['half2']['fsc_auc'])} | {_fmt(cross['merged']['fsc_auc'])} | "
            f"{_fmt(gt['recovar']['merged']['fsc_auc'])} | {_fmt(gt['relion']['merged']['fsc_auc'])} | "
            f"{_fmt(row['merged_gt_fsc_auc_delta'])} |"
        )
    lines.extend(
        [
            "",
            "Half1/half2 GT curves and FSC-AUC values are retained in the JSON and shellwise NPZ.",
        ]
    )
    final = report.get("final")
    if final:
        lines.extend(["", "## Final products", ""])
        for label, metric in final.get("cross_engine", {}).items():
            lines.append(f"- {label} cross-engine FSC-AUC: `{_fmt(metric.get('fsc_auc'))}`")
        delta = final.get("merged_gt_fsc_auc_delta")
        if delta is not None:
            lines.append(f"- Merged RECOVAR-minus-RELION GT FSC-AUC: `{_fmt(delta)}`")
    failures = report.get("failures") or []
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--recovar-dir", type=Path)
    parser.add_argument("--relion-dir", type=Path)
    parser.add_argument("--gt-volume", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-shellwise-npz", type=Path)
    parser.add_argument(
        "--gt-sign-mode",
        choices=("auto", "signed", "sign-invariant"),
        default="auto",
        help="Auto uses sign-invariant GT FSC-AUC only for case_config dataset_params_option=noctf.",
    )
    parser.add_argument("--min-cross-merged-fsc-auc", type=float, default=0.995)
    parser.add_argument("--min-merged-gt-delta", type=float, default=-0.002)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    analysis_dir = args.case_root.resolve() / "trajectory_analysis"
    output_json = (args.output_json or analysis_dir / "k1_fsc_trajectory.json").resolve()
    output_markdown = (args.output_markdown or analysis_dir / "k1_fsc_trajectory.md").resolve()
    output_npz = (args.output_shellwise_npz or analysis_dir / "k1_fsc_trajectory_shellwise.npz").resolve()
    for path in (output_json, output_markdown, output_npz):
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        report, shellwise = audit_case(args)
    except AuditError as exc:
        report = {
            "schema": SCHEMA,
            "status": "error",
            "quality_metric_policy": "FSC curves and normalized FSC-AUC only; correlation is not computed",
            "failures": [str(exc)],
            "earliest_failure": str(exc),
            "numbered_iterations": [],
            "final": None,
        }
        shellwise = {}

    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output_markdown.write_text(render_markdown(report))
    np.savez_compressed(output_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
