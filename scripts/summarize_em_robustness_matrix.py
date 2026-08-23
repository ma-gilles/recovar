#!/usr/bin/env python3
"""Summarize pending/completed EM robustness matrix scratch roots.

This is intentionally lightweight and tolerant: it reads case tables, per-case
JSON summaries, walltime JSON, and logs when present, but it still reports cases
whose Slurm jobs have only created job scripts so far.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "em_robustness_matrix_summary_v1"

TARGET_FILENAMES = {
    "case_config.json",
    "case_table.tsv",
    "selected_cases.tsv",
    "summary_metrics.json",
    "summary.md",
    "kclass_gt_fsc_auc.json",
    "kclass_gt_fsc.json",
    "relion_kclass_gt_fsc_auc.json",
    "relion_kclass_gt_fsc.json",
    "slurm_walltime.json",
    "prepare_walltime.json",
    "run_full_refinement.log",
    "prepare.log",
    "relion_autorefine.log",
    "relion_class3d.log",
    "relion_evaluate_kclass_gt.log",
    "evaluate_kclass_gt.log",
    "k1_robustness_matrix_summary.json",
    "k4_mini_summary.json",
}
PRUNE_DIRS = {
    ".git",
    ".pixi",
    "__pycache__",
    ".pytest_cache",
    "cuda",
    "cuda_cache",
    "data",
    "pixi_home",
    "rattler_cache",
    "summaries",
    "tmp",
}
FAILURE_RE = re.compile(
    r"(traceback|exception|resource_exhausted|(?:^|\s)(?:error|failed)(?::|\b)|(?:^|\W)oom(?:\W|$)|out of memory|time.?limit|cancelled|killed|segmentation fault)",
    re.IGNORECASE,
)
DIAGNOSTIC_ROOT_RE = re.compile(
    r"(?:probe|diag|dump|dense[_-]?pass2|forced[_-]?sparse|true[_-]?sparse|lazy[_-]?mask|caps[_-]?retry|highmem[_-]?budget|budget[_-]?patch|reuse[_-]?noise)",
    re.IGNORECASE,
)
FINAL_RES_RE = re.compile(r"final resolution:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*A", re.IGNORECASE)
RELION_ITER_RE = re.compile(r"(?:Auto-refine:\s*)?Iteration=\s*([0-9]+)", re.IGNORECASE)
RELION_EXPECTATION_ITER_RE = re.compile(r"Expectation iteration\s+([0-9]+)", re.IGNORECASE)
RELION_RES_RE = re.compile(
    r"Auto-refine:\s*Resolution=\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
    re.IGNORECASE,
)
RELION_PROGRESS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?/[0-9]+(?:\.[0-9]+)?\s+(?:hrs|min))")
RECOVAR_ITER_START_RE = re.compile(
    r"=== RELION Iteration\s+([0-9]+)(?:/([0-9]+))?:\s+current_size=([0-9]+)",
    re.IGNORECASE,
)
RECOVAR_ITER_DONE_RE = re.compile(
    r"RELION Iteration\s+([0-9]+):\s+current_size=([0-9]+),.*?time=([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)
RECOVAR_SPARSE_GROUP_RE = re.compile(
    r"Sparse(?: fused)?(?: K-class)? pass-2 bucket group (start|done):.*?(?:pair_)?bucket_size=([0-9]+).*?chunks=([0-9]+).*?images=([0-9]+)(?:.*?wall=([0-9]+(?:\.[0-9]+)?)s)?",
    re.IGNORECASE,
)
RECOVAR_SPARSE_DONE_RE = re.compile(
    r"Sparse(?: fused)?(?: K-class)? pass-2(?: \(bucketed\))?:\s+([0-9]+)\s+images(?:,\s+[0-9]+\s+classes)?,\s+([0-9]+)\s+buckets,\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)
RECOVAR_COMPLETE_RE = re.compile(r"Refinement complete in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
SLURM_JOB_HEADER_RE = re.compile(r"^Slurm job:\s*([0-9]+)\b", re.IGNORECASE)


@dataclass
class CaseSummary:
    scratch_root: Path
    case_root: Path
    case_id: str | None = None
    case_name: str | None = None
    status: str = "pending"
    job_id: str | None = None
    slurm_state: str | None = None
    slurm_exit_code: str | None = None
    n_classes: int | None = None
    n_images: int | None = None
    grid_size: int | None = None
    noise_level: float | None = None
    noise_model: str | None = None
    poses: str | None = None
    wall_s: float | None = None
    relion_wall_s: float | None = None
    exit_status: int | None = None
    relion_iteration: int | None = None
    relion_resolution_A: float | None = None
    relion_latest_progress: str | None = None
    recovar_iteration: int | None = None
    recovar_total_iterations: int | None = None
    recovar_current_size: int | None = None
    recovar_latest_stage: str | None = None
    recovar_last_iteration_time_s: float | None = None
    recovar_n_iterations: int | None = None
    recovar_convergence_iteration: int | None = None
    recovar_convergence_has_converged: bool | None = None
    recovar_final_all_data_ran: bool | None = None
    recovar_final_all_data_fsc_auc: float | None = None
    recovar_has_final_all_data_poses: bool | None = None
    fsc_auc_vs_gt: float | None = None
    relion_fsc_auc_vs_gt: float | None = None
    fsc_auc_delta_vs_relion: float | None = None
    final_resolution_A: float | None = None
    map_corr_vs_gt: float | None = None
    failure_reason: str | None = None
    failure_log: str | None = None
    log_excerpt: str | None = None
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "scratch_root": str(self.scratch_root),
            "case_root": str(self.case_root),
            "case_id": self.case_id,
            "case_name": self.case_name,
            "status": self.status,
            "job_id": self.job_id,
            "slurm_state": self.slurm_state,
            "slurm_exit_code": self.slurm_exit_code,
            "n_classes": self.n_classes,
            "n_images": self.n_images,
            "grid_size": self.grid_size,
            "noise_level": self.noise_level,
            "noise_model": self.noise_model,
            "poses": self.poses,
            "wall_s": self.wall_s,
            "relion_wall_s": self.relion_wall_s,
            "exit_status": self.exit_status,
            "relion_iteration": self.relion_iteration,
            "relion_resolution_A": self.relion_resolution_A,
            "relion_latest_progress": self.relion_latest_progress,
            "recovar_iteration": self.recovar_iteration,
            "recovar_total_iterations": self.recovar_total_iterations,
            "recovar_current_size": self.recovar_current_size,
            "recovar_latest_stage": self.recovar_latest_stage,
            "recovar_last_iteration_time_s": self.recovar_last_iteration_time_s,
            "recovar_n_iterations": self.recovar_n_iterations,
            "recovar_convergence_iteration": self.recovar_convergence_iteration,
            "recovar_convergence_has_converged": self.recovar_convergence_has_converged,
            "recovar_final_all_data_ran": self.recovar_final_all_data_ran,
            "recovar_final_all_data_fsc_auc": self.recovar_final_all_data_fsc_auc,
            "recovar_has_final_all_data_poses": self.recovar_has_final_all_data_poses,
            "fsc_auc_vs_gt": self.fsc_auc_vs_gt,
            "relion_fsc_auc_vs_gt": self.relion_fsc_auc_vs_gt,
            "fsc_auc_delta_vs_relion": self.fsc_auc_delta_vs_relion,
            "final_resolution_A": self.final_resolution_A,
            "map_corr_vs_gt": self.map_corr_vs_gt,
            "failure_reason": self.failure_reason,
            "failure_log": self.failure_log,
            "log_excerpt": self.log_excerpt,
            "notes": self.notes,
            "artifacts": dict(sorted(self.artifacts.items())),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scratch_roots", nargs="+", type=Path, help="Scratch roots to scan recursively.")
    parser.add_argument("--output-markdown", required=True, type=Path, help="Markdown report path.")
    parser.add_argument("--output-json", required=True, type=Path, help="JSON report path.")
    parser.add_argument("--log-lines", type=int, default=30, help="Maximum failed-log excerpt lines.")
    parser.add_argument(
        "--dedupe-case-reruns",
        action="store_true",
        help="For duplicate case id/name rows, keep one active row, preferring completed rows and later roots.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(()).item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "yes", "y", "1"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
    return None


def choose_number(*values: Any) -> float | None:
    for value in values:
        parsed = as_float(value)
        if parsed is not None:
            return parsed
    return None


def mean_finite(values: list[Any]) -> float | None:
    parsed = [v for v in (as_float(value) for value in values) if v is not None]
    if not parsed:
        return None
    return float(sum(parsed) / len(parsed))


def mean_fsc_auc(curve: Any) -> float | None:
    if not isinstance(curve, list) or len(curve) <= 1:
        return None
    return mean_finite(curve[1:])


def normalized_curve_auc(curve: Any) -> float | None:
    try:
        values = np.asarray(curve, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if values.size <= 1:
        return None
    finite = np.isfinite(values)
    finite[0] = False
    if int(finite.sum()) == 0:
        return None
    x = np.arange(values.size, dtype=np.float64)[finite]
    y = values[finite]
    if y.size == 1:
        return float(y[0])
    span = float(x[-1] - x[0])
    if span <= 0.0 or not math.isfinite(span):
        return float(np.mean(y))
    x = (x - x[0]) / span
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x))


def walk_artifacts(root: Path) -> list[Path]:
    paths: list[Path] = []
    if not root.exists():
        return paths
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in PRUNE_DIRS]
        base = Path(dirpath)
        for filename in filenames:
            if filename in TARGET_FILENAMES:
                paths.append(base / filename)
            elif filename.startswith(("em_k1_matrix_", "em_k4_mini_")) and filename.endswith((".out", ".err")):
                paths.append(base / filename)
    return paths


def path_key(path: Path) -> str:
    return str(path.expanduser().resolve() if path.exists() else path.expanduser().absolute())


def unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path_key(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def get_case(
    cases: dict[str, CaseSummary],
    scratch_root: Path,
    case_root: Path,
) -> CaseSummary:
    key = path_key(case_root)
    if key not in cases:
        cases[key] = CaseSummary(scratch_root=scratch_root, case_root=case_root)
    return cases[key]


def artifact_case_root(path: Path) -> Path | None:
    name = path.name
    if name in {
        "case_config.json",
        "summary_metrics.json",
        "summary.md",
        "kclass_gt_fsc_auc.json",
        "kclass_gt_fsc.json",
        "relion_kclass_gt_fsc_auc.json",
        "relion_kclass_gt_fsc.json",
    }:
        return path.parent
    if name in {
        "prepare.log",
        "relion_autorefine.log",
        "relion_class3d.log",
        "evaluate_kclass_gt.log",
        "relion_evaluate_kclass_gt.log",
        "prepare_walltime.json",
    }:
        return path.parent
    if name in {"slurm_walltime.json", "run_full_refinement.log"} and path.parent.name in {
        "recovar",
        "relion_ref",
        "relion_class3d",
    }:
        return path.parent.parent
    return None


def parse_table(path: Path, scratch_root: Path, cases: dict[str, CaseSummary]) -> None:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    if not lines:
        return
    header = lines[0].split("|")
    for raw in lines[1:]:
        if not raw.strip():
            continue
        parts = raw.split("|")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        case_root_raw = row.get("case_root")
        if case_root_raw:
            case_root = Path(case_root_raw)
        else:
            case_root = scratch_root / "cases" / f"{row.get('index', '').strip()}_{row.get('name', '').strip()}"
        case = get_case(cases, scratch_root, case_root)
        case.artifacts[path.name] = str(path)
        merge_case_metadata(case, row)
        case.job_id = row.get("case_job_id") or row.get("job_id") or case.job_id
        script = row.get("script")
        if script:
            case.artifacts["job_script"] = script
            add_central_logs_from_script(case, Path(script))


def merge_case_metadata(case: CaseSummary, data: dict[str, Any]) -> None:
    case.case_id = str(data.get("index") or data.get("idx") or case.case_id or "").strip() or case.case_id
    case.case_name = str(data.get("name") or data.get("case") or case.case_name or "").strip() or case.case_name
    case.n_classes = as_int(data.get("n_classes") or data.get("K")) or case.n_classes
    case.n_images = as_int(data.get("n_images")) or case.n_images
    case.grid_size = as_int(data.get("grid_size") or data.get("grid")) or case.grid_size
    case.noise_level = choose_number(data.get("noise_level"), case.noise_level)
    case.noise_model = str(data.get("noise_model") or case.noise_model or "").strip() or case.noise_model
    case.poses = str(data.get("dataset_params_option") or data.get("poses") or case.poses or "").strip() or case.poses


def add_central_logs_from_script(case: CaseSummary, script_path: Path) -> None:
    if not script_path.name.endswith(".sh"):
        return
    scratch = script_path.parent.parent
    stem = script_path.stem
    for suffix, label in ((".out", "slurm_stdout"), (".err", "slurm_stderr")):
        log_path = scratch / f"{stem}{suffix}"
        if log_path.exists():
            case.artifacts[label] = str(log_path)


def add_default_case_logs(case: CaseSummary) -> None:
    case_dirname = case.case_root.name
    scratch = case.case_root.parent.parent if case.case_root.parent.name == "cases" else case.scratch_root
    for prefix in ("em_k1_matrix_", "em_k4_mini_"):
        for suffix, label in ((".out", "slurm_stdout"), (".err", "slurm_stderr")):
            log_path = scratch / f"{prefix}{case_dirname}{suffix}"
            if log_path.exists():
                case.artifacts.setdefault(label, str(log_path))


def merge_aggregate_summary(path: Path, scratch_root: Path, cases: dict[str, CaseSummary]) -> None:
    data = read_json(path)
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_root_raw = row.get("case_root") or row.get("path")
        if not case_root_raw:
            continue
        case = get_case(cases, scratch_root, Path(case_root_raw))
        case.artifacts[path.name] = str(path)
        merge_case_metadata(case, row)
        case.status = str(row.get("status") or case.status)
        case.job_id = str(row.get("job_id") or case.job_id or "").strip() or case.job_id
        case.wall_s = choose_number(row.get("recovar_wall_s"), row.get("wall_s"), case.wall_s)
        case.relion_wall_s = choose_number(row.get("relion_wall_s"), case.relion_wall_s)
        case.exit_status = as_int(row.get("recovar_exit")) if row.get("recovar_exit") is not None else case.exit_status
        case.fsc_auc_vs_gt = choose_number(
            row.get("recovar_vs_gt_fsc_auc"),
            row.get("recovar_mean_fsc_auc_1_nyquist"),
            row.get("mean_fsc_auc_1_nyquist"),
            row.get("recovar_mean_fsc_auc_1_16"),
            row.get("mean_fsc_auc_1_16"),
            case.fsc_auc_vs_gt,
        )
        case.relion_fsc_auc_vs_gt = choose_number(
            row.get("relion_vs_gt_fsc_auc"),
            row.get("relion_mean_fsc_auc_1_nyquist"),
            row.get("relion_mean_fsc_auc_1_16"),
            case.relion_fsc_auc_vs_gt,
        )
        case.map_corr_vs_gt = choose_number(row.get("recovar_vs_gt_corr"), row.get("mean_corr"), case.map_corr_vs_gt)
        notes = row.get("notes")
        if isinstance(notes, list):
            case.notes.extend(str(note) for note in notes)


def merge_case_config(case: CaseSummary, path: Path) -> None:
    data = read_json(path)
    if isinstance(data, dict):
        merge_case_metadata(case, data)
        case.artifacts["case_config"] = str(path)


def merge_walltime(case: CaseSummary, path: Path) -> None:
    data = read_json(path)
    if not isinstance(data, dict):
        return
    label = "slurm_walltime" if path.name == "slurm_walltime.json" else path.name
    if path.parent.name == "recovar" or path.name == "prepare_walltime.json":
        case.artifacts[label] = str(path)
    if path.parent.name == "recovar":
        case.wall_s = choose_number(data.get("external_wall_s"), data.get("wall_s"), case.wall_s)
        case.exit_status = as_int(data.get("exit_status")) if data.get("exit_status") is not None else case.exit_status
        case.job_id = str(data.get("slurm_job_id") or case.job_id or "").strip() or case.job_id
    elif path.parent.name in {"relion_ref", "relion_class3d"}:
        case.artifacts["relion_slurm_walltime"] = str(path)
        case.relion_wall_s = choose_number(data.get("external_wall_s"), data.get("wall_s"), case.relion_wall_s)


def metric_fsc_auc(values: dict[str, Any], *, sign_invariant: bool = False) -> float | None:
    if sign_invariant:
        parsed = choose_number(
            values.get("fsc_auc_sign_invariant"),
            values.get("mean_fsc_auc_sign_invariant"),
            values.get("mean_fsc_auc_1_nyquist_sign_invariant"),
            values.get("mean_fsc_auc_1_16_sign_invariant"),
        )
        if parsed is not None:
            return parsed
    return choose_number(
        values.get("fsc_auc"),
        values.get("mean_fsc_auc"),
        values.get("mean_fsc_auc_1_nyquist"),
        values.get("mean_fsc_auc_1_16"),
        normalized_curve_auc(values.get("fsc")),
        normalized_curve_auc(values.get("fsc_vs_gt")),
        mean_fsc_auc(values.get("fsc")),
        mean_fsc_auc(values.get("fsc_vs_gt")),
    )


def summary_section_allows_global_sign(section: dict[str, Any]) -> bool:
    sign_ambiguity = section.get("sign_ambiguity")
    return isinstance(sign_ambiguity, dict) and bool(sign_ambiguity.get("allow_global_sign"))


def summary_section_for_metrics(data: dict[str, Any]) -> dict[str, Any] | None:
    sections = [section for key in ("k1", "k4") if isinstance((section := data.get(key)), dict)]
    if not sections:
        return None

    def section_score(section: dict[str, Any]) -> tuple[int, int, int]:
        status = str(section.get("status") or "").strip().lower()
        terminal_score = {
            "ok": 5,
            "completed": 5,
            "success": 5,
            "passed": 5,
            "failed": 4,
            "timing_probe": 3,
            "running": 2,
            "pending": 1,
            "missing": 1,
            "skipped": 0,
        }.get(status, 1 if status else 0)
        metrics = section.get("metrics")
        timing = section.get("timing")
        has_metrics = isinstance(metrics, dict) and bool(metrics)
        has_timing = isinstance(timing, dict) and bool(timing)
        return (terminal_score, int(has_metrics), int(has_timing))

    return max(sections, key=section_score)


def merge_summary_metrics(case: CaseSummary, path: Path) -> None:
    data = read_json(path)
    if not isinstance(data, dict):
        return
    case.artifacts["summary_metrics"] = str(path)
    section = summary_section_for_metrics(data)
    if not isinstance(section, dict):
        return
    case.status = str(section.get("status") or case.status)
    notes = section.get("notes")
    if isinstance(notes, list):
        fresh_notes = [str(note) for note in notes]
        if not any("runtime default guard" in note for note in fresh_notes):
            case.notes = [
                note
                for note in case.notes
                if "runtime default guard" not in note
                and "required artifacts or metrics are missing" not in note
            ]
        case.notes.extend(fresh_notes)
    convergence = section.get("recovar_convergence")
    if isinstance(convergence, dict):
        case.recovar_n_iterations = as_int(convergence.get("n_iterations")) or case.recovar_n_iterations
        case.recovar_convergence_iteration = (
            as_int(convergence.get("convergence_iteration")) or case.recovar_convergence_iteration
        )
        parsed_converged = as_bool(convergence.get("convergence_has_converged"))
        if parsed_converged is not None:
            case.recovar_convergence_has_converged = parsed_converged
    metrics = section.get("metrics")
    allow_global_sign = summary_section_allows_global_sign(section)
    if isinstance(metrics, dict):
        rec_gt = (
            metrics.get("recovar_merged_vs_gt")
            or metrics.get("recovar_vs_gt")
            or metrics.get("recovar_primary_vs_gt")
        )
        if isinstance(rec_gt, dict):
            case.fsc_auc_vs_gt = choose_number(
                metric_fsc_auc(rec_gt, sign_invariant=allow_global_sign),
                case.fsc_auc_vs_gt,
            )
            case.map_corr_vs_gt = choose_number(
                rec_gt.get("abs_corr") if allow_global_sign else None,
                rec_gt.get("corr"),
                rec_gt.get("corr_vs_gt"),
                rec_gt.get("mean_corr"),
                case.map_corr_vs_gt,
            )
            case.final_resolution_A = choose_number(
                rec_gt.get("resolution_0143_A"),
                rec_gt.get("final_resolution_A"),
                case.final_resolution_A,
            )
        rel_gt = (
            metrics.get("relion_merged_vs_gt")
            or metrics.get("relion_vs_gt")
            or metrics.get("relion_primary_vs_gt")
        )
        if isinstance(rel_gt, dict):
            case.relion_fsc_auc_vs_gt = choose_number(
                metric_fsc_auc(rel_gt, sign_invariant=allow_global_sign),
                case.relion_fsc_auc_vs_gt,
            )
    timing = section.get("timing")
    if isinstance(timing, dict):
        case.wall_s = choose_number(timing.get("recovar_walltime_s"), case.wall_s)
        case.relion_wall_s = choose_number(timing.get("relion_walltime_s"), case.relion_wall_s)
        rows = timing.get("recovar_iteration_rows") or timing.get("iteration_timing_rows")
        if isinstance(rows, list):
            for row in reversed(rows):
                if isinstance(row, dict):
                    case.final_resolution_A = choose_number(row.get("res_ang"), case.final_resolution_A)
                    if case.final_resolution_A is not None:
                        break


def merge_kclass_auc(case: CaseSummary, path: Path) -> None:
    data = read_json(path)
    if not isinstance(data, dict):
        return
    case.artifacts["kclass_gt_fsc_auc"] = str(path)
    case.status = "ok"
    case.fsc_auc_vs_gt = choose_number(
        data.get("mean_fsc_auc"),
        data.get("mean_fsc_auc_1_nyquist"),
        data.get("mean_fsc_auc_1_16"),
        case.fsc_auc_vs_gt,
    )


def merge_kclass_gt(case: CaseSummary, path: Path) -> None:
    data = read_json(path)
    if not isinstance(data, dict):
        return
    case.artifacts["kclass_gt_fsc"] = str(path)
    primary = data.get("primary")
    if not isinstance(primary, dict):
        primary = data
    per_class = primary.get("per_class")
    if not isinstance(per_class, list):
        return
    per_class_auc = [
        normalized_curve_auc(row.get("fsc_vs_gt")) for row in per_class if isinstance(row, dict)
    ]
    case.status = "ok"
    case.fsc_auc_vs_gt = choose_number(
        case.fsc_auc_vs_gt,
        primary.get("mean_fsc_auc"),
        primary.get("mean_fsc_auc_1_nyquist"),
        mean_finite(per_class_auc),
        primary.get("best_mean_fsc_1_8"),
        mean_finite([mean_fsc_auc(row.get("fsc_vs_gt")) for row in per_class if isinstance(row, dict)]),
    )
    case.map_corr_vs_gt = choose_number(
        mean_finite([row.get("corr") for row in per_class if isinstance(row, dict)]),
        case.map_corr_vs_gt,
    )
    case.final_resolution_A = choose_number(
        mean_finite([row.get("resolution_0143_A") for row in per_class if isinstance(row, dict)]),
        case.final_resolution_A,
    )


def merge_relion_kclass_auc(case: CaseSummary, path: Path) -> None:
    data = read_json(path)
    if not isinstance(data, dict):
        return
    case.artifacts["relion_kclass_gt_fsc_auc"] = str(path)
    case.relion_fsc_auc_vs_gt = choose_number(
        data.get("mean_fsc_auc"),
        data.get("mean_fsc_auc_1_nyquist"),
        data.get("mean_fsc_auc_1_16"),
        case.relion_fsc_auc_vs_gt,
    )


def merge_relion_kclass_gt(case: CaseSummary, path: Path) -> None:
    data = read_json(path)
    if not isinstance(data, dict):
        return
    case.artifacts["relion_kclass_gt_fsc"] = str(path)
    primary = data.get("primary")
    if not isinstance(primary, dict):
        primary = data
    per_class = primary.get("per_class")
    if not isinstance(per_class, list):
        return
    per_class_auc = [
        normalized_curve_auc(row.get("fsc_vs_gt")) for row in per_class if isinstance(row, dict)
    ]
    case.relion_fsc_auc_vs_gt = choose_number(
        case.relion_fsc_auc_vs_gt,
        primary.get("mean_fsc_auc"),
        primary.get("mean_fsc_auc_1_nyquist"),
        mean_finite(per_class_auc),
        primary.get("best_mean_fsc_1_8"),
        mean_finite([mean_fsc_auc(row.get("fsc_vs_gt")) for row in per_class if isinstance(row, dict)]),
    )


def read_tail(path: Path, max_lines: int = 80) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


def read_head(path: Path, max_lines: int = 120) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return lines[:max_lines]


def first_failure_line(lines: list[str]) -> str | None:
    for line in reversed(lines):
        text = line.strip()
        if text and FAILURE_RE.search(text):
            return text
    return None


def log_paths(case: CaseSummary) -> list[Path]:
    labels = (
        "slurm_stderr",
        "run_full_refinement_log",
        "slurm_stdout",
        "prepare_log",
        "relion_autorefine_log",
        "relion_class3d_log",
        "evaluate_kclass_gt_log",
        "relion_evaluate_kclass_gt_log",
    )
    paths: list[Path] = []
    for label in labels:
        raw = case.artifacts.get(label)
        if raw:
            paths.append(Path(raw))
    rec_log = case.case_root / "recovar" / "run_full_refinement.log"
    if rec_log.exists() and rec_log not in paths:
        paths.append(rec_log)
    return paths


def merge_log_signals(case: CaseSummary, max_excerpt_lines: int) -> None:
    add_default_case_logs(case)
    merge_slurm_job_header(case)
    rec_log = case.case_root / "recovar" / "run_full_refinement.log"
    if rec_log.exists():
        case.artifacts["run_full_refinement_log"] = str(rec_log)
    prep_log = case.case_root / "prepare.log"
    if prep_log.exists():
        case.artifacts["prepare_log"] = str(prep_log)
    relion_log = case.case_root / "relion_autorefine.log"
    if relion_log.exists():
        case.artifacts["relion_autorefine_log"] = str(relion_log)
    relion_class3d_log = case.case_root / "relion_class3d.log"
    if relion_class3d_log.exists():
        case.artifacts["relion_class3d_log"] = str(relion_class3d_log)
    eval_log = case.case_root / "evaluate_kclass_gt.log"
    if eval_log.exists():
        case.artifacts["evaluate_kclass_gt_log"] = str(eval_log)
    relion_eval_log = case.case_root / "relion_evaluate_kclass_gt.log"
    if relion_eval_log.exists():
        case.artifacts["relion_evaluate_kclass_gt_log"] = str(relion_eval_log)
    merge_relion_progress(case)
    merge_recovar_progress(case)

    failure_lines: list[str] = []
    excerpt: list[str] = []
    for path in log_paths(case):
        lines = read_tail(path)
        for line in lines:
            match = FINAL_RES_RE.search(line)
            if match:
                case.final_resolution_A = choose_number(match.group(1), case.final_resolution_A)
        failure = first_failure_line(lines)
        if failure:
            failure_lines.append(f"{path.name}: {failure}")
            case.failure_log = case.failure_log or str(path)
            case.artifacts.setdefault("failure_log", str(path))
            excerpt = lines[-max_excerpt_lines:]

    if case.exit_status not in (None, 0):
        case.status = "failed"
        case.failure_reason = case.failure_reason or f"exit_status={case.exit_status}"
    if failure_lines and case.status in {"failed", "missing", "pending", "running"}:
        case.failure_reason = case.failure_reason or failure_lines[0]
        if case.status != "failed":
            case.status = "failed"
    if case.status == "failed" and excerpt and not case.log_excerpt:
        case.log_excerpt = "\n".join(excerpt[-max_excerpt_lines:])
    if case.status == "failed" and not case.failure_log:
        for path in log_paths(case):
            if path.exists():
                case.failure_log = str(path)
                case.artifacts.setdefault("failure_log", str(path))
                break


def merge_slurm_job_header(case: CaseSummary) -> None:
    """Use the current Slurm job printed by case scripts over stale tables."""

    for label in ("slurm_stdout", "slurm_stderr"):
        raw = case.artifacts.get(label)
        if not raw:
            continue
        for line in read_head(Path(raw), max_lines=200):
            match = SLURM_JOB_HEADER_RE.search(line.strip())
            if match:
                case.job_id = match.group(1)
                return


def merge_relion_progress(case: CaseSummary) -> None:
    for label in ("relion_autorefine_log", "relion_class3d_log"):
        raw = case.artifacts.get(label)
        if not raw:
            continue
        for line in read_tail(Path(raw), max_lines=5000):
            text = line.strip()
            match = RELION_ITER_RE.search(text) or RELION_EXPECTATION_ITER_RE.search(text)
            if match:
                case.relion_iteration = as_int(match.group(1)) or case.relion_iteration
            match = RELION_RES_RE.search(text)
            if match:
                case.relion_resolution_A = choose_number(match.group(1), case.relion_resolution_A)
            match = RELION_PROGRESS_RE.search(text)
            if match:
                case.relion_latest_progress = match.group(1)


def merge_recovar_progress(case: CaseSummary) -> None:
    raw = case.artifacts.get("run_full_refinement_log")
    if not raw:
        return
    for line in read_tail(Path(raw), max_lines=5000):
        text = line.strip()
        match = RECOVAR_ITER_START_RE.search(text)
        if match:
            case.recovar_iteration = as_int(match.group(1)) or case.recovar_iteration
            case.recovar_total_iterations = as_int(match.group(2)) or case.recovar_total_iterations
            case.recovar_current_size = as_int(match.group(3)) or case.recovar_current_size
            total = f"/{case.recovar_total_iterations}" if case.recovar_total_iterations is not None else ""
            case.recovar_latest_stage = f"iter {case.recovar_iteration}{total} setup"
            continue
        match = RECOVAR_SPARSE_GROUP_RE.search(text)
        if match:
            phase = match.group(1).lower()
            bucket_size = match.group(2)
            chunks = match.group(3)
            images = match.group(4)
            wall = match.group(5)
            if phase == "start":
                case.recovar_latest_stage = f"sparse pass-2 bucket {bucket_size}: {chunks} chunks/{images} images"
            else:
                suffix = f", {wall}s" if wall else ""
                case.recovar_latest_stage = (
                    f"sparse pass-2 bucket {bucket_size} done: {chunks} chunks/{images} images{suffix}"
                )
            continue
        match = RECOVAR_SPARSE_DONE_RE.search(text)
        if match:
            case.recovar_latest_stage = (
                f"sparse pass-2 done: {match.group(1)} images/{match.group(2)} buckets/{match.group(3)}s"
            )
            continue
        match = RECOVAR_ITER_DONE_RE.search(text)
        if match:
            case.recovar_iteration = as_int(match.group(1)) or case.recovar_iteration
            case.recovar_current_size = as_int(match.group(2)) or case.recovar_current_size
            case.recovar_last_iteration_time_s = choose_number(match.group(3), case.recovar_last_iteration_time_s)
            total = f"/{case.recovar_total_iterations}" if case.recovar_total_iterations is not None else ""
            case.recovar_latest_stage = (
                f"iter {case.recovar_iteration}{total} complete in {case.recovar_last_iteration_time_s:g}s"
                if case.recovar_last_iteration_time_s is not None
                else f"iter {case.recovar_iteration}{total} complete"
            )
            continue
        match = RECOVAR_COMPLETE_RE.search(text)
        if match:
            case.recovar_latest_stage = f"complete in {match.group(1)}s"


def merge_recovar_convergence(case: CaseSummary) -> None:
    path = case.case_root / "recovar" / "refinement_results.npz"
    if not path.exists():
        return
    case.artifacts["recovar_refinement_results"] = str(path)
    try:
        with np.load(path) as data:
            case.recovar_n_iterations = as_int(data.get("n_iterations")) or case.recovar_n_iterations
            case.recovar_convergence_iteration = (
                as_int(data.get("convergence_iteration")) or case.recovar_convergence_iteration
            )
            parsed_converged = as_bool(data.get("convergence_has_converged"))
            if parsed_converged is not None:
                case.recovar_convergence_has_converged = parsed_converged
            parsed_final_all_data_ran = as_bool(data.get("final_all_data_ran"))
            if parsed_final_all_data_ran is not None:
                case.recovar_final_all_data_ran = parsed_final_all_data_ran
            if "fsc_final_all_data" in data.files:
                case.recovar_final_all_data_fsc_auc = choose_number(
                    normalized_curve_auc(data["fsc_final_all_data"]),
                    case.recovar_final_all_data_fsc_auc,
                )
            elif case.recovar_final_all_data_ran is not False:
                case.notes.append("missing RECOVAR fsc_final_all_data")
            case.recovar_has_final_all_data_poses = bool(
                "best_rotation_eulers_final_all_data_by_image" in data.files
                and "best_translations_final_all_data_by_image" in data.files
            )
            if not case.recovar_has_final_all_data_poses and case.recovar_final_all_data_ran is not False:
                case.notes.append("missing RECOVAR final all-data pose arrays")
            if case.recovar_final_all_data_ran is False:
                case.notes.append("RECOVAR final_class maps are last numbered iteration maps; final all-data did not run")
            if (
                case.n_classes is not None
                and int(case.n_classes) > 1
                and case.recovar_final_all_data_ran is True
                and case.recovar_convergence_has_converged is False
            ):
                case.notes.append(
                    "diagnostic K-class final all-data ran without convergence; "
                    "do not treat this row as default parity evidence"
                )
    except Exception as exc:
        case.notes.append(f"failed to read RECOVAR convergence metadata: {exc}")


def collect_slurm_accounting(cases: list[CaseSummary]) -> dict[str, tuple[str, str | None]]:
    job_ids = sorted({str(case.job_id).strip() for case in cases if str(case.job_id or "").strip().isdigit()})
    if not job_ids:
        return {}
    try:
        proc = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(job_ids),
                "-X",
                "-P",
                "-n",
                "-o",
                "JobIDRaw,State,ExitCode",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}

    states: dict[str, tuple[str, str | None]] = {}
    for raw in proc.stdout.splitlines():
        parts = raw.split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].strip()
        state = parts[1].strip()
        exit_code = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None
        if job_id and state:
            states[job_id] = (state, exit_code)
    return states


def apply_slurm_accounting(case: CaseSummary, state_info: tuple[str, str | None] | None) -> None:
    if state_info is None:
        return
    state, exit_code = state_info
    state_upper = state.strip().upper()
    state_word = state_upper.split()[0] if state_upper else ""
    case.slurm_state = state
    case.slurm_exit_code = exit_code
    if state_word in {"CANCELLED", "CANCELLED+"}:
        case.status = "cancelled"
        note = f"Slurm job {case.job_id} was cancelled"
        if note not in case.notes:
            case.notes.append(note)
        return
    if state_word in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}:
        case.status = "failed"
        case.failure_reason = case.failure_reason or f"Slurm job {case.job_id} state={state}"
        return
    if state_word == "RUNNING" and case.status == "pending":
        case.status = "running"


def append_comparison_caveats(case: CaseSummary) -> None:
    if case.recovar_convergence_has_converged is not False:
        return
    if case.fsc_auc_vs_gt is None or case.relion_fsc_auc_vs_gt is None:
        return
    if case.n_classes is not None and int(case.n_classes) > 1:
        note = (
            "RECOVAR did not converge before its iteration cap; K-class RECOVAR final_class maps "
            "and RELION run_itNNN_class maps are last-numbered iteration maps. This row is "
            "iteration-cap K-class GT FSC-AUC evidence, not post-convergence final all-data evidence."
        )
    else:
        note = (
            "RECOVAR did not converge before its iteration cap; RELION-vs-GT metrics may use final maps "
            "while RECOVAR-vs-GT metrics are pre-final."
        )
    if note not in case.notes:
        case.notes.append(note)


def is_diagnostic_case(case: CaseSummary) -> bool:
    notes = " ".join(str(note) for note in case.notes)
    text = " ".join(
        [
            case.case_name or "",
            case.case_id or "",
            case.case_root.name,
            case.scratch_root.name,
            notes,
        ]
    )
    if "timing probe only" in notes.lower():
        return True
    return bool(DIAGNOSTIC_ROOT_RE.search(text))


def mark_diagnostic_status(case: CaseSummary) -> None:
    if not is_diagnostic_case(case):
        return
    if case.status == "ok":
        case.status = "diagnostic_ok"
    elif case.status == "running":
        case.status = "diagnostic_running"
    elif case.status == "failed":
        case.status = "diagnostic_failed"
    elif case.status in {"pending", "missing", "missing_metrics", "completed_missing_metrics"}:
        case.status = "diagnostic_pending"


def expects_relion_quality(case: CaseSummary) -> bool:
    if case.relion_fsc_auc_vs_gt is not None or case.relion_wall_s is not None:
        return True
    if case.relion_iteration is not None or case.relion_resolution_A is not None:
        return True
    if any(label.startswith("relion") for label in case.artifacts):
        return True
    relion_paths = (
        case.case_root / "relion_ref",
        case.case_root / "relion_class3d",
        case.case_root / "relion_autorefine.log",
        case.case_root / "relion_class3d.log",
        case.case_root / "relion_evaluate_kclass_gt.log",
    )
    return any(path.exists() for path in relion_paths)


def has_complete_quality_metrics(case: CaseSummary) -> bool:
    if case.fsc_auc_vs_gt is None:
        return False
    if expects_relion_quality(case):
        return case.relion_fsc_auc_vs_gt is not None
    return True


def finalize_status(case: CaseSummary) -> None:
    if case.fsc_auc_vs_gt is not None and case.relion_fsc_auc_vs_gt is not None:
        case.fsc_auc_delta_vs_relion = case.fsc_auc_vs_gt - case.relion_fsc_auc_vs_gt

    if case.exit_status not in (None, 0):
        case.status = "failed"
    if case.status in {"ok", "pass", "passed", "success", "completed"}:
        case.status = "ok"
    elif case.status in {"skipped", "missing"}:
        case.status = "pending" if case.job_id and not case.case_root.exists() else case.status
    elif case.status not in {"failed", "pending", "running", "missing", "ok", "dryrun", "cancelled"}:
        case.status = str(case.status)

    running_logs = (
        case.case_root / "prepare.log",
        case.case_root / "relion_autorefine.log",
        case.case_root / "relion_class3d.log",
        case.case_root / "recovar" / "run_full_refinement.log",
    )
    if case.status == "pending" and any(path.exists() for path in running_logs):
        case.status = "running"
    if (
        case.status == "pending"
        and case.exit_status is None
        and any(case.artifacts.get(label) for label in ("slurm_stdout", "slurm_stderr"))
    ):
        case.status = "running"
    if case.status == "pending" and case.wall_s is not None and case.exit_status == 0:
        case.status = "completed_missing_metrics"
    if case.status == "ok" and case.fsc_auc_vs_gt is not None and not has_complete_quality_metrics(case):
        state_word = (case.slurm_state or "").strip().upper().split()[0] if case.slurm_state else ""
        case.status = "running" if state_word == "RUNNING" or case.exit_status is None else "completed_missing_metrics"
    if case.status == "running" and has_complete_quality_metrics(case):
        case.status = "ok"
    if case.status == "pending" and is_dryrun_case(case):
        case.status = "dryrun"

    if not case.case_name:
        match = re.match(r"(\d+)_(.+)", case.case_root.name)
        if match:
            case.case_id = case.case_id or match.group(1)
            case.case_name = match.group(2)
        else:
            case.case_name = case.case_root.name
    if not case.failure_reason and case.status == "failed" and case.notes:
        case.failure_reason = case.notes[0]
    mark_diagnostic_status(case)


def is_dryrun_case(case: CaseSummary) -> bool:
    job_id = str(case.job_id or "").strip().upper()
    if job_id == "DRYRUN":
        return True
    return any("dryrun" in part.lower() for part in case.case_root.parts)


def natural_chunks(text: str) -> tuple[Any, ...]:
    chunks: list[Any] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        chunks.append(int(part) if part.isdigit() else part.lower())
    return tuple(chunks)


def sort_key(case: CaseSummary) -> tuple[Any, ...]:
    raw_id = case.case_id or ""
    if raw_id.isdigit():
        return (0, int(raw_id), natural_chunks(case.case_name or ""), str(case.scratch_root))
    match = re.match(r"(\d+)[_-](.*)", case.case_name or case.case_root.name)
    if match:
        return (0, int(match.group(1)), natural_chunks(match.group(2)), str(case.scratch_root))
    return (1, natural_chunks(case.case_name or case.case_root.name), str(case.scratch_root))


def mark_superseded_duplicate_failures(cases: list[CaseSummary]) -> None:
    ok_case_names = {case.case_name for case in cases if case.status == "ok" and case.case_name}
    supersedable_statuses = {"failed", "missing", "missing_metrics", "completed_missing_metrics"}
    for case in cases:
        if not case.case_name or case.case_name not in ok_case_names:
            continue
        if case.status not in supersedable_statuses:
            continue
        case.status = "superseded"
        case.notes.append("superseded duplicate; another row for this case has completed metrics")


def duplicate_case_key(case: CaseSummary) -> tuple[str, str, str] | None:
    if not case.case_id or not case.case_name:
        return None
    return (str(case.case_id), str(case.case_name), "diagnostic" if is_diagnostic_case(case) else "default")


def dedupe_cases_by_requested_root(
    cases: list[CaseSummary],
    root_order: dict[str, int],
) -> list[CaseSummary]:
    status_order = {
        "superseded": 0,
        "dryrun": 1,
        "missing": 2,
        "cancelled": 2,
        "diagnostic_pending": 2,
        # A submitted rerun may still be pending when an older duplicate has
        # already failed. Treat pending and failed as equal so the requested
        # root order decides which row is visible.
        "pending": 5,
        "completed_missing_metrics": 4,
        "diagnostic_failed": 4,
        "failed": 5,
        # Active reruns should stay visible in monitors even when an older
        # duplicate already has completed metrics. This includes jobs that
        # are submitted but still pending on Slurm resources. Once the active
        # row completes, root order still selects the later requested root.
        "diagnostic_running": 8,
        "running": 8,
        "diagnostic_ok": 7,
        "timing_probe": 7,
        "ok": 7,
    }
    grouped: dict[tuple[str, str, str], list[CaseSummary]] = {}
    undeduped: list[CaseSummary] = []
    for case in cases:
        key = duplicate_case_key(case)
        if key is None:
            undeduped.append(case)
            continue
        grouped.setdefault(key, []).append(case)

    out = list(undeduped)
    for duplicates in grouped.values():
        if len(duplicates) == 1:
            out.append(duplicates[0])
            continue
        active_duplicates = [
            case
            for case in duplicates
            if case.status in {"pending", "diagnostic_pending", "running", "diagnostic_running"}
        ]
        ok_duplicates = [case for case in duplicates if case.status in {"ok", "diagnostic_ok", "timing_probe"}]
        candidates = active_duplicates or ok_duplicates or duplicates
        selected = max(
            candidates,
            key=lambda case: (
                status_order.get(case.status, 0),
                root_order.get(path_key(case.scratch_root), -1),
            ),
        )
        selected.notes.append(f"omitted {len(duplicates) - 1} duplicate row(s) from earlier requested roots")
        out.append(selected)
    return out


def discover_cases(
    roots: list[Path],
    max_excerpt_lines: int,
    *,
    dedupe_case_reruns: bool = False,
) -> list[CaseSummary]:
    cases: dict[str, CaseSummary] = {}
    root_order: dict[str, int] = {}
    for raw_root in unique_paths(roots):
        scratch_root = raw_root.expanduser().resolve()
        root_order[path_key(scratch_root)] = len(root_order)
        artifacts = walk_artifacts(scratch_root)
        for path in artifacts:
            if path.name in {"case_table.tsv", "selected_cases.tsv"}:
                parse_table(path, scratch_root, cases)
            elif path.name in {"k1_robustness_matrix_summary.json", "k4_mini_summary.json"}:
                merge_aggregate_summary(path, scratch_root, cases)

        for path in artifacts:
            case_root = artifact_case_root(path)
            if case_root is None:
                continue
            case = get_case(cases, scratch_root, case_root)
            if path.name == "case_config.json":
                merge_case_config(case, path)
            elif path.name in {"slurm_walltime.json", "prepare_walltime.json"}:
                merge_walltime(case, path)
            elif path.name == "summary_metrics.json":
                merge_summary_metrics(case, path)
            elif path.name == "kclass_gt_fsc_auc.json":
                merge_kclass_auc(case, path)
            elif path.name == "kclass_gt_fsc.json":
                merge_kclass_gt(case, path)
            elif path.name == "relion_kclass_gt_fsc_auc.json":
                merge_relion_kclass_auc(case, path)
            elif path.name == "relion_kclass_gt_fsc.json":
                merge_relion_kclass_gt(case, path)
            elif path.name == "summary.md":
                case.artifacts["summary_markdown"] = str(path)

    case_list = list(cases.values())
    slurm_accounting = collect_slurm_accounting(case_list)
    for case in case_list:
        merge_log_signals(case, max_excerpt_lines=max_excerpt_lines)
        merge_recovar_convergence(case)
        apply_slurm_accounting(case, slurm_accounting.get(str(case.job_id or "").strip()))
        append_comparison_caveats(case)
        finalize_status(case)
    out = case_list
    mark_superseded_duplicate_failures(out)
    if dedupe_case_reruns:
        out = dedupe_cases_by_requested_root(out, root_order)
    return sorted(out, key=sort_key)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}g}"
    return str(value)


def escape_md(value: Any) -> str:
    return fmt(value).replace("|", "\\|").replace("\n", "<br>")


def escape_notes(notes: list[str], *, limit: int = 3) -> str:
    if not notes:
        return "-"
    trimmed = notes[: max(1, int(limit))]
    suffix = "" if len(notes) <= len(trimmed) else f"; +{len(notes) - len(trimmed)} more"
    return escape_md("; ".join(trimmed) + suffix)


def render_markdown(roots: list[Path], cases: list[CaseSummary], json_path: Path) -> str:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.status] = counts.get(case.status, 0) + 1
    lines = [
        "# EM Robustness Matrix Summary",
        "",
        f"- JSON: `{json_path}`",
        f"- cases: **{len(cases)}**",
    ]
    if counts:
        lines.append("- status counts: " + ", ".join(f"`{key}`={counts[key]}" for key in sorted(counts)))
    lines.append("- roots:")
    for root in roots:
        lines.append(f"  - `{root}`")
    lines.extend(
        [
            "",
            "| # | Case | K | N | Grid | Noise | Poses | Status | RECOVAR iter | RECOVAR converged | Final all-data ran | Final all-data FSC AUC | Final all-data poses | RECOVAR stage | RECOVAR wall s | RECOVAR FSC AUC vs GT | RELION wall s | RELION FSC AUC vs GT | Delta vs RELION | RELION iter | RELION res A | RELION progress | Final res A | Map corr | Notes | Job | Case root | Failure reason | Failure log |",
            "|---:|---|---:|---:|---:|---|---|---|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|---|---|",
        ]
    )
    for case in cases:
        failure_log = f"`{case.failure_log}`" if case.failure_log else "-"
        recovar_iter = None
        if case.recovar_iteration is not None:
            recovar_iter = (
                f"{case.recovar_iteration}/{case.recovar_total_iterations}"
                if case.recovar_total_iterations is not None
                else case.recovar_iteration
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(case.case_id),
                    escape_md(case.case_name),
                    escape_md(case.n_classes),
                    escape_md(case.n_images),
                    escape_md(case.grid_size),
                    escape_md(
                        f"{case.noise_model} {case.noise_level:g}"
                        if case.noise_model is not None and case.noise_level is not None
                        else case.noise_model or case.noise_level
                    ),
                    escape_md(case.poses),
                    escape_md(case.status),
                    escape_md(recovar_iter),
                    escape_md(case.recovar_convergence_has_converged),
                    escape_md(case.recovar_final_all_data_ran),
                    escape_md(case.recovar_final_all_data_fsc_auc),
                    escape_md(case.recovar_has_final_all_data_poses),
                    escape_md(case.recovar_latest_stage),
                    escape_md(case.wall_s),
                    escape_md(case.fsc_auc_vs_gt),
                    escape_md(case.relion_wall_s),
                    escape_md(case.relion_fsc_auc_vs_gt),
                    escape_md(case.fsc_auc_delta_vs_relion),
                    escape_md(case.relion_iteration),
                    escape_md(case.relion_resolution_A),
                    escape_md(case.relion_latest_progress),
                    escape_md(case.final_resolution_A),
                    escape_md(case.map_corr_vs_gt),
                    escape_notes(case.notes),
                    escape_md(case.job_id),
                    f"`{case.case_root}`",
                    escape_md(case.failure_reason),
                    failure_log,
                ]
            )
            + " |"
        )

    failed = [case for case in cases if case.status == "failed" and case.log_excerpt]
    if failed:
        lines.extend(["", "## Failure Log Excerpts", ""])
        for case in failed:
            label = f"{case.case_id or '-'} {case.case_name or case.case_root.name}".strip()
            lines.append(f"### {label}")
            if case.failure_reason:
                lines.append(f"- reason: `{case.failure_reason}`")
            lines.append("")
            lines.append("```text")
            lines.append(case.log_excerpt.rstrip())
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(roots: list[Path], cases: list[CaseSummary], markdown_path: Path, json_path: Path) -> None:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.status] = counts.get(case.status, 0) + 1
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scratch_roots": [str(root) for root in roots],
        "counts": dict(sorted(counts.items())),
        "cases": [case.to_json() for case in cases],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(roots, cases, json_path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = unique_paths([root.expanduser().resolve() for root in args.scratch_roots])
    cases = discover_cases(
        roots,
        max_excerpt_lines=max(1, int(args.log_lines)),
        dedupe_case_reruns=bool(args.dedupe_case_reruns),
    )
    write_outputs(roots, cases, args.output_markdown, args.output_json)
    print(f"Markdown: {args.output_markdown}")
    print(f"JSON: {args.output_json}")
    print(f"Cases: {len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
