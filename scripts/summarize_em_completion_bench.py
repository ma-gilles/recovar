#!/usr/bin/env python
"""Summarize 100k/256 EM completion benchmark outputs.

The script is intentionally CPU-only: it loads final MRC volumes, computes
centered map correlations and NumPy FFT-shell FSCs, reads lightweight timing
ledgers, and writes JSON/Markdown reports. Missing benchmark products are
reported as skipped unless ``--require-all`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

# This reporter only reads files and computes NumPy FSCs. Force CPU before
# importing RECOVAR helpers so JAX does not initialize a busy Slurm GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
if os.environ.get("JAX_PLATFORMS", "").lower() == "cpu":
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

from recovar.utils import helpers


LoadFn = Callable[[Path], np.ndarray]
DEFAULT_FSC_AUC_PARITY_TOL = 1e-4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--k1-recovar-dir", type=Path, help="K=1 RECOVAR output directory.")
    parser.add_argument("--k1-relion-dir", type=Path, help="K=1 RELION output directory.")
    parser.add_argument("--k1-fixture-dir", type=Path, help="K=1 fixture directory with GT/metadata.")
    parser.add_argument("--k4-recovar-dir", type=Path, help="K=4 RECOVAR output directory.")
    parser.add_argument("--k4-relion-dir", type=Path, help="K=4 RELION output directory.")
    parser.add_argument("--k4-fixture-dir", type=Path, help="K=4 fixture directory with GT/metadata.")
    parser.add_argument("--output-json", type=Path, help="Optional path for machine-readable JSON summary.")
    parser.add_argument("--output-markdown", type=Path, help="Optional path for Markdown summary.")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit nonzero if any expected input product is missing.",
    )
    parser.add_argument(
        "--require-k1",
        action="store_true",
        help="Exit nonzero if selected K=1 RECOVAR/RELION products or metrics are missing.",
    )
    parser.add_argument(
        "--require-k4",
        action="store_true",
        help="Exit nonzero if selected K=4 RECOVAR/RELION products or metrics are missing.",
    )
    parser.add_argument(
        "--fsc-auc-parity-tol",
        type=float,
        default=DEFAULT_FSC_AUC_PARITY_TOL,
        help=(
            "For required cases, exit nonzero if RECOVAR GT FSC-AUC is below "
            "RELION GT FSC-AUC by more than this tolerance."
        ),
    )
    return parser.parse_args(argv)


def _load_recovar_volume(path: Path) -> np.ndarray:
    return np.asarray(helpers.load_mrc(str(path)), dtype=np.float64)


def _load_relion_volume(path: Path) -> np.ndarray:
    return np.asarray(helpers.load_relion_volume(str(path)), dtype=np.float64)


def _existing_path(base: Path | None, names: list[str]) -> Path | None:
    if base is None:
        return None
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def _latest_relion_path(base: Path | None, pattern: str) -> Path | None:
    if base is None:
        return None
    candidates = sorted(base.glob(pattern))
    if not candidates:
        return None

    def key(path: Path) -> tuple[int, str]:
        match = re.search(r"run_it(\d+)", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return max(candidates, key=key)


def _latest_relion_iteration_class_path(base: Path | None, class_idx: int) -> Path | None:
    if base is None:
        return None
    class_idx = int(class_idx)
    pattern = f"run_it*_class{class_idx:03d}.mrc"
    exact_name_re = re.compile(rf"^run_it(\d+)_class{class_idx:03d}\.mrc$")
    candidates = [path for path in base.glob(pattern) if exact_name_re.match(path.name)]
    if not candidates:
        return None

    def key(path: Path) -> tuple[int, str]:
        match = exact_name_re.match(path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return max(candidates, key=key)


def _relion_iteration_from_path(path: Path | None, default: int = 15) -> int:
    if path is None:
        return int(default)
    match = re.search(r"run_it(\d+)", path.name)
    return int(match.group(1)) if match else int(default)


def _relion_data_path(base: Path | None, relion_iter: int) -> Path | None:
    exact = _existing_path(base, [f"run_it{int(relion_iter):03d}_data.star"])
    return exact or _latest_relion_path(base, "run_it*_data.star")


def _relion_final_data_path(base: Path | None) -> Path | None:
    return _existing_path(base, ["run_data.star"])


def _relion_iteration_exists(base: Path | None, relion_iter: int) -> bool:
    if base is None:
        return False
    prefix = f"run_it{int(relion_iter):03d}_"
    return any(base.glob(prefix + "*"))


def _load_optional(path: Path | None, load_fn: LoadFn, label: str, notes: list[str]) -> np.ndarray | None:
    if path is None:
        notes.append(f"missing {label}")
        return None
    try:
        return load_fn(path)
    except Exception as exc:
        notes.append(f"failed to load {label} at {path}: {exc}")
        return None


def centered_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=np.float64).reshape(-1)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        return float("nan")
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.dot(a, b) / denom)


def shell_fsc(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return canonical RECOVAR FSC shells, excluding Nyquist edges."""
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or len(set(a.shape)) != 1:
        return np.asarray([], dtype=np.float64)

    n = int(a.shape[0])
    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)
    freqs = np.fft.fftfreq(n) * n
    z, y, x = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    shells = np.rint(np.sqrt(x * x + y * y + z * z)).astype(np.int32).ravel()
    product = (fa * np.conj(fb)).ravel()
    numerator = np.bincount(shells, weights=np.real(product))
    lhs_power = np.bincount(shells, weights=(np.abs(fa) ** 2).ravel())
    rhs_power = np.bincount(shells, weights=(np.abs(fb) ** 2).ravel())
    denom = np.sqrt(lhs_power * rhs_power)
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denom, out=out, where=denom > 0.0)
    return out[: n // 2 - 1]


def integer_shift_to_align_lhs_to_rhs(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    """Estimate the integer voxel roll that best aligns ``lhs`` to ``rhs``."""
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3:
        return {}

    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)
    cross_power = fb * np.conj(fa)
    denom = np.abs(cross_power)
    if not np.any(denom > 0.0):
        return {}
    cross_power = np.divide(cross_power, denom, out=np.zeros_like(cross_power), where=denom > 0.0)
    phase_corr = np.fft.ifftn(cross_power)
    peak_index = np.unravel_index(int(np.argmax(np.abs(phase_corr))), phase_corr.shape)
    shift = []
    for idx, size in zip(peak_index, a.shape, strict=True):
        value = int(idx)
        if value > size // 2:
            value -= int(size)
        shift.append(value)
    return {
        "integer_shift_lhs_to_rhs_zyx": shift,
        "integer_shift_norm_voxels": float(np.linalg.norm(np.asarray(shift, dtype=np.float64))),
        "phase_correlation_peak": float(np.abs(phase_corr[peak_index])),
    }


def mean_shell_value(values: np.ndarray, first_shell: int, last_shell: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    lo = max(0, int(first_shell))
    hi = min(values.size, int(last_shell) + 1)
    if hi <= lo:
        return float("nan")
    finite = values[lo:hi][np.isfinite(values[lo:hi])]
    return float(np.mean(finite)) if finite.size else float("nan")


def finite_max(*values: Any) -> float:
    finite: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            finite.append(parsed)
    return max(finite) if finite else float("nan")


def first_shell_below(values: np.ndarray, threshold: float) -> int | None:
    values = np.asarray(values, dtype=np.float64)
    for shell in range(1, values.size):
        if np.isfinite(values[shell]) and float(values[shell]) < float(threshold):
            return int(shell)
    return None


def normalized_fsc_auc(values: Any, axis: Any | None = None) -> float:
    """Integrate an FSC curve over a normalized shell/radius axis."""
    fsc = np.asarray(values, dtype=np.float64).reshape(-1)
    if fsc.size == 0:
        return float("nan")

    if axis is None:
        x = np.arange(fsc.size, dtype=np.float64)
    else:
        x = np.asarray(axis, dtype=np.float64).reshape(-1)
        if x.size != fsc.size:
            return float("nan")

    finite = np.isfinite(fsc) & np.isfinite(x)
    if finite.size:
        finite[0] = False  # Shell 0/DC is excluded from the existing FSC shell summaries.
    x = x[finite]
    y = fsc[finite]
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return float(y[0])

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0.0 or not math.isfinite(span):
        return float(np.mean(y))
    x_norm = (x - x[0]) / span
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x_norm))


def map_metrics(lhs: np.ndarray, rhs: np.ndarray, *, include_fsc: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"corr": centered_corr(lhs, rhs)}
    if np.isfinite(out["corr"]):
        out["abs_corr"] = abs(float(out["corr"]))
    out.update(integer_shift_to_align_lhs_to_rhs(lhs, rhs))
    if include_fsc:
        fsc = shell_fsc(lhs, rhs)
        shells = np.arange(fsc.size, dtype=np.float64)
        fsc_auc = normalized_fsc_auc(fsc, shells)
        flipped_fsc = -fsc
        flipped_fsc_auc = normalized_fsc_auc(flipped_fsc, shells)
        finite_auc = [
            (float(value), sign)
            for value, sign in ((fsc_auc, 1), (flipped_fsc_auc, -1))
            if np.isfinite(value)
        ]
        if finite_auc:
            sign_invariant_fsc_auc, best_sign = max(finite_auc, key=lambda item: item[0])
        else:
            sign_invariant_fsc_auc = float("nan")
            best_sign = None
        out.update(
            {
                "fsc_auc": fsc_auc,
                "fsc_auc_sign_flipped": flipped_fsc_auc,
                "fsc_auc_sign_invariant": sign_invariant_fsc_auc,
                "sign_invariant_best_sign": best_sign,
                "mean_fsc_1_8": mean_shell_value(fsc, 1, 8),
                "mean_fsc_1_16": mean_shell_value(fsc, 1, 16),
                "mean_fsc_1_8_sign_invariant": finite_max(
                    mean_shell_value(fsc, 1, 8),
                    mean_shell_value(flipped_fsc, 1, 8),
                ),
                "mean_fsc_1_16_sign_invariant": finite_max(
                    mean_shell_value(fsc, 1, 16),
                    mean_shell_value(flipped_fsc, 1, 16),
                ),
                "shell_05": first_shell_below(fsc, 0.5),
                "shell_0143": first_shell_below(fsc, 0.143),
                "fsc_shells": [float(v) for v in shells],
                "fsc": [float(v) for v in fsc],
            }
        )
    return out


def _best_permutation(score_matrix: np.ndarray) -> list[int]:
    scores = np.nan_to_num(np.asarray(score_matrix, dtype=np.float64), nan=-2.0, posinf=-2.0, neginf=-2.0)
    n_rows, n_cols = scores.shape
    if n_rows != n_cols:
        raise ValueError(f"expected square score matrix, got {scores.shape}")
    if n_rows <= 8:
        best_score = -np.inf
        best_perm: tuple[int, ...] | None = None
        for perm in itertools.permutations(range(n_cols)):
            score = float(sum(scores[row, perm[row]] for row in range(n_rows)))
            if score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None
        return [int(v) for v in best_perm]

    perm = [-1] * n_rows
    unused = set(range(n_cols))
    for row in range(n_rows):
        col = max(unused, key=lambda candidate: scores[row, candidate])
        perm[row] = int(col)
        unused.remove(col)
    return perm


def best_permutation_summary(
    lhs_vols: list[np.ndarray],
    rhs_vols: list[np.ndarray],
    *,
    rhs_label: str,
    include_fsc: bool,
    score_key: str | None = None,
) -> dict[str, Any]:
    if score_key is None:
        score_key = "fsc_auc" if include_fsc else "corr"

    pair_metrics: list[list[dict[str, Any]]] = []
    score_matrix = np.full((len(lhs_vols), len(rhs_vols)), np.nan, dtype=np.float64)
    for i, lhs in enumerate(lhs_vols):
        row = []
        for j, rhs in enumerate(rhs_vols):
            metrics = map_metrics(lhs, rhs, include_fsc=include_fsc)
            row.append(metrics)
            score_matrix[i, j] = float(metrics.get(score_key, np.nan))
        pair_metrics.append(row)

    perm = _best_permutation(score_matrix)
    chosen = []
    for i, j in enumerate(perm):
        metrics = dict(pair_metrics[i][j])
        metrics["lhs_class"] = int(i + 1)
        metrics[f"matched_{rhs_label}_class"] = int(j + 1)
        chosen.append(metrics)

    return {
        "permutation_lhs_to_rhs": perm,
        "permutation_score_key": score_key,
        "score_matrix": score_matrix.tolist(),
        "per_class": chosen,
        "mean_corr": _finite_mean([row["corr"] for row in chosen]),
        "mean_fsc_auc": _finite_mean([row.get("fsc_auc", float("nan")) for row in chosen]),
        "mean_fsc_1_8": _finite_mean([row.get("mean_fsc_1_8", float("nan")) for row in chosen]),
        "mean_fsc_1_16": _finite_mean([row.get("mean_fsc_1_16", float("nan")) for row in chosen]),
        "chosen_nonfinite_corr_count": int(sum(not np.isfinite(row["corr"]) for row in chosen)),
    }


def _finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _k1_sign_ambiguity_metadata(fixture_dir: Path | None) -> dict[str, Any]:
    """Return whether K=1 fixture metadata makes global sign arbitrary.

    No-CTF synthetic controls have a contrast/sign ambiguity. RECOVAR and
    RELION can reconstruct the same density with opposite global sign; signed
    FSC-AUC remains useful telemetry, but it is not a valid required-run
    failure gate for that fixture family.
    """

    out: dict[str, Any] = {"allow_global_sign": False}
    if fixture_dir is None:
        return out

    candidate_paths = [
        fixture_dir / "generation_config.json",
        fixture_dir.parent / "case_config.json",
    ]
    for path in candidate_paths:
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        option = str(data.get("dataset_params_option", "")).strip().lower()
        name = str(data.get("name", "")).strip().lower()
        if option == "noctf" or "noctf" in name:
            out.update(
                {
                    "allow_global_sign": True,
                    "reason": "dataset_params_option=noctf" if option == "noctf" else "case name contains noctf",
                    "metadata_path": str(path),
                }
            )
            return out
    return out


def _read_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _parse_duration_seconds(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    for pattern in (
        r"(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)",
        r"(?P<m>\d+):(?P<s>\d{1,2}(?:\.\d+)?)",
    ):
        match = re.search(pattern, text)
        if match:
            h = float(match.groupdict().get("h") or 0.0)
            m = float(match.groupdict().get("m") or 0.0)
            s = float(match.group("s"))
            return float(h * 3600.0 + m * 60.0 + s)
    labeled = re.search(r"(?i)(?:elapsed|walltime|wall_time|real|seconds|time_s)[^0-9]*(\d+(?:\.\d+)?)", text)
    if labeled:
        return float(labeled.group(1))
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def _csv_column(fieldnames: list[str] | None, key: str) -> str | None:
    if not fieldnames:
        return None
    key = key.lower()
    for fieldname in fieldnames:
        normalized = fieldname.strip().lower()
        if normalized == key or normalized.startswith(f"{key} ") or key in normalized:
            return fieldname
    return None


def _parse_mib(value: Any) -> float | None:
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        parsed = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _read_gpu_monitor(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None

    summary: dict[str, Any] = {
        "path": str(path),
        "sample_count": 0,
        "gpu_count": 0,
        "peak_memory_mib": None,
        "peak_memory_gib": None,
        "peak_memory_total_mib": None,
        "peak_memory_total_gib": None,
        "peak_device_index": None,
        "peak_device_name": None,
        "peak_timestamp": None,
        "notes": [],
    }
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            used_col = _csv_column(reader.fieldnames, "memory.used")
            if used_col is None:
                summary["notes"].append("missing memory.used column")
                return summary
            total_col = _csv_column(reader.fieldnames, "memory.total")
            index_col = _csv_column(reader.fieldnames, "index")
            name_col = _csv_column(reader.fieldnames, "name")
            timestamp_col = _csv_column(reader.fieldnames, "timestamp")

            peak_mib: float | None = None
            peak_row: dict[str, Any] = {}
            device_indices: set[str] = set()
            for row in reader:
                used_mib = _parse_mib(row.get(used_col))
                if used_mib is None:
                    continue
                summary["sample_count"] += 1
                if index_col is not None:
                    device_index = str(row.get(index_col, "")).strip()
                    if device_index:
                        device_indices.add(device_index)
                if peak_mib is None or used_mib > peak_mib:
                    peak_mib = used_mib
                    peak_row = row
    except Exception as exc:
        summary["notes"].append(f"failed to read gpu_monitor.csv: {exc}")
        return summary

    summary["gpu_count"] = len(device_indices)
    if peak_mib is None:
        summary["notes"].append("no parseable memory.used samples")
        return summary

    total_mib = _parse_mib(peak_row.get(total_col)) if total_col is not None else None
    summary["peak_memory_mib"] = float(peak_mib)
    summary["peak_memory_gib"] = float(peak_mib / 1024.0)
    summary["peak_memory_total_mib"] = float(total_mib) if total_mib is not None else None
    summary["peak_memory_total_gib"] = float(total_mib / 1024.0) if total_mib is not None else None
    summary["peak_device_index"] = str(peak_row.get(index_col, "")).strip() if index_col is not None else None
    summary["peak_device_name"] = str(peak_row.get(name_col, "")).strip() if name_col is not None else None
    summary["peak_timestamp"] = str(peak_row.get(timestamp_col, "")).strip() if timestamp_col is not None else None
    return summary


def _completion_metadata(recovar_dir: Path | None, label: str) -> dict[str, Any]:
    if recovar_dir is None:
        return {}

    case_root = recovar_dir.parent
    scratch_dir = next(
        (candidate for candidate in (case_root, *case_root.parents) if (candidate / "submission.env").is_file()),
        case_root,
    )
    env_path = scratch_dir / "submission.env"
    env = _read_env_file(env_path)
    job_key = "K1_JOB_ID" if label == "k1" else "K4_JOB_ID"
    log_prefix = "em_completion_k1_100k256" if label == "k1" else "em_completion_k4_100k256"
    script_name = f"{log_prefix}.sh"
    slurm_walltime_path = recovar_dir / "slurm_walltime.json"
    slurm_walltime = _read_json(slurm_walltime_path) if slurm_walltime_path.exists() else None
    job_id = env.get(job_key) or (slurm_walltime or {}).get("slurm_job_id")

    matrix_row: dict[str, str] | None = None
    case_table_path = scratch_dir / "selected_cases.tsv"
    if label == "k1" and case_table_path.is_file():
        try:
            with case_table_path.open(newline="") as stream:
                for row in csv.DictReader(stream, delimiter="|"):
                    row_root = row.get("case_root")
                    if row_root and Path(row_root).resolve() == case_root.resolve():
                        matrix_row = row
                        break
        except (OSError, csv.Error):
            matrix_row = None

    if matrix_row is not None:
        log_prefix = f"em_k1_matrix_{matrix_row['index']}_{matrix_row['name']}"
        script_name = f"{log_prefix}.sh"
        job_id = job_id or matrix_row.get("case_job_id")

    job_provenance_dir = scratch_dir / "job_provenance" / f"{log_prefix}_{job_id}"
    git_head_path = job_provenance_dir / "git_head.txt"
    git_branch_path = job_provenance_dir / "git_branch.txt"
    git_status_path = job_provenance_dir / "git_status_porcelain.txt"

    def _read_optional_text(path: Path) -> str | None:
        if not path.is_file():
            return None
        value = path.read_text().strip()
        return value or None

    known_paths = {
        "submission_env": env_path,
        "slurm_stdout": scratch_dir / f"{log_prefix}.out",
        "slurm_stderr": scratch_dir / f"{log_prefix}.err",
        "job_script": scratch_dir / "jobs" / script_name,
        "run_log": recovar_dir / "run_full_refinement.log",
        "slurm_walltime": slurm_walltime_path,
    }
    if matrix_row is not None:
        known_paths.update(
            {
                "job_provenance": job_provenance_dir,
                "git_head": git_head_path,
                "git_branch": git_branch_path,
                "git_status_porcelain": git_status_path,
            }
        )
    missing_artifacts = [name for name, path in known_paths.items() if not path.exists()]
    return {
        "submission_env_path": str(env_path) if env_path.exists() else None,
        "repo_root": env.get("REPO_ROOT"),
        "head": _read_optional_text(git_head_path) or env.get("HEAD"),
        "branch": _read_optional_text(git_branch_path) or env.get("BRANCH"),
        "scratch_dir": env.get("SCRATCH_DIR") or str(scratch_dir),
        "setup_job_id": env.get("SETUP_JOB_ID"),
        "job_id": job_id,
        "summary_job_id": env.get("SUMMARY_JOB_ID"),
        "slurm_stdout": str(scratch_dir / f"{log_prefix}.out") if (scratch_dir / f"{log_prefix}.out").exists() else None,
        "slurm_stderr": str(scratch_dir / f"{log_prefix}.err") if (scratch_dir / f"{log_prefix}.err").exists() else None,
        "job_script": str(scratch_dir / "jobs" / script_name)
        if (scratch_dir / "jobs" / script_name).exists()
        else None,
        "run_log": str(recovar_dir / "run_full_refinement.log")
        if (recovar_dir / "run_full_refinement.log").exists()
        else None,
        "slurm_walltime_path": str(slurm_walltime_path) if slurm_walltime_path.exists() else None,
        "slurm_walltime": slurm_walltime,
        "job_provenance_dir": str(job_provenance_dir) if job_provenance_dir.exists() else None,
        "git_status_porcelain": _read_optional_text(git_status_path),
        "known_paths": {key: str(path) for key, path in known_paths.items()},
        "missing_artifacts": missing_artifacts,
        "env": {
            key: env.get(key)
            for key in (
                "RUN_K4_FUSED_SPARSE_PASS2",
                "K1_MAX_ITER",
                "K1_MEM",
                "K1_TIME_LIMIT",
                "K4_MAX_ITER",
                "K4_MEM",
                "K4_TIME_LIMIT",
                "RECOVAR_SPARSE_KCLASS_FUSED",
                "RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES",
                "RECOVAR_SPARSE_PASS2_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES",
                "RECOVAR_SPARSE_PASS2_SMALL_BUCKET_THRESHOLD",
                "RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES",
                "RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES",
                "RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS",
                "RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES",
                "RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES",
                "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES",
                "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION",
                "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS",
                "RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS",
                "RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT",
                "RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS",
                "RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS",
                "RECOVAR_SPARSE_KCLASS_GROUP_TIMING",
                "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS",
                "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE",
                "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL",
                "RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE",
                "RECOVAR_K_CLASS_DENSE_PASS2_SUPPORT_FRACTION",
                "RECOVAR_K_CLASS_DENSE_PASS2_MEAN_SUPPORT_FRACTION",
                "RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_IMAGES",
                "RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_MEAN_SUPPORT_FRACTION",
                "RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION",
                "RECOVAR_PASS1_FUSED",
                "RECOVAR_DISABLE_LOCAL_BIG_JIT",
                "RECOVAR_LOCAL_BUCKET_QUANTUM",
                "EM_COMPLETION_TIMING_PROBE",
                "TF_GPU_ALLOCATOR",
            )
            if env.get(key) not in (None, "")
        },
    }


def _env_flag(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _section_has_timing_probe_env(section: dict[str, Any]) -> bool:
    metadata = section.get("metadata")
    if not isinstance(metadata, dict):
        return False
    env = metadata.get("env")
    if not isinstance(env, dict):
        return False
    return _env_flag(env.get("EM_COMPLETION_TIMING_PROBE"))


def _annotate_timing_probe_status(summary: dict[str, Any]) -> None:
    for case in ("k1", "k4"):
        section = summary.get(case)
        if not isinstance(section, dict) or not _section_has_timing_probe_env(section):
            continue
        label = "K=1" if case == "k1" else "K=4"
        note = (
            f"{label} timing probe only: not a correctness acceptance run; "
            "GT FSC-AUC gates are diagnostic because max_iter may be truncated"
        )
        notes = section.setdefault("notes", [])
        if note not in notes:
            notes.append(note)
        if section.get("status") == "ok":
            section["status"] = "timing_probe"


def _parse_value_token(raw_value: str) -> Any:
    raw = raw_value.strip()
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    match = re.match(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", raw)
    if match is None:
        return raw
    number_text = match.group(0)
    try:
        value = float(number_text)
    except ValueError:
        return raw
    if math.isfinite(value) and re.fullmatch(r"[-+]?\d+", number_text):
        return int(value)
    return value


def _parse_key_value_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=([^,;\)\s]*)", text):
        key = match.group(1)
        value = match.group(2).strip()
        if value == "":
            continue
        fields[key] = _parse_value_token(value)
    return fields


def _parse_relion_iteration_result(line: str, line_number: int) -> dict[str, Any] | None:
    match = re.search(r"RELION Iteration\s+(?P<iteration>\d+):\s*(?P<body>.*)$", line)
    if match is None:
        return None
    fields = _parse_key_value_fields(match.group("body"))
    row: dict[str, Any] = {
        "iteration": int(match.group("iteration")),
        "line_number": int(line_number),
    }
    key_map = {
        "current_size": "current_size",
        "pixel_res": "pixel_res",
        "res": "res_ang",
        "ave_Pmax": "ave_pmax",
        "healpix_order": "healpix_order",
        "converged": "converged",
        "time": "wall_time_s",
    }
    for source_key, output_key in key_map.items():
        if source_key in fields:
            row[output_key] = fields[source_key]
    return row


def _parse_iteration_start(line: str) -> dict[str, Any] | None:
    match = re.search(
        r"RELION Iteration\s+(?P<iteration>\d+)/(?:\d+):\s+"
        r"current_size=(?P<current_size>\d+),\s+"
        r"healpix_order=(?P<healpix_order>\d+),\s+"
        r"local_search=(?P<local_search>True|False|true|false)",
        line,
    )
    if match is None:
        return None
    return {
        "iteration": int(match.group("iteration")),
        "current_size": int(match.group("current_size")),
        "healpix_order": int(match.group("healpix_order")),
        "local_search": match.group("local_search").lower() == "true",
    }


def _parse_batch_sizing_line(
    line: str,
    *,
    current_iteration: int | None,
    current_size: int | None,
    line_number: int,
) -> dict[str, Any] | None:
    sizing_re = re.compile(
        r"requested image_batch_size=(?P<requested_image_batch_size>\d+)\s+"
        r"rotation_block_size=(?P<requested_rotation_block_size>\d+);\s+"
        r"using image_batch_size=(?P<image_batch_size>\d+)\s+"
        r"rotation_block_size=(?P<rotation_block_size>\d+).*?"
        r"n_rot=(?P<n_rot>\d+)\s+n_trans=(?P<n_trans>\d+)\s+K=(?P<n_classes>\d+)"
    )
    sizing_match = sizing_re.search(line)
    if sizing_match is None:
        return None
    event = {key: int(value) for key, value in sizing_match.groupdict().items()}
    event["iteration"] = current_iteration
    event["current_size"] = current_size
    event["line_number"] = int(line_number)
    for key, value in _parse_key_value_fields(line).items():
        if key in {
            "score_budget",
            "score_pixels",
            "projection_tile",
            "active_score_tile",
            "pose_pixel_tile",
            "translation_tile",
            "persistent_est",
            "usable_est",
            "gpu_used_est",
        }:
            event[key] = value
    return event


def _parse_slash_numbers(text: str, expected_count: int) -> list[float] | None:
    parts = text.strip().split("/")
    if len(parts) != expected_count:
        return None
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            return None
    return values


def _assign_int_alias(event: dict[str, Any], source_key: str, alias: str) -> None:
    value = event.get(source_key)
    if isinstance(value, (int, np.integer)):
        event[alias] = int(value)


def _parse_sparse_bucketing_line(
    line: str,
    *,
    current_iteration: int | None,
    current_size: int | None,
    line_number: int,
) -> dict[str, Any] | None:
    if "Sparse fused K-class pass-2 bucketing:" in line:
        match = re.search(
            r"Sparse fused K-class pass-2 bucketing:\s+"
            r"(?P<n_images>\d+) images x (?P<n_classes>\d+) classes -> (?P<buckets>\d+) buckets",
            line,
        )
        fused = True
    elif "Sparse pass-2 bucketing:" in line:
        match = re.search(
            r"Sparse pass-2 bucketing:\s+(?P<n_images>\d+) images -> (?P<buckets>\d+) buckets",
            line,
        )
        fused = False
    else:
        return None
    if match is None:
        return None

    event: dict[str, Any] = {
        "event": "bucketing",
        "fused": fused,
        "iteration": current_iteration,
        "current_size": current_size,
        "line_number": int(line_number),
        "n_images": int(match.group("n_images")),
        "buckets": int(match.group("buckets")),
    }
    if fused:
        event["n_classes"] = int(match.group("n_classes"))

    bucket_match = re.search(r"bucket_size min/med/mean/max=([^,\)]*)", line)
    if bucket_match is not None:
        values = _parse_slash_numbers(bucket_match.group(1), 4)
        if values is not None:
            event.update(
                {
                    "bucket_size_min": int(values[0]),
                    "bucket_size_median": float(values[1]),
                    "bucket_size_mean": float(values[2]),
                    "bucket_size_max": int(values[3]),
                }
            )
    images_match = re.search(r"images_per_bucket med/max=([^,\)]*)", line)
    if images_match is not None:
        values = _parse_slash_numbers(images_match.group(1), 2)
        if values is not None:
            event["images_per_bucket_median"] = float(values[0])
            event["images_per_bucket_max"] = int(values[1])
    top_match = re.search(r"top_bucket_counts=\[(.*?)\];", line)
    if top_match is not None:
        event["top_bucket_counts"] = top_match.group(1)

    for key, value in _parse_key_value_fields(line).items():
        if key in {
            "max_hypotheses_per_microbatch",
            "max_images_per_microbatch",
            "small_bucket_threshold",
            "small_bucket_max_images_per_microbatch",
            "max_projected_rotations_per_projection_call",
            "max_translation_tile_bytes",
            "n_score_pixels",
            "device_memory_gib",
        }:
            event[key] = value
    _assign_int_alias(event, "max_projected_rotations_per_projection_call", "projected_rotation_cap")
    _assign_int_alias(event, "max_translation_tile_bytes", "translation_tile_cap_bytes")
    return event


def _parse_sparse_runtime_line(
    line: str,
    *,
    current_iteration: int | None,
    current_size: int | None,
    line_number: int,
) -> dict[str, Any] | None:
    if "Sparse fused K-class pass-2:" in line and "bucketing" not in line:
        match = re.search(
            r"Sparse fused K-class pass-2:\s+"
            r"(?P<n_images>\d+) images,\s+(?P<n_classes>\d+) classes,\s+"
            r"(?P<buckets>\d+) buckets,\s+(?P<e_m_time_s>[-+]?\d+(?:\.\d+)?)s E\+M;\s+"
            r"median local rot=(?P<median_local_rot>[-+]?\d+(?:\.\d+)?),\s+"
            r"mean local rot=(?P<mean_local_rot>[-+]?\d+(?:\.\d+)?),\s+"
            r"median valid candidates/image=(?P<median_valid_candidates_per_image>[-+]?\d+(?:\.\d+)?)",
            line,
        )
        fused = True
    elif "Sparse pass-2 (bucketed):" in line:
        match = re.search(
            r"Sparse pass-2 \(bucketed\):\s+"
            r"(?P<n_images>\d+) images,\s+(?P<buckets>\d+) buckets,\s+"
            r"(?P<e_m_time_s>[-+]?\d+(?:\.\d+)?)s E\+M;\s+"
            r"median local rot=(?P<median_local_rot>[-+]?\d+(?:\.\d+)?),\s+"
            r"mean local rot=(?P<mean_local_rot>[-+]?\d+(?:\.\d+)?),\s+"
            r"median valid candidates/image=(?P<median_valid_candidates_per_image>[-+]?\d+(?:\.\d+)?)",
            line,
        )
        fused = False
    else:
        return None
    if match is None:
        return None

    event: dict[str, Any] = {
        "event": "runtime",
        "fused": fused,
        "iteration": current_iteration,
        "current_size": current_size,
        "line_number": int(line_number),
        "n_images": int(match.group("n_images")),
        "buckets": int(match.group("buckets")),
        "e_m_time_s": float(match.group("e_m_time_s")),
        "median_local_rot": float(match.group("median_local_rot")),
        "mean_local_rot": float(match.group("mean_local_rot")),
        "median_valid_candidates_per_image": float(match.group("median_valid_candidates_per_image")),
    }
    if fused:
        event["n_classes"] = int(match.group("n_classes"))
    return event


def _parse_sparse_group_timing_line(
    line: str,
    *,
    current_iteration: int | None,
    current_size: int | None,
    line_number: int,
) -> dict[str, Any] | None:
    if "Sparse fused K-class pass-2 bucket group timing:" not in line:
        return None
    fields = _parse_key_value_fields(line)
    key_match = re.search(
        r"bucket group timing:\s+mode=(?P<mode>\S+)\s+"
        r"(?P<group_key>[A-Za-z_][A-Za-z0-9_]*)=(?P<group_value>\d+)",
        line,
    )
    event: dict[str, Any] = {
        "iteration": current_iteration,
        "current_size": current_size,
        "line_number": int(line_number),
        "mode": fields.get("mode"),
    }
    if key_match is not None:
        event["mode"] = key_match.group("mode")
        event["group_key"] = key_match.group("group_key")
        event["group_value"] = int(key_match.group("group_value"))
    for key in (
        "build",
        "fetch",
        "prepare",
        "score",
        "mstep_noise_stats",
        "mstep_weighted_sums",
        "mstep_adjoint",
        "noise",
        "stats",
        "total_profiled",
        "wall",
    ):
        if key in fields:
            event[f"{key}_s"] = fields[key]
    return event


def _parse_compact_pair_planner_line(
    line: str,
    *,
    current_iteration: int | None,
    current_size: int | None,
    line_number: int,
) -> dict[str, Any] | None:
    if "Sparse fused K-class compact-pair planner:" not in line:
        return None
    fields = _parse_key_value_fields(line)
    event: dict[str, Any] = {
        "iteration": current_iteration,
        "current_size": current_size,
        "line_number": int(line_number),
    }
    for key in (
        "valid_pair_candidates",
        "padded_pair_candidates",
        "rectangular_candidates",
        "valid_reduction",
        "padded_reduction",
        "compact_buckets",
        "plan_time",
    ):
        if key in fields:
            event[key] = fields[key]

    pair_stats_match = re.search(
        r"median_valid_pairs/image=(?P<median>[-+]?\d+(?:\.\d+)?),\s+"
        r"mean_valid_pairs/image=(?P<mean>[-+]?\d+(?:\.\d+)?),\s+"
        r"max_valid_pairs/image=(?P<max>[-+]?\d+(?:\.\d+)?)",
        line,
    )
    if pair_stats_match is not None:
        event["median_valid_pairs_per_image"] = _parse_value_token(pair_stats_match.group("median"))
        event["mean_valid_pairs_per_image"] = _parse_value_token(pair_stats_match.group("mean"))
        event["max_valid_pairs_per_image"] = _parse_value_token(pair_stats_match.group("max"))
    return event


def _parse_local_significant_support_line(
    line: str,
    *,
    current_iteration: int | None,
    current_size: int | None,
    line_number: int,
) -> dict[str, Any] | None:
    if "Exact local significant-support summary:" not in line:
        return None
    fields = _parse_key_value_fields(line)
    event: dict[str, Any] = {
        "iteration": current_iteration,
        "current_size": current_size,
        "line_number": int(line_number),
    }
    for key in (
        "chunks",
        "big_jit_buckets",
        "sparse_big_jit_buckets",
        "reconstruction_rows",
        "padded_rows",
        "significant_samples",
        "mean_reconstruction_rows_per_image",
        "mean_significant_samples_per_image",
    ):
        if key in fields:
            event[key] = fields[key]
    reconstruction_rows = event.get("reconstruction_rows")
    padded_rows = event.get("padded_rows")
    significant_samples = event.get("significant_samples")
    if isinstance(padded_rows, (int, float)) and float(padded_rows) > 0.0:
        if isinstance(reconstruction_rows, (int, float)):
            event["padding_to_reconstruction_ratio"] = float(padded_rows) / max(float(reconstruction_rows), 1.0)
        if isinstance(significant_samples, (int, float)):
            event["samples_per_padded_row"] = float(significant_samples) / max(float(padded_rows), 1.0)
    return event


def _merge_sparse_runtime_event(events: list[dict[str, Any]], runtime: dict[str, Any]) -> None:
    for event in reversed(events):
        if event.get("event") not in ("bucketing", "bucketing+runtime"):
            continue
        if event.get("e_m_time_s") is not None:
            continue
        if event.get("fused") != runtime.get("fused"):
            continue
        if event.get("iteration") != runtime.get("iteration"):
            continue
        if event.get("n_images") != runtime.get("n_images"):
            continue
        if event.get("buckets") != runtime.get("buckets"):
            continue
        event.update({key: value for key, value in runtime.items() if key not in ("event", "line_number")})
        event["event"] = "bucketing+runtime"
        event["runtime_line_number"] = runtime.get("line_number")
        return
    events.append(runtime)


def _parse_run_log_telemetry(log_path: Path | None) -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "path": str(log_path) if log_path is not None else None,
        "status": "missing",
        "missing_artifacts": [],
        "notes": [],
        "iteration_rows": [],
        "batch_sizing_events": [],
        "sparse_pass2_events": [],
        "sparse_group_timing_events": [],
        "compact_pair_planner_events": [],
        "local_search_events": [],
        "counts": {
            "iteration_rows": 0,
            "batch_sizing_events": 0,
            "sparse_pass2_events": 0,
            "sparse_group_timing_events": 0,
            "compact_pair_planner_events": 0,
            "local_search_events": 0,
        },
    }
    if log_path is None:
        telemetry["missing_artifacts"].append("run_full_refinement.log")
        telemetry["notes"].append("RECOVAR run log path was not provided")
        return telemetry
    if not log_path.exists():
        telemetry["missing_artifacts"].append(str(log_path))
        telemetry["notes"].append(f"missing RECOVAR run log at {log_path}")
        return telemetry

    current_iteration: int | None = None
    current_size: int | None = None
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except Exception as exc:
        telemetry["status"] = "failed"
        telemetry["notes"].append(f"failed to read RECOVAR run log at {log_path}: {exc}")
        return telemetry

    telemetry["status"] = "ok"
    for line_number, line in enumerate(lines, start=1):
        start = _parse_iteration_start(line)
        if start is not None:
            current_iteration = int(start["iteration"])
            current_size = int(start["current_size"])
            continue

        iteration_row = _parse_relion_iteration_result(line, line_number)
        if iteration_row is not None:
            telemetry["iteration_rows"].append(iteration_row)
            continue

        batch_event = _parse_batch_sizing_line(
            line,
            current_iteration=current_iteration,
            current_size=current_size,
            line_number=line_number,
        )
        if batch_event is not None:
            telemetry["batch_sizing_events"].append(batch_event)
            continue

        sparse_bucketing = _parse_sparse_bucketing_line(
            line,
            current_iteration=current_iteration,
            current_size=current_size,
            line_number=line_number,
        )
        if sparse_bucketing is not None:
            telemetry["sparse_pass2_events"].append(sparse_bucketing)
            continue

        sparse_runtime = _parse_sparse_runtime_line(
            line,
            current_iteration=current_iteration,
            current_size=current_size,
            line_number=line_number,
        )
        if sparse_runtime is not None:
            _merge_sparse_runtime_event(telemetry["sparse_pass2_events"], sparse_runtime)
            continue

        sparse_group_timing = _parse_sparse_group_timing_line(
            line,
            current_iteration=current_iteration,
            current_size=current_size,
            line_number=line_number,
        )
        if sparse_group_timing is not None:
            telemetry["sparse_group_timing_events"].append(sparse_group_timing)
            continue

        compact_pair_planner = _parse_compact_pair_planner_line(
            line,
            current_iteration=current_iteration,
            current_size=current_size,
            line_number=line_number,
        )
        if compact_pair_planner is not None:
            telemetry["compact_pair_planner_events"].append(compact_pair_planner)
            continue

        local_event = _parse_local_significant_support_line(
            line,
            current_iteration=current_iteration,
            current_size=current_size,
            line_number=line_number,
        )
        if local_event is not None:
            telemetry["local_search_events"].append(local_event)

    telemetry["counts"] = {
        "iteration_rows": len(telemetry["iteration_rows"]),
        "batch_sizing_events": len(telemetry["batch_sizing_events"]),
        "sparse_pass2_events": len(telemetry["sparse_pass2_events"]),
        "sparse_group_timing_events": len(telemetry["sparse_group_timing_events"]),
        "compact_pair_planner_events": len(telemetry["compact_pair_planner_events"]),
        "local_search_events": len(telemetry["local_search_events"]),
    }
    if telemetry["counts"]["iteration_rows"] == 0:
        telemetry["notes"].append("no completed RELION Iteration telemetry rows found in RECOVAR run log")
    if telemetry["counts"]["batch_sizing_events"] == 0:
        telemetry["notes"].append("no RELION EM batch sizing rows found in RECOVAR run log")
    if telemetry["counts"]["sparse_pass2_events"] == 0:
        telemetry["notes"].append("no sparse pass-2 telemetry rows found in RECOVAR run log")
    return telemetry


def _parse_batch_sizing(log_path: Path | None) -> list[dict[str, Any]]:
    return list(_parse_run_log_telemetry(log_path).get("batch_sizing_events") or [])


def _sparse_pass2_aggregate(
    events: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    timed_events = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("e_m_time_s") is not None
        and math.isfinite(float(event.get("e_m_time_s")))
    ]
    if not timed_events:
        return None

    event_times = [float(event["e_m_time_s"]) for event in timed_events]
    completed_iterations = {
        int(row["iteration"])
        for row in iteration_rows
        if isinstance(row, dict)
        and row.get("iteration") is not None
        and row.get("wall_time_s") is not None
        and math.isfinite(float(row.get("wall_time_s")))
    }
    matched_times = [
        float(event["e_m_time_s"])
        for event in timed_events
        if event.get("iteration") is not None and int(event["iteration"]) in completed_iterations
    ]
    wall_times = [
        float(row["wall_time_s"])
        for row in iteration_rows
        if isinstance(row, dict)
        and row.get("iteration") is not None
        and int(row["iteration"]) in completed_iterations
        and row.get("wall_time_s") is not None
        and math.isfinite(float(row.get("wall_time_s")))
    ]
    matched_total = float(sum(matched_times)) if matched_times else None
    wall_total = float(sum(wall_times)) if wall_times else None
    fraction = None
    if matched_total is not None and wall_total is not None and wall_total > 0.0:
        fraction = float(matched_total / wall_total)

    fused_count = sum(1 for event in timed_events if event.get("fused") is True)
    return {
        "sparse_pass2_count": int(len(timed_events)),
        "sparse_pass2_fused_count": int(fused_count),
        "sparse_pass2_nonfused_count": int(len(timed_events) - fused_count),
        "sparse_pass2_total_s": float(sum(event_times)),
        "sparse_pass2_mean_s": float(np.mean(event_times)),
        "sparse_pass2_median_s": float(np.median(event_times)),
        "sparse_pass2_max_s": float(max(event_times)),
        "sparse_pass2_completed_iteration_total_s": matched_total,
        "sparse_pass2_completed_iteration_wall_s": wall_total,
        "sparse_pass2_fraction_of_completed_iteration_wall": fraction,
    }


def _finite_event_float(event: dict[str, Any], key: str) -> float | None:
    try:
        value = float(event.get(key))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _sparse_group_timing_aggregate(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [event for event in events if isinstance(event, dict)]
    if not valid:
        return None

    stage_keys = (
        "build_s",
        "fetch_s",
        "prepare_s",
        "score_s",
        "mstep_noise_stats_s",
        "mstep_weighted_sums_s",
        "mstep_adjoint_s",
        "noise_s",
        "stats_s",
        "total_profiled_s",
        "wall_s",
    )
    output_names = {
        "build_s": "sparse_kclass_group_build_total_s",
        "fetch_s": "sparse_kclass_group_fetch_total_s",
        "prepare_s": "sparse_kclass_group_prepare_total_s",
        "score_s": "sparse_kclass_group_score_total_s",
        "mstep_noise_stats_s": "sparse_kclass_group_mstep_noise_stats_total_s",
        "mstep_weighted_sums_s": "sparse_kclass_group_mstep_weighted_sums_total_s",
        "mstep_adjoint_s": "sparse_kclass_group_mstep_adjoint_total_s",
        "noise_s": "sparse_kclass_group_noise_total_s",
        "stats_s": "sparse_kclass_group_stats_total_s",
        "total_profiled_s": "sparse_kclass_group_profiled_total_s",
        "wall_s": "sparse_kclass_group_wall_total_s",
    }

    totals = {key: 0.0 for key in stage_keys}
    counts = {key: 0 for key in stage_keys}
    by_mode: dict[str, dict[str, Any]] = {}
    for event in valid:
        mode = str(event.get("mode") or "unknown")
        mode_row = by_mode.setdefault(mode, {"count": 0})
        mode_row["count"] = int(mode_row["count"]) + 1
        for key in stage_keys:
            value = _finite_event_float(event, key)
            if value is None:
                continue
            totals[key] += value
            counts[key] += 1
            mode_row[output_names[key].replace("sparse_kclass_group_", "")] = float(
                mode_row.get(output_names[key].replace("sparse_kclass_group_", ""), 0.0) + value
            )

    if not any(counts.values()):
        return None

    aggregate: dict[str, Any] = {
        "sparse_kclass_group_timing_count": int(len(valid)),
        "sparse_kclass_group_timing_numeric_count": int(max(counts.values())),
    }
    for key in stage_keys:
        if counts[key] > 0:
            aggregate[output_names[key]] = float(totals[key])

    wall_total = aggregate.get("sparse_kclass_group_wall_total_s")
    if isinstance(wall_total, (int, float)) and float(wall_total) > 0.0:
        for key in stage_keys:
            if key == "wall_s" or counts[key] <= 0:
                continue
            fraction_key = output_names[key].replace("_total_s", "_fraction_of_group_wall")
            aggregate[fraction_key] = float(totals[key] / float(wall_total))
        for mode_row in by_mode.values():
            mode_wall = mode_row.get("wall_total_s")
            if isinstance(mode_wall, (int, float)) and float(mode_wall) > 0.0:
                for short_key in (
                    "build_total_s",
                    "fetch_total_s",
                    "prepare_total_s",
                    "score_total_s",
                    "mstep_noise_stats_total_s",
                    "mstep_weighted_sums_total_s",
                    "mstep_adjoint_total_s",
                    "noise_total_s",
                    "stats_total_s",
                    "profiled_total_s",
                ):
                    if short_key in mode_row:
                        mode_row[short_key.replace("_total_s", "_fraction_of_wall")] = float(
                            mode_row[short_key] / float(mode_wall)
                        )
    aggregate["sparse_kclass_group_by_mode"] = by_mode
    return aggregate


def _global_profile_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if isinstance(row, dict)]
    if not valid:
        return None

    phase_counts: dict[str, int] = {}
    time_totals: dict[str, float] = {}
    kclass_rows = 0
    for row in valid:
        phase = str(row.get("phase") or "unknown")
        phase_counts[phase] = int(phase_counts.get(phase, 0) + 1)
        if bool(row.get("k_class_enabled")):
            kclass_rows += 1
        for key, value in row.items():
            if not str(key).endswith("_s"):
                continue
            numeric = _finite_event_float(row, str(key))
            if numeric is None:
                continue
            time_totals[str(key)] = float(time_totals.get(str(key), 0.0) + numeric)

    if not time_totals and not phase_counts:
        return None

    top_time_keys = sorted(time_totals.items(), key=lambda item: item[1], reverse=True)
    return {
        "global_profile_row_count": int(len(valid)),
        "global_profile_kclass_row_count": int(kclass_rows),
        "global_profile_phase_counts": phase_counts,
        "global_profile_time_totals_s": time_totals,
        "global_profile_top_time_totals_s": [
            {"key": key, "seconds": float(value)} for key, value in top_time_keys[:12]
        ],
    }


def _timing_summary(recovar_dir: Path | None, relion_dir: Path | None) -> dict[str, Any]:
    notes: list[str] = []
    ledger_path = recovar_dir / "benchmark_ledger.json" if recovar_dir is not None else None
    ledger = _read_json(ledger_path) if ledger_path is not None else None
    run_log_path = recovar_dir / "run_full_refinement.log" if recovar_dir is not None else None
    log_telemetry = _parse_run_log_telemetry(run_log_path)
    recovar_walltime = None
    n_images = None
    if ledger is None:
        notes.append("missing RECOVAR benchmark_ledger.json")
    else:
        recovar_walltime = _first_number(ledger, ("total_time_s", "elapsed_s", "walltime_s"))
        n_images = _first_number(ledger, ("n_images", "num_images"))
        if n_images is None:
            data = ledger.get("dataset", {}) if isinstance(ledger.get("dataset"), dict) else {}
            n_images = _first_number(data, ("n_images", "num_images"))

    relion_walltime = None
    relion_walltime_path = relion_dir / "relion_walltime.txt" if relion_dir is not None else None
    relion_slurm_walltime_path = relion_dir / "slurm_walltime.json" if relion_dir is not None else None
    if relion_dir is None:
        relion_walltime_path = None
    elif relion_walltime_path is None or not relion_walltime_path.exists():
        relion_slurm_walltime = (
            _read_json(relion_slurm_walltime_path)
            if relion_slurm_walltime_path is not None and relion_slurm_walltime_path.exists()
            else None
        )
        relion_walltime = (
            _first_number(relion_slurm_walltime, ("external_wall_s", "walltime_s", "elapsed_s"))
            if relion_slurm_walltime is not None
            else None
        )
        if relion_walltime is not None:
            relion_walltime_path = relion_slurm_walltime_path
            notes.append("using RELION slurm_walltime.json")
        else:
            notes.append("missing RELION relion_walltime.txt")
    else:
        relion_walltime = _parse_duration_seconds(relion_walltime_path.read_text())

    recovar_throughput = _throughput(n_images, recovar_walltime)
    relion_throughput = _throughput(n_images, relion_walltime)
    speed_ratio = None
    recovar_relion_wall_ratio = None
    if recovar_walltime and relion_walltime and recovar_walltime > 0.0:
        speed_ratio = float(relion_walltime / recovar_walltime)
    if recovar_walltime and relion_walltime and relion_walltime > 0.0:
        recovar_relion_wall_ratio = float(recovar_walltime / relion_walltime)
    timing = {
        "recovar_ledger_path": str(ledger_path) if ledger_path is not None and ledger_path.exists() else None,
        "relion_walltime_path": str(relion_walltime_path)
        if relion_walltime_path is not None and relion_walltime_path.exists()
        else None,
        "n_images": int(n_images) if n_images is not None and float(n_images).is_integer() else n_images,
        "recovar_walltime_s": recovar_walltime,
        "relion_walltime_s": relion_walltime,
        "speed_ratio_relion_over_recovar": speed_ratio,
        "recovar_relion_wall_ratio": recovar_relion_wall_ratio,
        "recovar_throughput_images_per_s": recovar_throughput,
        "relion_throughput_images_per_s": relion_throughput,
        "notes": notes,
        "run_log_telemetry": {
            "path": log_telemetry.get("path"),
            "status": log_telemetry.get("status"),
            "missing_artifacts": log_telemetry.get("missing_artifacts") or [],
            "notes": log_telemetry.get("notes") or [],
            "counts": log_telemetry.get("counts") or {},
        },
        "recovar_iteration_rows": log_telemetry.get("iteration_rows") or [],
        "batch_sizing_events": log_telemetry.get("batch_sizing_events") or [],
        "sparse_pass2_events": log_telemetry.get("sparse_pass2_events") or [],
        "sparse_group_timing_events": log_telemetry.get("sparse_group_timing_events") or [],
        "compact_pair_planner_events": log_telemetry.get("compact_pair_planner_events") or [],
        "local_search_events": log_telemetry.get("local_search_events") or [],
    }
    if log_telemetry.get("status") != "ok":
        notes.extend(str(note) for note in log_telemetry.get("notes") or [])
    if ledger is not None:
        for output_key, ledger_key in (
            ("wall_times_trajectory_s", "wall_times_trajectory"),
            ("current_sizes", "current_sizes"),
            ("pixel_resolutions", "pixel_resolutions"),
            ("ave_pmax_trajectory", "ave_Pmax_trajectory"),
        ):
            values = _json_list(ledger.get(ledger_key))
            if values is not None:
                timing[output_key] = values
        timing_rows = ledger.get("timing_rows")
        if isinstance(timing_rows, list):
            timing["iteration_timing_rows"] = timing_rows
        timing_summary = ledger.get("timing_summary")
        if isinstance(timing_summary, dict):
            timing["iteration_timing_summary"] = timing_summary
        global_profile_rows = ledger.get("global_profile_rows")
        if isinstance(global_profile_rows, list):
            timing["global_profile_rows"] = global_profile_rows
            global_profile_summary = _global_profile_aggregate(global_profile_rows)
            if global_profile_summary is not None:
                timing.update(global_profile_summary)
    iteration_rows = timing["recovar_iteration_rows"]
    if iteration_rows:
        if "wall_times_trajectory_s" not in timing:
            timing["wall_times_trajectory_s"] = [
                row.get("wall_time_s") for row in iteration_rows if row.get("wall_time_s") is not None
            ]
        if "current_sizes" not in timing:
            timing["current_sizes"] = [
                row.get("current_size") for row in iteration_rows if row.get("current_size") is not None
            ]
        if "pixel_resolutions" not in timing:
            timing["pixel_resolutions"] = [
                row.get("pixel_res") for row in iteration_rows if row.get("pixel_res") is not None
            ]
        if "ave_pmax_trajectory" not in timing:
            timing["ave_pmax_trajectory"] = [
                row.get("ave_pmax") for row in iteration_rows if row.get("ave_pmax") is not None
            ]
        wall_times = [float(row["wall_time_s"]) for row in iteration_rows if row.get("wall_time_s") is not None]
        timing["recovar_iteration_walltime_total_s"] = float(sum(wall_times)) if wall_times else None
        timing["recovar_iteration_walltime_mean_s"] = float(np.mean(wall_times)) if wall_times else None
    sparse_summary = _sparse_pass2_aggregate(timing["sparse_pass2_events"], timing["recovar_iteration_rows"])
    if sparse_summary is not None:
        timing.update(sparse_summary)
    sparse_group_summary = _sparse_group_timing_aggregate(timing["sparse_group_timing_events"])
    if sparse_group_summary is not None:
        timing.update(sparse_group_summary)
    for prefix, run_dir in (("recovar", recovar_dir), ("relion", relion_dir)):
        monitor = _read_gpu_monitor(run_dir / "gpu_monitor.csv" if run_dir is not None else None)
        if monitor is None:
            continue
        timing[f"{prefix}_gpu_monitor"] = monitor
        timing[f"{prefix}_peak_gpu_memory_mib"] = monitor.get("peak_memory_mib")
        timing[f"{prefix}_peak_gpu_memory_gib"] = monitor.get("peak_memory_gib")
    return timing


def _json_list(value: Any) -> list[Any] | None:
    if not isinstance(value, list):
        return None
    return [_jsonable(item) for item in value]


def _first_number(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in mapping:
            continue
        try:
            value = float(mapping[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _throughput(n_images: float | None, walltime_s: float | None) -> float | None:
    if n_images is None or walltime_s is None or walltime_s <= 0.0:
        return None
    return float(n_images / walltime_s)


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> Any | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def _latest_npz_key(data: np.lib.npyio.NpzFile, prefix: str) -> str | None:
    best: tuple[int, str] | None = None
    pattern = re.compile(re.escape(prefix) + r"_(\d+)$")
    for key in data.files:
        match = pattern.match(key)
        if match is None:
            continue
        candidate = (int(match.group(1)), key)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1] if best is not None else None


def _recovar_convergence_metadata(recovar_dir: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": None,
        "status": "missing",
        "n_iterations": None,
        "convergence_iteration": None,
        "convergence_has_converged": None,
    }
    if recovar_dir is None:
        return out
    path = recovar_dir / "refinement_results.npz"
    out["path"] = str(path)
    if not path.exists():
        return out
    try:
        with np.load(path, allow_pickle=True) as data:
            out["status"] = "ok"
            for key in ("n_iterations", "convergence_iteration", "convergence_has_converged"):
                out[key] = _npz_scalar(data, key)
    except Exception as exc:
        out["status"] = "error"
        out["error"] = str(exc)
    return out


def _k1_final_all_data_metadata(recovar_dir: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": None,
        "status": "missing",
        "final_all_data_ran": None,
        "fsc_final_all_data_present": False,
        "fsc_final_all_data_finite": False,
        "failures": [],
    }
    if recovar_dir is None:
        out["failures"].append("missing K=1 RECOVAR directory")
        return out

    path = recovar_dir / "refinement_results.npz"
    out["path"] = str(path)
    if not path.exists():
        out["failures"].append("missing refinement_results.npz")
        return out

    try:
        with np.load(path, allow_pickle=True) as data:
            out["status"] = "ok"
            if "final_all_data_ran" not in data.files:
                out["failures"].append("missing final_all_data_ran")
            else:
                final_all_data_ran = bool(np.asarray(data["final_all_data_ran"]).reshape(()))
                out["final_all_data_ran"] = final_all_data_ran
                if not final_all_data_ran:
                    out["failures"].append("final_all_data_ran is false")

            if "fsc_final_all_data" not in data.files:
                out["failures"].append("missing fsc_final_all_data")
            else:
                final_fsc = np.asarray(data["fsc_final_all_data"], dtype=np.float64).reshape(-1)
                out["fsc_final_all_data_present"] = bool(final_fsc.size)
                out["fsc_final_all_data_finite"] = bool(final_fsc.size and np.all(np.isfinite(final_fsc)))
                if not final_fsc.size:
                    out["failures"].append("empty fsc_final_all_data")
                elif not np.all(np.isfinite(final_fsc)):
                    out["failures"].append("non-finite fsc_final_all_data")
    except Exception as exc:
        out["status"] = "error"
        out["failures"].append(f"failed to read refinement_results.npz: {exc}")

    return out


def _stack_index_from_image_name(name: Any) -> int:
    match = re.match(r"(\d+)@", str(name))
    return int(match.group(1)) - 1 if match else -1


def _star_particles(path: Path):
    import starfile

    data = starfile.read(str(path))
    return data["particles"] if isinstance(data, dict) and "particles" in data else data


def _ordered_relion_particles(relion_data_path: Path, dataset_star_path: Path):
    relion_df = _star_particles(relion_data_path)
    dataset_df = _star_particles(dataset_star_path)
    relion_names = [str(name) for name in relion_df["rlnImageName"]]
    relion_by_name = {name: row for row, name in enumerate(relion_names)}
    rows = []
    for name in [str(value) for value in dataset_df["rlnImageName"]]:
        row = relion_by_name.get(name)
        if row is None:
            break
        rows.append(row)
    else:
        return relion_df.iloc[np.asarray(rows, dtype=np.int64)].reset_index(drop=True)

    relion_by_stack = {_stack_index_from_image_name(name): row for row, name in enumerate(relion_names)}
    rows = []
    for name in list(dataset_df["rlnImageName"]):
        stack_idx = _stack_index_from_image_name(name)
        row = relion_by_stack.get(stack_idx)
        if row is None:
            return None
        rows.append(row)
    return relion_df.iloc[np.asarray(rows, dtype=np.int64)].reset_index(drop=True)


def _summary_stats(diff: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(diff, dtype=np.float64)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return {"mean": None, "median": None, "max": None, "p95": None}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _corr_or_none(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return None
    a = lhs[valid] - float(np.mean(lhs[valid]))
    b = rhs[valid] - float(np.mean(rhs[valid]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 and math.isfinite(denom) else None


def _rotation_angle_errors_deg(recovar_eulers: np.ndarray, relion_eulers: np.ndarray) -> np.ndarray:
    from recovar import utils

    rec = np.asarray(recovar_eulers, dtype=np.float64)
    rel = np.asarray(relion_eulers, dtype=np.float64)
    rec_r = np.asarray(utils.R_from_relion(rec, degrees=True), dtype=np.float64)
    rel_r = np.asarray(utils.R_from_relion(rel, degrees=True), dtype=np.float64)
    delta = np.einsum("...ji,...jk->...ik", rel_r, rec_r)
    trace = np.trace(delta, axis1=-2, axis2=-1)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _class_assignments_by_image(data: np.lib.npyio.NpzFile, n_images: int) -> np.ndarray | None:
    if "class_assignments_half0" not in data.files:
        return None
    half1 = np.asarray(data["half1_indices"], dtype=np.int64) if "half1_indices" in data.files else None
    half2 = np.asarray(data["half2_indices"], dtype=np.int64) if "half2_indices" in data.files else None
    out = np.full(int(n_images), -1, dtype=np.int32)
    arr1 = np.asarray(data["class_assignments_half0"], dtype=np.int32)
    if half1 is not None:
        out[half1] = arr1[: half1.size]
    if "class_assignments_half1" in data.files and half2 is not None and half2.size:
        arr2 = np.asarray(data["class_assignments_half1"], dtype=np.int32)
        out[half2] = arr2[: half2.size]
    return out


def _preferred_npz_key(data: np.lib.npyio.NpzFile, preferred: str, fallback: str) -> str | None:
    if preferred in data.files:
        return preferred
    if fallback in data.files:
        return fallback
    return None


def _particle_metrics(
    *,
    recovar_dir: Path | None,
    relion_dir: Path | None,
    fixture_dir: Path | None,
    relion_iter: int,
    class_permutation: list[int] | None = None,
) -> dict[str, Any] | None:
    if recovar_dir is None or relion_dir is None or fixture_dir is None:
        return None
    npz_path = recovar_dir / "refinement_results.npz"
    dataset_star_path = fixture_dir / "particles.star"
    relion_data_path = _relion_data_path(relion_dir, relion_iter)
    relion_final_data_path = _relion_final_data_path(relion_dir)
    if not npz_path.exists() or relion_data_path is None or not dataset_star_path.exists():
        return None

    out: dict[str, Any] = {
        "recovar_npz": str(npz_path),
        "relion_data_star": str(relion_data_path),
        "relion_final_data_star": str(relion_final_data_path) if relion_final_data_path is not None else None,
        "relion_iteration": int(relion_iter),
        "notes": [],
    }
    try:
        relion_df = _ordered_relion_particles(relion_data_path, dataset_star_path)
    except Exception as exc:
        out["notes"].append(f"failed to read/order RELION data STAR: {exc}")
        return out
    if relion_df is None:
        out["notes"].append("failed to align RELION data STAR to fixture particle order")
        return out

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            n_images = int(_npz_scalar(data, "n_images") or len(relion_df))
            if len(relion_df) != n_images:
                out["notes"].append(f"RELION particle count {len(relion_df)} != RECOVAR n_images {n_images}")

            final_relion_df = None
            final_relion_load_attempted = False
            final_relion_note_emitted = False

            def relion_particles_for_pose_key(npz_key: str) -> tuple[Any, Path | None]:
                nonlocal final_relion_df, final_relion_load_attempted, final_relion_note_emitted
                if npz_key.endswith("_final_all_data_by_image"):
                    if relion_final_data_path is None:
                        if not final_relion_note_emitted:
                            out["notes"].append(
                                "missing RELION run_data.star; final all-data pose metrics use iteration data"
                            )
                            final_relion_note_emitted = True
                        return relion_df, relion_data_path
                    if not final_relion_load_attempted:
                        final_relion_load_attempted = True
                        try:
                            final_relion_df = _ordered_relion_particles(relion_final_data_path, dataset_star_path)
                        except Exception as exc:
                            out["notes"].append(f"failed to read/order RELION final data STAR: {exc}")
                            final_relion_df = None
                    if final_relion_df is not None:
                        return final_relion_df, relion_final_data_path
                    if not final_relion_note_emitted:
                        out["notes"].append("failed to align RELION final data STAR; final pose metrics use iteration data")
                        final_relion_note_emitted = True
                return relion_df, relion_data_path

            if "fsc_final_all_data" in data.files:
                final_fsc = np.asarray(data["fsc_final_all_data"], dtype=np.float64).reshape(-1)
                out["final_all_data_fsc"] = {
                    "npz_key": "fsc_final_all_data",
                    "n_shells": int(final_fsc.size),
                    "fsc_auc": normalized_fsc_auc(final_fsc, np.arange(final_fsc.size, dtype=np.float64)),
                    "mean_fsc_1_16": mean_shell_value(final_fsc, 1, 16),
                    "shell_05": first_shell_below(final_fsc, 0.5),
                    "shell_0143": first_shell_below(final_fsc, 0.143),
                }
            else:
                out["notes"].append("missing RECOVAR final all-data FSC")

            if "final_all_data_sampling_perturbation" in data.files:
                out["final_all_data_sampling"] = {
                    "perturbation": float(np.asarray(data["final_all_data_sampling_perturbation"])),
                    "applied": bool(np.asarray(data["final_all_data_sampling_perturbation_applied"]))
                    if "final_all_data_sampling_perturbation_applied" in data.files
                    else None,
                    "relion_iteration": int(np.asarray(data["final_all_data_sampling_relion_iteration"]))
                    if "final_all_data_sampling_relion_iteration" in data.files
                    else None,
                    "sampling_star": str(np.asarray(data["final_all_data_sampling_star"]).item())
                    if "final_all_data_sampling_star" in data.files
                    else None,
                    "sampling_star_source": str(np.asarray(data["final_all_data_sampling_star_source"]).item())
                    if "final_all_data_sampling_star_source" in data.files
                    else None,
                    "offset_range_px": float(np.asarray(data["final_all_data_sampling_offset_range"]))
                    if "final_all_data_sampling_offset_range" in data.files
                    else None,
                    "offset_step_px": float(np.asarray(data["final_all_data_sampling_offset_step"]))
                    if "final_all_data_sampling_offset_step" in data.files
                    else None,
                }
            if "final_all_data_grid_correct" in data.files or "final_all_data_gridding_correct" in data.files:
                out["final_all_data_reconstruction"] = {
                    "grid_correct": bool(np.asarray(data["final_all_data_grid_correct"]))
                    if "final_all_data_grid_correct" in data.files
                    else None,
                    "gridding_correct": str(np.asarray(data["final_all_data_gridding_correct"]).item())
                    if "final_all_data_gridding_correct" in data.files
                    else None,
                }

            pmax_key = "pmax_final_all_data_by_image" if "pmax_final_all_data_by_image" in data.files else None
            pmax_relion_df = None
            pmax_relion_data_path = None
            recovar_pmax = None
            if pmax_key is not None:
                pmax_relion_df, pmax_relion_data_path = relion_particles_for_pose_key(pmax_key)
                recovar_pmax = np.asarray(data[pmax_key], dtype=np.float64).reshape(-1)[:n_images]
            else:
                pmax_key = _latest_npz_key(data, "pmax_per_image_by_image_iter")
                if pmax_key is not None:
                    pmax_relion_df = relion_df
                    pmax_relion_data_path = relion_data_path
                    recovar_pmax = np.asarray(data[pmax_key], dtype=np.float64).reshape(-1)[:n_images]
                else:
                    pmax_key = _latest_npz_key(data, "pmax_per_image_iter")
                    if pmax_key is not None:
                        pmax_relion_df = relion_df
                        pmax_relion_data_path = relion_data_path
                        recovar_half_order = np.asarray(data[pmax_key], dtype=np.float64)
                        recovar_pmax = np.full(n_images, np.nan, dtype=np.float64)
                        half1 = np.asarray(data["half1_indices"], dtype=np.int64) if "half1_indices" in data.files else None
                        half2 = np.asarray(data["half2_indices"], dtype=np.int64) if "half2_indices" in data.files else None
                        cursor = 0
                        if half1 is not None:
                            recovar_pmax[half1] = recovar_half_order[cursor : cursor + half1.size]
                            cursor += half1.size
                        if half2 is not None and half2.size:
                            recovar_pmax[half2] = recovar_half_order[cursor : cursor + half2.size]
            if (
                pmax_key is not None
                and recovar_pmax is not None
                and pmax_relion_df is not None
                and "rlnMaxValueProbDistribution" in pmax_relion_df.columns
            ):
                relion_pmax = np.asarray(pmax_relion_df["rlnMaxValueProbDistribution"], dtype=np.float64)[:n_images]
                valid = np.isfinite(recovar_pmax) & np.isfinite(relion_pmax)
                diff = recovar_pmax[valid] - relion_pmax[valid]
                out["pmax"] = {
                    "npz_key": pmax_key,
                    "relion_data_star": str(pmax_relion_data_path) if pmax_relion_data_path is not None else None,
                    "matched_count": int(valid.sum()),
                    "recovar_mean": float(np.mean(recovar_pmax[valid])) if valid.any() else None,
                    "relion_mean": float(np.mean(relion_pmax[valid])) if valid.any() else None,
                    "gap_mean_recovar_minus_relion": float(np.mean(diff)) if diff.size else None,
                    "abs_diff": _summary_stats(np.abs(diff)),
                    "corr": _corr_or_none(recovar_pmax, relion_pmax),
                }
            else:
                out["notes"].append("missing RECOVAR pmax history or RELION Pmax column")

            euler_key = _preferred_npz_key(
                data,
                "best_rotation_eulers_final_all_data_by_image",
                "best_rotation_eulers_final_by_image",
            )
            pose_relion_df = None
            pose_relion_data_path = None
            if euler_key is not None:
                pose_relion_df, pose_relion_data_path = relion_particles_for_pose_key(euler_key)
            if euler_key is not None and all(
                col in pose_relion_df.columns for col in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
            ):
                rec_eulers = np.asarray(data[euler_key], dtype=np.float64)[:n_images]
                rel_eulers = np.stack(
                    [
                        np.asarray(pose_relion_df[col], dtype=np.float64)[:n_images]
                        for col in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
                    ],
                    axis=1,
                )
                valid = np.isfinite(rec_eulers).all(axis=1) & np.isfinite(rel_eulers).all(axis=1)
                angle_errors = _rotation_angle_errors_deg(rec_eulers[valid], rel_eulers[valid])
                out["pose_rotation_deg"] = {
                    "npz_key": euler_key,
                    "relion_data_star": str(pose_relion_data_path) if pose_relion_data_path is not None else None,
                    "matched_count": int(valid.sum()),
                    "angle_error": _summary_stats(angle_errors),
                    "within_1deg_fraction": float(np.mean(angle_errors <= 1.0)) if angle_errors.size else None,
                    "within_5deg_fraction": float(np.mean(angle_errors <= 5.0)) if angle_errors.size else None,
                }
            else:
                out["notes"].append("missing RECOVAR final eulers or RELION angle columns")

            trans_key = _preferred_npz_key(
                data,
                "best_translations_final_all_data_by_image",
                "best_translations_final_by_image",
            )
            trans_relion_df = None
            trans_relion_data_path = None
            if trans_key is not None:
                trans_relion_df, trans_relion_data_path = relion_particles_for_pose_key(trans_key)
            if trans_key is not None and "voxel_size" in data.files and all(
                col in trans_relion_df.columns for col in ("rlnOriginXAngst", "rlnOriginYAngst")
            ):
                voxel_size = float(_npz_scalar(data, "voxel_size"))
                rec_trans = np.asarray(data[trans_key], dtype=np.float64)[:n_images]
                rel_trans = np.stack(
                    [
                        np.asarray(trans_relion_df["rlnOriginXAngst"], dtype=np.float64)[:n_images] / voxel_size,
                        np.asarray(trans_relion_df["rlnOriginYAngst"], dtype=np.float64)[:n_images] / voxel_size,
                    ],
                    axis=1,
                )
                valid = np.isfinite(rec_trans).all(axis=1) & np.isfinite(rel_trans).all(axis=1)
                trans_error = np.linalg.norm(rec_trans[valid] - rel_trans[valid], axis=1)
                out["translation_px"] = {
                    "npz_key": trans_key,
                    "relion_data_star": str(trans_relion_data_path) if trans_relion_data_path is not None else None,
                    "matched_count": int(valid.sum()),
                    "l2_error": _summary_stats(trans_error),
                    "within_0_5px_fraction": float(np.mean(trans_error <= 0.5)) if trans_error.size else None,
                    "within_1px_fraction": float(np.mean(trans_error <= 1.0)) if trans_error.size else None,
                }
            else:
                out["notes"].append("missing RECOVAR final translations or RELION origin columns")

            if "rlnClassNumber" in relion_df.columns:
                rec_class = _class_assignments_by_image(data, n_images)
                if rec_class is not None:
                    valid = rec_class >= 0
                    rel_class_zero_based = np.asarray(relion_df["rlnClassNumber"], dtype=np.int32)[:n_images] - 1
                    raw_agreement = float(np.mean(rec_class[valid] == rel_class_zero_based[valid])) if valid.any() else None
                    out["class_assignment"] = {
                        "matched_count": int(valid.sum()),
                        "raw_agreement": raw_agreement,
                    }
                    if class_permutation is not None:
                        perm = np.asarray(class_permutation, dtype=np.int32)
                        mapped = np.full_like(rec_class, -1)
                        in_range = valid & (rec_class < perm.size)
                        mapped[in_range] = perm[rec_class[in_range]]
                        out["class_assignment"]["permutation_lhs_to_relion"] = [int(v) for v in perm.tolist()]
                        out["class_assignment"]["permuted_agreement"] = (
                            float(np.mean(mapped[in_range] == rel_class_zero_based[in_range]))
                            if in_range.any()
                            else None
                        )
    except Exception as exc:
        out["notes"].append(f"failed to compute per-particle metrics: {exc}")
    return out


def _check_numeric_default(
    *,
    data: np.lib.npyio.NpzFile,
    key: str,
    expected: float,
    values: dict[str, Any],
    failures: list[str],
    missing_fields: list[str],
    atol: float = 1.0e-6,
    required: bool = True,
) -> None:
    value = _npz_scalar(data, key)
    if value is None:
        missing_fields.append(key)
        if required:
            failures.append(f"missing {key}")
        return
    values[key] = value
    try:
        observed = float(value)
    except (TypeError, ValueError):
        failures.append(f"{key}={value!r}, expected {expected:g}")
        return
    if not math.isfinite(observed) or abs(observed - float(expected)) > float(atol):
        failures.append(f"{key}={observed:g}, expected {expected:g}")


def _check_bool_default(
    *,
    data: np.lib.npyio.NpzFile,
    key: str,
    expected: bool,
    values: dict[str, Any],
    failures: list[str],
    missing_fields: list[str],
    required: bool = True,
) -> None:
    value = _npz_scalar(data, key)
    if value is None:
        missing_fields.append(key)
        if required:
            failures.append(f"missing {key}")
        return
    observed = bool(value)
    values[key] = observed
    if observed is not bool(expected):
        failures.append(f"{key}={observed}, expected {bool(expected)}")


def _check_log_contains(
    *,
    recovar_dir: Path,
    log_name: str,
    pattern: str,
    label: str,
    values: dict[str, Any],
    failures: list[str],
    missing_fields: list[str],
) -> None:
    path = recovar_dir / log_name
    values[label] = False
    if not path.exists():
        missing_fields.append(log_name)
        failures.append(f"missing {log_name}")
        return
    try:
        found = pattern in path.read_text(errors="replace")
    except Exception as exc:
        failures.append(f"failed to read {log_name}: {exc}")
        return
    values[label] = bool(found)
    if not found:
        failures.append(f"{label}=False, expected log line containing {pattern!r}")


def _check_log_line_match(
    *,
    recovar_dir: Path,
    log_name: str,
    matcher: Callable[[str], bool],
    description: str,
    label: str,
    values: dict[str, Any],
    failures: list[str],
    missing_fields: list[str],
) -> None:
    path = recovar_dir / log_name
    values[label] = False
    if not path.exists():
        missing_fields.append(log_name)
        failures.append(f"missing {log_name}")
        return
    try:
        found = any(matcher(line) for line in path.read_text(errors="replace").splitlines())
    except Exception as exc:
        failures.append(f"failed to read {log_name}: {exc}")
        return
    values[label] = bool(found)
    if not found:
        failures.append(f"{label}=False, expected log line matching {description}")


def _k1_adaptive_fine_pass_sparse_route_line(line: str) -> bool:
    if "K=1" not in line or "run_dense_k_class_em_adaptive" not in line:
        return False
    return (
        "through sparse run_dense_k_class_em_adaptive" in line
        or "pass2_backend=sparse" in line
    )


def _check_k1_refinement_runtime_defaults(recovar_dir: Path | None) -> dict[str, Any] | None:
    if recovar_dir is None:
        return None

    path = recovar_dir / "refinement_results.npz"
    out: dict[str, Any] = {
        "path": str(path),
        "status": "missing",
        "values": {},
        "missing_fields": [],
        "failures": [],
    }
    if not path.exists():
        out["failures"].append("missing refinement_results.npz")
        return out

    try:
        with np.load(path, allow_pickle=True) as data:
            values = out["values"]
            failures = out["failures"]
            missing_fields = out["missing_fields"]

            _check_numeric_default(
                data=data,
                key="n_classes",
                expected=1,
                values=values,
                failures=failures,
                missing_fields=missing_fields,
                required=False,
            )
            for key, expected in (
                ("tau2_fudge", 1.0),
                ("healpix_order", 3.0),
                ("coarse_healpix_order", 3.0),
                ("finest_healpix_order", 4.0),
                ("adaptive_oversampling", 1.0),
                ("particle_diameter_ang", 200.0),
            ):
                _check_numeric_default(
                    data=data,
                    key=key,
                    expected=expected,
                    values=values,
                    failures=failures,
                    missing_fields=missing_fields,
                )
            for key in ("firstiter_cc", "apply_initial_lowpass"):
                _check_bool_default(
                    data=data,
                    key=key,
                    expected=True,
                    values=values,
                    failures=failures,
                    missing_fields=missing_fields,
                    required=False,
                )

            n_images_value = _npz_scalar(data, "n_images")
            if n_images_value is None:
                missing_fields.append("n_images")
                failures.append("missing n_images")
                n_images = None
            else:
                values["n_images"] = n_images_value
                n_images = int(n_images_value)
            if "half1_indices" not in data.files:
                missing_fields.append("half1_indices")
                failures.append("missing half1_indices")
                half1_count = None
            else:
                half1_count = int(np.asarray(data["half1_indices"]).size)
                values["half1_count"] = half1_count
            if "half2_indices" not in data.files:
                missing_fields.append("half2_indices")
                failures.append("missing half2_indices")
                half2_count = None
            else:
                half2_count = int(np.asarray(data["half2_indices"]).size)
                values["half2_count"] = half2_count
            if n_images is not None and half1_count is not None and half2_count is not None:
                if half1_count + half2_count != n_images:
                    failures.append(
                        f"half split total={half1_count + half2_count}, expected {n_images} particles"
                    )
                if half1_count <= 0 or half2_count <= 0:
                    failures.append(
                        f"K=1 AutoRefine half splits must both be nonempty, got {half1_count}/{half2_count}"
                    )
            _check_log_line_match(
                recovar_dir=recovar_dir,
                log_name="run_full_refinement.log",
                matcher=_k1_adaptive_fine_pass_sparse_route_line,
                description=(
                    "K=1 run_dense_k_class_em_adaptive sparse route "
                    "(legacy 'through sparse ...' or current pass2_backend=sparse)"
                ),
                label="k1_adaptive_fine_pass_route",
                values=values,
                failures=failures,
                missing_fields=missing_fields,
            )
    except Exception as exc:
        out["failures"].append(f"failed to read refinement_results.npz: {exc}")

    out["status"] = "ok" if not out["failures"] else "failed"
    return out


def _check_k4_refinement_runtime_defaults(recovar_dir: Path | None) -> dict[str, Any] | None:
    if recovar_dir is None:
        return None

    path = recovar_dir / "refinement_results.npz"
    out: dict[str, Any] = {
        "path": str(path),
        "status": "missing",
        "values": {},
        "missing_fields": [],
        "failures": [],
    }
    if not path.exists():
        out["failures"].append("missing refinement_results.npz")
        return out

    try:
        with np.load(path, allow_pickle=True) as data:
            values = out["values"]
            failures = out["failures"]
            missing_fields = out["missing_fields"]

            # n_classes was added as explicit metadata after older completion
            # runs existed. Treat a missing field as a warning; the four class
            # maps still make the active run auditable, and future outputs
            # should carry this scalar.
            _check_numeric_default(
                data=data,
                key="n_classes",
                expected=4,
                values=values,
                failures=failures,
                missing_fields=missing_fields,
                required=False,
            )
            for key, expected in (
                ("tau2_fudge", 4.0),
                ("healpix_order", 1.0),
                ("coarse_healpix_order", 1.0),
                ("finest_healpix_order", 2.0),
                ("adaptive_oversampling", 1.0),
                ("particle_diameter_ang", 380.0),
            ):
                _check_numeric_default(
                    data=data,
                    key=key,
                    expected=expected,
                    values=values,
                    failures=failures,
                    missing_fields=missing_fields,
                )

            n_images_value = _npz_scalar(data, "n_images")
            if n_images_value is None:
                missing_fields.append("n_images")
                failures.append("missing n_images")
                n_images = None
            else:
                values["n_images"] = n_images_value
                n_images = int(n_images_value)
            if "half1_indices" not in data.files:
                missing_fields.append("half1_indices")
                failures.append("missing half1_indices")
                half1_count = None
            else:
                half1_count = int(np.asarray(data["half1_indices"]).size)
                values["half1_count"] = half1_count
            if "half2_indices" not in data.files:
                missing_fields.append("half2_indices")
                failures.append("missing half2_indices")
                half2_count = None
            else:
                half2_count = int(np.asarray(data["half2_indices"]).size)
                values["half2_count"] = half2_count
            if n_images is not None and half1_count is not None and half1_count != n_images:
                failures.append(f"half1_count={half1_count}, expected all {n_images} K=4 particles")
            if half2_count is not None and half2_count != 0:
                failures.append(f"half2_count={half2_count}, expected 0 for RELION Class3D all-data split")
    except Exception as exc:
        out["failures"].append(f"failed to read refinement_results.npz: {exc}")

    out["status"] = "ok" if not out["failures"] else "failed"
    return out


def _k1_relion_half_path(relion_dir: Path | None, half: int) -> Path | None:
    latest = _latest_relion_path(relion_dir, f"run_it*_half{half}_class001.mrc")
    if latest is not None:
        return latest
    return _existing_path(
        relion_dir,
        [
            f"run_it015_half{half}_class001.mrc",
            f"run_it008_half{half}_class001.mrc",
            f"run_it003_half{half}_class001.mrc",
        ],
    )


def _k1_relion_final_map_path(relion_dir: Path | None) -> Path | None:
    final_map = _existing_path(relion_dir, ["run_class001.mrc"])
    if final_map is not None:
        return final_map
    latest = _latest_relion_iteration_class_path(relion_dir, 1)
    if latest is not None:
        return latest
    return _existing_path(relion_dir, ["run_it015_class001.mrc"])


def _completion_status_from_metrics(
    metrics: dict[str, Any],
    *,
    recovar_metric_keys: tuple[str, ...],
    notes: list[str],
) -> str:
    """Return section status without treating RELION-only baseline rows as done."""

    if any(key in metrics for key in recovar_metric_keys):
        return "ok"
    if any(str(note).startswith("failed ") for note in notes):
        return "failed"
    return "pending"


def summarize_k1(recovar_dir: Path | None, relion_dir: Path | None, fixture_dir: Path | None) -> dict[str, Any]:
    notes: list[str] = []
    if recovar_dir is None or fixture_dir is None:
        return {"status": "skipped", "notes": ["missing one or more K=1 RECOVAR/fixture input directories"]}
    relion_selected = relion_dir is not None
    sign_ambiguity = _k1_sign_ambiguity_metadata(fixture_dir)
    if sign_ambiguity.get("allow_global_sign"):
        notes.append(
            "K=1 no-CTF fixture: global sign is ambiguous; required GT FSC-AUC gate uses "
            "sign-invariant FSC-AUC while signed metrics are still reported"
        )

    runtime_defaults = _check_k1_refinement_runtime_defaults(recovar_dir)
    if runtime_defaults is not None and runtime_defaults.get("failures"):
        notes.extend(f"K=1 runtime default guard: {msg}" for msg in runtime_defaults["failures"])
    recovar_convergence = _recovar_convergence_metadata(recovar_dir)
    final_all_data = _k1_final_all_data_metadata(recovar_dir)

    rec_merged_path = _existing_path(recovar_dir, ["final_merged.mrc", "recovar_final_merged.mrc"])
    rec_unfiltered_paths = [
        _existing_path(recovar_dir, [f"final_half{half}_unfil.mrc"])
        for half in (1, 2)
    ]
    rel_unfiltered_paths = [
        _existing_path(relion_dir, [f"run_half{half}_class001_unfil.mrc"])
        if relion_selected
        else None
        for half in (1, 2)
    ]
    if any(path is not None for path in rec_unfiltered_paths + rel_unfiltered_paths):
        rec_h1_path, rec_h2_path = rec_unfiltered_paths
        rel_h1_path, rel_h2_path = rel_unfiltered_paths
        if not all(path is not None for path in rec_unfiltered_paths + rel_unfiltered_paths):
            notes.append(
                "K=1 final unfiltered half products are incomplete; regularized and unfiltered halves "
                "will not be cross-compared"
            )
    else:
        rec_h1_path = _existing_path(recovar_dir, ["final_half1.mrc", "recovar_final_half1.mrc"])
        rec_h2_path = _existing_path(recovar_dir, ["final_half2.mrc", "recovar_final_half2.mrc"])
        rel_h1_path = _k1_relion_half_path(relion_dir, 1) if relion_selected else None
        rel_h2_path = _k1_relion_half_path(relion_dir, 2) if relion_selected else None
    rel_final_map_path = _k1_relion_final_map_path(relion_dir) if relion_selected else None
    gt_path = _existing_path(fixture_dir, ["reference_gt.mrc", "reference_gt_class001.mrc", "gt.mrc"])

    rec = _load_optional(rec_merged_path, _load_recovar_volume, "K=1 RECOVAR final_merged.mrc", notes)
    rec_h1 = (
        _load_optional(rec_h1_path, _load_recovar_volume, "K=1 RECOVAR final_half1.mrc", notes)
        if relion_selected
        else None
    )
    rec_h2 = (
        _load_optional(rec_h2_path, _load_recovar_volume, "K=1 RECOVAR final_half2.mrc", notes)
        if relion_selected
        else None
    )
    rel_h1 = (
        _load_optional(rel_h1_path, _load_relion_volume, "K=1 RELION final half1 map", notes)
        if relion_selected
        else None
    )
    rel_h2 = (
        _load_optional(rel_h2_path, _load_relion_volume, "K=1 RELION final half2 map", notes)
        if relion_selected
        else None
    )
    rel_final_map = (
        _load_optional(rel_final_map_path, _load_relion_volume, "K=1 RELION final merged map", notes)
        if relion_selected
        else None
    )
    gt = _load_optional(gt_path, _load_recovar_volume, "K=1 GT reference", notes)

    metrics: dict[str, Any] = {}
    rel_halfavg = None
    if rel_h1 is not None and rel_h2 is not None and rel_h1.shape == rel_h2.shape:
        rel_halfavg = (rel_h1 + rel_h2) / 2.0
    elif rel_h1 is not None and rel_h2 is not None:
        notes.append(f"K=1 RELION half shapes differ: {rel_h1.shape} vs {rel_h2.shape}")

    if rec_h1 is not None and rel_h1 is not None:
        metrics["recovar_half1_vs_relion_half1"] = map_metrics(rec_h1, rel_h1)
    if rec_h2 is not None and rel_h2 is not None:
        metrics["recovar_half2_vs_relion_half2"] = map_metrics(rec_h2, rel_h2)
    if rec is not None and rel_final_map is not None:
        metrics["recovar_merged_vs_relion_merged"] = map_metrics(rec, rel_final_map)
        metrics["recovar_merged_vs_relion_final_map"] = metrics["recovar_merged_vs_relion_merged"]
    elif rec is not None and rel_halfavg is not None:
        metrics["recovar_merged_vs_relion_merged"] = map_metrics(rec, rel_halfavg)
        notes.append("K=1 RELION final merged map missing; recovar_merged_vs_relion_merged uses half-map average")
    if rec is not None and rel_halfavg is not None:
        metrics["recovar_merged_vs_relion_halfavg"] = map_metrics(rec, rel_halfavg)
    if rec is not None and gt is not None:
        metrics["recovar_merged_vs_gt"] = map_metrics(rec, gt)
    if rel_final_map is not None and gt is not None:
        metrics["relion_merged_vs_gt"] = map_metrics(rel_final_map, gt)
        metrics["relion_final_map_vs_gt"] = metrics["relion_merged_vs_gt"]
    elif rel_halfavg is not None and gt is not None:
        metrics["relion_merged_vs_gt"] = map_metrics(rel_halfavg, gt)
        notes.append("K=1 RELION final merged map missing; relion_merged_vs_gt uses half-map average")
    if rel_halfavg is not None and gt is not None:
        metrics["relion_halfavg_vs_gt"] = map_metrics(rel_halfavg, gt)
    if (
        relion_selected
        and recovar_convergence.get("convergence_has_converged") is False
        and rel_h1_path is not None
        and rel_h2_path is not None
        and "run_it" not in rel_h1_path.name
        and "run_it" not in rel_h2_path.name
    ):
        notes.append(
            "K=1 RECOVAR did not converge before its iteration cap; RELION final half maps are being used, "
            "so RECOVAR-vs-GT and RELION-vs-GT rows may compare pre-final RECOVAR maps to final RELION maps."
        )

    particle_metrics = None
    if relion_selected:
        relion_iter = max(_relion_iteration_from_path(rel_h1_path), _relion_iteration_from_path(rel_h2_path))
        relion_has_next_iteration = _relion_iteration_exists(relion_dir, relion_iter + 1)
        relion_half_maps_are_numbered = (
            rel_h1_path is not None
            and rel_h2_path is not None
            and "run_it" in rel_h1_path.name
            and "run_it" in rel_h2_path.name
        )
        final_guard_failures = [str(msg) for msg in final_all_data.get("failures", [])]
        optional_max_iter_final_failures = {"final_all_data_ran is false", "missing fsc_final_all_data"}
        if (
            final_guard_failures
            and set(final_guard_failures).issubset(optional_max_iter_final_failures)
            and recovar_convergence.get("convergence_has_converged") is False
            and relion_half_maps_are_numbered
            and not relion_has_next_iteration
        ):
            notes.append(
                "optional K=1 final all-data guard: RECOVAR ended at max_iter without convergence "
                f"and RELION fixture has no run_it{relion_iter + 1:03d} products"
            )
        elif final_guard_failures:
            notes.extend(f"K=1 final all-data guard: {msg}" for msg in final_guard_failures)
        particle_metrics = _particle_metrics(
            recovar_dir=recovar_dir,
            relion_dir=relion_dir,
            fixture_dir=fixture_dir,
            relion_iter=relion_iter,
        )
    elif final_all_data.get("failures"):
        notes.extend(f"K=1 final all-data guard: {msg}" for msg in final_all_data["failures"])
    if particle_metrics is not None and particle_metrics.get("notes"):
        for msg in particle_metrics["notes"]:
            if (
                msg == "missing RECOVAR final all-data FSC"
                and recovar_convergence.get("convergence_has_converged") is False
                and not relion_has_next_iteration
            ):
                notes.append(
                    "optional K=1 particle metrics: missing RECOVAR final all-data FSC; "
                    f"RECOVAR ended at max_iter without convergence and RELION fixture has no run_it{relion_iter + 1:03d} products"
                )
            else:
                notes.append(f"K=1 particle metrics: {msg}")

    status = _completion_status_from_metrics(
        metrics,
        recovar_metric_keys=(
            "recovar_half1_vs_relion_half1",
            "recovar_half2_vs_relion_half2",
            "recovar_merged_vs_relion_merged",
            "recovar_merged_vs_gt",
        ),
        notes=notes,
    )
    return {
        "status": status,
        "metadata": _completion_metadata(recovar_dir, "k1"),
        "recovar_convergence": recovar_convergence,
        "final_all_data": final_all_data,
        "paths": {
            "recovar_merged": str(rec_merged_path) if rec_merged_path else None,
            "recovar_half1": str(rec_h1_path) if rec_h1_path else None,
            "recovar_half2": str(rec_h2_path) if rec_h2_path else None,
            "relion_half1": str(rel_h1_path) if rel_h1_path else None,
            "relion_half2": str(rel_h2_path) if rel_h2_path else None,
            "gt": str(gt_path) if gt_path else None,
        },
        "metrics": metrics,
        "sign_ambiguity": sign_ambiguity,
        "particle_metrics": particle_metrics,
        "timing": _timing_summary(recovar_dir, relion_dir),
        "runtime_defaults": runtime_defaults,
        "notes": notes,
    }


def _k4_relion_path(relion_dir: Path | None, class_idx: int) -> Path | None:
    exact = _existing_path(relion_dir, [f"run_it015_class{class_idx:03d}.mrc"])
    return exact or _latest_relion_iteration_class_path(relion_dir, class_idx)


def summarize_k4(recovar_dir: Path | None, relion_dir: Path | None, fixture_dir: Path | None) -> dict[str, Any]:
    notes: list[str] = []
    if recovar_dir is None or relion_dir is None or fixture_dir is None:
        return {"status": "skipped", "notes": ["missing one or more K=4 input directories"]}

    recovar_convergence = _recovar_convergence_metadata(recovar_dir)
    runtime_defaults = _check_k4_refinement_runtime_defaults(recovar_dir)
    if runtime_defaults is not None and runtime_defaults.get("failures"):
        notes.extend(f"K=4 runtime default guard: {msg}" for msg in runtime_defaults["failures"])

    rec_paths = [_existing_path(recovar_dir, [f"final_class{i:03d}.mrc"]) for i in range(1, 5)]
    rel_paths = [_k4_relion_path(relion_dir, i) for i in range(1, 5)]
    gt_paths = [_existing_path(fixture_dir, [f"reference_gt_class{i:03d}.mrc"]) for i in range(1, 5)]

    rec_vols = [
        vol
        for vol in (
            _load_optional(path, _load_recovar_volume, f"K=4 RECOVAR class {idx}", notes)
            for idx, path in enumerate(rec_paths, start=1)
        )
        if vol is not None
    ]
    rel_vols = [
        vol
        for vol in (
            _load_optional(path, _load_relion_volume, f"K=4 RELION class {idx}", notes)
            for idx, path in enumerate(rel_paths, start=1)
        )
        if vol is not None
    ]
    gt_vols = [
        vol
        for vol in (
            _load_optional(path, _load_recovar_volume, f"K=4 GT class {idx}", notes)
            for idx, path in enumerate(gt_paths, start=1)
        )
        if vol is not None
    ]

    metrics: dict[str, Any] = {}
    if len(rec_vols) == 4 and len(rel_vols) == 4:
        metrics["recovar_vs_relion"] = best_permutation_summary(
            rec_vols,
            rel_vols,
            rhs_label="relion",
            include_fsc=False,
        )
    if len(rec_vols) == 4 and len(gt_vols) == 4:
        metrics["recovar_vs_gt"] = best_permutation_summary(
            rec_vols,
            gt_vols,
            rhs_label="gt",
            include_fsc=True,
        )
    if len(rel_vols) == 4 and len(gt_vols) == 4:
        metrics["relion_vs_gt"] = best_permutation_summary(
            rel_vols,
            gt_vols,
            rhs_label="gt",
            include_fsc=True,
        )

    relion_iter_candidates = [_relion_iteration_from_path(path) for path in rel_paths if path is not None]
    relion_iter = max(relion_iter_candidates) if relion_iter_candidates else 15
    class_permutation = None
    if "recovar_vs_relion" in metrics:
        class_permutation = metrics["recovar_vs_relion"].get("permutation_lhs_to_rhs")
    particle_metrics = _particle_metrics(
        recovar_dir=recovar_dir,
        relion_dir=relion_dir,
        fixture_dir=fixture_dir,
        relion_iter=relion_iter,
        class_permutation=class_permutation,
    )
    if particle_metrics is not None and particle_metrics.get("notes"):
        relion_has_next_iteration = _relion_iteration_exists(relion_dir, relion_iter + 1)
        for msg in particle_metrics["notes"]:
            if (
                msg == "missing RECOVAR final all-data FSC"
                and recovar_convergence.get("convergence_has_converged") is False
                and not relion_has_next_iteration
            ):
                notes.append(
                    "optional K=4 particle metrics: missing RECOVAR final all-data FSC; "
                    f"RECOVAR ended at max_iter without convergence and RELION fixture has no run_it{relion_iter + 1:03d} products"
                )
            else:
                notes.append(f"K=4 particle metrics: {msg}")

    status = _completion_status_from_metrics(
        metrics,
        recovar_metric_keys=("recovar_vs_relion", "recovar_vs_gt"),
        notes=notes,
    )
    return {
        "status": status,
        "metadata": _completion_metadata(recovar_dir, "k4"),
        "recovar_convergence": recovar_convergence,
        "paths": {
            "recovar_classes": [str(path) if path else None for path in rec_paths],
            "relion_classes": [str(path) if path else None for path in rel_paths],
            "gt_classes": [str(path) if path else None for path in gt_paths],
        },
        "metrics": metrics,
        "particle_metrics": particle_metrics,
        "timing": _timing_summary(recovar_dir, relion_dir),
        "runtime_defaults": runtime_defaults,
        "notes": notes,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_markdown(summary: dict[str, Any]) -> str:
    lines = ["# EM Completion Benchmark Summary", ""]
    for label in ("k1", "k4"):
        section = summary[label]
        lines.append(f"## {label.upper()}")
        lines.append("")
        lines.append(f"Status: `{section.get('status', 'unknown')}`")
        _append_metadata(lines, section.get("metadata", {}))
        timing = section.get("timing", {})
        if timing:
            lines.append("")
            lines.append("| Timing | Value |")
            lines.append("|---|---:|")
            for key in (
                "n_images",
                "recovar_walltime_s",
                "relion_walltime_s",
                "speed_ratio_relion_over_recovar",
                "recovar_relion_wall_ratio",
                "recovar_throughput_images_per_s",
                "relion_throughput_images_per_s",
            ):
                lines.append(f"| {key} | {_format_value(timing.get(key))} |")
            for key in ("recovar_peak_gpu_memory_gib", "relion_peak_gpu_memory_gib"):
                if key in timing:
                    lines.append(f"| {key} | {_format_value(timing.get(key))} |")
            for key in ("recovar_iteration_walltime_total_s", "recovar_iteration_walltime_mean_s"):
                if key in timing:
                    lines.append(f"| {key} | {_format_value(timing.get(key))} |")
            for key in (
                "sparse_pass2_count",
                "sparse_pass2_total_s",
                "sparse_pass2_mean_s",
                "sparse_pass2_median_s",
                "sparse_pass2_completed_iteration_total_s",
                "sparse_pass2_fraction_of_completed_iteration_wall",
                "sparse_kclass_group_timing_count",
                "sparse_kclass_group_wall_total_s",
                "sparse_kclass_group_profiled_total_s",
                "sparse_kclass_group_score_total_s",
                "sparse_kclass_group_mstep_noise_stats_total_s",
                "sparse_kclass_group_mstep_weighted_sums_total_s",
                "sparse_kclass_group_mstep_adjoint_total_s",
                "sparse_kclass_group_noise_total_s",
                "sparse_kclass_group_stats_total_s",
                "sparse_kclass_group_score_fraction_of_group_wall",
                "sparse_kclass_group_mstep_noise_stats_fraction_of_group_wall",
                "sparse_kclass_group_profiled_fraction_of_group_wall",
                "global_profile_row_count",
                "global_profile_kclass_row_count",
            ):
                if key in timing:
                    lines.append(f"| {key} | {_format_value(timing.get(key))} |")
            _append_run_log_telemetry(lines, timing)
            _append_iteration_timing(lines, timing)
            _append_global_profile_timing(lines, timing)
            _append_batch_sizing(lines, timing)
            _append_sparse_pass2(lines, timing)
            _append_sparse_group_timing(lines, timing)
            _append_compact_pair_planner(lines, timing)
            _append_local_search(lines, timing)
        _append_correctness_gate(
            lines,
            label,
            section.get("metrics", {}),
            sign_invariant_gt=(label == "k1" and _section_allows_global_sign(section)),
        )
        if label == "k1":
            _append_k1_metrics(lines, section.get("metrics", {}))
        else:
            _append_k4_metrics(lines, section.get("metrics", {}))
        _append_particle_metrics(lines, section.get("particle_metrics"))
        notes = section.get("notes") or []
        timing_notes = list(section.get("timing", {}).get("notes") or [])
        for monitor_key in ("recovar_gpu_monitor", "relion_gpu_monitor"):
            for note in section.get("timing", {}).get(monitor_key, {}).get("notes") or []:
                timing_notes.append(f"{monitor_key}: {note}")
        all_notes = [*notes, *timing_notes]
        if all_notes:
            lines.append("")
            lines.append("Notes:")
            for note in all_notes:
                lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _append_metadata(lines: list[str], metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    keys = (
        "head",
        "branch",
        "job_id",
        "setup_job_id",
        "summary_job_id",
        "scratch_dir",
        "run_log",
        "slurm_stdout",
        "slurm_stderr",
        "job_script",
        "submission_env_path",
        "slurm_walltime_path",
    )
    lines.append("")
    lines.append("| Metadata | Value |")
    lines.append("|---|---|")
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            lines.append(f"| {key} | `{value}` |")
    missing = metadata.get("missing_artifacts")
    if isinstance(missing, list) and missing:
        lines.append(f"| missing_artifacts | `{', '.join(str(value) for value in missing)}` |")
    env = metadata.get("env")
    if isinstance(env, dict):
        for key, value in env.items():
            lines.append(f"| env.{key} | `{value}` |")


def _append_run_log_telemetry(lines: list[str], timing: dict[str, Any]) -> None:
    telemetry = timing.get("run_log_telemetry")
    if not isinstance(telemetry, dict):
        return
    counts = telemetry.get("counts") if isinstance(telemetry.get("counts"), dict) else {}
    lines.append("")
    lines.append("| Log Telemetry | Value |")
    lines.append("|---|---|")
    for key in ("status", "path"):
        lines.append(f"| {key} | {_format_value(telemetry.get(key))} |")
    for key in (
        "iteration_rows",
        "batch_sizing_events",
        "sparse_pass2_events",
        "sparse_group_timing_events",
        "compact_pair_planner_events",
        "local_search_events",
    ):
        lines.append(f"| count.{key} | {_format_value(counts.get(key))} |")
    missing = telemetry.get("missing_artifacts")
    if isinstance(missing, list) and missing:
        lines.append(f"| missing_artifacts | `{', '.join(str(value) for value in missing)}` |")


def _append_iteration_timing(lines: list[str], timing: dict[str, Any]) -> None:
    parsed_rows = timing.get("recovar_iteration_rows")
    rows = timing.get("iteration_timing_rows")
    wall_times = timing.get("wall_times_trajectory_s")
    if isinstance(parsed_rows, list) and parsed_rows:
        rows = parsed_rows
    if not isinstance(rows, list):
        rows = []
    if not isinstance(wall_times, list):
        wall_times = []
    if not rows and not wall_times:
        return

    current_sizes = timing.get("current_sizes") if isinstance(timing.get("current_sizes"), list) else []
    pixel_resolutions = (
        timing.get("pixel_resolutions") if isinstance(timing.get("pixel_resolutions"), list) else []
    )
    ave_pmax = timing.get("ave_pmax_trajectory") if isinstance(timing.get("ave_pmax_trajectory"), list) else []
    stage_by_iter = (
        timing.get("iteration_timing_summary", {}).get("stage_delta_by_relion_iter", {})
        if isinstance(timing.get("iteration_timing_summary"), dict)
        else {}
    )

    n_rows = max(len(rows), len(wall_times), len(current_sizes), len(pixel_resolutions), len(ave_pmax))
    lines.append("")
    lines.append(
        "| Iter | Wall s | E-step s | Recon s | FSC s | Noise s | Convergence s | Current size | Pixel res | Res A | Healpix | Ave Pmax |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx in range(n_rows):
        row = rows[idx] if idx < len(rows) and isinstance(rows[idx], dict) else {}
        relion_iter = row.get("iteration") or row.get("relion_iteration") or idx + 1
        wall_time = row.get("wall_time_s") if row else None
        if wall_time is None:
            wall_time = _list_get(wall_times, idx)
        stages = stage_by_iter.get(str(relion_iter), {}) if isinstance(stage_by_iter, dict) else {}
        current_size = row.get("current_size") if row else None
        pixel_res = row.get("pixel_res") if row else None
        ave_pmax_value = row.get("ave_pmax") if row else None
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(relion_iter),
                    _format_value(wall_time),
                    _format_value(stages.get("e_step") if isinstance(stages, dict) else None),
                    _format_value(stages.get("recon") if isinstance(stages, dict) else None),
                    _format_value(stages.get("fsc") if isinstance(stages, dict) else None),
                    _format_value(stages.get("noise_update") if isinstance(stages, dict) else None),
                    _format_value(stages.get("convergence") if isinstance(stages, dict) else None),
                    _format_value(current_size if current_size is not None else _list_get(current_sizes, idx)),
                    _format_value(pixel_res if pixel_res is not None else _list_get(pixel_resolutions, idx)),
                    _format_value(row.get("res_ang") if row else None),
                    _format_value(row.get("healpix_order") if row else None),
                    _format_value(ave_pmax_value if ave_pmax_value is not None else _list_get(ave_pmax, idx)),
                ]
            )
            + " |"
        )


def _append_global_profile_timing(lines: list[str], timing: dict[str, Any]) -> None:
    top_times = timing.get("global_profile_top_time_totals_s")
    if not isinstance(top_times, list) or not top_times:
        return
    lines.append("")
    lines.append("| Global Profile Key | Total s |")
    lines.append("|---|---:|")
    for row in top_times[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(f"| {_format_value(row.get('key'))} | {_format_value(row.get('seconds'))} |")


def _append_batch_sizing(lines: list[str], timing: dict[str, Any]) -> None:
    events = timing.get("batch_sizing_events")
    if not isinstance(events, list) or not events:
        return
    compact: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        key = (
            event.get("iteration"),
            event.get("current_size"),
            event.get("image_batch_size"),
            event.get("rotation_block_size"),
            event.get("n_rot"),
            event.get("n_trans"),
            event.get("n_classes"),
            event.get("score_pixels"),
        )
        if key in seen:
            continue
        seen.add(key)
        compact.append(event)
    lines.append("")
    lines.append("| Batch Iter | Current size | Image batch | Rotation block | Rotations | Translations | K | Score pixels |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for event in compact[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(event.get("iteration")),
                    _format_value(event.get("current_size")),
                    _format_value(event.get("image_batch_size")),
                    _format_value(event.get("rotation_block_size")),
                    _format_value(event.get("n_rot")),
                    _format_value(event.get("n_trans")),
                    _format_value(event.get("n_classes")),
                    _format_value(event.get("score_pixels")),
                ]
            )
            + " |"
        )
    if len(compact) > 20:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | {len(compact) - 20} more rows |")


def _append_sparse_pass2(lines: list[str], timing: dict[str, Any]) -> None:
    events = timing.get("sparse_pass2_events")
    if not isinstance(events, list) or not events:
        return
    compact: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        key = (
            event.get("iteration"),
            event.get("current_size"),
            event.get("fused"),
            event.get("n_images"),
            event.get("n_classes"),
            event.get("buckets"),
            event.get("max_images_per_microbatch"),
            event.get("small_bucket_threshold"),
            event.get("small_bucket_max_images_per_microbatch"),
            event.get("max_translation_tile_bytes"),
            event.get("max_projected_rotations_per_projection_call"),
            event.get("n_score_pixels"),
            event.get("e_m_time_s"),
        )
        if key in seen:
            continue
        seen.add(key)
        compact.append(event)
    lines.append("")
    lines.append(
        "| Sparse Iter | Fused | Images | K | Buckets | E+M s | Max images/MB | Small bucket threshold | Small bucket max images/MB | Translation tile bytes | Projected rotations cap | Score pixels | Median valid candidates/image |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for event in compact[:24]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(event.get("iteration")),
                    _format_value(event.get("fused")),
                    _format_value(event.get("n_images")),
                    _format_value(event.get("n_classes")),
                    _format_value(event.get("buckets")),
                    _format_value(event.get("e_m_time_s")),
                    _format_value(event.get("max_images_per_microbatch")),
                    _format_value(event.get("small_bucket_threshold")),
                    _format_value(event.get("small_bucket_max_images_per_microbatch")),
                    _format_value(event.get("max_translation_tile_bytes")),
                    _format_value(event.get("max_projected_rotations_per_projection_call")),
                    _format_value(event.get("n_score_pixels")),
                    _format_value(event.get("median_valid_candidates_per_image")),
                ]
            )
            + " |"
        )
    if len(compact) > 24:
        lines.append(
            f"| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | {len(compact) - 24} more rows |"
        )


def _append_sparse_group_timing(lines: list[str], timing: dict[str, Any]) -> None:
    events = timing.get("sparse_group_timing_events")
    if not isinstance(events, list) or not events:
        return
    by_mode = timing.get("sparse_kclass_group_by_mode")
    if isinstance(by_mode, dict) and by_mode:
        lines.append("")
        lines.append(
            "| K4 Group Mode | Count | Wall s | Profiled s | Score s | M-step/noise/stats s | Weighted sums s | Adjoint s | Noise s | Stats s | Score/Wall | M-step/Wall |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mode, row in sorted(by_mode.items()):
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_value(mode),
                        _format_value(row.get("count")),
                        _format_value(row.get("wall_total_s")),
                        _format_value(row.get("profiled_total_s")),
                        _format_value(row.get("score_total_s")),
                        _format_value(row.get("mstep_noise_stats_total_s")),
                        _format_value(row.get("mstep_weighted_sums_total_s")),
                        _format_value(row.get("mstep_adjoint_total_s")),
                        _format_value(row.get("noise_total_s")),
                        _format_value(row.get("stats_total_s")),
                        _format_value(row.get("score_fraction_of_wall")),
                        _format_value(row.get("mstep_noise_stats_fraction_of_wall")),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append(
        "| K4 Group Iter | Current size | Mode | Group | Build s | Fetch s | Prepare s | Score s | M-step/noise/stats s | Weighted sums s | Adjoint s | Noise s | Stats s | Profiled s | Wall s |"
    )
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for event in events[:32]:
        if not isinstance(event, dict):
            continue
        group_label = None
        if event.get("group_key") is not None and event.get("group_value") is not None:
            group_label = f"{event.get('group_key')}={event.get('group_value')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(event.get("iteration")),
                    _format_value(event.get("current_size")),
                    _format_value(event.get("mode")),
                    _format_value(group_label),
                    _format_value(event.get("build_s")),
                    _format_value(event.get("fetch_s")),
                    _format_value(event.get("prepare_s")),
                    _format_value(event.get("score_s")),
                    _format_value(event.get("mstep_noise_stats_s")),
                    _format_value(event.get("mstep_weighted_sums_s")),
                    _format_value(event.get("mstep_adjoint_s")),
                    _format_value(event.get("noise_s")),
                    _format_value(event.get("stats_s")),
                    _format_value(event.get("total_profiled_s")),
                    _format_value(event.get("wall_s")),
                ]
            )
            + " |"
        )
    if len(events) > 32:
        lines.append(
            f"| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | {len(events) - 32} more rows |"
        )


def _append_compact_pair_planner(lines: list[str], timing: dict[str, Any]) -> None:
    events = timing.get("compact_pair_planner_events")
    if not isinstance(events, list) or not events:
        return
    lines.append("")
    lines.append(
        "| Compact Pair Iter | Current size | Valid pairs | Padded pairs | Rectangular pairs | Valid reduction | Padded reduction | Compact buckets | Median valid pairs/image | Mean valid pairs/image | Max valid pairs/image | Plan s |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for event in events[:24]:
        if not isinstance(event, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(event.get("iteration")),
                    _format_value(event.get("current_size")),
                    _format_value(event.get("valid_pair_candidates")),
                    _format_value(event.get("padded_pair_candidates")),
                    _format_value(event.get("rectangular_candidates")),
                    _format_value(event.get("valid_reduction")),
                    _format_value(event.get("padded_reduction")),
                    _format_value(event.get("compact_buckets")),
                    _format_value(event.get("median_valid_pairs_per_image")),
                    _format_value(event.get("mean_valid_pairs_per_image")),
                    _format_value(event.get("max_valid_pairs_per_image")),
                    _format_value(event.get("plan_time")),
                ]
            )
            + " |"
        )
    if len(events) > 24:
        lines.append(
            f"| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | {len(events) - 24} more rows |"
        )


def _append_local_search(lines: list[str], timing: dict[str, Any]) -> None:
    events = timing.get("local_search_events")
    if not isinstance(events, list) or not events:
        return
    lines.append("")
    lines.append(
        "| Local Iter | Current size | Chunks | Big-JIT buckets | Sparse buckets | Reconstruction rows | Padded rows | Padding/recon | Significant samples | Samples/padded row |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for event in events[:24]:
        if not isinstance(event, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(event.get("iteration")),
                    _format_value(event.get("current_size")),
                    _format_value(event.get("chunks")),
                    _format_value(event.get("big_jit_buckets")),
                    _format_value(event.get("sparse_big_jit_buckets")),
                    _format_value(event.get("reconstruction_rows")),
                    _format_value(event.get("padded_rows")),
                    _format_value(event.get("padding_to_reconstruction_ratio")),
                    _format_value(event.get("significant_samples")),
                    _format_value(event.get("samples_per_padded_row")),
                ]
            )
            + " |"
        )
    if len(events) > 24:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... | ... | {len(events) - 24} more rows |")


def _list_get(values: list[Any], index: int) -> Any | None:
    return values[index] if index < len(values) else None


def _ordered_metric_items(metrics: dict[str, Any], preferred: tuple[str, ...]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for name in preferred:
        values = metrics.get(name)
        if isinstance(values, dict):
            items.append((name, values))
            seen.add(name)
    for name, values in metrics.items():
        if name not in seen and isinstance(values, dict):
            items.append((name, values))
    return items


def _metric_fsc_auc(values: dict[str, Any]) -> float:
    existing = values.get("fsc_auc")
    if existing is not None:
        try:
            existing_value = float(existing)
        except (TypeError, ValueError):
            existing_value = float("nan")
        if np.isfinite(existing_value):
            return existing_value

    curve = values.get("fsc")
    if curve is None:
        return float("nan")
    for axis_key in ("fsc_shells", "shells", "shell", "radii", "radius", "freq", "frequency"):
        axis = values.get(axis_key)
        if axis is not None:
            auc = normalized_fsc_auc(curve, axis)
            if np.isfinite(auc):
                return auc
    return normalized_fsc_auc(curve)


def _metric_sign_invariant_fsc_auc(values: dict[str, Any]) -> float:
    existing = values.get("fsc_auc_sign_invariant")
    if existing is not None:
        try:
            existing_value = float(existing)
        except (TypeError, ValueError):
            existing_value = float("nan")
        if np.isfinite(existing_value):
            return existing_value

    signed = _metric_fsc_auc(values)
    flipped = values.get("fsc_auc_sign_flipped")
    try:
        flipped_value = float(flipped)
    except (TypeError, ValueError):
        flipped_value = float("nan")
    if np.isfinite(signed) and np.isfinite(flipped_value):
        return max(float(signed), float(flipped_value))

    curve = values.get("fsc")
    if curve is None:
        return abs(float(signed)) if np.isfinite(signed) else float("nan")
    fsc = np.asarray(curve, dtype=np.float64)
    for axis_key in ("fsc_shells", "shells", "shell", "radii", "radius", "freq", "frequency"):
        axis = values.get(axis_key)
        if axis is not None:
            auc = normalized_fsc_auc(fsc, axis)
            flipped_auc = normalized_fsc_auc(-fsc, axis)
            if np.isfinite(auc) and np.isfinite(flipped_auc):
                return max(float(auc), float(flipped_auc))
    auc = normalized_fsc_auc(fsc)
    flipped_auc = normalized_fsc_auc(-fsc)
    if np.isfinite(auc) and np.isfinite(flipped_auc):
        return max(float(auc), float(flipped_auc))
    return float("nan")


def _summary_mean_fsc_auc(values: dict[str, Any]) -> float:
    existing = values.get("mean_fsc_auc")
    if existing is not None:
        try:
            existing_value = float(existing)
        except (TypeError, ValueError):
            existing_value = float("nan")
        if np.isfinite(existing_value):
            return existing_value
    per_class = values.get("per_class")
    if not isinstance(per_class, list):
        return float("nan")
    return _finite_mean([_metric_fsc_auc(row) for row in per_class if isinstance(row, dict)])


def _finite_delta(lhs: float, rhs: float) -> float:
    if not np.isfinite(lhs) or not np.isfinite(rhs):
        return float("nan")
    return float(lhs - rhs)


def _section_allows_global_sign(section: dict[str, Any]) -> bool:
    sign_ambiguity = section.get("sign_ambiguity")
    return isinstance(sign_ambiguity, dict) and bool(sign_ambiguity.get("allow_global_sign"))


def _correctness_gate_rows(
    label: str,
    metrics: dict[str, Any],
    *,
    sign_invariant_gt: bool = False,
) -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []
    if label == "k1":
        rec = metrics.get("recovar_merged_vs_gt")
        rel = metrics.get("relion_merged_vs_gt")
        if isinstance(rec, dict) or isinstance(rel, dict):
            auc_fn = _metric_sign_invariant_fsc_auc if sign_invariant_gt else _metric_fsc_auc
            rec_auc = auc_fn(rec) if isinstance(rec, dict) else float("nan")
            rel_auc = auc_fn(rel) if isinstance(rel, dict) else float("nan")
            row_name = "merged_vs_gt_sign_invariant" if sign_invariant_gt else "merged_vs_gt"
            rows.append((row_name, rec_auc, rel_auc, _finite_delta(rec_auc, rel_auc)))
    elif label == "k4":
        rec = metrics.get("recovar_vs_gt")
        rel = metrics.get("relion_vs_gt")
        if isinstance(rec, dict) or isinstance(rel, dict):
            rec_auc = _summary_mean_fsc_auc(rec) if isinstance(rec, dict) else float("nan")
            rel_auc = _summary_mean_fsc_auc(rel) if isinstance(rel, dict) else float("nan")
            rows.append(("class_mean_vs_gt", rec_auc, rel_auc, _finite_delta(rec_auc, rel_auc)))

    return rows


def _append_correctness_gate(
    lines: list[str],
    label: str,
    metrics: dict[str, Any],
    *,
    sign_invariant_gt: bool = False,
) -> None:
    """Report the FSC-AUC acceptance metric before correlation-heavy details."""

    rows = _correctness_gate_rows(label, metrics, sign_invariant_gt=sign_invariant_gt)
    if not rows:
        return

    lines.append("")
    lines.append("| Correctness Gate | RECOVAR GT FSC AUC | RELION GT FSC AUC | Delta |")
    lines.append("|---|---:|---:|---:|")
    for name, rec_auc, rel_auc, delta in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_value(rec_auc),
                    _format_value(rel_auc),
                    _format_value(delta),
                ]
            )
            + " |"
        )


def _append_k1_metrics(lines: list[str], metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    lines.append("")
    lines.append(
        "| Comparison | FSC AUC | Corr | Shift z,y,x | Shift norm | FSC 1-8 | FSC 1-16 | Shell 0.5 | Shell 0.143 |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for name, values in _ordered_metric_items(
        metrics,
        (
            "recovar_merged_vs_gt",
            "relion_merged_vs_gt",
            "relion_halfavg_vs_gt",
            "recovar_merged_vs_relion_merged",
            "recovar_merged_vs_relion_halfavg",
            "recovar_half1_vs_relion_half1",
            "recovar_half2_vs_relion_half2",
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_value(_metric_fsc_auc(values)),
                    _format_value(values.get("corr")),
                    _format_shift(values.get("integer_shift_lhs_to_rhs_zyx")),
                    _format_value(values.get("integer_shift_norm_voxels")),
                    _format_value(values.get("mean_fsc_1_8")),
                    _format_value(values.get("mean_fsc_1_16")),
                    _format_value(values.get("shell_05")),
                    _format_value(values.get("shell_0143")),
                ]
            )
            + " |"
        )


def _append_k4_metrics(lines: list[str], metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    lines.append("")
    lines.append("| Comparison | Permutation | Mean FSC AUC | Mean Corr | Mean FSC 1-8 | Mean FSC 1-16 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name, values in _ordered_metric_items(metrics, ("recovar_vs_gt", "relion_vs_gt", "recovar_vs_relion")):
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(values.get("permutation_lhs_to_rhs")),
                    _format_value(_summary_mean_fsc_auc(values)),
                    _format_value(values.get("mean_corr")),
                    _format_value(values.get("mean_fsc_1_8")),
                    _format_value(values.get("mean_fsc_1_16")),
                ]
            )
            + " |"
        )


def _append_particle_metrics(lines: list[str], metrics: dict[str, Any] | None) -> None:
    if not isinstance(metrics, dict):
        return
    rows: list[tuple[str, Any]] = []
    pmax = metrics.get("pmax")
    if isinstance(pmax, dict):
        rows.extend(
            [
                ("pmax_matched_count", pmax.get("matched_count")),
                ("pmax_recovar_mean", pmax.get("recovar_mean")),
                ("pmax_relion_mean", pmax.get("relion_mean")),
                ("pmax_gap_mean_recovar_minus_relion", pmax.get("gap_mean_recovar_minus_relion")),
                ("pmax_corr", pmax.get("corr")),
                ("pmax_abs_diff_mean", (pmax.get("abs_diff") or {}).get("mean")),
                ("pmax_abs_diff_p95", (pmax.get("abs_diff") or {}).get("p95")),
                ("pmax_abs_diff_max", (pmax.get("abs_diff") or {}).get("max")),
            ]
        )
    final_fsc = metrics.get("final_all_data_fsc")
    if isinstance(final_fsc, dict):
        rows.extend(
            [
                ("final_all_data_fsc_npz_key", final_fsc.get("npz_key")),
                ("final_all_data_fsc_auc", final_fsc.get("fsc_auc")),
                ("final_all_data_fsc_1_16", final_fsc.get("mean_fsc_1_16")),
                ("final_all_data_fsc_shell_0_5", final_fsc.get("shell_05")),
                ("final_all_data_fsc_shell_0_143", final_fsc.get("shell_0143")),
            ]
        )
    final_sampling = metrics.get("final_all_data_sampling")
    if isinstance(final_sampling, dict):
        rows.extend(
            [
                ("final_all_data_sampling_perturbation", final_sampling.get("perturbation")),
                ("final_all_data_sampling_applied", final_sampling.get("applied")),
                ("final_all_data_sampling_relion_iteration", final_sampling.get("relion_iteration")),
                ("final_all_data_sampling_star", final_sampling.get("sampling_star")),
                ("final_all_data_sampling_star_source", final_sampling.get("sampling_star_source")),
                ("final_all_data_sampling_offset_range_px", final_sampling.get("offset_range_px")),
                ("final_all_data_sampling_offset_step_px", final_sampling.get("offset_step_px")),
            ]
        )
    final_reconstruction = metrics.get("final_all_data_reconstruction")
    if isinstance(final_reconstruction, dict):
        rows.extend(
            [
                ("final_all_data_grid_correct", final_reconstruction.get("grid_correct")),
                ("final_all_data_gridding_correct", final_reconstruction.get("gridding_correct")),
            ]
        )
    pose = metrics.get("pose_rotation_deg")
    if isinstance(pose, dict):
        rows.extend(
            [
                ("pose_npz_key", pose.get("npz_key")),
                ("pose_matched_count", pose.get("matched_count")),
                ("pose_angle_error_mean_deg", (pose.get("angle_error") or {}).get("mean")),
                ("pose_angle_error_p95_deg", (pose.get("angle_error") or {}).get("p95")),
                ("pose_within_1deg_fraction", pose.get("within_1deg_fraction")),
                ("pose_within_5deg_fraction", pose.get("within_5deg_fraction")),
            ]
        )
    trans = metrics.get("translation_px")
    if isinstance(trans, dict):
        rows.extend(
            [
                ("translation_npz_key", trans.get("npz_key")),
                ("translation_matched_count", trans.get("matched_count")),
                ("translation_l2_error_mean_px", (trans.get("l2_error") or {}).get("mean")),
                ("translation_l2_error_p95_px", (trans.get("l2_error") or {}).get("p95")),
                ("translation_within_0_5px_fraction", trans.get("within_0_5px_fraction")),
                ("translation_within_1px_fraction", trans.get("within_1px_fraction")),
            ]
        )
    cls = metrics.get("class_assignment")
    if isinstance(cls, dict):
        rows.extend(
            [
                ("class_assignment_matched_count", cls.get("matched_count")),
                ("class_assignment_raw_agreement", cls.get("raw_agreement")),
                ("class_assignment_permuted_agreement", cls.get("permuted_agreement")),
            ]
        )
    if not rows:
        return
    lines.append("")
    lines.append("| Particle Metric | Value |")
    lines.append("|---|---:|")
    for key, value in rows:
        lines.append(f"| {key} | {_format_value(value)} |")


def _format_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def _format_shift(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        values = [int(v) for v in value]
    except (TypeError, ValueError):
        return "missing"
    if len(values) != 3:
        return "missing"
    return ",".join(str(v) for v in values)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    summary = {
        "schema": "em_completion_bench_summary_v1",
        "k1": summarize_k1(args.k1_recovar_dir, args.k1_relion_dir, args.k1_fixture_dir),
        "k4": summarize_k4(args.k4_recovar_dir, args.k4_relion_dir, args.k4_fixture_dir),
    }
    _annotate_timing_probe_status(summary)
    return summary


def _required_cases(args: argparse.Namespace) -> tuple[str, ...]:
    cases: list[str] = []
    if bool(args.require_all or args.require_k1):
        cases.append("k1")
    if bool(args.require_all or args.require_k4):
        cases.append("k4")
    return tuple(cases)


_ALLOWED_REQUIRED_NOTES = {
    "missing K=1 RELION final merged map",
    "K=1 RELION final merged map missing; recovar_merged_vs_relion_merged uses half-map average",
    "K=1 RELION final merged map missing; relion_merged_vs_gt uses half-map average",
    "K=1 no-CTF fixture: global sign is ambiguous; required GT FSC-AUC gate uses sign-invariant FSC-AUC while signed metrics are still reported",
    "using RELION slurm_walltime.json",
}


def _section_has_missing_required_products(section: dict[str, Any]) -> bool:
    if section.get("status") != "ok":
        return True
    notes = [str(note) for note in section.get("notes", [])]
    for note in notes:
        if note.startswith("optional ") or note in _ALLOWED_REQUIRED_NOTES:
            continue
        return True
    return False


def _has_missing(summary: dict[str, Any], required_cases: tuple[str, ...]) -> bool:
    return any(_section_has_missing_required_products(summary[case]) for case in required_cases)


def _fsc_auc_gate_failure_notes(case: str, section: dict[str, Any], parity_tol: float) -> list[str]:
    label = "K=1" if case == "k1" else "K=4"
    rows = _correctness_gate_rows(
        case,
        section.get("metrics", {}),
        sign_invariant_gt=(case == "k1" and _section_allows_global_sign(section)),
    )
    if not rows:
        return [f"{label} was selected as required, but GT FSC-AUC correctness gate metrics are missing"]

    failures: list[str] = []
    for name, rec_auc, rel_auc, delta in rows:
        if not (np.isfinite(rec_auc) and np.isfinite(rel_auc)):
            failures.append(
                f"{label} GT FSC-AUC correctness gate {name} is non-finite "
                f"(RECOVAR={_format_value(rec_auc)}, RELION={_format_value(rel_auc)})"
            )
        elif rec_auc + parity_tol < rel_auc:
            failures.append(
                f"{label} GT FSC-AUC correctness gate failed for {name}: "
                f"RECOVAR={_format_value(rec_auc)}, RELION={_format_value(rel_auc)}, "
                f"delta={_format_value(delta)}, tolerance={_format_value(parity_tol)}"
            )
    return failures


def _mark_required_failures(
    summary: dict[str, Any],
    required_cases: tuple[str, ...],
    *,
    fsc_auc_parity_tol: float,
) -> None:
    for case in required_cases:
        section = summary[case]
        missing_required_products = _section_has_missing_required_products(section)
        gate_failures = [] if missing_required_products else _fsc_auc_gate_failure_notes(
            case,
            section,
            fsc_auc_parity_tol,
        )
        if not missing_required_products and not gate_failures:
            continue
        section["status"] = "failed"
        notes = section.setdefault("notes", [])
        label = "K=1" if case == "k1" else "K=4"
        if missing_required_products:
            message = f"{label} was selected as required, but required artifacts or metrics are missing"
            if message not in notes:
                notes.append(message)
        for message in gate_failures:
            if message not in notes:
                notes.append(message)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    required_cases = _required_cases(args)
    summary = summarize(args)
    _mark_required_failures(summary, required_cases, fsc_auc_parity_tol=float(args.fsc_auc_parity_tol))
    json_text = json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text)
    else:
        print(json_text)

    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(build_markdown(summary))

    if required_cases and _has_missing(summary, required_cases):
        labels = ", ".join(required_cases)
        print(
            f"summarize_em_completion_bench.py: required checks failed for {labels}; see summary notes",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
