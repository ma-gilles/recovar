#!/usr/bin/env python3
"""Audit one numbered K=1 RECOVAR checkpoint against a control run.

The audit separates execution invariants (particle decisions, grids, metadata,
and the shellwise half-map FSC) from floating-point products.  Large MRC maps
are compared in bounded-memory slabs so box-800 checkpoints can be audited on a
login node without materializing either volume in RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

SCHEMA = "recovar.em_k1_checkpoint_equivalence.v1"
DISCRETE_NPY_SUFFIXES = (
    "coarse_ha_half1.npy",
    "coarse_ha_half2.npy",
    "fsc.npy",
    "ha_half1.npy",
    "ha_half2.npy",
    "rotations.npy",
    "translations.npy",
)
DISCRETE_NPZ_SUFFIXES = (
    "particle_state_half1.npz",
    "particle_state_half2.npz",
)
FLOAT_NPY_SUFFIXES = (
    "Ft_ctf_0.npy",
    "Ft_ctf_1.npy",
    "Ft_y_0.npy",
    "Ft_y_1.npy",
    "noise.npy",
    "noise_half1.npy",
    "noise_half2.npy",
    "tau2.npy",
)
MAP_SUFFIXES = (
    "half1_reg.mrc",
    "half2_reg.mrc",
)


class AuditError(RuntimeError):
    """Raised when checkpoint evidence is missing or structurally invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"missing regular checkpoint artifact: {path}")
    return path


def _artifact_pair(
    baseline_dir: Path,
    candidate_dir: Path,
    prefix: str,
    suffix: str,
) -> tuple[Path, Path]:
    return (
        _require_file(baseline_dir / f"{prefix}_{suffix}"),
        _require_file(candidate_dir / f"{prefix}_{suffix}"),
    )


def _chunk_slices(shape: tuple[int, ...], itemsize: int, target_bytes: int = 64 * 1024**2):
    if not shape:
        yield ()
        return
    row_elements = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows = max(1, target_bytes // max(1, 2 * itemsize * row_elements))
    for start in range(0, shape[0], rows):
        yield (slice(start, min(start + rows, shape[0])), *([slice(None)] * (len(shape) - 1)))


def _numeric_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AuditError(
            "numeric artifact shape/dtype mismatch: "
            f"{left.shape}/{left.dtype} versus {right.shape}/{right.dtype}"
        )
    if not np.issubdtype(left.dtype, np.number):
        raise AuditError(f"expected numeric artifact, got {left.dtype}")

    exact = True
    finite = True
    count = 0
    difference_squared = 0.0
    left_squared = 0.0
    max_absolute = 0.0
    real_sums = np.zeros(5, dtype=np.float64) if not np.iscomplexobj(left) else None

    for key in _chunk_slices(left.shape, left.dtype.itemsize):
        left_chunk = np.asarray(left[key]) if key else np.asarray(left).reshape(1)
        right_chunk = np.asarray(right[key]) if key else np.asarray(right).reshape(1)
        exact &= bool(np.array_equal(left_chunk, right_chunk, equal_nan=True))
        finite &= bool(np.isfinite(left_chunk).all() and np.isfinite(right_chunk).all())
        left_wide = left_chunk.astype(
            np.complex128 if np.iscomplexobj(left_chunk) else np.float64,
            copy=False,
        )
        right_wide = right_chunk.astype(
            np.complex128 if np.iscomplexobj(right_chunk) else np.float64,
            copy=False,
        )
        difference = left_wide - right_wide
        absolute_difference = np.abs(difference)
        difference_squared += float(np.sum(absolute_difference * absolute_difference, dtype=np.float64))
        left_absolute = np.abs(left_wide)
        left_squared += float(np.sum(left_absolute * left_absolute, dtype=np.float64))
        max_absolute = max(max_absolute, float(np.max(absolute_difference, initial=0.0)))
        count += int(left_chunk.size)
        if real_sums is not None:
            left64 = left_wide
            right64 = right_wide
            real_sums += (
                float(np.sum(left64, dtype=np.float64)),
                float(np.sum(right64, dtype=np.float64)),
                float(np.sum(left64 * left64, dtype=np.float64)),
                float(np.sum(right64 * right64, dtype=np.float64)),
                float(np.sum(left64 * right64, dtype=np.float64)),
            )

    if count == 0:
        raise AuditError("checkpoint artifact is empty")
    relative_l2 = math.sqrt(difference_squared / left_squared) if left_squared else (0.0 if not difference_squared else math.inf)
    metrics: dict[str, Any] = {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "element_count": count,
        "element_exact": exact,
        "finite": finite,
        "max_absolute_difference": max_absolute,
        "rmse": math.sqrt(difference_squared / count),
        "relative_l2_difference": relative_l2,
    }
    if real_sums is not None:
        left_sum, right_sum, left_sum2, right_sum2, cross_sum = real_sums
        covariance = cross_sum - left_sum * right_sum / count
        left_variance = max(0.0, left_sum2 - left_sum * left_sum / count)
        right_variance = max(0.0, right_sum2 - right_sum * right_sum / count)
        denominator = math.sqrt(left_variance * right_variance)
        metrics["centered_correlation"] = covariance / denominator if denominator else None
    return metrics


def _numeric_file_record(left: Path, right: Path) -> dict[str, Any]:
    try:
        left_array = np.load(left, allow_pickle=False, mmap_mode="r")
        right_array = np.load(right, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, EOFError) as error:
        raise AuditError(f"failed to load numeric checkpoint artifact: {error}") from error
    return {
        "baseline": {"path": str(left), "sha256": sha256_file(left)},
        "candidate": {"path": str(right), "sha256": sha256_file(right)},
        "metrics": _numeric_metrics(left_array, right_array),
    }


def _objects_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        equal_nan = np.issubdtype(left.dtype, np.inexact)
        return bool(np.array_equal(left, right, equal_nan=equal_nan))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(_objects_equal(left[key], right[key]) for key in left)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            return False
        return len(left) == len(right) and all(
            _objects_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        return bool((np.isnan(left) and np.isnan(right)) or left == right)
    return bool(left == right)


def _object_file_record(left: Path, right: Path) -> dict[str, Any]:
    try:
        left_value = np.load(left, allow_pickle=True)
        right_value = np.load(right, allow_pickle=True)
    except (OSError, ValueError, EOFError) as error:
        raise AuditError(f"failed to load checkpoint metadata: {error}") from error
    if left_value.shape != () or right_value.shape != ():
        raise AuditError("checkpoint metadata must be an object scalar")
    return {
        "baseline": {"path": str(left), "sha256": sha256_file(left)},
        "candidate": {"path": str(right), "sha256": sha256_file(right)},
        "object_exact": _objects_equal(left_value.item(), right_value.item()),
    }


def _npz_file_record(left: Path, right: Path) -> dict[str, Any]:
    try:
        with np.load(left, allow_pickle=False) as left_archive, np.load(
            right, allow_pickle=False
        ) as right_archive:
            if left_archive.files != right_archive.files:
                raise AuditError(
                    f"NPZ members differ: {left_archive.files} versus {right_archive.files}"
                )
            members = {
                name: _numeric_metrics(
                    np.asarray(left_archive[name]),
                    np.asarray(right_archive[name]),
                )
                for name in left_archive.files
            }
    except (OSError, ValueError, EOFError) as error:
        raise AuditError(f"failed to load particle-state archive: {error}") from error
    return {
        "baseline": {"path": str(left), "sha256": sha256_file(left)},
        "candidate": {"path": str(right), "sha256": sha256_file(right)},
        "members": members,
        "all_members_exact": all(item["element_exact"] for item in members.values()),
    }


def _mrc_file_record(left: Path, right: Path) -> dict[str, Any]:
    try:
        with mrcfile.mmap(left, mode="r", permissive=True) as left_mrc, mrcfile.mmap(
            right, mode="r", permissive=True
        ) as right_mrc:
            metrics = _numeric_metrics(left_mrc.data, right_mrc.data)
    except (OSError, ValueError) as error:
        raise AuditError(f"failed to memory-map checkpoint volume: {error}") from error
    return {
        "baseline": {"path": str(left), "sha256": sha256_file(left)},
        "candidate": {"path": str(right), "sha256": sha256_file(right)},
        "metrics": metrics,
    }


def run_audit(
    baseline_dir: Path,
    candidate_dir: Path,
    iteration_zero_based: int,
    *,
    max_relative_l2: float,
) -> dict[str, Any]:
    baseline_dir = baseline_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    if not baseline_dir.is_dir() or not candidate_dir.is_dir():
        raise AuditError("baseline and candidate intermediate roots must be directories")
    if iteration_zero_based < 0:
        raise AuditError("iteration must be nonnegative")
    if not math.isfinite(max_relative_l2) or max_relative_l2 < 0.0:
        raise AuditError("maximum relative L2 must be finite and nonnegative")

    prefix = f"it{iteration_zero_based:03d}"
    discrete = {
        suffix: _numeric_file_record(
            *_artifact_pair(baseline_dir, candidate_dir, prefix, suffix)
        )
        for suffix in DISCRETE_NPY_SUFFIXES
    }
    particle_state = {
        suffix: _npz_file_record(
            *_artifact_pair(baseline_dir, candidate_dir, prefix, suffix)
        )
        for suffix in DISCRETE_NPZ_SUFFIXES
    }
    metadata = _object_file_record(
        *_artifact_pair(baseline_dir, candidate_dir, prefix, "meta.npy")
    )
    floating = {
        suffix: _numeric_file_record(
            *_artifact_pair(baseline_dir, candidate_dir, prefix, suffix)
        )
        for suffix in FLOAT_NPY_SUFFIXES
    }
    maps = {
        suffix: _mrc_file_record(
            *_artifact_pair(baseline_dir, candidate_dir, prefix, suffix)
        )
        for suffix in MAP_SUFFIXES
    }

    exact_execution_state = (
        all(record["metrics"]["element_exact"] for record in discrete.values())
        and all(record["all_members_exact"] for record in particle_state.values())
        and metadata["object_exact"]
    )
    floating_metrics = [record["metrics"] for record in (*floating.values(), *maps.values())]
    all_floating_finite = all(record["finite"] for record in floating_metrics)
    maximum_relative_l2 = max(record["relative_l2_difference"] for record in floating_metrics)
    accepted = exact_execution_state and all_floating_finite and maximum_relative_l2 <= max_relative_l2
    return {
        "schema": SCHEMA,
        "baseline_intermediates": str(baseline_dir),
        "candidate_intermediates": str(candidate_dir),
        "iteration_zero_based": iteration_zero_based,
        "thresholds": {"maximum_relative_l2_difference": max_relative_l2},
        "discrete_arrays": discrete,
        "particle_state": particle_state,
        "metadata": metadata,
        "floating_arrays": floating,
        "maps": maps,
        "summary": {
            "exact_execution_state": exact_execution_state,
            "all_floating_finite": all_floating_finite,
            "maximum_relative_l2_difference": maximum_relative_l2,
            "accepted": accepted,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-intermediates", type=Path, required=True)
    parser.add_argument("--candidate-intermediates", type=Path, required=True)
    parser.add_argument("--iteration-zero-based", type=int, required=True)
    parser.add_argument("--max-relative-l2", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run_audit(
            args.baseline_intermediates,
            args.candidate_intermediates,
            args.iteration_zero_based,
            max_relative_l2=args.max_relative_l2,
        )
    except AuditError as error:
        report = {
            "schema": SCHEMA,
            "iteration_zero_based": args.iteration_zero_based,
            "summary": {"accepted": False},
            "failures": [str(error)],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output.resolve()}")
    return 0 if report["summary"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
