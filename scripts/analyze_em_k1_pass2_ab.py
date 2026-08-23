#!/usr/bin/env python3
"""Compare two source-ID-aligned K=1 pass-2 intermediary panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

SCHEMA = "recovar.em_k1_pass2_ab.v1"
FILENAME = re.compile(r"pass2_orig(?P<particle>[0-9]{6})_cs(?P<size>[0-9]{3})[.]npz")
GROUP_FIELDS = {
    # ``local_index`` is an execution coordinate, not an immutable identity.
    # It is expected to differ when a replay boundary preserves source order
    # while a fresh run uses RELION physical order.
    "identity": ("original_index", "current_size", "n_fine_trans"),
    "candidate_geometry": (
        "fine_translations",
        "rotations",
        "oversampled_rot_indices",
        "parent_map",
        "candidate_mask",
        "window_indices",
        "recon_window_indices",
    ),
    "score_operands": (
        "shifted_corrected",
        "ctf2_over_nv_score",
        "proj_half",
        "half_weights",
    ),
    "raw_score": ("scores_pre_prior",),
    "priors": ("rotation_log_prior", "translation_log_prior"),
    "unnormalized_score": ("scores_with_prior",),
    "posterior": ("probs",),
    "reconstruction_operands": ("shifted_recon", "ctf2_over_nv_recon"),
    "significant_support": (
        "reconstruction_mask",
        "reconstruction_probs",
        "reconstruction_n_significant",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover(directory: Path, expected_count: int, current_size: int) -> dict[int, Path]:
    _require(directory.is_dir(), f"pass-2 directory does not exist: {directory}")
    result = {}
    for path in sorted(directory.glob("*.npz")):
        match = FILENAME.fullmatch(path.name)
        _require(match is not None, f"unexpected pass-2 filename: {path.name}")
        particle = int(match.group("particle"))
        _require(int(match.group("size")) == current_size, f"current size changed in {path.name}")
        _require(particle not in result, f"duplicate particle {particle}")
        result[particle] = path
    _require(len(result) == expected_count, f"expected {expected_count} particles; found {len(result)}")
    return result


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def array_metric(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left)
    right = np.asarray(right)
    shape_equal = left.shape == right.shape
    dtype_equal = left.dtype == right.dtype
    result: dict[str, object] = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "byte_exact": False,
        "correlation_used": False,
    }
    if not shape_equal or not dtype_equal:
        return result
    byte_exact = bool(np.array_equal(left, right, equal_nan=True))
    result["byte_exact"] = byte_exact
    if np.issubdtype(left.dtype, np.number):
        finite_left = np.isfinite(left)
        finite_right = np.isfinite(right)
        same_finite = bool(np.array_equal(finite_left, finite_right))
        result["same_finite_mask"] = same_finite
        if same_finite:
            metric_dtype = np.complex128 if np.iscomplexobj(left) else np.float64
            left64 = left[finite_left].astype(metric_dtype)
            right64 = right[finite_right].astype(metric_dtype)
            delta = right64 - left64
            delta_l2 = float(np.linalg.norm(delta))
            denominator = float(np.linalg.norm(left64))
            result.update(
                {
                    "finite_pair_count": int(delta.size),
                    "nonzero_finite_delta_count": int(np.count_nonzero(delta)),
                    "max_abs_finite_delta": float(np.max(np.abs(delta), initial=0.0)),
                    "finite_delta_l2": delta_l2,
                    "relative_l2_to_left": (
                        delta_l2 / denominator if denominator > 0.0 else (0.0 if delta_l2 == 0.0 else math.inf)
                    ),
                }
            )
    return result


def analyze(left_dir: Path, right_dir: Path, expected_count: int, current_size: int) -> dict[str, object]:
    left_paths = _discover(left_dir, expected_count, current_size)
    right_paths = _discover(right_dir, expected_count, current_size)
    _require(set(left_paths) == set(right_paths), "particle panels differ")
    required_fields = {field for fields in GROUP_FIELDS.values() for field in fields}
    group_exact = {group: True for group in GROUP_FIELDS}
    particles = []
    for particle in sorted(left_paths):
        left_path = left_paths[particle]
        right_path = right_paths[particle]
        left = _load(left_path)
        right = _load(right_path)
        _require(required_fields.issubset(left), f"particle {particle} is missing required fields")
        _require(required_fields.issubset(right), f"particle {particle} is missing required fields")
        groups = {}
        for group, fields in GROUP_FIELDS.items():
            metrics = {field: array_metric(left[field], right[field]) for field in fields}
            exact = all(bool(metric["byte_exact"]) for metric in metrics.values())
            group_exact[group] &= exact
            groups[group] = {"exact": exact, "fields": metrics}
        particles.append(
            {
                "original_index": particle,
                "left_path": str(left_path.resolve()),
                "right_path": str(right_path.resolve()),
                "left_sha256": _sha256(left_path),
                "right_sha256": _sha256(right_path),
                "left_pmax": float(np.max(left["probs"], initial=0.0)),
                "right_pmax": float(np.max(right["probs"], initial=0.0)),
                "left_probability_mass": float(np.sum(left["probs"], dtype=np.float64)),
                "right_probability_mass": float(np.sum(right["probs"], dtype=np.float64)),
                "left_local_index": int(left["local_index"]),
                "right_local_index": int(right["local_index"]),
                "left_only_optional_fields": sorted(set(left) - set(right)),
                "right_only_optional_fields": sorted(set(right) - set(left)),
                "groups": groups,
            }
        )
    first_unequal = next((group for group, exact in group_exact.items() if not exact), "all_fields_exact")
    return {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "direct byte and relative-L2 comparisons; no correlation",
        "classification": f"first_unequal_group__{first_unequal}",
        "first_unequal_group": first_unequal,
        "fixed_metric": {
            "passing": sum(group_exact.values()),
            "evaluated": len(group_exact),
            "groups": group_exact,
        },
        "left_directory": str(left_dir.resolve()),
        "right_directory": str(right_dir.resolve()),
        "particles": particles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--expected-particles", type=int, required=True)
    parser.add_argument("--current-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.left_dir, args.right_dir, args.expected_particles, args.current_size)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["fixed_metric"], sort_keys=True))
    print(f"first_unequal_group={report['first_unequal_group']}")


if __name__ == "__main__":
    main()
