#!/usr/bin/env python3
"""Compare one live K=4 checkpoint with a sealed RECOVAR control.

This diagnostic covers persisted execution topology, particle-aligned hard
assignments, noise, and wall time.  It does not compare maps: K=4 map quality
remains gated by shellwise FSC/FSC-AUC in ``audit_k4_fsc_trajectory.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "recovar.em_k4_live_checkpoint.v1"
META_KEYS = (
    "iteration",
    "current_size",
    "n_rotations",
    "n_translations",
    "healpix_order",
    "local_search",
    "sigma_rot",
)


class AuditError(RuntimeError):
    """Raised when a required checkpoint artifact is missing or malformed."""


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise AuditError(f"missing checkpoint artifact: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_meta(path: Path) -> dict[str, Any]:
    try:
        value = np.load(_require_file(path), allow_pickle=True)
    except (OSError, ValueError, EOFError) as exc:
        raise AuditError(f"failed to load checkpoint metadata {path}: {exc}") from exc
    if value.shape != () or value.dtype != np.dtype(object):
        raise AuditError(f"checkpoint metadata must be one object scalar: {path}")
    payload = value.item()
    if not isinstance(payload, dict) or tuple(payload) != META_KEYS:
        raise AuditError(f"checkpoint metadata has unexpected keys: {path}: {tuple(payload)}")
    for key in META_KEYS:
        scalar = payload[key]
        if isinstance(scalar, (float, np.floating)) and not np.isfinite(float(scalar)):
            raise AuditError(f"checkpoint metadata {key} is non-finite: {path}")
    return payload


def _load_array(path: Path, *, integer: bool = False) -> np.ndarray:
    try:
        value = np.asarray(np.load(_require_file(path), allow_pickle=False))
    except (OSError, ValueError, EOFError) as exc:
        raise AuditError(f"failed to load checkpoint array {path}: {exc}") from exc
    if value.size == 0 or value.dtype == np.dtype(object):
        raise AuditError(f"checkpoint array must be nonempty and numeric: {path}")
    if integer and (value.ndim != 1 or not np.issubdtype(value.dtype, np.integer)):
        raise AuditError(f"hard assignments must be one-dimensional integers: {path}")
    if not integer:
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise AuditError(f"checkpoint array must contain finite numeric values: {path}")
    return value


def _load_timing(path: Path, *, iteration: int) -> dict[str, float | int]:
    try:
        with np.load(_require_file(path), allow_pickle=False) as payload:
            required = {"iteration", "relion_iteration", "wall_time_s"}
            if not required.issubset(payload.files):
                raise AuditError(f"timing artifact lacks {sorted(required - set(payload.files))}: {path}")
            zero_based = int(np.asarray(payload["iteration"]).reshape(()))
            relion_iteration = int(np.asarray(payload["relion_iteration"]).reshape(()))
            wall_time_s = float(np.asarray(payload["wall_time_s"]).reshape(()))
    except (OSError, ValueError, EOFError) as exc:
        raise AuditError(f"failed to load checkpoint timing {path}: {exc}") from exc
    if zero_based != iteration - 1 or relion_iteration != iteration:
        raise AuditError(
            f"timing iteration mismatch in {path}: iteration={zero_based}, "
            f"relion_iteration={relion_iteration}, requested={iteration}"
        )
    if not np.isfinite(wall_time_s) or wall_time_s <= 0.0:
        raise AuditError(f"wall_time_s must be finite and positive: {path}")
    return {
        "iteration": zero_based,
        "relion_iteration": relion_iteration,
        "wall_time_s": wall_time_s,
    }


def _paired_array(live: Path, sealed: Path, *, integer: bool = False) -> tuple[np.ndarray, np.ndarray]:
    lhs = _load_array(live, integer=integer)
    rhs = _load_array(sealed, integer=integer)
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise AuditError(
            f"checkpoint arrays differ in shape or dtype: {live}={lhs.shape}/{lhs.dtype}, "
            f"{sealed}={rhs.shape}/{rhs.dtype}"
        )
    return lhs, rhs


def _hard_assignment_comparison(live: Path, sealed: Path) -> tuple[dict[str, Any], np.ndarray]:
    lhs, rhs = _paired_array(live, sealed, integer=True)
    mismatch = lhs != rhs
    count = int(np.count_nonzero(mismatch))
    return {
        "particle_count": int(lhs.size),
        "mismatch_count": count,
        "mismatch_fraction": float(count / lhs.size),
    }, mismatch


def _mismatch_dynamics(previous: np.ndarray, current: np.ndarray) -> dict[str, int]:
    if previous.shape != current.shape:
        raise AuditError(f"hard-assignment mismatch masks changed shape: {previous.shape} versus {current.shape}")
    return {
        "previous": int(np.count_nonzero(previous)),
        "current": int(np.count_nonzero(current)),
        "persistent": int(np.count_nonzero(previous & current)),
        "new": int(np.count_nonzero(~previous & current)),
        "resolved": int(np.count_nonzero(previous & ~current)),
    }


def audit(*, live_control: Path, sealed_control: Path, iteration: int) -> dict[str, Any]:
    """Audit one completed one-based RELION iteration checkpoint."""
    if iteration < 1:
        raise AuditError(f"iteration must be positive, got {iteration}")
    live_control = live_control.expanduser().resolve()
    sealed_control = sealed_control.expanduser().resolve()
    for root in (live_control, sealed_control):
        if not root.is_dir():
            raise AuditError(f"missing control directory: {root}")

    artifact_iteration = iteration - 1
    tag = f"it{artifact_iteration:03d}"
    timing_tag = f"iter_{iteration:03d}.npz"
    live_intermediates = live_control / "intermediates"
    sealed_intermediates = sealed_control / "intermediates"

    live_meta_path = live_intermediates / f"{tag}_meta.npy"
    sealed_meta_path = sealed_intermediates / f"{tag}_meta.npy"
    live_meta = _load_meta(live_meta_path)
    sealed_meta = _load_meta(sealed_meta_path)
    if int(live_meta["iteration"]) != artifact_iteration or int(sealed_meta["iteration"]) != artifact_iteration:
        raise AuditError(f"metadata iteration does not match requested iteration {iteration}")

    hard: dict[str, Any] = {}
    for label, suffix in (("coarse", "coarse_ha_half1"), ("final", "ha_half1")):
        comparison, mismatch = _hard_assignment_comparison(
            live_intermediates / f"{tag}_{suffix}.npy",
            sealed_intermediates / f"{tag}_{suffix}.npy",
        )
        hard[label] = comparison
        if iteration > 1:
            previous_tag = f"it{artifact_iteration - 1:03d}"
            _, previous = _hard_assignment_comparison(
                live_intermediates / f"{previous_tag}_{suffix}.npy",
                sealed_intermediates / f"{previous_tag}_{suffix}.npy",
            )
            hard[label]["dynamics_from_previous"] = _mismatch_dynamics(previous, mismatch)

    exact_arrays: dict[str, bool] = {}
    artifact_paths = [live_meta_path]
    for label, suffix in (("rotations", "rotations"), ("translations", "translations")):
        live_path = live_intermediates / f"{tag}_{suffix}.npy"
        sealed_path = sealed_intermediates / f"{tag}_{suffix}.npy"
        lhs, rhs = _paired_array(live_path, sealed_path)
        exact_arrays[label] = bool(np.array_equal(lhs, rhs))
        artifact_paths.append(live_path)

    live_noise_path = live_intermediates / f"{tag}_noise.npy"
    live_noise, sealed_noise = _paired_array(
        live_noise_path,
        sealed_intermediates / f"{tag}_noise.npy",
    )
    noise_delta = live_noise.astype(np.float64) - sealed_noise.astype(np.float64)
    artifact_paths.extend(
        [
            live_intermediates / f"{tag}_coarse_ha_half1.npy",
            live_intermediates / f"{tag}_ha_half1.npy",
            live_noise_path,
        ]
    )

    live_timing_path = live_control / "timing" / timing_tag
    sealed_timing_path = sealed_control / "timing" / timing_tag
    live_timing = _load_timing(live_timing_path, iteration=iteration)
    sealed_timing = _load_timing(sealed_timing_path, iteration=iteration)
    artifact_paths.append(live_timing_path)

    topology_exact = live_meta == sealed_meta and all(exact_arrays.values())
    return {
        "schema": SCHEMA,
        "status": "pass" if topology_exact else "fail",
        "scope": "persisted checkpoint topology and state diagnostics; map quality not evaluated",
        "quality_metric_policy": (
            "maps require shellwise FSC/FSC-AUC; correlation is not computed and this report is not a map gate"
        ),
        "iteration": iteration,
        "artifact_iteration": artifact_iteration,
        "paths": {
            "live_control": str(live_control),
            "sealed_control": str(sealed_control),
        },
        "topology": {
            "exact": topology_exact,
            "metadata_exact": live_meta == sealed_meta,
            "live_metadata": live_meta,
            "sealed_metadata": sealed_meta,
            "arrays_exact": exact_arrays,
        },
        "hard_assignments": hard,
        "noise": {
            "shape": list(live_noise.shape),
            "dtype": str(live_noise.dtype),
            "byte_exact": bool(np.array_equal(live_noise, sealed_noise)),
            "max_abs_delta": float(np.max(np.abs(noise_delta))),
            "l2_delta": float(np.linalg.norm(noise_delta)),
        },
        "timing": {
            "live_wall_time_s": live_timing["wall_time_s"],
            "sealed_wall_time_s": sealed_timing["wall_time_s"],
            "live_minus_sealed_fraction": float(
                live_timing["wall_time_s"] / sealed_timing["wall_time_s"] - 1.0
            ),
        },
        "live_artifact_sha256": {str(path.relative_to(live_control)): _sha256(path) for path in artifact_paths},
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-control", type=Path, required=True)
    parser.add_argument("--sealed-control", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = audit(
            live_control=args.live_control,
            sealed_control=args.sealed_control,
            iteration=args.iteration,
        )
    except AuditError as exc:
        report = {
            "schema": SCHEMA,
            "status": "error",
            "scope": "persisted checkpoint topology and state diagnostics; map quality not evaluated",
            "quality_metric_policy": "maps require shellwise FSC/FSC-AUC; correlation is not computed",
            "failures": [str(exc)],
        }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
