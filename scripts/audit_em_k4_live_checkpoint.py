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
import re
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "recovar.em_k4_live_checkpoint.v2"
META_KEYS = (
    "iteration",
    "current_size",
    "n_rotations",
    "n_translations",
    "healpix_order",
    "local_search",
    "sigma_rot",
)
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ITERATION_START_RE = re.compile(
    r"=== RELION Iteration (?P<iteration>\d+)/(?P<total>\d+): "
    r"current_size=(?P<current_size>\d+), healpix_order=(?P<healpix_order>\d+), "
    r"local_search=(?P<local_search>True|False) ==="
)
PMAX_RE = re.compile(
    rf"Class3D optimizer Pmax: value=(?P<value>{FLOAT_PATTERN}) "
    rf"numerator=(?P<numerator>{FLOAT_PATTERN}) "
    rf"half1_mstep_posterior_mass=(?P<mass>{FLOAT_PATTERN}) "
    r"half1_particle_count=(?P<count>\d+)"
)
ITERATION_COMPLETE_RE = re.compile(
    rf"RELION Iteration (?P<iteration>\d+): current_size=(?P<current_size>\d+), "
    rf"pixel_res=(?P<pixel_res>{FLOAT_PATTERN}), res=(?P<resolution>{FLOAT_PATTERN}) A, "
    rf"ave_Pmax=(?P<average_pmax>{FLOAT_PATTERN}), healpix_order=(?P<healpix_order>\d+), "
    rf"converged=(?P<converged>True|False), time=(?P<wall_time>{FLOAT_PATTERN})s"
)
DISPATCH_RE = re.compile(
    r"Strict RELION dynamic dispatch: numbered_iter=(?P<iteration>\d+) "
    r"rank_particle_counts=\[(?P<rank1>\d+), (?P<rank2>\d+)\]"
)
CURRENT_SIZE_RE = re.compile(
    rf"RELION current-size decision: iter=(?P<iteration>\d+) prev=(?P<previous>\d+) "
    rf"res_shell=(?P<resolution_shell>\d+) incr_size=(?P<increment>\d+) "
    rf"high_fsc_at_limit=(?P<high_fsc>True|False) ave_Pmax=(?P<average_pmax>{FLOAT_PATTERN}) "
    rf"raw=(?P<raw>{FLOAT_PATTERN}) quantized=(?P<quantized>\d+)"
)
EXPECTED_ACCURACY_RE = re.compile(
    rf"RELION exact expected accuracy: acc_rot=(?P<rotation>{FLOAT_PATTERN}) deg, "
    rf"acc_trans=(?P<translation>{FLOAT_PATTERN}) A \(trials=(?P<trials>\d+), "
    r"first_particle_ids=\[(?P<particle_ids>[^\]]*)\]\)"
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


def _unique_match(
    pattern: re.Pattern[str],
    text: str,
    *,
    label: str,
    path: Path,
) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise AuditError(f"expected one {label} record in {path}, found {len(matches)}")
    return matches[0]


def _as_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise AuditError(f"unexpected boolean token: {value}")
    return value == "True"


def _load_log_iteration(path: Path, *, iteration: int) -> dict[str, Any]:
    try:
        text = _require_file(path).read_text()
    except (OSError, UnicodeError) as exc:
        raise AuditError(f"failed to read refinement log {path}: {exc}") from exc

    starts = list(ITERATION_START_RE.finditer(text))
    requested = [match for match in starts if int(match.group("iteration")) == iteration]
    if len(requested) != 1:
        raise AuditError(f"expected one iteration-{iteration} start record in {path}, found {len(requested)}")
    start = requested[0]
    following_starts = [match for match in starts if match.start() > start.start()]
    end = following_starts[0].start() if following_starts else len(text)
    segment = text[start.start() : end]

    total_iterations = int(start.group("total"))
    if not 1 <= iteration <= total_iterations:
        raise AuditError(f"invalid iteration range in {path}: iteration={iteration}, total={total_iterations}")
    pmax = _unique_match(PMAX_RE, segment, label="optimizer Pmax", path=path)
    completed = _unique_match(
        ITERATION_COMPLETE_RE,
        segment,
        label="iteration completion",
        path=path,
    )
    if int(completed.group("iteration")) != iteration:
        raise AuditError(f"iteration completion does not match iteration {iteration}: {path}")

    optimizer_pmax = float(pmax.group("value"))
    completion_average_pmax = float(completed.group("average_pmax"))
    values_to_check = (
        optimizer_pmax,
        float(pmax.group("numerator")),
        float(pmax.group("mass")),
        float(completed.group("pixel_res")),
        float(completed.group("resolution")),
        completion_average_pmax,
        float(completed.group("wall_time")),
    )
    if not np.isfinite(values_to_check).all():
        raise AuditError(f"non-finite controller telemetry in {path} for iteration {iteration}")
    if abs(optimizer_pmax - completion_average_pmax) > 5.1e-5:
        raise AuditError(f"optimizer and displayed Pmax disagree in {path} for iteration {iteration}")

    start_record = {
        "iteration": iteration,
        "total_iterations": total_iterations,
        "current_size": int(start.group("current_size")),
        "healpix_order": int(start.group("healpix_order")),
        "local_search": _as_bool(start.group("local_search")),
    }
    completion_record = {
        "current_size": int(completed.group("current_size")),
        "pixel_resolution_shell": float(completed.group("pixel_res")),
        "resolution_angstrom": float(completed.group("resolution")),
        "average_pmax_display": completion_average_pmax,
        "healpix_order": int(completed.group("healpix_order")),
        "converged": _as_bool(completed.group("converged")),
        "wall_time_s_display": float(completed.group("wall_time")),
    }
    if (
        start_record["current_size"] != completion_record["current_size"]
        or start_record["healpix_order"] != completion_record["healpix_order"]
    ):
        raise AuditError(f"iteration start/completion topology disagrees in {path}")

    next_record = None
    if iteration < total_iterations:
        dispatch = _unique_match(DISPATCH_RE, segment, label="next dispatch", path=path)
        size = _unique_match(CURRENT_SIZE_RE, segment, label="next current-size decision", path=path)
        accuracy = _unique_match(
            EXPECTED_ACCURACY_RE,
            segment,
            label="next expected accuracy",
            path=path,
        )
        next_iteration = iteration + 1
        if int(dispatch.group("iteration")) != next_iteration or int(size.group("iteration")) != next_iteration:
            raise AuditError(f"next controller records do not target iteration {next_iteration}: {path}")
        if int(size.group("previous")) != completion_record["current_size"]:
            raise AuditError(f"next current-size decision has the wrong previous size: {path}")
        size_average_pmax = float(size.group("average_pmax"))
        expected_rotation = float(accuracy.group("rotation"))
        expected_translation = float(accuracy.group("translation"))
        if not np.isfinite((size_average_pmax, expected_rotation, expected_translation)).all():
            raise AuditError(f"non-finite next-controller telemetry in {path}")
        if abs(optimizer_pmax - size_average_pmax) > 5.1e-7:
            raise AuditError(f"optimizer and controller Pmax disagree in {path}")
        particle_ids_text = accuracy.group("particle_ids").strip()
        particle_ids = [] if not particle_ids_text else [int(value.strip()) for value in particle_ids_text.split(",")]
        next_record = {
            "iteration": next_iteration,
            "dispatch_counts": [int(dispatch.group("rank1")), int(dispatch.group("rank2"))],
            "previous_size": int(size.group("previous")),
            "resolution_shell": int(size.group("resolution_shell")),
            "size_increment": int(size.group("increment")),
            "high_fsc_at_limit": _as_bool(size.group("high_fsc")),
            "average_pmax_display": size_average_pmax,
            "raw_size": float(size.group("raw")),
            "quantized_size": int(size.group("quantized")),
            "expected_rotation_accuracy_deg": expected_rotation,
            "expected_translation_accuracy_angstrom": expected_translation,
            "trial_count": int(accuracy.group("trials")),
            "first_particle_ids": particle_ids,
        }

    return {
        "segment_sha256": hashlib.sha256(segment.encode()).hexdigest(),
        "start": start_record,
        "optimizer": {
            "average_pmax": optimizer_pmax,
            "numerator": float(pmax.group("numerator")),
            "posterior_mass": float(pmax.group("mass")),
            "particle_count": int(pmax.group("count")),
        },
        "completion": completion_record,
        "next": next_record,
    }


def _controller_topology(log: dict[str, Any]) -> dict[str, Any]:
    next_record = log["next"]
    return {
        "start": {
            key: log["start"][key]
            for key in ("iteration", "total_iterations", "current_size", "healpix_order", "local_search")
        },
        "completion": {key: log["completion"][key] for key in ("current_size", "healpix_order", "converged")},
        "next": (
            None
            if next_record is None
            else {
                key: next_record[key]
                for key in (
                    "iteration",
                    "dispatch_counts",
                    "previous_size",
                    "resolution_shell",
                    "size_increment",
                    "high_fsc_at_limit",
                    "quantized_size",
                    "trial_count",
                    "first_particle_ids",
                )
            }
        ),
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

    live_log_path = live_control / "run_full_refinement.stderr"
    sealed_log_path = sealed_control / "run_full_refinement.stderr"
    live_log = _load_log_iteration(live_log_path, iteration=iteration)
    sealed_log = _load_log_iteration(sealed_log_path, iteration=iteration)

    for label, metadata, log_record, timing in (
        ("live", live_meta, live_log, live_timing),
        ("sealed", sealed_meta, sealed_log, sealed_timing),
    ):
        start = log_record["start"]
        completion = log_record["completion"]
        if (
            int(metadata["current_size"]) != start["current_size"]
            or int(metadata["healpix_order"]) != start["healpix_order"]
            or bool(metadata["local_search"]) is not start["local_search"]
        ):
            raise AuditError(f"{label} metadata and log topology disagree")
        if round(float(timing["wall_time_s"]), 1) != completion["wall_time_s_display"]:
            raise AuditError(f"{label} timing artifact and displayed wall time disagree")

    live_controller_topology = _controller_topology(live_log)
    sealed_controller_topology = _controller_topology(sealed_log)
    controller_topology_exact = live_controller_topology == sealed_controller_topology
    persisted_topology_exact = live_meta == sealed_meta and all(exact_arrays.values())
    topology_exact = persisted_topology_exact and controller_topology_exact
    live_next = live_log["next"]
    sealed_next = sealed_log["next"]
    expected_accuracy_exact = (
        live_next is None
        and sealed_next is None
        or live_next is not None
        and sealed_next is not None
        and live_next["expected_rotation_accuracy_deg"] == sealed_next["expected_rotation_accuracy_deg"]
        and live_next["expected_translation_accuracy_angstrom"] == sealed_next["expected_translation_accuracy_angstrom"]
    )
    artifact_hashes = {str(path.relative_to(live_control)): _sha256(path) for path in artifact_paths}
    artifact_hashes[f"run_full_refinement.stderr#iteration={iteration}"] = live_log["segment_sha256"]
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
            "persisted_exact": persisted_topology_exact,
            "controller_exact": controller_topology_exact,
            "metadata_exact": live_meta == sealed_meta,
            "live_metadata": live_meta,
            "sealed_metadata": sealed_meta,
            "arrays_exact": exact_arrays,
            "live_controller": live_controller_topology,
            "sealed_controller": sealed_controller_topology,
        },
        "controller_state": {
            "optimizer_pmax": {
                "live": live_log["optimizer"]["average_pmax"],
                "sealed": sealed_log["optimizer"]["average_pmax"],
                "live_minus_sealed": float(
                    live_log["optimizer"]["average_pmax"] - sealed_log["optimizer"]["average_pmax"]
                ),
            },
            "optimizer_operands": {
                "live": live_log["optimizer"],
                "sealed": sealed_log["optimizer"],
            },
            "completion": {
                "live": live_log["completion"],
                "sealed": sealed_log["completion"],
            },
            "next": {
                "live": live_next,
                "sealed": sealed_next,
                "expected_accuracy_exact": expected_accuracy_exact,
            },
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
            "live_minus_sealed_fraction": float(live_timing["wall_time_s"] / sealed_timing["wall_time_s"] - 1.0),
        },
        "live_artifact_sha256": artifact_hashes,
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
