#!/usr/bin/env python3
"""Generic two-tier audit for a paired default-versus-threshold K-class run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
from scipy import fft as scipy_fft
from scipy.spatial.transform import Rotation as SciPyRotation

ITERATION_TIME_RE = re.compile(r"RELION Iteration (?P<iteration>\d+):.*time=(?P<seconds>[0-9.]+)s")
GROUP_WALL_RE = re.compile(r"Sparse fused K-class pass-2 bucket group done:.*\bwall=(?P<seconds>[0-9.]+)s")
CONTROLLER_KEYS = ("current_sizes", "pixel_resolutions", "healpix_order_trajectory")


def default_fft_workers() -> int:
    """Use the allocated CPU count without oversubscribing login or Slurm nodes."""

    raw = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def maximum_absolute_delta(lhs: np.ndarray, rhs: np.ndarray) -> float:
    require(lhs.shape == rhs.shape, f"array shapes differ: {lhs.shape} != {rhs.shape}")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))


def relion_eulers_to_rotation_matrices(eulers_deg: np.ndarray) -> np.ndarray:
    """Convert RELION Euler rows to physical rotation matrices in NumPy."""

    angles = np.asarray(eulers_deg, dtype=np.float64).copy()
    require(angles.ndim == 2 and angles.shape[1] == 3, f"invalid RELION Euler shape: {angles.shape}")
    angles[:, 0] += 90.0
    angles[:, 2] -= 90.0
    matrices = SciPyRotation.from_euler("zxz", angles, degrees=True).as_matrix()
    frame_adjustment = np.asarray(
        [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]],
        dtype=np.float64,
    )
    return matrices * frame_adjustment


def maximum_physical_rotation_delta_deg(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return the largest geodesic SO(3) distance between RELION Euler rows."""

    require(lhs.shape == rhs.shape, f"Euler shapes differ: {lhs.shape} != {rhs.shape}")
    if lhs.size == 0 or np.array_equal(lhs, rhs):
        return 0.0
    left = relion_eulers_to_rotation_matrices(lhs)
    right = relion_eulers_to_rotation_matrices(rhs)
    relative = left @ np.swapaxes(right, 1, 2)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.max(np.rad2deg(np.arccos(cosine))))


def compare_saved_state_arrays(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    n_iterations: int,
) -> dict[str, Any]:
    """Pure array comparison used by both the CLI and synthetic self-test."""

    controller_exact = {
        key: bool(np.array_equal(np.asarray(baseline[key]), np.asarray(candidate[key]))) for key in CONTROLLER_KEYS
    }
    rows: dict[str, list[dict[str, Any]]] = {
        "assignments": [],
        "significant_counts": [],
        "pmax": [],
        "eulers": [],
        "translations": [],
    }
    for iteration in range(n_iterations):
        suffix = f"{iteration:03d}"
        pairs = {
            "assignments": f"class_assignments_by_image_iter_{suffix}",
            "significant_counts": f"sig_counts_by_image_iter_{suffix}",
            "pmax": f"pmax_per_image_by_image_iter_{suffix}",
            "eulers": f"best_rotation_eulers_by_image_iter_{suffix}",
            "translations": f"best_translations_by_image_iter_{suffix}",
        }
        for group, key in pairs.items():
            left = np.asarray(baseline[key])
            right = np.asarray(candidate[key])
            require(left.shape == right.shape, f"{key} shapes differ: {left.shape} != {right.shape}")
            row: dict[str, Any] = {
                "iteration": iteration + 1,
                "array_exact": bool(np.array_equal(left, right)),
                "max_absolute_delta": maximum_absolute_delta(left, right),
            }
            if group in {"assignments", "significant_counts"}:
                row["mismatch_count"] = int(np.count_nonzero(left != right))
                row["row_count"] = int(left.size)
            elif group == "eulers":
                row["mismatch_row_count"] = int(np.count_nonzero(np.any(left != right, axis=1)))
                row["max_physical_rotation_delta_deg"] = maximum_physical_rotation_delta_deg(left, right)
            rows[group].append(row)
    return {
        "controller_exact": controller_exact,
        "trajectory_pmax_max_absolute_delta": maximum_absolute_delta(
            np.asarray(baseline["ave_Pmax_trajectory"]),
            np.asarray(candidate["ave_Pmax_trajectory"]),
        ),
        **rows,
    }


def class_occupancy_rows(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    n_iterations: int,
    n_classes: int,
) -> list[dict[str, Any]]:
    """Pure no-collapse check with per-iteration, per-arm class counts."""

    expected_labels = list(range(n_classes))
    rows: list[dict[str, Any]] = []
    for iteration in range(n_iterations):
        key = f"class_assignments_by_image_iter_{iteration:03d}"
        arm_counts: dict[str, dict[str, int]] = {}
        arm_complete: dict[str, bool] = {}
        for label, arrays in (("baseline", baseline), ("candidate", candidate)):
            unique, counts = np.unique(np.asarray(arrays[key]), return_counts=True)
            arm_counts[label] = {
                str(int(class_label)): int(count) for class_label, count in zip(unique, counts, strict=True)
            }
            arm_complete[label] = sorted(int(value) for value in unique) == expected_labels
        rows.append(
            {
                "iteration": iteration + 1,
                "baseline_count_by_class": arm_counts["baseline"],
                "candidate_count_by_class": arm_counts["candidate"],
                "baseline_all_classes_occupied": arm_complete["baseline"],
                "candidate_all_classes_occupied": arm_complete["candidate"],
            }
        )
    return rows


def compare_npz_state(
    baseline_path: Path,
    candidate_path: Path,
    *,
    n_iterations: int,
    n_classes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with (
        np.load(baseline_path, allow_pickle=False) as baseline,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        state = compare_saved_state_arrays(baseline, candidate, n_iterations=n_iterations)
        occupancy = class_occupancy_rows(
            baseline,
            candidate,
            n_iterations=n_iterations,
            n_classes=n_classes,
        )
    return state, occupancy


class ShellFscCalculator:
    """Memory-bounded signed FSC using a Hermitian half spectrum."""

    def __init__(self, box_size: int, *, workers: int = 1):
        self.box_size = int(box_size)
        self.workers = max(1, int(workers))
        full = (np.fft.fftfreq(box_size) * box_size).astype(np.float32)
        half = (np.fft.rfftfreq(box_size) * box_size).astype(np.float32)
        first_two = full[:, None] ** 2 + full[None, :] ** 2
        shells = np.empty((box_size, box_size, half.size), dtype=np.int16)
        for index in range(box_size):
            shells[index] = np.rint(np.sqrt(first_two[index, :, None] + half[None, :] ** 2)).astype(np.int16)
        self.shells = shells.reshape(-1)
        self.spectrum_shape = shells.shape
        self.last_weighted_plane = -1 if box_size % 2 == 0 else None

    def weighted_sum(self, values: np.ndarray) -> np.ndarray:
        require(values.shape == self.spectrum_shape, "unexpected spectrum shape")
        values[:, :, 1 : self.last_weighted_plane] *= np.float32(2.0)
        return np.bincount(self.shells, weights=values.reshape(-1))

    def curve(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        left = scipy_fft.rfftn(np.asarray(lhs, dtype=np.float32), workers=self.workers)
        right = scipy_fft.rfftn(np.asarray(rhs, dtype=np.float32), workers=self.workers)
        numerator = self.weighted_sum(np.real(left * np.conj(right)))
        left_power = np.abs(left)
        np.square(left_power, out=left_power)
        right_power = np.abs(right)
        np.square(right_power, out=right_power)
        denominator = np.sqrt(self.weighted_sum(left_power) * self.weighted_sum(right_power))
        curve = np.full(numerator.shape, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=curve, where=denominator > 0.0)
        return curve[: self.box_size // 2 - 1]


def load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        volume = np.array(handle.data, dtype=np.float32, copy=True)
    require(
        volume.ndim == 3 and volume.shape[0] == volume.shape[1] == volume.shape[2],
        f"invalid map shape in {path}: {volume.shape}",
    )
    require(np.all(np.isfinite(volume)), f"non-finite map {path}")
    return volume


def relative_l2(lhs: np.ndarray, rhs: np.ndarray, block_size: int = 4_000_000) -> float:
    left = lhs.reshape(-1)
    right = rhs.reshape(-1)
    numerator = 0.0
    denominator = 0.0
    for start in range(0, left.size, block_size):
        stop = min(left.size, start + block_size)
        left_block = left[start:stop].astype(np.float64)
        right_block = right[start:stop].astype(np.float64)
        delta = left_block - right_block
        numerator += float(np.dot(delta, delta))
        denominator += float(np.dot(left_block, left_block))
    require(denominator > 0.0, "baseline map has zero norm")
    return math.sqrt(numerator / denominator)


def compare_maps(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    n_classes: int,
    fft_workers: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    calculator: ShellFscCalculator | None = None
    for class_index in range(1, n_classes + 1):
        filename = f"final_class{class_index:03d}.mrc"
        baseline_path = baseline_dir / filename
        candidate_path = candidate_dir / filename
        baseline = load_mrc(baseline_path)
        candidate = load_mrc(candidate_path)
        require(candidate.shape == baseline.shape, f"map shapes differ for {filename}")
        if calculator is None:
            calculator = ShellFscCalculator(int(baseline.shape[0]), workers=fft_workers)
        curve = calculator.curve(baseline, candidate)
        non_dc = curve[1:]
        rows.append(
            {
                "class": class_index,
                "baseline_sha256": sha256(baseline_path),
                "candidate_sha256": sha256(candidate_path),
                "relative_l2": relative_l2(baseline, candidate),
                "signed_non_dc_fsc_auc": float(np.nanmean(non_dc)),
                "signed_non_dc_fsc_min": float(np.nanmin(non_dc)),
                "fsc_shell_count_non_dc": int(np.count_nonzero(np.isfinite(non_dc))),
            }
        )
    return rows


def parse_log(path: Path, *, n_iterations: int) -> dict[str, Any]:
    iteration_times: dict[str, float] = {}
    group_walls: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        if match := ITERATION_TIME_RE.search(line):
            iteration_times[match.group("iteration")] = float(match.group("seconds"))
        if match := GROUP_WALL_RE.search(line):
            group_walls.append(float(match.group("seconds")))
    require(
        len(iteration_times) == n_iterations,
        f"expected {n_iterations} completed iterations in {path}, got {len(iteration_times)}",
    )
    require(group_walls, f"no sparse K-class group timings in {path}")
    return {
        "iteration_times_s": iteration_times,
        "iteration_time_sum_s": float(sum(iteration_times.values())),
        "sparse_group_count": len(group_walls),
        "sparse_group_wall_sum_s": float(sum(group_walls)),
    }


def peak_hbm_mib(path: Path) -> float:
    values: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                values.append(float(row["memory_used_mib"]))
            except (KeyError, TypeError, ValueError):
                continue
    require(values, f"no valid HBM samples in {path}")
    return max(values)


def summarize_pair(
    state: dict[str, Any],
    occupancy: list[dict[str, Any]],
    maps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pure reduction from direct comparison rows to decision metrics."""

    return {
        "controller_exact": bool(all(state["controller_exact"].values())),
        "assignment_mismatch_count": sum(int(row["mismatch_count"]) for row in state["assignments"]),
        "significant_count_mismatch_count": sum(int(row["mismatch_count"]) for row in state["significant_counts"]),
        "max_pmax_absolute_delta": max(float(row["max_absolute_delta"]) for row in state["pmax"]),
        "max_euler_absolute_delta": max(float(row["max_absolute_delta"]) for row in state["eulers"]),
        "euler_mismatch_row_count": sum(int(row["mismatch_row_count"]) for row in state["eulers"]),
        "max_physical_rotation_delta_deg": max(
            float(row["max_physical_rotation_delta_deg"]) for row in state["eulers"]
        ),
        "max_translation_absolute_delta": max(float(row["max_absolute_delta"]) for row in state["translations"]),
        "poses_exact": bool(
            all(row["array_exact"] for row in state["eulers"])
            and all(row["array_exact"] for row in state["translations"])
        ),
        "no_class_collapse": bool(
            all(row["baseline_all_classes_occupied"] and row["candidate_all_classes_occupied"] for row in occupancy)
        ),
        "min_final_map_signed_non_dc_fsc_auc": min(float(row["signed_non_dc_fsc_auc"]) for row in maps),
        "max_final_map_relative_l2": max(float(row["relative_l2"]) for row in maps),
    }


def evaluate_tiers(
    summary: Mapping[str, Any],
    performance: Mapping[str, Any],
    formal_thresholds: Mapping[str, Any],
    science_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure, independent formal and scientific-equivalence decisions."""

    common_formal = {
        "controller_exact": bool(summary["controller_exact"]),
        "assignments_exact": int(summary["assignment_mismatch_count"]) == 0,
        "map_fsc": float(summary["min_final_map_signed_non_dc_fsc_auc"])
        >= float(formal_thresholds["final_map_min_signed_non_dc_fsc_auc"]),
        "map_l2": float(summary["max_final_map_relative_l2"]) <= float(formal_thresholds["final_map_max_relative_l2"]),
        "sampled_hbm": float(performance["sampled_hbm_ratio"]) <= float(formal_thresholds["max_sampled_hbm_ratio"]),
        "sparse_group_wall": float(performance["sparse_group_wall_ratio"])
        <= float(formal_thresholds["max_sparse_group_wall_ratio"]),
    }
    science = {
        "controller_exact": bool(summary["controller_exact"]),
        "assignments_exact": int(summary["assignment_mismatch_count"]) == 0,
        "poses_exact": bool(summary["poses_exact"]),
        "no_class_collapse": bool(summary["no_class_collapse"]),
        "map_fsc": float(summary["min_final_map_signed_non_dc_fsc_auc"])
        >= float(science_thresholds["final_map_min_signed_non_dc_fsc_auc"]),
        "map_l2": float(summary["max_final_map_relative_l2"]) <= float(science_thresholds["final_map_max_relative_l2"]),
        "sampled_hbm": float(performance["sampled_hbm_ratio"]) <= float(science_thresholds["max_sampled_hbm_ratio"]),
        "sparse_group_wall": float(performance["sparse_group_wall_ratio"])
        <= float(science_thresholds["max_sparse_group_wall_ratio"]),
    }
    return {
        "formal": {
            "decision": "accept" if all(common_formal.values()) else "reject",
            "gates": common_formal,
        },
        "scientific_equivalence": {
            "decision": "accept" if all(science.values()) else "reject",
            "gates": science,
        },
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    performance = report["performance"]
    tiers = report["tiers"]
    return "\n".join(
        [
            f"# {report['label']} paired audit",
            "",
            f"Formal decision: **{tiers['formal']['decision'].upper()}**. Scientific-equivalence classification: **{tiers['scientific_equivalence']['decision'].upper()}**.",
            "",
            "The tiers are independent; scientific equivalence cannot rewrite the formal result.",
            "",
            "| Metric | Observed |",
            "| --- | ---: |",
            f"| Assignment mismatches | {summary['assignment_mismatch_count']} |",
            f"| Poses exact | {summary['poses_exact']} |",
            f"| Euler rows changed | {summary['euler_mismatch_row_count']} |",
            f"| Maximum physical rotation delta | {summary['max_physical_rotation_delta_deg']:.9g} deg |",
            f"| No class collapse | {summary['no_class_collapse']} |",
            f"| Minimum map FSC-AUC | {summary['min_final_map_signed_non_dc_fsc_auc']:.12f} |",
            f"| Maximum map relL2 | {summary['max_final_map_relative_l2']:.9g} |",
            f"| Sampled HBM ratio | {performance['sampled_hbm_ratio']:.6f} |",
            f"| Sparse-group wall ratio | {performance['sparse_group_wall_ratio']:.6f} |",
            "",
        ]
    )


def synthetic_self_test() -> None:
    n_iterations = 2
    n_classes = 4
    baseline: dict[str, np.ndarray] = {
        "current_sizes": np.asarray([64, 72]),
        "pixel_resolutions": np.asarray([3.0, 2.5]),
        "healpix_order_trajectory": np.asarray([1, 1]),
        "ave_Pmax_trajectory": np.asarray([0.2, 0.3]),
    }
    for iteration in range(n_iterations):
        suffix = f"{iteration:03d}"
        baseline[f"class_assignments_by_image_iter_{suffix}"] = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        baseline[f"sig_counts_by_image_iter_{suffix}"] = np.arange(8, dtype=np.int32)
        baseline[f"pmax_per_image_by_image_iter_{suffix}"] = np.linspace(0, 1, 8)
        baseline[f"best_rotation_eulers_by_image_iter_{suffix}"] = np.zeros((8, 3))
        baseline[f"best_translations_by_image_iter_{suffix}"] = np.zeros((8, 2))
    candidate = {key: value.copy() for key, value in baseline.items()}
    state = compare_saved_state_arrays(baseline, candidate, n_iterations=n_iterations)
    occupancy = class_occupancy_rows(
        baseline,
        candidate,
        n_iterations=n_iterations,
        n_classes=n_classes,
    )
    maps = [{"signed_non_dc_fsc_auc": 1.0, "relative_l2": 0.0}]
    summary = summarize_pair(state, occupancy, maps)
    require(summary["poses_exact"], "exact synthetic poses were rejected")
    require(summary["no_class_collapse"], "balanced synthetic classes collapsed")
    candidate["best_translations_by_image_iter_001"][0, 0] = 1.0
    candidate["class_assignments_by_image_iter_001"][:] = 0
    changed_state = compare_saved_state_arrays(baseline, candidate, n_iterations=n_iterations)
    changed_occupancy = class_occupancy_rows(
        baseline,
        candidate,
        n_iterations=n_iterations,
        n_classes=n_classes,
    )
    changed_summary = summarize_pair(changed_state, changed_occupancy, maps)
    require(not changed_summary["poses_exact"], "pose mutation was not detected")
    require(not changed_summary["no_class_collapse"], "class collapse was not detected")
    print("synthetic self-test: PASS")


def run_audit(args: argparse.Namespace) -> int:
    contract = read_json(args.contract)
    if args.expected_contract_sha256:
        require(
            sha256(args.contract) == args.expected_contract_sha256,
            "contract hash differs from --expected-contract-sha256",
        )
    baseline_root = args.baseline_root.resolve()
    candidate_root = args.candidate_root.resolve()
    baseline_output = baseline_root / args.output_relative
    candidate_output = candidate_root / args.output_relative
    baseline_results = baseline_output / args.results_name
    candidate_results = candidate_output / args.results_name
    required = [
        baseline_results,
        candidate_results,
        baseline_output / args.log_name,
        candidate_output / args.log_name,
        args.baseline_gpu_monitor,
        args.candidate_gpu_monitor,
        args.baseline_walltime,
        args.candidate_walltime,
    ]
    for path in required:
        require(path.is_file(), f"missing required artifact {path}")

    state, occupancy = compare_npz_state(
        baseline_results,
        candidate_results,
        n_iterations=args.n_iterations,
        n_classes=args.n_classes,
    )
    maps = compare_maps(
        baseline_output,
        candidate_output,
        n_classes=args.n_classes,
        fft_workers=args.fft_workers,
    )
    summary = summarize_pair(state, occupancy, maps)
    baseline_log = parse_log(baseline_output / args.log_name, n_iterations=args.n_iterations)
    candidate_log = parse_log(candidate_output / args.log_name, n_iterations=args.n_iterations)
    baseline_hbm = peak_hbm_mib(args.baseline_gpu_monitor)
    candidate_hbm = peak_hbm_mib(args.candidate_gpu_monitor)
    baseline_wall = float(read_json(args.baseline_walltime)["external_wall_s"])
    candidate_wall = float(read_json(args.candidate_walltime)["external_wall_s"])
    performance = {
        "baseline": baseline_log,
        "candidate": candidate_log,
        "baseline_external_wall_s": baseline_wall,
        "candidate_external_wall_s": candidate_wall,
        "external_wall_ratio_descriptive": candidate_wall / baseline_wall,
        "baseline_peak_sampled_hbm_mib": baseline_hbm,
        "candidate_peak_sampled_hbm_mib": candidate_hbm,
        "sampled_hbm_ratio": candidate_hbm / baseline_hbm,
        "sparse_group_wall_ratio": candidate_log["sparse_group_wall_sum_s"] / baseline_log["sparse_group_wall_sum_s"],
    }
    tiers = evaluate_tiers(
        summary,
        performance,
        contract["formal_thresholds"],
        contract["scientific_equivalence_thresholds"],
    )
    tiers["scientific_equivalence"]["prospective"] = args.science_prospective
    if args.prior_formal_report:
        prior = read_json(args.prior_formal_report)
        prior_row = next(row for row in prior["candidates"] if row["label"] == args.prior_candidate_label)
        require(
            tiers["formal"]["decision"] == prior_row["decision"],
            "recomputed formal decision differs from prior frozen report",
        )
        require(
            tiers["formal"]["gates"] == prior_row["gates"],
            "recomputed formal gates differ from prior frozen report",
        )
    report = {
        "schema": "recovar.em.kclass_default_threshold_pair_audit.v1",
        "label": args.label,
        "baseline": {
            "root": str(baseline_root),
            "job_id": args.baseline_job_id,
            "gpu_monitor": str(args.baseline_gpu_monitor.resolve()),
            "walltime": str(args.baseline_walltime.resolve()),
        },
        "candidate": {
            "root": str(candidate_root),
            "job_id": args.candidate_job_id,
            "gpu_monitor": str(args.candidate_gpu_monitor.resolve()),
            "walltime": str(args.candidate_walltime.resolve()),
        },
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256(args.contract),
        "formal_thresholds": contract["formal_thresholds"],
        "scientific_equivalence_thresholds": contract["scientific_equivalence_thresholds"],
        "summary": summary,
        "saved_state": state,
        "class_occupancy": occupancy,
        "maps": maps,
        "performance": performance,
        "tiers": tiers,
        "artifact_sha256": {
            "baseline_results": sha256(baseline_results),
            "candidate_results": sha256(candidate_results),
            "baseline_log": sha256(baseline_output / args.log_name),
            "candidate_log": sha256(candidate_output / args.log_name),
            "baseline_gpu_monitor": sha256(args.baseline_gpu_monitor),
            "candidate_gpu_monitor": sha256(args.candidate_gpu_monitor),
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(markdown(report))
    print(json.dumps(tiers, indent=2, sort_keys=True))
    return (
        0 if tiers["formal"]["decision"] == "accept" and tiers["scientific_equivalence"]["decision"] == "accept" else 3
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="run an in-memory synthetic helper test")
    audit = subparsers.add_parser("audit", help="audit one completed pair")
    audit.add_argument("--label", required=True)
    audit.add_argument("--baseline-root", type=Path, required=True)
    audit.add_argument("--candidate-root", type=Path, required=True)
    audit.add_argument("--baseline-job-id", type=int, required=True)
    audit.add_argument("--candidate-job-id", type=int, required=True)
    audit.add_argument("--baseline-gpu-monitor", type=Path, required=True)
    audit.add_argument("--candidate-gpu-monitor", type=Path, required=True)
    audit.add_argument("--baseline-walltime", type=Path, required=True)
    audit.add_argument("--candidate-walltime", type=Path, required=True)
    audit.add_argument("--contract", type=Path, required=True)
    audit.add_argument("--expected-contract-sha256")
    audit.add_argument("--science-prospective", action=argparse.BooleanOptionalAction, default=True)
    audit.add_argument("--prior-formal-report", type=Path)
    audit.add_argument("--prior-candidate-label", default="threshold128")
    audit.add_argument("--n-iterations", type=int, default=8)
    audit.add_argument("--n-classes", type=int, default=4)
    audit.add_argument("--fft-workers", type=int, default=default_fft_workers())
    audit.add_argument("--output-relative", type=Path, default=Path("half1/recovar"))
    audit.add_argument("--results-name", default="refinement_results.npz")
    audit.add_argument("--log-name", default="run.log")
    audit.add_argument("--output-json", type=Path, required=True)
    audit.add_argument("--output-md", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-test":
        synthetic_self_test()
        return
    raise SystemExit(run_audit(args))


if __name__ == "__main__":
    main()
