#!/usr/bin/env python3
"""Analyze the focused stable-Fourier-shape VDAM trajectory gate.

This gate compares two ordinary-shape and two stable-shape runs from fresh
processes and fresh JAX caches.  It deliberately reuses the mature EM volume
loader and GPU-monitor parser; no scientific update is reimplemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_stable_fourier_window_shape_plan,
)
from scripts.analyze_vdam_coarse_combined_true200 import (
    GateSetupError,
    _load_json,
    _load_particles,
    _require,
)
from scripts.summarize_em_completion_bench import _load_relion_volume, _read_gpu_monitor

SCHEMA = "recovar.vdam_stable_shape_trajectory.v3"
ARM_ORDER = ("stable_off_1", "stable_on_1", "stable_on_2", "stable_off_2")
CONTROL_ARMS = ("stable_off_1", "stable_off_2")
CANDIDATE_ARMS = ("stable_on_1", "stable_on_2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_array(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GateSetupError(f"{label} is not numeric") from exc
    _require(np.all(np.isfinite(array)), f"{label} contains non-finite values")
    return array


def normalized_l2(left: Any, right: Any) -> float:
    """Symmetric normalized L2 distance used by the mature true-200 gate."""

    left_array = _numeric_array(left, "left operand").reshape(-1)
    right_array = _numeric_array(right, "right operand").reshape(-1)
    _require(left_array.shape == right_array.shape, "numeric operand shapes differ")
    numerator = float(np.linalg.norm(left_array - right_array))
    denominator = math.sqrt(
        0.5 * (float(np.vdot(left_array, left_array).real) + float(np.vdot(right_array, right_array).real))
    )
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def summarize_pair_distances(
    values: dict[str, Any],
    *,
    distance=normalized_l2,
) -> dict[str, Any]:
    """Summarize repeat and cross-mode distances without fitting a tolerance."""

    _require(set(values) == set(ARM_ORDER), "distance panel has the wrong arms")

    def pair(left: str, right: str) -> dict[str, Any]:
        return {"left": left, "right": right, "distance": float(distance(values[left], values[right]))}

    control_repeat = pair(*CONTROL_ARMS)
    candidate_repeat = pair(*CANDIDATE_ARMS)
    cross = [pair(left, right) for left in CONTROL_ARMS for right in CANDIDATE_ARMS]
    all_rows = [control_repeat, candidate_repeat, *cross]
    return {
        "control_repeat": control_repeat,
        "candidate_repeat": candidate_repeat,
        "cross_mode": cross,
        "maximum_repeat_distance": max(control_repeat["distance"], candidate_repeat["distance"]),
        "maximum_cross_mode_distance": max(row["distance"] for row in cross),
        "maximum_any_pair_distance": max(row["distance"] for row in all_rows),
    }


def summarize_runtime_panel(
    rows: dict[str, dict[str, float]],
    *,
    minimum_speedup_percent: float,
    maximum_memory_increase_percent: float,
) -> dict[str, Any]:
    """Apply the predeclared median runtime and memory gates."""

    _require(set(rows) == set(ARM_ORDER), "runtime panel has the wrong arms")

    def median(names: Sequence[str], key: str) -> float:
        values = np.asarray([rows[name][key] for name in names], dtype=np.float64)
        _require(np.all(np.isfinite(values)) and np.all(values > 0.0), f"invalid runtime metric {key}")
        return float(np.median(values))

    control_wall = median(CONTROL_ARMS, "end_to_end_wall_s")
    candidate_wall = median(CANDIDATE_ARMS, "end_to_end_wall_s")
    control_expectation = median(CONTROL_ARMS, "expectation_stage_s")
    candidate_expectation = median(CANDIDATE_ARMS, "expectation_stage_s")
    control_memory = median(CONTROL_ARMS, "peak_gpu_memory_mib")
    candidate_memory = median(CANDIDATE_ARMS, "peak_gpu_memory_mib")

    def percent_change(candidate: float, control: float) -> float:
        return 100.0 * (candidate / control - 1.0)

    wall_change = percent_change(candidate_wall, control_wall)
    expectation_change = percent_change(candidate_expectation, control_expectation)
    memory_change = percent_change(candidate_memory, control_memory)
    wall_pass = wall_change <= -float(minimum_speedup_percent)
    expectation_pass = expectation_change <= -float(minimum_speedup_percent)
    memory_pass = memory_change <= float(maximum_memory_increase_percent)
    return {
        "arms": rows,
        "median_control_end_to_end_wall_s": control_wall,
        "median_candidate_end_to_end_wall_s": candidate_wall,
        "median_end_to_end_percent_change": wall_change,
        "median_control_expectation_stage_s": control_expectation,
        "median_candidate_expectation_stage_s": candidate_expectation,
        "median_expectation_stage_percent_change": expectation_change,
        "median_control_peak_gpu_memory_mib": control_memory,
        "median_candidate_peak_gpu_memory_mib": candidate_memory,
        "median_peak_gpu_memory_percent_change": memory_change,
        "minimum_required_speedup_percent": float(minimum_speedup_percent),
        "maximum_allowed_memory_increase_percent": float(maximum_memory_increase_percent),
        "end_to_end_pass": wall_pass,
        "expectation_stage_pass": expectation_pass,
        "memory_pass": memory_pass,
        "pass": wall_pass and expectation_pass and memory_pass,
    }


def _native_options(
    root: Path,
    arm: str,
    *,
    candidate_enabled: bool,
    stable_fourier_window_shapes: bool,
    stable_flat_row_capacity: bool,
) -> dict[str, Any]:
    path = root / "runs" / arm / "output" / "run_native_options.json"
    options = _load_json(path, f"{arm} native options")
    expected_stable_shapes = bool(
        candidate_enabled and stable_fourier_window_shapes
    )
    _require(
        bool(options.get("stable_fourier_window_shapes"))
        is expected_stable_shapes,
        f"{arm} stable-shape option differs",
    )
    command_path = root / "runs" / arm / "command.json"
    command = _load_json(command_path, f"{arm} command")
    expected_flat_capacity = bool(candidate_enabled and stable_flat_row_capacity)
    _require(
        bool(command.get("stable_flat_row_capacity", False)) is expected_flat_capacity,
        f"{arm} stable-flat command contract differs",
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "command_path": str(command_path.resolve()),
        "command_sha256": _sha256(command_path),
        "stable_shapes": expected_stable_shapes,
        "stable_flat_row_capacity": expected_flat_capacity,
    }


def validate_stable_flat_capacity_execution(
    meta: dict[str, Any],
    *,
    expected: bool,
    arm: str,
    iteration: int,
) -> dict[str, int | bool]:
    """Validate and summarize the fixed-`B * R` packed-row execution contract."""

    for key in (
        "requested_stable_flat_row_capacity",
        "effective_stable_flat_row_capacity",
    ):
        _require(key in meta, f"{arm} iteration {iteration} omitted {key}")
        _require(
            bool(meta[key]) is bool(expected),
            f"{arm} iteration {iteration} reported {key}={meta[key]!r}, expected {expected!r}",
        )

    profile_count = 0
    chunk_count = 0
    strict_reduction_count = 0
    flat_row_sum = 0
    padded_row_sum = 0
    for key, value in sorted(meta.items()):
        if not isinstance(value, dict) or "chunk_flat_score_rows" not in value:
            continue
        profile_count += 1
        _require(
            bool(value.get("flat_local_rows_enabled", False)),
            f"{arm} iteration {iteration} profile {key} did not execute flat rows",
        )
        _require(
            "stable_flat_row_capacity_enabled" in value,
            f"{arm} iteration {iteration} profile {key} omitted its stable-flat flag",
        )
        _require(
            bool(value["stable_flat_row_capacity_enabled"]) is bool(expected),
            f"{arm} iteration {iteration} profile {key} stable-flat flag differs",
        )
        flat_rows = np.asarray(value["chunk_flat_score_rows"], dtype=np.int64)
        padded_rows = np.asarray(value.get("chunk_padded_rotations"), dtype=np.int64)
        _require(
            flat_rows.ndim == 1 and flat_rows.size > 0,
            f"{arm} iteration {iteration} profile {key} has no row chunks",
        )
        _require(
            flat_rows.shape == padded_rows.shape,
            f"{arm} iteration {iteration} profile {key} row shapes differ",
        )
        _require(
            bool(np.all(flat_rows <= padded_rows)),
            f"{arm} iteration {iteration} profile {key} exceeds B * R capacity",
        )
        if expected:
            _require(
                bool(np.array_equal(flat_rows, padded_rows)),
                f"{arm} iteration {iteration} profile {key} did not use fixed B * R rows",
            )
        chunk_count += int(flat_rows.size)
        strict_reduction_count += int(np.count_nonzero(flat_rows < padded_rows))
        flat_row_sum += int(flat_rows.sum())
        padded_row_sum += int(padded_rows.sum())

    _require(profile_count > 0, f"{arm} iteration {iteration} has no local-engine profile")
    return {
        "enabled": bool(expected),
        "profile_count": profile_count,
        "chunk_count": chunk_count,
        "strict_reduction_count": strict_reduction_count,
        "flat_row_sum": flat_row_sum,
        "padded_row_sum": padded_row_sum,
    }


def _arm_runtime(root: Path, arm: str, iterations: Sequence[int]) -> dict[str, float]:
    run_root = root / "runs" / arm
    timing = _load_json(run_root / "timing.json", f"{arm} timing")
    wall = float(timing.get("external_wall_s", math.nan))
    _require(math.isfinite(wall) and wall > 0.0, f"{arm} wall time is invalid")
    expectation = 0.0
    for iteration in iterations:
        if iteration == 0:
            continue
        meta = _load_json(
            run_root / "output" / f"run_it{iteration:03d}_recovar_meta.json",
            f"{arm} iteration {iteration} metadata",
        )
        profile = meta.get("vdam_iteration_profile_summary")
        _require(isinstance(profile, dict), f"{arm} iteration {iteration} has no profile")
        value = float(profile.get("expectation_time_s", math.nan))
        _require(math.isfinite(value) and value >= 0.0, f"{arm} expectation time is invalid")
        expectation += value
    monitor = _read_gpu_monitor(run_root / "gpu_monitor.csv")
    _require(isinstance(monitor, dict), f"{arm} GPU monitor is missing")
    peak = float(monitor.get("peak_memory_mib", math.nan))
    _require(
        int(monitor.get("sample_count", 0)) >= 2 and math.isfinite(peak) and peak > 0.0,
        f"{arm} GPU monitor is incomplete",
    )
    return {
        "end_to_end_wall_s": wall,
        "expectation_stage_s": expectation,
        "peak_gpu_memory_mib": peak,
    }


def analyze(
    root: Path,
    acceptance_path: Path,
    *,
    last_iteration: int,
    stable_fourier_window_shapes: bool,
    stable_flat_row_capacity: bool,
    numerical_tolerance: float,
    minimum_speedup_percent: float,
    maximum_memory_increase_percent: float,
) -> dict[str, Any]:
    acceptance = _load_json(acceptance_path, "true-200 acceptance contract")
    state_contract = acceptance.get("state_contract")
    _require(isinstance(state_contract, dict), "acceptance state contract is missing")
    exact_keys = tuple(state_contract["joint_exact_complete_path_metadata_keys"])
    numeric_keys = tuple(state_contract["serial_leave_one_out_numeric_keys"])
    identity_columns = tuple(state_contract["star_identity_columns"])
    numeric_groups = dict(state_contract["star_numeric_groups"])
    iterations = tuple(range(0, int(last_iteration) + 1))
    _require(iterations[-1] >= 2, "trajectory must include at least two VDAM iterations")
    _require(
        math.isfinite(numerical_tolerance) and numerical_tolerance > 0.0,
        "numerical tolerance must be finite and positive",
    )
    _require(
        stable_fourier_window_shapes or stable_flat_row_capacity,
        "trajectory candidate must enable at least one stable ABI",
    )

    native_options = {
        arm: _native_options(
            root,
            arm,
            candidate_enabled=arm in CANDIDATE_ARMS,
            stable_fourier_window_shapes=stable_fourier_window_shapes,
            stable_flat_row_capacity=stable_flat_row_capacity,
        )
        for arm in ARM_ORDER
    }
    runtime_rows = {arm: _arm_runtime(root, arm, iterations) for arm in ARM_ORDER}

    exact_failures: list[dict[str, Any]] = []
    numerical_failures: list[dict[str, Any]] = []
    checked_identity_columns: set[str] = set()
    uniformly_absent_identity_columns: set[str] = set()
    checkpoint_rows: list[dict[str, Any]] = []
    map_squared_sum = {f"{left}__vs__{right}": 0.0 for left in ARM_ORDER for right in ARM_ORDER if left < right}
    logical_sizes: list[int] = []
    physical_sizes: list[int] = []
    flat_capacity_execution = {
        arm: {
            "enabled": bool(stable_flat_row_capacity and arm in CANDIDATE_ARMS),
            "profile_count": 0,
            "chunk_count": 0,
            "strict_reduction_count": 0,
            "flat_row_sum": 0,
            "padded_row_sum": 0,
        }
        for arm in ARM_ORDER
    }

    for iteration in iterations:
        metas: dict[str, dict[str, Any]] = {}
        stars = {}
        maps = {}
        for arm in ARM_ORDER:
            output = root / "runs" / arm / "output"
            tag = f"run_it{iteration:03d}"
            for suffix in ("class001.mrc", "data.star", "model.star", "recovar_meta.json"):
                _require((output / f"{tag}_{suffix}").is_file(), f"missing {arm} {tag}_{suffix}")
            metas[arm] = _load_json(output / f"{tag}_recovar_meta.json", f"{arm} iteration {iteration}")
            stars[arm] = _load_particles(output / f"{tag}_data.star", f"{arm} iteration {iteration} STAR")
            maps[arm] = _load_relion_volume(output / f"{tag}_class001.mrc")

        logical_size = None
        plan = None
        if iteration > 0:
            sizes = {int(metas[arm]["current_size"]) for arm in ARM_ORDER}
            _require(len(sizes) == 1, f"iteration {iteration} current-size schedule differs")
            logical_size = sizes.pop()
            logical_sizes.append(logical_size)
            plan = make_stable_fourier_window_shape_plan(
                maps[ARM_ORDER[0]].shape[:2],
                logical_size,
                maps[ARM_ORDER[0]].shape[0] * (maps[ARM_ORDER[0]].shape[0] // 2 + 1),
                enabled=stable_fourier_window_shapes,
            )
            physical_sizes.append(int(plan.physical_current_size))
            for arm in ARM_ORDER:
                candidate_enabled = arm in CANDIDATE_ARMS
                stable_shapes_enabled = bool(
                    candidate_enabled and stable_fourier_window_shapes
                )
                observed = bool(metas[arm].get("effective_stable_fourier_window_shapes"))
                if observed is not stable_shapes_enabled:
                    exact_failures.append({"iteration": iteration, "feature": "stable_shape_execution", "arm": arm})
                profile = metas[arm].get("halfset_0_profile_summary", {})
                expected_windowed = (
                    plan.physical_score_pixels
                    if stable_shapes_enabled
                    else plan.logical_score_pixels
                )
                if int(profile.get("n_windowed", -1)) != int(expected_windowed):
                    exact_failures.append(
                        {
                            "iteration": iteration,
                            "feature": "window_capacity",
                            "arm": arm,
                            "expected": int(expected_windowed),
                            "observed": int(profile.get("n_windowed", -1)),
                        }
                    )
                if stable_flat_row_capacity:
                    observation = validate_stable_flat_capacity_execution(
                        metas[arm],
                        expected=candidate_enabled,
                        arm=arm,
                        iteration=iteration,
                    )
                    for key in (
                        "profile_count",
                        "chunk_count",
                        "strict_reduction_count",
                        "flat_row_sum",
                        "padded_row_sum",
                    ):
                        flat_capacity_execution[arm][key] += int(observation[key])

        numeric_max = 0.0
        if iteration > 0:
            for key in exact_keys:
                values = {arm: metas[arm].get(key) for arm in ARM_ORDER}
                reference = np.asarray(values[ARM_ORDER[0]])
                for arm in ARM_ORDER[1:]:
                    if not np.array_equal(reference, np.asarray(values[arm])):
                        exact_failures.append({"iteration": iteration, "feature": f"meta:{key}", "arm": arm})

            for key in numeric_keys:
                values = {arm: metas[arm].get(key) for arm in ARM_ORDER}
                distances = summarize_pair_distances(values)
                numeric_max = max(numeric_max, distances["maximum_any_pair_distance"])
                if distances["maximum_any_pair_distance"] > numerical_tolerance:
                    numerical_failures.append(
                        {
                            "iteration": iteration,
                            "feature": f"meta:{key}",
                            "distance": distances["maximum_any_pair_distance"],
                        }
                    )

        for column in identity_columns:
            availability = [column in stars[arm].columns for arm in ARM_ORDER]
            _require(
                len(set(availability)) == 1,
                f"iteration {iteration} STAR identity availability differs for {column}",
            )
            if not availability[0]:
                uniformly_absent_identity_columns.add(column)
                continue
            checked_identity_columns.add(column)
            reference = stars[ARM_ORDER[0]][column].to_numpy()
            for arm in ARM_ORDER[1:]:
                if not np.array_equal(reference, stars[arm][column].to_numpy()):
                    exact_failures.append({"iteration": iteration, "feature": f"STAR:{column}", "arm": arm})
        for group, columns in numeric_groups.items():
            values = {arm: stars[arm][columns].to_numpy(dtype=np.float64, copy=False) for arm in ARM_ORDER}
            distances = summarize_pair_distances(values)
            numeric_max = max(numeric_max, distances["maximum_any_pair_distance"])
            if distances["maximum_any_pair_distance"] > numerical_tolerance:
                numerical_failures.append(
                    {
                        "iteration": iteration,
                        "feature": f"STAR:{group}",
                        "distance": distances["maximum_any_pair_distance"],
                    }
                )

        map_distances = summarize_pair_distances(maps)
        for row in (
            map_distances["control_repeat"],
            map_distances["candidate_repeat"],
            *map_distances["cross_mode"],
        ):
            key = "__vs__".join(sorted((row["left"], row["right"])))
            map_squared_sum[key] += float(row["distance"]) ** 2
        if map_distances["maximum_any_pair_distance"] > numerical_tolerance:
            numerical_failures.append(
                {
                    "iteration": iteration,
                    "feature": "map",
                    "distance": map_distances["maximum_any_pair_distance"],
                }
            )
        checkpoint_rows.append(
            {
                "iteration": iteration,
                "logical_current_size": logical_size,
                "physical_current_size": (None if plan is None else int(plan.physical_current_size)),
                "map": map_distances,
                "maximum_numeric_state_distance": numeric_max,
            }
        )

    map_trajectory_rms = {key: math.sqrt(value / len(iterations)) for key, value in map_squared_sum.items()}
    maximum_map_distance = max(row["map"]["maximum_any_pair_distance"] for row in checkpoint_rows)
    maximum_numeric_distance = max(row["maximum_numeric_state_distance"] for row in checkpoint_rows)
    _require(
        checked_identity_columns.isdisjoint(uniformly_absent_identity_columns),
        "STAR identity column availability changed during the trajectory",
    )
    if stable_flat_row_capacity:
        for arm in CONTROL_ARMS:
            _require(
                int(flat_capacity_execution[arm]["strict_reduction_count"]) > 0,
                f"{arm} never exercised ordinary packed-row reduction",
            )
        for arm in CANDIDATE_ARMS:
            _require(
                int(flat_capacity_execution[arm]["strict_reduction_count"]) == 0,
                f"{arm} did not retain the fixed B * R row ABI",
            )
    science = {
        "fixed_numerical_tolerance": numerical_tolerance,
        "exact_failure_count": len(exact_failures),
        "exact_failures": exact_failures,
        "numerical_failure_count": len(numerical_failures),
        "numerical_failures": numerical_failures,
        "maximum_checkpoint_map_normalized_l2": maximum_map_distance,
        "maximum_numeric_state_normalized_l2": maximum_numeric_distance,
        "map_whole_trajectory_rms_normalized_l2": map_trajectory_rms,
        "star_identity_columns_checked": sorted(checked_identity_columns),
        "star_identity_columns_uniformly_absent": sorted(uniformly_absent_identity_columns),
        "halfset_identity_covered_by_exact_metadata": "halfset_ids" in exact_keys,
        "pass": not exact_failures and not numerical_failures,
    }
    runtime = summarize_runtime_panel(
        runtime_rows,
        minimum_speedup_percent=minimum_speedup_percent,
        maximum_memory_increase_percent=maximum_memory_increase_percent,
    )
    return {
        "schema": SCHEMA,
        "classification": (
            "focused_stable_fourier_and_flat_row_abi_multi_size_abba_gate"
            if stable_fourier_window_shapes and stable_flat_row_capacity
            else (
                "focused_stable_fourier_abi_multi_size_abba_gate"
                if stable_fourier_window_shapes
                else "focused_stable_flat_row_abi_multi_size_abba_gate"
            )
        ),
        "trajectory_mode": (
            "stable_fourier_and_flat_row_abi"
            if stable_fourier_window_shapes and stable_flat_row_capacity
            else (
                "stable_fourier_abi_only"
                if stable_fourier_window_shapes
                else "stable_flat_row_abi_only"
            )
        ),
        "arm_order": list(ARM_ORDER),
        "fresh_process_and_jax_cache_per_arm_required": True,
        "only_configuration_delta": [
            *(
                ["--stable-fourier-window-shapes"]
                if stable_fourier_window_shapes
                else []
            ),
            *(
                ["RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY=1"]
                if stable_flat_row_capacity
                else []
            ),
        ],
        "acceptance_config": str(acceptance_path.resolve()),
        "acceptance_config_sha256": _sha256(acceptance_path),
        "last_iteration": int(last_iteration),
        "logical_current_size_sequence": logical_sizes,
        "physical_current_size_sequence": physical_sizes,
        "unique_logical_current_sizes": sorted(set(logical_sizes)),
        "unique_physical_current_sizes": sorted(set(physical_sizes)),
        "native_options": native_options,
        "flat_row_capacity_execution": (
            flat_capacity_execution if stable_flat_row_capacity else None
        ),
        "science": science,
        "runtime": runtime,
        "checkpoints": checkpoint_rows,
        "pass": science["pass"] and runtime["pass"],
    }


def _markdown(report: dict[str, Any]) -> str:
    science = report["science"]
    runtime = report["runtime"]
    status = "PASS" if report["pass"] else "FAIL"
    title = (
        "Stable Fourier + flat-row ABI trajectory gate"
        if report["trajectory_mode"] == "stable_fourier_and_flat_row_abi"
        else (
            "Stable Fourier-shape trajectory gate"
            if report["trajectory_mode"] == "stable_fourier_abi_only"
            else "Stable flat-row ABI trajectory gate"
        )
    )
    return "\n".join(
        (
            f"# {title}: {status}",
            "",
            "| Gate | Result |",
            "|---|---:|",
            f"| Iterations | 0--{report['last_iteration']} |",
            f"| Logical / physical size classes | {len(report['unique_logical_current_sizes'])} / {len(report['unique_physical_current_sizes'])} |",
            f"| Exact-state failures | {science['exact_failure_count']} |",
            f"| Numerical-bound failures | {science['numerical_failure_count']} |",
            f"| Maximum map normalized L2 | {science['maximum_checkpoint_map_normalized_l2']:.3e} |",
            f"| Fixed numerical bound | {science['fixed_numerical_tolerance']:.3e} |",
            f"| End-to-end change | {runtime['median_end_to_end_percent_change']:+.2f}% |",
            f"| Expectation-stage change | {runtime['median_expectation_stage_percent_change']:+.2f}% |",
            f"| Peak-memory change | {runtime['median_peak_gpu_memory_percent_change']:+.2f}% |",
            "",
        )
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--last-iteration", type=int, default=50)
    parser.add_argument(
        "--stable-fourier-window-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require candidate arms to use stable Fourier-window shapes",
    )
    parser.add_argument(
        "--stable-flat-row-capacity",
        action="store_true",
        help="require candidate arms to use the fixed B * R packed-row ABI",
    )
    parser.add_argument("--numerical-tolerance", type=float, default=2.0**-21)
    parser.add_argument("--minimum-speedup-percent", type=float, default=5.0)
    parser.add_argument("--maximum-memory-increase-percent", type=float, default=5.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = analyze(
            args.root.resolve(),
            args.acceptance.resolve(),
            last_iteration=args.last_iteration,
            stable_fourier_window_shapes=args.stable_fourier_window_shapes,
            stable_flat_row_capacity=args.stable_flat_row_capacity,
            numerical_tolerance=args.numerical_tolerance,
            minimum_speedup_percent=args.minimum_speedup_percent,
            maximum_memory_increase_percent=args.maximum_memory_increase_percent,
        )
    except GateSetupError as exc:
        print(f"SETUP ERROR: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(_markdown(report))
    print(_markdown(report), end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
