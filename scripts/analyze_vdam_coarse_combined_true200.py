#!/usr/bin/env python3
"""Analyze the sealed replicated VDAM mature combined-coarse 0--200 gate.

Scientific map quality is evaluated only with the mature EM shellwise FSC and
FSC-AUC helpers.  This module adds replicated-panel statistics, serial
leave-one-out envelopes, and fail-closed artifact/provenance checks; it does
not implement or replay any EM/VDAM scientific update.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import starfile

from scripts.analyze_em_k1_map_amplitude_trajectory import (
    centered_fourier,
    summarize_fourier_pair,
)
from scripts.audit_em_particle_state_distribution import _angular_error_deg
from scripts.summarize_em_completion_bench import (
    _load_relion_volume,
    _read_gpu_monitor,
    first_shell_below,
    normalized_fsc_auc,
    shell_fsc,
)

SCHEMA = "recovar.vdam_coarse_combined_true200_analysis.v2"
_ITERATION_ARTIFACT_RE = re.compile(
    r"^run_it(?P<iteration>\d{3})_(?P<suffix>class001\.mrc|data\.star|model\.star|recovar_meta\.json)$"
)


class GateSetupError(RuntimeError):
    """Raised when evidence is incomplete, contaminated, or not sealed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateSetupError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateSetupError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateSetupError(f"{label} must contain one JSON object: {path}")
    return value


def _finite_array(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GateSetupError(f"{label} is not numeric") from exc
    if not np.all(np.isfinite(array)):
        raise GateSetupError(f"{label} contains non-finite values")
    return array


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    left = _finite_array(left, "left array").reshape(-1)
    right = _finite_array(right, "right array").reshape(-1)
    if left.shape != right.shape:
        raise GateSetupError(f"array shapes differ: {left.shape} != {right.shape}")
    denominator = math.sqrt(0.5 * (float(np.vdot(left, left).real) + float(np.vdot(right, right).real)))
    numerator = float(np.linalg.norm(left - right))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _inside(value: float, bound: float) -> bool:
    """Compare derived nonnegative metrics without inventing a science tolerance."""

    if not math.isfinite(value) or not math.isfinite(bound) or bound < 0.0:
        return False
    if bound == 0.0:
        return value == 0.0
    return value <= np.nextafter(bound, math.inf)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _distance_matrix(values: Sequence[Any], distance: Callable[[Any, Any], float]) -> np.ndarray:
    count = len(values)
    result = np.zeros((count, count), dtype=np.float64)
    for left, right in itertools.combinations(range(count), 2):
        value = float(distance(values[left], values[right]))
        if not math.isfinite(value) or value < 0.0:
            raise GateSetupError("distance function returned an invalid value")
        result[left, right] = value
        result[right, left] = value
    return result


def _serial_loo_from_distances(
    distances: np.ndarray,
    serial_indices: Sequence[int],
    lane_indices: Sequence[int],
    *,
    multiplier: float,
    minimum_radius: float = 0.0,
) -> dict[str, Any]:
    serial = tuple(int(value) for value in serial_indices)
    lanes = tuple(int(value) for value in lane_indices)
    _require(len(serial) >= 3 and lanes, "serial LOO requires at least three controls and one lane arm")
    _require(multiplier >= 1.0, "serial LOO multiplier must be at least one")
    _require(
        math.isfinite(float(minimum_radius)) and float(minimum_radius) >= 0.0,
        "serial LOO minimum radius must be finite and non-negative",
    )
    serial_nearest = []
    for index in serial:
        peers = [distances[index, peer] for peer in serial if peer != index]
        serial_nearest.append(float(min(peers)))
    empirical_radius = float(max(serial_nearest)) * float(multiplier)
    radius = max(empirical_radius, float(minimum_radius))
    lane_nearest = [float(min(distances[index, peer] for peer in serial)) for index in lanes]
    lane_pass = [_inside(value, radius) for value in lane_nearest]
    return {
        "serial_leave_one_out_nearest": serial_nearest,
        "serial_leave_one_out_empirical_radius": empirical_radius,
        "minimum_accepted_radius": float(minimum_radius),
        "serial_leave_one_out_radius": radius,
        "lane_nearest_serial": lane_nearest,
        "lane_inside": lane_pass,
        "pass": all(lane_pass),
    }


def _group_scatter(gram: np.ndarray, indices: Sequence[int]) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    return max(
        float(np.sum(np.diag(gram)[idx]) - np.sum(gram[np.ix_(idx, idx)]) / idx.size),
        0.0,
    )


def _balanced_assignments(blocks: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    flattened = tuple(int(index) for block in blocks for index in block)
    _require(bool(blocks), "blocked design has no blocks")
    _require(all(len(block) >= 2 and len(block) % 2 == 0 for block in blocks), "each block must be nonempty and even")
    _require(len(set(flattened)) == len(flattened), "blocked design repeats an arm index")
    _require(set(flattened) == set(range(len(flattened))), "blocked design must cover every arm exactly once")
    choices = [tuple(itertools.combinations(tuple(block), len(block) // 2)) for block in blocks]
    return [tuple(sorted(itertools.chain.from_iterable(selected))) for selected in itertools.product(*choices)]


def _blocked_permutation_statistics(
    gram: np.ndarray,
    blocks: Sequence[Sequence[int]],
    observed_serial_indices: Sequence[int],
) -> dict[str, Any]:
    count = int(gram.shape[0])
    _require(gram.shape == (count, count), "trajectory Gram matrix must be square")
    assignments = _balanced_assignments(blocks)
    observed = tuple(sorted(int(value) for value in observed_serial_indices))
    _require(observed in assignments, "observed labels do not satisfy the blocked design")
    all_indices = tuple(range(count))
    distance_sq = np.maximum(
        np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram,
        0.0,
    )
    distances = np.sqrt(distance_sq)
    endpoints: list[tuple[float, float, float, float]] = []
    details: list[dict[str, float]] = []
    for selected in assignments:
        complement = tuple(index for index in all_indices if index not in selected)
        weight = np.zeros(count, dtype=np.float64)
        weight[list(complement)] = 1.0 / len(complement)
        weight[list(selected)] = -1.0 / len(selected)
        centroid = math.sqrt(max(float(weight @ gram @ weight), 0.0))
        serial_scatter = _group_scatter(gram, selected)
        lane_scatter = _group_scatter(gram, complement)
        pooled_rms = math.sqrt((serial_scatter + lane_scatter) / max(count - 2, 1))
        centroid_effect = _ratio(centroid, pooled_rms)
        within = [
            distances[left, right]
            for group in (selected, complement)
            for left, right in itertools.combinations(group, 2)
        ]
        cross = [distances[left, right] for left in selected for right in complement]
        cross_within = _ratio(float(np.median(cross)), float(np.median(within)))
        serial_variance = serial_scatter / max(len(selected) - 1, 1)
        lane_variance = lane_scatter / max(len(complement) - 1, 1)
        variance_inflation = _ratio(lane_variance, serial_variance)
        block_effects = []
        for block in blocks:
            block_serial = tuple(index for index in block if index in selected)
            block_lane = tuple(index for index in block if index in complement)
            block_weight = np.zeros(count, dtype=np.float64)
            block_weight[list(block_lane)] = 1.0 / len(block_lane)
            block_weight[list(block_serial)] = -1.0 / len(block_serial)
            block_effects.append(math.sqrt(max(float(block_weight @ gram @ block_weight), 0.0)))
        block_heterogeneity = float(np.std(block_effects))
        endpoints.append((centroid_effect, cross_within, variance_inflation, block_heterogeneity))
        details.append(
            {
                "centroid_over_pooled_run_rms": centroid_effect,
                "cross_within_median_distance_ratio": cross_within,
                "lane_variance_inflation": variance_inflation,
                "arm_by_block_heterogeneity": block_heterogeneity,
            }
        )
    values = np.asarray(endpoints, dtype=np.float64)
    finite = np.isfinite(values)
    location = np.zeros(values.shape[1], dtype=np.float64)
    scale = np.ones(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        column_values = values[finite[:, column], column]
        if column_values.size:
            location[column] = float(np.mean(column_values))
            observed_scale = float(np.std(column_values, ddof=1)) if column_values.size > 1 else 0.0
            scale[column] = observed_scale if observed_scale > 0.0 else 1.0
    standardized = (values - location) / scale
    standardized[~finite] = math.inf
    joint = np.max(standardized, axis=1)
    observed_row = assignments.index(observed)
    observed_joint = float(joint[observed_row])
    p_value = float(np.count_nonzero(joint >= observed_joint) / len(joint))
    return {
        "n_blocked_whole_trajectory_permutations": len(assignments),
        "observed": details[observed_row],
        "observed_joint_max": observed_joint,
        "joint_max_one_sided_permutation_p": p_value,
    }


def _linear_slope(iterations: np.ndarray, values: np.ndarray) -> float:
    centered = iterations.astype(np.float64) - float(np.mean(iterations))
    denominator = float(np.vdot(centered, centered).real)
    if denominator == 0.0:
        return 0.0
    return float(np.vdot(centered, values - float(np.mean(values))).real / denominator)


def _leave_one_out_radius(gram: np.ndarray, indices: Sequence[int]) -> float:
    """Return the largest nearest-neighbour distance inside one reference group."""

    count = int(gram.shape[0])
    selected = np.asarray(tuple(int(value) for value in indices), dtype=np.int64)
    _require(gram.shape == (count, count), "LOO Gram matrix is invalid")
    _require(selected.size >= 2 and np.unique(selected).size == selected.size, "LOO reference group is invalid")
    _require(np.all((selected >= 0) & (selected < count)), "LOO reference index is out of range")
    distance_sq = np.maximum(
        np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram,
        0.0,
    )
    distances = np.sqrt(distance_sq[np.ix_(selected, selected)])
    np.fill_diagonal(distances, math.inf)
    return float(np.max(np.min(distances, axis=1)))


def _pooled_leave_one_out_radius(gram: np.ndarray) -> float:
    """Compatibility helper for all-arm focused geometry tests."""

    return _leave_one_out_radius(gram, range(gram.shape[0]))


def _positive_energy_increment(current: float, reference: float) -> float:
    """Magnitude added in quadrature, avoiding norm-difference cancellation."""

    current_value = float(current)
    reference_value = float(reference)
    _require(
        math.isfinite(current_value)
        and math.isfinite(reference_value)
        and current_value >= 0.0
        and reference_value >= 0.0,
        "no-growth excursion energy is invalid",
    )
    if current_value <= reference_value:
        return 0.0
    ratio = reference_value / current_value
    increment = current_value * math.sqrt(max((1.0 - ratio) * (1.0 + ratio), 0.0))
    _require(math.isfinite(increment), "no-growth excursion energy overflowed")
    return increment


def classify_no_growth(
    iteration_grams: np.ndarray,
    iterations: Sequence[int],
    *,
    blocks: Sequence[Sequence[int]],
    observed_serial_indices: Sequence[int],
    phase_segments: Sequence[Sequence[int]],
    phase_boundary_windows: Sequence[Sequence[int]],
    late_window: Sequence[int],
    primary_family_endpoints: dict[str, str],
    familywise_alpha: float,
    materiality_loo_multiplier: float,
    normalized_numerical_noise_floor: float,
) -> dict[str, Any]:
    """Test predeclared growth families for both direction and materiality."""

    grams = _finite_array(iteration_grams, "iteration trajectory Gram matrices")
    iteration_array = np.asarray(iterations, dtype=np.int64)
    _require(grams.ndim == 3 and grams.shape[0] == iteration_array.size, "growth Gram count differs")
    _require(grams.shape[1] == grams.shape[2], "growth Gram matrices must be square")
    assignments = _balanced_assignments(blocks)
    observed = tuple(sorted(int(value) for value in observed_serial_indices))
    _require(observed in assignments, "observed growth labels do not satisfy blocked design")
    count = grams.shape[1]
    all_indices = tuple(range(count))
    by_iteration = {int(value): index for index, value in enumerate(iteration_array)}
    early_end, terminal = (int(value) for value in late_window)
    _require(early_end in by_iteration and terminal in by_iteration, "late window is incomplete")
    early_mask = (iteration_array >= 1) & (iteration_array <= 59)
    late_mask = (iteration_array >= early_end) & (iteration_array <= terminal)
    endpoint_suffixes = [f"positive_slope_{start}_{stop}" for start, stop in phase_segments]
    endpoint_suffixes.extend(
        f"boundary_peak_{left}_{center}_{right}" for left, center, right in phase_boundary_windows
    )
    endpoint_suffixes.extend((f"late_growth_{early_end}_{terminal}", "late_vs_early_median_growth"))
    endpoint_names = [
        f"{metric}:{suffix}"
        for metric in ("centroid_separation", "lane_variance_excess_rms")
        for suffix in endpoint_suffixes
    ]
    _require(
        isinstance(primary_family_endpoints, dict)
        and len(primary_family_endpoints) == 2
        and len(set(primary_family_endpoints.values())) == 2,
        "no-growth acceptance requires exactly two distinct primary families",
    )
    _require(0.0 < float(familywise_alpha) < 1.0, "no-growth familywise alpha is invalid")
    _require(
        float(materiality_loo_multiplier) > 0.0,
        "no-growth materiality LOO multiplier must be positive",
    )
    _require(
        float(normalized_numerical_noise_floor) > 0.0,
        "no-growth normalized numerical-noise floor must be positive",
    )
    diagnostic_rows = []
    centroid_endpoint_rows = []
    variance_gap_endpoint_rows = []
    centroid_primary_rows = []
    variance_gap_primary_rows = []
    centroid_separations = []
    lane_variance_excesses = []

    primary_suffixes = [f"phase_excursion_{start}_{stop}" for start, stop in phase_segments]
    primary_suffixes.extend(
        f"boundary_excursion_{left}_{center}_{right}"
        for left, center, right in phase_boundary_windows
    )
    primary_suffixes.append(f"late_excursion_{early_end}_{terminal}")
    primary_reference_iterations: dict[str, tuple[int, ...]] = {}
    for start, stop in phase_segments:
        reference = int(start) - 1 if int(start) - 1 in by_iteration else int(start)
        primary_reference_iterations[f"phase_excursion_{start}_{stop}"] = (reference,)
    for left, center, right in phase_boundary_windows:
        primary_reference_iterations[f"boundary_excursion_{left}_{center}_{right}"] = (
            int(left),
            int(right),
        )
    primary_reference_iterations[f"late_excursion_{early_end}_{terminal}"] = (early_end,)

    def growth_endpoints(series: np.ndarray) -> list[float]:
        endpoints = []
        for start, stop in phase_segments:
            mask = (iteration_array >= int(start)) & (iteration_array <= int(stop))
            _require(int(np.count_nonzero(mask)) >= 2, f"phase segment {start}--{stop} is incomplete")
            endpoints.append(max(_linear_slope(iteration_array[mask], series[mask]), 0.0))
        for left, center, right in phase_boundary_windows:
            _require(
                all(int(value) in by_iteration for value in (left, center, right)),
                "phase boundary is incomplete",
            )
            endpoints.append(
                max(
                    float(series[by_iteration[center]])
                    - max(float(series[by_iteration[left]]), float(series[by_iteration[right]])),
                    0.0,
                )
            )
        endpoints.append(
            max(float(series[by_iteration[terminal]]) - float(series[by_iteration[early_end]]), 0.0)
        )
        endpoints.append(max(float(np.median(series[late_mask])) - float(np.median(series[early_mask])), 0.0))
        return endpoints

    def primary_excursion_endpoints(series: np.ndarray) -> list[float]:
        endpoints = []
        for start, stop in phase_segments:
            mask = (iteration_array >= int(start)) & (iteration_array <= int(stop))
            reference = int(start) - 1 if int(start) - 1 in by_iteration else int(start)
            peak = float(np.max(series[mask]))
            endpoints.append(_positive_energy_increment(peak, float(series[by_iteration[reference]])))
        for left, center, right in phase_boundary_windows:
            reference = max(
                float(series[by_iteration[int(left)]]),
                float(series[by_iteration[int(right)]]),
            )
            endpoints.append(
                _positive_energy_increment(float(series[by_iteration[int(center)]]), reference)
            )
        endpoints.append(
            _positive_energy_increment(
                float(np.max(series[late_mask])),
                float(series[by_iteration[early_end]]),
            )
        )
        return endpoints

    for selected in assignments:
        complement = tuple(index for index in all_indices if index not in selected)
        weight = np.zeros(count, dtype=np.float64)
        weight[list(complement)] = 1.0 / len(complement)
        weight[list(selected)] = -1.0 / len(selected)
        separation = np.sqrt(np.maximum(np.einsum("i,tij,j->t", weight, grams, weight), 0.0))
        serial_variance = np.asarray(
            [_group_scatter(gram, selected) / max(len(selected) - 1, 1) for gram in grams],
            dtype=np.float64,
        )
        lane_variance = np.asarray(
            [_group_scatter(gram, complement) / max(len(complement) - 1, 1) for gram in grams],
            dtype=np.float64,
        )
        lane_variance_excess = np.maximum(np.sqrt(lane_variance) - np.sqrt(serial_variance), 0.0)
        variance_gap = np.abs(np.sqrt(lane_variance) - np.sqrt(serial_variance))
        centroid_endpoints = growth_endpoints(separation)
        lane_variance_endpoints = growth_endpoints(lane_variance_excess)
        variance_gap_endpoints = growth_endpoints(variance_gap)
        centroid_primary = primary_excursion_endpoints(separation)
        variance_gap_primary = primary_excursion_endpoints(variance_gap)
        diagnostic_rows.append(centroid_endpoints + lane_variance_endpoints)
        centroid_endpoint_rows.append(centroid_endpoints)
        variance_gap_endpoint_rows.append(variance_gap_endpoints)
        centroid_primary_rows.append(centroid_primary)
        variance_gap_primary_rows.append(variance_gap_primary)
        centroid_separations.append(separation)
        lane_variance_excesses.append(lane_variance_excess)
    endpoint_values = np.asarray(diagnostic_rows, dtype=np.float64)
    location = np.mean(endpoint_values, axis=0)
    scale = np.std(endpoint_values, axis=0, ddof=1)
    standardized = np.divide(
        endpoint_values - location,
        scale,
        out=np.zeros_like(endpoint_values),
        where=scale > 0.0,
    )
    joint = np.max(standardized, axis=1)
    observed_row = assignments.index(observed)
    observed_joint = float(joint[observed_row])
    p_value = float(np.count_nonzero(joint >= observed_joint) / len(joint))
    endpoint_report = {}
    for column, name in enumerate(endpoint_names):
        values = endpoint_values[:, column]
        observed_value = float(values[observed_row])
        endpoint_report[name] = {
            "observed": observed_value,
            "blocked_null_p95": float(np.quantile(values, 0.95, method="higher")),
            "one_sided_permutation_p": float(np.count_nonzero(values >= observed_value) / len(values)),
        }
    primary_catalog: dict[str, np.ndarray] = {}
    for metric, values in (
        ("centroid_separation", np.asarray(centroid_endpoint_rows, dtype=np.float64)),
        ("between_group_variance_rms_gap", np.asarray(variance_gap_endpoint_rows, dtype=np.float64)),
    ):
        for column, suffix in enumerate(endpoint_suffixes):
            primary_catalog[f"{metric}:{suffix}"] = values[:, column]
    primary_endpoint_catalog = {
        "centroid_separation": np.asarray(centroid_primary_rows, dtype=np.float64),
        "between_group_variance_rms_gap": np.asarray(variance_gap_primary_rows, dtype=np.float64),
    }
    for metric, values in primary_endpoint_catalog.items():
        primary_catalog[f"{metric}:max_declared_excursion"] = np.max(values, axis=1)
    unknown_primary = sorted(set(primary_family_endpoints.values()).difference(primary_catalog))
    _require(not unknown_primary, f"unknown no-growth primary endpoints: {unknown_primary}")

    materiality_by_endpoint = {}
    for suffix in primary_suffixes:
        reference_iterations = primary_reference_iterations[suffix]
        serial_radius = max(
            _leave_one_out_radius(grams[by_iteration[iteration]], observed)
            for iteration in reference_iterations
        )
        materiality_by_endpoint[suffix] = {
            "reference_iterations": list(reference_iterations),
            "serial_control_leave_one_out_radius": serial_radius,
            "threshold": max(
                float(materiality_loo_multiplier) * serial_radius,
                float(normalized_numerical_noise_floor),
            ),
        }

    assignment_rows = {selected: index for index, selected in enumerate(assignments)}
    complement_rows = np.asarray(
        [assignment_rows[tuple(index for index in all_indices if index not in selected)] for selected in assignments],
        dtype=np.int64,
    )
    for endpoint_name in primary_family_endpoints.values():
        values = primary_catalog[endpoint_name]
        _require(
            np.array_equal(values, values[complement_rows]),
            f"no-growth primary endpoint is not label/complement symmetric: {endpoint_name}",
        )

    per_family_alpha = float(familywise_alpha) / len(primary_family_endpoints)
    primary_families = {}
    for family, endpoint_name in primary_family_endpoints.items():
        values = primary_catalog[endpoint_name]
        observed_value = float(values[observed_row])
        family_p = float(np.count_nonzero(values >= observed_value) / len(values))
        statistically_directional = family_p < per_family_alpha
        metric = endpoint_name.split(":", 1)[0]
        endpoint_values_for_metric = primary_endpoint_catalog[metric]
        observed_endpoint_values = endpoint_values_for_metric[observed_row]
        endpoint_materiality = {}
        for column, suffix in enumerate(primary_suffixes):
            endpoint_observed = float(observed_endpoint_values[column])
            threshold = float(materiality_by_endpoint[suffix]["threshold"])
            endpoint_materiality[suffix] = {
                "observed": endpoint_observed,
                **materiality_by_endpoint[suffix],
                "drives_family_max": endpoint_observed == observed_value,
                "materially_larger_than_repeat_noise": endpoint_observed > threshold,
            }
        materially_larger_than_repeat_noise = any(
            row["drives_family_max"] and row["materially_larger_than_repeat_noise"]
            for row in endpoint_materiality.values()
        )
        family_pass = not (statistically_directional and materially_larger_than_repeat_noise)
        primary_families[family] = {
            "endpoint": endpoint_name,
            "observed": observed_value,
            "blocked_null_p95": float(np.quantile(values, 0.95, method="higher")),
            "exact_blocked_one_sided_permutation_p": family_p,
            "bonferroni_alpha": per_family_alpha,
            "statistically_directional": statistically_directional,
            "materiality_endpoints": endpoint_materiality,
            "materially_larger_than_repeat_noise": materially_larger_than_repeat_noise,
            "pass": family_pass,
        }
    primary_pass = all(row["pass"] for row in primary_families.values())
    return {
        "policy": (
            "two predeclared label/complement-symmetric primary families use exact blocked "
            "permutation p-values over the maximum declared phase/boundary excursion with "
            "Bonferroni familywise control; rejection also requires excursion energy above a "
            "serial-control LOO envelope and a sealed numerical floor"
        ),
        "n_blocked_whole_trajectory_permutations": len(assignments),
        "phase_boundaries": [list(map(int, row)) for row in phase_boundary_windows],
        "observed_centroid_separation_by_iteration": centroid_separations[observed_row].tolist(),
        "observed_lane_variance_excess_rms_by_iteration": lane_variance_excesses[observed_row].tolist(),
        "endpoints": endpoint_report,
        "diagnostic_endpoint_count": len(endpoint_names),
        "observed_joint_max": observed_joint,
        "joint_max_one_sided_permutation_p": p_value,
        "joint_max_used_for_acceptance": False,
        "primary_familywise_gate": {
            "method": "bonferroni_two_exact_blocked_max_excursion_families_with_control_materiality",
            "familywise_alpha": float(familywise_alpha),
            "per_family_alpha": per_family_alpha,
            "permutation_statistic_label_and_complement_symmetric": True,
            "materiality": {
                "scale": "serial_control_leave_one_out_radius_by_declared_endpoint",
                "multiplier": float(materiality_loo_multiplier),
                "normalized_numerical_noise_floor": float(normalized_numerical_noise_floor),
                "candidate_arms_used_for_scale": False,
                "excursion_measure": "positive_sqrt_current_energy_minus_reference_energy",
                "by_endpoint": materiality_by_endpoint,
                "failure_requires_significance_and_materiality": True,
            },
            "families": primary_families,
            "pass": primary_pass,
        },
        "pass": primary_pass,
    }


def _panel_indices(labels: Sequence[str]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    serial = tuple(index for index, label in enumerate(labels) if label.startswith("control_serial_"))
    lanes = tuple(index for index, label in enumerate(labels) if label.startswith("combined_candidate_"))
    _require(len(serial) == len(lanes) and len(serial) >= 3, "true-200 panel requires balanced replicated arms")
    _require(len(serial) + len(lanes) == len(labels), "true-200 panel contains an unknown arm label")
    return serial, lanes


def analyze_map_panel_from_loader(
    load_maps: Callable[[int], Sequence[np.ndarray]],
    labels: Sequence[str],
    blocks: Sequence[Sequence[int]],
    iterations: Sequence[int],
    *,
    thresholds: dict[str, Any],
    loo_multiplier: float,
    exact_witness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze maps one checkpoint at a time so true-200 volumes are never retained."""

    labels = tuple(str(value) for value in labels)
    iteration_values = tuple(int(value) for value in iterations)
    serial, lanes = _panel_indices(labels)
    _require(len(labels) == len(serial) + len(lanes), "true-200 map panel is unbalanced")
    _require(iteration_values, "map panel has no iterations")
    serial_distance = np.zeros((len(iteration_values), len(serial)), dtype=np.float64)
    lane_distance = np.zeros((len(iteration_values), len(lanes)), dtype=np.float64)
    trajectory_gram = np.zeros((len(labels), len(labels)), dtype=np.float64)
    iteration_grams = np.zeros((len(iteration_values), len(labels), len(labels)), dtype=np.float64)
    checkpoints: list[dict[str, Any]] = []
    expected_shape: tuple[int, ...] | None = None

    for row_index, iteration in enumerate(iteration_values):
        values = [
            _finite_array(value, f"iteration {iteration} map {label}")
            for label, value in zip(labels, load_maps(iteration), strict=True)
        ]
        _require(len(values) == len(labels), f"iteration {iteration} loader returned the wrong number of maps")
        shapes = {tuple(value.shape) for value in values}
        _require(len(shapes) == 1, f"iteration {iteration} maps have different shapes: {sorted(shapes)}")
        shape = next(iter(shapes))
        _require(len(shape) == 3 and len(set(shape)) == 1, f"iteration {iteration} maps are not cubic volumes")
        if expected_shape is None:
            expected_shape = shape
        _require(shape == expected_shape, f"map shape changed at iteration {iteration}: {shape} != {expected_shape}")
        flattened = [value.reshape(-1) for value in values]
        deviations = np.stack([value - flattened[0] for value in flattened])
        deviation_gram = deviations @ deviations.T
        numerator_sq = np.maximum(
            np.diag(deviation_gram)[:, None] + np.diag(deviation_gram)[None, :] - 2.0 * deviation_gram,
            0.0,
        )
        norm_sq = np.asarray([float(np.vdot(value, value).real) for value in flattened], dtype=np.float64)
        denominator_sq = 0.5 * (norm_sq[:, None] + norm_sq[None, :])
        distances = np.zeros_like(numerator_sq)
        np.divide(
            np.sqrt(numerator_sq),
            np.sqrt(denominator_sq),
            out=distances,
            where=denominator_sq > 0.0,
        )
        zero_energy_mismatch = (denominator_sq == 0.0) & (numerator_sq > 0.0)
        _require(
            not np.any(zero_energy_mismatch),
            f"iteration {iteration} has a zero-energy map mismatch",
        )
        loo = _serial_loo_from_distances(
            distances,
            serial,
            lanes,
            multiplier=float(loo_multiplier),
        )
        serial_distance[row_index] = loo["serial_leave_one_out_nearest"]
        lane_distance[row_index] = loo["lane_nearest_serial"]
        checkpoints.append(
            {
                "iteration": iteration,
                **loo,
            }
        )

        pooled_norms = np.asarray([np.linalg.norm(value) for value in flattened])
        positive_norms = pooled_norms[pooled_norms > 0.0]
        _require(positive_norms.size, f"all maps have zero energy at iteration {iteration}")
        scale = float(np.median(positive_norms))
        checkpoints[-1]["pooled_all_arm_map_norm_scale"] = scale
        centering = np.eye(len(labels)) - np.ones((len(labels), len(labels))) / len(labels)
        iteration_grams[row_index] = centering @ deviation_gram @ centering / (scale * scale)
        trajectory_gram += iteration_grams[row_index] / len(iteration_values)

    symmetric_iteration_grams = 0.5 * (
        iteration_grams + np.swapaxes(iteration_grams, 1, 2)
    )
    iteration_squared_distances = np.maximum(
        np.diagonal(symmetric_iteration_grams, axis1=1, axis2=2)[:, :, None]
        + np.diagonal(symmetric_iteration_grams, axis1=1, axis2=2)[:, None, :]
        - 2.0 * symmetric_iteration_grams,
        0.0,
    )
    for matrix in iteration_squared_distances:
        np.fill_diagonal(matrix, 0.0)
    witness_coupled = None
    if exact_witness is not None:
        witness_coupled = classify_witness_coupled_map_trajectory(
            iteration_squared_distances,
            labels,
            exact_witness,
            maximum_normalized_distance=float(
                thresholds["witness_map_normalized_distance_at_most"]
            ),
        )
    permutation = _blocked_permutation_statistics(trajectory_gram, blocks, serial)
    observed = permutation["observed"]
    no_growth = classify_no_growth(
        iteration_grams,
        iteration_values,
        blocks=blocks,
        observed_serial_indices=serial,
        phase_segments=thresholds["phase_segments"],
        phase_boundary_windows=thresholds["phase_boundary_windows"],
        late_window=thresholds["late_window"],
        primary_family_endpoints=thresholds["no_growth_primary_family_endpoints"],
        familywise_alpha=float(thresholds["no_growth_familywise_alpha"]),
        materiality_loo_multiplier=float(thresholds["no_growth_materiality_loo_multiplier"]),
        normalized_numerical_noise_floor=float(
            thresholds["no_growth_normalized_numerical_noise_floor"]
        ),
    )
    gates = {
        "centroid_over_pooled_run_rms": observed["centroid_over_pooled_run_rms"] <= float(thresholds["centroid_max"]),
        "cross_within_median_distance_ratio": observed["cross_within_median_distance_ratio"]
        <= float(thresholds["cross_within_max"]),
        "lane_variance_inflation": observed["lane_variance_inflation"] <= float(thresholds["variance_inflation_max"]),
        "blocked_whole_trajectory_joint_permutation": permutation["joint_max_one_sided_permutation_p"]
        >= float(thresholds["permutation_p_min"]),
        "no_growth_against_serial_leave_one_out": no_growth["pass"],
    }
    if witness_coupled is not None:
        gates["joint_exact_witness_coupled_bounded_map_trajectory"] = witness_coupled["pass"]
    return {
        "map_shape": list(expected_shape or ()),
        "checkpoint_scale_policy": "median positive map norm pooled over every arm before label permutation",
        "iterations": list(iteration_values),
        "checkpoints": checkpoints,
        "pointwise_serial_leave_one_out_diagnostic_only": {
            "all_lane_maps_inside_nearest_serial_radius": all(row["pass"] for row in checkpoints),
            "used_for_candidate_acceptance": False,
            "reason": "a high-multiplicity pointwise conjunction is not calibrated under an exchangeable repeat null",
        },
        "joint_exact_witness_coupled_map_trajectory": witness_coupled,
        "trajectory_permutation": permutation,
        "no_growth": no_growth,
        "gates": gates,
        "pass": all(gates.values()),
    }


def analyze_map_panel(
    maps: np.ndarray,
    labels: Sequence[str],
    blocks: Sequence[Sequence[int]],
    iterations: Sequence[int],
    *,
    thresholds: dict[str, Any],
    loo_multiplier: float = 1.0,
    exact_witness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Small-array adapter used by focused analyzer tests."""

    values = _finite_array(maps, "map panel")
    _require(values.ndim >= 3, "map panel must have arm, iteration, and feature dimensions")
    _require(values.shape[0] == len(labels), "map panel arm count differs from labels")
    _require(values.shape[1] == len(iterations), "map panel iteration count differs")
    return analyze_map_panel_from_loader(
        lambda iteration: values[:, tuple(iterations).index(iteration)],
        labels,
        blocks,
        iterations,
        thresholds=thresholds,
        loo_multiplier=loo_multiplier,
        exact_witness=exact_witness,
    )


def _has_nonfinite_numeric(value: Any) -> bool:
    """Return whether any numeric leaf is NaN or infinite.

    Exact-state metadata includes nested mappings, lists, and occasionally
    object arrays.  Checking only the outer NumPy dtype lets a non-finite
    numeric leaf inside an object container compare equal to itself, which can
    manufacture an exact serial witness.  Recurse through every supported
    container before performing exact equality.
    """

    if isinstance(value, Mapping):
        return any(
            _has_nonfinite_numeric(key) or _has_nonfinite_numeric(item)
            for key, item in value.items()
        )
    if isinstance(value, np.ndarray):
        if value.dtype.fields is not None or value.dtype == object:
            return any(_has_nonfinite_numeric(item) for item in value.reshape(-1).tolist())
        if np.issubdtype(value.dtype, np.number):
            return not bool(np.all(np.isfinite(value)))
        return False
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_has_nonfinite_numeric(item) for item in value)
    if isinstance(value, (np.number, int, float, complex)) and not isinstance(value, (bool, np.bool_)):
        try:
            return not bool(np.isfinite(value))
        except TypeError:
            return False
    return False


def _values_equal(left: Any, right: Any) -> bool:
    """Recursively compare exact state while rejecting every non-finite leaf."""

    if _has_nonfinite_numeric(left) or _has_nonfinite_numeric(right):
        return False
    if left is None or right is None:
        return left is right
    if isinstance(left, (str, bytes, bool, np.bool_)) or isinstance(
        right, (str, bytes, bool, np.bool_)
    ):
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if left.keys() != right.keys():
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
    except (TypeError, ValueError):
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False
    if left_array.shape != right_array.shape:
        return False
    if left_array.dtype == object or right_array.dtype == object:
        if left_array.ndim == 0:
            try:
                return bool(left == right)
            except (TypeError, ValueError):
                return False
        return all(
            _values_equal(left_item, right_item)
            for left_item, right_item in zip(
                left_array.reshape(-1).tolist(),
                right_array.reshape(-1).tolist(),
                strict=True,
            )
        )
    return bool(np.array_equal(left_array, right_array))


def _exact_any_serial(values: Sequence[Any], serial: Sequence[int], lanes: Sequence[int]) -> dict[str, Any]:
    serial_matches = [
        any(_values_equal(values[index], values[peer]) for peer in serial if peer != index) for index in serial
    ]
    lane_matches = [any(_values_equal(values[index], values[peer]) for peer in serial) for index in lanes]
    return {
        "serial_has_exact_peer": serial_matches,
        "lane_has_exact_serial": lane_matches,
        "pass": all(serial_matches) and all(lane_matches),
    }


def _exact_mismatch_squared_distances(values: Sequence[Any]) -> np.ndarray:
    """Encode exact checkpoint equality as a squared Hamming geometry."""

    count = len(values)
    _require(count >= 2, "exact-state trajectory requires at least two arms")
    mismatches = np.zeros((count, count), dtype=np.float64)
    for left, right in itertools.combinations(range(count), 2):
        mismatch = 0.0 if _values_equal(values[left], values[right]) else 1.0
        mismatches[left, right] = mismatch
        mismatches[right, left] = mismatch
    return mismatches


def classify_joint_exact_serial_witnesses(
    squared_mismatches_by_feature: dict[str, Sequence[np.ndarray]],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Build one exact complete-path witness matrix over every discrete feature.

    A witness pair must be exactly equal at every checkpoint for every
    declared metadata key and STAR identity column.  Taking one conjunction
    before looking for witnesses prevents checkpoint switching, cross-key
    chimeras, and matching a different serial control for each identity
    column.  No empirical radius is used for exact state.
    """

    labels = tuple(str(value) for value in labels)
    serial, lanes = _panel_indices(labels)
    _require(bool(squared_mismatches_by_feature), "joint exact trajectory has no declared features")
    joint_equal = np.ones((len(labels), len(labels)), dtype=bool)
    checkpoint_counts: dict[str, int] = {}
    for feature in sorted(squared_mismatches_by_feature):
        matrices = _finite_array(
            squared_mismatches_by_feature[feature],
            f"joint exact mismatch history {feature}",
        )
        _require(
            matrices.ndim == 3
            and matrices.shape[0] >= 1
            and matrices.shape[1:] == (len(labels), len(labels)),
            f"joint exact mismatch history {feature} has invalid layout",
        )
        _require(
            np.array_equal(matrices, np.swapaxes(matrices, 1, 2)),
            f"joint exact mismatch history {feature} is not symmetric",
        )
        _require(
            np.array_equal(
                np.diagonal(matrices, axis1=1, axis2=2),
                np.zeros((matrices.shape[0], len(labels)), dtype=matrices.dtype),
            ),
            f"joint exact mismatch history {feature} has a nonzero diagonal",
        )
        _require(
            np.all((matrices == 0.0) | (matrices == 1.0)),
            f"joint exact mismatch history {feature} is not binary",
        )
        joint_equal &= np.all(matrices == 0.0, axis=0)
        checkpoint_counts[feature] = int(matrices.shape[0])
    _require(
        len(set(checkpoint_counts.values())) == 1,
        "joint exact trajectory feature histories have different checkpoint counts",
    )
    np.fill_diagonal(joint_equal, True)

    def witnesses(index: int, allowed: Sequence[int], *, exclude_self: bool) -> list[int]:
        return [
            int(peer)
            for peer in allowed
            if (not exclude_self or peer != index) and bool(joint_equal[index, peer])
        ]

    serial_witness_indices = {
        labels[index]: witnesses(index, serial, exclude_self=True) for index in serial
    }
    lane_witness_indices = {
        labels[index]: witnesses(index, serial, exclude_self=False) for index in lanes
    }
    serial_has_peer = {
        label: bool(indices) for label, indices in serial_witness_indices.items()
    }
    lane_has_witness = {
        label: bool(indices) for label, indices in lane_witness_indices.items()
    }
    return {
        "policy": (
            "one exact complete-path equality conjunction over all checkpoints, "
            "metadata keys, and STAR identity columns; each candidate has one "
            "serial witness and each serial has a distinct complete-path serial peer"
        ),
        "feature_checkpoint_counts": checkpoint_counts,
        "checkpoint_count": next(iter(checkpoint_counts.values())),
        "complete_path_equal_matrix": joint_equal.tolist(),
        "serial_witness_indices": serial_witness_indices,
        "serial_witness_labels": {
            label: [labels[index] for index in indices]
            for label, indices in serial_witness_indices.items()
        },
        "lane_witness_indices": lane_witness_indices,
        "lane_witness_labels": {
            label: [labels[index] for index in indices]
            for label, indices in lane_witness_indices.items()
        },
        "serial_has_complete_path_peer": serial_has_peer,
        "lane_has_complete_path_serial_witness": lane_has_witness,
        "pass": all(serial_has_peer.values()) and all(lane_has_witness.values()),
    }


def classify_witness_coupled_map_trajectory(
    normalized_squared_distances_by_checkpoint: np.ndarray,
    labels: Sequence[str],
    exact_witness: dict[str, Any],
    *,
    maximum_normalized_distance: float,
) -> dict[str, Any]:
    """Require maps to stay numerically close to the same exact-state witness.

    Both the maximum checkpoint distance and the whole-path RMS distance must
    fit the sealed bound.  The bound is fixed by the contract, never learned
    from a potentially divergent serial panel, so unstable or scale-growing
    differences cannot enlarge their own acceptance envelope.
    """

    labels = tuple(str(value) for value in labels)
    serial, lanes = _panel_indices(labels)
    squared = _finite_array(
        normalized_squared_distances_by_checkpoint,
        "witness-coupled normalized map distances",
    )
    _require(
        squared.ndim == 3
        and squared.shape[0] >= 1
        and squared.shape[1:] == (len(labels), len(labels)),
        "witness-coupled map distances have invalid layout",
    )
    _require(
        np.array_equal(squared, np.swapaxes(squared, 1, 2)),
        "witness-coupled map distances are not symmetric",
    )
    _require(
        np.array_equal(
            np.diagonal(squared, axis1=1, axis2=2),
            np.zeros((squared.shape[0], len(labels)), dtype=squared.dtype),
        ),
        "witness-coupled map distance diagonal is nonzero",
    )
    _require(np.all(squared >= 0.0), "witness-coupled map distance is negative")
    tolerance = float(maximum_normalized_distance)
    _require(math.isfinite(tolerance) and tolerance > 0.0, "map witness tolerance is invalid")
    matrix = np.asarray(exact_witness.get("complete_path_equal_matrix"), dtype=bool)
    _require(
        matrix.shape == (len(labels), len(labels))
        and np.array_equal(matrix, matrix.T)
        and np.all(np.diag(matrix)),
        "joint exact witness matrix is invalid",
    )
    checkpoint_distances = np.sqrt(squared)
    rms = np.sqrt(np.mean(squared, axis=0, dtype=np.float64))
    maximum = np.max(checkpoint_distances, axis=0)

    def arm_row(index: int, allowed: Sequence[int], *, exclude_self: bool) -> dict[str, Any]:
        exact_peers = [
            int(peer)
            for peer in allowed
            if (not exclude_self or peer != index) and bool(matrix[index, peer])
        ]
        peer_rows = [
            {
                "index": peer,
                "label": labels[peer],
                "whole_trajectory_rms_normalized_distance": float(rms[index, peer]),
                "maximum_checkpoint_normalized_distance": float(maximum[index, peer]),
                "inside_sealed_bound": (
                    _inside(float(rms[index, peer]), tolerance)
                    and _inside(float(maximum[index, peer]), tolerance)
                ),
            }
            for peer in exact_peers
        ]
        accepted = [row for row in peer_rows if row["inside_sealed_bound"]]
        return {
            "exact_state_witnesses": peer_rows,
            "accepted_witness_labels": [row["label"] for row in accepted],
            "pass": bool(accepted),
        }

    serial_rows = {
        labels[index]: arm_row(index, serial, exclude_self=True) for index in serial
    }
    lane_rows = {
        labels[index]: arm_row(index, serial, exclude_self=False) for index in lanes
    }
    exact_pass = bool(exact_witness.get("pass"))
    return {
        "policy": (
            "map witnesses are restricted to the joint exact complete-path serial witness set; "
            "both every-checkpoint and trajectory-RMS normalized distances must fit one sealed bound"
        ),
        "checkpoint_count": int(squared.shape[0]),
        "maximum_normalized_distance": tolerance,
        "tolerance_source": "sealed fixed numerical-noise bound; not fitted from this panel",
        "joint_exact_witness_gate_pass": exact_pass,
        "serial_controls": serial_rows,
        "combined_candidates": lane_rows,
        "pass": (
            exact_pass
            and all(row["pass"] for row in serial_rows.values())
            and all(row["pass"] for row in lane_rows.values())
        ),
    }


def _numeric_serial_loo(
    values: Sequence[Any],
    serial: Sequence[int],
    lanes: Sequence[int],
    *,
    multiplier: float,
    distance: Callable[[Any, Any], float] = _relative_l2,
    trajectory_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    shapes = [tuple(np.asarray(value).shape) for value in values]
    serial_shapes = {shapes[index] for index in serial}
    if len(serial_shapes) != 1 or any(shapes[index] not in serial_shapes for index in lanes):
        return {
            "shapes": [list(value) for value in shapes],
            "failure_reason": "candidate or serial numeric state shapes differ",
            "pass": False,
        }
    distances = _distance_matrix(values, distance)
    result = _serial_loo_from_distances(
        distances,
        serial,
        lanes,
        multiplier=float(multiplier),
    )
    result["distance_matrix"] = distances.tolist()
    trajectory_values = [
        _finite_array(value, "continuous trajectory operand")
        for value in values
    ]
    if trajectory_transform is not None:
        trajectory_values = [
            _finite_array(trajectory_transform(value), "transformed continuous trajectory operand")
            for value in trajectory_values
        ]
    transformed_shapes = {tuple(value.shape) for value in trajectory_values}
    if len(transformed_shapes) != 1:
        result["trajectory_failure_reason"] = "transformed continuous-state operand shapes differ"
        result["pass"] = False
        return result
    squared = np.zeros((len(trajectory_values), len(trajectory_values)), dtype=np.float64)
    flattened = [value.reshape(-1) for value in trajectory_values]
    for left, right in itertools.combinations(range(len(flattened)), 2):
        delta = flattened[left] - flattened[right]
        value = float(np.vdot(delta, delta).real)
        _require(math.isfinite(value) and value >= 0.0, "continuous trajectory squared distance is invalid")
        squared[left, right] = value
        squared[right, left] = value
    result["trajectory_squared_euclidean_distance_matrix"] = squared.tolist()
    return result


def classify_metadata_iteration(
    metadata: Sequence[dict[str, Any]],
    labels: Sequence[str],
    state_contract: dict[str, Any],
) -> dict[str, Any]:
    """Classify one complete metadata checkpoint against serial repeats."""

    _require(len(metadata) == len(labels), "metadata panel arm count differs")
    serial, lanes = _panel_indices(labels)
    exact: dict[str, Any] = {}
    numeric: dict[str, Any] = {}
    for key in state_contract["joint_exact_complete_path_metadata_keys"]:
        _require(all(key in row for row in metadata), f"metadata is missing declared exact key {key}")
        exact[key] = _exact_any_serial([row[key] for row in metadata], serial, lanes)
    multiplier = float(state_contract["serial_leave_one_out_multiplier"])
    for key in state_contract["serial_leave_one_out_numeric_keys"]:
        _require(all(key in row for row in metadata), f"metadata is missing declared numeric key {key}")
        values = [_finite_array(row[key], f"metadata {key}") for row in metadata]
        numeric[key] = _numeric_serial_loo(
            values,
            serial,
            lanes,
            multiplier=multiplier,
        )
    return {
        "exact": exact,
        "numeric": numeric,
        "continuous_serial_loo_diagnostic_only_pass": all(row["pass"] for row in numeric.values()),
        "pass": all(row["pass"] for row in exact.values()),
    }


_HALFSET_PROFILE_RE = re.compile(r"^halfset_(?P<halfset>\d+)_profile_summary$")


def _validate_coarse_selector_profile_audits(
    metadata: dict[str, Any],
    *,
    label: str,
    iteration: int,
    multistream_workers: int,
    native_atomic_reduction: int,
) -> list[dict[str, Any]]:
    """Prove the configured coarse selector actually ran at one checkpoint."""

    from recovar.em.dense_single_volume.helpers.significance import (
        _validate_coarse_selector_audit,
    )

    profile_keys = sorted(
        key for key in metadata if _HALFSET_PROFILE_RE.fullmatch(str(key))
    )
    expected_profile_keys = [
        "halfset_0_profile_summary",
        "halfset_1_profile_summary",
    ]
    _require(
        profile_keys == expected_profile_keys,
        f"{label} iteration {iteration} halfset profile topology differs: "
        f"expected={expected_profile_keys}, observed={profile_keys}",
    )
    multistream = int(multistream_workers) > 0
    expected_wrapper = (
        "relion_coarse_diff2_projector_multistream_f32"
        if multistream
        else "relion_coarse_diff2_projector_f32"
    )
    expected_target = (
        "cuda_relion_coarse_diff2_projector_multistream_f32"
        if multistream
        else "cuda_relion_coarse_diff2_projector_f32"
    )
    expected_fields = {
        "score_mode": "gaussian",
        "requested_fused": True,
        "effective_fused": True,
        "requested_workers": int(multistream_workers),
        "effective_workers": int(multistream_workers),
        "requested_atomic": bool(native_atomic_reduction),
        "effective_atomic": bool(native_atomic_reduction),
        "wrapper": expected_wrapper,
        "target": expected_target,
    }
    rows = []
    for key in profile_keys:
        profile = metadata[key]
        _require(isinstance(profile, dict), f"{label} iteration {iteration} {key} is invalid")
        try:
            audit = _validate_coarse_selector_audit(profile.get("coarse_selector_audit"))
        except (TypeError, ValueError) as exc:
            raise GateSetupError(
                f"{label} iteration {iteration} {key} coarse selector audit is invalid: {exc}"
            ) from exc
        mismatches = {
            name: {"expected": expected, "observed": audit.get(name)}
            for name, expected in expected_fields.items()
            if audit.get(name) != expected
        }
        _require(
            not mismatches,
            f"{label} iteration {iteration} {key} coarse selector differs: {mismatches}",
        )
        _require(
            audit["translation_count"] == int(metadata["n_translations"]),
            f"{label} iteration {iteration} {key} selector translation count differs",
        )
        counts = audit["counts"]
        _require(
            counts["fused_calls"] > 0
            and counts["actual_rows"] > 0
            and counts["multistream_calls"]
            == (counts["fused_calls"] if multistream else 0)
            and counts["native_atomic_selected_calls"]
            == (counts["fused_calls"] if native_atomic_reduction else 0),
            f"{label} iteration {iteration} {key} selector execution counts are invalid",
        )
        match = _HALFSET_PROFILE_RE.fullmatch(key)
        _require(match is not None, f"{label} iteration {iteration} profile key is invalid")
        rows.append(
            {
                "label": label,
                "iteration": int(iteration),
                "halfset": int(match.group("halfset")),
                "audit": audit,
            }
        )
    return rows


def _particle_table_values(table: Any, state_contract: dict[str, Any], label: str) -> dict[str, Any]:
    identity_columns = tuple(str(value) for value in state_contract["star_identity_columns"])
    _require(
        identity_columns == ("rlnImageName", "rlnRandomSubset"),
        "particle STAR identity contract must be rlnImageName plus rlnRandomSubset",
    )
    numeric_groups = state_contract["star_numeric_groups"]
    required = set(identity_columns)
    for columns in numeric_groups.values():
        required.update(str(value) for value in columns)
    missing = sorted(required.difference(table.columns))
    _require(not missing, f"{label} particle STAR is missing columns {missing}")
    identity_series = table["rlnImageName"]
    raw_identities = identity_series.to_numpy()
    identities = np.asarray([str(value) for value in raw_identities], dtype=str)
    _require(
        not bool(identity_series.isna().any())
        and not _has_nonfinite_numeric(raw_identities)
        and all(value.strip() for value in identities),
        f"{label} has an empty or non-finite image identity",
    )
    _require(
        len(set(identities.tolist())) == identities.size,
        f"{label} has duplicate image identities",
    )
    try:
        random_subset_numeric = table["rlnRandomSubset"].astype(float).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GateSetupError(f"{label} rlnRandomSubset is not numeric") from exc
    _require(
        np.all(np.isfinite(random_subset_numeric)),
        f"{label} rlnRandomSubset is non-finite",
    )
    _require(
        np.all(random_subset_numeric == np.floor(random_subset_numeric)),
        f"{label} rlnRandomSubset is not integral",
    )
    random_subsets = random_subset_numeric.astype(np.int64)
    _require(
        set(random_subsets.tolist()).issubset({1, 2}),
        f"{label} rlnRandomSubset is outside the RELION halfset domain {{1, 2}}",
    )
    order = np.argsort(identities, kind="stable")
    result: dict[str, Any] = {
        "identities": identities[order],
        "identity_columns": {
            "rlnImageName": identities[order],
            "rlnRandomSubset": random_subsets[order],
        },
        "numeric_groups": {},
    }
    for group, columns in numeric_groups.items():
        try:
            values = table[list(columns)].astype(float).to_numpy(dtype=np.float64)[order]
        except (TypeError, ValueError) as exc:
            raise GateSetupError(f"{label} STAR group {group} is not numeric") from exc
        _require(np.all(np.isfinite(values)), f"{label} STAR group {group} is non-finite")
        result["numeric_groups"][group] = values
    return result


def _rotation_rms_deg(left: np.ndarray, right: np.ndarray) -> float:
    values = _angular_error_deg(left, right)
    return float(np.sqrt(np.mean(np.square(values))))


def _translation_rms_angstrom(left: np.ndarray, right: np.ndarray) -> float:
    delta = _finite_array(left, "left translation") - _finite_array(right, "right translation")
    return float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))


def classify_particle_tables(
    tables: Sequence[Any],
    labels: Sequence[str],
    state_contract: dict[str, Any],
) -> dict[str, Any]:
    """Gate particle identity, halfset, pose, translation, and support state."""

    _require(len(tables) == len(labels), "particle STAR panel arm count differs")
    serial, lanes = _panel_indices(labels)
    parsed = [
        _particle_table_values(table, state_contract, str(label)) for table, label in zip(tables, labels, strict=True)
    ]
    exact: dict[str, Any] = {}
    for column in state_contract["star_identity_columns"]:
        exact[column] = _exact_any_serial([row["identity_columns"][column] for row in parsed], serial, lanes)
    multiplier = float(state_contract["serial_leave_one_out_multiplier"])
    groups: dict[str, Any] = {}
    distance_functions: dict[str, Callable[[Any, Any], float]] = {
        "rotation_euler_deg": _rotation_rms_deg,
        "translation_angstrom": _translation_rms_angstrom,
        "support_probability": _relative_l2,
    }
    from recovar.em.sampling import _relion_euler_angles_to_matrix

    trajectory_transforms: dict[str, Callable[[np.ndarray], np.ndarray] | None] = {
        "rotation_euler_deg": _relion_euler_angles_to_matrix,
        "translation_angstrom": None,
        "support_probability": None,
    }
    for group in state_contract["star_numeric_groups"]:
        _require(group in distance_functions, f"no declared STAR distance for {group}")
        _require(group in trajectory_transforms, f"no Euclidean trajectory embedding for STAR group {group}")
        groups[group] = _numeric_serial_loo(
            [row["numeric_groups"][group] for row in parsed],
            serial,
            lanes,
            multiplier=multiplier,
            distance=distance_functions[group],
            trajectory_transform=trajectory_transforms[group],
        )
    return {
        "identity": exact,
        "numeric_groups": groups,
        "continuous_serial_loo_diagnostic_only_pass": all(row["pass"] for row in groups.values()),
        "pass": all(row["pass"] for row in exact.values()),
    }


def _squared_euclidean_distances_to_gram(squared_distances: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Recover a centered Gram matrix without projecting or clipping its spectrum."""

    squared = _finite_array(squared_distances, "squared Euclidean distance matrix")
    _require(squared.ndim == 2 and squared.shape[0] == squared.shape[1], "squared distance is not square")
    _require(np.array_equal(squared, squared.T), "squared distance matrix is not exactly symmetric")
    _require(np.array_equal(np.diag(squared), np.zeros(squared.shape[0])), "squared distance diagonal is nonzero")
    _require(np.all(squared >= 0.0), "squared distance matrix contains a negative value")
    count = squared.shape[0]
    centering = np.eye(count) - np.ones((count, count), dtype=np.float64) / count
    gram = -0.5 * centering @ squared @ centering
    gram = (gram + gram.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(gram)
    negative_energy = float(-np.sum(eigenvalues[eigenvalues < 0.0]))
    positive_energy = float(np.sum(eigenvalues[eigenvalues > 0.0]))
    total_energy = positive_energy + negative_energy
    negative_energy_fraction = 0.0 if total_energy == 0.0 else negative_energy / total_energy
    reconstructed = np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram
    denominator = float(np.linalg.norm(squared))
    reconstruction_relative_l2 = float(np.linalg.norm(reconstructed - squared)) / max(
        denominator,
        np.finfo(np.float64).tiny,
    )
    roundoff_limit = 4096.0 * np.finfo(np.float64).eps * max(count, 1)
    _require(
        negative_energy_fraction <= roundoff_limit and reconstruction_relative_l2 <= roundoff_limit,
        "continuous-state distances are not squared-Euclidean within float64 roundoff",
    )
    return gram, {
        "negative_energy": negative_energy,
        "positive_energy": positive_energy,
        "negative_energy_fraction": negative_energy_fraction,
        "distance_reconstruction_relative_l2": reconstruction_relative_l2,
        "roundoff_limit": roundoff_limit,
    }


def _direct_sum_gram(
    squared_distances_by_feature: dict[str, np.ndarray],
    feature_scales: dict[str, float],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Build the normalized feature direct sum from squared-Euclidean factors."""

    _require(set(squared_distances_by_feature) == set(feature_scales), "direct-sum feature topology differs")
    _require(bool(squared_distances_by_feature), "direct-sum geometry has no features")
    grams = []
    diagnostics: dict[str, dict[str, float]] = {}
    for name in sorted(squared_distances_by_feature):
        scale = float(feature_scales[name])
        _require(math.isfinite(scale) and scale > 0.0, f"direct-sum feature scale is invalid for {name}")
        gram, report = _squared_euclidean_distances_to_gram(
            squared_distances_by_feature[name] / (scale * scale)
        )
        grams.append(gram)
        diagnostics[name] = report
    return np.mean(np.stack(grams), axis=0), diagnostics


def classify_continuous_state_trajectory(
    iteration_rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    blocks: Sequence[Sequence[int]],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Jointly gate continuous state without a high-multiplicity pointwise conjunction."""

    serial, _ = _panel_indices(labels)
    iterations = []
    incomplete = []
    diagnostic_failures = 0
    per_iteration: list[dict[str, np.ndarray]] = []
    feature_history: dict[str, list[np.ndarray]] = {}
    for row in iteration_rows:
        matrices: dict[str, np.ndarray] = {}
        groups = [(f"metadata:{name}", result) for name, result in row["metadata"]["numeric"].items()]
        groups += [(f"particle_star:{name}", result) for name, result in row["particle_star"]["numeric_groups"].items()]
        for name, result in groups:
            matrix = result.get("trajectory_squared_euclidean_distance_matrix")
            if matrix is None:
                incomplete.append({"iteration": row["iteration"], "group": name})
                continue
            squared_distances = _finite_array(
                matrix,
                f"iteration {row['iteration']} state squared-Euclidean distance {name}",
            )
            matrices[name] = squared_distances
            feature_history.setdefault(name, []).append(squared_distances)
            if not result["pass"]:
                diagnostic_failures += 1
        if incomplete:
            continue
        _require(matrices, f"iteration {row['iteration']} has no continuous state evidence")
        per_iteration.append(matrices)
        iterations.append(int(row["iteration"]))
    if incomplete:
        return {
            "incomplete_numeric_evidence": incomplete,
            "pointwise_loo_diagnostic_failure_count": diagnostic_failures,
            "pass": False,
        }
    scales = {}
    for name, matrices in feature_history.items():
        upper_values = np.concatenate(
            [np.sqrt(matrix[np.triu_indices(matrix.shape[0], 1)]) for matrix in matrices]
        )
        positive = upper_values[upper_values > 0.0]
        scales[name] = float(np.median(positive)) if positive.size else 1.0
    iteration_grams = []
    geometry_diagnostics: dict[str, dict[str, float]] = {}
    for matrices in per_iteration:
        _require(set(matrices) == set(scales), "continuous state feature topology changed")
        combined_gram, iteration_geometry = _direct_sum_gram(matrices, scales)
        iteration_grams.append(combined_gram)
        for name, row in iteration_geometry.items():
            aggregate = geometry_diagnostics.setdefault(
                name,
                {
                    "maximum_negative_energy": 0.0,
                    "maximum_negative_energy_fraction": 0.0,
                    "maximum_distance_reconstruction_relative_l2": 0.0,
                    "roundoff_limit": row["roundoff_limit"],
                },
            )
            aggregate["maximum_negative_energy"] = max(aggregate["maximum_negative_energy"], row["negative_energy"])
            aggregate["maximum_negative_energy_fraction"] = max(
                aggregate["maximum_negative_energy_fraction"],
                row["negative_energy_fraction"],
            )
            aggregate["maximum_distance_reconstruction_relative_l2"] = max(
                aggregate["maximum_distance_reconstruction_relative_l2"],
                row["distance_reconstruction_relative_l2"],
            )
    grams = np.stack(iteration_grams)
    aggregate = np.mean(grams, axis=0)
    permutation = _blocked_permutation_statistics(aggregate, blocks, serial)
    observed = permutation["observed"]
    no_growth = classify_no_growth(
        grams,
        iterations,
        blocks=blocks,
        observed_serial_indices=serial,
        phase_segments=thresholds["phase_segments"],
        phase_boundary_windows=thresholds["phase_boundary_windows"],
        late_window=thresholds["late_window"],
        primary_family_endpoints=thresholds["no_growth_primary_family_endpoints"],
        familywise_alpha=float(thresholds["no_growth_familywise_alpha"]),
        materiality_loo_multiplier=float(thresholds["no_growth_materiality_loo_multiplier"]),
        normalized_numerical_noise_floor=float(
            thresholds["no_growth_normalized_numerical_noise_floor"]
        ),
    )
    gates = {
        "centroid_over_pooled_run_rms": observed["centroid_over_pooled_run_rms"] <= float(thresholds["centroid_max"]),
        "cross_within_median_distance_ratio": observed["cross_within_median_distance_ratio"]
        <= float(thresholds["cross_within_max"]),
        "lane_variance_inflation": observed["lane_variance_inflation"] <= float(thresholds["variance_inflation_max"]),
        "blocked_joint_permutation": permutation["joint_max_one_sided_permutation_p"]
        >= float(thresholds["permutation_p_min"]),
        "no_growth": no_growth["pass"],
    }
    return {
        "policy": (
            "continuous per-checkpoint LOO is diagnostic; acceptance uses a normalized direct sum "
            "of trajectory-specific squared-Euclidean features without PSD clipping"
        ),
        "trajectory_feature_scales": scales,
        "direct_sum_geometry": geometry_diagnostics,
        "pointwise_loo_diagnostic_failure_count": diagnostic_failures,
        "trajectory_permutation": permutation,
        "no_growth": no_growth,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _one_sided_quality_gate(
    serial_values: Sequence[float],
    lane_values: Sequence[float],
    *,
    higher_is_better: bool,
    outer_degradation: float = 0.0,
) -> dict[str, Any]:
    serial = _finite_array(serial_values, "serial quality values").reshape(-1)
    lanes = _finite_array(lane_values, "lane quality values").reshape(-1)
    _require(
        serial.size == lanes.size and serial.size >= 3,
        "quality gate requires balanced replicated values",
    )
    serial_worst = float(np.min(serial) if higher_is_better else np.max(serial))
    if higher_is_better:
        outer = serial_worst - float(outer_degradation)
        passed = bool(np.all(lanes >= np.nextafter(outer, -math.inf)))
        lane_worst = float(np.min(lanes))
    else:
        outer = serial_worst + float(outer_degradation)
        passed = bool(np.all(lanes <= np.nextafter(outer, math.inf)))
        lane_worst = float(np.max(lanes))
    return {
        "higher_is_better": higher_is_better,
        "serial_values": serial.tolist(),
        "lane_values": lanes.tolist(),
        "serial_worst": serial_worst,
        "outer_accepted_bound": outer,
        "maximum_outer_degradation": float(outer_degradation),
        "lane_worst": lane_worst,
        "pass": passed,
    }


def _two_sided_serial_envelope(serial_values: Sequence[float], lane_values: Sequence[float]) -> dict[str, Any]:
    serial = _finite_array(serial_values, "serial envelope values").reshape(-1)
    lanes = _finite_array(lane_values, "lane envelope values").reshape(-1)
    _require(
        serial.size == lanes.size and serial.size >= 3,
        "serial envelope requires balanced replicated values",
    )
    lower = float(np.min(serial))
    upper = float(np.max(serial))
    passed = bool(np.all(lanes >= np.nextafter(lower, -math.inf)) and np.all(lanes <= np.nextafter(upper, math.inf)))
    return {
        "serial_values": serial.tolist(),
        "lane_values": lanes.tolist(),
        "serial_min": lower,
        "serial_max": upper,
        "pass": passed,
    }


def classify_final_quality(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    blocks: Sequence[Sequence[int]],
    fsc_auc_outer_degradation: float,
    resolution_shell_outer_degradation: int,
    model_resolution_relative_degradation: float,
    scale_relative_deviation: float,
    permutation_p_min: float,
) -> dict[str, Any]:
    """Gate final quality with practical guards plus one blocked joint test."""

    _require(len(rows) == len(labels), "final quality panel arm count differs")
    serial, lanes = _panel_indices(labels)
    reference_names = tuple(sorted(rows[0]["references"]))
    _require(reference_names, "final quality panel has no frozen references")
    _require(
        all(tuple(sorted(row["references"])) == reference_names for row in rows),
        "final quality reference sets differ across arms",
    )
    references: dict[str, Any] = {}
    outer_gates: dict[str, bool] = {}
    for reference in reference_names:
        metrics = [row["references"][reference] for row in rows]
        auc = _one_sided_quality_gate(
            [metrics[index]["fsc_auc"] for index in serial],
            [metrics[index]["fsc_auc"] for index in lanes],
            higher_is_better=True,
            outer_degradation=float(fsc_auc_outer_degradation),
        )
        resolution_shell = _one_sided_quality_gate(
            [metrics[index]["resolution_shell_0p143"] for index in serial],
            [metrics[index]["resolution_shell_0p143"] for index in lanes],
            higher_is_better=True,
            outer_degradation=float(resolution_shell_outer_degradation),
        )
        serial_scale = _finite_array(
            [metrics[index]["global_scale_candidate_to_reference"] for index in serial],
            f"{reference} serial scale",
        )
        lane_scale = _finite_array(
            [metrics[index]["global_scale_candidate_to_reference"] for index in lanes],
            f"{reference} lane scale",
        )
        scale_center = float(np.median(serial_scale))
        _require(scale_center > 0.0, f"{reference} serial scale center is invalid")
        serial_relative = np.abs(serial_scale / scale_center - 1.0)
        accepted_scale_deviation = max(float(np.max(serial_relative)), float(scale_relative_deviation))
        lane_relative = np.abs(lane_scale / scale_center - 1.0)
        scale = {
            "serial_values": serial_scale.tolist(),
            "lane_values": lane_scale.tolist(),
            "serial_median": scale_center,
            "serial_maximum_relative_deviation": float(np.max(serial_relative)),
            "predeclared_practical_relative_deviation": float(scale_relative_deviation),
            "accepted_relative_deviation": accepted_scale_deviation,
            "lane_maximum_relative_deviation": float(np.max(lane_relative)),
            "pass": bool(np.all(lane_relative <= np.nextafter(accepted_scale_deviation, math.inf))),
        }
        references[reference] = {
            "fsc_auc": auc,
            "resolution_shell_0p143": resolution_shell,
            "resolution_angstrom_0p143_values": [metric["resolution_angstrom_0p143"] for metric in metrics],
            "global_scale_candidate_to_reference": scale,
        }
        outer_gates[f"{reference}:fsc_auc"] = auc["pass"]
        outer_gates[f"{reference}:resolution_shell_0p143"] = resolution_shell["pass"]
        outer_gates[f"{reference}:global_scale"] = scale["pass"]
    serial_model = _finite_array(
        [rows[index]["model_resolution_angstrom"] for index in serial], "serial model resolution"
    )
    lane_model = _finite_array([rows[index]["model_resolution_angstrom"] for index in lanes], "lane model resolution")
    model_bound = float(np.max(serial_model)) * (1.0 + float(model_resolution_relative_degradation))
    model_resolution = {
        "serial_values": serial_model.tolist(),
        "lane_values": lane_model.tolist(),
        "serial_worst_angstrom": float(np.max(serial_model)),
        "relative_outer_degradation": float(model_resolution_relative_degradation),
        "accepted_upper_bound_angstrom": model_bound,
        "pass": bool(np.all(lane_model <= np.nextafter(model_bound, math.inf))),
    }
    outer_gates["model_resolution_angstrom"] = model_resolution["pass"]

    assignments = _balanced_assignments(blocks)
    observed = tuple(sorted(serial))
    _require(observed in assignments, "observed final-quality labels do not satisfy blocked design")
    all_indices = tuple(range(len(labels)))
    harmful_rows = []
    endpoint_names = []
    for reference in reference_names:
        endpoint_names.extend(
            (
                f"{reference}:fsc_auc_degradation",
                f"{reference}:resolution_shell_degradation",
                f"{reference}:scale_location_shift",
            )
        )
    endpoint_names.append("model_resolution_degradation")
    for selected in assignments:
        complement = tuple(index for index in all_indices if index not in selected)
        endpoints = []
        for reference in reference_names:
            metrics = [row["references"][reference] for row in rows]
            auc_values = np.asarray([metric["fsc_auc"] for metric in metrics], dtype=np.float64)
            shell_values = np.asarray([metric["resolution_shell_0p143"] for metric in metrics], dtype=np.float64)
            scale_values = np.asarray(
                [metric["global_scale_candidate_to_reference"] for metric in metrics], dtype=np.float64
            )
            endpoints.extend(
                (
                    max(float(np.mean(auc_values[list(selected)])) - float(np.mean(auc_values[list(complement)])), 0.0),
                    max(
                        float(np.mean(shell_values[list(selected)])) - float(np.mean(shell_values[list(complement)])),
                        0.0,
                    ),
                    abs(float(np.mean(scale_values[list(complement)])) - float(np.mean(scale_values[list(selected)]))),
                )
            )
        model_values = np.asarray([row["model_resolution_angstrom"] for row in rows])
        endpoints.append(
            max(float(np.mean(model_values[list(complement)])) - float(np.mean(model_values[list(selected)])), 0.0)
        )
        harmful_rows.append(endpoints)
    values = np.asarray(harmful_rows, dtype=np.float64)
    location = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    standardized = np.divide(values - location, scale, out=np.zeros_like(values), where=scale > 0.0)
    joint = np.max(standardized, axis=1)
    observed_row = assignments.index(observed)
    observed_joint = float(joint[observed_row])
    p_value = float(np.count_nonzero(joint >= observed_joint) / len(joint))
    permutation = {
        "n_blocked_whole_run_permutations": len(assignments),
        "endpoint_names": endpoint_names,
        "observed_harmful_endpoints": values[observed_row].tolist(),
        "observed_joint_max": observed_joint,
        "joint_max_one_sided_permutation_p": p_value,
        "required_p_min": float(permutation_p_min),
        "pass": p_value >= float(permutation_p_min),
    }
    gates = {**outer_gates, "blocked_joint_quality_permutation": permutation["pass"]}
    return {
        "policy": (
            "One blocked joint whole-run permutation controls multiplicity; explicit outer guards "
            "catch material FSC-AUC, resolution, model-resolution, or scale regressions."
        ),
        "references": references,
        "model_resolution_angstrom": model_resolution,
        "blocked_joint_permutation": permutation,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _percent_change(candidate: float, control: float) -> float:
    _require(math.isfinite(candidate) and math.isfinite(control) and control > 0.0, "invalid runtime values")
    return 100.0 * (candidate / control - 1.0)


def classify_runtime(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    relion_wall_s: Sequence[float],
    *,
    wall_percent_max: float,
    expectation_percent_max: float,
    peak_memory_percent_max: float,
    relion_ratio_target: float,
) -> dict[str, Any]:
    """Classify replicated end-to-end, stage, memory, and absolute runtime."""

    _require(len(rows) == len(labels), "runtime panel arm count differs")
    serial, lanes = _panel_indices(labels)

    def median(key: str, indices: Sequence[int]) -> float:
        values = _finite_array([rows[index][key] for index in indices], f"runtime {key}")
        _require(np.all(values > 0.0), f"runtime {key} values must be positive")
        return float(np.median(values))

    metrics: dict[str, Any] = {}
    for key, threshold in (
        ("end_to_end_wall_s", wall_percent_max),
        ("expectation_stage_s", expectation_percent_max),
        ("peak_gpu_memory_mib", peak_memory_percent_max),
    ):
        serial_median = median(key, serial)
        lane_median = median(key, lanes)
        change = _percent_change(lane_median, serial_median)
        metrics[key] = {
            "serial_median": serial_median,
            "lane_median": lane_median,
            "lane_percent_change": change,
            "required_percent_change_at_most": float(threshold),
            "pass": change <= float(threshold),
        }
    reference = _finite_array(relion_wall_s, "RELION runtimes").reshape(-1)
    _require(reference.size == 4 and np.all(reference > 0.0), "four positive RELION runtimes are required")
    relion_median = float(np.median(reference))
    absolute_ratio = metrics["end_to_end_wall_s"]["lane_median"] / relion_median
    program_gate = {
        "relion_repeat_wall_s": reference.tolist(),
        "relion_median_wall_s": relion_median,
        "lane_to_relion_runtime_ratio": absolute_ratio,
        "program_target_at_most": float(relion_ratio_target),
        "pass": absolute_ratio <= float(relion_ratio_target),
        "independent_of_candidate_qualification": True,
    }
    candidate_gates = {
        "median_end_to_end_lane_gain": metrics["end_to_end_wall_s"]["pass"],
        "median_expectation_stage_lane_gain": metrics["expectation_stage_s"]["pass"],
        "median_peak_gpu_memory": metrics["peak_gpu_memory_mib"]["pass"],
    }
    return {
        "arms": list(rows),
        "replicated_metrics": metrics,
        "candidate_gates": candidate_gates,
        "candidate_pass": all(candidate_gates.values()),
        "absolute_relion_program_gate": program_gate,
    }


def _manifest_text(base: Path, paths: Sequence[Path]) -> str:
    rows = []
    for path in sorted((value.resolve() for value in paths), key=lambda value: str(value.relative_to(base.resolve()))):
        _require(path.is_file(), f"manifest input is missing: {path}")
        rows.append(f"{_sha256(path)}  {path.relative_to(base.resolve())}\n")
    return "".join(rows)


def _manifest_digest(base: Path, paths: Sequence[Path]) -> str:
    return hashlib.sha256(_manifest_text(base, paths).encode()).hexdigest()


def _validate_saved_manifest(path: Path, base: Path, expected_paths: Sequence[Path]) -> str:
    _require(path.is_file(), f"missing artifact manifest: {path}")
    expected = _manifest_text(base, expected_paths)
    try:
        observed = path.read_text()
    except OSError as exc:
        raise GateSetupError(f"cannot read artifact manifest {path}: {exc}") from exc
    _require(observed == expected, f"artifact manifest is stale or incomplete: {path}")
    return _sha256(path)


def _expected_iteration_artifacts(output: Path, iterations: Sequence[int], suffixes: Sequence[str]) -> list[Path]:
    return [output / f"run_it{int(iteration):03d}_{suffix}" for iteration in iterations for suffix in suffixes]


def _expected_runtime_artifacts(run_root: Path) -> list[Path]:
    """Files whose hashes make one completed arm immutable on resume."""

    return [
        run_root / "command.json",
        run_root / "timing.json",
        run_root / "gpu_monitor.csv",
        run_root / "process.time",
        run_root / "runner.stdout",
        run_root / "runner.stderr",
        run_root / "arm.json",
        run_root / "science_environment.json",
        run_root / "output" / "run_native_options.json",
    ]


def _validate_runtime_artifact_manifest(run_root: Path) -> str:
    return _validate_saved_manifest(
        run_root / "runtime_artifact_manifest.sha256",
        run_root,
        _expected_runtime_artifacts(run_root),
    )


def _validate_science_environment(
    path: Path,
    *,
    root: Path,
    run_root: Path,
    label: str,
    multistream_workers: int,
    native_atomic_reduction: int,
    arm: dict[str, Any],
    expected: dict[str, Any],
    runtime_contract: dict[str, Any],
) -> dict[str, str]:
    payload = _load_json(path, f"arm science environment for {label}")
    _require(
        payload.get("schema") == "recovar.vdam_coarse_combined_true200_science_environment.v2",
        f"arm {label} science-environment schema differs",
    )
    _require(payload.get("label") == label, f"arm {label} science-environment label differs")
    job_id = str(arm.get("job_id", ""))
    _require(job_id.isdigit() and payload.get("job_id") == job_id, f"arm {label} job ID is invalid")
    environment = payload.get("environment")
    _require(
        isinstance(environment, dict)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()),
        f"arm {label} science environment is not a string mapping",
    )
    try:
        from scripts.resolve_vdam_coarse_combined_true200_launch import (
            LaunchResolutionError,
            _science_environment_snapshot,
        )

        canonical = _science_environment_snapshot(environment, runtime_contract)
    except (LaunchResolutionError, KeyError, TypeError, ValueError) as exc:
        raise GateSetupError(f"arm {label} science environment is invalid: {exc}") from exc
    _require(canonical == environment, f"arm {label} science environment is incomplete or non-canonical")

    fixed = runtime_contract.get("fixed_science_environment")
    _require(isinstance(fixed, dict), "runtime contract has no fixed science environment")
    attempt_runtime = root / "runtime" / "attempts" / job_id
    required = {
        **{str(key): str(value) for key, value in fixed.items()},
        "CUDA_VISIBLE_DEVICES": str(expected["gpu_uuid"]),
        "JAX_COMPILATION_CACHE_DIR": str((run_root / "jax_cache").resolve()),
        "LD_PRELOAD": str(Path(expected["cusparse_path"]).resolve()),
        "PIXI_HOME": str((attempt_runtime / "pixi_home").resolve()),
        "PYTHONPATH": str(Path(expected["repo_root"]).resolve()),
        "RATTLER_CACHE_DIR": str((attempt_runtime / "rattler_cache").resolve()),
        "RECOVAR_CUDA_LIB": str(Path(expected["cuda_path"]).resolve()),
        "RECOVAR_EXPECTED_REPO_ROOT": str(Path(expected["repo_root"]).resolve()),
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": str(int(multistream_workers)),
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": str(int(native_atomic_reduction)),
        "RECOVAR_RELION_BIND_BUILD_DIR": str(Path(expected["relion_bind_path"]).resolve().parent),
        "RECOVAR_SELECTED_GPU_UUID": str(expected["gpu_uuid"]),
        "TMPDIR": str((run_root / "tmp").resolve()),
        "VDAM_ALLOCATED_GPU_UUIDS_CSV": str(expected["gpu_uuid"]),
        "VDAM_SELECTED_GPU_UUID": str(expected["gpu_uuid"]),
        "VDAM_VISIBLE_GPU_UUIDS_CSV": str(expected["visible_gpu_uuids_csv"]),
    }
    mismatches = {
        key: {"expected": value, "observed": environment.get(key)}
        for key, value in required.items()
        if environment.get(key) != value
    }
    _require(not mismatches, f"arm {label} effective science environment differs: {mismatches}")
    visible_gpu_uuids = environment.get("VDAM_VISIBLE_GPU_UUIDS_CSV", "").split(",")
    _require(
        visible_gpu_uuids
        and len(visible_gpu_uuids) == len(set(visible_gpu_uuids))
        and all(value.startswith("GPU-") for value in visible_gpu_uuids)
        and str(expected["gpu_uuid"]) in visible_gpu_uuids,
        f"arm {label} visible GPU UUID evidence is invalid",
    )
    if "VDAM_TRUE200_ROOT" in environment:
        _require(
            Path(environment["VDAM_TRUE200_ROOT"]).resolve() == root.resolve(),
            f"arm {label} VDAM_TRUE200_ROOT differs",
        )
    if "VDAM_TRUE200_RESUME" in environment:
        _require(
            environment["VDAM_TRUE200_RESUME"] in {"0", "1"},
            f"arm {label} VDAM_TRUE200_RESUME is invalid",
        )
    if "VDAM_GF46_FIXTURE_DIR" in environment:
        _require(
            Path(environment["VDAM_GF46_FIXTURE_DIR"]).resolve()
            == Path(expected["fixture_dir"]).resolve(),
            f"arm {label} fixture environment differs",
        )
    if "VDAM_NATIVE_REFERENCE_ROOT" in environment:
        _require(
            Path(environment["VDAM_NATIVE_REFERENCE_ROOT"]).resolve()
            == Path(expected["native_reference_root"]).resolve(),
            f"arm {label} native-reference environment differs",
        )
    if "VDAM_TRUE200_LAUNCH_MANIFEST" in environment:
        launch_manifest = Path(environment["VDAM_TRUE200_LAUNCH_MANIFEST"])
        _require(
            launch_manifest.is_file()
            and _sha256(launch_manifest) == expected["launch_manifest_sha256"],
            f"arm {label} external launch-manifest environment differs",
        )
    return environment


def _validate_arm_artifacts(
    root: Path,
    label: str,
    multistream_workers: int,
    native_atomic_reduction: int,
    iterations: Sequence[int],
    suffixes: Sequence[str],
    *,
    expected: dict[str, Any],
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    run_root = root / "runs" / label
    output = run_root / "output"
    _require((run_root / "SCIENCE_COMPLETED").is_file(), f"arm {label} did not complete science")
    _require(output.is_dir(), f"arm {label} has no output directory")
    runtime_manifest_sha = _validate_runtime_artifact_manifest(run_root)
    arm = _load_json(run_root / "arm.json", f"arm provenance for {label}")
    exact_fields = {
        "schema": "recovar.vdam_coarse_combined_true200_arm.v2",
        "label": label,
        "multistream_workers": int(multistream_workers),
        "native_atomic_reduction": int(native_atomic_reduction),
        "git_head": expected["git_head"],
        "production_candidate_head": expected["production_candidate_head"],
        "gpu_uuid": expected["gpu_uuid"],
        "node": expected["node"],
        "allocated_gpu_uuids_csv": expected["allocated_gpu_uuids_csv"],
        "visible_gpu_uuids_csv": expected["visible_gpu_uuids_csv"],
        "cuda_sha256": expected["cuda_sha256"],
        "relion_bind_sha256": expected["relion_bind_sha256"],
        "interpreter_sha256": expected["interpreter_sha256"],
        "cusparse_sha256": expected["cusparse_sha256"],
        "scorecard_sha256": expected["scorecard_sha256"],
        "acceptance_config_sha256": expected["acceptance_config_sha256"],
        "analyzer_source_sha256": expected["analyzer_source_sha256"],
        "science_contract_sha256": expected["science_contract_sha256"],
        "exit_status": 0,
        "fresh_process": True,
        "fresh_jax_cache": True,
        "initial_model_profile": True,
    }
    mismatches = {
        key: {"expected": value, "observed": arm.get(key)}
        for key, value in exact_fields.items()
        if arm.get(key) != value
    }
    _require(not mismatches, f"arm {label} provenance differs: {mismatches}")
    _require(Path(arm["output_dir"]).resolve() == output.resolve(), f"arm {label} output path differs")
    cache = Path(arm["jax_cache_dir"]).resolve()
    _require(cache == (run_root / "jax_cache").resolve(), f"arm {label} JAX cache path differs")
    science_environment_path = run_root / "science_environment.json"
    _require(
        arm.get("science_environment_sha256") == _sha256(science_environment_path),
        f"arm {label} science-environment hash differs",
    )
    science_environment = _validate_science_environment(
        science_environment_path,
        root=root,
        run_root=run_root,
        label=label,
        multistream_workers=multistream_workers,
        native_atomic_reduction=native_atomic_reduction,
        arm=arm,
        expected=expected,
        runtime_contract=runtime_contract,
    )
    expected_paths = _expected_iteration_artifacts(output, iterations, suffixes)
    expected_names = {path.name for path in expected_paths}
    observed_names = {
        path.name for path in output.iterdir() if path.is_file() and _ITERATION_ARTIFACT_RE.match(path.name)
    }
    _require(
        observed_names == expected_names,
        f"arm {label} numbered artifact topology differs: "
        f"missing={sorted(expected_names - observed_names)[:8]}, extra={sorted(observed_names - expected_names)[:8]}",
    )
    for path in expected_paths:
        _require(path.is_file() and path.stat().st_size > 0, f"arm {label} artifact is empty: {path}")
    manifest_sha = _validate_saved_manifest(run_root / "artifact_manifest.sha256", output, expected_paths)
    return {
        "label": label,
        "multistream_workers": int(multistream_workers),
        "native_atomic_reduction": int(native_atomic_reduction),
        "arm_provenance_sha256": _sha256(run_root / "arm.json"),
        "artifact_manifest_sha256": manifest_sha,
        "runtime_artifact_manifest_sha256": runtime_manifest_sha,
        "science_environment_sha256": _sha256(science_environment_path),
        "science_environment": science_environment,
        "artifact_count": len(expected_paths),
        "output_dir": str(output.resolve()),
        "jax_cache_dir": str(cache),
    }


def _load_particles(path: Path, label: str) -> Any:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        value = starfile.read(path, always_dict=True)
    except Exception as exc:
        raise GateSetupError(f"cannot read {label} at {path}: {exc}") from exc
    _require(isinstance(value, dict) and "particles" in value, f"{label} has no particles table")
    return value["particles"]


def _model_resolution(path: Path, label: str) -> float:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        value = starfile.read(path, always_dict=True)
    except Exception as exc:
        raise GateSetupError(f"cannot read {label} at {path}: {exc}") from exc
    general = value.get("model_general") if isinstance(value, dict) else None
    _require(isinstance(general, dict), f"{label} has no model_general mapping")
    candidates = ("rlnCurrentResolution", "_rlnCurrentResolution")
    key = next((name for name in candidates if name in general), None)
    _require(key is not None, f"{label} has no current resolution")
    try:
        result = float(general[key])
    except (TypeError, ValueError) as exc:
        raise GateSetupError(f"{label} current resolution is not numeric") from exc
    _require(math.isfinite(result) and result > 0.0, f"{label} current resolution is invalid")
    return result


def _pixel_size_from_fixture_star(path: Path) -> float:
    _require(path.is_file(), f"missing fixture STAR: {path}")
    try:
        value = starfile.read(path, always_dict=True)
    except Exception as exc:
        raise GateSetupError(f"cannot read fixture STAR {path}: {exc}") from exc
    optics = value.get("optics") if isinstance(value, dict) else None
    _require(optics is not None, "fixture STAR has no optics table")
    column = next(
        (name for name in ("rlnImagePixelSize", "_rlnImagePixelSize") if name in optics.columns),
        None,
    )
    _require(column is not None and len(optics) == 1, "fixture STAR must define one image pixel size")
    result = float(optics.iloc[0][column])
    _require(math.isfinite(result) and result > 0.0, "fixture pixel size is invalid")
    return result


def _fsc_quality_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    pixel_size_angstrom: float,
) -> tuple[dict[str, Any], np.ndarray]:
    candidate = _finite_array(candidate, "candidate quality map")
    reference = _finite_array(reference, "reference quality map")
    _require(candidate.shape == reference.shape, "candidate and quality reference shapes differ")
    curve = np.asarray(shell_fsc(candidate, reference), dtype=np.float64)
    _require(curve.size > 1 and np.any(np.isfinite(curve[1:])), "FSC curve has no finite non-DC shells")
    auc = float(normalized_fsc_auc(curve))
    _require(math.isfinite(auc), "FSC-AUC is non-finite")
    crossing = first_shell_below(curve, 0.143)
    resolution_shell = int(curve.size if crossing is None else crossing)
    resolution_shell = max(resolution_shell, 1)
    resolution_angstrom = float(pixel_size_angstrom * candidate.shape[0] / resolution_shell)
    try:
        amplitude = summarize_fourier_pair(centered_fourier(candidate), centered_fourier(reference))
    except (AssertionError, ValueError) as exc:
        raise GateSetupError(f"cannot compute mature Fourier scale metric: {exc}") from exc
    return (
        {
            "fsc_auc": auc,
            "resolution_shell_0p143": resolution_shell,
            "resolution_angstrom_0p143": resolution_angstrom,
            "global_scale_candidate_to_reference": float(amplitude["global_scale_recovar_to_relion"]),
            "relative_l2_after_global_scale": float(amplitude["relative_l2_after_global_scale"]),
        },
        curve,
    )


def _validate_native_options(path: Path, definition: dict[str, Any]) -> dict[str, Any]:
    options = _load_json(path, "native InitialModel options")
    expected_pairs = {
        "nr_classes": definition["nr_classes"],
        "nr_iter": definition["nr_iter"],
        "random_seed": definition["random_seed"],
        "tau2_fudge": definition["tau2_fudge"],
        "healpix_order": definition["healpix_order"],
        "oversampling": definition["oversampling"],
        "offset_range_px": definition["offset_range_px"],
        "offset_step_px": definition["offset_step_px"],
        "padding_factor": definition["padding_factor"],
        "particle_diameter": definition["particle_diameter_angstrom"],
        "sym_name": definition.get("symmetry", "C1"),
        "do_run_C1": definition.get("do_run_C1", True),
        "grad_write_iter": 1,
    }
    mismatches = {}
    for key, expected in expected_pairs.items():
        actual = options.get(key)
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                matches = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=0.0)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            mismatches[key] = {"expected": expected, "observed": actual}
    _require(not mismatches, f"native InitialModel options differ from the frozen case: {mismatches}")
    return expected_pairs


def _native_repeat_roots(native_root: Path, count: int) -> tuple[Path, ...]:
    roots = tuple(native_root / f"repeat-{index:02d}" / "vdam-gf46" for index in range(1, count + 1))
    _require(all(path.is_dir() for path in roots), "frozen native repeat roots are incomplete")
    return roots


def _native_command_option(argv: Sequence[str], name: str, label: str) -> str:
    indices = [index for index, value in enumerate(argv) if value == name]
    _require(len(indices) == 1, f"{label} must contain exactly one {name}")
    index = indices[0]
    _require(index + 1 < len(argv), f"{label} has no value for {name}")
    return str(argv[index + 1])


def _validate_native_command(
    path: Path,
    *,
    repeat: Path,
    provenance: dict[str, Any],
    acceptance: dict[str, Any],
    index: int,
) -> list[str]:
    command = _load_json(path, f"native repeat {index} RELION command")
    argv = command.get("argv")
    _require(
        isinstance(argv, list)
        and len(argv) >= 2
        and all(isinstance(value, str) and value and "\x00" not in value and "\n" not in value for value in argv),
        f"native repeat {index} RELION command argv is invalid",
    )
    reference = provenance.get("relion_reference")
    _require(isinstance(reference, dict), f"native repeat {index} has no RELION reference provenance")
    _require(argv[0] == reference.get("executable"), f"native repeat {index} RELION command executable differs")
    definition = acceptance["science_contract"]["definition"]
    expected_options = {
        "--o": str((repeat / "relion" / "run").resolve()),
        "--iter": str(int(definition["nr_iter"])),
        "--grad_write_iter": str(int(acceptance["science_contract"]["grad_write_iter"])),
        "--K": str(int(definition["nr_classes"])),
        "--sym": str(definition.get("symmetry", "C1")),
        "--pad": str(int(definition["padding_factor"])),
        "--particle_diameter": str(float(definition["particle_diameter_angstrom"])),
        "--oversampling": str(int(definition["oversampling"])),
        "--healpix_order": str(int(definition["healpix_order"])),
        "--offset_range": str(int(definition["offset_range_px"])),
        "--offset_step": str(int(definition["offset_step_px"])),
        "--tau2_fudge": str(int(definition["tau2_fudge"])),
        "--random_seed": str(int(definition["random_seed"])),
        "--gpu": str(acceptance["science_contract"]["gpu_argument"]),
    }
    mismatches = {
        option: {"expected": expected, "observed": _native_command_option(argv, option, f"native repeat {index}")}
        for option, expected in expected_options.items()
        if _native_command_option(argv, option, f"native repeat {index}") != expected
    }
    _require(not mismatches, f"native repeat {index} RELION command options differ: {mismatches}")
    input_star = Path(_native_command_option(argv, "--i", f"native repeat {index}"))
    _require(
        input_star.is_file()
        and _sha256(input_star) == acceptance["case"]["particle_star_sha256"],
        f"native repeat {index} RELION command input STAR differs",
    )
    required_flags = {
        "--grad",
        "--denovo_3dref",
        "--ctf",
        "--flatten_solvent",
        "--zero_mask",
        "--dont_combine_weights_via_disc",
        "--auto_sampling",
    }
    _require(
        all(argv.count(flag) == 1 for flag in required_flags),
        f"native repeat {index} RELION command flags differ",
    )
    return argv


def _validate_native_reference(native_root: Path, acceptance: dict[str, Any]) -> dict[str, Any]:
    contract = acceptance["native_reference"]
    _require(native_root.is_dir(), f"native reference root is missing: {native_root}")
    science_manifest = native_root / "science_manifest.json"
    _require(
        _sha256(science_manifest) == contract["science_manifest_sha256"],
        "native reference science manifest hash differs",
    )
    science = _load_json(science_manifest, "native science manifest")
    _require(science.get("status") == "science_complete", "native science manifest is not complete")
    repeat_count = int(contract["repeat_count"])
    repeats = _native_repeat_roots(native_root, repeat_count)
    hash_contract_names = (
        "paired_gpu_uuid_sha256",
        "run_provenance_sha256",
        "relion_timing_sha256",
        "relion_command_sha256",
    )
    for name in hash_contract_names:
        values = contract.get(name)
        _require(
            isinstance(values, list)
            and len(values) == repeat_count
            and all(isinstance(value, str) and len(value) == 64 for value in values),
            f"native reference {name} contract is invalid",
        )
    map_paths = [
        repeat / "relion" / f"run_it{iteration:03d}_class001.mrc" for repeat in repeats for iteration in range(201)
    ]
    _require(len(map_paths) == 804, "native reference map topology is not 4 x 201")
    observed_map_digest = _manifest_digest(native_root, map_paths)
    _require(
        observed_map_digest == contract["reference_map_manifest_sha256"],
        "native reference map aggregate manifest differs",
    )
    audit_hashes = []
    relion_times = []
    repeat_rows = []
    for index, (repeat, expected_audit_sha) in enumerate(
        zip(repeats, contract["trajectory_audit_sha256"], strict=True), start=1
    ):
        audit = repeat / "trajectory_audit.json"
        audit_sha = _sha256(audit)
        _require(audit_sha == expected_audit_sha, f"native repeat {index} trajectory audit differs")
        audit_hashes.append(audit_sha)
        paired_gpu_path = repeat / "paired_gpu_uuid.json"
        run_provenance_path = repeat / "run_provenance.json"
        timing_path = repeat / "relion" / "relion.timing.json"
        command_path = repeat / "relion" / "relion_command.json"
        sealed_paths = {
            "paired_gpu_uuid_sha256": paired_gpu_path,
            "run_provenance_sha256": run_provenance_path,
            "relion_timing_sha256": timing_path,
            "relion_command_sha256": command_path,
        }
        sealed_hashes = {}
        for name, sealed_path in sealed_paths.items():
            observed_sha = _sha256(sealed_path)
            expected_sha = contract[name][index - 1]
            _require(observed_sha == expected_sha, f"native repeat {index} {name} differs")
            sealed_hashes[name] = observed_sha
        gpu = _load_json(paired_gpu_path, f"native repeat {index} GPU provenance")
        gpu_values = [gpu.get(key) for key in ("physical_gpu_uuid", "relion_gpu_uuid", "recovar_gpu_uuid")]
        _require(
            len(set(gpu_values)) == 1 and gpu_values[0] == contract["physical_gpu_uuid"],
            f"native repeat {index} physical GPU differs",
        )
        provenance = _load_json(run_provenance_path, f"native repeat {index} provenance")
        reference = provenance.get("relion_reference", {})
        _require(
            reference.get("executable_sha256") == contract["relion_executable_sha256"],
            f"native repeat {index} RELION executable differs",
        )
        _validate_native_command(
            command_path,
            repeat=repeat,
            provenance=provenance,
            acceptance=acceptance,
            index=index,
        )
        timing = _load_json(timing_path, f"native repeat {index} timing")
        wall_s = float(timing.get("external_wall_s", math.nan))
        _require(math.isfinite(wall_s) and wall_s > 0.0, f"native repeat {index} runtime is invalid")
        relion_times.append(wall_s)
        repeat_rows.append(
            {
                "repeat": index,
                "root": str(repeat.resolve()),
                "trajectory_audit_sha256": audit_sha,
                **sealed_hashes,
                "relion_wall_s": wall_s,
            }
        )
    return {
        "root": str(native_root.resolve()),
        "science_manifest_sha256": _sha256(science_manifest),
        "reference_map_manifest_sha256": observed_map_digest,
        "physical_gpu_uuid": contract["physical_gpu_uuid"],
        "trajectory_audit_sha256": audit_hashes,
        "relion_wall_s": relion_times,
        "repeats": repeat_rows,
        "final_map_paths": [str(repeat / "relion" / "run_it200_class001.mrc") for repeat in repeats],
    }


def _arm_runtime(
    root: Path,
    label: str,
    iterations: Sequence[int],
    *,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    run_root = root / "runs" / label
    runtime_manifest_sha = _validate_runtime_artifact_manifest(run_root)
    timing = _load_json(run_root / "timing.json", f"arm timing for {label}")
    wall_s = float(timing.get("external_wall_s", math.nan))
    _require(math.isfinite(wall_s) and wall_s > 0.0, f"arm {label} wall time is invalid")
    expectation = 0.0
    for iteration in iterations:
        if iteration == 0:
            continue
        meta = _load_json(
            run_root / "output" / f"run_it{iteration:03d}_recovar_meta.json",
            f"arm {label} iteration {iteration} metadata",
        )
        profile = meta.get("vdam_iteration_profile_summary")
        _require(isinstance(profile, dict), f"arm {label} iteration {iteration} has no stage profile")
        value = float(profile.get("expectation_time_s", math.nan))
        _require(
            math.isfinite(value) and value >= 0.0, f"arm {label} iteration {iteration} expectation time is invalid"
        )
        expectation += value
    monitor = _read_gpu_monitor(run_root / "gpu_monitor.csv")
    _require(isinstance(monitor, dict), f"arm {label} GPU monitor is missing")
    peak = monitor.get("peak_memory_mib")
    _require(
        int(monitor.get("sample_count", 0)) >= 2
        and int(monitor.get("gpu_count", 0)) == 1
        and peak is not None
        and math.isfinite(float(peak))
        and float(peak) > 0.0,
        f"arm {label} GPU monitor is incomplete: {monitor}",
    )
    _require(
        monitor.get("gpu_uuids") == [expected_gpu_uuid]
        and int(monitor.get("gpu_uuid_count", 0)) == 1
        and monitor.get("peak_device_uuid") == expected_gpu_uuid,
        f"arm {label} GPU monitor UUID evidence differs: {monitor}",
    )
    return {
        "label": label,
        "end_to_end_wall_s": wall_s,
        "expectation_stage_s": expectation,
        "peak_gpu_memory_mib": float(peak),
        "timing_sha256": _sha256(run_root / "timing.json"),
        "gpu_monitor_sha256": _sha256(run_root / "gpu_monitor.csv"),
        "runtime_artifact_manifest_sha256": runtime_manifest_sha,
        "gpu_monitor_sample_count": int(monitor["sample_count"]),
        "gpu_uuids": monitor["gpu_uuids"],
        "gpu_uuid_count": int(monitor["gpu_uuid_count"]),
        "peak_device_uuid": monitor["peak_device_uuid"],
    }


def _validate_source_manifest(path: Path, repo: Path, source_files: Sequence[str]) -> str:
    expected_paths = []
    for relative in source_files:
        candidate = Path(relative)
        _require(not candidate.is_absolute() and ".." not in candidate.parts, "source manifest path escapes repo")
        expected_paths.append(repo / candidate)
    return _validate_saved_manifest(path, repo, expected_paths)


def _validate_run_provenance(
    root: Path,
    acceptance_path: Path,
    scorecard_path: Path,
    acceptance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance_dir = root / "provenance"
    run_path = provenance_dir / "run.json"
    run = _load_json(run_path, "true-200 run provenance")
    labels = acceptance["panel"]["execution_order"]
    workers = acceptance["panel"]["coarse_multistream_workers"]
    atomic = acceptance["panel"]["coarse_native_atomic_reduction"]
    expected = {
        "schema": "recovar.vdam_coarse_combined_true200_run.v2",
        "git_head": run.get("git_head"),
        "production_candidate_head": acceptance["qualified_candidate"]["production_head"],
        "gpu_uuid": acceptance["native_reference"]["physical_gpu_uuid"],
        "allocated_gpu_uuids_csv": acceptance["native_reference"]["physical_gpu_uuid"],
        "cuda_sha256": acceptance["qualified_candidate"]["cuda_sha256"],
        "relion_bind_sha256": acceptance["qualified_candidate"]["relion_bind_sha256"],
        "interpreter_sha256": acceptance["qualified_candidate"]["interpreter_sha256"],
        "scorecard_sha256": acceptance["case"]["scorecard_sha256"],
        "acceptance_config_sha256": _sha256(acceptance_path),
        "analyzer_source_sha256": _sha256(Path(__file__).resolve()),
        "execution_order": labels,
        "coarse_multistream_workers": workers,
        "coarse_native_atomic_reduction": atomic,
        "fresh_process_and_jax_cache_per_arm": True,
        "only_configuration_delta": (
            "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=0/8 + "
            "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=0/1"
        ),
        "initial_model_profile": True,
    }
    mismatches = {
        key: {"expected": value, "observed": run.get(key)} for key, value in expected.items() if run.get(key) != value
    }
    _require(not mismatches, f"run provenance differs from the sealed contract: {mismatches}")
    visible_gpu_uuids = str(run.get("visible_gpu_uuids_csv", "")).split(",")
    _require(
        visible_gpu_uuids
        and len(visible_gpu_uuids) == len(set(visible_gpu_uuids))
        and all(value.startswith("GPU-") for value in visible_gpu_uuids)
        and expected["gpu_uuid"] in visible_gpu_uuids,
        "run visible-GPU UUID evidence is invalid",
    )
    _require(
        run["git_head"] != acceptance["qualified_candidate"]["production_head"],
        "true-200 run must use a review overlay rather than mutate the production candidate",
    )
    repo = Path(run["repo_root"]).resolve()
    _require(repo.is_dir(), f"recorded repo root is missing: {repo}")
    try:
        current_head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        tracked_status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain=v1"], text=True
        ).strip()
        ancestor_status = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                acceptance["qualified_candidate"]["production_head"],
                current_head,
            ],
            check=False,
        ).returncode
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GateSetupError(f"cannot validate recorded source checkout: {exc}") from exc
    _require(run["git_head"] == current_head, "recorded source HEAD differs from analysis checkout")
    _require(not tracked_status, "analysis checkout is dirty")
    _require(ancestor_status == 0, "production candidate is not an ancestor of the analysis checkout")
    _require(Path(run["scorecard_path"]).resolve() == scorecard_path.resolve(), "scorecard path differs")
    _require(_sha256(scorecard_path) == expected["scorecard_sha256"], "scorecard hash differs")
    _require(Path(run["acceptance_path"]).resolve() == acceptance_path.resolve(), "acceptance path differs")
    launch_manifest = provenance_dir / "launch_manifest.json"
    _require(
        Path(run.get("launch_manifest_path", "")).resolve() == launch_manifest.resolve(),
        "launch manifest path differs",
    )
    try:
        from scripts.resolve_vdam_coarse_combined_true200_launch import (
            LaunchResolutionError,
            validate_launch_manifest,
        )

        launch = validate_launch_manifest(launch_manifest, str(run.get("launch_manifest_sha256", "")))
    except (LaunchResolutionError, KeyError, TypeError, ValueError) as exc:
        raise GateSetupError(f"launch manifest validation failed: {exc}") from exc
    _require(launch["expected_overlay_head"] == run["git_head"], "launch manifest source HEAD differs")
    _require(launch["target_gpu_uuid"] == run["gpu_uuid"], "launch manifest GPU differs")
    _require(
        launch["source_manifest"]["sha256"] == run["source_manifest_sha256"],
        "launch source-manifest seal differs",
    )
    _require(run.get("runtime_contract") == acceptance["runtime_contract"], "runtime contract differs")
    _require(
        run.get("cusparse_sha256") == launch["files"]["cusparse_library"]["sha256"],
        "runtime cuSPARSE hash differs from the launch seal",
    )
    science_contract = provenance_dir / "science_contract.json"
    _require(
        _sha256(science_contract) == run.get("science_contract_sha256"),
        "science contract hash differs",
    )
    science = _load_json(science_contract, "science contract")
    _require(science == acceptance["science_contract"], "science contract differs from acceptance")
    _require(science.get("definition") == run.get("definition"), "science definition differs across provenance")
    source_manifest = provenance_dir / "source_manifest.sha256"
    source_manifest_sha = _validate_source_manifest(source_manifest, repo, run["source_files"])
    _require(source_manifest_sha == run.get("source_manifest_sha256"), "source manifest hash differs")
    runtime_files = {
        "cuda": Path(run["cuda_path"]),
        "relion_bind": Path(run["relion_bind_path"]),
        "interpreter": Path(run["interpreter_path"]),
        "cusparse": Path(run["cusparse_path"]),
    }
    for key, path in runtime_files.items():
        _require(path.is_file(), f"recorded {key} runtime file is missing: {path}")
    _require(_sha256(runtime_files["cuda"]) == expected["cuda_sha256"], "runtime CUDA binary differs")
    _require(
        _sha256(runtime_files["relion_bind"]) == expected["relion_bind_sha256"],
        "runtime RELION binding differs",
    )
    _require(
        _sha256(runtime_files["interpreter"]) == expected["interpreter_sha256"],
        "runtime interpreter differs",
    )
    _require(
        _sha256(runtime_files["cusparse"]) == run["cusparse_sha256"],
        "runtime cuSPARSE differs",
    )
    runtime_library_manifest = provenance_dir / "runtime_libraries.json"
    _require(
        Path(run.get("runtime_libraries_path", "")).resolve()
        == runtime_library_manifest.resolve(),
        "runtime-library manifest path differs",
    )
    _require(
        _sha256(runtime_library_manifest) == run.get("runtime_libraries_sha256"),
        "runtime-library manifest hash differs",
    )
    runtime_libraries = _load_json(runtime_library_manifest, "runtime-library manifest")
    _require(
        runtime_libraries.get("schema") == "recovar.vdam_combined_true200_runtime_libraries.v1",
        "runtime-library manifest schema differs",
    )
    _require(
        runtime_libraries.get("transitive_runtime_library_closure_claimed") is False
        and acceptance["runtime_contract"]["transitive_runtime_library_closure_claimed"] is False,
        "runtime-library closure claim differs",
    )
    runtime_entries = runtime_libraries.get("direct_and_ldd_visible_artifacts")
    _require(isinstance(runtime_entries, list) and runtime_entries, "runtime-library manifest is empty")
    seen_runtime_paths: set[str] = set()
    for entry in runtime_entries:
        _require(isinstance(entry, dict), "runtime-library entry is invalid")
        runtime_path = Path(str(entry.get("path", "")))
        _require(runtime_path.is_absolute() and runtime_path.is_file(), "runtime-library path is invalid")
        canonical = str(runtime_path.resolve())
        _require(canonical not in seen_runtime_paths, "runtime-library manifest repeats a path")
        seen_runtime_paths.add(canonical)
        _require(_sha256(runtime_path) == entry.get("sha256"), f"runtime library differs: {runtime_path}")
    _require(
        {str(path.resolve()) for path in runtime_files.values()}.issubset(seen_runtime_paths),
        "direct runtime artifact is absent from runtime-library manifest",
    )
    return run, {
        "run_json_sha256": _sha256(run_path),
        "science_contract_sha256": _sha256(science_contract),
        "source_manifest_sha256": source_manifest_sha,
        "launch_manifest_sha256": _sha256(launch_manifest),
        "runtime_libraries_sha256": _sha256(runtime_library_manifest),
        "repo_root": str(repo),
        "git_head": run["git_head"],
        "node": run["node"],
        "gpu_uuid": run["gpu_uuid"],
    }


def _case_definition(scorecard: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [row for row in scorecard.get("cases", []) if row.get("id") == case_id]
    _require(len(matches) == 1, f"scorecard does not contain exactly one {case_id} case")
    definition = matches[0].get("definition")
    _require(isinstance(definition, dict), f"scorecard case {case_id} has no definition")
    return definition


def _quality_panel(
    root: Path,
    labels: Sequence[str],
    native: dict[str, Any],
    fixture_dir: Path,
    *,
    blocks: Sequence[Sequence[int]],
    final_iteration: int,
    gate_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    input_star = fixture_dir / "particles.star"
    pixel_size = _pixel_size_from_fixture_star(input_star)
    references = {"gt": _load_relion_volume(fixture_dir / "reference_gt_relion.mrc")}
    for index, path in enumerate(native["final_map_paths"], start=1):
        references[f"relion_repeat_{index}"] = _load_relion_volume(Path(path))
    shellwise: dict[str, np.ndarray] = {}
    rows = []
    for label in labels:
        output = root / "runs" / label / "output"
        candidate = _load_relion_volume(output / f"run_it{final_iteration:03d}_class001.mrc")
        metrics = {}
        for reference_name, reference in references.items():
            row, curve = _fsc_quality_metrics(
                candidate,
                reference,
                pixel_size_angstrom=pixel_size,
            )
            metrics[reference_name] = row
            shellwise[f"{label}__{reference_name}"] = curve
        rows.append(
            {
                "label": label,
                "model_resolution_angstrom": _model_resolution(
                    output / f"run_it{final_iteration:03d}_model.star",
                    f"{label} final model",
                ),
                "references": metrics,
            }
        )
    return (
        {
            "pixel_size_angstrom": pixel_size,
            "final_iteration": final_iteration,
            "arms": rows,
            "classification": classify_final_quality(
                rows,
                labels,
                blocks=blocks,
                fsc_auc_outer_degradation=float(gate_contract["final_fsc_auc_outer_degradation_at_most"]),
                resolution_shell_outer_degradation=int(
                    gate_contract["final_resolution_shell_outer_degradation_at_most"]
                ),
                model_resolution_relative_degradation=float(
                    gate_contract["final_model_resolution_relative_degradation_at_most"]
                ),
                scale_relative_deviation=float(gate_contract["final_scale_relative_deviation_at_most"]),
                permutation_p_min=float(gate_contract["final_quality_blocked_joint_permutation_p_at_least"]),
            ),
        },
        shellwise,
    )


def _markdown(report: dict[str, Any]) -> str:
    gates = report["gates"]
    runtime = report["runtime"]
    lines = [
        "# VDAM mature combined coarse: sealed true-200 gate",
        "",
        f"Decision: **{report['decision']['classification']}**",
        "",
        report["numerical_acceptance_policy"],
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in gates.items())
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| Metric | Serial median | Lane median | Change | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, row in runtime["replicated_metrics"].items():
        lines.append(
            f"| {name} | {row['serial_median']:.3f} | {row['lane_median']:.3f} | "
            f"{row['lane_percent_change']:+.2f}% | {'PASS' if row['pass'] else 'FAIL'} |"
        )
    program = runtime["absolute_relion_program_gate"]
    lines.extend(
        [
            "",
            f"Independent absolute RELION runtime ratio: **{program['lane_to_relion_runtime_ratio']:.3f}x** "
            f"(program target <= {program['program_target_at_most']:.3f}x; "
            f"{'PASS' if program['pass'] else 'FAIL'}).",
            "",
            "A science PASS qualifies only the mature default-off combined candidate; "
            "it does not enable the path by default or alter the frozen v3 correctness score.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = args.root.resolve()
    acceptance_path = args.acceptance.resolve()
    scorecard_path = args.scorecard.resolve()
    acceptance = _load_json(acceptance_path, "acceptance contract")
    _require(
        acceptance.get("schema") == "recovar.vdam_coarse_combined_true200_acceptance.v2",
        "unsupported acceptance schema",
    )
    _require((root / "RUNS_COMPLETED").is_file(), "panel arms are not marked complete")
    run, provenance = _validate_run_provenance(root, acceptance_path, scorecard_path, acceptance)
    native = _validate_native_reference(args.native_reference_root.resolve(), acceptance)
    fixture_dir = args.fixture_dir.resolve()
    fixture_files = {
        "fixture_manifest_sha256": fixture_dir / "fixture_materialization.json",
        "particle_star_sha256": fixture_dir / "particles.star",
        "particle_stack_sha256": fixture_dir / "particles.128.mrcs",
        "gt_relion_mrc_sha256": fixture_dir / "reference_gt_relion.mrc",
    }
    for hash_key, path in fixture_files.items():
        _require(path.is_file(), f"fixture input is missing: {path}")
        _require(_sha256(path) == acceptance["case"][hash_key], f"fixture hash differs for {path.name}")
    scorecard = _load_json(scorecard_path, "scorecard")
    definition = _case_definition(scorecard, acceptance["case"]["id"])
    _require(definition == run["definition"], "run definition differs from the frozen scorecard")
    trajectory = acceptance["trajectory"]
    iterations = tuple(range(int(trajectory["first_iteration"]), int(trajectory["last_iteration"]) + 1))
    state_iterations = tuple(
        range(
            int(trajectory["exact_state_first_iteration"]),
            int(trajectory["exact_state_last_iteration"]) + 1,
        )
    )
    _require(
        state_iterations == iterations[1:],
        "exact state checkpoints must cover every post-initialization iteration",
    )
    labels = tuple(acceptance["panel"]["execution_order"])
    workers = tuple(int(value) for value in acceptance["panel"]["coarse_multistream_workers"])
    atomic = tuple(int(value) for value in acceptance["panel"]["coarse_native_atomic_reduction"])
    arm_expected = {
        **run,
        "production_candidate_head": acceptance["qualified_candidate"]["production_head"],
    }
    arms = []
    for label, worker_count, atomic_mode in zip(labels, workers, atomic, strict=True):
        arms.append(
            _validate_arm_artifacts(
                root,
                label,
                worker_count,
                atomic_mode,
                iterations,
                trajectory["required_artifact_suffixes"],
                expected=arm_expected,
                runtime_contract=acceptance["runtime_contract"],
            )
        )
        arms[-1]["validated_native_options"] = _validate_native_options(
            root / "runs" / label / "output" / "run_native_options.json",
            definition,
        )

    block_indices = tuple(
        tuple(labels.index(label) for label in block)
        for block in acceptance["panel"]["whole_trajectory_permutation_blocks"]
    )
    gate_contract = acceptance["required_gates"]
    map_thresholds = {
        "phase_segments": trajectory["phase_segments"],
        "phase_boundary_windows": trajectory["phase_boundary_windows"],
        "late_window": trajectory["late_window"],
        "centroid_max": gate_contract["whole_trajectory_centroid_over_pooled_run_rms_at_most"],
        "cross_within_max": gate_contract["whole_trajectory_cross_within_median_ratio_at_most"],
        "variance_inflation_max": gate_contract["whole_trajectory_lane_variance_inflation_at_most"],
        "permutation_p_min": gate_contract["blocked_whole_trajectory_joint_permutation_p_at_least"],
        "no_growth_primary_family_endpoints": trajectory["no_growth_primary_family_endpoints"],
        "no_growth_familywise_alpha": gate_contract["no_growth_familywise_alpha"],
        "no_growth_materiality_loo_multiplier": gate_contract[
            "no_growth_materiality_serial_control_loo_multiplier"
        ],
        "no_growth_normalized_numerical_noise_floor": gate_contract[
            "no_growth_normalized_numerical_noise_floor"
        ],
        "witness_map_normalized_distance_at_most": gate_contract[
            "witness_map_normalized_distance_at_most"
        ],
    }

    def load_iteration_maps(iteration: int) -> list[np.ndarray]:
        return [
            _load_relion_volume(root / "runs" / label / "output" / f"run_it{iteration:03d}_class001.mrc")
            for label in labels
        ]

    state_rows = []
    exact_path_mismatches: dict[str, list[np.ndarray]] = {
        f"metadata:{key}": []
        for key in acceptance["state_contract"]["joint_exact_complete_path_metadata_keys"]
    }
    exact_path_mismatches.update(
        {
            f"particle_star_identity:{column}": []
            for column in acceptance["state_contract"]["star_identity_columns"]
        }
    )
    coarse_selector_audit_rows: list[dict[str, Any]] = []
    for iteration in state_iterations:
        metadata = [
            _load_json(
                root / "runs" / label / "output" / f"run_it{iteration:03d}_recovar_meta.json",
                f"{label} iteration {iteration} metadata",
            )
            for label in labels
        ]
        tables = [
            _load_particles(
                root / "runs" / label / "output" / f"run_it{iteration:03d}_data.star",
                f"{label} iteration {iteration} particles",
            )
            for label in labels
        ]
        for label, worker_count, atomic_mode, meta in zip(
            labels,
            workers,
            atomic,
            metadata,
            strict=True,
        ):
            coarse_selector_audit_rows.extend(
                _validate_coarse_selector_profile_audits(
                    meta,
                    label=label,
                    iteration=iteration,
                    multistream_workers=worker_count,
                    native_atomic_reduction=atomic_mode,
                )
            )
        metadata_result = classify_metadata_iteration(metadata, labels, acceptance["state_contract"])
        particle_result = classify_particle_tables(tables, labels, acceptance["state_contract"])
        parsed_tables = [
            _particle_table_values(table, acceptance["state_contract"], label)
            for table, label in zip(tables, labels, strict=True)
        ]
        for key in acceptance["state_contract"]["joint_exact_complete_path_metadata_keys"]:
            exact_path_mismatches[f"metadata:{key}"].append(
                _exact_mismatch_squared_distances([row[key] for row in metadata])
            )
        for column in acceptance["state_contract"]["star_identity_columns"]:
            exact_path_mismatches[f"particle_star_identity:{column}"].append(
                _exact_mismatch_squared_distances(
                    [row["identity_columns"][column] for row in parsed_tables]
                )
            )
        state_rows.append(
            {
                "iteration": iteration,
                "metadata": metadata_result,
                "particle_star": particle_result,
                "pass": metadata_result["pass"] and particle_result["pass"],
            }
        )
    continuous_state = classify_continuous_state_trajectory(
        state_rows,
        labels,
        block_indices,
        thresholds=map_thresholds,
    )
    joint_exact_state = classify_joint_exact_serial_witnesses(
        exact_path_mismatches,
        labels,
    )
    expected_selector_checkpoints = {
        (label, iteration) for label in labels for iteration in state_iterations
    }
    observed_selector_checkpoints = {
        (row["label"], row["iteration"]) for row in coarse_selector_audit_rows
    }
    _require(
        observed_selector_checkpoints == expected_selector_checkpoints,
        "coarse selector audits do not cover every arm and post-initialization checkpoint",
    )
    selector_by_arm = {}
    for label in labels:
        arm_rows = [row for row in coarse_selector_audit_rows if row["label"] == label]
        _require(
            len(arm_rows) == 2 * len(state_iterations)
            and sorted({row["halfset"] for row in arm_rows}) == [0, 1],
            f"{label} coarse selector halfset audit coverage differs",
        )
        selector_by_arm[label] = {
            "checkpoint_count": len({row["iteration"] for row in arm_rows}),
            "halfsets": sorted({row["halfset"] for row in arm_rows}),
            "audit_count": len(arm_rows),
            "total_fused_calls": sum(row["audit"]["counts"]["fused_calls"] for row in arm_rows),
            "total_actual_rows": sum(row["audit"]["counts"]["actual_rows"] for row in arm_rows),
            "total_multistream_calls": sum(
                row["audit"]["counts"]["multistream_calls"] for row in arm_rows
            ),
            "total_native_atomic_selected_calls": sum(
                row["audit"]["counts"]["native_atomic_selected_calls"] for row in arm_rows
            ),
        }
    coarse_selector_execution = {
        "policy": (
            "every sealed post-initialization halfset profile must prove the requested and "
            "effective selector, exact wrapper/target, and host-observed execution counts; "
            "serial controls must use the serial fused wrapper with zero multistream/native-atomic counts"
        ),
        "checkpoint_count": len(state_iterations),
        "audit_count": len(coarse_selector_audit_rows),
        "by_arm": selector_by_arm,
        "rows": coarse_selector_audit_rows,
        "pass": True,
    }
    state = {
        "iterations": state_rows,
        "continuous_panel": continuous_state,
        "joint_exact_complete_path_serial_witness": joint_exact_state,
        "coarse_selector_execution": coarse_selector_execution,
        "checkpoint_exact_any_serial_diagnostic_only_pass": all(
            row["pass"] for row in state_rows
        ),
        "pass": (
            continuous_state["pass"]
            and joint_exact_state["pass"]
            and coarse_selector_execution["pass"]
        ),
    }
    map_analysis = analyze_map_panel_from_loader(
        load_iteration_maps,
        labels,
        block_indices,
        iterations,
        thresholds=map_thresholds,
        loo_multiplier=float(acceptance["state_contract"]["serial_leave_one_out_multiplier"]),
        exact_witness=joint_exact_state,
    )
    quality, shellwise = _quality_panel(
        root,
        labels,
        native,
        fixture_dir,
        blocks=block_indices,
        final_iteration=iterations[-1],
        gate_contract=gate_contract,
    )
    runtime_rows = [
        _arm_runtime(
            root,
            label,
            iterations,
            expected_gpu_uuid=acceptance["native_reference"]["physical_gpu_uuid"],
        )
        for label in labels
    ]
    runtime = classify_runtime(
        runtime_rows,
        labels,
        native["relion_wall_s"],
        wall_percent_max=float(gate_contract["median_end_to_end_lane_percent_change_at_most"]),
        expectation_percent_max=float(gate_contract["median_stage_expectation_lane_percent_change_at_most"]),
        peak_memory_percent_max=float(gate_contract["median_peak_gpu_memory_percent_change_at_most"]),
        relion_ratio_target=float(gate_contract["absolute_relion_runtime_ratio_program_target"]),
    )
    gates = {
        "artifact_and_provenance_complete": True,
        "coarse_selector_executed_as_sealed_at_every_checkpoint": coarse_selector_execution[
            "pass"
        ],
        "state_joint_exact_path_and_continuous_panel": state["pass"],
        "whole_trajectory_unbiased_and_repeat_stable": all(
            map_analysis["gates"][key]
            for key in (
                "centroid_over_pooled_run_rms",
                "cross_within_median_distance_ratio",
                "lane_variance_inflation",
                "blocked_whole_trajectory_joint_permutation",
            )
        ),
        "no_growth_against_serial_leave_one_out": map_analysis["no_growth"]["pass"],
        "joint_exact_witness_coupled_bounded_map_trajectory": map_analysis[
            "joint_exact_witness_coupled_map_trajectory"
        ]["pass"],
        "final_quality_no_material_degradation": quality["classification"]["pass"],
        "material_runtime_gain": runtime["candidate_pass"],
    }
    candidate_pass = all(gates.values())
    report = {
        "schema": SCHEMA,
        "numerical_acceptance_policy": acceptance["numerical_acceptance_policy"]["statement"],
        "provenance": provenance,
        "native_reference": native,
        "fixture": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in fixture_files.items()},
        "arms": arms,
        "map_trajectory": map_analysis,
        "state_trajectory": state,
        "final_quality": quality,
        "runtime": runtime,
        "gates": gates,
        "decision": {
            "classification": (
                "accept_default_off_true200_candidate" if candidate_pass else "reject_true200_candidate"
            ),
            "candidate_qualification_pass": candidate_pass,
            "absolute_relion_program_target_pass": runtime["absolute_relion_program_gate"]["pass"],
            "default_enablement_allowed": False,
            "frozen_v3_score_changed": False,
        },
    }
    return report, shellwise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--native-reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--shells-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report, shellwise = analyze(args)
    except GateSetupError as exc:
        print(f"SETUP_FAILURE: {exc}")
        return 2
    ready = _json_ready(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ready, indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.markdown_output.write_text(_markdown(ready))
    np.savez_compressed(args.shells_output, **shellwise)
    print(json.dumps(ready["decision"], indent=2, sort_keys=True))
    return 0 if ready["decision"]["candidate_qualification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
