#!/usr/bin/env python3
"""Fail-closed analysis for the crossed late-VDAM local donation A/B."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import starfile

from scripts.run_local_mstep_donation_ab import (
    GF46_CHECKPOINT_ITERATION,
    GF46_NR_ITER_SCHEDULE,
    GF46_PROFILED_ITERATION,
    INPUT_MANIFEST_SCHEMA,
    SEALED_ARM_ENVIRONMENT_NAMES,
    SEALED_ARM_ENVIRONMENT_SCHEMA,
    SEALED_NORMALIZED_OPTIONS,
    SEALED_NORMALIZED_RECOVAR_ARGV,
)
from scripts.run_local_mstep_donation_ab import SCHEMA as ARM_SCHEMA

SCHEMA = "recovar.local_mstep_donation_ab_analysis.v3"
EXPECTED_REPEATS_PER_ARM = 3
EXPECTED_CROSSED_ORDER = {
    1: ("01", "donated"),
    2: ("01", "control"),
    3: ("02", "control"),
    4: ("02", "donated"),
    5: ("03", "donated"),
    6: ("03", "control"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    return parser.parse_args()


# These are the only non-science values removed from recovar_meta.  In
# particular, this is deliberately not an ``endswith('_s')`` rule: a new
# field is science until reviewed and explicitly added here.
_ITERATION_PROFILE_TIMING_KEYS = frozenset(
    {
        "expectation_time_s",
        "mstep_time_s",
        "pre_artifact_time_s",
        "projector_refresh_time_s",
        "schedule_time_s",
        "state_update_time_s",
        "subset_time_s",
    }
)
_SPARSE_PASS2_TIMING_KEYS = frozenset({"pass1_time_s", "pass2_time_s"})
_HALFSET_PROFILE_TIMING_KEYS = frozenset(
    {
        "accounted_em_time_s",
        "batch_fetch_time_s",
        "big_jit_bucket_s",
        "bucket_build_time_s",
        "em_time_s",
        "fused_score_mstep_s",
        "local_backproject_ctf_s",
        "local_backproject_y_s",
        "local_final_accumulator_s",
        "local_host_stats_s",
        "local_mstep_s",
        "local_noise_s",
        "local_normalize_s",
        "local_pack_s",
        "local_postprocess_s",
        "local_score_s",
        "local_significance_s",
        "local_stats_finalize_s",
        "preprocess_cache_build_s",
        "preprocess_cache_fetch_s",
        "preprocess_ctf_s",
        "preprocess_integer_shift_s",
        "preprocess_norm_s",
        "preprocess_recon_process_s",
        "preprocess_score_process_s",
        "preprocess_tile_shift_recon_s",
        "preprocess_tile_shift_score_s",
        "preprocess_time_s",
        "preprocess_translation_phase_s",
        "projection_time_s",
        "raw_cache_build_time_s",
        "relion_projection_cache_build_s",
        "transfer_final_noise_to_host_s",
        "transfer_mstep_posterior_sum_to_host_s",
        "transfer_postprocess_argmax_to_host_s",
        "transfer_postprocess_posterior_to_host_s",
        "transfer_postprocess_scores_to_host_s",
        "transfer_reconstruction_mask_to_host_s",
        "transfer_total_to_host_s",
        "unattributed_em_time_s",
    }
)
_FORBIDDEN_META_KEY_FRAGMENTS = frozenset({"iref_replay"})


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode() + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def mrc_science_snapshot(path: Path) -> dict[str, Any]:
    """Return exact file bytes, voxels, and reconstruction-relevant headers."""

    with mrcfile.open(path, permissive=False) as handle:
        data = np.array(handle.data, copy=True)
        header = handle.header
        header_fields = {
            "nx_ny_nz": [int(header.nx), int(header.ny), int(header.nz)],
            "mode": int(header.mode),
            "nxstart_nystart_nzstart": [int(header.nxstart), int(header.nystart), int(header.nzstart)],
            "mx_my_mz": [int(header.mx), int(header.my), int(header.mz)],
            "cella": [float(header.cella.x), float(header.cella.y), float(header.cella.z)],
            "cellb": [float(header.cellb.alpha), float(header.cellb.beta), float(header.cellb.gamma)],
            "mapc_mapr_maps": [int(header.mapc), int(header.mapr), int(header.maps)],
            "dmin_dmax_dmean": [float(header.dmin), float(header.dmax), float(header.dmean)],
            "origin": [float(header.origin.x), float(header.origin.y), float(header.origin.z)],
            "ispg": int(header.ispg),
            "nsymbt": int(header.nsymbt),
            "rms": float(header.rms),
        }
    return {
        "file_sha256": _bytes_sha256(path.read_bytes()),
        "shape": list(data.shape),
        "dtype": data.dtype.str,
        "voxel_sha256": _array_sha256(data),
        "header": header_fields,
    }


def _normalize_output_path(value: str, output_prefix: Path) -> str:
    return value.replace(str(output_prefix), "<OUTPUT_PREFIX>")


def _series_snapshot(series: Any, output_prefix: Path) -> dict[str, Any]:
    values = series.to_numpy()
    if values.dtype.kind in "biufcMm":
        return {
            "kind": "numeric",
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "sha256": _array_sha256(values),
        }
    normalized = [
        None if value is None else _normalize_output_path(str(value), output_prefix) for value in values.tolist()
    ]
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "kind": "string",
        "shape": list(values.shape),
        "sha256": _bytes_sha256(encoded),
        "values": normalized,
    }


def _scalar_snapshot(value: Any, output_prefix: Path) -> dict[str, Any]:
    """Snapshot one scalar from a non-loop STAR block without coercion loss."""

    array = np.asarray(value)
    if array.ndim == 0 and array.dtype.kind in "biufcMm":
        return {
            "kind": "numeric_scalar",
            "dtype": array.dtype.str,
            "sha256": _array_sha256(array),
        }
    if value is None or array.ndim == 0:
        normalized = None if value is None else _normalize_output_path(str(value), output_prefix)
        encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()
        return {
            "kind": "string_scalar",
            "sha256": _bytes_sha256(encoded),
            "value": normalized,
        }
    raise TypeError(f"STAR scalar block contains a non-scalar value: {type(value)!r}")


def star_science_snapshot(path: Path, output_prefix: Path) -> dict[str, Any]:
    """Parse every STAR block/row/column, normalizing only the output prefix."""

    document = starfile.read(path, always_dict=True)
    blocks: dict[str, Any] = {}
    for block_name, frame in document.items():
        if isinstance(frame, dict):
            keys = [str(key) for key in frame]
            blocks[str(block_name)] = {
                "kind": "scalar_mapping",
                "entry_count": len(frame),
                "keys": keys,
                "values": {str(key): _scalar_snapshot(frame[key], output_prefix) for key in frame},
            }
            continue
        columns = [str(column) for column in frame.columns]
        blocks[str(block_name)] = {
            "kind": "table",
            "row_count": int(len(frame)),
            "columns": columns,
            "series": {column: _series_snapshot(frame[column], output_prefix) for column in columns},
        }
    return {"block_order": [str(name) for name in document], "blocks": blocks}


def _recovar_meta_science(value: Any, path: tuple[str, ...] = ()) -> tuple[Any, list[str]]:
    if isinstance(value, dict):
        normalized = {}
        excluded: list[str] = []
        container = path[-1] if path else ""
        for key, item in value.items():
            key = str(key)
            if any(fragment in key.casefold() for fragment in _FORBIDDEN_META_KEY_FRAGMENTS):
                raise RuntimeError(
                    "IREF replay is forbidden by the sealed donation A/B gate: "
                    f"/{'/'.join((*path, key))}"
                )
            excluded_here = bool(
                (container == "vdam_iteration_profile_summary" and key in _ITERATION_PROFILE_TIMING_KEYS)
                or (container == "sparse_pass2_profile_summary" and key in _SPARSE_PASS2_TIMING_KEYS)
                or (
                    container.startswith("halfset_")
                    and container.endswith("_profile_summary")
                    and key in _HALFSET_PROFILE_TIMING_KEYS
                )
            )
            child_path = (*path, key)
            if excluded_here:
                excluded.append("/" + "/".join(child_path))
                continue
            normalized_item, child_excluded = _recovar_meta_science(item, child_path)
            normalized[key] = normalized_item
            excluded.extend(child_excluded)
        return normalized, excluded
    if isinstance(value, list):
        normalized_items = []
        excluded = []
        for index, item in enumerate(value):
            normalized_item, child_excluded = _recovar_meta_science(item, (*path, str(index)))
            normalized_items.append(normalized_item)
            excluded.extend(child_excluded)
        return normalized_items, excluded
    return value, []


def recovar_meta_science_snapshot(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    science, excluded_paths = _recovar_meta_science(raw)
    encoded = json.dumps(science, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "science_sha256": _bytes_sha256(encoded),
        "top_level_keys": sorted(science),
        "excluded_paths": sorted(excluded_paths),
    }


def science_snapshot(report: dict[str, Any], phase: str) -> dict[str, Any]:
    outputs = report[phase]["science_outputs"]
    prefix = Path(outputs["output_prefix"]["path"])
    snapshot = {
        "class_map": mrc_science_snapshot(Path(outputs["class_map"]["path"])),
        "initial_model_map": mrc_science_snapshot(Path(outputs["initial_model_map"]["path"])),
        "stars": {
            name: star_science_snapshot(Path(outputs[name]["path"]), prefix) for name in ("data_star", "model_star")
        },
        "recovar_meta": recovar_meta_science_snapshot(Path(outputs["recovar_meta"]["path"])),
    }
    return snapshot


def _exact_array_snapshot(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }


def _collect_json_components(
    value: Any,
    *,
    path: str,
    discrete: dict[str, Any],
    continuous: dict[str, np.ndarray],
) -> None:
    if isinstance(value, dict):
        # JSON object order is serialization detail, while the key set is
        # science topology.  Sort both the recorded topology and traversal so
        # equivalent writers cannot manufacture a discrete mismatch merely by
        # changing object insertion order.
        discrete[f"{path}/<keys>"] = sorted(value)
        for key in sorted(value):
            item = value[key]
            _collect_json_components(
                item,
                path=f"{path}/{key}",
                discrete=discrete,
                continuous=continuous,
            )
        return
    if isinstance(value, list):
        try:
            array = np.asarray(value)
        except ValueError:
            # NumPy rejects ragged nested sequences.  They still have an exact
            # ordered topology, so recurse rather than losing any field.
            array = np.asarray(value, dtype=object)
        if array.dtype.kind in "iu" and array.dtype != np.dtype(bool):
            discrete[path] = _exact_array_snapshot(array)
            return
        if array.dtype.kind in "fc":
            if not np.all(np.isfinite(array)):
                raise RuntimeError(f"non-finite continuous science field: {path}")
            continuous[path] = np.ascontiguousarray(array)
            return
        discrete[f"{path}/<length>"] = len(value)
        for index, item in enumerate(value):
            _collect_json_components(
                item,
                path=f"{path}/{index}",
                discrete=discrete,
                continuous=continuous,
            )
        return
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        discrete[path] = value
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite continuous science field: {path}")
        continuous[path] = np.asarray([value], dtype=np.float64)
        return
    raise TypeError(f"unsupported recovar_meta science value at {path}: {type(value)!r}")


def _mrc_components(
    path: Path,
    *,
    label: str,
    discrete: dict[str, Any],
    continuous: dict[str, np.ndarray],
) -> None:
    with mrcfile.open(path, permissive=False) as handle:
        data = np.array(handle.data, copy=True)
        header = handle.header
        discrete[f"{label}/topology"] = {
            "shape": list(data.shape),
            "dtype": data.dtype.str,
            "mode": int(header.mode),
            "nx_ny_nz": [int(header.nx), int(header.ny), int(header.nz)],
            "nxstart_nystart_nzstart": [int(header.nxstart), int(header.nystart), int(header.nzstart)],
            "mx_my_mz": [int(header.mx), int(header.my), int(header.mz)],
            "mapc_mapr_maps": [int(header.mapc), int(header.mapr), int(header.maps)],
            "ispg": int(header.ispg),
            "nsymbt": int(header.nsymbt),
        }
        continuous[f"{label}/voxels"] = np.ascontiguousarray(data)
        continuous[f"{label}/header"] = np.asarray(
            [
                float(header.cella.x),
                float(header.cella.y),
                float(header.cella.z),
                float(header.cellb.alpha),
                float(header.cellb.beta),
                float(header.cellb.gamma),
                float(header.dmin),
                float(header.dmax),
                float(header.dmean),
                float(header.origin.x),
                float(header.origin.y),
                float(header.origin.z),
                float(header.rms),
            ],
            dtype=np.float64,
        )


def _star_components(
    path: Path,
    output_prefix: Path,
    *,
    label: str,
    discrete: dict[str, Any],
    continuous: dict[str, np.ndarray],
) -> None:
    document = starfile.read(path, always_dict=True)
    discrete[f"{label}/block_order"] = [str(name) for name in document]
    for block_name, frame in document.items():
        block_path = f"{label}/{block_name}"
        if isinstance(frame, dict):
            discrete[f"{block_path}/keys"] = [str(key) for key in frame]
            items = frame.items()
        else:
            discrete[f"{block_path}/columns"] = [str(column) for column in frame.columns]
            discrete[f"{block_path}/row_count"] = int(len(frame))
            items = ((column, frame[column].to_numpy()) for column in frame.columns)
        for name, raw_value in items:
            field_path = f"{block_path}/{name}"
            array = np.asarray(raw_value)
            if array.dtype.kind in "iu" and array.dtype != np.dtype(bool):
                discrete[field_path] = _exact_array_snapshot(array)
            elif array.dtype.kind in "fc":
                if not np.all(np.isfinite(array)):
                    raise RuntimeError(f"non-finite STAR science field: {field_path}")
                continuous[field_path] = np.ascontiguousarray(array)
            else:
                values = array.reshape(-1).tolist() if array.ndim else [array.item()]
                discrete[field_path] = [
                    None
                    if value is None
                    else _normalize_output_path(str(value), output_prefix)
                    for value in values
                ]


def science_components(report: dict[str, Any], phase: str) -> dict[str, Any]:
    outputs = report[phase]["science_outputs"]
    prefix = Path(outputs["output_prefix"]["path"])
    discrete: dict[str, Any] = {}
    continuous: dict[str, np.ndarray] = {}
    for name in ("class_map", "initial_model_map"):
        _mrc_components(
            Path(outputs[name]["path"]),
            label=f"mrc/{name}",
            discrete=discrete,
            continuous=continuous,
        )
    for name in ("data_star", "model_star"):
        _star_components(
            Path(outputs[name]["path"]),
            prefix,
            label=f"star/{name}",
            discrete=discrete,
            continuous=continuous,
        )
    raw_meta = json.loads(Path(outputs["recovar_meta"]["path"]).read_text())
    meta_science, excluded_paths = _recovar_meta_science(raw_meta)
    discrete["meta/<excluded_timing_or_path_fields>"] = sorted(excluded_paths)
    _collect_json_components(
        meta_science,
        path="meta",
        discrete=discrete,
        continuous=continuous,
    )
    return {
        "discrete": discrete,
        "continuous": continuous,
    }


def _continuous_distance_matrix(samples: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    field_names = sorted(samples[0]["continuous"])
    if not field_names:
        raise RuntimeError("science contract contains no continuous fields")
    for sample in samples[1:]:
        if sorted(sample["continuous"]) != field_names:
            raise RuntimeError("continuous science field topology differs across phase-matched arms")
    field_scales: dict[str, float] = {}
    for field_name in field_names:
        arrays = [np.asarray(sample["continuous"][field_name]) for sample in samples]
        reference = arrays[0]
        for array in arrays[1:]:
            if array.shape != reference.shape or array.dtype != reference.dtype:
                raise RuntimeError(f"continuous science shape/dtype differs: {field_name}")
        comparison_dtype = np.complex128 if np.iscomplexobj(reference) else np.float64
        field_scales[field_name] = max(
            float(np.linalg.norm(array.astype(comparison_dtype).ravel()))
            for array in arrays
        )

    matrix = np.zeros((len(samples), len(samples)), dtype=np.float64)
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            squared = []
            for field_name in field_names:
                left_raw = np.asarray(samples[left]["continuous"][field_name])
                right_raw = np.asarray(samples[right]["continuous"][field_name])
                comparison_dtype = np.complex128 if np.iscomplexobj(left_raw) else np.float64
                left_value = left_raw.astype(comparison_dtype)
                right_value = right_raw.astype(comparison_dtype)
                scale = field_scales[field_name]
                delta = float(np.linalg.norm((left_value - right_value).ravel()))
                squared.append(0.0 if scale == 0.0 and delta == 0.0 else (delta / max(scale, np.finfo(float).tiny)) ** 2)
            matrix[left, right] = matrix[right, left] = math.sqrt(sum(squared) / len(squared))
    return matrix, {
        "field_count": len(field_names),
        "field_names": field_names,
        "field_scales": field_scales,
    }


def _within_values(matrix: np.ndarray, indices: tuple[int, ...]) -> list[float]:
    return [float(matrix[left, right]) for left, right in itertools.combinations(indices, 2)]


def _cross_values(matrix: np.ndarray, left: tuple[int, ...], right: tuple[int, ...]) -> list[float]:
    return [float(matrix[i, j]) for i in left for j in right]


def _phase_numeric_gate(samples_by_arm: dict[str, list[dict[str, Any]]], phase: str) -> dict[str, Any]:
    controls = samples_by_arm["control"]
    donated = samples_by_arm["donated"]
    if len(controls) != EXPECTED_REPEATS_PER_ARM or len(donated) != EXPECTED_REPEATS_PER_ARM:
        raise RuntimeError(f"{phase} does not contain the sealed 3+3 repeat panel")
    samples = [*controls, *donated]
    reference_discrete = samples[0]["discrete"]
    for sample in samples[1:]:
        if sample["discrete"] != reference_discrete:
            raise RuntimeError(f"{phase} discrete choices/identity/topology are not exact")
    exact_forensic_products = all(sample["snapshot"] == samples[0]["snapshot"] for sample in samples[1:])

    matrix, field_report = _continuous_distance_matrix(samples)
    control_indices = (0, 1, 2)
    donated_indices = (3, 4, 5)
    control_within = _within_values(matrix, control_indices)
    donated_within = _within_values(matrix, donated_indices)
    cross = _cross_values(matrix, control_indices, donated_indices)
    exact_continuous = bool(np.all(matrix == 0.0))

    observed_energy = float(
        2.0 * statistics.mean(cross)
        - statistics.mean(control_within)
        - statistics.mean(donated_within)
    )
    observed_variance_delta = float(statistics.mean(donated_within) - statistics.mean(control_within))
    energy_permutations = []
    variance_permutations = []
    all_indices = tuple(range(6))
    for candidate_control in itertools.combinations(all_indices, 3):
        candidate_donated = tuple(index for index in all_indices if index not in candidate_control)
        perm_control = _within_values(matrix, candidate_control)
        perm_donated = _within_values(matrix, candidate_donated)
        perm_cross = _cross_values(matrix, candidate_control, candidate_donated)
        energy_permutations.append(
            2.0 * statistics.mean(perm_cross)
            - statistics.mean(perm_control)
            - statistics.mean(perm_donated)
        )
        variance_permutations.append(statistics.mean(perm_donated) - statistics.mean(perm_control))
    energy_p = sum(value >= observed_energy for value in energy_permutations) / len(energy_permutations)
    variance_inflation_p = (
        sum(value >= observed_variance_delta for value in variance_permutations)
        / len(variance_permutations)
    )
    complete_separation = bool(
        min(cross) > max([*control_within, *donated_within])
    )
    nondirectional_distribution_failure = bool(energy_p <= 0.10 and complete_separation)
    variance_inflation_failure = bool(variance_inflation_p <= 0.05 and observed_variance_delta > 0.0)
    hard_failure = bool(nondirectional_distribution_failure or variance_inflation_failure)
    # The typed contract covers every MRC voxel/reconstruction header, every
    # STAR row/column, and every non-timing recovar_meta value.  Byte-container
    # hashes remain useful forensic evidence, but are not mathematical values
    # and therefore cannot turn typed equality into a numerical failure.
    exact_parsed_science = exact_continuous
    if exact_parsed_science:
        status = "exact_pass"
    elif hard_failure:
        status = "fail"
    else:
        status = "inconclusive_3x3"
    return {
        "phase": phase,
        "status": status,
        "exact_parsed_science": exact_parsed_science,
        "exact_continuous_science": exact_continuous,
        "exact_forensic_product_snapshots": exact_forensic_products,
        "exact_discrete_choices_identity_topology": True,
        "repeat_count_per_arm": EXPECTED_REPEATS_PER_ARM,
        "continuous_distance": {
            "normalization": "per-field_l2_divided_by_phase_pooled_max_l2_then_equal-field-rms",
            "matrix": matrix.tolist(),
            **field_report,
        },
        "control_control_distances": control_within,
        "donated_donated_distances": donated_within,
        "donated_control_cross_distances": cross,
        "energy_statistic": observed_energy,
        "energy_exact_permutation_p": energy_p,
        "variance_delta": observed_variance_delta,
        "variance_inflation_exact_permutation_p": variance_inflation_p,
        "complete_cross_within_separation": complete_separation,
        "nondirectional_distribution_failure": nondirectional_distribution_failure,
        "variance_inflation_failure": variance_inflation_failure,
        "hard_failure": hard_failure,
        "qualification_power": "insufficient_for_nonexact_acceptance_with_3_controls_and_3_donated",
        "recommended_repeat_count_per_arm": 8,
    }


def _load_samples(path: Path, start_ns: int, end_ns: int) -> dict[str, Any]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("unix_ns"):
            continue
        timestamp, memory_mib = line.split("\t")
        timestamp_ns = int(timestamp)
        if start_ns <= timestamp_ns <= end_ns:
            rows.append((timestamp_ns, int(memory_mib)))
    if not rows:
        raise RuntimeError(f"no nvidia-smi samples overlap the warm interval: {path}")
    memory = [row[1] for row in rows]
    return {
        "sample_count": len(rows),
        "first_mib": memory[0],
        "min_mib": min(memory),
        "peak_mib": max(memory),
        "first_unix_ns": rows[0][0],
        "last_unix_ns": rows[-1][0],
    }


def _profile_metrics(report: dict[str, Any]) -> dict[str, float]:
    warm = report["warm"]
    iteration = warm["iteration_profile"]
    halfsets = warm["halfset_profiles"].values()
    return {
        "e2e_wall_s": float(warm["wall_s"]),
        "expectation_s": float(iteration["expectation_time_s"]),
        "mstep_s": float(iteration["mstep_time_s"]),
        "big_jit_bucket_s": float(sum(float(profile["big_jit_bucket_s"]) for profile in halfsets)),
        "outer_backproject_y_s": float(sum(float(profile["local_backproject_y_s"]) for profile in halfsets)),
        "final_accumulator_s": float(sum(float(profile["local_final_accumulator_s"]) for profile in halfsets)),
    }


def _jax_peak_bytes(report: dict[str, Any]) -> int:
    if report["warm"].get("jax_memory_scope") != "whole_process_cumulative_not_warm_isolated":
        raise RuntimeError("arm report does not label cumulative JAX peak scope")
    stats = report["warm"]["jax_memory_after"]
    if "peak_bytes_in_use" not in stats:
        raise RuntimeError(f"JAX memory stats lack peak_bytes_in_use: {sorted(stats)}")
    return int(stats["peak_bytes_in_use"])


def _median_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {name: float(statistics.median(value[name] for value in values)) for name in values[0]}


def _ratio(candidate: float, control: float) -> float:
    if control <= 0.0:
        raise RuntimeError(f"control metric must be positive, got {control}")
    return float(candidate / control)


def _load_runs(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(root.glob("runs/repeat-*/**/donation_arm_summary.json"))
    runs = [(path, json.loads(path.read_text())) for path in paths]
    if len(runs) != 2 * EXPECTED_REPEATS_PER_ARM:
        raise RuntimeError(f"expected {2 * EXPECTED_REPEATS_PER_ARM} arm reports, got {len(runs)}")
    return runs


def analyze(
    root: Path,
    *,
    expected_repo_head: str,
    expected_gpu_uuid: str,
    expected_input_manifest_sha256: str,
) -> dict[str, Any]:
    runs = _load_runs(root)
    by_arm: dict[str, list[dict[str, Any]]] = {"control": [], "donated": []}
    phase_samples: dict[str, dict[str, list[dict[str, Any]]]] = {
        phase: {"control": [], "donated": []} for phase in ("cold", "warm")
    }
    run_rows = []
    reference_contract_source = None
    reference_input_manifest = None
    cache_paths: set[str] = set()
    launch_ordinals: set[int] = set()
    program_tables: dict[str, dict[str, dict[str, Any]]] = {
        "control": {},
        "donated": {},
    }
    for path, report in runs:
        if report.get("schema") != ARM_SCHEMA or report.get("passed") is not True:
            raise RuntimeError(f"invalid arm report: {path}")
        if report.get("git_head") != expected_repo_head:
            raise RuntimeError(f"arm report source head drifted: {path}")
        if report.get("gpu_uuid") != expected_gpu_uuid:
            raise RuntimeError(f"arm report GPU drifted: {path}")
        if report.get("checkpoint_iteration") != GF46_CHECKPOINT_ITERATION:
            raise RuntimeError(f"arm report checkpoint schedule drifted: {path}")
        if report.get("profiled_iteration") != GF46_PROFILED_ITERATION:
            raise RuntimeError(f"arm report profiled iteration drifted: {path}")
        if report.get("nr_iter_schedule") != GF46_NR_ITER_SCHEDULE:
            raise RuntimeError(f"arm report nr_iter schedule drifted: {path}")
        if report.get("normalized_options") != SEALED_NORMALIZED_OPTIONS:
            raise RuntimeError(f"arm report normalized options drifted: {path}")
        if report.get("normalized_recovar_argv") != SEALED_NORMALIZED_RECOVAR_ARGV:
            raise RuntimeError(f"arm report normalized argv drifted: {path}")
        for phase in ("cold", "warm"):
            if report.get(phase, {}).get("normalized_argv") != SEALED_NORMALIZED_RECOVAR_ARGV:
                raise RuntimeError(f"{phase} normalized argv drifted: {path}")

        input_manifest = report.get("input_manifest")
        if not isinstance(input_manifest, dict):
            raise RuntimeError(f"arm report lacks its GF46 input manifest: {path}")
        if input_manifest.get("schema") != INPUT_MANIFEST_SCHEMA:
            raise RuntimeError(f"arm report input-manifest schema drifted: {path}")
        if input_manifest.get("sha256") != expected_input_manifest_sha256:
            raise RuntimeError(f"arm report GF46 input-manifest SHA drifted: {path}")
        if report.get("input_hashes") != input_manifest.get("hashes"):
            raise RuntimeError(f"arm report input hashes disagree with its manifest: {path}")
        input_hashes = input_manifest["hashes"]
        if report.get("checkpoint_optimiser_sha256") != input_hashes.get("checkpoint/optimiser.star"):
            raise RuntimeError(f"arm checkpoint hash disagrees with its GF46 input manifest: {path}")
        input_star_key = f"input/{Path(report.get('input_star', '')).name}"
        if report.get("input_star_sha256") != input_hashes.get(input_star_key):
            raise RuntimeError(f"arm input STAR hash disagrees with its GF46 input manifest: {path}")
        particle_key = f"particles/{Path(report.get('particle_stack', '')).name}"
        if report.get("particle_stack_sha256") != input_hashes.get(particle_key):
            raise RuntimeError(f"arm particle-stack hash disagrees with its GF46 input manifest: {path}")
        if reference_input_manifest is None:
            reference_input_manifest = {
                key: input_manifest[key] for key in ("sha256", "schema", "entries", "hashes")
            }
        elif {
            key: input_manifest[key] for key in ("sha256", "schema", "entries", "hashes")
        } != reference_input_manifest:
            raise RuntimeError(f"GF46 input manifest differs across arms: {path}")

        runtime = report.get("runtime_provenance", {})
        if runtime.get("parity_ancestors_verified") is not True:
            raise RuntimeError(f"arm did not verify required parity ancestors: {path}")
        repo_root = Path(runtime.get("repo_root", "")).resolve()
        pixi_env = Path(runtime.get("pixi_env", "")).resolve()
        if pixi_env != repo_root / ".pixi" / "envs" / "default":
            raise RuntimeError(f"arm did not use the exact worktree pixi env: {path}")
        for key in ("python_executable", "python_prefix", "jax_path"):
            if not Path(runtime.get(key, "")).resolve().is_relative_to(pixi_env):
                raise RuntimeError(f"arm {key} is outside the exact worktree pixi env: {path}")
        if not Path(runtime.get("recovar_path", "")).resolve().is_relative_to(repo_root):
            raise RuntimeError(f"arm RECOVAR import is outside the exact worktree: {path}")

        cache = report.get("jax_persistent_cache", {})
        cache_path = Path(cache.get("path", "")).resolve()
        run_dir = path.parents[1]
        expected_cache_path = (run_dir / "jax_cache").resolve()
        if cache_path != expected_cache_path:
            raise RuntimeError(f"arm JAX cache is not private to its run directory: {path}")
        if cache.get("initial_empty") is not True or cache.get("initial_entry_count") != 0:
            raise RuntimeError(f"arm JAX cache was not fresh at launch: {path}")
        if not isinstance(cache.get("final", {}).get("manifest_sha256"), str):
            raise RuntimeError(f"arm JAX cache lacks its sealed final manifest: {path}")
        sealed_environment = report.get("sealed_environment", {})
        if sealed_environment.get("schema") != SEALED_ARM_ENVIRONMENT_SCHEMA:
            raise RuntimeError(f"arm environment-contract schema drifted: {path}")
        if sealed_environment.get("exact_allowlist") is not True:
            raise RuntimeError(f"arm did not run under the exact environment allowlist: {path}")
        if sealed_environment.get("iref_replay_forbidden") is not True:
            raise RuntimeError(f"arm environment did not forbid IREF replay: {path}")
        if sealed_environment.get("unexpected_names") != []:
            raise RuntimeError(f"arm environment contains unexpected variables: {path}")
        if sealed_environment.get("allowed_names") != sorted(SEALED_ARM_ENVIRONMENT_NAMES):
            raise RuntimeError(f"arm environment allowlist drifted: {path}")
        observed_environment = sealed_environment.get("observed", {})
        if set(observed_environment) != SEALED_ARM_ENVIRONMENT_NAMES:
            raise RuntimeError(f"arm observed environment names drifted: {path}")
        expected_environment_values = {
            "CUDA_VISIBLE_DEVICES": expected_gpu_uuid,
            "JAX_COMPILATION_CACHE_DIR": str(cache_path),
            "RECOVAR_EXPECTED_REPO_ROOT": str(repo_root),
        }
        if any(
            observed_environment.get(name) != expected
            for name, expected in expected_environment_values.items()
        ):
            raise RuntimeError(f"arm observed environment values drifted: {path}")
        cache_paths.add(str(cache_path))
        arm = str(report["arm"])
        if arm not in by_arm:
            raise RuntimeError(f"unknown arm in {path}: {arm}")
        launch_path = run_dir / "launch_contract.json"
        if not launch_path.is_file():
            raise RuntimeError(f"arm lacks its pre-launch contract: {path}")
        launch = json.loads(launch_path.read_text())
        if launch.get("schema") != "recovar.local_mstep_donation_launch.v1":
            raise RuntimeError(f"arm launch-contract schema drifted: {path}")
        if launch.get("arm") != arm or launch.get("repeat") != run_dir.parent.name.removeprefix("repeat-"):
            raise RuntimeError(f"arm launch-contract identity drifted: {path}")
        if Path(launch.get("jax_cache_dir", "")).resolve() != cache_path:
            raise RuntimeError(f"arm launch-contract cache drifted: {path}")
        if launch.get("jax_cache_initial_empty") is not True:
            raise RuntimeError(f"arm launch contract did not seal an empty cache: {path}")
        if launch.get("input_manifest_sha256") != expected_input_manifest_sha256:
            raise RuntimeError(f"arm launch-contract input manifest drifted: {path}")
        if {
            key: launch.get(key)
            for key in ("checkpoint_iteration", "profiled_iteration", "nr_iter_schedule")
        } != {
            "checkpoint_iteration": GF46_CHECKPOINT_ITERATION,
            "profiled_iteration": GF46_PROFILED_ITERATION,
            "nr_iter_schedule": GF46_NR_ITER_SCHEDULE,
        }:
            raise RuntimeError(f"arm launch-contract schedule drifted: {path}")
        launch_ordinals.add(int(launch["ordinal"]))
        if EXPECTED_CROSSED_ORDER.get(int(launch["ordinal"])) != (
            launch["repeat"],
            launch["arm"],
        ):
            raise RuntimeError(f"arm launch contract violates the crossed execution order: {path}")
        contract = report["compiled_local_programs"]["contract"]
        if not contract["same_wrapped_numeric_source"]:
            raise RuntimeError(f"arm does not use the shared numeric source: {path}")
        source_sha = contract["numeric_source_sha256"]
        if reference_contract_source is None:
            reference_contract_source = source_sha
        elif source_sha != reference_contract_source:
            raise RuntimeError("numeric source SHA differs across arms")

        phase_science: dict[str, dict[str, Any]] = {}
        for phase in ("cold", "warm"):
            snapshot = science_snapshot(report, phase)
            components = science_components(report, phase)
            phase_science[phase] = snapshot
            phase_samples[phase][arm].append(
                {
                    "report": str(path.resolve()),
                    "snapshot": snapshot,
                    **components,
                }
            )

        samples = _load_samples(
            run_dir / "nvidia_smi.tsv",
            int(report["warm"]["timed_start_unix_ns"]),
            int(report["warm"]["timed_end_unix_ns"]),
        )
        metrics = _profile_metrics(report)
        row = {
            "report": str(path.resolve()),
            "arm": arm,
            "metrics": metrics,
            "whole_process_jax_peak_bytes": _jax_peak_bytes(report),
            "sampled_memory": samples,
            "raw_science_sha256": {
                phase: {
                    name: value["sha256"]
                    for name, value in report[phase]["science_outputs"].items()
                    if "sha256" in value
                }
                for phase in ("cold", "warm")
            },
            "parsed_science_snapshots": phase_science,
        }
        run_rows.append(row)
        by_arm[arm].append(row)
        for program in report["compiled_local_programs"]["programs"]:
            prior = program_tables[arm].setdefault(program["program_key"], program)
            if program != prior:
                raise RuntimeError(f"compiled program metadata changed within {arm}")

    for arm, values in by_arm.items():
        if len(values) != EXPECTED_REPEATS_PER_ARM:
            raise RuntimeError(f"expected {EXPECTED_REPEATS_PER_ARM} {arm} runs, got {len(values)}")
    if len(cache_paths) != 2 * EXPECTED_REPEATS_PER_ARM:
        raise RuntimeError(f"expected six unique per-run JAX caches, got {sorted(cache_paths)}")
    if launch_ordinals != set(range(1, 2 * EXPECTED_REPEATS_PER_ARM + 1)):
        raise RuntimeError(f"pre-launch ordinals are not exactly 1..6: {sorted(launch_ordinals)}")
    if set(program_tables["control"]) != set(program_tables["donated"]):
        raise RuntimeError("control/donated compiled program signatures differ")

    phase_numeric_gates = {
        phase: _phase_numeric_gate(phase_samples[phase], phase)
        for phase in ("cold", "warm")
    }
    if any(gate["status"] == "fail" for gate in phase_numeric_gates.values()):
        numeric_qualification_status = "fail"
    elif all(gate["status"] == "exact_pass" for gate in phase_numeric_gates.values()):
        numeric_qualification_status = "exact_pass"
    else:
        numeric_qualification_status = "inconclusive_3x3"
    numeric_qualification_allowed = numeric_qualification_status == "exact_pass"

    qualifying = []
    for key in sorted(program_tables["control"]):
        control = program_tables["control"][key]
        donated = program_tables["donated"][key]
        if (
            donated["mstep_relion_x_half"]
            and donated["adjoints_enabled"]
            and int(donated["accumulator_bytes"]["total"]) > 0
        ):
            accumulator_bytes = int(donated["accumulator_bytes"]["total"])
            if control["accumulator_bytes"] != donated["accumulator_bytes"]:
                raise RuntimeError("control/donated accumulator shapes or dtypes differ")
            control_alias = int(control["memory_analysis"]["alias_size_in_bytes"])
            donated_alias = int(donated["memory_analysis"]["alias_size_in_bytes"])
            qualifying.append(
                {
                    "program_key": key,
                    "accumulator_bytes": accumulator_bytes,
                    "control_alias_bytes": control_alias,
                    "donated_alias_bytes": donated_alias,
                    "alias_gain_bytes": donated_alias - control_alias,
                }
            )
    if not qualifying:
        raise RuntimeError("gate observed no production-sized RELION x-half local program with enabled adjoints")
    alias_contract_passed = any(row["alias_gain_bytes"] >= row["accumulator_bytes"] for row in qualifying)

    arm_summary = {}
    for arm, values in by_arm.items():
        arm_summary[arm] = {
            "median_metrics": _median_metrics([value["metrics"] for value in values]),
            "median_whole_process_jax_peak_bytes": int(
                statistics.median(value["whole_process_jax_peak_bytes"] for value in values)
            ),
            "median_sampled_peak_mib": float(
                statistics.median(value["sampled_memory"]["peak_mib"] for value in values)
            ),
        }
    ratios = {
        name: _ratio(
            arm_summary["donated"]["median_metrics"][name],
            arm_summary["control"]["median_metrics"][name],
        )
        for name in arm_summary["control"]["median_metrics"]
    }
    ratios["jax_whole_process_peak"] = _ratio(
        arm_summary["donated"]["median_whole_process_jax_peak_bytes"],
        arm_summary["control"]["median_whole_process_jax_peak_bytes"],
    )
    ratios["sampled_peak"] = _ratio(
        arm_summary["donated"]["median_sampled_peak_mib"],
        arm_summary["control"]["median_sampled_peak_mib"],
    )
    material_runtime_win = bool(ratios["e2e_wall_s"] <= 0.90)
    material_stage_win = bool(ratios["expectation_s"] <= 0.90 or ratios["big_jit_bucket_s"] <= 0.90)
    sampled_memory_diagnostic_below_0_95 = bool(ratios["sampled_peak"] <= 0.95)
    # The 50 ms nvidia-smi poller can miss a short peak entirely (including a
    # one-row trace).  Keep its ratio visible, but never use it for acceptance
    # or a memory claim until a defensible process-HWM measurement exists.
    material_memory_win = False
    no_material_regression = bool(
        ratios["e2e_wall_s"] <= 1.10
        and ratios["expectation_s"] <= 1.10
        and ratios["big_jit_bucket_s"] <= 1.10
    )
    passed = bool(
        alias_contract_passed
        and no_material_regression
        and numeric_qualification_status != "fail"
    )
    return {
        "schema": SCHEMA,
        "classification": "diagnostic_performance_only",
        "passed": passed,
        "same_source_exact_science_outputs": numeric_qualification_status == "exact_pass",
        "cold_and_warm_are_separate_phase_matched_cohorts": True,
        "phase_matched_complete_science_outputs_exact": {
            phase: gate["exact_parsed_science"]
            for phase, gate in phase_numeric_gates.items()
        },
        "phase_matched_forensic_product_snapshots_exact": {
            phase: gate["exact_forensic_product_snapshots"]
            for phase, gate in phase_numeric_gates.items()
        },
        "strict_exactness_is_strong_evidence_not_universal_requirement": True,
        "same_declared_discrete_science_metadata": True,
        "phase_numeric_gates": phase_numeric_gates,
        "numeric_qualification_status": numeric_qualification_status,
        "numeric_qualification_allowed": numeric_qualification_allowed,
        "nonexact_3x3_policy": "inconclusive_not_pass_or_fail_unless_distribution_or_variance_gate_rejects",
        "recommended_repeat_count_per_arm": 8,
        "recovar_meta_exclusion_policy": {
            "explicit_timing_key_count": (
                len(_ITERATION_PROFILE_TIMING_KEYS)
                + len(_SPARSE_PASS2_TIMING_KEYS)
                + len(_HALFSET_PROFILE_TIMING_KEYS)
            ),
            "explicit_timing_keys_by_container": {
                "vdam_iteration_profile_summary": sorted(_ITERATION_PROFILE_TIMING_KEYS),
                "sparse_pass2_profile_summary": sorted(_SPARSE_PASS2_TIMING_KEYS),
                "halfset_*_profile_summary": sorted(_HALFSET_PROFILE_TIMING_KEYS),
            },
            "forbidden_meta_key_fragments": sorted(_FORBIDDEN_META_KEY_FRAGMENTS),
            "iref_replay_forbidden": True,
            "suffix_or_substring_exclusion_rules_used": False,
        },
        "arithmetic_changed": False,
        "broader_optimized_arithmetic_policy": {
            "stable_repeat_bounded_noise_allowed": True,
            "directional_bias_allowed": False,
            "iteration_amplified_drift_allowed": False,
            "exact_discrete_choices_and_trajectory_basin_required": True,
            "material_final_quality_loss_allowed": False,
            "material_end_to_end_runtime_gain_required": True,
        },
        "production_xhalf_adjoint_program_observed": True,
        "alias_contract_passed": alias_contract_passed,
        "no_material_regression": no_material_regression,
        "material_runtime_win": material_runtime_win,
        "material_stage_win": material_stage_win,
        "material_memory_win": material_memory_win,
        "sampled_memory_diagnostic_below_0_95": sampled_memory_diagnostic_below_0_95,
        "sampled_memory_acceptance_use": "diagnostic_only_not_used_for_acceptance",
        "memory_claim_policy": "disabled_until_defensible_process_high_water_mark",
        "speed_claim_allowed": bool(passed and numeric_qualification_allowed and material_runtime_win),
        "memory_claim_allowed": False,
        "default_promotion_allowed": False,
        "repo_head": expected_repo_head,
        "gpu_uuid": expected_gpu_uuid,
        "input_manifest_sha256": expected_input_manifest_sha256,
        "input_manifest": reference_input_manifest,
        "unique_fresh_jax_cache_count": len(cache_paths),
        "numeric_source_sha256": reference_contract_source,
        "qualifying_programs": qualifying,
        "arm_summary": arm_summary,
        "donated_over_control_ratios": ratios,
        "runs": run_rows,
    }


def main() -> int:
    args = _parse_args()
    payload = analyze(
        args.root.resolve(strict=True),
        expected_repo_head=args.expected_repo_head,
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_input_manifest_sha256=args.expected_input_manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
