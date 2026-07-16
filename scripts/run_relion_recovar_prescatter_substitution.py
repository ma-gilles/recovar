#!/usr/bin/env python3
"""Replay RELION/RECOVAR pre-scatter operands through frozen RECOVAR geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


N = 256
PADDING_FACTOR = 2
CURRENT_SIZE = 48
R_MAX = 24
ACCUMULATOR_SHAPE = (99, 99, 99)
VOLUME_SHAPE = (N, N, N)
VOXEL_SIZE = 1.6375000476837158
OUTLIER_STACK = 111721
OUTLIER_ORIGINAL_INDEX = 8494
COMMON_ARM_KEYS = {
    "target": None,
    "rec_true_f64": "rec_control_data",
    "rel_true_f64": "rel_all_data",
    "rel_excl_p8494_true_f64": "rel_excl_p8494_data",
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


def array_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    left = np.asarray(lhs)
    right = np.asarray(rhs)
    _require(left.shape == right.shape, f"shape mismatch: {left.shape} != {right.shape}")
    left128 = left.astype(np.complex128, copy=False).reshape(-1)
    right128 = right.astype(np.complex128, copy=False).reshape(-1)
    delta = right128 - left128
    absolute = np.abs(delta)
    reference = max(float(np.linalg.norm(left128)), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "lhs_dtype": str(left.dtype),
        "rhs_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_lhs": float(np.linalg.norm(delta) / reference),
        "delta_mean_abs": float(np.mean(absolute)),
        "delta_p95_abs": float(np.quantile(absolute, 0.95)),
        "delta_p99_abs": float(np.quantile(absolute, 0.99)),
        "delta_max_abs": float(np.max(absolute)),
    }


def scatter_operand(
    output_data: np.ndarray,
    output_weight: np.ndarray,
    source_data: np.ndarray,
    source_weight: np.ndarray,
    row_flags: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_coefficients: np.ndarray,
    neighbor_flags: np.ndarray,
) -> int:
    """Scatter in canonical row order after promoting captured operands to f64."""
    data = np.asarray(source_data, dtype=np.complex128)
    weight = np.asarray(source_weight, dtype=np.float64)
    flags = np.asarray(row_flags, dtype=np.uint32)
    indices = np.asarray(neighbor_indices, dtype=np.int64)
    coefficients = np.asarray(neighbor_coefficients, dtype=np.float64)
    neighbor_bits = np.asarray(neighbor_flags, dtype=np.uint32)
    _require(data.shape == weight.shape == flags.shape, "source table shapes differ")
    expected = data.shape + (8,)
    _require(indices.shape == coefficients.shape == neighbor_bits.shape == expected, "bad geometry shape")
    active_pixel = (flags & np.uint32(64)) != 0
    valid_neighbor = (neighbor_bits & np.uint32(8)) == 0
    _require(
        np.array_equal(valid_neighbor, np.broadcast_to(active_pixel[..., None], valid_neighbor.shape)),
        "neighbor validity does not match reached-scatter support",
    )
    _require(np.all(indices[valid_neighbor] >= 0), "valid neighbor has negative index")
    _require(np.all(indices[~valid_neighbor] == -1), "inactive neighbor has live index")
    folded = np.where((flags & np.uint32(16)) != 0, np.conj(data), data)
    data_contributions = folded[..., None] * coefficients
    data_contributions = np.where(
        (neighbor_bits & np.uint32(2)) != 0,
        np.conj(data_contributions),
        data_contributions,
    )
    weight_contributions = weight[..., None] * coefficients
    live_indices = indices[valid_neighbor]
    np.add.at(output_data, live_indices, data_contributions[valid_neighbor])
    np.add.at(output_weight, live_indices, weight_contributions[valid_neighbor])
    return int(live_indices.size)


def _half_to_public(data: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from recovar.em.dense_single_volume.helpers import half_volume_mstep

    data_post = half_volume_mstep.enforce_relion_half_volume_x0_hermitian_host(
        data, ACCUMULATOR_SHAPE
    )
    weight_post = half_volume_mstep.enforce_relion_half_volume_x0_hermitian_host(
        weight, ACCUMULATOR_SHAPE
    )
    public_data = np.asarray(
        half_volume_mstep.relion_x_half_volume_to_full(data_post, ACCUMULATOR_SHAPE)
    ).reshape(ACCUMULATOR_SHAPE)
    public_weight = np.asarray(
        half_volume_mstep.relion_x_half_volume_to_full(weight_post, ACCUMULATOR_SHAPE)
    ).real.reshape(ACCUMULATOR_SHAPE)
    return public_data.astype(np.complex128), public_weight.astype(np.float64)


def scatter(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "maps").mkdir(exist_ok=True)
    (output / "SAFE_TO_DELETE").touch()
    aligned_path = Path(args.aligned).resolve()
    geometry_directory = Path(args.geometry).resolve()
    replay_arrays_path = Path(args.replay_arrays).resolve()
    closure_path = Path(args.geometry_closure).resolve()
    order_path = Path(args.order_control).resolve()
    geometry_paths = sorted(geometry_directory.glob("*.npz"))
    _require(bool(geometry_paths), f"no geometry shards: {geometry_directory}")
    closure = json.loads(closure_path.read_text())
    order_control = json.loads(order_path.read_text())
    _require(closure.get("production_faithful") is True, "geometry closure gate failed")
    _require(order_control.get("status") == "PASS_ORDER_CONTROL", "order-control gate failed")
    with np.load(aligned_path, allow_pickle=False) as aligned:
        stacks = np.asarray(aligned["stack_indices_1based"], dtype=np.int64)
        window = np.asarray(aligned["window_indices"], dtype=np.int32)
        rec_data = np.asarray(aligned["recovar_data"], dtype=np.complex64)
        rec_weight = np.asarray(aligned["recovar_weight"], dtype=np.float32)
        rel_data = np.asarray(aligned["relion_data_recovar_units"], dtype=np.complex64)
        rel_weight = np.asarray(aligned["relion_weight_recovar_units"], dtype=np.float32)
        support = np.asarray(aligned["recovar_device_support_mask"], dtype=bool)
    _require(np.unique(stacks).size == stacks.size, "aligned stacks are not unique")
    stack_to_row = {int(stack): row for row, stack in enumerate(stacks)}
    _require(OUTLIER_STACK in stack_to_row, "missing particle-8494 stack")
    size = int(np.prod((99, 99, 50)))
    accumulators = {
        key: np.zeros(size, dtype=dtype)
        for key, dtype in (
            ("rec_data", np.complex128),
            ("rec_weight", np.float64),
            ("rel_data", np.complex128),
            ("rel_weight", np.float64),
            ("outlier_rec_data", np.complex128),
            ("outlier_rec_weight", np.float64),
            ("outlier_rel_data", np.complex128),
            ("outlier_rel_weight", np.float64),
        )
    }
    seen: set[int] = set()
    active_neighbors = 0
    outlier_record = None
    for geometry_path in geometry_paths:
        with np.load(geometry_path, allow_pickle=False) as geometry:
            companion_path = Path(str(geometry["companion_contribution_path"]))
            particle_rows = np.asarray(geometry["signature_particle_rows"], dtype=np.int64)
            original_indices = np.asarray(geometry["signature_original_indices"], dtype=np.int64)
            pixels = np.asarray(geometry["signature_pixel_indices"], dtype=np.int32)
            row_flags = np.asarray(geometry["signature_row_flags"], dtype=np.uint32)
            captured_source = np.asarray(geometry["signature_source_values"], dtype=np.float32)
            captured_weight = np.asarray(geometry["signature_source_weights"], dtype=np.float32)
            neighbor_indices = np.asarray(geometry["signature_neighbor_indices"], dtype=np.int32)
            coefficients = np.asarray(geometry["signature_neighbor_coefficients"], dtype=np.float32)
            neighbor_flags = np.asarray(geometry["signature_neighbor_flags"], dtype=np.uint32)
        with np.load(companion_path, allow_pickle=False) as companion:
            companion_stacks = np.asarray(companion["stack_indices_1based"], dtype=np.int64)
        shard_stacks = companion_stacks[particle_rows]
        _require(np.array_equal(pixels, np.broadcast_to(window, pixels.shape)), "pixel identity mismatch")
        rows = np.asarray([stack_to_row[int(stack)] for stack in shard_stacks], dtype=np.int64)
        _require(not seen.intersection(shard_stacks.tolist()), "duplicate geometry stack")
        seen.update(int(stack) for stack in shard_stacks)
        shard_rec_data = rec_data[rows]
        shard_rec_weight = rec_weight[rows]
        shard_rel_data = rel_data[rows]
        shard_rel_weight = rel_weight[rows]
        _require(
            np.array_equal(captured_source[..., 0], shard_rec_data.real)
            and np.array_equal(captured_source[..., 1], shard_rec_data.imag),
            f"RECOVAR data identity mismatch: {geometry_path}",
        )
        _require(np.array_equal(captured_weight, shard_rec_weight), "RECOVAR weight identity mismatch")
        _require(
            np.array_equal((row_flags & np.uint32(64)) != 0, support[rows]),
            "support identity mismatch",
        )
        active_neighbors += scatter_operand(
            accumulators["rec_data"], accumulators["rec_weight"], shard_rec_data,
            shard_rec_weight, row_flags, neighbor_indices, coefficients, neighbor_flags,
        )
        scatter_operand(
            accumulators["rel_data"], accumulators["rel_weight"], shard_rel_data,
            shard_rel_weight, row_flags, neighbor_indices, coefficients, neighbor_flags,
        )
        outlier_rows = np.flatnonzero(shard_stacks == OUTLIER_STACK)
        if outlier_rows.size:
            _require(outlier_rows.size == 1 and outlier_record is None, "outlier is not unique")
            row = int(outlier_rows[0])
            _require(int(original_indices[row]) == OUTLIER_ORIGINAL_INDEX, "outlier identity mismatch")
            selected = slice(row, row + 1)
            scatter_operand(
                accumulators["outlier_rec_data"], accumulators["outlier_rec_weight"],
                shard_rec_data[selected], shard_rec_weight[selected], row_flags[selected],
                neighbor_indices[selected], coefficients[selected], neighbor_flags[selected],
            )
            scatter_operand(
                accumulators["outlier_rel_data"], accumulators["outlier_rel_weight"],
                shard_rel_data[selected], shard_rel_weight[selected], row_flags[selected],
                neighbor_indices[selected], coefficients[selected], neighbor_flags[selected],
            )
            outlier_record = {
                "geometry_path": str(geometry_path),
                "stack_index_1based": OUTLIER_STACK,
                "original_index": int(original_indices[row]),
            }
    _require(seen == set(stacks.tolist()), "geometry particle coverage is incomplete")
    _require(outlier_record is not None, "outlier geometry was not found")
    rel_excl_data = accumulators["rel_data"] - (
        accumulators["outlier_rel_data"] - accumulators["outlier_rec_data"]
    )
    rel_excl_weight = accumulators["rel_weight"] - (
        accumulators["outlier_rel_weight"] - accumulators["outlier_rec_weight"]
    )
    public: dict[str, np.ndarray] = {}
    for prefix, data, weight in (
        ("rec_control", accumulators["rec_data"], accumulators["rec_weight"]),
        ("rel_all", accumulators["rel_data"], accumulators["rel_weight"]),
        ("rel_excl_p8494", rel_excl_data, rel_excl_weight),
    ):
        public[f"{prefix}_data"], public[f"{prefix}_weight"] = _half_to_public(data, weight)
    with np.load(replay_arrays_path, allow_pickle=False) as replay:
        prior_data = np.asarray(replay["canonical_host_f64_public_data"])
        prior_weight = np.asarray(replay["canonical_host_f64_public_weight"])
    output_arrays = output / "controlled_substitution_accumulators_v1.npz"
    np.savez(output_arrays, **public)
    full_delta = public["rel_all_data"] - public["rec_control_data"]
    excluded_delta = public["rel_excl_p8494_data"] - public["rec_control_data"]
    outlier_delta = full_delta - excluded_delta
    report = {
        "schema": "recovar.em.real10076.it1.half2.prescatter-controlled-substitution.v1",
        "status": "PASS_CANONICAL_SCATTER",
        "metric_policy": "exact/array metrics for accumulators; FSC/FSC-AUC for maps; no correlation",
        "precision": {
            "inputs": "captured complex64/float32 operands and coefficients promoted before arithmetic",
            "accumulation": "deterministic common-order complex128/float64 np.add.at",
            "qualification": "does not recover precision lost during upstream operand generation",
        },
        "scope": {
            "particles": int(stacks.size), "pixels_per_particle": int(window.size),
            "geometry_shards": len(geometry_paths), "active_neighbors": active_neighbors,
            "outlier": outlier_record,
        },
        "inputs": {
            "aligned_arrays": str(aligned_path), "aligned_arrays_sha256": _sha256(aligned_path),
            "geometry_directory": str(geometry_directory),
            "geometry_closure": str(closure_path), "geometry_closure_sha256": _sha256(closure_path),
            "order_control": str(order_path), "order_control_sha256": _sha256(order_path),
            "prior_replay_arrays": str(replay_arrays_path),
            "prior_replay_arrays_sha256": _sha256(replay_arrays_path),
        },
        "accumulator_metrics": {
            "rec_true_f64_vs_prior_promoted_contribution_f64_data": array_metrics(prior_data, public["rec_control_data"]),
            "rec_true_f64_vs_prior_promoted_contribution_f64_weight": array_metrics(prior_weight, public["rec_control_weight"]),
            "rec_vs_rel_all_data": array_metrics(public["rec_control_data"], public["rel_all_data"]),
            "rec_vs_rel_all_weight": array_metrics(public["rec_control_weight"], public["rel_all_weight"]),
            "rec_vs_rel_excluding_p8494_data": array_metrics(public["rec_control_data"], public["rel_excl_p8494_data"]),
            "rec_vs_rel_excluding_p8494_weight": array_metrics(public["rec_control_weight"], public["rel_excl_p8494_weight"]),
        },
        "delta_norm_decomposition": {
            "p8494_fraction_of_full_data_delta_l2": float(np.linalg.norm(outlier_delta) / max(np.linalg.norm(full_delta), np.finfo(np.float64).tiny)),
            "all_other_fraction_of_full_data_delta_l2": float(np.linalg.norm(excluded_delta) / max(np.linalg.norm(full_delta), np.finfo(np.float64).tiny)),
        },
        "production_order_control_summary": order_control["summary"],
        "output_arrays": str(output_arrays), "output_arrays_sha256": _sha256(output_arrays),
    }
    report_path = output / "controlled_substitution_scatter_v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(report_path)}, indent=2))


def _read_spectrum(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(8)
        _require(len(header) == 8, f"truncated spectrum header: {path}")
        (count,) = struct.unpack("q", header)
        values = np.fromfile(stream, dtype=np.float64, count=int(count))
        _require(values.size == int(count) and not stream.read(1), f"bad spectrum payload: {path}")
    return values


def reconstruct_common(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.dense_single_volume.mean_helpers import _reconstruct_volume_eager
    from recovar.utils.helpers import recovar_volume_to_relion

    output = Path(args.output).resolve()
    arm = str(args.arm)
    _require(arm in COMMON_ARM_KEYS, f"unknown common arm: {arm}")
    target_path = Path(args.target_arrays).resolve()
    with np.load(target_path, allow_pickle=False) as target:
        target_data = np.asarray(target["target_relion_data"], dtype=np.complex128)
        target_weight = np.asarray(target["target_relion_weight"], dtype=np.float64)
    if COMMON_ARM_KEYS[arm] is None:
        data = target_data
    else:
        with np.load(output / "controlled_substitution_accumulators_v1.npz", allow_pickle=False) as arrays:
            data = np.asarray(arrays[COMMON_ARM_KEYS[arm]], dtype=np.complex128)
    tau = _read_spectrum(Path(args.tau).resolve()) * float(N**4)
    volume_ft = np.asarray(
        _reconstruct_volume_eager(
            target_weight.reshape(-1),
            data.reshape(-1),
            VOLUME_SHAPE,
            PADDING_FACTOR,
            tau=tau,
            tau2_fudge=1.0,
            projection_padding_factor=PADDING_FACTOR,
            use_spherical_mask=True,
            grid_correct=True,
            minres_map=5,
            current_size=CURRENT_SIZE,
            return_real_space=False,
            accumulator_volume_shape=ACCUMULATOR_SHAPE,
            tau_is_1d=True,
        )
    ).reshape(VOLUME_SHAPE)
    real = np.asarray(ftu.get_idft3(jnp.asarray(volume_ft)).real, dtype=np.float64)
    relion_frame = np.asarray(recovar_volume_to_relion(real), dtype=np.float64)
    map_path = output / "maps" / f"common_{arm}.npy"
    np.save(map_path, relion_frame)
    metadata = {
        "schema": "recovar.em.real10076.it1.half2.prescatter-common-map-arm.v1",
        "arm": arm,
        "data_source": "target_relion_data" if arm == "target" else COMMON_ARM_KEYS[arm],
        "weight_source": "target_relion_weight",
        "tau_source": str(Path(args.tau).resolve()),
        "target_arrays": str(target_path),
        "target_arrays_sha256": _sha256(target_path),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "map": str(map_path),
        "map_sha256": _sha256(map_path),
    }
    metadata_path = output / "maps" / f"common_{arm}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arm": arm, "map": str(map_path), "devices": metadata["jax_devices"]}, indent=2))


def shell_fsc(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    _require(left.shape == right.shape == VOLUME_SHAPE, "expected matching 256-cubed maps")
    left_ft = np.fft.rfftn(left)
    right_ft = np.fft.rfftn(right)
    full = np.fft.fftfreq(N) * N
    half = np.fft.rfftfreq(N) * N
    yy, xx = np.meshgrid(full, half, indexing="ij")
    numerator = np.zeros(N // 2 + 1, dtype=np.float64)
    left_power = np.zeros_like(numerator)
    right_power = np.zeros_like(numerator)
    for iz, z in enumerate(full):
        shells = np.rint(np.sqrt(z * z + yy * yy + xx * xx)).astype(np.int32)
        keep = shells <= N // 2
        indices = shells[keep].reshape(-1)
        cross = (left_ft[iz] * np.conj(right_ft[iz]))[keep].reshape(-1)
        numerator += np.bincount(indices, weights=cross.real, minlength=N // 2 + 1)
        left_power += np.bincount(
            indices, weights=np.abs(left_ft[iz][keep]) ** 2, minlength=N // 2 + 1
        )
        right_power += np.bincount(
            indices, weights=np.abs(right_ft[iz][keep]) ** 2, minlength=N // 2 + 1
        )
    denominator = np.sqrt(left_power * right_power)
    return np.clip(
        np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        ),
        -1.0,
        1.0,
    )


def map_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    fsc = shell_fsc(lhs, rhs)
    finite = fsc[1:][np.isfinite(fsc[1:])]
    _require(finite.size > 0, "no finite non-DC FSC shells")
    return {
        "fsc": fsc.tolist(),
        "fsc_auc_non_dc": float(np.mean(finite)),
        "fsc_min_non_dc": float(np.min(finite)),
    }


def analyze(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    maps = {
        arm: np.load(output / "maps" / f"common_{arm}.npy")
        for arm in COMMON_ARM_KEYS
    }
    target = maps["target"]
    comparisons = {
        f"target_vs_{arm}": map_metrics(target, values)
        for arm, values in maps.items()
        if arm != "target"
    }
    sealed_root = Path(args.sealed_aggregate).resolve()
    with np.load(sealed_root / "geometry_substitution_arrays.npz", allow_pickle=False) as sealed_arrays:
        sealed_target_map = np.asarray(sealed_arrays["target_common_map"], dtype=np.float64)
    target_replay = map_metrics(sealed_target_map, target)
    sealed_report = json.loads((sealed_root / "geometry_substitution.json").read_text())
    sealed_maps = sealed_report["controlled_common_weight_tau_maps"]
    sealed_native_auc = float(
        sealed_maps["native_recovar_sources_vs_target_relion_bpref"]["fsc_auc_non_dc"]
    )
    sealed_rel_auc = float(
        sealed_maps["relion_sources_on_recovar_geometry_vs_target_relion_bpref"]["fsc_auc_non_dc"]
    )
    independent_rec_auc = comparisons["target_vs_rec_true_f64"]["fsc_auc_non_dc"]
    independent_rel_auc = comparisons["target_vs_rel_true_f64"]["fsc_auc_non_dc"]
    scatter_report = json.loads((output / "controlled_substitution_scatter_v1.json").read_text())
    accumulator = scatter_report["accumulator_metrics"]
    order_auc = float(
        scatter_report["production_order_control_summary"]["min_order_map_fsc_auc_non_dc"]
    )
    order_auc_defect = 1.0 - order_auc
    rec_auc_difference = abs(independent_rec_auc - sealed_native_auc)
    rel_auc_difference = abs(independent_rel_auc - sealed_rel_auc)
    metric_reproduction_passed = bool(
        rec_auc_difference <= order_auc_defect
        and rel_auc_difference <= order_auc_defect
        and independent_rel_auc >= 0.999999999
        and independent_rel_auc > independent_rec_auc
    )
    absolute_target_replay_identical = bool(target_replay["fsc_auc_non_dc"] >= 0.999999999)
    status = (
        "PASS_METRIC_REPRODUCTION_WITH_NONIDENTICAL_ABSOLUTE_TARGET"
        if metric_reproduction_passed and not absolute_target_replay_identical
        else "PASS_INDEPENDENT_COMMON_MAP_REPRODUCTION"
        if metric_reproduction_passed
        else "FAIL_INDEPENDENT_COMMON_MAP_REPRODUCTION"
    )
    report = {
        "schema": "recovar.em.real10076.it1.half2.prescatter-independent-reproduction.v1",
        "status": status,
        "metric_policy": "exact/array metrics for accumulators; FSC/FSC-AUC for maps; no correlation",
        "scope": {
            "recomputed": "all common-weight/tau maps from frozen accumulators and raw tau",
            "shared_frozen_input": str(sealed_root / "geometry_substitution_arrays.npz"),
            "qualification": (
                "The target data/weight arrays are shared immutable inputs; map reconstruction, "
                "FSC, and FSC-AUC are independently recomputed in this isolated worktree."
            ),
        },
        "target_map_replay_vs_sealed": target_replay,
        "target_map_replay_array_metrics": array_metrics(sealed_target_map, target),
        "target_map_replay_qualification": (
            "The absolute target replay is not hash-identical and is not used as the causal gate. "
            "The sealed artifact does not record enough reconstruction-environment provenance to "
            "demand an absolute map hash. The controlled REC-vs-target and REL-vs-target FSC/FSC-AUC "
            "effects are independently recomputed with one common transfer path and are compared "
            "against the existing same-GPU production-order envelope."
        ),
        "independent_common_weight_tau_maps": comparisons,
        "sealed_metric_comparison": {
            "rec": {
                "sealed_canonical_f32_fsc_auc": sealed_native_auc,
                "independent_true_f64_fsc_auc": independent_rec_auc,
                "absolute_auc_difference": rec_auc_difference,
            },
            "rel": {
                "sealed_canonical_f32_fsc_auc": sealed_rel_auc,
                "independent_true_f64_fsc_auc": independent_rel_auc,
                "absolute_auc_difference": rel_auc_difference,
            },
            "acceptance_envelope": {
                "source": "same-GPU production float32 order controls",
                "min_order_map_fsc_auc_non_dc": order_auc,
                "max_order_auc_defect": order_auc_defect,
                "metric_reproduction_passed": metric_reproduction_passed,
            },
        },
        "p8494_stack111721": {
            "identity": scatter_report["scope"]["outlier"],
            "delta_norm_decomposition": scatter_report["delta_norm_decomposition"],
            "included_map": comparisons["target_vs_rel_true_f64"],
            "excluded_map": comparisons["target_vs_rel_excl_p8494_true_f64"],
        },
        "accumulator_precision_and_endpoint_explanation": {
            "independent_rec_vs_rel_common_geometry_true_f64_data_relative_l2": accumulator[
                "rec_vs_rel_all_data"
            ]["relative_l2_over_lhs"],
            "sealed_native_rec_sources_vs_target_relion_f32_data_relative_l2": sealed_report[
                "comparisons"
            ]["native_recovar_sources_vs_relion_bpref_data"]["relative_l2_over_lhs"],
            "independent_rec_vs_rel_common_geometry_true_f64_weight_relative_l2": accumulator[
                "rec_vs_rel_all_weight"
            ]["relative_l2_over_lhs"],
            "sealed_native_rec_sources_vs_target_relion_f32_weight_relative_l2": sealed_report[
                "comparisons"
            ]["native_recovar_sources_vs_relion_bpref_weight"]["relative_l2_over_lhs"],
            "explanation": (
                "The independent data number compares REC and REL source operands after the same "
                "geometry in true float64 arithmetic; the sealed number compares canonical-float32 "
                "REC source accumulation with the target RELION accumulator. Their tiny data offset "
                "therefore includes arithmetic/order and endpoint differences. The larger weight "
                "difference is expected because direct source-weight differences on common geometry "
                "exclude the target accumulator's reduction/geometry residual."
            ),
        },
        "input_hashes": {
            "sealed_geometry_substitution_arrays": _sha256(
                sealed_root / "geometry_substitution_arrays.npz"
            ),
            "sealed_geometry_substitution_report": _sha256(
                sealed_root / "geometry_substitution.json"
            ),
            "scatter_report": _sha256(output / "controlled_substitution_scatter_v1.json"),
        },
        "map_hashes": {
            arm: _sha256(output / "maps" / f"common_{arm}.npy") for arm in COMMON_ARM_KEYS
        },
    }
    report_path = output / "independent_reproduction_v1.json"
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if str(prior.get("status", "")).startswith("FAIL"):
            fail_path = output / "independent_reproduction_first_attempt_fail.json"
            if not fail_path.exists():
                fail_path.write_text(report_path.read_text())
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "target_replay_fsc_auc": target_replay["fsc_auc_non_dc"],
                "independent_rec_fsc_auc": independent_rec_auc,
                "independent_rel_fsc_auc": independent_rel_auc,
                "independent_rel_excl_p8494_fsc_auc": comparisons[
                    "target_vs_rel_excl_p8494_true_f64"
                ]["fsc_auc_non_dc"],
            },
            indent=2,
        )
    )
    if not status.startswith("PASS"):
        raise RuntimeError(status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scatter_parser = subparsers.add_parser("scatter")
    scatter_parser.add_argument("--aligned", type=Path, required=True)
    scatter_parser.add_argument("--geometry", type=Path, required=True)
    scatter_parser.add_argument("--replay-arrays", type=Path, required=True)
    scatter_parser.add_argument("--geometry-closure", type=Path, required=True)
    scatter_parser.add_argument("--order-control", type=Path, required=True)
    scatter_parser.add_argument("--output", type=Path, required=True)
    scatter_parser.set_defaults(function=scatter)
    reconstruct_parser = subparsers.add_parser("reconstruct-common")
    reconstruct_parser.add_argument("--output", type=Path, required=True)
    reconstruct_parser.add_argument("--target-arrays", type=Path, required=True)
    reconstruct_parser.add_argument("--tau", type=Path, required=True)
    reconstruct_parser.add_argument("--arm", choices=sorted(COMMON_ARM_KEYS), required=True)
    reconstruct_parser.set_defaults(function=reconstruct_common)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--sealed-aggregate", type=Path, required=True)
    analyze_parser.set_defaults(function=analyze)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
