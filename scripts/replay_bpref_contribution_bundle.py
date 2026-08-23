#!/usr/bin/env python3
"""Replay one exact-local BPref row bundle across order and precision controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from recovar.em.bpref_contribution_replay import (
    BPrefAccumulatorReplay,
    accumulator_replay_metrics,
    load_bpref_contribution_bundle,
    replay_relion_double,
    sha256_file,
    summarize_bpref_contribution_bundle,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--relion-bind-dir", type=Path)
    parser.add_argument("--skip-relion", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--gpu-repeats", type=int, default=2)
    return parser.parse_args(argv)


def _replay_gpu(
    bundle,
    *,
    order: str,
    precision: str,
) -> BPrefAccumulatorReplay:
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject

    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU BPref replay requires a JAX GPU backend")
    if precision == "float32":
        complex_dtype, real_dtype = jnp.complex64, jnp.float32
    elif precision == "float64":
        complex_dtype, real_dtype = jnp.complex128, jnp.float64
    else:
        raise ValueError(f"unknown BPref replay precision {precision!r}")

    rows = bundle.concatenate(order)
    boundary = bundle.boundary_values
    image_shape = tuple(int(value) for value in np.asarray(boundary["image_shape"]))
    volume_shape = tuple(int(value) for value in np.asarray(boundary["volume_shape"]))
    half_shape = (*volume_shape[:2], volume_shape[2] // 2 + 1)
    accumulator_size = int(np.prod(half_shape))
    pixel_indices = jnp.asarray(boundary["window_indices"], dtype=jnp.int32)
    data = jnp.zeros((accumulator_size,), dtype=complex_dtype)
    weight = jnp.zeros((accumulator_size,), dtype=real_dtype)

    # Keep the captured execution-shard partition sizes fixed for both row
    # orders. Schema v3 does not contain production packed-zero rows, so this
    # topology is explicitly a captured-active-row control rather than a claim
    # of bitwise production-launch replay.
    start = 0
    for row_count in bundle.shard_row_counts:
        stop = start + int(row_count)
        rotations = jnp.asarray(rows["active_rotations"][start:stop], dtype=real_dtype)
        data = cuda_backproject.backproject_indexed(
            data,
            jnp.asarray(rows["active_summed"][start:stop], dtype=complex_dtype),
            pixel_indices,
            rotations,
            image_shape=image_shape,
            volume_shape=volume_shape,
            order=1,
            half_volume=True,
            half_image=True,
            max_r=float(int(np.asarray(boundary["current_size"]).item()) // 2),
            relion_x_half=True,
        )
        weight = cuda_backproject.backproject_indexed(
            weight,
            jnp.asarray(rows["active_ctf_probs"][start:stop], dtype=real_dtype),
            pixel_indices,
            rotations,
            image_shape=image_shape,
            volume_shape=volume_shape,
            order=1,
            half_volume=True,
            half_image=True,
            max_r=float(int(np.asarray(boundary["current_size"]).item()) // 2),
            relion_x_half=True,
        )
        start = stop
    data, weight = jax.device_get((data, weight))
    return BPrefAccumulatorReplay(
        data=np.asarray(data).reshape(half_shape),
        weight=np.asarray(weight).reshape(half_shape),
        backend="recovar_cuda_backproject_indexed",
        order=order,
        precision=("complex64/float32" if precision == "float32" else "complex128/float64"),
        launch_topology="captured_active_rows_shard_partitions",
    )


def _replay_key(replay: BPrefAccumulatorReplay, repeat: int = 0) -> str:
    backend = "gpu" if replay.backend.startswith("recovar_cuda") else "relion"
    precision = "f32" if replay.precision == "complex64/float32" else "f64"
    return f"{backend}_{precision}_{replay.order}_repeat{repeat}"


def _sha256_arrays(data: np.ndarray, weight: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (np.asarray(data), np.asarray(weight)):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _comparison(replays, left_key, right_key):
    return accumulator_replay_metrics(replays[left_key], replays[right_key])


def _reconstruct_unregularized_map(replay, boundary):
    """Transfer one replay through the shared no-prior reconstruction path."""

    from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
        relion_x_half_volume_to_native_half,
    )
    from recovar.em.dense_single_volume.local_backprojection import (
        enforce_relion_half_volume_x0_hermitian_host,
    )
    from recovar.em.dense_single_volume.mean_helpers import _reconstruct_volume_eager

    accumulator_shape = tuple(int(value) for value in np.asarray(boundary["volume_shape"]))
    image_shape = tuple(int(value) for value in np.asarray(boundary["image_shape"]))
    if image_shape[0] != image_shape[1]:
        raise ValueError("BPref replay reconstruction requires a square particle image")
    volume_shape = (image_shape[0],) * 3
    data = enforce_relion_half_volume_x0_hermitian_host(
        np.asarray(replay.data).reshape(-1), accumulator_shape
    )
    weight = enforce_relion_half_volume_x0_hermitian_host(
        np.asarray(replay.weight).reshape(-1), accumulator_shape
    )
    data = relion_x_half_volume_to_native_half(data, accumulator_shape)
    weight = relion_x_half_volume_to_native_half(weight, accumulator_shape)
    reconstructed = _reconstruct_volume_eager(
        weight,
        data,
        volume_shape,
        int(np.asarray(boundary["reconstruction_padding_factor"]).item()),
        tau=None,
        tau2_fudge=1.0,
        projection_padding_factor=int(np.asarray(boundary["projection_padding_factor"]).item()),
        use_spherical_mask=False,
        grid_correct=False,
        minres_map=0,
        current_size=int(np.asarray(boundary["current_size"]).item()),
        return_real_space=True,
        accumulator_volume_shape=accumulator_shape,
    )
    reconstructed = np.asarray(reconstructed).real.reshape(volume_shape)
    if not np.all(np.isfinite(reconstructed)):
        raise ValueError("BPref replay reconstruction contains nonfinite values")
    return reconstructed


def _shell_fsc(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or len(set(left.shape)) != 1:
        raise ValueError("map FSC requires equal cubic 3-D arrays")
    size = int(left.shape[0])
    left_ft = np.fft.fftn(left)
    right_ft = np.fft.fftn(right)
    axis = np.fft.fftfreq(size) * size
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    shells = np.rint(np.sqrt(x * x + y * y + z * z)).astype(np.int32).reshape(-1)
    product = (left_ft * np.conj(right_ft)).reshape(-1)
    numerator = np.bincount(shells, weights=np.real(product))
    left_power = np.bincount(shells, weights=(np.abs(left_ft) ** 2).reshape(-1))
    right_power = np.bincount(shells, weights=(np.abs(right_ft) ** 2).reshape(-1))
    denominator = np.sqrt(left_power * right_power)
    fsc = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=fsc, where=denominator > 0.0)
    return fsc[: size // 2 - 1]


def _normalized_fsc_auc(fsc):
    values = np.asarray(fsc, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if finite.size:
        finite[0] = False
    if np.count_nonzero(finite) < 2:
        return float("nan")
    axis = np.flatnonzero(finite).astype(np.float64)
    span = float(axis[-1] - axis[0])
    if span <= 0.0 or not math.isfinite(span):
        return float("nan")
    axis = (axis - axis[0]) / span
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(values[finite], axis))


def _map_fsc_metrics(left, right):
    fsc = _shell_fsc(left, right)
    finite_non_dc = np.isfinite(fsc)
    if finite_non_dc.size:
        finite_non_dc[0] = False
    return {
        "fsc_auc": _normalized_fsc_auc(fsc),
        "min_fsc_non_dc": float(np.min(fsc[finite_non_dc], initial=np.inf)),
        "fsc": fsc.tolist(),
    }


def _classify_replay_difference(comparisons, map_comparisons):
    """Classify the earliest supported scatter difference from control scales."""

    required = {
        "control_repeat_f32_canonical_1",
        "gpu_order_only_f32",
        "gpu_precision_canonical",
        "gpu_vs_relion_f64_canonical",
    }
    if not required.issubset(comparisons):
        return {
            "classification": "unresolved",
            "reason": "both GPU repeats and the RELION double backend are required",
        }

    def scale(name):
        comparison = comparisons[name]
        return max(
            float(comparison["data"]["relative_l2"]),
            float(comparison["weight"]["relative_l2"]),
        )

    repeat_scale = scale("control_repeat_f32_canonical_1")
    order_scale = scale("gpu_order_only_f32")
    precision_scale = scale("gpu_precision_canonical")
    cross_backend_f64_scale = scale("gpu_vs_relion_f64_canonical")
    numerical_floor = max(1e-12, 100.0 * cross_backend_f64_scale)
    if cross_backend_f64_scale > 1e-10:
        classification = "geometry_or_backend_arithmetic"
        reason = "common canonical float64 replay does not close near machine precision"
    elif precision_scale > 100.0 * max(repeat_scale, order_scale, numerical_floor):
        classification = "scatter_precision"
        reason = (
            "common canonical float64 replay closes across backends while the float32/float64 "
            "difference dominates repeat and order controls"
        )
    elif order_scale > 10.0 * max(repeat_scale, numerical_floor):
        classification = "reduction_order"
        reason = "row-order variation dominates control-repeat variation"
    elif repeat_scale > numerical_floor:
        classification = "nondeterministic_reduction"
        reason = "control-repeat variation is the leading resolved difference"
    else:
        classification = "numerical_floor"
        reason = "all resolved differences are at the configured numerical floor"
    result = {
        "classification": classification,
        "reason": reason,
        "control_repeat_f32_relative_l2_max": repeat_scale,
        "order_only_f32_relative_l2_max": order_scale,
        "precision_canonical_relative_l2_max": precision_scale,
        "cross_backend_f64_canonical_relative_l2_max": cross_backend_f64_scale,
    }
    map_metrics = map_comparisons.get("gpu_precision_canonical")
    if map_metrics is not None:
        result["precision_control_unregularized_map_fsc_auc"] = float(
            map_metrics["fsc_auc"]
        )
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.skip_gpu and args.skip_relion:
        raise ValueError("at least one replay backend must be enabled")
    if args.gpu_repeats <= 0:
        raise ValueError("--gpu-repeats must be positive")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.relion_bind_dir is not None:
        os.environ["RECOVAR_RELION_BIND_BUILD_DIR"] = str(args.relion_bind_dir.resolve())

    bundle = load_bpref_contribution_bundle(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replays = {}
    if not args.skip_gpu:
        for precision in ("float32", "float64"):
            for order in ("execution", "canonical"):
                for repeat in range(args.gpu_repeats):
                    replay = _replay_gpu(bundle, order=order, precision=precision)
                    replays[_replay_key(replay, repeat)] = replay
    if not args.skip_relion:
        from recovar.relion_bind._relion_bind_core import TRILINEAR, get_backprojector_data

        for order in ("execution", "canonical"):
            replay = replay_relion_double(
                bundle,
                order=order,
                get_backprojector_data=get_backprojector_data,
                interpolator=TRILINEAR,
            )
            replays[_replay_key(replay)] = replay

    arrays_path = args.output_dir / "bpref_accumulator_replays_v1.npz"
    arrays = {}
    replay_inventory = {}
    reconstructed_maps = {}
    for key, replay in replays.items():
        arrays[f"{key}_data"] = replay.data
        arrays[f"{key}_weight"] = replay.weight
        reconstructed_maps[key] = _reconstruct_unregularized_map(replay, bundle.boundary_values)
        arrays[f"{key}_unregularized_map"] = reconstructed_maps[key]
        replay_inventory[key] = {
            "backend": replay.backend,
            "order": replay.order,
            "precision": replay.precision,
            "launch_topology": replay.launch_topology,
            "data_dtype": str(replay.data.dtype),
            "weight_dtype": str(replay.weight.dtype),
            "shape": list(replay.data.shape),
            "arrays_sha256": _sha256_arrays(replay.data, replay.weight),
            "unregularized_map_dtype": str(reconstructed_maps[key].dtype),
            "unregularized_map_shape": list(reconstructed_maps[key].shape),
            "unregularized_map_sha256": hashlib.sha256(
                np.ascontiguousarray(reconstructed_maps[key]).tobytes()
            ).hexdigest(),
        }
    np.savez(arrays_path, **arrays)

    comparisons = {}
    if not args.skip_gpu:
        for precision in ("f32", "f64"):
            for order in ("execution", "canonical"):
                base = f"gpu_{precision}_{order}_repeat0"
                for repeat in range(1, args.gpu_repeats):
                    comparisons[f"control_repeat_{precision}_{order}_{repeat}"] = _comparison(
                        replays, base, f"gpu_{precision}_{order}_repeat{repeat}"
                    )
            comparisons[f"gpu_order_only_{precision}"] = _comparison(
                replays,
                f"gpu_{precision}_execution_repeat0",
                f"gpu_{precision}_canonical_repeat0",
            )
        comparisons["gpu_precision_canonical"] = _comparison(
            replays, "gpu_f32_canonical_repeat0", "gpu_f64_canonical_repeat0"
        )
    if not args.skip_relion:
        comparisons["relion_order_only_f64"] = _comparison(
            replays, "relion_f64_execution_repeat0", "relion_f64_canonical_repeat0"
        )
    if not args.skip_gpu and not args.skip_relion:
        comparisons["gpu_vs_relion_f64_execution"] = _comparison(
            replays, "gpu_f64_execution_repeat0", "relion_f64_execution_repeat0"
        )
        comparisons["gpu_vs_relion_f64_canonical"] = _comparison(
            replays, "gpu_f64_canonical_repeat0", "relion_f64_canonical_repeat0"
        )

    map_comparisons = {}
    if not args.skip_gpu:
        for precision in ("f32", "f64"):
            for order in ("execution", "canonical"):
                base = f"gpu_{precision}_{order}_repeat0"
                for repeat in range(1, args.gpu_repeats):
                    other = f"gpu_{precision}_{order}_repeat{repeat}"
                    map_comparisons[f"control_repeat_{precision}_{order}_{repeat}"] = (
                        _map_fsc_metrics(reconstructed_maps[base], reconstructed_maps[other])
                    )
            map_comparisons[f"gpu_order_only_{precision}"] = _map_fsc_metrics(
                reconstructed_maps[f"gpu_{precision}_execution_repeat0"],
                reconstructed_maps[f"gpu_{precision}_canonical_repeat0"],
            )
        map_comparisons["gpu_precision_canonical"] = _map_fsc_metrics(
            reconstructed_maps["gpu_f32_canonical_repeat0"],
            reconstructed_maps["gpu_f64_canonical_repeat0"],
        )
    if not args.skip_relion:
        map_comparisons["relion_order_only_f64"] = _map_fsc_metrics(
            reconstructed_maps["relion_f64_execution_repeat0"],
            reconstructed_maps["relion_f64_canonical_repeat0"],
        )
    if not args.skip_gpu and not args.skip_relion:
        map_comparisons["gpu_vs_relion_f64_canonical"] = _map_fsc_metrics(
            reconstructed_maps["gpu_f64_canonical_repeat0"],
            reconstructed_maps["relion_f64_canonical_repeat0"],
        )

    repo_root = Path(__file__).resolve().parents[1]
    report = {
        "schema": "recovar-bpref-accumulator-replay-report-v1",
        "status": "COMPLETE",
        "bundle": summarize_bpref_contribution_bundle(bundle),
        "replays": replay_inventory,
        "comparisons": comparisons,
        "map_fsc_comparisons": map_comparisons,
        "automatic_classification": _classify_replay_difference(
            comparisons, map_comparisons
        ),
        "artifacts": {
            "arrays_npz": str(arrays_path.resolve()),
            "arrays_npz_sha256": sha256_file(arrays_path),
        },
        "provenance": {
            "repo_root": str(repo_root),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip(),
            "git_status_porcelain": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True
            ).splitlines(),
            "devices": [],
        },
        "precision_provenance": (
            "Captured active_summed/active_ctf_probs retain their native complex128/float64 "
            "upstream values; float32 modes cast them at the scatter boundary. Captured "
            "rotation matrices are float32, so double geometry promotes those matrices and "
            "cannot recover rotation precision lost upstream."
        ),
        "topology_limitation": (
            "Schema v3 captures active rows and execution-shard boundaries but not the exact "
            "production packed-zero row layout. GPU execution replay preserves shard partition "
            "counts and active-row order; it is not claimed as bitwise production launch replay."
        ),
        "quality_gate": "exact/array accumulator metrics only; map comparisons require FSC/FSC-AUC",
        "map_reconstruction_control": (
            "Shared unregularized no-mask reconstruction with current-size support, tau=None, "
            "and grid correction off; intended only to measure whether scatter-boundary "
            "differences survive into maps."
        ),
    }
    if not args.skip_gpu:
        import jax

        report["provenance"]["devices"] = [str(device) for device in jax.devices()]
    report_path = args.output_dir / "bpref_accumulator_replay_report_v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
