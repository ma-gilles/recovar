#!/usr/bin/env python3
"""Rebuild a sealed RELION projector from its captured float64 Iref operand."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.relion_projector_capture import (
    build_relion_projector_replay_state,
    load_relion_projector_iref_state,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_bind_path(bind_path: Path, requested_bind_dir: str) -> Path:
    bind_path = Path(bind_path).resolve()
    if requested_bind_dir:
        requested_bind_path = Path(requested_bind_dir).expanduser().resolve()
        if not bind_path.is_relative_to(requested_bind_path):
            raise RuntimeError(
                "loaded RELION binding is outside RECOVAR_RELION_BIND_BUILD_DIR: "
                f"module={bind_path}, requested={requested_bind_path}"
            )
    return bind_path


def _ordered_float32_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return np.where(
        (bits & np.uint32(0x80000000)) != 0,
        ~bits,
        bits ^ np.uint32(0x80000000),
    ).astype(np.uint64)


def _array_metrics(rebuilt64: np.ndarray, captured32: np.ndarray) -> dict[str, object]:
    rebuilt64 = np.asarray(rebuilt64, dtype=np.complex128)
    captured32 = np.asarray(captured32, dtype=np.complex64)
    rebuilt32 = rebuilt64.astype(np.complex64)
    if rebuilt64.shape != captured32.shape:
        return {
            "rebuilt_shape": list(rebuilt64.shape),
            "captured_shape": list(captured32.shape),
            "shape_equal": False,
            "exact_after_complex64": False,
            "finite": False,
        }
    delta32 = np.abs(rebuilt32 - captured32)
    delta64 = np.abs(rebuilt64 - captured32.astype(np.complex128))
    rebuilt_real_bits = _ordered_float32_bits(rebuilt32.real)
    captured_real_bits = _ordered_float32_bits(captured32.real)
    rebuilt_imag_bits = _ordered_float32_bits(rebuilt32.imag)
    captured_imag_bits = _ordered_float32_bits(captured32.imag)
    real_ulp = np.maximum(rebuilt_real_bits, captured_real_bits) - np.minimum(
        rebuilt_real_bits, captured_real_bits
    )
    imag_ulp = np.maximum(rebuilt_imag_bits, captured_imag_bits) - np.minimum(
        rebuilt_imag_bits, captured_imag_bits
    )
    return {
        "rebuilt_shape": list(rebuilt64.shape),
        "captured_shape": list(captured32.shape),
        "shape_equal": True,
        "finite": bool(np.isfinite(rebuilt64).all() and np.isfinite(captured32).all()),
        "exact_after_complex64": bool(np.array_equal(rebuilt32, captured32)),
        "n_unequal_after_complex64": int(np.count_nonzero(rebuilt32 != captured32)),
        "max_real_ulp_after_complex64": int(real_ulp.max(initial=0)),
        "max_imag_ulp_after_complex64": int(imag_ulp.max(initial=0)),
        "n_real_components_over_one_ulp": int(np.count_nonzero(real_ulp > 1)),
        "n_imag_components_over_one_ulp": int(np.count_nonzero(imag_ulp > 1)),
        "within_one_ulp_after_complex64": bool(np.all(real_ulp <= 1) and np.all(imag_ulp <= 1)),
        "max_abs_after_complex64": float(delta32.max(initial=0.0)),
        "mean_abs_after_complex64": float(delta32.mean()),
        "max_abs_rebuilt64_vs_captured32": float(delta64.max(initial=0.0)),
        "mean_abs_rebuilt64_vs_captured32": float(delta64.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--current-size", type=int, required=True)
    parser.add_argument("--volume-size", type=int, required=True)
    parser.add_argument("--n-classes", type=int, default=1)
    parser.add_argument("--interpolator", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Return success even when the projector mismatch remains unresolved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    volume_shape = (int(args.volume_size),) * 3
    projector_state = build_relion_projector_replay_state(
        args.capture_dir,
        manifest_path=args.manifest,
        iteration=args.iteration,
        current_size=args.current_size,
        volume_shape=volume_shape,
        n_classes=args.n_classes,
    )
    iref_state = load_relion_projector_iref_state(
        args.capture_dir,
        manifest_path=args.manifest,
        iteration=args.iteration,
        volume_shape=volume_shape,
        n_classes=args.n_classes,
    )
    if iref_state["source_manifest_sha256"] != projector_state["source_manifest_sha256"]:
        raise RuntimeError("Iref and projector payloads were not loaded from the same manifest")

    from recovar.relion_bind import _relion_bind_core as bind

    requested_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR", "")
    bind_path = _validate_bind_path(Path(bind.__file__), requested_bind_dir)

    rows = []
    passed = True
    all_exact = True
    for half in (1, 2):
        for class_id in range(int(args.n_classes)):
            iref = iref_state["iref_by_half"][half - 1][class_id]
            result = bind.compute_fourier_transform_map(
                iref,
                int(args.volume_size),
                int(projector_state["padding_factor"]),
                int(args.interpolator),
                int(args.current_size),
                True,
                2,
            )
            rebuilt, _power, ori_size, padding_factor, r_max, _r_min_nn, interpolator = result
            captured = projector_state["projector_half_by_half"][half - 1][class_id]
            metadata_equal = bool(
                int(ori_size) == int(args.volume_size)
                and int(padding_factor) == int(projector_state["padding_factor"])
                and int(r_max) == int(projector_state["projector_r_max_by_half"][half - 1])
                and int(interpolator) == int(args.interpolator)
            )
            metrics = _array_metrics(rebuilt, captured)
            row_exact = bool(metrics.get("exact_after_complex64"))
            row_passed = bool(
                metadata_equal and metrics.get("finite") and metrics.get("within_one_ulp_after_complex64")
            )
            passed = passed and row_passed
            all_exact = all_exact and row_exact
            rows.append(
                {
                    "half": half,
                    "class_id": class_id,
                    "metadata_equal": metadata_equal,
                    "ori_size": int(ori_size),
                    "padding_factor": int(padding_factor),
                    "r_max": int(r_max),
                    "interpolator": int(interpolator),
                    "metrics": metrics,
                    "status": (
                        "exact" if row_exact else "numerical_one_ulp" if row_passed else "mismatch"
                    ),
                }
            )

    report = {
        "schema": "recovar-relion-projector-captured-iref-rebuild-v1",
        "status": "pass_exact" if passed and all_exact else "pass_numerical" if passed else "mismatch",
        "classification": (
            "exact"
            if passed and all_exact
            else "precision_or_reduction_order"
            if passed
            else "unresolved"
        ),
        "metric_policy": (
            "report exact complex64 arrays, float32 ULP distances, and complex128/float64 "
            "pre-cast deltas; classify non-exact results only when every component is within one ULP"
        ),
        "capture_dir": str(args.capture_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": projector_state["source_manifest_sha256"],
        "relion_bind_build_dir": requested_bind_dir,
        "relion_bind_module": str(bind_path),
        "relion_bind_module_sha256": _sha256_file(bind_path),
        "do_gridding": True,
        "data_dim": 2,
        "iteration": int(args.iteration),
        "current_size": int(args.current_size),
        "volume_shape": list(volume_shape),
        "n_classes": int(args.n_classes),
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(rendered)
        os.replace(temporary, args.output_json)
    print(rendered, end="")
    return 0 if passed or args.report_only else 3


if __name__ == "__main__":
    raise SystemExit(main())
