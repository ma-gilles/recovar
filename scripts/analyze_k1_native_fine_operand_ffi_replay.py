#!/usr/bin/env python3
"""Replay RELION fine-operand captures through RECOVAR's exact CUDA reducer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from scripts.validate_relion_fine_operand_capture import (
    _cuda_fine_production_lanes,
    _reduce_lanes,
    load_fine_operand_capture,
)


SCHEMA = "recovar.em.k1_native_fine_operand_ffi_replay.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ulp_distance(left: np.float32, right: np.float32) -> int:
    def ordered(value: np.float32) -> int:
        bits = int(np.asarray(value, dtype=np.float32).view(np.uint32))
        return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000

    return abs(ordered(left) - ordered(right))


def _capture_row(path: Path) -> dict[str, object]:
    capture = load_fine_operand_capture(path)
    if capture.candidates.size != 1:
        raise ValueError("focused FFI replay requires exactly one captured candidate")
    candidate = capture.candidates[0]
    pixels = capture.pixels.reshape(1, capture.image_size)[0]
    reference = np.asarray(
        pixels["reference_real"] + np.complex64(1j) * pixels["reference_imag"],
        dtype=np.complex64,
    )
    shifted = np.asarray(
        pixels["shifted_real"] + np.complex64(1j) * pixels["shifted_imag"],
        dtype=np.complex64,
    )
    correction = np.asarray(pixels["corr"], dtype=np.float32)

    reduced = cuda_backproject.relion_fine_diff2_pairs_f32(
        jnp.asarray(reference[None, None, :]),
        jnp.asarray(shifted[None, None, :]),
        jnp.asarray(correction[None, :]),
        jnp.arange(capture.image_size, dtype=jnp.int32),
    )
    reduced = np.asarray(jax.block_until_ready(reduced), dtype=np.float32).reshape(-1)
    if reduced.shape != (1,):
        raise ValueError(f"focused FFI replay returned shape {reduced.shape}")

    sum_init = np.float32(candidate["sum_init"])
    ffi_raw = np.add(reduced[0], sum_init, dtype=np.float32)
    diff_real = np.subtract(reference.real, shifted.real, dtype=np.float32)
    diff_imag = np.subtract(reference.imag, shifted.imag, dtype=np.float32)
    host_raw = np.add(
        _reduce_lanes(
            _cuda_fine_production_lanes(diff_real, diff_imag, correction)
        ),
        sum_init,
        dtype=np.float32,
    )
    production = np.float32(candidate["production_raw_diff2"])
    isolated = np.float32(candidate["replay_raw_diff2"])

    return {
        "capture": str(path.resolve()),
        "capture_sha256": _sha256(path),
        "stack_index_one_based": int(capture.stack_index),
        "rotation_local": int(candidate["rotation_local"]),
        "translation_id": int(candidate["translation_id"]),
        "image_size": int(capture.image_size),
        "sum_init": float(sum_init),
        "production_raw_diff2": float(production),
        "capture_isolated_replay_raw_diff2": float(isolated),
        "recovar_custom_cuda_raw_diff2": float(ffi_raw),
        "host_sass_replay_raw_diff2": float(host_raw),
        "capture_isolated_ulp_from_production": _ulp_distance(isolated, production),
        "recovar_custom_cuda_ulp_from_production": _ulp_distance(ffi_raw, production),
        "host_sass_ulp_from_production": _ulp_distance(host_raw, production),
        "capture_isolated_exact_production": bool(isolated == production),
        "recovar_custom_cuda_exact_production": bool(ffi_raw == production),
        "host_sass_exact_production": bool(host_raw == production),
    }


def analyze(captures: list[Path]) -> dict[str, object]:
    if jax.default_backend() != "gpu":
        raise RuntimeError("native fine-operand FFI replay requires a GPU")
    rows = [_capture_row(path) for path in captures]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "exact float32 words and ULP distance; no correlation",
        "device": str(jax.devices()[0]),
        "cuda_library": os.environ.get("RECOVAR_CUDA_LIB"),
        "all_custom_cuda_exact_production": all(
            bool(row["recovar_custom_cuda_exact_production"]) for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.capture)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
