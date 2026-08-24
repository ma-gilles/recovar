#!/usr/bin/env python3
"""Recover one native RELION float32 image-normalization factor.

The input is RELION's passively captured ``normalized_shifted_real`` image.
The diagnostic applies the known zero-padded integer shift to the immutable
source image and searches adjacent positive float32 factors.  Acceptance is
exact bytes; relative L2 is retained only as a localization metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np


SCHEMA = "recovar.em.k1_native_normalization_factor.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_flat_real(path: Path) -> np.ndarray:
    with Path(path).open("rb") as stream:
        count_raw = stream.read(4)
        if len(count_raw) != 4:
            raise ValueError(f"truncated flat-real header: {path}")
        count = int(np.frombuffer(count_raw, dtype="<i4", count=1)[0])
        payload = stream.read()
    if count < 0 or len(payload) != count * 8:
        raise ValueError(f"invalid flat-real payload: {path}")
    return np.frombuffer(payload, dtype="<f8", count=count).astype(np.float32)


def zero_padded_integer_shift(
    image: np.ndarray,
    *,
    shift_x: int,
    shift_y: int,
) -> np.ndarray:
    """Match RELION's real-space integer translation without wraparound."""

    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"source image must be two-dimensional, got {source.shape}")
    height, width = source.shape
    x0 = max(0, -int(shift_x))
    x1 = min(width, width - int(shift_x))
    y0 = max(0, -int(shift_y))
    y1 = min(height, height - int(shift_y))
    shifted = np.zeros_like(source)
    if x1 > x0 and y1 > y0:
        shifted[
            y0 + int(shift_y) : y1 + int(shift_y),
            x0 + int(shift_x) : x1 + int(shift_x),
        ] = source[y0:y1, x0:x1]
    return shifted


def recover_normalization_factor(
    shifted_source: np.ndarray,
    native_normalized: np.ndarray,
    *,
    search_ulp: int = 64,
) -> dict[str, Any]:
    """Find the adjacent float32 factor that best reproduces native bytes."""

    source = np.asarray(shifted_source, dtype=np.float32)
    native = np.asarray(native_normalized, dtype=np.float32)
    if source.shape != native.shape or source.size == 0:
        raise ValueError("shifted source and native image must be nonempty and aligned")
    if search_ulp < 0:
        raise ValueError("search_ulp must be nonnegative")
    denominator = float(np.vdot(source.astype(np.float64), source.astype(np.float64)).real)
    if denominator <= 0.0:
        raise ValueError("shifted source image has zero norm")
    least_squares = float(
        np.vdot(source.astype(np.float64), native.astype(np.float64)).real
        / denominator
    )
    center = np.float32(least_squares)
    if not np.isfinite(center) or center <= 0.0:
        raise ValueError("recovered factor is not finite and positive")
    center_bits = int(center.view(np.uint32))
    native64 = native.astype(np.float64)
    native_norm = float(np.linalg.norm(native64.reshape(-1)))
    candidates = []
    for offset in range(-search_ulp, search_ulp + 1):
        bits = center_bits + offset
        if bits <= 0 or bits >= 0x7F800000:
            continue
        factor = np.asarray(bits, dtype=np.uint32).view(np.float32)
        predicted = np.multiply(source, factor, dtype=np.float32)
        delta = predicted.astype(np.float64) - native64
        candidates.append(
            (
                int(np.count_nonzero(predicted != native)),
                float(np.linalg.norm(delta.reshape(-1))),
                abs(offset),
                bits,
                factor,
                predicted,
            )
        )
    mismatch_count, delta_norm, _, bits, factor, predicted = min(candidates)
    max_abs = float(
        np.max(np.abs(predicted.astype(np.float64) - native64), initial=0.0)
    )
    return {
        "least_squares_float64": least_squares,
        "factor_float32": float(factor),
        "factor_float32_bits": f"0x{bits:08x}",
        "pixel_count": int(native.size),
        "bit_exact_count": int(native.size - mismatch_count),
        "mismatch_count": mismatch_count,
        "relative_l2": delta_norm / native_norm if native_norm else delta_norm,
        "max_abs": max_abs,
    }


def analyze(
    *,
    particles_mrcs: Path,
    source_index: int,
    native_normalized_shifted: Path,
    shift_x: int,
    shift_y: int,
    serialized_factor: float,
    search_ulp: int = 64,
) -> dict[str, Any]:
    particles_mrcs = Path(particles_mrcs).resolve()
    native_normalized_shifted = Path(native_normalized_shifted).resolve()
    if source_index < 0:
        raise ValueError("source_index must be nonnegative")
    with mrcfile.mmap(particles_mrcs, permissive=False, mode="r") as stack:
        if source_index >= stack.data.shape[0]:
            raise ValueError("source_index is outside the particle stack")
        source = np.asarray(stack.data[source_index], dtype=np.float32).copy()
    native_flat = _load_flat_real(native_normalized_shifted)
    if source.size != native_flat.size:
        raise ValueError("source and native captured image sizes differ")
    native = native_flat.reshape(source.shape)
    shifted = zero_padded_integer_shift(
        source,
        shift_x=shift_x,
        shift_y=shift_y,
    )
    recovered = recover_normalization_factor(
        shifted,
        native,
        search_ulp=search_ulp,
    )
    serialized = np.float32(serialized_factor)
    if not np.isfinite(serialized) or serialized <= 0.0:
        raise ValueError("serialized_factor must be finite and positive")
    recovered_bits = int(recovered["factor_float32_bits"], 16)
    serialized_bits = int(serialized.view(np.uint32))
    return {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "exact float32 bytes and relative L2; no correlation",
        "source_index_zero_based": source_index,
        "stack_index_one_based": source_index + 1,
        "integer_shift_xy": [int(shift_x), int(shift_y)],
        "recovered": recovered,
        "serialized": {
            "factor_float32": float(serialized),
            "factor_float32_bits": f"0x{serialized_bits:08x}",
            "serialized_minus_recovered_ulp": serialized_bits - recovered_bits,
            "relative_delta": float(
                (np.float64(serialized) - np.float64(recovered["factor_float32"]))
                / np.float64(recovered["factor_float32"])
            ),
        },
        "inputs": {
            "particles_mrcs": str(particles_mrcs),
            "particles_mrcs_sha256": _sha256(particles_mrcs),
            "native_normalized_shifted": str(native_normalized_shifted),
            "native_normalized_shifted_sha256": _sha256(native_normalized_shifted),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles-mrcs", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--native-normalized-shifted", type=Path, required=True)
    parser.add_argument("--shift-x", type=int, required=True)
    parser.add_argument("--shift-y", type=int, required=True)
    parser.add_argument("--serialized-factor", type=float, required=True)
    parser.add_argument("--search-ulp", type=int, default=64)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        particles_mrcs=args.particles_mrcs,
        source_index=args.source_index,
        native_normalized_shifted=args.native_normalized_shifted,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        serialized_factor=args.serialized_factor,
        search_ulp=args.search_ulp,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
