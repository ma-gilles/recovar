#!/usr/bin/env python3
"""Compare one native RELION and RECOVAR in-memory Projector::data boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat(path: Path, dtype: np.dtype) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 4, f"truncated flat array: {path}")
    count = struct.unpack_from("<i", payload)[0]
    values = np.frombuffer(payload, dtype=dtype, offset=4).copy()
    _require(values.size == count, f"flat-array size mismatch: {path}")
    return values


def _scalar(path: Path) -> float:
    payload = path.read_bytes()
    _require(len(payload) == 8, f"scalar size mismatch: {path}")
    return float(struct.unpack("<d", payload)[0])


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    _require(candidate.shape == reference.shape and candidate.size > 0, "metric topology mismatch")
    residual = candidate.astype(np.complex128) - reference.astype(np.complex128)
    denominator = max(float(np.linalg.norm(reference.astype(np.complex128))), np.finfo(float).tiny)
    return {
        "count": int(candidate.size),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "median_abs": float(np.median(np.abs(residual))),
        "p95_abs": float(np.percentile(np.abs(residual), 95)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def analyze(native_directory: Path, recovar_capture: Path, *, native_prefix: str) -> dict[str, object]:
    prefix = native_directory / native_prefix
    dims = _flat(Path(f"{prefix}dims.bin"), np.dtype("<i4"))
    _require(dims.size == 7, "native Projector dimensions changed")
    xdim, ydim, zdim, xinit, yinit, zinit, r_max = (int(value) for value in dims)
    native_real = _flat(Path(f"{prefix}real.bin"), np.dtype("<f8"))
    native_imag = _flat(Path(f"{prefix}imag.bin"), np.dtype("<f8"))
    _require(native_real.size == native_imag.size == xdim * ydim * zdim, "native Projector payload changed")
    native = (native_real + 1j * native_imag).astype(np.complex64).reshape(zdim, ydim, xdim)
    padding_factor = _scalar(Path(f"{prefix}padding_factor.bin"))

    with np.load(recovar_capture, allow_pickle=False) as payload:
        recovar = np.asarray(payload["projector_half"], dtype=np.complex64)
        if recovar.ndim == 4:
            _require(recovar.shape[0] == 1, "RECOVAR Projector capture is not K=1")
            recovar = recovar[0]
        recovar_r_max = int(payload["projector_r_max"])
        recovar_current_size = int(payload["current_size"])
        recovar_padding = int(payload["padding_factor"])
        volume_shape = np.asarray(payload["volume_shape"], dtype=np.int64).tolist()
        n_classes = int(payload["n_classes"])
    _require(recovar.shape == native.shape, "native and RECOVAR Projector shapes differ")
    _require(recovar_r_max == r_max, "native and RECOVAR r_max differ")
    _require(float(recovar_padding) == padding_factor, "native and RECOVAR padding factors differ")

    native_bits = native.view(np.float32).view(np.uint32)
    recovar_bits = recovar.view(np.float32).view(np.uint32)
    component_exact = native_bits == recovar_bits
    nonzero = (native != 0.0) | (recovar != 0.0)
    native_magnitude = np.abs(native[nonzero]).astype(np.float64)
    recovar_magnitude = np.abs(recovar[nonzero]).astype(np.float64)
    jointly_positive = (native_magnitude > 0.0) & (recovar_magnitude > 0.0)
    ratios = recovar_magnitude[jointly_positive] / native_magnitude[jointly_positive]
    _require(ratios.size > 0, "Projector capture has no jointly positive entries")

    return {
        "schema": "recovar.em.k1_projector_boundary.v1",
        "identity": {
            "shape_zyx": list(native.shape),
            "origin_xyz": [xinit, yinit, zinit],
            "r_max": r_max,
            "padding_factor": padding_factor,
            "current_size": recovar_current_size,
            "volume_shape": volume_shape,
            "n_classes": n_classes,
        },
        "complex_values": {
            **_metric(recovar, native),
            "bit_exact_float32_component_count": int(np.count_nonzero(component_exact)),
            "float32_component_count": int(component_exact.size),
            "nonzero_union_count": int(np.count_nonzero(nonzero)),
            "magnitude_ratio_median": float(np.median(ratios)),
            "magnitude_ratio_p05": float(np.percentile(ratios, 5)),
            "magnitude_ratio_p95": float(np.percentile(ratios, 95)),
            "native_norm": float(np.linalg.norm(native.astype(np.complex128))),
            "recovar_norm": float(np.linalg.norm(recovar.astype(np.complex128))),
            "recovar_over_native_norm": float(
                np.linalg.norm(recovar.astype(np.complex128))
                / np.linalg.norm(native.astype(np.complex128))
            ),
        },
        "artifacts": {
            "native_directory": str(native_directory.resolve()),
            "native_files_sha256": {
                suffix: _sha256(Path(f"{prefix}{suffix}.bin"))
                for suffix in ("dims", "padding_factor", "real", "imag")
            },
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
        },
        "classification": (
            "Projector::data differs before candidate projection"
            if _metric(recovar, native)["relative_l2"] > 1e-7
            else "Projector::data agrees; texture projection arithmetic is the next boundary"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument(
        "--native-prefix",
        default="img0_part109_storeWavg_wavg_ppref_",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.recovar_capture,
        native_prefix=args.native_prefix,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
