#!/usr/bin/env python3
"""Compare native iteration-1 PPref with fresh and serialized RECOVAR sources."""

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


def _native_ppref(native_directory: Path, prefix: str) -> tuple[np.ndarray, int, int]:
    base = native_directory / prefix
    dims = _flat(Path(f"{base}dims.bin"), np.dtype("<i4"))
    _require(dims.size == 7, "native Projector dimensions changed")
    xdim, ydim, zdim, _, _, _, r_max = (int(value) for value in dims)
    real = _flat(Path(f"{base}real.bin"), np.dtype("<f8"))
    imag = _flat(Path(f"{base}imag.bin"), np.dtype("<f8"))
    _require(real.size == imag.size == xdim * ydim * zdim, "native Projector payload changed")
    ppref = (real + 1j * imag).astype(np.complex64).reshape(zdim, ydim, xdim)
    padding_payload = Path(f"{base}padding_factor.bin").read_bytes()
    _require(len(padding_payload) == 8, "native padding factor payload changed")
    padding_factor = int(struct.unpack("<d", padding_payload)[0])
    return ppref, r_max, padding_factor


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.complex64)
    reference = np.asarray(reference, dtype=np.complex64)
    _require(candidate.shape == reference.shape, "PPref topology mismatch")
    residual = candidate.astype(np.complex128) - reference.astype(np.complex128)
    denominator = max(float(np.linalg.norm(reference.astype(np.complex128))), np.finfo(float).tiny)
    exact_components = candidate.view(np.float32).view(np.uint32) == reference.view(np.float32).view(np.uint32)
    return {
        "array_equal": bool(np.array_equal(candidate, reference)),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "max_abs": float(np.max(np.abs(residual))),
        "bit_exact_float32_components": int(np.count_nonzero(exact_components)),
        "float32_components": int(exact_components.size),
    }


def classify(*, fresh_relative_l2: float, serialized_relative_l2: float) -> str:
    if fresh_relative_l2 <= 1.0e-7 and serialized_relative_l2 > 100.0 * max(
        fresh_relative_l2, np.finfo(float).tiny
    ):
        return "serialized_it000_replay_is_the_ppref_source_mismatch"
    if fresh_relative_l2 > 1.0e-7:
        return "fresh_initial_reference_to_ppref_boundary_remains_open"
    return "fresh_and_serialized_sources_are_not_separated"


def analyze(
    *,
    native_directory: Path,
    native_prefix: str,
    fresh_reference_mrc: Path,
    serialized_it000_mrc: Path,
    pixel_size: float,
    ini_high: float,
    current_size: int,
) -> dict[str, object]:
    from recovar.em.initial_model.dense_adapter import reference_to_relion_projector_half_maps
    from recovar.utils import helpers
    from scripts.run_multi_iter_parity import filter_fresh_initial_reference

    native, native_r_max, padding_factor = _native_ppref(native_directory, native_prefix)
    _require(padding_factor == 2, "expected RELION projector padding factor 2")

    fresh_source = np.asarray(helpers.load_mrc(fresh_reference_mrc), dtype=np.float64)
    fresh_filtered = filter_fresh_initial_reference(
        fresh_source,
        pixel_size=pixel_size,
        ini_high_angstrom=ini_high,
    )
    fresh_ppref, fresh_r_max = reference_to_relion_projector_half_maps(
        fresh_filtered[None],
        current_size=current_size,
        padding_factor=padding_factor,
    )

    serialized_source = np.asarray(helpers.load_relion_volume(str(serialized_it000_mrc)), dtype=np.float64)
    serialized_ppref, serialized_r_max = reference_to_relion_projector_half_maps(
        serialized_source[None],
        current_size=current_size,
        padding_factor=padding_factor,
    )
    _require(
        native_r_max == fresh_r_max == serialized_r_max,
        "native/fresh/serialized r_max mismatch",
    )
    fresh_metrics = _metrics(fresh_ppref[0], native)
    serialized_metrics = _metrics(serialized_ppref[0], native)
    return {
        "schema": "recovar.em.k1_firstiter_ppref_sources.v1",
        "classification": classify(
            fresh_relative_l2=float(fresh_metrics["relative_l2"]),
            serialized_relative_l2=float(serialized_metrics["relative_l2"]),
        ),
        "identity": {
            "shape_zyx": list(native.shape),
            "r_max": native_r_max,
            "padding_factor": padding_factor,
            "current_size": int(current_size),
            "pixel_size": float(pixel_size),
            "ini_high_angstrom": float(ini_high),
        },
        "fresh_filtered_ppref_vs_native": fresh_metrics,
        "serialized_it000_ppref_vs_native": serialized_metrics,
        "artifacts": {
            "native_directory": str(native_directory.resolve()),
            "fresh_reference_mrc": str(fresh_reference_mrc.resolve()),
            "fresh_reference_sha256": _sha256(fresh_reference_mrc),
            "serialized_it000_mrc": str(serialized_it000_mrc.resolve()),
            "serialized_it000_sha256": _sha256(serialized_it000_mrc),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-prefix", default="pass1_class0_ppref_")
    parser.add_argument("--fresh-reference-mrc", type=Path, required=True)
    parser.add_argument("--serialized-it000-mrc", type=Path, required=True)
    parser.add_argument("--pixel-size", type=float, required=True)
    parser.add_argument("--ini-high", type=float, required=True)
    parser.add_argument("--current-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        native_directory=args.native_directory,
        native_prefix=args.native_prefix,
        fresh_reference_mrc=args.fresh_reference_mrc,
        serialized_it000_mrc=args.serialized_it000_mrc,
        pixel_size=args.pixel_size,
        ini_high=args.ini_high,
        current_size=args.current_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
