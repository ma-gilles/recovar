#!/usr/bin/env python3
"""Replay dumped native RELION fine operands through RECOVAR's CUDA reducer."""

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
from scripts.analyze_k1_native_fine_operand_boundary import (
    _center,
    _load_flat_real,
    _metric,
    _native_to_recovar_compact,
)

SCHEMA = "recovar.em.k1_native_fine_operand_gpu_replay.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(
    native_dir: Path,
    *,
    recovar_capture: Path | None = None,
    physical_image_size: int | None = None,
) -> dict[str, object]:
    corr = _load_flat_real(native_dir / "pass1_img0_corr_img.bin").astype(np.float32)
    native_raw = _load_flat_real(
        native_dir / "pass1_exp_Mweight_raw_preprior.bin"
    ).astype(np.float32)
    candidate_count = int(native_raw.size)
    pixel_count = int(corr.size)

    def complex_operand(stem: str) -> np.ndarray:
        real = _load_flat_real(native_dir / f"pass1_class0_{stem}_real.bin")
        imag = _load_flat_real(native_dir / f"pass1_class0_{stem}_imag.bin")
        if real.size != candidate_count * pixel_count or imag.size != real.size:
            raise ValueError(f"invalid {stem} operand shape")
        return (real + 1j * imag).astype(np.complex64).reshape(candidate_count, pixel_count)

    reference = complex_operand("fine_ref")
    shifted = complex_operand("fine_shifted")
    full_to_compact = np.arange(pixel_count, dtype=np.int32)

    def device_reduce(pixel_weight: np.ndarray) -> np.ndarray:
        values = cuda_backproject.relion_fine_diff2_pairs_f32(
            jnp.asarray(reference[None, :, :]),
            jnp.asarray(shifted[None, :, :]),
            jnp.asarray(pixel_weight[None, :]),
            jnp.asarray(full_to_compact),
        )
        return np.asarray(jax.block_until_ready(values), dtype=np.float32).reshape(-1)

    device_raw_without_highres = device_reduce(corr)
    inferred_highres = np.subtract(
        native_raw, device_raw_without_highres, dtype=np.float32
    )
    inferred_median = np.float32(np.median(inferred_highres))
    replayed_raw = np.add(
        device_raw_without_highres, inferred_median, dtype=np.float32
    )
    files = sorted(native_dir.glob("pass1_*.bin"))
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "exact bytes and relative L2; no correlation",
        "candidate_count": candidate_count,
        "pixel_count": pixel_count,
        "devices": [str(device) for device in jax.devices()],
        "cuda_library": os.environ.get("RECOVAR_CUDA_LIB"),
        "inferred_highres_xi2_half_median": float(inferred_median),
        "inferred_highres_xi2_half_range": [
            float(np.min(inferred_highres)),
            float(np.max(inferred_highres)),
        ],
        "raw_score": _metric(native_raw, replayed_raw),
        "centered_raw_score": _metric(_center(native_raw), _center(replayed_raw)),
        "artifacts": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in files
        ],
    }
    if recovar_capture is not None:
        if physical_image_size is None:
            raise ValueError("physical_image_size is required with recovar_capture")
        with np.load(recovar_capture, allow_pickle=False) as archive:
            recovar_corr_compact = np.asarray(
                archive["raw_operand_corr_img_score"], dtype=np.float32
            )
            recovar_full_to_compact = np.asarray(
                archive["raw_operand_relion_full_to_compact"], dtype=np.int64
            )
        native_to_compact = _native_to_recovar_compact(
            native_image_size=pixel_count,
            recovar_full_to_compact=recovar_full_to_compact,
        )
        common = (native_to_compact >= 0) & (corr != 0.0)
        scale = np.float32(physical_image_size * physical_image_size)
        recovar_corr_native_units = np.zeros_like(corr)
        recovar_corr_native_units[common] = np.multiply(
            recovar_corr_compact[native_to_compact[common]],
            np.multiply(scale, scale, dtype=np.float32),
            dtype=np.float32,
        )
        recovar_corr_raw = device_reduce(recovar_corr_native_units)
        recovar_corr_replayed = np.add(
            recovar_corr_raw, inferred_median, dtype=np.float32
        )
        report["recovar_corr_counterfactual"] = {
            "common_contributing_pixel_count": int(np.count_nonzero(common)),
            "fourier_scale": float(scale),
            "corr_img_common_pixels": _metric(
                corr[common], recovar_corr_native_units[common]
            ),
            "raw_score": _metric(native_raw, recovar_corr_replayed),
            "centered_raw_score": _metric(
                _center(native_raw), _center(recovar_corr_replayed)
            ),
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--recovar-capture", type=Path)
    parser.add_argument("--physical-image-size", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = replay(
        args.native_dir,
        recovar_capture=args.recovar_capture,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
