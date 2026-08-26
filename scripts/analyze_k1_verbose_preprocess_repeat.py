#!/usr/bin/env python3
"""Replay one verbose RELION preprocessing boundary, including atomic repeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np

from recovar.cuda_backproject import relion_preprocess_real_f32
from recovar.data_io.image_backends import _centered_rfft2_jax_per_image

try:
    from scripts.analyze_k1_native_fine_operand_boundary import (
        _complex_metric,
        _load_flat_complex,
        _load_flat_real,
        _load_scalar,
        _metric,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from analyze_k1_native_fine_operand_boundary import (  # type: ignore[no-redef]
        _complex_metric,
        _load_flat_complex,
        _load_flat_real,
        _load_scalar,
        _metric,
    )


def _summarize_repeats(
    records: list[dict[str, Any]], native_background: np.float32
) -> dict[str, Any]:
    backgrounds = np.asarray(
        [record["background"] for record in records], dtype=np.float32
    )
    bits = backgrounds.view(np.uint32)
    native_bits = np.asarray([native_background], dtype=np.float32).view(np.uint32)[0]
    stages = ("normalized_shifted_real", "masked_real", "masked_fourier_pre_optics")
    return {
        "repeat_count": len(records),
        "native_background": float(native_background),
        "native_background_bits": int(native_bits),
        "unique_background_bits": [int(value) for value in np.unique(bits)],
        "exact_native_background_hits": int(np.count_nonzero(bits == native_bits)),
        "stage_envelopes": {
            stage: {
                "min_relative_l2": min(
                    float(record[stage]["relative_l2_over_reference"])
                    for record in records
                ),
                "max_relative_l2": max(
                    float(record[stage]["relative_l2_over_reference"])
                    for record in records
                ),
                "min_max_abs": min(float(record[stage]["max_abs"]) for record in records),
                "max_max_abs": max(float(record[stage]["max_abs"]) for record in records),
                "min_value_mismatch_count": min(
                    int(record[stage]["value_mismatch_count"]) for record in records
                ),
                "max_value_mismatch_count": max(
                    int(record[stage]["value_mismatch_count"]) for record in records
                ),
            }
            for stage in stages
        },
    }


def _load_native_stages(directory: Path) -> dict[str, np.ndarray | np.float32]:
    return {
        "normalized_shifted_real": _load_flat_real(
            directory / "preprocess_img0_normalized_shifted_real.bin"
        ).astype(np.float32),
        "masked_real": _load_flat_real(
            directory / "preprocess_img0_masked_real.bin"
        ).astype(np.float32),
        "masked_fourier_pre_optics": _load_flat_complex(
            directory / "preprocess_img0_masked_fourier_pre_optics_real.bin",
            directory / "preprocess_img0_masked_fourier_pre_optics_imag.bin",
        ),
        "masked_fourier_post_optics": _load_flat_complex(
            directory / "preprocess_img0_masked_fourier_post_optics_real.bin",
            directory / "preprocess_img0_masked_fourier_post_optics_imag.bin",
        ),
        "background": np.float32(
            _load_scalar(directory / "preprocess_img0_softmask_background.bin")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-verbose-dir", type=Path, required=True)
    parser.add_argument("--recovar-pass2", type=Path, required=True)
    parser.add_argument("--particles-mrcs", type=Path, required=True)
    parser.add_argument("--stack-index-one-based", type=int, required=True)
    parser.add_argument("--radius-pixels", type=float, required=True)
    parser.add_argument("--mask-edge-pixels", type=float, default=5.0)
    parser.add_argument("--atomic-repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stack_index_one_based <= 0:
        raise ValueError("--stack-index-one-based must be positive")
    if args.atomic_repeats <= 0:
        raise ValueError("--atomic-repeats must be positive")
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {devices}")

    native = _load_native_stages(args.native_verbose_dir)
    with np.load(args.recovar_pass2, allow_pickle=False) as capture:
        normalization = np.asarray(
            [capture["relion_preprocess_normalization_factor"]], dtype=np.float32
        )
        integer_shift = np.asarray(
            capture["relion_integer_pre_shift"], dtype=np.int32
        ).reshape(1, 2)
    with mrcfile.mmap(args.particles_mrcs, permissive=False, mode="r") as stack:
        raw = np.asarray(
            stack.data[args.stack_index_one_based - 1], dtype=np.float32
        ).copy()[None]

    image_size = int(raw.shape[-1])
    if raw.shape != (1, image_size, image_size):
        raise ValueError(f"particle image is not square: {raw.shape}")
    native_normalized = np.asarray(native["normalized_shifted_real"]).reshape(raw.shape)
    native_masked = np.asarray(native["masked_real"]).reshape(raw.shape)
    native_fourier = np.asarray(native["masked_fourier_pre_optics"])
    current_size = next(
        (
            size
            for size in range(2, image_size + 1, 2)
            if size * (size // 2 + 1) == native_fourier.size
        ),
        0,
    )
    current_half_width = current_size // 2 + 1
    if current_size == 0:
        raise ValueError(f"invalid native current-size Fourier shape: {native_fourier.size}")
    logical_rows = np.where(
        np.arange(current_size) <= current_size // 2,
        np.arange(current_size),
        np.arange(current_size) - current_size,
    )
    centered_indices = (
        (logical_rows + image_size // 2)[:, None] * (image_size // 2 + 1)
        + np.arange(current_half_width)[None, :]
    ).reshape(-1)
    fourier_scale = np.float32(1.0 / (image_size * image_size))

    raw_device = jnp.asarray(raw, dtype=jnp.float32)
    normalization_device = jnp.asarray(normalization, dtype=jnp.float32)
    shift_device = jnp.asarray(integer_shift, dtype=jnp.int32)

    def replay(*, native_lane: bool = False, native_atomic: bool = False) -> dict[str, Any]:
        normalized, masked = relion_preprocess_real_f32(
            raw_device,
            normalization_device,
            shift_device,
            args.radius_pixels,
            args.mask_edge_pixels,
            True,
            native_lane_reduction=native_lane,
            native_atomic_reduction=native_atomic,
        )
        transformed = _centered_rfft2_jax_per_image(masked)
        normalized_np, masked_np, transformed_np = (
            np.asarray(value)
            for value in jax.block_until_ready((normalized, masked, transformed))
        )
        fourier_np = np.asarray(
            transformed_np.reshape(-1)[centered_indices] * fourier_scale,
            dtype=np.complex64,
        )
        return {
            "background": float(np.float32(masked_np[0, 0, 0])),
            "normalized_shifted_real": _metric(native_normalized, normalized_np),
            "masked_real": _metric(native_masked, masked_np),
            "masked_fourier_pre_optics": _complex_metric(native_fourier, fourier_np),
        }

    deterministic = replay()
    native_lane = replay(native_lane=True)
    atomic_records = [replay(native_atomic=True) for _ in range(args.atomic_repeats)]
    native_background = np.float32(native["background"])
    report = {
        "schema": "recovar.em.k1_verbose_preprocess_repeat.v1",
        "jax_devices": [str(device) for device in devices],
        "inputs": {
            "native_verbose_dir": str(args.native_verbose_dir.resolve()),
            "recovar_pass2": str(args.recovar_pass2.resolve()),
            "particles_mrcs": str(args.particles_mrcs.resolve()),
            "stack_index_one_based": args.stack_index_one_based,
            "normalization_factor": float(normalization[0]),
            "integer_shift": integer_shift[0].tolist(),
            "radius_pixels": args.radius_pixels,
            "mask_edge_pixels": args.mask_edge_pixels,
            "image_size": image_size,
            "current_size": current_size,
            "fourier_scale": float(fourier_scale),
        },
        "native": {
            "background": float(native_background),
            "pre_vs_post_optics": _complex_metric(
                np.asarray(native["masked_fourier_pre_optics"]),
                np.asarray(native["masked_fourier_post_optics"]),
            ),
        },
        "deterministic": deterministic,
        "native_lane": native_lane,
        "native_atomic_repeats": _summarize_repeats(
            atomic_records, native_background
        ),
        "native_atomic_records": atomic_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["native_atomic_repeats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
