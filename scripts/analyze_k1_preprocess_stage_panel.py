#!/usr/bin/env python3
"""Locate the first native-RELION/RECOVAR preprocessing stage mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.cuda_backproject import relion_preprocess_real_f32
from recovar.data_io.image_backends import _centered_rfft2_jax_per_image

if __package__:
    from scripts.analyze_k1_native_wavg_pixels import (
        _comparison,
        _complex_comparison,
        _native_standard_half_indices,
    )
    from scripts.validate_relion_preprocess_capture import load_artifact
else:
    from analyze_k1_native_wavg_pixels import (  # type: ignore[no-redef]
        _comparison,
        _complex_comparison,
        _native_standard_half_indices,
    )
    from validate_relion_preprocess_capture import load_artifact  # type: ignore[no-redef]


def _real_comparison(native: np.ndarray, recovar: np.ndarray) -> dict[str, object]:
    native_f32 = np.asarray(native, dtype=np.float32).reshape(-1)
    recovar_f32 = np.asarray(recovar, dtype=np.float32).reshape(-1)
    difference = recovar_f32.astype(np.float64) - native_f32.astype(np.float64)
    native_norm = np.linalg.norm(native_f32.astype(np.float64))
    mismatch = native_f32.view(np.uint32) != recovar_f32.view(np.uint32)
    return {
        "pixel_count": int(native_f32.size),
        "bit_exact_count": int(np.count_nonzero(~mismatch)),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "relative_l2": float(np.linalg.norm(difference) / native_norm) if native_norm else 0.0,
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-preprocess-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-lane-reduction", action="store_true")
    parser.add_argument("--native-atomic-reduction", action="store_true")
    args = parser.parse_args()

    artifacts = [
        load_artifact(path)
        for path in sorted(args.native_preprocess_dir.glob("*.preprocess-v1.bin"))
    ]
    if len(artifacts) != 17:
        raise ValueError(f"expected 17 native preprocess artifacts, found {len(artifacts)}")
    raw = np.concatenate([artifact.raw_input_real for artifact in artifacts], axis=0)
    factors = np.asarray([artifact.norm_correction for artifact in artifacts], dtype=np.float32)
    shifts = np.stack([artifact.old_offset[:2] for artifact in artifacts]).astype(np.int32)
    radius = artifacts[0].mask_parameters["radius"]
    cosine_width = artifacts[0].mask_parameters["cosine_width"]
    if any(
        artifact.mask_parameters["radius"] != radius
        or artifact.mask_parameters["cosine_width"] != cosine_width
        for artifact in artifacts
    ):
        raise ValueError("native preprocess panel does not share one mask geometry")

    normalized, masked = relion_preprocess_real_f32(
        jnp.asarray(raw, dtype=jnp.float32),
        jnp.asarray(factors, dtype=jnp.float32),
        jnp.asarray(shifts, dtype=jnp.int32),
        radius,
        cosine_width,
        True,
        native_lane_reduction=args.native_lane_reduction,
        native_atomic_reduction=args.native_atomic_reduction,
    )
    transformed = _centered_rfft2_jax_per_image(masked)
    normalized_np, masked_np, transformed_np = (
        np.asarray(value)
        for value in jax.block_until_ready((normalized, masked, transformed))
    )
    current_size = int(artifacts[0].header[17])
    image_size = int(artifacts[0].header[12])
    current_half_width = current_size // 2 + 1
    logical_rows = np.where(
        np.arange(current_size) <= current_size // 2,
        np.arange(current_size),
        np.arange(current_size) - current_size,
    )
    centered_indices = (
        (logical_rows + image_size // 2)[:, None] * (image_size // 2 + 1)
        + np.arange(current_half_width)[None, :]
    ).reshape(-1)
    valid = np.ones(centered_indices.size, dtype=bool)
    fourier_scale = np.float32(1.0 / (image_size * image_size))

    records = []
    for row, artifact in enumerate(artifacts):
        recovar_fourier = np.asarray(
            transformed_np[row].reshape(-1)[centered_indices] * fourier_scale,
            dtype=np.complex64,
        )
        native_pre = artifact.masked_fourier_pre_optics.reshape(-1)
        native_post = artifact.masked_fourier_post_optics.reshape(-1)
        records.append(
            {
                "part_id": artifact.part_id,
                "source_index": artifact.stack_index - 1,
                "normalized_shifted_real": _real_comparison(
                    artifact.normalized_shifted_real,
                    normalized_np[row],
                ),
                "masked_real": _real_comparison(artifact.masked_real, masked_np[row]),
                "mask_background": _comparison(
                    np.asarray([artifact.mask_parameters["background"]], dtype=np.float32),
                    np.asarray([masked_np[row, 0, 0]], dtype=np.float32),
                    np.asarray([True]),
                ),
                "native_mask_background": artifact.mask_parameters["background"],
                "recovar_mask_background": float(masked_np[row, 0, 0]),
                "masked_fourier_pre_optics": _complex_comparison(
                    native_pre,
                    recovar_fourier,
                    valid,
                ),
                "masked_fourier_post_optics": _complex_comparison(
                    native_post,
                    recovar_fourier,
                    valid,
                ),
                "capture": str(artifact.path.resolve()),
            }
        )

    stage_names = (
        "normalized_shifted_real",
        "masked_real",
        "masked_fourier_pre_optics",
        "masked_fourier_post_optics",
    )
    summaries = {}
    summaries["mask_background"] = {
        "particle_count": len(records),
        "bit_exact_values": sum(
            int(record["mask_background"]["bit_exact_count"])
            for record in records
        ),
        "value_count": len(records),
        "max_ulp": max(int(record["mask_background"]["max_ulp"]) for record in records),
        "max_abs": max(float(record["mask_background"]["max_abs"]) for record in records),
    }
    for stage in stage_names:
        complex_stage = "fourier" in stage
        summaries[stage] = {
            "particle_count": len(records),
            "bit_exact_values": sum(
                int(
                    record[stage][
                        "complex_bit_exact_count" if complex_stage else "bit_exact_count"
                    ]
                )
                for record in records
            ),
            "value_count": sum(
                int(
                    record[stage]["real"]["valid_pixel_count"]
                    if complex_stage
                    else record[stage]["pixel_count"]
                )
                for record in records
            ),
            "max_relative_l2": max(float(record[stage]["relative_l2"]) for record in records),
        }
    report = {
        "schema": "recovar.em.k1_preprocess_stage_panel.v1",
        "jax_devices": [str(device) for device in jax.devices()],
        "native_lane_reduction": args.native_lane_reduction,
        "native_atomic_reduction": args.native_atomic_reduction,
        "fourier_operand_scale": float(fourier_scale),
        "native_standard_index_count": int(
            _native_standard_half_indices(current_size, image_size).size
        ),
        "summaries": summaries,
        "particles": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
