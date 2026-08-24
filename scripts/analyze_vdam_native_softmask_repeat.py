#!/usr/bin/env python3
"""Audit RELION soft-mask reduction repeatability and RECOVAR replay modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar import cuda_backproject  # noqa: E402
from scripts.analyze_vdam_native_translation_boundary import _flat_real_dump  # noqa: E402


def _float32_bits(value: float | np.floating) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    residual = candidate.astype(np.float64) - reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference.astype(np.float64).reshape(-1)))
    return {
        "exact_count": int(np.count_nonzero(candidate == reference)),
        "value_count": int(reference.size),
        "relative_l2": float(np.linalg.norm(residual.reshape(-1)) / denominator),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _capture(capture_dir: Path, full_size: int) -> tuple[np.ndarray, np.ndarray, np.float32]:
    prefix = Path(capture_dir) / "preprocess_img0_"
    normalized = _flat_real_dump(Path(f"{prefix}normalized_shifted_real.bin"))
    masked = _flat_real_dump(Path(f"{prefix}masked_real.bin"))
    background = np.fromfile(Path(f"{prefix}softmask_background.bin"), dtype=np.float64)
    expected = full_size * full_size
    if normalized.size != expected or masked.size != expected or background.shape != (1,):
        raise ValueError(f"invalid preprocessing capture in {capture_dir}")
    return (
        normalized.astype(np.float32).reshape(full_size, full_size),
        masked.astype(np.float32).reshape(full_size, full_size),
        np.float32(background[0]),
    )


def _replay(
    normalized: np.ndarray,
    *,
    radius: float,
    cosine_width: float,
    native_lane: bool = False,
    native_atomic: bool = False,
) -> np.ndarray:
    _, masked = cuda_backproject.relion_preprocess_real_f32(
        jnp.asarray(normalized[None], dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.zeros((1, 2), dtype=jnp.int32),
        radius,
        cosine_width,
        apply_mask=True,
        native_lane_reduction=native_lane,
        native_atomic_reduction=native_atomic,
    )
    return np.asarray(jax.block_until_ready(masked), dtype=np.float32)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=8)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--mask-radius", type=float, required=True)
    parser.add_argument("--mask-cosine-width", type=float, default=5.0)
    parser.add_argument("--native-atomic-samples", type=int, default=32)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat_count < 2 or args.native_atomic_samples < 1:
        parser.error("repeat-count must be at least 2 and native-atomic-samples must be positive")

    captures = [
        _capture(args.panel_root / f"repeat-{index:02d}" / "capture", args.full_image_size)
        for index in range(1, args.repeat_count + 1)
    ]
    normalized_reference = captures[0][0]
    normalized_metrics = [_metric(normalized_reference, item[0]) for item in captures]
    if any(metric["exact_count"] != metric["value_count"] for metric in normalized_metrics):
        raise ValueError("repeat panel did not preserve an identical normalized input")

    native_backgrounds = np.asarray([item[2] for item in captures], dtype=np.float32)
    native_bits = native_backgrounds.view(np.uint32)
    unique_native_bits, native_counts = np.unique(native_bits, return_counts=True)

    replay_masks: dict[str, list[np.ndarray]] = {
        "block_first": [
            _replay(
                normalized_reference,
                radius=args.mask_radius,
                cosine_width=args.mask_cosine_width,
            )
        ],
        "native_lane": [
            _replay(
                normalized_reference,
                radius=args.mask_radius,
                cosine_width=args.mask_cosine_width,
                native_lane=True,
            )
        ],
        "native_atomic": [
            _replay(
                normalized_reference,
                radius=args.mask_radius,
                cosine_width=args.mask_cosine_width,
                native_atomic=True,
            )
            for _ in range(args.native_atomic_samples)
        ],
    }
    modes: dict[str, object] = {}
    for mode, masks in replay_masks.items():
        backgrounds = np.asarray([mask[0, 0] for mask in masks], dtype=np.float32)
        bits = backgrounds.view(np.uint32)
        unique_bits, counts = np.unique(bits, return_counts=True)
        modes[mode] = {
            "sample_count": int(bits.size),
            "background_bits": [int(value) for value in bits],
            "unique_background_bits": [int(value) for value in unique_bits],
            "unique_background_counts": [int(value) for value in counts],
            "inside_native_observed_range": [
                bool(int(native_bits.min()) <= int(value) <= int(native_bits.max()))
                for value in bits
            ],
            "minimum_ulp_distance_to_native": [
                int(np.min(np.abs(native_bits.astype(np.int64) - int(value)))) for value in bits
            ],
            "masked_vs_native_repeats": [
                _metric(native_masked, mask)
                for mask in masks
                for _, native_masked, _ in captures
            ],
        }

    payload = {
        "schema": "recovar.vdam_native_softmask_repeat.v1",
        "status": "complete",
        "panel_root": str(args.panel_root.resolve()),
        "repeat_count": args.repeat_count,
        "normalized_inputs_exact": True,
        "native": {
            "background_values": [float(value) for value in native_backgrounds],
            "background_bits": [int(value) for value in native_bits],
            "unique_background_bits": [int(value) for value in unique_native_bits],
            "unique_background_counts": [int(value) for value in native_counts],
            "ulp_span": int(native_bits.max() - native_bits.min()),
            "masked_repeat_metrics_vs_first": [
                _metric(captures[0][1], item[1]) for item in captures
            ],
        },
        "replay_modes": modes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
