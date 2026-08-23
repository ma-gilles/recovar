#!/usr/bin/env python3
"""Reproduce RELION Wavg translation and rotation accumulation levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
    make_frequency_coords_half_np,
)
from scripts.analyze_k1_projected_power_boundary import (
    _native_probabilities,
    _raw_f32,
)
from scripts.analyze_k1_scale_aa_candidates import _metric, _real, _scalar, _sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _flat_int(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 4, f"truncated integer array: {path}")
    count = int(np.frombuffer(payload, dtype="<i4", count=1)[0])
    values = np.frombuffer(payload, dtype="<i4", offset=4).copy()
    _require(values.size == count, f"integer-array size mismatch: {path}")
    return values


def _translation_loop_mass(raw: np.ndarray, sum_weight: np.float32, threshold: np.float32) -> np.ndarray:
    result = np.zeros(raw.shape[0], dtype=np.float32)
    for translation in range(raw.shape[1]):
        weights = raw[:, translation]
        normalized = np.where(
            weights >= threshold,
            (weights / sum_weight).astype(np.float32),
            np.float32(0.0),
        )
        result = (result + normalized).astype(np.float32)
    return result


def _native_pixel_aa(path: Path, *, iteration: int, half: int, part_id: int) -> dict[int, float]:
    prefix = f"acc_scale_pixel\titer={iteration}\tpart_id={part_id}\thalfset={half}\t"
    rows: dict[int, float] = {}
    with path.open() as stream:
        for line in stream:
            if not line.startswith(prefix):
                continue
            fields = {
                key: value
                for key, value in (item.split("=", 1) for item in line.rstrip().split("\t")[1:])
            }
            pixel = int(fields["j"])
            _require(pixel not in rows, f"duplicate native pixel {pixel}")
            rows[pixel] = float(fields["aa"])
    _require(rows, "native target pixel selection is empty")
    return rows


def analyze(
    recovar_capture: Path,
    native_directory: Path,
    native_pixels: Path,
    *,
    native_prefix: str,
    recovar_term_divisor: float,
    image_size: int,
) -> dict[str, object]:
    with np.load(recovar_capture, allow_pickle=False) as payload:
        recovar_probabilities = np.asarray(payload["candidate_posterior_probs"], dtype=np.float32)
        recovar_aa = np.asarray(payload["scale_aa_per_pixel"], dtype=np.float64)
        recovar_mask = np.asarray(payload["scale_correction_pixel_mask"], dtype=bool)
        iteration = int(payload["iteration"])
        half = int(payload["half"])
        part_id = int(payload["group_id"])
        current_size = int(payload["current_size"])

    prefix = native_directory / native_prefix
    native_probabilities, _, native_rotation_mass_float64 = _native_probabilities(prefix)
    orientation_num, translation_num = native_probabilities.shape
    raw = _real(Path(f"{prefix}sorted_weights.bin")).astype(np.float32).reshape(
        orientation_num,
        translation_num,
    )
    sum_weight = np.float32(_scalar(Path(f"{prefix}sum_weight.bin")))
    threshold = np.float32(_scalar(Path(f"{prefix}significant_weight.bin")))
    native_rotation_mass_wavg = _translation_loop_mass(raw, sum_weight, threshold)

    panel_pixels = _flat_int(Path(f"{prefix}project_panel_pixels.bin")).astype(np.int64)
    panel_shells = _flat_int(Path(f"{prefix}project_panel_shells.bin")).astype(np.int32)
    panel_size = panel_pixels.size
    native_real = _raw_f32(Path(f"{prefix}project_panel_ref_real.f32")).reshape(
        orientation_num,
        panel_size,
    )
    native_imag = _raw_f32(Path(f"{prefix}project_panel_ref_imag.f32")).reshape(
        orientation_num,
        panel_size,
    )
    native_ctf_all = _real(Path(f"{prefix}ctfs.bin")).astype(np.float32)
    native_ctf = native_ctf_all[panel_pixels]
    scaled_real = (native_real * native_ctf[None, :]).astype(np.float32)
    scaled_imag = (native_imag * native_ctf[None, :]).astype(np.float32)
    projected_power = (scaled_real * scaled_real + scaled_imag * scaled_imag).astype(np.float32)
    contributions = (projected_power * native_rotation_mass_wavg[:, None]).astype(np.float32)

    outer_float64 = np.sum(contributions, axis=0, dtype=np.float64)
    outer_pairwise_float32 = np.sum(contributions, axis=0, dtype=np.float32).astype(np.float64)
    outer_forward_float32 = np.cumsum(contributions, axis=0, dtype=np.float32)[-1].astype(np.float64)
    outer_reverse_float32 = np.cumsum(contributions[::-1], axis=0, dtype=np.float32)[-1].astype(
        np.float64
    )

    native_rows = _native_pixel_aa(
        native_pixels,
        iteration=iteration,
        half=half,
        part_id=part_id,
    )
    _require(all(int(pixel) in native_rows for pixel in panel_pixels), "native AA misses panel pixels")
    native_aa = np.asarray([native_rows[int(pixel)] for pixel in panel_pixels], dtype=np.float64)
    recovar_active_rows = np.flatnonzero(recovar_mask)
    _require(recovar_active_rows.size == panel_size, "RECOVAR and native panel sizes differ")
    window_indices, _ = make_fourier_window_indices_np(
        (image_size, image_size),
        current_size,
        square=False,
        include_dc=True,
        exact_radius=True,
    )
    recovar_coordinates = np.rint(
        make_frequency_coords_half_np((image_size, image_size))[window_indices]
    ).astype(np.int32)
    recovar_by_coordinate = {
        tuple(recovar_coordinates[row]): recovar_aa[row] / recovar_term_divisor
        for row in recovar_active_rows.tolist()
    }
    native_xdim = current_size // 2 + 1
    native_ydim = int(native_ctf_all.size // native_xdim)
    panel_coordinates = []
    for pixel in panel_pixels.tolist():
        row = int(pixel) // native_xdim
        x = int(pixel) % native_xdim
        y = row if row <= native_ydim // 2 else row - native_ydim
        panel_coordinates.append((x, y))
    _require(
        all(coordinate in recovar_by_coordinate for coordinate in panel_coordinates),
        "RECOVAR AA misses a native panel coordinate",
    )
    recovar_aa_native_units = np.asarray(
        [recovar_by_coordinate[coordinate] for coordinate in panel_coordinates],
        dtype=np.float64,
    )

    naive_rotation_mass_float32 = np.sum(native_probabilities, axis=1, dtype=np.float32)
    exact_inner_delta = native_rotation_mass_wavg.astype(np.float64) - naive_rotation_mass_float32
    positive_mass = native_rotation_mass_wavg > 0.0
    return {
        "schema": "recovar.em.k1_wavg_accumulation_boundary.v1",
        "identity": {
            "iteration": iteration,
            "half": half,
            "part_id": part_id,
            "orientation_count": orientation_num,
            "translation_count": translation_num,
            "panel_pixel_count": panel_size,
            "shell_ids": np.unique(panel_shells).tolist(),
        },
        "translation_loop": {
            "wavg_sequential_vs_numpy_float32_rotation_mass": _metric(
                native_rotation_mass_wavg[positive_mass],
                naive_rotation_mass_float32[positive_mass],
            ),
            "wavg_sequential_vs_float64_rotation_mass": _metric(
                native_rotation_mass_wavg[positive_mass],
                native_rotation_mass_float64[positive_mass],
            ),
            "max_signed_delta_vs_numpy_float32": float(np.max(np.abs(exact_inner_delta))),
        },
        "rotation_accumulation_vs_native_pixel_aa": {
            "float64_sum": _metric(outer_float64, native_aa),
            "numpy_pairwise_float32_sum": _metric(outer_pairwise_float32, native_aa),
            "forward_orientation_float32_sum": _metric(outer_forward_float32, native_aa),
            "reverse_orientation_float32_sum": _metric(outer_reverse_float32, native_aa),
            "captured_recovar": _metric(recovar_aa_native_units, native_aa),
        },
        "artifacts": {
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
            "native_directory": str(native_directory.resolve()),
            "native_pixels": str(native_pixels.resolve()),
            "native_pixels_sha256": _sha256(native_pixels),
        },
        "classification": "the first material inequality is native Wavg rotation accumulation order",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-pixels", type=Path, required=True)
    parser.add_argument("--native-prefix", default="img0_part109_storeWavg_")
    parser.add_argument("--recovar-term-divisor", type=float, default=float(128**4))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.recovar_capture,
        args.native_directory,
        args.native_pixels,
        native_prefix=args.native_prefix,
        recovar_term_divisor=args.recovar_term_divisor,
        image_size=args.image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
