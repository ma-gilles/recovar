#!/usr/bin/env python3
"""Join one RECOVAR scale-AA capture to native RELION pixels by Fourier coordinate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
    make_frequency_coords_half_np,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    _require(candidate.shape == reference.shape and candidate.size > 0, "metric topology mismatch")
    residual = candidate - reference
    denominator = max(float(np.linalg.norm(reference)), float(np.finfo(np.float64).tiny))
    return {
        "count": int(candidate.size),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "median_signed": float(np.median(residual)),
        "median_abs": float(np.median(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _native_pixels(
    path: Path,
    *,
    iteration: int,
    half: int,
    part_id: int,
) -> dict[tuple[int, int], tuple[int, float, float]]:
    rows: dict[tuple[int, int], tuple[int, float, float]] = {}
    prefix = f"acc_scale_pixel\titer={iteration}\tpart_id={part_id}\thalfset={half}\t"
    with path.open() as stream:
        for line in stream:
            if not line.startswith(prefix):
                continue
            fields = {
                key: value
                for key, value in (item.split("=", 1) for item in line.rstrip().split("\t")[1:])
            }
            coordinate = (int(fields["x"]), int(fields["y"]))
            _require(coordinate not in rows, f"duplicate native coordinate {coordinate}")
            rows[coordinate] = (int(fields["shell"]), float(fields["aa"]), float(fields["xa"]))
    _require(rows, "native target selection is empty")
    return rows


def _native_direct_residual_shells(
    path: Path,
    *,
    iteration: int,
    half: int,
    part_id: int,
) -> dict[int, float]:
    rows: dict[int, float] = {}
    prefix = f"acc_components\titer={iteration}\tpart_id={part_id}\thalfset={half}\t"
    with path.open() as stream:
        for line in stream:
            if not line.startswith(prefix):
                continue
            fields = {
                key: value
                for key, value in (item.split("=", 1) for item in line.rstrip().split("\t")[1:])
            }
            shell = int(fields["shell"])
            _require(shell not in rows, f"duplicate native direct-residual shell {shell}")
            rows[shell] = float(fields["direct_residual"])
    _require(rows, "native direct-residual target selection is empty")
    return rows


def analyze(
    recovar_capture: Path,
    native_pixels: Path,
    *,
    native_noise_components: Path | None = None,
    expected_iteration: int,
    expected_half: int,
    expected_part_id: int,
    expected_original_index: int,
    image_size: int,
    recovar_term_divisor: float,
    term_source: str = "scale",
) -> dict[str, object]:
    _require(image_size > 0 and image_size % 2 == 0, "image size must be positive and even")
    _require(np.isfinite(recovar_term_divisor) and recovar_term_divisor > 0.0, "invalid divisor")
    with np.load(recovar_capture, allow_pickle=False) as payload:
        schema = str(payload["schema"].item())
        _require(term_source in {"scale", "norm"}, f"unsupported term source {term_source}")
        supported_schemas = {
            "recovar-k1-scale-aa-chunked-v1",
            "recovar-k1-scale-xa-aa-chunked-v2",
            "recovar-k1-scale-xa-aa-chunked-v3",
            "recovar-k1-scale-xa-aa-chunked-v4",
        }
        _require(schema in supported_schemas, f"unsupported schema {schema}")
        iteration = int(payload["iteration"])
        half = int(payload["half"])
        original_index = int(payload["original_index"])
        part_id = int(payload["group_id"])
        current_size = int(payload["current_size"])
        shell_indices = np.asarray(payload["scale_shell_indices"], dtype=np.int32)
        if term_source == "norm":
            _require(
                "norm_a2_per_pixel_by_chunk" in payload
                and "norm_xa_per_pixel_by_chunk" in payload,
                "normalization pixel chunks are absent",
            )
            mask = np.ones(shell_indices.shape, dtype=bool)
            aa_chunks = np.asarray(payload["norm_a2_per_pixel_by_chunk"], dtype=np.float32)
            xa_chunks = np.asarray(payload["norm_xa_per_pixel_by_chunk"], dtype=np.float32)
            aa_per_pixel = np.sum(aa_chunks, axis=0, dtype=np.float32).astype(np.float64)
            xa_per_pixel = np.sum(xa_chunks, axis=0, dtype=np.float32).astype(np.float64)
            aa_per_shell = np.asarray(
                [
                    np.sum(aa_per_pixel[shell_indices == shell], dtype=np.float64)
                    for shell in range(current_size // 2 + 1)
                ],
                dtype=np.float64,
            )
            captured_a2_scalar = float(payload["norm_a2_per_image"])
            captured_xa_scalar = float(payload["norm_xa_per_image"])
        else:
            mask = np.asarray(payload["scale_correction_pixel_mask"], dtype=bool)
            aa_per_pixel = np.asarray(payload["scale_aa_per_pixel"], dtype=np.float64)
            aa_per_shell = np.asarray(payload["scale_aa_per_shell"], dtype=np.float64)
            xa_per_pixel = (
                np.asarray(payload["scale_xa_per_pixel"], dtype=np.float64)
                if "scale_xa_per_pixel" in payload
                else None
            )
            captured_a2_scalar = None
            captured_xa_scalar = None
        atomic_aa_per_pixel = (
            np.asarray(payload["scale_aa_atomic_per_pixel"], dtype=np.float64)
            if "scale_aa_atomic_per_pixel" in payload
            else None
        )
        atomic_xa_per_pixel = (
            np.asarray(payload["scale_xa_atomic_per_pixel"], dtype=np.float64)
            if "scale_xa_atomic_per_pixel" in payload
            else None
        )
        atomic_diff2_per_shell = (
            np.asarray(payload["wavg_diff2_atomic_per_shell"], dtype=np.float64)
            if "wavg_diff2_atomic_per_shell" in payload
            else None
        )
    _require(iteration == expected_iteration and half == expected_half, "iteration/half identity changed")
    _require(original_index == expected_original_index, "source-particle identity changed")
    _require(part_id == expected_part_id, "RECOVAR group and native part identity differ")
    _require(mask.shape == shell_indices.shape == aa_per_pixel.shape, "RECOVAR pixel topology changed")

    window_indices, _ = make_fourier_window_indices_np(
        (image_size, image_size),
        current_size,
        square=False,
        include_dc=True,
        exact_radius=True,
    )
    _require(window_indices.size == aa_per_pixel.size, "window reconstruction does not match capture")
    coordinates = np.rint(make_frequency_coords_half_np((image_size, image_size))).astype(np.int32)
    recovar_coordinates = coordinates[window_indices]
    _require(len({tuple(row) for row in recovar_coordinates}) == recovar_coordinates.shape[0], "RECOVAR coordinates repeat")

    native = _native_pixels(
        native_pixels,
        iteration=expected_iteration,
        half=expected_half,
        part_id=expected_part_id,
    )
    active_rows = np.flatnonzero(mask)
    missing = [tuple(recovar_coordinates[row]) for row in active_rows if tuple(recovar_coordinates[row]) not in native]
    _require(not missing, f"native capture misses {len(missing)} active RECOVAR coordinates")

    native_shell = np.asarray([native[tuple(recovar_coordinates[row])][0] for row in active_rows], dtype=np.int32)
    native_aa = np.asarray([native[tuple(recovar_coordinates[row])][1] for row in active_rows], dtype=np.float64)
    native_xa = np.asarray([native[tuple(recovar_coordinates[row])][2] for row in active_rows], dtype=np.float64)
    recovar_shell = shell_indices[active_rows]
    recovar_aa = aa_per_pixel[active_rows] / float(recovar_term_divisor)
    _require(np.array_equal(recovar_shell, native_shell), "native and RECOVAR shell labels differ")

    pixel_metric = _metric(recovar_aa, native_aa)
    positive = (recovar_aa > 0.0) & (native_aa > 0.0)
    _require(np.any(positive), "no jointly positive AA pixels")
    ratios = recovar_aa[positive] / native_aa[positive]
    residual = recovar_aa - native_aa
    ranked = np.argsort(np.abs(residual))[::-1]

    active_shells = np.unique(recovar_shell)
    recovar_shell_reduced = np.asarray(
        [np.sum(recovar_aa[recovar_shell == shell], dtype=np.float64) for shell in active_shells]
    )
    native_shell_reduced = np.asarray(
        [np.sum(native_aa[native_shell == shell], dtype=np.float64) for shell in active_shells]
    )
    shell_metric = _metric(recovar_shell_reduced, native_shell_reduced)
    captured_shell_reference = aa_per_shell[active_shells] / float(recovar_term_divisor)
    atomic_report = None
    if atomic_aa_per_pixel is not None:
        _require(atomic_aa_per_pixel.shape == mask.shape, "atomic AA pixel topology changed")
        atomic_active = atomic_aa_per_pixel[active_rows] / float(recovar_term_divisor)
        atomic_shell_reduced = np.asarray(
            [
                np.sum(atomic_active[recovar_shell == shell], dtype=np.float64)
                for shell in active_shells
            ]
        )
        atomic_report = {
            "pixel": _metric(atomic_active, native_aa),
            "fixed_order_shell_reduction": _metric(
                atomic_shell_reduced,
                native_shell_reduced,
            ),
        }
    xa_report = None
    if xa_per_pixel is not None:
        _require(xa_per_pixel.shape == mask.shape, "XA pixel topology changed")
        xa_active = xa_per_pixel[active_rows] / float(recovar_term_divisor)
        xa_shell_reduced = np.asarray(
            [
                np.sum(xa_active[recovar_shell == shell], dtype=np.float64)
                for shell in active_shells
            ]
        )
        native_xa_shell_reduced = np.asarray(
            [
                np.sum(native_xa[native_shell == shell], dtype=np.float64)
                for shell in active_shells
            ]
        )
        xa_report = {
            "pixel": _metric(xa_active, native_xa),
            "fixed_order_shell_reduction": _metric(
                xa_shell_reduced,
                native_xa_shell_reduced,
            ),
            "atomic": None,
        }
        if atomic_xa_per_pixel is not None:
            _require(atomic_xa_per_pixel.shape == mask.shape, "atomic XA pixel topology changed")
            atomic_xa_active = atomic_xa_per_pixel[active_rows] / float(recovar_term_divisor)
            atomic_xa_shell_reduced = np.asarray(
                [
                    np.sum(atomic_xa_active[recovar_shell == shell], dtype=np.float64)
                    for shell in active_shells
                ]
            )
            xa_report["atomic"] = {
                "pixel": _metric(atomic_xa_active, native_xa),
                "fixed_order_shell_reduction": _metric(
                    atomic_xa_shell_reduced,
                    native_xa_shell_reduced,
                ),
            }

    direct_residual_report = None
    if native_noise_components is not None:
        _require(
            atomic_diff2_per_shell is not None,
            "native noise components require captured Wavg diff2 atomics",
        )
        native_direct = _native_direct_residual_shells(
            native_noise_components,
            iteration=expected_iteration,
            half=expected_half,
            part_id=expected_part_id,
        )
        compared_shells = np.asarray(
            sorted(
                shell
                for shell in native_direct
                if 0 <= shell < atomic_diff2_per_shell.size
            ),
            dtype=np.int32,
        )
        _require(compared_shells.size > 0, "no common Wavg direct-residual shells")
        recovar_direct = (
            atomic_diff2_per_shell[compared_shells] / float(recovar_term_divisor)
        )
        native_direct_values = np.asarray(
            [native_direct[int(shell)] for shell in compared_shells],
            dtype=np.float64,
        )
        direct_residual_report = {
            **_metric(recovar_direct, native_direct_values),
            "shells": compared_shells.tolist(),
            "recovar_native_units": recovar_direct.tolist(),
            "native": native_direct_values.tolist(),
        }

    return {
        "schema": "recovar.em.k1_scale_aa_pixels.v1",
        "identity": {
            "iteration": iteration,
            "half": half,
            "part_id": part_id,
            "original_index_zero_based": original_index,
            "image_size": image_size,
            "current_size": current_size,
            "native_pixel_count": len(native),
            "recovar_window_pixel_count": int(mask.size),
            "active_pixel_count": int(active_rows.size),
            "active_shells": active_shells.tolist(),
            "recovar_term_divisor": float(recovar_term_divisor),
            "term_source": term_source,
        },
        "coordinate_join": {
            "missing_active_recovar_coordinates": len(missing),
            "shell_labels_exact": True,
        },
        "pixel_aa": {
            **pixel_metric,
            "ratio_median": float(np.median(ratios)),
            "ratio_p05": float(np.percentile(ratios, 5)),
            "ratio_p95": float(np.percentile(ratios, 95)),
            "largest_abs_residual_pixels": [
                {
                    "x": int(recovar_coordinates[active_rows[row], 0]),
                    "y": int(recovar_coordinates[active_rows[row], 1]),
                    "shell": int(recovar_shell[row]),
                    "native_aa": float(native_aa[row]),
                    "native_xa": float(native_xa[row]),
                    "recovar_aa_native_units": float(recovar_aa[row]),
                    "signed_delta": float(residual[row]),
                }
                for row in ranked[:20]
            ],
        },
        "fixed_order_shell_reduction": {
            **shell_metric,
            "recovar_reduction_vs_capture": _metric(
                recovar_shell_reduced,
                captured_shell_reference,
            ),
        },
        "recovar_pixel_sum_vs_captured_scalar": (
            None
            if captured_a2_scalar is None
            else {
                "a2_pixel_sum": float(np.sum(aa_per_pixel, dtype=np.float64)),
                "a2_captured_scalar": captured_a2_scalar,
                "a2_signed_delta": float(np.sum(aa_per_pixel, dtype=np.float64) - captured_a2_scalar),
                "xa_pixel_sum": float(np.sum(xa_per_pixel, dtype=np.float64)),
                "xa_captured_scalar": captured_xa_scalar,
                "xa_signed_delta": float(np.sum(xa_per_pixel, dtype=np.float64) - captured_xa_scalar),
            }
        ),
        "atomic_aa": atomic_report,
        "xa": xa_report,
        "wavg_direct_residual": direct_residual_report,
        "artifacts": {
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
            "native_pixels": str(native_pixels.resolve()),
            "native_pixels_sha256": _sha256(native_pixels),
            "native_noise_components": (
                None
                if native_noise_components is None
                else str(native_noise_components.resolve())
            ),
            "native_noise_components_sha256": (
                None
                if native_noise_components is None
                else _sha256(native_noise_components)
            ),
        },
        "classification": (
            "atomic Wavg XA/AA treatment captured"
            if atomic_report is not None and xa_report is not None and xa_report["atomic"] is not None
            else "atomic Wavg AA treatment captured"
            if atomic_report is not None
            else "AA differs before shell reduction"
            if pixel_metric["relative_l2"] > 1e-6
            else "per-pixel AA agrees at capture precision; inspect particle-local reduction"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--native-pixels", type=Path, required=True)
    parser.add_argument("--native-noise-components", type=Path)
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--half", type=int, default=1)
    parser.add_argument("--part-id", type=int, required=True)
    parser.add_argument("--original-index", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--recovar-term-divisor", type=float, default=float(128**4))
    parser.add_argument("--term-source", choices=("scale", "norm"), default="scale")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.recovar_capture,
        args.native_pixels,
        native_noise_components=args.native_noise_components,
        expected_iteration=args.iteration,
        expected_half=args.half,
        expected_part_id=args.part_id,
        expected_original_index=args.original_index,
        image_size=args.image_size,
        recovar_term_divisor=args.recovar_term_divisor,
        term_source=args.term_source,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
