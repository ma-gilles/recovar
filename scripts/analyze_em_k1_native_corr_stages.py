#!/usr/bin/env python3
"""Compare native RELION K=1 ``corr_img`` arithmetic stage by stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_flat(path: Path, dtype: np.dtype) -> np.ndarray:
    with path.open("rb") as stream:
        count = int(np.fromfile(stream, dtype=np.uint64, count=1)[0])
        values = np.fromfile(stream, dtype=dtype, count=count)
    if values.size != count:
        raise ValueError(f"truncated flat capture: {path}")
    return values


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"stage shapes differ: {left.shape} != {right.shape}")
    delta = left.astype(np.float64) - right.astype(np.float64)
    denominator = float(np.linalg.norm(right.astype(np.float64)))
    ulp = np.abs(
        left.view(np.uint32).astype(np.int64) - right.view(np.uint32).astype(np.int64)
    )
    return {
        "count": int(left.size),
        "exact_count": int(np.count_nonzero(left == right)),
        "relative_l2": float(np.linalg.norm(delta) / denominator) if denominator else 0.0,
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "max_ulp": int(np.max(ulp, initial=0)),
    }


def analyze(capture_dir: Path) -> dict:
    capture_dir = Path(capture_dir)
    prefix = capture_dir / "img0_part634_fineCorr_"

    def read(name: str, dtype: np.dtype) -> np.ndarray:
        return _read_flat(Path(f"{prefix}{name}.bin"), dtype)

    minvsigma2 = read("minvsigma2", np.float32)
    fctf_float = read("fctf", np.float32)
    fctf_rfloat = read("fctf_rfloat", np.float64)
    captured_float_square = read("fctf_squared", np.float32)
    captured_rfloat_square = read("fctf_squared_rfloat", np.float64)
    captured_float_path = read("after_ctf", np.float32)
    captured_source_path = read("after_ctf_source_semantics", np.float32)
    corr_img = read("corr_img", np.float32)
    arrays = (
        fctf_float,
        fctf_rfloat,
        captured_float_square,
        captured_rfloat_square,
        captured_float_path,
        captured_source_path,
        corr_img,
    )
    if any(values.shape != minvsigma2.shape for values in arrays):
        raise ValueError("corr_img stage captures have different lengths")

    float_square = np.asarray(fctf_float * fctf_float, dtype=np.float32)
    rfloat_square = fctf_rfloat * fctf_rfloat
    float_path = np.asarray(minvsigma2 * float_square, dtype=np.float32)
    source_path = np.asarray(
        minvsigma2.astype(np.float64) * rfloat_square,
        dtype=np.float32,
    )
    comparisons = {
        "captured_float_square_vs_replay": _comparison(captured_float_square, float_square),
        "captured_rfloat_square_vs_replay_cast_f32": _comparison(
            captured_rfloat_square.astype(np.float32), rfloat_square.astype(np.float32)
        ),
        "captured_float_path_vs_replay": _comparison(captured_float_path, float_path),
        "captured_source_path_vs_replay": _comparison(captured_source_path, source_path),
        "corr_img_vs_float_path": _comparison(corr_img, float_path),
        "corr_img_vs_source_path": _comparison(corr_img, source_path),
        "corr_img_vs_captured_source_path": _comparison(corr_img, captured_source_path),
    }
    fimg_classification = None
    fimg_actual_real_path = Path(f"{prefix}fimg_corrected_actual_real.bin")
    if fimg_actual_real_path.exists():
        raw_real = read("fimg_raw_real_rfloat", np.float64)
        raw_imag = read("fimg_raw_imag_rfloat", np.float64)
        pixel_correction = read("pixel_correction", np.float32)
        expected_real = read("fimg_corrected_expected_real", np.float32)
        expected_imag = read("fimg_corrected_expected_imag", np.float32)
        actual_real = read("fimg_corrected_actual_real", np.float32)
        actual_imag = read("fimg_corrected_actual_imag", np.float32)
        scale = np.float32(read("scale", np.float64)[0])
        initial_correction = np.float32(1.0 / float(scale))
        ctf_nonzero = np.abs(fctf_rfloat) > 1e-8
        source_correction = np.where(
            ctf_nonzero,
            np.asarray(initial_correction.astype(np.float64) / fctf_rfloat, dtype=np.float32),
            initial_correction,
        ).astype(np.float32)
        float_correction = np.where(
            ctf_nonzero,
            np.asarray(initial_correction / fctf_float, dtype=np.float32),
            initial_correction,
        ).astype(np.float32)
        replay_real = np.asarray(raw_real * source_correction.astype(np.float64), dtype=np.float32)
        replay_imag = np.asarray(raw_imag * source_correction.astype(np.float64), dtype=np.float32)
        comparisons.update(
            {
                "pixel_correction_vs_source_replay": _comparison(
                    pixel_correction, source_correction
                ),
                "pixel_correction_vs_float_ctf_path": _comparison(
                    pixel_correction, float_correction
                ),
                "fimg_expected_real_vs_source_replay": _comparison(
                    expected_real, replay_real
                ),
                "fimg_expected_imag_vs_source_replay": _comparison(
                    expected_imag, replay_imag
                ),
                "fimg_actual_real_vs_expected": _comparison(actual_real, expected_real),
                "fimg_actual_imag_vs_expected": _comparison(actual_imag, expected_imag),
            }
        )
        fimg_source_exact = (
            comparisons["pixel_correction_vs_source_replay"]["exact_count"]
            == int(pixel_correction.size)
            and comparisons["fimg_actual_real_vs_expected"]["exact_count"]
            == int(pixel_correction.size)
            and comparisons["fimg_actual_imag_vs_expected"]["exact_count"]
            == int(pixel_correction.size)
        )
        fimg_float_exact = (
            comparisons["pixel_correction_vs_float_ctf_path"]["exact_count"]
            == int(pixel_correction.size)
        )
        if fimg_source_exact and not fimg_float_exact:
            fimg_classification = (
                "native_corrected_fimg_requires_rfloat_ctf_division_before_xfloat_cast"
            )
        elif fimg_source_exact:
            fimg_classification = "native_corrected_fimg_matches_both_tested_ctf_paths"
        else:
            fimg_classification = "native_corrected_fimg_has_an_unmodelled_stage_difference"
    count = int(corr_img.size)
    source_exact = comparisons["corr_img_vs_source_path"]["exact_count"] == count
    float_exact = comparisons["corr_img_vs_float_path"]["exact_count"] == count
    if source_exact and not float_exact:
        classification = "native_corr_img_requires_rfloat_ctf_square_before_xfloat_cast"
    elif source_exact:
        classification = "native_corr_img_matches_both_tested_paths"
    else:
        classification = "native_corr_img_has_an_earlier_or_unmodelled_stage_difference"
    return {
        "schema": 1,
        "particle_id": 634,
        "pixel_count": count,
        "classification": classification,
        "fimg_classification": fimg_classification,
        "comparisons": comparisons,
    }


def _markdown(report: dict) -> str:
    rows = [
        "# Native K=1 corr_img stage comparison",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "| Comparison | Exact | Count | Relative L2 | Max abs | Max ULP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["comparisons"].items():
        rows.append(
            f"| `{name}` | {values['exact_count']} | {values['count']} | "
            f"{values['relative_l2']:.9g} | {values['max_abs']:.9g} | {values['max_ulp']} |"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.capture_dir)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
