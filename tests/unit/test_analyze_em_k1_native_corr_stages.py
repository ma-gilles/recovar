from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_em_k1_native_corr_stages as analyzer


def _write_flat(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    with path.open("wb") as stream:
        np.asarray([values.size], dtype=np.uint64).tofile(stream)
        values.tofile(stream)


@pytest.mark.unit
def test_classifies_rfloat_ctf_square_boundary(tmp_path: Path) -> None:
    prefix = tmp_path / "img0_part634_fineCorr_"
    minvsigma2 = np.asarray([0.13333298, 1.750001], dtype=np.float32)
    fctf_rfloat = np.asarray(
        [0.07354116995482596, 0.1265216380265534], dtype=np.float64
    )
    fctf_float = fctf_rfloat.astype(np.float32)
    float_square = np.asarray(fctf_float * fctf_float, dtype=np.float32)
    rfloat_square = fctf_rfloat * fctf_rfloat
    float_path = np.asarray(minvsigma2 * float_square, dtype=np.float32)
    source_path = np.asarray(minvsigma2.astype(np.float64) * rfloat_square, dtype=np.float32)
    pixel_correction = np.asarray(1.0 / fctf_rfloat, dtype=np.float32)
    raw_real = np.asarray([2.123456789, -0.765432198], dtype=np.float64)
    raw_imag = np.asarray([-1.234567891, 0.456789123], dtype=np.float64)
    corrected_real = np.asarray(
        raw_real * pixel_correction.astype(np.float64), dtype=np.float32
    )
    corrected_imag = np.asarray(
        raw_imag * pixel_correction.astype(np.float64), dtype=np.float32
    )
    assert np.any(float_path != source_path)

    captures = {
        "minvsigma2": minvsigma2,
        "fctf": fctf_float,
        "fctf_rfloat": fctf_rfloat,
        "fctf_squared": float_square,
        "fctf_squared_rfloat": rfloat_square,
        "after_ctf": float_path,
        "after_ctf_source_semantics": source_path,
        "corr_img": source_path,
        "scale": np.asarray([1.0], dtype=np.float64),
        "fimg_raw_real_rfloat": raw_real,
        "fimg_raw_imag_rfloat": raw_imag,
        "pixel_correction": pixel_correction,
        "fimg_corrected_expected_real": corrected_real,
        "fimg_corrected_expected_imag": corrected_imag,
        "fimg_corrected_actual_real": corrected_real,
        "fimg_corrected_actual_imag": corrected_imag,
    }
    for name, values in captures.items():
        _write_flat(Path(f"{prefix}{name}.bin"), values)

    report = analyzer.analyze(tmp_path)
    assert report["classification"] == (
        "native_corr_img_requires_rfloat_ctf_square_before_xfloat_cast"
    )
    assert report["fimg_classification"] == (
        "native_corrected_fimg_requires_rfloat_ctf_division_before_xfloat_cast"
    )
    assert report["comparisons"]["corr_img_vs_source_path"]["exact_count"] == 2
    assert report["comparisons"]["corr_img_vs_float_path"]["exact_count"] < 2
    assert report["comparisons"]["pixel_correction_vs_source_replay"]["exact_count"] == 2
    assert report["comparisons"]["fimg_actual_real_vs_expected"]["exact_count"] == 2
