"""Unit tests for the VDAM native translation-boundary analyzer."""

import numpy as np
import pytest

from scripts.analyze_vdam_native_translation_boundary import (
    _captured_native_current_size,
    _centered_diff2_replay_stats,
    _current_crop_to_compact,
    _flat_real_dump,
    _metric,
    _native_crop_rows,
    _native_current_fft_rows,
    _preprocess_capture,
)

pytestmark = pytest.mark.unit


def test_native_crop_rows_map_centered_half_to_relion_fftw_crop():
    full_size = 8
    current_size = 4
    half_width = full_size // 2 + 1
    # Centered rows ky=-1, 0, 1 map to native FFTW rows 3, 0, 1.
    score_indices = np.asarray([3 * half_width + 1, 4 * half_width, 5 * half_width + 2])

    crop = _native_crop_rows(score_indices, full_size, current_size)

    np.testing.assert_array_equal(crop, np.asarray([10, 0, 5], dtype=np.int32))


def test_current_crop_to_compact_preserves_native_pixel_lanes():
    lookup = _current_crop_to_compact(np.asarray([10, 0, 5]), current_size=4)

    assert lookup.shape == (12,)
    np.testing.assert_array_equal(lookup[[0, 5, 10]], np.asarray([1, 2, 0]))
    assert np.count_nonzero(lookup < 0) == 9


def test_native_current_fft_rows_map_standard_rows_into_centered_full_fft():
    rows = _native_current_fft_rows(full_size=8, current_size=4)

    np.testing.assert_array_equal(
        rows.reshape(4, 3),
        np.asarray([[20, 21, 22], [25, 26, 27], [30, 31, 32], [15, 16, 17]]),
    )


def test_metric_reports_exact_complex_values_and_residual():
    reference = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64)
    candidate = reference.copy()
    candidate[1] += np.complex64(1j)

    result = _metric(reference, candidate)

    assert result["exact_count"] == 1
    assert result["value_count"] == 2
    assert result["max_abs"] == pytest.approx(1.0)


def test_centered_diff2_replay_factors_out_constant_highres_addend():
    replay = np.asarray([10.0, 11.5, 13.0], dtype=np.float32)
    native = np.add(replay, np.float32(0.25), dtype=np.float32)

    result = _centered_diff2_replay_stats(native, replay)

    assert result["rms"] == 0.0
    assert result["max_abs"] == 0.0
    assert result["inferred_highres_mode"] == pytest.approx(0.25)
    assert result["inferred_highres_mode_count"] == 3
    assert result["inferred_highres_unique_count"] == 1


def _write_flat_real(path, values):
    values = np.asarray(values, dtype=np.float64)
    with path.open("wb") as stream:
        np.asarray([values.size], dtype=np.int32).tofile(stream)
        values.tofile(stream)


def test_captured_native_current_size_reads_exact_scalar(tmp_path):
    np.asarray([34.0], dtype=np.float64).tofile(
        tmp_path / "pass1_img0_exp_current_image_size.bin"
    )

    assert _captured_native_current_size(tmp_path) == 34


def test_flat_real_dump_rejects_trailing_payload(tmp_path):
    path = tmp_path / "values.bin"
    _write_flat_real(path, [1.0, 2.0])
    assert _flat_real_dump(path).tolist() == [1.0, 2.0]
    with path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="payload"):
        _flat_real_dump(path)


def test_preprocess_capture_loads_relion_verbose_boundaries(tmp_path):
    full_size = 4
    current_size = 2
    real = np.arange(full_size**2, dtype=np.float64)
    fourier_size = current_size * (current_size // 2 + 1)
    _write_flat_real(tmp_path / "preprocess_img0_normalized_shifted_real.bin", real)
    _write_flat_real(tmp_path / "preprocess_img0_masked_real.bin", real + 1)
    _write_flat_real(
        tmp_path / "preprocess_img0_masked_fourier_pre_optics_real.bin",
        np.arange(fourier_size),
    )
    _write_flat_real(
        tmp_path / "preprocess_img0_masked_fourier_pre_optics_imag.bin",
        np.arange(fourier_size) + 2,
    )
    _write_flat_real(
        tmp_path / "preprocess_img0_masked_fourier_post_optics_real.bin",
        np.arange(fourier_size) + 4,
    )
    _write_flat_real(
        tmp_path / "preprocess_img0_masked_fourier_post_optics_imag.bin",
        np.arange(fourier_size) + 6,
    )
    np.asarray([3.5], dtype=np.float64).tofile(
        tmp_path / "preprocess_img0_softmask_background.bin"
    )

    capture = _preprocess_capture(
        tmp_path,
        full_size=full_size,
        current_size=current_size,
    )

    assert capture["normalized_shifted"].shape == (4, 4)
    assert capture["masked"].shape == (4, 4)
    assert capture["masked_fourier_pre_optics"].shape == (4,)
    assert capture["masked_fourier_post_optics"].shape == (4,)
    np.testing.assert_array_equal(
        capture["masked_fourier_post_optics"],
        np.arange(fourier_size, dtype=np.float32)
        + 4
        + np.complex64(1j) * (np.arange(fourier_size, dtype=np.float32) + 6),
    )
    assert capture["background"] == pytest.approx(3.5)
