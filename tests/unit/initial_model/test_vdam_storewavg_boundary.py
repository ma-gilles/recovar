from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_vdam_storewavg_boundary import (
    _complex_long_3d,
    _fftw_window_to_native_crop,
    _load_unmasked_image,
    _match_rotations,
    _metric,
    _native_gradient_rows,
    _select_recovar_particle_rows,
)

pytestmark = pytest.mark.unit


def test_fftw_window_maps_full_box_rows_to_native_crop_and_centered_rows():
    half_width = 65
    fftw_indices = np.asarray(
        [0 * half_width + 0, 1 * half_width + 2, 127 * half_width + 3],
        dtype=np.int32,
    )

    crop, centered = _fftw_window_to_native_crop(
        fftw_indices,
        physical_image_size=128,
        current_size=38,
    )

    np.testing.assert_array_equal(crop, np.asarray([0, 22, 743], dtype=np.int32))
    np.testing.assert_array_equal(
        centered,
        np.asarray([64 * half_width, 65 * half_width + 2, 63 * half_width + 3], dtype=np.int32),
    )


def test_match_rotations_is_tolerance_bounded_and_one_to_one():
    native = np.stack((np.eye(3), np.diag([-1.0, -1.0, 1.0]))).astype(np.float32)
    recovar = native[::-1].copy()
    recovar[0, 0, 0] += np.float32(5.0e-7)

    np.testing.assert_array_equal(_match_rotations(native, recovar, 1.0e-6), np.asarray([1, 0]))
    with pytest.raises(ValueError, match="absent"):
        _match_rotations(native, recovar, 1.0e-8)


def test_native_gradient_rows_replays_relion_residual_formula():
    probabilities = np.asarray([[0.25, 0.75], [0.5, 0.0]], dtype=np.float32)
    translated = np.asarray([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    projections = np.asarray([[0.5 + 0.25j, 1 + 0.5j], [2 + 1j, 3 + 1.5j]], dtype=np.complex64)
    ctf = np.asarray([2.0, -3.0], dtype=np.float32)
    inverse_noise = np.asarray([4.0, 5.0], dtype=np.float32)

    data, weight = _native_gradient_rows(probabilities, translated, projections, ctf, inverse_noise)

    mass = probabilities.sum(axis=1)
    expected_weight = mass[:, None] * (ctf * ctf * inverse_noise)[None, :]
    expected_data = (
        probabilities @ translated * (ctf * inverse_noise)[None, :]
        - projections * expected_weight
    )
    np.testing.assert_array_equal(weight, expected_weight.astype(np.float32))
    np.testing.assert_allclose(data, expected_data.astype(np.complex64), rtol=1.0e-7, atol=1.0e-7)


def test_metric_uses_relative_l2_and_complex_inner_product():
    reference = np.asarray([1 + 1j, 2 - 1j], dtype=np.complex64)
    result = _metric(reference, reference.copy())

    assert result["relative_l2"] == 0.0
    assert result["cosine"] == pytest.approx(1.0)
    assert result["max_abs"] == 0.0


def test_complex_long_3d_reads_relion_multidimarray_dump(tmp_path):
    path = tmp_path / "p0_Fimg_nomask.bin"
    dimensions = np.asarray([1, 2, 3], dtype=np.int_)
    values = np.arange(6, dtype=np.float64).astype(np.complex128).reshape(1, 2, 3)
    path.write_bytes(dimensions.tobytes() + values.tobytes())

    np.testing.assert_array_equal(_complex_long_3d(path), values)
    np.testing.assert_array_equal(_load_unmasked_image(path), values)


def test_load_unmasked_image_rejects_masked_scoring_operand(tmp_path):
    path = tmp_path / "Fimg_unweighted.bin"
    path.write_bytes(b"not used")

    with pytest.raises(ValueError, match="refusing masked"):
        _load_unmasked_image(path)


def test_select_recovar_particle_rows_uses_original_identity():
    capture = {
        "original_indices": np.asarray([100, 0], dtype=np.int64),
        "active_particle_rows": np.asarray([0, 1, 0, 1, 1], dtype=np.int32),
    }

    slot, row_mask = _select_recovar_particle_rows(capture, 0)

    assert slot == 1
    np.testing.assert_array_equal(row_mask, np.asarray([False, True, False, True, True]))


def test_select_recovar_particle_rows_requires_identity_for_panel():
    capture = {
        "original_indices": np.asarray([0, 100], dtype=np.int64),
        "active_particle_rows": np.asarray([0, 1], dtype=np.int32),
    }

    with pytest.raises(ValueError, match="required for a panel"):
        _select_recovar_particle_rows(capture, None)
