"""Unit tests for the VDAM native-PPref coarse boundary analyzer."""

import numpy as np
import pytest

from scripts.analyze_vdam_coarse_projector_boundary import (
    _centered_metric,
    _complex_metric,
    _flat_dump,
    _load_projector,
    _native_current_fft_rows,
)

pytestmark = pytest.mark.unit


def _write_flat(path, values, dtype):
    values = np.asarray(values, dtype=dtype)
    with path.open("wb") as stream:
        np.asarray([values.size], dtype=np.int32).tofile(stream)
        values.tofile(stream)


def test_load_projector_reads_count_prefixed_relion_payload(tmp_path):
    prefix = "pass1_class0_ppref_"
    _write_flat(tmp_path / f"{prefix}dims.bin", [2, 2, 1, 0, -1, 0, 3], np.int32)
    _write_flat(tmp_path / f"{prefix}real.bin", [1, 2, 3, 4], np.float64)
    _write_flat(tmp_path / f"{prefix}imag.bin", [5, 6, 7, 8], np.float64)

    projector, r_max = _load_projector(tmp_path, prefix)

    assert projector.shape == (1, 2, 2)
    assert projector.dtype == np.complex64
    assert r_max == 3
    np.testing.assert_array_equal(
        projector.reshape(-1),
        np.asarray([1 + 5j, 2 + 6j, 3 + 7j, 4 + 8j], dtype=np.complex64),
    )


def test_flat_dump_rejects_trailing_payload(tmp_path):
    path = tmp_path / "values.bin"
    _write_flat(path, [1.0, 2.0], np.float64)
    assert _flat_dump(path, np.dtype("<f8")).tolist() == [1.0, 2.0]
    with path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="payload"):
        _flat_dump(path, np.dtype("<f8"))


def test_centered_metric_removes_only_a_common_offset():
    reference = np.asarray([10.0, 12.0, 15.0], dtype=np.float32)
    candidate = np.asarray([20.0, 22.0, 26.0], dtype=np.float32)

    result = _centered_metric(reference, candidate)

    assert result["exact_count"] == 2
    assert result["max_abs"] == pytest.approx(1.0)
    assert result["rms"] == pytest.approx(1.0 / np.sqrt(3.0))


def test_complex_metric_reports_exact_values_and_relative_error():
    reference = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64)
    candidate = reference.copy()
    candidate[1] += np.complex64(1j)

    result = _complex_metric(reference, candidate)

    assert result["exact_count"] == 1
    assert result["max_abs"] == pytest.approx(1.0)


def test_native_current_fft_rows_map_native_order_into_centered_full_rows():
    rows = _native_current_fft_rows(full_size=8, current_size=4)

    np.testing.assert_array_equal(
        rows.reshape(4, 3),
        np.asarray([[20, 21, 22], [25, 26, 27], [30, 31, 32], [15, 16, 17]]),
    )
