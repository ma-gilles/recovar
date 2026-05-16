"""Unit tests for parsing RELION operand dump directories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.parse_relion_dump_dir import parse_dump_dir


pytestmark = pytest.mark.unit


def _write_real_2d(path: Path, arr):
    arr = np.asarray(arr, dtype=np.float64)
    with path.open("wb") as f:
        f.write(np.int32(arr.shape[0]).tobytes())
        f.write(np.int32(arr.shape[1]).tobytes())
        f.write(arr.tobytes())


def _write_complex_2d(path: Path, arr):
    arr = np.asarray(arr, dtype=np.complex128)
    with path.open("wb") as f:
        f.write(np.int32(arr.shape[0]).tobytes())
        f.write(np.int32(arr.shape[1]).tobytes())
        f.write(arr.tobytes())


def _write_flat_real(path: Path, arr):
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    with path.open("wb") as f:
        f.write(np.int32(arr.size).tobytes())
        f.write(arr.tobytes())


def _write_flat_int(path: Path, arr):
    arr = np.asarray(arr, dtype=np.int32).reshape(-1)
    with path.open("wb") as f:
        f.write(np.int32(arr.size).tobytes())
        f.write(arr.tobytes())


def _write_scalar(path: Path, value):
    np.array(float(value), dtype=np.float64).tofile(path)


def test_parse_relion_dump_dir_reads_known_file_types(tmp_path):
    (tmp_path / "dimensions.txt").write_text(
        "nr_dir=4\nnr_psi=2\nnr_trans=3\ncurrent_size=80\npixel_size=4.25\n"
    )
    _write_real_2d(tmp_path / "Fctf.bin", np.arange(6).reshape(2, 3))
    _write_complex_2d(tmp_path / "Fimg_unweighted.bin", np.arange(6).reshape(2, 3) + 1j)
    _write_flat_real(tmp_path / "exp_Mweight_posterior.bin", [1.0, 2.0, 3.0])
    _write_flat_real(tmp_path / "candidate_weight_normalized.bin", [0.1, 0.2, 0.3])
    _write_flat_real(tmp_path / "candidate_combined_log_prior.bin", [-3.0, -2.0, -1.0])
    _write_flat_real(tmp_path / "candidate_translation_x.bin", [0.0, 1.5, 0.0])
    _write_flat_real(tmp_path / "candidate_translation_y.bin", [0.0, -0.5, 1.5])
    _write_flat_real(tmp_path / "directions_prior.bin", [0.25, 0.75])
    _write_flat_real(tmp_path / "pdf_offset.bin", [-0.5, -0.25, 0.0])
    _write_flat_real(tmp_path / "pdf_orientation.bin", [-1.5, -1.0, -0.5, 0.0])
    _write_flat_int(tmp_path / "pointer_dir_nonzeroprior.bin", [5, 8])
    _write_flat_int(tmp_path / "candidate_in_denominator_set.bin", [1, 1, 1])
    _write_flat_int(tmp_path / "candidate_coarse_trans_idx.bin", [0, 0, 1])
    _write_scalar(tmp_path / "Pmax.bin", 0.6)

    parsed = parse_dump_dir(tmp_path)

    assert int(parsed["header_nr_dir"]) == 4
    assert float(parsed["header_pixel_size"]) == pytest.approx(4.25)
    np.testing.assert_array_equal(parsed["Fctf"], np.arange(6, dtype=np.float64).reshape(2, 3))
    np.testing.assert_array_equal(parsed["Fimg_unweighted"], np.arange(6, dtype=np.float64).reshape(2, 3) + 1j)
    np.testing.assert_array_equal(parsed["exp_Mweight_posterior"], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(parsed["candidate_weight_normalized"], np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(parsed["candidate_combined_log_prior"], np.array([-3.0, -2.0, -1.0]))
    np.testing.assert_array_equal(parsed["candidate_translation_x"], np.array([0.0, 1.5, 0.0]))
    np.testing.assert_array_equal(parsed["candidate_translation_y"], np.array([0.0, -0.5, 1.5]))
    np.testing.assert_array_equal(parsed["directions_prior"], np.array([0.25, 0.75]))
    np.testing.assert_array_equal(parsed["pdf_offset"], np.array([-0.5, -0.25, 0.0]))
    np.testing.assert_array_equal(parsed["pdf_orientation"], np.array([-1.5, -1.0, -0.5, 0.0]))
    np.testing.assert_array_equal(parsed["pointer_dir_nonzeroprior"], np.array([5, 8], dtype=np.int32))
    np.testing.assert_array_equal(parsed["candidate_in_denominator_set"], np.array([1, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(parsed["candidate_coarse_trans_idx"], np.array([0, 0, 1], dtype=np.int32))
    assert float(parsed["Pmax"]) == pytest.approx(0.6)
