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


def _write_flat_split_complex(path: Path, arr):
    arr = np.asarray(arr, dtype=np.complex128).reshape(-1)
    split = np.concatenate([arr.real, arr.imag]).astype(np.float64)
    with path.open("wb") as f:
        f.write(np.int32(split.size).tobytes())
        f.write(split.tobytes())


def _write_flat_int(path: Path, arr):
    arr = np.asarray(arr, dtype=np.int32).reshape(-1)
    with path.open("wb") as f:
        f.write(np.int32(arr.size).tobytes())
        f.write(arr.tobytes())


def _write_scalar(path: Path, value):
    np.array(float(value), dtype=np.float64).tofile(path)


def _write_scalar_i32(path: Path, value):
    np.array(int(value), dtype=np.int32).tofile(path)


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
    _write_flat_int(tmp_path / "candidate_class_idx.bin", [0, 2, 1])
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
    np.testing.assert_array_equal(parsed["candidate_class_idx"], np.array([0, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(parsed["candidate_coarse_trans_idx"], np.array([0, 0, 1], dtype=np.int32))
    assert float(parsed["Pmax"]) == pytest.approx(0.6)


def test_parse_relion_dump_dir_classifies_pass_prefixed_files(tmp_path):
    _write_real_2d(tmp_path / "pass0_Fctf.bin", np.arange(6).reshape(2, 3))
    _write_real_2d(tmp_path / "pass0_over0_sigma2_noise.bin", np.arange(4).reshape(1, 4))
    _write_flat_real(tmp_path / "pass0_candidate_weight_normalized.bin", [0.7, 0.2, 0.1])
    _write_flat_real(tmp_path / "pass1_exp_Mweight_raw_preprior.bin", [-3.0, -2.0])
    _write_flat_real(tmp_path / "pass0_img0_corr_img.bin", [4.0, 5.0])
    _write_flat_real(tmp_path / "pass0_img0_Fimg_corrected_real.bin", [1.0, 2.0])
    _write_flat_real(tmp_path / "pass0_img0_Fimg_corrected_imag.bin", [-1.0, -2.0])
    _write_flat_int(tmp_path / "pass0_candidate_in_fine_threshold_set.bin", [1, 0, 1])
    _write_flat_int(tmp_path / "pass0_coarse_candidate_rot_idx.bin", [4, 5, 6])
    _write_scalar_i32(tmp_path / "pass0_acc_iter.bin", 2)

    parsed = parse_dump_dir(tmp_path)

    np.testing.assert_array_equal(parsed["pass0_Fctf"], np.arange(6, dtype=np.float64).reshape(2, 3))
    np.testing.assert_array_equal(parsed["pass0_over0_sigma2_noise"], np.arange(4, dtype=np.float64).reshape(1, 4))
    np.testing.assert_array_equal(parsed["pass0_candidate_weight_normalized"], np.array([0.7, 0.2, 0.1]))
    np.testing.assert_array_equal(parsed["pass1_exp_Mweight_raw_preprior"], np.array([-3.0, -2.0]))
    np.testing.assert_array_equal(parsed["pass0_img0_corr_img"], np.array([4.0, 5.0]))
    np.testing.assert_array_equal(parsed["pass0_img0_Fimg_corrected_real"], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(parsed["pass0_img0_Fimg_corrected_imag"], np.array([-1.0, -2.0]))
    np.testing.assert_array_equal(
        parsed["pass0_candidate_in_fine_threshold_set"],
        np.array([1, 0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(parsed["pass0_coarse_candidate_rot_idx"], np.array([4, 5, 6], dtype=np.int32))
    assert int(parsed["pass0_acc_iter"]) == 2


def test_parse_relion_dump_dir_classifies_firstiter_cc_files(tmp_path):
    _write_flat_real(tmp_path / "pass1_firstiter_cc_exp_Mweight_raw_preonehot.bin", [-4.0, -5.0, -3.0])
    _write_flat_int(tmp_path / "pass1_firstiter_cc_raw_rot_idx.bin", [8, 9, 10])
    _write_flat_int(tmp_path / "pass1_firstiter_cc_raw_trans_idx.bin", [80, 81, 82])
    _write_flat_int(tmp_path / "pass1_firstiter_cc_raw_rot_id.bin", [18, 19, 20])
    _write_flat_int(tmp_path / "pass1_firstiter_cc_raw_ihidden_overs.bin", [0, 1, 2])
    _write_flat_int(tmp_path / "pass1_firstiter_cc_weight_dims.bin", [7, 1, 1, 1, 3, 1, 1])

    parsed = parse_dump_dir(tmp_path)

    np.testing.assert_array_equal(
        parsed["pass1_firstiter_cc_exp_Mweight_raw_preonehot"],
        np.array([-4.0, -5.0, -3.0]),
    )
    np.testing.assert_array_equal(parsed["pass1_firstiter_cc_raw_rot_idx"], np.array([8, 9, 10], dtype=np.int32))
    np.testing.assert_array_equal(parsed["pass1_firstiter_cc_raw_trans_idx"], np.array([80, 81, 82], dtype=np.int32))
    np.testing.assert_array_equal(parsed["pass1_firstiter_cc_raw_rot_id"], np.array([18, 19, 20], dtype=np.int32))
    np.testing.assert_array_equal(
        parsed["pass1_firstiter_cc_raw_ihidden_overs"],
        np.array([0, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        parsed["pass1_firstiter_cc_weight_dims"],
        np.array([7, 1, 1, 1, 3, 1, 1], dtype=np.int32),
    )


def test_parse_relion_dump_dir_classifies_part_specific_acc_files(tmp_path):
    _write_flat_real(tmp_path / "img0_part7778_pass1_class0_pass1_diff2_weights.bin", [-0.4, -0.3])
    _write_flat_split_complex(tmp_path / "img0_part7778_pass1_class0_pass1_Fimg.bin", [1.0 + 2.0j, -1.0 - 2.0j])
    _write_flat_real(tmp_path / "img0_part7778_pass1_class0_pass1_corr_img.bin", [3.0, 4.0])
    _write_flat_real(
        tmp_path / "img0_part7778_pass1_class0_pass1_eulers_matrices.bin",
        np.arange(18, dtype=np.float64),
    )
    _write_flat_real(tmp_path / "img0_part7778_pass1_class0_pass1_trans_xyz_phases.bin", [0.1, 0.2, 0.3])
    _write_scalar(tmp_path / "img0_part7778_pass1_class0_pass1_orientation_num.bin", 2)
    _write_scalar(tmp_path / "img0_part7778_pass1_class0_pass1_translation_num.bin", 1)

    parsed = parse_dump_dir(tmp_path)

    np.testing.assert_array_equal(
        parsed["img0_part7778_pass1_class0_pass1_diff2_weights"],
        np.array([-0.4, -0.3]),
    )
    np.testing.assert_array_equal(
        parsed["img0_part7778_pass1_class0_pass1_Fimg"],
        np.array([1.0 + 2.0j, -1.0 - 2.0j], dtype=np.complex128),
    )
    np.testing.assert_array_equal(parsed["img0_part7778_pass1_class0_pass1_corr_img"], np.array([3.0, 4.0]))
    np.testing.assert_array_equal(
        parsed["img0_part7778_pass1_class0_pass1_eulers_matrices"],
        np.arange(18, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        parsed["img0_part7778_pass1_class0_pass1_trans_xyz_phases"],
        np.array([0.1, 0.2, 0.3]),
    )
    assert int(parsed["img0_part7778_pass1_class0_pass1_orientation_num"]) == 2
    assert int(parsed["img0_part7778_pass1_class0_pass1_translation_num"]) == 1


def test_parse_relion_dump_dir_classifies_projector_and_component_files(tmp_path):
    _write_flat_int(tmp_path / "pass1_class0_ppref_dims.bin", [58, 115, 115, 0, -57, -57, 28])
    _write_flat_real(tmp_path / "pass1_class0_ppref_real.bin", [1.0, 2.0, 3.0])
    _write_flat_real(tmp_path / "pass1_class0_ppref_imag.bin", [-1.0, -2.0, -3.0])
    _write_flat_real(tmp_path / "pass1_class0_fine_ref_real.bin", [0.5, 0.25])
    _write_flat_real(tmp_path / "pass1_class0_fine_ref_imag.bin", [-0.5, -0.25])
    _write_flat_real(tmp_path / "pass1_class0_fine_shifted_real.bin", [1.5, 1.25])
    _write_flat_real(tmp_path / "pass1_class0_fine_shifted_imag.bin", [-1.5, -1.25])
    _write_flat_real(tmp_path / "pass1_class0_fine_eulers.bin", np.arange(18, dtype=np.float64))
    _write_flat_real(tmp_path / "pass1_class0_fine_rots.bin", [10.0, 20.0])
    _write_flat_real(tmp_path / "pass1_class0_fine_tilts.bin", [30.0, 40.0])
    _write_flat_real(tmp_path / "pass1_class0_fine_psis.bin", [50.0, 60.0])
    _write_flat_int(tmp_path / "pass1_class0_fine_iorientclasses.bin", [0, 1])
    _write_flat_int(tmp_path / "pass1_class0_fine_iover_rots.bin", [2, 3])
    _write_flat_int(tmp_path / "pass1_class0_fine_class_entries.bin", [4])
    _write_flat_int(tmp_path / "pass1_class0_fine_class_idx.bin", [5, 6])
    _write_flat_real(tmp_path / "pass0_class0_cc_component_weight.bin", [7.0, 8.0])
    _write_flat_real(tmp_path / "pass0_class0_cc_component_norm.bin", [9.0, 10.0])

    parsed = parse_dump_dir(tmp_path)

    np.testing.assert_array_equal(
        parsed["pass1_class0_ppref_dims"],
        np.array([58, 115, 115, 0, -57, -57, 28], dtype=np.int32),
    )
    np.testing.assert_array_equal(parsed["pass1_class0_ppref_real"], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(parsed["pass1_class0_ppref_imag"], np.array([-1.0, -2.0, -3.0]))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_ref_real"], np.array([0.5, 0.25]))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_ref_imag"], np.array([-0.5, -0.25]))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_shifted_real"], np.array([1.5, 1.25]))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_shifted_imag"], np.array([-1.5, -1.25]))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_eulers"], np.arange(18, dtype=np.float64))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_rots"], np.array([10.0, 20.0]))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_tilts"], np.array([30.0, 40.0]))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_psis"], np.array([50.0, 60.0]))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_iorientclasses"], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_iover_rots"], np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_class_entries"], np.array([4], dtype=np.int32))
    np.testing.assert_array_equal(parsed["pass1_class0_fine_class_idx"], np.array([5, 6], dtype=np.int32))
    np.testing.assert_array_equal(parsed["pass0_class0_cc_component_weight"], np.array([7.0, 8.0]))
    np.testing.assert_array_equal(parsed["pass0_class0_cc_component_norm"], np.array([9.0, 10.0]))


def test_parse_relion_dump_dir_classifies_store_wavg_and_candidate_files(tmp_path):
    _write_flat_real(tmp_path / "img0_part2355_storeWavg_sorted_weights.bin", [0.9, 0.5, 0.1])
    _write_complex_2d(tmp_path / "store_candidate0_Fimg_store.bin", np.ones((2, 3), dtype=np.complex128))
    _write_complex_2d(tmp_path / "store_candidate0_Frefctf.bin", np.arange(6).reshape(2, 3) + 2j)
    _write_real_2d(tmp_path / "store_candidate0_Mctf.bin", np.arange(6).reshape(2, 3))
    _write_real_2d(tmp_path / "store_candidate0_Minvsigma2.bin", np.arange(6).reshape(2, 3) + 0.5)
    _write_scalar(tmp_path / "store_candidate0_weight_normalized.bin", 0.25)

    parsed = parse_dump_dir(tmp_path)

    np.testing.assert_array_equal(
        parsed["img0_part2355_storeWavg_sorted_weights"],
        np.array([0.9, 0.5, 0.1]),
    )
    np.testing.assert_array_equal(
        parsed["store_candidate0_Fimg_store"],
        np.ones((2, 3), dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        parsed["store_candidate0_Frefctf"],
        np.arange(6, dtype=np.float64).reshape(2, 3) + 2j,
    )
    np.testing.assert_array_equal(
        parsed["store_candidate0_Mctf"],
        np.arange(6, dtype=np.float64).reshape(2, 3),
    )
    np.testing.assert_array_equal(
        parsed["store_candidate0_Minvsigma2"],
        np.arange(6, dtype=np.float64).reshape(2, 3) + 0.5,
    )
    assert float(parsed["store_candidate0_weight_normalized"]) == pytest.approx(0.25)
