"""The K=1 pass-2 dump writer captures its optional RELION operands through one owner."""

import inspect

import numpy as np

from recovar.em.diagnostics import pass2 as pass2_diagnostics


def test_absent_operands_are_recorded_as_absent():
    fields = pass2_diagnostics._optional_operand_row_fields(
        1, shifted_corrected=None, direct_score_input=None, direct_preprocessed_score_input=None, direct_pixel_correction=None,
        direct_inverse_noise_score=None, direct_ctf_rfloat_score=None, direct_preprocess_normalization_factors=None,
        direct_integer_pre_shifts=None, direct_batch_image_corrections=None, direct_batch_scale_corrections=None,
    )
    assert set(fields) == {
        "shifted_corrected", "direct_score_input", "direct_preprocessed_score_input", "direct_pixel_correction", "direct_inverse_noise_score",
        "direct_ctf_rfloat_score", "relion_preprocess_normalization_factor", "relion_integer_pre_shift", "batch_image_correction", "batch_scale_correction",
    }
    for name, dtype in (("shifted_corrected", np.complex64), ("direct_score_input", np.complex64), ("direct_preprocessed_score_input", np.complex64),
                        ("direct_pixel_correction", np.float32), ("direct_inverse_noise_score", np.float32), ("direct_ctf_rfloat_score", np.float64),
                        ("relion_integer_pre_shift", np.int32)):
        assert fields[name].shape == (0,) and fields[name].dtype == dtype, name
    for name in ("relion_preprocess_normalization_factor", "batch_image_correction", "batch_scale_correction"):
        assert np.isnan(fields[name]) and fields[name].dtype == np.float32


def test_present_operands_take_the_row_with_their_capture_dtypes():
    shifted = (np.arange(12).reshape(2, 2, 3) + 0.5j).astype(np.complex64)
    ctf = np.arange(6, dtype=np.float32).reshape(2, 3)
    fields = pass2_diagnostics._optional_operand_row_fields(
        1, shifted_corrected=shifted, direct_score_input=shifted * 2, direct_preprocessed_score_input=shifted * 3, direct_pixel_correction=ctf,
        direct_inverse_noise_score=np.asarray([1.0, 2.0, 3.0], dtype=np.float32), direct_ctf_rfloat_score=ctf * 2,
        direct_preprocess_normalization_factors=np.asarray([0.9, 1.1], dtype=np.float32), direct_integer_pre_shifts=np.asarray([[1, -1], [2, 0]]),
        direct_batch_image_corrections=np.asarray([1.5, 2.5], dtype=np.float32), direct_batch_scale_corrections=np.asarray([0.5, 0.75], dtype=np.float32),
    )
    assert np.array_equal(fields["shifted_corrected"], shifted[1]) and np.array_equal(fields["direct_score_input"], shifted[1] * 2)
    assert np.array_equal(fields["direct_pixel_correction"], ctf[1])
    assert np.array_equal(fields["direct_inverse_noise_score"], [1.0, 2.0, 3.0])
    assert fields["direct_ctf_rfloat_score"].dtype == np.float64 and np.array_equal(fields["direct_ctf_rfloat_score"], ctf[1] * 2)
    assert fields["relion_preprocess_normalization_factor"] == np.float32(1.1)
    assert fields["relion_integer_pre_shift"].dtype == np.int32 and fields["relion_integer_pre_shift"].tolist() == [2, 0]
    assert fields["batch_image_correction"] == np.float32(2.5) and fields["batch_scale_correction"] == np.float32(0.75)


def test_both_dump_schemas_use_the_owner():
    source = inspect.getsource(pass2_diagnostics._maybe_dump_pass2_bucket)
    assert source.count("**_optional_operand_row_fields(") == 2
    # the ten score operands and their absent encodings live in the owner
    assert "np.float32(np.nan)" not in source
    for name in ("shifted_corrected=(", "direct_score_input=(", "relion_integer_pre_shift=(", "batch_scale_correction=("):
        assert name not in source, name
    # the reconstruction-side operands stay with each schema on purpose
    assert source.count("shifted_recon=(") == 2
    assert source.count("ctf2_over_nv_recon=(") == 2
