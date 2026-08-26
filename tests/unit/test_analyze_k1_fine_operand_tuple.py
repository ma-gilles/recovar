import numpy as np
import pytest

from scripts.analyze_k1_fine_operand_tuple import (
    _factorial_operand_substitutions,
    _largest_mismatches,
    _masked_shifted_substitutions,
    _optimal_scalar_fit,
    _score_window_rows_from_relion_full,
    _shifted_source_phase_substitutions,
)


@pytest.mark.unit
def test_largest_mismatches_preserves_full_indices_for_selected_pixels():
    relion = np.zeros(6, dtype=np.float32)
    recovar = np.asarray([100.0, 2.0, 200.0, 4.0, 300.0, 1.0], dtype=np.float32)

    mismatches = _largest_mismatches(
        relion,
        recovar,
        flat_indices=np.asarray([1, 3, 5]),
        limit=2,
    )

    assert [row["flat_index"] for row in mismatches] == [3, 1]
    assert [row["abs_delta"] for row in mismatches] == [4.0, 2.0]


@pytest.mark.unit
def test_largest_mismatches_rejects_selected_pixels_outside_operand():
    values = np.zeros(3, dtype=np.float32)

    with pytest.raises(ValueError, match="outside the flattened operand"):
        _largest_mismatches(values, values, flat_indices=np.asarray([3]))


@pytest.mark.unit
def test_score_window_mapping_handles_even_y_nyquist_and_permuted_rows():
    # current_size=4 uses FFTW rows ky=[0, 1, +2, -1].  The physical score
    # window is centered in an 8x8 image, so these selected rows map to
    # physical half-plane indices [20, 26, 32, 16].  In particular, ky=+2
    # must not be interpreted as ky=-2, which is the projector convention.
    supported_full = np.asarray([0, 4, 8, 10], dtype=np.int64)
    window_indices = np.asarray([32, 20, 16, 26], dtype=np.int64)

    rows = _score_window_rows_from_relion_full(
        supported_full=supported_full,
        window_indices=window_indices,
        image_shape=(8, 8),
        current_size=4,
    )

    assert rows.tolist() == [1, 3, 0, 2]
    assert window_indices[rows].tolist() == [20, 26, 32, 16]


@pytest.mark.unit
def test_score_window_mapping_rejects_missing_or_duplicate_pixels():
    supported_full = np.asarray([0, 8], dtype=np.int64)

    with pytest.raises(ValueError, match="duplicate physical pixels"):
        _score_window_rows_from_relion_full(
            supported_full=supported_full,
            window_indices=np.asarray([20, 20], dtype=np.int64),
            image_shape=(8, 8),
            current_size=4,
        )

    with pytest.raises(ValueError, match="score pixels and RECOVAR score window differ"):
        _score_window_rows_from_relion_full(
            supported_full=supported_full,
            window_indices=np.asarray([20, 31], dtype=np.int64),
            image_shape=(8, 8),
            current_size=4,
        )


@pytest.mark.unit
def test_factorial_operand_substitutions_covers_all_combinations():
    relion_reference = np.asarray([1.0 + 2.0j], dtype=np.complex64)
    recovar_reference = np.asarray([3.0 + 4.0j], dtype=np.complex64)
    relion_shifted = np.asarray([0.5 + 0.25j], dtype=np.complex64)
    recovar_shifted = np.asarray([0.75 + 0.125j], dtype=np.complex64)
    relion_correction = np.asarray([2.0], dtype=np.float32)
    recovar_correction = np.asarray([1.0], dtype=np.float32)

    results = _factorial_operand_substitutions(
        relion_reference=relion_reference,
        relion_shifted=relion_shifted,
        relion_correction=relion_correction,
        relion_highres=np.float32(7.0),
        recovar_reference=recovar_reference,
        recovar_shifted=recovar_shifted,
        recovar_correction=recovar_correction,
        recovar_highres=np.float32(5.0),
    )

    assert len(results) == 16
    assert "recovar" in results
    assert "native_reference_shifted_correction_highres" in results
    assert results["recovar"] != results["native_reference_shifted_correction_highres"]


@pytest.mark.unit
def test_masked_shifted_substitutions_separates_inside_and_outside_pixels():
    recovar_shifted = np.asarray([0.0, 0.0, 0.0], dtype=np.complex64)
    native_shifted = np.asarray([1.0, 2.0, 4.0], dtype=np.complex64)
    common = {
        "relion_reference": np.zeros(3, dtype=np.complex64),
        "relion_shifted": native_shifted,
        "relion_correction": np.ones(3, dtype=np.float32),
        "relion_highres": np.float32(0.0),
        "recovar_reference": np.zeros(3, dtype=np.complex64),
        "recovar_shifted": recovar_shifted,
        "recovar_correction": np.ones(3, dtype=np.float32),
        "recovar_highres": np.float32(0.0),
    }

    result = _masked_shifted_substitutions(
        **common,
        mask=np.asarray([False, True, False]),
    )

    assert result["recovar"] == 0.0
    # The native fine-score kernel applies the explicit x-half multiplicity,
    # so these three x=0 pixels each contribute one half of their square.
    assert result["native_shifted_inside_mask_only"] == 2.0
    assert result["native_shifted_outside_mask_only"] == 8.5
    assert result["native_shifted_all"] == 10.5
    assert result["native_all"] == 10.5


@pytest.mark.unit
def test_shifted_source_phase_substitutions_separates_operands():
    recovar_source = np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64)
    native_source = np.asarray([3.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex64)
    recovar_phase = np.asarray([1.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex64)
    native_phase = np.asarray([0.0 + 1.0j, -1.0 + 0.0j], dtype=np.complex64)
    result, observed_native, observed_recovar, valid = (
        _shifted_source_phase_substitutions(
            relion_reference=np.zeros(2, dtype=np.complex64),
            relion_unshifted=native_source,
            relion_shifted=native_source * native_phase,
            recovar_reference=np.zeros(2, dtype=np.complex64),
            recovar_unshifted=recovar_source,
            recovar_shifted=recovar_source * recovar_phase,
            recovar_correction=np.ones(2, dtype=np.float32),
            recovar_highres=np.float32(0.0),
            active_mask=np.ones(2, dtype=bool),
        )
    )

    assert valid.tolist() == [True, True]
    assert np.array_equal(observed_native, native_phase)
    assert np.array_equal(observed_recovar, recovar_phase)
    assert result["valid_pixel_count"] == 2
    assert result["native_unshifted_source_only"] != result["recovar"]
    assert result["native_translation_phase_only"] == result["recovar"]
    assert result["native_unshifted_source_and_phase"] == result["native_shifted_direct"]


@pytest.mark.unit
def test_optimal_scalar_fit_separates_scale_from_pixel_residual():
    recovar = np.asarray([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)
    native = np.complex64(1.25 - 0.5j) * recovar

    fit = _optimal_scalar_fit(native, recovar)

    assert np.isclose(fit["native_over_recovar_complex_scale_real"], 1.25)
    assert np.isclose(fit["native_over_recovar_complex_scale_imag"], -0.5)
    assert fit["complex_scaled_relative_l2"] < 1e-15
    assert fit["real_scaled_relative_l2"] > 0.0
