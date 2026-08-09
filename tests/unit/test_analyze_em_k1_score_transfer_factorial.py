from __future__ import annotations

import numpy as np
import pytest

from scripts import analyze_em_k1_score_transfer_factorial as analyzer


def test_builds_fixed_two_by_two_score_transfer_bases() -> None:
    postoptics = np.asarray([2 + 3j, -1 + 0.5j])
    relion_pixel = np.asarray([4 + 6j, -2 + 1j])
    relion_corr = np.asarray([5.0, 7.0])
    recovar_ctf = np.asarray([-0.5, 0.25])
    recovar_ctf2 = np.asarray([0.2, 0.4])
    half_weights = np.asarray([1.0, 2.0])
    size = 4
    got = analyzer.score_transfer_factorial_bases(
        relion_postoptics_native=postoptics,
        relion_pixel_corrected_native=relion_pixel,
        relion_corr_img=relion_corr,
        recovar_ctf=recovar_ctf,
        recovar_ctf2_data=recovar_ctf2,
        half_weights=half_weights,
        full_image_size=size,
    )
    recovar_pixel = -1.0 / recovar_ctf
    recovar_corr = size**4 * half_weights * recovar_ctf2

    def expected(image: np.ndarray, correction: np.ndarray) -> np.ndarray:
        return -image * correction / (size**2 * half_weights)

    assert np.allclose(
        got["actual_relion"],
        expected(relion_pixel, relion_corr),
    )
    assert np.allclose(
        got["recovar_pixel_correction_only"],
        expected(postoptics * recovar_pixel, relion_corr),
    )
    assert np.allclose(
        got["recovar_corr_img_only"],
        expected(relion_pixel, recovar_corr),
    )
    assert np.allclose(
        got["recovar_pixel_and_corr_img"],
        expected(postoptics * recovar_pixel, recovar_corr),
    )


def test_infers_fixed_two_by_two_bases_without_preprocess_capture() -> None:
    relion_pixel = np.asarray([4 + 6j, -2 + 1j])
    relion_corr = np.asarray([5.0, 7.0])
    recovar_pixel = np.asarray([2 - 1j, -3 + 0.25j])
    recovar_ctf2 = np.asarray([0.2, 0.4])
    half_weights = np.asarray([1.0, 2.0])
    size = 4
    recovar_corr = size**4 * half_weights * recovar_ctf2

    def expected(image: np.ndarray, correction: np.ndarray) -> np.ndarray:
        return -image * correction / (size**2 * half_weights)

    recovar_base = expected(recovar_pixel, recovar_corr)
    got = analyzer.inferred_score_transfer_factorial_bases(
        relion_pixel_corrected_native=relion_pixel,
        relion_corr_img=relion_corr,
        recovar_base_corrected=recovar_base,
        recovar_ctf2_data=recovar_ctf2,
        half_weights=half_weights,
        full_image_size=size,
    )
    assert np.allclose(got["actual_relion"], expected(relion_pixel, relion_corr))
    assert np.allclose(
        got["recovar_pixel_correction_only"],
        expected(recovar_pixel, relion_corr),
    )
    assert np.allclose(
        got["recovar_corr_img_only"],
        expected(relion_pixel, recovar_corr),
    )
    assert np.allclose(got["recovar_pixel_and_corr_img"], recovar_base)


def test_inferred_factorial_rejects_zero_recovar_corr_img() -> None:
    with pytest.raises(ValueError, match="corr_img must be finite and nonzero"):
        analyzer.inferred_score_transfer_factorial_bases(
            relion_pixel_corrected_native=np.ones(2),
            relion_corr_img=np.ones(2),
            recovar_base_corrected=np.ones(2),
            recovar_ctf2_data=np.asarray([1.0, 0.0]),
            half_weights=np.ones(2),
            full_image_size=4,
        )


def test_rejects_zero_ctf_in_fixed_cohort() -> None:
    with pytest.raises(ValueError, match="zero-CTF score pixel"):
        analyzer.score_transfer_factorial_bases(
            relion_postoptics_native=np.ones(2),
            relion_pixel_corrected_native=np.ones(2),
            relion_corr_img=np.ones(2),
            recovar_ctf=np.asarray([1.0, 0.0]),
            recovar_ctf2_data=np.ones(2),
            half_weights=np.ones(2),
            full_image_size=4,
        )


def test_classifies_corr_img_specific_fixed_pattern() -> None:
    dominated = {
        "actual_relion": 14,
        "recovar_pixel_correction_only": 14,
        "recovar_corr_img_only": 0,
        "recovar_pixel_and_corr_img": 0,
    }
    assert (
        analyzer.classify_score_transfer_factorial(
            qualified=True,
            dominated=dominated,
            expected_particles=14,
        )
        == analyzer.CLASSIFICATION
    )


def test_classification_fails_closed_on_unqualified_inputs() -> None:
    dominated = {
        "actual_relion": 14,
        "recovar_pixel_correction_only": 14,
        "recovar_corr_img_only": 0,
        "recovar_pixel_and_corr_img": 0,
    }
    assert (
        analyzer.classify_score_transfer_factorial(
            qualified=False,
            dominated=dominated,
            expected_particles=14,
        )
        == "score_transfer_factorial_inputs_not_qualified"
    )


def test_classification_reports_mixed_pattern() -> None:
    dominated = {
        "actual_relion": 14,
        "recovar_pixel_correction_only": 3,
        "recovar_corr_img_only": 1,
        "recovar_pixel_and_corr_img": 0,
    }
    assert (
        analyzer.classify_score_transfer_factorial(
            qualified=True,
            dominated=dominated,
            expected_particles=14,
        )
        == "raw_coarse_residual_has_mixed_pixel_correction_corr_img_effect"
    )
