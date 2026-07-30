from __future__ import annotations

import numpy as np
import pytest

from scripts import analyze_em_k1_corr_img_conditioning as analyzer


def test_conditioned_factorial_retains_actual_values_outside_mask() -> None:
    relion_ctf = np.asarray([0.5, 0.005, -0.25])
    recovar_ctf = np.asarray([0.4, 0.02, -0.2])
    relion_inverse_noise = np.asarray([2.0, 3.0, 4.0])
    recovar_inverse_noise = np.asarray([5.0, 6.0, 7.0])
    relion_corr = relion_inverse_noise * relion_ctf**2
    recovar_corr = recovar_inverse_noise * recovar_ctf**2

    values, valid = analyzer.conditioned_corr_img_factorial_values(
        relion_corr_img=relion_corr,
        relion_effective_ctf=relion_ctf,
        recovar_corr_img=recovar_corr,
        recovar_effective_ctf=recovar_ctf,
        effective_ctf_threshold=0.01,
    )

    assert np.array_equal(valid, np.asarray([True, False, True]))
    assert np.allclose(values["actual_relion"], relion_corr)
    assert np.allclose(
        values["recovar_inverse_noise_only"],
        np.asarray(
            [
                recovar_inverse_noise[0] * relion_ctf[0] ** 2,
                relion_corr[1],
                recovar_inverse_noise[2] * relion_ctf[2] ** 2,
            ]
        ),
    )
    assert np.allclose(
        values["recovar_ctf_scale_squared_only"],
        np.asarray(
            [
                relion_inverse_noise[0] * recovar_ctf[0] ** 2,
                relion_corr[1],
                relion_inverse_noise[2] * recovar_ctf[2] ** 2,
            ]
        ),
    )
    assert np.allclose(
        values["recovar_inverse_noise_and_ctf_scale_squared"],
        np.asarray([recovar_corr[0], relion_corr[1], recovar_corr[2]]),
    )


def test_conditioned_factorial_rejects_threshold_excluding_all_pixels() -> None:
    with pytest.raises(ValueError, match="excludes every pixel"):
        analyzer.conditioned_corr_img_factorial_values(
            relion_corr_img=np.ones(2),
            relion_effective_ctf=np.asarray([0.01, 0.02]),
            recovar_corr_img=np.ones(2),
            recovar_effective_ctf=np.asarray([0.01, 0.02]),
            effective_ctf_threshold=0.02,
        )


def test_conditioned_factorial_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        analyzer.conditioned_corr_img_factorial_values(
            relion_corr_img=np.ones(2),
            relion_effective_ctf=np.ones(2),
            recovar_corr_img=np.ones(2),
            recovar_effective_ctf=np.ones(2),
            effective_ctf_threshold=-1.0,
        )


def test_classifies_fixed_threshold_stability() -> None:
    expected = {
        "actual_relion": 14,
        "recovar_inverse_noise_only": 0,
        "recovar_ctf_scale_squared_only": 14,
        "recovar_inverse_noise_and_ctf_scale_squared": 0,
    }
    dominated = {
        analyzer._threshold_label(threshold): expected.copy()
        for threshold in analyzer.FIXED_EFFECTIVE_CTF_THRESHOLDS
    }
    assert (
        analyzer.classify_conditioning_audit(
            qualified=True,
            dominated_by_threshold=dominated,
            thresholds=analyzer.FIXED_EFFECTIVE_CTF_THRESHOLDS,
            expected_particles=14,
        )
        == analyzer.CLASSIFICATION
    )


def test_classification_fails_closed_on_one_threshold() -> None:
    expected = {
        "actual_relion": 14,
        "recovar_inverse_noise_only": 0,
        "recovar_ctf_scale_squared_only": 14,
        "recovar_inverse_noise_and_ctf_scale_squared": 0,
    }
    dominated = {
        analyzer._threshold_label(threshold): expected.copy()
        for threshold in analyzer.FIXED_EFFECTIVE_CTF_THRESHOLDS
    }
    dominated["0.01"]["recovar_inverse_noise_only"] = 1
    assert (
        analyzer.classify_conditioning_audit(
            qualified=True,
            dominated_by_threshold=dominated,
            thresholds=analyzer.FIXED_EFFECTIVE_CTF_THRESHOLDS,
            expected_particles=14,
        )
        == "inverse_noise_attribution_is_not_stable_across_ctf_thresholds"
    )
