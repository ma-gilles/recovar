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


def test_shell_partition_swaps_only_predeclared_shell_cohort() -> None:
    relion_ctf = np.asarray([0.5, 0.4, 0.3, 0.2, 0.1, 0.005])
    recovar_ctf = np.asarray([0.45, 0.35, 0.25, 0.15, 0.05, 0.02])
    shells = np.asarray([1, 2, 4, 5, 8, 3])
    relion_inverse_noise = np.asarray([2, 3, 4, 5, 6, 7], dtype=float)
    recovar_inverse_noise = np.asarray([12, 13, 14, 15, 16, 17], dtype=float)
    relion_corr = relion_inverse_noise * relion_ctf**2
    recovar_corr = recovar_inverse_noise * recovar_ctf**2

    values, valid = analyzer.inverse_noise_shell_partition_values(
        relion_corr_img=relion_corr,
        relion_effective_ctf=relion_ctf,
        recovar_corr_img=recovar_corr,
        recovar_effective_ctf=recovar_ctf,
        shell_indices=shells,
        effective_ctf_threshold=0.01,
    )

    assert np.array_equal(
        valid,
        np.asarray([True, True, True, True, True, False]),
    )
    assert np.allclose(values["actual_relion"], relion_corr)
    assert np.allclose(
        values["recovar_all"],
        np.concatenate([recovar_corr[:5], relion_corr[5:]]),
    )
    assert np.allclose(
        values["relion_inverse_noise_all"][:5],
        relion_inverse_noise[:5] * recovar_ctf[:5] ** 2,
    )
    assert values["relion_inverse_noise_all"][5] == relion_corr[5]
    expected_low = np.where(
        np.isin(shells[:5], [1, 2, 3, 4]),
        relion_inverse_noise[:5],
        recovar_inverse_noise[:5],
    )
    assert np.allclose(
        values["relion_inverse_noise_shells_1_through_4"][:5],
        expected_low * recovar_ctf[:5] ** 2,
    )
    expected_high = np.where(
        shells[:5] >= 5,
        relion_inverse_noise[:5],
        recovar_inverse_noise[:5],
    )
    assert np.allclose(
        values["relion_inverse_noise_shells_5_plus"][:5],
        expected_high * recovar_ctf[:5] ** 2,
    )


def test_shell_partition_rejects_origin() -> None:
    with pytest.raises(ValueError, match="excluded origin"):
        analyzer.inverse_noise_shell_partition_values(
            relion_corr_img=np.ones(2),
            relion_effective_ctf=np.ones(2),
            recovar_corr_img=np.ones(2),
            recovar_effective_ctf=np.ones(2),
            shell_indices=np.asarray([0, 1]),
            effective_ctf_threshold=0.01,
        )


def test_classifies_fixed_decimal_shell_partition() -> None:
    dominated = {
        "actual_relion": 14,
        "recovar_all": 0,
        "relion_inverse_noise_all": 14,
        "relion_inverse_noise_shells_1_through_4": 14,
        "relion_inverse_noise_shells_5_plus": 0,
    }
    assert (
        analyzer.classify_shell_partition(
            qualified=True,
            dominated=dominated,
            expected_particles=14,
        )
        == analyzer.SHELL_PARTITION_CLASSIFICATION
    )


def test_shell_partition_classification_fails_closed() -> None:
    dominated = {
        "actual_relion": 14,
        "recovar_all": 0,
        "relion_inverse_noise_all": 14,
        "relion_inverse_noise_shells_1_through_4": 13,
        "relion_inverse_noise_shells_5_plus": 0,
    }
    assert (
        analyzer.classify_shell_partition(
            qualified=True,
            dominated=dominated,
            expected_particles=14,
        )
        == "inverse_noise_residual_is_not_confined_to_fixed_decimal_shells"
    )


def test_validates_star_precision_partition(tmp_path) -> None:
    model_star = tmp_path / "model.star"
    model_star.write_text(
        """
data_model_optics_group_1

loop_
_rlnSpectralIndex #1
_rlnResolution #2
_rlnSigma2Noise #3
0 0.0 0.011668
1 0.1 0.005805
2 0.2 0.003661
3 0.3 0.002303
4 0.4 0.001193
5 0.5 7.535662e-04

data_model_pdf_orient_class_1
""".lstrip()
    )

    partition = analyzer._validate_star_precision_partition(model_star)

    assert partition["fixed_decimal_shells"] == [1, 2, 3, 4]
    assert partition["first_scientific_shell"] == 5
    assert partition["raw_sigma2_noise_tokens"]["4"] == "0.001193"
    assert partition["raw_sigma2_noise_tokens"]["5"] == "7.535662e-04"


def test_star_precision_partition_rejects_wrong_shell5_format(tmp_path) -> None:
    model_star = tmp_path / "model.star"
    model_star.write_text(
        """
data_model_optics_group_1
_rlnSigma2Noise #3
0 0.0 0.011668
1 0.1 0.005805
2 0.2 0.003661
3 0.3 0.002303
4 0.4 0.001193
5 0.5 0.000754
""".lstrip()
    )

    with pytest.raises(ValueError, match="shell 5"):
        analyzer._validate_star_precision_partition(model_star)
