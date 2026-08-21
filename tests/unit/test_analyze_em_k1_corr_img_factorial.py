from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_em_k1_corr_img_factorial as analyzer


def test_resolves_exact_stack_group_scales() -> None:
    model = {
        "model_general": {
            "rlnOriginalImageSize": 8,
            "rlnNrClasses": 1,
            "rlnNrBodies": 1,
            "rlnNrGroups": 2,
        },
        "model_groups": pd.DataFrame(
            {
                "rlnGroupNumber": [1, 2],
                "rlnGroupScaleCorrection": [0.75, 1.25],
            }
        ),
    }
    data = {
        "particles": pd.DataFrame(
            {
                "rlnImageName": ["2@stack.mrcs", "1@stack.mrcs"],
                "rlnGroupNumber": [2, 1],
                "rlnRandomSubset": [1, 2],
            }
        )
    }
    assert analyzer._scale_by_stack(model, data, full_image_size=8) == {
        1: (0.75, 2),
        2: (1.25, 1),
    }


def test_rejects_duplicate_stack_identity() -> None:
    model = {
        "model_general": {
            "rlnOriginalImageSize": 8,
            "rlnNrClasses": 1,
            "rlnNrBodies": 1,
            "rlnNrGroups": 1,
        },
        "model_groups": pd.DataFrame(
            {
                "rlnGroupNumber": [1],
                "rlnGroupScaleCorrection": [1.0],
            }
        ),
    }
    data = {
        "particles": pd.DataFrame(
            {
                "rlnImageName": ["1@a.mrcs", "1@a.mrcs"],
                "rlnGroupNumber": [1, 1],
                "rlnRandomSubset": [1, 1],
            }
        )
    }
    with pytest.raises(ValueError, match="duplicate RELION stack"):
        analyzer._scale_by_stack(model, data, full_image_size=8)


def test_builds_inverse_noise_ctf_scale_factorial() -> None:
    relion_ctf = np.asarray([0.5, -0.25])
    recovar_ctf = np.asarray([0.4, -0.2])
    relion_inverse_noise = np.asarray([2.0, 3.0])
    recovar_inverse_noise = np.asarray([5.0, 7.0])
    relion_corr = relion_inverse_noise * relion_ctf**2
    recovar_corr = recovar_inverse_noise * recovar_ctf**2
    values, got_relion_noise, got_recovar_noise = (
        analyzer.corr_img_factorial_values(
            relion_corr_img=relion_corr,
            relion_effective_ctf=relion_ctf,
            recovar_corr_img=recovar_corr,
            recovar_effective_ctf=recovar_ctf,
        )
    )
    assert np.allclose(got_relion_noise, relion_inverse_noise)
    assert np.allclose(got_recovar_noise, recovar_inverse_noise)
    assert np.allclose(values["actual_relion"], relion_corr)
    assert np.allclose(
        values["recovar_inverse_noise_only"],
        recovar_inverse_noise * relion_ctf**2,
    )
    assert np.allclose(
        values["recovar_ctf_scale_squared_only"],
        relion_inverse_noise * recovar_ctf**2,
    )
    assert np.allclose(
        values["recovar_inverse_noise_and_ctf_scale_squared"],
        recovar_corr,
    )


def test_rejects_zero_effective_ctf() -> None:
    with pytest.raises(ValueError, match="zero effective-CTF"):
        analyzer.corr_img_factorial_values(
            relion_corr_img=np.ones(2),
            relion_effective_ctf=np.asarray([1.0, 0.0]),
            recovar_corr_img=np.ones(2),
            recovar_effective_ctf=np.ones(2),
        )


def test_classifies_inverse_noise_specific_pattern() -> None:
    dominated = {
        "actual_relion": 14,
        "recovar_inverse_noise_only": 0,
        "recovar_ctf_scale_squared_only": 14,
        "recovar_inverse_noise_and_ctf_scale_squared": 0,
    }
    assert (
        analyzer.classify_corr_img_factorial(
            qualified=True,
            dominated=dominated,
            expected_particles=14,
        )
        == analyzer.CLASSIFICATION
    )


def test_classification_fails_closed() -> None:
    assert (
        analyzer.classify_corr_img_factorial(
            qualified=False,
            dominated={},
            expected_particles=14,
        )
        == "corr_img_factorial_inputs_not_qualified"
    )
