from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_em_k1_postoptics_score_transfer as analyzer


def _relion_layout(centered: np.ndarray, full_size: int) -> np.ndarray:
    return np.fft.ifftshift(
        centered / np.float64(full_size**2),
        axes=0,
    )


def test_recovar_weighted_base_uses_fixed_ctf_noise_transfer() -> None:
    full_size = 4
    centered = (
        np.arange(12, dtype=np.float64).reshape(4, 3)
        + np.complex128(2j)
    )
    relion = _relion_layout(centered, full_size)
    window_indices = np.arange(centered.size, dtype=np.int64)
    ctf = np.linspace(-0.9, 0.8, centered.size, dtype=np.float64)
    ctf2_data = np.linspace(0.1, 0.3, centered.size, dtype=np.float64)
    got = analyzer.recovar_weighted_base_from_relion_postoptics(
        relion,
        window_indices=window_indices,
        ctf_half=ctf,
        ctf2_data=ctf2_data,
        full_image_size=full_size,
        current_size=full_size,
    )
    assert np.allclose(got, centered.reshape(-1) * ctf2_data / ctf)


def test_zero_ctf_with_zero_weight_maps_to_zero() -> None:
    full_size = 4
    centered = np.ones((4, 3), dtype=np.complex128)
    ctf = np.ones(centered.size, dtype=np.float64)
    ctf[5] = 0.0
    ctf2_data = np.ones(centered.size, dtype=np.float64)
    ctf2_data[5] = 0.0
    got = analyzer.recovar_weighted_base_from_relion_postoptics(
        _relion_layout(centered, full_size),
        window_indices=np.arange(centered.size),
        ctf_half=ctf,
        ctf2_data=ctf2_data,
        full_image_size=full_size,
        current_size=full_size,
    )
    assert got[5] == 0.0


def test_zero_ctf_rejects_nonzero_score_weight() -> None:
    full_size = 4
    ctf = np.ones(12, dtype=np.float64)
    ctf[5] = 0.0
    with pytest.raises(ValueError, match="nonzero at a zero-CTF"):
        analyzer.recovar_weighted_base_from_relion_postoptics(
            np.ones((4, 3), dtype=np.complex128),
            window_indices=np.arange(12),
            ctf_half=ctf,
            ctf2_data=np.ones(12, dtype=np.float64),
            full_image_size=full_size,
            current_size=full_size,
        )


def test_classification_requires_transfer_specific_fixed_pattern() -> None:
    assert (
        analyzer.classify_postoptics_transfer(
            qualified=True,
            actual_live_dominated=14,
            hybrid_live_dominated=0,
            hybrid_within_material_threshold=14,
            expected_particles=14,
        )
        == "raw_coarse_residual_is_postoptics_score_weight_transfer_"
        "dominated_not_preprocessing"
    )


def test_classification_rejects_unqualified_inputs() -> None:
    assert (
        analyzer.classify_postoptics_transfer(
            qualified=False,
            actual_live_dominated=14,
            hybrid_live_dominated=0,
            hybrid_within_material_threshold=14,
            expected_particles=14,
        )
        == "postoptics_inputs_not_qualified"
    )


def test_capture_qualification_does_not_gate_on_target_residual() -> None:
    assert analyzer.capture_inputs_qualified(
        preprocess_validation={"status": "pass"},
        operand_validation={"status": "pass"},
    )
    assert not analyzer.capture_inputs_qualified(
        preprocess_validation={"status": "rejected"},
        operand_validation={"status": "pass"},
    )
    assert not analyzer.capture_inputs_qualified(
        preprocess_validation={"status": "pass"},
        operand_validation={"status": "rejected"},
    )


def test_load_ctf_half_rejects_wrong_image_size(tmp_path: Path) -> None:
    path = tmp_path / "ctf.pkl"
    values = np.asarray(
        [[8, 1.5, 10000, 11000, 0, 300, 2.7, 0.1, 0]],
        dtype=np.float32,
    )
    with path.open("wb") as stream:
        pickle.dump(values, stream)
    with pytest.raises(ValueError, match="image size differs"):
        analyzer._load_ctf_half(path, full_image_size=4)


def test_relative_l2_remains_scale_sensitive() -> None:
    target = np.ones(8, dtype=np.float64)
    assert analyzer._relative_l2(2.0 * target, target) == 1.0
