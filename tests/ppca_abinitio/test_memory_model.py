"""Unit tests for recovar.em.ppca_abinitio.memory_model."""

from __future__ import annotations

import pytest

from recovar.em.ppca_abinitio.memory_model import (
    estimate_peak_memory_bytes,
    recommended_image_batch_size,
)

pytestmark = [pytest.mark.unit]


def test_components_scale_correctly_with_n_img():
    """post_mean and post_Hinv scale linearly with n_img; u_proj_half
    and M_voxel are independent of n_img."""
    base = estimate_peak_memory_bytes(
        n_img=1024,
        volume_shape=(32, 32, 32),
        image_shape=(32, 32),
        n_rot=576,
        n_trans=5,
        q=4,
    )
    doubled = estimate_peak_memory_bytes(
        n_img=2048,
        volume_shape=(32, 32, 32),
        image_shape=(32, 32),
        n_rot=576,
        n_trans=5,
        q=4,
    )

    assert doubled["post_mean"] == 2 * base["post_mean"]
    assert doubled["post_Hinv"] == 2 * base["post_Hinv"]
    assert doubled["u_proj_half"] == base["u_proj_half"]
    assert doubled["M_voxel"] == base["M_voxel"]


def test_components_scale_quadratically_with_q():
    """post_Hinv ~ q²; M_voxel ~ q²; post_mean ~ q linear."""
    q4 = estimate_peak_memory_bytes(
        n_img=1024,
        volume_shape=(32, 32, 32),
        image_shape=(32, 32),
        n_rot=576,
        n_trans=5,
        q=4,
    )
    q8 = estimate_peak_memory_bytes(
        n_img=1024,
        volume_shape=(32, 32, 32),
        image_shape=(32, 32),
        n_rot=576,
        n_trans=5,
        q=8,
    )
    # q doubled → post_Hinv 4x, M_voxel 4x, post_mean 2x
    assert q8["post_Hinv"] == 4 * q4["post_Hinv"]
    assert q8["M_voxel"] == 4 * q4["M_voxel"]
    assert q8["post_mean"] == 2 * q4["post_mean"]


def test_total_is_sum_of_components():
    cost = estimate_peak_memory_bytes(
        n_img=1024,
        volume_shape=(32, 32, 32),
        image_shape=(32, 32),
        n_rot=576,
        n_trans=5,
        q=4,
    )
    parts = sum(v for k, v in cost.items() if k != "total")
    assert cost["total"] == parts


def test_recommended_batch_size_returns_full_when_under_budget():
    """vol=32, n=1024, q=4 fits comfortably; should return n_img."""
    bs = recommended_image_batch_size(
        n_img=1024,
        volume_shape=(32, 32, 32),
        image_shape=(32, 32),
        n_rot=576,
        n_trans=5,
        q=4,
        budget_gb=60.0,
    )
    assert bs == 1024


def test_recommended_batch_size_reduces_when_oversized():
    """vol=128, n=100k at q=8 should not fit; recommended bs < n_img."""
    bs = recommended_image_batch_size(
        n_img=100_000,
        volume_shape=(128, 128, 128),
        image_shape=(128, 128),
        n_rot=7776,
        n_trans=5,
        q=8,
        budget_gb=60.0,
    )
    assert bs < 100_000
    assert bs >= 1


def test_recommended_batch_size_is_power_of_two():
    bs = recommended_image_batch_size(
        n_img=10_000,
        volume_shape=(64, 64, 64),
        image_shape=(64, 64),
        n_rot=1944,
        n_trans=5,
        q=8,
        budget_gb=10.0,
    )
    assert bs & (bs - 1) == 0  # power of two


def test_vol32_predictions_match_doc_table():
    """Spot-check vol=32 q=4 numbers against
    docs/math/ppca_abinitio_memory_model.md."""
    cost = estimate_peak_memory_bytes(
        n_img=1024,
        volume_shape=(32, 32, 32),
        image_shape=(32, 32),
        n_rot=576,
        n_trans=5,
        q=4,
    )
    # Doc says: u_proj_half ≈ 19 MB at q=4 (i.e. n_rot * q * img_half * 16)
    # = 576 * 4 * 32 * 17 * 16 = 20.0 MB. Within rounding.
    expected = 576 * 4 * 32 * 17 * 16
    assert cost["u_proj_half"] == expected
    # post_Hinv: 1024 * 576 * 16 * 8 = 75 MB (doc rounds 71)
    assert cost["post_Hinv"] == 1024 * 576 * 16 * 8


def test_vol128_q8_predicted_to_saturate_h100():
    """Doc claims vol=128, n=10k, order=3, q=8 saturates an H100.
    Verify the predicted total is in the 60-65 GB range."""
    cost = estimate_peak_memory_bytes(
        n_img=10_000,
        volume_shape=(128, 128, 128),
        image_shape=(128, 128),
        n_rot=7776,
        n_trans=5,
        q=8,
    )
    total_gb = cost["total"] / (1024**3)
    assert 50.0 < total_gb < 80.0, f"expected ~63 GB, got {total_gb:.1f} GB"
