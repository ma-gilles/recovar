"""Exact published CUDA outputs after an alternative FP64 certificate center."""

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import significance
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import plan_coarse_gemm_certificate_topology

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.mark.parametrize("dyadic", [False, True])
@pytest.mark.parametrize("actual_images", [2, 3])
def test_real_cross_preserves_selected_exact_scores(dyadic, actual_images, custom_cuda_lib, gpu_device):
    rng = np.random.default_rng(593)
    batch, rotations, translations, pixels = 3, 256, 3, 33

    def values(shape):
        if dyadic:
            return ((rng.integers(-4, 5, shape) + 1j * rng.integers(-4, 5, shape)) / 16).astype(np.complex64)
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64) / np.float32(8)

    reference = values((rotations, pixels)) + np.complex64(10 + 10j)
    reference[:2] = values((2, pixels))
    shifted = values((batch, translations, pixels))
    weight = np.ones((batch, pixels), np.float32)
    initial = np.full(batch, 0.5, np.float32)
    if actual_images < batch:
        shifted[actual_images:] = np.nan
        weight[actual_images:] = np.inf
        initial[actual_images:] = -1
    topology = plan_coarse_gemm_certificate_topology(
        np.arange(pixels, dtype=np.int32), compact_pixel_count=pixels, translation_count=translations
    )
    kwargs = dict(
        topology=topology, actual_image_count=actual_images, class_log_prior=np.float32(-0.25),
        rotation_log_prior=np.linspace(-0.2, 0.1, rotations, dtype=np.float32),
        translation_log_prior=np.linspace(-0.1, 0.1, batch * translations, dtype=np.float32).reshape(batch, translations),
        certificate_chunk_rows=64, block_capacity=4, compact_posterior=True, capture_selected_diff2=True,
    )
    results = [significance._compute_coarse_gaussian_gemm_hybrid_batch(
        reference[None], shifted, weight, initial, **kwargs, real_cross=enabled
    ) for enabled in (False, True)]
    for result in results:
        assert result.used_selected_rescore
        assert result.fallback_reason is None
        assert result.selection.eligible
        assert np.all(result.selection.block_count[:actual_images] == 1)
    control, candidate = results
    for name in ("block_ids", "block_count", "posterior_block_count", "raw_max_block_count"):
        np.testing.assert_array_equal(getattr(control.selection, name), getattr(candidate.selection, name))
    for left, right in zip(control.compact_scores, candidate.compact_scores, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(control.diagnostic_selected_diff2, candidate.diagnostic_selected_diff2)
