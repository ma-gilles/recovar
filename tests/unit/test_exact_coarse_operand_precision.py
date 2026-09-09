"""Exact coarse assembly preserves both precision and physical batch padding."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers import significance, sparse_pass2_bucketed

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("use_float64", [False, True])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("scale_enabled", [False, True])
def test_exact_coarse_assembly_precision_and_padding(
    monkeypatch,
    use_float64,
    batch_size,
    compact,
    scale_enabled,
):
    real_dtype = jnp.float64 if use_float64 else jnp.float32
    complex_dtype = jnp.complex128 if use_float64 else jnp.complex64
    score_indices = np.asarray([11, 0, 5, 5, 1, 2, 3, 4], dtype=np.int32) if compact else np.arange(12, dtype=np.int32)
    ctf = np.asarray([[1.125] * 12, [1.375] * 12], dtype=np.float64)
    scales = np.ones(batch_size, dtype=np.float64)
    active_mask = np.ones(len(score_indices), dtype=bool)
    if compact:
        ctf[:, :6] = np.asarray([-1.125, -1e-9, 0.0, 1e-9, -1e-8, 1.0000000001])
        ctf[1] *= 1.25
        scales[:2] = [0.875, 1.25]
        active_mask[3] = False  # Repeated score position used as inactive padding.
    scale_operand = jnp.asarray(scales[:, None], dtype=real_dtype)
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_exact_ctf_half_from_source_star_host",
        lambda *_args, pixel_indices: ctf[:, pixel_indices],
    )
    seen = []

    def translate(images, angles, indices, image_shape):
        assert images.dtype == complex_dtype
        assert angles.dtype == real_dtype
        assert image_shape == (4, 4)
        np.testing.assert_array_equal(indices, score_indices)
        seen.append(images)
        return jnp.broadcast_to(images[:, None, :], (batch_size, 2, len(score_indices)))

    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f64" if use_float64 else "relion_translate_score_f32",
        translate,
    )
    source_images = jnp.asarray(np.arange(batch_size * 12).reshape(batch_size, 12) + 1j, dtype=complex_dtype)

    def full_powerclass(images, **_kwargs):
        assert images is source_images
        assert images.shape == (batch_size, 12)
        return jnp.zeros(images.shape[0], dtype=real_dtype)

    result = significance._assemble_relion_exact_coarse_gaussian_operands(
        object(),
        source_images,
        np.arange(2),
        batch_scale_np=scales,
        actual_batch_size=2,
        batch_size=batch_size,
        score_indices=jnp.asarray(score_indices),
        score_indices_np=score_indices,
        score_active_mask=jnp.asarray(active_mask),
        translations_source=np.zeros((2, 2), dtype=np.float64),
        image_shape=(4, 4),
        noise_variance_half=jnp.full(12, 2.0, dtype=jnp.float64),
        scale_corrections_enabled=scale_enabled,
        half_weights=jnp.ones(12, dtype=real_dtype),
        powerclass=full_powerclass,
        current_size=4,
        use_float64_scoring=use_float64,
    )
    expected_ctf = ctf if batch_size == 2 else np.concatenate((ctf, ctf[:1]), axis=0)
    correction = sparse_pass2_bucketed._relion_cuda_pixel_correction_from_rfloat_ctf(
        scale_operand,
        jnp.asarray(expected_ctf),
        output_dtype=real_dtype,
    )
    np.testing.assert_array_equal(
        result.unshifted_corrected,
        jnp.where(active_mask[None, :], (source_images * correction)[:, score_indices], 0.0),
    )
    np.testing.assert_array_equal(result.shifted_corrected[:, 0], result.unshifted_corrected)
    np.testing.assert_array_equal(result.shifted_corrected[:, 1], result.unshifted_corrected)
    assert len(seen) == 1
    assert result.pixel_weight.dtype == real_dtype
    assert result.unshifted_corrected.dtype == complex_dtype
    if use_float64:
        full_corr = sparse_pass2_bucketed._relion_cuda_corr_img_from_rfloat_ctf(
            jnp.full((1, 12), 0.5, dtype=jnp.float64),
            jnp.asarray(expected_ctf),
            scale_operand if scale_enabled else None,
            output_dtype=real_dtype,
        )
    else:
        full_corr = sparse_pass2_bucketed._relion_cuda_corr_img_from_native_noise_variance(
            jnp.full((1, 12), 2.0, dtype=jnp.float64),
            jnp.asarray(expected_ctf),
            (4, 4),
            scale_operand if scale_enabled else None,
        )
    np.testing.assert_array_equal(
        result.pixel_weight,
        jnp.where(active_mask[None, :], full_corr[:, score_indices], 0.0),
    )
    assert result.pixel_weight.shape == (batch_size, len(score_indices))
    assert bool(jnp.all(jnp.isfinite(result.pixel_weight)))
