"""Exact coarse assembly preserves both precision and physical batch padding."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers import significance, sparse_pass2_bucketed

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("use_float64", [False, True])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_exact_coarse_assembly_precision_and_padding(monkeypatch, use_float64, batch_size):
    real_dtype = jnp.float64 if use_float64 else jnp.float32
    complex_dtype = jnp.complex128 if use_float64 else jnp.complex64
    ctf = np.asarray([[1.125] * 12, [1.375] * 12], dtype=np.float64)
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_exact_ctf_half_from_source_star_host",
        lambda *_args: ctf,
    )
    seen = []

    def translate(images, angles, indices, image_shape):
        assert images.dtype == complex_dtype
        assert angles.dtype == real_dtype
        assert image_shape == (4, 4)
        np.testing.assert_array_equal(indices, np.arange(12))
        seen.append(images)
        return jnp.broadcast_to(images[:, None, :], (batch_size, 2, 12))

    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f64" if use_float64 else "relion_translate_score_f32",
        translate,
    )
    source_images = jnp.asarray(np.arange(batch_size * 12).reshape(batch_size, 12) + 1j, dtype=complex_dtype)
    result = significance._assemble_relion_exact_coarse_gaussian_operands(
        object(),
        source_images,
        np.arange(2),
        batch_scale_np=np.ones(batch_size, dtype=np.float64),
        actual_batch_size=2,
        batch_size=batch_size,
        score_indices=jnp.arange(12, dtype=jnp.int32),
        score_active_mask=jnp.ones(12, dtype=bool),
        translations_source=np.zeros((2, 2), dtype=np.float64),
        image_shape=(4, 4),
        noise_variance_half=jnp.full(12, 2.0, dtype=jnp.float64),
        scale_corrections_enabled=True,
        half_weights=jnp.ones(12, dtype=real_dtype),
        powerclass=lambda images, **_kwargs: jnp.zeros(images.shape[0], dtype=real_dtype),
        current_size=4,
        use_float64_scoring=use_float64,
    )
    expected_ctf = ctf if batch_size == 2 else np.concatenate((ctf, ctf[:1]), axis=0)
    correction = sparse_pass2_bucketed._relion_cuda_pixel_correction_from_rfloat_ctf(
        jnp.ones((batch_size, 1), dtype=real_dtype),
        jnp.asarray(expected_ctf),
        output_dtype=real_dtype,
    )
    np.testing.assert_array_equal(result.unshifted_corrected, source_images * correction)
    np.testing.assert_array_equal(result.shifted_corrected[:, 0], result.unshifted_corrected)
    np.testing.assert_array_equal(result.shifted_corrected[:, 1], result.unshifted_corrected)
    assert len(seen) == 1
    assert result.pixel_weight.dtype == real_dtype
    assert result.unshifted_corrected.dtype == complex_dtype
    assert result.pixel_weight.shape == (batch_size, 12)
    assert bool(jnp.all(jnp.isfinite(result.pixel_weight)))
