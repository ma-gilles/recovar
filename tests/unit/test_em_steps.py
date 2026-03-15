import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em import e_step, m_step

pytestmark = pytest.mark.unit


def test_compute_probability_from_residual_normal_squared_one_image_normalizes_rows():
    residual = jnp.array([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]], dtype=jnp.float32)
    probs = np.asarray(e_step.compute_probability_from_residual_normal_squared_one_image(residual))

    assert probs.shape == (2, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones((2,), dtype=np.float32), atol=1e-6)
    assert probs[0, 0] > probs[0, 1] > probs[0, 2]
    np.testing.assert_allclose(probs[1], np.ones((3,), dtype=np.float32) / 3.0, atol=1e-6)


def test_compute_probability_from_residual_normal_squared_vmap_normalizes_each_sample():
    residual = jnp.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[3.0, 3.0], [0.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    probs = np.asarray(e_step.compute_probability_from_residual_normal_squared(residual))

    assert probs.shape == residual.shape
    np.testing.assert_allclose(np.sum(probs, axis=2), np.ones((2, 2), dtype=np.float32), atol=1e-6)


def test_compute_residuals_many_poses_fft_branch_uses_translation_indices(monkeypatch):
    def _fake_slice(_volumes, _rotations, _image_shape, _volume_shape, _disc_type):
        return jnp.ones((2, 1, 2, 4), dtype=jnp.complex64)

    def _fake_norm_squared(_projected_volumes, _images, _image_shape):
        template = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
        return jnp.broadcast_to(template, _projected_volumes.shape)

    monkeypatch.setattr(e_step, "batch_vol_rot_slice_volume", _fake_slice)
    monkeypatch.setattr(e_step, "norm_squared_residuals_from_ft", _fake_norm_squared)
    monkeypatch.setattr(
        e_step,
        "translations_to_indices",
        lambda _translations, _image_shape: jnp.array([[1, 3], [0, 2]], dtype=jnp.int32),
    )

    volumes = jnp.zeros((1, 8), dtype=jnp.complex64)
    images = jnp.ones((2, 4), dtype=jnp.complex64)
    rotations = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None, None], (2, 2, 1, 1))
    translations = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    ctf_params = jnp.zeros((2, 9), dtype=jnp.float32)
    noise_variance = jnp.ones((4,), dtype=jnp.float32)

    out = np.asarray(
        e_step.compute_residuals_many_poses.__wrapped__(
            volumes,
            images,
            rotations,
            translations,
            ctf_params,
            noise_variance,
            1.0,
            (2, 2, 2),
            (2, 2),
            "linear_interp",
            lambda params, _shape, _voxel: jnp.ones((params.shape[0], 4), dtype=jnp.complex64),
            translation_fn="fft",
        )
    )

    expected = np.array(
        [
            [[[0.0, -16.0], [0.0, -16.0]]],
            [[[8.0, -8.0], [8.0, -8.0]]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_compute_residuals_many_poses_nofft_branch_uses_translated_images(monkeypatch):
    monkeypatch.setattr(
        e_step,
        "batch_vol_rot_slice_volume",
        lambda *_args, **_kwargs: jnp.zeros((2, 1, 1, 4), dtype=jnp.complex64),
    )
    monkeypatch.setattr(
        e_step.core,
        "batch_trans_translate_images",
        lambda _images, _translations, _image_shape: jnp.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=jnp.complex64,
        ),
    )

    volumes = jnp.zeros((1, 8), dtype=jnp.complex64)
    images = jnp.ones((2, 4), dtype=jnp.complex64)
    rotations = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None, None], (2, 1, 1, 1))
    translations = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    ctf_params = jnp.zeros((2, 9), dtype=jnp.float32)
    noise_variance = jnp.ones((4,), dtype=jnp.float32)

    out = np.asarray(
        e_step.compute_residuals_many_poses.__wrapped__(
            volumes,
            images,
            rotations,
            translations,
            ctf_params,
            noise_variance,
            1.0,
            (2, 2, 2),
            (2, 2),
            "linear_interp",
            lambda params, _shape, _voxel: jnp.ones((params.shape[0], 4), dtype=jnp.complex64),
            translation_fn="nofft",
        )
    )

    assert out.shape == (2, 1, 1, 2)
    np.testing.assert_allclose(out, np.ones((2, 1, 1, 2), dtype=np.float32), atol=1e-6)


def test_sum_up_translate_one_image_fft_accumulates_duplicate_translation_bins(monkeypatch):
    monkeypatch.setattr(
        m_step,
        "translations_to_indices",
        lambda _translations, _shape: jnp.array([0, 2, 0], dtype=jnp.int32),
    )
    monkeypatch.setattr(m_step.fourier_transform_utils, "get_dft2", lambda x: x)

    image = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    probs = jnp.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
        ],
        dtype=jnp.float32,
    )
    out = np.asarray(
        m_step.sum_up_translate_one_image(
            image,
            probs,
            jnp.zeros((3, 2), dtype=jnp.float32),
            (2, 2),
            translation_fn="fft",
        )
    )

    expected_probs = np.zeros((2, 2, 4), dtype=np.float32)
    expected_probs[..., 0] = np.asarray(probs[..., 0] + probs[..., 2])
    expected_probs[..., 2] = np.asarray(probs[..., 1])
    expected = expected_probs * np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_sum_up_translate_one_image_nofft_weighted_sum(monkeypatch):
    monkeypatch.setattr(
        m_step.core,
        "batch_trans_translate_images",
        lambda _image, _translations, _shape: jnp.array(
            [[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]],
            dtype=jnp.float32,
        ),
    )

    probs = jnp.array(
        [
            [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]],
            [[0.5, 0.25, 0.25], [0.0, 0.4, 0.6]],
        ],
        dtype=jnp.float32,
    )
    out = np.asarray(
        m_step.sum_up_translate_one_image(
            jnp.ones((4,), dtype=jnp.float32),
            probs,
            jnp.zeros((3, 2), dtype=jnp.float32),
            (2, 2),
            translation_fn="nofft",
        )
    )

    weighted = np.asarray(probs[..., 0] + 2.0 * probs[..., 1] + 3.0 * probs[..., 2])
    expected = np.repeat(weighted[..., None], 4, axis=-1)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_M_with_precompute_handles_small_rotation_count_without_zero_batch(monkeypatch):
    from recovar import utils as rec_utils

    class _Dataset:
        image_size = 4
        image_shape = (2, 2)
        n_images = 1
        grid_size = 8
        volume_size = 5
        dtype = jnp.float32
        CTF_params = np.zeros((1, 9), dtype=np.float32)
        ctf_evaluator = staticmethod(lambda params, _shape, _voxel: jnp.ones((params.shape[0], 4), dtype=jnp.float32))
        voxel_size = 1.0
        volume_shape = (2, 2, 2)

        class image_stack:
            @staticmethod
            def process_images(batch, apply_image_mask=False):
                _ = apply_image_mask
                return batch

        @staticmethod
        def get_dataset_subset_generator(batch_size, subset_indices):
            assert batch_size >= 1
            _ = subset_indices
            yield jnp.ones((1, 4), dtype=jnp.float32), None, np.array([0], dtype=np.int32)

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda: 1)
    monkeypatch.setattr(rec_utils, "get_image_batch_size", lambda _grid_size, _gpu_memory: 1)
    monkeypatch.setattr(
        rec_utils,
        "index_batch_iter",
        lambda n_units, batch_size: [np.arange(n_units, dtype=np.int32)] if batch_size >= 1 else (_ for _ in ()).throw(AssertionError("batch_size must be >= 1")),
    )
    monkeypatch.setattr(
        m_step,
        "sum_up_images_fixed_rots_eqx",
        lambda _config, *_args, Ft_y=0, Ft_ctf=0, **_kwargs: (Ft_y + jnp.ones_like(Ft_y), Ft_ctf + 2.0 * jnp.ones_like(Ft_ctf)),
    )

    probs = jnp.array([[[0.6, 0.4]]], dtype=jnp.float32)
    rots = jnp.eye(3, dtype=jnp.float32)[None]
    trans = jnp.zeros((2, 2), dtype=jnp.float32)
    noise = jnp.ones((4,), dtype=jnp.float32)

    ft_y, ft_ctf = m_step.M_with_precompute(_Dataset(), probs, rots, trans, noise, "linear_interp")

    np.testing.assert_allclose(np.asarray(ft_y), np.ones((5,), dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(ft_ctf), np.ones((5,), dtype=np.float32) * 2.0, atol=1e-6)


def test_E_with_precompute_rejects_empty_rotations_or_translations():
    ds = type(
        "DS",
        (),
        {
            "image_shape": (2, 2),
            "image_size": 4,
            "n_images": 1,
        },
    )()

    with pytest.raises(ValueError, match="at least one rotation"):
        e_step.E_with_precompute(
            ds,
            volume=jnp.zeros((8,), dtype=jnp.complex64),
            rotations=jnp.zeros((0, 3, 3), dtype=jnp.float32),
            translations=jnp.zeros((1, 2), dtype=jnp.float32),
            noise_variance=jnp.ones((4,), dtype=jnp.float32),
            disc_type="linear_interp",
        )

    with pytest.raises(ValueError, match="at least one translation"):
        e_step.E_with_precompute(
            ds,
            volume=jnp.zeros((8,), dtype=jnp.complex64),
            rotations=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            translations=jnp.zeros((0, 2), dtype=jnp.float32),
            noise_variance=jnp.ones((4,), dtype=jnp.float32),
            disc_type="linear_interp",
        )


def test_M_with_precompute_rejects_empty_rotations_or_translations():
    ds = type(
        "DS",
        (),
        {
            "image_size": 4,
            "image_shape": (2, 2),
            "n_images": 1,
        },
    )()

    with pytest.raises(ValueError, match="at least one rotation"):
        m_step.M_with_precompute(
            ds,
            probabilities=jnp.zeros((1, 0, 1), dtype=jnp.float32),
            rotations=jnp.zeros((0, 3, 3), dtype=jnp.float32),
            translations=jnp.zeros((1, 2), dtype=jnp.float32),
            noise_variance=jnp.ones((4,), dtype=jnp.float32),
            disc_type="linear_interp",
        )

    with pytest.raises(ValueError, match="at least one translation"):
        m_step.M_with_precompute(
            ds,
            probabilities=jnp.zeros((1, 1, 0), dtype=jnp.float32),
            rotations=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            translations=jnp.zeros((0, 2), dtype=jnp.float32),
            noise_variance=jnp.ones((4,), dtype=jnp.float32),
            disc_type="linear_interp",
        )


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax


@pytest.mark.gpu
def test_compute_probability_one_image_gpu(gpu_device):
    residual = jnp.array([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]], dtype=jnp.float32)

    cpu_out = np.asarray(e_step.compute_probability_from_residual_normal_squared_one_image(residual))

    with jax.default_device(gpu_device):
        residual_g = jax.device_put(residual, gpu_device)
        gpu_out = np.asarray(e_step.compute_probability_from_residual_normal_squared_one_image(residual_g))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_compute_probability_vmap_gpu(gpu_device):
    residual = jnp.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[3.0, 3.0], [0.0, 2.0]],
        ],
        dtype=jnp.float32,
    )

    cpu_out = np.asarray(e_step.compute_probability_from_residual_normal_squared(residual))

    with jax.default_device(gpu_device):
        residual_g = jax.device_put(residual, gpu_device)
        gpu_out = np.asarray(e_step.compute_probability_from_residual_normal_squared(residual_g))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)
