import numpy as np
import pytest
import jax.numpy as jnp

pytest.importorskip("jax")
pytest.importorskip("healpy")

import recovar.em.heterogeneity as hetero

pytestmark = pytest.mark.unit


class _TinyImageSource:
    @staticmethod
    def process_images(batch, apply_image_mask=False):
        _ = apply_image_mask
        return batch


class _TinyDataset:
    image_shape = (2, 2)
    image_size = 4
    volume_shape = (2, 2, 2)
    volume_size = 8
    grid_size = 8
    dtype_real = jnp.float32
    dtype = jnp.complex64
    voxel_size = 1.0
    CTF_params = np.zeros((1, 9), dtype=np.float32)
    image_source = _TinyImageSource()

    def process_images(self, batch, apply_image_mask=False):
        return self.image_source.process_images(batch, apply_image_mask=apply_image_mask)

    @staticmethod
    def ctf_evaluator(params, _image_shape, _voxel_size):
        return jnp.ones((params.shape[0], 4), dtype=jnp.float32)

    def iter_batches(self, batch_size, *, indices=None, **kwargs):
        _ = (batch_size, indices, kwargs)
        yield (
            jnp.ones((1, 4), dtype=jnp.complex64),
            jnp.zeros((1, 3, 3), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.float32),
            self.CTF_params[:1],
            None,
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
        )


def test_compute_H_B_rejects_empty_rotations_or_translations():
    ds = _TinyDataset()
    mean = np.zeros((8,), dtype=np.complex64)
    probs = np.zeros((1, 1, 1), dtype=np.float32)
    picked = np.array([0], dtype=np.int32)
    noise = np.ones((4,), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one rotation"):
        hetero.compute_H_B(
            ds,
            mean,
            probs,
            rotations=np.zeros((0, 3, 3), dtype=np.float32),
            translations=np.zeros((1, 2), dtype=np.float32),
            noise_variance=noise,
            volume_mask=None,
            picked_frequency_indices=picked,
            image_indices=np.array([0], dtype=np.int32),
            mean_disc="linear_interp",
        )

    with pytest.raises(ValueError, match="at least one translation"):
        hetero.compute_H_B(
            ds,
            mean,
            probs,
            rotations=np.zeros((1, 3, 3), dtype=np.float32),
            translations=np.zeros((0, 2), dtype=np.float32),
            noise_variance=noise,
            volume_mask=None,
            picked_frequency_indices=picked,
            image_indices=np.array([0], dtype=np.int32),
            mean_disc="linear_interp",
        )


def test_compute_H_B_small_rotation_count_avoids_zero_internal_batches(monkeypatch):
    ds = _TinyDataset()
    mean = np.zeros((8,), dtype=np.complex64)
    probs = np.ones((1, 1, 2), dtype=np.float32) / 2.0
    picked = np.array([0, 3], dtype=np.int32)
    noise = np.ones((4,), dtype=np.float32)
    rotations = np.zeros((1, 3, 3), dtype=np.float32)
    translations = np.zeros((2, 2), dtype=np.float32)

    from recovar import utils as rec_utils

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 1)
    monkeypatch.setattr(rec_utils, "get_image_batch_size", lambda _grid_size, _gpu_memory: 0)
    monkeypatch.setattr(
        rec_utils,
        "index_batch_iter",
        lambda n_units, batch_size: (
            [np.arange(n_units, dtype=np.int32)]
            if batch_size >= 1
            else (_ for _ in ()).throw(AssertionError("batch_size must be >= 1"))
        ),
    )
    monkeypatch.setattr(
        hetero.core,
        "slice_volume",
        lambda _mean, rot, _image_shape, _volume_shape: np.zeros((len(rot), 4), dtype=np.complex64),
    )
    monkeypatch.setattr(
        hetero.core,
        "vec_indices_to_vol_indices",
        lambda vec, _shape: np.zeros((len(vec), 3), dtype=np.int32),
    )
    monkeypatch.setattr(
        hetero.core,
        "batch_get_gridpoint_coords",
        lambda rot, _image_shape, _volume_shape: np.zeros((len(rot), 4, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        hetero,
        "sum_up_images_fixed_rots_covariance_precompute_eqx",
        lambda _config, _images, _translations, ctf_params: (
            jnp.zeros((len(ctf_params), 2, 4), dtype=jnp.complex64),
            jnp.ones((len(ctf_params), 4), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        hetero,
        "sum_up_images_fixed_rots_covariance_with_precompute_eqx",
        lambda _config, _shifted, _mean_proj, _ctf, _grid, _prob, _rots, _noise, _picked, H=0, B=0, **_k: (
            H + jnp.ones_like(H),
            B + (1.0 + 0j) * jnp.ones_like(B),
        ),
    )

    H, B = hetero.compute_H_B(
        ds,
        mean,
        probs,
        rotations=rotations,
        translations=translations,
        noise_variance=noise,
        volume_mask=None,
        picked_frequency_indices=picked,
        image_indices=np.array([0], dtype=np.int32),
        mean_disc="linear_interp",
    )

    assert len(H) == 2
    assert len(B) == 2
    np.testing.assert_allclose(np.asarray(H[0]), np.ones((8,), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(H[1]), np.ones((8,), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(B[0]), np.ones((8,), dtype=np.complex64))
    np.testing.assert_allclose(np.asarray(B[1]), np.ones((8,), dtype=np.complex64))


def test_compute_projected_covariance_rhs_lhs_rejects_empty_rotations_or_translations():
    ds = _TinyDataset()
    mean = np.zeros((8,), dtype=np.complex64)
    basis = np.zeros((8, 2), dtype=np.complex64)
    probs = np.zeros((1, 1, 1), dtype=np.float32)
    noise = np.ones((4,), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one rotation"):
        hetero.compute_projected_covariance_rhs_lhs(
            ds,
            mean,
            basis,
            rotations=np.zeros((0, 3, 3), dtype=np.float32),
            translations=np.zeros((1, 2), dtype=np.float32),
            probabilities=probs,
            volume_mask=None,
            noise_variance=noise,
            disc_type_mean="linear_interp",
            disc_type_u="linear_interp",
            image_indices=np.array([0], dtype=np.int32),
        )

    with pytest.raises(ValueError, match="at least one translation"):
        hetero.compute_projected_covariance_rhs_lhs(
            ds,
            mean,
            basis,
            rotations=np.zeros((1, 3, 3), dtype=np.float32),
            translations=np.zeros((0, 2), dtype=np.float32),
            probabilities=probs,
            volume_mask=None,
            noise_variance=noise,
            disc_type_mean="linear_interp",
            disc_type_u="linear_interp",
            image_indices=np.array([0], dtype=np.int32),
        )


def test_compute_projected_covariance_rhs_lhs_small_rotation_count_avoids_zero_batches(monkeypatch):
    ds = _TinyDataset()
    mean = np.zeros((8,), dtype=np.complex64)
    basis = np.ones((8, 2), dtype=np.complex64)
    probs = np.ones((1, 1, 2), dtype=np.float32) / 2.0
    noise = np.ones((4,), dtype=np.float32)
    rotations = np.zeros((1, 3, 3), dtype=np.float32)
    translations = np.zeros((2, 2), dtype=np.float32)

    from recovar import utils as rec_utils

    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 1)
    monkeypatch.setattr(rec_utils, "get_image_batch_size", lambda _grid_size, _gpu_memory: 0)
    monkeypatch.setattr(
        rec_utils,
        "index_batch_iter",
        lambda n_units, batch_size: (
            [np.arange(n_units, dtype=np.int32)]
            if batch_size >= 1
            else (_ for _ in ()).throw(AssertionError("batch_size must be >= 1"))
        ),
    )
    monkeypatch.setattr(
        hetero.core,
        "slice_volume",
        lambda _mean, rot, _image_shape, _volume_shape: np.zeros((len(rot), 4), dtype=np.complex64),
    )
    monkeypatch.setattr(
        hetero,
        "batch_vol_slice_volume",
        lambda basis_local, rot, _image_shape, _volume_shape: np.zeros(
            (len(rot), basis_local.array.shape[0] if hasattr(basis_local, "array") else basis_local.shape[0], 4),
            dtype=np.complex64,
        ),
    )
    monkeypatch.setattr(
        hetero,
        "reduce_covariance_est_inner_eqx",
        lambda _config, _mean_proj, _u_proj, _prob, _batch, _translations, _ctf_params, _noise: (
            jnp.eye(2, dtype=jnp.float32),
            jnp.ones((2, 2), dtype=jnp.float32),
        ),
    )

    lhs, rhs = hetero.compute_projected_covariance_rhs_lhs(
        ds,
        mean,
        basis,
        rotations=rotations,
        translations=translations,
        probabilities=probs,
        volume_mask=None,
        noise_variance=noise,
        disc_type_mean="linear_interp",
        disc_type_u="linear_interp",
        image_indices=np.array([0], dtype=np.int32),
    )

    np.testing.assert_allclose(np.asarray(lhs), np.eye(2, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(rhs), np.ones((2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# GPU tests – verify guard functions still reject on GPU-placed arrays
# ---------------------------------------------------------------------------

import jax


@pytest.mark.gpu
def test_compute_H_B_rejects_empty_rotations_gpu(gpu_device):
    """Guard validation works identically with GPU-placed arrays."""
    ds = _TinyDataset()
    mean = np.zeros((8,), dtype=np.complex64)
    probs = np.zeros((1, 1, 1), dtype=np.float32)
    picked = np.array([0], dtype=np.int32)
    noise_arr = np.ones((4,), dtype=np.float32)

    with jax.default_device(gpu_device):
        with pytest.raises(ValueError, match="at least one rotation"):
            hetero.compute_H_B(
                ds,
                mean,
                probs,
                rotations=jax.device_put(jnp.zeros((0, 3, 3), dtype=jnp.float32), gpu_device),
                translations=jnp.zeros((1, 2), dtype=jnp.float32),
                noise_variance=noise_arr,
                volume_mask=None,
                picked_frequency_indices=picked,
                image_indices=np.array([0], dtype=np.int32),
                mean_disc="linear_interp",
            )

        with pytest.raises(ValueError, match="at least one translation"):
            hetero.compute_H_B(
                ds,
                mean,
                probs,
                rotations=jnp.zeros((1, 3, 3), dtype=jnp.float32),
                translations=jax.device_put(jnp.zeros((0, 2), dtype=jnp.float32), gpu_device),
                noise_variance=noise_arr,
                volume_mask=None,
                picked_frequency_indices=picked,
                image_indices=np.array([0], dtype=np.int32),
                mean_disc="linear_interp",
            )


@pytest.mark.gpu
def test_compute_projected_covariance_rhs_lhs_rejects_empty_gpu(gpu_device):
    """Guard validation works identically with GPU-placed arrays."""
    ds = _TinyDataset()
    mean = np.zeros((8,), dtype=np.complex64)
    basis = np.zeros((8, 2), dtype=np.complex64)
    probs = np.zeros((1, 1, 1), dtype=np.float32)
    noise_arr = np.ones((4,), dtype=np.float32)

    with jax.default_device(gpu_device):
        with pytest.raises(ValueError, match="at least one rotation"):
            hetero.compute_projected_covariance_rhs_lhs(
                ds,
                mean,
                basis,
                rotations=jax.device_put(jnp.zeros((0, 3, 3), dtype=jnp.float32), gpu_device),
                translations=jnp.zeros((1, 2), dtype=jnp.float32),
                probabilities=probs,
                volume_mask=None,
                noise_variance=noise_arr,
                disc_type_mean="linear_interp",
                disc_type_u="linear_interp",
                image_indices=np.array([0], dtype=np.int32),
            )

        with pytest.raises(ValueError, match="at least one translation"):
            hetero.compute_projected_covariance_rhs_lhs(
                ds,
                mean,
                basis,
                rotations=jnp.zeros((1, 3, 3), dtype=jnp.float32),
                translations=jax.device_put(jnp.zeros((0, 2), dtype=jnp.float32), gpu_device),
                probabilities=probs,
                volume_mask=None,
                noise_variance=noise_arr,
                disc_type_mean="linear_interp",
                disc_type_u="linear_interp",
                image_indices=np.array([0], dtype=np.int32),
            )
