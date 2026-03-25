import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.heterogeneity import adaptive_kernel_discretization as akd
from recovar.heterogeneity import ppca
from recovar.core import linalg

pytestmark = pytest.mark.unit


def test_batch_vec_and_unvec_roundtrip_real():
    x = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    v = akd.batch_vec(x)
    x_rt = akd.batch_unvec(v)
    assert v.shape == (2, 9)
    assert x_rt.shape == x.shape
    assert np.allclose(x_rt, x)


def test_batch_vec_and_unvec_roundtrip_complex():
    rng = np.random.default_rng(0)
    xr = rng.normal(size=(4, 2, 2)).astype(np.float32)
    xi = rng.normal(size=(4, 2, 2)).astype(np.float32)
    x = xr + 1j * xi
    v = akd.batch_vec(x)
    x_rt = akd.batch_unvec(v)
    assert v.shape == (4, 4)
    assert x_rt.shape == x.shape
    assert np.allclose(x_rt, x)


# ---------------------------------------------------------------------------
# ppca module tests — M_step_batch, M_step (EM requires full dataset)
# ---------------------------------------------------------------------------


def test_M_step_batch_runs_and_accumulates():
    """Verify M_step_batch runs without shape errors and accumulates non-trivially."""
    from recovar import core

    rng = np.random.default_rng(10)
    grid_size = 4
    image_shape = (grid_size, grid_size)
    volume_shape = (grid_size, grid_size, grid_size)
    volume_size = int(np.prod(volume_shape))
    n_images = 3
    basis_size = 2
    voxel_size = 1.0

    n_pixels = int(np.prod(image_shape))
    images = (rng.normal(size=(n_images, n_pixels)) + 1j * rng.normal(size=(n_images, n_pixels))).astype(np.complex64)
    # Realistic CTF params: DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT, BFACTOR, CONTRAST
    CTF_params = np.zeros((n_images, 9), dtype=np.float32)
    CTF_params[:, 0] = 15000.0  # DFU (Angstrom)
    CTF_params[:, 1] = 15000.0  # DFV (Angstrom)
    CTF_params[:, 3] = 300.0  # VOLT (kV)
    CTF_params[:, 4] = 2.7  # CS (mm)
    CTF_params[:, 5] = 0.1  # W (amplitude contrast)
    CTF_params[:, 8] = 1.0  # CONTRAST
    rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    translations = np.zeros((n_images, 2), dtype=np.float32)
    noise_variance = np.ones((n_images, n_pixels), dtype=np.float32)

    latent_means = rng.normal(size=(n_images, basis_size)).astype(np.float32)
    latent_covs = np.tile(np.eye(basis_size, dtype=np.float32), (n_images, 1, 1)) * 0.1

    lhs = jnp.zeros((volume_size, basis_size * basis_size), dtype=np.complex64)
    rhs = jnp.zeros((volume_size, basis_size), dtype=np.complex64)

    lhs_out, rhs_out = ppca.M_step_batch(
        images,
        lhs,
        rhs,
        latent_means,
        latent_covs,
        CTF_params,
        rotation_matrices,
        translations,
        image_shape,
        volume_shape,
        grid_size,
        voxel_size,
        noise_variance,
        core.CTFEvaluator(),
    )

    assert lhs_out.shape == (volume_size, basis_size * basis_size)
    assert rhs_out.shape == (volume_size, basis_size)
    assert np.all(np.isfinite(np.asarray(lhs_out)))
    assert np.all(np.isfinite(np.asarray(rhs_out)))
    # At least some voxels should have non-zero accumulations
    assert np.any(np.asarray(lhs_out) != 0)
    assert np.any(np.asarray(rhs_out) != 0)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_batch_vec_unvec_roundtrip_gpu(gpu_device):
    x = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)

    cpu_v = np.asarray(akd.batch_vec(x))
    cpu_rt = np.asarray(akd.batch_unvec(cpu_v))

    with jax.default_device(gpu_device):
        x_g = jax.device_put(jnp.array(x), gpu_device)
        gpu_v = np.asarray(akd.batch_vec(x_g))
        gpu_rt = np.asarray(akd.batch_unvec(gpu_v))

    np.testing.assert_allclose(cpu_v, gpu_v, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_rt, gpu_rt, atol=1e-5, rtol=1e-5)
