"""Numerical unit tests for recovar.em.heterogeneity pure functions.

Tests functions that require NO experiment_dataset mock:
  - compute_UPLambdainvPU
  - compute_bLambdainvPU_terms
  - solve_covariance
  - compute_bHb_terms  (with trivial CTF/process_images)
"""

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

import recovar.em.heterogeneity as hetero

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared tiny-data dimensions
# ---------------------------------------------------------------------------
N_ROT = 3
N_PC = 2
IMG_SZ = 4  # 2x2 image flattened
IMAGE_SHAPE = (2, 2)
N_IMG = 2
N_TRANS = 1
RNG = np.random.default_rng(42)


def _rand_complex(shape):
    return (RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Tests for compute_UPLambdainvPU
# ---------------------------------------------------------------------------


def test_UPLambdainvPU_output_shape():
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    CTF = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    noise = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    H = hetero.compute_UPLambdainvPU(u_proj, CTF, noise)
    assert H.shape == (N_ROT, N_ROT, N_PC, N_PC)


def test_UPLambdainvPU_output_is_real():
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    CTF = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    noise = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    H = hetero.compute_UPLambdainvPU(u_proj, CTF, noise)
    assert jnp.issubdtype(H.dtype, jnp.floating)


def test_UPLambdainvPU_no_nan():
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    CTF = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    noise = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    H = hetero.compute_UPLambdainvPU(u_proj, CTF, noise)
    assert jnp.all(jnp.isfinite(H))


def test_UPLambdainvPU_single_pc_diagonal():
    """With n_pc=1, identity CTF, unit noise: H[r, r, 0, 0] = ||u_proj[r]||^2."""
    rng = np.random.default_rng(99)
    n_rot, n_pc = 2, 1
    u_raw = (rng.standard_normal((n_rot, n_pc, IMG_SZ)) + 1j * rng.standard_normal((n_rot, n_pc, IMG_SZ))).astype(
        np.complex64
    )
    u_proj = jnp.array(u_raw)
    CTF = jnp.ones((n_rot, IMG_SZ), dtype=jnp.float32)
    noise = jnp.ones((n_rot, IMG_SZ), dtype=jnp.float32)
    H = hetero.compute_UPLambdainvPU(u_proj, CTF, noise)
    for r in range(n_rot):
        expected = float(np.real(np.dot(u_raw[r, 0], np.conj(u_raw[r, 0]))))
        np.testing.assert_allclose(float(H[r, r, 0, 0]), expected, rtol=1e-4)


def test_UPLambdainvPU_symmetric_in_pc_axes():
    """The (n_pc x n_pc) block must be symmetric for each (r1, r2) pair."""
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    CTF = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    noise = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    H = np.array(hetero.compute_UPLambdainvPU(u_proj, CTF, noise))
    for r1 in range(N_ROT):
        for r2 in range(N_ROT):
            np.testing.assert_allclose(H[r1, r2], H[r1, r2].T, atol=1e-5)


def test_UPLambdainvPU_noise_scaling():
    """Doubling noise_variance halves the output."""
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    CTF = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    noise1 = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    noise2 = 2.0 * noise1
    H1 = hetero.compute_UPLambdainvPU(u_proj, CTF, noise1)
    H2 = hetero.compute_UPLambdainvPU(u_proj, CTF, noise2)
    np.testing.assert_allclose(np.array(H1), 2.0 * np.array(H2), rtol=1e-4)


def test_UPLambdainvPU_zero_u_gives_zero():
    u_proj = jnp.zeros((N_ROT, N_PC, IMG_SZ), dtype=jnp.complex64)
    CTF = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    noise = jnp.ones((N_ROT, IMG_SZ), dtype=jnp.float32)
    H = hetero.compute_UPLambdainvPU(u_proj, CTF, noise)
    np.testing.assert_allclose(np.array(H), 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Tests for compute_bLambdainvPU_terms
#
# Shape contract: CTF.shape[0] must equal images.shape[0] (one CTF per image).
# n_rot (from u_projections.shape[0]) is independent.
# Output: (n_rot, n_images, n_pc, n_translations)
# ---------------------------------------------------------------------------


def test_bLambdainvPU_output_shape():
    mean_proj = jnp.array(_rand_complex((N_ROT, IMG_SZ)))
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    images = jnp.array(_rand_complex((N_IMG, IMG_SZ)))
    trans = jnp.zeros((N_TRANS, 2), dtype=jnp.float32)
    # CTF must have shape (n_images, image_size) to match batch
    CTF = jnp.ones((N_IMG, IMG_SZ), dtype=jnp.float32)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float32)
    b = hetero.compute_bLambdainvPU_terms(mean_proj, u_proj, images, trans, CTF, noise_var, IMAGE_SHAPE)
    assert b.shape == (N_ROT, N_IMG, N_PC, N_TRANS)


def test_bLambdainvPU_output_is_real():
    mean_proj = jnp.array(_rand_complex((N_ROT, IMG_SZ)))
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    images = jnp.array(_rand_complex((N_IMG, IMG_SZ)))
    trans = jnp.zeros((N_TRANS, 2), dtype=jnp.float32)
    CTF = jnp.ones((N_IMG, IMG_SZ), dtype=jnp.float32)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float32)
    b = hetero.compute_bLambdainvPU_terms(mean_proj, u_proj, images, trans, CTF, noise_var, IMAGE_SHAPE)
    assert jnp.issubdtype(b.dtype, jnp.floating)


def test_bLambdainvPU_no_nan():
    mean_proj = jnp.array(_rand_complex((N_ROT, IMG_SZ)))
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    images = jnp.array(_rand_complex((N_IMG, IMG_SZ)))
    trans = jnp.zeros((N_TRANS, 2), dtype=jnp.float32)
    CTF = jnp.ones((N_IMG, IMG_SZ), dtype=jnp.float32)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float32)
    b = hetero.compute_bLambdainvPU_terms(mean_proj, u_proj, images, trans, CTF, noise_var, IMAGE_SHAPE)
    assert jnp.all(jnp.isfinite(b))


def test_bLambdainvPU_zero_images_zero_mean():
    """With zero images and zero mean, b must be zero."""
    mean_proj = jnp.zeros((N_ROT, IMG_SZ), dtype=jnp.complex64)
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    images = jnp.zeros((N_IMG, IMG_SZ), dtype=jnp.complex64)
    trans = jnp.zeros((N_TRANS, 2), dtype=jnp.float32)
    CTF = jnp.ones((N_IMG, IMG_SZ), dtype=jnp.float32)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float32)
    b = hetero.compute_bLambdainvPU_terms(mean_proj, u_proj, images, trans, CTF, noise_var, IMAGE_SHAPE)
    np.testing.assert_allclose(np.array(b), 0.0, atol=1e-6)


def test_bLambdainvPU_zero_u_gives_zero():
    """If U=0, b must be zero regardless of image values."""
    mean_proj = jnp.zeros((N_ROT, IMG_SZ), dtype=jnp.complex64)
    u_proj = jnp.zeros((N_ROT, N_PC, IMG_SZ), dtype=jnp.complex64)
    images = jnp.array(_rand_complex((N_IMG, IMG_SZ)))
    trans = jnp.zeros((N_TRANS, 2), dtype=jnp.float32)
    CTF = jnp.ones((N_IMG, IMG_SZ), dtype=jnp.float32)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float32)
    b = hetero.compute_bLambdainvPU_terms(mean_proj, u_proj, images, trans, CTF, noise_var, IMAGE_SHAPE)
    np.testing.assert_allclose(np.array(b), 0.0, atol=1e-6)


def test_bLambdainvPU_noise_scaling_mean_term():
    """With zero images, only b2 survives. Doubling noise halves b2."""
    rng = np.random.default_rng(77)
    n_rot, n_pc, n_img = 2, 1, 1
    mean_proj = jnp.array(
        (rng.standard_normal((n_rot, IMG_SZ)) + 1j * rng.standard_normal((n_rot, IMG_SZ))).astype(np.complex64)
    )
    u_proj = jnp.array(
        (rng.standard_normal((n_rot, n_pc, IMG_SZ)) + 1j * rng.standard_normal((n_rot, n_pc, IMG_SZ))).astype(
            np.complex64
        )
    )
    images = jnp.zeros((n_img, IMG_SZ), dtype=jnp.complex64)
    trans = jnp.zeros((1, 2), dtype=jnp.float32)
    CTF = jnp.ones((n_img, IMG_SZ), dtype=jnp.float32)
    b1 = hetero.compute_bLambdainvPU_terms(mean_proj, u_proj, images, trans, CTF, jnp.ones(IMG_SZ), IMAGE_SHAPE)
    b2 = hetero.compute_bLambdainvPU_terms(mean_proj, u_proj, images, trans, CTF, 2.0 * jnp.ones(IMG_SZ), IMAGE_SHAPE)
    np.testing.assert_allclose(np.array(b1), 2.0 * np.array(b2), rtol=1e-4)


def test_bLambdainvPU_multiple_translations_shape():
    n_trans = 3
    mean_proj = jnp.array(_rand_complex((N_ROT, IMG_SZ)))
    u_proj = jnp.array(_rand_complex((N_ROT, N_PC, IMG_SZ)))
    images = jnp.array(_rand_complex((N_IMG, IMG_SZ)))
    trans = jnp.zeros((n_trans, 2), dtype=jnp.float32)
    CTF = jnp.ones((N_IMG, IMG_SZ), dtype=jnp.float32)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float32)
    b = hetero.compute_bLambdainvPU_terms(mean_proj, u_proj, images, trans, CTF, noise_var, IMAGE_SHAPE)
    assert b.shape == (N_ROT, N_IMG, N_PC, n_trans)


# ---------------------------------------------------------------------------
# Tests for solve_covariance
# ---------------------------------------------------------------------------


def test_solve_covariance_identity_lhs():
    """lhs = I => covar = rhs."""
    n = 3
    rng = np.random.default_rng(50)
    rhs = jnp.array(rng.standard_normal((n, n)).astype(np.float32))
    lhs = jnp.eye(n * n, dtype=jnp.float32)
    covar = hetero.solve_covariance(lhs, rhs)
    np.testing.assert_allclose(np.array(covar), np.array(rhs), atol=1e-5)


def test_solve_covariance_scalar():
    """n_pc=1: lhs (1,1), rhs (1,1). covar = rhs / lhs."""
    lhs = jnp.array([[4.0]], dtype=jnp.float32)
    rhs = jnp.array([[8.0]], dtype=jnp.float32)
    covar = hetero.solve_covariance(lhs, rhs)
    np.testing.assert_allclose(float(covar[0, 0]), 2.0, atol=1e-5)


def test_solve_covariance_2x2_known():
    """n_pc=2. lhs = 2*I_4, rhs = [[1,2],[3,4]]. covar = rhs/2."""
    lhs = 2.0 * jnp.eye(4, dtype=jnp.float32)
    rhs = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    covar = hetero.solve_covariance(lhs, rhs)
    expected = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(covar), expected, atol=1e-5)


def test_solve_covariance_output_shape():
    for n in [1, 2, 4]:
        lhs = jnp.eye(n * n, dtype=jnp.float32)
        rhs = jnp.zeros((n, n), dtype=jnp.float32)
        covar = hetero.solve_covariance(lhs, rhs)
        assert covar.shape == (n, n), f"n={n}: expected ({n},{n}), got {covar.shape}"


def test_solve_covariance_zero_rhs():
    """rhs = 0 => covar = 0."""
    n = 3
    lhs = jnp.eye(n * n, dtype=jnp.float32) * 5.0
    rhs = jnp.zeros((n, n), dtype=jnp.float32)
    covar = hetero.solve_covariance(lhs, rhs)
    np.testing.assert_allclose(np.array(covar), 0.0, atol=1e-6)


def test_solve_covariance_pd_lhs_finite():
    """PD lhs with random rhs should give finite output."""
    rng = np.random.default_rng(88)
    n = 2
    A = rng.standard_normal((n * n, n * n)).astype(np.float32)
    lhs = jnp.array(A.T @ A + 0.1 * np.eye(n * n, dtype=np.float32))
    rhs = jnp.ones((n, n), dtype=jnp.float32)
    covar = hetero.solve_covariance(lhs, rhs)
    assert covar.shape == (n, n)
    assert jnp.all(jnp.isfinite(covar))


# ---------------------------------------------------------------------------
# Tests for compute_bHb_terms
# ---------------------------------------------------------------------------


def _identity_ctf(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = image_shape[0] * image_shape[1]
    return jnp.ones((n, sz), dtype=jnp.float32)


def _identity_process(batch, apply_image_mask=False):
    return batch


def _make_bHb_inputs(rng, n_rot=2, n_pc=2, n_img=2, n_trans=1):
    """Build inputs for compute_bHb_terms.

    CTF_params.shape[0] == n_img (one CTF per image).
    mean_projections / u_projections have n_rot (separate dimension).
    """
    mean_proj = jnp.array(
        (rng.standard_normal((n_rot, IMG_SZ)) + 1j * rng.standard_normal((n_rot, IMG_SZ))).astype(np.complex64)
    )
    u_proj = jnp.array(
        0.01
        * (rng.standard_normal((n_rot, n_pc, IMG_SZ)) + 1j * rng.standard_normal((n_rot, n_pc, IMG_SZ))).astype(
            np.complex64
        )
    )
    s = jnp.ones(n_pc, dtype=jnp.float32) * 2.0
    batch = jnp.array(
        (rng.standard_normal((n_img, IMG_SZ)) + 1j * rng.standard_normal((n_img, IMG_SZ))).astype(np.complex64)
    )
    trans = jnp.zeros((n_trans, 2), dtype=jnp.float32)
    # CTF_params indexed by image, not rotation
    CTF_params = jnp.zeros((n_img, 9), dtype=jnp.float32)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float32)
    return mean_proj, u_proj, s, batch, trans, CTF_params, noise_var


def test_bHb_output_shape():
    n_rot, n_pc, n_img, n_trans = 2, 2, 3, 1
    mean_proj, u_proj, s, batch, trans, CTF_params, noise_var = _make_bHb_inputs(
        np.random.default_rng(7), n_rot, n_pc, n_img, n_trans
    )
    result = hetero.compute_bHb_terms(
        mean_proj, u_proj, s, batch, trans, CTF_params, _identity_ctf, noise_var, 1.0, IMAGE_SHAPE, _identity_process
    )
    assert result.shape == (n_img, n_rot, n_trans)


def test_bHb_output_is_real():
    mean_proj, u_proj, s, batch, trans, CTF_params, noise_var = _make_bHb_inputs(np.random.default_rng(8))
    result = hetero.compute_bHb_terms(
        mean_proj, u_proj, s, batch, trans, CTF_params, _identity_ctf, noise_var, 1.0, IMAGE_SHAPE, _identity_process
    )
    assert jnp.issubdtype(result.dtype, jnp.floating)


def test_bHb_no_nan():
    mean_proj, u_proj, s, batch, trans, CTF_params, noise_var = _make_bHb_inputs(np.random.default_rng(9))
    result = hetero.compute_bHb_terms(
        mean_proj, u_proj, s, batch, trans, CTF_params, _identity_ctf, noise_var, 1.0, IMAGE_SHAPE, _identity_process
    )
    assert jnp.all(jnp.isfinite(result))


def test_bHb_zero_u_cholesky_succeeds():
    """With u_projections=0, H=diag(1/s) which is PD. Cholesky must work."""
    n_rot, n_pc, n_img = 2, 2, 2
    mean_proj = jnp.zeros((n_rot, IMG_SZ), dtype=jnp.complex64)
    u_proj = jnp.zeros((n_rot, n_pc, IMG_SZ), dtype=jnp.complex64)
    s = jnp.array([1.0, 2.0], dtype=jnp.float32)
    batch = jnp.zeros((n_img, IMG_SZ), dtype=jnp.complex64)
    trans = jnp.zeros((1, 2), dtype=jnp.float32)
    CTF_params = jnp.zeros((n_img, 9), dtype=jnp.float32)
    noise_var = jnp.ones(IMG_SZ, dtype=jnp.float32)
    result = hetero.compute_bHb_terms(
        mean_proj, u_proj, s, batch, trans, CTF_params, _identity_ctf, noise_var, 1.0, IMAGE_SHAPE, _identity_process
    )
    assert result.shape == (n_img, n_rot, 1)
    assert jnp.all(jnp.isfinite(result))


def test_bHb_changing_s_changes_output():
    """Changing prior variance s must change the ELBO term."""
    mean_proj, u_proj, s, batch, trans, CTF_params, noise_var = _make_bHb_inputs(np.random.default_rng(10))
    r1 = hetero.compute_bHb_terms(
        mean_proj, u_proj, s, batch, trans, CTF_params, _identity_ctf, noise_var, 1.0, IMAGE_SHAPE, _identity_process
    )
    r2 = hetero.compute_bHb_terms(
        mean_proj,
        u_proj,
        s * 10,
        batch,
        trans,
        CTF_params,
        _identity_ctf,
        noise_var,
        1.0,
        IMAGE_SHAPE,
        _identity_process,
    )
    assert not jnp.allclose(r1, r2)


def test_bHb_multiple_translations_shape():
    n_rot, n_pc, n_img, n_trans = 2, 2, 2, 3
    mean_proj, u_proj, s, batch, trans, CTF_params, noise_var = _make_bHb_inputs(
        np.random.default_rng(11), n_rot, n_pc, n_img, n_trans
    )
    result = hetero.compute_bHb_terms(
        mean_proj, u_proj, s, batch, trans, CTF_params, _identity_ctf, noise_var, 1.0, IMAGE_SHAPE, _identity_process
    )
    assert result.shape == (n_img, n_rot, n_trans)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_UPLambdainvPU_on_gpu(gpu_device):
    rng = np.random.default_rng(200)
    u_raw = (rng.standard_normal((N_ROT, N_PC, IMG_SZ)) + 1j * rng.standard_normal((N_ROT, N_PC, IMG_SZ))).astype(
        np.complex64
    )
    CTF_raw = np.ones((N_ROT, IMG_SZ), dtype=np.float32)
    noise_raw = np.ones((N_ROT, IMG_SZ), dtype=np.float32)
    cpu_out = np.array(hetero.compute_UPLambdainvPU(jnp.array(u_raw), jnp.array(CTF_raw), jnp.array(noise_raw)))
    with jax.default_device(gpu_device):
        gpu_out = np.array(hetero.compute_UPLambdainvPU(jnp.array(u_raw), jnp.array(CTF_raw), jnp.array(noise_raw)))
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_solve_covariance_on_gpu(gpu_device):
    rng = np.random.default_rng(201)
    n = 3
    A = rng.standard_normal((n * n, n * n)).astype(np.float32)
    lhs_np = A.T @ A + 0.1 * np.eye(n * n, dtype=np.float32)
    rhs_np = rng.standard_normal((n, n)).astype(np.float32)
    cpu_out = np.array(hetero.solve_covariance(jnp.array(lhs_np), jnp.array(rhs_np)))
    with jax.default_device(gpu_device):
        gpu_out = np.array(hetero.solve_covariance(jnp.array(lhs_np), jnp.array(rhs_np)))
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_bHb_on_gpu(gpu_device):
    rng = np.random.default_rng(202)
    n_rot, n_pc, n_img = 2, 2, 2
    mean_proj = (rng.standard_normal((n_rot, IMG_SZ)) + 1j * rng.standard_normal((n_rot, IMG_SZ))).astype(np.complex64)
    u_proj = (
        0.01 * (rng.standard_normal((n_rot, n_pc, IMG_SZ)) + 1j * rng.standard_normal((n_rot, n_pc, IMG_SZ)))
    ).astype(np.complex64)
    s = np.ones(n_pc, dtype=np.float32) * 2.0
    batch = (rng.standard_normal((n_img, IMG_SZ)) + 1j * rng.standard_normal((n_img, IMG_SZ))).astype(np.complex64)
    trans = np.zeros((1, 2), dtype=np.float32)
    # CTF_params indexed by image, not rotation
    CTF_params = np.zeros((n_img, 9), dtype=np.float32)
    noise_var = np.ones(IMG_SZ, dtype=np.float32)
    cpu_out = np.array(
        hetero.compute_bHb_terms(
            jnp.array(mean_proj),
            jnp.array(u_proj),
            jnp.array(s),
            jnp.array(batch),
            jnp.array(trans),
            jnp.array(CTF_params),
            _identity_ctf,
            jnp.array(noise_var),
            1.0,
            IMAGE_SHAPE,
            _identity_process,
        )
    )
    with jax.default_device(gpu_device):
        gpu_out = np.array(
            hetero.compute_bHb_terms(
                jnp.array(mean_proj),
                jnp.array(u_proj),
                jnp.array(s),
                jnp.array(batch),
                jnp.array(trans),
                jnp.array(CTF_params),
                _identity_ctf,
                jnp.array(noise_var),
                1.0,
                IMAGE_SHAPE,
                _identity_process,
            )
        )
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-4, rtol=1e-4)
