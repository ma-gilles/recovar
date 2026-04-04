"""Tests for sketched normal-operator products (ppca.compute_normal_residual_sketches).

All primitives operate in half-image (rfft2) convention for efficiency.

Tests verify:
1. Dense reference: S_L @ G_dense == left_sketch_helper, G_dense @ Q_R == right_sketch_helper
2. Cryo-EM path consistency on a tiny batch
3. Zero residual → zero output
4. Linearity in residual
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core
import recovar.core.forward as core_forward
import recovar.core.fourier_transform_utils as ftu
from recovar.core import linalg
from recovar.core.configs import ForwardModelConfig
from recovar.core.slicing import adjoint_slice_volume
from recovar.heterogeneity import covariance_core
from recovar.heterogeneity.sketched_normal import (
    compute_residual_batch_from_factors,
    left_sketch_normal_residual_batch,
    right_sketch_normal_residual_batch,
)
# left_sketch_normal_residual_batch now takes config as first arg

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ones_ctf(ctf_params, image_shape, voxel_size, **kw):
    """CTF that is identically 1 everywhere."""
    half_image = kw.get("half_image", False)
    if half_image:
        shape = ftu.image_shape_to_half_image_shape(image_shape)
    else:
        shape = image_shape
    return np.ones((ctf_params.shape[0], int(np.prod(shape))), dtype=np.float32)


def _hermitian_images(rng, n_images, image_shape):
    """Generate Hermitian-symmetric Fourier images (from real-space data)."""
    real_space = rng.normal(size=(n_images, *image_shape)).astype(np.float32)
    fourier = np.fft.fft2(np.fft.fftshift(real_space, axes=(-2, -1)))
    return fourier.reshape(n_images, -1).astype(np.complex64)


def _hermitian_volume(rng, volume_shape):
    """Generate Hermitian-symmetric Fourier volume (from real-space data)."""
    real_space = rng.normal(size=volume_shape).astype(np.float32)
    fourier = np.fft.fftn(np.fft.fftshift(real_space))
    return fourier.reshape(-1).astype(np.complex64)


def _make_config(grid_size=6, disc_type="nearest"):
    image_shape = (grid_size, grid_size)
    volume_shape = (grid_size, grid_size, grid_size)
    return ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=grid_size,
        voxel_size=1.0,
        padding=0,
        disc_type=disc_type,
        ctf=recovar.core.as_ctf_evaluator(_ones_ctf),
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
    )


def _random_rotation_matrices(rng, n):
    """Generate random rotation matrices via QR decomposition."""
    mats = rng.normal(size=(n, 3, 3)).astype(np.float32)
    rotations = []
    for i in range(n):
        q, r = np.linalg.qr(mats[i])
        q = q * np.sign(np.linalg.det(q))
        rotations.append(q)
    return np.stack(rotations)


def _build_dense_gradient_batch_full(
    config, U_X, sigma_X, V_X_batch, images_batch, mean,
    rotation_matrices, translations, ctf_params, noise_variance,
):
    """Build dense G(X)[:,batch] in FULL-spectrum space for reference.

    Returns (volume_size, batch_size) array.
    """
    # Compute residual in half-image, then convert to full for reference
    residual_half, CTF_w_half = compute_residual_batch_from_factors(
        config, U_X, sigma_X, V_X_batch, images_batch, mean,
        rotation_matrices, translations, ctf_params, noise_variance,
    )
    # Convert to full spectrum for per-column adjoint
    residual_full = ftu.half_image_to_full_image(residual_half, config.image_shape)
    CTF_w_full = ftu.half_image_to_full_image(CTF_w_half, config.image_shape)

    batch_size = residual_full.shape[0]
    G_cols = []
    for i in range(batch_size):
        adjoint_input = CTF_w_full[i:i+1] * residual_full[i:i+1]
        g_i = adjoint_slice_volume(
            adjoint_input,
            rotation_matrices[i:i+1],
            config.image_shape,
            config.volume_shape,
            config.disc_type,
        )
        G_cols.append(g_i)
    return jnp.stack(G_cols, axis=-1)  # (volume_size, batch_size)


# ---------------------------------------------------------------------------
# Test 1: Dense reference against sketches
# ---------------------------------------------------------------------------


class TestDenseReference:
    """Compare sketch helpers against explicitly formed dense G(X)."""

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(42)
        grid_size = 6
        config = _make_config(grid_size=grid_size, disc_type="nearest")
        volume_size = int(np.prod(config.volume_shape))
        image_size = int(np.prod(config.image_shape))
        n_images = 5
        rank = 2
        sketch_rank = 3
        qrank = 4

        # Hermitian-symmetric volumes and images (from real-space data)
        U_X = jnp.array(np.stack([_hermitian_volume(rng, config.volume_shape) for _ in range(rank)], axis=-1))
        sigma_X = jnp.array(rng.uniform(0.5, 2.0, size=(rank,)).astype(np.float32))
        V_X_batch = jnp.array(rng.normal(size=(n_images, rank)).astype(np.float32))

        images = jnp.array(_hermitian_images(rng, n_images, config.image_shape))
        mean = jnp.array(_hermitian_volume(rng, config.volume_shape))
        rotation_matrices = jnp.array(_random_rotation_matrices(rng, n_images))
        translations = jnp.zeros((n_images, 2), dtype=np.float32)
        ctf_params = jnp.zeros((n_images, 9), dtype=np.float32)
        noise_variance = jnp.ones((n_images, image_size), dtype=np.float32)

        # Sketch matrices (S_left in volume space — also Hermitian)
        S_left = jnp.array(np.stack([_hermitian_volume(rng, config.volume_shape) for _ in range(sketch_rank)]))
        Q_right = jnp.array(rng.normal(size=(n_images, qrank)).astype(np.float32))

        hermitian_weights = linalg.rfft2_hermitian_weights(config.image_shape)

        return dict(
            config=config, U_X=U_X, sigma_X=sigma_X, V_X_batch=V_X_batch,
            images=images, mean=mean, rotation_matrices=rotation_matrices,
            translations=translations, ctf_params=ctf_params,
            noise_variance=noise_variance, S_left=S_left, Q_right=Q_right,
            hermitian_weights=hermitian_weights,
            n_images=n_images, volume_size=volume_size, sketch_rank=sketch_rank,
            qrank=qrank,
        )

    def test_right_sketch_matches_dense(self, setup):
        s = setup
        G_dense = _build_dense_gradient_batch_full(
            s["config"], s["U_X"], s["sigma_X"], s["V_X_batch"],
            s["images"], s["mean"], s["rotation_matrices"],
            s["translations"], s["ctf_params"], s["noise_variance"],
        )
        expected = G_dense @ s["Q_right"]  # (volume_size, qrank)

        residual, CTF_w = compute_residual_batch_from_factors(
            s["config"], s["U_X"], s["sigma_X"], s["V_X_batch"],
            s["images"], s["mean"], s["rotation_matrices"],
            s["translations"], s["ctf_params"], s["noise_variance"],
        )
        actual = right_sketch_normal_residual_batch(
            residual, s["Q_right"], CTF_w, s["rotation_matrices"],
            s["config"].image_shape, s["config"].volume_shape, s["config"].disc_type,
        )
        actual = actual.T  # (qrank, vol) -> (vol, qrank)

        # Tolerance for float32 differences between full-spectrum per-column
        # adjoint and half-image batch adjoint CUDA paths.
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected),
            atol=5e-3, rtol=5e-3,
            err_msg="Right sketch does not match dense reference",
        )

    def test_chunking_matches_unchunked(self, setup):
        """Verify chunked path (chunk_size=1) gives same result as unchunked."""
        s = setup
        residual, CTF_w = compute_residual_batch_from_factors(
            s["config"], s["U_X"], s["sigma_X"], s["V_X_batch"],
            s["images"], s["mean"], s["rotation_matrices"],
            s["translations"], s["ctf_params"], s["noise_variance"],
            pc_batch_size=1,
        )
        residual_unchunked, _ = compute_residual_batch_from_factors(
            s["config"], s["U_X"], s["sigma_X"], s["V_X_batch"],
            s["images"], s["mean"], s["rotation_matrices"],
            s["translations"], s["ctf_params"], s["noise_variance"],
            pc_batch_size=999,
        )
        np.testing.assert_allclose(
            np.asarray(residual), np.asarray(residual_unchunked),
            atol=1e-5, rtol=1e-5, err_msg="Chunked residual differs",
        )

        right_c = right_sketch_normal_residual_batch(
            residual, s["Q_right"], CTF_w, s["rotation_matrices"],
            s["config"].image_shape, s["config"].volume_shape, s["config"].disc_type,
            sketch_chunk_size=1,
        )
        right_u = right_sketch_normal_residual_batch(
            residual, s["Q_right"], CTF_w, s["rotation_matrices"],
            s["config"].image_shape, s["config"].volume_shape, s["config"].disc_type,
            sketch_chunk_size=999,
        )
        np.testing.assert_allclose(
            np.asarray(right_c), np.asarray(right_u),
            atol=1e-5, rtol=1e-5, err_msg="Chunked right sketch differs",
        )

        left_c = left_sketch_normal_residual_batch(
            s["config"], s["S_left"], residual, CTF_w, s["rotation_matrices"],
            s["ctf_params"], s["hermitian_weights"], sketch_chunk_size=1,
        )
        left_u = left_sketch_normal_residual_batch(
            s["config"], s["S_left"], residual, CTF_w, s["rotation_matrices"],
            s["ctf_params"], s["hermitian_weights"], sketch_chunk_size=999,
        )
        np.testing.assert_allclose(
            np.asarray(left_c), np.asarray(left_u),
            atol=1e-5, rtol=1e-5, err_msg="Chunked left sketch differs",
        )

    def test_left_sketch_matches_dense(self, setup):
        s = setup
        G_dense = _build_dense_gradient_batch_full(
            s["config"], s["U_X"], s["sigma_X"], s["V_X_batch"],
            s["images"], s["mean"], s["rotation_matrices"],
            s["translations"], s["ctf_params"], s["noise_variance"],
        )
        expected = s["S_left"] @ G_dense  # (sketch_rank, n_images)

        residual, CTF_w = compute_residual_batch_from_factors(
            s["config"], s["U_X"], s["sigma_X"], s["V_X_batch"],
            s["images"], s["mean"], s["rotation_matrices"],
            s["translations"], s["ctf_params"], s["noise_variance"],
        )
        actual = left_sketch_normal_residual_batch(
            s["config"], s["S_left"], residual, CTF_w, s["rotation_matrices"],
            s["ctf_params"], s["hermitian_weights"],
        )

        # Left sketch output is real (Hermitian contraction); expected is also real.
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected.real),
            atol=5e-3, rtol=5e-3,
            err_msg="Left sketch does not match dense reference",
        )


# ---------------------------------------------------------------------------
# Test 2: Cryo-EM path consistency — explicit per-column construction
# ---------------------------------------------------------------------------


def test_cryoem_path_explicit_columns():
    """Build G(X) columns explicitly and compare with sketch helpers."""
    rng = np.random.default_rng(99)
    grid_size = 6
    config = _make_config(grid_size=grid_size, disc_type="nearest")
    volume_size = int(np.prod(config.volume_shape))
    image_size = int(np.prod(config.image_shape))
    n_images = 3
    rank = 2

    # Hermitian-symmetric data
    U_X = jnp.array(np.stack([_hermitian_volume(rng, config.volume_shape) for _ in range(rank)], axis=-1))
    sigma_X = jnp.array(np.array([1.0, 0.5], dtype=np.float32))
    V_X_batch = jnp.array(rng.normal(size=(n_images, rank)).astype(np.float32))

    images = jnp.array(_hermitian_images(rng, n_images, config.image_shape))
    mean = jnp.zeros(volume_size, dtype=np.complex64)
    rotation_matrices = jnp.array(
        np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    )
    translations = jnp.zeros((n_images, 2), dtype=np.float32)
    ctf_params = jnp.zeros((n_images, 9), dtype=np.float32)
    noise_variance = jnp.ones((n_images, image_size), dtype=np.float32)

    # Explicitly construct each column and backproject in full-spectrum
    G_explicit = []
    for i in range(n_images):
        x_i = U_X @ jnp.diag(sigma_X) @ V_X_batch[i, :]
        predicted = core_forward.forward_model(
            config, x_i, ctf_params[i:i+1], rotation_matrices[i:i+1], skip_ctf=False
        )
        projected_mean = core_forward.forward_model(
            config, mean, ctf_params[i:i+1], rotation_matrices[i:i+1], skip_ctf=False
        )
        r_i = predicted - (images[i:i+1] - projected_mean)
        # CTF=1, noise=1 so CTF_w=1
        g_i = core_forward.adjoint_forward_model(
            config, r_i, ctf_params[i:i+1], rotation_matrices[i:i+1], skip_ctf=False
        )
        G_explicit.append(g_i)

    G_explicit = jnp.stack(G_explicit, axis=-1)

    # Compare with our helper (builds via half-image, converts back for reference)
    G_helper = _build_dense_gradient_batch_full(
        config, U_X, sigma_X, V_X_batch, images, mean,
        rotation_matrices, translations, ctf_params, noise_variance,
    )

    # Tolerance for float32 differences between code paths.
    np.testing.assert_allclose(
        np.asarray(G_helper), np.asarray(G_explicit),
        atol=2e-3, rtol=2e-3,
        err_msg="Helper gradient columns don't match explicit per-column construction",
    )


# ---------------------------------------------------------------------------
# Test 3: Zero residual gives zero output
# ---------------------------------------------------------------------------


def test_zero_residual_gives_zero():
    """If A(X) = b exactly, both sketch outputs must be zero."""
    rng = np.random.default_rng(7)
    grid_size = 6
    config = _make_config(grid_size=grid_size, disc_type="nearest")
    volume_size = int(np.prod(config.volume_shape))
    image_size = int(np.prod(config.image_shape))
    n_images = 4
    rank = 2
    sketch_rank = 3
    qrank = 2

    # Hermitian-symmetric volumes
    U_X = jnp.array(np.stack([_hermitian_volume(rng, config.volume_shape) for _ in range(rank)], axis=-1))
    sigma_X = jnp.array(np.array([1.0, 1.0], dtype=np.float32))
    V_X_batch = jnp.array(rng.normal(size=(n_images, rank)).astype(np.float32))

    mean = jnp.zeros(volume_size, dtype=np.complex64)
    rotation_matrices = jnp.array(
        np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    )
    translations = jnp.zeros((n_images, 2), dtype=np.float32)
    ctf_params = jnp.zeros((n_images, 9), dtype=np.float32)
    noise_variance = jnp.ones((n_images, image_size), dtype=np.float32)
    hermitian_weights = linalg.rfft2_hermitian_weights(config.image_shape)

    # Construct images so that A(X) = b exactly (forward model produces Hermitian output)
    images_list = []
    for i in range(n_images):
        x_i = U_X @ jnp.diag(sigma_X) @ V_X_batch[i, :]
        y_i = core_forward.forward_model(
            config, x_i, ctf_params[i:i+1], rotation_matrices[i:i+1], skip_ctf=False
        )
        images_list.append(y_i[0])
    images = jnp.stack(images_list, axis=0)

    residual, CTF_w = compute_residual_batch_from_factors(
        config, U_X, sigma_X, V_X_batch, images, mean,
        rotation_matrices, translations, ctf_params, noise_variance,
    )

    # Residual should be near zero.  Tolerance accounts for float32 differences
    # between per-image forward_model and batched batch_vol_forward_from_map,
    # plus half-image roundtrip.
    np.testing.assert_allclose(
        np.asarray(residual), 0.0, atol=2e-2,
        err_msg="Residual not zero when A(X)=b",
    )

    # Right sketch should be near zero
    Q_right = jnp.array(rng.normal(size=(n_images, qrank)).astype(np.float32))
    right_out = right_sketch_normal_residual_batch(
        residual, Q_right, CTF_w, rotation_matrices,
        config.image_shape, config.volume_shape, config.disc_type,
    )
    np.testing.assert_allclose(
        np.asarray(right_out), 0.0, atol=5e-2,
        err_msg="Right sketch not zero when residual is zero",
    )

    # Left sketch should be near zero
    S_left = jnp.array(np.stack([_hermitian_volume(rng, config.volume_shape) for _ in range(sketch_rank)]))
    left_out = left_sketch_normal_residual_batch(
        config, S_left, residual, CTF_w, rotation_matrices, ctf_params,
        hermitian_weights,
    )
    # Left sketch amplifies residual error by sketch norm * image_size,
    # so tolerance scales with sketch size.
    np.testing.assert_allclose(
        np.asarray(left_out), 0.0, atol=5e-1,
        err_msg="Left sketch not zero when residual is zero",
    )


# ---------------------------------------------------------------------------
# Test 4: Linearity in residual
# ---------------------------------------------------------------------------


class TestLinearity:
    """Check homogeneity and additivity of the sketch operators w.r.t. residual."""

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(123)
        grid_size = 6
        config = _make_config(grid_size=grid_size, disc_type="nearest")
        volume_size = int(np.prod(config.volume_shape))
        half_image_size = int(np.prod(ftu.image_shape_to_half_image_shape(config.image_shape)))
        n_images = 4
        sketch_rank = 3
        qrank = 2

        rotation_matrices = jnp.array(_random_rotation_matrices(rng, n_images))
        ctf_params = jnp.zeros((n_images, 9), dtype=np.float32)

        CTF_w = jnp.ones((n_images, half_image_size), dtype=np.float32)
        hermitian_weights = linalg.rfft2_hermitian_weights(config.image_shape)

        # Hermitian residuals in half-image format (from real-space data)
        r1_full = _hermitian_images(rng, n_images, config.image_shape)
        r2_full = _hermitian_images(rng, n_images, config.image_shape)
        r1 = jnp.array(ftu.full_image_to_half_image(r1_full, config.image_shape))
        r2 = jnp.array(ftu.full_image_to_half_image(r2_full, config.image_shape))

        Q_right = jnp.array(rng.normal(size=(n_images, qrank)).astype(np.float32))
        S_left = jnp.array(np.stack([_hermitian_volume(rng, config.volume_shape) for _ in range(sketch_rank)]))

        return dict(
            config=config, rotation_matrices=rotation_matrices,
            ctf_params=ctf_params, CTF_w=CTF_w,
            hermitian_weights=hermitian_weights,
            r1=r1, r2=r2, Q_right=Q_right, S_left=S_left,
        )

    def test_right_sketch_homogeneity(self, setup):
        s = setup
        alpha = 2.5

        out1 = right_sketch_normal_residual_batch(
            s["r1"], s["Q_right"], s["CTF_w"], s["rotation_matrices"],
            s["config"].image_shape, s["config"].volume_shape, s["config"].disc_type,
        )
        out_scaled = right_sketch_normal_residual_batch(
            alpha * s["r1"], s["Q_right"], s["CTF_w"], s["rotation_matrices"],
            s["config"].image_shape, s["config"].volume_shape, s["config"].disc_type,
        )

        np.testing.assert_allclose(
            np.asarray(out_scaled), alpha * np.asarray(out1),
            atol=1e-5, rtol=1e-5,
        )

    def test_right_sketch_additivity(self, setup):
        s = setup

        out1 = right_sketch_normal_residual_batch(
            s["r1"], s["Q_right"], s["CTF_w"], s["rotation_matrices"],
            s["config"].image_shape, s["config"].volume_shape, s["config"].disc_type,
        )
        out2 = right_sketch_normal_residual_batch(
            s["r2"], s["Q_right"], s["CTF_w"], s["rotation_matrices"],
            s["config"].image_shape, s["config"].volume_shape, s["config"].disc_type,
        )
        out_sum = right_sketch_normal_residual_batch(
            s["r1"] + s["r2"], s["Q_right"], s["CTF_w"], s["rotation_matrices"],
            s["config"].image_shape, s["config"].volume_shape, s["config"].disc_type,
        )

        np.testing.assert_allclose(
            np.asarray(out_sum), np.asarray(out1 + out2),
            atol=1e-5, rtol=1e-5,
        )

    def test_left_sketch_homogeneity(self, setup):
        s = setup
        alpha = 3.0

        out1 = left_sketch_normal_residual_batch(
            s["config"], s["S_left"], s["r1"], s["CTF_w"], s["rotation_matrices"],
            s["ctf_params"], s["hermitian_weights"],
        )
        out_scaled = left_sketch_normal_residual_batch(
            s["config"], s["S_left"], alpha * s["r1"], s["CTF_w"], s["rotation_matrices"],
            s["ctf_params"], s["hermitian_weights"],
        )

        np.testing.assert_allclose(
            np.asarray(out_scaled), alpha * np.asarray(out1),
            atol=1e-4, rtol=1e-4,
        )

    def test_left_sketch_additivity(self, setup):
        s = setup

        out1 = left_sketch_normal_residual_batch(
            s["config"], s["S_left"], s["r1"], s["CTF_w"], s["rotation_matrices"],
            s["ctf_params"], s["hermitian_weights"],
        )
        out2 = left_sketch_normal_residual_batch(
            s["config"], s["S_left"], s["r2"], s["CTF_w"], s["rotation_matrices"],
            s["ctf_params"], s["hermitian_weights"],
        )
        out_sum = left_sketch_normal_residual_batch(
            s["config"], s["S_left"], s["r1"] + s["r2"], s["CTF_w"], s["rotation_matrices"],
            s["ctf_params"], s["hermitian_weights"],
        )

        np.testing.assert_allclose(
            np.asarray(out_sum), np.asarray(out1 + out2),
            atol=1e-4, rtol=1e-4,
        )
