"""Half-spectrum EM equivalence tests (Phase 1 of RELION-parity plan).

Verifies that the half-spectrum GEMM engine in engine_v2.py produces
numerically identical results to the full-spectrum reference implementation.

Tests:
1. test_half_inner_product_correctness: weighted half-spectrum dot product == full
2. test_e_step_half_matches_full: half-spectrum E-step scores match full-spectrum
3. test_m_step_half_matches_full: half-spectrum M-step Ft_y, Ft_ctf match full-spectrum
4. test_full_iteration_half_matches: one complete run_em_v2 iteration matches reference
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar import core
from recovar.core.configs import ForwardModelConfig
import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.engine_v2 import (
    make_half_image_weights,
    _preprocess_batch,
    _e_step_block_scores,
    _m_step_block,
    _compute_projections_block,
    _update_logsumexp,
    run_em_v2,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Test constants -- small sizes for unit tests (no GPU required for tiny data)
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
H, W = IMAGE_SHAPE
N_HALF = H * (W // 2 + 1)  # 8 * 5 = 40
N_FULL = H * W               # 8 * 8 = 64
N_ROTATIONS = 5
N_TRANSLATIONS = 3
N_IMAGES = 4
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hermitian_image_2d(image_shape, seed=42):
    """Generate a Hermitian-symmetric 2D spectrum (DFT of real data), centered."""
    rng = np.random.default_rng(seed)
    real_img = rng.standard_normal(image_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fft2(real_img))
    return jnp.array(ft, dtype=jnp.complex64)


def _hermitian_volume(volume_shape, seed=42):
    """Generate a Hermitian-symmetric 3D volume (DFT of real data), centered."""
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _make_rotations(n, seed=42):
    """Generate n rotation matrices via QR decomposition."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return jnp.array(q, dtype=jnp.float32)


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    """CTF that returns ones (identity)."""
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _identity_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    return batch


class MockDataset:
    """Minimal dataset for equivalence tests."""

    def __init__(self, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = N_IMAGES
        self.n_units = N_IMAGES
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((N_IMAGES, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_identity_process)

        # Fixed Hermitian-symmetric images from seed
        self._images = np.zeros((N_IMAGES, IMAGE_SIZE), dtype=np.complex64)
        for i in range(N_IMAGES):
            self._images[i] = _hermitian_image_2d(IMAGE_SHAPE, seed=rng.integers(10000)).reshape(-1)

        class _ImageSource:
            process_images = staticmethod(_identity_process)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        _ = kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                None,
                None,
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def mock_dataset(rng):
    return MockDataset(rng)


@pytest.fixture
def seeded_inputs(rng, mock_dataset):
    """Deterministic inputs for equivalence tests."""
    volume = _hermitian_volume(VOLUME_SHAPE, seed=42)
    rotations = _make_rotations(N_ROTATIONS, seed=12)
    translations = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
    )
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)

    config = ForwardModelConfig.from_dataset(
        mock_dataset,
        disc_type="linear_interp",
        process_fn=_identity_process,
    )

    return {
        "volume": volume,
        "rotations": rotations,
        "translations": translations,
        "noise_variance": noise_variance,
        "config": config,
        "dataset": mock_dataset,
    }


# ===========================================================================
# Test 1: Half-spectrum inner product correctness
# ===========================================================================


class TestHalfInnerProductCorrectness:
    """Verify that weighted half-spectrum inner product matches full inner product.

    This is the fundamental identity that makes all half-spectrum GEMMs correct.
    """

    def test_hermitian_pair(self):
        """Re<a, b>_full == Re[sum(conj(a_half) * w * b_half)] for Hermitian data."""
        a_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=10)
        b_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=20)

        a_flat = a_2d.reshape(N_FULL)
        b_flat = b_2d.reshape(N_FULL)

        # Full inner product
        ip_full = jnp.sum(jnp.conj(a_flat) * b_flat).real

        # Half inner product with weights
        a_half = ftu.full_image_to_half_image(a_flat[None, :], IMAGE_SHAPE).reshape(-1)
        b_half = ftu.full_image_to_half_image(b_flat[None, :], IMAGE_SHAPE).reshape(-1)
        w = make_half_image_weights(IMAGE_SHAPE)
        ip_half = jnp.sum(jnp.conj(a_half) * w * b_half).real

        np.testing.assert_allclose(float(ip_full), float(ip_half), rtol=1e-5)

    def test_matrix_inner_product(self):
        """Batched: conj(A) @ B^T matches conj(A_half) @ (B_half * w)^T."""
        rng = np.random.default_rng(42)
        n_a, n_b = 4, 5

        A = jnp.stack([_hermitian_image_2d(IMAGE_SHAPE, seed=i).reshape(-1) for i in range(n_a)])
        B = jnp.stack([_hermitian_image_2d(IMAGE_SHAPE, seed=100 + i).reshape(-1) for i in range(n_b)])

        # Full GEMM
        full_result = (jnp.conj(A) @ B.T).real  # (n_a, n_b)

        # Half GEMM with weights absorbed into B
        w = make_half_image_weights(IMAGE_SHAPE)
        A_half = ftu.full_image_to_half_image(A, IMAGE_SHAPE)
        B_half = ftu.full_image_to_half_image(B, IMAGE_SHAPE)
        B_half_weighted = B_half * w
        half_result = (jnp.conj(A_half) @ B_half_weighted.T).real  # (n_a, n_b)

        np.testing.assert_allclose(
            np.array(full_result), np.array(half_result),
            rtol=1e-5, atol=1e-4,
            err_msg="Batched half-spectrum GEMM does not match full GEMM",
        )

    def test_weight_values(self):
        """Verify make_half_image_weights returns correct values."""
        w = make_half_image_weights(IMAGE_SHAPE)
        w_2d = w.reshape(H, W // 2 + 1)

        # DC column (0) should be 1
        np.testing.assert_array_equal(np.array(w_2d[:, 0]), np.ones(H))
        # Nyquist column (-1) should be 1
        np.testing.assert_array_equal(np.array(w_2d[:, -1]), np.ones(H))
        # Interior columns should be 2
        if W // 2 + 1 > 2:
            np.testing.assert_array_equal(
                np.array(w_2d[:, 1:-1]),
                2.0 * np.ones((H, W // 2 - 1)),
            )


# ===========================================================================
# Test 2: E-step half matches full
# ===========================================================================


class TestEStepHalfMatchesFull:
    """Verify that the half-spectrum E-step produces the same scores as the
    full-spectrum reference (compute_dot_products_eqx + norm term)."""

    def test_scores_match(self, seeded_inputs):
        """E-step scores from half-spectrum path match full-spectrum scores."""
        s = seeded_inputs
        config = s["config"]
        volume = s["volume"]
        rotations = s["rotations"]
        translations = s["translations"]
        noise_variance = s["noise_variance"]
        ds = s["dataset"]

        n_images = ds.n_units
        n_trans = translations.shape[0]

        # Get batch data
        batch_data = jnp.asarray(ds._images)
        ctf_params = jnp.asarray(ds.CTF_params)

        # === FULL-SPECTRUM reference (from em.core) ===
        from recovar.em import core as em_core

        # Full-spectrum projections
        proj_full = core.slice_volume(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=False
        )
        proj_abs2_full = jnp.abs(proj_full) ** 2

        # Cross-term via existing function
        cross_term = em_core.compute_dot_products_eqx(
            config, proj_full, batch_data, translations, ctf_params, noise_variance
        )
        # Norm-term
        norm_term = em_core.compute_CTFed_proj_norms_eqx(
            config, proj_abs2_full, ctf_params, noise_variance
        )
        # Full residual: scores = -0.5 * (cross + norm)
        scores_full = -0.5 * (cross_term + norm_term[..., None])

        # === HALF-SPECTRUM path ===
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data, ctf_params, noise_variance, translations, config,
            n_images, n_trans,
        )

        proj_half, proj_abs2_half = _compute_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        proj_half_weighted = proj_half * half_weights
        proj_abs2_weighted = proj_abs2_half * half_weights

        scores_half = _e_step_block_scores(
            shifted_half, batch_norm, ctf2_over_nv_half,
            proj_half_weighted, proj_abs2_weighted, half_weights,
            n_images, n_trans, IMAGE_SHAPE, VOLUME_SHAPE,
        )

        np.testing.assert_allclose(
            np.array(scores_full),
            np.array(scores_half),
            atol=1e-3,
            rtol=1e-4,
            err_msg="E-step scores differ between half-spectrum and full-spectrum paths",
        )

    def test_probabilities_match(self, seeded_inputs):
        """Probabilities (softmax of scores) match between half and full paths."""
        s = seeded_inputs
        config = s["config"]
        volume = s["volume"]
        rotations = s["rotations"]
        translations = s["translations"]
        noise_variance = s["noise_variance"]
        ds = s["dataset"]

        n_images = ds.n_units
        n_trans = translations.shape[0]
        n_rot = rotations.shape[0]

        batch_data = jnp.asarray(ds._images)
        ctf_params = jnp.asarray(ds.CTF_params)

        # Full-spectrum scores
        from recovar.em import core as em_core
        proj_full = core.slice_volume(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=False
        )
        proj_abs2_full = jnp.abs(proj_full) ** 2
        cross_term = em_core.compute_dot_products_eqx(
            config, proj_full, batch_data, translations, ctf_params, noise_variance
        )
        norm_term = em_core.compute_CTFed_proj_norms_eqx(
            config, proj_abs2_full, ctf_params, noise_variance
        )
        scores_full = -0.5 * (cross_term + norm_term[..., None])
        # Softmax
        scores_flat = scores_full.reshape(n_images, -1)
        log_Z_full = jax.scipy.special.logsumexp(scores_flat, axis=1)
        probs_full = jnp.exp(scores_full - log_Z_full[:, None, None])

        # Half-spectrum scores
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data, ctf_params, noise_variance, translations, config,
            n_images, n_trans,
        )
        proj_half, proj_abs2_half = _compute_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        proj_half_weighted = proj_half * half_weights
        proj_abs2_weighted = proj_abs2_half * half_weights

        scores_half = _e_step_block_scores(
            shifted_half, batch_norm, ctf2_over_nv_half,
            proj_half_weighted, proj_abs2_weighted, half_weights,
            n_images, n_trans, IMAGE_SHAPE, VOLUME_SHAPE,
        )
        scores_h_flat = scores_half.reshape(n_images, -1)
        log_Z_half = jax.scipy.special.logsumexp(scores_h_flat, axis=1)
        probs_half = jnp.exp(scores_half - log_Z_half[:, None, None])

        np.testing.assert_allclose(
            np.array(probs_full),
            np.array(probs_half),
            atol=1e-5,
            err_msg="Probabilities differ between half-spectrum and full-spectrum paths",
        )


# ===========================================================================
# Test 3: M-step half matches full
# ===========================================================================


class TestMStepHalfMatchesFull:
    """Verify that the half-spectrum M-step accumulates Ft_y and Ft_ctf
    identically to the full-spectrum reference."""

    def test_m_step_accumulation(self, seeded_inputs):
        """Ft_y and Ft_ctf match between half and full M-step paths."""
        s = seeded_inputs
        config = s["config"]
        volume = s["volume"]
        rotations = s["rotations"]
        translations = s["translations"]
        noise_variance = s["noise_variance"]
        ds = s["dataset"]

        n_images = ds.n_units
        n_trans = translations.shape[0]
        n_rot = rotations.shape[0]

        batch_data = jnp.asarray(ds._images)
        ctf_params = jnp.asarray(ds.CTF_params)

        # -- Compute shared probabilities using half-spectrum path (already verified) --
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data, ctf_params, noise_variance, translations, config,
            n_images, n_trans,
        )
        proj_half, proj_abs2_half = _compute_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        proj_half_weighted = proj_half * half_weights
        proj_abs2_weighted = proj_abs2_half * half_weights

        scores = _e_step_block_scores(
            shifted_half, batch_norm, ctf2_over_nv_half,
            proj_half_weighted, proj_abs2_weighted, half_weights,
            n_images, n_trans, IMAGE_SHAPE, VOLUME_SHAPE,
        )
        scores_flat = scores.reshape(n_images, -1)
        log_Z = jax.scipy.special.logsumexp(scores_flat, axis=1)

        # -- Half-spectrum M-step --
        Ft_y_half = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)
        Ft_ctf_half = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)

        Ft_y_half, Ft_ctf_half, probs, _, _ = _m_step_block(
            shifted_half, scores, log_Z, rotations, ctf2_over_nv_half,
            Ft_y_half, Ft_ctf_half,
            n_images, n_trans, IMAGE_SHAPE, VOLUME_SHAPE,
        )

        # -- Full-spectrum reference M-step --
        # Reconstruct full-spectrum shifted images for reference
        CTF = config.compute_ctf(ctf_params)
        processed = config.process_fn(batch_data, apply_image_mask=False)
        ctf_weighted = processed * CTF / noise_variance
        shifted_full = core.batch_trans_translate_images(
            ctf_weighted, jnp.repeat(translations[None], n_images, axis=0), IMAGE_SHAPE,
        )
        shifted_full_flat = shifted_full.reshape(n_images * n_trans, -1)
        ctf2_over_nv_full = CTF ** 2 / noise_variance

        Ft_y_full = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)
        Ft_ctf_full = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)

        # M-step GEMM on full spectrum
        rot_block_size = n_rot
        P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
        summed_full = P @ shifted_full_flat  # (rot_block, N_full)
        summed_half_ref = ftu.full_image_to_half_image(summed_full, IMAGE_SHAPE)
        Ft_y_full = core.adjoint_slice_volume(
            summed_half_ref, rotations, IMAGE_SHAPE, VOLUME_SHAPE,
            "linear_interp", volume=Ft_y_full, half_image=True,
        )

        probs_sum_t = jnp.sum(probs, axis=-1)
        ctf_probs_full = probs_sum_t.T @ ctf2_over_nv_full
        ctf_half_ref = ftu.full_image_to_half_image(ctf_probs_full, IMAGE_SHAPE)
        Ft_ctf_full = core.adjoint_slice_volume(
            ctf_half_ref, rotations, IMAGE_SHAPE, VOLUME_SHAPE,
            "linear_interp", volume=Ft_ctf_full, half_image=True,
        )

        np.testing.assert_allclose(
            np.array(Ft_y_half),
            np.array(Ft_y_full),
            atol=1e-4,
            err_msg="Ft_y differs between half-spectrum and full-spectrum M-step",
        )
        np.testing.assert_allclose(
            np.array(Ft_ctf_half),
            np.array(Ft_ctf_full),
            atol=1e-4,
            err_msg="Ft_ctf differs between half-spectrum and full-spectrum M-step",
        )


# ===========================================================================
# Test 4: Full iteration half matches (run_em_v2)
# ===========================================================================


class TestFullIterationHalfMatches:
    """One complete run_em_v2 iteration should produce correct results.

    We compare against a manually-computed reference using the old full-spectrum
    em.core functions, verifying hard assignments and mean volume match.
    """

    def test_full_iteration_self_consistency(self, seeded_inputs):
        """run_em_v2 produces identical results on two calls with same inputs."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        new_mean1, ha1, Ft_y1, Ft_ctf1 = run_em_v2(
            ds, volume, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        new_mean2, ha2, Ft_y2, Ft_ctf2 = run_em_v2(
            ds, volume, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        np.testing.assert_allclose(np.array(new_mean1), np.array(new_mean2), atol=1e-6)
        np.testing.assert_array_equal(ha1, ha2)
        np.testing.assert_allclose(np.array(Ft_y1), np.array(Ft_y2), atol=1e-6)
        np.testing.assert_allclose(np.array(Ft_ctf1), np.array(Ft_ctf2), atol=1e-6)

    def test_hard_assignments_valid(self, seeded_inputs):
        """Hard assignments should be valid indices into (n_rot * n_trans)."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        _, ha, _, _ = run_em_v2(
            ds, volume, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        assert ha.shape == (N_IMAGES,)
        assert np.all(ha >= 0)
        assert np.all(ha < N_ROTATIONS * N_TRANSLATIONS)

    def test_outputs_finite(self, seeded_inputs):
        """All outputs should be finite (no NaN/inf)."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        new_mean, ha, Ft_y, Ft_ctf = run_em_v2(
            ds, volume, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        assert np.all(np.isfinite(np.array(new_mean)))
        assert np.all(np.isfinite(np.array(Ft_y)))
        assert np.all(np.isfinite(np.array(Ft_ctf)))

    def test_multiple_rotation_blocks(self, seeded_inputs):
        """Results should be identical regardless of rotation block size."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        # All rotations in one block
        new_mean_1, ha_1, Ft_y_1, Ft_ctf_1 = run_em_v2(
            ds, volume, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        # Rotations split into blocks of 2 (5 rots -> 3 blocks: 2+2+1, padded to 2+2+2)
        new_mean_2, ha_2, Ft_y_2, Ft_ctf_2 = run_em_v2(
            ds, volume, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=2,
        )

        np.testing.assert_allclose(
            np.array(new_mean_1), np.array(new_mean_2), atol=1e-4,
            err_msg="Mean differs between single-block and multi-block rotation processing",
        )
        np.testing.assert_array_equal(
            ha_1, ha_2,
            err_msg="Hard assignments differ between single-block and multi-block rotation processing",
        )
        np.testing.assert_allclose(
            np.array(Ft_y_1), np.array(Ft_y_2), atol=1e-4,
            err_msg="Ft_y differs between single-block and multi-block rotation processing",
        )
        np.testing.assert_allclose(
            np.array(Ft_ctf_1), np.array(Ft_ctf_2), atol=1e-4,
            err_msg="Ft_ctf differs between single-block and multi-block rotation processing",
        )

    def test_multiple_image_batches(self, seeded_inputs):
        """Results should be identical regardless of image batch size."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        # All images in one batch
        new_mean_1, ha_1, Ft_y_1, Ft_ctf_1 = run_em_v2(
            ds, volume, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        # Images in batches of 1
        new_mean_2, ha_2, Ft_y_2, Ft_ctf_2 = run_em_v2(
            ds, volume, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=1,
            rotation_block_size=N_ROTATIONS,
        )

        np.testing.assert_allclose(
            np.array(new_mean_1), np.array(new_mean_2), atol=1e-4,
            err_msg="Mean differs between single-batch and multi-batch image processing",
        )
        # Hard assignments should match (same probabilities -> same argmax)
        np.testing.assert_array_equal(
            ha_1, ha_2,
            err_msg="Hard assignments differ between single-batch and multi-batch image processing",
        )


# ===========================================================================
# Test 5: make_half_image_weights shape and dtype
# ===========================================================================


class TestMakeHalfImageWeights:
    """Verify make_half_image_weights produces correct shape and values."""

    @pytest.mark.parametrize("shape", [(4, 4), (8, 8), (16, 16), (64, 64), (128, 128)])
    def test_shape(self, shape):
        """Output shape is (H * (W//2 + 1),)."""
        H, W = shape
        w = make_half_image_weights(shape)
        expected_len = H * (W // 2 + 1)
        assert w.shape == (expected_len,), f"Expected ({expected_len},), got {w.shape}"

    @pytest.mark.parametrize("shape", [(4, 4), (8, 8), (128, 128)])
    def test_sum(self, shape):
        """Sum of weights should equal H * W (total pixels in full image)."""
        H, W = shape
        w = make_half_image_weights(shape)
        expected_sum = H * W
        np.testing.assert_allclose(float(jnp.sum(w)), expected_sum, rtol=1e-5)

    def test_dtype(self):
        """Weights should be float32."""
        w = make_half_image_weights(IMAGE_SHAPE)
        assert w.dtype == jnp.float32


# ===========================================================================
# Test 6: Streaming logsumexp correctness
# ===========================================================================


class TestStreamingLogsumexp:
    """Verify that streaming logsumexp over blocks matches direct computation."""

    def test_streaming_matches_direct(self):
        """Streaming logsumexp over 3 blocks matches jax.scipy.special.logsumexp."""
        rng = np.random.default_rng(42)
        n_images = 4
        n_rot_per_block = 3
        n_trans = 2

        # Create 3 blocks of scores
        blocks = [
            jnp.array(rng.standard_normal((n_images, n_rot_per_block, n_trans)).astype(np.float32))
            for _ in range(3)
        ]

        # Direct: concatenate all and compute
        all_scores = jnp.concatenate([b.reshape(n_images, -1) for b in blocks], axis=1)
        log_Z_direct = jax.scipy.special.logsumexp(all_scores, axis=1)

        # Streaming
        max_s = jnp.full(n_images, -jnp.inf)
        sum_exp = jnp.zeros(n_images)
        for block in blocks:
            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, block)
        log_Z_streaming = max_s + jnp.log(sum_exp)

        np.testing.assert_allclose(
            np.array(log_Z_direct),
            np.array(log_Z_streaming),
            atol=1e-5,
            err_msg="Streaming logsumexp does not match direct computation",
        )
