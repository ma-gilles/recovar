"""Half-spectrum EM equivalence tests (Phase 1 of RELION-parity plan).

Verifies that the half-spectrum GEMM engine in em_engine.py produces
numerically identical results to the full-spectrum reference implementation.

Tests:
1. test_half_inner_product_correctness: weighted half-spectrum dot product == full
2. test_e_step_half_matches_full: half-spectrum E-step scores match full-spectrum
3. test_m_step_half_matches_full: half-spectrum M-step Ft_y, Ft_ctf match full-spectrum
4. test_full_iteration_half_matches: one complete run_em iteration matches reference
"""

import logging

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.core.configs import ForwardModelConfig
import recovar.em.dense_single_volume.em_engine as em_engine_module
import recovar.em.dense_single_volume.iteration_loop as iteration_loop_module
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_half as _adjoint_slice_volume_half,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
from recovar.em.dense_single_volume.helpers.preprocessing import (
    preprocess_batch as _preprocess_batch,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_projections_block as _compute_projections_block,
)
from recovar.em.dense_single_volume.helpers.scoring import (
    _e_step_block_scores,
    _m_step_block_compute,
    _update_logsumexp,
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
N_FULL = H * W  # 8 * 8 = 64
N_ROTATIONS = 5
N_TRANSLATIONS = 3
N_IMAGES = 4
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_real_image_2d(image_shape, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(image_shape).astype(np.float32)


def _hermitian_image_2d(image_shape, seed=42):
    """Generate a Hermitian-symmetric 2D spectrum (DFT of real data), centered."""
    ft = np.fft.fftshift(np.fft.fft2(_raw_real_image_2d(image_shape, seed=seed)))
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


def _raw_real_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _mask_for_score_only_process(batch, apply_image_mask=False):
    if apply_image_mask:
        return jnp.zeros((jnp.asarray(batch).shape[0], IMAGE_SIZE), dtype=jnp.complex64)
    return _raw_real_process(batch, apply_image_mask=False)


def _mask_for_score_only_process_half(batch, apply_image_mask=False):
    if apply_image_mask:
        return jnp.zeros((jnp.asarray(batch).shape[0], N_HALF), dtype=jnp.complex64)
    return _raw_real_process_half(batch, apply_image_mask=False)


def _constant_half_noise_variance(noise_variance):
    noise_variance = jnp.asarray(noise_variance)
    if noise_variance.shape[-1] == N_HALF:
        return noise_variance
    values = np.asarray(noise_variance).reshape(-1)
    if values.size != IMAGE_SIZE or not np.allclose(values, values[0]):
        raise ValueError("test helper only supports scalar full-image noise variance")
    return jnp.full((N_HALF,), values[0], dtype=noise_variance.dtype)


def _preprocess_test_batch(
    dataset,
    batch_data,
    ctf_params,
    noise_variance,
    translations,
    config,
    *,
    score_with_masked_images=False,
):
    return _preprocess_batch(
        dataset,
        batch_data,
        ctf_params,
        _constant_half_noise_variance(noise_variance),
        translations,
        config,
        score_with_masked_images=score_with_masked_images,
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        norm_real_dtype=jnp.float64,
    )


def _compute_jax_projections_block(volume, rotations, image_shape, volume_shape, disc_type):
    return _compute_projections_block(
        volume,
        rotations,
        image_shape,
        volume_shape,
        disc_type,
        relion_texture_interp=False,
    )


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
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)

        self._images = np.zeros((N_IMAGES, *IMAGE_SHAPE), dtype=np.float32)
        for i in range(N_IMAGES):
            self._images[i] = _raw_real_image_2d(IMAGE_SHAPE, seed=rng.integers(10000))

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)

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


class MaskedScoringDataset(MockDataset):
    """Dataset where masked scoring sees zeros but reconstruction sees data."""

    def __init__(self, rng):
        super().__init__(rng)
        self.process_images = staticmethod(_mask_for_score_only_process)
        self.process_images_half = staticmethod(_mask_for_score_only_process_half)

        class _ImageSource:
            process_images = staticmethod(_mask_for_score_only_process)
            process_images_half = staticmethod(_mask_for_score_only_process_half)

        self.image_source = _ImageSource()


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
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)

    config = ForwardModelConfig.from_dataset(
        mock_dataset,
        disc_type="linear_interp",
        process_fn=mock_dataset.process_images,
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
            np.array(full_result),
            np.array(half_result),
            rtol=1e-5,
            atol=1e-4,
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
        proj_full = core.slice_volume(volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=False)
        proj_abs2_full = jnp.abs(proj_full) ** 2

        # Cross-term via existing function
        cross_term = em_core.compute_dot_products_eqx(
            config, proj_full, batch_data, translations, ctf_params, noise_variance
        )
        # Norm-term
        norm_term = em_core.compute_CTFed_proj_norms_eqx(config, proj_abs2_full, ctf_params, noise_variance)
        # Full residual: scores = -0.5 * (cross + norm)
        scores_full = -0.5 * (cross_term + norm_term[..., None])

        # === HALF-SPECTRUM path ===
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_test_batch(
            ds,
            batch_data,
            ctf_params,
            noise_variance,
            translations,
            config,
        )

        proj_half, proj_abs2_half = _compute_jax_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        proj_half_weighted = proj_half * half_weights
        proj_abs2_weighted = proj_abs2_half * half_weights

        scores_half = _e_step_block_scores(
            shifted_half,
            batch_norm,
            ctf2_over_nv_half,
            proj_half_weighted,
            proj_abs2_weighted,
            half_weights,
            n_images,
            n_trans,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
        )

        score_offset = 0.5 * np.asarray(batch_norm).reshape(n_images, 1, 1)
        np.testing.assert_allclose(
            np.array(scores_full) + score_offset,
            np.array(scores_half),
            atol=1e-3,
            rtol=5e-4,
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

        proj_full = core.slice_volume(volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=False)
        proj_abs2_full = jnp.abs(proj_full) ** 2
        cross_term = em_core.compute_dot_products_eqx(
            config, proj_full, batch_data, translations, ctf_params, noise_variance
        )
        norm_term = em_core.compute_CTFed_proj_norms_eqx(config, proj_abs2_full, ctf_params, noise_variance)
        scores_full = -0.5 * (cross_term + norm_term[..., None])
        # Softmax
        scores_flat = scores_full.reshape(n_images, -1)
        log_Z_full = jax.scipy.special.logsumexp(scores_flat, axis=1)
        probs_full = jnp.exp(scores_full - log_Z_full[:, None, None])

        # Half-spectrum scores
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_test_batch(
            ds,
            batch_data,
            ctf_params,
            noise_variance,
            translations,
            config,
        )
        proj_half, proj_abs2_half = _compute_jax_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        proj_half_weighted = proj_half * half_weights
        proj_abs2_weighted = proj_abs2_half * half_weights

        scores_half = _e_step_block_scores(
            shifted_half,
            batch_norm,
            ctf2_over_nv_half,
            proj_half_weighted,
            proj_abs2_weighted,
            half_weights,
            n_images,
            n_trans,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
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
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_test_batch(
            ds,
            batch_data,
            ctf_params,
            noise_variance,
            translations,
            config,
        )
        proj_half, proj_abs2_half = _compute_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        proj_half_weighted = proj_half * half_weights
        proj_abs2_weighted = proj_abs2_half * half_weights

        scores = _e_step_block_scores(
            shifted_half,
            batch_norm,
            ctf2_over_nv_half,
            proj_half_weighted,
            proj_abs2_weighted,
            half_weights,
            n_images,
            n_trans,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
        )
        scores_flat = scores.reshape(n_images, -1)
        log_Z = jax.scipy.special.logsumexp(scores_flat, axis=1)

        # -- Half-spectrum M-step --
        Ft_y_half = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)
        Ft_ctf_half = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)

        probs, _, _, summed_half, ctf_probs_half = _m_step_block_compute(
            shifted_half,
            scores,
            log_Z,
            rotations,
            ctf2_over_nv_half,
            n_images,
            n_trans,
        )
        Ft_y_half = _adjoint_slice_volume_half(
            summed_half,
            rotations,
            Ft_y_half,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            True,
        )
        Ft_ctf_half = _adjoint_slice_volume_half(
            ctf_probs_half,
            rotations,
            Ft_ctf_half,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            True,
        )

        # -- Full-spectrum reference M-step --
        # Reconstruct full-spectrum shifted images for reference
        CTF = config.compute_ctf(ctf_params)
        processed = config.process_fn(batch_data, apply_image_mask=False)
        ctf_weighted = processed * CTF / noise_variance
        shifted_full = core.batch_trans_translate_images(
            ctf_weighted,
            jnp.repeat(translations[None], n_images, axis=0),
            IMAGE_SHAPE,
        )
        shifted_full_flat = shifted_full.reshape(n_images * n_trans, -1)
        ctf2_over_nv_full = CTF**2 / noise_variance

        Ft_y_full = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)
        Ft_ctf_full = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)

        # M-step GEMM on full spectrum
        rot_block_size = n_rot
        P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
        summed_full = P @ shifted_full_flat  # (rot_block, N_full)
        summed_half_ref = ftu.full_image_to_half_image(summed_full, IMAGE_SHAPE)
        Ft_y_full = core.adjoint_slice_volume(
            summed_half_ref,
            rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            volume=Ft_y_full,
            half_image=True,
        )

        probs_sum_t = jnp.sum(probs, axis=-1)
        ctf_probs_full = probs_sum_t.T @ ctf2_over_nv_full
        ctf_half_ref = ftu.full_image_to_half_image(ctf_probs_full, IMAGE_SHAPE)
        Ft_ctf_full = core.adjoint_slice_volume(
            ctf_half_ref,
            rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            volume=Ft_ctf_full,
            half_image=True,
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
# Test 4: Full iteration half matches (run_em)
# ===========================================================================


class TestFullIterationHalfMatches:
    """One complete run_em iteration should produce correct results.

    We compare against a manually-computed reference using the old full-spectrum
    em.core functions, verifying hard assignments and mean volume match.
    """

    def test_full_iteration_self_consistency(self, seeded_inputs):
        """run_em produces identical results on two calls with same inputs."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        new_mean1, ha1, Ft_y1, Ft_ctf1 = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        new_mean2, ha2, Ft_y2, Ft_ctf2 = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
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

        _, ha, _, _ = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
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

        new_mean, ha, Ft_y, Ft_ctf = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        assert np.all(np.isfinite(np.array(new_mean)))
        assert np.all(np.isfinite(np.array(Ft_y)))
        assert np.all(np.isfinite(np.array(Ft_ctf)))

    def test_return_stats_matches_direct_pmax(self, seeded_inputs):
        """Optional return_stats path should expose the exact batchwise E-step maxima."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        _, hard_assignments, _, _, stats = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            return_stats=True,
        )

        batch_data, _, _, ctf_params, _, _, _ = next(
            ds.iter_batches(N_IMAGES, indices=np.arange(N_IMAGES), by_image=False)
        )
        config = ForwardModelConfig.from_dataset(
            ds,
            disc_type="linear_interp",
            process_fn=ds.process_images,
        )
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_test_batch(
            ds,
            jnp.asarray(batch_data),
            jnp.asarray(ctf_params),
            noise_variance,
            translations,
            config,
        )
        proj_half, proj_abs2_half = _compute_projections_block(
            volume,
            rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        scores = _e_step_block_scores(
            shifted_half,
            batch_norm,
            ctf2_over_nv_half,
            proj_half * half_weights,
            proj_abs2_half * half_weights,
            half_weights,
            N_IMAGES,
            N_TRANSLATIONS,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
        )
        scores_flat = np.asarray(scores).reshape(N_IMAGES, -1)
        expected_best = np.max(scores_flat, axis=1)
        max_scores = np.max(scores_flat, axis=1, keepdims=True)
        expected_log_z = max_scores[:, 0] + np.log(np.sum(np.exp(scores_flat - max_scores), axis=1))
        log_score_offset = -0.5 * np.asarray(batch_norm).reshape(N_IMAGES)
        expected_pmax = np.exp(expected_best - expected_log_z)

        np.testing.assert_allclose(
            np.asarray(stats.best_log_score_per_image),
            expected_best + log_score_offset,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(stats.log_evidence_per_image),
            expected_log_z + log_score_offset,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(stats.max_posterior_per_image),
            expected_pmax,
            atol=1e-5,
        )
        expected_rotation_mass = np.sum(
            np.exp(np.asarray(scores) - expected_log_z[:, None, None]),
            axis=(0, 2),
        )
        np.testing.assert_allclose(
            np.asarray(stats.rotation_posterior_sums),
            expected_rotation_mass,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            float(np.sum(np.asarray(stats.rotation_posterior_sums))),
            float(N_IMAGES),
            atol=1e-5,
        )
        np.testing.assert_array_equal(hard_assignments, np.argmax(scores_flat, axis=1))
        assert np.all(np.asarray(stats.max_posterior_per_image) >= 0.0)
        assert np.all(np.asarray(stats.max_posterior_per_image) <= 1.0)

    def test_class_log_prior_shifts_evidence_without_changing_single_class_posterior(self, seeded_inputs):
        """A uniform class prior shifts scores but cancels in a single-class posterior."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0
        class_log_prior = float(np.log(0.25))

        _, ha_base, Ft_y_base, Ft_ctf_base, stats_base, _ = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            return_stats=True,
            accumulate_noise=True,
            sparse_pass2=False,
        )
        _, ha_prior, Ft_y_prior, Ft_ctf_prior, stats_prior = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            class_log_prior=class_log_prior,
            return_stats=True,
            sparse_pass2=False,
        )

        np.testing.assert_array_equal(ha_prior, ha_base)
        np.testing.assert_allclose(np.asarray(Ft_y_prior), np.asarray(Ft_y_base), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(Ft_ctf_prior), np.asarray(Ft_ctf_base), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(stats_prior.log_evidence_per_image),
            np.asarray(stats_base.log_evidence_per_image) + class_log_prior,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(stats_prior.best_log_score_per_image),
            np.asarray(stats_base.best_log_score_per_image) + class_log_prior,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(stats_prior.max_posterior_per_image),
            np.asarray(stats_base.max_posterior_per_image),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_external_log_evidence_normalizes_dense_mstep_against_global_posterior(self, seeded_inputs):
        """External evidence normalizer scales this class's posterior mass."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        _, ha_base, Ft_y_base, Ft_ctf_base, stats_base, _ = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            return_stats=True,
            accumulate_noise=True,
            sparse_pass2=False,
        )
        batch_data, _, _, ctf_params, _, _, _ = next(
            ds.iter_batches(N_IMAGES, indices=np.arange(N_IMAGES), by_image=False)
        )
        config = ForwardModelConfig.from_dataset(
            ds,
            disc_type="linear_interp",
            process_fn=ds.process_images,
        )
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_test_batch(
            ds,
            jnp.asarray(batch_data),
            jnp.asarray(ctf_params),
            noise_variance,
            translations,
            config,
        )
        proj_half, proj_abs2_half = _compute_projections_block(
            volume,
            rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        scores = _e_step_block_scores(
            shifted_half,
            batch_norm,
            ctf2_over_nv_half,
            proj_half * half_weights,
            proj_abs2_half * half_weights,
            half_weights,
            N_IMAGES,
            N_TRANSLATIONS,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
        )
        scores_flat = np.asarray(scores).reshape(N_IMAGES, -1)
        max_scores = np.max(scores_flat, axis=1)
        direct_log_z = max_scores + np.log(
            np.sum(np.exp((scores_flat - max_scores[:, None]).astype(np.float64)), axis=1),
        )
        log_score_offset = -0.5 * np.asarray(batch_norm).reshape(N_IMAGES)
        external_log_evidence = direct_log_z + log_score_offset + np.log(2.0)
        _, ha_scaled, Ft_y_scaled, Ft_ctf_scaled, stats_scaled, _ = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            normalization_log_evidence=external_log_evidence,
            return_stats=True,
            accumulate_noise=True,
            sparse_pass2=False,
        )

        np.testing.assert_array_equal(ha_scaled, ha_base)
        np.testing.assert_allclose(np.asarray(Ft_y_scaled), 0.5 * np.asarray(Ft_y_base), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(Ft_ctf_scaled), 0.5 * np.asarray(Ft_ctf_base), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(stats_scaled.log_evidence_per_image),
            external_log_evidence,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(stats_scaled.max_posterior_per_image),
            0.5 * np.asarray(stats_base.max_posterior_per_image),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_return_stats_stable_with_large_image_norm_constant(self, seeded_inputs):
        """Large per-image norm constants should not collapse the posterior to uniform."""
        s = seeded_inputs
        ds = s["dataset"]
        ds._images = (np.asarray(ds._images) * 1e4).astype(np.float32)
        volume = s["volume"] * np.complex64(1e-6)
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        _, _, _, _, stats = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            return_stats=True,
        )

        batch_data, _, _, ctf_params, _, _, _ = next(
            ds.iter_batches(N_IMAGES, indices=np.arange(N_IMAGES), by_image=False)
        )
        config = ForwardModelConfig.from_dataset(
            ds,
            disc_type="linear_interp",
            process_fn=ds.process_images,
        )
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_test_batch(
            ds,
            jnp.asarray(batch_data),
            jnp.asarray(ctf_params),
            noise_variance,
            translations,
            config,
        )
        proj_half, proj_abs2_half = _compute_projections_block(
            volume,
            rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)

        shifted_np = np.asarray(shifted_half, dtype=np.complex128)
        ctf2_np = np.asarray(ctf2_over_nv_half, dtype=np.float64)
        proj_weighted_np = np.asarray(proj_half * half_weights, dtype=np.complex128)
        proj_abs2_np = np.asarray(proj_abs2_half * half_weights, dtype=np.float64)

        cross = -2.0 * np.real(np.conj(shifted_np) @ proj_weighted_np.T)
        cross = cross.reshape(N_IMAGES, N_TRANSLATIONS, N_ROTATIONS).swapaxes(1, 2)
        norms = ctf2_np @ proj_abs2_np.T
        rel_scores = -0.5 * (cross + norms[..., None])
        rel_scores_flat = rel_scores.reshape(N_IMAGES, -1)
        rel_best = np.max(rel_scores_flat, axis=1)
        rel_max = np.max(rel_scores_flat, axis=1, keepdims=True)
        rel_log_z = rel_max[:, 0] + np.log(np.sum(np.exp(rel_scores_flat - rel_max), axis=1))
        expected_pmax = np.exp(rel_best - rel_log_z)

        uniform_pmax = np.full_like(
            expected_pmax,
            1.0 / float(N_ROTATIONS * N_TRANSLATIONS),
        )
        assert np.max(np.abs(expected_pmax - uniform_pmax)) > 1e-8
        np.testing.assert_allclose(
            np.asarray(stats.max_posterior_per_image),
            expected_pmax,
            atol=1e-3,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            np.asarray(stats.log_evidence_per_image),
            rel_log_z - 0.5 * np.asarray(batch_norm).reshape(N_IMAGES),
            atol=1e-5,
            rtol=1e-6,
        )

    def test_masked_scoring_unmasked_reconstruction_split(self, rng):
        """Masked likelihood path should not zero out the reconstruction path."""
        ds = MaskedScoringDataset(rng)
        mean = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32)
        noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
        rotations = np.asarray(_make_rotations(2, seed=9))
        translations = np.array([[0.0, 0.0]], dtype=np.float32)

        _, _, Ft_y, _, stats = run_em(
            ds,
            mean,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=2,
            score_with_masked_images=True,
            return_stats=True,
        )

        np.testing.assert_allclose(
            np.asarray(stats.max_posterior_per_image),
            np.full(ds.n_units, 0.5, dtype=np.float32),
            atol=1e-6,
            rtol=1e-6,
        )
        assert np.max(np.abs(np.asarray(Ft_y))) > 0.0

    def test_subset_image_indices_matches_restricted_dataset(self, seeded_inputs):
        """Subset E-step/M-step via image_indices should match an explicitly restricted dataset."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0
        subset = np.array([0, 2, 3], dtype=np.int64)

        class _SubsetDataset:
            def __init__(self, base_ds, image_indices):
                self.image_shape = base_ds.image_shape
                self.image_size = base_ds.image_size
                self.grid_size = base_ds.grid_size
                self.volume_shape = base_ds.volume_shape
                self.volume_size = base_ds.volume_size
                self.voxel_size = base_ds.voxel_size
                self.dtype = base_ds.dtype
                self.ctf_evaluator = base_ds.ctf_evaluator
                self.process_images = base_ds.process_images
                self.process_images_half = base_ds.process_images_half
                self.image_source = base_ds.image_source
                self._images = np.asarray(base_ds._images)[image_indices].copy()
                self.CTF_params = np.asarray(base_ds.CTF_params)[image_indices].copy()
                self.n_images = len(image_indices)
                self.n_units = len(image_indices)

            def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
                _ = kwargs
                _ = by_image
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

        ds_subset = _SubsetDataset(ds, subset)

        mean_subset, ha_subset, Ft_y_subset, Ft_ctf_subset = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=2,
            rotation_block_size=2,
            image_indices=subset,
        )

        mean_restricted, ha_restricted, Ft_y_restricted, Ft_ctf_restricted = run_em(
            ds_subset,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=2,
            rotation_block_size=2,
        )

        np.testing.assert_allclose(np.array(mean_subset), np.array(mean_restricted), atol=1e-5)
        np.testing.assert_array_equal(ha_subset, ha_restricted)
        np.testing.assert_allclose(np.array(Ft_y_subset), np.array(Ft_y_restricted), atol=1e-5)
        np.testing.assert_allclose(np.array(Ft_ctf_subset), np.array(Ft_ctf_restricted), atol=1e-5)

    def test_image_specific_rotation_log_prior_controls_assignments(self, seeded_inputs):
        """run_em should support image-specific rotation priors."""
        s = seeded_inputs
        log_prior = np.array(
            [
                [0.0, -1e6],
                [-1e6, 0.0],
                [0.0, -1e6],
                [-1e6, 0.0],
            ],
            dtype=np.float32,
        )

        _, hard_assignments, _, _, stats = run_em(
            s["dataset"],
            s["volume"],
            np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0,
            s["noise_variance"],
            np.array(s["rotations"][:2]),
            np.array([[0.0, 0.0]], dtype=np.float32),
            "linear_interp",
            image_batch_size=2,
            rotation_block_size=2,
            rotation_log_prior=log_prior,
            return_stats=True,
        )

        np.testing.assert_array_equal(
            hard_assignments,
            np.array([0, 1, 0, 1], dtype=np.int32),
        )
        assert np.all(np.asarray(stats.max_posterior_per_image) > 0.99)

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
        new_mean_1, ha_1, Ft_y_1, Ft_ctf_1 = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        # Rotations split into blocks of 2 (5 rots -> 3 blocks: 2+2+1, padded to 2+2+2)
        new_mean_2, ha_2, Ft_y_2, Ft_ctf_2 = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=2,
        )

        np.testing.assert_allclose(
            np.array(new_mean_1),
            np.array(new_mean_2),
            atol=1e-4,
            err_msg="Mean differs between single-block and multi-block rotation processing",
        )
        np.testing.assert_array_equal(
            ha_1,
            ha_2,
            err_msg="Hard assignments differ between single-block and multi-block rotation processing",
        )
        np.testing.assert_allclose(
            np.array(Ft_y_1),
            np.array(Ft_y_2),
            atol=1e-4,
            err_msg="Ft_y differs between single-block and multi-block rotation processing",
        )
        np.testing.assert_allclose(
            np.array(Ft_ctf_1),
            np.array(Ft_ctf_2),
            atol=1e-4,
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
        new_mean_1, ha_1, Ft_y_1, Ft_ctf_1 = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
        )

        # Images in batches of 1
        new_mean_2, ha_2, Ft_y_2, Ft_ctf_2 = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=N_ROTATIONS,
        )

        np.testing.assert_allclose(
            np.array(new_mean_1),
            np.array(new_mean_2),
            atol=1e-4,
            err_msg="Mean differs between single-batch and multi-batch image processing",
        )
        # Hard assignments should match (same probabilities -> same argmax)
        np.testing.assert_array_equal(
            ha_1,
            ha_2,
            err_msg="Hard assignments differ between single-batch and multi-batch image processing",
        )

    def test_sparse_pass2_matches_dense_pass2_and_logs_skips(self, seeded_inputs, caplog):
        """Sparse pass 2 should preserve outputs while reporting skipped blocks."""
        s = seeded_inputs
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0
        rotation_log_prior = np.array([0.0, 0.0, -1e6, -1e6, -1e6], dtype=np.float32)

        dense_mean, dense_hard, dense_ft_y, dense_ft_ctf = run_em(
            s["dataset"],
            s["volume"],
            mean_variance,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=2,
            rotation_log_prior=rotation_log_prior,
            sparse_pass2=False,
        )

        with caplog.at_level(logging.INFO):
            sparse_mean, sparse_hard, sparse_ft_y, sparse_ft_ctf = run_em(
                s["dataset"],
                s["volume"],
                mean_variance,
                s["noise_variance"],
                np.array(s["rotations"]),
                np.array(s["translations"]),
                "linear_interp",
                image_batch_size=N_IMAGES,
                rotation_block_size=2,
                rotation_log_prior=rotation_log_prior,
                sparse_pass2=True,
            )

        np.testing.assert_allclose(np.asarray(sparse_mean), np.asarray(dense_mean), atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(sparse_hard, dense_hard)
        np.testing.assert_allclose(np.asarray(sparse_ft_y), np.asarray(dense_ft_y), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(sparse_ft_ctf), np.asarray(dense_ft_ctf), atol=1e-6, rtol=1e-6)
        assert any(
            "Sparse pass2 skipped 2 / 3 pass2 rotation blocks" in record.getMessage()
            and "omitted posterior mass upper bound" in record.getMessage()
            for record in caplog.records
        )

    def test_profile_stats_report_total_wall_and_unattributed_time(self, seeded_inputs):
        """Profile stats should include total wall time and explicit unattributed time."""
        s = seeded_inputs
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0
        rotation_log_prior = np.array([0.0, 0.0, -1e6, -1e6, -1e6], dtype=np.float32)

        _, _, _, _, em_profile = run_em(
            s["dataset"],
            s["volume"],
            mean_variance,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=2,
            rotation_log_prior=rotation_log_prior,
            sparse_pass2=True,
            return_profile=True,
        )

        stage_sum = (
            em_profile.batch_fetch_s
            + em_profile.preprocess_s
            + em_profile.score_prep_s
            + em_profile.pass1_projection_s
            + em_profile.pass1_score_s
            + em_profile.pass1_postprocess_s
            + em_profile.pass1_logsumexp_s
            + em_profile.pass2_skipmask_s
            + em_profile.pass2_projection_s
            + em_profile.pass2_score_s
            + em_profile.pass2_postprocess_s
            + em_profile.mstep_s
            + em_profile.window_scatter_s
            + em_profile.adjoint_y_s
            + em_profile.adjoint_ctf_s
            + em_profile.noise_s
            + em_profile.assignment_s
            + em_profile.stats_finalize_s
            + em_profile.host_stats_s
            + em_profile.solve_s
        )
        np.testing.assert_allclose(em_profile.accounted_s, stage_sum, atol=1e-6, rtol=1e-6)
        assert em_profile.total_wall_s >= stage_sum
        np.testing.assert_allclose(
            em_profile.unattributed_s,
            em_profile.total_wall_s - stage_sum,
            atol=1e-6,
            rtol=1e-6,
        )
        assert em_profile.sparse_pass2_total_blocks == 3
        assert em_profile.sparse_pass2_skipped_blocks == 2
        assert em_profile.sparse_pass2_omitted_mass_upper_mean >= 0.0
        assert em_profile.sparse_pass2_omitted_mass_upper_max >= em_profile.sparse_pass2_omitted_mass_upper_mean
        assert em_profile.sparse_pass2_omitted_mass_upper_sum >= em_profile.sparse_pass2_omitted_mass_upper_max

    def test_windowed_adjoint_ablation_flags_zero_expected_accumulators(self, seeded_inputs):
        """Windowed adjoint ablations should zero only the targeted accumulator."""
        s = seeded_inputs
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0
        common_kwargs = dict(
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            current_size=4,
            sparse_pass2=False,
        )

        _, _, base_ft_y, base_ft_ctf = run_em(
            s["dataset"],
            s["volume"],
            mean_variance,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            **common_kwargs,
        )
        _, _, no_y_ft_y, no_y_ft_ctf = run_em(
            s["dataset"],
            s["volume"],
            mean_variance,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            disable_adjoint_y=True,
            **common_kwargs,
        )
        _, _, no_ctf_ft_y, no_ctf_ft_ctf = run_em(
            s["dataset"],
            s["volume"],
            mean_variance,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            disable_adjoint_ctf=True,
            **common_kwargs,
        )

        assert np.linalg.norm(np.asarray(base_ft_y)) > 0.0
        assert np.linalg.norm(np.asarray(base_ft_ctf)) > 0.0
        np.testing.assert_allclose(np.asarray(no_y_ft_y), 0.0, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(np.asarray(no_y_ft_ctf), np.asarray(base_ft_ctf), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(no_ctf_ft_y), np.asarray(base_ft_y), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(no_ctf_ft_ctf), 0.0, atol=1e-7, rtol=1e-7)

    def test_score_only_probe_matches_full_disabled_adjoint_scores(self, seeded_inputs):
        """K-class score probes should skip pass 2 without changing evidence."""
        s = seeded_inputs
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0
        common_kwargs = dict(
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            current_size=4,
            sparse_pass2=False,
            return_stats=True,
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            relion_firstiter_score_mode="normalized_cc",
            relion_firstiter_winner_take_all=True,
        )

        full = run_em(
            s["dataset"],
            s["volume"],
            mean_variance,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            **common_kwargs,
        )
        probe = run_em(
            s["dataset"],
            s["volume"],
            mean_variance,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            score_only=True,
            **common_kwargs,
        )

        assert probe[0] is None
        np.testing.assert_array_equal(np.asarray(probe[1]), np.asarray(full[1]))
        np.testing.assert_allclose(np.asarray(probe[2]), 0.0, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(np.asarray(probe[3]), 0.0, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(probe[4].log_evidence_per_image, full[4].log_evidence_per_image)
        np.testing.assert_allclose(probe[4].best_log_score_per_image, full[4].best_log_score_per_image)
        np.testing.assert_allclose(probe[4].max_posterior_per_image, full[4].max_posterior_per_image)

    def test_score_only_skips_reconstruction_preprocess(self, seeded_inputs, monkeypatch):
        s = seeded_inputs

        def _fail_prepare_reconstruction_batch(*_args, **_kwargs):
            raise AssertionError("score_only should not prepare reconstruction batches")

        monkeypatch.setattr(
            em_engine_module,
            "_prepare_reconstruction_batch",
            _fail_prepare_reconstruction_batch,
        )

        result = run_em(
            s["dataset"],
            s["volume"],
            np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            current_size=4,
            sparse_pass2=False,
            return_stats=True,
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            relion_firstiter_score_mode="normalized_cc",
            relion_firstiter_winner_take_all=True,
            score_only=True,
        )

        assert result[0] is None

    def test_score_only_requires_disabled_adjoints(self, seeded_inputs):
        s = seeded_inputs
        with pytest.raises(ValueError, match="score_only requires"):
            run_em(
                s["dataset"],
                s["volume"],
                np.ones(VOLUME_SIZE, dtype=np.float32),
                s["noise_variance"],
                np.array(s["rotations"]),
                np.array(s["translations"]),
                "linear_interp",
                score_only=True,
            )

    def test_profile_timing_sync_runs_only_when_requested(self, seeded_inputs, monkeypatch):
        """Timing barriers should be inserted only for profiling runs."""
        s = seeded_inputs
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0
        sync_calls = []

        def _spy_block_until_ready(*values):
            sync_calls.append(len(values))

        monkeypatch.setattr(em_engine_module, "_block_until_ready", _spy_block_until_ready)

        run_em(
            s["dataset"],
            s["volume"],
            mean_variance,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            sparse_pass2=False,
            return_profile=False,
        )
        assert sync_calls == []

        run_em(
            s["dataset"],
            s["volume"],
            mean_variance,
            s["noise_variance"],
            np.array(s["rotations"]),
            np.array(s["translations"]),
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            sparse_pass2=False,
            return_profile=True,
        )
        assert sync_calls

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
            jnp.array(rng.standard_normal((n_images, n_rot_per_block, n_trans)).astype(np.float32)) for _ in range(3)
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
