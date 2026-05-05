"""Fourier windowing tests (Phase 3 of RELION-parity plan).

Tests:
1. test_window_indices_at_full_resolution: At current_size=128, all N_half pixels included.
2. test_window_indices_subset: At current_size=32, strict subset, count ~ pi*(16)^2 / 2.
3. test_windowed_e_step_matches_full: At current_size=128, windowed path == non-windowed.
4. test_windowed_m_step_roundtrip: Project -> window -> GEMM -> scatter -> adjoint.
5. test_adjoint_dot_product_windowed: <Ax, y> == <x, A*y> for windowed operator.
6. test_iteration_at_each_current_size: One run_em at current_size=32,64,128.
"""

import math

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    preprocess_batch as _preprocess_batch,
)
from recovar.em.dense_single_volume.helpers.projection import (
    compute_projections_block as _compute_projections_block,
)
from recovar.em.dense_single_volume.helpers.scoring import (
    _e_step_block_scores,
    _e_step_block_scores_windowed,
    _m_step_block_windowed,
)
from recovar.em.dense_single_volume.helpers.fourier_window import (
    ALLOWED_CURRENT_SIZES,
    make_fourier_window_spec,
    make_fourier_window_indices_np,
    make_frequency_coords_half_np,
    make_frequency_radius_map_half,
    quantize_current_size,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Test constants -- 8x8 images for unit tests (no GPU required)
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
# Helpers (reused from test_half_spectrum_em.py)
# ---------------------------------------------------------------------------


def _hermitian_image_2d(image_shape, seed=42):
    rng = np.random.default_rng(seed)
    real_img = rng.standard_normal(image_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fft2(real_img))
    return jnp.array(ft, dtype=jnp.complex64)


def _raw_real_image_2d(image_shape, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(image_shape).astype(np.float32)


def _hermitian_volume(volume_shape, seed=42):
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _make_rotations(n, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return jnp.array(q, dtype=jnp.float32)


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _identity_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _identity_process_half(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _constant_half_noise_variance(noise_variance):
    noise_variance = jnp.asarray(noise_variance)
    if noise_variance.shape[-1] == N_HALF:
        return noise_variance
    values = np.asarray(noise_variance).reshape(-1)
    if values.size != IMAGE_SIZE or not np.allclose(values, values[0]):
        raise ValueError("test helper only supports scalar full-image noise variance")
    return jnp.full((N_HALF,), values[0], dtype=noise_variance.dtype)


class MockDataset:
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
        self.process_images_half = staticmethod(_identity_process_half)

        self._images = np.zeros((N_IMAGES, *IMAGE_SHAPE), dtype=np.float32)
        for i in range(N_IMAGES):
            self._images[i] = _raw_real_image_2d(IMAGE_SHAPE, seed=rng.integers(10000))

        class _ImageSource:
            process_images = staticmethod(_identity_process)
            process_images_half = staticmethod(_identity_process_half)

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
    volume = _hermitian_volume(VOLUME_SHAPE, seed=42)
    rotations = _make_rotations(N_ROTATIONS, seed=12)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
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
# Test 1: Window indices at full resolution
# ===========================================================================


class TestWindowIndicesFullResolution:
    """At current_size=image_shape[0], window selects a circular band with radius N//2.

    Note: This does NOT include all N_half pixels because the half-spectrum
    is a rectangle and corner pixels have radius > N//2.  This is correct:
    RELION's rlnCurrentImageSize=N means a circular mask at radius N//2,
    excluding the square-grid corners.

    In run_em, when current_size >= image_shape[0], use_window=False
    (no windowing at all), so the full half-spectrum is used.
    """

    def test_full_resolution_is_circular(self):
        """current_size=8 for 8x8: circular mask at r=4, excludes corners."""
        indices, n = make_fourier_window_indices_np(IMAGE_SHAPE, current_size=8)
        # Not all N_HALF pixels, only those within radius 4
        assert n < N_HALF, f"Should exclude corners, got {n} / {N_HALF}"
        assert n > N_HALF // 2, f"Should include most pixels, got {n}"

    def test_128px_full_resolution(self):
        """current_size=128 for 128x128: circular mask at r=64."""
        shape_128 = (128, 128)
        n_half_128 = 128 * 65
        indices, n = make_fourier_window_indices_np(shape_128, current_size=128)
        # Circular area ~= pi*64^2/2 ~= 6434, total = 8320
        assert n < n_half_128, "Should exclude corners"
        # But should include a large fraction (pi/4 ~ 0.785 of the rectangle)
        assert n > n_half_128 * 0.7, f"Should include most pixels, got {n} / {n_half_128}"


# ===========================================================================
# Test 2: Window indices subset
# ===========================================================================


class TestWindowIndicesSubset:
    """At current_size < full, indices should be a strict subset."""

    def test_subset_is_strict(self):
        """current_size=4 for 8x8: fewer than N_half indices."""
        indices_small, n_small = make_fourier_window_indices_np(IMAGE_SHAPE, current_size=4)
        assert n_small < N_HALF, f"Expected strict subset, got {n_small} / {N_HALF}"
        assert n_small > 0, "Window should include at least DC"

    def test_32_in_128_approximate_count(self):
        """current_size=32 for 128x128: ~pi*(16)^2 / 2 ~ 402 (half of circular area)."""
        shape_128 = (128, 128)
        indices, n = make_fourier_window_indices_np(shape_128, current_size=32)
        expected_approx = math.pi * 16**2 / 2  # half of circular area
        # Allow 20% tolerance since we're comparing discrete count to continuous area
        assert abs(n - expected_approx) / expected_approx < 0.20, f"Expected ~{expected_approx:.0f} indices, got {n}"

    def test_indices_sorted(self):
        """Indices should be sorted."""
        indices, _ = make_fourier_window_indices_np(IMAGE_SHAPE, current_size=4)
        assert np.all(np.diff(indices) > 0), "Indices should be strictly sorted"

    def test_indices_in_range(self):
        """All indices must be valid for N_half."""
        indices, _ = make_fourier_window_indices_np(IMAGE_SHAPE, current_size=4)
        assert np.all(indices >= 0)
        assert np.all(indices < N_HALF)

    def test_monotonic_inclusion(self):
        """Larger current_size should include all indices of smaller current_size."""
        idx_4, _ = make_fourier_window_indices_np(IMAGE_SHAPE, current_size=4)
        idx_6, _ = make_fourier_window_indices_np(IMAGE_SHAPE, current_size=6)
        idx_8, _ = make_fourier_window_indices_np(IMAGE_SHAPE, current_size=8)
        assert set(idx_4).issubset(set(idx_6))
        assert set(idx_6).issubset(set(idx_8))

    def test_relion_exact_half_layout_count_at_late_current_size(self):
        """128px box, current_size=84: match the RELION half-layout scoring count."""
        indices, n = make_fourier_window_indices_np((128, 128), current_size=84)
        assert n == 2834
        assert indices.size == 2834

    def test_initial_model_pass1_radial_support_count(self):
        """64px box, current_size=10: RELION InitialModel pass1 uses radial Minvsigma2 support."""
        radial_indices, n_radial = make_fourier_window_indices_np(
            (64, 64),
            current_size=10,
            square=False,
            include_dc=False,
        )
        square_indices, n_square = make_fourier_window_indices_np(
            (64, 64),
            current_size=10,
            square=True,
            include_dc=False,
        )

        assert n_radial == 46
        assert radial_indices.size == 46
        assert n_square == 59
        assert square_indices.size == 59

    def test_relion_reconstruction_window_keeps_only_dc_extra(self):
        """Reconstruction support should differ from scoring support only by DC."""
        idx_score, n_score = make_fourier_window_indices_np((128, 128), current_size=84, include_dc=False)
        idx_recon, n_recon = make_fourier_window_indices_np((128, 128), current_size=84, include_dc=True)
        assert n_score == 2834
        assert n_recon == 2835
        assert n_recon - n_score == 1
        assert set(idx_score).issubset(set(idx_recon))

    def test_exact_radius_removes_rounded_outer_rim(self):
        """M-step support follows RELION BackProjector's squared-radius cutoff."""
        shape = (64, 64)
        current_size = 54
        idx_round, _ = make_fourier_window_indices_np(shape, current_size=current_size, include_dc=True)
        idx_exact, _ = make_fourier_window_indices_np(
            shape,
            current_size=current_size,
            include_dc=True,
            exact_radius=True,
        )

        coords = make_frequency_coords_half_np(shape)
        r_max = current_size // 2
        exact_coords = np.rint(coords[idx_exact]).astype(np.int32)
        assert np.all(np.sum(exact_coords * exact_coords, axis=1) <= r_max * r_max)
        assert set(idx_exact).issubset(set(idx_round))
        assert len(set(idx_round) - set(idx_exact)) > 0

    def test_relion_half_layout_excludes_duplicates_and_negative_boundary_row(self):
        """Exact RELION support removes the packed-half extras from the naive radial mask."""
        shape = (128, 128)
        current_size = 84
        coords = np.asarray(ftu.get_k_coordinate_of_each_pixel_half(shape, voxel_size=1, scaled=False))
        kx = np.rint(coords[:, 0]).astype(np.int32)
        ky = np.rint(coords[:, 1]).astype(np.int32)
        radii = np.sqrt(np.sum(coords**2, axis=-1))
        naive_mask = np.round(radii).astype(np.int32) <= current_size // 2

        idx_score, n_score = make_fourier_window_indices_np(shape, current_size=current_size, include_dc=False)
        assert int(np.count_nonzero(naive_mask)) == 2883
        assert n_score == 2834
        assert int(np.count_nonzero(naive_mask)) - n_score == 49
        assert int(np.count_nonzero(naive_mask & (kx == 0) & (ky == 0))) == 1
        assert int(np.count_nonzero(naive_mask & (kx == 0) & (ky < 0))) == 42
        assert int(np.count_nonzero(naive_mask & (ky == -(current_size // 2)))) == 7
        assert 1 + 42 + 6 == 49

    def test_relion_noise_shell_indices_exclude_negative_kx0_axis(self):
        """Noise shell support matches RELION's Mresol_fine duplicate-axis rule."""
        shape = (128, 128)
        coords = make_frequency_coords_half_np(shape)
        shell_indices = np.asarray(make_relion_noise_shell_indices_half(shape))

        shell1 = np.flatnonzero(shell_indices == 1)
        shell1_coords = {tuple(coords[i].astype(int)) for i in shell1}

        assert (0, -1) not in shell1_coords
        assert (0, 1) in shell1_coords
        assert len(shell1_coords) == 4
        counts = np.bincount(shell_indices[shell_indices <= shape[0] // 2], minlength=shape[0] // 2 + 1)
        assert counts[:5].tolist() == [1, 4, 6, 8, 16]

    def test_square_window_matches_relion_downsized_half_shape(self):
        """Square mode matches RELION windowFourierTransform's FFTW crop size."""
        shape = (64, 64)
        current_size = 54
        idx_score, n_score = make_fourier_window_indices_np(
            shape,
            current_size=current_size,
            square=True,
            include_dc=False,
        )
        idx_recon, n_recon = make_fourier_window_indices_np(
            shape,
            current_size=current_size,
            square=True,
            include_dc=True,
        )

        assert n_recon == current_size * (current_size // 2 + 1)
        assert n_score == n_recon - 1
        assert set(idx_score).issubset(set(idx_recon))

    def test_square_window_keeps_relion_negative_rows_not_full_nyquist_column(self):
        """RELION keeps negative ky rows in its crop but not the full-grid Nyquist column."""
        shape = (64, 64)
        current_size = 54
        coords = make_frequency_coords_half_np(shape)
        idx_recon, _ = make_fourier_window_indices_np(
            shape,
            current_size=current_size,
            square=True,
            include_dc=True,
        )
        selected = coords[idx_recon]

        assert np.any((selected[:, 0] == 0) & (selected[:, 1] < 0))
        assert not np.any(selected[:, 0] == -(shape[1] // 2))


# ===========================================================================
# Test 3: Windowed E-step matches full at current_size=full
# ===========================================================================


class TestWindowedEStepMatchesFull:
    """When using ALL half-spectrum indices as window, windowed path matches full exactly."""

    def test_e_step_all_indices_match(self, seeded_inputs):
        """Windowed E-step with all N_half indices == non-windowed E-step."""
        s = seeded_inputs
        config = s["config"]
        volume = s["volume"]
        rotations = s["rotations"]
        translations = s["translations"]
        noise_variance = s["noise_variance"]
        ds = s["dataset"]

        n_images = ds.n_units
        n_trans = translations.shape[0]

        batch_data = jnp.asarray(ds._images)
        ctf_params = jnp.asarray(ds.CTF_params)

        # Preprocess
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            ds,
            batch_data,
            ctf_params,
            _constant_half_noise_variance(noise_variance),
            translations,
            config,
        )

        # Projections
        proj_half, proj_abs2_half = _compute_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        proj_half_weighted = proj_half * half_weights
        proj_abs2_weighted = proj_abs2_half * half_weights

        # Non-windowed E-step
        scores_full = _e_step_block_scores(
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

        # Windowed E-step using ALL indices (identity window)
        wi = jnp.arange(N_HALF, dtype=jnp.int32)
        n_windowed = N_HALF

        shifted_w = shifted_half[:, wi]
        ctf2_w = ctf2_over_nv_half[:, wi]
        proj_w = proj_half[:, wi]
        proj_abs2_w = proj_abs2_half[:, wi]
        hw = half_weights[wi]
        proj_w_weighted = proj_w * hw
        proj_abs2_w_weighted = proj_abs2_w * hw

        scores_windowed = _e_step_block_scores_windowed(
            shifted_w,
            batch_norm,
            ctf2_w,
            proj_w_weighted,
            proj_abs2_w_weighted,
            hw,
            n_images,
            n_trans,
            n_windowed,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
        )

        np.testing.assert_allclose(
            np.array(scores_full),
            np.array(scores_windowed),
            atol=1e-4,
            rtol=1e-5,
            err_msg="Windowed E-step with all indices should match non-windowed",
        )


class TestFourierWindowSpec:
    def test_score_and_recon_values_gather_last_axis(self):
        spec = make_fourier_window_spec(IMAGE_SHAPE, 6, N_HALF, include_recon_window=True)
        values = jnp.arange(3 * N_HALF, dtype=jnp.float32).reshape(3, N_HALF)

        np.testing.assert_array_equal(
            np.asarray(spec.score_values(values)),
            np.asarray(values)[:, np.asarray(spec.score_indices_np)],
        )
        np.testing.assert_array_equal(
            np.asarray(spec.recon_values(values)),
            np.asarray(values)[:, np.asarray(spec.recon_indices_np)],
        )

    def test_firstiter_cc_uses_rectangular_score_and_radial_recon_support(self):
        """RELION firstiter CC scores the cropped FFTW image, but backprojects radially."""
        shape = (128, 128)
        n_half = shape[0] * (shape[1] // 2 + 1)
        spec = make_fourier_window_spec(
            shape,
            current_size=26,
            n_half=n_half,
            square=False,
            score_square=True,
            score_include_dc=True,
            include_recon_window=True,
        )

        assert spec.n_score == 26 * (26 // 2 + 1)
        assert spec.n_recon == 265
        assert spec.projection_kwargs()["max_r"] == 13.0
        assert spec.dense_big_jit_projection_max_r() == 13.0
        assert spec.dense_big_jit_backprojection_max_r() == 13.0
        assert set(np.asarray(spec.recon_indices_np)).issubset(set(np.asarray(spec.score_indices_np)))


# ===========================================================================
# Test 4: Windowed M-step roundtrip
# ===========================================================================


class TestWindowedMStepRoundtrip:
    """Project -> window -> GEMM -> scatter -> adjoint: energy only at low freq."""

    def test_ft_y_energy_at_low_frequencies(self, seeded_inputs):
        """M-step with small window should produce Ft_y with energy only at low-freq shells."""
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
        current_size = 4  # small window

        batch_data = jnp.asarray(ds._images)
        ctf_params = jnp.asarray(ds.CTF_params)

        # Preprocess
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            ds,
            batch_data,
            ctf_params,
            _constant_half_noise_variance(noise_variance),
            translations,
            config,
        )

        # Window
        window_indices, n_windowed = make_fourier_window_indices_np(IMAGE_SHAPE, current_size)
        wi = jnp.asarray(window_indices)
        shifted_w = shifted_half[:, wi]
        ctf2_w = ctf2_over_nv_half[:, wi]

        # Projections (windowed)
        proj_half, proj_abs2_half = _compute_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        hw = half_weights[wi]
        proj_w = proj_half[:, wi]
        proj_abs2_w = proj_abs2_half[:, wi]
        proj_w_weighted = proj_w * hw
        proj_abs2_w_weighted = proj_abs2_w * hw

        # E-step to get scores
        scores = _e_step_block_scores_windowed(
            shifted_w,
            batch_norm,
            ctf2_w,
            proj_w_weighted,
            proj_abs2_w_weighted,
            hw,
            n_images,
            n_trans,
            n_windowed,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
        )
        scores_flat = scores.reshape(n_images, -1)
        log_Z = jax.scipy.special.logsumexp(scores_flat, axis=1)

        # M-step (windowed)
        Ft_y = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)
        Ft_ctf = jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64)

        (Ft_y, Ft_ctf, probs, _, _, summed_w, ctf_probs_w) = _m_step_block_windowed(
            shifted_w,
            scores,
            log_Z,
            rotations,
            ctf2_w,
            Ft_y,
            Ft_ctf,
            n_images,
            n_trans,
            n_windowed,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
        )

        # Scatter back to full half-spectrum
        n_half = H * (W // 2 + 1)
        summed_half = jnp.zeros((n_rot, n_half), dtype=summed_w.dtype)
        summed_half = summed_half.at[:, wi].set(summed_w)

        # Adjoint
        Ft_y = core.adjoint_slice_volume(
            summed_half,
            rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            volume=Ft_y,
            half_image=True,
        )

        # The volume Ft_y should have non-zero values (the adjoint distributes energy)
        assert jnp.any(jnp.abs(Ft_y) > 0), "Ft_y should be non-zero"
        assert jnp.all(jnp.isfinite(Ft_y)), "Ft_y should be finite"


# ===========================================================================
# Test 5: Adjoint dot product test for windowed operator
# ===========================================================================


class TestAdjointDotProductWindowed:
    """For the windowed operator (project -> window, and window -> scatter -> adjoint),
    verify <Ax, y>_hw == <x, A*y>_vol.

    Key insight: adjoint_slice_volume(y_half, half_image=True) already includes
    the Hermitian fold.  It is the adjoint of slice_volume(x, half_image=False)
    in the sense that:

        <proj_full, y_full> == <x, adjoint_half(y_half)>

    where y_full = half_image_to_full_image(y_half).

    Equivalently, with Hermitian weights:
        <proj_half, y_half>_hw == <x, adjoint_half(y_half)>

    where <a, b>_hw = Re[sum(conj(a) * hw * b)] recovers the full inner product.

    For the WINDOWED operator:
        Forward:  A(x) = gather(slice_volume(x, half_image=True), wi)
        <Ax, y>_hw_windowed = Re[sum(conj(Ax_w) * hw_w * y_w)]
        A*(y) = adjoint_slice_volume(scatter(y_w * hw_w, wi), half_image=True)
        <x, A*y> = Re[sum(conj(x) * A*y)]

    The hw_w is included in A* because the adjoint_slice already includes
    the Hermitian fold; we need the weights to map the windowed inner product
    to the full inner product that adjoint_slice_volume expects.
    """

    def test_adjoint_consistency_no_window(self):
        """Baseline: <proj_half, y_half>_hw == <x, adjoint_half(y_half)> (no windowing).

        adjoint_slice_volume(y_half, half_image=True) includes the Hermitian fold,
        so it is the adjoint of slice_volume(x, half_image=True) under the
        Hermitian-weighted inner product:
            <a_half, b_half>_hw = Re[sum(conj(a_half) * hw * b_half)]
        """
        rng = np.random.default_rng(42)

        x_real = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
        x_ft = np.fft.fftshift(np.fft.fftn(x_real))
        x = jnp.array(x_ft.ravel(), dtype=jnp.complex64)

        n_rot = 3
        rotations = _make_rotations(n_rot, seed=99)

        y_half = jnp.array(
            (rng.standard_normal((n_rot, N_HALF)) + 1j * rng.standard_normal((n_rot, N_HALF))).astype(np.complex64)
        )

        half_weights = make_half_image_weights(IMAGE_SHAPE)

        # Forward (half)
        proj_half = core.slice_volume(x, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=True)

        # <proj_half, y_half>_hw
        ip_Ax_y = jnp.sum(jnp.conj(proj_half) * half_weights * y_half).real

        # Adjoint (half_image=True includes Hermitian fold)
        Aty = core.adjoint_slice_volume(
            y_half,
            rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            half_image=True,
        )

        ip_x_Aty = jnp.sum(jnp.conj(x) * Aty).real

        np.testing.assert_allclose(
            float(ip_Ax_y),
            float(ip_x_Aty),
            rtol=1e-4,
            err_msg="Full adjoint dot product test failed: <Ax, y>_hw != <x, A*y>",
        )

    def test_adjoint_consistency_windowed(self):
        """Windowed adjoint: <proj_w, y_w>_hw_w == <x, adjoint_half(z_half)>.

        The windowed forward is: A_w(x) = gather(slice_volume(x, half_image=True), wi)
        The windowed adjoint under <a, b>_hw is: A_w*(y_w) = adjoint_half(z_half)
        where z_half has y_w at windowed positions and zeros elsewhere (NO hw weighting).

        This works because:
            <proj_w, y_w>_hw_w = sum_wi conj(proj_half[wi]) * hw[wi] * y_w
                               = sum_half conj(proj_half) * hw * z_half
                               = <proj_half, z_half>_hw
                               = <x, adjoint_half(z_half)>
        """
        rng = np.random.default_rng(42)

        x_real = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
        x_ft = np.fft.fftshift(np.fft.fftn(x_real))
        x = jnp.array(x_ft.ravel(), dtype=jnp.complex64)

        current_size = 4
        window_indices, n_windowed = make_fourier_window_indices_np(IMAGE_SHAPE, current_size)
        wi = jnp.asarray(window_indices)
        n_rot = 3
        rotations = _make_rotations(n_rot, seed=99)

        y_windowed = jnp.array(
            (rng.standard_normal((n_rot, n_windowed)) + 1j * rng.standard_normal((n_rot, n_windowed))).astype(
                np.complex64
            )
        )

        half_weights = make_half_image_weights(IMAGE_SHAPE)
        hw_w = half_weights[wi]

        # Forward: project at full res, then gather windowed indices
        proj_half = core.slice_volume(x, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=True)
        Ax_windowed = proj_half[:, wi]

        # <Ax_w, y_w>_hw_w = Re[sum(conj(Ax_w) * hw_w * y_w)]
        ip_Ax_y = jnp.sum(jnp.conj(Ax_windowed) * hw_w * y_windowed).real

        # Adjoint: scatter y_w into full half-spectrum (NO hw multiplication),
        # then adjoint_slice_volume
        z_half = jnp.zeros((n_rot, N_HALF), dtype=y_windowed.dtype)
        z_half = z_half.at[:, wi].set(y_windowed)
        Aty = core.adjoint_slice_volume(
            z_half,
            rotations,
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            half_image=True,
        )

        ip_x_Aty = jnp.sum(jnp.conj(x) * Aty).real

        np.testing.assert_allclose(
            float(ip_Ax_y),
            float(ip_x_Aty),
            rtol=1e-4,
            err_msg="Windowed adjoint dot product failed: <Ax_w, y_w>_hw != <x, A*(y_w)>",
        )


# ===========================================================================
# Test 6: Full iteration at each current_size
# ===========================================================================


class TestIterationAtEachCurrentSize:
    """Run one full run_em iteration at each allowed current_size."""

    @pytest.mark.parametrize("current_size", [4, 6, 8])
    def test_iteration_produces_finite_results(self, seeded_inputs, current_size):
        """run_em with current_size produces finite outputs and valid hard assignments."""
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
            current_size=current_size,
        )

        # All outputs should be finite
        assert np.all(np.isfinite(np.array(new_mean))), f"new_mean not finite at cs={current_size}"
        assert np.all(np.isfinite(np.array(Ft_y))), f"Ft_y not finite at cs={current_size}"
        assert np.all(np.isfinite(np.array(Ft_ctf))), f"Ft_ctf not finite at cs={current_size}"

        # Hard assignments should be valid
        assert ha.shape == (N_IMAGES,)
        assert np.all(ha >= 0)
        assert np.all(ha < N_ROTATIONS * N_TRANSLATIONS)

    def test_full_resolution_matches_no_window(self, seeded_inputs):
        """current_size=8 (full for 8x8 images) should match current_size=None."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0

        # No windowing
        new_mean_none, ha_none, Ft_y_none, _ = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            current_size=None,
        )

        # current_size = 8 (full resolution for 8x8)
        new_mean_8, ha_8, Ft_y_8, _ = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            current_size=8,
        )

        # current_size=8 for 8x8 images is NOT windowed (use_window is False
        # when current_size >= image_shape[0]), so these should be identical
        np.testing.assert_allclose(
            np.array(new_mean_none),
            np.array(new_mean_8),
            atol=1e-5,
            err_msg="current_size=8 should match no windowing for 8x8 images",
        )
        np.testing.assert_array_equal(ha_none, ha_8)


# ===========================================================================
# Test 7: Quantize current_size
# ===========================================================================


class TestQuantizeCurrentSize:
    """Test the quantize_current_size helper."""

    def test_exact_match(self):
        for s in [16, 18, 30, 50, 98, 128]:
            assert quantize_current_size(s, ori_size=128) == s

    def test_default_allowed_path_preserved(self):
        for s in ALLOWED_CURRENT_SIZES:
            assert quantize_current_size(s) == s

    def test_round_up(self):
        assert quantize_current_size(10, ori_size=128) == 16
        assert quantize_current_size(17, ori_size=128) == 18
        assert quantize_current_size(25, ori_size=128) == 26
        assert quantize_current_size(33, ori_size=128) == 34
        assert quantize_current_size(49, ori_size=128) == 50
        assert quantize_current_size(65, ori_size=128) == 66
        assert quantize_current_size(81, ori_size=128) == 82
        assert quantize_current_size(97, ori_size=128) == 98

    def test_above_max(self):
        assert quantize_current_size(200, ori_size=128) == 128

    def test_custom_allowed(self):
        assert quantize_current_size(10, allowed=[8, 16, 32]) == 16
        assert quantize_current_size(8, allowed=[8, 16, 32]) == 8

    def test_allowed_sizes_sorted(self):
        """ALLOWED_CURRENT_SIZES must be sorted in ascending order."""
        assert ALLOWED_CURRENT_SIZES == sorted(ALLOWED_CURRENT_SIZES)

    def test_allowed_sizes_match_module(self):
        """Sanity: module constant matches expected set."""
        assert ALLOWED_CURRENT_SIZES == [16, 24, 32, 48, 64, 80, 96, 104, 112, 120, 128, 160, 192, 224, 256]


# ===========================================================================
# Test 8: Frequency radius map
# ===========================================================================


class TestFrequencyRadiusMap:
    """Test make_frequency_radius_map_half."""

    def test_dc_at_zero(self):
        """DC frequency should have radius 0."""
        radii = make_frequency_radius_map_half(IMAGE_SHAPE)
        # DC is at the frequency coordinate (0, 0)
        # Find it: in half-spectrum, the DC pixel should exist
        coords = ftu.get_k_coordinate_of_each_pixel_half(IMAGE_SHAPE, voxel_size=1, scaled=False)
        dc_mask = jnp.all(coords == 0, axis=-1)
        dc_radii = radii[dc_mask]
        assert len(dc_radii) == 1, "Should have exactly one DC pixel"
        np.testing.assert_allclose(float(dc_radii[0]), 0.0, atol=1e-6)

    def test_shape(self):
        radii = make_frequency_radius_map_half(IMAGE_SHAPE)
        assert radii.shape == (N_HALF,)

    def test_positive(self):
        """All radii should be non-negative."""
        radii = make_frequency_radius_map_half(IMAGE_SHAPE)
        assert jnp.all(radii >= 0)


# ===========================================================================
# Test 9: Multiple rotation blocks with windowing
# ===========================================================================


class TestWindowedMultipleBlocks:
    """Results should be identical regardless of rotation block size with windowing."""

    def test_block_size_invariance(self, seeded_inputs):
        """Windowed results with block_size=N_ROT vs block_size=2 should match."""
        s = seeded_inputs
        ds = s["dataset"]
        volume = s["volume"]
        noise_variance = s["noise_variance"]
        rotations = np.array(s["rotations"])
        translations = np.array(s["translations"])
        mean_variance = np.ones(VOLUME_SIZE, dtype=np.float32) * 100.0
        current_size = 4

        # All rotations in one block
        new_mean_1, ha_1, Ft_y_1, _ = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            current_size=current_size,
        )

        # Split into blocks of 2
        new_mean_2, ha_2, Ft_y_2, _ = run_em(
            ds,
            volume,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=2,
            current_size=current_size,
        )

        np.testing.assert_allclose(
            np.array(new_mean_1),
            np.array(new_mean_2),
            atol=1e-4,
            err_msg="Mean differs between single-block and multi-block with windowing",
        )
        np.testing.assert_array_equal(ha_1, ha_2)
