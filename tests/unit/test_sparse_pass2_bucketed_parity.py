"""Parity tests: bucketed sparse pass-2 vs per-image reference path.

The bucketed batched implementation in
``compute_pass2_stats_sparse_bucketed`` must produce numerically
identical outputs (modulo floating-point rounding) to the legacy
per-image loop preserved as
``_compute_pass2_stats_sparse_perimage_reference``.

These tests construct synthetic single-class data with varied per-image
significant rotation counts (the trigger for the JIT-recompile bug the
batched path is designed to fix), then call both paths and compare the
M-step accumulators ``Ft_y`` / ``Ft_ctf``, hard assignments, and per-image
RELION stats.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.helpers.oversampling import (
    _compute_pass2_stats_sparse_perimage_reference,
    compute_pass2_stats_sparse,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _normalize_pass2_bucket,
    _normalize_pass2_bucket_with_log_z,
    _score_pass2_bucket_normalized_cc,
    _score_pass2_bucket_relion_gpu_diff2,
    _winner_take_all_bucket_probs,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Mock dataset (mirrors test_adaptive_oversampling.MockDataset)
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512


def _raw_real_image_2d(image_shape, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(image_shape).astype(np.float32)


def _hermitian_volume(volume_shape, seed=42):
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _unit_image_mask(dtype=jnp.float32):
    return jnp.linspace(0.2, 1.0, IMAGE_SIZE, dtype=dtype).reshape(IMAGE_SHAPE)


def _raw_real_process(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    if apply_image_mask:
        images = images * _unit_image_mask(images.dtype)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    if apply_image_mask:
        images = images * _unit_image_mask(images.dtype)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


class MockDataset:
    def __init__(self, n_images=10, seed=42):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = n_images
        self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)

        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)
        self.premultiplied_ctf = False

        rng = np.random.default_rng(seed)
        self._images = np.zeros((n_images, *IMAGE_SHAPE), dtype=np.float32)
        for i in range(n_images):
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
                jnp.asarray(self.rotation_matrices[idx]),
                jnp.asarray(self.translations[idx]),
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)

    def update_poses(self, rotations, translations):
        self.rotation_matrices = np.asarray(rotations)
        self.translations = np.asarray(translations)


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------


def _max_relative_error(a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.maximum(denom, eps)
    return float(np.max(diff / denom))


def _compare_outputs(out_ref, out_bucket, atol=1e-5, rtol=1e-5):
    """Compare per-image and accumulated outputs with tight tolerance."""
    (
        Ft_y_ref,
        Ft_ctf_ref,
        ha_ref,
        best_rot_ref,
        best_tr_ref,
        best_idx_ref,
        stats_ref,
    ) = out_ref[:7]
    (
        Ft_y_b,
        Ft_ctf_b,
        ha_b,
        best_rot_b,
        best_tr_b,
        best_idx_b,
        stats_b,
    ) = out_bucket[:7]

    # M-step accumulators must be very close.
    np.testing.assert_allclose(np.asarray(Ft_y_ref), np.asarray(Ft_y_b), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(Ft_ctf_ref), np.asarray(Ft_ctf_b), atol=atol, rtol=rtol)

    # Hard assignments must match exactly (decoded from probs argmax).
    np.testing.assert_array_equal(np.asarray(ha_ref), np.asarray(ha_b))
    np.testing.assert_array_equal(np.asarray(best_idx_ref), np.asarray(best_idx_b))
    np.testing.assert_allclose(np.asarray(best_rot_ref), np.asarray(best_rot_b), atol=1e-6)
    np.testing.assert_allclose(np.asarray(best_tr_ref), np.asarray(best_tr_b), atol=1e-6)

    # RELION stats must match within float32 precision.
    np.testing.assert_allclose(
        np.asarray(stats_ref.log_evidence_per_image),
        np.asarray(stats_b.log_evidence_per_image),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(stats_ref.best_log_score_per_image),
        np.asarray(stats_b.best_log_score_per_image),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(stats_ref.max_posterior_per_image),
        np.asarray(stats_b.max_posterior_per_image),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(stats_ref.rotation_posterior_sums),
        np.asarray(stats_b.rotation_posterior_sums),
        atol=atol,
        rtol=rtol,
    )

    # Noise stats (when present)
    if len(out_ref) == 8 and len(out_bucket) == 8:
        ns_ref = out_ref[7]
        ns_b = out_bucket[7]
        np.testing.assert_allclose(
            np.asarray(ns_ref.wsum_sigma2_noise),
            np.asarray(ns_b.wsum_sigma2_noise),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            np.asarray(ns_ref.wsum_img_power),
            np.asarray(ns_b.wsum_img_power),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            ns_ref.wsum_sigma2_offset,
            ns_b.wsum_sigma2_offset,
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(ns_ref.sumw, ns_b.sumw, atol=atol)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_sparse_pass2_score_matches_direct_diff2_for_finite_inputs():
    """Sparse pass-2 scores must preserve direct diff2 algebra."""
    rng = np.random.default_rng(123)
    batch, n_rot, n_trans, n_half = 2, 3, 4, 5
    image = (
        rng.normal(size=(batch, n_trans, n_half)) + 1j * rng.normal(size=(batch, n_trans, n_half))
    ).astype(np.complex64)
    ctf = rng.uniform(0.35, 1.25, size=(batch, n_half)).astype(np.float32)
    noise = rng.uniform(0.7, 2.0, size=(batch, n_half)).astype(np.float32)
    proj = (
        rng.normal(size=(batch, n_rot, n_half)) + 1j * rng.normal(size=(batch, n_rot, n_half))
    ).astype(np.complex64)
    half_weights = np.array([1.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float32)
    rot_prior = rng.normal(scale=0.1, size=(batch, n_rot)).astype(np.float32)
    trans_prior = rng.normal(scale=0.1, size=(batch, n_trans)).astype(np.float32)
    mask = np.ones((batch, n_rot, n_trans), dtype=bool)

    corr_img_score = ctf * ctf / noise
    shifted_corrected = image / ctf[:, None, :]
    scores = _score_pass2_bucket_relion_gpu_diff2(
        jnp.asarray(shifted_corrected),
        jnp.asarray(corr_img_score),
        jnp.asarray(proj),
        jnp.asarray(half_weights),
        jnp.asarray(rot_prior),
        jnp.asarray(trans_prior),
        jnp.asarray(mask),
    )

    expected = np.empty((batch, n_rot, n_trans), dtype=np.float32)
    for b in range(batch):
        weights = corr_img_score[b] * half_weights
        image_constant = 0.5 * np.sum(np.abs(shifted_corrected[b]) ** 2 * weights[None, :], axis=-1)
        for r in range(n_rot):
            diff = proj[b, r][None, :] - shifted_corrected[b]
            diff2 = 0.5 * np.sum(np.abs(diff) ** 2 * weights[None, :], axis=-1)
            expected[b, r] = -diff2 + image_constant + rot_prior[b, r] + trans_prior[b]

    np.testing.assert_allclose(np.asarray(scores), expected, atol=5e-5, rtol=5e-5)


def test_sparse_pass2_normalized_cc_score_matches_dense_formula():
    rng = np.random.default_rng(456)
    batch, n_rot, n_trans, n_half = 2, 3, 4, 5
    shifted = (
        rng.normal(size=(batch, n_trans, n_half)) + 1j * rng.normal(size=(batch, n_trans, n_half))
    ).astype(np.complex64)
    score_weight = rng.uniform(0.2, 1.4, size=(batch, n_half)).astype(np.float32)
    proj = (
        rng.normal(size=(batch, n_rot, n_half)) + 1j * rng.normal(size=(batch, n_rot, n_half))
    ).astype(np.complex64)
    half_weights = np.array([1.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float32)
    mask = np.ones((batch, n_rot, n_trans), dtype=bool)
    mask[1, 2, 3] = False

    scores = _score_pass2_bucket_normalized_cc(
        jnp.asarray(shifted),
        jnp.asarray(score_weight),
        jnp.asarray(proj),
        jnp.asarray(half_weights),
        jnp.asarray(mask),
    )

    expected = np.empty((batch, n_rot, n_trans), dtype=np.float32)
    for b in range(batch):
        weighted_proj = proj[b] * half_weights[None, :]
        cross = -2.0 * np.einsum("tn,rn->rt", np.conj(shifted[b]), weighted_proj).real
        norms = np.einsum("n,rn,n->r", score_weight[b], np.abs(proj[b]) ** 2, half_weights)
        expected[b] = (-0.5 * cross) / np.sqrt(np.maximum(norms[:, None], 1e-30))
    expected = np.where(mask, expected, -np.inf)

    np.testing.assert_allclose(np.asarray(scores), expected, atol=5e-5, rtol=5e-5)
    assert np.isneginf(float(np.asarray(scores[1, 2, 3])))


def test_sparse_pass2_winner_take_all_bucket_probs_are_one_hot():
    scores = jnp.array(
        [
            [[1.0, 4.0], [3.0, 2.0]],
            [[-jnp.inf, -jnp.inf], [-jnp.inf, -jnp.inf]],
        ],
        dtype=jnp.float32,
    )
    _, _, best_log_score, best_argmax, _ = _normalize_pass2_bucket(scores)

    probs = _winner_take_all_bucket_probs(scores, best_argmax, best_log_score)

    np.testing.assert_array_equal(np.asarray(probs[0]), np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(probs[1]), np.zeros((2, 2), dtype=np.float32))


def test_sparse_pass2_score_masks_nonfinite_direct_diff2_candidates():
    """Overflowed direct-diff candidates should become impossible, not NaN."""
    scores = _score_pass2_bucket_relion_gpu_diff2(
        jnp.array([[[jnp.inf + 0.0j]]], dtype=jnp.complex64),
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.zeros((1, 1, 1), dtype=jnp.complex64),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1, 1), dtype=bool),
    )

    assert np.isneginf(float(np.asarray(scores[0, 0, 0])))


def test_sparse_pass2_normalization_masks_nonfinite_scores():
    """Non-finite sparse candidates must not poison probabilities or noise sums."""
    scores = jnp.array(
        [
            [[jnp.nan, -jnp.inf], [1.0, -2.0]],
            [[-jnp.inf, jnp.nan], [jnp.inf, -jnp.inf]],
        ],
        dtype=jnp.float32,
    )

    log_z, probs, best_log_score, best_argmax, max_posterior = _normalize_pass2_bucket(scores)

    np.testing.assert_allclose(np.asarray(probs[0]).sum(), 1.0, atol=1e-7)
    assert np.all(np.isfinite(np.asarray(probs)))
    assert np.isfinite(float(np.asarray(log_z[0])))
    assert float(np.asarray(best_log_score[0])) == pytest.approx(1.0)
    assert int(np.asarray(best_argmax[0])) == 2
    assert float(np.asarray(max_posterior[0])) > 0.0

    # The second image has no finite sparse candidate.  Keep it finite and give
    # it zero posterior mass instead of emitting NaNs into the M-step/noise sums.
    np.testing.assert_allclose(np.asarray(probs[1]).sum(), 0.0, atol=1e-7)
    assert float(np.asarray(log_z[1])) == pytest.approx(0.0)
    assert np.isneginf(float(np.asarray(best_log_score[1])))
    assert int(np.asarray(best_argmax[1])) == 0
    assert float(np.asarray(max_posterior[1])) == pytest.approx(0.0)


def test_sparse_pass2_logz_normalization_masks_nonfinite_scores():
    """Precomputed-log-Z sparse normalization must also stay finite."""
    scores = jnp.array([[[jnp.nan, -jnp.inf]], [[0.0, -1.0]]], dtype=jnp.float32)
    log_z = jnp.array([jnp.nan, 0.5], dtype=jnp.float32)

    safe_log_z, probs, best_log_score, best_argmax, max_posterior = _normalize_pass2_bucket_with_log_z(scores, log_z)

    assert float(np.asarray(safe_log_z[0])) == pytest.approx(0.0)
    np.testing.assert_allclose(np.asarray(probs[0]).sum(), 0.0, atol=1e-7)
    assert np.isneginf(float(np.asarray(best_log_score[0])))
    assert int(np.asarray(best_argmax[0])) == 0
    assert float(np.asarray(max_posterior[0])) == pytest.approx(0.0)
    assert np.all(np.isfinite(np.asarray(probs)))
    assert float(np.asarray(safe_log_z[1])) == pytest.approx(0.5)


class TestSparsePass2Bucketed:
    """Bucketed pass-2 must match the per-image reference exactly."""

    def _common_args(self, sig_indices):
        nside_level = 1
        n_images = len(sig_indices)
        ds = MockDataset(n_images=n_images, seed=11)
        volume = _hermitian_volume(VOLUME_SHAPE, seed=17)
        mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
        noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
        translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
        return ds, volume, mean_variance, noise_variance, translations, nside_level

    def _run_both(
        self,
        sig_indices,
        *,
        oversampling_order=1,
        rotation_log_prior=None,
        translation_log_prior=None,
        return_stats=True,
        accumulate_noise=False,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        use_float64_scoring=False,
        image_corrections=None,
        scale_corrections=None,
        image_pre_shifts=None,
        translation_prior_centers=None,
        current_size=None,
        relion_firstiter_score_mode="gaussian",
        relion_firstiter_winner_take_all=False,
    ):
        ds, vol, mv, nv, trans, nside = self._common_args(sig_indices)

        common_kwargs = dict(
            nside_level=nside,
            disc_type="linear_interp",
            oversampling_order=oversampling_order,
            current_size=current_size,
            rotation_log_prior=rotation_log_prior,
            translation_log_prior=translation_log_prior,
            return_stats=return_stats,
            accumulate_noise=accumulate_noise,
            score_with_masked_images=score_with_masked_images,
            half_spectrum_scoring=half_spectrum_scoring,
            use_float64_scoring=use_float64_scoring,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            translation_prior_centers=translation_prior_centers,
            relion_firstiter_score_mode=relion_firstiter_score_mode,
            relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
        )

        out_ref = _compute_pass2_stats_sparse_perimage_reference(ds, vol, mv, nv, trans, sig_indices, **common_kwargs)
        out_bucket = compute_pass2_stats_sparse(ds, vol, mv, nv, trans, sig_indices, **common_kwargs)
        return out_ref, out_bucket

    def test_uniform_counts_match(self):
        """All images have the same number of significant samples (single bucket)."""
        n_images = 4
        sig_indices = [np.array([0, 1, 2], dtype=np.int32)] * n_images
        out_ref, out_bucket = self._run_both(sig_indices, return_stats=True)
        _compare_outputs(out_ref, out_bucket)

    def test_varied_counts_match(self):
        """Per-image rotation counts vary -> multiple buckets."""
        # Different number of significant samples per image, designed to land
        # in different bucket sizes after _exact_bucket_rotation_size quantization.
        sig_indices = [
            np.array([0, 1], dtype=np.int32),  # 2 sigs
            np.array([0, 1, 2, 3, 4], dtype=np.int32),  # 5 sigs
            np.array([5], dtype=np.int32),  # 1 sig
            np.arange(20, dtype=np.int32),  # 20 sigs
            np.array([0, 4, 8, 12], dtype=np.int32),  # 4 sigs
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),  # 9 sigs
        ]
        out_ref, out_bucket = self._run_both(sig_indices, return_stats=True)
        _compare_outputs(out_ref, out_bucket)

    def test_with_translation_grid_match(self):
        """Significance includes both rotation AND translation indices."""
        # n_coarse_trans = 2 in the mock setup. (rot, trans) flat index = rot*2 + trans.
        # Pick a few specific (rot, trans) pairs per image.
        sig_indices = [
            np.array([0, 3], dtype=np.int32),  # (0,0), (1,1)
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
            np.array([7], dtype=np.int32),  # (3,1)
            np.array([0, 2, 4, 6, 8, 10], dtype=np.int32),  # 6 sigs
        ]
        out_ref, out_bucket = self._run_both(sig_indices, return_stats=True)
        _compare_outputs(out_ref, out_bucket)

    def test_with_rotation_log_prior_match(self):
        """Per-rotation log-prior should propagate identically."""
        n_coarse_rot = 48  # rotation_grid_size(nside_level=1) = 48 for nside=2
        from recovar.em.sampling import rotation_grid_size

        n_coarse_rot = rotation_grid_size(1)
        rotation_log_prior = np.linspace(-2.0, 2.0, n_coarse_rot, dtype=np.float32)
        sig_indices = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
            np.array([2, 3, 8, 12], dtype=np.int32),
        ]
        out_ref, out_bucket = self._run_both(
            sig_indices,
            rotation_log_prior=rotation_log_prior,
            return_stats=True,
        )
        _compare_outputs(out_ref, out_bucket)

    def test_with_per_image_translation_log_prior_match(self):
        """Per-image translation log-prior must be reindexed identically."""
        # Coarse trans count = 2 in mock, fine = 2 * 4^1 = 8
        n_images = 3
        sig_indices = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 3, 4, 5], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
        ]
        # Per-image translation prior: shape (n_images, n_coarse_trans=2)
        translation_log_prior = np.array(
            [
                [-1.0, 0.0],
                [0.5, -0.5],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        out_ref, out_bucket = self._run_both(
            sig_indices,
            translation_log_prior=translation_log_prior,
            return_stats=True,
        )
        _compare_outputs(out_ref, out_bucket)

    def test_with_accumulate_noise_match(self):
        """Noise accumulators must match identically too."""
        sig_indices = [
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
            np.array([2], dtype=np.int32),
        ]
        out_ref, out_bucket = self._run_both(
            sig_indices,
            return_stats=True,
            accumulate_noise=True,
            half_spectrum_scoring=True,
            use_float64_scoring=True,
        )
        _compare_outputs(out_ref, out_bucket, atol=1e-4, rtol=1e-4)

    def test_with_translation_prior_centers_noise_match(self):
        """Bucketed pass-2 must accumulate RELION sigma-offset posterior mass."""
        sig_indices = [
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.array([0, 3, 5], dtype=np.int32),
        ]
        translation_prior_centers = np.array(
            [
                [0.25, -0.25],
                [0.50, 0.00],
                [-0.25, 0.25],
                [0.00, 0.50],
            ],
            dtype=np.float32,
        )
        out_ref, out_bucket = self._run_both(
            sig_indices,
            return_stats=True,
            accumulate_noise=True,
            half_spectrum_scoring=True,
            use_float64_scoring=True,
            translation_prior_centers=translation_prior_centers,
        )
        assert out_ref[7].wsum_sigma2_offset > 0.0
        _compare_outputs(out_ref, out_bucket, atol=1e-4, rtol=1e-4)

    def test_full_candidate_lists_match(self):
        """``sig_samples is None`` (full coarse grid) per image."""
        n_images = 3
        sig_indices = [None] * n_images
        out_ref, out_bucket = self._run_both(sig_indices, return_stats=True)
        # The bucketed path accumulates all images in one GPU reduction, while
        # the reference adds per-image reductions.  The remaining difference is
        # float32 accumulation order in Ft_y, not candidate selection logic.
        _compare_outputs(out_ref, out_bucket, atol=5e-5, rtol=5e-5)

    def test_relion_mode_kwargs_match(self):
        """Match the exact production call: half_spectrum + float64 + masked + noise."""
        sig_indices = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
            np.array([3, 7, 11], dtype=np.int32),
            np.array([0, 4], dtype=np.int32),
        ]
        out_ref, out_bucket = self._run_both(
            sig_indices,
            return_stats=True,
            accumulate_noise=True,
            score_with_masked_images=True,
            half_spectrum_scoring=True,
            use_float64_scoring=True,
        )
        _compare_outputs(out_ref, out_bucket, atol=1e-4, rtol=1e-4)
        assert np.asarray(out_bucket[6].log_evidence_per_image).dtype == np.float64
        assert np.asarray(out_bucket[6].best_log_score_per_image).dtype == np.float64

    def test_with_image_corrections_match(self):
        """Per-image image_corrections + scale_corrections + pre_shifts must match.

        These exercise the parity-critical RELION corrections path (avg_norm/normcorr,
        rlnGroupScaleCorrection, old_offset pre-centering).  The bucketed
        path needs to reproduce the per-image scaling exactly, including the
        ``batch_norm`` and noise image-power image-only correction conventions.
        """
        sig_indices = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
            np.array([3, 7, 11], dtype=np.int32),
            np.array([0, 4], dtype=np.int32),
        ]
        n_images = len(sig_indices)
        rng = np.random.default_rng(31)
        image_corrections = (1.0 + 0.1 * rng.standard_normal(n_images)).astype(np.float32)
        scale_corrections = (1.0 + 0.05 * rng.standard_normal(n_images)).astype(np.float32)
        image_pre_shifts = (0.5 * rng.standard_normal((n_images, 2))).astype(np.float32)
        out_ref, out_bucket = self._run_both(
            sig_indices,
            return_stats=True,
            accumulate_noise=True,
            score_with_masked_images=True,
            half_spectrum_scoring=True,
            use_float64_scoring=True,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
        )
        _compare_outputs(out_ref, out_bucket, atol=1e-4, rtol=1e-4)

    def test_firstiter_cc_winner_take_all_match(self):
        """Sparse normalized-CC WTA must match dense per-image run_em."""

        sig_indices = [
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([2], dtype=np.int32),
        ]
        image_corrections = np.asarray([1.0, 0.9, 1.1], dtype=np.float32)
        scale_corrections = np.asarray([1.0, 1.05, 0.95], dtype=np.float32)
        out_ref, out_bucket = self._run_both(
            sig_indices,
            return_stats=True,
            score_with_masked_images=True,
            half_spectrum_scoring=True,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            current_size=6,
            relion_firstiter_score_mode="normalized_cc",
            relion_firstiter_winner_take_all=True,
        )
        _compare_outputs(out_ref, out_bucket, atol=2e-5, rtol=2e-5)
