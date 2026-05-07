import jax
import jax.numpy as jnp
import numpy as np
import pytest

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout
from recovar.em.dense_single_volume.iteration_loop import run_dense_ppca_refinement_with_kclass_schedule
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
from recovar.em.dense_single_volume.helpers.scoring import _e_step_block_scores
from recovar.em.ppca_refinement import (
    HalfsetMeanComparison,
    PoseMarginalPPCAEMState,
    dense_pose_ppca_E_step_blocked,
    iter_dense_ppca_dataset_blocks,
    run_dense_ppca_refinement_loop,
    run_dense_ppca_fused_em_iteration,
    run_dense_ppca_halfset_fused_em_iteration,
    run_local_ppca_fused_em_iteration,
    run_local_ppca_refinement_loop,
)


pytestmark = pytest.mark.unit

IMAGE_SHAPE = (4, 4)
VOLUME_SHAPE = (4, 4, 4)
N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
HALF_VOL = int(np.prod(ftu.volume_shape_to_half_volume_shape(VOLUME_SHAPE)))


def _identity_ctf(params, image_shape, voxel_size, *, half_image=False):
    del voxel_size
    if half_image:
        n_pix = image_shape[0] * (image_shape[1] // 2 + 1)
    else:
        n_pix = image_shape[0] * image_shape[1]
    return jnp.ones((params.shape[0], n_pix), dtype=jnp.float32)


def _make_half_fourier_volume(seed):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
    full = np.fft.fftshift(np.fft.fftn(real)).astype(np.complex64)
    return np.asarray(ftu.full_volume_to_half_volume(full, VOLUME_SHAPE), dtype=np.complex64).reshape(-1)


class _TinyPPCAData:
    def __init__(self, images, *, image_offset=0):
        self.image_shape = IMAGE_SHAPE
        self.volume_shape = VOLUME_SHAPE
        self.grid_size = IMAGE_SHAPE[0]
        self.voxel_size = 1.0
        self.n_images = int(images.shape[0])
        self.n_units = self.n_images
        self.dtype = jnp.complex64
        self.ctf_evaluator = _identity_ctf
        self.CTF_params = np.zeros((self.n_images, 9), dtype=np.float32)
        self._images = np.asarray(images, dtype=np.complex64)
        self._image_offset = int(image_offset)
        self.image_source = self

    def process_images(self, images, apply_image_mask=False):
        del apply_image_mask
        return images

    def process_images_half(self, images, apply_image_mask=False):
        del apply_image_mask
        return images

    @property
    def already_prefetches(self):
        return True

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        del by_image, kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices, dtype=np.int64)
        for start in range(0, indices.size, int(batch_size)):
            idx = indices[start : start + int(batch_size)]
            yield (
                jnp.asarray(self._images[idx]),
                None,
                None,
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx + self._image_offset,
                idx + self._image_offset,
            )

    def get_halfset(self, halfset_id):
        if halfset_id == 0:
            return _TinyPPCAData(self._images[::2], image_offset=0)
        if halfset_id == 1:
            return _TinyPPCAData(self._images[1::2], image_offset=1)
        raise ValueError("halfset_id must be 0 or 1")


@pytest.fixture
def tiny_inputs():
    rng = np.random.default_rng(7)
    images = (
        rng.standard_normal((4, N_HALF)) + 1j * rng.standard_normal((4, N_HALF))
    ).astype(np.complex64)
    dataset = _TinyPPCAData(images)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy()
    translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    mu = _make_half_fourier_volume(10)
    W = (_make_half_fourier_volume(11)[:, None] * np.asarray(0.05, dtype=np.float32)).astype(np.complex64)
    return dataset, mu, W, rotations, translations


def test_dataset_blocks_match_homogeneous_dense_q0_score_convention(tiny_inputs):
    dataset, mu, _W, rotations, translations = tiny_inputs
    block = next(
        iter(
            iter_dense_ppca_dataset_blocks(
                dataset,
                mu,
                None,
                noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
                rotations=rotations,
                translations=translations,
                image_batch_size=2,
                rotation_block_size=2,
                q=0,
                volume_domain="fourier_half",
            )
        )
    )

    _stats, diag = dense_pose_ppca_E_step_blocked(
        block.Y1,
        block.proj_aug,
        block.ctf2_over_noise,
        block.y_norm,
        block.pose_log_prior,
    )
    half_weights = make_half_image_weights(IMAGE_SHAPE)
    proj = block.proj_aug[:, 0, :]
    dense_scores = _e_step_block_scores(
        block.Y1_recon.reshape(block.Y1_recon.shape[0] * block.Y1_recon.shape[1], -1),
        block.y_norm[:, None],
        block.ctf2_over_noise_recon,
        proj * half_weights[None, :],
        (jnp.abs(proj) ** 2) * half_weights[None, :],
        half_weights,
        block.Y1_recon.shape[0],
        block.Y1_recon.shape[1],
        IMAGE_SHAPE,
        VOLUME_SHAPE,
    )
    expected_logz = jax.scipy.special.logsumexp(
        dense_scores.reshape(block.Y1_recon.shape[0], -1),
        axis=-1,
    ) - 0.5 * block.y_norm
    np.testing.assert_allclose(np.asarray(diag.logZ), np.asarray(expected_logz), rtol=5e-4, atol=5e-2)


def test_dataset_backed_dense_ppca_iteration_returns_finite_update(tiny_inputs):
    dataset, mu, W, rotations, translations = tiny_inputs
    result = run_dense_ppca_fused_em_iteration(
        dataset,
        mu,
        W,
        mean_prior=jnp.ones((HALF_VOL,), dtype=jnp.float32) * 10.0,
        W_prior=jnp.ones((HALF_VOL, 1), dtype=jnp.float32) * 5.0,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        rotations=rotations,
        translations=translations,
        image_batch_size=2,
        rotation_block_size=1,
        current_size=4,
        volume_domain="fourier_half",
        enforce_x0=False,
    )

    assert result.mu_half.shape == (HALF_VOL,)
    assert result.W_half.shape == (HALF_VOL, 1)
    assert result.stats.n_images == dataset.n_images
    assert np.isfinite(result.stats.log_likelihood)
    assert np.all(np.isfinite(np.asarray(result.mu_half)))
    assert np.all(np.isfinite(np.asarray(result.W_half)))


def test_dataset_backed_dense_ppca_iteration_is_invariant_to_rotation_blocking(tiny_inputs):
    dataset, mu, W, rotations, translations = tiny_inputs
    mean_prior = jnp.ones((HALF_VOL,), dtype=jnp.float32) * 10.0
    W_prior = jnp.ones((HALF_VOL, 1), dtype=jnp.float32) * 5.0
    common_kwargs = dict(
        mean_prior=mean_prior,
        W_prior=W_prior,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        rotations=rotations,
        translations=translations,
        image_batch_size=2,
        current_size=4,
        volume_domain="fourier_half",
        enforce_x0=False,
        mstep_chunk_size=8,
    )

    unsplit = run_dense_ppca_fused_em_iteration(
        dataset,
        mu,
        W,
        rotation_block_size=rotations.shape[0],
        **common_kwargs,
    )
    split = run_dense_ppca_fused_em_iteration(
        dataset,
        mu,
        W,
        rotation_block_size=1,
        **common_kwargs,
    )

    np.testing.assert_allclose(split.stats.log_likelihood, unsplit.stats.log_likelihood, rtol=5e-4, atol=0.25)
    np.testing.assert_allclose(np.asarray(split.stats.rhs), np.asarray(unsplit.stats.rhs), rtol=3e-3, atol=2e-3)
    np.testing.assert_allclose(
        np.asarray(split.stats.lhs_tri),
        np.asarray(unsplit.stats.lhs_tri),
        rtol=3e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(np.asarray(split.mu_half), np.asarray(unsplit.mu_half), rtol=3e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(split.W_half), np.asarray(unsplit.W_half), rtol=3e-3, atol=2e-3)
    np.testing.assert_array_equal(
        np.asarray(split.diagnostics["best_rotation_idx"]),
        np.asarray(unsplit.diagnostics["best_rotation_idx"]),
    )
    np.testing.assert_array_equal(
        np.asarray(split.diagnostics["best_translation_idx"]),
        np.asarray(unsplit.diagnostics["best_translation_idx"]),
    )


def test_halfset_dense_ppca_iteration_updates_scoring_state(tiny_inputs):
    dataset, mu, W, rotations, translations = tiny_inputs
    state = PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu), jnp.asarray(mu)),
        W_half=(jnp.asarray(W), jnp.asarray(W)),
        mu_score=jnp.asarray(mu),
        W_score=jnp.asarray(W),
        W_prior=jnp.ones((HALF_VOL, 1), dtype=jnp.float32) * 5.0,
        mean_prior=jnp.ones((HALF_VOL,), dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((1,), dtype=jnp.float32),
        schedule_state=None,
    )

    updated = run_dense_ppca_halfset_fused_em_iteration(
        state,
        dataset,
        rotations=rotations,
        translations=translations,
        image_batch_size=2,
        rotation_block_size=1,
        current_size=4,
    )

    assert updated.mu_score.shape == (HALF_VOL,)
    assert updated.W_score.shape == (HALF_VOL, 1)
    assert "delta_rms_mu" in updated.pose_diagnostics
    assert np.isfinite(updated.pose_diagnostics["delta_rms_mu"])


def test_dense_ppca_refinement_loop_advances_current_size_when_gates_pass(tiny_inputs):
    dataset, mu, W, rotations, translations = tiny_inputs
    state = PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu), jnp.asarray(mu)),
        W_half=(jnp.asarray(W), jnp.asarray(W)),
        mu_score=jnp.asarray(mu),
        W_score=jnp.asarray(W),
        W_prior=jnp.ones((HALF_VOL, 1), dtype=jnp.float32) * 5.0,
        mean_prior=jnp.ones((HALF_VOL,), dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((1,), dtype=jnp.float32),
        schedule_state=None,
    )

    def comparator(_state, proposed_current_size):
        return HalfsetMeanComparison(
            means_aligned=True,
            resolution_supports=True,
            no_halfset_drift=True,
            fsc=np.ones((proposed_current_size // 2,), dtype=np.float32),
        )

    final_state, records = run_dense_ppca_refinement_loop(
        state,
        dataset,
        rotations=rotations,
        translations=translations,
        n_iterations=1,
        image_batch_size=2,
        rotation_block_size=1,
        init_current_size=2,
        max_current_size=4,
        halfset_comparator=comparator,
        pose_stability_threshold=1.0,
        mstep_chunk_size=8,
    )

    assert final_state.schedule_state.current_size == 4
    assert len(records) == 1
    assert records[0].resolution_decision.allow_increase


def test_dense_ppca_refinement_loop_blocks_on_halfset_gate(tiny_inputs):
    dataset, mu, W, rotations, translations = tiny_inputs
    state = PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu), jnp.asarray(mu)),
        W_half=(jnp.asarray(W), jnp.asarray(W)),
        mu_score=jnp.asarray(mu),
        W_score=jnp.asarray(W),
        W_prior=jnp.ones((HALF_VOL, 1), dtype=jnp.float32) * 5.0,
        mean_prior=jnp.ones((HALF_VOL,), dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((1,), dtype=jnp.float32),
        schedule_state=None,
    )

    def comparator(_state, proposed_current_size):
        return HalfsetMeanComparison(
            means_aligned=True,
            resolution_supports=False,
            no_halfset_drift=True,
            fsc=np.zeros((proposed_current_size // 2,), dtype=np.float32),
        )

    final_state, records = run_dense_ppca_refinement_loop(
        state,
        dataset,
        rotations=rotations,
        translations=translations,
        n_iterations=1,
        image_batch_size=2,
        rotation_block_size=1,
        init_current_size=2,
        max_current_size=4,
        halfset_comparator=comparator,
        pose_stability_threshold=1.0,
        mstep_chunk_size=8,
    )

    assert final_state.schedule_state.current_size == 2
    assert not records[0].resolution_decision.allow_increase
    assert "halfset mean comparison below requested resolution" in records[0].resolution_decision.reasons


def _all_retained_local_layout(dataset, rotations, translations):
    n_images = dataset.n_images
    n_rot = rotations.shape[0]
    offsets = np.arange(n_images + 1, dtype=np.int64) * n_rot
    return LocalHypothesisLayout(
        n_global_rotations=n_rot,
        n_pixels=n_rot,
        n_psi=1,
        rotation_offsets=offsets,
        rotation_ids_flat=np.tile(np.arange(n_rot, dtype=np.int32), n_images),
        rotations_flat=np.tile(rotations[None, :, :, :], (n_images, 1, 1, 1)).reshape(n_images * n_rot, 3, 3),
        rotation_log_priors_flat=np.zeros((n_images * n_rot,), dtype=np.float32),
        rotation_counts=np.full((n_images,), n_rot, dtype=np.int32),
        translation_grid=translations,
        translation_log_priors=np.zeros((n_images, translations.shape[0]), dtype=np.float32),
    )


def test_exact_local_all_retained_support_matches_dense(tiny_inputs):
    dataset, mu, W, rotations, translations = tiny_inputs
    mean_prior = jnp.ones((HALF_VOL,), dtype=jnp.float32) * 10.0
    W_prior = jnp.ones((HALF_VOL, 1), dtype=jnp.float32) * 5.0
    dense = run_dense_ppca_fused_em_iteration(
        dataset,
        mu,
        W,
        mean_prior=mean_prior,
        W_prior=W_prior,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        rotations=rotations,
        translations=translations,
        image_batch_size=1,
        rotation_block_size=rotations.shape[0],
        volume_domain="fourier_half",
        enforce_x0=False,
        mstep_chunk_size=8,
    )
    local = run_local_ppca_fused_em_iteration(
        dataset,
        mu,
        W,
        mean_prior=mean_prior,
        W_prior=W_prior,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        local_layout=_all_retained_local_layout(dataset, rotations, translations),
        volume_domain="fourier_half",
        enforce_x0=False,
        mstep_chunk_size=8,
    )

    np.testing.assert_allclose(np.asarray(local.stats.rhs), np.asarray(dense.stats.rhs), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(local.stats.lhs_tri), np.asarray(dense.stats.lhs_tri), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(local.mu_half), np.asarray(dense.mu_half), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(local.W_half), np.asarray(dense.W_half), rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(
        np.asarray(local.diagnostics["best_rotation_id"]),
        np.asarray(dense.diagnostics["best_rotation_idx"]),
    )


def test_exact_local_refinement_loop_uses_same_resolution_gate(tiny_inputs):
    dataset, mu, W, rotations, translations = tiny_inputs
    halfsets = (dataset.get_halfset(0), dataset.get_halfset(1))
    layouts = tuple(_all_retained_local_layout(ds, rotations, translations) for ds in halfsets)
    state = PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu), jnp.asarray(mu)),
        W_half=(jnp.asarray(W), jnp.asarray(W)),
        mu_score=jnp.asarray(mu),
        W_score=jnp.asarray(W),
        W_prior=jnp.ones((HALF_VOL, 1), dtype=jnp.float32) * 5.0,
        mean_prior=jnp.ones((HALF_VOL,), dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((1,), dtype=jnp.float32),
        schedule_state=None,
    )

    def comparator(_state, proposed_current_size):
        return HalfsetMeanComparison(
            means_aligned=True,
            resolution_supports=True,
            no_halfset_drift=True,
            fsc=np.ones((proposed_current_size // 2,), dtype=np.float32),
        )

    final_state, records = run_local_ppca_refinement_loop(
        state,
        halfsets,
        layouts,
        n_iterations=1,
        init_current_size=2,
        max_current_size=4,
        halfset_comparator=comparator,
        pose_stability_threshold=1.0,
        mstep_chunk_size=8,
    )

    assert final_state.schedule_state.current_size == 4
    assert records[0].resolution_decision.allow_increase
    assert records[0].diagnostics["path"] == "exact_local"


def test_dense_ppca_wrapper_uses_production_kclass_schedule_bridge(tiny_inputs):
    dataset, mu, W, rotations, translations = tiny_inputs
    state = PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu), jnp.asarray(mu)),
        W_half=(jnp.asarray(W), jnp.asarray(W)),
        mu_score=jnp.asarray(mu),
        W_score=jnp.asarray(W),
        W_prior=jnp.ones((HALF_VOL, 1), dtype=jnp.float32) * 5.0,
        mean_prior=jnp.ones((HALF_VOL,), dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((1,), dtype=jnp.float32),
        schedule_state=None,
    )

    def comparator(_state, proposed_current_size):
        return HalfsetMeanComparison(
            means_aligned=True,
            resolution_supports=True,
            no_halfset_drift=True,
            fsc=np.ones((proposed_current_size // 2,), dtype=np.float32),
        )

    final_state, records, bridge = run_dense_ppca_refinement_with_kclass_schedule(
        state,
        dataset,
        rotations=rotations,
        translations=translations,
        n_iterations=1,
        image_batch_size=2,
        rotation_block_size=1,
        init_current_size=2,
        max_current_size=4,
        halfset_comparator=comparator,
        pose_stability_threshold=1.0,
        mstep_chunk_size=8,
        init_healpix_order=2,
        max_healpix_order=3,
    )

    assert final_state.schedule_state.current_size == 4
    assert records[0].resolution_decision.allow_increase
    assert len(bridge.history) == 1
    assert bridge.history[0]["allowed"]
    assert bridge.state.iteration == 1
