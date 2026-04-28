"""Smoke tests for refine_single_volume's RELION-only path.

Verifies:
1. RELION mode runs without error on a tiny dataset (4 images, 8px, 2 iters)
2. Returns the expected dict keys (including RELION-specific ones)
3. Legacy mode is rejected
4. Invalid mode raises ValueError
5. Convergence state is a RefinementState instance
6. data_vs_prior_trajectory and ave_Pmax_trajectory are populated
"""

from pathlib import Path
import inspect

import numpy as np
import pytest

pytest.importorskip("jax")
import healpy as hp
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.core.configs import ForwardModelConfig
import recovar.em.dense_single_volume.iteration_loop as iteration_loop_module
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.dense_single_volume.local_em_engine import (
    _fetch_indexed_batch,
    _pad_local_big_jit_image_axis,
    _prepare_local_exact_bucket,
    _reorder_bucket_to_indices,
    _try_process_masked_and_unmasked_half_together,
    run_local_em_exact,
)
from recovar.em.dense_single_volume.local_debug import maybe_write_debug_score_dump
from recovar.em.dense_single_volume.local_layout import (
    LocalBucketSpec,
    LocalHypothesisLayout,
    _selected_rotation_matrices,
    build_local_hypothesis_layout,
    build_pass2_hypothesis_layout,
    bucket_local_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_score_pass import (
    compute_reconstruction_support,
    normalize_local_scores,
    score_local_bucket,
    score_local_bucket_abs2_weighted_on_demand,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
from recovar.em.dense_single_volume.iteration_loop import (
    _align_fourier_volume_sign_to_reference,
    _normalize_noise_variance_per_half,
    _replay_control_model_iteration,
    refine_single_volume,
)
from recovar.em.dense_single_volume.helpers.convergence import RefinementState
from recovar.em.dense_single_volume.helpers.local_search import _local_search_engine_rotation_block_size
from recovar.em.dense_single_volume.helpers.resolution import (
    _bootstrap_current_size_relion,
    bootstrap_current_size_from_ini_high_relion,
    clamp_relion_coarse_image_size,
    compute_coarse_image_size,
    should_skip_adaptive_pass2,
)
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    collapse_rotation_posterior_to_direction_prior,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    normalize_direction_prior_per_half,
    relion_translation_prior_center,
    relion_translation_search_base,
)
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    integer_pre_shifts_or_none,
)
from recovar.em.dense_single_volume.helpers.significance import (
    _compute_significance_batched,
)
from recovar.em.dense_single_volume.helpers.oversampling import (
    _compute_pass2_stats_sparse_perimage_reference,
)
from recovar.em.dense_single_volume.helpers.types import NoiseStats, RelionStats
from recovar.em.sampling import (
    apply_relion_rotation_perturbation,
    apply_relion_rotation_perturbation_to_eulers,
    build_local_search_grid_metadata,
    get_local_rotation_grid_fast,
    get_translation_grid,
    get_relion_rotation_grid,
    get_relion_rotation_grid_eulers,
    relion_angular_sampling_deg,
    rotation_grid_n_in_planes,
    rotation_grid_size,
)
from recovar.data_io.image_backends import ParticleImageDataset

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Test constants -- 8x8 images for fast unit tests
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
H, W = IMAGE_SHAPE
N_ROTATIONS = 5
N_TRANSLATIONS = 3
N_IMAGES = 4  # tiny: 2 per half-set
SEED = 42


# ---------------------------------------------------------------------------
# Helpers (same as test_fsc_resolution_loop.py)
# ---------------------------------------------------------------------------


def _hermitian_image_2d(image_shape, seed=42):
    rng = np.random.default_rng(seed)
    real_img = rng.standard_normal(image_shape).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fft2(real_img))
    return jnp.array(ft, dtype=jnp.complex64)


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
    return q.astype(np.float32)


def test_local_search_engine_rotation_block_size_caps_dense_tiles():
    assert _local_search_engine_rotation_block_size(64) == 64
    assert _local_search_engine_rotation_block_size(1024) == 1024
    assert _local_search_engine_rotation_block_size(5000) == 1024


def test_build_local_hypothesis_layout_and_bucketization_preserve_per_image_support(monkeypatch):
    import recovar.em.dense_single_volume.local_layout as local_layout_mod

    call_count = {"value": 0}

    def fake_selector(
        prior_rotation_indices,
        sigma_rot,
        sigma_psi,
        healpix_order,
        sigma_cutoff=3.0,
        *,
        per_image=False,
        grid_metadata=None,
    ):
        _ = (prior_rotation_indices, sigma_rot, sigma_psi, healpix_order, sigma_cutoff, grid_metadata)
        assert per_image
        image_idx = call_count["value"]
        call_count["value"] += 1
        if image_idx == 0:
            return np.array([1, 3], dtype=np.int64), np.array([[0.0, -1.0]], dtype=np.float32)
        return np.array([2, 4, 5], dtype=np.int64), np.array([[0.0, -1.0, -2.0]], dtype=np.float32)

    monkeypatch.setattr(local_layout_mod, "get_local_rotation_grid_fast", fake_selector)
    monkeypatch.setattr(
        local_layout_mod,
        "make_relion_translation_log_prior",
        lambda *args, **kwargs: np.zeros((2, 3), dtype=np.float32),
    )

    prior_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
    rotation_grid_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 8, axis=0)
    translations = np.zeros((3, 2), dtype=np.float32)
    prior_translations = np.zeros((2, 2), dtype=np.float32)

    layout = build_local_hypothesis_layout(
        prior_rotations,
        rotation_grid_rotations,
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=4,
        translations=translations,
        prior_translations=prior_translations,
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata={"mode": "full", "n_pixels": np.int64(192), "n_psi": np.int64(16)},
    )

    np.testing.assert_array_equal(layout.rotation_offsets, np.array([0, 2, 5], dtype=np.int64))
    np.testing.assert_array_equal(layout.rotation_counts, np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(layout.rotation_ids_flat, np.array([1, 3, 2, 4, 5], dtype=np.int32))

    buckets = bucket_local_hypothesis_layout(layout, image_batch_size=2, rotation_block_size=16, max_hypotheses_per_microbatch=64)
    assert len(buckets) == 1
    assert buckets[0].bucket_image_count == 2
    np.testing.assert_array_equal(buckets[0].actual_rotation_counts, np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(buckets[0].local_rotation_ids[0, :2], np.array([1, 3], dtype=np.int32))
    assert not np.any(buckets[0].local_rotation_mask[0, 2:])
    np.testing.assert_array_equal(buckets[0].local_rotation_ids[1, :3], np.array([2, 4, 5], dtype=np.int32))


def test_build_pass2_hypothesis_layout_preserves_sparse_rotation_translation_mask():
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    significant_samples = [
        np.array([0, 3], dtype=np.int32),  # rot 0/trans 0 and rot 1/trans 1
        np.array([2], dtype=np.int32),  # rot 1/trans 0
    ]

    layout = build_pass2_hypothesis_layout(
        significant_samples,
        n_coarse_rotations=rotation_grid_size(0),
        n_coarse_translations=2,
        nside_level=0,
        translations=translations,
        oversampling_order=0,
        rotation_log_prior=np.arange(rotation_grid_size(0), dtype=np.float32),
        translation_log_prior=np.array([0.0, -2.0], dtype=np.float32),
    )

    assert layout.n_images == 2
    np.testing.assert_array_equal(layout.rotation_counts, np.array([2, 1], dtype=np.int32))
    np.testing.assert_array_equal(layout.rotation_offsets, np.array([0, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(layout.rotation_posterior_ids_flat, np.array([0, 1, 1], dtype=np.int32))
    assert layout.sample_mask_flat.shape == (3, 2)
    np.testing.assert_array_equal(
        layout.sample_mask_flat,
        np.array(
            [
                [True, False],
                [False, True],
                [True, False],
            ],
            dtype=bool,
        ),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=2,
        rotation_block_size=4,
        max_hypotheses_per_microbatch=64,
    )
    assert len(buckets) == 1
    row_for_image0 = int(np.flatnonzero(buckets[0].image_indices == 0)[0])
    np.testing.assert_array_equal(
        buckets[0].local_rotation_posterior_ids[row_for_image0, :2],
        np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(buckets[0].local_sample_mask[row_for_image0, :2], layout.sample_mask_flat[:2])
    assert not np.any(buckets[0].local_sample_mask[row_for_image0, 2:])


def test_score_local_bucket_honors_rotation_translation_sample_mask():
    scores = score_local_bucket(
        shifted=jnp.zeros((1, 2, 1), dtype=jnp.complex64),
        ctf2_over_nv=jnp.zeros((1, 1), dtype=jnp.float32),
        proj_weighted=jnp.zeros((1, 2, 1), dtype=jnp.complex64),
        proj_abs2_weighted=jnp.zeros((1, 2, 1), dtype=jnp.float32),
        rotation_log_prior=jnp.array([[0.0, 10.0]], dtype=jnp.float32),
        translation_log_prior=jnp.array([[0.0, 1.0]], dtype=jnp.float32),
        rotation_mask=jnp.array([[True, True]]),
        sample_mask=jnp.array([[[False, True], [True, False]]]),
    )

    scores_np = np.asarray(scores)
    assert np.isneginf(scores_np[0, 0, 0])
    assert scores_np[0, 0, 1] == pytest.approx(1.0)
    assert scores_np[0, 1, 0] == pytest.approx(10.0)
    assert np.isneginf(scores_np[0, 1, 1])


def test_build_local_hypothesis_layout_factorized_matches_per_image_selector():
    healpix_order = 3
    grid_metadata = build_local_search_grid_metadata(healpix_order)
    rotation_grid = get_relion_rotation_grid(healpix_order).astype(np.float32)
    prior_eulers = np.array(
        [
            [12.0, 40.0, 3.0],
            [91.0, 65.0, 29.0],
            [177.0, 23.0, 144.0],
        ],
        dtype=np.float32,
    )

    layout = build_local_hypothesis_layout(
        prior_eulers,
        rotation_grid,
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=healpix_order,
        translations=np.zeros((9, 2), dtype=np.float32),
        prior_translations=np.zeros((3, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata=grid_metadata,
    )

    for image_idx in range(prior_eulers.shape[0]):
        local_ids_ref, local_log_prior_ref = get_local_rotation_grid_fast(
            prior_eulers[image_idx : image_idx + 1],
            np.deg2rad(7.5),
            np.deg2rad(7.5),
            healpix_order,
            sigma_cutoff=3.0,
            per_image=True,
            grid_metadata=grid_metadata,
        )
        start = int(layout.rotation_offsets[image_idx])
        stop = int(layout.rotation_offsets[image_idx + 1])
        np.testing.assert_array_equal(layout.rotation_ids_flat[start:stop], np.asarray(local_ids_ref, dtype=np.int32))
        np.testing.assert_allclose(
            layout.rotation_log_priors_flat[start:stop],
            np.asarray(local_log_prior_ref[0], dtype=np.float32),
        )


def test_selected_rotation_matrices_match_full_perturbed_grid():
    healpix_order = 2
    random_perturbation = 0.25
    angular_sampling_deg = relion_angular_sampling_deg(healpix_order)
    grid_metadata = build_local_search_grid_metadata(healpix_order)
    full_eulers = get_relion_rotation_grid_eulers(healpix_order).astype(np.float32)
    full_perturbed_rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        full_eulers,
        random_perturbation,
        angular_sampling_deg,
    )
    rotation_ids = np.array([0, 3, 17, rotation_grid_size(healpix_order) - 1], dtype=np.int32)

    selected_rotations = _selected_rotation_matrices(
        rotation_ids,
        None,
        grid_metadata,
        random_perturbation=random_perturbation,
        angular_sampling_deg=angular_sampling_deg,
    )

    np.testing.assert_allclose(
        selected_rotations,
        np.asarray(full_perturbed_rotations, dtype=np.float32)[rotation_ids],
        atol=1e-6,
        rtol=1e-6,
    )


def test_exact_local_fine_grid_precompute_auto_policy(monkeypatch):
    monkeypatch.delenv("RECOVAR_RELION_EXACT_LOCAL_PRECOMPUTE_FINE_GRID", raising=False)
    monkeypatch.delenv("RECOVAR_RELION_EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS", raising=False)

    assert iteration_loop_module._precompute_exact_local_fine_grid_enabled(5)
    assert not iteration_loop_module._precompute_exact_local_fine_grid_enabled(6)

    monkeypatch.setenv("RECOVAR_RELION_EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS", "100")
    assert not iteration_loop_module._precompute_exact_local_fine_grid_enabled(5)

    monkeypatch.setenv("RECOVAR_RELION_EXACT_LOCAL_PRECOMPUTE_FINE_GRID", "1")
    assert iteration_loop_module._precompute_exact_local_fine_grid_enabled(6)

    monkeypatch.setenv("RECOVAR_RELION_EXACT_LOCAL_PRECOMPUTE_FINE_GRID", "0")
    assert not iteration_loop_module._precompute_exact_local_fine_grid_enabled(5)


def test_bucket_local_hypothesis_layout_coarsens_large_exact_neighborhoods():
    layout = LocalHypothesisLayout(
        n_global_rotations=2000,
        n_pixels=768,
        n_psi=16,
        rotation_offsets=np.array([0, 1368, 2760, 4176], dtype=np.int64),
        rotation_ids_flat=np.arange(4176, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4176, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4176, dtype=np.float32),
        rotation_counts=np.array([1368, 1392, 1416], dtype=np.int32),
        translation_grid=np.zeros((9, 2), dtype=np.float32),
        translation_log_priors=np.zeros((3, 9), dtype=np.float32),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
    )

    bucket_sizes = sorted(int(bucket.bucket_rotation_count) for bucket in buckets)
    assert bucket_sizes == [1408, 1536]
    assert [int(bucket.bucket_image_count) for bucket in buckets] == [10, 10]
    np.testing.assert_array_equal(buckets[0].actual_rotation_counts, np.array([1368, 1392], dtype=np.int32))
    np.testing.assert_array_equal(buckets[1].actual_rotation_counts, np.array([1416], dtype=np.int32))
    assert buckets[0].local_rotation_mask[0, :1368].all()
    assert not buckets[0].local_rotation_mask[0, 1368:].any()


def test_pad_local_big_jit_image_axis_masks_dummy_rows():
    bucket = LocalBucketSpec(
        image_indices=np.array([2], dtype=np.int32),
        bucket_image_count=3,
        bucket_rotation_count=2,
        actual_rotation_counts=np.array([2], dtype=np.int32),
        local_rotation_ids=np.array([[5, 7]], dtype=np.int32),
        local_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (1, 2, 3, 3)).copy(),
        local_rotation_log_prior=np.zeros((1, 2), dtype=np.float32),
        local_rotation_mask=np.ones((1, 2), dtype=bool),
        translation_log_prior=np.ones((1, 4), dtype=np.float32),
    )
    batch_data = np.ones((1, 8, 8), dtype=np.float32)
    ctf_params = np.ones((1, 9), dtype=np.float32)

    padded, padded_batch, padded_ctf, valid_mask, padded_batch_size = _pad_local_big_jit_image_axis(
        bucket,
        batch_data,
        ctf_params,
    )

    assert padded_batch_size == 3
    assert padded.bucket_image_count == 3
    np.testing.assert_array_equal(valid_mask, np.array([True, False, False]))
    assert padded_batch.shape == (3, 8, 8)
    assert padded_ctf.shape == (3, 9)
    np.testing.assert_array_equal(padded.local_rotation_mask[0], np.array([True, True]))
    assert not np.any(padded.local_rotation_mask[1:])
    np.testing.assert_array_equal(padded.local_rotation_ids[1:], -np.ones((2, 2), dtype=np.int32))
    np.testing.assert_allclose(
        padded.local_rotations[1:, 0],
        np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)),
    )
    np.testing.assert_allclose(padded_ctf[1:], np.broadcast_to(ctf_params[0], (2, 9)))


def test_local_score_debug_dump_records_attempted_pose_metadata(tmp_path):
    class _Dataset:
        def original_image_indices_from_local(self, indices):
            _ = indices
            return np.array([123], dtype=np.int64)

    layout = LocalHypothesisLayout(
        n_global_rotations=16,
        n_pixels=8,
        n_psi=2,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([5, 7], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        translation_log_priors=np.zeros((1, 3), dtype=np.float32),
    )
    bucket = LocalBucketSpec(
        image_indices=np.array([0], dtype=np.int32),
        bucket_image_count=1,
        bucket_rotation_count=2,
        actual_rotation_counts=np.array([2], dtype=np.int32),
        local_rotation_ids=np.array([[5, 7]], dtype=np.int32),
        local_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (1, 2, 3, 3)).copy(),
        local_rotation_log_prior=np.zeros((1, 2), dtype=np.float32),
        local_rotation_mask=np.ones((1, 2), dtype=bool),
        translation_log_prior=np.zeros((1, 3), dtype=np.float32),
    )
    def write_dump(*, current_size):
        return maybe_write_debug_score_dump(
            experiment_dataset=_Dataset(),
            local_layout=layout,
            bucket=bucket,
            image_pre_shifts=np.array([[2.0, -1.0]], dtype=np.float32),
            scores=np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32),
            probs=np.array([[[0.05, 0.10, 0.15], [0.20, 0.25, 0.25]]], dtype=np.float32),
            log_Z=np.array([7.0], dtype=np.float32),
            best_log_score=np.array([6.0], dtype=np.float32),
            max_posterior=np.array([0.25], dtype=np.float32),
            reconstruction_sample_mask=np.ones((1, 2, 3), dtype=bool),
            reconstruction_rotation_mask=np.ones((1, 2), dtype=bool),
            n_significant_samples=np.array([6], dtype=np.int32),
            current_size=current_size,
            debug_iteration=9,
            dump_dir=tmp_path,
            pending_targets={123},
            requested_current_sizes={8},
        )

    pending = write_dump(current_size=7)
    assert pending == {123}
    assert not (tmp_path / "local_score_it009_image_123.npz").exists()

    pending = write_dump(current_size=8)

    assert pending == set()
    with np.load(tmp_path / "local_score_it009_image_123.npz") as dump:
        np.testing.assert_array_equal(dump["local_rotation_indices"], np.array([5, 7], dtype=np.int32))
        np.testing.assert_array_equal(dump["debug_iteration"], np.array([9], dtype=np.int32))
        assert dump["local_rotation_eulers"].shape == (2, 3)
        assert dump["local_rotation_matrices"].shape == (2, 3, 3)
        np.testing.assert_array_equal(
            dump["candidate_pose_rotation_indices"],
            np.array([[5, 5, 5], [7, 7, 7]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            dump["candidate_pose_translation_indices"],
            np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32),
        )
        np.testing.assert_array_equal(dump["best_score_rotation_global_id"], np.array([7], dtype=np.int32))
        np.testing.assert_array_equal(dump["best_score_translation_index"], np.array([2], dtype=np.int32))
        np.testing.assert_allclose(dump["best_score_translation"], np.array([[0.0, 1.0]], dtype=np.float32))


def test_run_local_search_iteration_dispatches_exact_engine(monkeypatch, rng):
    mock_dataset = MockDataset(N_IMAGES, rng)
    called = {"engine": None}

    def fake_exact(*args, **kwargs):
        _ = args
        called["engine"] = "exact_v1"
        if kwargs.get("accumulate_noise", False):
            return (
                jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
                jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
                np.zeros(mock_dataset.n_units, dtype=np.int32),
                RelionStats(
                    log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.zeros(6, dtype=jnp.float32),
                ),
                NoiseStats(
                    wsum_sigma2_noise=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                    wsum_img_power=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=0.0,
                ),
            )
        raise AssertionError("test expects accumulate_noise=True")

    monkeypatch.setattr(iteration_loop_module, "_run_local_search_iteration_exact_v1", fake_exact)

    prior_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], mock_dataset.n_units, axis=0)
    rotation_grid_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 6, axis=0)
    rotation_grid_eulers = np.zeros((6, 3), dtype=np.float32)
    translations = np.zeros((N_TRANSLATIONS, 2), dtype=np.float32)
    prior_translations = np.zeros((mock_dataset.n_units, 2), dtype=np.float32)

    outputs = iteration_loop_module._run_local_search_iteration(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order=1,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=translations,
        prior_translations=prior_translations,
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        disc_type="linear_interp",
        image_batch_size=2,
        rotation_block_size=4,
        current_size=4,
        accumulate_noise=True,
    )

    assert called["engine"] == "exact_v1"
    assert len(outputs) == 5

def test_run_local_search_iteration_exact_engine_uses_translation_prior_reference_grid(monkeypatch, rng):
    mock_dataset = MockDataset(1, rng)
    captured = {}

    def fake_exact(*args, **kwargs):
        _ = (
            args,
        )
        captured["translation_prior_reference_translations"] = (
            None
            if kwargs.get("translation_prior_reference_translations") is None
            else np.asarray(kwargs["translation_prior_reference_translations"], dtype=np.float32).copy()
        )
        captured["translation_prior_centers"] = (
            None
            if kwargs.get("translation_prior_centers") is None
            else np.asarray(kwargs["translation_prior_centers"], dtype=np.float32).copy()
        )
        return (
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            np.zeros(mock_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(6, dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_img_power=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=0.0,
            ),
        )

    monkeypatch.setattr(iteration_loop_module, "_run_local_search_iteration_exact_v1", fake_exact)

    prior_rotations = np.zeros((1, 3), dtype=np.float32)
    rotation_grid_rotations = get_relion_rotation_grid(0).astype(np.float32)
    rotation_grid_eulers = get_relion_rotation_grid_eulers(0).astype(np.float32)
    translations = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float32)
    reference_translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    prior_centers = np.array([[0.25, -0.5]], dtype=np.float32)

    outputs = iteration_loop_module._run_local_search_iteration(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=translations,
        prior_translations=np.zeros((1, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        disc_type="linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=4,
        accumulate_noise=True,
        translation_prior_reference_translations=reference_translations,
        translation_prior_centers=prior_centers,
    )

    np.testing.assert_allclose(
        captured["translation_prior_reference_translations"],
        reference_translations,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        captured["translation_prior_centers"],
        prior_centers,
        atol=1e-6,
    )
    assert len(outputs) == 5


def test_run_local_search_iteration_exact_engine_uses_model_sigma_for_translation_prior(monkeypatch, rng):
    mock_dataset = MockDataset(1, rng)
    captured = {}

    def fake_build_local_hypothesis_layout(
        prior_rotations,
        rotation_grid_rotations,
        sigma_rot,
        sigma_psi,
        healpix_order,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        voxel_size,
        *,
        grid_metadata,
        translation_prior_reference_translations=None,
        rotation_grid_random_perturbation=0.0,
        rotation_grid_angular_sampling_deg=None,
    ):
        captured["offset_range_pixels"] = offset_range_pixels
        captured["sigma_offset_angstrom"] = sigma_offset_angstrom
        captured["rotation_grid_random_perturbation"] = rotation_grid_random_perturbation
        captured["rotation_grid_angular_sampling_deg"] = rotation_grid_angular_sampling_deg
        captured["translation_prior_reference_translations"] = (
            None
            if translation_prior_reference_translations is None
            else np.asarray(translation_prior_reference_translations, dtype=np.float32).copy()
        )
        return LocalHypothesisLayout(
            n_global_rotations=1,
            n_pixels=1,
            n_psi=1,
            rotation_offsets=np.array([0, 1], dtype=np.int64),
            rotation_ids_flat=np.array([0], dtype=np.int32),
            rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 1, axis=0),
            rotation_log_priors_flat=np.zeros(1, dtype=np.float32),
            rotation_counts=np.array([1], dtype=np.int32),
            translation_grid=np.asarray(translations, dtype=np.float32),
            translation_log_priors=np.zeros((1, np.asarray(translations).shape[0]), dtype=np.float32),
        )

    def fake_run_local_em_exact(*args, **kwargs):
        _ = args
        captured["reconstruct_significant_only"] = kwargs.get("reconstruct_significant_only")
        captured["adaptive_fraction"] = kwargs.get("adaptive_fraction")
        captured["max_significants"] = kwargs.get("max_significants")
        return (
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            np.zeros(mock_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(1, dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_img_power=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=0.0,
            ),
        )

    monkeypatch.setattr(iteration_loop_module, "build_local_hypothesis_layout", fake_build_local_hypothesis_layout)
    monkeypatch.setattr(iteration_loop_module, "run_local_em_exact", fake_run_local_em_exact)

    prior_rotations = np.zeros((1, 3), dtype=np.float32)
    rotation_grid_rotations = get_relion_rotation_grid(0).astype(np.float32)
    rotation_grid_eulers = get_relion_rotation_grid_eulers(0).astype(np.float32)
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    reference_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    outputs = iteration_loop_module._run_local_search_iteration_exact_v1(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=translations,
        prior_translations=np.zeros((1, 2), dtype=np.float32),
        sigma_offset_angstrom=1.25,
        offset_range_pixels=3.5,
        disc_type="linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=4,
        accumulate_noise=True,
        translation_prior_reference_translations=reference_translations,
    )

    assert captured["offset_range_pixels"] is None
    assert captured["sigma_offset_angstrom"] == 1.25
    assert captured["rotation_grid_random_perturbation"] == 0.0
    assert captured["rotation_grid_angular_sampling_deg"] is None
    assert captured["reconstruct_significant_only"] is True
    assert captured["adaptive_fraction"] == pytest.approx(0.999)
    assert captured["max_significants"] == -1
    np.testing.assert_allclose(
        captured["translation_prior_reference_translations"],
        reference_translations,
        atol=1e-6,
    )
    assert len(outputs) == 5


def test_run_local_search_iteration_exact_engine_uses_factorized_prior_metadata_for_perturbed_grid(
    monkeypatch,
    rng,
):
    from recovar import utils

    mock_dataset = MockDataset(1, rng)
    captured = {}

    def fake_build_local_hypothesis_layout(
        prior_rotations,
        rotation_grid_rotations,
        sigma_rot,
        sigma_psi,
        healpix_order,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        voxel_size,
        *,
        grid_metadata,
        translation_prior_reference_translations=None,
        rotation_grid_random_perturbation=0.0,
        rotation_grid_angular_sampling_deg=None,
    ):
        _ = (
            prior_rotations,
            sigma_rot,
            sigma_psi,
            healpix_order,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            voxel_size,
            translation_prior_reference_translations,
        )
        captured["grid_metadata_mode"] = grid_metadata["mode"]
        captured["n_pixels"] = int(grid_metadata["n_pixels"])
        captured["n_psi"] = int(grid_metadata["n_psi"])
        captured["rotation_grid_random_perturbation"] = rotation_grid_random_perturbation
        captured["rotation_grid_angular_sampling_deg"] = rotation_grid_angular_sampling_deg
        captured["scored_rotations"] = np.asarray(rotation_grid_rotations, dtype=np.float32).copy()
        return LocalHypothesisLayout(
            n_global_rotations=rotation_grid_rotations.shape[0],
            n_pixels=1,
            n_psi=1,
            rotation_offsets=np.array([0, 1], dtype=np.int64),
            rotation_ids_flat=np.array([0], dtype=np.int32),
            rotations_flat=np.asarray(rotation_grid_rotations[:1], dtype=np.float32),
            rotation_log_priors_flat=np.zeros(1, dtype=np.float32),
            rotation_counts=np.array([1], dtype=np.int32),
            translation_grid=np.asarray(translations, dtype=np.float32),
            translation_log_priors=np.zeros((1, np.asarray(translations).shape[0]), dtype=np.float32),
        )

    def fake_run_local_em_exact(*args, **kwargs):
        _ = args
        captured["max_significants"] = kwargs.get("max_significants")
        return (
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            np.zeros(mock_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(rotation_grid_size(1), dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_img_power=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=0.0,
            ),
        )

    monkeypatch.setattr(iteration_loop_module, "build_local_hypothesis_layout", fake_build_local_hypothesis_layout)
    monkeypatch.setattr(iteration_loop_module, "run_local_em_exact", fake_run_local_em_exact)

    healpix_order = 1
    canonical_rotations = get_relion_rotation_grid(healpix_order).astype(np.float32)
    perturbed_rotations = apply_relion_rotation_perturbation(
        canonical_rotations,
        random_perturbation=0.3,
        angular_sampling_deg=relion_angular_sampling_deg(healpix_order),
    ).astype(np.float32)
    perturbed_eulers = utils.R_to_relion(perturbed_rotations, degrees=True).astype(np.float32)
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    # A perturbed full Euler table no longer factorizes. RELION still builds
    # local priors from the canonical Healpix direction and psi axes, then
    # applies SamplingPerturbation only when scoring the trial rotations.
    assert build_local_search_grid_metadata(
        healpix_order,
        grid_eulers=perturbed_eulers,
        grid_rotations=perturbed_rotations,
    )["mode"] == "full"

    outputs = iteration_loop_module._run_local_search_iteration_exact_v1(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        np.zeros((1, 3), dtype=np.float32),
        perturbed_rotations,
        perturbed_eulers,
        healpix_order=healpix_order,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=translations,
        prior_translations=np.zeros((1, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=2.0,
        disc_type="linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=4,
        accumulate_noise=True,
    )

    assert captured["grid_metadata_mode"] == "factorized"
    assert captured["n_pixels"] == hp.nside2npix(2**healpix_order)
    assert captured["n_psi"] == rotation_grid_n_in_planes(healpix_order)
    assert captured["max_significants"] == -1
    assert captured["rotation_grid_random_perturbation"] == 0.0
    assert captured["rotation_grid_angular_sampling_deg"] is None
    np.testing.assert_allclose(captured["scored_rotations"], perturbed_rotations)
    assert len(outputs) == 5


def test_run_local_em_exact_matches_dense_engine_on_single_image_local_grid(rng):
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=101)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(2, seed=99)
    translations = np.zeros((1, 2), dtype=np.float32)
    rotation_log_prior = np.zeros(2, dtype=np.float32)
    translation_log_prior = np.zeros((1, 1), dtype=np.float32)

    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.asarray(local_rotations, dtype=np.float32),
        rotation_log_priors_flat=np.asarray(rotation_log_prior, dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.asarray(translation_log_prior, dtype=np.float32),
    )

    exact_outputs = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        accumulate_noise=True,
        reconstruct_significant_only=False,
        return_profile=False,
    )
    _, ha_dense, Ft_y_dense, Ft_ctf_dense, stats_dense, noise_dense = run_em(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        np.asarray(local_rotations, dtype=np.float32),
        np.asarray(translations, dtype=np.float32),
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        rotation_log_prior=rotation_log_prior[None, :],
        translation_log_prior=translation_log_prior,
        image_indices=np.array([0], dtype=np.int32),
        score_with_masked_images=True,
        return_stats=True,
        accumulate_noise=True,
        sparse_pass2=False,
    )

    Ft_y_exact, Ft_ctf_exact, ha_exact, stats_exact, noise_exact = exact_outputs
    assert np.asarray(Ft_y_exact).size == VOLUME_SIZE
    assert np.asarray(Ft_ctf_exact).size == VOLUME_SIZE
    config = ForwardModelConfig.from_dataset(dataset, disc_type="linear_interp", process_fn=dataset.process_images)
    noise_variance_half = jnp.ones(dataset.image_shape[0] * (dataset.image_shape[1] // 2 + 1), dtype=jnp.float32)
    half_weights = make_half_image_weights(dataset.image_shape)
    bucket = bucket_local_hypothesis_layout(
        local_layout,
        image_batch_size=1,
        rotation_block_size=4,
        max_hypotheses_per_microbatch=32768,
    )[0]
    batch_data, ctf_params, fetched_indices = _fetch_indexed_batch(dataset, bucket.image_indices)
    bucket = _reorder_bucket_to_indices(bucket, fetched_indices)
    (
        shifted_score_half,
        shifted_recon_half,
        _batch_norm,
        ctf2_over_nv_half,
        _processed_score_half,
        _real_space_pre_shift_applied,
    ) = (
        _prepare_local_exact_bucket(
            dataset,
            batch_data,
            ctf_params,
            bucket.image_indices,
            noise_variance_half,
            jnp.asarray(local_layout.translation_grid),
            config,
            half_weights,
            batch_size=1,
            n_trans=1,
            score_with_masked_images=True,
        )
    )
    flat_rotations = flatten_bucket_rotations(jnp.asarray(bucket.local_rotations))
    n_half = dataset.image_shape[0] * (dataset.image_shape[1] // 2 + 1)
    proj_half_flat = core.slice_volume(
        mean,
        flat_rotations,
        dataset.image_shape,
        dataset.volume_shape,
        "linear_interp",
        half_image=True,
    )
    proj_abs2_half_flat = jnp.abs(proj_half_flat) ** 2
    proj_half = proj_half_flat.reshape(1, bucket.bucket_rotation_count, n_half)
    proj_abs2 = proj_abs2_half_flat.reshape(1, bucket.bucket_rotation_count, n_half)
    proj_weighted = proj_half * half_weights[None, None, :]
    proj_abs2_weighted = proj_abs2 * half_weights[None, None, :]
    scores = score_local_bucket(
        shifted_score_half.reshape(1, 1, -1),
        ctf2_over_nv_half,
        proj_weighted,
        proj_abs2_weighted,
        jnp.asarray(bucket.local_rotation_log_prior),
        jnp.asarray(bucket.translation_log_prior),
        jnp.asarray(bucket.local_rotation_mask),
    )
    _log_Z, probs, _best_log_score, _best_argmax, _max_posterior = normalize_local_scores(scores)
    shifted_recon_split = shifted_recon_half.reshape(1, 1, -1)
    manual_summed = compute_local_weighted_sums(probs, shifted_recon_split)
    manual_ctf_probs = compute_local_ctf_sums(probs, ctf2_over_nv_half)
    Ft_y_manual = core.adjoint_slice_volume(
        flatten_bucket_rows(manual_summed),
        flat_rotations,
        dataset.image_shape,
        dataset.volume_shape,
        "linear_interp",
        half_image=True,
        half_volume=False,
    )
    Ft_ctf_manual = core.adjoint_slice_volume(
        flatten_bucket_rows(manual_ctf_probs),
        flat_rotations,
        dataset.image_shape,
        dataset.volume_shape,
        "linear_interp",
        half_image=True,
        half_volume=False,
    )

    np.testing.assert_allclose(np.asarray(Ft_y_exact), np.asarray(Ft_y_manual), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_exact), np.asarray(Ft_ctf_manual), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_y_exact), np.asarray(Ft_y_dense), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_exact), np.asarray(Ft_ctf_dense), atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(ha_exact, ha_dense)
    np.testing.assert_allclose(
        np.asarray(stats_exact.log_evidence_per_image),
        np.asarray(stats_dense.log_evidence_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_exact.rotation_posterior_sums[:2]),
        np.asarray(stats_dense.rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_exact.wsum_sigma2_noise),
        np.asarray(noise_dense.wsum_sigma2_noise),
        atol=1e-5,
        rtol=1e-5,
    )


def test_combined_masked_preprocess_matches_separate_relion_fft(monkeypatch, rng):
    monkeypatch.setenv("RECOVAR_RELION_NUMPY_IMAGE_FFT", "1")

    backend = object.__new__(ParticleImageDataset)
    backend.image_size = 8
    backend.image_shape = (8, 8)
    backend.D = 8
    backend.padding = 0
    backend.dtype = np.complex64
    backend.image_mask_mode = "relion_background_fill"
    backend.image_mask = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    backend.mask = backend.image_mask
    backend.mult = -1
    backend.data_multiplier = -1

    class _ImageSource:
        pass

    class _Dataset:
        process_images_half = backend.process_images_half
        image_source = _ImageSource()
        image_mask = backend.image_mask

    _Dataset.image_source.backend = backend
    batch = rng.standard_normal((3, 8, 8)).astype(np.float32)

    combined = _try_process_masked_and_unmasked_half_together(_Dataset(), batch)
    assert combined is not None
    score_half, recon_half = combined

    expected_score = backend.process_images_half(batch, apply_image_mask=True)
    expected_recon = backend.process_images_half(batch, apply_image_mask=False)
    np.testing.assert_array_equal(np.asarray(score_half), np.asarray(expected_score))
    np.testing.assert_array_equal(np.asarray(recon_half), np.asarray(expected_recon))


def test_weighted_abs2_on_demand_scores_match_materialized(rng):
    batch_size = 2
    n_rot = 3
    n_trans = 2
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    shifted = (
        rng.standard_normal((batch_size, n_trans, n_half))
        + 1j * rng.standard_normal((batch_size, n_trans, n_half))
    ).astype(np.complex64)
    proj = (
        rng.standard_normal((batch_size, n_rot, n_half))
        + 1j * rng.standard_normal((batch_size, n_rot, n_half))
    ).astype(np.complex64)
    ctf2_over_nv = rng.uniform(0.1, 2.0, size=(batch_size, n_half)).astype(np.float32)
    half_weights = np.asarray(make_half_image_weights(IMAGE_SHAPE), dtype=np.float32)
    proj_weighted = proj * half_weights[None, None, :]
    proj_abs2_weighted = (np.abs(proj) ** 2).astype(np.float32) * half_weights[None, None, :]
    rotation_log_prior = rng.standard_normal((batch_size, n_rot)).astype(np.float32) * 0.01
    translation_log_prior = rng.standard_normal((batch_size, n_trans)).astype(np.float32) * 0.01
    rotation_mask = np.ones((batch_size, n_rot), dtype=bool)

    expected = score_local_bucket(
        jnp.asarray(shifted),
        jnp.asarray(ctf2_over_nv),
        jnp.asarray(proj_weighted),
        jnp.asarray(proj_abs2_weighted),
        jnp.asarray(rotation_log_prior),
        jnp.asarray(translation_log_prior),
        jnp.asarray(rotation_mask),
    )
    actual = score_local_bucket_abs2_weighted_on_demand(
        jnp.asarray(shifted),
        jnp.asarray(ctf2_over_nv),
        jnp.asarray(proj_weighted),
        jnp.asarray(half_weights),
        jnp.asarray(rotation_log_prior),
        jnp.asarray(translation_log_prior),
        jnp.asarray(rotation_mask),
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_run_local_em_exact_windowed_path_computes_reconstruction_abs2_without_full_buffer(rng):
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=201)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(2, seed=109)
    translations = np.zeros((1, 2), dtype=np.float32)
    rotation_log_prior = np.zeros(2, dtype=np.float32)
    translation_log_prior = np.zeros((1, 1), dtype=np.float32)

    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.asarray(local_rotations, dtype=np.float32),
        rotation_log_priors_flat=np.asarray(rotation_log_prior, dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.asarray(translation_log_prior, dtype=np.float32),
    )

    outputs = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=4,
        accumulate_noise=False,
        reconstruct_significant_only=False,
        return_profile=False,
    )

    Ft_y_exact, Ft_ctf_exact, ha_exact, stats_exact = outputs
    assert Ft_y_exact.shape == (VOLUME_SIZE,)
    assert Ft_ctf_exact.shape == (VOLUME_SIZE,)
    assert ha_exact.shape == (1,)
    assert stats_exact.max_posterior_per_image.shape == (1,)


def test_run_local_em_exact_windowed_with_pre_shifts_matches_dense_engine(rng):
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=211)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(5, seed=219)
    translations = np.array(
        [
            [0.0, 0.0],
            [0.5, -0.5],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rotation_log_prior = np.linspace(0.0, -1.0, 5, dtype=np.float32)
    translation_log_prior = np.array([[0.2, -0.1, -0.4]], dtype=np.float32)

    local_layout = LocalHypothesisLayout(
        n_global_rotations=5,
        n_pixels=5,
        n_psi=1,
        rotation_offsets=np.array([0, 5], dtype=np.int64),
        rotation_ids_flat=np.arange(5, dtype=np.int32),
        rotations_flat=np.asarray(local_rotations, dtype=np.float32),
        rotation_log_priors_flat=np.asarray(rotation_log_prior, dtype=np.float32),
        rotation_counts=np.array([5], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.asarray(translation_log_prior, dtype=np.float32),
    )

    exact_outputs = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=6,
        accumulate_noise=True,
        reconstruct_significant_only=False,
        return_profile=False,
        score_with_masked_images=True,
        half_spectrum_scoring=True,
        image_corrections=np.array([1.3], dtype=np.float32),
        scale_corrections=np.array([0.7], dtype=np.float32),
        image_pre_shifts=np.array([[0.5, -1.0]], dtype=np.float32),
    )

    _, ha_dense, Ft_y_dense, Ft_ctf_dense, stats_dense, noise_dense = run_em(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        np.asarray(local_rotations, dtype=np.float32),
        np.asarray(translations, dtype=np.float32),
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        rotation_log_prior=rotation_log_prior[None, :],
        translation_log_prior=translation_log_prior,
        image_indices=np.array([0], dtype=np.int32),
        score_with_masked_images=True,
        return_stats=True,
        accumulate_noise=True,
        sparse_pass2=False,
        half_spectrum_scoring=True,
        image_corrections=np.array([1.3], dtype=np.float32),
        scale_corrections=np.array([0.7], dtype=np.float32),
        image_pre_shifts=np.array([[0.5, -1.0]], dtype=np.float32),
        current_size=6,
    )

    Ft_y_exact, Ft_ctf_exact, ha_exact, stats_exact, noise_exact = exact_outputs
    np.testing.assert_array_equal(ha_exact, ha_dense)
    np.testing.assert_allclose(np.asarray(Ft_y_exact), np.asarray(Ft_y_dense), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_exact), np.asarray(Ft_ctf_dense), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(stats_exact.log_evidence_per_image),
        np.asarray(stats_dense.log_evidence_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_exact.best_log_score_per_image),
        np.asarray(stats_dense.best_log_score_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_exact.max_posterior_per_image),
        np.asarray(stats_dense.max_posterior_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_exact.rotation_posterior_sums),
        np.asarray(stats_dense.rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_exact.wsum_sigma2_noise),
        np.asarray(noise_dense.wsum_sigma2_noise),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_exact.wsum_img_power),
        np.asarray(noise_dense.wsum_img_power),
        atol=1e-5,
        rtol=1e-5,
    )


def test_run_local_em_exact_batched_matches_single_image_chunks(rng):
    dataset = MockDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=231)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(6, seed=233)
    translations = np.array(
        [
            [0.0, 0.0],
            [0.5, -0.5],
        ],
        dtype=np.float32,
    )
    rotation_ids = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
        np.array([0, 2, 4, 5], dtype=np.int32),
    ]
    rotation_counts = np.asarray([ids.size for ids in rotation_ids], dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    rotation_ids_flat = np.concatenate(rotation_ids).astype(np.int32)
    rotation_log_priors_flat = np.linspace(0.0, -0.8, rotation_ids_flat.size, dtype=np.float32)
    translation_log_prior = np.array(
        [
            [0.0, -0.5],
            [-0.2, 0.1],
            [0.3, -0.4],
        ],
        dtype=np.float32,
    )
    local_layout = LocalHypothesisLayout(
        n_global_rotations=all_rotations.shape[0],
        n_pixels=6,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=np.asarray(all_rotations[rotation_ids_flat], dtype=np.float32),
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=rotation_counts,
        translation_grid=translations,
        translation_log_priors=translation_log_prior,
    )

    batched = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=True,
        reconstruct_significant_only=False,
        return_profile=True,
        score_with_masked_images=True,
        half_spectrum_scoring=True,
        image_corrections=np.array([1.3, 0.8, 1.1], dtype=np.float32),
        scale_corrections=np.array([0.7, 1.2, 0.9], dtype=np.float32),
        image_pre_shifts=np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    )
    single = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=True,
        reconstruct_significant_only=False,
        return_profile=True,
        score_with_masked_images=True,
        half_spectrum_scoring=True,
        image_corrections=np.array([1.3, 0.8, 1.1], dtype=np.float32),
        scale_corrections=np.array([0.7, 1.2, 0.9], dtype=np.float32),
        image_pre_shifts=np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    )

    Ft_y_b, Ft_ctf_b, ha_b, stats_b, noise_b, profile_b = batched
    Ft_y_s, Ft_ctf_s, ha_s, stats_s, noise_s, profile_s = single
    assert int(profile_b["n_chunks"]) < int(profile_s["n_chunks"])
    np.testing.assert_array_equal(ha_b, ha_s)
    np.testing.assert_allclose(np.asarray(Ft_y_b), np.asarray(Ft_y_s), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_b), np.asarray(Ft_ctf_s), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(stats_b.log_evidence_per_image),
        np.asarray(stats_s.log_evidence_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_b.best_log_score_per_image),
        np.asarray(stats_s.best_log_score_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_b.max_posterior_per_image),
        np.asarray(stats_s.max_posterior_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_b.rotation_posterior_sums),
        np.asarray(stats_s.rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_b.wsum_sigma2_noise),
        np.asarray(noise_s.wsum_sigma2_noise),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_b.wsum_img_power),
        np.asarray(noise_s.wsum_img_power),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_b.wsum_sigma2_offset == pytest.approx(noise_s.wsum_sigma2_offset, abs=1e-5)
    assert noise_b.sumw == pytest.approx(noise_s.sumw, abs=1e-5)


def test_run_local_em_exact_default_fused_path_matches_materialized_split(monkeypatch, rng):
    dataset = MockDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=531)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(6, seed=533)
    translations = np.array(
        [
            [0.0, 0.0],
            [0.5, -0.5],
        ],
        dtype=np.float32,
    )
    rotation_ids = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
        np.array([0, 2, 4, 5], dtype=np.int32),
    ]
    rotation_counts = np.asarray([ids.size for ids in rotation_ids], dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    rotation_ids_flat = np.concatenate(rotation_ids).astype(np.int32)
    rotation_log_priors_flat = np.linspace(0.0, -0.8, rotation_ids_flat.size, dtype=np.float32)
    translation_log_prior = np.array(
        [
            [0.0, -0.5],
            [-0.2, 0.1],
            [0.3, -0.4],
        ],
        dtype=np.float32,
    )
    local_layout = LocalHypothesisLayout(
        n_global_rotations=all_rotations.shape[0],
        n_pixels=6,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=np.asarray(all_rotations[rotation_ids_flat], dtype=np.float32),
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=rotation_counts,
        translation_grid=translations,
        translation_log_priors=translation_log_prior,
    )
    common_kwargs = dict(
        image_batch_size=3,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=True,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=True,
        half_spectrum_scoring=False,
        image_corrections=np.array([1.3, 0.8, 1.1], dtype=np.float32),
        scale_corrections=np.array([0.7, 1.2, 0.9], dtype=np.float32),
        image_pre_shifts=np.array([[0.5, -1.0], [-1.0, 1.25], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
    )

    monkeypatch.delenv("RECOVAR_RELION_EXACT_LOCAL_MATERIALIZE_PROJECTION_ABS2", raising=False)
    monkeypatch.delenv("RECOVAR_RELION_EXACT_LOCAL_FUSED_SCORE_MSTEP", raising=False)
    default = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )
    monkeypatch.setenv("RECOVAR_RELION_EXACT_LOCAL_MATERIALIZE_PROJECTION_ABS2", "1")
    monkeypatch.setenv("RECOVAR_RELION_EXACT_LOCAL_FUSED_SCORE_MSTEP", "0")
    materialized = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    Ft_y_default, Ft_ctf_default, hard_default, stats_default, noise_default, profile_default = default
    Ft_y_mat, Ft_ctf_mat, hard_mat, stats_mat, noise_mat, profile_mat = materialized
    assert bool(profile_default["materialize_projection_abs2"]) is False
    assert bool(profile_default["fused_score_mstep_enabled"]) is True
    assert bool(profile_mat["materialize_projection_abs2"]) is True
    assert bool(profile_mat["fused_score_mstep_enabled"]) is False
    np.testing.assert_array_equal(hard_default, hard_mat)
    np.testing.assert_allclose(np.asarray(Ft_y_default), np.asarray(Ft_y_mat), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_default), np.asarray(Ft_ctf_mat), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(stats_default.log_evidence_per_image),
        np.asarray(stats_mat.log_evidence_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_default.max_posterior_per_image),
        np.asarray(stats_mat.max_posterior_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_default.rotation_posterior_sums),
        np.asarray(stats_mat.rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_default.wsum_sigma2_noise),
        np.asarray(noise_mat.wsum_sigma2_noise),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_default.sumw == pytest.approx(noise_mat.sumw, abs=1e-5)


def test_run_local_em_exact_big_jit_bucket_matches_legacy(monkeypatch, rng):
    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=551)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(5, seed=553)
    translations = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    rotation_ids = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
        np.array([0, 2, 4], dtype=np.int32),
    ]
    rotation_counts = np.asarray([ids.size for ids in rotation_ids], dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    rotation_ids_flat = np.concatenate(rotation_ids).astype(np.int32)
    local_layout = LocalHypothesisLayout(
        n_global_rotations=all_rotations.shape[0],
        n_pixels=6,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=np.asarray(all_rotations[rotation_ids_flat], dtype=np.float32),
        rotation_log_priors_flat=np.linspace(0.0, -0.7, rotation_ids_flat.size, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=translations,
        translation_log_priors=np.array(
            [[0.0, -0.5], [-0.2, 0.1], [0.3, -0.4]],
            dtype=np.float32,
        ),
    )
    common_kwargs = dict(
        image_batch_size=3,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=True,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        image_pre_shifts=np.array([[0.25, -0.5], [-0.75, 0.5], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
    )

    monkeypatch.setenv("RECOVAR_RELION_EXACT_LOCAL_BIG_JIT", "1")
    big = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )
    monkeypatch.setenv("RECOVAR_RELION_EXACT_LOCAL_BIG_JIT", "0")
    legacy = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    Ft_y_big, Ft_ctf_big, hard_big, stats_big, noise_big, profile_big = big
    Ft_y_legacy, Ft_ctf_legacy, hard_legacy, stats_legacy, noise_legacy, profile_legacy = legacy
    assert bool(profile_big["big_jit_enabled"]) is True
    assert int(profile_big["big_jit_bucket_count"]) == 1
    assert bool(profile_legacy["big_jit_enabled"]) is False
    np.testing.assert_array_equal(hard_big, hard_legacy)
    np.testing.assert_allclose(np.asarray(Ft_y_big), np.asarray(Ft_y_legacy), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_big), np.asarray(Ft_ctf_legacy), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(stats_big.log_evidence_per_image),
        np.asarray(stats_legacy.log_evidence_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_big.max_posterior_per_image),
        np.asarray(stats_legacy.max_posterior_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_big.rotation_posterior_sums),
        np.asarray(stats_legacy.rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_big.wsum_sigma2_noise),
        np.asarray(noise_legacy.wsum_sigma2_noise),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_big.sumw == pytest.approx(noise_legacy.sumw, abs=1e-5)


def test_compute_reconstruction_support_matches_relion_style_threshold():
    probs = jnp.asarray(
        [
            [
                [0.70, 0.20],
                [0.05, 0.05],
            ]
        ],
        dtype=jnp.float32,
    )

    sig_samples, sig_rots, n_sig = compute_reconstruction_support(
        probs,
        adaptive_fraction=0.9,
        max_significants=-1,
    )

    np.testing.assert_array_equal(np.asarray(n_sig), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(sig_samples),
        np.array([[[True, True], [False, False]]]),
    )
    np.testing.assert_array_equal(
        np.asarray(sig_rots),
        np.array([[True, False]]),
    )


def test_tracked_local_engine_todo_ids_are_present():
    repo_root = Path(__file__).resolve().parents[2]
    iteration_loop_path = repo_root / "recovar" / "em" / "dense_single_volume" / "iteration_loop.py"
    em_engine_path = repo_root / "recovar" / "em" / "dense_single_volume" / "em_engine.py"
    docs_path = repo_root / "docs" / "relion_local_engine_refactor.md"

    iteration_text = iteration_loop_path.read_text(encoding="utf-8")
    em_engine_text = em_engine_path.read_text(encoding="utf-8")
    docs_text = docs_path.read_text(encoding="utf-8")

    documented_ids = [
        "RELION_LOCAL_ENGINE/T001",
        "RELION_LOCAL_ENGINE/T002",
        "RELION_LOCAL_ENGINE/T003",
        "RELION_LOCAL_ENGINE/T004",
        "DENSE_ENGINE_BOUNDARY/E001",
        "DENSE_ENGINE_BOUNDARY/E002",
        "DENSE_ENGINE_BOUNDARY/E003",
        "DENSE_ENGINE_BOUNDARY/E004",
        "DENSE_ENGINE_BOUNDARY/E005",
        "DENSE_ENGINE_BOUNDARY/E006",
    ]
    for todo_id in documented_ids:
        assert todo_id in docs_text
    assert "RELION_LOCAL_ENGINE/T001" not in iteration_text
    assert "DENSE_ENGINE_BOUNDARY/E001" in em_engine_text
    assert "DENSE_ENGINE_BOUNDARY/E002" in em_engine_text
    assert "DENSE_ENGINE_BOUNDARY/E004" in em_engine_text
    assert "DENSE_ENGINE_BOUNDARY/E005" in em_engine_text
    assert "DENSE_ENGINE_BOUNDARY/E003" not in em_engine_text
    assert "DENSE_ENGINE_BOUNDARY/E006" not in em_engine_text


def test_local_engine_selector_is_removed():
    assert "local_engine" not in inspect.signature(refine_single_volume).parameters
    assert "local_engine" not in inspect.signature(iteration_loop_module._run_local_search_iteration).parameters


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
    """Minimal mock of CryoEMDataset for unit testing."""

    def __init__(self, n_images, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.padding = 0
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
        self.image_mask = np.asarray(_unit_image_mask(np.float32), dtype=np.float32)
        self.premultiplied_ctf = False

        self._images = rng.standard_normal((n_images, *IMAGE_SHAPE)).astype(np.float32)

        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)

        class _Backend:
            image_mask = np.asarray(_unit_image_mask(np.float32), dtype=np.float32)
            image_mask_mode = "multiply"

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)
            backend = _Backend()

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
                self.rotation_matrices[idx],
                self.translations[idx],
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def update_poses(self, rots, trans):
        self.rotation_matrices = np.asarray(rots)
        self.translations = np.asarray(trans)

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)

    def original_image_indices_from_local(self, indices):
        return np.asarray(indices, dtype=np.int64)


class RawRealImageDataset:
    """Minimal raw real-space dataset for native half-preprocess tests."""

    def __init__(self, n_images, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.padding = 0
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = n_images
        self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = np.float32
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)
        self.premultiplied_ctf = False
        self._images = rng.standard_normal((n_images, *IMAGE_SHAPE)).astype(np.float32)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)

        class _Backend:
            image_mask = None
            image_mask_mode = "multiply"

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)
            backend = _Backend()

        self.image_source = _ImageSource()

    @property
    def image_mask(self):
        return None

    @property
    def data_multiplier(self):
        return 1.0

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        _ = by_image, kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                self.rotation_matrices[idx],
                self.translations[idx],
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def update_poses(self, rots, trans):
        self.rotation_matrices = np.asarray(rots)
        self.translations = np.asarray(trans)

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)

    def original_image_indices_from_local(self, indices):
        return np.asarray(indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def half_datasets(rng):
    ds0 = MockDataset(N_IMAGES // 2, rng)
    ds1 = MockDataset(N_IMAGES // 2, rng)
    return [ds0, ds1]


@pytest.fixture
def init_volume():
    return _hermitian_volume(VOLUME_SHAPE, seed=42)


@pytest.fixture
def rotations():
    return _make_rotations(N_ROTATIONS, seed=12)


@pytest.fixture
def translations():
    return jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)


def test_sparse_pass2_local_search_matches_per_image_reference(rng, init_volume, translations):
    """Exact-local sparse pass 2 preserves the legacy per-image pass-2 contract."""

    dataset = MockDataset(2, rng)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    significant_samples = [
        np.array([0, 4], dtype=np.int32),
        np.array([2, 3, 5], dtype=np.int32),
    ]
    common_kwargs = dict(
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        translation_step=1.0,
        return_stats=True,
        accumulate_noise=True,
        half_spectrum_scoring=True,
        use_float64_scoring=True,
    )

    reference = _compute_pass2_stats_sparse_perimage_reference(
        dataset,
        init_volume,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        **common_kwargs,
    )
    exact_local = iteration_loop_module._run_sparse_pass2_local_search_iteration(
        dataset,
        init_volume,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        common_kwargs["nside_level"],
        common_kwargs["disc_type"],
        oversampling_order=common_kwargs["oversampling_order"],
        current_size=common_kwargs["current_size"],
        translation_step=common_kwargs["translation_step"],
        return_stats=common_kwargs["return_stats"],
        accumulate_noise=common_kwargs["accumulate_noise"],
        half_spectrum_scoring=common_kwargs["half_spectrum_scoring"],
        use_float64_scoring=common_kwargs["use_float64_scoring"],
    )

    np.testing.assert_allclose(np.asarray(exact_local[0]), np.asarray(reference[0]), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(exact_local[1]), np.asarray(reference[1]), atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(np.asarray(exact_local[2]), np.asarray(reference[2]))
    np.testing.assert_allclose(np.asarray(exact_local[3]), np.asarray(reference[3]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(exact_local[4]), np.asarray(reference[4]), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(exact_local[5]), np.asarray(reference[5]))
    np.testing.assert_allclose(
        np.asarray(exact_local[6].log_evidence_per_image),
        np.asarray(reference[6].log_evidence_per_image),
        atol=5e-4,
        rtol=5e-4,
    )
    np.testing.assert_allclose(
        np.asarray(exact_local[6].max_posterior_per_image),
        np.asarray(reference[6].max_posterior_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(exact_local[6].rotation_posterior_sums),
        np.asarray(reference[6].rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(exact_local[7].wsum_sigma2_noise),
        np.asarray(reference[7].wsum_sigma2_noise),
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(exact_local[7].wsum_img_power),
        np.asarray(reference[7].wsum_img_power),
        atol=1e-5,
        rtol=1e-5,
    )
    assert exact_local[7].sumw == pytest.approx(reference[7].sumw)


# ===========================================================================
# Test 1: RELION mode smoke test -- runs without error
# ===========================================================================


class TestRelionModeSmokeTest:
    """Call refine_single_volume with mode='relion' and verify it runs."""

    def test_relion_bootstrap_current_size_matches_benchmark_case(self):
        """128px, 4.25A/px, ini_high=30A should bootstrap from 36 -> 56."""
        assert _bootstrap_current_size_relion(36, 128) == 56

    def test_relion_bootstrap_current_size_from_ini_high_matches_benchmark_case(self):
        assert bootstrap_current_size_from_ini_high_relion(128, 4.25, 30.0) == 56

    def test_align_fourier_volume_sign_to_reference_flips_negative_overlap(self):
        ref = np.array([1.0 + 0.0j, -2.0 + 0.0j], dtype=np.complex64)
        vol = -ref
        aligned, flipped = _align_fourier_volume_sign_to_reference(vol, ref, (2, 1, 1))
        assert flipped is True
        np.testing.assert_allclose(aligned, ref)

    def test_compute_coarse_image_size_uses_particle_diameter(self):
        """RELION coarse_size should depend on particle diameter, not box size."""
        coarse_from_particle = compute_coarse_image_size(
            14.7,
            4.25,
            128,
            particle_diameter=200.0,
        )
        coarse_from_box = compute_coarse_image_size(
            14.7,
            4.25,
            128,
        )
        assert coarse_from_particle == 52
        assert coarse_from_box == 20
        assert coarse_from_particle > coarse_from_box

    def test_clamp_relion_coarse_image_size_caps_at_current_size(self):
        """RELION clamps coarse_size to current_size, not current_size/2."""
        coarse_size = compute_coarse_image_size(
            7.5,
            4.25,
            128,
            particle_diameter=200.0,
        )
        assert coarse_size == 100
        assert clamp_relion_coarse_image_size(coarse_size, current_size=60, ori_size=128) == 60

    def test_make_relion_direction_log_prior_matches_canonical_grid_indices(self):
        order = 2
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.linspace(1.0, float(n_pixels), n_pixels, dtype=np.float32)
        direction_prior /= direction_prior.sum()
        rotations = np.asarray(get_relion_rotation_grid(order), dtype=np.float32)
        view_dirs = rotations[:, 2, :].astype(np.float64)
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
        expected_pixels = hp.vec2pix(
            2**order,
            view_dirs[:, 0],
            view_dirs[:, 1],
            view_dirs[:, 2],
        )

        got = make_relion_direction_log_prior(
            direction_prior,
            order,
            rotations=rotations,
        )
        expected = np.log(direction_prior[expected_pixels]).astype(np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_make_relion_direction_log_prior_tracks_perturbed_view_directions(self):
        order = 3
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.linspace(1.0, float(n_pixels), n_pixels, dtype=np.float32)
        direction_prior /= direction_prior.sum()

        perturbed_rotations = apply_relion_rotation_perturbation(
            np.asarray(get_relion_rotation_grid(order), dtype=np.float32),
            random_perturbation=0.3,
            angular_sampling_deg=360.0 / (6 * 2**order),
        ).astype(np.float32)
        view_dirs = perturbed_rotations[:, 2, :].astype(np.float64)
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
        expected_pixels = hp.vec2pix(
            2**order,
            view_dirs[:, 0],
            view_dirs[:, 1],
            view_dirs[:, 2],
        )

        got = make_relion_direction_log_prior(
            direction_prior,
            order,
            rotations=perturbed_rotations,
        )
        expected = np.log(direction_prior[expected_pixels]).astype(np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_make_relion_direction_log_prior_default_keeps_sample_index_prior(self):
        order = 3
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.linspace(1.0, float(n_pixels), n_pixels, dtype=np.float32)
        direction_prior /= direction_prior.sum()

        got = make_relion_direction_log_prior(direction_prior, order)
        expected = np.log(np.repeat(direction_prior[None, :], rotation_grid_n_in_planes(order), axis=0).reshape(-1))
        np.testing.assert_allclose(got, expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    def test_make_relion_direction_log_prior_preserves_zero_prior_as_hard_mask(self):
        order = 2
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.ones(n_pixels, dtype=np.float32)
        direction_prior[3] = 0.0
        direction_prior /= direction_prior.sum()

        got = make_relion_direction_log_prior(direction_prior, order)
        zero_direction_rows = np.arange(n_rot, dtype=np.int64) % n_pixels == 3

        assert np.isneginf(got[zero_direction_rows]).all()
        assert np.isfinite(got[~zero_direction_rows]).all()

    def test_normalize_direction_prior_per_half_preserves_relion_half_models(self):
        half1 = np.array([0.7, 0.3], dtype=np.float32)
        half2 = np.array([0.2, 0.8], dtype=np.float32)

        got = normalize_direction_prior_per_half([half1, half2])

        np.testing.assert_allclose(got[0], half1)
        np.testing.assert_allclose(got[1], half2)

    def test_normalize_direction_prior_per_half_keeps_legacy_shared_prior(self):
        shared = np.array([0.7, 0.3], dtype=np.float32)

        got = normalize_direction_prior_per_half(shared)

        np.testing.assert_allclose(got[0], shared)
        np.testing.assert_allclose(got[1], shared)
        assert got[0] is not got[1]

    def test_normalize_noise_variance_per_half_keeps_legacy_shared_noise(self):
        shared = jnp.arange(IMAGE_SIZE, dtype=jnp.float32) + 1.0

        got = _normalize_noise_variance_per_half(shared, n_halves=2)

        assert len(got) == 2
        np.testing.assert_allclose(np.asarray(got[0]), np.asarray(shared))
        np.testing.assert_allclose(np.asarray(got[1]), np.asarray(shared))

    def test_normalize_noise_variance_per_half_preserves_relion_half_models(self):
        half1 = np.arange(IMAGE_SIZE, dtype=np.float32) + 1.0
        half2 = half1 * 2.0

        got = _normalize_noise_variance_per_half(np.stack([half1, half2]), n_halves=2)

        np.testing.assert_allclose(np.asarray(got[0]), half1)
        np.testing.assert_allclose(np.asarray(got[1]), half2)

    def test_relion_mode_runs_2_iterations(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """mode='relion' completes 2 iterations on a tiny dataset."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        # Basic return dict structure
        assert "mean" in result
        assert "means" in result
        assert "fsc" in result
        assert "hard_assignments" in result
        assert "current_sizes" in result
        assert "fsc_history" in result
        assert "pixel_resolutions" in result
        assert "wall_times" in result

        # RELION-specific keys
        assert "convergence_state" in result
        assert "data_vs_prior_trajectory" in result
        assert "healpix_order_trajectory" in result
        assert "ave_Pmax_trajectory" in result

    def test_relion_mode_does_not_finalize_after_max_iter_exhaustion(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """RELION does not run final all-data iteration just because max_iter ended."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=4,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
        )

        assert result["convergence_state"].has_converged is False
        assert len(result["wall_times"]) == 1
        assert len(result["current_sizes"]) == 1

    def test_relion_final_iteration_scores_half_maps_after_convergence(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """The final joined reconstruction still scores each half against its own map."""
        original_update = iteration_loop_module.update_refinement_state
        original_run_em = iteration_loop_module.run_em
        run_em_mean_ids = []

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def spy_run_em(dataset, mean, *args, **kwargs):
            _ = dataset
            run_em_mean_ids.append(id(mean))
            return original_run_em(dataset, mean, *args, **kwargs)

        monkeypatch.setattr(
            iteration_loop_module,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        monkeypatch.setattr(iteration_loop_module, "run_em", spy_run_em)

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=4,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
        )

        assert result["convergence_state"].has_converged is True
        assert len(result["wall_times"]) == 2
        assert len(run_em_mean_ids) == 4
        assert run_em_mean_ids[-2:] == [id(result["means"][0]), id(result["means"][1])]

    def test_relion_mode_finite_outputs(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """RELION mode produces finite volumes and valid assignments."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        # Final mean should be finite
        assert np.all(np.isfinite(np.array(result["mean"]))), "Mean not finite"
        # FSC should be computed
        assert result["fsc"] is not None
        # Hard assignments valid
        for k in range(2):
            ha = result["hard_assignments"][k]
            assert ha is not None
            assert np.all(ha >= 0)

    def test_relion_mode_uses_engine_pmax(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """ave_Pmax should aggregate the engine's posterior maxima across both half-sets."""
        init_noise = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
        init_tau = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0

        expected_per_half = []
        for dataset in half_datasets:
            _, _, _, _, stats = run_em(
                dataset,
                init_volume,
                init_tau,
                init_noise,
                rotations,
                translations,
                "linear_interp",
                image_batch_size=N_IMAGES,
                rotation_block_size=N_ROTATIONS,
                return_stats=True,
            )
            expected_per_half.append(np.asarray(stats.max_posterior_per_image))
        expected_ave_pmax = float(np.mean(np.concatenate(expected_per_half)))

        result = refine_single_volume(
            half_datasets,
            init_volume,
            init_noise,
            init_tau,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            adaptive_oversampling=0,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        assert result["ave_Pmax_trajectory"] == pytest.approx(
            [expected_ave_pmax],
            abs=1e-6,
        )
        assert result["convergence_state"].ave_Pmax == pytest.approx(
            expected_ave_pmax,
            abs=1e-6,
        )

    def test_relion_mode_forwards_particle_diameter_to_coarse_size(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Adaptive RELION mode should pass the explicit particle diameter through."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        recorded = {"particle_diameter": None}
        original_compute_coarse_image_size = refine_mod.compute_coarse_image_size

        def wrap_compute_coarse_image_size(*args, **kwargs):
            recorded["particle_diameter"] = kwargs.get("particle_diameter")
            return original_compute_coarse_image_size(*args, **kwargs)

        monkeypatch.setattr(
            refine_mod,
            "compute_coarse_image_size",
            wrap_compute_coarse_image_size,
        )

        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            _make_rotations(20, seed=123),
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=20,
            init_current_size=16,
            adaptive_oversampling=1,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
            particle_diameter_ang=200.0,
        )

        assert recorded["particle_diameter"] == pytest.approx(200.0)

    def test_adaptive_pass1_receives_relion_corrections_and_preshifts(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Adaptive pass 1 must receive RELION image corrections and pre-shifts."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        recorded = {
            "image_corrections": [],
            "scale_corrections": [],
            "image_pre_shifts": [],
        }

        def fake_significance(*args, **kwargs):
            recorded["image_corrections"].append(kwargs.get("image_corrections"))
            recorded["scale_corrections"].append(kwargs.get("scale_corrections"))
            recorded["image_pre_shifts"].append(kwargs.get("image_pre_shifts"))
            n_rot = args[3].shape[0]
            n_img = args[0].n_units
            return (
                np.ones(n_rot, dtype=bool),
                np.full(n_img, n_rot * translations.shape[0], dtype=np.int32),
                np.zeros(n_img, dtype=np.int32),
                [None] * n_img,
            )

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)
        monkeypatch.setattr(refine_mod, "should_skip_adaptive_pass2", lambda *a, **k: (True, 1.0))

        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            _make_rotations(20, seed=123),
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=20,
            init_current_size=16,
            adaptive_oversampling=1,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
            init_image_corrections=[
                np.array([0.8, 1.2], dtype=np.float32),
                np.array([1.1, 0.9], dtype=np.float32),
            ],
            init_scale_corrections=[
                np.array([1.05, 0.95], dtype=np.float32),
                np.array([0.9, 1.1], dtype=np.float32),
            ],
            init_previous_best_translations=[
                np.array([[0.5, -0.25], [0.0, 0.75]], dtype=np.float32),
                np.array([[-0.4, 0.3], [0.6, -0.2]], dtype=np.float32),
            ],
        )

        assert len(recorded["image_corrections"]) == 2
        assert len(recorded["scale_corrections"]) == 2
        assert len(recorded["image_pre_shifts"]) == 2
        np.testing.assert_allclose(
            recorded["image_pre_shifts"][0],
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            recorded["image_pre_shifts"][1],
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_relion_translation_log_prior_matches_source_pdf_offset(self, translations):
        log_prior = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
        )
        translations_angstrom = np.asarray(translations) * 4.25
        expected = -0.5 * np.sum(translations_angstrom**2, axis=1) / (10.0**2)
        expected *= 4.25**2
        np.testing.assert_allclose(log_prior, expected, rtol=1e-6, atol=1e-6)
        assert int(np.argmax(log_prior)) == 0

    def test_relion_translation_log_prior_uses_offset_range_when_active(self, translations):
        log_prior_sigma = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
        )
        log_prior_range = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            offset_range_pixels=3.0,
        )
        # RELION uses sigma = offset_range / 3 while a finite search range is active.
        assert log_prior_range[0] == pytest.approx(0.0)
        assert log_prior_range[1] < log_prior_sigma[1]

    def test_relion_translation_search_base_uses_integer_prescoring_shift(self):
        prev = np.array([[0.5, -0.25], [-0.4, 0.3]], dtype=np.float32)
        expected = np.rint(prev).astype(np.float32)
        np.testing.assert_allclose(relion_translation_search_base(prev), expected, rtol=1e-6, atol=1e-6)
        assert relion_translation_search_base(np.array([], dtype=np.float32)).shape == (0, 2)

    def test_relion_integer_pre_shift_uses_zero_fill_real_space_convention(self):
        image = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        shifted = apply_relion_integer_pre_shifts(image, np.array([[1, -1]], dtype=np.int32))
        expected = np.array(
            [
                [
                    [0.0, 3.0, 4.0],
                    [0.0, 6.0, 7.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(shifted, expected)

    def test_integer_pre_shifts_only_selects_integral_offsets(self):
        shifts = np.array([[1.0, -1.0], [0.5, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            integer_pre_shifts_or_none(shifts, np.array([0], dtype=np.int32)),
            np.array([[1, -1]], dtype=np.int32),
        )
        assert integer_pre_shifts_or_none(shifts, np.array([1], dtype=np.int32)) is None

    def test_relion_translation_prior_center_matches_accelerated_pdf_offset_units(self):
        prev = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
        expected = np.array([[0.0, 1.0 / 4.25], [-1.0 / 4.25, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(
            relion_translation_prior_center(prev, voxel_size=4.25),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_direction_prior_round_trip_to_rotation_log_prior(self):
        healpix_order = 1
        n_dirs = 48
        direction_prior = np.zeros(n_dirs, dtype=np.float32)
        direction_prior[:3] = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        direction_prior[3:] = np.finfo(np.float32).tiny
        direction_prior /= direction_prior.sum()

        rotation_log_prior = make_relion_direction_log_prior(direction_prior, healpix_order)
        collapsed = collapse_rotation_posterior_to_direction_prior(
            np.exp(rotation_log_prior),
            healpix_order,
        )

        np.testing.assert_allclose(collapsed, direction_prior, rtol=1e-6, atol=1e-8)

    def test_engine_translation_log_prior_changes_pmax(self, half_datasets, init_volume):
        rotations = _make_rotations(1, seed=17)
        translations = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=jnp.float32,
        )
        half_datasets[0]._images = np.zeros_like(half_datasets[0]._images)
        uniform_stats = run_em(
            half_datasets[0],
            jnp.zeros_like(init_volume),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=1,
            return_stats=True,
        )[4]
        biased_prior = np.log(np.array([100.0, 1.0, 1.0], dtype=np.float32))
        biased_stats = run_em(
            half_datasets[0],
            jnp.zeros_like(init_volume),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            image_batch_size=N_IMAGES,
            rotation_block_size=1,
            translation_log_prior=biased_prior,
            return_stats=True,
        )[4]
        assert np.allclose(
            np.asarray(uniform_stats.max_posterior_per_image),
            1.0 / 3.0,
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(biased_stats.max_posterior_per_image),
            100.0 / 102.0,
            atol=1e-6,
        )

    def test_significance_batched_supports_padded_rotation_log_prior(
        self,
        half_datasets,
        init_volume,
        translations,
    ):
        """Rotation priors should work even when the last block is padded."""
        rotations = _make_rotations(5, seed=19)
        sig_rot_any, n_sig, ha = _compute_significance_batched(
            half_datasets[0],
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            0.999,
            -1,
            image_batch_size=N_IMAGES,
            rotation_block_size=rotations.shape[0] + 1,
            current_size=None,
            rotation_log_prior=np.zeros(rotations.shape[0], dtype=np.float32),
        )

        assert sig_rot_any.shape == (rotations.shape[0],)
        assert n_sig.shape == (half_datasets[0].n_units,)
        assert ha.shape == (half_datasets[0].n_units,)

    def test_significance_batched_matches_run_em_with_pre_shifts_scales_and_projection_padding(
        self,
        half_datasets,
        init_volume,
    ):
        """Adaptive pass 1 must score with the same corrections as the dense engine."""
        dataset = half_datasets[0]
        rotations = _make_rotations(5, seed=23)
        translations = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=jnp.float32,
        )
        init_noise = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
        init_tau = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0
        image_corrections = np.array([0.8, 1.2], dtype=np.float32)
        scale_corrections = np.array([1.1, 0.9], dtype=np.float32)
        image_pre_shifts = np.array([[1.5, -0.5], [-1.0, 1.25]], dtype=np.float32)
        current_size = 6

        _, expected_ha, _, _ = run_em(
            dataset,
            init_volume,
            init_tau,
            init_noise,
            rotations,
            translations,
            "linear_interp",
            image_batch_size=dataset.n_units,
            rotation_block_size=rotations.shape[0],
            current_size=current_size,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            score_with_masked_images=True,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
        )

        _, _, actual_ha = _compute_significance_batched(
            dataset,
            init_volume,
            init_noise,
            rotations,
            translations,
            "linear_interp",
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=rotations.shape[0],
            current_size=current_size,
            score_with_masked_images=True,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
        )

        np.testing.assert_array_equal(np.asarray(actual_ha), np.asarray(expected_ha))

    def test_relion_mode_convergence_state(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """Convergence state is a RefinementState with correct fields."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        state = result["convergence_state"]
        assert isinstance(state, RefinementState)
        # After 2 iterations, iteration counter should be at least 1
        assert state.iteration >= 1
        # ave_Pmax should be in [0, 1]
        assert 0.0 <= state.ave_Pmax <= 1.0

    def test_relion_mode_uses_tau2_from_weights_for_prior(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """RELION mode should compute tau2 from Ft_ctf weights + FSC (RELION order)."""
        from recovar.reconstruction import regularization

        called = {"tau2": 0}

        original_tau2 = regularization.compute_relion_tau2_from_weights

        def wrap_tau2(*args, **kwargs):
            called["tau2"] += 1
            return original_tau2(*args, **kwargs)

        monkeypatch.setattr(regularization, "compute_relion_tau2_from_weights", wrap_tau2)

        grid_size = int(np.sqrt(IMAGE_SIZE))
        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            adaptive_oversampling=0,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
            init_fsc=np.ones(grid_size // 2),
        )

        assert called["tau2"] >= 1

    def test_relion_mode_current_size_no_longer_uses_weight_based_data_vs_prior(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """RELION mode should derive current_size from FSC-derived SSNR logic."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        def fail_old_dvp(*args, **kwargs):
            raise AssertionError("RELION mode should not call compute_data_vs_prior")

        monkeypatch.setattr(refine_mod, "compute_data_vs_prior", fail_old_dvp, raising=False)

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            adaptive_oversampling=0,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        assert len(result["current_sizes"]) == 2

    def test_relion_mode_trajectories_populated(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """RELION-specific trajectories have correct lengths."""
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=2,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            mode="relion",
            init_healpix_order=2,
            max_healpix_order=3,
        )

        n_iters = len(result["current_sizes"])
        assert n_iters <= 2
        assert len(result["healpix_order_trajectory"]) == n_iters
        assert len(result["ave_Pmax_trajectory"]) == n_iters
        # data_vs_prior is populated starting from iteration 1
        assert len(result["data_vs_prior_trajectory"]) <= n_iters

    def test_should_skip_adaptive_pass2_threshold(self):
        """Adaptive pass 2 should be skipped when mean significant fraction >= 0.5."""
        skip, frac = should_skip_adaptive_pass2(
            np.array([60, 60], dtype=np.int32),
            n_rotations=20,
            n_translations=3,
        )
        assert skip is True
        assert frac == pytest.approx(1.0)

        skip, frac = should_skip_adaptive_pass2(
            np.array([12, 18], dtype=np.int32),
            n_rotations=20,
            n_translations=3,
        )
        assert skip is False
        assert frac == pytest.approx(0.25)

        skip, frac = should_skip_adaptive_pass2(
            np.array([60, 60], dtype=np.int32),
            n_rotations=20,
            n_translations=3,
            threshold=-1.0,
        )
        assert skip is False
        assert frac == pytest.approx(0.0)

    def test_relion_mode_skips_pass2_when_significance_fraction_is_high(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """When pass-1 significance is dense, RELION mode should bypass sparse pass 2."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        rotations_many = _make_rotations(20, seed=123)
        captured = {"n_rot": None}

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_rot = np.asarray(args[3]).shape[0]
            n_trans = len(np.asarray(translations))
            captured["n_rot"] = n_rot
            return (
                np.ones(n_rot, dtype=bool),
                np.full(n_images, n_rot * n_trans, dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [np.array([0], dtype=np.int32) for _ in range(n_images)],
            )

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)
        assert not hasattr(refine_mod, "compute_pass2_stats_sparse")

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=16,
            adaptive_oversampling=1,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        assert result["significant_counts"][0] is not None
        counts = np.asarray(result["significant_counts"][0])
        assert counts.shape == (half_datasets[0].n_units + half_datasets[1].n_units,)
        np.testing.assert_array_equal(
            counts,
            np.full_like(counts, captured["n_rot"] * len(np.asarray(translations))),
        )

    def test_relion_mode_uses_dense_exact_pass2_when_all_samples_significant(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Exact mode should batch dense pass 2 instead of calling sparse per image."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        rotations_many = _make_rotations(20, seed=456)
        dense_calls = {"count": 0}

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_rot = np.asarray(args[3]).shape[0]
            n_trans = len(np.asarray(translations))
            return (
                np.ones(n_rot, dtype=bool),
                np.full(n_images, n_rot * n_trans, dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [None] * n_images,
            )

        def fake_dense_pass2(*args, **kwargs):
            from recovar.em.dense_single_volume.helpers.types import NoiseStats

            dataset = args[0]
            n_images = dataset.n_units
            rotations_arg = np.asarray(args[4], dtype=np.float32)
            dense_calls["count"] += 1
            n_shells = dataset.image_shape[0] // 2 + 1
            rpf = kwargs.get("reconstruction_padding_factor", 1)
            recon_vol_size = VOLUME_SIZE * rpf**3
            result = (
                jnp.zeros(recon_vol_size, dtype=jnp.complex64),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
                np.zeros(n_images, dtype=np.int32),
                rotations_arg,
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(rotations_arg.shape[0], dtype=jnp.float32),
                ),
            )
            if kwargs.get("accumulate_noise", False):
                result = result + (
                    NoiseStats(
                        wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                        wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                        wsum_sigma2_offset=0.0,
                        sumw=float(n_images),
                    ),
                )
            return result

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)
        monkeypatch.setattr(refine_mod, "compute_pass2_stats", fake_dense_pass2)
        assert not hasattr(refine_mod, "compute_pass2_stats_sparse")

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=16,
            adaptive_oversampling=1,
            adaptive_pass2_skip_threshold=-1.0,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        assert dense_calls["count"] == 2
        assert result["significant_counts"][0] is not None

    def test_relion_mode_routes_sparse_adaptive_pass2_through_local_search_iteration(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Sparse adaptive pass 2 should use exact local-search machinery, not sparse bucketed fallback."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        rotations_many = _make_rotations(20, seed=789)
        captured = {"local_pass2_calls": 0}

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_rot = np.asarray(args[3]).shape[0]
            _ = kwargs
            return (
                np.ones(n_rot, dtype=bool),
                np.ones(n_images, dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [np.array([0], dtype=np.int32) for _ in range(n_images)],
            )

        def fake_local_pass2(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_shells = dataset.image_shape[0] // 2 + 1
            recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
            coarse_order = int(args[6])
            captured["local_pass2_calls"] += 1
            captured["adaptive_fraction"] = kwargs["adaptive_fraction"]
            captured["oversampling_order"] = kwargs["oversampling_order"]
            return (
                jnp.zeros(recon_vol_size, dtype=jnp.complex64),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
                np.zeros(n_images, dtype=np.int32),
                np.tile(np.eye(3, dtype=np.float32)[None, :, :], (n_images, 1, 1)),
                np.zeros((n_images, 2), dtype=np.float32),
                np.zeros(n_images, dtype=np.int32),
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(rotation_grid_size(coarse_order), dtype=jnp.float32),
                ),
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images),
                ),
            )

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)
        monkeypatch.setattr(refine_mod, "_run_sparse_pass2_local_search_iteration", fake_local_pass2)
        assert not hasattr(refine_mod, "compute_pass2_stats_sparse")

        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=16,
            adaptive_oversampling=1,
            adaptive_fraction=0.97,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        assert captured["local_pass2_calls"] == 2
        assert captured["adaptive_fraction"] == pytest.approx(0.97)
        assert captured["oversampling_order"] == 1

    def test_relion_mode_passes_adaptive_pruning_parameters(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """RELION mode should forward adaptive pruning kwargs to pass 1."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        rotations_many = _make_rotations(20, seed=321)
        captured = {}

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_rot = np.asarray(args[3]).shape[0]
            n_trans = len(np.asarray(translations))
            captured["adaptive_fraction"] = kwargs["adaptive_fraction"]
            captured["max_significants"] = kwargs["max_significants"]
            return (
                np.ones(n_rot, dtype=bool),
                np.full(n_images, n_rot * n_trans, dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [np.array([0], dtype=np.int32) for _ in range(n_images)],
            )

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)

        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=16,
            adaptive_oversampling=1,
            adaptive_fraction=0.97,
            max_significants=123,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        assert captured["adaptive_fraction"] == pytest.approx(0.97)
        assert captured["max_significants"] == 123

    def test_relion_mode_uses_global_significant_support_for_os0_replay(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """os=0 global search should still prune to RELION's significant subset."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        rotations_many = _make_rotations(20, seed=654)
        prev_h1 = np.array([[0.6, -1.2], [0.2, 0.9]], dtype=np.float32)
        prev_h2 = np.array([[-0.7, 1.6], [1.4, -0.1]], dtype=np.float32)
        captured = {
            "sig_calls": 0,
            "local_pass2_calls": 0,
            "run_em_calls": 0,
            "prior_centers": [],
            "normalization_log_z": [],
        }

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_rot = np.asarray(args[3]).shape[0]
            n_trans = len(np.asarray(translations))
            captured["sig_calls"] += 1
            full_stats = {
                "normalization_log_z": np.full(n_images, 7.5, dtype=np.float64),
                "log_evidence_per_image": np.full(n_images, 1.25, dtype=np.float32),
                "best_log_score_per_image": np.full(n_images, 2.5, dtype=np.float32),
                "max_posterior_per_image": np.full(n_images, 0.75, dtype=np.float32),
            }
            return (
                np.ones(n_rot, dtype=bool),
                np.full(n_images, min(3, n_rot * n_trans), dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [np.array([0, 1, 2], dtype=np.int32) for _ in range(n_images)],
                full_stats,
            )

        def fake_run_em(*args, **kwargs):
            experiment_dataset = args[0]
            rotations = args[4]
            n_images = experiment_dataset.n_units
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
            captured["run_em_calls"] += 1
            return (
                None,
                np.zeros(n_images, dtype=np.int32),
                jnp.zeros(recon_vol_size, dtype=jnp.complex64),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
                ),
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images),
                ),
            )

        def fake_local_pass2(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            n_shells = dataset.image_shape[0] // 2 + 1
            recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
            coarse_order = int(args[6])
            captured["local_pass2_calls"] += 1
            captured["prior_centers"].append(np.asarray(kwargs["translation_prior_centers"], dtype=np.float32).copy())
            captured["normalization_log_z"].append(np.asarray(kwargs["normalization_log_z"], dtype=np.float64).copy())
            assert kwargs["oversampling_order"] == 0
            return (
                jnp.zeros(recon_vol_size, dtype=jnp.complex64),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
                np.zeros(n_images, dtype=np.int32),
                np.tile(np.eye(3, dtype=np.float32)[None, :, :], (n_images, 1, 1)),
                np.zeros((n_images, 2), dtype=np.float32),
                np.zeros(n_images, dtype=np.int64),
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(rotation_grid_size(coarse_order), dtype=jnp.float32),
                ),
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images),
                ),
            )

        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)
        monkeypatch.setattr(refine_mod, "_run_sparse_pass2_local_search_iteration", fake_local_pass2)
        assert not hasattr(refine_mod, "compute_pass2_stats_sparse")
        monkeypatch.setattr(refine_mod, "run_em", fake_run_em)

        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=16,
            adaptive_oversampling=0,
            adaptive_fraction=0.97,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
            init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
        )

        assert captured["sig_calls"] == 2
        assert captured["local_pass2_calls"] == 2
        assert captured["run_em_calls"] == 0
        np.testing.assert_allclose(
            captured["prior_centers"][0],
            relion_translation_prior_center(prev_h1, half_datasets[0].voxel_size),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(captured["normalization_log_z"][0], np.full(prev_h1.shape[0], 7.5))
        np.testing.assert_array_equal(captured["normalization_log_z"][1], np.full(prev_h2.shape[0], 7.5))
        np.testing.assert_allclose(
            captured["prior_centers"][1],
            relion_translation_prior_center(prev_h2, half_datasets[1].voxel_size),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_relion_mode_updates_sigma_offset_from_posterior_noise_stats(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Posterior-weighted offset variance should drive sigma_offset in RELION mode."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        for ds in half_datasets:
            ds.voxel_size = 8.5
        rotations_many = _make_rotations(20, seed=888)
        noise_offset_wsums = [12.0, 20.0]
        call_idx = {"value": 0}

        def fake_run_em(
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            disc_type,
            **kwargs,
        ):
            _ = (mean, mean_variance, noise_variance, disc_type, kwargs)
            idx = call_idx["value"]
            call_idx["value"] += 1
            offset_wsum = noise_offset_wsums[min(idx, len(noise_offset_wsums) - 1)]
            n_images = experiment_dataset.n_units
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
            return (
                None,
                np.zeros(n_images, dtype=np.int32),
                jnp.zeros(recon_vol_size, dtype=jnp.complex64),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
                ),
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=offset_wsum,
                    sumw=float(n_images),
                ),
            )

        monkeypatch.setattr(refine_mod, "run_em", fake_run_em)

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=16,
            adaptive_oversampling=0,
            adaptive_fraction=1.0,
            nside_level=1,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        expected_sigma = np.sqrt((noise_offset_wsums[0] + noise_offset_wsums[1]) / (2.0 * N_IMAGES))
        assert result["sigma_offset_trajectory"][0] == pytest.approx(expected_sigma)

    def test_relion_mode_passes_per_half_noise_to_engine(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """RELION mode must score each half-set with its own sigma2_noise."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        rotations_many = _make_rotations(20, seed=777)
        half1_noise = np.arange(IMAGE_SIZE, dtype=np.float32) + 1.0
        half2_noise = half1_noise * 3.0
        captured_noise = []

        def fake_run_em(
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            disc_type,
            **kwargs,
        ):
            _ = (mean, mean_variance, translations, disc_type, kwargs)
            captured_noise.append(np.asarray(noise_variance, dtype=np.float32))
            n_images = experiment_dataset.n_units
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
            return (
                None,
                np.zeros(n_images, dtype=np.int32),
                jnp.zeros(recon_vol_size, dtype=jnp.complex64),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
                ),
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images),
                ),
            )

        monkeypatch.setattr(refine_mod, "run_em", fake_run_em)

        refine_single_volume(
            half_datasets,
            init_volume,
            np.stack([half1_noise, half2_noise]),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations_many,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(rotations_many),
            init_current_size=8,
            adaptive_oversampling=0,
            adaptive_fraction=1.0,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
            skip_final_iteration=True,
        )

        assert len(captured_noise) == 2
        np.testing.assert_allclose(captured_noise[0], half1_noise)
        np.testing.assert_allclose(captured_noise[1], half2_noise)

    def test_relion_mode_regenerates_initial_coarse_grid_from_healpix_state(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Pass 1 should start from the coarse RELION HEALPix order, not a fine caller grid."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        captured = {"grid_order": None}

        def fake_grid(order):
            captured["grid_order"] = int(order)
            n_rot = 4 if order == 1 else 9
            return np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))

        def fake_significance(*args, **kwargs):
            dataset = args[0]
            n_images = dataset.n_units
            rotations_arg = np.asarray(args[3], dtype=np.float32)
            n_trans = len(np.asarray(translations))
            return (
                np.ones(rotations_arg.shape[0], dtype=bool),
                np.full(n_images, rotations_arg.shape[0] * n_trans, dtype=np.int32),
                np.zeros(n_images, dtype=np.int32),
                [None] * n_images,
            )

        monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_grid)
        monkeypatch.setattr(refine_mod, "_compute_significance_batched", fake_significance)

        fine_rotations = _make_rotations(9, seed=999)
        refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            fine_rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=len(fine_rotations),
            init_current_size=16,
            adaptive_oversampling=1,
            adaptive_pass2_skip_threshold=-1.0,
            nside_level=3,
            mode="relion",
            init_healpix_order=1,
            max_healpix_order=2,
        )

        assert captured["grid_order"] == 1

# ===========================================================================
# Test 2: Legacy mode removed
# ===========================================================================


class TestLegacyModeRemoved:
    """Verify that mode='legacy' is no longer part of the API."""

    def test_legacy_mode_explicit_raises(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """Explicit mode='legacy' is rejected."""
        with pytest.raises(ValueError, match="expected 'relion'"):
            refine_single_volume(
                half_datasets,
                init_volume,
                jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
                jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
                rotations,
                translations,
                disc_type="linear_interp",
                max_iter=1,
                image_batch_size=N_IMAGES,
                rotation_block_size=N_ROTATIONS,
                mode="legacy",
            )

    def test_default_mode_is_relion(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """Calling without mode= uses the RELION path."""
        sentinel = {"convergence_state": object()}
        called = {"enabled_defaults": False, "ran_relion": False}

        def fake_enable_defaults():
            called["enabled_defaults"] = True

        def fake_relion_loop(**kwargs):
            called["ran_relion"] = True
            assert kwargs["experiment_datasets"] is half_datasets
            assert kwargs["relion_current_sizes"] is None
            assert kwargs["init_healpix_order"] == 2
            return sentinel

        monkeypatch.setattr(
            iteration_loop_module,
            "_enable_relion_parity_defaults",
            fake_enable_defaults,
        )
        monkeypatch.setattr(
            iteration_loop_module,
            "_run_relion_iteration_loop",
            fake_relion_loop,
        )

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            init_healpix_order=2,
            max_healpix_order=3,
        )

        assert result is sentinel
        assert called == {"enabled_defaults": True, "ran_relion": True}


# ===========================================================================
# Test 3: Local search oversampling regression
# ===========================================================================


def test_fused_score_normalize_mstep_matches_split_path():
    import jax

    from recovar.em.dense_single_volume.local_backprojection import (
        compute_local_ctf_sums,
        compute_local_weighted_sums,
    )
    from recovar.em.dense_single_volume.local_score_pass import (
        compute_reconstruction_support,
        fused_score_normalize_mstep_abs2_on_demand,
        normalize_local_scores,
        score_local_bucket_abs2_weighted_on_demand,
    )

    rng = np.random.default_rng(0)
    batch_size, n_rot, n_trans, n_score, n_recon = 3, 5, 4, 7, 6
    shifted_score = rng.normal(size=(batch_size, n_trans, n_score)) + 1j * rng.normal(
        size=(batch_size, n_trans, n_score)
    )
    shifted_recon = rng.normal(size=(batch_size, n_trans, n_recon)) + 1j * rng.normal(
        size=(batch_size, n_trans, n_recon)
    )
    proj_weighted = rng.normal(size=(batch_size, n_rot, n_score)) + 1j * rng.normal(
        size=(batch_size, n_rot, n_score)
    )
    ctf_score = np.abs(rng.normal(size=(batch_size, n_score))).astype(np.float32) + 0.1
    ctf_recon = np.abs(rng.normal(size=(batch_size, n_recon))).astype(np.float32) + 0.1
    half_weights = np.linspace(1.0, 2.0, n_score, dtype=np.float32)
    rotation_log_prior = rng.normal(size=(batch_size, n_rot)).astype(np.float32) * 0.01
    translation_log_prior = rng.normal(size=(batch_size, n_trans)).astype(np.float32) * 0.01
    rotation_mask = np.ones((batch_size, n_rot), dtype=bool)
    rotation_mask[0, -1] = False
    sample_mask = np.ones((batch_size, n_rot, n_trans), dtype=bool)
    sample_mask[1, 2, 3] = False

    shifted_score_j = jnp.asarray(shifted_score, dtype=jnp.complex64)
    shifted_recon_j = jnp.asarray(shifted_recon, dtype=jnp.complex64)
    proj_weighted_j = jnp.asarray(proj_weighted, dtype=jnp.complex64)
    ctf_score_j = jnp.asarray(ctf_score, dtype=jnp.float32)
    ctf_recon_j = jnp.asarray(ctf_recon, dtype=jnp.float32)
    half_weights_j = jnp.asarray(half_weights, dtype=jnp.float32)
    rotation_log_prior_j = jnp.asarray(rotation_log_prior)
    translation_log_prior_j = jnp.asarray(translation_log_prior)
    rotation_mask_j = jnp.asarray(rotation_mask)
    sample_mask_j = jnp.asarray(sample_mask)

    scores = score_local_bucket_abs2_weighted_on_demand(
        shifted_score_j,
        ctf_score_j,
        proj_weighted_j,
        half_weights_j,
        rotation_log_prior_j,
        translation_log_prior_j,
        rotation_mask_j,
        sample_mask_j,
    )
    log_z, probs, best_log_score, best_argmax, max_posterior = normalize_local_scores(scores)
    recon_mask, recon_rot_mask, n_sig = compute_reconstruction_support(
        probs,
        adaptive_fraction=0.99,
        max_significants=6,
    )
    recon_probs = jnp.where(recon_mask, probs, 0.0)
    probs_sum_t = jnp.sum(probs, axis=-1)
    recon_probs_sum_t = jnp.sum(recon_probs, axis=-1)
    summed = compute_local_weighted_sums(recon_probs, shifted_recon_j)
    ctf_probs = compute_local_ctf_sums(recon_probs, ctf_recon_j)

    fused = fused_score_normalize_mstep_abs2_on_demand(
        shifted_score_j,
        ctf_score_j,
        proj_weighted_j,
        half_weights_j,
        rotation_log_prior_j,
        translation_log_prior_j,
        rotation_mask_j,
        sample_mask_j,
        shifted_recon_j,
        ctf_recon_j,
        half_spectrum_scoring=False,
        use_float64_normalization=True,
        reconstruct_significant_only=True,
        adaptive_fraction=0.99,
        max_significants=6,
    )
    fused = jax.tree.map(np.asarray, fused)

    np.testing.assert_allclose(fused[0], np.asarray(log_z), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(fused[1], np.asarray(probs), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(fused[2], np.asarray(best_log_score), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(fused[3], np.asarray(best_argmax))
    np.testing.assert_allclose(fused[4], np.asarray(max_posterior), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(fused[5], np.asarray(recon_mask))
    np.testing.assert_array_equal(fused[6], np.asarray(recon_rot_mask))
    np.testing.assert_array_equal(fused[7], np.asarray(n_sig))
    np.testing.assert_allclose(fused[8], np.asarray(recon_probs), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(fused[9], np.asarray(probs_sum_t), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(fused[10], np.asarray(recon_probs_sum_t), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(fused[11], np.asarray(summed), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(fused[12], np.asarray(ctf_probs), rtol=2e-6, atol=2e-6)


def test_local_search_uses_selected_only_fine_rotation_grid_when_oversampling_is_enabled(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Exact local search selects from the fine-order grid without materializing it."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    order_sizes = {4: 4, 5: 9}
    grid_calls = []
    local_calls = []

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        grid_calls.append(("rot", order))
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        grid_calls.append(("euler", order))
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "rotations_is_none": rotation_grid_rotations is None,
                "eulers_is_none": rotation_grid_eulers is None,
                "rotation_grid_random_perturbation": kwargs.get("rotation_grid_random_perturbation"),
                "rotation_grid_angular_sampling_deg": kwargs.get("rotation_grid_angular_sampling_deg"),
            }
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(experiment_dataset.n_units),
        )
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_ids = np.zeros(experiment_dataset.n_units, dtype=np.int32)
            return base_outputs + (best_rots, best_trans, best_ids, relion_stats, noise_stats)
        return base_outputs + (relion_stats, noise_stats)

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(
                max(1, fake_rotation_grid_size(healpix_order)),
                dtype=np.float64,
            )
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=99),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
    )

    assert not any(kind == "rot" and order == 5 for kind, order in grid_calls)
    assert not any(kind == "euler" and order == 5 for kind, order in grid_calls)
    assert local_calls
    for call in local_calls:
        assert call["healpix_order"] == 5
        assert call["rotations_is_none"]
        assert call["eulers_is_none"]
        assert call["rotation_grid_random_perturbation"] == 0.0
        assert call["rotation_grid_angular_sampling_deg"] == pytest.approx(
            relion_angular_sampling_deg(5, adaptive_oversampling=0),
        )


def test_local_search_applies_perturbation_to_generated_fine_rotation_grid(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Selected-only fine local grids must carry the RELION perturbation metadata."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    order_sizes = {4: 4, 5: 9}
    perturb_calls = []
    local_calls = []

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_advance_relion_perturbation(current, perturb_factor, rng):
        _ = (current, perturb_factor, rng)
        return 0.25

    def fake_apply_relion_rotation_perturbation(rotations, random_perturbation, angular_sampling_deg):
        perturb_calls.append(
            {
                "n_rot": int(np.asarray(rotations).shape[0]),
                "random_perturbation": float(random_perturbation),
                "angular_sampling_deg": float(angular_sampling_deg),
            }
        )
        sentinel = np.zeros_like(np.asarray(rotations, dtype=np.float32))
        sentinel[:, 0, 0] = 7.0
        return sentinel

    def fake_apply_relion_rotation_perturbation_to_eulers(eulers, random_perturbation, angular_sampling_deg):
        perturb_calls.append(
            {
                "n_rot": int(np.asarray(eulers).shape[0]),
                "random_perturbation": float(random_perturbation),
                "angular_sampling_deg": float(angular_sampling_deg),
            }
        )
        sentinel_rotations = np.zeros((np.asarray(eulers).shape[0], 3, 3), dtype=np.float32)
        sentinel_rotations[:, 0, 0] = 7.0
        sentinel_eulers = np.full((np.asarray(eulers).shape[0], 3), 5.0, dtype=np.float32)
        return sentinel_rotations, sentinel_eulers

    def fake_r_to_relion(rotations, degrees=True):
        _ = degrees
        return np.full((np.asarray(rotations).shape[0], 3), 5.0, dtype=np.float32)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            prior_rotations,
            sigma_rot,
            sigma_psi,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            kwargs,
        )
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "rotations_is_none": rotation_grid_rotations is None,
                "eulers_is_none": rotation_grid_eulers is None,
                "rotation_grid_random_perturbation": kwargs.get("rotation_grid_random_perturbation"),
                "rotation_grid_angular_sampling_deg": kwargs.get("rotation_grid_angular_sampling_deg"),
            }
        )
        n_shells = half_datasets[0].image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(half_datasets[0].n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(half_datasets[0].n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(half_datasets[0].n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(half_datasets[0].n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(half_datasets[0].n_units),
        )
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], half_datasets[0].n_units, axis=0)
            best_trans = np.zeros((half_datasets[0].n_units, 2), dtype=np.float32)
            best_ids = np.zeros(half_datasets[0].n_units, dtype=np.int32)
            return base_outputs + (best_rots, best_trans, best_ids, relion_stats, noise_stats)
        return base_outputs + (relion_stats, noise_stats)

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "advance_relion_perturbation", fake_advance_relion_perturbation)
    monkeypatch.setattr(refine_mod, "apply_relion_rotation_perturbation", fake_apply_relion_rotation_perturbation)
    monkeypatch.setattr(
        refine_mod,
        "apply_relion_rotation_perturbation_to_eulers",
        fake_apply_relion_rotation_perturbation_to_eulers,
    )
    monkeypatch.setattr(refine_mod.utils, "R_to_relion", fake_r_to_relion)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(max(1, fake_rotation_grid_size(healpix_order)), dtype=np.float64)
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=111),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        perturb_factor=0.5,
    )

    assert any(
        call["n_rot"] == order_sizes[4]
        and np.isclose(call["angular_sampling_deg"], relion_angular_sampling_deg(4, adaptive_oversampling=0))
        for call in perturb_calls
    )
    assert not any(call["n_rot"] == order_sizes[5] for call in perturb_calls)
    assert local_calls
    assert local_calls[0]["rotations_is_none"]
    assert local_calls[0]["eulers_is_none"]
    assert local_calls[0]["rotation_grid_random_perturbation"] == pytest.approx(0.25)
    assert local_calls[0]["rotation_grid_angular_sampling_deg"] == pytest.approx(
        relion_angular_sampling_deg(5, adaptive_oversampling=0),
    )


def test_local_search_uses_negative_previous_offsets_for_translation_prior(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Local-search priors use RELION's pdf_offset units, not pre-shift pixels."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    order_sizes = {4: 4, 5: 9}
    prev_h1 = np.array([[0.5, -0.25], [1.0, 0.75]], dtype=np.float32)
    prev_h2 = np.array([[-0.75, 0.25], [0.25, -1.25]], dtype=np.float32)
    local_prior_translations = []

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_run_em(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            mean,
            mean_variance,
            noise_variance,
            prior_rotations,
            rotation_grid_rotations,
            rotation_grid_eulers,
            healpix_order,
            sigma_rot,
            sigma_psi,
            translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            kwargs,
        )
        local_prior_translations.append(np.asarray(prior_translations, dtype=np.float32).copy())
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "_compute_significance_batched",
        lambda *args, **kwargs: (
            np.ones(args[3].shape[0], dtype=bool),
            np.full(args[0].n_units, args[3].shape[0] * np.asarray(translations).shape[0], dtype=np.int32),
            np.zeros(args[0].n_units, dtype=np.int32),
            [None] * args[0].n_units,
        ),
    )
    monkeypatch.setattr(refine_mod, "should_skip_adaptive_pass2", lambda *a, **k: (True, 1.0))
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(
                max(1, fake_rotation_grid_size(healpix_order)),
                dtype=np.float64,
            )
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=123),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
    )

    assert len(local_prior_translations) == 2
    np.testing.assert_allclose(local_prior_translations[0], -np.rint(prev_h1) / 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(local_prior_translations[1], -np.rint(prev_h2) / 1.0, rtol=1e-6, atol=1e-6)


def test_local_search_coarse_translation_prior_mode_uses_unperturbed_base_grid(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    order_sizes = {4: 4, 5: 9}
    prev_h1 = np.zeros((half_datasets[0].n_units, 2), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 2), dtype=np.float32)
    recorded_translation_reference_grids = []

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_run_em(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            prior_rotations,
            rotation_grid_rotations,
            rotation_grid_eulers,
            healpix_order,
            sigma_rot,
            sigma_psi,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
        )
        recorded_translation_reference_grids.append(
            np.asarray(kwargs["translation_prior_reference_translations"], dtype=np.float32).copy()
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "_compute_significance_batched",
        lambda *args, **kwargs: (
            np.ones(args[3].shape[0], dtype=bool),
            np.full(args[0].n_units, args[3].shape[0] * np.asarray(translations).shape[0], dtype=np.int32),
            np.zeros(args[0].n_units, dtype=np.int32),
            [None] * args[0].n_units,
        ),
    )
    monkeypatch.setattr(refine_mod, "should_skip_adaptive_pass2", lambda *a, **k: (True, 1.0))
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(max(1, fake_rotation_grid_size(healpix_order)), dtype=np.float64)
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=123),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
        perturb_factor=0.5,
        perturb_seed=0,
        local_search_translation_prior_mode="coarse",
    )

    assert recorded_translation_reference_grids
    coarse_grid = np.asarray(translations, dtype=np.float32)
    for grid in recorded_translation_reference_grids:
        np.testing.assert_allclose(grid, coarse_grid, rtol=1e-6, atol=1e-6)


def test_local_search_os0_keeps_full_local_support_for_mstep(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """RELION os0 local search keeps all fine candidates in storeWeightedSums."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    order_sizes = {4: 4}
    reconstruct_flags = []

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        return np.tile(np.eye(3, dtype=np.float32), (fake_rotation_grid_size(order), 1, 1))

    def fake_get_grid_eulers(order):
        return np.zeros((fake_rotation_grid_size(order), 3), dtype=np.float32)

    def fake_run_em(experiment_dataset, mean, mean_variance, noise_variance, rotations, translations, disc_type, **kwargs):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_local_search(experiment_dataset, *args, **kwargs):
        _ = args
        reconstruct_flags.append(kwargs["reconstruct_significant_only"])
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            np.tile(np.eye(3, dtype=np.float32)[None, :, :], (experiment_dataset.n_units, 1, 1)),
            np.zeros((experiment_dataset.n_units, 2), dtype=np.float32),
            np.zeros(experiment_dataset.n_units, dtype=np.int64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[4], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(max(1, fake_rotation_grid_size(healpix_order)), dtype=np.float64)
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=222),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=0,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
    )

    assert reconstruct_flags == [False, False]


def _run_refine_with_stubbed_exact_local_batch_sizes(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
    *,
    env_override=None,
):
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    order_sizes = {4: 4}
    image_batch_sizes = []

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        return np.tile(np.eye(3, dtype=np.float32), (fake_rotation_grid_size(order), 1, 1))

    def fake_get_grid_eulers(order):
        return np.zeros((fake_rotation_grid_size(order), 3), dtype=np.float32)

    def fake_run_em(experiment_dataset, mean, mean_variance, noise_variance, rotations, translations, disc_type, **kwargs):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_local_search(experiment_dataset, *args, **kwargs):
        _ = args
        image_batch_sizes.append(int(kwargs["image_batch_size"]))
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            np.tile(np.eye(3, dtype=np.float32)[None, :, :], (experiment_dataset.n_units, 1, 1)),
            np.zeros((experiment_dataset.n_units, 2), dtype=np.float32),
            np.zeros(experiment_dataset.n_units, dtype=np.int64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[4], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    if env_override is None:
        monkeypatch.delenv("RECOVAR_RELION_EXACT_LOCAL_IMAGE_BATCH_SIZE", raising=False)
    else:
        monkeypatch.setenv("RECOVAR_RELION_EXACT_LOCAL_IMAGE_BATCH_SIZE", str(env_override))
    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(max(1, fake_rotation_grid_size(healpix_order)), dtype=np.float64)
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=226),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=0,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
    )

    return image_batch_sizes


def test_local_search_exact_path_uses_safe_multi_image_batches(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Exact local search should not fall back to one-image chunks by default."""
    image_batch_sizes = _run_refine_with_stubbed_exact_local_batch_sizes(
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    )
    assert image_batch_sizes == [N_IMAGES, N_IMAGES]


def test_local_search_exact_batch_size_env_override(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    image_batch_sizes = _run_refine_with_stubbed_exact_local_batch_sizes(
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
        env_override=2,
    )
    assert image_batch_sizes == [2, 2]


def test_local_search_coarse_translation_prior_mode_uses_replay_sampling_grid_when_available(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
    tmp_path,
):
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    order_sizes = {4: 4, 5: 9}
    prev_h1 = np.zeros((half_datasets[0].n_units, 2), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 2), dtype=np.float32)
    recorded_translation_reference_grids = []

    relion_pixel_size = 4.25
    for ds in half_datasets:
        ds.voxel_size = relion_pixel_size
    replay_offset_range = 2.411663
    replay_offset_step = 1.220812

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_run_em(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            prior_rotations,
            rotation_grid_rotations,
            rotation_grid_eulers,
            healpix_order,
            sigma_rot,
            sigma_psi,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
        )
        recorded_translation_reference_grids.append(
            np.asarray(kwargs["translation_prior_reference_translations"], dtype=np.float32).copy()
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "_compute_significance_batched",
        lambda *args, **kwargs: (
            np.ones(args[3].shape[0], dtype=bool),
            np.full(args[0].n_units, args[3].shape[0] * np.asarray(translations).shape[0], dtype=np.int32),
            np.zeros(args[0].n_units, dtype=np.int32),
            [None] * args[0].n_units,
        ),
    )
    monkeypatch.setattr(refine_mod, "should_skip_adaptive_pass2", lambda *a, **k: (True, 1.0))
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(max(1, fake_rotation_grid_size(healpix_order)), dtype=np.float64)
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )
    monkeypatch.setattr(
        refine_mod,
        "read_relion_sampling_metadata",
        lambda _path: {
            "random_perturbation": -0.13168,
            "perturbation_factor": 0.5,
            "healpix_order": 5,
            "offset_range": replay_offset_range,
            "offset_step": replay_offset_step,
        },
    )
    monkeypatch.setattr(refine_mod.os.path, "exists", lambda _path: False)

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=123),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
        perturb_factor=0.5,
        perturb_seed=0,
        local_search_translation_prior_mode="coarse",
        perturb_replay_relion_dir=str(tmp_path),
        init_relion_iteration=13,
    )

    assert recorded_translation_reference_grids
    replay_grid = get_translation_grid(
        replay_offset_range / relion_pixel_size,
        replay_offset_step / relion_pixel_size,
    ).astype(np.float32)
    for grid in recorded_translation_reference_grids:
        np.testing.assert_allclose(grid, replay_grid, rtol=1e-6, atol=1e-6)


def test_replay_current_size_uses_control_model_star():
    assert _replay_control_model_iteration(0, 0) == 1
    assert _replay_control_model_iteration(1, 0) == 2
    assert _replay_control_model_iteration(13, 0) == 14


def test_first_local_iteration_uses_previous_best_rotations_without_dense_bootstrap(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """hp4 should enter local search immediately when previous best rotations exist."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    dense_calls = []
    local_calls = []
    prev_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    def fake_run_em(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        dense_calls.append(int(np.asarray(rotations).shape[0]))
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            mean,
            mean_variance,
            noise_variance,
            sigma_rot,
            sigma_psi,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            kwargs,
        )
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "prior_shape": np.asarray(prior_rotations).shape,
                "grid_shape": np.asarray(rotation_grid_rotations).shape,
            }
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotation_grid_rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(
                max(1, rotation_grid_size(healpix_order)),
                dtype=np.float64,
            )
            / max(1, rotation_grid_size(healpix_order))
        ),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(rotation_grid_size(4), seed=7),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=512,
        init_current_size=16,
        adaptive_oversampling=0,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        replay_iteration_overrides=[
            {
                "local_search": True,
                "healpix_order": 4,
                "previous_best_rotation_eulers": [prev_h1, prev_h2],
            }
        ],
        skip_final_iteration=True,
    )

    assert local_calls
    assert not dense_calls
    for call in local_calls:
        assert call["healpix_order"] == 4
        assert call["prior_shape"][0] == half_datasets[0].n_units


def test_init_previous_best_rotation_eulers_seed_first_local_iteration(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Initial previous-best eulers should skip the dense hp4 bootstrap."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    dense_calls = []
    local_calls = []
    prev_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    def fake_run_em(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, translations, disc_type, kwargs)
        dense_calls.append(int(np.asarray(rotations).shape[0]))
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            mean,
            mean_variance,
            noise_variance,
            sigma_rot,
            sigma_psi,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            kwargs,
        )
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "prior_shape": np.asarray(prior_rotations).shape,
                "grid_shape": np.asarray(rotation_grid_rotations).shape,
            }
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotation_grid_rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(
                max(1, rotation_grid_size(healpix_order)),
                dtype=np.float64,
            )
            / max(1, rotation_grid_size(healpix_order))
        ),
    )

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(rotation_grid_size(4), seed=11),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=512,
        init_current_size=16,
        adaptive_oversampling=0,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_rotation_eulers=[prev_h1, prev_h2],
        skip_final_iteration=True,
    )

    assert local_calls
    assert not dense_calls
    for call in local_calls:
        assert call["healpix_order"] == 4
        assert call["prior_shape"][0] == half_datasets[0].n_units


def test_relion_mode_writes_absolute_translations_from_previous_offset(
    rng,
    init_volume,
    translations,
    monkeypatch,
):
    """RELION-mode writeback should use old_offset + delta."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    half_datasets = [MockDataset(1, rng), MockDataset(1, rng)]
    prev_h1 = np.array([[0.5, -0.25]], dtype=np.float32)
    prev_h2 = np.array([[-0.4, 0.3]], dtype=np.float32)
    chosen_trans = np.asarray(translations[1], dtype=np.float32)

    def fake_run_em(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, rotations, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        hard_assignment = np.full(experiment_dataset.n_units, 1, dtype=np.int32)
        return (
            None,
            hard_assignment,
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)

    result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(1, seed=123),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=1,
        init_current_size=16,
        adaptive_oversampling=0,
        nside_level=1,
        mode="relion",
        init_healpix_order=1,
        max_healpix_order=1,
        init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
        adaptive_fraction=1.0,
        skip_final_iteration=True,
    )

    expected_h1 = relion_translation_search_base(prev_h1) + chosen_trans[None, :]
    expected_h2 = relion_translation_search_base(prev_h2) + chosen_trans[None, :]
    np.testing.assert_allclose(half_datasets[0].translations, expected_h1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(half_datasets[1].translations, expected_h2, rtol=1e-6, atol=1e-6)

    best_hist = result["best_translations_history"]
    assert len(best_hist) == 1
    np.testing.assert_allclose(
        best_hist[0],
        np.concatenate([expected_h1, expected_h2], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_local_search_decodes_hard_assignments_on_fine_grid(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Oversampled local-search assignments must be decoded on the fine grid."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    order_sizes = {4: 4, 5: 9}
    fine_idx = order_sizes[5] - 1
    trans_idx = 1

    def fake_rotation_grid_size(order):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        mats = np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))
        for i in range(order_sizes[order]):
            mats[i, 0, 0] = 1.0 + i
        return mats

    def fake_get_grid_eulers(order):
        order = int(order)
        vals = np.arange(order_sizes[order], dtype=np.float32)
        return np.stack([vals, vals + 100.0, vals + 200.0], axis=1)

    def fake_run_em(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (mean, mean_variance, noise_variance, rotations, translations, disc_type, kwargs)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        return (
            None,
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        **kwargs,
    ):
        _ = (
            mean,
            mean_variance,
            noise_variance,
            prior_rotations,
            rotation_grid_rotations,
            rotation_grid_eulers,
            healpix_order,
            sigma_rot,
            sigma_psi,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            kwargs,
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        assignment = np.full(
            experiment_dataset.n_units,
            fine_idx * np.asarray(translations).shape[0] + trans_idx,
            dtype=np.int32,
        )
        return (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            assignment,
            RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(
                max(1, fake_rotation_grid_size(healpix_order)),
                dtype=np.float64,
            )
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=321),
        translations,
        disc_type="linear_interp",
        max_iter=2,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        mode="relion",
        init_healpix_order=4,
        max_healpix_order=4,
        perturb_factor=0.0,
    )

    expected_rotation = _selected_rotation_matrices(
        np.array([fine_idx], dtype=np.int32),
        None,
        build_local_search_grid_metadata(5),
    )
    expected_euler = iteration_loop_module.utils.R_to_relion(expected_rotation, degrees=True)[0].astype(np.float32)
    observed = np.asarray(result["best_rotation_eulers_history"][1], dtype=np.float32)
    assert observed.shape[0] == N_IMAGES
    np.testing.assert_allclose(
        observed,
        np.repeat(expected_euler[None, :], N_IMAGES, axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


# ===========================================================================
# Test 3: Invalid mode
# ===========================================================================


class TestInvalidMode:
    def test_invalid_mode_raises(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """Only RELION mode is supported."""
        with pytest.raises(ValueError, match="only 'relion' is supported"):
            refine_single_volume(
                half_datasets,
                init_volume,
                jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
                jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
                rotations,
                translations,
                mode="bogus",
                max_iter=1,
            )
