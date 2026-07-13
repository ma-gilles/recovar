"""Smoke tests for refine_single_volume's RELION-only path.

Verifies:
1. RELION mode runs without error on a tiny dataset (4 images, 8px, 2 iters)
2. Returns the expected dict keys (including RELION-specific ones)
3. Convergence state is a RefinementState instance
4. data_vs_prior_trajectory and ave_Pmax_trajectory are populated
"""

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
import healpy as hp
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
import recovar.em.dense_single_volume.iteration_loop as iteration_loop_module
import recovar.em.dense_single_volume.local_layout as local_layout_module
import recovar.reconstruction.regularization as regularization_module
from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.em_engine import _batch_parameter_rows, run_em
from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch as _fetch_indexed_batch
from recovar.em.dense_single_volume.helpers.convergence import (
    RefinementState,
    refine_angular_sampling,
    should_refine_angular_sampling,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    enforce_half_volume_x0,
    half_volume_accumulator_shape,
    half_volume_accumulators_to_full,
)
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    integer_pre_shifts_or_none,
)
from recovar.em.dense_single_volume.helpers.local_search import (
    _local_search_engine_rotation_block_size,
)
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    collapse_rotation_posterior_to_direction_prior,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    normalize_direction_prior_per_half,
    relion_local_translation_prior_center,
    relion_sigma_offset_prior_center,
    relion_translation_prior_center,
    relion_translation_search_base,
)
from recovar.em.dense_single_volume.helpers.preprocessing import resolve_image_mask_for_half_preprocess
from recovar.em.dense_single_volume.helpers.resolution import (
    _bootstrap_current_size_relion,
    bootstrap_current_size_from_ini_high_relion,
    clamp_relion_coarse_image_size,
    compute_coarse_image_size,
    shell_index_to_resolution_angstrom,
    should_skip_adaptive_pass2,
)
from recovar.em.dense_single_volume.helpers.score_constraints import DenseScoreConstraints
from recovar.em.dense_single_volume.helpers.significance import (
    _compute_k_class_significance_batched,
    _compute_significance_batched,
)
from recovar.em.dense_single_volume.helpers.types import NoiseStats, RelionStats
from recovar.em.dense_single_volume.iteration_loop import (
    _align_fourier_volume_sign_to_reference,
    _combined_class_direction_prior_from_halves,
    _combined_noise_stats,
    _estimate_relion_em_batch_sizes,
    _exhaustive_grid_order_for_state,
    _normalize_noise_variance_per_half,
    _replay_control_model_iteration,
    _rotation_eulers_for_canonical_or_custom_grid,
    refine_single_volume,
    update_relion_norm_scale_corrections,
)
from recovar.em.dense_single_volume.k_class import (
    KClassEMResult,
    _sum_noise_stats,
    run_dense_k_class_em,
    run_local_k_class_em,
)
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_weighted_sums,
    flatten_bucket_rotations,
    flatten_bucket_rows,
)
from recovar.em.dense_single_volume.local_debug import (
    current_size_matches_request,
    iteration_matches_request,
    maybe_write_debug_score_dump,
)
from recovar.em.dense_single_volume.local_em_engine import (
    EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP_ENV,
    EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV,
    EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV,
    EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV,
    EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV,
    EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM_ENV,
    EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV,
    EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION,
    EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR,
    EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV,
    EXACT_LOCAL_TARGET_ROW_PIXELS_ENV,
    _build_reconstruction_pack_indices,
    _exact_local_effective_max_hypotheses_per_microbatch,
    _exact_local_max_hypotheses_per_microbatch,
    _adjoint_slice_volume_maybe_windowed_row_chunks,
    _local_processed_half_cache_enabled,
    _local_raw_cache_enabled,
    _pad_local_big_jit_image_axis,
    _prepare_local_exact_bucket,
    _reorder_bucket_to_indices,
    run_local_em_exact,
)
from recovar.em.dense_single_volume.local_layout import (
    LocalBucketSpec,
    LocalHypothesisLayout,
    _selected_rotation_matrices,
    bucket_local_hypothesis_layout,
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
    build_pass2_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_score_pass import (
    compute_reconstruction_support,
    compute_reconstruction_support_from_threshold,
    fused_score_normalize_support_abs2_on_demand,
    normalize_local_scores,
    normalize_local_scores_float32,
    normalize_local_scores_with_log_z,
    normalize_local_scores_with_log_z_float32,
    score_local_bucket,
    score_local_bucket_abs2_weighted_on_demand,
)
from recovar.em.sampling import (
    apply_relion_rotation_perturbation,
    apply_relion_rotation_perturbation_to_eulers,
    build_local_search_grid_metadata,
    get_local_rotation_grid_fast,
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
    get_relion_rotation_grid,
    get_relion_rotation_grid_eulers,
    get_translation_grid,
    relion_angular_sampling_deg,
    rotation_grid_n_in_planes,
    rotation_grid_size,
)

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


def test_final_all_data_grid_correct_env_defaults_to_quality_mode(monkeypatch):
    env_name = "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT"
    monkeypatch.delenv(env_name, raising=False)
    assert iteration_loop_module._final_all_data_grid_correct_enabled() is False

    monkeypatch.setenv(env_name, "0")
    assert iteration_loop_module._final_all_data_grid_correct_enabled() is False

    monkeypatch.setenv(env_name, "false")
    assert iteration_loop_module._final_all_data_grid_correct_enabled() is False

    monkeypatch.setenv(env_name, "1")
    assert iteration_loop_module._final_all_data_grid_correct_enabled() is True

    monkeypatch.setenv(env_name, "unexpected")
    assert iteration_loop_module._final_all_data_grid_correct_enabled() is False


def test_final_all_data_after_max_iter_env_defaults_to_disabled(monkeypatch):
    env_name = "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER"
    monkeypatch.delenv(env_name, raising=False)
    assert iteration_loop_module._final_all_data_after_max_iter_enabled() is False
    assert (
        iteration_loop_module._should_run_final_all_data_iteration(
            has_converged=False,
            iteration=5,
            max_iter=5,
            force_max_iter_after_convergence=False,
        )
        is False
    )

    monkeypatch.setenv(env_name, "1")
    assert iteration_loop_module._final_all_data_after_max_iter_enabled() is True
    assert (
        iteration_loop_module._should_run_final_all_data_iteration(
            has_converged=False,
            iteration=5,
            max_iter=5,
            force_max_iter_after_convergence=False,
            k_class_enabled=False,
        )
        is True
    )
    assert (
        iteration_loop_module._should_run_final_all_data_iteration(
            has_converged=False,
            iteration=5,
            max_iter=5,
            force_max_iter_after_convergence=False,
            k_class_enabled=True,
        )
        is False
    )
    assert (
        iteration_loop_module._should_run_final_all_data_iteration(
            has_converged=True,
            iteration=5,
            max_iter=5,
            force_max_iter_after_convergence=False,
            k_class_enabled=True,
        )
        is True
    )

    assert (
        iteration_loop_module._should_run_final_all_data_iteration(
            has_converged=False,
            iteration=4,
            max_iter=5,
            force_max_iter_after_convergence=False,
        )
        is False
    )
    assert (
        iteration_loop_module._should_run_final_all_data_iteration(
            has_converged=False,
            iteration=5,
            max_iter=5,
            force_max_iter_after_convergence=True,
        )
        is False
    )


def test_final_local_sampling_orders_use_advanced_final_star_parent():
    """100k replay advances final parent hp6->hp7, so os1 must score fine hp8."""

    parent_order, fine_order = iteration_loop_module._final_local_sampling_orders(
        state_healpix_order=6,
        adaptive_oversampling=1,
        final_sampling_healpix_order=7,
    )

    assert parent_order == 7
    assert fine_order == 8


def test_final_local_sampling_orders_preserve_equal_order_and_state_fallback():
    """10k replay stays at hp6, and missing final metadata retains state hp6."""

    assert iteration_loop_module._final_local_sampling_orders(
        state_healpix_order=6,
        adaptive_oversampling=1,
        final_sampling_healpix_order=6,
    ) == (6, 7)
    assert iteration_loop_module._final_local_sampling_orders(
        state_healpix_order=6,
        adaptive_oversampling=1,
        final_sampling_healpix_order=None,
    ) == (6, 7)


def test_local_debug_current_size_minus_one_is_wildcard():
    assert current_size_matches_request(None, 80)
    assert current_size_matches_request({-1}, 80)
    assert not current_size_matches_request({-1}, None)
    assert current_size_matches_request({-2}, None)
    assert not current_size_matches_request({-2}, 80)
    assert current_size_matches_request({80}, 80)
    assert not current_size_matches_request({80}, 96)

    assert iteration_matches_request(None, 11)
    assert iteration_matches_request({11}, 11)
    assert not iteration_matches_request({11}, 10)


def test_k1_skip_significance_pruning_env_defaults_to_disabled(monkeypatch):
    env_name = "RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING"
    monkeypatch.delenv(env_name, raising=False)
    assert iteration_loop_module._k1_skip_significance_pruning_enabled() is False

    monkeypatch.setenv(env_name, "0")
    assert iteration_loop_module._k1_skip_significance_pruning_enabled() is False

    monkeypatch.setenv(env_name, "false")
    assert iteration_loop_module._k1_skip_significance_pruning_enabled() is False

    monkeypatch.setenv(env_name, "1")
    assert iteration_loop_module._k1_skip_significance_pruning_enabled() is True

    monkeypatch.setenv(env_name, "unexpected")
    assert iteration_loop_module._k1_skip_significance_pruning_enabled() is False


def test_kclass_final_reconstruction_does_not_predivide_class_accumulators():
    source = inspect.getsource(iteration_loop_module.refine_single_volume)

    assert "final_ft_ctf[class_idx] /" not in source
    assert "final_ft_y[class_idx] /" not in source


def test_replay_translation_grid_preserves_state_grid_for_subtolerance_star_rounding(monkeypatch, tmp_path):
    class State:
        healpix_order = 3
        max_healpix_order = 3
        auto_local_healpix_order = 4
        do_local_search = False
        sigma_rot = 0.0
        sigma_psi = 0.0
        translation_range = 3.0
        translation_step = 1.0

    class Cryo:
        voxel_size = 1.4166666666666667

    sampling_paths = []

    def fake_sampling_metadata(path):
        sampling_paths.append(path)
        return {
            "random_perturbation": -0.11451,
            "perturbation_factor": 0.5,
            "healpix_order": 3,
            "psi_step": 7.5,
            "offset_range": 4.25,
            "offset_step": 1.416667,
        }

    monkeypatch.setattr(iteration_loop_module, "read_relion_sampling_metadata", fake_sampling_metadata)

    state = State()
    state_grid = get_translation_grid(state.translation_range, state.translation_step)
    rounded_star_grid = get_translation_grid(4.25 / Cryo.voxel_size, 1.416667 / Cryo.voxel_size)
    assert state_grid.shape[0] == 29
    assert rounded_star_grid.shape[0] == 25

    result = iteration_loop_module.apply_iter_replay_overrides(
        iter_replay_override=None,
        perturb_replay_relion_dir=str(tmp_path),
        perturb_replay_relion_prefix="custom",
        init_relion_iteration=6,
        iteration=0,
        state=state,
        cs=64,
        cryo=Cryo(),
        k_class_enabled=False,
        n_classes=1,
        relion_half_inputs=iteration_loop_module._RelionHalfInputState.from_initial_values(
            previous_best_translations=None,
            previous_best_rotation_eulers=None,
            image_corrections=None,
            scale_corrections=None,
        ),
        previous_best_rotations=[None, None],
        noise_variance_per_half=[None, None],
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        previous_noise_radial_per_half=[None, None],
        previous_noise_radial=None,
        current_sigma_offset_angstrom=10.0,
        class_direction_prior_per_half=[None, None],
        class_direction_prior_order_per_half=[None, None],
        global_direction_prior_per_half=[None, None],
        global_direction_prior_order_per_half=[None, None],
    )

    replay_grid = np.asarray(result.prior_translations)
    assert replay_grid.shape[0] == 29
    assert get_translation_grid(state.translation_range, state.translation_step).shape[0] == 29
    assert state.translation_range == pytest.approx(3.0)
    assert state.translation_step == pytest.approx(1.0)
    assert sampling_paths == [str(tmp_path / "custom_it007_sampling.star")]
    np.testing.assert_allclose(replay_grid, state_grid, rtol=0.0, atol=1e-6)


def test_local_translation_prior_ignores_stale_replay_grid_shape():
    current_grid = get_translation_grid(3.0, 1.0).astype(np.float32)
    stale_replay_grid = get_translation_grid(3.0, 1.0000002352941175).astype(np.float32)
    assert current_grid.shape[0] == 29
    assert stale_replay_grid.shape[0] == 25

    chosen, source, mismatched = iteration_loop_module._local_translation_prior_reference_translations(
        current_translations=current_grid + 0.125,
        base_translations=current_grid,
        replay_prior_translations=stale_replay_grid,
    )

    assert source == "base"
    assert mismatched is True
    np.testing.assert_allclose(chosen, current_grid)


def test_replay_override_preserves_half_specific_sigma_offsets():
    class State:
        healpix_order = 1
        max_healpix_order = 1
        auto_local_healpix_order = 4
        do_local_search = False
        sigma_rot = 0.0
        sigma_psi = 0.0
        translation_range = 3.0
        translation_step = 1.0

    class Cryo:
        voxel_size = 1.0

    result = iteration_loop_module.apply_iter_replay_overrides(
        iter_replay_override={
            "translation_sigma_angstrom": 99.0,
            "translation_sigma_angstrom_per_half": [5.0, 7.0],
        },
        perturb_replay_relion_dir=None,
        init_relion_iteration=0,
        iteration=1,
        state=State(),
        cs=8,
        cryo=Cryo(),
        k_class_enabled=False,
        n_classes=1,
        relion_half_inputs=iteration_loop_module._RelionHalfInputState.from_initial_values(
            previous_best_translations=None,
            previous_best_rotation_eulers=None,
            image_corrections=None,
            scale_corrections=None,
        ),
        previous_best_rotations=[None, None],
        noise_variance_per_half=[None, None],
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        previous_noise_radial_per_half=[None, None],
        previous_noise_radial=None,
        current_sigma_offset_angstrom=10.0,
        class_direction_prior_per_half=[None, None],
        class_direction_prior_order_per_half=[None, None],
        global_direction_prior_per_half=[None, None],
        global_direction_prior_order_per_half=[None, None],
    )

    assert result.current_sigma_offset_angstrom == pytest.approx(6.0)
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([5.0, 7.0])


def test_replay_override_replaces_native_norm_scale_state_but_keeps_group_ids():
    class State:
        healpix_order = 1
        max_healpix_order = 1
        auto_local_healpix_order = 4
        do_local_search = False
        sigma_rot = 0.0
        sigma_psi = 0.0
        translation_range = 3.0
        translation_step = 1.0

    class Cryo:
        voxel_size = 1.0

    group_ids = [np.asarray([0, 1], dtype=np.int64), np.asarray([1], dtype=np.int64)]
    relion_half_inputs = iteration_loop_module._RelionHalfInputState.from_initial_values(
        previous_best_translations=None,
        previous_best_rotation_eulers=None,
        image_corrections=[np.asarray([9.0, 9.0], dtype=np.float32), np.asarray([8.0], dtype=np.float32)],
        scale_corrections=[np.asarray([7.0, 7.0], dtype=np.float32), np.asarray([6.0], dtype=np.float32)],
        group_ids=group_ids,
    )

    iteration_loop_module.apply_iter_replay_overrides(
        iter_replay_override={
            "image_corrections": [
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([3.0], dtype=np.float32),
            ],
            "scale_corrections": [
                np.asarray([4.0, 5.0], dtype=np.float32),
                np.asarray([6.0], dtype=np.float32),
            ],
        },
        perturb_replay_relion_dir=None,
        init_relion_iteration=0,
        iteration=1,
        state=State(),
        cs=8,
        cryo=Cryo(),
        k_class_enabled=False,
        n_classes=1,
        relion_half_inputs=relion_half_inputs,
        previous_best_rotations=[None, None],
        noise_variance_per_half=[None, None],
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        previous_noise_radial_per_half=[None, None],
        previous_noise_radial=None,
        current_sigma_offset_angstrom=10.0,
        class_direction_prior_per_half=[None, None],
        class_direction_prior_order_per_half=[None, None],
        global_direction_prior_per_half=[None, None],
        global_direction_prior_order_per_half=[None, None],
    )

    np.testing.assert_allclose(relion_half_inputs.image_corrections[0], [1.0, 2.0])
    np.testing.assert_allclose(relion_half_inputs.scale_corrections[0], [4.0, 5.0])
    np.testing.assert_array_equal(relion_half_inputs.group_ids[0], group_ids[0])
    np.testing.assert_array_equal(relion_half_inputs.group_ids[1], group_ids[1])


def _pack_fake_local_search_outputs(
    base_outputs,
    relion_stats,
    noise_stats,
    kwargs,
    n_units,
    best_pose_details=(),
):
    outputs = list(base_outputs)
    outputs.extend(best_pose_details)
    outputs.append(relion_stats)
    if kwargs.get("accumulate_noise", False):
        outputs.append(noise_stats)
    if kwargs.get("return_profile", False):
        outputs.append({"reconstruction_sample_indices_by_image": [None] * int(n_units)})
    return tuple(outputs)


class _RawCacheFakeLoader:
    def __init__(self, *, n=8, D=16, dtype=np.float32):
        self.num_images = n
        self.image_size = D
        self._dtype = np.dtype(dtype)
        self._cached = None
        self.load_count = 0

    def load_all(self):
        self.load_count += 1
        self._cached = np.zeros(
            (self.num_images, self.image_size, self.image_size),
            dtype=self._dtype,
        )


class _RawCacheFakeBackend:
    def __init__(self, loader):
        self.source = loader


class _RawCacheFakeImageSource:
    def __init__(self, loader):
        self.backend = _RawCacheFakeBackend(loader)


class _RawCacheFakeDataset:
    def __init__(self, loader):
        self.image_source = _RawCacheFakeImageSource(loader)


def test_relion_raw_image_cache_loads_unique_loaders(monkeypatch):
    loader = _RawCacheFakeLoader()
    monkeypatch.setenv("RECOVAR_EM_RAW_IMAGE_CACHE", "auto")
    monkeypatch.setenv("RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB", "1")

    iteration_loop_module._maybe_cache_raw_image_loaders([_RawCacheFakeDataset(loader), _RawCacheFakeDataset(loader)])

    assert loader.load_count == 1
    assert loader._cached is not None


def test_relion_raw_image_cache_respects_memory_guard(monkeypatch):
    loader = _RawCacheFakeLoader(n=1024, D=1024)
    monkeypatch.setenv("RECOVAR_EM_RAW_IMAGE_CACHE", "auto")
    monkeypatch.setenv("RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB", "0.001")

    iteration_loop_module._maybe_cache_raw_image_loaders([_RawCacheFakeDataset(loader)])

    assert loader.load_count == 0
    assert loader._cached is None


def test_relion_raw_image_cache_can_be_disabled(monkeypatch):
    loader = _RawCacheFakeLoader()
    monkeypatch.setenv("RECOVAR_EM_RAW_IMAGE_CACHE", "off")

    iteration_loop_module._maybe_cache_raw_image_loaders([_RawCacheFakeDataset(loader)])

    assert loader.load_count == 0
    assert loader._cached is None


def test_exact_local_raw_cache_default_covers_50k_256_float32(monkeypatch):
    monkeypatch.delenv(EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV, raising=False)

    assert _local_raw_cache_enabled(50_000, (256, 256), np.float32)


def test_exact_local_raw_cache_respects_memory_guard(monkeypatch):
    monkeypatch.setenv(EXACT_LOCAL_RAW_CACHE_MAX_GB_ENV, "1")

    assert not _local_raw_cache_enabled(50_000, (256, 256), np.float32)


def test_exact_local_processed_half_cache_disabled_by_default(monkeypatch):
    monkeypatch.delenv(EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV, raising=False)

    assert not _local_processed_half_cache_enabled(
        25_007,
        256 * (256 // 2 + 1),
        np.complex64,
        store_recon_half=True,
    )


def test_exact_local_processed_half_cache_opt_in_covers_50k_256_halfset(monkeypatch):
    monkeypatch.setenv(EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV, "16")

    assert _local_processed_half_cache_enabled(
        25_007,
        256 * (256 // 2 + 1),
        np.complex64,
        store_recon_half=True,
    )


def test_exact_local_processed_half_cache_respects_memory_guard(monkeypatch):
    monkeypatch.setenv(EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV, "1")

    assert not _local_processed_half_cache_enabled(
        25_007,
        256 * (256 // 2 + 1),
        np.complex64,
        store_recon_half=True,
    )


def _force_exact_local_standard_gpu_default(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    monkeypatch.setattr(local_em_engine, "_visible_gpu_memory_bytes", lambda: None)


def test_exact_local_microbatch_default_matches_profiled_256_window(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)

    assert _exact_local_max_hypotheses_per_microbatch(None, 8018) == 23696


def test_exact_local_microbatch_high_memory_gpu_default(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.setattr(local_em_engine, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    assert (
        _exact_local_max_hypotheses_per_microbatch(
            None,
            12861,
            n_trans=36,
            n_recon_windowed=12723,
        )
        == 19905
    )


def test_exact_local_score_only_cap_covers_100k_parent_tile(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)
    monkeypatch.setattr(local_em_engine, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    rotation_counts = np.asarray([198], dtype=np.int32)
    layout = LocalHypothesisLayout(
        n_global_rotations=198,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.asarray([0, 198], dtype=np.int64),
        rotation_ids_flat=np.arange(198, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (198, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(198, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((9, 2), dtype=np.float32),
        translation_log_priors=np.zeros((1, 9), dtype=np.float32),
    )
    runtime_free_bytes = int(46.4 * 1024**3)
    cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        12861,
        n_trans=9,
        n_recon_windowed=12723,
        local_layout=layout,
        image_batch_size=168,
        rotation_block_size=198,
        score_only=True,
        runtime_free_memory_bytes=runtime_free_bytes,
    )
    expected_tile_cap = int(
        runtime_free_bytes
        * EXACT_LOCAL_SCORE_TILE_FREE_MEMORY_FRACTION
        // (9 * 12861 * np.dtype(np.float32).itemsize * EXACT_LOCAL_SCORE_TILE_LIVE_FACTOR)
    )

    assert cap == expected_tile_cap
    assert cap < 168 * 198
    assert cap // 198 == 86


def test_exact_local_score_only_cap_preserves_smaller_bucket_shape(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)
    monkeypatch.setattr(local_em_engine, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    rotation_counts = np.asarray([198], dtype=np.int32)
    layout = LocalHypothesisLayout(
        n_global_rotations=198,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.asarray([0, 198], dtype=np.int64),
        rotation_ids_flat=np.arange(198, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (198, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(198, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((9, 2), dtype=np.float32),
        translation_log_priors=np.zeros((1, 9), dtype=np.float32),
    )
    cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        4003,
        n_trans=9,
        n_recon_windowed=3923,
        local_layout=layout,
        image_batch_size=168,
        rotation_block_size=198,
        score_only=True,
        runtime_free_memory_bytes=int(46.4 * 1024**3),
    )
    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=168,
        rotation_block_size=198,
        max_hypotheses_per_microbatch=cap,
    )

    assert cap >= 168 * 198
    assert len(buckets) == 1


def test_exact_local_xhalf_full_bpref_uses_conservative_high_memory_cap(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.setattr(local_em_engine, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    high_memory_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        8320,
        n_trans=36,
        n_recon_windowed=8320,
    )
    full_bpref_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        8320,
        n_trans=36,
        n_recon_windowed=8320,
        allow_high_memory_default=False,
    )

    assert high_memory_cap == 30769
    assert full_bpref_cap == 17127
    assert full_bpref_cap < high_memory_cap


def test_exact_local_xhalf_current_bpref_uses_conservative_high_memory_cap(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.setattr(local_em_engine, "_visible_gpu_memory_bytes", lambda: 80 * 1024**3)

    high_memory_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        4003,
        n_trans=52,
        n_recon_windowed=3923,
    )
    current_bpref_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        4003,
        n_trans=52,
        n_recon_windowed=3923,
        allow_high_memory_default=False,
    )

    assert high_memory_cap == 63952
    assert current_bpref_cap == 35933
    assert current_bpref_cap < high_memory_cap


def test_exact_local_microbatch_target_row_pixels_override(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.setenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "380000000")

    assert _exact_local_max_hypotheses_per_microbatch(None, 8018) == 47393


def test_exact_local_microbatch_caps_fused_mstep_matmul(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)

    assert (
        _exact_local_max_hypotheses_per_microbatch(
            None,
            1091,
            n_trans=21,
            n_recon_windowed=1134,
        )
        == 65536
    )


def test_exact_local_microbatch_batches_full_parent_256_pass2(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)

    cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=84,
        n_recon_windowed=33024,
    )
    assert cap >= 3072

    n_images = 12
    local_rotations = 1416
    rotation_counts = np.full(n_images, local_rotations, dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    total_rotations = int(rotation_offsets[-1])
    layout = LocalHypothesisLayout(
        n_global_rotations=total_rotations,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=np.arange(total_rotations, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (total_rotations, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(total_rotations, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((84, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, 84), dtype=np.float32),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=17,
        rotation_block_size=26,
        max_hypotheses_per_microbatch=cap,
    )

    assert max(int(bucket.image_indices.shape[0]) for bucket in buckets) >= 2
    assert len(buckets) <= (n_images + 1) // 2


def test_exact_local_microbatch_boosts_high_res_local_batch_without_full_floor(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_BUCKET_QUANTUM", raising=False)

    n_images = 13
    local_rotations = 1536
    rotation_counts = np.full(n_images, local_rotations, dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    total_rotations = int(rotation_offsets[-1])
    layout = LocalHypothesisLayout(
        n_global_rotations=total_rotations,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=np.arange(total_rotations, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (total_rotations, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(total_rotations, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((116, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, 116), dtype=np.float32),
    )

    base_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
    )
    assert base_cap < n_images * local_rotations

    cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
    )

    assert cap == base_cap * 2
    assert cap < n_images * local_rotations
    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=n_images,
        rotation_block_size=19,
        max_hypotheses_per_microbatch=cap,
    )
    assert {int(bucket.image_indices.shape[0]) for bucket in buckets[:-1]} == {5}
    assert len(buckets) == 3
    assert max(int(bucket.image_indices.shape[0]) for bucket in buckets) == 5


def test_exact_local_microbatch_boost_can_be_disabled_for_mstep_pass2(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_BUCKET_QUANTUM", raising=False)

    n_images = 13
    local_rotations = 1536
    rotation_counts = np.full(n_images, local_rotations, dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    total_rotations = int(rotation_offsets[-1])
    layout = LocalHypothesisLayout(
        n_global_rotations=total_rotations,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=np.arange(total_rotations, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (total_rotations, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(total_rotations, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((116, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, 116), dtype=np.float32),
    )

    base_cap = _exact_local_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
    )
    boosted_cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
    )
    capped_pass2_cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
        allow_auto_boost=False,
    )
    bounded_xhalf_cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
        auto_boost_factor=1.25,
    )

    assert boosted_cap > base_cap
    assert capped_pass2_cap == base_cap
    assert bounded_xhalf_cap == int(np.floor(base_cap * 1.25))


def test_exact_local_xhalf_mstep_uses_explicit_microbatch_boost_hook():
    source = inspect.getsource(run_local_em_exact)

    assert "allow_microbatch_auto_boost = True" in source
    assert "auto_boost_factor=xhalf_auto_microbatch_boost" in source
    assert "allow_high_memory_default=not xhalf_bpref_mstep" in source


def test_exact_local_microbatch_env_override_keeps_lower_cap(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.setenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "190000000")
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_AUTO_MICROBATCH_BOOST_ENV, raising=False)

    n_images = 13
    local_rotations = 1536
    rotation_counts = np.full(n_images, local_rotations, dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    total_rotations = int(rotation_offsets[-1])
    layout = LocalHypothesisLayout(
        n_global_rotations=total_rotations,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=np.arange(total_rotations, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (total_rotations, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(total_rotations, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((116, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, 116), dtype=np.float32),
    )

    cap = _exact_local_effective_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
        local_layout=layout,
        image_batch_size=n_images,
        rotation_block_size=19,
    )

    assert cap == _exact_local_max_hypotheses_per_microbatch(
        None,
        33024,
        n_trans=116,
        n_recon_windowed=33024,
    )
    assert cap < n_images * local_rotations


def test_local_search_outer_batch_sizing_uses_current_size_window():
    source = inspect.getsource(iteration_loop_module._score_half_local)

    assert "current_size_for_batch=cs_for_engine" in source


def test_exact_local_microbatch_matmul_cap_can_be_disabled(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.delenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, raising=False)
    monkeypatch.setenv(EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB_ENV, "0")

    assert (
        _exact_local_max_hypotheses_per_microbatch(
            None,
            1091,
            n_trans=21,
            n_recon_windowed=1134,
        )
        == 65536
    )


def test_exact_local_microbatch_target_row_pixels_rejects_invalid(monkeypatch):
    _force_exact_local_standard_gpu_default(monkeypatch)
    monkeypatch.setenv(EXACT_LOCAL_TARGET_ROW_PIXELS_ENV, "0")

    with pytest.raises(ValueError, match=EXACT_LOCAL_TARGET_ROW_PIXELS_ENV):
        _exact_local_max_hypotheses_per_microbatch(None, 8018)


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


def test_relion_em_batch_sizing_preserves_small_safe_requests():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=4,
        requested_rotation_block_size=8,
        n_rot=8,
        n_trans=5,
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 4
    assert plan.rotation_block_size == 8


def test_relion_em_batch_sizing_clamps_highres_projection_tiles():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=294912,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 4
    assert plan.rotation_block_size == 13
    assert plan.projection_block_gb <= plan.projection_budget_gb
    assert plan.pose_pixel_tile_gb < 3.0
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb
    assert plan.translation_tile_gb <= plan.translation_tile_budget_gb


def test_relion_em_batch_sizing_does_not_pad_beyond_actual_rotation_grid():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=4608,
        n_trans=29,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 46
    assert plan.rotation_block_size == 65
    assert plan.projection_block_gb <= plan.projection_budget_gb
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb


def test_relion_em_batch_sizing_uses_runtime_gpu_occupancy(monkeypatch):
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 60.0)

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=4608,
        n_trans=29,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )

    assert plan.gpu_used_estimate_gb == pytest.approx(60.0)
    assert plan.rotation_block_size < 4608
    assert plan.projection_block_gb <= plan.projection_budget_gb


def test_relion_em_batch_sizing_caps_runtime_highres_local_translation_tile(monkeypatch):
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 24.0)

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=4608,
        n_rot=4608,
        n_trans=116,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )

    assert plan.image_batch_size <= 56
    assert plan.translation_tile_gb <= plan.translation_tile_budget_gb
    assert plan.gpu_used_estimate_gb == pytest.approx(24.0)


def test_relion_em_batch_sizing_caps_dense_big_jit_score_workspace(monkeypatch):
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 41.0)

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )

    assert plan.image_batch_size == 50
    assert plan.rotation_block_size == 146
    assert plan.projection_block_gb <= plan.projection_budget_gb
    assert plan.pose_pixel_tile_gb < 3.0
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb
    assert plan.gpu_used_estimate_gb == pytest.approx(41.0)


def test_relion_em_batch_sizing_uses_active_window_for_dense_score_workspace(monkeypatch):
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 0.0)

    common = dict(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )
    low_res = _estimate_relion_em_batch_sizes(**common, current_size=56)
    high_res = _estimate_relion_em_batch_sizes(**common, current_size=248)

    assert low_res.rotation_block_size == 8192
    assert high_res.rotation_block_size < 8192
    assert low_res.score_pixel_count < high_res.score_pixel_count


def test_relion_em_batch_sizing_allows_larger_adaptive_pass1_blocks():
    common = dict(
        requested_image_batch_size=500,
        requested_rotation_block_size=5000,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    pass1_coarse = _estimate_relion_em_batch_sizes(**common, current_size=100)
    pass2_fine = _estimate_relion_em_batch_sizes(**common, current_size=154)

    assert pass1_coarse.score_pixel_count < pass2_fine.score_pixel_count
    assert pass1_coarse.rotation_block_size > pass2_fine.rotation_block_size
    assert pass1_coarse.pose_pixel_tile_gb <= pass1_coarse.projection_budget_gb
    assert pass2_fine.pose_pixel_tile_gb <= pass2_fine.projection_budget_gb


def test_relion_em_batch_sizing_projection_budget_override_expands_pass1_blocks(monkeypatch):
    common = dict(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
        current_size=100,
    )

    monkeypatch.delenv("RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION", raising=False)
    default = _estimate_relion_em_batch_sizes(**common)

    monkeypatch.setenv("RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION", "0.40")
    expanded = _estimate_relion_em_batch_sizes(**common)

    assert expanded.rotation_block_size > default.rotation_block_size
    assert expanded.rotation_block_size <= common["requested_rotation_block_size"]
    assert expanded.pose_pixel_tile_gb <= expanded.projection_budget_gb


def test_relion_em_batch_sizing_projection_budget_override_rejects_invalid(monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION", "0")

    with pytest.raises(ValueError, match="RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION"):
        _estimate_relion_em_batch_sizes(
            requested_image_batch_size=64,
            requested_rotation_block_size=8192,
            n_rot=36864,
            n_trans=29,
            image_shape=(256, 256),
            volume_shape=(256, 256, 256),
            padding_factor=2,
            n_classes=1,
            gpu_memory_gb=80.0,
            current_size=100,
        )


def test_relion_em_batch_sizing_clamps_highres_translation_tiles():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=1024,
        n_rot=1024,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 9
    assert plan.rotation_block_size == 13
    assert plan.pose_pixel_tile_gb < 3.0
    assert plan.translation_tile_gb <= plan.translation_tile_budget_gb


def test_relion_em_batch_sizing_accounts_for_k_classes():
    single = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=294912,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )
    k4 = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=294912,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=4,
        gpu_memory_gb=80.0,
    )

    assert k4.image_batch_size <= single.image_batch_size
    assert k4.rotation_block_size <= single.rotation_block_size


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

    buckets = bucket_local_hypothesis_layout(
        layout, image_batch_size=2, rotation_block_size=16, max_hypotheses_per_microbatch=64
    )
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


def test_build_pass2_hypothesis_layout_accepts_fine_translation_log_prior():
    translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    fine_prior = np.arange(8, dtype=np.float32)

    layout = build_pass2_hypothesis_layout(
        [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)],
        n_coarse_rotations=rotation_grid_size(0),
        n_coarse_translations=2,
        nside_level=0,
        translations=translations,
        oversampling_order=1,
        translation_step=2.0,
        fine_translation_log_prior=fine_prior,
    )

    assert layout.translation_grid.shape[0] == fine_prior.shape[0]
    np.testing.assert_allclose(
        layout.translation_log_priors,
        np.broadcast_to(fine_prior[None, :], (2, fine_prior.shape[0])),
    )


def test_build_pass2_hypothesis_layout_rejects_rotation_ids_outside_coarse_grid():
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="outside the coarse grid"):
        build_pass2_hypothesis_layout(
            [np.array([4], dtype=np.int32)],  # rot 2/trans 0 for a 2-rotation grid
            n_coarse_rotations=2,
            n_coarse_translations=2,
            nside_level=0,
            translations=translations,
            oversampling_order=0,
        )

    with pytest.raises(ValueError, match="outside the coarse grid"):
        build_pass2_hypothesis_layout(
            [np.array([-1], dtype=np.int32)],
            n_coarse_rotations=2,
            n_coarse_translations=2,
            nside_level=0,
            translations=translations,
            oversampling_order=0,
        )


def test_build_pass2_hypothesis_layout_can_keep_empty_class_support():
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    layout = build_pass2_hypothesis_layout(
        [np.array([], dtype=np.int32)],
        n_coarse_rotations=rotation_grid_size(0),
        n_coarse_translations=2,
        nside_level=0,
        translations=translations,
        oversampling_order=0,
        allow_empty=True,
    )

    assert layout.n_images == 1
    np.testing.assert_array_equal(layout.rotation_counts, np.array([1], dtype=np.int32))
    assert layout.sample_mask_flat.shape == (1, 2)
    assert not np.any(layout.sample_mask_flat)


def test_local_id_lookup_matches_dense_table_duplicate_semantics():
    ids = np.array([4, 2, 4, 9], dtype=np.int32)
    values = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    selected, matched = local_layout_module._lookup_values_by_id(ids, values, np.array([2, 4, 9], dtype=np.int32))

    np.testing.assert_array_equal(matched, np.array([True, True, True]))
    np.testing.assert_allclose(selected, np.array([0.5, 0.75, 1.0], dtype=np.float32))
    assert selected.dtype == np.float32


def test_build_local_adaptive_pass2_hypothesis_layout_masks_parent_pairs():
    parent_order = 0
    oversampling_order = 1
    coarse_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_order),
        n_pixels=12,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 3], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 1], dtype=np.int32),
        rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
        rotation_log_priors_flat=np.array([0.0, -2.0, -3.0], dtype=np.float32),
        rotation_counts=np.array([2, 1], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.array([[0.0, -1.0], [-4.0, -5.0]], dtype=np.float32),
    )
    significant_samples = [
        np.array([0, 3], dtype=np.int32),  # rot 0/trans 0 and rot 1/trans 1
        np.array([2], dtype=np.int32),  # rot 1/trans 0
    ]

    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        significant_samples,
        parent_order,
        oversampling_order=oversampling_order,
        translation_step=2.0,
    )

    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        2.0,
        oversampling_order=oversampling_order,
    )
    np.testing.assert_allclose(layout.translation_grid, fine_translations, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        layout.translation_log_priors,
        parent_layout.translation_log_priors[:, fine_translation_parent],
    )
    assert layout.n_global_rotations == rotation_grid_size(parent_order)
    np.testing.assert_array_equal(layout.rotation_counts, np.array([16, 8], dtype=np.int32))

    child_rots0, parent_map0, child_ids0 = get_oversampled_rotation_grid_from_samples(
        np.array([0, 1], dtype=np.int32),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    np.testing.assert_array_equal(layout.rotation_ids_flat[:16], child_ids0.astype(np.int32))
    np.testing.assert_array_equal(layout.rotation_posterior_ids_flat[:16], np.array([0, 1], dtype=np.int32)[parent_map0])
    np.testing.assert_allclose(layout.rotations_flat[:16], child_rots0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        layout.rotation_log_priors_flat[:16],
        np.array([0.0, -2.0], dtype=np.float32)[parent_map0],
    )
    expected0 = np.zeros((16, fine_translation_parent.size), dtype=bool)
    expected0[parent_map0 == 0] = fine_translation_parent == 0
    expected0[parent_map0 == 1] = fine_translation_parent == 1
    np.testing.assert_array_equal(layout.sample_mask_flat[:16], expected0)

    _, parent_map1, child_ids1 = get_oversampled_rotation_grid_from_samples(
        np.array([1], dtype=np.int32),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    np.testing.assert_array_equal(layout.rotation_ids_flat[16:], child_ids1.astype(np.int32))
    np.testing.assert_array_equal(layout.rotation_posterior_ids_flat[16:], np.full(parent_map1.shape, 1, dtype=np.int32))
    np.testing.assert_allclose(layout.rotation_log_priors_flat[16:], np.full(parent_map1.shape, -3.0))
    np.testing.assert_array_equal(
        layout.sample_mask_flat[16:],
        np.broadcast_to(fine_translation_parent[None, :] == 0, layout.sample_mask_flat[16:].shape),
    )


def test_build_local_adaptive_pass2_hypothesis_layout_empty_significant_samples_fallback_to_local_support():
    parent_order = 0
    oversampling_order = 1
    coarse_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_order),
        n_pixels=12,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        rotation_log_priors_flat=np.array([0.0, -2.0], dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.array([[0.0, -1.0]], dtype=np.float32),
    )

    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [np.array([], dtype=np.int64)],
        parent_order,
        oversampling_order=oversampling_order,
        translation_step=2.0,
    )

    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        2.0,
        oversampling_order=oversampling_order,
    )
    child_rots, parent_map, child_ids = get_oversampled_rotation_grid_from_samples(
        np.array([0, 1], dtype=np.int32),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )

    np.testing.assert_allclose(layout.translation_grid, fine_translations, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        layout.translation_log_priors,
        parent_layout.translation_log_priors[:, fine_translation_parent],
    )
    np.testing.assert_array_equal(layout.rotation_counts, np.array([16], dtype=np.int32))
    np.testing.assert_array_equal(layout.rotation_offsets, np.array([0, 16], dtype=np.int64))
    np.testing.assert_array_equal(layout.rotation_ids_flat, child_ids.astype(np.int32))
    np.testing.assert_array_equal(layout.rotation_posterior_ids_flat, np.array([0, 1], dtype=np.int32)[parent_map])
    np.testing.assert_allclose(layout.rotations_flat, child_rots, rtol=1e-6, atol=1e-6)
    assert layout.sample_mask_flat is None
    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=1,
        rotation_block_size=8,
        max_hypotheses_per_microbatch=128,
    )
    assert buckets
    assert all(bucket.local_sample_mask is None for bucket in buckets)


def test_build_local_adaptive_pass2_hypothesis_layout_none_uses_full_parent_support():
    parent_order = 0
    coarse_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_order),
        n_pixels=12,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        rotation_log_priors_flat=np.array([0.0, -2.0], dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.array([[0.0, -1.0]], dtype=np.float32),
    )

    layout_none = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [None],
        parent_order,
        oversampling_order=1,
        translation_step=2.0,
    )
    layout_empty = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [np.zeros(0, dtype=np.int64)],
        parent_order,
        oversampling_order=1,
        translation_step=2.0,
    )

    np.testing.assert_array_equal(layout_none.rotation_offsets, layout_empty.rotation_offsets)
    np.testing.assert_array_equal(layout_none.rotation_counts, layout_empty.rotation_counts)
    np.testing.assert_array_equal(layout_none.rotation_ids_flat, layout_empty.rotation_ids_flat)
    np.testing.assert_array_equal(layout_none.rotation_posterior_ids_flat, layout_empty.rotation_posterior_ids_flat)
    assert layout_none.sample_mask_flat is None
    assert layout_empty.sample_mask_flat is None


def test_expand_significant_samples_to_full_parent_translations_preserves_rotation_support():
    samples = [
        np.array([0, 3, 4], dtype=np.int64),  # rotations 0 and 1 with n_trans=3
        None,
        np.zeros(0, dtype=np.int64),
    ]

    expanded = iteration_loop_module._expand_significant_samples_to_full_parent_translations(
        samples,
        n_parent_translations=3,
    )

    np.testing.assert_array_equal(expanded[0], np.array([0, 1, 2, 3, 4, 5], dtype=np.int64))
    assert expanded[1] is None
    np.testing.assert_array_equal(expanded[2], np.zeros(0, dtype=np.int64))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("rotation_only", "rotation_only"),
        ("rotation", "rotation_only"),
        ("full_parent", "full_parent"),
        ("1", None),
        ("default", None),
    ],
)
def test_local_adaptive_pass2_denominator_support_mode(monkeypatch, value, expected):
    monkeypatch.setenv("RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT", value)

    assert iteration_loop_module._local_adaptive_pass2_denominator_support_mode() == expected


def test_build_local_adaptive_pass2_hypothesis_layout_accepts_int64_packed_samples():
    parent_order = 7
    oversampling_order = 1
    n_coarse_trans = 49
    parent_id = np.int64(50_000_000)
    coarse_trans = np.int64(17)
    significant_sample = parent_id * np.int64(n_coarse_trans) + coarse_trans
    coarse_translations = np.stack(
        [np.arange(n_coarse_trans, dtype=np.float32), np.zeros(n_coarse_trans, dtype=np.float32)],
        axis=1,
    )
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_order),
        n_pixels=12 * (2**parent_order) ** 2,
        n_psi=rotation_grid_n_in_planes(parent_order),
        rotation_offsets=np.array([0, 1], dtype=np.int64),
        rotation_ids_flat=np.array([parent_id], dtype=np.int32),
        rotations_flat=np.eye(3, dtype=np.float32)[None],
        rotation_log_priors_flat=np.array([-2.5], dtype=np.float32),
        rotation_counts=np.array([1], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.zeros((1, n_coarse_trans), dtype=np.float32),
    )

    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [np.array([significant_sample], dtype=np.int64)],
        parent_order,
        oversampling_order=oversampling_order,
        translation_step=1.0,
    )

    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        1.0,
        oversampling_order=oversampling_order,
    )
    _, parent_map, child_ids = get_oversampled_rotation_grid_from_samples(
        np.array([parent_id], dtype=np.int64),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    np.testing.assert_allclose(layout.translation_grid, fine_translations, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(layout.rotation_ids_flat, child_ids.astype(np.int32))
    np.testing.assert_allclose(layout.rotation_log_priors_flat, np.full(parent_map.shape, -2.5, dtype=np.float32))
    expected_mask = np.broadcast_to(
        fine_translation_parent[None, :] == int(coarse_trans),
        layout.sample_mask_flat.shape,
    )
    np.testing.assert_array_equal(layout.sample_mask_flat, expected_mask)


def test_build_pass2_hypothesis_layout_accepts_int64_packed_samples():
    parent_order = 7
    oversampling_order = 1
    n_coarse_trans = 49
    parent_id = np.int64(50_000_000)
    coarse_trans = np.int64(17)
    significant_sample = parent_id * np.int64(n_coarse_trans) + coarse_trans
    coarse_translations = np.stack(
        [np.arange(n_coarse_trans, dtype=np.float32), np.zeros(n_coarse_trans, dtype=np.float32)],
        axis=1,
    )

    layout = build_pass2_hypothesis_layout(
        [np.array([significant_sample], dtype=np.int64)],
        rotation_grid_size(parent_order),
        n_coarse_trans,
        parent_order,
        coarse_translations,
        oversampling_order=oversampling_order,
        translation_step=1.0,
    )

    _, parent_map, child_ids = get_oversampled_rotation_grid_from_samples(
        np.array([parent_id], dtype=np.int64),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    _, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        1.0,
        oversampling_order=oversampling_order,
    )
    np.testing.assert_array_equal(layout.rotation_ids_flat, child_ids.astype(np.int32))
    np.testing.assert_array_equal(layout.rotation_posterior_ids_flat, np.full(parent_map.shape, parent_id, dtype=np.int32))
    expected_mask = np.broadcast_to(
        fine_translation_parent[None, :] == int(coarse_trans),
        layout.sample_mask_flat.shape,
    )
    np.testing.assert_array_equal(layout.sample_mask_flat, expected_mask)


def test_pass2_layout_builders_do_not_allocate_global_rotation_lookup_tables(monkeypatch):
    n_global_rotations = 1_000_000
    original_full = local_layout_module.np.full

    def guarded_full(shape, *args, **kwargs):
        if shape == n_global_rotations or shape == (n_global_rotations,):
            raise AssertionError("pass-2 layout builder allocated a dense global rotation lookup table")
        return original_full(shape, *args, **kwargs)

    monkeypatch.setattr(local_layout_module.np, "full", guarded_full)

    parent_order = 0
    coarse_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=n_global_rotations,
        n_pixels=12,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        rotation_log_priors_flat=np.array([0.0, -2.0], dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.array([[0.0, -1.0]], dtype=np.float32),
    )
    adaptive_layout = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [np.array([0, 3], dtype=np.int32)],
        parent_order,
        oversampling_order=1,
        translation_step=2.0,
    )
    _, parent_map, _ = get_oversampled_rotation_grid_from_samples(
        np.array([0, 1], dtype=np.int32),
        parent_order,
        oversampling_order=1,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    np.testing.assert_allclose(
        adaptive_layout.rotation_log_priors_flat,
        np.array([0.0, -2.0], dtype=np.float32)[parent_map],
    )
    assert adaptive_layout.sample_mask_flat.shape == (16, 8)

    pass2_layout = build_pass2_hypothesis_layout(
        [np.array([0, 3], dtype=np.int32)],
        n_coarse_rotations=n_global_rotations,
        n_coarse_translations=2,
        nside_level=0,
        translations=coarse_translations,
        oversampling_order=1,
        translation_step=2.0,
        rotation_log_prior=np.arange(rotation_grid_size(parent_order), dtype=np.float32),
    )
    np.testing.assert_array_equal(pass2_layout.rotation_posterior_ids_flat, np.array([0, 1], dtype=np.int32)[parent_map])
    np.testing.assert_array_equal(pass2_layout.sample_mask_flat, adaptive_layout.sample_mask_flat)


def test_normalize_local_scores_zeroes_all_invalid_rows():
    scores = jnp.asarray(
        [
            [[2.0, -1.0], [-jnp.inf, 0.0]],
            [[-jnp.inf, -jnp.inf], [-jnp.inf, -jnp.inf]],
        ],
        dtype=jnp.float32,
    )

    for normalize in (normalize_local_scores, normalize_local_scores_float32):
        log_z, probs, best_log_score, best_argmax, max_posterior = normalize(scores)
        probs_np = np.asarray(probs)
        np.testing.assert_allclose(np.sum(probs_np[0]), 1.0, rtol=1e-6, atol=1e-6)
        assert np.isneginf(np.asarray(log_z)[1])
        assert np.isneginf(np.asarray(best_log_score)[1])
        assert int(np.asarray(best_argmax)[1]) == 0
        assert float(np.asarray(max_posterior)[1]) == 0.0
        np.testing.assert_allclose(probs_np[1], 0.0)
        assert not np.any(np.isnan(probs_np))

    external_log_z = jnp.asarray([3.0, 4.0], dtype=jnp.float32)
    for normalize in (normalize_local_scores_with_log_z, normalize_local_scores_with_log_z_float32):
        log_z, probs, best_log_score, best_argmax, max_posterior = normalize(scores, external_log_z)
        assert np.isfinite(np.asarray(log_z)[1])
        assert np.isneginf(np.asarray(best_log_score)[1])
        assert int(np.asarray(best_argmax)[1]) == 0
        assert float(np.asarray(max_posterior)[1]) == 0.0
        np.testing.assert_allclose(np.asarray(probs)[1], 0.0)
        assert not np.any(np.isnan(np.asarray(probs)))

    support_mask, support_rotations, n_significant = compute_reconstruction_support(probs)
    assert not np.any(np.asarray(support_mask)[1])
    assert not np.any(np.asarray(support_rotations)[1])
    assert int(np.asarray(n_significant)[1]) == 0


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


def test_build_local_hypothesis_layout_parent_expands_relion_coarse_support():
    parent_order = 2
    fine_order = 3
    oversampling_order = fine_order - parent_order
    parent_metadata = build_local_search_grid_metadata(parent_order)
    fine_metadata = build_local_search_grid_metadata(fine_order)
    prior_eulers = np.array(
        [
            [12.0, 40.0, 3.0],
            [91.0, 65.0, 29.0],
        ],
        dtype=np.float32,
    )
    translations = np.zeros((3, 2), dtype=np.float32)
    prior_translations = np.zeros((prior_eulers.shape[0], 2), dtype=np.float32)
    sigma_rot = np.deg2rad(7.5)
    sigma_psi = np.deg2rad(7.5)

    parent_layout = build_local_hypothesis_layout(
        prior_eulers,
        None,
        sigma_rot=sigma_rot,
        sigma_psi=sigma_psi,
        healpix_order=parent_order,
        translations=translations,
        prior_translations=prior_translations,
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata=parent_metadata,
    )
    expanded_layout = build_local_hypothesis_layout(
        prior_eulers,
        None,
        sigma_rot=sigma_rot,
        sigma_psi=sigma_psi,
        healpix_order=fine_order,
        translations=translations,
        prior_translations=prior_translations,
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata=fine_metadata,
        local_parent_oversampling_order=oversampling_order,
    )

    assert expanded_layout.n_global_rotations == rotation_grid_size(fine_order)
    np.testing.assert_array_equal(expanded_layout.rotation_counts, parent_layout.rotation_counts * 8)

    for image_idx in range(prior_eulers.shape[0]):
        p0, p1 = parent_layout.rotation_offsets[image_idx : image_idx + 2]
        c0, c1 = expanded_layout.rotation_offsets[image_idx : image_idx + 2]
        parent_ids = parent_layout.rotation_ids_flat[p0:p1]
        parent_log_prior = parent_layout.rotation_log_priors_flat[p0:p1]
        child_rots, parent_map, child_ids = get_oversampled_rotation_grid_from_samples(
            parent_ids,
            parent_order,
            oversampling_order=oversampling_order,
            return_rotation_indices=True,
            rotation_index_order="recovar",
        )

        np.testing.assert_array_equal(expanded_layout.rotation_ids_flat[c0:c1], child_ids.astype(np.int32))
        np.testing.assert_allclose(expanded_layout.rotation_log_priors_flat[c0:c1], parent_log_prior[parent_map])
        np.testing.assert_allclose(expanded_layout.rotations_flat[c0:c1], child_rots, rtol=1e-6, atol=1e-6)


def test_score_half_local_parent_layout_ignores_global_rotation_prior_for_adaptive_pass2(monkeypatch, rng):
    dataset = MockDataset(2, rng)
    captured = {}
    rotation_log_prior = np.linspace(-0.75, 0.25, rotation_grid_size(0), dtype=np.float32)

    class StopAfterParentLayout(Exception):
        pass

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
        rotation_log_prior=None,
        rotation_grid_random_perturbation=0.0,
        rotation_grid_angular_sampling_deg=None,
    ):
        _ = (
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
            grid_metadata,
            translation_prior_reference_translations,
            rotation_grid_random_perturbation,
            rotation_grid_angular_sampling_deg,
        )
        captured["rotation_log_prior"] = (
            None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=np.float32).copy()
        )
        raise StopAfterParentLayout

    monkeypatch.setattr(iteration_loop_module, "build_local_hypothesis_layout", fake_build_local_hypothesis_layout)

    with pytest.raises(StopAfterParentLayout):
        iteration_loop_module._score_half_local(
            k=0,
            experiment_dataset=dataset,
            means_k=jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
            mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            noise_variance_k=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            previous_best_rotation_eulers_k=np.zeros((dataset.n_units, 3), dtype=np.float32),
            local_search_rotations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
            local_search_rotation_eulers=np.zeros((2, 3), dtype=np.float32),
            local_search_order=1,
            sigma_rot=np.deg2rad(1.0),
            sigma_psi=np.deg2rad(1.0),
            current_translations=np.zeros((1, 2), dtype=np.float32),
            base_translations=np.zeros((1, 2), dtype=np.float32),
            trans_prior_center=np.zeros((dataset.n_units, 2), dtype=np.float32),
            trans_prior_center_for_engine=np.zeros((dataset.n_units, 2), dtype=np.float32),
            current_sigma_offset_angstrom=1.0,
            current_translation_range=1.0,
            disc_type="linear_interp",
            cs_for_engine=None,
            local_pass1_current_size=4,
            image_corrections_k=None,
            scale_corrections_k=None,
            translation_search_base=None,
            disable_adjoint_y=False,
            disable_adjoint_ctf=False,
            max_significants=None,
            state=type("State", (), {"adaptive_oversampling": 1})(),
            iteration=3,
            save_intermediates_dir=None,
            local_search_random_perturbation=0.0,
            local_search_angular_sampling_deg=relion_angular_sampling_deg(1),
            local_parent_oversampling_order=1,
            diagnostic_score_only=False,
            local_search_translation_prior_mode="coarse",
            replay_prior_translations=None,
            rotation_log_prior_k=rotation_log_prior,
            class_log_priors=None,
            k_class_enabled=False,
            collect_local_search_profile=False,
            safe_batch_sizes=lambda *args, **kwargs: (1, 1),
            class_assignments=[None],
            class_posterior_per_half=[None],
            class_full_posterior_per_half=[None],
            best_pose_rotations=[None],
            best_pose_rotation_eulers=[None],
            best_pose_translations=[None],
            local_profile_history=[],
        )

    assert captured["rotation_log_prior"] is None


def test_build_local_hypothesis_layout_parent_expands_translation_grid_and_priors():
    parent_order = 1
    fine_order = 2
    oversampling_order = fine_order - parent_order
    grid_metadata = build_local_search_grid_metadata(fine_order)
    prior_eulers = np.array(
        [
            [12.0, 40.0, 3.0],
            [91.0, 65.0, 29.0],
        ],
        dtype=np.float32,
    )
    translations = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    prior_translations = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )
    reference_translations = np.array(
        [
            [0.25, 0.25],
            [2.25, 0.25],
            [0.25, 2.25],
        ],
        dtype=np.float32,
    )

    layout = build_local_hypothesis_layout(
        prior_eulers,
        None,
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=fine_order,
        translations=translations,
        prior_translations=prior_translations,
        sigma_offset_angstrom=1.25,
        offset_range_pixels=None,
        voxel_size=1.0,
        grid_metadata=grid_metadata,
        translation_prior_reference_translations=reference_translations,
        local_parent_oversampling_order=oversampling_order,
    )

    fine_translations, translation_parent = get_oversampled_translation_grid(
        translations,
        2.0,
        oversampling_order=oversampling_order,
    )
    coarse_prior = make_relion_translation_log_prior(
        reference_translations,
        1.0,
        1.25,
        prior_translations,
        offset_range_pixels=None,
    )

    np.testing.assert_allclose(layout.translation_grid, fine_translations, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        layout.translation_log_priors,
        coarse_prior[:, translation_parent],
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_local_hypothesis_layout_factorized_chunking_preserves_support(monkeypatch):
    healpix_order = 3
    grid_metadata = build_local_search_grid_metadata(healpix_order)
    rotation_grid = get_relion_rotation_grid(healpix_order).astype(np.float32)
    prior_eulers = np.array(
        [
            [12.0, 40.0, 3.0],
            [91.0, 65.0, 29.0],
            [177.0, 23.0, 144.0],
            [240.0, 81.0, 271.0],
        ],
        dtype=np.float32,
    )
    kwargs = dict(
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=healpix_order,
        translations=np.zeros((9, 2), dtype=np.float32),
        prior_translations=np.zeros((prior_eulers.shape[0], 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata=grid_metadata,
    )

    monkeypatch.delenv("RECOVAR_LOCAL_SELECTOR_CHUNK_SIZE", raising=False)
    full_layout = build_local_hypothesis_layout(prior_eulers, rotation_grid, **kwargs)
    monkeypatch.setenv("RECOVAR_LOCAL_SELECTOR_CHUNK_SIZE", "1")
    chunked_layout = build_local_hypothesis_layout(prior_eulers, rotation_grid, **kwargs)

    np.testing.assert_array_equal(chunked_layout.rotation_offsets, full_layout.rotation_offsets)
    np.testing.assert_array_equal(chunked_layout.rotation_counts, full_layout.rotation_counts)
    np.testing.assert_array_equal(chunked_layout.rotation_ids_flat, full_layout.rotation_ids_flat)
    np.testing.assert_allclose(chunked_layout.rotation_log_priors_flat, full_layout.rotation_log_priors_flat)


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


def test_exact_local_fine_grid_precompute_auto_policy():
    assert iteration_loop_module._precompute_exact_local_fine_grid_enabled(5)
    assert not iteration_loop_module._precompute_exact_local_fine_grid_enabled(6)


def test_bucket_local_hypothesis_layout_coarsens_large_exact_neighborhoods_without_4096_floor():
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
    assert bucket_sizes == [2048]
    assert [int(bucket.bucket_image_count) for bucket in buckets] == [10]
    np.testing.assert_array_equal(buckets[0].actual_rotation_counts, np.array([1368, 1392, 1416], dtype=np.int32))
    assert buckets[0].local_rotation_mask[0, :1368].all()
    assert not buckets[0].local_rotation_mask[0, 1368:].any()


def test_bucket_local_hypothesis_layout_quantum_env_can_request_finer_tail_shapes(monkeypatch):
    monkeypatch.setenv("RECOVAR_LOCAL_BUCKET_QUANTUM", "128")
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

    assert sorted(int(bucket.bucket_rotation_count) for bucket in buckets) == [1408, 1536]


def test_bucket_local_hypothesis_layout_batches_moderate_local_search_neighborhoods(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_BUCKET_QUANTUM", raising=False)
    n_images = 120
    rotation_counts = np.full(n_images, 198, dtype=np.int32)
    rotation_offsets = np.concatenate([[0], np.cumsum(rotation_counts)]).astype(np.int64)
    n_total = int(rotation_counts.sum())
    layout = LocalHypothesisLayout(
        n_global_rotations=2359296,
        n_pixels=12288,
        n_psi=192,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=np.arange(n_total, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (n_total, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(n_total, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((25, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, 25), dtype=np.float32),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=60,
        rotation_block_size=89,
        max_hypotheses_per_microbatch=1211,
    )

    assert {int(bucket.bucket_rotation_count) for bucket in buckets} == {256}
    assert {int(bucket.image_indices.shape[0]) for bucket in buckets} == {4}
    assert len(buckets) == 30
    assert sum(int(bucket.image_indices.shape[0]) * int(bucket.bucket_rotation_count) for bucket in buckets) == 30720


def test_reconstruction_pack_uses_compact_default_quantum(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.delenv(EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM_ENV, raising=False)
    significant = np.ones((2, 1536), dtype=bool)
    local_mask = np.ones((2, 1536), dtype=bool)

    take_indices, pack_mask, actual_counts, row_count = _build_reconstruction_pack_indices(
        significant,
        local_mask,
        rotation_block_size=19,
    )

    assert take_indices.shape == (2, 1536)
    assert pack_mask.shape == (2, 1536)
    np.testing.assert_array_equal(actual_counts, np.array([1536, 1536], dtype=np.int32))
    assert row_count == 3072


def test_reconstruction_pack_quantum_env_can_coarsen(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.setenv(EXACT_LOCAL_RECONSTRUCTION_PACK_QUANTUM_ENV, "2048")
    significant = np.ones((2, 1536), dtype=bool)
    local_mask = np.ones((2, 1536), dtype=bool)

    take_indices, pack_mask, actual_counts, row_count = _build_reconstruction_pack_indices(
        significant,
        local_mask,
        rotation_block_size=19,
    )

    assert take_indices.shape == (2, 2048)
    assert pack_mask.shape == (2, 2048)
    np.testing.assert_array_equal(actual_counts, np.array([1536, 1536], dtype=np.int32))
    assert row_count == 3072


def test_bucket_local_hypothesis_layout_unify_env_collapses_shape_classes(monkeypatch):
    """RECOVAR_LOCAL_BUCKET_UNIFY=1 collapses ~13 unique bucket shape classes
    into one max-sized class so the JIT only compiles one shape per layout.
    Pins the 7.3× perf win measured on 50k/256 K=1 (commit 8e868d5e)."""

    rotation_counts = np.array([16, 16, 128, 256, 512, 1024, 1280, 1408], dtype=np.int32)
    rotation_ids = np.arange(int(rotation_counts.sum()), dtype=np.int32)
    rotation_offsets = np.concatenate([[0], np.cumsum(rotation_counts)]).astype(np.int64)
    n_total = int(rotation_counts.sum())
    layout = LocalHypothesisLayout(
        n_global_rotations=2000,
        n_pixels=768,
        n_psi=16,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids,
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (n_total, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(n_total, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((4, 2), dtype=np.float32),
        translation_log_priors=np.zeros((len(rotation_counts), 4), dtype=np.float32),
    )

    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_UNIFY", raising=False)
    default_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
    )
    default_unique_sizes = sorted({int(b.bucket_rotation_count) for b in default_buckets})
    assert len(default_unique_sizes) >= 6  # power-of-2 spread

    monkeypatch.setenv("RECOVAR_LOCAL_BUCKET_UNIFY", "1")
    unified_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
    )
    unified_sizes = {int(b.bucket_rotation_count) for b in unified_buckets}
    assert unified_sizes == {max(default_unique_sizes)}
    assert len(unified_buckets) == 1
    # All images must remain represented exactly once.
    served_indices = np.sort(np.concatenate([b.image_indices for b in unified_buckets]))
    np.testing.assert_array_equal(served_indices, np.arange(len(rotation_counts), dtype=np.int32))


def test_bucket_local_hypothesis_layout_unify_argument_collapses_shape_classes(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_UNIFY", raising=False)
    rotation_counts = np.array([16, 128, 512, 1408], dtype=np.int32)
    rotation_ids = np.arange(int(rotation_counts.sum()), dtype=np.int32)
    rotation_offsets = np.concatenate([[0], np.cumsum(rotation_counts)]).astype(np.int64)
    layout = LocalHypothesisLayout(
        n_global_rotations=2000,
        n_pixels=768,
        n_psi=16,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids,
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (int(rotation_counts.sum()), 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(int(rotation_counts.sum()), dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((4, 2), dtype=np.float32),
        translation_log_priors=np.zeros((len(rotation_counts), 4), dtype=np.float32),
    )

    default_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
        unify_bucket_sizes=False,
    )
    explicit_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
        unify_bucket_sizes=True,
    )

    default_unique_sizes = sorted({int(b.bucket_rotation_count) for b in default_buckets})
    assert len(default_unique_sizes) > 1
    assert {int(b.bucket_rotation_count) for b in explicit_buckets} == {max(default_unique_sizes)}
    np.testing.assert_array_equal(
        np.sort(np.concatenate([b.image_indices for b in explicit_buckets])),
        np.arange(len(rotation_counts), dtype=np.int32),
    )


def test_pad_local_big_jit_image_axis_masks_dummy_rows():
    mstep_rotations = np.broadcast_to(2.0 * np.eye(3, dtype=np.float32), (1, 2, 3, 3)).copy()
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
        local_mstep_rotations=mstep_rotations,
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
    np.testing.assert_array_equal(padded.local_mstep_rotations[0], mstep_rotations[0])
    np.testing.assert_allclose(
        padded.local_mstep_rotations[1:, 0],
        np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)),
    )
    np.testing.assert_allclose(padded_ctf[1:], np.broadcast_to(ctf_params[0], (2, 9)))


def test_project_local_bucket_accepts_singleton_class_relion_projector(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine
    from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec

    bucket = LocalBucketSpec(
        image_indices=np.array([0], dtype=np.int32),
        bucket_image_count=1,
        bucket_rotation_count=2,
        actual_rotation_counts=np.array([2], dtype=np.int32),
        local_rotation_ids=np.array([[0, 1]], dtype=np.int32),
        local_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (1, 2, 3, 3)).copy(),
        local_rotation_log_prior=np.zeros((1, 2), dtype=np.float32),
        local_rotation_mask=np.ones((1, 2), dtype=bool),
        translation_log_prior=np.zeros((1, 1), dtype=np.float32),
    )
    calls = []

    def fake_projector(projector_half, rotations, image_shape, **kwargs):
        calls.append((np.asarray(projector_half).shape, tuple(rotations.shape), dict(kwargs)))
        return jnp.ones((rotations.shape[0], image_shape[0] * (image_shape[0] // 2 + 1)), dtype=jnp.complex64), None

    monkeypatch.setattr(local_em_engine, "_compute_relion_projector_projections_block", fake_projector)

    block = local_em_engine._project_local_bucket(
        mean_for_proj=jnp.zeros((4, 4, 4), dtype=jnp.complex64),
        bucket=bucket,
        image_shape=(4, 4),
        proj_volume_shape=(4, 4, 4),
        disc_type="linear_interp",
        projection_kwargs={"relion_texture_interp": False},
        window_spec=make_fourier_window_spec((4, 4), 4, 12, include_recon_window=True),
        n_half=12,
        half_weights=jnp.ones(12, dtype=jnp.float32),
        precision_policy=DensePrecisionPolicy(use_float64_scoring=False),
        relion_projector_half=jnp.ones((1, 4, 4, 3), dtype=jnp.complex64),
        relion_projector_r_max=2,
        projection_padding_factor=1,
    )

    assert calls == [
        (
            (4, 4, 3),
            (2, 3, 3),
            {
                "r_max": 2,
                "padding_factor": 1,
                "return_abs2": False,
                "centered_rows": True,
                "dense_scale": True,
                "relion_texture_interp": False,
            },
        )
    ]
    assert block.proj_weighted.shape == (1, 2, 12)
    assert block.proj_for_noise.shape == (1, 2, 12)


def test_relion_projector_indexed_centered_rows_match_full_window(rng):
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
    from recovar.em.dense_single_volume.helpers.projection import compute_relion_projector_projections_block

    image_shape = (8, 8)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    window_spec = make_fourier_window_spec(image_shape, 6, n_half, include_recon_window=True)
    projector_half = (
        rng.standard_normal((8, 8, 5)) + 1j * rng.standard_normal((8, 8, 5))
    ).astype(np.complex64)
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (3, 3, 3))

    full, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector_half),
        rotations,
        image_shape,
        r_max=4,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=6,
    )
    indexed, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector_half),
        rotations,
        image_shape,
        r_max=4,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=6,
        pixel_indices=window_spec.projection_indices,
    )

    np.testing.assert_allclose(
        np.asarray(indexed),
        np.asarray(full)[:, window_spec.projection_indices_np],
        atol=1e-5,
        rtol=1e-5,
    )


def test_relion_projector_texture_full_embeds_positive_x_half():
    from recovar.em.dense_single_volume.helpers.projection import relion_projector_half_to_texture_full

    projector_half = (
        np.arange(5 * 5 * 3, dtype=np.float32).reshape(5, 5, 3)
        + 1j * np.arange(5 * 5 * 3, dtype=np.float32).reshape(5, 5, 3)[::-1]
    ).astype(np.complex64)
    full = np.asarray(relion_projector_half_to_texture_full(jnp.asarray(projector_half)))

    assert full.shape == (5, 5, 5)
    np.testing.assert_array_equal(full[2:], np.transpose(projector_half, (2, 1, 0)))
    np.testing.assert_array_equal(full[:2], np.zeros((2, 5, 5), dtype=np.complex64))


def test_relion_projector_texture_route_defaults_on_and_can_be_disabled(monkeypatch):
    from recovar.em.dense_single_volume.helpers import projection as projection_helpers

    projector_half = jnp.ones((5, 5, 3), dtype=jnp.complex64)
    rotations = jnp.eye(3, dtype=jnp.float32)[None]
    calls = []

    monkeypatch.delenv("RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP", raising=False)
    monkeypatch.setattr(projection_helpers, "_cuda_projection_available", lambda: True)
    assert projection_helpers._relion_projector_texture_enabled(
        projector_half,
        r_max=1,
        padding_factor=1,
    )
    monkeypatch.setenv("RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP", "1")

    def fake_texture(projector, rotations_block, image_shape, **kwargs):
        calls.append((tuple(projector.shape), tuple(rotations_block.shape), tuple(image_shape), dict(kwargs)))
        return jnp.full((rotations_block.shape[0], 4 * 3), 2.0 + 1.0j, dtype=jnp.complex64)

    monkeypatch.setattr(projection_helpers, "_project_relion_projector_texture", fake_texture)
    got, _ = projection_helpers.compute_relion_projector_projections_block(
        projector_half,
        rotations,
        (4, 4),
        r_max=1,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=2,
    )
    np.testing.assert_array_equal(np.asarray(got), np.full((1, 12), 2.0 + 1.0j, dtype=np.complex64))
    assert calls == [
        (
            (5, 5, 3),
            (1, 3, 3),
            (4, 4),
            {"r_max": 1, "projector_output_size": 2},
        )
    ]

    monkeypatch.setenv("RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP", "0")
    monkeypatch.setattr(
        projection_helpers,
        "project_relion_projector_half_spectrum_centered_rows",
        lambda *args, **kwargs: jnp.full((1, 12), 7.0 + 0.0j, dtype=jnp.complex64),
    )
    fallback, _ = projection_helpers.compute_relion_projector_projections_block(
        projector_half,
        rotations,
        (4, 4),
        r_max=1,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=2,
    )
    np.testing.assert_array_equal(np.asarray(fallback), np.full((1, 12), 7.0 + 0.0j, dtype=np.complex64))
    assert len(calls) == 1

    monkeypatch.setenv("RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP", "1")
    explicit_fallback, _ = projection_helpers.compute_relion_projector_projections_block(
        projector_half,
        rotations,
        (4, 4),
        r_max=1,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=2,
        relion_texture_interp=False,
    )
    np.testing.assert_array_equal(
        np.asarray(explicit_fallback),
        np.full((1, 12), 7.0 + 0.0j, dtype=np.complex64),
    )
    assert len(calls) == 1


def test_global_pass1_relion_projector_texture_defaults_to_texture(monkeypatch):
    from recovar.em.dense_single_volume.helpers import significance

    monkeypatch.delenv("RECOVAR_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP", raising=False)
    monkeypatch.setenv("RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP", "1")
    assert significance._global_pass1_relion_projector_texture_enabled()

    monkeypatch.setenv("RECOVAR_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP", "1")
    assert significance._global_pass1_relion_projector_texture_enabled()

    monkeypatch.setenv("RECOVAR_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP", "invalid")
    with pytest.raises(ValueError, match="RECOVAR_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP"):
        significance._global_pass1_relion_projector_texture_enabled()


def test_texture_centered_crop_masks_current_image_disk():
    from recovar.em.dense_single_volume.helpers.projection import _texture_centered_crop_to_full

    crop = jnp.ones((1, 4 * 3), dtype=jnp.complex64)
    got = np.asarray(
        _texture_centered_crop_to_full(
            crop,
            image_shape=(4, 4),
            projector_output_size=4,
        )
    ).reshape(1, 4, 3)
    expected = np.ones((1, 4, 3), dtype=np.complex64)
    expected[:, 0, 1:] = 0.0
    expected[:, 1, 2] = 0.0
    expected[:, 3, 2] = 0.0
    np.testing.assert_array_equal(got, expected)


def test_local_big_jit_relion_projector_matches_helper(rng):
    from recovar.em.dense_single_volume.helpers.projection import compute_relion_projector_projections_block
    from recovar.em.dense_single_volume.local_big_jit import _project_local_half_spectrum

    image_shape = (8, 8)
    projector_half = (
        rng.standard_normal((8, 8, 5)) + 1j * rng.standard_normal((8, 8, 5))
    ).astype(np.complex64)
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (3, 3, 3))

    got = _project_local_half_spectrum(
        jnp.zeros((1,), dtype=jnp.complex64),
        jnp.asarray(projector_half),
        rotations,
        None,
        image_shape,
        (8, 8, 8),
        "linear_interp",
        projection_half_volume=False,
        projection_max_r="auto",
        relion_projector_output_size=0,
        projection_relion_texture_interp=True,
        projection_force_jax=False,
        use_relion_projector=True,
        relion_projector_r_max=4,
        projection_padding_factor=1,
    )
    expected, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector_half),
        rotations,
        image_shape,
        r_max=4,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_project_local_bucket_windowed_relion_projector_uses_compact_indices(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine
    from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec

    bucket = LocalBucketSpec(
        image_indices=np.array([0], dtype=np.int32),
        bucket_image_count=1,
        bucket_rotation_count=2,
        actual_rotation_counts=np.array([2], dtype=np.int32),
        local_rotation_ids=np.array([[0, 1]], dtype=np.int32),
        local_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (1, 2, 3, 3)).copy(),
        local_rotation_log_prior=np.zeros((1, 2), dtype=np.float32),
        local_rotation_mask=np.ones((1, 2), dtype=bool),
        translation_log_prior=np.zeros((1, 1), dtype=np.float32),
    )
    calls = []

    def fake_projector(projector_half, rotations, image_shape, **kwargs):
        pixel_indices = kwargs.get("pixel_indices")
        calls.append(
            (
                np.asarray(projector_half).shape,
                tuple(rotations.shape),
                None if pixel_indices is None else int(np.asarray(pixel_indices).shape[0]),
                kwargs.get("projector_output_size"),
                kwargs.get("relion_texture_interp"),
            )
        )
        n_values = int(np.asarray(pixel_indices).shape[0])
        values = jnp.arange(rotations.shape[0] * n_values, dtype=jnp.float32).reshape(rotations.shape[0], n_values)
        return values.astype(jnp.complex64), None

    monkeypatch.setattr(local_em_engine, "_compute_relion_projector_projections_block", fake_projector)
    window_spec = make_fourier_window_spec((8, 8), 6, 40, include_recon_window=True)

    block = local_em_engine._project_local_bucket(
        mean_for_proj=jnp.zeros((8, 8, 8), dtype=jnp.complex64),
        bucket=bucket,
        image_shape=(8, 8),
        proj_volume_shape=(8, 8, 8),
        disc_type="linear_interp",
        projection_kwargs={"relion_texture_interp": True},
        window_spec=window_spec,
        n_half=40,
        half_weights=jnp.ones(40, dtype=jnp.float32),
        precision_policy=DensePrecisionPolicy(use_float64_scoring=False),
        relion_projector_half=jnp.ones((1, 8, 8, 5), dtype=jnp.complex64),
        relion_projector_r_max=4,
        projection_padding_factor=1,
    )

    assert calls == [((8, 8, 5), (2, 3, 3), window_spec.n_projection, 6, True)]
    assert block.proj_weighted.shape == (1, 2, window_spec.n_score)
    assert block.proj_for_noise.shape == (1, 2, window_spec.n_recon)


def test_packed_local_noise_projection_accepts_relion_projector(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine
    from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec

    calls = []

    def fake_projector(projector_half, rotations, image_shape, **kwargs):
        calls.append((np.asarray(projector_half).shape, tuple(rotations.shape), dict(kwargs)))
        n_half = image_shape[0] * (image_shape[0] // 2 + 1)
        values = jnp.arange(rotations.shape[0] * n_half, dtype=jnp.float32).reshape(rotations.shape[0], n_half)
        return values.astype(jnp.complex64), None

    monkeypatch.setattr(local_em_engine, "_compute_relion_projector_projections_block", fake_projector)
    window_spec = make_fourier_window_spec((4, 4), 4, 12, include_recon_window=True)

    packed = local_em_engine._project_packed_noise_rows(
        mean_for_proj=jnp.zeros((4, 4, 4), dtype=jnp.complex64),
        packed_flat_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 3, 3)),
        packed_rotation_count=2,
        batch_size=1,
        image_shape=(4, 4),
        proj_volume_shape=(4, 4, 4),
        disc_type="linear_interp",
        projection_kwargs={"relion_texture_interp": True},
        window_spec=window_spec,
        n_half=12,
        precision_policy=DensePrecisionPolicy(use_float64_scoring=False),
        reconstruction_pack_mask_jnp=jnp.array([[True, False]]),
        relion_projector_half=jnp.ones((1, 4, 4, 3), dtype=jnp.complex64),
        relion_projector_r_max=2,
        projection_padding_factor=1,
    )

    assert calls == [
        (
            (4, 4, 3),
            (2, 3, 3),
            {
                "r_max": 2,
                "padding_factor": 1,
                "return_abs2": False,
                "centered_rows": True,
                "dense_scale": True,
                "relion_texture_interp": True,
            },
        )
    ]
    assert packed.shape == (1, 2, 12)
    np.testing.assert_allclose(np.asarray(packed[0, 1]), 0.0)


def test_local_relion_projection_cache_forwards_texture_selection(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    bucket = LocalBucketSpec(
        image_indices=np.array([0], dtype=np.int32),
        bucket_image_count=1,
        bucket_rotation_count=2,
        actual_rotation_counts=np.array([2], dtype=np.int32),
        local_rotation_ids=np.array([[0, 1]], dtype=np.int32),
        local_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (1, 2, 3, 3)).copy(),
        local_rotation_log_prior=np.zeros((1, 2), dtype=np.float32),
        local_rotation_mask=np.ones((1, 2), dtype=bool),
        translation_log_prior=np.zeros((1, 1), dtype=np.float32),
    )
    calls = []

    def fake_projector(projector_half, rotations, image_shape, **kwargs):
        calls.append(dict(kwargs))
        return jnp.ones((rotations.shape[0], 12), dtype=jnp.complex64), None

    monkeypatch.setattr(local_em_engine, "_compute_relion_projector_projections_block", fake_projector)
    cache = local_em_engine._build_exact_local_relion_projection_cache_for_buckets(
        [bucket],
        jnp.ones((4, 4, 3), dtype=jnp.complex64),
        image_shape=(4, 4),
        n_projection_pixels=12,
        relion_projector_r_max=2,
        projection_padding_factor=1,
        projection_relion_texture_interp=True,
        projection_pixel_indices=None,
        projector_output_size=4,
        cache_row_capacity=2,
        max_global_rotation_id=1,
        group_index=0,
        n_groups=1,
    )

    assert cache.enabled
    assert calls[0]["relion_texture_interp"] is True


def test_packed_local_noise_projection_chunk_rows_env(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    monkeypatch.delenv(local_em_engine.EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS_ENV, raising=False)
    assert local_em_engine._packed_noise_projection_chunk_rows(12) == (
        local_em_engine.EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS // 12
    )

    monkeypatch.setenv(local_em_engine.EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS_ENV, "25")
    assert local_em_engine._packed_noise_projection_chunk_rows(12) == 2
    assert local_em_engine._packed_noise_projection_chunk_rows(12, batch_size=2) == 1

    monkeypatch.setenv(local_em_engine.EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS_ENV, "not_an_int")
    assert local_em_engine._packed_noise_projection_chunk_rows(12) == (
        local_em_engine.EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS // 12
    )


def test_packed_local_noise_projection_default_cap_is_memory_safe(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    monkeypatch.delenv(local_em_engine.EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS_ENV, raising=False)
    assert local_em_engine._packed_noise_projection_chunk_rows(4096, batch_size=64) <= 256
    assert local_em_engine._packed_noise_projection_chunk_rows(74112, batch_size=64) <= 14


def test_exact_local_progress_env_and_hook(monkeypatch):
    from recovar.em.dense_single_volume import local_em_engine

    monkeypatch.delenv(local_em_engine.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, raising=False)
    assert local_em_engine._optional_nonnegative_int_env(local_em_engine.EXACT_LOCAL_PROGRESS_CHUNKS_ENV) is None

    monkeypatch.setenv(local_em_engine.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, "0")
    assert local_em_engine._optional_nonnegative_int_env(local_em_engine.EXACT_LOCAL_PROGRESS_CHUNKS_ENV) == 0

    monkeypatch.setenv(local_em_engine.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, "25")
    assert local_em_engine._optional_nonnegative_int_env(local_em_engine.EXACT_LOCAL_PROGRESS_CHUNKS_ENV) == 25

    monkeypatch.setenv(local_em_engine.EXACT_LOCAL_PROGRESS_CHUNKS_ENV, "-1")
    with pytest.raises(ValueError, match="non-negative integer"):
        local_em_engine._optional_nonnegative_int_env(local_em_engine.EXACT_LOCAL_PROGRESS_CHUNKS_ENV)

    src = inspect.getsource(local_em_engine.run_local_em_exact)
    assert "Exact local bucket loop start" in src
    assert "Exact local bucket loop %s" in src
    assert "_mark_exact_local_bucket_done(bucket)" in src


def test_exact_local_noise_projection_chunks_packed_tail():
    from recovar.em.dense_single_volume import local_em_engine

    src = inspect.getsource(local_em_engine.run_local_em_exact)
    assert "_packed_noise_projection_chunk_rows" in src
    assert "for chunk_start in range(0, packed_rotation_count, chunk_rows)" in src
    assert "Exact local noise projection chunking" in src


def test_exact_local_cached_noise_projection_chunks_packed_tail():
    from recovar.em.dense_single_volume import local_em_engine

    src = inspect.getsource(local_em_engine.run_local_em_exact)
    marker = "Exact local cached noise projection chunking"
    assert marker in src
    cached_chunk_src = src[src.index(marker) :]
    cached_chunk_src = cached_chunk_src[: cached_chunk_src.index("if return_profile:")]
    assert "for chunk_start in range(0, packed_rotation_count, chunk_rows)" in cached_chunk_src
    assert "chunk_take_indices = reconstruction_take_indices_jnp[:, chunk_start:chunk_stop]" in cached_chunk_src
    assert "jnp.take_along_axis(" in cached_chunk_src


def test_exact_local_relion_projector_noise_projection_materializes_once():
    from recovar.em.dense_single_volume import local_em_engine

    src = inspect.getsource(local_em_engine.run_local_em_exact)
    defer_src = src[src.index("can_defer_local_noise_projection = (") :]
    defer_src = defer_src[: defer_src.index("if accumulate_noise:")]
    assert "relion_projector_half is None" in defer_src
    assert "relion_projector_half is not None or" not in defer_src
    assert "need_local_recon_projection = require_materialized_recon_projection or (" in src
    assert "accumulate_noise and not defer_local_noise_projection" in src
    assert "materialize_recon_projection=need_local_recon_projection" in src
    assert "noise_projection_pixels = int(n_half) if relion_projector_half is not None else int(n_recon_pixels)" in src
    assert "_packed_noise_projection_chunk_rows(noise_projection_pixels, batch_size=batch_size)" in src


def test_relion_projector_cache_reuses_cached_projector_data(monkeypatch, tmp_path):
    import recovar.em.initial_model.dense_adapter as dense_adapter

    calls = []

    def fake_projector_builder(refs_real, *, current_size, padding_factor):
        calls.append(np.asarray(refs_real).copy())
        projector_half = np.full((refs_real.shape[0], 3, 3, 2), 7.0 + len(calls), dtype=np.complex64)
        return projector_half, int(current_size // 2)

    monkeypatch.setattr(dense_adapter, "reference_to_relion_projector_half_maps", fake_projector_builder)
    monkeypatch.setenv("RECOVAR_RELION_PROJECTOR_CACHE_DIR", str(tmp_path))

    mean_ft = np.zeros((4, 4, 4), dtype=np.complex64)
    mean_ft[0, 0, 0] = 1.0
    first = iteration_loop_module._relion_projector_half_maps_for_scoring(
        mean_ft.reshape(-1),
        volume_shape=(4, 4, 4),
        current_size=4,
        padding_factor=2,
        n_classes=1,
    )
    second = iteration_loop_module._relion_projector_half_maps_for_scoring(
        mean_ft.reshape(-1),
        volume_shape=(4, 4, 4),
        current_size=4,
        padding_factor=2,
        n_classes=1,
    )

    assert len(calls) == 1
    assert first[1] == second[1] == 2
    np.testing.assert_array_equal(first[0], second[0])
    assert (tmp_path / "SAFE_TO_DELETE").exists()
    assert len(list(tmp_path.glob("projector_*.npz"))) == 1


def test_half0_local_relion_accumulators_offload_to_host():
    result = iteration_loop_module.HalfScoreResult(
        ha=np.zeros(1, dtype=np.int32),
        Ft_y=jnp.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=jnp.complex64),
        Ft_ctf=jnp.asarray([5.0, 6.0], dtype=jnp.float32),
        em_stats=object(),
        noise_stats=object(),
        mstep_full_half_axis=0,
        mstep_accumulator_shape=(3, 3, 3),
    )

    out = iteration_loop_module._maybe_host_offload_half0_local_accumulators(
        half_index=0,
        use_local=True,
        k_class_enabled=False,
        score_result=result,
    )

    assert out is result
    assert isinstance(out.Ft_y, np.ndarray)
    assert isinstance(out.Ft_ctf, np.ndarray)
    np.testing.assert_allclose(out.Ft_y, np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64))
    np.testing.assert_allclose(out.Ft_ctf, np.array([5.0, 6.0], dtype=np.float32))


def test_half0_local_relion_accumulator_offload_skips_non_x_half():
    result = iteration_loop_module.HalfScoreResult(
        ha=np.zeros(1, dtype=np.int32),
        Ft_y=jnp.asarray([1.0 + 0.0j], dtype=jnp.complex64),
        Ft_ctf=jnp.asarray([1.0], dtype=jnp.float32),
        em_stats=object(),
        noise_stats=object(),
        mstep_full_half_axis=None,
    )

    out = iteration_loop_module._maybe_host_offload_half0_local_accumulators(
        half_index=0,
        use_local=True,
        k_class_enabled=False,
        score_result=result,
    )

    assert out is result
    assert not isinstance(out.Ft_y, np.ndarray)


def test_exact_local_relion_x_half_full_support_mstep_uses_fftw_indices_and_radius():
    from recovar.em.dense_single_volume import local_em_engine

    image_shape = (128, 128)
    half_width = image_shape[1] // 2 + 1
    indices, max_r = local_em_engine._local_mstep_adjoint_window(
        image_shape,
        image_shape[0] * half_width,
        None,
        use_window=False,
        recon_window_indices=None,
        mstep_relion_x_half=True,
    )

    assert max_r == 64.0
    assert int(indices[64 * half_width]) == 0
    assert int(indices[36 * half_width + 5]) == 100 * half_width + 5
    assert int(indices[92 * half_width + 7]) == 28 * half_width + 7


def test_exact_local_relion_x_half_windowed_mstep_uses_fftw_indices_and_current_radius():
    from recovar.em.dense_single_volume import local_em_engine

    image_shape = (128, 128)
    half_width = image_shape[1] // 2 + 1
    recon_window_indices = jnp.asarray(
        [
            64 * half_width + 0,
            36 * half_width + 5,
            92 * half_width + 7,
        ],
        dtype=jnp.int32,
    )

    indices, max_r = local_em_engine._local_mstep_adjoint_window(
        image_shape,
        image_shape[0] * half_width,
        56,
        use_window=True,
        recon_window_indices=recon_window_indices,
        mstep_relion_x_half=True,
    )

    assert max_r == 28.0
    assert indices.tolist() == [
        0 * half_width + 0,
        100 * half_width + 5,
        28 * half_width + 7,
    ]


def test_exact_local_fused_posterior_missing_warning_respects_filters():
    from recovar.em.dense_single_volume import local_em_engine

    src = inspect.getsource(local_em_engine.run_local_em_exact)
    assert "debug_fused_posterior_dump_filter_matches = (" in src
    assert "current_size_matches_request(debug_fused_posterior_dump_current_sizes, current_size)" in src
    assert "iteration_matches_request(debug_fused_posterior_dump_iterations, debug_iteration)" in src
    assert (
        "debug_fused_posterior_dump_filter_matches\n"
        "        and debug_fused_posterior_dump_targets"
    ) in src


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
        rotation_log_prior=None,
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
        captured["use_float64_scoring"] = kwargs.get("use_float64_scoring")
        captured["use_float64_normalization"] = kwargs.get("use_float64_normalization")
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
    assert captured["use_float64_scoring"] is False
    assert captured["use_float64_normalization"] is True
    np.testing.assert_allclose(
        captured["translation_prior_reference_translations"],
        reference_translations,
        atol=1e-6,
    )
    assert len(outputs) == 5


def test_run_local_search_iteration_clamps_highres_local_batches(monkeypatch):
    # Pin GPU memory queries so the batch-size clamp is deterministic.
    # ``_estimate_relion_em_batch_sizes`` reads
    # ``iteration_loop.utils.get_gpu_memory_total/_used``; without pinning,
    # the clamp behavior depends on prior test JAX allocations — the test
    # passes in isolation (and on H100 / A100-80 in the dev branch where
    # ``get_gpu_memory_total`` returned ~42 GB) but fails after the full
    # suite when the queries return stale or inconsistent values. The 42 GB
    # / 0 GB-used pair is the historical isolation reading that this test
    # was originally calibrated against.
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 42.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 0.0)

    class HighresDataset:
        image_shape = (384, 384)
        image_size = 384 * 384
        volume_shape = (384, 384, 384)
        volume_size = 1
        n_images = 2
        n_units = 2
        voxel_size = 1.0
        dtype = jnp.complex64

    translation_grid = np.zeros((137, 2), dtype=np.float32)
    layout = LocalHypothesisLayout(
        n_global_rotations=1024,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 1024, 2048], dtype=np.int64),
        rotation_ids_flat=np.tile(np.arange(1024, dtype=np.int32), 2),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (2048, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(2048, dtype=np.float32),
        rotation_counts=np.array([1024, 1024], dtype=np.int32),
        translation_grid=translation_grid,
        translation_log_priors=np.zeros((2, translation_grid.shape[0]), dtype=np.float32),
    )
    captured = {}

    def fake_run_local_em_exact(*args, **kwargs):
        _ = args
        captured["image_batch_size"] = int(kwargs["image_batch_size"])
        captured["rotation_block_size"] = int(kwargs["rotation_block_size"])
        return (
            jnp.zeros(1, dtype=jnp.complex64),
            jnp.zeros(1, dtype=jnp.complex64),
            np.zeros(2, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(2, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(2, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(2, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(1024, dtype=jnp.float32),
            ),
        )

    monkeypatch.setattr(iteration_loop_module, "run_local_em_exact", fake_run_local_em_exact)

    outputs = iteration_loop_module._run_local_search_iteration(
        HighresDataset(),
        jnp.zeros(1, dtype=jnp.complex64),
        jnp.ones(1, dtype=jnp.float32),
        jnp.ones(384 * 384, dtype=jnp.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32), (1024, 3, 3)).copy(),
        None,
        healpix_order=2,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=translation_grid,
        prior_translations=np.zeros((2, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=None,
        disc_type="linear_interp",
        image_batch_size=250,
        rotation_block_size=5000,
        current_size=56,
        accumulate_noise=False,
        projection_padding_factor=2,
        reconstruction_padding_factor=2,
        pass2_layout=layout,
    )

    assert captured["image_batch_size"] < 250
    assert captured["rotation_block_size"] == 1024
    assert len(outputs) == 4


def test_run_local_search_iteration_relion_xhalf_uses_windowed_batch_guard_by_default(monkeypatch):
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_total", lambda: 42.0)
    monkeypatch.setattr(iteration_loop_module.utils, "get_gpu_memory_used", lambda: 0.0)

    class Dataset256:
        image_shape = (256, 256)
        image_size = 256 * 256
        volume_shape = (256, 256, 256)
        volume_size = 1
        n_images = 2
        n_units = 2
        voxel_size = 1.0
        dtype = jnp.complex64

    local_rotations = 1536
    n_trans = 116
    layout = LocalHypothesisLayout(
        n_global_rotations=local_rotations,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, local_rotations, 2 * local_rotations], dtype=np.int64),
        rotation_ids_flat=np.tile(np.arange(local_rotations, dtype=np.int32), 2),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (2 * local_rotations, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(2 * local_rotations, dtype=np.float32),
        rotation_counts=np.full(2, local_rotations, dtype=np.int32),
        translation_grid=np.zeros((n_trans, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, n_trans), dtype=np.float32),
    )
    captured = {}

    def fake_run_local_em_exact(*args, **kwargs):
        _ = args
        captured["image_batch_size"] = int(kwargs["image_batch_size"])
        captured["rotation_block_size"] = int(kwargs["rotation_block_size"])
        return (
            jnp.zeros(1, dtype=jnp.complex64),
            jnp.zeros(1, dtype=jnp.complex64),
            np.zeros(2, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(2, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(2, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(2, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(local_rotations, dtype=jnp.float32),
            ),
        )

    monkeypatch.setattr(iteration_loop_module, "run_local_em_exact", fake_run_local_em_exact)
    monkeypatch.delenv("RECOVAR_LOCAL_XHALF_BATCH_GUARD", raising=False)

    iteration_loop_module._run_local_search_iteration(
        Dataset256(),
        jnp.zeros(1, dtype=jnp.complex64),
        jnp.ones(1, dtype=jnp.float32),
        jnp.ones(256 * 129, dtype=jnp.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32), (local_rotations, 3, 3)).copy(),
        None,
        healpix_order=4,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=layout.translation_grid,
        prior_translations=np.zeros((2, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=None,
        disc_type="linear_interp",
        image_batch_size=41,
        rotation_block_size=58,
        current_size=166,
        accumulate_noise=False,
        projection_padding_factor=2,
        reconstruction_padding_factor=2,
        relion_projector_half=jnp.zeros((1, 1, 1), dtype=jnp.complex64),
        relion_projector_r_max=128,
        mstep_relion_x_half=True,
        pass2_layout=layout,
    )

    assert captured["image_batch_size"] == 41
    assert captured["rotation_block_size"] == 58

    monkeypatch.setenv("RECOVAR_LOCAL_XHALF_BATCH_GUARD", "full")
    iteration_loop_module._run_local_search_iteration(
        Dataset256(),
        jnp.zeros(1, dtype=jnp.complex64),
        jnp.ones(1, dtype=jnp.float32),
        jnp.ones(256 * 129, dtype=jnp.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32), (local_rotations, 3, 3)).copy(),
        None,
        healpix_order=4,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=layout.translation_grid,
        prior_translations=np.zeros((2, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=None,
        disc_type="linear_interp",
        image_batch_size=41,
        rotation_block_size=58,
        current_size=166,
        accumulate_noise=False,
        projection_padding_factor=2,
        reconstruction_padding_factor=2,
        relion_projector_half=jnp.zeros((1, 1, 1), dtype=jnp.complex64),
        relion_projector_r_max=128,
        mstep_relion_x_half=True,
        pass2_layout=layout,
    )

    assert captured["image_batch_size"] == 13
    assert captured["rotation_block_size"] == 19


def test_run_local_search_iteration_plumbs_score_only_to_exact_engine(monkeypatch, rng):
    mock_dataset = MockDataset(2, rng)
    layout = LocalHypothesisLayout(
        n_global_rotations=3,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 1, 2], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.zeros((2, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )
    captured = {}

    def fake_run_local_em_exact(*args, **kwargs):
        _ = args
        captured.update(kwargs)
        return (
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            np.zeros(mock_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(3, dtype=jnp.float32),
            ),
            {"score_only": kwargs["score_only"]},
        )

    monkeypatch.setattr(iteration_loop_module, "run_local_em_exact", fake_run_local_em_exact)

    outputs = iteration_loop_module._run_local_search_iteration(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy(),
        None,
        healpix_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=layout.translation_grid,
        prior_translations=np.zeros((2, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=None,
        disc_type="linear_interp",
        image_batch_size=2,
        rotation_block_size=8,
        current_size=4,
        accumulate_noise=False,
        pass2_layout=layout,
        return_profile=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        score_only=True,
    )

    assert captured["score_only"] is True
    assert captured["disable_adjoint_y"] is True
    assert captured["disable_adjoint_ctf"] is True
    assert captured["accumulate_noise"] is False
    assert outputs[-1]["score_only"] is True


def test_local_adaptive_parent_support_probe_is_score_only():
    source = Path(iteration_loop_module.__file__).read_text()
    start = source.index("parent_outputs = _run_local_search_iteration(")
    end = source.index("parent_profile = parent_outputs[-1]", start)
    parent_call = source[start:end]

    assert "disable_adjoint_y=True" in parent_call
    assert "disable_adjoint_ctf=True" in parent_call
    assert "return_reconstruction_sample_indices=True" in parent_call
    assert "score_only=True" in parent_call


def test_run_local_search_iteration_plumbs_normalization_log_evidence(monkeypatch, rng):
    mock_dataset = MockDataset(2, rng)
    layout = LocalHypothesisLayout(
        n_global_rotations=3,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 1, 2], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.zeros((2, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )
    captured = {}
    normalization_log_evidence = np.array([1.25, 2.5], dtype=np.float64)

    def fake_run_local_em_exact(*args, **kwargs):
        _ = args
        captured.update(kwargs)
        return (
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            np.zeros(mock_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(3, dtype=jnp.float32),
            ),
        )

    monkeypatch.setattr(iteration_loop_module, "run_local_em_exact", fake_run_local_em_exact)

    outputs = iteration_loop_module._run_local_search_iteration(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy(),
        None,
        healpix_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=layout.translation_grid,
        prior_translations=np.zeros((2, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=None,
        disc_type="linear_interp",
        image_batch_size=2,
        rotation_block_size=8,
        current_size=4,
        accumulate_noise=False,
        pass2_layout=layout,
        normalization_log_evidence=normalization_log_evidence,
    )

    np.testing.assert_allclose(captured["normalization_log_evidence"], normalization_log_evidence)
    assert len(outputs) == 4


def test_run_local_search_iteration_plumbs_stats_use_reconstruction_probs(monkeypatch, rng):
    mock_dataset = MockDataset(2, rng)
    layout = LocalHypothesisLayout(
        n_global_rotations=3,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 1, 2], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.zeros((2, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )
    captured = {}

    def fake_run_local_em_exact(*args, **kwargs):
        _ = args
        captured["stats_use_reconstruction_probs"] = kwargs["stats_use_reconstruction_probs"]
        return (
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            np.zeros(mock_dataset.n_units, dtype=np.int32),
            RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(3, dtype=jnp.float32),
            ),
        )

    monkeypatch.setattr(iteration_loop_module, "run_local_em_exact", fake_run_local_em_exact)

    outputs = iteration_loop_module._run_local_search_iteration(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy(),
        None,
        healpix_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=layout.translation_grid,
        prior_translations=np.zeros((2, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=None,
        disc_type="linear_interp",
        image_batch_size=2,
        rotation_block_size=8,
        current_size=4,
        accumulate_noise=False,
        pass2_layout=layout,
        reconstruct_significant_only=True,
        stats_use_reconstruction_probs=True,
    )

    assert captured["stats_use_reconstruction_probs"] is True
    assert len(outputs) == 4


def test_run_local_search_iteration_rejects_k_class_score_only(rng):
    mock_dataset = MockDataset(2, rng)
    layout = LocalHypothesisLayout(
        n_global_rotations=3,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 1, 2], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.zeros((2, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(NotImplementedError, match="K-class local search does not support score_only"):
        iteration_loop_module._run_local_search_iteration(
            mock_dataset,
            jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy(),
            None,
            healpix_order=0,
            sigma_rot=0.1,
            sigma_psi=0.1,
            translations=layout.translation_grid,
            prior_translations=np.zeros((2, 2), dtype=np.float32),
            sigma_offset_angstrom=1.0,
            offset_range_pixels=None,
            disc_type="linear_interp",
            image_batch_size=2,
            rotation_block_size=8,
            current_size=4,
            pass2_layout=layout,
            class_log_priors=np.zeros(2, dtype=np.float32),
            score_only=True,
        )


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
        rotation_log_prior=None,
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
        captured["use_float64_scoring"] = kwargs.get("use_float64_scoring")
        captured["use_float64_normalization"] = kwargs.get("use_float64_normalization")
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
    assert (
        build_local_search_grid_metadata(
            healpix_order,
            grid_eulers=perturbed_eulers,
            grid_rotations=perturbed_rotations,
        )["mode"]
        == "full"
    )

    outputs = iteration_loop_module._run_local_search_iteration(
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
    assert captured["use_float64_scoring"] is False
    assert captured["use_float64_normalization"] is True
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
        current_size=6,
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
    ) = _prepare_local_exact_bucket(
        dataset,
        batch_data,
        ctf_params,
        bucket.image_indices,
        noise_variance_half,
        jnp.asarray(local_layout.translation_grid),
        config,
        half_weights,
        score_with_masked_images=True,
    )
    flat_rotations = flatten_bucket_rotations(jnp.asarray(bucket.local_rotations))
    n_half = dataset.image_shape[0] * (dataset.image_shape[1] // 2 + 1)
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec

    window_spec = make_fourier_window_spec(dataset.image_shape, 6, n_half)
    proj_half_flat = core.slice_volume(
        mean,
        flat_rotations,
        dataset.image_shape,
        dataset.volume_shape,
        "linear_interp",
        half_image=True,
        max_r=window_spec.projection_max_r if window_spec.use_window else "auto",
    )
    proj_abs2_half_flat = jnp.abs(proj_half_flat) ** 2
    proj_half = window_spec.score_values(proj_half_flat).reshape(
        1,
        bucket.bucket_rotation_count,
        window_spec.n_score,
    )
    proj_abs2 = window_spec.score_values(proj_abs2_half_flat).reshape(
        1,
        bucket.bucket_rotation_count,
        window_spec.n_score,
    )
    score_half_weights = window_spec.score_values(half_weights)
    proj_weighted = proj_half * score_half_weights[None, None, :]
    proj_abs2_weighted = proj_abs2 * score_half_weights[None, None, :]
    shifted_score = window_spec.score_values(shifted_score_half)
    ctf2_over_nv_score = window_spec.score_values(ctf2_over_nv_half)
    scores = score_local_bucket(
        shifted_score.reshape(1, 1, -1),
        ctf2_over_nv_score,
        proj_weighted,
        proj_abs2_weighted,
        jnp.asarray(bucket.local_rotation_log_prior),
        jnp.asarray(bucket.translation_log_prior),
        jnp.asarray(bucket.local_rotation_mask),
    )
    _log_Z, probs, _best_log_score, _best_argmax, _max_posterior = normalize_local_scores(scores)
    shifted_recon = window_spec.recon_values(shifted_recon_half)
    ctf2_over_nv_recon = window_spec.recon_values(ctf2_over_nv_half)
    shifted_recon_split = shifted_recon.reshape(1, 1, -1)
    manual_summed = compute_local_weighted_sums(probs, shifted_recon_split)
    manual_ctf_probs = compute_local_ctf_sums(probs, ctf2_over_nv_recon)
    from recovar.em.dense_single_volume.helpers.adjoint import adjoint_slice_volume_maybe_windowed

    Ft_y_manual_half = adjoint_slice_volume_maybe_windowed(
        flatten_bucket_rows(manual_summed),
        window_spec.recon_indices,
        flat_rotations,
        jnp.zeros(int(np.prod(half_volume_accumulator_shape(dataset.volume_shape))), dtype=manual_summed.dtype),
        dataset.image_shape,
        dataset.volume_shape,
        "linear_interp",
        half_image=True,
        half_volume=True,
        use_window=window_spec.use_window,
        max_r=float(6 // 2) if window_spec.use_window else None,
    )
    Ft_ctf_manual_half = adjoint_slice_volume_maybe_windowed(
        flatten_bucket_rows(manual_ctf_probs),
        window_spec.recon_indices,
        flat_rotations,
        jnp.zeros(int(np.prod(half_volume_accumulator_shape(dataset.volume_shape))), dtype=manual_ctf_probs.dtype),
        dataset.image_shape,
        dataset.volume_shape,
        "linear_interp",
        half_image=True,
        half_volume=True,
        use_window=window_spec.use_window,
        max_r=float(6 // 2) if window_spec.use_window else None,
    )
    Ft_y_manual_half, Ft_ctf_manual_half = enforce_half_volume_x0(
        Ft_y_manual_half,
        Ft_ctf_manual_half,
        dataset.volume_shape,
        logger=iteration_loop_module.logger,
        label="test-local",
    )
    Ft_y_manual, Ft_ctf_manual = half_volume_accumulators_to_full(
        Ft_y_manual_half,
        Ft_ctf_manual_half,
        dataset.volume_shape,
    )

    np.testing.assert_allclose(np.asarray(Ft_y_exact), np.asarray(Ft_y_manual), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_exact), np.asarray(Ft_ctf_manual), atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(ha_exact, ha_dense)
    log_score_offset = -0.5 * np.asarray(_batch_norm).reshape(-1)
    np.testing.assert_allclose(
        np.asarray(stats_exact.log_evidence_per_image),
        np.asarray(_log_Z + log_score_offset, dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    manual_rotation_posteriors = np.zeros(2, dtype=np.float64)
    np.add.at(
        manual_rotation_posteriors,
        np.asarray(bucket.local_rotation_ids).reshape(-1),
        np.asarray(probs).sum(axis=2).reshape(-1),
    )
    np.testing.assert_allclose(
        np.asarray(stats_exact.rotation_posterior_sums[:2]),
        manual_rotation_posteriors,
        atol=1e-5,
        rtol=1e-5,
    )
    assert np.all(np.isfinite(np.asarray(noise_exact.wsum_sigma2_noise)))


def test_run_local_em_exact_can_return_half_volume_accumulators(rng, monkeypatch):
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=117)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(2, seed=119)
    translations = np.zeros((1, 2), dtype=np.float32)
    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.asarray(local_rotations, dtype=np.float32),
        rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((1, 1), dtype=np.float32),
    )

    full = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
    )
    half = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
        return_half_volume_accumulators=True,
    )

    half_shape = ftu.volume_shape_to_half_volume_shape(dataset.volume_shape)
    assert np.asarray(half[0]).size == int(np.prod(half_shape))
    assert np.asarray(half[1]).size == int(np.prod(half_shape))
    Ft_y_half_full, Ft_ctf_half_full = half_volume_accumulators_to_full(
        half[0],
        half[1],
        dataset.volume_shape,
    )
    np.testing.assert_allclose(np.asarray(Ft_y_half_full), np.asarray(full[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_half_full), np.asarray(full[1]), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(half[2]), np.asarray(full[2]))
    np.testing.assert_allclose(
        np.asarray(half[3].log_evidence_per_image),
        np.asarray(full[3].log_evidence_per_image),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sparse_adjoint_row_chunks_match_single_windowed_adjoint(rng, monkeypatch):
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    from recovar.em.dense_single_volume.helpers.adjoint import adjoint_slice_volume_maybe_windowed

    window_indices = jnp.asarray([0, 1, 2, 5, 8, 13, 21, 30], dtype=jnp.int32)
    n_rows = 5
    rotations = jnp.asarray(_make_rotations(n_rows, seed=219), dtype=jnp.float32)
    rows_np = (
        rng.normal(size=(n_rows, window_indices.shape[0]))
        + 1j * rng.normal(size=(n_rows, window_indices.shape[0]))
    ).astype(np.complex64)
    rows = jnp.asarray(rows_np)
    half_shape = half_volume_accumulator_shape(VOLUME_SHAPE)
    Ft0 = jnp.zeros(int(np.prod(half_shape)), dtype=jnp.complex64)

    Ft_single = adjoint_slice_volume_maybe_windowed(
        rows,
        window_indices,
        rotations,
        Ft0,
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        True,
        True,
        use_window=True,
        max_r=3.0,
    )
    Ft_chunked, n_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
        rows,
        window_indices,
        rotations,
        Ft0,
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        use_window=True,
        max_r=3.0,
        relion_x_half=False,
        target_rows=2,
    )

    assert n_chunks == 3
    np.testing.assert_allclose(np.asarray(Ft_chunked), np.asarray(Ft_single), atol=1e-5, rtol=1e-5)


def test_run_em_dense_can_return_half_volume_accumulators(rng, monkeypatch):
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=118)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    rotations = _make_rotations(2, seed=120)
    translations = np.zeros((1, 2), dtype=np.float32)

    full = run_em(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        score_with_masked_images=True,
        return_stats=True,
        sparse_pass2=False,
        relion_half_volume_mstep=True,
    )
    half = run_em(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        score_with_masked_images=True,
        return_stats=True,
        sparse_pass2=False,
        relion_half_volume_mstep=True,
        return_half_volume_accumulators=True,
    )

    half_shape = ftu.volume_shape_to_half_volume_shape(dataset.volume_shape)
    assert full[0] is None
    assert half[0] is None
    assert np.asarray(full[2]).size == VOLUME_SIZE
    assert np.asarray(full[3]).size == VOLUME_SIZE
    assert np.asarray(half[2]).size == int(np.prod(half_shape))
    assert np.asarray(half[3]).size == int(np.prod(half_shape))
    Ft_y_half_full, Ft_ctf_half_full = half_volume_accumulators_to_full(
        half[2],
        half[3],
        dataset.volume_shape,
    )
    np.testing.assert_allclose(np.asarray(Ft_y_half_full), np.asarray(full[2]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_half_full), np.asarray(full[3]), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(half[1]), np.asarray(full[1]))
    np.testing.assert_allclose(
        np.asarray(half[4].log_evidence_per_image),
        np.asarray(full[4].log_evidence_per_image),
        rtol=1e-5,
        atol=1e-5,
    )


def test_run_local_em_exact_class_log_prior_shifts_evidence_only(rng):
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=121)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(2, seed=129)
    translations = np.zeros((1, 2), dtype=np.float32)
    class_log_prior = float(np.log(0.25))

    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.asarray(local_rotations, dtype=np.float32),
        rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((1, 1), dtype=np.float32),
    )

    Ft_y_base, Ft_ctf_base, ha_base, stats_base = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
    )
    Ft_y_prior, Ft_ctf_prior, ha_prior, stats_prior = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
        class_log_prior=class_log_prior,
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
        np.asarray(stats_prior.max_posterior_per_image),
        np.asarray(stats_base.max_posterior_per_image),
        rtol=1e-5,
        atol=1e-6,
    )


def test_run_local_em_exact_external_log_evidence_scales_posterior(rng, monkeypatch):
    monkeypatch.setenv("RECOVAR_DISABLE_LOCAL_BIG_JIT", "1")
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=131)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(2, seed=139)
    translations = np.zeros((1, 2), dtype=np.float32)

    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.asarray(local_rotations, dtype=np.float32),
        rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((1, 1), dtype=np.float32),
    )

    Ft_y_base, Ft_ctf_base, ha_base, stats_base = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
    )
    external_log_evidence = np.asarray(stats_base.log_evidence_per_image, dtype=np.float64) + np.log(2.0)
    Ft_y_scaled, Ft_ctf_scaled, ha_scaled, stats_scaled = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
        normalization_log_evidence=external_log_evidence,
    )
    Ft_y_fallback, Ft_ctf_fallback, ha_fallback, stats_fallback = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
        normalization_log_evidence=external_log_evidence,
        reconstruction_probability_threshold=np.zeros(dataset.n_units, dtype=np.float64),
    )

    np.testing.assert_array_equal(ha_scaled, ha_base)
    np.testing.assert_array_equal(ha_scaled, ha_fallback)
    np.testing.assert_allclose(np.asarray(Ft_y_scaled), 0.5 * np.asarray(Ft_y_base), rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_scaled), 0.5 * np.asarray(Ft_ctf_base), rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_y_scaled), np.asarray(Ft_y_fallback), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(Ft_ctf_scaled), np.asarray(Ft_ctf_fallback), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(stats_scaled.log_evidence_per_image),
        external_log_evidence,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_scaled.max_posterior_per_image),
        0.5 * np.asarray(stats_base.max_posterior_per_image),
        rtol=5e-3,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(stats_scaled.max_posterior_per_image),
        np.asarray(stats_fallback.max_posterior_per_image),
        rtol=1e-5,
        atol=1e-6,
    )


def test_run_local_em_exact_deferred_packed_mstep_matches_fused(rng, monkeypatch):
    monkeypatch.setenv("RECOVAR_DISABLE_LOCAL_BIG_JIT", "1")
    dataset = MockDataset(2, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=137)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(3, seed=143)
    translations = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)

    local_layout = LocalHypothesisLayout(
        n_global_rotations=3,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 3, 6], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
        rotations_flat=np.tile(np.asarray(local_rotations, dtype=np.float32), (2, 1, 1)),
        rotation_log_priors_flat=np.zeros(6, dtype=np.float32),
        rotation_counts=np.array([3, 3], dtype=np.int32),
        translation_grid=translations,
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )

    Ft_y_base, Ft_ctf_base, ha_base, stats_base = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=True,
    )
    monkeypatch.setenv("RECOVAR_EXACT_LOCAL_DEFER_PACKED_MSTEP", "1")
    Ft_y_deferred, Ft_ctf_deferred, ha_deferred, stats_deferred = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=True,
    )

    np.testing.assert_array_equal(ha_deferred, ha_base)
    np.testing.assert_allclose(np.asarray(Ft_y_deferred), np.asarray(Ft_y_base), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(Ft_ctf_deferred), np.asarray(Ft_ctf_base), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(stats_deferred.log_evidence_per_image),
        np.asarray(stats_base.log_evidence_per_image),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(stats_deferred.max_posterior_per_image),
        np.asarray(stats_base.max_posterior_per_image),
        rtol=1e-5,
        atol=1e-6,
    )


def test_dense_k_class_identical_means_split_global_posterior(rng):
    dataset = MockDataset(2, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=141)
    means = jnp.stack([mean, mean], axis=0)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    rotations = _make_rotations(2, seed=149)
    translations = np.zeros((1, 2), dtype=np.float32)

    _, ha_base, Ft_y_base, Ft_ctf_base, stats_base = run_em(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=4,
        current_size=None,
        score_with_masked_images=True,
        return_stats=True,
        sparse_pass2=False,
    )
    result = run_dense_k_class_em(
        dataset,
        means,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=4,
        current_size=None,
        score_with_masked_images=True,
        sparse_pass2=False,
    )

    np.testing.assert_array_equal(np.asarray(result.per_class_hard_assignments[0]), ha_base)
    np.testing.assert_allclose(np.asarray(result.Ft_y[0]), 0.5 * np.asarray(Ft_y_base), rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result.Ft_ctf[0]), 0.5 * np.asarray(Ft_ctf_base), rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(result.Ft_y, axis=0)),
        np.asarray(Ft_y_base),
        rtol=5e-3,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.stats.log_evidence_per_image),
        np.asarray(stats_base.log_evidence_per_image),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.class_posterior_sums),
        np.full(2, dataset.n_images / 2.0, dtype=np.float32),
        rtol=5e-3,
        atol=1e-5,
    )


def test_local_k_class_identical_means_split_global_posterior(rng):
    dataset = MockDataset(2, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=151)
    means = jnp.stack([mean, mean], axis=0)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(2, seed=159)
    translations = np.zeros((1, 2), dtype=np.float32)
    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 0, 1], dtype=np.int32),
        rotations_flat=np.tile(np.asarray(local_rotations, dtype=np.float32), (2, 1, 1)),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((2, 1), dtype=np.float32),
    )

    Ft_y_base, Ft_ctf_base, ha_base, stats_base = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
    )
    result = run_local_k_class_em(
        dataset,
        means,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
    )

    np.testing.assert_array_equal(np.asarray(result.per_class_hard_assignments[0]), ha_base)
    np.testing.assert_allclose(np.asarray(result.Ft_y[0]), 0.5 * np.asarray(Ft_y_base), rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result.Ft_ctf[0]), 0.5 * np.asarray(Ft_ctf_base), rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(result.Ft_y, axis=0)),
        np.asarray(Ft_y_base),
        rtol=5e-3,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.stats.log_evidence_per_image),
        np.asarray(stats_base.log_evidence_per_image),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.class_posterior_sums),
        np.full(2, dataset.n_images / 2.0, dtype=np.float32),
        rtol=5e-3,
        atol=1e-5,
    )


def test_run_local_em_exact_can_report_significant_support_rotation_stats(rng):
    dataset = MockDataset(2, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=163)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = np.zeros((1, 2), dtype=np.float32)
    rotations_flat = np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy()
    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 0, 1], dtype=np.int32),
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=np.tile(np.log(np.asarray([0.75, 0.25], dtype=np.float32)), 2),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((2, 1), dtype=np.float32),
    )

    base = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=4,
        current_size=None,
        accumulate_noise=True,
        reconstruct_significant_only=True,
        max_significants=1,
    )
    support_stats = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=4,
        current_size=None,
        accumulate_noise=True,
        reconstruct_significant_only=True,
        max_significants=1,
        stats_use_reconstruction_probs=True,
    )

    stats_base = base[3]
    stats_support = support_stats[3]
    noise_support = support_stats[4]
    np.testing.assert_allclose(
        np.sum(np.asarray(stats_base.rotation_posterior_sums)),
        dataset.n_images,
        rtol=5e-3,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(stats_support.rotation_posterior_sums)),
        float(noise_support.sumw),
        rtol=5e-3,
        atol=1e-5,
    )
    assert float(noise_support.sumw) < float(dataset.n_images)


def test_run_local_em_exact_collects_unpruned_probability_values_for_global_threshold(rng):
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=167)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = np.zeros((1, 2), dtype=np.float32)
    rotations_flat = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy()
    local_layout = LocalHypothesisLayout(
        n_global_rotations=3,
        n_pixels=3,
        n_psi=1,
        rotation_offsets=np.array([0, 3], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 2], dtype=np.int32),
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=np.log(np.asarray([0.6, 0.3, 0.1], dtype=np.float32)),
        rotation_counts=np.array([3], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((1, 1), dtype=np.float32),
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
        current_size=None,
        reconstruct_significant_only=True,
        max_significants=1,
        return_profile=True,
        return_reconstruction_probability_values=True,
    )

    values = outputs[-1]["reconstruction_probability_values_by_image"][0]
    assert values.size == 3
    np.testing.assert_allclose(np.sum(values), 1.0, rtol=5e-3, atol=1e-5)


def test_run_local_em_exact_collects_global_reconstruction_sample_ids(rng):
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=171)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = np.zeros((1, 2), dtype=np.float32)
    rotations_flat = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy()
    local_layout = LocalHypothesisLayout(
        n_global_rotations=8,
        n_pixels=8,
        n_psi=1,
        rotation_offsets=np.array([0, 3], dtype=np.int64),
        rotation_ids_flat=np.array([2, 7, 4], dtype=np.int32),
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=np.log(np.asarray([0.1, 0.8, 0.1], dtype=np.float32)),
        rotation_counts=np.array([3], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((1, 1), dtype=np.float32),
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
        current_size=None,
        reconstruct_significant_only=True,
        max_significants=1,
        return_profile=True,
        return_reconstruction_sample_indices=True,
    )

    sample_ids = outputs[-1]["reconstruction_sample_indices_by_image"][0]
    np.testing.assert_array_equal(sample_ids, np.array([7], dtype=np.int32))


def test_run_local_em_exact_collapses_fine_children_to_parent_posterior_ids(rng):
    dataset = MockDataset(1, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=173)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = np.zeros((1, 2), dtype=np.float32)
    rotations_flat = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy()
    local_layout = LocalHypothesisLayout(
        n_global_rotations=8,
        n_pixels=8,
        n_psi=1,
        rotation_offsets=np.array([0, 3], dtype=np.int64),
        rotation_ids_flat=np.array([2, 7, 4], dtype=np.int32),
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=np.log(np.asarray([0.1, 0.8, 0.1], dtype=np.float32)),
        rotation_counts=np.array([3], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((1, 1), dtype=np.float32),
        rotation_posterior_ids_flat=np.array([1, 1, 3], dtype=np.int32),
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
        current_size=None,
        reconstruct_significant_only=True,
        max_significants=1,
        return_profile=True,
        return_reconstruction_sample_indices=True,
    )

    hard_assignment = np.asarray(outputs[2])
    stats = outputs[3]
    sample_ids = outputs[-1]["reconstruction_sample_indices_by_image"][0]
    np.testing.assert_array_equal(hard_assignment, np.array([7], dtype=np.int32))
    np.testing.assert_array_equal(sample_ids, np.array([1], dtype=np.int64))
    posterior_sums = np.asarray(stats.rotation_posterior_sums)
    assert posterior_sums[1] > posterior_sums[3]
    assert posterior_sums[0] == pytest.approx(0.0)
    assert posterior_sums[2] == pytest.approx(0.0)


def test_local_k_class_can_report_noise_support_class_sums(monkeypatch):
    import recovar.em.dense_single_volume.k_class as k_class_module

    dataset = type("Dataset", (), {"n_images": 2, "n_units": 2})()
    means = jnp.zeros((2, 4), dtype=jnp.complex64)
    mean_variance = jnp.ones((2, 4), dtype=jnp.float32)
    noise_variance = jnp.ones(4, dtype=jnp.float32)
    local_layout = LocalHypothesisLayout(
        n_global_rotations=1,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 1, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 0], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
        rotation_counts=np.array([1, 1], dtype=np.int32),
        translation_grid=np.zeros((1, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 1), dtype=np.float32),
    )
    class_log_evidence = np.log(np.asarray([[0.9, 0.9], [0.1, 0.1]], dtype=np.float64))
    support_sumw = [0.25, 1.75]
    calls = []

    def fake_run_local_em_exact(*args, **kwargs):
        class_index = len(calls)
        calls.append(kwargs)
        return (
            jnp.full(4, class_index + 1, dtype=jnp.complex64),
            jnp.full(4, class_index + 1, dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.int32),
            RelionStats(
                log_evidence_per_image=jnp.asarray(class_log_evidence[class_index], dtype=jnp.float32),
                best_log_score_per_image=jnp.full(2, class_index, dtype=jnp.float32),
                max_posterior_per_image=jnp.full(2, 0.5, dtype=jnp.float32),
                rotation_posterior_sums=jnp.asarray([support_sumw[class_index]], dtype=jnp.float32),
            ),
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(1, dtype=jnp.float32),
                wsum_img_power=jnp.ones(1, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=support_sumw[class_index],
            ),
        )

    monkeypatch.setattr(k_class_module, "run_local_em_exact", fake_run_local_em_exact)

    result = run_local_k_class_em(
        dataset,
        means,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=2,
        rotation_block_size=1,
        current_size=None,
        accumulate_noise=True,
        class_log_evidence=class_log_evidence,
        stats_use_reconstruction_probs=True,
        class_posterior_sums_from_noise=True,
    )

    assert [call["stats_use_reconstruction_probs"] for call in calls] == [True, True]
    np.testing.assert_allclose(
        np.asarray(result.class_responsibilities),
        [[0.9, 0.9], [0.1, 0.1]],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(result.class_posterior_sums), [1.8, 0.2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.class_mstep_posterior_sums), support_sumw)
    assert result.aggregate_noise_stats is not None
    np.testing.assert_allclose(float(result.aggregate_noise_stats.sumw), sum(support_sumw))


def test_local_k_class_uses_global_reconstruction_threshold(monkeypatch):
    import recovar.em.dense_single_volume.k_class as k_class_module

    dataset = type("Dataset", (), {"n_images": 1, "n_units": 1})()
    means = jnp.zeros((2, 4), dtype=jnp.complex64)
    mean_variance = jnp.ones((2, 4), dtype=jnp.float32)
    noise_variance = jnp.ones(4, dtype=jnp.float32)
    local_layout = LocalHypothesisLayout(
        n_global_rotations=1,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 1], dtype=np.int64),
        rotation_ids_flat=np.array([0], dtype=np.int32),
        rotations_flat=np.eye(3, dtype=np.float32)[None, :, :],
        rotation_log_priors_flat=np.zeros(1, dtype=np.float32),
        rotation_counts=np.array([1], dtype=np.int32),
        translation_grid=np.zeros((1, 2), dtype=np.float32),
        translation_log_priors=np.zeros((1, 1), dtype=np.float32),
    )
    class_masses = np.asarray([0.9997, 0.0003], dtype=np.float64)
    support_values = (
        (np.asarray([0.50015005, 0.30009003, 0.19005702, 0.00970291], dtype=np.float32),),
        (np.asarray([2.0 / 3.0, 1.0 / 3.0], dtype=np.float32),),
    )
    calls = []

    def fake_run_local_em_exact(*args, **kwargs):
        del args
        calls.append(kwargs)
        is_probe = kwargs.get("disable_adjoint_y", False)
        class_index = (sum(1 for call in calls if call.get("disable_adjoint_y", False)) - 1) if is_probe else (
            sum(1 for call in calls if not call.get("disable_adjoint_y", False)) - 1
        )
        stats = RelionStats(
            log_evidence_per_image=jnp.asarray([np.log(class_masses[class_index])], dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(1, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(1, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(1, dtype=jnp.float32),
        )
        base = (
            jnp.full(4, class_index + 1, dtype=jnp.complex64),
            jnp.full(4, class_index + 1, dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.int32),
            stats,
        )
        if kwargs.get("return_profile"):
            return base + ({"reconstruction_probability_values_by_image": support_values[class_index]},)
        return base

    monkeypatch.setattr(k_class_module, "run_local_em_exact", fake_run_local_em_exact)

    run_local_k_class_em(
        dataset,
        means,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=1,
        current_size=None,
        reconstruct_significant_only=True,
    )

    mstep_thresholds = [
        call.get("reconstruction_probability_threshold")
        for call in calls
        if not call.get("disable_adjoint_y", False)
    ]
    assert len(mstep_thresholds) == 2
    for threshold in mstep_thresholds:
        np.testing.assert_allclose(threshold, np.asarray([0.0097]), rtol=1e-3, atol=1e-6)


def test_local_search_iteration_k_class_returns_class_details(rng):
    dataset = MockDataset(2, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=161)
    means = jnp.stack([mean, mean], axis=0)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(2, seed=169)
    translations = np.zeros((1, 2), dtype=np.float32)
    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 0, 1], dtype=np.int32),
        rotations_flat=np.tile(np.asarray(local_rotations, dtype=np.float32), (2, 1, 1)),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((2, 1), dtype=np.float32),
    )

    outputs = iteration_loop_module._run_local_search_iteration(
        dataset,
        means,
        mean_variance,
        noise_variance,
        np.zeros((2, 3), dtype=np.float32),
        local_rotations,
        None,
        1,
        0.0,
        0.0,
        translations,
        np.zeros((2, 2), dtype=np.float32),
        1.0,
        None,
        "linear_interp",
        2,
        4,
        None,
        accumulate_noise=True,
        projection_padding_factor=1,
        reconstruction_padding_factor=1,
        half_spectrum_scoring=False,
        pass2_layout=local_layout,
        return_best_pose_details=True,
        class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        return_class_details=True,
    )

    (
        Ft_y,
        Ft_ctf,
        hard_assignment,
        best_rotations,
        best_translations,
        best_rotation_ids,
        stats,
        noise_stats,
        class_assignments_out,
        class_posterior_sums,
        class_full_posterior_sums,
    ) = outputs
    assert np.asarray(Ft_y).shape == (2, VOLUME_SIZE)
    assert np.asarray(Ft_ctf).shape == (2, VOLUME_SIZE)
    assert np.asarray(hard_assignment).shape == (2,)
    assert np.asarray(best_rotations).shape == (2, 3, 3)
    assert np.asarray(best_translations).shape == (2, 2)
    assert np.asarray(best_rotation_ids).shape == (2,)
    assert np.asarray(stats.log_evidence_per_image).shape == (2,)
    assert noise_stats is not None
    assert np.asarray(class_assignments_out).shape == (2,)
    assert np.asarray(class_posterior_sums).shape == (2,)
    np.testing.assert_allclose(np.sum(class_full_posterior_sums), dataset.n_images, rtol=5e-3, atol=1e-5)


def test_local_search_iteration_k_class_keeps_mstep_and_full_class_mass_separate(monkeypatch, rng):
    from types import SimpleNamespace

    import recovar.em.dense_single_volume.local_search_iteration as local_search_iteration

    dataset = MockDataset(2, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=171)
    means = jnp.stack([mean, mean], axis=0)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32)
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    local_rotations = _make_rotations(2, seed=173)
    translations = np.zeros((1, 2), dtype=np.float32)
    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 0, 1], dtype=np.int32),
        rotations_flat=np.tile(np.asarray(local_rotations, dtype=np.float32), (2, 1, 1)),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.asarray(translations, dtype=np.float32),
        translation_log_priors=np.zeros((2, 1), dtype=np.float32),
    )
    captured = {}

    def fake_run_local_k_class_em(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            Ft_y=jnp.zeros((2, VOLUME_SIZE), dtype=jnp.complex64),
            Ft_ctf=jnp.zeros((2, VOLUME_SIZE), dtype=jnp.float32),
            pose_assignments=np.zeros(2, dtype=np.int32),
            best_pose_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy(),
            best_pose_translations=np.zeros((2, 2), dtype=np.float32),
            best_pose_rotation_ids=np.zeros(2, dtype=np.int32),
            stats=RelionStats(
                log_evidence_per_image=jnp.zeros(2, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(2, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(2, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(2, dtype=jnp.float32),
            ),
            aggregate_noise_stats=NoiseStats(
                wsum_sigma2_noise=jnp.ones(1, dtype=jnp.float32),
                wsum_img_power=jnp.ones(1, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=2.0,
            ),
            class_assignments=np.array([0, 1], dtype=np.int32),
            class_posterior_sums=np.array([1.7, 0.3], dtype=np.float32),
            class_mstep_posterior_sums=np.array([1.2, 0.8], dtype=np.float32),
        )

    monkeypatch.setattr(local_search_iteration, "run_local_k_class_em", fake_run_local_k_class_em)

    outputs = iteration_loop_module._run_local_search_iteration(
        dataset,
        means,
        mean_variance,
        noise_variance,
        np.zeros((2, 3), dtype=np.float32),
        local_rotations,
        None,
        1,
        0.0,
        0.0,
        translations,
        np.zeros((2, 2), dtype=np.float32),
        1.0,
        None,
        "linear_interp",
        2,
        4,
        None,
        accumulate_noise=True,
        projection_padding_factor=1,
        reconstruction_padding_factor=1,
        half_spectrum_scoring=False,
        pass2_layout=local_layout,
        return_best_pose_details=True,
        reconstruct_significant_only=True,
        stats_use_reconstruction_probs=True,
        class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        return_class_details=True,
    )

    assert captured["class_posterior_sums_from_noise"] is True
    np.testing.assert_allclose(np.asarray(outputs[-2]), [1.2, 0.8], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(outputs[-1]), [1.7, 0.3], rtol=1e-6, atol=1e-6)


def test_native_half_preprocess_requires_mask_for_masked_score(rng):
    dataset = RawRealImageDataset(1, rng)

    with pytest.raises(ValueError, match="score_with_masked_images=True requires an image mask"):
        resolve_image_mask_for_half_preprocess(
            dataset,
            dataset.image_shape,
            require_mask=True,
        )


def test_native_half_preprocess_rejects_unsupported_mask_mode(rng):
    dataset = MockDataset(1, rng)
    dataset.image_source.backend.image_mask_mode = "unsupported"

    with pytest.raises(ValueError, match="Unsupported image_mask_mode"):
        resolve_image_mask_for_half_preprocess(
            dataset,
            dataset.image_shape,
            require_mask=True,
        )


def test_weighted_abs2_on_demand_scores_match_materialized(rng):
    batch_size = 2
    n_rot = 3
    n_trans = 2
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    shifted = (
        rng.standard_normal((batch_size, n_trans, n_half)) + 1j * rng.standard_normal((batch_size, n_trans, n_half))
    ).astype(np.complex64)
    proj = (
        rng.standard_normal((batch_size, n_rot, n_half)) + 1j * rng.standard_normal((batch_size, n_rot, n_half))
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
        current_size=None,
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
    assert np.asarray(Ft_y_exact).shape == np.asarray(Ft_y_dense).shape
    assert np.asarray(Ft_ctf_exact).shape == np.asarray(Ft_ctf_dense).shape
    assert np.all(np.isfinite(np.asarray(Ft_y_exact)))
    assert np.all(np.isfinite(np.asarray(Ft_ctf_exact)))
    assert np.all(np.isfinite(np.asarray(stats_exact.log_evidence_per_image)))
    assert np.all(np.isfinite(np.asarray(stats_exact.best_log_score_per_image)))
    assert np.all(np.asarray(stats_exact.max_posterior_per_image) <= 1.0)
    np.testing.assert_allclose(
        np.sum(np.asarray(stats_exact.rotation_posterior_sums)),
        np.array(1.0, dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    assert np.all(np.isfinite(np.asarray(noise_exact.wsum_sigma2_noise)))
    assert np.all(np.isfinite(np.asarray(noise_exact.wsum_img_power)))


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


def test_run_local_em_exact_default_path_matches_debug_split_path(monkeypatch, rng, tmp_path):
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
        group_ids=np.array([0, 1, 0], dtype=np.int64),
        image_pre_shifts=np.array([[0.5, -1.0], [-1.0, 1.25], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
    )

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", raising=False)
    default = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", str(tmp_path / "score_dump"))
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", "0")
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", "1")
    split = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    Ft_y_default, Ft_ctf_default, hard_default, stats_default, noise_default, profile_default = default
    Ft_y_split, Ft_ctf_split, hard_split, stats_split, noise_split, profile_split = split
    assert int(profile_default["big_jit_bucket_count"]) > 0
    assert int(profile_split["big_jit_bucket_count"]) == 0
    assert bool(profile_default["fused_score_mstep_enabled"]) is True
    assert bool(profile_split["fused_score_mstep_enabled"]) is True
    np.testing.assert_array_equal(hard_default, hard_split)
    np.testing.assert_allclose(np.asarray(Ft_y_default), np.asarray(Ft_y_split), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_default), np.asarray(Ft_ctf_split), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(stats_default.log_evidence_per_image),
        np.asarray(stats_split.log_evidence_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_default.max_posterior_per_image),
        np.asarray(stats_split.max_posterior_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_default.rotation_posterior_sums),
        np.asarray(stats_split.rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_default.wsum_sigma2_noise),
        np.asarray(noise_split.wsum_sigma2_noise),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_default.wsum_norm_correction is not None
    assert noise_split.wsum_norm_correction is not None
    np.testing.assert_allclose(
        np.asarray(noise_default.wsum_norm_correction),
        np.asarray(noise_split.wsum_norm_correction),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_default.wsum_scale_correction_xa is not None
    assert noise_default.wsum_scale_correction_aa is not None
    assert noise_split.wsum_scale_correction_xa is not None
    assert noise_split.wsum_scale_correction_aa is not None
    assert np.asarray(noise_default.wsum_scale_correction_xa).shape == (2,)
    np.testing.assert_allclose(
        np.asarray(noise_default.wsum_scale_correction_xa),
        np.asarray(noise_split.wsum_scale_correction_xa),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_default.wsum_scale_correction_aa),
        np.asarray(noise_split.wsum_scale_correction_aa),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_default.sumw == pytest.approx(noise_split.sumw, abs=1e-5)


def test_run_local_em_exact_big_jit_bucket_matches_debug_split(monkeypatch, rng, tmp_path):
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
        return_reconstruction_sample_indices=True,
    )

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", raising=False)
    big = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", str(tmp_path / "score_dump"))
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", "0")
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", "1")
    split = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    Ft_y_big, Ft_ctf_big, hard_big, stats_big, noise_big, profile_big = big
    Ft_y_split, Ft_ctf_split, hard_split, stats_split, noise_split, profile_split = split
    assert int(profile_big["big_jit_bucket_count"]) == 1
    assert int(profile_split["big_jit_bucket_count"]) == 0
    np.testing.assert_array_equal(hard_big, hard_split)
    np.testing.assert_allclose(np.asarray(Ft_y_big), np.asarray(Ft_y_split), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_big), np.asarray(Ft_ctf_split), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(stats_big.log_evidence_per_image),
        np.asarray(stats_split.log_evidence_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_big.max_posterior_per_image),
        np.asarray(stats_split.max_posterior_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_big.rotation_posterior_sums),
        np.asarray(stats_split.rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_big.wsum_sigma2_noise),
        np.asarray(noise_split.wsum_sigma2_noise),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_big.wsum_norm_correction is not None
    assert noise_split.wsum_norm_correction is not None
    np.testing.assert_allclose(
        np.asarray(noise_big.wsum_norm_correction),
        np.asarray(noise_split.wsum_norm_correction),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_big.sumw == pytest.approx(noise_split.sumw, abs=1e-5)
    for sample_big, sample_split in zip(
        profile_big["reconstruction_sample_indices_by_image"],
        profile_split["reconstruction_sample_indices_by_image"],
        strict=True,
    ):
        np.testing.assert_array_equal(sample_big, sample_split)


def test_run_local_em_exact_score_only_big_jit_matches_debug_split(monkeypatch, rng, tmp_path):
    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=561)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(5, seed=563)
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
        accumulate_noise=False,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        image_pre_shifts=np.array([[0.25, -0.5], [-0.75, 0.5], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        score_only=True,
    )

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", raising=False)
    big = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", str(tmp_path / "score_dump"))
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", "0")
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", "1")
    split = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    Ft_y_big, Ft_ctf_big, hard_big, stats_big, profile_big = big
    Ft_y_split, Ft_ctf_split, hard_split, stats_split, profile_split = split
    assert int(profile_big["big_jit_bucket_count"]) == 1
    assert int(profile_split["big_jit_bucket_count"]) == 0
    assert bool(profile_big["score_only"]) is True
    assert bool(profile_split["score_only"]) is True
    assert np.asarray(Ft_y_big).shape == (1,)
    assert np.asarray(Ft_ctf_big).shape == (1,)
    assert np.asarray(Ft_y_split).shape == (1,)
    assert np.asarray(Ft_ctf_split).shape == (1,)
    np.testing.assert_array_equal(hard_big, hard_split)
    np.testing.assert_allclose(np.asarray(Ft_y_big), np.asarray(Ft_y_split), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(Ft_ctf_big), np.asarray(Ft_ctf_split), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(stats_big.log_evidence_per_image),
        np.asarray(stats_split.log_evidence_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_big.max_posterior_per_image),
        np.asarray(stats_split.max_posterior_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_big.rotation_posterior_sums),
        np.asarray(stats_split.rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )


def test_local_score_debug_dump_defaults_to_big_jit(monkeypatch, rng, tmp_path):
    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=562)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(5, seed=564)
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

    score_dump_dir = tmp_path / "score_dump"
    fused_dump_dir = tmp_path / "fused_dump"
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", str(score_dump_dir))
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", "0")
    monkeypatch.setenv("RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR", str(fused_dump_dir))
    monkeypatch.setenv("RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES", "0")
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_OPERANDS", raising=False)

    result = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=3,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=False,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        image_pre_shifts=np.array([[0.25, -0.5], [-0.75, 0.5], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        score_only=True,
    )

    _, _, _, _, profile = result
    assert int(profile["big_jit_bucket_count"]) == 1
    assert int(profile["big_jit_debug_bucket_count"]) == 1
    score_dumps = sorted(score_dump_dir.glob("local_score_it*_image_0*.npz"))
    assert len(score_dumps) == 1
    with np.load(score_dumps[0]) as dump:
        assert dump["pass2_scores_total"].shape == (1, 3, 2)
        assert dump["posterior"].shape == (1, 3, 2)
        assert dump["best_score"].shape == (1,)
    fused_dumps = sorted(fused_dump_dir.glob("local_fused_posterior_it*_image_0*.npz"))
    assert len(fused_dumps) == 1
    with np.load(fused_dumps[0]) as dump:
        assert dump["posterior"].shape == (1, 3, 2)
        assert dump["best_score"].shape == (1,)


def test_local_score_debug_dump_operands_stay_on_big_jit(monkeypatch, rng, tmp_path):
    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=563)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(5, seed=565)
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
        accumulate_noise=False,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        image_pre_shifts=np.array([[0.25, -0.5], [-0.75, 0.5], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
    )

    big_dump_dir = tmp_path / "score_dump_big"
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", str(big_dump_dir))
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", "0")
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_OPERANDS", "1")
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", raising=False)
    big = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    split_dump_dir = tmp_path / "score_dump_split"
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", str(split_dump_dir))
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", "1")
    split = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    profile_big = big[-1]
    profile_split = split[-1]
    assert int(profile_big["big_jit_bucket_count"]) == 1
    assert int(profile_big["big_jit_debug_bucket_count"]) == 1
    assert int(profile_split["big_jit_bucket_count"]) == 0

    big_dump = next(big_dump_dir.glob("local_score_it*_image_0*.npz"))
    split_dump = next(split_dump_dir.glob("local_score_it*_image_0*.npz"))
    operand_keys = [
        "debug_shifted_score",
        "debug_shifted_recon",
        "debug_ctf2_over_nv",
        "debug_ctf2_over_nv_recon",
        "debug_proj_weighted",
        "debug_proj_for_recon",
    ]
    with np.load(big_dump) as big_npz, np.load(split_dump) as split_npz:
        for key in operand_keys:
            assert key in big_npz.files
            assert key in split_npz.files
            np.testing.assert_allclose(big_npz[key], split_npz[key], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(
            big_npz["pass2_scores_total"],
            split_npz["pass2_scores_total"],
            atol=2e-5,
            rtol=2e-5,
        )


def test_local_score_debug_dump_only_materializes_target_bucket(monkeypatch, rng, tmp_path):
    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=566)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(5, seed=567)
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

    score_dump_dir = tmp_path / "score_dump"
    fused_dump_dir = tmp_path / "fused_dump"
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", str(score_dump_dir))
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", "2")
    monkeypatch.setenv("RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR", str(fused_dump_dir))
    monkeypatch.setenv("RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES", "2")
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_OPERANDS", raising=False)

    result = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=False,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        image_pre_shifts=np.array([[0.25, -0.5], [-0.75, 0.5], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        score_only=True,
    )

    _, _, _, _, profile = result
    assert int(profile["big_jit_bucket_count"]) > 1
    assert int(profile["big_jit_debug_bucket_count"]) == 1
    assert sorted(path.name for path in score_dump_dir.glob("local_score_it*_image_*.npz")) == [
        "local_score_it-01_image_2.npz",
    ]
    assert sorted(path.name for path in fused_dump_dir.glob("local_fused_posterior_it*_image_*.npz")) == [
        "local_fused_posterior_it-01_image_2.npz",
    ]


def test_local_score_debug_force_split_only_splits_target_bucket(monkeypatch, rng, tmp_path):
    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=568)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(5, seed=570)
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
    score_dump_dir = tmp_path / "score_dump"
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", str(score_dump_dir))
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", "2")
    monkeypatch.setenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", "1")
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_OPERANDS", raising=False)

    result = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=1,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=False,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        image_pre_shifts=np.array([[0.25, -0.5], [-0.75, 0.5], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        score_only=True,
    )

    _, _, _, _, profile = result
    assert int(profile["n_chunks"]) > 1
    assert int(profile["big_jit_bucket_count"]) == int(profile["n_chunks"]) - 1
    assert int(profile["big_jit_debug_bucket_count"]) == 0
    assert sorted(path.name for path in score_dump_dir.glob("local_score_it*_image_*.npz")) == [
        "local_score_it-01_image_2.npz",
    ]


def test_local_score_debug_recon_projection_materialization_is_bucket_scoped():
    src = inspect.getsource(run_local_em_exact)
    require_block = src[
        src.index("require_materialized_recon_projection = bool(") :
        src.index("can_defer_local_noise_projection = (")
    ]
    assert "debug_score_dump_force_split" not in require_block
    assert "debug_score_dump_operands" not in require_block
    assert "need_local_recon_projection_for_bucket = bool(" in src
    assert "materialize_recon_projection=need_local_recon_projection_for_bucket" in src
    bucket_block = src[
        src.index("need_local_recon_projection_for_bucket = bool(") :
        src.index("translation_sqdist_ang = None")
    ]
    assert "debug_score_dump_operands" in bucket_block
    assert "debug_score_dump_force_split" not in bucket_block
    assert "if debug_score_dump_force_split and debug_score_dump_bucket_matches:" in src
    assert "jax.clear_caches()" in src


def test_local_fused_posterior_debug_does_not_request_scores_without_score_dump():
    src = inspect.getsource(run_local_em_exact)
    assert "return_big_jit_debug_scores = bool(score_debug_bucket_matches)" in src
    assert "return_debug_scores=return_big_jit_debug_scores" in src
    assert "if return_big_jit_debug_scores" in src
    assert "if fused_debug_bucket_matches and debug_fused_posterior_dump_targets:" in src
    assert "if score_debug_bucket_matches and debug_score_dump_targets:" in src


def test_local_big_jit_windowed_translation_slices_before_tiling():
    from recovar.em.dense_single_volume import local_big_jit

    src = inspect.getsource(local_big_jit.run_local_bucket_big_jit)
    assert "def _translate_weighted_half_window" in src
    assert "processed_half[:, pixel_indices]" in src
    assert "translation_phases_half[:, pixel_indices]" in src
    assert "shifted_half[:, window_indices]" not in src
    assert "shifted_recon_half[:, recon_window_indices]" not in src


def test_run_local_em_exact_windowed_relion_projector_big_jit_matches_split(monkeypatch, rng):
    from recovar.core.relion_project import centered_full_to_relion_half

    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=565)
    relion_projector_half = centered_full_to_relion_half(mean.reshape(VOLUME_SHAPE))[None]
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(5, seed=567)
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
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=4,
    )

    monkeypatch.delenv("RECOVAR_DISABLE_LOCAL_BIG_JIT", raising=False)
    big = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    monkeypatch.setenv("RECOVAR_DISABLE_LOCAL_BIG_JIT", "1")
    split = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    Ft_y_big, Ft_ctf_big, hard_big, stats_big, noise_big, profile_big = big
    Ft_y_split, Ft_ctf_split, hard_split, stats_split, noise_split, profile_split = split
    assert int(profile_big["big_jit_bucket_count"]) == 1
    assert int(profile_split["big_jit_bucket_count"]) == 0
    assert profile_big["projection_mode"].item() == "relion_projector"
    assert profile_split["projection_mode"].item() == "relion_projector"
    np.testing.assert_array_equal(hard_big, hard_split)
    np.testing.assert_allclose(np.asarray(Ft_y_big), np.asarray(Ft_y_split), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_big), np.asarray(Ft_ctf_split), atol=2e-5, rtol=2e-5)
    _assert_relion_stats_allclose(stats_big, stats_split)
    _assert_noise_stats_allclose(noise_big, noise_split)


def test_run_local_em_exact_relion_projection_cache_matches_uncached_big_jit(monkeypatch, rng):
    from recovar.core.relion_project import centered_full_to_relion_half

    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=568)
    relion_projector_half = centered_full_to_relion_half(mean.reshape(VOLUME_SHAPE))[None]
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(5, seed=569)
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
        accumulate_noise=False,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        image_pre_shifts=np.array([[0.25, -0.5], [-0.75, 0.5], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=4,
    )

    monkeypatch.delenv("RECOVAR_DISABLE_LOCAL_BIG_JIT", raising=False)
    monkeypatch.setenv(EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV, "1")
    cached = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    monkeypatch.setenv(EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB_ENV, "0")
    uncached = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    Ft_y_cached, Ft_ctf_cached, hard_cached, stats_cached, profile_cached = cached
    Ft_y_uncached, Ft_ctf_uncached, hard_uncached, stats_uncached, profile_uncached = uncached
    assert int(profile_cached["big_jit_bucket_count"]) == 1
    assert bool(profile_cached["relion_projection_cache_enabled"])
    assert int(profile_cached["relion_projection_cache_rows"]) == all_rotations.shape[0]
    assert not bool(profile_uncached["relion_projection_cache_enabled"])
    np.testing.assert_array_equal(hard_cached, hard_uncached)
    np.testing.assert_allclose(np.asarray(Ft_y_cached), np.asarray(Ft_y_uncached), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_cached), np.asarray(Ft_ctf_uncached), atol=2e-5, rtol=2e-5)
    _assert_relion_stats_allclose(stats_cached, stats_uncached)


def _sparse_big_jit_local_case(rng):
    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=571)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(6, seed=573)
    translations = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    rotation_ids = [
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array([0, 2, 4, 5], dtype=np.int32),
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
    return dataset, mean, mean_variance, noise_variance, local_layout


def _assert_relion_stats_allclose(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual.log_evidence_per_image),
        np.asarray(expected.log_evidence_per_image),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual.best_log_score_per_image),
        np.asarray(expected.best_log_score_per_image),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual.max_posterior_per_image),
        np.asarray(expected.max_posterior_per_image),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual.rotation_posterior_sums),
        np.asarray(expected.rotation_posterior_sums),
        rtol=1e-5,
        atol=1e-6,
    )


def _assert_noise_stats_allclose(actual, expected):
    assert actual is not None
    assert expected is not None
    for field in actual._fields:
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if actual_value is None or expected_value is None:
            assert actual_value is None
            assert expected_value is None
            continue
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=1e-5,
            atol=1e-6,
        )


def test_local_bucket_preserves_distinct_mstep_rotations(rng):
    _, _, _, _, local_layout = _sparse_big_jit_local_case(rng)
    distinct_mstep_rotations = np.roll(np.asarray(local_layout.rotations_flat), 1, axis=0)
    distinct_layout = replace(local_layout, mstep_rotations_flat=distinct_mstep_rotations)

    buckets = bucket_local_hypothesis_layout(
        distinct_layout,
        image_batch_size=2,
        rotation_block_size=8,
        max_hypotheses_per_microbatch=10_000,
    )

    for bucket in buckets:
        assert bucket.local_mstep_rotations is not None
        for row, image_idx in enumerate(np.asarray(bucket.image_indices, dtype=np.int64)):
            count = int(bucket.actual_rotation_counts[row])
            start = int(distinct_layout.rotation_offsets[image_idx])
            stop = start + count
            np.testing.assert_array_equal(
                np.asarray(bucket.local_rotations[row, :count]),
                np.asarray(distinct_layout.rotations_flat[start:stop]),
            )
            np.testing.assert_array_equal(
                np.asarray(bucket.local_mstep_rotations[row, :count]),
                distinct_mstep_rotations[start:stop],
            )

    reordered = _reorder_bucket_to_indices(buckets[0], np.asarray(buckets[0].image_indices)[::-1])
    np.testing.assert_array_equal(reordered.local_rotations, np.asarray(buckets[0].local_rotations)[::-1])
    np.testing.assert_array_equal(
        reordered.local_mstep_rotations,
        np.asarray(buckets[0].local_mstep_rotations)[::-1],
    )


@pytest.mark.parametrize("route", ["in_kernel_big_jit", "sparse_big_jit", "deferred_big_jit", "split"])
def test_local_mstep_rotation_override_changes_only_adjoint_outputs(monkeypatch, rng, route):
    dataset, mean, mean_variance, noise_variance, local_layout = _sparse_big_jit_local_case(rng)
    distinct_layout = replace(
        local_layout,
        mstep_rotations_flat=np.roll(np.asarray(local_layout.rotations_flat), 1, axis=0),
    )
    kwargs = dict(
        image_batch_size=3,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=False,
        reconstruct_significant_only=route != "in_kernel_big_jit",
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        max_significants=-1,
    )

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", raising=False)
    monkeypatch.delenv("RECOVAR_DISABLE_LOCAL_BIG_JIT", raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP_ENV, raising=False)
    monkeypatch.delenv(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, raising=False)
    if route == "sparse_big_jit":
        monkeypatch.setenv(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, "1000")
    elif route == "deferred_big_jit":
        monkeypatch.setenv(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, "0")
    elif route == "split":
        monkeypatch.setenv("RECOVAR_DISABLE_LOCAL_BIG_JIT", "1")

    baseline = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **kwargs,
    )
    overridden = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        distinct_layout,
        "linear_interp",
        **kwargs,
    )

    Ft_y_baseline, Ft_ctf_baseline, hard_baseline, stats_baseline, profile_baseline = baseline
    Ft_y_overridden, Ft_ctf_overridden, hard_overridden, stats_overridden, profile_overridden = overridden
    if route == "split":
        assert int(profile_baseline["big_jit_bucket_count"]) == 0
        assert int(profile_overridden["big_jit_bucket_count"]) == 0
    else:
        assert int(profile_baseline["big_jit_bucket_count"]) > 0
        assert int(profile_overridden["big_jit_bucket_count"]) > 0
    np.testing.assert_array_equal(hard_overridden, hard_baseline)
    _assert_relion_stats_allclose(stats_overridden, stats_baseline)
    assert np.max(np.abs(np.asarray(Ft_y_overridden) - np.asarray(Ft_y_baseline))) > 1e-6
    assert np.max(np.abs(np.asarray(Ft_ctf_overridden) - np.asarray(Ft_ctf_baseline))) > 1e-6


def test_run_local_em_exact_significant_support_uses_sparse_big_jit_packed_backprojection(monkeypatch, rng):
    dataset, mean, mean_variance, noise_variance, local_layout = _sparse_big_jit_local_case(rng)

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", raising=False)
    outputs = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        image_batch_size=3,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=True,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        max_significants=-1,
    )

    profile = outputs[-1]
    assert int(profile["big_jit_bucket_count"]) > 0
    assert int(profile["sparse_big_jit_bucket_count"]) > 0
    assert int(profile["sum_reconstruction_rows"]) < int(profile["sum_padded_rows"])


def test_run_local_em_exact_over_cap_significant_support_defaults_to_deferred_big_jit(monkeypatch, rng):
    dataset, mean, mean_variance, noise_variance, local_layout = _sparse_big_jit_local_case(rng)
    kwargs = dict(
        image_batch_size=3,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=True,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        max_significants=-1,
    )

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP_ENV, raising=False)
    monkeypatch.setenv(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, "1000")
    sparse = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **kwargs,
    )

    monkeypatch.setenv(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, "0")
    deferred = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **kwargs,
    )

    profile = deferred[-1]
    assert int(profile["big_jit_bucket_count"]) > 0
    assert int(profile["sparse_big_jit_bucket_count"]) > 0
    np.testing.assert_array_equal(deferred[2], sparse[2])
    np.testing.assert_allclose(np.asarray(deferred[0]), np.asarray(sparse[0]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(deferred[1]), np.asarray(sparse[1]), rtol=1e-5, atol=1e-6)
    _assert_relion_stats_allclose(deferred[3], sparse[3])
    _assert_noise_stats_allclose(deferred[4], sparse[4])


def test_run_local_em_exact_deferred_big_jit_no_noise_matches_sparse_big_jit(monkeypatch, rng):
    dataset, mean, mean_variance, noise_variance, local_layout = _sparse_big_jit_local_case(rng)
    kwargs = dict(
        image_batch_size=3,
        rotation_block_size=8,
        current_size=6,
        accumulate_noise=False,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        max_significants=-1,
    )

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", raising=False)
    monkeypatch.delenv(EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP_ENV, raising=False)
    monkeypatch.setenv(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, "1000")
    sparse = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **kwargs,
    )

    monkeypatch.setenv(EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, "0")
    deferred = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **kwargs,
    )

    sparse_profile = sparse[-1]
    deferred_profile = deferred[-1]
    assert int(sparse_profile["big_jit_bucket_count"]) > 0
    assert int(deferred_profile["big_jit_bucket_count"]) > 0
    assert int(sparse_profile["sparse_big_jit_bucket_count"]) > 0
    assert int(deferred_profile["sparse_big_jit_bucket_count"]) > 0
    assert len(sparse) == 5
    assert len(deferred) == 5
    np.testing.assert_array_equal(deferred[2], sparse[2])
    np.testing.assert_allclose(np.asarray(deferred[0]), np.asarray(sparse[0]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(deferred[1]), np.asarray(sparse[1]), rtol=1e-5, atol=1e-6)
    _assert_relion_stats_allclose(deferred[3], sparse[3])


def test_run_local_em_exact_processed_half_cache_matches_uncached_split(monkeypatch, rng):
    dataset = MockDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=581)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(6, seed=583)
    translations = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    rotation_ids = [
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array([0, 2, 4, 5], dtype=np.int32),
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
        current_size=None,
        accumulate_noise=True,
        reconstruct_significant_only=True,
        return_profile=True,
        score_with_masked_images=True,
        half_spectrum_scoring=False,
        image_pre_shifts=np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        max_significants=-1,
    )

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES", raising=False)
    monkeypatch.setenv(EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV, "0")
    uncached = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )
    monkeypatch.setenv(EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB_ENV, "1")
    cached = run_local_em_exact(
        dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        "linear_interp",
        **common_kwargs,
    )

    Ft_y_uncached, Ft_ctf_uncached, hard_uncached, stats_uncached, noise_uncached, profile_uncached = uncached
    Ft_y_cached, Ft_ctf_cached, hard_cached, stats_cached, noise_cached, profile_cached = cached
    assert bool(profile_uncached["processed_half_cache_enabled"]) is False
    assert bool(profile_cached["processed_half_cache_enabled"]) is True
    np.testing.assert_array_equal(hard_uncached, hard_cached)
    np.testing.assert_allclose(np.asarray(Ft_y_uncached), np.asarray(Ft_y_cached), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(Ft_ctf_uncached), np.asarray(Ft_ctf_cached), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(stats_uncached.log_evidence_per_image),
        np.asarray(stats_cached.log_evidence_per_image),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(stats_uncached.rotation_posterior_sums),
        np.asarray(stats_cached.rotation_posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(noise_uncached.wsum_sigma2_noise),
        np.asarray(noise_cached.wsum_sigma2_noise),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_uncached.wsum_norm_correction is not None
    assert noise_cached.wsum_norm_correction is not None
    np.testing.assert_allclose(
        np.asarray(noise_uncached.wsum_norm_correction),
        np.asarray(noise_cached.wsum_norm_correction),
        atol=1e-5,
        rtol=1e-5,
    )
    assert noise_uncached.sumw == pytest.approx(noise_cached.sumw, abs=1e-5)


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

    np.testing.assert_array_equal(np.asarray(n_sig), np.array([4], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(sig_samples),
        np.array([[[True, True], [True, True]]]),
    )
    np.testing.assert_array_equal(
        np.asarray(sig_rots),
        np.array([[True, True]]),
    )


def test_compute_reconstruction_support_from_global_threshold_drops_low_class_tail():
    probs = jnp.asarray(
        [
            [
                [0.0002, 0.0001],
                [0.0, 0.0],
            ],
            [
                [0.50, 0.30],
                [0.19, 0.0097],
            ],
        ],
        dtype=jnp.float32,
    )
    thresholds = jnp.asarray([0.0097, 0.0097], dtype=jnp.float32)

    sig_samples, sig_rots, n_sig = compute_reconstruction_support_from_threshold(probs, thresholds)

    np.testing.assert_array_equal(np.asarray(n_sig), np.array([0, 4], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(sig_samples[0]), np.zeros((2, 2), dtype=bool))
    np.testing.assert_array_equal(np.asarray(sig_rots[0]), np.array([False, False]))
    np.testing.assert_array_equal(np.asarray(sig_rots[1]), np.array([True, True]))


def test_tracked_local_engine_todo_ids_are_resolved():
    repo_root = Path(__file__).resolve().parents[2]
    iteration_loop_path = repo_root / "recovar" / "em" / "dense_single_volume" / "iteration_loop.py"
    em_engine_path = repo_root / "recovar" / "em" / "dense_single_volume" / "em_engine.py"
    half_spectrum_path = repo_root / "recovar" / "em" / "dense_single_volume" / "helpers" / "half_spectrum.py"
    docs_path = repo_root / "docs" / "relion_local_engine_refactor.md"

    iteration_text = iteration_loop_path.read_text(encoding="utf-8")
    em_engine_text = em_engine_path.read_text(encoding="utf-8")
    half_spectrum_text = half_spectrum_path.read_text(encoding="utf-8")
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
        assert f"`{todo_id}` | RESOLVED" in docs_text

    active_code = "\n".join([iteration_text, em_engine_text, half_spectrum_text])
    assert "TODO(RELION_LOCAL_ENGINE" not in active_code
    assert "TODO(DENSE_ENGINE_BOUNDARY" not in active_code
    assert "TODO(RELION-parity-debt" not in active_code


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


# ===========================================================================
# Test 1: RELION-parity smoke test -- runs without error
# ===========================================================================


class TestRelionModeSmokeTest:
    """Call refine_single_volume and verify it runs."""

    def test_relion_bootstrap_current_size_matches_benchmark_case(self):
        """128px, 4.25A/px, ini_high=30A should bootstrap from 36 -> 56."""
        assert _bootstrap_current_size_relion(36, 128) == 56

    def test_relion_bootstrap_current_size_from_ini_high_matches_benchmark_case(self):
        assert bootstrap_current_size_from_ini_high_relion(128, 4.25, 30.0) == 56

    def test_firstiter_cc_ini_high_tau2_taper_matches_relion_squared_cosine(self):
        taper = iteration_loop_module._firstiter_cc_ini_high_tau2_taper(
            65,
            128,
            4.25,
            30.0,
            filter_edgewidth=2,
        )

        radius = 128 * 4.25 / 30.0 - 1.0
        radius_p = radius + 2.0
        expected18 = (0.5 - 0.5 * np.cos(np.pi * (radius_p - 18.0) / 2.0)) ** 2
        expected19 = (0.5 - 0.5 * np.cos(np.pi * (radius_p - 19.0) / 2.0)) ** 2
        assert taper[17] == 1.0
        np.testing.assert_allclose(taper[18], expected18, rtol=0, atol=1e-15)
        np.testing.assert_allclose(taper[19], expected19, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(taper[20:], 0.0)

    def test_firstiter_cc_reconstructs_before_tau2_reporting_taper(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """The ini_high tau2 taper changes reported state, not reconstruction."""
        from recovar.em.dense_single_volume import mean_helpers as mean_helpers_module

        untapered_tau = [7.0, 11.0]
        taper = np.asarray([1.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
        tau2_call = 0
        reconstruction_tau = []

        def fake_tau2_from_weights(*_args, **_kwargs):
            nonlocal tau2_call
            value = untapered_tau[tau2_call]
            tau2_call += 1
            shells = jnp.full(taper.shape, value, dtype=jnp.float32)
            details = {
                "prior_shells": shells,
                "sigma2_shells": jnp.ones_like(shells),
                "avg_weight_shells": jnp.ones_like(shells),
                "shell_sum": jnp.ones_like(shells),
                "shell_count": jnp.ones_like(shells),
                "fsc_shells": jnp.ones_like(shells),
                "ssnr_shells": shells,
            }
            return jnp.full(VOLUME_SIZE, value, dtype=jnp.float32), shells, details

        def fake_reconstruct(*_args, **kwargs):
            reconstruction_tau.append(np.asarray(kwargs["tau"]))
            return jnp.ones(VOLUME_SIZE, dtype=jnp.complex64)

        monkeypatch.setattr(
            regularization_module,
            "compute_relion_tau2_from_weights",
            fake_tau2_from_weights,
        )
        monkeypatch.setattr(
            iteration_loop_module,
            "_firstiter_cc_ini_high_tau2_taper",
            lambda *_args, **_kwargs: taper,
        )
        monkeypatch.setattr(mean_helpers_module, "_reconstruct_volume_eager", fake_reconstruct)
        monkeypatch.setattr(
            mean_helpers_module,
            "_apply_relion_initial_lowpass_filter",
            lambda volume, *_args, **_kwargs: volume,
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
            init_current_size=4,
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
            emulate_relion_firstiter_cc=True,
            relion_firstiter_ini_high_angstrom=8.0,
            skip_final_iteration=True,
        )

        assert len(reconstruction_tau) == 2
        np.testing.assert_array_equal(reconstruction_tau[0], untapered_tau[0])
        np.testing.assert_array_equal(reconstruction_tau[1], untapered_tau[1])
        np.testing.assert_allclose(
            result["tau2_radial_trajectory"][0],
            untapered_tau[0] * taper,
            rtol=0.0,
            atol=0.0,
        )

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

    def test_clamp_relion_coarse_image_size_allows_small_even_sizes(self):
        """RELION allows adaptive coarse sizes below the generic current-size floor."""
        coarse_size = compute_coarse_image_size(
            30.0,
            4.25,
            128,
            particle_diameter=380.0,
        )
        assert coarse_size == 14
        assert clamp_relion_coarse_image_size(coarse_size, current_size=44, ori_size=128) == 14

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

    def test_normalize_direction_prior_per_half_keeps_shared_prior(self):
        shared = np.array([0.7, 0.3], dtype=np.float32)

        got = normalize_direction_prior_per_half(shared)

        np.testing.assert_allclose(got[0], shared)
        np.testing.assert_allclose(got[1], shared)
        assert got[0] is not got[1]

    def test_normalize_noise_variance_per_half_keeps_shared_noise(self):
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

    def test_combined_noise_stats_sums_half_sufficient_statistics(self):
        """Class3D combines half accumulators before one RELION sigma2 update."""
        stats0 = NoiseStats(
            wsum_sigma2_noise=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
            wsum_sigma2_offset=7.0,
            sumw=11.0,
            wsum_noise_a2=jnp.array([0.5, 1.0, 1.5], dtype=jnp.float32),
        )
        stats1 = NoiseStats(
            wsum_sigma2_noise=jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([40.0, 50.0, 60.0], dtype=jnp.float32),
            wsum_sigma2_offset=70.0,
            sumw=13.0,
            wsum_noise_xa=jnp.array([2.0, 4.0, 6.0], dtype=jnp.float32),
        )

        got = _combined_noise_stats([stats0, stats1])

        np.testing.assert_allclose(np.asarray(got.wsum_sigma2_noise), [11.0, 22.0, 33.0])
        np.testing.assert_allclose(np.asarray(got.wsum_img_power), [44.0, 55.0, 66.0])
        assert got.wsum_sigma2_offset == pytest.approx(77.0)
        assert got.sumw == pytest.approx(24.0)
        np.testing.assert_allclose(np.asarray(got.wsum_noise_a2), [0.5, 1.0, 1.5])
        np.testing.assert_allclose(np.asarray(got.wsum_noise_xa), [2.0, 4.0, 6.0])

    def test_relion_norm_scale_update_matches_relion_formula(self):
        """Native updater reconstructs RELION's normcorr/group-scale state."""
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=4.0,
            wsum_norm_correction=jnp.array([2.0, 8.0, 18.0, 0.5], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([1.0, 100.0, 1.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([1.0, 10.0, 0.0], dtype=jnp.float32),
        )
        group_ids = np.array([0, 1, 1, 2], dtype=np.int64)
        old_group_scale = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        old_scale = old_group_scale[group_ids]
        old_image_corr = np.array([2.0, 1.0, 4.0, 4.0], dtype=np.float64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            image_corrections_per_half=[old_image_corr, old_image_corr],
            scale_corrections_per_half=[old_scale, old_scale],
            group_ids_per_half=[group_ids, group_ids],
            group_scale_corrections_per_half=[old_group_scale, old_group_scale],
        )

        expected_normcorr = np.array([1.0, 8.0, 3.0, 1.0], dtype=np.float64)
        expected_avg_norm = float(np.mean(expected_normcorr))
        scale_target = np.array([1.0, 10.0, 1.0], dtype=np.float64)
        clipped = np.array([1.0, 5.0, 1.0], dtype=np.float64)
        expected_group_scale = clipped / ((1.0 * clipped[0] + 2.0 * clipped[1] + clipped[2]) / 4.0)
        expected_scale = expected_group_scale[group_ids]
        expected_image_corr = (expected_avg_norm / expected_normcorr) * expected_scale

        np.testing.assert_allclose(np.asarray(got.norm_corrections_per_half[0]), expected_normcorr, rtol=1e-6)
        assert got.avg_norm_correction_per_half[0] == pytest.approx(expected_avg_norm)
        np.testing.assert_allclose(np.asarray(got.group_scale_corrections_per_half[0]), expected_group_scale, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got.scale_corrections_per_half[0]), expected_scale, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got.image_corrections_per_half[0]), expected_image_corr, rtol=1e-6)

    def test_relion_norm_scale_update_accepts_single_active_class3d_half(self):
        """Class3D-style single-half runs still update active-half corrections."""
        active = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=2.0,
            wsum_norm_correction=jnp.array([2.0, 8.0], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([2.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([1.0], dtype=jnp.float32),
        )
        empty = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=0.0,
        )

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[active, empty],
            group_ids_per_half=[np.zeros(2, dtype=np.int64), np.zeros(0, dtype=np.int64)],
        )

        np.testing.assert_allclose(np.asarray(got.norm_corrections_per_half[0]), [2.0, 4.0], rtol=1e-6)
        assert got.avg_norm_correction_per_half[0] == pytest.approx(3.0)
        np.testing.assert_allclose(np.asarray(got.group_scale_corrections_per_half[0]), [1.0], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got.image_corrections_per_half[0]), [1.5, 0.75], rtol=1e-6)
        assert np.asarray(got.image_corrections_per_half[1]).shape == (0,)
        assert np.asarray(got.scale_corrections_per_half[1]).shape == (0,)

    def test_relion_norm_scale_update_skips_firstiter_cc_scale_only(self):
        """RELION firstiter-CC still updates normcorr but keeps old scales."""
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=2.0,
            wsum_norm_correction=jnp.array([2.0, 8.0], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([100.0, 1.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([10.0, 1.0], dtype=jnp.float32),
        )
        group_ids = np.array([0, 1], dtype=np.int64)
        old_group_scale = np.array([1.0, 2.0], dtype=np.float64)
        old_scale = old_group_scale[group_ids]
        old_image_corr = np.array([1.0, 1.0], dtype=np.float64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            image_corrections_per_half=[old_image_corr, old_image_corr],
            scale_corrections_per_half=[old_scale, old_scale],
            group_ids_per_half=[group_ids, group_ids],
            group_scale_corrections_per_half=[old_group_scale, old_group_scale],
            relion_firstiter_cc_this_iter=True,
        )

        expected_normcorr = np.array([2.0, 8.0], dtype=np.float64)
        expected_avg_norm = 5.0
        expected_image_corr = (expected_avg_norm / expected_normcorr) * old_scale
        np.testing.assert_allclose(np.asarray(got.group_scale_corrections_per_half[0]), old_group_scale)
        np.testing.assert_allclose(np.asarray(got.scale_corrections_per_half[0]), old_scale)
        np.testing.assert_allclose(np.asarray(got.image_corrections_per_half[0]), expected_image_corr, rtol=1e-6)

    def test_relion_norm_scale_update_changes_two_native_groups_differently(self):
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=4.0,
            wsum_norm_correction=jnp.ones(4, dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([2.0, 6.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([1.0, 2.0], dtype=jnp.float32),
        )
        group_ids = np.asarray([0, 1, 0, 1], dtype=np.int64)
        old_scale = np.ones(4, dtype=np.float64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            scale_corrections_per_half=[old_scale, old_scale],
            group_ids_per_half=[group_ids, group_ids],
            do_norm_correction=False,
        )

        group_scale = np.asarray(got.group_scale_corrections_per_half[0], dtype=np.float64)
        assert group_scale.shape == (2,)
        assert group_scale[0] != pytest.approx(group_scale[1])
        np.testing.assert_allclose(
            np.asarray(got.scale_corrections_per_half[0], dtype=np.float64),
            group_scale[group_ids],
            rtol=1e-6,
        )

    def test_relion_norm_scale_update_preserves_zero_norm_residual_rows(self):
        """Images with no posterior norm mass keep their previous finite correction."""
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=3.0,
            wsum_norm_correction=jnp.array([12.5, 0.0, 12.5], dtype=jnp.float32),
        )
        group_ids = np.zeros(3, dtype=np.int64)
        old_scale = np.ones(3, dtype=np.float64)
        old_image_corr = np.array([1.0, 2.0, 1.0], dtype=np.float64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            image_corrections_per_half=[old_image_corr, old_image_corr],
            scale_corrections_per_half=[old_scale, old_scale],
            group_ids_per_half=[group_ids, group_ids],
            avg_norm_correction_per_half=[5.0, 5.0],
            do_scale_correction=False,
        )

        np.testing.assert_allclose(np.asarray(got.image_corrections_per_half[0]), old_image_corr, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got.norm_corrections_per_half[0]), [5.0, 2.5, 5.0], rtol=1e-6)
        assert got.avg_norm_correction_per_half[0] == pytest.approx(5.0)
        assert got.zero_norm_residual_counts == [1, 1]
        assert np.all(np.isfinite(np.asarray(got.image_corrections_per_half[0])))

    def test_relion_noise_stats_carry_optional_norm_scale_fields(self):
        stats0 = NoiseStats(
            wsum_sigma2_noise=jnp.array([1.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([2.0], dtype=jnp.float32),
            wsum_sigma2_offset=3.0,
            sumw=4.0,
            wsum_norm_correction=jnp.array([5.0, 6.0], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([7.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([8.0], dtype=jnp.float32),
        )
        stats1 = stats0._replace(
            wsum_sigma2_noise=jnp.array([10.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([20.0], dtype=jnp.float32),
            wsum_norm_correction=jnp.array([50.0, 60.0], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([70.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([80.0], dtype=jnp.float32),
        )

        summed = _sum_noise_stats((stats0, stats1))
        combined = _combined_noise_stats([stats0, stats1])

        np.testing.assert_allclose(np.asarray(summed.wsum_norm_correction), [55.0, 66.0])
        np.testing.assert_allclose(np.asarray(summed.wsum_scale_correction_xa), [77.0])
        np.testing.assert_allclose(np.asarray(summed.wsum_scale_correction_aa), [88.0])
        assert combined.wsum_norm_correction is None
        assert combined.wsum_scale_correction_xa is None
        assert combined.wsum_scale_correction_aa is None

    def test_relion_refinement_runs_2_iterations(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """RELION-parity refinement completes 2 iterations on a tiny dataset."""
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
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
        )

        assert result["convergence_state"].has_converged is False
        assert len(result["wall_times"]) == 1
        assert len(result["current_sizes"]) == 1

    def test_relion_mode_joins_lowres_halves_on_first_iteration(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """RELION joins low-res half accumulators before the first local output iter."""

        join_calls = []
        original_join = regularization_module.join_halves_at_low_resolution

        def spy_join(*args, **kwargs):
            join_calls.append(kwargs.get("current_resolution_angstrom"))
            return original_join(*args, **kwargs)

        monkeypatch.setattr(
            regularization_module,
            "join_halves_at_low_resolution",
            spy_join,
        )

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
            init_current_size=4,
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=40.0,
            relion_firstiter_ini_high_angstrom=30.0,
        )

        expected_resolution = shell_index_to_resolution_angstrom(1, IMAGE_SHAPE[0], half_datasets[0].voxel_size)
        assert len(join_calls) == 1
        assert join_calls[0] == pytest.approx(expected_resolution)

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
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
        )

        assert result["convergence_state"].has_converged is True
        assert len(result["wall_times"]) == 2
        assert len(run_em_mean_ids) == 4
        assert run_em_mean_ids[-2] != run_em_mean_ids[-1]

    def test_relion_final_iteration_tau2_uses_half_accumulators(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """Final all-data tau2 uses half weights; only final reconstruction sums them."""
        original_update = iteration_loop_module.update_refinement_state
        original_tau2 = regularization_module.compute_relion_tau2_from_weights
        ctf_values = [2.0, 4.0, 7.0, 11.0]
        run_em_call = {"idx": 0}
        whole_tau2_calls = []

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def fake_run_em(
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            rotations_arg,
            translations_arg,
            disc_type,
            **kwargs,
        ):
            del mean, mean_variance, noise_variance, translations_arg, disc_type
            idx = run_em_call["idx"]
            run_em_call["idx"] += 1
            n_images = experiment_dataset.n_units
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
            ctf_value = ctf_values[idx]
            return (
                None,
                np.zeros(n_images, dtype=np.int32),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
                jnp.full(recon_vol_size, ctf_value, dtype=jnp.complex64),
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(rotations_arg).shape[0], dtype=jnp.float32),
                ),
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images),
                ),
            )

        def spy_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs):
            if kwargs.get("is_whole_instead_of_half", False):
                whole_tau2_calls.append(
                    (
                        np.asarray(Ft_ctf_0).real.copy(),
                        np.asarray(Ft_ctf_1).real.copy(),
                    )
                )
            return original_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs)

        monkeypatch.setattr(
            iteration_loop_module,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        monkeypatch.setattr(iteration_loop_module, "run_em", fake_run_em)
        monkeypatch.setattr(regularization_module, "compute_relion_tau2_from_weights", spy_tau2)

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
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
        )

        assert result["convergence_state"].has_converged is True
        assert run_em_call["idx"] == 4
        assert len(whole_tau2_calls) == 1
        final_half0, final_half1 = whole_tau2_calls[0]
        np.testing.assert_allclose(final_half0, ctf_values[2], atol=0.0)
        np.testing.assert_allclose(final_half1, ctf_values[3], atol=0.0)

    def test_relion_final_iteration_uses_learned_k1_direction_prior(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """The final K=1 all-data E-step uses the previous iter's pdf_direction."""
        original_update = iteration_loop_module.update_refinement_state
        original_run_em = iteration_loop_module.run_em
        custom_eulers = np.zeros((N_ROTATIONS, 3), dtype=np.float32)
        learned_direction_priors = [
            np.array([0.7, 0.3], dtype=np.float32),
            np.array([0.2, 0.8], dtype=np.float32),
        ]
        expected_rotation_log_priors = [
            np.linspace(0.0, -0.4, N_ROTATIONS, dtype=np.float32),
            np.linspace(-1.0, -1.4, N_ROTATIONS, dtype=np.float32),
        ]
        run_em_rotation_priors = []
        collapse_calls = []
        make_prior_calls = []

        monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def fake_rotation_grid_size(_order):
            return N_ROTATIONS

        def fake_collapse_rotation_posterior_to_direction_prior(rotation_posterior_sums, healpix_order):
            collapse_calls.append((np.asarray(rotation_posterior_sums).shape, int(healpix_order)))
            return learned_direction_priors[len(collapse_calls) - 1]

        def fake_make_relion_direction_log_prior(direction_prior, healpix_order):
            prior = np.asarray(direction_prior, dtype=np.float32)
            make_prior_calls.append((prior.copy(), int(healpix_order)))
            if np.array_equal(prior, learned_direction_priors[0]):
                return expected_rotation_log_priors[0]
            if np.array_equal(prior, learned_direction_priors[1]):
                return expected_rotation_log_priors[1]
            raise AssertionError(f"unexpected direction prior {prior}")

        def spy_run_em(dataset, mean, *args, **kwargs):
            _ = dataset, mean
            prior = kwargs.get("rotation_log_prior")
            run_em_rotation_priors.append(None if prior is None else np.asarray(prior).copy())
            return original_run_em(dataset, mean, *args, **kwargs)

        monkeypatch.setattr(
            iteration_loop_module,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        monkeypatch.setattr(iteration_loop_module, "run_em", spy_run_em)
        monkeypatch.setattr(iteration_loop_module, "rotation_grid_size", fake_rotation_grid_size)
        monkeypatch.setattr(
            iteration_loop_module,
            "get_relion_rotation_grid",
            lambda _order: np.asarray(rotations, dtype=np.float32),
        )
        monkeypatch.setattr(
            iteration_loop_module,
            "get_relion_rotation_grid_eulers",
            lambda _order: custom_eulers,
        )
        monkeypatch.setattr(
            iteration_loop_module,
            "collapse_rotation_posterior_to_direction_prior",
            fake_collapse_rotation_posterior_to_direction_prior,
        )
        monkeypatch.setattr(
            iteration_loop_module,
            "make_relion_direction_log_prior",
            fake_make_relion_direction_log_prior,
        )

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
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
        )

        assert result["convergence_state"].has_converged is True
        assert run_em_rotation_priors[:2] == [None, None]
        np.testing.assert_allclose(run_em_rotation_priors[-2], expected_rotation_log_priors[0])
        np.testing.assert_allclose(run_em_rotation_priors[-1], expected_rotation_log_priors[1])
        assert len(collapse_calls) == 2
        assert [call[0] for call in collapse_calls] == [(N_ROTATIONS,), (N_ROTATIONS,)]
        assert [call[1] for call in collapse_calls] == [2, 2]
        assert len(make_prior_calls) == 4
        np.testing.assert_array_equal(make_prior_calls[-2][0], learned_direction_priors[0])
        np.testing.assert_array_equal(make_prior_calls[-1][0], learned_direction_priors[1])

    def test_relion_final_iteration_keeps_translation_prior(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """The final all-data E-step still uses RELION's pdf_offset prior."""
        original_update = iteration_loop_module.update_refinement_state
        original_run_em = iteration_loop_module.run_em
        run_em_translation_priors = []
        run_em_translation_prior_centers = []

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def spy_run_em(dataset, mean, *args, **kwargs):
            _ = dataset, mean
            prior = kwargs.get("translation_log_prior")
            centers = kwargs.get("translation_prior_centers")
            run_em_translation_priors.append(None if prior is None else np.asarray(prior).copy())
            run_em_translation_prior_centers.append(None if centers is None else np.asarray(centers).copy())
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
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
        )

        assert result["convergence_state"].has_converged is True
        assert len(run_em_translation_priors) == 4
        assert len(run_em_translation_prior_centers) == 4
        for final_prior in run_em_translation_priors[-2:]:
            assert final_prior is not None
            assert final_prior.size > 0
            assert np.all(np.isfinite(final_prior))
        for final_centers in run_em_translation_prior_centers[-2:]:
            assert final_centers is not None
            assert final_centers.shape[-1] == 2
            assert np.all(np.isfinite(final_centers))

    def test_relion_final_iteration_uses_joined_first_half_noise_variance(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """The joined final E-step scores both particle halves with model half 1 noise."""
        original_update = iteration_loop_module.update_refinement_state
        original_run_em = iteration_loop_module.run_em
        replay_noise_h1 = np.linspace(2.0, 3.0, IMAGE_SIZE, dtype=np.float32)
        replay_noise_h2 = np.linspace(5.0, 6.0, IMAGE_SIZE, dtype=np.float32)
        run_em_noise = []

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def spy_run_em(dataset, mean, mean_variance, noise_variance, *args, **kwargs):
            run_em_noise.append(np.asarray(noise_variance, dtype=np.float32).copy())
            return original_run_em(dataset, mean, mean_variance, noise_variance, *args, **kwargs)

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
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
            replay_iteration_overrides=[
                None,
                {"noise_variance": [replay_noise_h1, replay_noise_h2]},
            ],
        )

        assert result["convergence_state"].has_converged is True
        assert len(run_em_noise) == 4
        np.testing.assert_allclose(run_em_noise[-2], replay_noise_h1, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(run_em_noise[-1], replay_noise_h1, rtol=0.0, atol=0.0)
        assert result["final_all_data_noise_source_half"] == 0

    def test_relion_final_iteration_uses_local_search_when_converged_state_is_local(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """The final all-data E-step follows RELION local-search state."""
        original_update = iteration_loop_module.update_refinement_state
        original_run_em = iteration_loop_module.run_em
        run_em_calls = []
        local_calls = []

        def fake_rotation_grid_size(_order):
            return N_ROTATIONS

        def fake_rotation_grid_n_in_planes(_order):
            return 1

        def fake_relion_rotation_grid_float32(order):
            n_rotations = fake_rotation_grid_size(order)
            return (
                np.repeat(np.eye(3, dtype=np.float32)[None, :, :], n_rotations, axis=0),
                np.zeros((n_rotations, 3), dtype=np.float32),
            )

        def fake_get_relion_rotation_grid(order, *args, **kwargs):
            del args, kwargs
            return fake_relion_rotation_grid_float32(order)[0]

        def fake_get_relion_rotation_grid_eulers(order, *args, **kwargs):
            del args, kwargs
            return fake_relion_rotation_grid_float32(order)[1]

        def force_converged_local_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            updated.do_local_search = True
            updated.healpix_order = max(updated.healpix_order, updated.auto_local_healpix_order)
            return updated

        def spy_run_em(dataset, mean, *args, **kwargs):
            run_em_calls.append((dataset.n_units, id(mean)))
            return original_run_em(dataset, mean, *args, **kwargs)

        def fake_local_search(
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
                mean_variance,
                noise_variance,
                rotation_grid_rotations,
                rotation_grid_eulers,
                sigma_rot,
                sigma_psi,
                translations,
                sigma_offset_angstrom,
                offset_range_pixels,
                disc_type,
                image_batch_size,
                rotation_block_size,
            )
            local_calls.append(
                {
                    "n_units": int(experiment_dataset.n_units),
                    "mean_id": id(mean),
                    "prior_rotations": np.asarray(prior_rotations, dtype=np.float32).copy(),
                    "prior_translations": np.asarray(prior_translations, dtype=np.float32).copy(),
                    "translation_prior_centers": np.asarray(
                        kwargs["translation_prior_centers"],
                        dtype=np.float32,
                    ).copy(),
                    "image_pre_shifts": np.asarray(kwargs["image_pre_shifts"], dtype=np.float32).copy(),
                    "healpix_order": int(healpix_order),
                    "current_size": current_size,
                    "accumulate_noise": kwargs["accumulate_noise"],
                    "return_best_pose_details": kwargs["return_best_pose_details"],
                }
            )
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
            base_outputs = (
                jnp.zeros(recon_vol_size, dtype=jnp.complex64),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
                np.zeros(experiment_dataset.n_units, dtype=np.int32),
            )
            best_pose_details = (
                np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0),
                np.zeros((experiment_dataset.n_units, 2), dtype=np.float32),
                np.zeros(experiment_dataset.n_units, dtype=np.int64),
            )
            relion_stats = RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(
                    iteration_loop_module.rotation_grid_size(healpix_order),
                    dtype=jnp.float32,
                ),
            )
            noise_stats = NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=1.0,
                sumw=float(experiment_dataset.n_units),
            )
            return _pack_fake_local_search_outputs(
                base_outputs,
                relion_stats,
                noise_stats,
                kwargs,
                experiment_dataset.n_units,
                best_pose_details,
            )

        monkeypatch.setattr(
            iteration_loop_module,
            "update_refinement_state",
            force_converged_local_after_first_iter,
        )
        monkeypatch.setattr(iteration_loop_module, "run_em", spy_run_em)
        monkeypatch.setattr(iteration_loop_module, "rotation_grid_size", fake_rotation_grid_size)
        monkeypatch.setattr(iteration_loop_module, "get_relion_rotation_grid", fake_get_relion_rotation_grid)
        monkeypatch.setattr(
            iteration_loop_module,
            "get_relion_rotation_grid_eulers",
            fake_get_relion_rotation_grid_eulers,
        )
        monkeypatch.setattr(
            iteration_loop_module,
            "_relion_rotation_grid_float32",
            fake_relion_rotation_grid_float32,
        )
        monkeypatch.setitem(
            make_relion_direction_log_prior.__globals__,
            "rotation_grid_size",
            fake_rotation_grid_size,
        )
        monkeypatch.setitem(
            make_relion_direction_log_prior.__globals__,
            "rotation_grid_n_in_planes",
            fake_rotation_grid_n_in_planes,
        )
        monkeypatch.setattr(iteration_loop_module, "_run_local_search_iteration", fake_local_search)

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
            init_healpix_order=2,
            max_healpix_order=4,
            auto_local_healpix_order=4,
            low_resol_join_halves_angstrom=0.0,
        )

        assert result["convergence_state"].has_converged is True
        assert len(run_em_calls) == 2
        assert len(local_calls) == 2
        assert local_calls[0]["mean_id"] != local_calls[1]["mean_id"]
        assert all(call["current_size"] == IMAGE_SHAPE[0] for call in local_calls)
        assert all(call["accumulate_noise"] for call in local_calls)
        assert all(call["return_best_pose_details"] for call in local_calls)
        assert all(call["healpix_order"] == 4 for call in local_calls)
        for call, dataset in zip(local_calls, half_datasets):
            assert call["prior_rotations"].shape == (dataset.n_units, 3)
            assert call["prior_translations"].shape == (dataset.n_units, 2)
            assert call["translation_prior_centers"].shape == (dataset.n_units, 2)
            assert call["image_pre_shifts"].shape == (dataset.n_units, 2)
            assert np.all(np.isfinite(call["prior_translations"]))
            assert np.all(np.isfinite(call["translation_prior_centers"]))
            assert np.all(np.isfinite(call["image_pre_shifts"]))

    def test_relion_final_iteration_supports_k_class(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """K-class refinement can run the final all-data iteration."""
        original_update = iteration_loop_module.update_refinement_state

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        monkeypatch.setattr(
            iteration_loop_module,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )

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
            init_healpix_order=2,
            max_healpix_order=2,
            low_resol_join_halves_angstrom=0.0,
            n_classes=2,
            init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        )

        assert result["convergence_state"].has_converged is True
        assert len(result["wall_times"]) == 2
        assert np.asarray(result["class_means"]).shape == (2, VOLUME_SIZE)
        np.testing.assert_allclose(np.sum(result["class_weights"]), 1.0, rtol=1e-6, atol=1e-6)
        for half_classes in result["class_assignments"]:
            assert np.asarray(half_classes).shape == (N_IMAGES // 2,)

    def test_relion_final_iteration_k_class_adaptive_uses_sparse_pass2_route(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """Adaptive K-class final all-data should not fall back to direct dense scoring."""
        original_update = iteration_loop_module.update_refinement_state

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        adaptive_calls = []

        def fail_direct_dense_k_class(*_args, **_kwargs):
            raise AssertionError("adaptive K-class final all-data routed to direct dense K-class scoring")

        def fake_adaptive_k_class(
            experiment_dataset,
            means,
            mean_variance,
            noise_variance,
            coarse_rotations,
            coarse_translations,
            fine_rotations,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
            disc_type,
            **kwargs,
        ):
            _ = (
                mean_variance,
                noise_variance,
                coarse_translations,
                trans_parent_map,
                disc_type,
            )
            adaptive_calls.append(
                {
                    "coarse_current_size": kwargs.get("coarse_current_size"),
                    "fine_current_size": kwargs.get("fine_current_size"),
                    "sparse_pass2": kwargs.get("sparse_pass2"),
                    "relion_fine_mstep_prune": kwargs.get("relion_fine_mstep_prune"),
                    "return_best_pose_details": kwargs.get("return_best_pose_details"),
                    "n_coarse_rot": int(np.asarray(coarse_rotations).shape[0]),
                    "n_fine_rot": int(np.asarray(fine_rotations).shape[0]),
                }
            )
            n_classes = int(np.asarray(means).shape[0])
            n_images = int(experiment_dataset.n_units)
            n_shells = int(experiment_dataset.image_shape[0]) // 2 + 1
            padding_factor = int(kwargs.get("reconstruction_padding_factor", 1))
            recon_vol_size = int(np.prod(experiment_dataset.volume_shape)) * (padding_factor**3)
            n_fine_rot = int(np.asarray(fine_rotations).shape[0])
            per_class_stats = tuple(
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(n_fine_rot, dtype=jnp.float32),
                )
                for _ in range(n_classes)
            )
            per_class_noise = tuple(
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images) / float(n_classes),
                )
                for _ in range(n_classes)
            )
            aggregate_noise = NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(n_images),
            )
            return KClassEMResult(
                new_means=None,
                Ft_y=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
                Ft_ctf=jnp.ones((n_classes, recon_vol_size), dtype=jnp.complex64),
                per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
                class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                class_responsibilities=jnp.full((n_classes, n_images), 1.0 / n_classes, dtype=jnp.float32),
                class_posterior_sums=jnp.full(n_classes, n_images / n_classes, dtype=jnp.float32),
                stats=per_class_stats[0],
                per_class_stats=per_class_stats,
                noise_stats=per_class_noise,
                aggregate_noise_stats=aggregate_noise,
                best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
                best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
                best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
                significant_counts=jnp.ones(n_images, dtype=jnp.int32),
            )

        monkeypatch.setattr(
            iteration_loop_module,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        monkeypatch.setattr(iteration_loop_module, "run_dense_k_class_em", fail_direct_dense_k_class)
        monkeypatch.setattr(iteration_loop_module, "run_dense_k_class_em_adaptive", fake_adaptive_k_class)
        monkeypatch.setattr(iteration_loop_module, "compute_coarse_image_size", lambda *_args, **_kwargs: 4)
        monkeypatch.setattr(iteration_loop_module, "clamp_relion_coarse_image_size", lambda coarse, *_args: int(coarse))

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
            init_healpix_order=1,
            max_healpix_order=1,
            adaptive_oversampling=1,
            low_resol_join_halves_angstrom=0.0,
            n_classes=2,
            init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        )

        assert result["convergence_state"].has_converged is True
        assert len(adaptive_calls) == 4
        final_calls = adaptive_calls[-2:]
        assert all(call["coarse_current_size"] == 4 for call in final_calls)
        assert all(call["fine_current_size"] == IMAGE_SHAPE[0] for call in final_calls)
        assert all(call["sparse_pass2"] is True for call in final_calls)
        assert all(call["relion_fine_mstep_prune"] is True for call in final_calls)
        assert all(call["n_fine_rot"] >= call["n_coarse_rot"] for call in final_calls)

    def test_relion_firstiter_k_class_dense_pass2_env_routes_dense(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        adaptive_calls = []

        def fake_adaptive_k_class(
            experiment_dataset,
            means,
            mean_variance,
            noise_variance,
            coarse_rotations,
            coarse_translations,
            fine_rotations,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
            disc_type,
            **kwargs,
        ):
            _ = (
                mean_variance,
                noise_variance,
                coarse_rotations,
                coarse_translations,
                rot_parent_map,
                trans_parent_map,
                disc_type,
            )
            adaptive_calls.append(kwargs.get("sparse_pass2"))
            n_classes = int(np.asarray(means).shape[0])
            n_images = int(experiment_dataset.n_units)
            n_shells = int(experiment_dataset.image_shape[0]) // 2 + 1
            padding_factor = int(kwargs.get("reconstruction_padding_factor", 1))
            recon_vol_size = int(np.prod(experiment_dataset.volume_shape)) * (padding_factor**3)
            n_fine_rot = int(np.asarray(fine_rotations).shape[0])
            per_class_stats = tuple(
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(n_fine_rot, dtype=jnp.float32),
                )
                for _ in range(n_classes)
            )
            per_class_noise = tuple(
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images) / float(n_classes),
                )
                for _ in range(n_classes)
            )
            aggregate_noise = NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(n_images),
            )
            return KClassEMResult(
                new_means=None,
                Ft_y=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
                Ft_ctf=jnp.ones((n_classes, recon_vol_size), dtype=jnp.complex64),
                per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
                class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                class_responsibilities=jnp.full((n_classes, n_images), 1.0 / n_classes, dtype=jnp.float32),
                class_posterior_sums=jnp.full(n_classes, n_images / n_classes, dtype=jnp.float32),
                stats=per_class_stats[0],
                per_class_stats=per_class_stats,
                noise_stats=per_class_noise,
                aggregate_noise_stats=aggregate_noise,
                best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
                best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
                best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
                significant_counts=jnp.ones(n_images, dtype=jnp.int32),
            )

        monkeypatch.setattr(iteration_loop_module, "run_dense_k_class_em_adaptive", fake_adaptive_k_class)
        monkeypatch.setattr(iteration_loop_module, "compute_coarse_image_size", lambda *_args, **_kwargs: 4)
        monkeypatch.setattr(iteration_loop_module, "clamp_relion_coarse_image_size", lambda coarse, *_args: int(coarse))

        def run_once():
            adaptive_calls.clear()
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
                init_current_size=4,
                init_healpix_order=1,
                max_healpix_order=1,
                adaptive_oversampling=1,
                low_resol_join_halves_angstrom=0.0,
                n_classes=2,
                init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
                skip_final_iteration=True,
            )
            return list(adaptive_calls)

        monkeypatch.delenv("RECOVAR_K_CLASS_DENSE_PASS2", raising=False)
        assert run_once() == [True, True]

        monkeypatch.setenv("RECOVAR_K_CLASS_DENSE_PASS2", "1")
        assert run_once() == [False, False]

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

    def test_relion_mode_dense_k_class_finite_outputs(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
    ):
        """Dense non-adaptive RELION loop supports an explicit class axis."""
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
            n_classes=2,
            init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        )

        assert np.all(np.isfinite(np.asarray(result["mean"])))
        assert np.asarray(result["class_means"]).shape == (2, VOLUME_SIZE)
        assert np.asarray(result["means"][0]).shape == (2, VOLUME_SIZE)
        assert np.asarray(result["means"][1]).shape == (2, VOLUME_SIZE)
        np.testing.assert_allclose(np.asarray(result["means"][0]), np.asarray(result["means"][1]))
        np.testing.assert_allclose(np.sum(result["class_weights"]), 1.0, rtol=1e-6, atol=1e-6)
        assert len(result["class_weight_trajectory"]) == 1
        assert len(result["class_assignment_history"]) == 1
        np.testing.assert_array_equal(
            result["class_assignment_history"][0],
            np.concatenate(
                [
                    np.asarray(result["class_assignments"][0], dtype=np.int32),
                    np.asarray(result["class_assignments"][1], dtype=np.int32),
                ],
            ),
        )
        for half_idx in range(2):
            assert result["class_assignments"][half_idx].shape == (half_datasets[half_idx].n_units,)
            assert np.all(result["class_assignments"][half_idx] >= 0)

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

        relion_rotations, _ = iteration_loop_module._relion_rotation_grid_float32(2)
        expected_per_half = []
        for dataset in half_datasets:
            _, _, _, _, stats, _ = run_em(
                dataset,
                init_volume,
                init_tau,
                init_noise,
                relion_rotations,
                translations,
                "linear_interp",
                image_batch_size=N_IMAGES,
                rotation_block_size=N_ROTATIONS,
                current_size=16,
                score_with_masked_images=True,
                half_spectrum_scoring=True,
                projection_padding_factor=iteration_loop_module.PROJECTION_PADDING_FACTOR,
                reconstruction_padding_factor=iteration_loop_module.PADDING_FACTOR,
                do_gridding_correction=True,
                square_window=iteration_loop_module.RELION_FOURIER_WINDOW_SQUARE,
                return_stats=True,
                accumulate_noise=True,
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
            init_healpix_order=1,
            max_healpix_order=2,
            particle_diameter_ang=200.0,
        )

        assert recorded["particle_diameter"] == pytest.approx(200.0)

    def test_relion_translation_log_prior_matches_source_pdf_offset(self, translations):
        log_prior = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            prior_centers=np.zeros(2, dtype=np.float32),
        )
        translations_px = np.asarray(translations)
        expected = -0.5 * np.sum(translations_px**2, axis=1) * (4.25**4) / (10.0**2)
        np.testing.assert_allclose(log_prior, expected, rtol=1e-6, atol=1e-6)
        assert int(np.argmax(log_prior)) == 0

    def test_relion_translation_log_prior_is_flat_without_offset_prior(self, translations):
        log_prior = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            prior_centers=None,
        )
        np.testing.assert_array_equal(log_prior, np.zeros(len(translations), dtype=np.float32))

    def test_relion_translation_log_prior_uses_offset_range_when_active(self, translations):
        log_prior_sigma = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            prior_centers=np.zeros(2, dtype=np.float32),
        )
        log_prior_range = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            prior_centers=np.zeros(2, dtype=np.float32),
            offset_range_pixels=3.0,
        )
        # RELION uses sigma = offset_range / 3 while a finite search range is active.
        assert log_prior_range[0] == pytest.approx(0.0)
        assert log_prior_range[1] < log_prior_sigma[1]

    def test_relion_translation_search_base_uses_integer_prescoring_shift(self):
        prev = np.array([[0.5, -0.5], [1.5, -1.5], [-0.49, 0.49]], dtype=np.float32)
        expected = np.array([[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(relion_translation_search_base(prev), expected, rtol=1e-6, atol=1e-6)
        near_half = np.array([[0.49999999, -0.49999999], [0.50000001, -0.50000001]], dtype=np.float64)
        np.testing.assert_allclose(
            relion_translation_search_base(near_half),
            np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
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

    def test_dense_batch_parameter_rows_accepts_selected_image_arrays(self):
        selected_values = np.asarray([[10.0], [11.0], [12.0], [13.0]], dtype=np.float32)
        selected_position = {10000: 0, 10002: 1, 10004: 2, 10006: 3}

        rows = _batch_parameter_rows(
            selected_values,
            batch_indices=np.asarray([10004, 10002], dtype=np.int64),
            start=1,
            end=3,
            selected_image_count=4,
            source_image_count=50000,
            selected_position=selected_position,
            name="image_pre_shifts",
        )

        np.testing.assert_array_equal(rows, selected_values[[2, 1]])

    def test_dense_batch_parameter_rows_accepts_full_dataset_arrays(self):
        full_values = np.arange(12, dtype=np.float32).reshape(6, 2)

        rows = _batch_parameter_rows(
            full_values,
            batch_indices=np.asarray([5, 2], dtype=np.int64),
            start=1,
            end=3,
            selected_image_count=4,
            source_image_count=6,
            name="image_corrections",
        )

        np.testing.assert_array_equal(rows, full_values[[5, 2]])

    def test_dense_score_constraints_use_selected_rows_for_per_image_priors(self):
        constraints = DenseScoreConstraints.from_inputs(
            rotation_log_prior=np.arange(12, dtype=np.float32).reshape(3, 4),
            translation_log_prior=np.arange(6, dtype=np.float32).reshape(3, 2) + 100.0,
            rotation_translation_mask=np.arange(24).reshape(3, 4, 2) % 3 == 0,
            n_images=3,
            n_rot=4,
            n_trans=2,
            n_rot_padded=4,
        )

        rotation_prior, translation_prior, candidate_mask, _valid = constraints.block_inputs(
            r0=1,
            r1=3,
            start=0,
            end=2,
            batch_count=2,
            rotation_block_size=2,
            rows=np.asarray([2, 0], dtype=np.int64),
        )

        np.testing.assert_array_equal(np.asarray(rotation_prior), np.asarray([[9, 10], [1, 2]], dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(translation_prior), np.asarray([[104, 105], [100, 101]], dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(candidate_mask),
            (np.arange(24).reshape(3, 4, 2) % 3 == 0)[[2, 0], 1:3, :],
        )

    def test_relion_translation_prior_center_matches_accelerated_pdf_offset_units(self):
        prev = np.array([[0.0, -1.0], [1.0, 0.0], [-0.82310355, -0.82310355]], dtype=np.float32)
        expected = np.array([[0.0, 1.0 / 4.25], [-1.0 / 4.25, 0.0], [1.0 / 4.25, 1.0 / 4.25]], dtype=np.float32)
        np.testing.assert_allclose(
            relion_translation_prior_center(prev, voxel_size=4.25),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )
        explicit_prior = np.zeros((1, 2), dtype=np.float32)
        np.testing.assert_allclose(
            relion_translation_prior_center(prev[:1], voxel_size=4.25, prior_offsets=explicit_prior),
            np.array([[0.0, 1.0 / 4.25]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_relion_local_translation_prior_center_keeps_angstrom_sampling_units(self):
        prev = np.array([[0.0, -1.0], [1.0, 0.0], [-0.82310355, -0.82310355]], dtype=np.float32)
        expected = np.array([[0.0, 1.0 / 4.25], [-1.0 / 4.25, 0.0], [1.0 / 4.25, 1.0 / 4.25]], dtype=np.float32)
        np.testing.assert_allclose(
            relion_local_translation_prior_center(prev, voxel_size=4.25),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_relion_sigma_offset_prior_center_matches_store_weighted_sums_units(self):
        prev = np.array([[0.0, -1.0], [1.0, 0.0], [-0.82310355, -0.82310355]], dtype=np.float32)
        expected = np.array([[0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(
            relion_sigma_offset_prior_center(prev),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )
        explicit_prior = np.zeros((1, 2), dtype=np.float32)
        np.testing.assert_allclose(
            relion_sigma_offset_prior_center(prev[:1], prior_offsets=explicit_prior),
            np.array([[0.0, 1.0]], dtype=np.float32),
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

    def test_kclass_direction_prior_combines_half_posteriors_before_collapsing(self):
        """Class3D has one pdf_direction update; RECOVAR halves are parallelism only."""
        healpix_order = 0
        n_rot = rotation_grid_size(healpix_order)
        n_dirs = n_rot // rotation_grid_n_in_planes(healpix_order)
        half0 = np.zeros((2, n_rot), dtype=np.float64)
        half1 = np.zeros((2, n_rot), dtype=np.float64)
        half0[0, 0] = 9.0
        half1[0, 1] = 3.0
        half0[1, 2] = 2.0
        half1[1, 3] = 6.0

        combined = _combined_class_direction_prior_from_halves(
            [half0, half1],
            n_classes=2,
            healpix_order=healpix_order,
        )

        expected = []
        for class_idx in range(2):
            expected.append(
                collapse_rotation_posterior_to_direction_prior(
                    half0[class_idx] + half1[class_idx],
                    healpix_order,
                )
            )
        expected = np.stack(expected, axis=0)
        assert combined.shape == (2, n_dirs)
        np.testing.assert_allclose(combined, expected, rtol=1e-6, atol=1e-8)

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

    def test_k_class_significance_manual_ppref_bypasses_texture_selector(
        self,
        half_datasets,
        monkeypatch,
    ):
        """Strict coarse scoring must call the manual PPref leaf directly."""
        from recovar.em.dense_single_volume.helpers import projection as projection_helpers

        manual_calls = []

        def fake_manual(projector_half, rotations, image_shape, r_max, padding_factor, output_size):
            manual_calls.append((projector_half.shape, rotations.shape, r_max, padding_factor, output_size))
            n_half = int(image_shape[0] * (image_shape[1] // 2 + 1))
            return jnp.ones((rotations.shape[0], n_half), dtype=jnp.complex64)

        def fail_texture_selector(*_args, **_kwargs):
            raise AssertionError("manual coarse PPref scoring reached the texture selector")

        monkeypatch.setattr(
            projection_helpers,
            "project_relion_projector_half_spectrum_centered_rows",
            fake_manual,
        )
        monkeypatch.setattr(
            projection_helpers,
            "compute_relion_projector_projections_block",
            fail_texture_selector,
        )

        dataset = half_datasets[0]
        rotations = _make_rotations(3, seed=20)
        _compute_k_class_significance_batched(
            dataset,
            jnp.zeros((1, VOLUME_SIZE), dtype=jnp.complex64),
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            jnp.zeros((1, 2), dtype=jnp.float32),
            "linear_interp",
            class_log_priors=np.zeros(1, dtype=np.float64),
            adaptive_fraction=1.0,
            max_significants=1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=None,
            relion_projector_half=jnp.zeros((1, 3, 3, 2), dtype=jnp.complex64),
            relion_projector_r_max=1,
            relion_projector_texture_interp=False,
            collect_significance=False,
            return_class_best=True,
        )

        assert manual_calls

    def test_significance_batched_matches_run_em_with_pre_shifts_scales_and_projection_padding(
        self,
        half_datasets,
        init_volume,
        monkeypatch,
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

        monkeypatch.setenv("RECOVAR_SIGNIFICANCE_SCORE_CACHE", "off")
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

        monkeypatch.setenv("RECOVAR_SIGNIFICANCE_SCORE_CACHE", "force")
        cached_result = _compute_significance_batched(
            dataset,
            init_volume,
            init_noise,
            rotations,
            translations,
            "linear_interp",
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=current_size,
            score_with_masked_images=True,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
            return_significant_sample_indices=True,
            return_full_stats=True,
        )
        monkeypatch.setenv("RECOVAR_SIGNIFICANCE_SCORE_CACHE", "off")
        uncached_result = _compute_significance_batched(
            dataset,
            init_volume,
            init_noise,
            rotations,
            translations,
            "linear_interp",
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=current_size,
            score_with_masked_images=True,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
            return_significant_sample_indices=True,
            return_full_stats=True,
        )
        for cached, uncached in zip(cached_result[:3], uncached_result[:3]):
            np.testing.assert_array_equal(np.asarray(cached), np.asarray(uncached))
        for cached_sig, uncached_sig in zip(cached_result[3], uncached_result[3]):
            if cached_sig is None or uncached_sig is None:
                assert cached_sig is uncached_sig
            else:
                np.testing.assert_array_equal(cached_sig, uncached_sig)
        for key, cached in cached_result[4].items():
            np.testing.assert_allclose(
                np.asarray(cached),
                np.asarray(uncached_result[4][key]),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_k_class_significance_score_cache_matches_uncached_path(
        self,
        half_datasets,
        init_volume,
        monkeypatch,
    ):
        """The K-class significance score cache must be exact, not approximate."""
        dataset = half_datasets[0]
        means = jnp.stack([init_volume, init_volume * jnp.asarray(1.01, dtype=init_volume.dtype)])
        rotations = _make_rotations(5, seed=31)
        translations = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
            dtype=jnp.float32,
        )
        class_log_priors = np.log(np.array([0.55, 0.45], dtype=np.float64))
        rotation_log_prior = np.stack(
            [
                np.linspace(0.0, -0.2, rotations.shape[0], dtype=np.float32),
                np.linspace(-0.1, 0.1, rotations.shape[0], dtype=np.float32),
            ],
            axis=0,
        )
        translation_log_prior = np.array([0.0, -0.05, -0.2], dtype=np.float32)
        common_kwargs = dict(
            class_log_priors=class_log_priors,
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=6,
            score_with_masked_images=True,
            rotation_log_prior=rotation_log_prior,
            translation_log_prior=translation_log_prior,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
        )

        monkeypatch.setenv("RECOVAR_SIGNIFICANCE_SCORE_CACHE", "force")
        cached_result = _compute_k_class_significance_batched(
            dataset,
            means,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            **common_kwargs,
        )
        monkeypatch.setenv("RECOVAR_SIGNIFICANCE_SCORE_CACHE", "off")
        uncached_result = _compute_k_class_significance_batched(
            dataset,
            means,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            **common_kwargs,
        )

        for cached, uncached in zip(cached_result[:4], uncached_result[:4]):
            np.testing.assert_array_equal(np.asarray(cached), np.asarray(uncached))
        for cached_by_class, uncached_by_class in zip(cached_result[4], uncached_result[4]):
            for cached_sig, uncached_sig in zip(cached_by_class, uncached_by_class):
                if cached_sig is None or uncached_sig is None:
                    assert cached_sig is uncached_sig
                else:
                    np.testing.assert_array_equal(cached_sig, uncached_sig)
        for key, cached in cached_result[5].items():
            np.testing.assert_allclose(
                np.asarray(cached),
                np.asarray(uncached_result[5][key]),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_k_class_pass1_fused_matches_unfused_significance(
        self,
        half_datasets,
        init_volume,
        monkeypatch,
    ):
        """The opt-in fused pass1 path must preserve K-class significance outputs."""
        dataset = half_datasets[0]
        means = jnp.stack([init_volume, init_volume * jnp.asarray(1.01, dtype=init_volume.dtype)])
        rotations = _make_rotations(5, seed=43)
        translations = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
            dtype=jnp.float32,
        )
        class_log_priors = np.log(np.array([0.55, 0.45], dtype=np.float64))
        rotation_log_prior = np.stack(
            [
                np.linspace(0.0, -0.2, rotations.shape[0], dtype=np.float32),
                np.linspace(-0.1, 0.1, rotations.shape[0], dtype=np.float32),
            ],
            axis=0,
        )
        translation_log_prior = np.array([0.0, -0.05, -0.2], dtype=np.float32)
        common_kwargs = dict(
            class_log_priors=class_log_priors,
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=6,
            score_with_masked_images=True,
            rotation_log_prior=rotation_log_prior,
            translation_log_prior=translation_log_prior,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
            collect_significance=True,
            return_class_best=True,
            score_mode="gaussian",
        )

        monkeypatch.setenv("RECOVAR_PASS1_FUSED", "0")
        unfused_result = _compute_k_class_significance_batched(
            dataset,
            means,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            **common_kwargs,
        )
        monkeypatch.setenv("RECOVAR_PASS1_FUSED", "1")
        fused_result = _compute_k_class_significance_batched(
            dataset,
            means,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            **common_kwargs,
        )

        for fused, unfused in zip(fused_result[:4], unfused_result[:4]):
            np.testing.assert_array_equal(np.asarray(fused), np.asarray(unfused))
        for fused_by_class, unfused_by_class in zip(fused_result[4], unfused_result[4]):
            for fused_sig, unfused_sig in zip(fused_by_class, unfused_by_class):
                if fused_sig is None or unfused_sig is None:
                    assert fused_sig is unfused_sig
                else:
                    np.testing.assert_array_equal(fused_sig, unfused_sig)
        for key, fused in fused_result[5].items():
            np.testing.assert_allclose(
                np.asarray(fused),
                np.asarray(unfused_result[5][key]),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_k_class_firstiter_cc_significance_matches_per_class_run_em(
        self,
        half_datasets,
        init_volume,
    ):
        """Joint firstiter-CC significance must match the old per-class probe."""
        dataset = half_datasets[0]
        means = jnp.stack([init_volume, init_volume * jnp.asarray(1.01, dtype=init_volume.dtype)])
        rotations = _make_rotations(5, seed=41)
        translations = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
            dtype=jnp.float32,
        )
        noise = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
        tau = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0
        class_log_priors = np.log(np.array([0.55, 0.45], dtype=np.float64))
        rotation_log_prior = np.stack(
            [
                np.linspace(0.0, -0.2, rotations.shape[0], dtype=np.float32),
                np.linspace(-0.1, 0.1, rotations.shape[0], dtype=np.float32),
            ],
            axis=0,
        )
        image_corrections = np.array([0.8, 1.2], dtype=np.float32)
        scale_corrections = np.array([1.1, 0.9], dtype=np.float32)
        image_pre_shifts = np.array([[1.5, -0.5], [-1.0, 1.25]], dtype=np.float32)
        common_kwargs = dict(
            image_batch_size=dataset.n_units,
            rotation_block_size=rotations.shape[0],
            current_size=6,
            score_with_masked_images=True,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
        )

        expected_hard = []
        expected_best = []
        for class_index in range(means.shape[0]):
            result = run_em(
                dataset,
                means[class_index],
                tau,
                noise,
                rotations,
                translations,
                "linear_interp",
                return_stats=True,
                accumulate_noise=False,
                disable_adjoint_y=True,
                disable_adjoint_ctf=True,
                score_only=True,
                class_log_prior=float(class_log_priors[class_index]),
                rotation_log_prior=rotation_log_prior[class_index],
                relion_firstiter_score_mode="normalized_cc",
                relion_firstiter_winner_take_all=True,
                **common_kwargs,
            )
            expected_hard.append(np.asarray(result[1], dtype=np.int32))
            expected_best.append(np.asarray(result[4].best_log_score_per_image, dtype=np.float32))

        *_, full_stats = _compute_k_class_significance_batched(
            dataset,
            means,
            noise,
            rotations,
            translations,
            "linear_interp",
            class_log_priors=class_log_priors,
            adaptive_fraction=1.0,
            max_significants=1,
            rotation_log_prior=rotation_log_prior,
            collect_significance=False,
            return_class_best=True,
            score_mode="normalized_cc",
            **common_kwargs,
        )

        np.testing.assert_array_equal(
            np.asarray(full_stats["class_hard_assignments"], dtype=np.int32),
            np.stack(expected_hard, axis=0),
        )
        np.testing.assert_allclose(
            np.asarray(full_stats["class_best_log_score_per_image"], dtype=np.float32),
            np.stack(expected_best, axis=0),
            rtol=1e-6,
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "with_image_corr,with_scale_corr,with_pre_shifts",
        [
            (False, False, False),  # baseline
            (True,  False, False),  # image only
            (False, True,  False),  # scale only
            (False, False, True),   # shifts only
            (True,  True,  False),  # image + scale
            (True,  False, True),   # image + shifts
            (False, True,  True),   # scale + shifts
            (True,  True,  True),   # all three (original failing case)
        ],
    )
    def test_k_class_gaussian_significance_matches_per_class_run_em(
        self,
        half_datasets,
        init_volume,
        with_image_corr,
        with_scale_corr,
        with_pre_shifts,
    ):
        """Joint gaussian K-class significance must match per-class run_em scoring.

        Companion to test_k_class_firstiter_cc_significance_matches_per_class_run_em,
        but for the gaussian score_mode that InitialModel uses for iter-2+ scoring.
        Parametrized across (corrections, priors) to isolate where the K-class
        gaussian path diverges from em_engine.run_em on identical inputs.
        """
        dataset = half_datasets[0]
        means = jnp.stack([init_volume, init_volume * jnp.asarray(1.01, dtype=init_volume.dtype)])
        rotations = _make_rotations(5, seed=41)
        translations = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
            dtype=jnp.float32,
        )
        noise = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
        tau = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0
        # Priors are always on — the [False-True] case in the prior bisect
        # established that priors do NOT break parity, so the residual is
        # purely in the corrections.
        class_log_priors = np.log(np.array([0.55, 0.45], dtype=np.float64))
        rotation_log_prior = np.stack(
            [
                np.linspace(0.0, -0.2, rotations.shape[0], dtype=np.float32),
                np.linspace(-0.1, 0.1, rotations.shape[0], dtype=np.float32),
            ],
            axis=0,
        )
        image_corrections = np.array([0.8, 1.2], dtype=np.float32) if with_image_corr else None
        scale_corrections = np.array([1.1, 0.9], dtype=np.float32) if with_scale_corr else None
        image_pre_shifts = np.array([[1.5, -0.5], [-1.0, 1.25]], dtype=np.float32) if with_pre_shifts else None
        common_kwargs = dict(
            image_batch_size=dataset.n_units,
            rotation_block_size=rotations.shape[0],
            current_size=6,
            score_with_masked_images=True,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            use_float64_scoring=True,
        )

        expected_hard = []
        expected_best = []
        for class_index in range(means.shape[0]):
            result = run_em(
                dataset,
                means[class_index],
                tau,
                noise,
                rotations,
                translations,
                "linear_interp",
                return_stats=True,
                accumulate_noise=False,
                disable_adjoint_y=True,
                disable_adjoint_ctf=True,
                score_only=True,
                class_log_prior=float(class_log_priors[class_index]),
                rotation_log_prior=rotation_log_prior[class_index],
                **common_kwargs,
            )
            expected_hard.append(np.asarray(result[1], dtype=np.int32))
            expected_best.append(np.asarray(result[4].best_log_score_per_image, dtype=np.float32))

        *_, full_stats = _compute_k_class_significance_batched(
            dataset,
            means,
            noise,
            rotations,
            translations,
            "linear_interp",
            class_log_priors=class_log_priors,
            adaptive_fraction=1.0,
            max_significants=1,
            rotation_log_prior=rotation_log_prior,
            collect_significance=False,
            return_class_best=True,
            score_mode="gaussian",
            **common_kwargs,
        )

        np.testing.assert_array_equal(
            np.asarray(full_stats["class_hard_assignments"], dtype=np.int32),
            np.stack(expected_hard, axis=0),
        )
        np.testing.assert_allclose(
            np.asarray(full_stats["class_best_log_score_per_image"], dtype=np.float32),
            np.stack(expected_best, axis=0),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_k_class_significance_dump_emits_target_files(
        self,
        half_datasets,
        init_volume,
        monkeypatch,
        tmp_path,
    ):
        """K-class significance pass writes per-image .npz dumps when env vars target an image.

        Regression for codex_k2_dump_20260508_064420_5026: prior to wiring
        ``_maybe_dump_k_class_significance_batch`` into the K-class branch, the
        InitialModel K=2 sparse pass-2 emitted no significance debug files even
        with ``RECOVAR_SIGNIFICANCE_DUMP_DIR`` and the matching original-index
        target set.
        """
        dataset = half_datasets[0]
        means = jnp.stack([init_volume, init_volume * jnp.asarray(1.01, dtype=init_volume.dtype)])
        rotations = _make_rotations(5, seed=31)
        translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
        class_log_priors = np.log(np.array([0.55, 0.45], dtype=np.float64))

        dump_dir = tmp_path / "k_class_sig_dump"
        target_local = 0
        monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
        monkeypatch.setenv(
            "RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES",
            str(target_local),
        )

        _compute_k_class_significance_batched(
            dataset,
            means,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            class_log_priors=class_log_priors,
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=6,
            score_with_masked_images=False,
        )

        files = sorted(dump_dir.glob("*.npz"))
        assert files, f"K-class significance dump produced no files in {dump_dir}"
        payload = np.load(files[0])
        assert int(payload["n_classes"]) == 2
        assert int(payload["n_rot"]) == int(rotations.shape[0])
        assert int(payload["n_trans"]) == int(translations.shape[0])
        weights_per_class = payload["weights_per_class"]
        assert weights_per_class.shape == (2, int(rotations.shape[0]) * int(translations.shape[0]))
        assert weights_per_class.sum() == pytest.approx(1.0, abs=1e-6)
        class_log_z = payload["class_log_z"]
        assert class_log_z.shape == (2,)
        assert int(payload["class_assignment"]) in (0, 1)

    def test_k_class_score_probe_ignores_dump_env_when_significance_is_not_collected(
        self,
        half_datasets,
        init_volume,
        monkeypatch,
        tmp_path,
    ):
        """Score-only K-class probes must not abort when a dump env is set."""
        dataset = half_datasets[0]
        means = jnp.stack([init_volume, init_volume * jnp.asarray(1.01, dtype=init_volume.dtype)])
        rotations = _make_rotations(5, seed=37)
        translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
        class_log_priors = np.log(np.array([0.55, 0.45], dtype=np.float64))

        dump_dir = tmp_path / "k_class_score_probe_dump"
        monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
        monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "0")

        *_, full_stats = _compute_k_class_significance_batched(
            dataset,
            means,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            class_log_priors=class_log_priors,
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=6,
            score_with_masked_images=False,
            collect_significance=False,
            return_class_best=True,
        )

        assert not list(dump_dir.glob("*.npz"))
        assert full_stats["class_hard_assignments"].shape == (2, dataset.n_units)

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
            init_healpix_order=2,
            max_healpix_order=3,
            auto_local_healpix_order=3,
        )

        state = result["convergence_state"]
        assert isinstance(state, RefinementState)
        assert state.auto_local_healpix_order == 3
        # After 2 iterations, iteration counter should be at least 1
        assert state.iteration >= 1
        # ave_Pmax should be in [0, 1]
        assert 0.0 <= state.ave_Pmax <= 1.0

    def test_refinement_state_uses_configured_auto_local_healpix_order(self):
        default_state = RefinementState(healpix_order=3)
        assert not default_state.should_do_local_search
        assert not default_state.do_local_search

        local_state = RefinementState(healpix_order=3, auto_local_healpix_order=3)
        assert local_state.should_do_local_search
        assert local_state.do_local_search

    def test_refine_angular_sampling_uses_configured_auto_local_healpix_order(self):
        state = RefinementState(
            healpix_order=2,
            adaptive_oversampling=1,
            translation_range=10.0,
            translation_step=2.0,
            max_healpix_order=7,
            auto_local_healpix_order=3,
        )

        refined = refine_angular_sampling(state)

        assert refined.healpix_order == 3
        assert refined.auto_local_healpix_order == 3
        assert refined.do_local_search
        assert refined.sigma_rot > 0.0
        assert refined.sigma_psi > 0.0

    def test_low_pmax_refinement_guard_is_opt_in(self, monkeypatch):
        state = RefinementState(
            healpix_order=4,
            max_healpix_order=7,
            auto_local_healpix_order=4,
            current_resolution=36.0,
            previous_resolution=36.0,
            nr_iter_wo_resol_gain=4,
            nr_iter_wo_large_hidden_variable_changes=1,
            smallest_changes_optimal_orientations=2.0,
            smallest_changes_optimal_offsets_angstrom=0.5,
            smallest_changes_optimal_classes=0,
            ave_Pmax=0.10,
            acc_rot=float("inf"),
        )

        monkeypatch.delenv("RECOVAR_EM_LOW_PMAX_REFINE_GUARD", raising=False)
        assert should_refine_angular_sampling(state)

        monkeypatch.setenv("RECOVAR_EM_LOW_PMAX_REFINE_GUARD", "1")
        assert not should_refine_angular_sampling(state)

        confident = RefinementState(
            healpix_order=4,
            max_healpix_order=7,
            auto_local_healpix_order=4,
            current_resolution=36.0,
            previous_resolution=36.0,
            nr_iter_wo_resol_gain=4,
            nr_iter_wo_large_hidden_variable_changes=1,
            smallest_changes_optimal_orientations=2.0,
            smallest_changes_optimal_offsets_angstrom=0.5,
            smallest_changes_optimal_classes=0,
            ave_Pmax=0.30,
            acc_rot=float("inf"),
        )
        assert should_refine_angular_sampling(confident)

    def test_low_pmax_refinement_guard_can_cover_prelocal_transition(self, monkeypatch):
        state = RefinementState(
            healpix_order=3,
            max_healpix_order=7,
            auto_local_healpix_order=4,
            current_resolution=36.0,
            previous_resolution=36.0,
            nr_iter_wo_resol_gain=4,
            nr_iter_wo_large_hidden_variable_changes=1,
            smallest_changes_optimal_orientations=2.0,
            smallest_changes_optimal_offsets_angstrom=0.5,
            smallest_changes_optimal_classes=0,
            ave_Pmax=0.10,
            acc_rot=float("inf"),
        )
        assert not state.do_local_search

        monkeypatch.setenv("RECOVAR_EM_LOW_PMAX_REFINE_GUARD", "1")
        monkeypatch.delenv("RECOVAR_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL", raising=False)
        assert should_refine_angular_sampling(state)

        monkeypatch.setenv("RECOVAR_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL", "0")
        assert not should_refine_angular_sampling(state)

    def test_local_search_keeps_exhaustive_grid_at_last_prelocal_order(self):
        state = RefinementState(healpix_order=4, auto_local_healpix_order=4)

        assert state.do_local_search
        assert _exhaustive_grid_order_for_state(state) == 3

        nonlocal_state = RefinementState(healpix_order=4, auto_local_healpix_order=5)
        assert not nonlocal_state.do_local_search
        assert _exhaustive_grid_order_for_state(nonlocal_state) == 4

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
            init_healpix_order=2,
            max_healpix_order=3,
            init_fsc=np.ones(grid_size // 2),
        )

        assert called["tau2"] >= 1

    def test_k1_solvent_corrected_fsc_disabled_uses_raw_tau2_fsc(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """GUI auto-refine default does not solvent-correct FSC for tau2."""
        from recovar.reconstruction import regularization

        grid_size = int(np.sqrt(IMAGE_SIZE))
        n_shells = grid_size // 2 + 1
        raw_fsc = np.linspace(0.95, 0.55, n_shells, dtype=np.float32)
        raw_fsc[0] = 1.0
        tau2_fsc_inputs = []
        corrected_called = {"value": False}

        monkeypatch.setattr(
            regularization,
            "compute_relion_fsc_from_backprojector",
            lambda *_args, **_kwargs: jnp.asarray(raw_fsc),
        )

        def fail_corrected_fsc(*_args, **_kwargs):
            corrected_called["value"] = True
            raise AssertionError("solvent FSC correction should be disabled")

        monkeypatch.setattr(
            regularization,
            "compute_relion_solvent_corrected_true_fsc",
            fail_corrected_fsc,
        )

        original_tau2 = regularization.compute_relion_tau2_from_weights

        def wrap_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs):
            tau2_fsc_inputs.append(np.asarray(fsc, dtype=np.float32).copy())
            return original_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs)

        monkeypatch.setattr(regularization, "compute_relion_tau2_from_weights", wrap_tau2)

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
            adaptive_oversampling=0,
            init_healpix_order=2,
            max_healpix_order=3,
            particle_diameter_ang=200.0,
        )

        assert len(tau2_fsc_inputs) == 2
        for tau2_fsc in tau2_fsc_inputs:
            np.testing.assert_allclose(tau2_fsc, raw_fsc, atol=1e-7)
        np.testing.assert_allclose(np.asarray(result["fsc_history"][0]), raw_fsc, atol=1e-7)
        assert corrected_called["value"] is False

    def test_k1_solvent_corrected_fsc_enabled_feeds_tau2_and_growth(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """If RELION enables solvent FSC correction, the corrected curve drives tau2."""
        from recovar.reconstruction import regularization

        grid_size = int(np.sqrt(IMAGE_SIZE))
        n_shells = grid_size // 2 + 1
        raw_fsc = np.linspace(0.95, 0.55, n_shells, dtype=np.float32)
        raw_fsc[0] = 1.0
        corrected_fsc = np.linspace(0.90, 0.05, n_shells, dtype=np.float32)
        corrected_fsc[0] = 1.0
        tau2_fsc_inputs = []

        monkeypatch.setattr(
            regularization,
            "compute_relion_fsc_from_backprojector",
            lambda *_args, **_kwargs: jnp.asarray(raw_fsc),
        )

        def fake_corrected_fsc(*_args, **_kwargs):
            return jnp.asarray(corrected_fsc), {
                "randomize_at": 1,
                "fsc_unmasked": raw_fsc,
                "fsc_masked": corrected_fsc,
                "fsc_random_masked": np.zeros_like(corrected_fsc),
                "fsc_true": corrected_fsc,
            }

        monkeypatch.setattr(
            regularization,
            "compute_relion_solvent_corrected_true_fsc",
            fake_corrected_fsc,
        )

        original_tau2 = regularization.compute_relion_tau2_from_weights

        def wrap_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs):
            tau2_fsc_inputs.append(np.asarray(fsc, dtype=np.float32).copy())
            return original_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs)

        monkeypatch.setattr(regularization, "compute_relion_tau2_from_weights", wrap_tau2)

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
            adaptive_oversampling=0,
            init_healpix_order=2,
            max_healpix_order=3,
            particle_diameter_ang=200.0,
            do_solvent_fsc_correction=True,
        )

        assert len(tau2_fsc_inputs) == 2
        for tau2_fsc in tau2_fsc_inputs:
            np.testing.assert_allclose(tau2_fsc, corrected_fsc, atol=1e-7)
        np.testing.assert_allclose(np.asarray(result["fsc_history"][0]), raw_fsc, atol=1e-7)

    def test_k1_current_size_scheduling_raw_fsc_matches_gui_default(self):
        """GUI-default K=1 scheduling uses raw FSC-derived DVP."""
        raw_fsc = np.ones(129, dtype=np.float32) * 0.9
        raw_fsc[0] = 1.0
        raw_fsc[28:] = 0.0

        dvp = iteration_loop_module._k1_data_vs_prior_for_scheduling(
            raw_fsc=raw_fsc,
            corrected_data_vs_prior=None,
            current_size=56,
            grid_size=256,
            tau2_fudge=1.0,
        )
        shell = regularization_module.resolution_from_data_vs_prior(
            dvp,
            allow_high_res_recovery=True,
        )
        current_size = regularization_module.compute_current_size_relion(
            shell,
            256,
            ave_Pmax=1.0,
            has_high_fsc_at_limit=True,
        )

        assert shell == 27
        assert current_size == 118

    def test_k1_current_size_scheduling_keeps_boundary_shell(self):
        """Raw-FSC and corrected-DVP scheduling must agree at current_size//2."""
        current_size = 56
        boundary_shell = current_size // 2
        raw_fsc = np.zeros(129, dtype=np.float32)
        corrected_dvp = np.zeros_like(raw_fsc)
        raw_fsc[:boundary_shell] = 0.05
        corrected_dvp[:boundary_shell] = 0.05
        raw_fsc[boundary_shell] = 0.9
        corrected_dvp[boundary_shell] = 10.0

        raw_dvp = iteration_loop_module._k1_data_vs_prior_for_scheduling(
            raw_fsc=raw_fsc,
            corrected_data_vs_prior=None,
            current_size=current_size,
            grid_size=256,
            tau2_fudge=1.0,
        )
        corrected = iteration_loop_module._k1_data_vs_prior_for_scheduling(
            raw_fsc=raw_fsc,
            corrected_data_vs_prior=corrected_dvp,
            current_size=current_size,
            grid_size=256,
            tau2_fudge=1.0,
        )

        assert raw_dvp[boundary_shell] > 1.0
        assert corrected[boundary_shell] > 1.0
        assert raw_dvp[boundary_shell + 1] < 1.0
        assert corrected[boundary_shell + 1] == 0.0
        assert regularization_module.resolution_from_data_vs_prior(
            raw_dvp,
            allow_high_res_recovery=True,
        ) == boundary_shell
        assert regularization_module.resolution_from_data_vs_prior(
            corrected,
            allow_high_res_recovery=True,
        ) == boundary_shell

    def test_firstiter_cc_scheduling_uses_ini_high_shell(self):
        """RELION iter-1 firstiter_cc grows from ini_high, not DVP."""
        shell = iteration_loop_module._firstiter_cc_ini_high_resolution_shell(256, 2.125, 30.0)
        current_size = regularization_module.compute_current_size_relion(
            shell,
            256,
            ave_Pmax=1.0,
            has_high_fsc_at_limit=True,
        )
        assert shell == 18
        assert current_size == 100

        raw_fsc = np.ones(129, dtype=np.float32) * 0.9
        raw_fsc[0] = 1.0
        raw_fsc[28:] = 0.0
        dvp = iteration_loop_module._k1_data_vs_prior_for_scheduling(
            raw_fsc=raw_fsc,
            corrected_data_vs_prior=None,
            current_size=56,
            grid_size=256,
            tau2_fudge=1.0,
        )
        assert regularization_module.resolution_from_data_vs_prior(dvp, allow_high_res_recovery=True) == 27
        assert dvp[29] < 1.0

    def test_firstiter_cc_lowpass_runs_before_solvent_flatten(self, monkeypatch):
        """RELION applies iter-1 ini_high low-pass before solvent flatten."""
        from types import SimpleNamespace

        from recovar.em.dense_single_volume import mean_helpers as mean_helpers_module

        events = []

        def fake_reconstruct(*_args, **_kwargs):
            return jnp.ones(VOLUME_SIZE, dtype=jnp.complex64)

        def fake_lowpass(volume_ft_flat, *_args, **_kwargs):
            events.append("lowpass")
            return volume_ft_flat

        def fake_idft3(volume_ft):
            events.append("flatten_idft")
            return jnp.ones(VOLUME_SHAPE, dtype=jnp.float32)

        def fake_dft3(volume_real):
            events.append("flatten_dft")
            return jnp.asarray(volume_real, dtype=jnp.complex64)

        monkeypatch.setattr(mean_helpers_module, "_reconstruct_volume_eager", fake_reconstruct)
        monkeypatch.setattr(mean_helpers_module, "_apply_relion_initial_lowpass_filter", fake_lowpass)
        monkeypatch.setattr(mean_helpers_module.fourier_transform_utils, "get_idft3", fake_idft3)
        monkeypatch.setattr(mean_helpers_module.fourier_transform_utils, "get_dft3", fake_dft3)

        means = [None, None]
        mean_helpers_module._reconstruct_and_postprocess_means(
            means,
            Ft_y_0=jnp.ones(VOLUME_SIZE, dtype=jnp.complex64),
            Ft_y_1=jnp.ones(VOLUME_SIZE, dtype=jnp.complex64),
            Ft_ctf_0=jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            Ft_ctf_1=jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            Ft_y_combined=None,
            Ft_ctf_combined=None,
            mean_signal_variance=None,
            mean_signal_variance_shells=None,
            mean_signal_variance_per_half=[jnp.ones(VOLUME_SIZE), jnp.ones(VOLUME_SIZE)],
            n_classes=1,
            k_class_enabled=False,
            cs=8,
            iteration=0,
            grid_size=8,
            cryo=SimpleNamespace(voxel_size=1.0),
            volume_shape=VOLUME_SHAPE,
            tau2_fudge=1.0,
            padding_factor=1,
            projection_padding_factor=1,
            relion_minres_map=1,
            particle_diameter_ang=4.0,
            relion_firstiter_cc_this_iter=True,
            relion_firstiter_ini_high_angstrom=30.0,
            relion_width_mask_edge=5,
            relion_fmask_edge=2,
        )

        assert events[:3] == ["lowpass", "flatten_idft", "flatten_dft"]

    def test_kclass_reconstruction_uses_1d_tau_shell_prior(self, monkeypatch):
        """K-class M-step reconstruction should index RELION tau2 as shells."""
        from types import SimpleNamespace

        from recovar.em.dense_single_volume import mean_helpers as mean_helpers_module

        calls = []

        def fake_reconstruct(*_args, **kwargs):
            calls.append(kwargs)
            return jnp.ones(VOLUME_SIZE, dtype=jnp.complex64)

        monkeypatch.setattr(mean_helpers_module, "_reconstruct_volume_eager", fake_reconstruct)

        n_classes = 2
        n_shells = VOLUME_SHAPE[0] // 2 + 1
        tau_full = jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32) * 11.0
        tau_shells = jnp.stack(
            [
                jnp.arange(n_shells, dtype=jnp.float32) + 101.0,
                jnp.arange(n_shells, dtype=jnp.float32) + 201.0,
            ],
            axis=0,
        )
        means = [None, None]
        mean_helpers_module._reconstruct_and_postprocess_means(
            means,
            Ft_y_0=None,
            Ft_y_1=None,
            Ft_ctf_0=None,
            Ft_ctf_1=None,
            Ft_y_combined=jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.complex64),
            Ft_ctf_combined=jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32),
            mean_signal_variance=tau_full,
            mean_signal_variance_shells=tau_shells,
            mean_signal_variance_per_half=None,
            n_classes=n_classes,
            k_class_enabled=True,
            cs=8,
            iteration=0,
            grid_size=8,
            cryo=SimpleNamespace(voxel_size=1.0),
            volume_shape=VOLUME_SHAPE,
            tau2_fudge=4.0,
            padding_factor=1,
            projection_padding_factor=1,
            relion_minres_map=0,
            particle_diameter_ang=None,
            relion_firstiter_cc_this_iter=False,
            relion_firstiter_ini_high_angstrom=None,
            relion_width_mask_edge=5,
            relion_fmask_edge=2,
        )

        assert len(calls) == n_classes
        assert all(call["tau_is_1d"] is True for call in calls)
        np.testing.assert_allclose(np.asarray(calls[0]["tau"]), np.asarray(tau_shells[0]))
        np.testing.assert_allclose(np.asarray(calls[1]["tau"]), np.asarray(tau_shells[1]))
        assert means[0].shape == (n_classes, VOLUME_SIZE)
        np.testing.assert_array_equal(np.asarray(means[0]), np.asarray(means[1]))

    def test_k1_save_intermediates_reconstructs_unregularized_half_maps(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        tmp_path,
    ):
        """K=1 diagnostic unregularized maps use full half accumulators, not class indexing."""

        out_dir = tmp_path / "intermediates"
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
            adaptive_oversampling=0,
            init_healpix_order=2,
            max_healpix_order=3,
            save_intermediates_dir=str(out_dir),
        )

        assert len(result["current_sizes"]) == 1
        assert (out_dir / "it000_half1_unreg.mrc").exists()
        assert (out_dir / "it000_half2_unreg.mrc").exists()

    def test_k1_save_intermediates_can_skip_unregularized_half_maps(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        tmp_path,
    ):
        """Fast forensic dumps can keep regularized maps without unreg reconstruction."""

        out_dir = tmp_path / "intermediates"
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
            adaptive_oversampling=0,
            init_healpix_order=2,
            max_healpix_order=3,
            save_intermediates_dir=str(out_dir),
            save_intermediates_skip_unregularized=True,
        )

        assert len(result["current_sizes"]) == 1
        assert (out_dir / "it000_half1_reg.mrc").exists()
        assert (out_dir / "it000_half2_reg.mrc").exists()
        assert not (out_dir / "it000_half1_unreg.mrc").exists()
        assert not (out_dir / "it000_half2_unreg.mrc").exists()

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
            nside_level=1,
            init_healpix_order=1,
            max_healpix_order=2,
        )

        expected_per_half = [
            np.sqrt(noise_offset_wsums[0] / (2.0 * half_datasets[0].n_units)),
            np.sqrt(noise_offset_wsums[1] / (2.0 * half_datasets[1].n_units)),
        ]
        expected_sigma = float(np.mean(expected_per_half))
        assert result["sigma_offset_trajectory"][0] == pytest.approx(expected_sigma)
        assert result["sigma_offset_per_half_trajectory"][0] == pytest.approx(expected_per_half)

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
            init_healpix_order=1,
            max_healpix_order=2,
            skip_final_iteration=True,
        )

        assert len(captured_noise) == 2
        np.testing.assert_allclose(captured_noise[0], half1_noise)
        np.testing.assert_allclose(captured_noise[1], half2_noise)

    def test_k1_adaptive_significant_counts_feed_accuracy_estimate(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """K=1 adaptive pass-2 counts should make acc_rot finite."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        monkeypatch.delenv("RECOVAR_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE", raising=False)
        counts_by_half = [
            np.array([4, 5], dtype=np.int32),
            np.array([6, 7], dtype=np.int32),
        ]
        call_idx = {"value": 0}
        fine_mstep_prune_values = []
        rotations_many = _make_rotations(20, seed=333)

        monkeypatch.setattr(
            refine_mod,
            "_relion_rotation_grid_float32",
            lambda _order: (rotations_many, np.zeros((len(rotations_many), 3), dtype=np.float32)),
        )
        monkeypatch.setattr(
            refine_mod,
            "_relion_projector_half_maps_for_scoring",
            lambda *_args, **_kwargs: (None, None),
        )

        def fake_build_pass2_grids(
            effective_rotations,
            current_translations,
            base_translations,
            current_healpix_order,
            adaptive_oversampling,
            translation_step,
            random_perturbation,
        ):
            _ = (base_translations, current_healpix_order, adaptive_oversampling, translation_step, random_perturbation)
            coarse_rot = np.asarray(effective_rotations, dtype=np.float32)
            coarse_trans = np.asarray(current_translations, dtype=np.float32)
            rot_parent = np.arange(coarse_rot.shape[0], dtype=np.int32)
            trans_parent = np.arange(coarse_trans.shape[0], dtype=np.int32)
            return coarse_rot, coarse_trans, coarse_rot, coarse_trans, rot_parent, trans_parent

        def fake_adaptive_k1(
            experiment_dataset,
            means,
            mean_variance,
            noise_variance,
            coarse_rotations,
            coarse_translations,
            fine_rotations,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
            disc_type,
            **kwargs,
        ):
            _ = (
                means,
                mean_variance,
                noise_variance,
                coarse_translations,
                rot_parent_map,
                trans_parent_map,
                disc_type,
            )
            half_idx = call_idx["value"]
            call_idx["value"] += 1
            fine_mstep_prune_values.append(kwargs.get("relion_fine_mstep_prune"))
            n_images = int(experiment_dataset.n_units)
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
            counts = counts_by_half[half_idx]
            assert counts.shape == (n_images,)
            return KClassEMResult(
                new_means=None,
                Ft_y=jnp.zeros((1, recon_vol_size), dtype=jnp.complex64),
                Ft_ctf=jnp.ones((1, recon_vol_size), dtype=jnp.complex64),
                per_class_hard_assignments=jnp.zeros((1, n_images), dtype=jnp.int32),
                class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                class_responsibilities=jnp.ones((1, n_images), dtype=jnp.float32),
                class_posterior_sums=jnp.array([float(n_images)], dtype=jnp.float32),
                stats=RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(coarse_rotations).shape[0], dtype=jnp.float32),
                ),
                per_class_stats=(
                    RelionStats(
                        log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                        best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                        max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                        rotation_posterior_sums=jnp.ones(np.asarray(coarse_rotations).shape[0], dtype=jnp.float32),
                    ),
                ),
                noise_stats=(
                    NoiseStats(
                        wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                        wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                        wsum_sigma2_offset=0.0,
                        sumw=float(n_images),
                    ),
                ),
                aggregate_noise_stats=NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images),
                ),
                best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
                best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
                best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
                significant_counts=jnp.asarray(counts, dtype=jnp.int32),
            )

        monkeypatch.setattr(refine_mod, "_build_firstiter_cc_pass2_grids", fake_build_pass2_grids)
        monkeypatch.setattr(refine_mod, "run_dense_k_class_em_adaptive", fake_adaptive_k1)

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
            rotation_block_size=20,
            relion_current_sizes=[8],
            adaptive_oversampling=1,
            init_healpix_order=1,
            max_healpix_order=1,
            skip_final_iteration=True,
        )

        assert call_idx["value"] == 2
        assert fine_mstep_prune_values == [True, True]
        np.testing.assert_array_equal(
            np.asarray(result["significant_counts"][0], dtype=np.int32),
            np.concatenate(counts_by_half),
        )
        assert np.isfinite(result["acc_rot_trajectory"][0])
        assert np.isinf(result["convergence_state"].acc_rot)

    def test_k1_zero_oversampling_skips_adaptive_engine(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """The accepted K=1 os=0 path must remain on direct dense EM."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        def fail_adaptive(*_args, **_kwargs):
            raise AssertionError("adaptive_oversampling=0 entered the adaptive K-class engine")

        rotations_many = _make_rotations(20, seed=334)
        monkeypatch.setattr(refine_mod, "run_dense_k_class_em_adaptive", fail_adaptive)
        monkeypatch.setattr(
            refine_mod,
            "_relion_rotation_grid_float32",
            lambda _order: (rotations_many, np.zeros((len(rotations_many), 3), dtype=np.float32)),
        )

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
            rotation_block_size=20,
            relion_current_sizes=[8],
            adaptive_oversampling=0,
            init_healpix_order=1,
            max_healpix_order=1,
            skip_final_iteration=True,
        )

        assert np.asarray(result["mean"]).shape == (VOLUME_SIZE,)

    def test_k_class_adaptive_significant_counts_are_recorded_without_convergence_effect(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """K-class significant counts are diagnostics, not convergence inputs."""
        import recovar.em.dense_single_volume.iteration_loop as refine_mod

        monkeypatch.delenv("RECOVAR_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE", raising=False)
        counts_by_half = [
            np.array([8, 9], dtype=np.int32),
            np.array([10, 11], dtype=np.int32),
        ]
        call_idx = {"value": 0}
        fine_mstep_prune_values = []

        def fake_build_pass2_grids(
            effective_rotations,
            current_translations,
            base_translations,
            current_healpix_order,
            adaptive_oversampling,
            translation_step,
            random_perturbation,
        ):
            _ = (base_translations, current_healpix_order, adaptive_oversampling, translation_step, random_perturbation)
            coarse_rot = np.asarray(effective_rotations, dtype=np.float32)
            coarse_trans = np.asarray(current_translations, dtype=np.float32)
            rot_parent = np.arange(coarse_rot.shape[0], dtype=np.int32)
            trans_parent = np.arange(coarse_trans.shape[0], dtype=np.int32)
            return coarse_rot, coarse_trans, coarse_rot, coarse_trans, rot_parent, trans_parent

        def fake_adaptive_k_class(
            experiment_dataset,
            means,
            mean_variance,
            noise_variance,
            coarse_rotations,
            coarse_translations,
            fine_rotations,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
            disc_type,
            **kwargs,
        ):
            _ = (
                means,
                mean_variance,
                noise_variance,
                coarse_translations,
                fine_translations,
                rot_parent_map,
                trans_parent_map,
                disc_type,
            )
            half_idx = call_idx["value"]
            call_idx["value"] += 1
            fine_mstep_prune_values.append(kwargs.get("relion_fine_mstep_prune"))
            n_images = int(experiment_dataset.n_units)
            n_classes = 2
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
            counts = counts_by_half[half_idx]
            assert counts.shape == (n_images,)
            per_class_stats = tuple(
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(coarse_rotations).shape[0], dtype=jnp.float32),
                )
                for _ in range(n_classes)
            )
            noise_stats = tuple(
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images) / float(n_classes),
                )
                for _ in range(n_classes)
            )
            return KClassEMResult(
                new_means=None,
                Ft_y=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
                Ft_ctf=jnp.ones((n_classes, recon_vol_size), dtype=jnp.complex64),
                per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
                class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                class_responsibilities=jnp.full((n_classes, n_images), 0.5, dtype=jnp.float32),
                class_posterior_sums=jnp.array([float(n_images) / 2.0, float(n_images) / 2.0], dtype=jnp.float32),
                stats=RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(coarse_rotations).shape[0], dtype=jnp.float32),
                ),
                per_class_stats=per_class_stats,
                noise_stats=noise_stats,
                aggregate_noise_stats=NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images),
                ),
                best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
                best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
                best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
                significant_counts=jnp.asarray(counts, dtype=jnp.int32),
            )

        monkeypatch.setattr(refine_mod, "_build_firstiter_cc_pass2_grids", fake_build_pass2_grids)
        monkeypatch.setattr(refine_mod, "run_dense_k_class_em_adaptive", fake_adaptive_k_class)
        monkeypatch.setattr(iteration_loop_module, "compute_coarse_image_size", lambda *_args, **_kwargs: 4)
        monkeypatch.setattr(iteration_loop_module, "clamp_relion_coarse_image_size", lambda coarse, *_args: int(coarse))

        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            _make_rotations(20, seed=334),
            translations,
            disc_type="linear_interp",
            max_iter=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=20,
            relion_current_sizes=[8],
            adaptive_oversampling=1,
            init_healpix_order=1,
            max_healpix_order=1,
            low_resol_join_halves_angstrom=0.0,
            n_classes=2,
            init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
            skip_final_iteration=True,
        )

        assert call_idx["value"] == 2
        assert fine_mstep_prune_values == [True, True]
        np.testing.assert_array_equal(
            np.asarray(result["significant_counts"][0], dtype=np.int32),
            np.concatenate(counts_by_half),
        )
        assert np.isnan(result["acc_rot_trajectory"][0])
        assert np.isinf(result["convergence_state"].acc_rot)

    def test_approx_acc_rot_convergence_policy_guards_confident_prelocal_runs(self, monkeypatch):
        import recovar.em.dense_single_volume.iteration_loop as refine_mod
        from recovar.em.dense_single_volume.helpers.convergence import RefinementState

        for name in (
            "RECOVAR_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE",
            "RECOVAR_EM_DISABLE_APPROX_ACC_ROT_FOR_CONVERGENCE",
            "RECOVAR_EM_APPROX_ACC_ROT_MAX_AVE_PMAX",
            "RECOVAR_EM_APPROX_ACC_ROT_MIN_ITER",
        ):
            monkeypatch.delenv(name, raising=False)

        state = RefinementState(
            healpix_order=3,
            auto_local_healpix_order=4,
            current_resolution=23.65,
            particle_diameter_angstrom=200.0,
        )

        allow, reason = refine_mod._approx_acc_rot_policy_for_convergence(
            state=state,
            iteration_number=5,
            ave_pmax=0.96,
            new_resolution_angstrom=23.65,
        )

        assert not allow
        assert "high-pmax" in reason

    def test_approx_acc_rot_convergence_policy_is_diagnostic_by_default(self, monkeypatch):
        import recovar.em.dense_single_volume.iteration_loop as refine_mod
        from recovar.em.dense_single_volume.helpers.convergence import RefinementState

        for name in (
            "RECOVAR_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE",
            "RECOVAR_EM_DISABLE_APPROX_ACC_ROT_FOR_CONVERGENCE",
            "RECOVAR_EM_APPROX_ACC_ROT_MAX_AVE_PMAX",
            "RECOVAR_EM_APPROX_ACC_ROT_MIN_ITER",
        ):
            monkeypatch.delenv(name, raising=False)

        state = RefinementState(
            healpix_order=3,
            auto_local_healpix_order=4,
            current_resolution=36.27,
            particle_diameter_angstrom=200.0,
        )

        allow, reason = refine_mod._approx_acc_rot_policy_for_convergence(
            state=state,
            iteration_number=5,
            ave_pmax=0.77,
            new_resolution_angstrom=36.27,
        )

        assert not allow
        assert reason == "diagnostic-only-default"

    def test_approx_acc_rot_convergence_policy_env_overrides(self, monkeypatch):
        import recovar.em.dense_single_volume.iteration_loop as refine_mod
        from recovar.em.dense_single_volume.helpers.convergence import RefinementState

        state = RefinementState(
            healpix_order=3,
            auto_local_healpix_order=4,
            current_resolution=20.0,
        )

        monkeypatch.setenv("RECOVAR_EM_DISABLE_APPROX_ACC_ROT_FOR_CONVERGENCE", "1")
        monkeypatch.delenv("RECOVAR_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE", raising=False)
        allow, reason = refine_mod._approx_acc_rot_policy_for_convergence(
            state=state,
            iteration_number=5,
            ave_pmax=0.5,
            new_resolution_angstrom=20.0,
        )
        assert not allow
        assert reason == "disabled-by-env"

        monkeypatch.setenv("RECOVAR_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE", "1")
        monkeypatch.delenv("RECOVAR_EM_DISABLE_APPROX_ACC_ROT_FOR_CONVERGENCE", raising=False)
        allow, reason = refine_mod._approx_acc_rot_policy_for_convergence(
            state=state,
            iteration_number=1,
            ave_pmax=1.0,
            new_resolution_angstrom=10.0,
        )
        assert allow
        assert reason == "forced-by-env"


class TestRelionDefault:
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
        called = {"ran_relion": False}

        def fake_relion_loop(**kwargs):
            called["ran_relion"] = True
            assert kwargs["experiment_datasets"] is half_datasets
            assert kwargs["relion_current_sizes"] is None
            assert kwargs["init_healpix_order"] == 2
            return sentinel

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
        assert called == {"ran_relion": True}

    def test_refinement_options_struct_overrides_kwargs(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """Passing ``options=RefinementOptions(...)`` overrides individual kwargs."""
        from recovar.em.dense_single_volume import (
            KClassOptions,
            RefinementOptions,
            RefinementSchedule,
            RelionParityOptions,
        )

        sentinel = {"convergence_state": object()}
        captured: dict = {}

        def fake_relion_loop(**kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(
            iteration_loop_module,
            "_run_relion_iteration_loop",
            fake_relion_loop,
        )

        opts = RefinementOptions(
            schedule=RefinementSchedule(max_iter=7, init_healpix_order=3, max_healpix_order=4),
            parity=RelionParityOptions(
                tau2_fudge=4.0,
                perturb_replay_relion_prefix="custom",
                emulate_relion_firstiter_cc=True,
                do_solvent_fsc_correction=True,
            ),
            k_class=KClassOptions(n_classes=4),
        )
        result = refine_single_volume(
            half_datasets,
            init_volume,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
            rotations,
            translations,
            # These individual kwargs would normally win, but the struct overrides.
            max_iter=1,
            init_healpix_order=2,
            max_healpix_order=3,
            tau2_fudge=1.0,
            emulate_relion_firstiter_cc=False,
            n_classes=1,
            image_batch_size=N_IMAGES,
            rotation_block_size=N_ROTATIONS,
            init_current_size=16,
            options=opts,
        )

        assert result is sentinel
        assert captured["max_iter"] == 7
        assert captured["init_healpix_order"] == 3
        assert captured["max_healpix_order"] == 4
        assert captured["tau2_fudge"] == 4.0
        assert captured["perturb_replay_relion_prefix"] == "custom"
        assert captured["emulate_relion_firstiter_cc"] is True
        assert captured["do_solvent_fsc_correction"] is True
        assert captured["n_classes"] == 4

    def test_canonical_rotation_grid_reuses_relion_euler_table(self, monkeypatch):
        """The auto-refine setup path must not convert canonical grids via SciPy."""
        order = 1
        canonical_rotations = np.asarray(get_relion_rotation_grid(order), dtype=np.float32)
        expected_eulers = np.asarray(get_relion_rotation_grid_eulers(order), dtype=np.float32)

        def fail_r_to_relion(*_args, **_kwargs):
            raise AssertionError("generic R_to_relion should not be called for canonical grids")

        monkeypatch.setattr(iteration_loop_module.utils, "R_to_relion", fail_r_to_relion)

        got = _rotation_eulers_for_canonical_or_custom_grid(canonical_rotations, order)
        np.testing.assert_allclose(got, expected_eulers, rtol=0.0, atol=0.0)


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
    proj_weighted = rng.normal(size=(batch_size, n_rot, n_score)) + 1j * rng.normal(size=(batch_size, n_rot, n_score))
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

    support_only = fused_score_normalize_support_abs2_on_demand(
        shifted_score_j,
        ctf_score_j,
        proj_weighted_j,
        half_weights_j,
        rotation_log_prior_j,
        translation_log_prior_j,
        rotation_mask_j,
        sample_mask_j,
        half_spectrum_scoring=False,
        use_float64_normalization=True,
        reconstruct_significant_only=True,
        adaptive_fraction=0.99,
        max_significants=6,
    )
    support_only = jax.tree.map(np.asarray, support_only)

    np.testing.assert_allclose(support_only[0], np.asarray(log_z), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(support_only[1], np.asarray(probs), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(support_only[2], np.asarray(best_log_score), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(support_only[3], np.asarray(best_argmax))
    np.testing.assert_allclose(support_only[4], np.asarray(max_posterior), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(support_only[5], np.asarray(recon_mask))
    np.testing.assert_array_equal(support_only[6], np.asarray(recon_rot_mask))
    np.testing.assert_array_equal(support_only[7], np.asarray(n_sig))
    np.testing.assert_allclose(support_only[8], np.asarray(probs_sum_t), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(support_only[9], np.asarray(recon_probs_sum_t), rtol=2e-6, atol=2e-6)


def test_local_big_jit_sample_mask_none_matches_full_support():
    import jax

    from recovar.em.dense_single_volume.local_big_jit import _score_normalize_support

    rng = np.random.default_rng(123)
    batch_size, n_rot, n_trans, n_score = 2, 4, 3, 5
    shifted_score = rng.normal(size=(batch_size, n_trans, n_score)) + 1j * rng.normal(
        size=(batch_size, n_trans, n_score)
    )
    proj_weighted = rng.normal(size=(batch_size, n_rot, n_score)) + 1j * rng.normal(
        size=(batch_size, n_rot, n_score)
    )
    ctf_score = np.abs(rng.normal(size=(batch_size, n_score))).astype(np.float32) + 0.1
    half_weights = np.linspace(1.0, 2.0, n_score, dtype=np.float32)
    rotation_log_prior = rng.normal(size=(batch_size, n_rot)).astype(np.float32) * 0.01
    translation_log_prior = rng.normal(size=(batch_size, n_trans)).astype(np.float32) * 0.01
    rotation_mask = np.ones((batch_size, n_rot), dtype=bool)
    rotation_mask[0, -1] = False
    valid_image_mask = np.ones((batch_size,), dtype=bool)

    common_args = (
        jnp.asarray(shifted_score, dtype=jnp.complex64),
        jnp.asarray(ctf_score, dtype=jnp.float32),
        jnp.asarray(proj_weighted, dtype=jnp.complex64),
        jnp.asarray(half_weights, dtype=jnp.float32),
        jnp.asarray(rotation_log_prior),
        jnp.asarray(translation_log_prior),
        jnp.asarray(rotation_mask),
    )
    explicit_full = _score_normalize_support(
        *common_args,
        jnp.ones((batch_size, n_rot, n_trans), dtype=bool),
        jnp.asarray(valid_image_mask),
        jnp.zeros((batch_size,), dtype=jnp.float32),
        None,
        has_normalization_log_z=False,
        has_reconstruction_probability_threshold=False,
        half_spectrum_scoring=False,
        use_float64_normalization=True,
        reconstruct_significant_only=False,
        adaptive_fraction=0.999,
        max_significants=-1,
    )
    implicit_full = _score_normalize_support(
        *common_args,
        None,
        jnp.asarray(valid_image_mask),
        jnp.zeros((batch_size,), dtype=jnp.float32),
        None,
        has_normalization_log_z=False,
        has_reconstruction_probability_threshold=False,
        half_spectrum_scoring=False,
        use_float64_normalization=True,
        reconstruct_significant_only=False,
        adaptive_fraction=0.999,
        max_significants=-1,
    )

    explicit_full = jax.tree.map(np.asarray, explicit_full)
    implicit_full = jax.tree.map(np.asarray, implicit_full)
    for expected, actual in zip(explicit_full, implicit_full):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert implicit_full[6].shape == (batch_size, n_rot, n_trans)
    np.testing.assert_array_equal(
        implicit_full[8],
        np.sum(rotation_mask, axis=1, dtype=np.int32) * n_trans,
    )


def test_local_search_uses_lazy_parent_expanded_fine_rotation_grid_when_oversampling_is_enabled(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Adaptive local search expands RELION coarse parents without materializing the full fine grid."""
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
                "local_parent_oversampling_order": kwargs.get("local_parent_oversampling_order"),
                "adaptive_fraction": kwargs.get("adaptive_fraction"),
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
        outputs = list(base_outputs)
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_ids = np.zeros(experiment_dataset.n_units, dtype=np.int32)
            outputs.extend([best_rots, best_trans, best_ids])
        outputs.append(relion_stats)
        if kwargs.get("accumulate_noise", False):
            outputs.append(noise_stats)
        if kwargs.get("return_profile", False):
            outputs.append({"reconstruction_sample_indices_by_image": [None] * experiment_dataset.n_units})
        return tuple(outputs)

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "_precompute_exact_local_fine_grid_enabled", lambda order: False)
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
    prev_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=99),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_rotation_eulers=[prev_h1, prev_h2],
    )

    assert not any(kind == "rot" and order == 5 for kind, order in grid_calls)
    assert not any(kind == "euler" and order == 5 for kind, order in grid_calls)
    assert local_calls
    parent_calls = [call for call in local_calls if call["healpix_order"] == 4]
    fine_calls = [call for call in local_calls if call["healpix_order"] == 5]
    assert len(parent_calls) == 2
    assert len(fine_calls) == 2
    for call in fine_calls:
        assert call["healpix_order"] == 5
        assert call["rotations_is_none"]
        assert call["eulers_is_none"]
        assert call["rotation_grid_random_perturbation"] == 0.0
        assert call["rotation_grid_angular_sampling_deg"] == pytest.approx(
            relion_angular_sampling_deg(5, adaptive_oversampling=0),
        )
        assert call["local_parent_oversampling_order"] == 1
        assert call["adaptive_fraction"] == pytest.approx(0.999)


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
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], half_datasets[0].n_units, axis=0)
            best_trans = np.zeros((half_datasets[0].n_units, 2), dtype=np.float32)
            best_ids = np.zeros(half_datasets[0].n_units, dtype=np.int32)
            best_pose_details = (best_rots, best_trans, best_ids)
        return _pack_fake_local_search_outputs(
            base_outputs,
            relion_stats,
            noise_stats,
            kwargs,
            half_datasets[0].n_units,
            best_pose_details,
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "_precompute_exact_local_fine_grid_enabled", lambda order: False)
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
            np.ones(12 * (2 ** int(healpix_order)) ** 2, dtype=np.float64)
            / (12 * (2 ** int(healpix_order)) ** 2)
        ),
    )
    prev_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=111),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        init_healpix_order=4,
        max_healpix_order=4,
        perturb_factor=0.5,
        init_previous_best_rotation_eulers=[prev_h1, prev_h2],
    )

    assert any(
        call["n_rot"] == order_sizes[4]
        and np.isclose(call["angular_sampling_deg"], relion_angular_sampling_deg(4, adaptive_oversampling=0))
        for call in perturb_calls
    )
    assert not any(call["n_rot"] == order_sizes[5] for call in perturb_calls)
    assert local_calls
    fine_calls = [call for call in local_calls if call["healpix_order"] == 5]
    assert fine_calls
    assert fine_calls[0]["rotations_is_none"]
    assert fine_calls[0]["eulers_is_none"]
    assert fine_calls[0]["rotation_grid_random_perturbation"] == pytest.approx(0.25)
    assert fine_calls[0]["rotation_grid_angular_sampling_deg"] == pytest.approx(
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
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_ids = np.zeros(experiment_dataset.n_units, dtype=np.int32)
            best_pose_details = (best_rots, best_trans, best_ids)
        return _pack_fake_local_search_outputs(
            base_outputs,
            relion_stats,
            noise_stats,
            kwargs,
            experiment_dataset.n_units,
            best_pose_details,
        )

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "_precompute_exact_local_fine_grid_enabled", lambda order: False)
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

    prev_eulers_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_eulers_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=123),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_rotation_eulers=[prev_eulers_h1, prev_eulers_h2],
        init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
    )

    assert len(local_prior_translations) == 4
    expected_h1 = -relion_translation_search_base(prev_h1) / 1.0
    expected_h2 = -relion_translation_search_base(prev_h2) / 1.0
    np.testing.assert_allclose(local_prior_translations[0], expected_h1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(local_prior_translations[1], expected_h1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(local_prior_translations[2], expected_h2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(local_prior_translations[3], expected_h2, rtol=1e-6, atol=1e-6)


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
        outputs = list(base_outputs)
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_ids = np.zeros(experiment_dataset.n_units, dtype=np.int32)
            outputs.extend([best_rots, best_trans, best_ids])
        outputs.append(relion_stats)
        if kwargs.get("accumulate_noise", False):
            outputs.append(noise_stats)
        if kwargs.get("return_profile", False):
            outputs.append({"reconstruction_sample_indices_by_image": [None] * experiment_dataset.n_units})
        return tuple(outputs)

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "_precompute_exact_local_fine_grid_enabled", lambda order: False)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
    monkeypatch.setattr(
        refine_mod,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order: (
            np.ones(12 * (2 ** int(healpix_order)) ** 2, dtype=np.float64)
            / (12 * (2 ** int(healpix_order)) ** 2)
        ),
    )

    prev_eulers_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_eulers_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=123),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_rotation_eulers=[prev_eulers_h1, prev_eulers_h2],
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

    def fake_run_em(
        experiment_dataset, mean, mean_variance, noise_variance, rotations, translations, disc_type, **kwargs
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
        init_healpix_order=4,
        max_healpix_order=4,
    )

    assert reconstruct_flags == [False, False]


def _run_refine_with_stubbed_exact_local_batch_sizes(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
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

    def fake_run_em(
        experiment_dataset, mean, mean_variance, noise_variance, rotations, translations, disc_type, **kwargs
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
        outputs = list(base_outputs)
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_ids = np.zeros(experiment_dataset.n_units, dtype=np.int32)
            outputs.extend([best_rots, best_trans, best_ids])
        outputs.append(relion_stats)
        if kwargs.get("accumulate_noise", False):
            outputs.append(noise_stats)
        if kwargs.get("return_profile", False):
            outputs.append({"reconstruction_sample_indices_by_image": [None] * experiment_dataset.n_units})
        return tuple(outputs)

    monkeypatch.setattr(refine_mod, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid", fake_get_grid)
    monkeypatch.setattr(refine_mod, "get_relion_rotation_grid_eulers", fake_get_grid_eulers)
    monkeypatch.setattr(refine_mod, "run_em", fake_run_em)
    monkeypatch.setattr(refine_mod, "_run_local_search_iteration", fake_grouped_local_search)
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

    prev_eulers_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_eulers_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=123),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_rotation_eulers=[prev_eulers_h1, prev_eulers_h2],
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
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(np.asarray(rotation_grid_rotations).shape[0], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(experiment_dataset.n_units),
        )
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_ids = np.zeros(experiment_dataset.n_units, dtype=np.int32)
            best_pose_details = (best_rots, best_trans, best_ids)
        return _pack_fake_local_search_outputs(
            base_outputs,
            relion_stats,
            noise_stats,
            kwargs,
            experiment_dataset.n_units,
            best_pose_details,
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
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(np.asarray(rotation_grid_rotations).shape[0], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(experiment_dataset.n_units),
        )
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_ids = np.zeros(experiment_dataset.n_units, dtype=np.int32)
            best_pose_details = (best_rots, best_trans, best_ids)
        return _pack_fake_local_search_outputs(
            base_outputs,
            relion_stats,
            noise_stats,
            kwargs,
            experiment_dataset.n_units,
            best_pose_details,
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
    for ds in half_datasets:
        ds.voxel_size = 4.25
    prev_h1 = np.array([[1.6, -2.4]], dtype=np.float32)
    prev_h2 = np.array([[-1.6, 2.4]], dtype=np.float32)
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
        init_healpix_order=1,
        max_healpix_order=1,
        init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
        skip_final_iteration=True,
    )

    expected_h1 = relion_translation_search_base(prev_h1) + chosen_trans[None, :]
    expected_h2 = relion_translation_search_base(prev_h2) + chosen_trans[None, :]

    best_hist = result["best_translations_history"]
    assert len(best_hist) == 1
    np.testing.assert_allclose(
        np.concatenate(best_hist[0], axis=0),
        np.concatenate([expected_h1, expected_h2], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_kclass_recomputes_mstep_tau2_from_iref_power_spectrum(
    rng,
    init_volume,
    monkeypatch,
):
    """Class3D M-step tau2 comes from current Iref power, not previous model.star."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    half_datasets = [MockDataset(1, rng), MockDataset(1, rng)]
    n_classes = 2
    grid_scale = float(VOLUME_SHAPE[0]) ** 4
    class_tau2 = np.asarray(
        [
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ],
        dtype=np.float64,
    )
    iref_tau2 = np.asarray(
        [
            [101.0, 102.0, 103.0, 104.0, 105.0],
            [201.0, 202.0, 203.0, 204.0, 205.0],
        ],
        dtype=np.float64,
    )
    iref_tau2_calls = []

    def fake_iref_tau2(*_args, **kwargs):
        assert kwargs.get("return_details") is True
        class_idx = len(iref_tau2_calls) % n_classes
        iref_tau2_calls.append(class_idx)
        tau2_shells_relion = iref_tau2[class_idx] / grid_scale
        return (
            jnp.full(VOLUME_SIZE, tau2_shells_relion[0], dtype=jnp.float32),
            {"tau2_shells": tau2_shells_relion},
        )

    def fake_run_dense_k_class_em(
        experiment_dataset,
        means,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        del mean_variance, noise_variance, translations, disc_type
        n_images = int(experiment_dataset.n_units)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        per_class_stats = tuple(
            RelionStats(
                log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            )
            for _ in range(n_classes)
        )
        per_class_noise = tuple(
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(n_images) / float(n_classes),
            )
            for _ in range(n_classes)
        )
        aggregate_noise = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(n_images),
        )
        return KClassEMResult(
            new_means=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
            Ft_y=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
            Ft_ctf=jnp.ones((n_classes, recon_vol_size), dtype=jnp.complex64),
            per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
            class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            class_responsibilities=jnp.full((n_classes, n_images), 1.0 / n_classes, dtype=jnp.float32),
            class_posterior_sums=jnp.full(n_classes, n_images / n_classes, dtype=jnp.float32),
            stats=per_class_stats[0],
            per_class_stats=per_class_stats,
            noise_stats=per_class_noise,
            aggregate_noise_stats=aggregate_noise,
            best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
            best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
            best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
        )

    monkeypatch.setattr(regularization_module, "compute_relion_tau2_from_iref_power_spectrum", fake_iref_tau2)
    monkeypatch.setattr(refine_mod, "run_dense_k_class_em", fake_run_dense_k_class_em)

    result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32),
        _make_rotations(1, seed=123),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=1,
        init_current_size=4,
        adaptive_oversampling=0,
        nside_level=1,
        init_healpix_order=1,
        max_healpix_order=1,
        n_classes=n_classes,
        init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        replay_iteration_overrides=[{"class_tau2": class_tau2}],
        skip_final_iteration=True,
    )

    assert len(result["tau2_radial_trajectory"]) == 1
    np.testing.assert_allclose(result["tau2_radial_trajectory"][0], iref_tau2, rtol=0.0, atol=1e-5)

    init_tau2_volume = jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32) * 3.0
    init_result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        init_tau2_volume,
        _make_rotations(1, seed=456),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=1,
        init_current_size=4,
        adaptive_oversampling=0,
        nside_level=1,
        init_healpix_order=1,
        max_healpix_order=1,
        n_classes=n_classes,
        init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        skip_final_iteration=True,
    )
    assert len(init_result["tau2_radial_trajectory"]) == 1
    np.testing.assert_allclose(init_result["tau2_radial_trajectory"][0], iref_tau2, rtol=0.0, atol=1e-5)
    assert iref_tau2_calls == [0, 1, 0, 1]

    same_iter_tau2 = class_tau2 + 1000.0
    monkeypatch.setenv("RECOVAR_KCLASS_REPLAY_TAU2", "1")
    replay_result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32),
        _make_rotations(1, seed=789),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=1,
        init_current_size=4,
        adaptive_oversampling=0,
        nside_level=1,
        init_healpix_order=1,
        max_healpix_order=1,
        n_classes=n_classes,
        init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        replay_iteration_overrides=[{"class_tau2": class_tau2}, {"class_tau2": same_iter_tau2}],
        skip_final_iteration=True,
    )

    assert len(replay_result["tau2_radial_trajectory"]) == 1
    np.testing.assert_allclose(replay_result["tau2_radial_trajectory"][0], same_iter_tau2, rtol=0.0, atol=1e-5)
    assert iref_tau2_calls == [0, 1, 0, 1]

    monkeypatch.setenv("RECOVAR_KCLASS_REPLAY_TAU2_SAME_ITER", "1")
    same_iter_replay_result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32),
        _make_rotations(1, seed=790),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=1,
        init_current_size=4,
        adaptive_oversampling=0,
        nside_level=1,
        init_healpix_order=1,
        max_healpix_order=1,
        n_classes=n_classes,
        init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        replay_iteration_overrides=[{"class_tau2": class_tau2}, {"class_tau2": same_iter_tau2}],
        skip_final_iteration=True,
    )

    assert len(same_iter_replay_result["tau2_radial_trajectory"]) == 1
    np.testing.assert_allclose(
        same_iter_replay_result["tau2_radial_trajectory"][0],
        same_iter_tau2,
        rtol=0.0,
        atol=1e-5,
    )
    assert iref_tau2_calls == [0, 1, 0, 1]


def test_relion_mode_dense_k_class_writes_absolute_translations_from_previous_offset(
    rng,
    init_volume,
    monkeypatch,
):
    """Dense K-class RELION-mode writeback should use old_offset + selected delta."""
    import recovar.em.dense_single_volume.iteration_loop as refine_mod

    half_datasets = [MockDataset(1, rng), MockDataset(1, rng)]
    for ds in half_datasets:
        ds.voxel_size = 4.25
    prev_h1 = np.array([[1.6, -2.4]], dtype=np.float32)
    prev_h2 = np.array([[-1.6, 2.4]], dtype=np.float32)
    selected_by_half = [
        np.array([[0.25, -0.5]], dtype=np.float32),
        np.array([[-0.75, 1.5]], dtype=np.float32),
    ]
    dense_calls = []

    def fake_run_dense_k_class_em(
        experiment_dataset,
        means,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        **kwargs,
    ):
        _ = (means, mean_variance, noise_variance, translations, disc_type)
        half_idx = len(dense_calls)
        dense_calls.append(kwargs)
        n_classes = int(np.asarray(means).shape[0])
        n_images = int(experiment_dataset.n_units)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = VOLUME_SIZE * kwargs.get("reconstruction_padding_factor", 1) ** 3
        selected = np.broadcast_to(selected_by_half[half_idx], (n_images, 2)).astype(np.float32)
        per_class_stats = tuple(
            RelionStats(
                log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            )
            for _ in range(n_classes)
        )
        per_class_noise = tuple(
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(n_images) / float(n_classes),
            )
            for _ in range(n_classes)
        )
        aggregate_noise = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(n_images),
        )
        return KClassEMResult(
            new_means=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
            Ft_y=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
            Ft_ctf=jnp.ones((n_classes, recon_vol_size), dtype=jnp.complex64),
            per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
            class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            class_responsibilities=jnp.full((n_classes, n_images), 1.0 / n_classes, dtype=jnp.float32),
            class_posterior_sums=jnp.full(n_classes, n_images / n_classes, dtype=jnp.float32),
            stats=per_class_stats[0],
            per_class_stats=per_class_stats,
            noise_stats=per_class_noise,
            aggregate_noise_stats=aggregate_noise,
            best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
            best_pose_translations=jnp.asarray(selected, dtype=jnp.float32),
            best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
        )

    monkeypatch.setattr(refine_mod, "run_dense_k_class_em", fake_run_dense_k_class_em)

    result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(1, seed=123),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=1,
        init_current_size=16,
        adaptive_oversampling=0,
        nside_level=1,
        init_healpix_order=1,
        max_healpix_order=1,
        init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
        n_classes=2,
        init_class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        skip_final_iteration=True,
    )

    expected_h1 = relion_translation_search_base(prev_h1) + selected_by_half[0]
    expected_h2 = relion_translation_search_base(prev_h2) + selected_by_half[1]
    np.testing.assert_allclose(
        dense_calls[0]["translation_prior_centers"],
        relion_sigma_offset_prior_center(prev_h1),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        dense_calls[1]["translation_prior_centers"],
        relion_sigma_offset_prior_center(prev_h2),
        rtol=1e-6,
        atol=1e-6,
    )
    expected_translation_log_prior = make_relion_translation_log_prior(
        np.array([[0.0, 0.0]], dtype=np.float32),
        half_datasets[0].voxel_size,
        sigma_offset_angstrom=10.0,
        prior_centers=relion_translation_prior_center(prev_h1, half_datasets[0].voxel_size),
        offset_range_pixels=None,
    )
    np.testing.assert_allclose(
        dense_calls[0]["translation_log_prior"],
        expected_translation_log_prior,
        rtol=1e-6,
        atol=1e-6,
    )
    assert all(call["relion_half_volume_mstep"] is False for call in dense_calls)
    best_hist = result["best_translations_history"]
    assert len(best_hist) == 1
    np.testing.assert_allclose(best_hist[0][0], expected_h1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(best_hist[0][1], expected_h2, rtol=1e-6, atol=1e-6)
    assert len(dense_calls) == 2


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
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            assignment,
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
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            fine_rot = _selected_rotation_matrices(
                np.array([fine_idx], dtype=np.int32),
                None,
                build_local_search_grid_metadata(int(healpix_order)),
            )[0].astype(np.float32)
            best_rots = np.repeat(fine_rot[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.repeat(
                np.asarray(translations)[trans_idx : trans_idx + 1], experiment_dataset.n_units, axis=0
            )
            best_ids = np.full(experiment_dataset.n_units, fine_idx, dtype=np.int32)
            best_pose_details = (best_rots, best_trans, best_ids)
        return _pack_fake_local_search_outputs(
            base_outputs,
            relion_stats,
            noise_stats,
            kwargs,
            experiment_dataset.n_units,
            best_pose_details,
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

    prev_eulers_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_eulers_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    result = refine_single_volume(
        half_datasets,
        init_volume,
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        _make_rotations(order_sizes[4], seed=321),
        translations,
        disc_type="linear_interp",
        max_iter=1,
        image_batch_size=N_IMAGES,
        rotation_block_size=order_sizes[4],
        init_current_size=16,
        adaptive_oversampling=1,
        nside_level=4,
        init_healpix_order=4,
        max_healpix_order=4,
        init_previous_best_rotation_eulers=[prev_eulers_h1, prev_eulers_h2],
        perturb_factor=0.0,
    )

    expected_rotation = _selected_rotation_matrices(
        np.array([fine_idx], dtype=np.int32),
        None,
        build_local_search_grid_metadata(5),
    )
    expected_euler = iteration_loop_module.utils.R_to_relion(expected_rotation, degrees=True)[0].astype(np.float32)
    observed = np.asarray(result["best_rotation_eulers_history"][0], dtype=np.float32).reshape(-1, 3)
    assert observed.shape[0] == N_IMAGES
    np.testing.assert_allclose(
        observed,
        np.repeat(expected_euler[None, :], N_IMAGES, axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
