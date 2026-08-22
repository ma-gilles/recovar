"""Sanity perf test: bucketed sparse pass-2 must NOT recompile per image.

The original ``compute_pass2_stats_sparse`` has a Python for-loop over
particles that calls ``run_em(..., image_batch_size=1, ...)`` once per
image, with a different XLA shape each time.  On the 5k fixture this
caused thousands of separate JIT compiles and made iter-1 take >50 min.

This test:
  * builds a synthetic dataset with N images that have *varied* per-image
    significant-rotation counts (the trigger for the recompile bug);
  * monkey-patches ``jax.jit`` so we count the number of distinct
    compiled trace cache keys produced during one call;
  * asserts the bucketed path produces ≪ N compiled programs (i.e.,
    bounded by the number of bucket sizes), whereas the per-image
    reference path would scale with the number of distinct rotation
    counts.

We use a tiny mock dataset so the test is fast on a login node.
"""

from __future__ import annotations

import gc
import inspect
import logging
import os
import weakref

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.core as core
import recovar.core.fourier_transform_utils as ftu
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
from recovar.em.dense_single_volume.helpers.oversampling import (
    compute_pass2_stats_sparse,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    apply_half_translation_phases,
    half_translation_phase_table,
)
from recovar.em.dense_single_volume.helpers.projection import compute_noise_block
from recovar.em.dense_single_volume.helpers.significance import (
    ComplementSignificantSampleIndices,
    compact_significant_sample_indices_from_mask,
    significant_sample_count,
    significant_sample_ids,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    SparseCandidateMask,
    _accumulate_adjoint_block_chunked,
    _active_flat_row_indices_from_probs_sum_t,
    _active_image_indices_for_rotation_rows,
    _active_row_grouping_for_canonical_matmul,
    _active_row_grouping_shape,
    _adjoint_block_chunk_rows,
    _best_compact_pair_from_scores,
    _bucket_pass2_inputs,
    _bucket_sparse_k_class_compact_pair_counts,
    _bucket_sparse_k_class_compact_pair_inputs,
    _bucket_sparse_k_class_pass2_inputs,
    _build_compact_pair_bucket_arrays,
    _build_compact_pair_bucket_arrays_from_per_image_inputs,
    _build_k_class_bucket_arrays,
    _candidate_mask_count,
    _coalesce_tail_bucket_sizes,
    _compact_k_class_pair_plan_stats,
    _compact_k_class_pair_plan_stats_from_counts,
    _compact_pair_buckets_for_execution_threshold,
    _compact_pair_counts_from_candidate_masks,
    _compact_pair_dense_mstep_max_bytes_for_pass,
    _compact_pair_dense_probs_and_reductions,
    _compact_pair_execution_enabled_for_pass,
    _compact_pair_execution_mask_excluding_full_support,
    _compact_pair_hybrid_threshold_reports,
    _compact_pair_image_mask_for_threshold,
    _compact_pair_max_images_per_microbatch_for_pass,
    _compact_pair_min_bucket_size_for_pass,
    _compact_pair_mstep_mode_for_pass,
    _compact_pair_prepare_max_images_per_microbatch,
    _compact_pair_tail_bucket_coalesce_params_for_pass,
    _compact_pair_weighted_image_sums,
    _compact_pair_weighted_image_sums_dense,
    _compact_pair_weighted_image_sums_pair_sparse,
    _compact_pair_weighted_rotation_and_image_sums,
    _compact_pair_weighted_rotation_and_image_sums_pair_sparse,
    _compact_pair_weighted_rotation_sums,
    _compact_pair_weighted_rotation_sums_dense,
    _compact_pair_weighted_rotation_sums_pair_sparse,
    _compute_active_noise_rows_chunked,
    _compute_noise_block_and_norm_residual_chunked,
    _compute_noise_block_chunked,
    _compute_sparse_pass2_projections_block,
    _compute_sparse_pass2_windowed_projections_block,
    _exact_raw_diff2_cache_estimated_bytes,
    _exact_raw_diff2_cache_fits_budget,
    _exact_raw_diff2_cache_limit_bytes,
    _flat_image_indices_for_rotation_rows,
    _half_translation_phase_table_for_indices,
    _hybrid_k_class_compact_pair_execution_buckets,
    _logsumexp_pass2_bucket_score_only,
    _logsumexp_pass2_pairs_score_only,
    _max_adjoint_block_bytes_for_pass,
    _max_hypotheses_per_microbatch_for_pass,
    _max_images_for_sparse_pass2_translation_tile,
    _max_images_for_translation_tile,
    _max_noise_block_bytes_for_pass,
    _max_projected_rotations_per_call_for_pass,
    _max_projection_gather_bytes_for_pass,
    _max_translation_tile_bytes_for_pass,
    _maybe_prepare_sparse_k_class_compact_pair_plan,
    _normalize_pass2_bucket,
    _normalize_pass2_bucket_score_only,
    _normalize_pass2_bucket_with_log_z,
    _normalize_pass2_pairs_score_only,
    _normalize_pass2_pairs_with_log_z,
    _nvidia_smi_visible_device_memory_bytes,
    _pass2_conservative_dump_execution_enabled,
    _pass2_dump_enabled,
    _prepare_bucket_io,
    _prepare_per_image_compact_candidate_pairs,
    _prepare_per_image_pass2_inputs,
    _projection_budget_pixels_for_pass,
    _projection_cache_budget_complex_dtype,
    _projection_cache_enabled_for_pass,
    _projection_cache_fits_budget,
    _projection_cache_max_bytes_for_pass,
    _projection_cache_transient_bytes,
    _projection_gather_bytes_per_rotation_row,
    _projection_rotation_chunk_size,
    _rectangular_active_prematmul_is_efficient,
    _rectangular_active_weighted_sums_or_none,
    _relion_cuda_corr_img_from_rfloat_ctf,
    _relion_cuda_pixel_correction_from_rfloat_ctf,
    _relion_fine_mstep_prune_mode,
    _relion_joint_winner_take_all_masks,
    _relion_pass2_reconstruction_joint_masks,
    _relion_pass2_reconstruction_pair_probs,
    _relion_pass2_reconstruction_probs,
    _relion_translation_angles_f32,
    _score_pass2_bucket_normalized_cc,
    _score_pass2_bucket_relion_gpu_diff2,
    _score_pass2_bucket_relion_gpu_diff2_raw,
    _score_pass2_pairs_normalized_cc,
    _score_pass2_pairs_relion_gpu_diff2,
    _score_pass2_pairs_relion_gpu_diff2_raw,
    _select_active_flat_rows,
    _select_active_flat_values,
    _select_active_noise_rows,
    _small_bucket_coalesce_size_for_pass,
    _split_compact_pair_buckets_by_projection_gather_budget,
    _tail_bucket_coalesce_params_for_pass,
    _translation_tile_half_pixels_for_budget,
    _validate_k_class_execution_bucket_partition,
    _weighted_image_power_shells_and_per_image,
    _windowed_translation_tile_cap_enabled_for_pass,
    _winner_take_all_bucket_probs_from_global_argmax,
)
from recovar.em.dense_single_volume.k_class import (
    _build_fine_grid_significance_mask,
    _fine_support_stats,
    _k_class_fused_relion_fine_mstep_prune_mode_override,
    _run_sparse_k_class_adaptive_pass2,
    _use_fused_sparse_k_class_pass2,
)
from recovar.em.dense_single_volume.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_mstep_sums,
    compute_local_weighted_sums,
    flatten_bucket_rows,
)
from scripts import validate_bpref_device_signature as bpref_signature_validator

pytestmark = pytest.mark.unit


def test_relion_corr_img_squares_rfloat_ctf_before_xfloat_cast():
    inverse_noise = np.asarray([0.13333298, 1.750001], dtype=np.float32)
    ctf_rfloat = np.asarray([0.994443123456, -0.7135792468], dtype=np.float64)
    expected = np.asarray(
        inverse_noise.astype(np.float64) * (ctf_rfloat * ctf_rfloat),
        dtype=np.float32,
    )
    float_ctf = ctf_rfloat.astype(np.float32)
    rejected_float_path = np.asarray(
        inverse_noise * (float_ctf * float_ctf),
        dtype=np.float32,
    )
    assert np.any(expected != rejected_float_path)

    actual = np.asarray(
        _relion_cuda_corr_img_from_rfloat_ctf(inverse_noise, ctf_rfloat)
    )
    np.testing.assert_array_equal(actual, expected)


def test_relion_pixel_correction_divides_by_rfloat_ctf_before_xfloat_cast():
    scale = np.asarray([[1.0]], dtype=np.float32)
    ctf_rfloat = np.asarray(
        [[0.07354116995482596, 0.1265216380265534]], dtype=np.float64
    )
    initial = np.asarray(1.0 / scale, dtype=np.float32)
    expected = np.asarray(
        initial.astype(np.float64) / ctf_rfloat,
        dtype=np.float32,
    )
    rejected_float_path = np.asarray(
        initial / ctf_rfloat.astype(np.float32),
        dtype=np.float32,
    )
    assert np.any(expected != rejected_float_path)

    actual = np.asarray(
        _relion_cuda_pixel_correction_from_rfloat_ctf(scale, ctf_rfloat)
    )
    np.testing.assert_array_equal(actual, expected)


# Mock dataset (mirrors test_sparse_pass2_bucketed_parity.MockDataset).
IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512


def test_compute_local_ctf_sums_from_probs_sum_t_matches_dense_helper():
    rng = np.random.default_rng(112358)
    probs = rng.random((3, 5, 7), dtype=np.float32)
    probs[0, 2, :] = 0.0
    ctf2_over_nv = 0.25 + rng.random((3, 11), dtype=np.float32)

    dense = compute_local_ctf_sums(jnp.asarray(probs), jnp.asarray(ctf2_over_nv))
    from_probs_sum = compute_local_ctf_sums_from_probs_sum_t(
        jnp.sum(jnp.asarray(probs), axis=-1),
        jnp.asarray(ctf2_over_nv),
    )

    np.testing.assert_allclose(np.asarray(from_probs_sum), np.asarray(dense), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(from_probs_sum)[0, 2], np.zeros_like(np.asarray(from_probs_sum)[0, 2]))


def test_k_class_pass2_dump_stop_is_env_gated_diagnostic_only():
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed

    source = inspect.getsource(sparse_pass2_bucketed)

    assert "class Pass2DumpComplete" in source
    assert 'RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET' in source
    assert "if bucket_dump_count:" in source
    assert "raise Pass2DumpComplete" in source


def test_k1_pass2_dump_progress_requires_complete_target_set(tmp_path):
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _k1_pass2_dump_progress,
    )

    targets = {7, 42, 105}
    first = tmp_path / "pass2_orig000007_cs100.npz"
    first.touch()
    assert _k1_pass2_dump_progress(
        dump_dir=tmp_path,
        target_original_indices=targets,
        current_size=100,
    ) == (1, 3)

    (tmp_path / "pass2_orig000042_cs100.npz").touch()
    (tmp_path / "pass2_orig000105_cs100.npz").touch()
    assert _k1_pass2_dump_progress(
        dump_dir=tmp_path,
        target_original_indices=targets,
        current_size=100,
    ) == (3, 3)


def test_pass2_dump_does_not_change_planner_without_conservative_opt_in(monkeypatch):
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_CONSERVATIVE_EXECUTION", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES", raising=False)

    def planned_cap():
        return _max_hypotheses_per_microbatch_for_pass(
            score_only=True,
            use_window=True,
            has_external_normalization=False,
            conservative_dump_execution=_pass2_conservative_dump_execution_enabled(),
            n_score_pixels=652,
            device_memory_bytes=80 * 1024**3,
        )

    production_cap = planned_cap()
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", "/tmp/pass2-dump")
    assert _pass2_dump_enabled()
    assert not _pass2_conservative_dump_execution_enabled()
    assert planned_cap() == production_cap

    monkeypatch.setenv("RECOVAR_PASS2_DUMP_CONSERVATIVE_EXECUTION", "1")
    assert _pass2_conservative_dump_execution_enabled()
    assert planned_cap() < production_cap


def test_pass2_projection_cache_override_supports_matched_dump_ab(monkeypatch):
    fine_rotations = np.zeros((3, 3, 3), dtype=np.float32)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE", raising=False)

    assert _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=False,
    )
    assert not _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=True,
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE", "on")
    assert _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=False,
    )
    assert _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=True,
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE", "off")
    assert not _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=False,
    )
    assert not _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=True,
    )
    assert not _projection_cache_enabled_for_pass(
        fine_rotations_override=None,
        dump_pass2_operands=False,
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE", "invalid")
    with pytest.raises(ValueError, match="must be 'auto', 'on', or 'off'"):
        _projection_cache_enabled_for_pass(
            fine_rotations_override=fine_rotations,
            dump_pass2_operands=False,
        )


def _assert_relion_stats_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(
        np.asarray(actual.log_evidence_per_image),
        np.asarray(expected.log_evidence_per_image),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.best_log_score_per_image),
        np.asarray(expected.best_log_score_per_image),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.max_posterior_per_image),
        np.asarray(expected.max_posterior_per_image),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.rotation_posterior_sums),
        np.asarray(expected.rotation_posterior_sums),
        rtol=rtol,
        atol=atol,
    )


def _assert_noise_stats_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    assert len(actual) == len(expected)
    for actual_stats, expected_stats in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_stats.wsum_sigma2_noise),
            np.asarray(expected_stats.wsum_sigma2_noise),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(actual_stats.wsum_img_power),
            np.asarray(expected_stats.wsum_img_power),
            rtol=rtol,
            atol=atol,
        )
        actual_norm = getattr(actual_stats, "wsum_norm_correction", None)
        expected_norm = getattr(expected_stats, "wsum_norm_correction", None)
        if actual_norm is None or expected_norm is None:
            assert actual_norm is None and expected_norm is None
        else:
            np.testing.assert_allclose(
                np.asarray(actual_norm),
                np.asarray(expected_norm),
                rtol=rtol,
                atol=atol,
            )
        for field in ("wsum_scale_correction_xa", "wsum_scale_correction_aa"):
            actual_scale = getattr(actual_stats, field, None)
            expected_scale = getattr(expected_stats, field, None)
            if actual_scale is None or expected_scale is None:
                assert actual_scale is None and expected_scale is None
            else:
                np.testing.assert_allclose(
                    np.asarray(actual_scale),
                    np.asarray(expected_scale),
                    rtol=rtol,
                    atol=atol,
                )
        assert actual_stats.wsum_sigma2_offset == pytest.approx(expected_stats.wsum_sigma2_offset, abs=atol)
        assert actual_stats.sumw == pytest.approx(expected_stats.sumw, abs=atol)


def _assert_noise_residual_terms_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    assert len(actual) == len(expected)
    for actual_stats, expected_stats in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_stats.wsum_sigma2_noise),
            np.asarray(expected_stats.wsum_sigma2_noise),
            rtol=rtol,
            atol=atol,
        )
        assert actual_stats.wsum_sigma2_offset == pytest.approx(expected_stats.wsum_sigma2_offset, abs=atol)


def _assert_k_class_noise_sumw_matches_class_mass(result, *, rtol=1e-5, atol=1e-5):
    assert result.noise_stats is not None
    class_mass_source = result.class_mstep_posterior_sums
    if class_mass_source is None:
        class_mass_source = result.class_posterior_sums
    class_mass = np.asarray(class_mass_source, dtype=np.float64)
    for stats, expected_mass in zip(result.noise_stats, class_mass, strict=True):
        assert float(stats.sumw) == pytest.approx(float(expected_mass), rel=rtol, abs=atol)
    if result.aggregate_noise_stats is not None:
        assert float(result.aggregate_noise_stats.sumw) == pytest.approx(float(np.sum(class_mass)), rel=rtol, abs=atol)
        np.testing.assert_allclose(
            np.asarray(result.aggregate_noise_stats.wsum_img_power),
            np.sum([np.asarray(stats.wsum_img_power) for stats in result.noise_stats], axis=0),
            rtol=rtol,
            atol=atol,
        )


def test_active_row_selection_padding_masks_dummy_rows() -> None:
    probs_sum_t = jnp.asarray(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    active_indices, active_mask, active_count = _active_flat_row_indices_from_probs_sum_t(
        probs_sum_t,
        pad_multiple=4,
    )

    assert active_count == 3
    assert active_indices.shape == (4,)
    np.testing.assert_array_equal(active_indices[:active_count], np.asarray([0, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(active_mask, np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32))

    values = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2)
    rotations = jnp.arange(18, dtype=jnp.float32).reshape(6, 3)
    active_values, active_rotations = _select_active_flat_rows(
        values,
        rotations,
        active_indices,
        active_mask,
    )
    active_scalars = _select_active_flat_values(
        jnp.arange(6, dtype=jnp.float32).reshape(2, 3, 1),
        active_indices,
        active_mask,
    )

    np.testing.assert_allclose(
        np.asarray(active_values[:active_count]),
        np.asarray(flatten_bucket_rows(values))[[0, 2, 4]],
    )
    np.testing.assert_allclose(np.asarray(active_values[active_count:]), 0.0)
    np.testing.assert_allclose(np.asarray(active_scalars[:active_count, 0]), np.asarray([0.0, 2.0, 4.0]))
    np.testing.assert_allclose(np.asarray(active_scalars[active_count:]), 0.0)
    np.testing.assert_allclose(np.asarray(active_rotations[:active_count]), np.asarray(rotations)[[0, 2, 4]])


def test_active_row_grouping_preserves_slots_for_padded_unsorted_rows() -> None:
    image_indices, active_slots, grouped_rows = _active_row_grouping_for_canonical_matmul(
        np.asarray([7, 1, 4, 2, 9, 7], dtype=np.int32),
        np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        n_images=3,
        n_rotation_rows=4,
    )

    np.testing.assert_array_equal(image_indices, np.asarray([1, 0, 1, 0, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(active_slots, np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        grouped_rows,
        np.asarray(
            [
                [1, 2],
                [3, 0],
                [1, 0],
            ],
            dtype=np.int32,
        ),
    )

    _, empty_slots, empty_grouped_rows = _active_row_grouping_for_canonical_matmul(
        np.asarray([0, 0], dtype=np.int32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        n_images=3,
        n_rotation_rows=4,
    )
    np.testing.assert_array_equal(empty_slots, np.zeros(2, dtype=np.int32))
    np.testing.assert_array_equal(empty_grouped_rows, np.zeros((3, 1), dtype=np.int32))


def test_active_row_grouping_shape_detects_dense_expansion() -> None:
    active_count, active_slots, grouped_rows = _active_row_grouping_shape(
        np.asarray([0, 1, 2, 3, 4, 8], dtype=np.int32),
        np.ones(6, dtype=np.float32),
        n_images=3,
        n_rotation_rows=4,
    )

    assert active_count == 6
    assert active_slots == 4
    assert grouped_rows == 12

    active_count, active_slots, grouped_rows = _active_row_grouping_shape(
        np.asarray([0, 4, 8, 0], dtype=np.int32),
        np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        n_images=3,
        n_rotation_rows=4,
    )

    assert active_count == 3
    assert active_slots == 1
    assert grouped_rows == 3


def test_rectangular_active_prematmul_guard_rejects_near_dense_grouping() -> None:
    use_prematmul, active_count, active_slots, grouped_rows, dense_rows, grouped_dense_ratio = (
        _rectangular_active_prematmul_is_efficient(
            np.asarray([0, 1, 2, 3, 4, 8], dtype=np.int32),
            np.ones(6, dtype=np.float32),
            n_images=3,
            n_rotation_rows=4,
            max_grouped_dense_ratio=0.5,
        )
    )

    assert use_prematmul is False
    assert active_count == 6
    assert active_slots == 4
    assert grouped_rows == 12
    assert dense_rows == 12
    assert grouped_dense_ratio == 1.0

    use_prematmul, active_count, active_slots, grouped_rows, dense_rows, grouped_dense_ratio = (
        _rectangular_active_prematmul_is_efficient(
            np.asarray([0, 4, 8], dtype=np.int32),
            np.ones(3, dtype=np.float32),
            n_images=3,
            n_rotation_rows=4,
            max_grouped_dense_ratio=0.5,
        )
    )

    assert use_prematmul is True
    assert active_count == 3
    assert active_slots == 1
    assert grouped_rows == 3
    assert dense_rows == 12
    assert grouped_dense_ratio == 0.25


def test_active_row_selection_does_not_flatten_full_bucket(monkeypatch) -> None:
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    values = jnp.arange(3 * 4 * 2, dtype=jnp.float32).reshape(3, 4, 2)
    scalar_values = jnp.arange(3 * 4, dtype=jnp.float32).reshape(3, 4, 1)
    rotations = jnp.arange(3 * 4 * 9, dtype=jnp.float32).reshape(3 * 4, 3, 3)
    active_indices = np.asarray([0, 3, 4, 9, 11, 0], dtype=np.int32)
    active_mask = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    expected_values = np.asarray(values).reshape(12, 2)[active_indices] * active_mask[:, None]
    expected_scalars = np.asarray(scalar_values).reshape(12, 1)[active_indices] * active_mask[:, None]

    def fail_if_full_flattened(_values):
        raise AssertionError("active-row selection should not flatten the full bucket")

    monkeypatch.setattr(bucketed_mod, "flatten_bucket_rows", fail_if_full_flattened)

    active_values, active_rotations = bucketed_mod._select_active_flat_rows(
        values,
        rotations,
        active_indices,
        active_mask,
    )
    active_scalars = bucketed_mod._select_active_flat_values(
        scalar_values,
        active_indices,
        active_mask,
    )

    np.testing.assert_allclose(np.asarray(active_values), expected_values)
    np.testing.assert_allclose(np.asarray(active_scalars), expected_scalars)
    np.testing.assert_allclose(np.asarray(active_rotations), np.asarray(rotations)[active_indices])


def test_flat_image_indices_follow_rotation_rows_not_pair_rows() -> None:
    active_indices = np.asarray([0, 2, 3, 5], dtype=np.int32)
    active_mask = np.ones(active_indices.shape, dtype=np.float32)

    image_indices = _select_active_flat_values(
        _flat_image_indices_for_rotation_rows(batch=2, n_rotation_rows=3)[..., None],
        active_indices,
        active_mask,
    )

    np.testing.assert_array_equal(
        np.asarray(image_indices)[:, 0],
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(_active_image_indices_for_rotation_rows(active_indices, active_mask, n_rotation_rows=3)),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )


def test_active_image_indices_masks_padded_rows() -> None:
    active_indices = np.asarray([5, 7, 5, 5], dtype=np.int32)
    active_mask = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    np.testing.assert_array_equal(
        np.asarray(_active_image_indices_for_rotation_rows(active_indices, active_mask, n_rotation_rows=3)),
        np.asarray([1, 2, 0, 0], dtype=np.int32),
    )


def test_select_active_noise_rows_matches_separate_gathers() -> None:
    rng = np.random.default_rng(31)
    batch = 3
    n_rot = 4
    n_pixels = 5
    shape = (batch, n_rot, n_pixels)
    proj = (
        rng.standard_normal(shape).astype(np.float32)
        + 1j * rng.standard_normal(shape).astype(np.float32)
    ).astype(np.complex64)
    proj_abs2 = np.abs(proj).astype(np.float32) ** 2
    summed = (
        rng.standard_normal(shape).astype(np.float32)
        + 1j * rng.standard_normal(shape).astype(np.float32)
    ).astype(np.complex64)
    ctf_probs = rng.random(shape, dtype=np.float32)
    active_indices = np.asarray([0, 5, 11, 2, 5], dtype=np.int32)
    active_mask = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    fused = _select_active_noise_rows(
        jnp.asarray(proj),
        jnp.asarray(proj_abs2),
        jnp.asarray(summed),
        jnp.asarray(ctf_probs),
        active_indices,
        active_mask,
        n_rotation_rows=n_rot,
    )

    np.testing.assert_allclose(
        np.asarray(fused[0]),
        np.asarray(_select_active_flat_values(jnp.asarray(proj), active_indices, active_mask)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(fused[1]),
        np.asarray(_select_active_flat_values(jnp.asarray(proj_abs2), active_indices, active_mask)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(fused[2]),
        np.asarray(_select_active_flat_values(jnp.asarray(summed), active_indices, active_mask)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(fused[3]),
        np.asarray(_select_active_flat_values(jnp.asarray(ctf_probs), active_indices, active_mask)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(
        np.asarray(fused[4]),
        np.asarray(_active_image_indices_for_rotation_rows(active_indices, active_mask, n_rotation_rows=n_rot)),
    )


def _assert_best_pose_outputs_close(actual, expected, *, rtol=1e-6, atol=1e-6):
    for field in (
        "per_class_best_pose_rotations",
        "per_class_best_pose_translations",
        "per_class_best_pose_rotation_ids",
        "best_pose_rotations",
        "best_pose_translations",
        "best_pose_rotation_ids",
    ):
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if actual_value is None or expected_value is None:
            assert actual_value is None and expected_value is None
            continue
        if isinstance(actual_value, tuple):
            assert len(actual_value) == len(expected_value)
            for actual_item, expected_item in zip(actual_value, expected_value, strict=True):
                np.testing.assert_allclose(
                    np.asarray(actual_item),
                    np.asarray(expected_item),
                    rtol=rtol,
                    atol=atol,
                )
        else:
            np.testing.assert_allclose(
                np.asarray(actual_value),
                np.asarray(expected_value),
                rtol=rtol,
                atol=atol,
            )


def _assert_k_class_extra_outputs_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    _assert_relion_stats_close(actual.stats, expected.stats, rtol=rtol, atol=atol)
    assert len(actual.per_class_stats) == len(expected.per_class_stats)
    for actual_stats, expected_stats in zip(actual.per_class_stats, expected.per_class_stats, strict=True):
        _assert_relion_stats_close(actual_stats, expected_stats, rtol=rtol, atol=atol)
    _assert_noise_stats_close(actual.noise_stats, expected.noise_stats, rtol=rtol, atol=atol)
    _assert_noise_stats_close(
        None if actual.aggregate_noise_stats is None else (actual.aggregate_noise_stats,),
        None if expected.aggregate_noise_stats is None else (expected.aggregate_noise_stats,),
        rtol=rtol,
        atol=atol,
    )
    _assert_best_pose_outputs_close(actual, expected)


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


def _raw_real_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
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


def test_bucket_count_bounded_under_varied_per_image_rotation_counts():
    """Number of buckets must be bounded by the number of unique quantized sizes,
    not by the number of distinct per-image counts (and certainly not by N_images).
    """
    rng = np.random.default_rng(7)
    n_images = 500
    n_coarse_rot = 48
    n_coarse_trans = 2
    n_fine_trans = 2 * (4**1)  # = 8

    # Random per-image significant counts in [1, 20] — many distinct counts.
    counts = rng.integers(low=1, high=21, size=n_images)
    # Build (rot * n_coarse_trans + trans) flat indices: pick coarse rot, pair with trans 0
    # so candidate_mask is non-empty (fine trans 0 is parent of trans 0).
    sig_indices = [
        (rng.choice(n_coarse_rot, size=int(c), replace=False).astype(np.int32) * n_coarse_trans).astype(np.int32)
        for c in counts
    ]

    # Build per-image inputs the way compute_pass2_stats_sparse_bucketed does.
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import _prepare_per_image_pass2_inputs

    # fine_translation_parent maps fine trans -> coarse trans. With oversampling=1
    # in 2D, each coarse trans expands to 4 children, so trans 0..3 map to coarse 0,
    # trans 4..7 map to coarse 1.
    fine_trans_parent = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

    per_image = _prepare_per_image_pass2_inputs(
        sig_indices,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_trans_parent,
        rotation_log_prior=None,
        random_perturbation=0.0,
    )

    buckets = _bucket_pass2_inputs(per_image, n_fine_trans=n_fine_trans)
    n_distinct_counts = len({int(rots.shape[0]) for rots in per_image["oversampled_rots"]})

    # Quantization should collapse many distinct counts into a few buckets.
    assert len(buckets) < n_distinct_counts + 1, (
        f"Expected fewer than {n_distinct_counts + 1} buckets after quantization, got {len(buckets)}."
    )
    # Must be much smaller than n_images — that's the whole point.
    assert len(buckets) < n_images / 10, f"Got {len(buckets)} buckets for {n_images} images — bucketing too granular."


def test_default_sparse_pass2_budget_keeps_broad_support_batched():
    """Broad soft K-class supports must not fall back to one image per launch."""

    n_images = 26
    n_fine_trans = 116
    n_rot = 1024
    per_image = {
        "oversampled_rots": [np.zeros((n_rot, 3, 3), dtype=np.float32) for _ in range(n_images)],
    }

    default_buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_images_per_microbatch=13,
    )
    old_cap_buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=100_000,
        max_images_per_microbatch=13,
    )

    assert [len(bucket["image_indices"]) for bucket in default_buckets] == [8, 8, 8, 2]
    assert len(old_cap_buckets) == n_images


def test_single_class_sparse_pass2_can_coalesce_small_bucket_tail(monkeypatch):
    """Small sparse tails can opt into fewer execution shapes."""

    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    n_fine_trans = 116
    counts = [16] * 12 + [32] * 7 + [64] * 5 + [128] * 3 + [256]
    per_image = {
        "oversampled_rots": [
            np.zeros((int(count), 3, 3), dtype=np.float32)
            for count in counts
        ],
    }

    baseline = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    coalesced = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
        small_bucket_coalesce_size=128,
    )

    assert sorted({int(bucket["bucket_size"]) for bucket in baseline}) == [16, 32, 64, 128, 256]
    assert sorted({int(bucket["bucket_size"]) for bucket in coalesced}) == [128, 256]
    assert len(coalesced) < len(baseline)
    assert sum(len(bucket["image_indices"]) for bucket in coalesced) == len(counts)


def test_sparse_pass2_auto_small_bucket_coalescing_is_small_dataset_only(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_SMALL_BUCKET_COALESCE_SIZE", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES", raising=False)

    assert _small_bucket_coalesce_size_for_pass(1_000) == 128
    assert _small_bucket_coalesce_size_for_pass(100_000) is None

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES", "0")
    assert _small_bucket_coalesce_size_for_pass(1_000) is None

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_SMALL_BUCKET_COALESCE_SIZE", "256")
    assert _small_bucket_coalesce_size_for_pass(100_000) == 256


def test_sparse_pass2_tail_bucket_coalescing_is_opt_in(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE", raising=False)

    assert _tail_bucket_coalesce_params_for_pass(fused_k_class=False) == (None, None, None)
    assert _tail_bucket_coalesce_params_for_pass(fused_k_class=True) == (None, None, None)

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES", "0")
    assert _tail_bucket_coalesce_params_for_pass(fused_k_class=True) == (None, None, None)

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES", "7")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION", "1.25")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE", "8192")
    assert _tail_bucket_coalesce_params_for_pass(fused_k_class=False) == (7, 1.25, 8192)
    assert _tail_bucket_coalesce_params_for_pass(fused_k_class=True) == (7, 1.25, 8192)


def test_compact_pair_tail_bucket_coalescing_defaults_to_bounded_tail(monkeypatch):
    for name in (
        "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES",
        "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION",
        "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE",
        "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES",
        "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION",
        "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert _tail_bucket_coalesce_params_for_pass(fused_k_class=True) == (None, None, None)
    assert _compact_pair_tail_bucket_coalesce_params_for_pass() == (19, 2.0, 4096)
    assert _compact_pair_tail_bucket_coalesce_params_for_pass(
        default_max_images=1024,
        default_max_inflation=8.0,
        default_min_bucket_size=1,
    ) == (1024, 8.0, 1)

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES", "0")
    assert _compact_pair_tail_bucket_coalesce_params_for_pass() == (None, None, None)

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES", "5")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION", "1.5")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE", "8192")
    assert _compact_pair_tail_bucket_coalesce_params_for_pass() == (5, 1.5, 8192)
    assert _compact_pair_tail_bucket_coalesce_params_for_pass(
        default_max_images=1024,
        default_max_inflation=8.0,
        default_min_bucket_size=1,
    ) == (5, 1.5, 8192)


def test_sparse_pass2_tail_bucket_coalescing_merges_only_bounded_high_tail():
    bucket_sizes = np.asarray(
        [4096] * 20 + [8192] * 2 + [12288] * 3 + [16384] * 2,
        dtype=np.int64,
    )

    coalesced = _coalesce_tail_bucket_sizes(
        bucket_sizes,
        max_images=8,
        max_inflation=2.0,
        min_bucket_size=4096,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
        n_fine_trans=116,
        n_classes=4,
    )

    assert sorted(np.unique(coalesced).astype(int).tolist()) == [4096, 16384]
    assert np.count_nonzero(coalesced == 4096) == 20
    assert np.count_nonzero(coalesced == 16384) == 7


def test_sparse_pass2_tail_bucket_coalescing_respects_inflation_cap():
    bucket_sizes = np.asarray([4096, 4096, 16384], dtype=np.int64)

    coalesced = _coalesce_tail_bucket_sizes(
        bucket_sizes,
        max_images=3,
        max_inflation=1.2,
        min_bucket_size=4096,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
        n_fine_trans=116,
        n_classes=4,
    )

    np.testing.assert_array_equal(coalesced, bucket_sizes)


def test_score_only_sparse_pass_uses_larger_default_bucket_budget(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES", raising=False)

    device_memory = 80 * 1024**3
    n_score_pixels = 652
    assert (
        _max_hypotheses_per_microbatch_for_pass(
            score_only=True,
            use_window=True,
            has_external_normalization=False,
            conservative_dump_execution=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
        > _max_hypotheses_per_microbatch_for_pass(
            score_only=False,
            use_window=True,
            has_external_normalization=False,
            conservative_dump_execution=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
    )
    assert (
        _max_hypotheses_per_microbatch_for_pass(
            score_only=True,
            use_window=True,
            has_external_normalization=False,
            conservative_dump_execution=False,
            n_score_pixels=n_score_pixels * 2,
            device_memory_bytes=device_memory,
        )
        < _max_hypotheses_per_microbatch_for_pass(
            score_only=True,
            use_window=True,
            has_external_normalization=False,
            conservative_dump_execution=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES", "12345")
    assert (
        _max_hypotheses_per_microbatch_for_pass(
            score_only=True,
            use_window=True,
            has_external_normalization=False,
            conservative_dump_execution=False,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
        == 12345
    )


def test_sparse_pass2_auto_hypothesis_cap_matches_80gb_probe_scale(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES", raising=False)

    device_memory = 80 * 1024**3
    n_score_pixels = 652
    cap = _max_hypotheses_per_microbatch_for_pass(
        score_only=True,
        use_window=True,
        has_external_normalization=False,
        conservative_dump_execution=False,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory,
    )

    assert 9_500_000 <= cap <= 10_900_000


def test_sparse_pass2_hypothesis_cap_accounts_for_score_dtype(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES", raising=False)

    device_memory = 80 * 1024**3
    n_score_pixels = 1103
    cap32 = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=True,
        has_external_normalization=False,
        conservative_dump_execution=False,
        fused_k_class=True,
        fused_k_class_count=4,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory,
        score_complex_dtype=jnp.complex64,
    )
    cap64 = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=True,
        has_external_normalization=False,
        conservative_dump_execution=False,
        fused_k_class=True,
        fused_k_class_count=4,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory,
        score_complex_dtype=jnp.complex128,
    )

    assert cap64 == pytest.approx(cap32 / 2, rel=1e-6, abs=1)


def test_fused_k_class_sparse_pass2_reserves_extra_headroom(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", raising=False)

    device_memory = 80 * 1024**3
    n_score_pixels = 1103
    single_class_cap = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=True,
        has_external_normalization=False,
        conservative_dump_execution=False,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory,
    )
    fused_cap = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=True,
        has_external_normalization=False,
        conservative_dump_execution=False,
        fused_k_class=True,
        fused_k_class_count=4,
        n_score_pixels=n_score_pixels,
        device_memory_bytes=device_memory,
    )

    assert fused_cap < single_class_cap
    assert 1_700_000 <= fused_cap <= 2_100_000

    # The 100k K=4/256 fixture has 652 active score pixels.  Keep the
    # nominal float32 candidate-by-pixel block at or below 8 GiB.  The former
    # 6,587,373-candidate cap requested a contiguous 17.04 GiB scorer temporary
    # after earlier JIT shapes had fragmented the A100 allocator.
    low_resolution_cap = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=True,
        has_external_normalization=False,
        conservative_dump_execution=False,
        fused_k_class=True,
        fused_k_class_count=4,
        n_score_pixels=652,
        device_memory_bytes=device_memory,
    )
    assert 3_000_000 <= low_resolution_cap <= 3_400_000
    assert low_resolution_cap // (4 * 24_576) == 33
    per_class_candidates = low_resolution_cap // 4
    estimated_two_gather_bytes = (
        per_class_candidates
        * 652
        * np.dtype(np.complex64).itemsize
        * 2
    )
    assert estimated_two_gather_bytes <= 8 * 1024**3

    for n_classes in (1, 2, 4):
        class_aware_cap = _max_hypotheses_per_microbatch_for_pass(
            score_only=False,
            use_window=True,
            has_external_normalization=False,
            conservative_dump_execution=False,
            fused_k_class=True,
            fused_k_class_count=n_classes,
            n_score_pixels=652,
            device_memory_bytes=device_memory,
        )
        estimated_two_gather_bytes = (
            (class_aware_cap // n_classes)
            * 652
            * np.dtype(np.complex64).itemsize
            * 2
        )
        assert estimated_two_gather_bytes <= 0.10 * device_memory

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", "12345")
    assert (
        _max_hypotheses_per_microbatch_for_pass(
            score_only=False,
            use_window=True,
            has_external_normalization=False,
            conservative_dump_execution=False,
            fused_k_class=True,
            fused_k_class_count=4,
            n_score_pixels=n_score_pixels,
            device_memory_bytes=device_memory,
        )
        == 12345
    )


def test_sparse_pass2_warns_when_env_cap_is_below_auto(monkeypatch, caplog):
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", "2000000")

    caplog.set_level(logging.WARNING, logger="recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed")
    cap = _max_hypotheses_per_microbatch_for_pass(
        score_only=False,
        use_window=True,
        has_external_normalization=False,
        conservative_dump_execution=False,
        fused_k_class=True,
        fused_k_class_count=4,
        n_score_pixels=652,
        device_memory_bytes=80 * 1024**3,
    )

    assert cap == 2_000_000
    assert "below the auto sparse pass-2 cap" in caplog.text
    assert "fragment buckets and slow pass-2" in caplog.text


def test_sparse_pass2_memory_budgets_auto_scale_with_device_memory(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", raising=False)

    small_gpu = 20 * 1024**3
    mid_gpu = 40 * 1024**3
    large_gpu = 80 * 1024**3

    assert _max_translation_tile_bytes_for_pass(large_gpu) == pytest.approx(
        4 * _max_translation_tile_bytes_for_pass(small_gpu),
        rel=1e-6,
    )
    assert _max_translation_tile_bytes_for_pass(
        large_gpu,
        has_external_normalization=True,
    ) < _max_translation_tile_bytes_for_pass(large_gpu)
    assert _max_translation_tile_bytes_for_pass(
        large_gpu,
        fused_k_class=True,
    ) < _max_translation_tile_bytes_for_pass(
        large_gpu,
        has_external_normalization=True,
    )
    assert _projection_cache_max_bytes_for_pass(large_gpu) == pytest.approx(
        4 * _projection_cache_max_bytes_for_pass(small_gpu),
        rel=1e-6,
    )
    assert _max_projection_gather_bytes_for_pass(mid_gpu) == pytest.approx(
        2 * _max_projection_gather_bytes_for_pass(small_gpu),
        rel=1e-6,
    )
    assert _max_projection_gather_bytes_for_pass(large_gpu) == pytest.approx(
        4 * _max_projection_gather_bytes_for_pass(small_gpu),
        rel=1e-6,
    )
    assert _max_noise_block_bytes_for_pass(mid_gpu) == pytest.approx(
        2 * _max_noise_block_bytes_for_pass(small_gpu),
        rel=1e-6,
    )
    assert _max_noise_block_bytes_for_pass(large_gpu) == pytest.approx(
        4 * _max_noise_block_bytes_for_pass(small_gpu),
        rel=1e-6,
    )
    assert _max_adjoint_block_bytes_for_pass(large_gpu) == pytest.approx(
        4 * _max_adjoint_block_bytes_for_pass(small_gpu),
        rel=1e-6,
    )
    assert (
        _max_images_for_translation_tile(
            (256, 256),
            116,
            max_tile_bytes=_max_translation_tile_bytes_for_pass(large_gpu),
        )
        >= 50
    )
    assert (
        _max_images_for_translation_tile(
            (256, 256),
            116,
            max_tile_bytes=_max_translation_tile_bytes_for_pass(
                large_gpu,
                has_external_normalization=True,
            ),
        )
        >= 35
    )
    assert (
        _max_images_for_translation_tile(
            (256, 256),
            116,
            max_tile_bytes=_max_translation_tile_bytes_for_pass(small_gpu),
        )
        >= 13
    )
    assert 17 <= _max_images_for_translation_tile(
        (256, 256),
        116,
        max_tile_bytes=_max_translation_tile_bytes_for_pass(large_gpu, fused_k_class=True),
    ) <= 22

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES", "123456")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", "345678")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES", "234567")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES", "456789")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", "654321")
    assert _max_translation_tile_bytes_for_pass(large_gpu) == 123456
    assert _max_projection_gather_bytes_for_pass(large_gpu) == 345678
    assert _max_noise_block_bytes_for_pass(large_gpu) == 234567
    assert _max_adjoint_block_bytes_for_pass(large_gpu) == 456789
    assert _projection_cache_max_bytes_for_pass(large_gpu) == 654321

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", "0")
    assert _projection_cache_max_bytes_for_pass(large_gpu) == 0


def test_compact_pair_projection_gather_budget_splits_large_bucket():
    n_images = 10
    bucket_size = 8192
    per_image_inputs_by_class = [
        {"oversampled_rots": [np.zeros((bucket_size, 3, 3), dtype=np.float32) for _ in range(n_images)]}
        for _ in range(4)
    ]
    bucket = {"pair_bucket_size": bucket_size, "image_indices": np.arange(n_images, dtype=np.int64)}
    row_bytes = _projection_gather_bytes_per_rotation_row(
        n_score_pixels=17,
        n_recon_pixels=19,
        projection_complex_dtype=np.complex64,
        include_recon_noise=True,
    )

    split = _split_compact_pair_buckets_by_projection_gather_budget(
        [bucket],
        per_image_inputs_by_class,
        n_score_pixels=17,
        n_recon_pixels=19,
        projection_complex_dtype=np.complex64,
        max_gather_bytes=2 * bucket_size * row_bytes + 1,
        rotation_block_size_for_quantization=5000,
    )

    assert len(split) == 5
    assert [len(chunk["image_indices"]) for chunk in split] == [2, 2, 2, 2, 2]
    np.testing.assert_array_equal(
        np.concatenate([chunk["image_indices"] for chunk in split]),
        np.arange(n_images, dtype=np.int64),
    )
    assert all(int(chunk["pair_bucket_size"]) == bucket_size for chunk in split)


def test_relion_windowed_projection_budget_accounts_for_centered_full_half_transient(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", raising=False)

    large_gpu = 80 * 1024**3
    n_half = 128 * (128 // 2 + 1)
    default_pixels = _projection_budget_pixels_for_pass(
        n_half,
        use_window=True,
        use_relion_projector=False,
    )
    relion_pixels = _projection_budget_pixels_for_pass(
        n_half,
        use_window=True,
        use_relion_projector=True,
    )

    assert default_pixels == n_half
    assert relion_pixels == 8 * n_half

    default_cap = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=large_gpu,
        n_projection_pixels=default_pixels,
        projection_complex_dtype=np.complex64,
        include_abs2=False,
    )
    relion_dtype = _projection_cache_budget_complex_dtype(
        np.complex64,
        np.complex64,
        use_relion_projector=True,
    )
    relion_cap = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=large_gpu,
        n_projection_pixels=relion_pixels,
        projection_complex_dtype=relion_dtype,
        include_abs2=False,
    )

    assert default_cap == 51622
    assert relion_cap == 3226

    default_cap64 = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=large_gpu,
        n_projection_pixels=default_pixels,
        projection_complex_dtype=np.complex128,
        include_abs2=False,
    )
    relion_cap64 = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=large_gpu,
        n_projection_pixels=relion_pixels,
        projection_complex_dtype=np.complex128,
        include_abs2=False,
    )

    assert default_cap64 == 25811
    assert relion_cap64 == 3226

    relion_chunk_bytes = _projection_cache_transient_bytes(
        relion_cap,
        n_half,
        projection_complex_dtype=relion_dtype,
        include_abs2=False,
    )
    assert relion_chunk_bytes <= 512 * 1024**2

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", "0")
    assert (
        _max_projected_rotations_per_call_for_pass(
            device_memory_bytes=large_gpu,
            n_projection_pixels=relion_pixels,
            projection_complex_dtype=np.complex64,
            include_abs2=False,
        )
        is None
    )


def test_compact_pair_prepare_tile_budget_splits_large_bucket():
    n_images = 10
    bucket_size = 8192
    per_image_inputs_by_class = [
        {"oversampled_rots": [np.zeros((bucket_size, 3, 3), dtype=np.float32) for _ in range(n_images)]}
        for _ in range(4)
    ]
    bucket = {"pair_bucket_size": 256, "image_indices": np.arange(n_images, dtype=np.int64)}

    split = _split_compact_pair_buckets_by_projection_gather_budget(
        [bucket],
        per_image_inputs_by_class,
        n_score_pixels=17,
        n_recon_pixels=19,
        projection_complex_dtype=np.complex64,
        max_gather_bytes=10**18,
        max_prepare_images_per_microbatch=3,
        rotation_block_size_for_quantization=5000,
    )

    assert [len(chunk["image_indices"]) for chunk in split] == [3, 3, 3, 1]
    np.testing.assert_array_equal(
        np.concatenate([chunk["image_indices"] for chunk in split]),
        np.arange(n_images, dtype=np.int64),
    )
    assert all(int(chunk["pair_bucket_size"]) == 256 for chunk in split)


def test_hybrid_compact_pair_execution_buckets_partition_images_once():
    dense_buckets = [
        {"bucket_size": 128, "image_indices": np.asarray([0, 1, 2, 3], dtype=np.int64)},
        {"bucket_size": 4096, "image_indices": np.asarray([4, 5, 6, 7], dtype=np.int64)},
    ]
    compact_pair_buckets = [
        {"pair_bucket_size": 512, "image_indices": np.asarray([0, 2], dtype=np.int64)},
        {"pair_bucket_size": 4096, "image_indices": np.asarray([4, 6], dtype=np.int64)},
        {"pair_bucket_size": 8192, "image_indices": np.asarray([5, 7], dtype=np.int64)},
    ]

    execution_buckets = _hybrid_k_class_compact_pair_execution_buckets(
        dense_buckets,
        compact_pair_buckets,
        min_pair_bucket_size=4096,
    )

    rectangular = [bucket for bucket in execution_buckets if bucket["_execution_mode"] == "rectangular"]
    compact = [bucket for bucket in execution_buckets if bucket["_execution_mode"] == "compact_pair"]
    assert [int(bucket["bucket_size"]) for bucket in rectangular] == [128]
    assert [int(bucket["pair_bucket_size"]) for bucket in compact] == [4096, 8192]
    np.testing.assert_array_equal(rectangular[0]["image_indices"], np.asarray([0, 1, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(
        np.sort(np.concatenate([bucket["image_indices"] for bucket in compact])),
        np.asarray([4, 5, 6, 7], dtype=np.int64),
    )
    all_images = np.concatenate([bucket["image_indices"] for bucket in execution_buckets])
    np.testing.assert_array_equal(np.sort(all_images), np.arange(8, dtype=np.int64))
    assert np.unique(all_images).size == all_images.size
    _validate_k_class_execution_bucket_partition(execution_buckets, n_images=8)


def test_compact_pair_execution_threshold_filters_before_split():
    compact_pair_buckets = [
        {"pair_bucket_size": 512, "image_indices": np.asarray([0, 2], dtype=np.int64)},
        {"pair_bucket_size": 4096, "image_indices": np.asarray([4, 6], dtype=np.int64)},
        {"pair_bucket_size": 8192, "image_indices": np.asarray([5, 7], dtype=np.int64)},
    ]

    selected = _compact_pair_buckets_for_execution_threshold(
        compact_pair_buckets,
        min_pair_bucket_size=4096,
    )

    assert [int(bucket["pair_bucket_size"]) for bucket in selected] == [4096, 8192]
    np.testing.assert_array_equal(
        np.concatenate([bucket["image_indices"] for bucket in selected]),
        np.asarray([4, 6, 5, 7], dtype=np.int64),
    )
    assert _compact_pair_buckets_for_execution_threshold(compact_pair_buckets, None) == compact_pair_buckets


def test_compact_pair_materialization_prefilter_skips_below_threshold_images():
    masks = []
    for count in (2, 1024, 2048):
        mask = np.zeros((2048, 1), dtype=bool)
        mask.reshape(-1)[:count] = True
        masks.append(mask)
    per_image_inputs = {
        "candidate_mask": masks,
        "oversampled_rot_indices": [np.arange(2048, dtype=np.int64) for _ in masks],
        "log_prior": [np.arange(2048, dtype=np.float32) for _ in masks],
    }
    per_image_inputs_by_class = [per_image_inputs, per_image_inputs]

    pair_counts_by_class = _compact_pair_counts_from_candidate_masks(per_image_inputs_by_class)
    image_mask = _compact_pair_image_mask_for_threshold(pair_counts_by_class, 1024)
    compact_inputs = _prepare_per_image_compact_candidate_pairs(per_image_inputs, image_mask=image_mask)

    np.testing.assert_array_equal(pair_counts_by_class[0], np.asarray([2, 1024, 2048], dtype=np.int64))
    np.testing.assert_array_equal(image_mask, np.asarray([False, True, True]))
    np.testing.assert_array_equal(compact_inputs["pair_counts"], np.asarray([0, 1024, 2048], dtype=np.int32))
    assert compact_inputs["local_rotation_row"][0].size == 0
    assert compact_inputs["translation_idx"][0].size == 0
    assert compact_inputs["local_rotation_row"][1].size == 1024
    assert compact_inputs["translation_idx"][2].size == 2048


def test_compact_pair_execution_filter_routes_full_support_rectangular():
    n_rows = 1024
    n_fine_trans = 4
    full = SparseCandidateMask(
        mode="full",
        n_rows=n_rows,
        n_fine_trans=n_fine_trans,
        count=n_rows * n_fine_trans,
    )
    sparse = SparseCandidateMask(
        mode="coarse",
        n_rows=n_rows,
        n_fine_trans=n_fine_trans,
        count=(n_rows * n_fine_trans) // 2,
    )
    per_image_inputs_by_class = [
        {"candidate_mask": [full, sparse, sparse]},
        {"candidate_mask": [sparse, sparse, full]},
    ]
    pair_counts_by_class = _compact_pair_counts_from_candidate_masks(per_image_inputs_by_class)
    threshold_mask = _compact_pair_image_mask_for_threshold(pair_counts_by_class, 1)

    image_mask, excluded = _compact_pair_execution_mask_excluding_full_support(
        per_image_inputs_by_class,
        threshold_mask,
    )

    assert excluded == 2
    np.testing.assert_array_equal(image_mask, np.asarray([False, True, False]))

    dense_buckets = [
        {"bucket_size": n_rows, "image_indices": np.asarray([0, 1, 2], dtype=np.int64)},
    ]
    stats = _compact_k_class_pair_plan_stats_from_counts(
        pair_counts_by_class,
        dense_buckets,
        n_fine_trans=n_fine_trans,
        max_pair_candidates_per_microbatch=10**9,
        max_images_per_microbatch=10,
        image_mask=image_mask,
    )
    execution_buckets = _hybrid_k_class_compact_pair_execution_buckets(
        dense_buckets,
        stats.buckets,
        min_pair_bucket_size=1,
    )

    rectangular = [bucket for bucket in execution_buckets if bucket["_execution_mode"] == "rectangular"]
    compact = [bucket for bucket in execution_buckets if bucket["_execution_mode"] == "compact_pair"]
    assert len(rectangular) == 1
    assert len(compact) == 1
    np.testing.assert_array_equal(rectangular[0]["image_indices"], np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(compact[0]["image_indices"], np.asarray([1], dtype=np.int64))
    _validate_k_class_execution_bucket_partition(execution_buckets, n_images=3)


def test_compact_pair_bucket_arrays_can_be_materialized_per_bucket():
    candidate_mask = np.asarray(
        [
            [
                [True, False, True],
                [False, True, False],
            ],
            [
                [False, True, False],
                [True, True, False],
            ],
            [
                [True, False, False],
                [False, False, True],
            ],
        ],
        dtype=bool,
    )
    per_image_inputs = {
        "candidate_mask": [candidate_mask[0], candidate_mask[1], candidate_mask[2]],
        "oversampled_rot_indices": [
            np.asarray([10, 11], dtype=np.int64),
            np.asarray([20, 21], dtype=np.int64),
            np.asarray([30, 31], dtype=np.int64),
        ],
        "log_prior": [
            np.asarray([0.1, 0.2], dtype=np.float32),
            np.asarray([0.3, 0.4], dtype=np.float32),
            np.asarray([0.5, 0.6], dtype=np.float32),
        ],
    }
    compact_inputs = _prepare_per_image_compact_candidate_pairs(per_image_inputs)
    bucket = {
        "pair_bucket_size": 5,
        "image_indices": np.asarray([2, 0], dtype=np.int64),
    }

    precomputed = _build_compact_pair_bucket_arrays(bucket, compact_inputs)
    on_demand = _build_compact_pair_bucket_arrays_from_per_image_inputs(bucket, per_image_inputs)

    assert precomputed.keys() == on_demand.keys()
    for key in precomputed:
        np.testing.assert_array_equal(on_demand[key], precomputed[key])


def test_compact_pair_tail_coalescing_respects_execution_image_mask():
    counts = (4096, 8192, 12288)
    max_count = max(counts)
    masks = []
    for count in counts:
        mask = np.zeros((max_count, 1), dtype=bool)
        mask.reshape(-1)[:count] = True
        masks.append(mask)
    per_image_inputs = {
        "candidate_mask": masks,
        "oversampled_rot_indices": [np.arange(max_count, dtype=np.int64) for _ in masks],
        "log_prior": [np.arange(max_count, dtype=np.float32) for _ in masks],
    }
    per_image_inputs_by_class = [per_image_inputs, per_image_inputs]
    pair_counts_by_class = _compact_pair_counts_from_candidate_masks(per_image_inputs_by_class)
    image_mask = _compact_pair_image_mask_for_threshold(pair_counts_by_class, 8192)
    dense_buckets = [
        {"bucket_size": 4096, "image_indices": np.asarray([0], dtype=np.int64)},
        {"bucket_size": 8192, "image_indices": np.asarray([1], dtype=np.int64)},
        {"bucket_size": 12288, "image_indices": np.asarray([2], dtype=np.int64)},
    ]

    stats = _compact_k_class_pair_plan_stats(
        per_image_inputs_by_class,
        dense_buckets,
        n_fine_trans=1,
        pair_block_size_for_quantization=5000,
        max_pair_candidates_per_microbatch=10**9,
        max_images_per_microbatch=10,
        tail_bucket_coalesce_max_images=8,
        tail_bucket_coalesce_max_inflation=2.0,
        tail_bucket_coalesce_min_bucket_size=4096,
        image_mask=image_mask,
    )

    np.testing.assert_array_equal(image_mask, np.asarray([False, True, True]))
    assert len(stats.buckets) == 1
    assert int(stats.buckets[0]["pair_bucket_size"]) == 12288
    np.testing.assert_array_equal(stats.buckets[0]["image_indices"], np.asarray([1, 2], dtype=np.int64))
    assert stats.valid_pair_candidates == 2 * (8192 + 12288)
    assert stats.rectangular_candidates == 2 * (8192 + 12288)


def test_k_class_execution_bucket_partition_validation_rejects_bad_routes():
    valid = [
        {
            "bucket_size": 128,
            "image_indices": np.asarray([0, 2], dtype=np.int64),
            "_execution_mode": "rectangular",
            "_execution_size_key": "bucket_size",
            "_execution_bucket_size": 128,
        },
        {
            "pair_bucket_size": 4096,
            "image_indices": np.asarray([1, 3], dtype=np.int64),
            "_execution_mode": "compact_pair",
            "_execution_size_key": "pair_bucket_size",
            "_execution_bucket_size": 4096,
        },
    ]
    _validate_k_class_execution_bucket_partition(valid, n_images=4)

    duplicate = [dict(valid[0]), dict(valid[1])]
    duplicate[1]["image_indices"] = np.asarray([1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="partition images exactly once"):
        _validate_k_class_execution_bucket_partition(duplicate, n_images=4)

    missing = [dict(valid[0]), dict(valid[1])]
    missing[1]["image_indices"] = np.asarray([1], dtype=np.int64)
    with pytest.raises(ValueError, match="coverage count mismatch"):
        _validate_k_class_execution_bucket_partition(missing, n_images=4)

    out_of_range = [dict(valid[0]), dict(valid[1])]
    out_of_range[1]["image_indices"] = np.asarray([1, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="out of range"):
        _validate_k_class_execution_bucket_partition(out_of_range, n_images=4)


def test_compact_pair_hybrid_threshold_reports_candidate_slots():
    dense_buckets = [
        {"bucket_size": 128, "image_indices": np.asarray([0, 1, 2, 3], dtype=np.int64)},
        {"bucket_size": 4096, "image_indices": np.asarray([4, 5, 6, 7], dtype=np.int64)},
    ]
    compact_pair_buckets = [
        {"pair_bucket_size": 512, "image_indices": np.asarray([0, 2], dtype=np.int64)},
        {"pair_bucket_size": 4096, "image_indices": np.asarray([4, 6], dtype=np.int64)},
        {"pair_bucket_size": 8192, "image_indices": np.asarray([5, 7], dtype=np.int64)},
    ]

    reports = _compact_pair_hybrid_threshold_reports(
        dense_buckets,
        compact_pair_buckets,
        thresholds=(4096, 8192),
        n_classes=4,
        n_fine_trans=3,
    )

    assert [report["threshold"] for report in reports] == [4096, 8192]
    assert reports[0]["compact_buckets"] == 2
    assert reports[0]["compact_images"] == 4
    assert reports[0]["rectangular_buckets"] == 1
    assert reports[0]["rectangular_images"] == 4
    assert reports[0]["rectangular_candidate_slots"] == 4 * 4 * 128 * 3
    assert reports[0]["compact_candidate_slots"] == 4 * 2 * 4096 + 4 * 2 * 8192
    assert reports[0]["total_candidate_slots"] == (
        reports[0]["rectangular_candidate_slots"] + reports[0]["compact_candidate_slots"]
    )

    baseline_candidate_slots = 4 * 4 * 128 * 3 + 4 * 4 * 4096 * 3
    assert reports[0]["slot_reduction"] == pytest.approx(
        baseline_candidate_slots / reports[0]["total_candidate_slots"],
    )
    assert reports[1]["compact_buckets"] == 1
    assert reports[1]["compact_images"] == 2
    assert reports[1]["rectangular_buckets"] == 2
    assert reports[1]["rectangular_images"] == 6


def test_sparse_pass2_projection_rotation_chunk_size_bounds_high_tail():
    max_gather_bytes = 512 * 1024**2
    chunk_size = _projection_rotation_chunk_size(
        batch_size=3,
        n_score_pixels=1867,
        n_recon_pixels=1813,
        projection_complex_dtype=np.complex64,
        include_recon_noise=True,
        max_gather_bytes=max_gather_bytes,
        max_projected_rotations=25811,
    )
    row_bytes = _projection_gather_bytes_per_rotation_row(
        n_score_pixels=1867,
        n_recon_pixels=1813,
        projection_complex_dtype=np.complex64,
        include_recon_noise=True,
    )

    assert chunk_size is not None
    assert chunk_size < 184320
    assert 3 * chunk_size * row_bytes <= max_gather_bytes


def test_sparse_pass2_winner_take_all_global_argmax_respects_chunk_boundary():
    scores = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    probs = _winner_take_all_bucket_probs_from_global_argmax(
        scores,
        jnp.asarray([5 * 2 + 1, 3 * 2 + 1], dtype=jnp.int32),
        jnp.asarray(4, dtype=jnp.int32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
    )

    expected = np.zeros((2, 3, 2), dtype=np.float32)
    expected[0, 1, 1] = 1.0
    np.testing.assert_array_equal(np.asarray(probs), expected)


def test_sparse_pass2_noise_block_chunking_matches_unchunked():
    rng = np.random.default_rng(123)
    n_rows = 9
    n_pixels = 13
    n_shells = 5
    proj_half = jnp.asarray(
        rng.standard_normal((n_rows, n_pixels)) + 1j * rng.standard_normal((n_rows, n_pixels)),
        dtype=jnp.complex64,
    )
    proj_abs2_half = jnp.abs(proj_half) ** 2
    summed_masked = jnp.asarray(
        rng.standard_normal((n_rows, n_pixels)) + 1j * rng.standard_normal((n_rows, n_pixels)),
        dtype=jnp.complex64,
    )
    ctf_probs = jnp.asarray(rng.random((n_rows, n_pixels)), dtype=jnp.float32)
    noise_variance_half = jnp.asarray(rng.random(n_pixels) + 0.1, dtype=jnp.float32)
    shell_indices = jnp.asarray(np.arange(n_pixels) % n_shells, dtype=jnp.int32)

    expected = compute_noise_block(
        proj_half,
        proj_abs2_half,
        summed_masked,
        ctf_probs,
        noise_variance_half,
        shell_indices,
        n_shells,
    )
    chunked = _compute_noise_block_chunked(
        proj_half,
        proj_abs2_half,
        summed_masked,
        ctf_probs,
        noise_variance_half,
        shell_indices,
        n_shells,
        max_block_bytes=1,
    )

    for actual_part, expected_part in zip(chunked, expected, strict=True):
        np.testing.assert_allclose(np.asarray(actual_part), np.asarray(expected_part), rtol=2e-6, atol=2e-6)


def test_sparse_pass2_fused_noise_norm_chunking_matches_unchunked():
    rng = np.random.default_rng(124)
    n_rows = 11
    n_pixels = 13
    n_shells = 5
    batch_size = 4
    proj_half = jnp.asarray(
        rng.standard_normal((n_rows, n_pixels)) + 1j * rng.standard_normal((n_rows, n_pixels)),
        dtype=jnp.complex64,
    )
    proj_abs2_half = jnp.abs(proj_half) ** 2
    summed_masked = jnp.asarray(
        rng.standard_normal((n_rows, n_pixels)) + 1j * rng.standard_normal((n_rows, n_pixels)),
        dtype=jnp.complex64,
    )
    ctf_probs_np = rng.random((n_rows, n_pixels), dtype=np.float32)
    ctf_probs_np[1::3, ::4] = 0.0
    ctf_probs = jnp.asarray(ctf_probs_np, dtype=jnp.float32)
    noise_variance_half = jnp.asarray(rng.random(n_pixels, dtype=np.float32) + 0.1, dtype=jnp.float32)
    shell_indices = jnp.asarray(np.arange(n_pixels) % n_shells, dtype=jnp.int32)
    flat_image_indices_np = np.asarray([0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2], dtype=np.int32)
    flat_image_indices = jnp.asarray(flat_image_indices_np, dtype=jnp.int32)

    expected_noise, _, _ = compute_noise_block(
        proj_half,
        proj_abs2_half,
        summed_masked,
        ctf_probs,
        noise_variance_half,
        shell_indices,
        n_shells,
    )
    unchunked_noise, unchunked_norm = _compute_noise_block_and_norm_residual_chunked(
        proj_half,
        proj_abs2_half,
        summed_masked,
        ctf_probs,
        noise_variance_half,
        shell_indices,
        flat_image_indices,
        shell_count=n_shells,
        batch_size=batch_size,
        max_block_bytes=None,
    )
    chunked_noise, chunked_norm = _compute_noise_block_and_norm_residual_chunked(
        proj_half,
        proj_abs2_half,
        summed_masked,
        ctf_probs,
        noise_variance_half,
        shell_indices,
        flat_image_indices,
        shell_count=n_shells,
        batch_size=batch_size,
        max_block_bytes=1,
    )

    proj_abs2_np = np.asarray(proj_abs2_half)
    summed_masked_np = np.asarray(summed_masked)
    proj_half_np = np.asarray(proj_half)
    noise_variance_np = np.asarray(noise_variance_half)
    ctf_has_mass = ctf_probs_np != 0.0
    ctf_probs_raw = np.where(ctf_has_mass, ctf_probs_np * noise_variance_np[None, :], 0.0)
    a2_terms = np.where(ctf_has_mass, proj_abs2_np * ctf_probs_raw, 0.0)
    cross_terms = np.where(summed_masked_np != 0.0, proj_half_np * np.conj(summed_masked_np), 0.0)
    residual_per_row = np.sum(a2_terms, axis=1) - 2.0 * np.sum(
        noise_variance_np[None, :] * cross_terms.real,
        axis=1,
    )
    expected_norm = np.zeros(batch_size, dtype=np.float32)
    np.add.at(expected_norm, flat_image_indices_np, residual_per_row.astype(np.float32))

    np.testing.assert_allclose(np.asarray(unchunked_noise), np.asarray(expected_noise), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(chunked_noise), np.asarray(expected_noise), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(unchunked_norm), expected_norm, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(chunked_norm), expected_norm, rtol=2e-6, atol=2e-6)


def test_sparse_pass2_compact_active_noise_gather_chunking_matches_full_gather():
    rng = np.random.default_rng(126)
    batch = 4
    n_rot = 5
    n_rows = batch * n_rot
    n_pixels = 11
    n_shells = 5
    shape = (batch, n_rot, n_pixels)
    proj = jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        dtype=jnp.complex64,
    )
    proj_abs2 = jnp.abs(proj) ** 2
    summed = jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        dtype=jnp.complex64,
    )
    ctf_probs_np = rng.random(shape, dtype=np.float32)
    ctf_probs_np[1, 2, ::3] = 0.0
    ctf_probs = jnp.asarray(ctf_probs_np, dtype=jnp.float32)
    active_indices = np.asarray([0, 2, 7, 11, 16, 19, 2, 2], dtype=np.int32)
    assert np.all(active_indices < n_rows)
    active_mask = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    noise_variance_half = jnp.asarray(rng.random(n_pixels, dtype=np.float32) + 0.1, dtype=jnp.float32)
    shell_indices = jnp.asarray(np.arange(n_pixels) % n_shells, dtype=jnp.int32)

    full_gather = _select_active_noise_rows(
        proj,
        proj_abs2,
        summed,
        ctf_probs,
        active_indices,
        active_mask,
        n_rotation_rows=n_rot,
    )
    expected = _compute_noise_block_and_norm_residual_chunked(
        full_gather[0],
        full_gather[1],
        full_gather[2],
        full_gather[3],
        noise_variance_half,
        shell_indices,
        full_gather[4],
        shell_count=n_shells,
        batch_size=batch,
        max_block_bytes=None,
    )
    chunked = _compute_active_noise_rows_chunked(
        proj,
        proj_abs2,
        summed,
        ctf_probs,
        active_indices,
        active_mask,
        noise_variance_half,
        shell_indices,
        n_rotation_rows=n_rot,
        shell_count=n_shells,
        batch_size=batch,
        max_block_bytes=1,
    )

    np.testing.assert_allclose(np.asarray(chunked[0]), np.asarray(expected[0]), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(chunked[1]), np.asarray(expected[1]), rtol=2e-6, atol=2e-6)


def test_sparse_pass2_residual_terms_fused_matches_legacy_nonfinite_masks(monkeypatch):
    rng = np.random.default_rng(125)
    n_rows = 6
    n_pixels = 7
    n_shells = 4
    batch_size = 3
    proj_half_np = (
        rng.standard_normal((n_rows, n_pixels)) + 1j * rng.standard_normal((n_rows, n_pixels))
    ).astype(np.complex64)
    proj_abs2_np = np.abs(proj_half_np).astype(np.float32) ** 2
    summed_masked_np = (
        rng.standard_normal((n_rows, n_pixels)) + 1j * rng.standard_normal((n_rows, n_pixels))
    ).astype(np.complex64)
    ctf_probs_np = rng.random((n_rows, n_pixels), dtype=np.float32)

    ctf_probs_np[0, 0] = 0.0
    proj_abs2_np[0, 0] = np.nan
    summed_masked_np[1, 1] = 0.0
    proj_half_np[1, 1] = np.nan + 1j * np.nan

    proj_half = jnp.asarray(proj_half_np)
    proj_abs2_half = jnp.asarray(proj_abs2_np)
    summed_masked = jnp.asarray(summed_masked_np)
    ctf_probs = jnp.asarray(ctf_probs_np)
    noise_variance_half = jnp.asarray(rng.random(n_pixels, dtype=np.float32) + 0.1)
    shell_indices = jnp.asarray(np.arange(n_pixels) % n_shells, dtype=jnp.int32)
    flat_image_indices = jnp.asarray([0, 1, 2, 0, 1, 2], dtype=jnp.int32)

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED", "0")
    legacy_noise, legacy_norm = _compute_noise_block_and_norm_residual_chunked(
        proj_half,
        proj_abs2_half,
        summed_masked,
        ctf_probs,
        noise_variance_half,
        shell_indices,
        flat_image_indices,
        shell_count=n_shells,
        batch_size=batch_size,
        max_block_bytes=None,
    )
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED", "1")
    fused_noise, fused_norm = _compute_noise_block_and_norm_residual_chunked(
        proj_half,
        proj_abs2_half,
        summed_masked,
        ctf_probs,
        noise_variance_half,
        shell_indices,
        flat_image_indices,
        shell_count=n_shells,
        batch_size=batch_size,
        max_block_bytes=None,
    )

    np.testing.assert_allclose(np.asarray(fused_noise), np.asarray(legacy_noise), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(fused_norm), np.asarray(legacy_norm), rtol=2e-6, atol=2e-6)
    assert np.all(np.isfinite(np.asarray(fused_noise)))
    assert np.all(np.isfinite(np.asarray(fused_norm)))


def test_sparse_pass2_adjoint_block_chunking_accumulates_all_rows(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    flat_block = jnp.arange(30, dtype=jnp.float32).reshape(10, 3)
    rotations = jnp.zeros((10, 3, 3), dtype=jnp.float32)
    volume = jnp.zeros(2, dtype=jnp.float32)
    calls = []

    def fake_adjoint_slice_volume_half(
        half_block,
        rotations_block,
        volume_in,
        image_shape,
        volume_shape,
        disc_type,
        half_image,
        half_volume=False,
    ):
        del rotations_block, image_shape, volume_shape, disc_type, half_image, half_volume
        calls.append(int(half_block.shape[0]))
        return volume_in + jnp.sum(half_block)

    monkeypatch.setattr(bucketed_mod, "_adjoint_slice_volume_half", fake_adjoint_slice_volume_half)

    row_bytes = flat_block.shape[1] * np.dtype(np.float32).itemsize
    assert _adjoint_block_chunk_rows(flat_block, max_block_bytes=4 * row_bytes + 1) == 4
    actual = _accumulate_adjoint_block_chunked(
        flat_block,
        rotations,
        volume,
        use_windowed_adjoint=False,
        image_shape=(8, 8),
        volume_shape=(8, 8, 8),
        disc_type="linear_interp",
        half_image=True,
        half_volume=False,
        max_r=None,
        relion_x_half=False,
        max_block_bytes=4 * row_bytes + 1,
        log_label="test",
    )

    assert calls == [4, 4, 2]
    np.testing.assert_allclose(np.asarray(actual), np.asarray(volume + jnp.sum(flat_block)))


def test_relion_x_half_bp_per_particle_launch_is_off_by_default(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", raising=False)
    assert bucketed_mod.relion_x_half_bp_per_particle_launch_enabled() is False
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", "1")
    assert bucketed_mod.relion_x_half_bp_per_particle_launch_enabled() is True


def test_scoped_bpref_ownership_gate_ignores_unrelated_bucket_order():
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    image_indices = np.asarray([20, 5, 13], dtype=np.int64)
    target_rows = np.asarray([0, 2], dtype=np.int64)

    scoped = bucketed_mod._bpref_diagnostic_ownership_indices(
        image_indices,
        target_rows,
        device_signature_requested=True,
    )
    unscoped = bucketed_mod._bpref_diagnostic_ownership_indices(
        image_indices,
        target_rows,
        device_signature_requested=False,
    )

    np.testing.assert_array_equal(scoped, np.asarray([20, 13], dtype=np.int64))
    np.testing.assert_array_equal(unscoped, image_indices)
    assert np.unique(scoped).size == scoped.size
    assert not np.all(np.diff(scoped) > 0)
    assert not np.all(np.diff(unscoped) > 0)
    bucketed_mod._validate_bpref_diagnostic_ownership(
        scoped,
        device_signature_requested=True,
    )
    with pytest.raises(RuntimeError, match="unique particle ownership"):
        bucketed_mod._validate_bpref_diagnostic_ownership(
            np.asarray([20, 20], dtype=np.int64),
            device_signature_requested=True,
        )
    with pytest.raises(RuntimeError, match="strictly increasing particle ownership order"):
        bucketed_mod._validate_bpref_diagnostic_ownership(
            unscoped,
            device_signature_requested=False,
        )


def test_relion_x_half_bp_fused_atomics_is_off_by_default(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS", raising=False)
    assert bucketed_mod.relion_x_half_bp_fused_atomics_enabled() is False
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS", "1")
    assert bucketed_mod.relion_x_half_bp_fused_atomics_enabled() is True


def test_relion_x_half_bp_per_particle_launch_preserves_ownership_and_order(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    values = jnp.arange(2 * 3 * 2, dtype=jnp.float32).reshape(2, 3, 2).astype(jnp.complex64)
    ctf_values = (100.0 + jnp.arange(2 * 3 * 2, dtype=jnp.float32)).reshape(2, 3, 2)
    rotations = jnp.arange(2 * 3 * 9, dtype=jnp.float32).reshape(2, 3, 3, 3)
    actual_counts = np.asarray([2, 1], dtype=np.int32)
    calls = []

    def fake_adjoint_slice_volume_windowed(
        half_block,
        window_indices,
        rotations_block,
        volume_in,
        image_shape,
        volume_shape,
        disc_type,
        half_image,
        half_volume=False,
        max_r=None,
        relion_x_half=False,
    ):
        del window_indices, image_shape, volume_shape, disc_type, half_volume, max_r
        assert half_image is True and relion_x_half is True
        calls.append((np.asarray(half_block).copy(), np.asarray(rotations_block).copy()))
        return volume_in + jnp.sum(jnp.real(half_block))

    monkeypatch.setattr(bucketed_mod, "_adjoint_slice_volume_windowed", fake_adjoint_slice_volume_windowed)
    y_volume, ctf_volume = bucketed_mod._accumulate_relion_x_half_per_particle_launches(
        values,
        ctf_values,
        rotations,
        actual_counts,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(10.0, dtype=jnp.float32),
        window_indices=jnp.arange(2, dtype=jnp.int32),
        image_shape=(8, 8),
        volume_shape=(8, 8, 8),
        disc_type="linear_interp",
        half_volume=True,
        max_r=2.0,
        log_label_prefix="test",
    )

    assert [call[0].shape[0] for call in calls] == [2, 2, 1, 1]
    np.testing.assert_array_equal(calls[0][0], np.asarray(values[0, :2]))
    np.testing.assert_array_equal(calls[0][1], np.asarray(rotations[0, :2]))
    np.testing.assert_array_equal(calls[1][0], np.asarray(ctf_values[0, :2]))
    np.testing.assert_array_equal(calls[2][0], np.asarray(values[1, :1]))
    np.testing.assert_array_equal(calls[3][0], np.asarray(ctf_values[1, :1]))
    expected_y = np.asarray(values[0, :2].real).sum() + np.asarray(values[1, :1].real).sum()
    expected_ctf = 10.0 + np.asarray(ctf_values[0, :2]).sum() + np.asarray(ctf_values[1, :1]).sum()
    np.testing.assert_allclose(np.asarray(y_volume), expected_y)
    np.testing.assert_allclose(np.asarray(ctf_volume), expected_ctf)


def test_relion_x_half_bp_fused_atomics_threads_both_accumulators_per_particle(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    values = jnp.arange(2 * 3 * 2, dtype=jnp.float32).reshape(2, 3, 2).astype(jnp.complex64)
    ctf_values = (100.0 + jnp.arange(2 * 3 * 2, dtype=jnp.float32)).reshape(2, 3, 2)
    rotations = jnp.arange(2 * 3 * 9, dtype=jnp.float32).reshape(2, 3, 3, 3)
    actual_counts = np.asarray([2, 1], dtype=np.int32)
    calls = []

    def fake_fused(
        y_volume,
        ctf_volume,
        particle_values,
        particle_ctf_values,
        window_indices,
        particle_rotations,
        **kwargs,
    ):
        calls.append(
            (
                np.asarray(particle_values).copy(),
                np.asarray(particle_ctf_values).copy(),
                np.asarray(particle_rotations).copy(),
                np.asarray(window_indices).copy(),
                kwargs,
            )
        )
        return (
            y_volume + jnp.sum(jnp.real(particle_values)),
            ctf_volume + jnp.sum(particle_ctf_values),
        )

    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", "1")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS", "1")
    monkeypatch.setattr(cuda_backproject, "relion_fused_x_half_backproject_indexed", fake_fused)

    y_volume, ctf_volume = bucketed_mod._accumulate_relion_x_half_per_particle_launches(
        values,
        ctf_values,
        rotations,
        actual_counts,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(10.0, dtype=jnp.float32),
        window_indices=jnp.arange(2, dtype=jnp.int32),
        image_shape=(8, 8),
        volume_shape=(8, 8, 8),
        disc_type="linear_interp",
        half_volume=True,
        max_r=2.0,
        log_label_prefix="test-fused",
    )

    assert [call[0].shape[0] for call in calls] == [2, 1]
    np.testing.assert_array_equal(calls[0][0], np.asarray(values[0, :2]))
    np.testing.assert_array_equal(calls[0][1], np.asarray(ctf_values[0, :2]))
    np.testing.assert_array_equal(calls[0][2], np.asarray(rotations[0, :2]))
    np.testing.assert_array_equal(calls[1][0], np.asarray(values[1, :1]))
    expected_y = np.asarray(values[0, :2].real).sum() + np.asarray(values[1, :1].real).sum()
    expected_ctf = 10.0 + np.asarray(ctf_values[0, :2]).sum() + np.asarray(ctf_values[1, :1]).sum()
    np.testing.assert_allclose(np.asarray(y_volume), expected_y)
    np.testing.assert_allclose(np.asarray(ctf_volume), expected_ctf)


def test_fresh_k1_particle_pool_preserves_consecutive_particle_order(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    values = jnp.arange(2 * 3 * 2, dtype=jnp.float32).reshape(2, 3, 2).astype(jnp.complex64)
    ctf_values = (100.0 + jnp.arange(2 * 3 * 2, dtype=jnp.float32)).reshape(2, 3, 2)
    rotations = jnp.arange(2 * 3 * 9, dtype=jnp.float32).reshape(2, 3, 3, 3)
    actual_counts = np.asarray([2, 1], dtype=np.int32)
    calls = []

    def fake_fused(
        y_volume,
        ctf_volume,
        particle_values,
        particle_ctf_values,
        window_indices,
        particle_rotations,
        **kwargs,
    ):
        calls.append(
            (
                np.asarray(particle_values).copy(),
                np.asarray(particle_ctf_values).copy(),
                np.asarray(particle_rotations).copy(),
            )
        )
        return y_volume, ctf_volume

    monkeypatch.setenv("RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE", "2")
    monkeypatch.setattr(cuda_backproject, "relion_fused_x_half_backproject_indexed", fake_fused)

    bucketed_mod._accumulate_relion_x_half_per_particle_launches(
        values,
        ctf_values,
        rotations,
        actual_counts,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(10.0, dtype=jnp.float32),
        window_indices=jnp.arange(2, dtype=jnp.int32),
        image_shape=(8, 8),
        volume_shape=(8, 8, 8),
        disc_type="linear_interp",
        half_volume=True,
        max_r=2.0,
        log_label_prefix="test-pool",
        winner_take_all=True,
        strict_particle_order=True,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0][0],
        np.concatenate((np.asarray(values[0, :2]), np.asarray(values[1, :1])), axis=0),
    )
    np.testing.assert_array_equal(
        calls[0][1],
        np.concatenate((np.asarray(ctf_values[0, :2]), np.asarray(ctf_values[1, :1])), axis=0),
    )
    np.testing.assert_array_equal(
        calls[0][2],
        np.concatenate((np.asarray(rotations[0, :2]), np.asarray(rotations[1, :1])), axis=0),
    )


def test_particle_pool_rejects_non_fresh_k1_path(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE", "3")
    with pytest.raises(RuntimeError, match="fresh K=1 winner-take-all"):
        bucketed_mod._accumulate_relion_x_half_per_particle_launches(
            jnp.ones((1, 1, 1), dtype=jnp.complex64),
            jnp.ones((1, 1, 1), dtype=jnp.float32),
            jnp.eye(3, dtype=jnp.float32).reshape(1, 1, 3, 3),
            np.asarray([1], dtype=np.int32),
            jnp.zeros(1, dtype=jnp.complex64),
            jnp.zeros(1, dtype=jnp.float32),
            window_indices=jnp.asarray([0], dtype=jnp.int32),
            image_shape=(8, 8),
            volume_shape=(7, 7, 7),
            disc_type="linear_interp",
            half_volume=True,
            max_r=2.0,
            log_label_prefix="test-invalid-pool",
        )


def test_fresh_k1_firstiter_cc_uses_fused_atomics_without_diagnostic_env(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", raising=False)
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", raising=False)
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS", raising=False)
    calls = []

    def fake_fused(
        y_volume,
        ctf_volume,
        particle_values,
        particle_ctf_values,
        window_indices,
        particle_rotations,
        **kwargs,
    ):
        calls.append((np.asarray(particle_values), np.asarray(particle_ctf_values)))
        return y_volume, ctf_volume

    monkeypatch.setattr(cuda_backproject, "relion_fused_x_half_backproject_indexed", fake_fused)

    bucketed_mod._accumulate_relion_x_half_per_particle_launches(
        jnp.ones((1, 1, 1), dtype=jnp.complex64),
        jnp.ones((1, 1, 1), dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32).reshape(1, 1, 3, 3),
        np.asarray([1], dtype=np.int32),
        jnp.zeros(1, dtype=jnp.complex64),
        jnp.zeros(1, dtype=jnp.float32),
        window_indices=jnp.asarray([0], dtype=jnp.int32),
        image_shape=(8, 8),
        volume_shape=(7, 7, 7),
        disc_type="linear_interp",
        half_volume=True,
        max_r=2.0,
        log_label_prefix="fresh-k1-firstiter",
        winner_take_all=True,
        strict_particle_order=True,
    )

    assert len(calls) == 1


def test_relion_x_half_bp_fused_atomics_requires_block_topology(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", "1")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS", "1")
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", raising=False)
    assert cuda_backproject.relion_x_half_bp_block_topology_enabled() is False

    with pytest.raises(RuntimeError, match="BLOCK_TOPOLOGY=1"):
        bucketed_mod._accumulate_relion_x_half_per_particle_launches(
            jnp.ones((1, 1, 1), dtype=jnp.complex64),
            jnp.ones((1, 1, 1), dtype=jnp.float32),
            jnp.eye(3, dtype=jnp.float32).reshape(1, 1, 3, 3),
            np.asarray([1], dtype=np.int32),
            jnp.zeros(1, dtype=jnp.complex64),
            jnp.zeros(1, dtype=jnp.float32),
            window_indices=jnp.asarray([0], dtype=jnp.int32),
            image_shape=(8, 8),
            volume_shape=(7, 7, 7),
            disc_type="linear_interp",
            half_volume=True,
            max_r=2.0,
            log_label_prefix="test-prerequisite",
        )


@pytest.mark.gpu
def test_later_soft_posterior_scoped_fused_signature_fixture(
    monkeypatch,
    tmp_path,
    custom_cuda_lib,
    gpu_device,
):
    """Separate translation-order, scatter-topology, and signature effects."""

    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS", raising=False)
    monkeypatch.delenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", raising=False)

    image_shape = (8, 8)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    max_r = 2.0
    probs_np = np.asarray(
        [
            [[0.19, 0.07, 0.03], [0.11, 0.13, 0.05], [0.09, 0.17, 0.16]],
            [[0.04, 0.15, 0.09], [0.18, 0.06, 0.12], [0.10, 0.14, 0.12]],
        ],
        dtype=np.float32,
    )
    shifted_np = np.asarray(
        [
            [
                [1.25 + 0.75j, -0.50 + 1.50j, 0.20 - 0.40j],
                [-0.70 + 0.10j, 0.30 - 1.20j, 1.10 + 0.60j],
                [0.45 - 0.90j, 1.40 + 0.20j, -0.80 + 0.35j],
            ],
            [
                [-1.10 + 0.40j, 0.65 + 1.30j, 0.15 - 0.75j],
                [0.90 - 0.55j, -1.25 + 0.30j, 0.70 + 0.80j],
                [0.35 + 1.10j, 0.50 - 0.60j, -0.45 + 0.95j],
            ],
        ],
        dtype=np.complex64,
    )
    ctf2_over_nv_np = np.asarray(
        [[0.75, 1.25, 0.40], [1.10, 0.55, 1.35]],
        dtype=np.float32,
    )

    def _z_rotation(degrees):
        radians = np.deg2rad(np.asarray(degrees, dtype=np.float32))
        cosine = np.cos(radians).astype(np.float32)
        sine = np.sin(radians).astype(np.float32)
        result = np.zeros((len(degrees), 3, 3), dtype=np.float32)
        result[:, 0, 0] = cosine
        result[:, 0, 1] = -sine
        result[:, 1, 0] = sine
        result[:, 1, 1] = cosine
        result[:, 2, 2] = 1.0
        return result

    rotations_np = np.stack(
        (_z_rotation([0.0, 37.0, -53.0]), _z_rotation([19.0, -31.0, 71.0])),
        axis=0,
    )
    window_indices_np = np.asarray([1, 12, 36], dtype=np.int32)
    actual_counts = np.asarray([3, 3], dtype=np.int32)
    initial_data_np = (
        np.linspace(-1.0e-4, 1.0e-4, volume_size, dtype=np.float32)
        * np.complex64(1.0 + 0.25j)
    ).astype(np.complex64)
    initial_weight_np = np.linspace(1.0e-5, 2.0e-4, volume_size, dtype=np.float32)

    frozen_sources = {
        "probs": probs_np.copy(),
        "shifted": shifted_np.copy(),
        "ctf2_over_nv": ctf2_over_nv_np.copy(),
        "rotations": rotations_np.copy(),
        "window_indices": window_indices_np.copy(),
        "actual_counts": actual_counts.copy(),
        "initial_data": initial_data_np.copy(),
        "initial_weight": initial_weight_np.copy(),
    }

    def _accumulate_separate(data_rows, weight_rows, label):
        return bucketed_mod._accumulate_relion_x_half_per_particle_launches(
            data_rows,
            weight_rows,
            rotations,
            actual_counts,
            jnp.asarray(initial_data_np),
            jnp.asarray(initial_weight_np),
            window_indices=window_indices,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type="linear_interp",
            half_volume=True,
            max_r=max_r,
            log_label_prefix=label,
        )

    with cuda_backproject.jax.default_device(gpu_device):
        probs = jnp.asarray(probs_np)
        shifted = jnp.asarray(shifted_np)
        ctf2_over_nv = jnp.asarray(ctf2_over_nv_np)
        rotations = jnp.asarray(rotations_np)
        window_indices = jnp.asarray(window_indices_np)

        # A: production reduction, separate numerator/denominator adjoints.
        ordinary_rows = compute_local_mstep_sums(
            probs,
            shifted,
            ctf2_over_nv,
            relion_x_half=True,
            sequential_translation_reduction=False,
        )
        a_data, a_weight = _accumulate_separate(*ordinary_rows, "fixture-A-ordinary")

        # B: only translation reduction changes; scatter remains separate.
        sequential_rows = compute_local_mstep_sums(
            probs,
            shifted,
            ctf2_over_nv,
            relion_x_half=True,
            sequential_translation_reduction=True,
        )
        frozen_sequential_rows = tuple(np.asarray(value).copy() for value in sequential_rows)
        b_data, b_weight = _accumulate_separate(*sequential_rows, "fixture-B-sequential")

        # C: reuse the exact B rows and change only to fused data/weight atomics.
        monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", "1")
        monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS", "1")
        monkeypatch.setenv(
            "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
            str(tmp_path / "device-signatures"),
        )
        with cuda_backproject.bpref_device_signature_scope(True):
            c_data, c_weight = bucketed_mod._accumulate_relion_x_half_per_particle_launches(
                *sequential_rows,
                rotations,
                actual_counts,
                jnp.asarray(initial_data_np),
                jnp.asarray(initial_weight_np),
                window_indices=window_indices,
                image_shape=image_shape,
                volume_shape=volume_shape,
                disc_type="linear_interp",
                half_volume=True,
                max_r=max_r,
                log_label_prefix="fixture-C-fused",
            )

        # D: replay C's exact rows and particle launch boundaries through the
        # signature FFI.  Each call enforces bitwise accumulator-shadow and
        # prepared-operand inertness internally.
        d_data = jnp.asarray(initial_data_np)
        d_weight = jnp.asarray(initial_weight_np)
        with cuda_backproject.bpref_device_signature_scope(True):
            for particle_index, count in enumerate(actual_counts.tolist()):
                outputs = cuda_backproject.relion_fused_x_half_backproject_signature_indexed(
                    d_data,
                    d_weight,
                    data_rows=sequential_rows[0][particle_index, :count],
                    weight_rows=sequential_rows[1][particle_index, :count],
                    pixel_indices=window_indices,
                    rotation_matrices=rotations[particle_index, :count],
                    canonical_rotation_keys=jnp.asarray(
                        particle_index * 100 + np.arange(count),
                        dtype=jnp.int32,
                    ),
                    signature_row_indices=jnp.arange(count, dtype=jnp.int32),
                    image_shape=image_shape,
                    volume_shape=volume_shape,
                    max_r=max_r,
                )
                d_data, d_weight = outputs[:2]

    arrays = {
        "a_data": np.asarray(a_data),
        "a_weight": np.asarray(a_weight),
        "b_data": np.asarray(b_data),
        "b_weight": np.asarray(b_weight),
        "c_data": np.asarray(c_data),
        "c_weight": np.asarray(c_weight),
        "d_data": np.asarray(d_data),
        "d_weight": np.asarray(d_weight),
    }

    # Recompute the row operands from the frozen pre-reduction sources.  This
    # is deliberately not a cast of A/B: it retains the exact float32 input
    # values but performs every multiply and reduction in complex128/float64.
    probs_64 = probs_np.astype(np.float64)
    shifted_128 = shifted_np.astype(np.complex128)
    ctf2_over_nv_64 = ctf2_over_nv_np.astype(np.float64)
    canonical_rows = (
        np.einsum("brt,btn->brn", probs_64, shifted_128, optimize=False),
        np.sum(probs_64, axis=-1, dtype=np.float64)[..., None]
        * ctf2_over_nv_64[:, None, :],
    )
    ordinary_rows_np = tuple(np.asarray(value) for value in ordinary_rows)
    sequential_rows_np = tuple(np.asarray(value) for value in sequential_rows)

    def _row_metrics(observed, canonical):
        difference = np.abs(observed.astype(canonical.dtype) - canonical)
        canonical_l1 = max(
            float(np.sum(np.abs(canonical), dtype=np.float64)),
            np.finfo(np.float64).tiny,
        )
        row_denominator = np.maximum(
            np.sum(np.abs(canonical), axis=-1, dtype=np.float64),
            np.finfo(np.float64).tiny,
        )
        return {
            "rel_l1": float(np.sum(difference, dtype=np.float64) / canonical_l1),
            "max_abs": float(np.max(difference)),
            "max_row_rel_l1": float(
                np.max(np.sum(difference, axis=-1, dtype=np.float64) / row_denominator)
            ),
        }

    canonical_row_metrics = {
        "A_data_vs_complex128": _row_metrics(ordinary_rows_np[0], canonical_rows[0]),
        "A_weight_vs_float64": _row_metrics(ordinary_rows_np[1], canonical_rows[1]),
        "B_data_vs_complex128": _row_metrics(sequential_rows_np[0], canonical_rows[0]),
        "B_weight_vs_float64": _row_metrics(sequential_rows_np[1], canonical_rows[1]),
    }

    # Error-bound classification for the frozen operands.  A GPU dot may use
    # TF32 products with FP32 accumulation; B explicitly performs FP32
    # multiply/adds in translation order.  Bound real and imaginary channels
    # independently, then combine them into a complex absolute-error bound.
    n_translations = int(probs_np.shape[-1])
    u_float32 = 2.0**-24
    u_tf32 = 2.0**-11

    def _gamma(operation_count, unit_roundoff):
        product = operation_count * unit_roundoff
        return product / (1.0 - product)

    fp32_coefficient = _gamma(2 * n_translations + 2, u_float32)
    tf32_coefficient = (
        (1.0 + u_tf32) ** 2
        * (1.0 + _gamma(n_translations + 1, u_float32))
        - 1.0
    )
    abs_weighted_real = np.einsum(
        "brt,btn->brn",
        np.abs(probs_64),
        np.abs(shifted_128.real),
        optimize=False,
    )
    abs_weighted_imag = np.einsum(
        "brt,btn->brn",
        np.abs(probs_64),
        np.abs(shifted_128.imag),
        optimize=False,
    )
    fp32_data_bound = fp32_coefficient * np.hypot(abs_weighted_real, abs_weighted_imag)
    tf32_data_bound = tf32_coefficient * np.hypot(abs_weighted_real, abs_weighted_imag)
    abs_weight_terms = (
        np.sum(np.abs(probs_64), axis=-1)[..., None]
        * np.abs(ctf2_over_nv_64[:, None, :])
    )
    fp32_weight_bound = fp32_coefficient * abs_weight_terms
    bound_slack = 1.10

    def _within_bound(observed, canonical, bound):
        error = np.abs(observed.astype(canonical.dtype) - canonical)
        absolute_slack = 8.0 * np.finfo(np.float32).eps * np.finfo(np.float32).tiny
        return bool(np.all(error <= bound_slack * bound + absolute_slack))

    a_data_fp32_compatible = _within_bound(
        ordinary_rows_np[0], canonical_rows[0], fp32_data_bound
    )
    a_data_tf32_compatible = _within_bound(
        ordinary_rows_np[0], canonical_rows[0], tf32_data_bound
    )
    b_data_fp32_compatible = _within_bound(
        sequential_rows_np[0], canonical_rows[0], fp32_data_bound
    )
    a_weight_fp32_compatible = _within_bound(
        ordinary_rows_np[1], canonical_rows[1], fp32_weight_bound
    )
    b_weight_fp32_compatible = _within_bound(
        sequential_rows_np[1], canonical_rows[1], fp32_weight_bound
    )
    if a_data_fp32_compatible:
        reduction_classification = "A-and-B-fp32-order-compatible"
    elif a_data_tf32_compatible:
        reduction_classification = "A-tf32-compatible-B-fp32-order-compatible"
    else:
        reduction_classification = "unresolved-outside-tf32-error-bound"

    assert b_data_fp32_compatible, canonical_row_metrics
    assert a_weight_fp32_compatible, canonical_row_metrics
    assert b_weight_fp32_compatible, canonical_row_metrics
    assert a_data_tf32_compatible, canonical_row_metrics

    def _transition(left, right):
        denominator = max(
            float(np.sum(np.abs(left), dtype=np.float64)),
            np.finfo(np.float64).tiny,
        )
        fsc_auc, min_fsc, fsc = bpref_signature_validator._accumulator_fsc(
            left,
            right,
            volume_shape,
        )
        return {
            "rel_l1": float(np.sum(np.abs(right - left), dtype=np.float64) / denominator),
            "max_abs": float(np.max(np.abs(right - left))),
            "fsc_auc": fsc_auc,
            "min_finite_shell_fsc": min_fsc,
            "finite_non_dc_shell_count": int(np.count_nonzero(np.isfinite(fsc[1:]))),
        }

    metrics = {
        "A_to_B_data": _transition(arrays["a_data"], arrays["b_data"]),
        "A_to_B_weight": _transition(arrays["a_weight"], arrays["b_weight"]),
        "B_to_C_data": _transition(arrays["b_data"], arrays["c_data"]),
        "B_to_C_weight": _transition(arrays["b_weight"], arrays["c_weight"]),
        "C_to_D_data_cross_launch": _transition(arrays["c_data"], arrays["d_data"]),
        "C_to_D_weight_cross_launch": _transition(arrays["c_weight"], arrays["d_weight"]),
    }
    for transition, values in metrics.items():
        assert np.all(np.isfinite(list(values.values()))), (transition, values)
    # A/B is the explicitly classified translation-reduction transition.  Its
    # array error is reported rather than hidden behind a loosened tolerance;
    # map-space agreement remains an FSC quality sanity check.
    for transition in ("A_to_B_data", "A_to_B_weight"):
        values = metrics[transition]
        assert values["fsc_auc"] > 0.9999, (transition, values)
        assert values["min_finite_shell_fsc"] > 0.9999, (transition, values)
    # B/C reuses bitwise-identical rows, so this remains the strict topology-
    # only numerical gate.
    for transition in ("B_to_C_data", "B_to_C_weight"):
        values = metrics[transition]
        assert values["rel_l1"] < 1.0e-5, (transition, values)
        assert values["max_abs"] < 1.0e-5, (transition, values)
        assert values["fsc_auc"] > 0.99999, (transition, values)
        assert values["min_finite_shell_fsc"] > 0.99999, (transition, values)
    # C/D are separate atomic launches and therefore may differ at the normal
    # control-repeat scale.  Do not use a one-repeat envelope as a pass/fail
    # gate.  Signature inertness is instead proved inside each D call by the
    # stream-ordered bitwise accumulator shadow and exact operand snapshots.
    for transition in (
        "C_to_D_data_cross_launch",
        "C_to_D_weight_cross_launch",
    ):
        values = metrics[transition]
        assert values["fsc_auc"] > 0.9999, (transition, values)
        assert values["min_finite_shell_fsc"] > 0.9999, (transition, values)

    import json

    print(
        "BPREF_SOFT_POSTERIOR_FIXTURE_METRICS "
        + json.dumps(
            {
                "reduction_classification": reduction_classification,
                "signature_internal_bitwise_gate_passed": True,
                "canonical_row_metrics": canonical_row_metrics,
                "accumulator_transitions": metrics,
            },
            sort_keys=True,
        )
    )

    for observed, expected in zip(sequential_rows, frozen_sequential_rows, strict=True):
        np.testing.assert_array_equal(np.asarray(observed), expected)
    for name, expected in frozen_sources.items():
        observed = {
            "probs": probs,
            "shifted": shifted,
            "ctf2_over_nv": ctf2_over_nv,
            "rotations": rotations,
            "window_indices": window_indices,
            "actual_counts": actual_counts,
            "initial_data": initial_data_np,
            "initial_weight": initial_weight_np,
        }[name]
        np.testing.assert_array_equal(np.asarray(observed), expected)


def test_sparse_pass2_active_flat_row_gather_chunking_matches_full_gather(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    batch = 3
    n_rot = 5
    n_pixels = 4
    values = jnp.arange(batch * n_rot * n_pixels, dtype=jnp.float32).reshape(batch, n_rot, n_pixels)
    ctf_values = (100.0 + values).astype(jnp.float32)
    rotations = jnp.arange(batch * n_rot * 9, dtype=jnp.float32).reshape(batch * n_rot, 3, 3)
    active_indices = np.asarray([0, 3, 7, 11, 14, 3, 3, 3], dtype=np.int32)
    active_mask = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    full_values, _full_rotations = _select_active_flat_rows(
        values,
        rotations,
        active_indices,
        active_mask,
    )
    full_ctf_values = _select_active_flat_values(
        ctf_values,
        active_indices,
        active_mask,
    )

    calls = []

    def fake_accumulate_adjoint_block_chunked(
        flat_block,
        flat_rotations,
        volume,
        **kwargs,
    ):
        del flat_rotations, kwargs
        calls.append(int(flat_block.shape[0]))
        return volume + jnp.sum(flat_block)

    monkeypatch.setattr(bucketed_mod, "_accumulate_adjoint_block_chunked", fake_accumulate_adjoint_block_chunked)

    row_bytes = (
        n_pixels * np.dtype(np.float32).itemsize
        + n_pixels * np.dtype(np.float32).itemsize
        + 9 * np.dtype(np.float32).itemsize
    )
    y_volume, ctf_volume = bucketed_mod._accumulate_active_flat_rows_adjoint_chunked(
        values,
        ctf_values,
        rotations,
        active_indices,
        active_mask,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(10.0, dtype=jnp.float32),
        use_windowed_adjoint=False,
        image_shape=(8, 8),
        volume_shape=(8, 8, 8),
        disc_type="linear_interp",
        half_image=True,
        half_volume=False,
        max_r=None,
        relion_x_half=False,
        max_block_bytes=3 * row_bytes + 1,
        log_label_prefix="test-active",
    )

    assert calls == [3, 3, 3, 3, 2, 2]
    np.testing.assert_allclose(np.asarray(y_volume), np.asarray(jnp.sum(full_values)))
    np.testing.assert_allclose(np.asarray(ctf_volume), np.asarray(10.0 + jnp.sum(full_ctf_values)))


def test_sparse_pass2_rotation_chunked_xhalf_uses_relion_recon_indices():
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    source = inspect.getsource(bucketed_mod.compute_pass2_stats_sparse_bucketed)
    marker = "mstep_window_indices = relion_x_half_recon_indices if use_relion_x_half_mstep else recon_window_indices"
    marker_idx = source.index(marker)
    chunk_start = source.rfind(
        "if rotation_chunk_size is not None and int(rotation_chunk_size) < bucket_size:",
        0,
        marker_idx,
    )
    chunk_stop = source.index("if projection_cache is not None:", marker_idx)
    chunked_branch = source[chunk_start:chunk_stop]

    assert marker in chunked_branch
    assert chunked_branch.count("window_indices=mstep_window_indices") >= 2


def test_sparse_pass2_translation_tile_accounts_for_score_dtype():
    image_shape = (8, 8)
    n_fine_trans = 2
    half_image_size = image_shape[0] * (image_shape[1] // 2 + 1)
    max_tile_bytes = 7 * n_fine_trans * half_image_size * np.dtype(np.complex128).itemsize

    assert (
        _max_images_for_translation_tile(
            image_shape,
            n_fine_trans,
            max_tile_bytes=max_tile_bytes,
            complex_dtype=jnp.complex64,
        )
        == 14
    )
    assert (
        _max_images_for_translation_tile(
            image_shape,
            n_fine_trans,
            max_tile_bytes=max_tile_bytes,
            complex_dtype=jnp.complex128,
        )
        == 7
    )


def test_sparse_pass2_translation_tile_can_budget_window_pixels():
    image_shape = (8, 8)
    n_fine_trans = 2
    half_image_size = image_shape[0] * (image_shape[1] // 2 + 1)
    window_pixels = 7
    max_tile_bytes = 5 * n_fine_trans * half_image_size * np.dtype(np.complex64).itemsize

    full_cap = _max_images_for_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_tile_bytes,
        complex_dtype=jnp.complex64,
    )
    window_cap = _max_images_for_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_tile_bytes,
        complex_dtype=jnp.complex64,
        n_half_pixels=window_pixels,
    )

    assert full_cap == 5
    assert window_cap == 5 * half_image_size // window_pixels


def test_sparse_pass2_windowed_translation_tile_budget_defaults_to_active_pixels(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP", raising=False)

    assert (
        _translation_tile_half_pixels_for_budget(
            use_window=True,
            n_score_pixels=4003,
            n_recon_pixels=3923,
        )
        == 4003
    )
    assert (
        _translation_tile_half_pixels_for_budget(
            use_window=False,
            n_score_pixels=4003,
            n_recon_pixels=3923,
        )
        is None
    )

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP", "0")
    assert (
        _translation_tile_half_pixels_for_budget(
            use_window=True,
            n_score_pixels=4003,
            n_recon_pixels=3923,
        )
        is None
    )

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", "0")
    assert (
        _translation_tile_half_pixels_for_budget(
            use_window=True,
            n_score_pixels=4003,
            n_recon_pixels=3923,
        )
        is None
    )


def test_sparse_pass2_windowed_translation_tile_image_cap_is_bounded(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER", raising=False)

    image_shape = (8, 8)
    n_fine_trans = 2
    half_image_size = image_shape[0] * (image_shape[1] // 2 + 1)
    window_pixels = 5
    max_tile_bytes = 5 * n_fine_trans * half_image_size * np.dtype(np.complex64).itemsize

    chosen, full_cap, window_cap, multiplier = _max_images_for_sparse_pass2_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_tile_bytes,
        complex_dtype=jnp.complex64,
        translation_tile_half_pixels=window_pixels,
    )

    assert full_cap == 5
    assert window_cap == 5 * half_image_size // window_pixels
    assert multiplier == 4
    assert chosen == 20

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER", "2")
    chosen, full_cap, window_cap, multiplier = _max_images_for_sparse_pass2_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_tile_bytes,
        complex_dtype=jnp.complex64,
        translation_tile_half_pixels=window_pixels,
    )

    assert full_cap == 5
    assert window_cap == 5 * half_image_size // window_pixels
    assert multiplier == 2
    assert chosen == 10


def test_sparse_kclass_windowed_translation_tile_cap_defaults_on(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP", raising=False)
    assert _windowed_translation_tile_cap_enabled_for_pass() is True

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP", "0")
    assert _windowed_translation_tile_cap_enabled_for_pass() is False

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP", "1")
    assert _windowed_translation_tile_cap_enabled_for_pass() is True


def test_sparse_pass2_projection_cache_estimate_accounts_for_projection_dtype():
    n_rot = 11
    n_half = 17

    assert _projection_cache_budget_complex_dtype(jnp.complex64, jnp.complex64) == np.dtype(np.complex64)
    assert _projection_cache_budget_complex_dtype(jnp.complex64, jnp.complex128) == np.dtype(np.complex128)
    assert _projection_cache_budget_complex_dtype(jnp.complex128, jnp.complex64) == np.dtype(np.complex128)
    assert _projection_cache_budget_complex_dtype(
        jnp.complex64,
        jnp.complex64,
        use_relion_projector=True,
    ) == np.dtype(np.complex128)
    assert _projection_cache_transient_bytes(
        n_rot,
        n_half,
        projection_complex_dtype=jnp.complex64,
        include_abs2=True,
    ) == n_rot * n_half * (np.dtype(np.complex64).itemsize + np.dtype(np.float32).itemsize)
    assert _projection_cache_transient_bytes(
        n_rot,
        n_half,
        projection_complex_dtype=jnp.complex128,
        include_abs2=True,
    ) == n_rot * n_half * (np.dtype(np.complex128).itemsize + np.dtype(np.float64).itemsize)
    assert _projection_cache_transient_bytes(
        n_rot,
        n_half,
        projection_complex_dtype=jnp.complex128,
        include_abs2=False,
    ) == n_rot * n_half * np.dtype(np.complex128).itemsize


def test_sparse_pass2_projection_cache_budget_accounts_for_k_classes():
    assert _projection_cache_fits_budget(100, 250, n_classes=2)
    assert not _projection_cache_fits_budget(100, 250, n_classes=3)


def test_relion_windowed_projection_cache_estimate_admits_retained_window_cache():
    n_fine_rot = 147_456
    n_half = 128 * (128 // 2 + 1)
    n_windowed = 1_624
    n_recon_windowed = 1_227
    cap_12g = 12 * 1024**3

    retained_window_cache_estimate = _projection_cache_transient_bytes(
        n_fine_rot,
        n_windowed,
        projection_complex_dtype=np.complex64,
        include_abs2=False,
    ) + _projection_cache_transient_bytes(
        n_fine_rot,
        n_recon_windowed,
        projection_complex_dtype=np.complex64,
        include_abs2=True,
    )
    relion_centered_transient_estimate = _projection_cache_transient_bytes(
        n_fine_rot,
        _projection_budget_pixels_for_pass(
            n_half,
            use_window=True,
            use_relion_projector=True,
        ),
        projection_complex_dtype=_projection_cache_budget_complex_dtype(
            np.complex64,
            np.complex64,
            use_relion_projector=True,
        ),
        include_abs2=False,
    )

    assert retained_window_cache_estimate < cap_12g
    assert relion_centered_transient_estimate > cap_12g
    assert _projection_cache_fits_budget(retained_window_cache_estimate, cap_12g)

    per_call_cap = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=80 * 1024**3,
        n_projection_pixels=_projection_budget_pixels_for_pass(
            n_half,
            use_window=True,
            use_relion_projector=True,
        ),
        projection_complex_dtype=_projection_cache_budget_complex_dtype(
            np.complex64,
            np.complex64,
            use_relion_projector=True,
        ),
        include_abs2=False,
    )
    assert per_call_cap is not None
    assert per_call_cap < n_fine_rot


def test_fused_k_class_sparse_pass2_uses_coarse_tail_bucket_quantum(monkeypatch):
    """Pin the coarse default tail bucketing used by local sparse pass-2."""

    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    counts = [1025, 1152, 1537, 2049]

    def per_class_inputs():
        return {
            "oversampled_rots": [
                np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
                for count in counts
            ],
        }

    buckets = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(4)],
        n_fine_trans=116,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    default_sizes = sorted({int(bucket["bucket_size"]) for bucket in buckets})
    assert default_sizes == [4096]

    monkeypatch.setenv("RECOVAR_LOCAL_BUCKET_QUANTUM", "128")
    finer_buckets = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(4)],
        n_fine_trans=116,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    finer_sizes = sorted({int(bucket["bucket_size"]) for bucket in finer_buckets})
    assert finer_sizes == [1152, 1664, 2176]
    assert default_sizes[0] > finer_sizes[-1]


def test_single_class_sparse_pass2_bucket_quantum_can_coarsen_pathological_tail(monkeypatch):
    """Document tail coarsening for outlier-heavy K=1 cases."""

    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    n_fine_trans = 116
    counts = [1812, 4461, 9728, 12801, 23041, 45057, 76800]
    per_image_inputs = {
        "oversampled_rots": [
            np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
            for count in counts
        ],
    }

    default_buckets = _bucket_pass2_inputs(
        per_image_inputs,
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    default_sizes = [int(bucket["bucket_size"]) for bucket in default_buckets]

    monkeypatch.setenv("RECOVAR_LOCAL_BUCKET_QUANTUM", "512")
    fine_buckets = _bucket_pass2_inputs(
        per_image_inputs,
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    fine_sizes = [int(bucket["bucket_size"]) for bucket in fine_buckets]

    assert default_sizes == [4096, 8192, 12288, 16384, 24576, 49152, 77824]
    assert fine_sizes == [2048, 4608, 9728, 13312, 23552, 45568, 76800]
    assert len(set(default_sizes)) <= len(set(fine_sizes))
    assert default_sizes[-1] >= counts[-1]


def test_sparse_pass2_auto_projection_cap_prevents_one_image_tail_oversize(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", raising=False)
    n_half = 128 * (128 // 2 + 1)
    cap = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=80 * 1024**3,
        n_projection_pixels=n_half,
        projection_complex_dtype=np.complex64,
        include_abs2=False,
    )
    assert cap is not None
    assert cap < 90_112
    assert cap > 10_000

    cap_with_abs2 = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=80 * 1024**3,
        n_projection_pixels=n_half,
        projection_complex_dtype=np.complex64,
        include_abs2=True,
    )
    assert cap_with_abs2 is not None
    assert cap_with_abs2 < cap

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "1234")
    assert (
        _max_projected_rotations_per_call_for_pass(
            device_memory_bytes=80 * 1024**3,
            n_projection_pixels=n_half,
            projection_complex_dtype=np.complex64,
            include_abs2=False,
        )
        == 1234
    )


def _minimal_per_image_inputs(rotation_counts, n_fine_trans):
    oversampled_rots = [
        np.broadcast_to(np.eye(3, dtype=np.float32), (int(count), 3, 3)).copy()
        for count in rotation_counts
    ]
    return {
        "oversampled_rots": oversampled_rots,
        "oversampled_mstep_rots": oversampled_rots,
        "oversampled_rot_indices": [
            np.arange(int(count), dtype=np.int64)
            for count in rotation_counts
        ],
        "unique_rot": [
            np.arange(int(count), dtype=np.int32)
            for count in rotation_counts
        ],
        "parent_map": [
            np.arange(int(count), dtype=np.int32)
            for count in rotation_counts
        ],
        "log_prior": [
            np.zeros(int(count), dtype=np.float32)
            for count in rotation_counts
        ],
        "candidate_mask": [
            np.ones((int(count), n_fine_trans), dtype=bool)
            for count in rotation_counts
        ],
    }


def test_compact_fused_k_class_bucket_arrays_reduce_rectangular_padding(monkeypatch):
    """Opt-in compact fused buckets avoid padding every class to the largest class."""

    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    n_fine_trans = 4
    bucket = {
        "bucket_size": 128,
        "image_indices": np.asarray([0, 1, 2], dtype=np.int64),
    }
    per_class = [
        _minimal_per_image_inputs([17, 18, 17], n_fine_trans),
        _minimal_per_image_inputs([65, 66, 70], n_fine_trans),
    ]

    rectangular = _build_k_class_bucket_arrays(
        bucket,
        per_class,
        n_fine_trans,
        compact_buckets=False,
    )
    compact = _build_k_class_bucket_arrays(
        bucket,
        per_class,
        n_fine_trans,
        compact_buckets=True,
    )

    assert [int(arrays["bucket_size"]) for arrays in rectangular] == [128, 128]
    assert [int(arrays["bucket_size"]) for arrays in compact] == [32, 128]
    assert sum(int(arrays["bucket_size"]) for arrays in compact) < sum(
        int(arrays["bucket_size"]) for arrays in rectangular
    )
    np.testing.assert_array_equal(compact[0]["actual_counts"], np.asarray([17, 18, 17], dtype=np.int32))
    np.testing.assert_array_equal(compact[1]["actual_counts"], np.asarray([65, 66, 70], dtype=np.int32))
    assert not np.any(compact[0]["candidate_mask"][:, 18:, :])


def test_compact_pair_execution_bucket_arrays_skip_unused_dense_score_fields(monkeypatch):
    """Compact-pair execution does not need dense R x T score masks."""

    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    n_fine_trans = 4
    bucket = {
        "bucket_size": 128,
        "image_indices": np.asarray([0, 1, 2], dtype=np.int64),
    }
    per_class = [
        _minimal_per_image_inputs([17, 18, 17], n_fine_trans),
        _minimal_per_image_inputs([65, 66, 70], n_fine_trans),
    ]

    compact_pair_execution = _build_k_class_bucket_arrays(
        bucket,
        per_class,
        n_fine_trans,
        compact_buckets=True,
        include_dense_score_fields=False,
    )

    assert [int(arrays["bucket_size"]) for arrays in compact_pair_execution] == [32, 128]
    for arrays in compact_pair_execution:
        assert arrays["candidate_mask"] is None
        assert arrays["log_prior"] is None
        assert arrays["parent_map"] is None
        assert arrays["rotations"].shape[:2] == (
            bucket["image_indices"].shape[0],
            int(arrays["bucket_size"]),
        )
        assert arrays["rotation_indices"].shape == (
            bucket["image_indices"].shape[0],
            int(arrays["bucket_size"]),
        )
    np.testing.assert_array_equal(
        compact_pair_execution[0]["actual_counts"],
        np.asarray([17, 18, 17], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        compact_pair_execution[1]["actual_counts"],
        np.asarray([65, 66, 70], dtype=np.int32),
    )


def test_fused_k_class_sparse_pass2_projection_cap_does_not_fragment_buckets(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    counts = [1025] * 5 + [1537] * 5 + [2049] * 5

    def per_class_inputs():
        return {
            "oversampled_rots": [
                np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
                for count in counts
            ],
        }

    uncapped_buckets = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(4)],
        n_fine_trans=116,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    capped_buckets = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(4)],
        n_fine_trans=116,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )

    assert max(len(bucket["image_indices"]) for bucket in uncapped_buckets) == 15
    assert [(int(b["bucket_size"]), len(b["image_indices"])) for b in capped_buckets] == [
        (4096, 15),
    ]


def test_fused_k_class_sparse_pass2_budget_caps_real_score_tensors(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    n_classes = 4
    n_fine_trans = 116
    max_hypotheses = 10_001_475
    max_images = 8
    counts = [16, 1025, 1537, 2049, 3585, 4097] * 11

    def per_class_inputs():
        return {
            "oversampled_rots": [
                np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
                for count in counts
            ],
        }

    buckets = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(n_classes)],
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=max_hypotheses,
        max_images_per_microbatch=max_images,
    )

    assert buckets
    assert max(len(bucket["image_indices"]) for bucket in buckets) <= max_images
    for bucket in buckets:
        n_images = len(bucket["image_indices"])
        bucket_size = int(bucket["bucket_size"])
        assert n_classes * n_images * bucket_size * n_fine_trans <= max_hypotheses


def test_fused_k_class_sparse_pass2_can_chunk_small_buckets_larger(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    n_classes = 4
    n_fine_trans = 116
    counts = [16] * 40 + [1024] * 40

    def per_class_inputs():
        return {
            "oversampled_rots": [
                np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
                for count in counts
            ],
        }

    baseline = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(n_classes)],
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=8,
    )
    hybrid = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(n_classes)],
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=8,
        small_bucket_threshold=128,
        small_bucket_max_images_per_microbatch=19,
    )

    baseline_small_chunks = [
        len(bucket["image_indices"])
        for bucket in baseline
        if int(bucket["bucket_size"]) <= 128
    ]
    hybrid_small_chunks = [
        len(bucket["image_indices"])
        for bucket in hybrid
        if int(bucket["bucket_size"]) <= 128
    ]
    hybrid_large_chunks = [
        len(bucket["image_indices"])
        for bucket in hybrid
        if int(bucket["bucket_size"]) > 128
    ]

    assert max(baseline_small_chunks) == 8
    assert max(hybrid_small_chunks) == 19
    assert len(hybrid_small_chunks) < len(baseline_small_chunks)
    assert max(hybrid_large_chunks) == 8


def test_fused_k_class_sparse_pass2_can_coalesce_small_bucket_tail(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    n_classes = 4
    n_fine_trans = 116
    counts = [16] * 7 + [32] * 12 + [64] * 23 + [128] * 64 + [256] * 3

    def per_class_inputs():
        return {
            "oversampled_rots": [
                np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
                for count in counts
            ],
        }

    baseline = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(n_classes)],
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    coalesced = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(n_classes)],
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
        small_bucket_coalesce_size=128,
    )

    assert sorted({int(bucket["bucket_size"]) for bucket in baseline}) == [16, 32, 64, 128, 256]
    assert sorted({int(bucket["bucket_size"]) for bucket in coalesced}) == [128, 256]
    assert len(coalesced) < len(baseline)
    assert sum(len(bucket["image_indices"]) for bucket in coalesced) == len(counts)


def test_fused_k_class_sparse_pass2_can_coalesce_high_bucket_tail(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    n_classes = 4
    n_fine_trans = 116
    counts = [4097] * 2 + [8193] * 3 + [12289] * 2

    def per_class_inputs():
        return {
            "oversampled_rots": [
                np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
                for count in counts
            ],
        }

    baseline = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(n_classes)],
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    coalesced = _bucket_sparse_k_class_pass2_inputs(
        [per_class_inputs() for _ in range(n_classes)],
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
        tail_bucket_coalesce_max_images=8,
        tail_bucket_coalesce_max_inflation=2.0,
        tail_bucket_coalesce_min_bucket_size=4096,
    )

    assert sorted({int(bucket["bucket_size"]) for bucket in baseline}) == [8192, 12288, 16384]
    assert sorted({int(bucket["bucket_size"]) for bucket in coalesced}) == [16384]
    assert sum(len(bucket["image_indices"]) for bucket in coalesced) == len(counts)


def test_compact_candidate_pair_builder_matches_dense_mask_nonzero():
    masks = [
        np.asarray(
            [
                [False, True, False, False],
                [True, False, True, False],
                [False, False, False, True],
            ],
            dtype=bool,
        ),
        np.ones((2, 4), dtype=bool),
        np.zeros((1, 4), dtype=bool),
    ]
    rotation_indices = [
        np.asarray([10, 11, 12], dtype=np.int64),
        np.asarray([20, 21], dtype=np.int64),
        np.asarray([30], dtype=np.int64),
    ]
    log_priors = [
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([3.0], dtype=np.float32),
    ]
    per_image_inputs = {
        "candidate_mask": masks,
        "oversampled_rot_indices": rotation_indices,
        "log_prior": log_priors,
    }

    compact = _prepare_per_image_compact_candidate_pairs(per_image_inputs)

    for image_idx, dense_mask in enumerate(masks):
        expected_rows, expected_trans = np.nonzero(dense_mask)
        np.testing.assert_array_equal(compact["local_rotation_row"][image_idx], expected_rows)
        np.testing.assert_array_equal(compact["translation_idx"][image_idx], expected_trans)
        np.testing.assert_array_equal(
            compact["rotation_index"][image_idx],
            rotation_indices[image_idx][expected_rows],
        )
        np.testing.assert_array_equal(
            compact["log_prior"][image_idx],
            log_priors[image_idx][expected_rows],
        )
        np.testing.assert_array_equal(
            compact["pair_mask"][image_idx],
            np.ones(expected_rows.shape[0], dtype=bool),
        )
        assert int(compact["pair_counts"][image_idx]) == int(dense_mask.sum())


def test_compact_pair_gaussian_scores_and_log_z_match_dense_masked_bucket():
    rng = np.random.default_rng(114)
    batch = 2
    n_rot = 3
    n_trans = 4
    n_pixels = 5
    shifted = rng.standard_normal((batch, n_trans, n_pixels)).astype(np.float32) + 1j * rng.standard_normal(
        (batch, n_trans, n_pixels),
    ).astype(np.float32)
    proj = rng.standard_normal((batch, n_rot, n_pixels)).astype(np.float32) + 1j * rng.standard_normal(
        (batch, n_rot, n_pixels),
    ).astype(np.float32)
    corr_img_score = (0.5 + rng.random((batch, n_pixels))).astype(np.float32)
    half_weights = np.linspace(0.25, 1.25, n_pixels, dtype=np.float32)
    rotation_log_prior = rng.normal(loc=-0.2, scale=0.3, size=(batch, n_rot)).astype(np.float32)
    translation_log_prior = rng.normal(loc=-0.1, scale=0.2, size=(batch, n_trans)).astype(np.float32)
    candidate_mask = np.asarray(
        [
            [
                [True, False, True, False],
                [False, True, False, False],
                [False, False, True, True],
            ],
            [
                [False, True, False, False],
                [True, False, True, False],
                [False, False, False, True],
            ],
        ],
        dtype=bool,
    )
    per_image_inputs = {
        "candidate_mask": [candidate_mask[0], candidate_mask[1]],
        "oversampled_rot_indices": [
            np.asarray([10, 11, 12], dtype=np.int64),
            np.asarray([20, 21, 22], dtype=np.int64),
        ],
        "log_prior": [rotation_log_prior[0], rotation_log_prior[1]],
    }
    compact = _prepare_per_image_compact_candidate_pairs(per_image_inputs)
    arrays = _build_compact_pair_bucket_arrays(
        {
            "pair_bucket_size": int(max(compact["pair_counts"])) + 3,
            "image_indices": np.asarray([0, 1], dtype=np.int64),
        },
        compact,
    )

    dense_scores = _score_pass2_bucket_relion_gpu_diff2(
        jnp.asarray(shifted),
        jnp.asarray(corr_img_score),
        jnp.asarray(proj),
        jnp.asarray(half_weights),
        jnp.asarray(rotation_log_prior),
        jnp.asarray(translation_log_prior),
        jnp.asarray(candidate_mask),
    )
    compact_scores = _score_pass2_pairs_relion_gpu_diff2(
        jnp.asarray(shifted),
        jnp.asarray(corr_img_score),
        jnp.asarray(proj),
        jnp.asarray(half_weights),
        jnp.asarray(arrays["log_prior"]),
        jnp.asarray(translation_log_prior),
        jnp.asarray(arrays["local_rotation_row"]),
        jnp.asarray(arrays["translation_idx"]),
        jnp.asarray(arrays["pair_mask"]),
    )

    dense_scores_np = np.asarray(dense_scores)
    compact_scores_np = np.asarray(compact_scores)
    for image_idx in range(batch):
        rows, trans = np.nonzero(candidate_mask[image_idx])
        count = int(compact["pair_counts"][image_idx])
        np.testing.assert_allclose(
            compact_scores_np[image_idx, :count],
            dense_scores_np[image_idx, rows, trans],
            rtol=1e-6,
            atol=1e-6,
        )
        assert np.all(np.isneginf(compact_scores_np[image_idx, count:]))

    dense_log_z = _logsumexp_pass2_bucket_score_only(dense_scores)
    compact_log_z = _logsumexp_pass2_pairs_score_only(
        compact_scores,
        jnp.asarray(arrays["pair_mask"]),
    )
    np.testing.assert_allclose(np.asarray(compact_log_z), np.asarray(dense_log_z), rtol=1e-6, atol=1e-6)


def test_compact_pair_normalized_cc_scores_match_dense_masked_bucket():
    rng = np.random.default_rng(214)
    batch = 2
    n_rot = 3
    n_trans = 4
    n_pixels = 5
    shifted = rng.standard_normal((batch, n_trans, n_pixels)).astype(np.float32) + 1j * rng.standard_normal(
        (batch, n_trans, n_pixels),
    ).astype(np.float32)
    proj = rng.standard_normal((batch, n_rot, n_pixels)).astype(np.float32) + 1j * rng.standard_normal(
        (batch, n_rot, n_pixels),
    ).astype(np.float32)
    score_weight = (0.5 + rng.random((batch, n_pixels))).astype(np.float32)
    half_weights = np.linspace(0.25, 1.25, n_pixels, dtype=np.float32)
    candidate_mask = np.asarray(
        [
            [
                [True, False, True, False],
                [False, True, False, False],
                [False, False, True, True],
            ],
            [
                [False, True, False, False],
                [True, False, True, False],
                [False, False, False, True],
            ],
        ],
        dtype=bool,
    )
    per_image_inputs = {
        "candidate_mask": [candidate_mask[0], candidate_mask[1]],
        "oversampled_rot_indices": [
            np.asarray([10, 11, 12], dtype=np.int64),
            np.asarray([20, 21, 22], dtype=np.int64),
        ],
        "log_prior": [np.zeros(n_rot, dtype=np.float32), np.zeros(n_rot, dtype=np.float32)],
    }
    compact = _prepare_per_image_compact_candidate_pairs(per_image_inputs)
    arrays = _build_compact_pair_bucket_arrays(
        {
            "pair_bucket_size": int(max(compact["pair_counts"])) + 3,
            "image_indices": np.asarray([0, 1], dtype=np.int64),
        },
        compact,
    )

    dense_scores = _score_pass2_bucket_normalized_cc(
        jnp.asarray(shifted),
        jnp.asarray(score_weight),
        jnp.asarray(proj),
        jnp.asarray(half_weights),
        jnp.asarray(candidate_mask),
    )
    compact_scores = _score_pass2_pairs_normalized_cc(
        jnp.asarray(shifted),
        jnp.asarray(score_weight),
        jnp.asarray(proj),
        jnp.asarray(half_weights),
        jnp.asarray(arrays["local_rotation_row"]),
        jnp.asarray(arrays["translation_idx"]),
        jnp.asarray(arrays["pair_mask"]),
    )

    dense_scores_np = np.asarray(dense_scores)
    compact_scores_np = np.asarray(compact_scores)
    for image_idx in range(batch):
        rows, trans = np.nonzero(candidate_mask[image_idx])
        count = int(compact["pair_counts"][image_idx])
        np.testing.assert_allclose(
            compact_scores_np[image_idx, :count],
            dense_scores_np[image_idx, rows, trans],
            rtol=1e-6,
            atol=1e-6,
        )
        assert np.all(np.isneginf(compact_scores_np[image_idx, count:]))

    dense_log_z = _logsumexp_pass2_bucket_score_only(dense_scores)
    compact_log_z = _logsumexp_pass2_pairs_score_only(
        compact_scores,
        jnp.asarray(arrays["pair_mask"]),
    )
    np.testing.assert_allclose(np.asarray(compact_log_z), np.asarray(dense_log_z), rtol=1e-6, atol=1e-6)


def test_compact_pair_global_log_z_normalization_matches_dense_valid_pairs():
    rng = np.random.default_rng(231)
    batch = 3
    n_rot = 4
    n_trans = 5
    dense_scores = rng.normal(loc=-0.5, scale=2.0, size=(batch, n_rot, n_trans)).astype(np.float32)
    candidate_mask = np.asarray(
        [
            [
                [True, False, False, True, False],
                [False, True, False, False, False],
                [False, False, True, False, True],
                [False, False, False, False, False],
            ],
            [
                [False, True, False, False, False],
                [True, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
            np.zeros((n_rot, n_trans), dtype=bool),
        ],
        dtype=bool,
    )
    dense_scores = np.where(candidate_mask, dense_scores, -np.inf)
    per_image_inputs = {
        "candidate_mask": [candidate_mask[i] for i in range(batch)],
        "oversampled_rot_indices": [np.arange(n_rot, dtype=np.int64) + 10 * i for i in range(batch)],
        "log_prior": [np.zeros(n_rot, dtype=np.float32) for _ in range(batch)],
    }
    compact = _prepare_per_image_compact_candidate_pairs(per_image_inputs)
    arrays = _build_compact_pair_bucket_arrays(
        {
            "pair_bucket_size": int(max(compact["pair_counts"])) + 2,
            "image_indices": np.arange(batch, dtype=np.int64),
        },
        compact,
    )
    pair_scores = np.full_like(arrays["log_prior"], -np.inf, dtype=np.float32)
    for image_idx in range(batch):
        count = int(compact["pair_counts"][image_idx])
        if count == 0:
            continue
        rows = arrays["local_rotation_row"][image_idx, :count]
        trans = arrays["translation_idx"][image_idx, :count]
        pair_scores[image_idx, :count] = dense_scores[image_idx, rows, trans]

    dense_local_log_z = _logsumexp_pass2_bucket_score_only(jnp.asarray(dense_scores))
    other_class_log_z = jnp.asarray([1.2, -0.3, 0.7], dtype=jnp.float32)
    global_log_z = jnp.logaddexp(dense_local_log_z, other_class_log_z)
    dense_safe_log_z, dense_probs, dense_best_score, dense_best_argmax, dense_max_post = (
        _normalize_pass2_bucket_with_log_z(jnp.asarray(dense_scores), global_log_z)
    )
    safe_log_z, pair_probs, best_log_score, best_pair_argmax, max_posterior = (
        _normalize_pass2_pairs_with_log_z(
            jnp.asarray(pair_scores),
            jnp.asarray(arrays["pair_mask"]),
            global_log_z,
        )
    )

    np.testing.assert_allclose(np.asarray(safe_log_z), np.asarray(dense_safe_log_z), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(best_log_score), np.asarray(dense_best_score), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(max_posterior), np.asarray(dense_max_post), rtol=1e-6, atol=1e-6)
    for image_idx in range(batch):
        count = int(compact["pair_counts"][image_idx])
        rows = arrays["local_rotation_row"][image_idx, :count]
        trans = arrays["translation_idx"][image_idx, :count]
        np.testing.assert_allclose(
            np.asarray(pair_probs)[image_idx, :count],
            np.asarray(dense_probs)[image_idx, rows, trans],
            rtol=1e-6,
            atol=1e-6,
        )
        assert np.all(np.asarray(pair_probs)[image_idx, count:] == 0.0)
        if count:
            dense_flat_argmax = int(np.asarray(dense_best_argmax)[image_idx])
            best_row, best_trans = divmod(dense_flat_argmax, n_trans)
            pair_argmax = int(np.asarray(best_pair_argmax)[image_idx])
            assert int(rows[pair_argmax]) == best_row
            assert int(trans[pair_argmax]) == best_trans
        else:
            assert int(np.asarray(best_pair_argmax)[image_idx]) == 0


def test_compact_pair_weighted_rotation_sums_match_dense_mstep_helpers():
    rng = np.random.default_rng(232)
    batch = 3
    n_rot = 4
    n_trans = 5
    n_pixels = 6
    candidate_mask = np.asarray(
        [
            [
                [True, False, False, True, False],
                [False, True, False, False, False],
                [False, False, True, False, True],
                [False, False, False, False, False],
            ],
            [
                [False, True, False, False, False],
                [True, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
            np.zeros((n_rot, n_trans), dtype=bool),
        ],
        dtype=bool,
    )
    dense_probs = (rng.random((batch, n_rot, n_trans)) * candidate_mask).astype(np.float32)
    shifted = rng.standard_normal((batch, n_trans, n_pixels)).astype(np.float32) + 1j * rng.standard_normal(
        (batch, n_trans, n_pixels),
    ).astype(np.float32)
    ctf2_over_nv = (0.25 + rng.random((batch, n_pixels))).astype(np.float32)
    per_image_inputs = {
        "candidate_mask": [candidate_mask[i] for i in range(batch)],
        "oversampled_rot_indices": [np.arange(n_rot, dtype=np.int64) + 20 * i for i in range(batch)],
        "log_prior": [np.zeros(n_rot, dtype=np.float32) for _ in range(batch)],
    }
    compact = _prepare_per_image_compact_candidate_pairs(per_image_inputs)
    arrays = _build_compact_pair_bucket_arrays(
        {
            "pair_bucket_size": int(max(compact["pair_counts"])) + 3,
            "image_indices": np.arange(batch, dtype=np.int64),
        },
        compact,
    )
    pair_probs = np.zeros_like(arrays["log_prior"], dtype=np.float32)
    for image_idx in range(batch):
        count = int(compact["pair_counts"][image_idx])
        rows = arrays["local_rotation_row"][image_idx, :count]
        trans = arrays["translation_idx"][image_idx, :count]
        pair_probs[image_idx, :count] = dense_probs[image_idx, rows, trans]

    dense_summed = compute_local_weighted_sums(jnp.asarray(dense_probs), jnp.asarray(shifted))
    dense_ctf_probs = compute_local_ctf_sums(jnp.asarray(dense_probs), jnp.asarray(ctf2_over_nv))
    dense_probs_sum_t = jnp.sum(jnp.asarray(dense_probs), axis=-1)
    dense_translation_posterior = jnp.sum(jnp.asarray(dense_probs), axis=1)
    compact_dense_probs = _compact_pair_dense_probs_and_reductions(
        jnp.asarray(pair_probs),
        jnp.asarray(arrays["local_rotation_row"]),
        jnp.asarray(arrays["translation_idx"]),
        jnp.asarray(arrays["pair_mask"]),
        n_rotation_rows=n_rot,
        n_trans=n_trans,
    )
    compact_summed, compact_ctf_probs, compact_probs_sum_t, compact_translation_posterior = (
        _compact_pair_weighted_rotation_sums(
            jnp.asarray(pair_probs),
            jnp.asarray(arrays["local_rotation_row"]),
            jnp.asarray(arrays["translation_idx"]),
            jnp.asarray(arrays["pair_mask"]),
            jnp.asarray(shifted),
            jnp.asarray(ctf2_over_nv),
            n_rotation_rows=n_rot,
        )
    )

    np.testing.assert_allclose(np.asarray(compact_summed), np.asarray(dense_summed), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(compact_ctf_probs), np.asarray(dense_ctf_probs), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(compact_dense_probs), dense_probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(compact_dense_probs, axis=-1)),
        np.asarray(dense_probs_sum_t),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.sum(compact_dense_probs, axis=1)),
        np.asarray(dense_translation_posterior),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(compact_probs_sum_t),
        np.asarray(dense_probs_sum_t),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(compact_translation_posterior),
        np.asarray(dense_translation_posterior),
        rtol=1e-6,
        atol=1e-6,
    )
    active_indices, active_mask, active_count = _active_flat_row_indices_from_probs_sum_t(
        compact_probs_sum_t,
        pad_multiple=4,
    )
    flat_rotations = jnp.arange(batch * n_rot * 9, dtype=jnp.float32).reshape(batch * n_rot, 3, 3)
    active_summed, active_ctf_probs, active_rotations = _rectangular_active_weighted_sums_or_none(
        compact_dense_probs,
        compact_probs_sum_t,
        jnp.asarray(shifted),
        jnp.asarray(ctf2_over_nv),
        flat_rotations,
        active_indices,
        active_mask,
    )
    selected_summed, selected_rotations = _select_active_flat_rows(
        compact_summed,
        flat_rotations,
        active_indices,
        active_mask,
    )
    selected_ctf_probs = _select_active_flat_values(
        compact_ctf_probs,
        active_indices,
        active_mask,
    )

    assert active_count > 0
    np.testing.assert_allclose(np.asarray(active_summed), np.asarray(selected_summed), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(active_ctf_probs), np.asarray(selected_ctf_probs), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(active_rotations), np.asarray(selected_rotations))


def test_compact_pair_weighted_rotation_and_image_sums_match_separate_helpers(monkeypatch):
    rng = np.random.default_rng(237)
    batch = 3
    n_rot = 4
    n_trans = 5
    n_pixels = 6
    candidate_mask = np.asarray(
        [
            [
                [True, False, False, True, False],
                [False, True, False, False, False],
                [False, False, True, False, True],
                [False, False, False, False, False],
            ],
            [
                [False, True, False, False, False],
                [True, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
            np.zeros((n_rot, n_trans), dtype=bool),
        ],
        dtype=bool,
    )
    dense_probs = (rng.random((batch, n_rot, n_trans)) * candidate_mask).astype(np.float32)
    shifted_recon = rng.standard_normal((batch, n_trans, n_pixels)).astype(np.float32) + 1j * rng.standard_normal(
        (batch, n_trans, n_pixels),
    ).astype(np.float32)
    shifted_noise = rng.standard_normal((batch, n_trans, n_pixels)).astype(np.float32) + 1j * rng.standard_normal(
        (batch, n_trans, n_pixels),
    ).astype(np.float32)
    ctf2_over_nv = (0.25 + rng.random((batch, n_pixels))).astype(np.float32)
    per_image_inputs = {
        "candidate_mask": [candidate_mask[i] for i in range(batch)],
        "oversampled_rot_indices": [np.arange(n_rot, dtype=np.int64) + 20 * i for i in range(batch)],
        "log_prior": [np.zeros(n_rot, dtype=np.float32) for _ in range(batch)],
    }
    compact = _prepare_per_image_compact_candidate_pairs(per_image_inputs)
    arrays = _build_compact_pair_bucket_arrays(
        {
            "pair_bucket_size": int(max(compact["pair_counts"])) + 3,
            "image_indices": np.arange(batch, dtype=np.int64),
        },
        compact,
    )
    pair_probs = np.zeros_like(arrays["log_prior"], dtype=np.float32)
    for image_idx in range(batch):
        count = int(compact["pair_counts"][image_idx])
        rows = arrays["local_rotation_row"][image_idx, :count]
        trans = arrays["translation_idx"][image_idx, :count]
        pair_probs[image_idx, :count] = dense_probs[image_idx, rows, trans]

    separate_summed, separate_ctf_probs, separate_probs_sum_t, separate_translation_posterior = (
        _compact_pair_weighted_rotation_sums(
            jnp.asarray(pair_probs),
            jnp.asarray(arrays["local_rotation_row"]),
            jnp.asarray(arrays["translation_idx"]),
            jnp.asarray(arrays["pair_mask"]),
            jnp.asarray(shifted_recon),
            jnp.asarray(ctf2_over_nv),
            n_rotation_rows=n_rot,
        )
    )
    separate_image_summed = _compact_pair_weighted_image_sums(
        jnp.asarray(pair_probs),
        jnp.asarray(arrays["local_rotation_row"]),
        jnp.asarray(arrays["translation_idx"]),
        jnp.asarray(arrays["pair_mask"]),
        jnp.asarray(shifted_noise),
        n_rotation_rows=n_rot,
    )
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS", "0")
    legacy_summed, legacy_image_summed, legacy_ctf_probs, legacy_probs_sum_t, legacy_translation_posterior = (
        _compact_pair_weighted_rotation_and_image_sums(
            jnp.asarray(pair_probs),
            jnp.asarray(arrays["local_rotation_row"]),
            jnp.asarray(arrays["translation_idx"]),
            jnp.asarray(arrays["pair_mask"]),
            jnp.asarray(shifted_recon),
            jnp.asarray(shifted_noise),
            jnp.asarray(ctf2_over_nv),
            n_rotation_rows=n_rot,
        )
    )

    np.testing.assert_allclose(np.asarray(legacy_summed), np.asarray(separate_summed), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(legacy_image_summed),
        np.asarray(separate_image_summed),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(legacy_ctf_probs), np.asarray(separate_ctf_probs), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(legacy_probs_sum_t),
        np.asarray(separate_probs_sum_t),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(legacy_translation_posterior),
        np.asarray(separate_translation_posterior),
        rtol=1e-6,
        atol=1e-6,
    )

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS", "1")
    (
        fused_combined_summed,
        fused_combined_image_summed,
        fused_combined_ctf_probs,
        fused_combined_probs_sum_t,
        fused_combined_translation_posterior,
    ) = (
        _compact_pair_weighted_rotation_and_image_sums(
            jnp.asarray(pair_probs),
            jnp.asarray(arrays["local_rotation_row"]),
            jnp.asarray(arrays["translation_idx"]),
            jnp.asarray(arrays["pair_mask"]),
            jnp.asarray(shifted_recon),
            jnp.asarray(shifted_noise),
            jnp.asarray(ctf2_over_nv),
            n_rotation_rows=n_rot,
        )
    )
    np.testing.assert_allclose(np.asarray(fused_combined_summed), np.asarray(separate_summed), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(fused_combined_image_summed),
        np.asarray(separate_image_summed),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused_combined_ctf_probs),
        np.asarray(separate_ctf_probs),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused_combined_probs_sum_t),
        np.asarray(separate_probs_sum_t),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused_combined_translation_posterior),
        np.asarray(separate_translation_posterior),
        rtol=1e-6,
        atol=1e-6,
    )


def _make_compact_pair_sparse_mstep_case(dtype=np.float64):
    rng = np.random.default_rng(241)
    batch = 4
    n_rot = 5
    n_trans = 6
    n_pairs = 12
    n_pixels = 7
    local_rotation_row = np.asarray(
        [
            [1, 1, 3, 4, -1, 0, 2, 2, 5, 1, 0, 0],
            [0, 2, 2, 2, 3, 3, 1, 1, 4, -2, 0, 0],
            [4, 4, 4, 1, 1, 0, 2, 3, 3, 0, 0, 0],
            [2, 2, 0, 0, 1, 4, 4, 3, 3, 3, 0, 0],
        ],
        dtype=np.int32,
    )
    translation_idx = np.asarray(
        [
            [2, 2, 1, 4, 3, 0, 5, 5, 0, 6, 1, 0],
            [1, 3, 3, 4, 0, 0, 5, 5, 2, 2, -1, 0],
            [0, 0, 2, 2, 2, 1, 4, 4, 4, 0, 0, 0],
            [5, 5, 2, 2, 3, 1, 1, 4, 0, 6, 0, 0],
        ],
        dtype=np.int32,
    )
    pair_mask = np.asarray(
        [
            [True, True, True, True, True, False, True, True, True, True, False, False],
            [True, True, True, True, True, True, True, True, True, True, True, False],
            [True, True, True, True, True, False, True, True, True, False, False, False],
            [True, True, True, True, True, True, True, True, False, True, False, False],
        ],
        dtype=bool,
    )
    pair_probs = rng.random((batch, n_pairs)).astype(dtype)
    pair_probs[0, 5] = np.nan
    pair_probs[0, 10] = np.inf
    pair_probs[1, 9] = np.nan
    pair_probs[1, 10] = np.inf
    pair_probs[2, 9:] = np.nan
    pair_probs[3, 8] = np.inf
    pair_probs[3, 10:] = np.nan
    complex_dtype = np.complex128 if dtype == np.float64 else np.complex64
    shifted_recon = (
        rng.standard_normal((batch, n_trans, n_pixels)).astype(dtype)
        + 1j * rng.standard_normal((batch, n_trans, n_pixels)).astype(dtype)
    ).astype(complex_dtype)
    shifted_image = (
        rng.standard_normal((batch, n_trans, n_pixels)).astype(dtype)
        + 1j * rng.standard_normal((batch, n_trans, n_pixels)).astype(dtype)
    ).astype(complex_dtype)
    ctf2_over_nv = (0.25 + rng.random((batch, n_pixels))).astype(dtype)
    return {
        "pair_probs": pair_probs,
        "local_rotation_row": local_rotation_row,
        "translation_idx": translation_idx,
        "pair_mask": pair_mask,
        "shifted_recon": shifted_recon,
        "shifted_image": shifted_image,
        "ctf2_over_nv": ctf2_over_nv,
        "n_rot": n_rot,
    }


def _call_rotation_sums(fn, case):
    return fn(
        jnp.asarray(case["pair_probs"]),
        jnp.asarray(case["local_rotation_row"]),
        jnp.asarray(case["translation_idx"]),
        jnp.asarray(case["pair_mask"]),
        jnp.asarray(case["shifted_recon"]),
        jnp.asarray(case["ctf2_over_nv"]),
        n_rotation_rows=case["n_rot"],
    )


def _call_image_sums(fn, case):
    return fn(
        jnp.asarray(case["pair_probs"]),
        jnp.asarray(case["local_rotation_row"]),
        jnp.asarray(case["translation_idx"]),
        jnp.asarray(case["pair_mask"]),
        jnp.asarray(case["shifted_image"]),
        n_rotation_rows=case["n_rot"],
    )


def _call_rotation_and_image_sums(fn, case):
    return fn(
        jnp.asarray(case["pair_probs"]),
        jnp.asarray(case["local_rotation_row"]),
        jnp.asarray(case["translation_idx"]),
        jnp.asarray(case["pair_mask"]),
        jnp.asarray(case["shifted_recon"]),
        jnp.asarray(case["shifted_image"]),
        jnp.asarray(case["ctf2_over_nv"]),
        n_rotation_rows=case["n_rot"],
    )


def _assert_tree_allclose(actual, expected, *, rtol, atol):
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=rtol,
            atol=atol,
        )


def test_compact_pair_pair_sparse_rotation_sums_match_dense_cpu_float64():
    case = _make_compact_pair_sparse_mstep_case(dtype=np.float64)
    with jax.default_device(jax.devices("cpu")[0]):
        dense = _call_rotation_sums(_compact_pair_weighted_rotation_sums_dense, case)
        pair_sparse = _call_rotation_sums(_compact_pair_weighted_rotation_sums_pair_sparse, case)

    _assert_tree_allclose(pair_sparse, dense, rtol=1e-12, atol=1e-12)


def test_compact_pair_pair_sparse_image_and_combined_sums_match_dense_cpu_float64(monkeypatch):
    case = _make_compact_pair_sparse_mstep_case(dtype=np.float64)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP", raising=False)
    with jax.default_device(jax.devices("cpu")[0]):
        dense_image = _call_image_sums(_compact_pair_weighted_image_sums_dense, case)
        pair_sparse_image = _call_image_sums(_compact_pair_weighted_image_sums_pair_sparse, case)
        dense_combined = _call_rotation_and_image_sums(_compact_pair_weighted_rotation_and_image_sums, case)
        pair_sparse_combined = _call_rotation_and_image_sums(
            _compact_pair_weighted_rotation_and_image_sums_pair_sparse,
            case,
        )

    np.testing.assert_allclose(np.asarray(pair_sparse_image), np.asarray(dense_image), rtol=1e-12, atol=1e-12)
    _assert_tree_allclose(pair_sparse_combined, dense_combined, rtol=1e-12, atol=1e-12)


def test_compact_pair_mstep_pair_sparse_env_matches_dense_cpu_float64(monkeypatch):
    case = _make_compact_pair_sparse_mstep_case(dtype=np.float64)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP", "pair_sparse")
    assert _compact_pair_mstep_mode_for_pass() == "pair_sparse"
    with jax.default_device(jax.devices("cpu")[0]):
        expected = _call_rotation_sums(_compact_pair_weighted_rotation_sums_dense, case)
        actual = _call_rotation_sums(_compact_pair_weighted_rotation_sums, case)

    _assert_tree_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_compact_pair_mstep_default_remains_dense(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    case = _make_compact_pair_sparse_mstep_case(dtype=np.float64)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP", raising=False)
    assert _compact_pair_mstep_mode_for_pass() == "dense"

    def fail_pair_sparse(*_args, **_kwargs):
        raise AssertionError("pair-sparse compact-pair M-step path should be opt-in")

    monkeypatch.setattr(
        bucketed_mod,
        "_compact_pair_weighted_rotation_sums_pair_sparse",
        fail_pair_sparse,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        expected = _call_rotation_sums(bucketed_mod._compact_pair_weighted_rotation_sums_dense, case)
        actual = _call_rotation_sums(bucketed_mod._compact_pair_weighted_rotation_sums, case)

    _assert_tree_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_compact_pair_relion_reconstruction_threshold_matches_dense_valid_pairs():
    rng = np.random.default_rng(233)
    batch = 3
    n_rot = 4
    n_trans = 5
    candidate_mask = np.asarray(
        [
            [
                [True, False, False, True, False],
                [False, True, False, False, False],
                [False, False, True, False, True],
                [False, False, False, False, False],
            ],
            [
                [False, True, False, False, False],
                [True, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
            np.zeros((n_rot, n_trans), dtype=bool),
        ],
        dtype=bool,
    )
    dense_probs = (rng.random((batch, n_rot, n_trans)) * candidate_mask).astype(np.float32)
    dense_probs = dense_probs / np.maximum(dense_probs.reshape(batch, -1).sum(axis=1), 1.0)[:, None, None]
    per_image_inputs = {
        "candidate_mask": [candidate_mask[i] for i in range(batch)],
        "oversampled_rot_indices": [np.arange(n_rot, dtype=np.int64) + 30 * i for i in range(batch)],
        "log_prior": [np.zeros(n_rot, dtype=np.float32) for _ in range(batch)],
    }
    compact = _prepare_per_image_compact_candidate_pairs(per_image_inputs)
    arrays = _build_compact_pair_bucket_arrays(
        {
            "pair_bucket_size": int(max(compact["pair_counts"])) + 3,
            "image_indices": np.arange(batch, dtype=np.int64),
        },
        compact,
    )
    pair_probs = np.zeros_like(arrays["log_prior"], dtype=np.float32)
    for image_idx in range(batch):
        count = int(compact["pair_counts"][image_idx])
        rows = arrays["local_rotation_row"][image_idx, :count]
        trans = arrays["translation_idx"][image_idx, :count]
        pair_probs[image_idx, :count] = dense_probs[image_idx, rows, trans]

    dense_thresholded, _dense_mask, _dense_n = _relion_pass2_reconstruction_probs(
        jnp.asarray(dense_probs),
        adaptive_fraction=0.7,
    )
    compact_thresholded, _compact_mask, _compact_n = _relion_pass2_reconstruction_pair_probs(
        jnp.asarray(pair_probs),
        jnp.asarray(arrays["pair_mask"]),
        adaptive_fraction=0.7,
    )

    dense_thresholded_np = np.asarray(dense_thresholded)
    compact_thresholded_np = np.asarray(compact_thresholded)
    for image_idx in range(batch):
        count = int(compact["pair_counts"][image_idx])
        rows = arrays["local_rotation_row"][image_idx, :count]
        trans = arrays["translation_idx"][image_idx, :count]
        np.testing.assert_allclose(
            compact_thresholded_np[image_idx, :count],
            dense_thresholded_np[image_idx, rows, trans],
            rtol=1e-6,
            atol=1e-6,
        )
        assert np.all(compact_thresholded_np[image_idx, count:] == 0.0)


def test_relion_reconstruction_threshold_excludes_zero_probability_tail():
    probs = jnp.asarray(
        [
            [
                [0.50, 0.00, 0.25, 0.00],
                [0.00, 0.25, 0.00, 0.00],
            ]
        ],
        dtype=jnp.float32,
    )
    pair_probs = jnp.asarray([[0.50, 0.00, 0.25, 0.00, 0.25, 0.00]], dtype=jnp.float32)
    pair_mask = jnp.asarray([[True, True, True, False, True, False]], dtype=bool)

    _dense_thresholded, dense_mask, dense_n = _relion_pass2_reconstruction_probs(
        probs,
        adaptive_fraction=1.0,
    )
    _pair_thresholded, pair_sig_mask, pair_n = _relion_pass2_reconstruction_pair_probs(
        pair_probs,
        pair_mask,
        adaptive_fraction=1.0,
    )

    np.testing.assert_array_equal(
        np.asarray(dense_mask),
        np.array([[[True, False, True, False], [False, True, False, False]]]),
    )
    np.testing.assert_array_equal(
        np.asarray(pair_sig_mask),
        np.array([[True, False, True, False, True, False]]),
    )
    np.testing.assert_array_equal(np.asarray(dense_n), np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(pair_n), np.array([3], dtype=np.int32))


def test_relion_joint_winner_take_all_masks_choose_one_global_class_pose():
    class0_scores = jnp.asarray(
        [
            [0.1, 0.2, 2.0],
            [5.0, 1.0, -jnp.inf],
            [-jnp.inf, -jnp.inf, -jnp.inf],
        ],
        dtype=jnp.float32,
    )
    class1_scores = jnp.asarray(
        [
            [3.0, 0.5],
            [4.0, 4.5],
            [-jnp.inf, -jnp.inf],
        ],
        dtype=jnp.float32,
    )

    mask0, mask1 = _relion_joint_winner_take_all_masks([class0_scores, class1_scores])

    np.testing.assert_array_equal(
        np.asarray(mask0),
        np.asarray(
            [
                [False, False, False],
                [True, False, False],
                [False, False, False],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(mask1),
        np.asarray(
            [
                [True, False],
                [False, False],
                [False, False],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(mask0).sum(axis=1) + np.asarray(mask1).sum(axis=1),
        np.asarray([1, 1, 0]),
    )


def test_relion_joint_mstep_prune_drops_weak_class_tail_that_per_class_keeps():
    class0 = jnp.asarray([[0.9995, 0.0]], dtype=jnp.float32)
    class1 = jnp.asarray([[0.0005, 0.0]], dtype=jnp.float32)
    pair_mask = jnp.ones_like(class0, dtype=bool)

    per_class0, _, _ = _relion_pass2_reconstruction_pair_probs(
        class0,
        pair_mask,
        adaptive_fraction=0.999,
    )
    per_class1, _, _ = _relion_pass2_reconstruction_pair_probs(
        class1,
        pair_mask,
        adaptive_fraction=0.999,
    )
    joint_mask0, joint_mask1 = _relion_pass2_reconstruction_joint_masks(
        [class0, class1],
        adaptive_fraction=0.999,
    )
    joint0 = jnp.where(joint_mask0, class0, 0.0)
    joint1 = jnp.where(joint_mask1, class1, 0.0)

    assert float(jnp.sum(per_class0)) == pytest.approx(0.9995)
    assert float(jnp.sum(per_class1)) == pytest.approx(0.0005)
    assert float(jnp.sum(joint0)) == pytest.approx(0.9995)
    assert float(jnp.sum(joint1)) == pytest.approx(0.0)


def test_relion_fine_mstep_prune_mode_override_beats_env(monkeypatch):
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE", "per_class")

    assert (
        _relion_fine_mstep_prune_mode(
            use_relion_x_half_mstep=False,
            mode_override="joint",
        )
        == "joint"
    )
    assert (
        _relion_fine_mstep_prune_mode(
            use_relion_x_half_mstep=False,
            mode_override="none",
        )
        == "none"
    )


def test_k_class_fused_prune_mode_allows_explicit_env_override(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE", raising=False)
    assert (
        _k_class_fused_relion_fine_mstep_prune_mode_override(relion_fine_mstep_prune=False)
        is None
    )
    assert (
        _k_class_fused_relion_fine_mstep_prune_mode_override(relion_fine_mstep_prune=True)
        == "joint"
    )

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE", "none")
    assert (
        _k_class_fused_relion_fine_mstep_prune_mode_override(relion_fine_mstep_prune=True)
        is None
    )


def test_weighted_image_power_shells_uses_per_image_support_mass():
    processed_half = jnp.asarray(
        [
            [1.0 + 1.0j, 2.0 + 0.0j, 3.0 + 0.0j],
            [4.0 + 0.0j, 0.0 + 5.0j, 6.0 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    support_mass = jnp.asarray([0.25, 1.5], dtype=jnp.float32)
    shell_indices = jnp.asarray([0, 1, 1], dtype=jnp.int32)

    shells, per_image = _weighted_image_power_shells_and_per_image(
        processed_half,
        shell_indices,
        support_mass,
        shell_count=2,
    )

    power = np.abs(np.asarray(processed_half)) ** 2
    expected_per_image = power.sum(axis=1) * np.asarray(support_mass)
    expected_shells = np.zeros(2, dtype=np.float32)
    expected_shells[0] = power[0, 0] * 0.25 + power[1, 0] * 1.5
    expected_shells[1] = (power[0, 1] + power[0, 2]) * 0.25 + (power[1, 1] + power[1, 2]) * 1.5

    np.testing.assert_allclose(np.asarray(per_image), expected_per_image, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(shells), expected_shells, rtol=1e-6, atol=1e-6)


def test_weighted_image_power_uses_unweighted_high_shells_for_noise_and_normcorr():
    processed_half = jnp.asarray(
        [
            [1.0 + 1.0j, 2.0 + 0.0j, 3.0 + 0.0j],
            [4.0 + 0.0j, 0.0 + 5.0j, 6.0 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    support_mass = jnp.asarray([0.25, 0.0], dtype=jnp.float32)
    shell_indices = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    shells, per_image = _weighted_image_power_shells_and_per_image(
        processed_half,
        shell_indices,
        support_mass,
        shell_count=3,
        norm_unweighted_shell_cutoff=1,
    )

    power = np.abs(np.asarray(processed_half)) ** 2
    expected_shells = np.zeros(3, dtype=np.float32)
    expected_shells[0] = power[0, 0] * 0.25
    expected_shells[1] = power[0, 1] * 0.25
    expected_shells[2] = power[0, 2] + power[1, 2]
    expected_per_image = np.array(
        [
            (power[0, 0] + power[0, 1]) * 0.25 + power[0, 2],
            power[1, 2],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(np.asarray(shells), expected_shells, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(per_image), expected_per_image, rtol=1e-6, atol=1e-6)


def test_weighted_image_power_replaces_only_unweighted_high_shell_normcorr():
    processed_half = jnp.asarray(
        [
            [1.0 + 1.0j, 2.0 + 0.0j, 3.0 + 0.0j],
            [4.0 + 0.0j, 0.0 + 5.0j, 6.0 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    support_mass = jnp.asarray([0.25, 0.0], dtype=jnp.float32)
    shell_indices = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    powerclass_high = jnp.asarray([123.0, 456.0], dtype=jnp.float32)

    shells, per_image = _weighted_image_power_shells_and_per_image(
        processed_half,
        shell_indices,
        support_mass,
        shell_count=3,
        norm_unweighted_shell_cutoff=1,
        norm_unweighted_high_shell=powerclass_high,
    )

    power = np.abs(np.asarray(processed_half)) ** 2
    expected_shells = np.zeros(3, dtype=np.float32)
    expected_shells[0] = power[0, 0] * 0.25
    expected_shells[1] = power[0, 1] * 0.25
    expected_shells[2] = power[0, 2] + power[1, 2]
    expected_per_image = np.asarray(
        [
            (power[0, 0] + power[0, 1]) * 0.25 + powerclass_high[0],
            powerclass_high[1],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(np.asarray(shells), expected_shells)
    np.testing.assert_array_equal(np.asarray(per_image), expected_per_image)


def test_weighted_image_power_assigns_shared_high_shell_once_across_classes():
    processed_half = jnp.asarray(
        [
            [1.0 + 1.0j, 2.0 + 0.0j, 3.0 + 0.0j],
            [4.0 + 0.0j, 0.0 + 5.0j, 6.0 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    shell_indices = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    support_mass_by_class = np.asarray(
        [[0.2, 0.7], [0.3, 0.1], [0.5, 0.2]],
        dtype=np.float32,
    )
    powerclass_high = jnp.asarray([123.0, 456.0], dtype=jnp.float32)

    per_class_norm = []
    per_class_shells = []
    for class_index, support_mass in enumerate(support_mass_by_class):
        shells, per_image = _weighted_image_power_shells_and_per_image(
            processed_half,
            shell_indices,
            jnp.asarray(support_mass),
            shell_count=3,
            norm_unweighted_shell_cutoff=0,
            norm_unweighted_high_shell=powerclass_high,
            include_unweighted_high_shell=class_index == 0,
        )
        per_class_shells.append(np.asarray(shells))
        per_class_norm.append(np.asarray(per_image))

    pixel_power = np.abs(np.asarray(processed_half)) ** 2
    total_mass = support_mass_by_class.sum(axis=0)
    expected_norm = pixel_power[:, 0] * total_mass + np.asarray(powerclass_high)
    expected_shells = np.asarray(
        [
            np.sum(pixel_power[:, 0] * total_mass),
            np.sum(pixel_power[:, 1]),
            np.sum(pixel_power[:, 2]),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.sum(per_class_norm, axis=0), expected_norm, rtol=1e-7, atol=5e-5)
    np.testing.assert_allclose(np.sum(per_class_shells, axis=0), expected_shells, rtol=1e-7, atol=5e-5)


def test_weighted_image_power_excludes_sentinel_from_support_weighted_normcorr():
    processed_half = jnp.asarray(
        [[1.0 + 0.0j, 100.0 + 0.0j], [2.0 + 0.0j, 200.0 + 0.0j]],
        dtype=jnp.complex64,
    )
    support_mass = jnp.asarray([0.25, 1.5], dtype=jnp.float32)
    shell_indices = jnp.asarray([0, 2], dtype=jnp.int32)

    shells, per_image = _weighted_image_power_shells_and_per_image(
        processed_half,
        shell_indices,
        support_mass,
        shell_count=2,
    )

    np.testing.assert_array_equal(np.asarray(shells), np.asarray([6.25, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(per_image), np.asarray([0.25, 6.0], dtype=np.float32))


def test_weighted_image_power_excludes_sentinel_from_normcorr_but_keeps_valid_outer_shell():
    processed_half = jnp.asarray(
        [
            [1.0 + 0.0j, 2.0 + 0.0j, 100.0 + 0.0j],
            [3.0 + 0.0j, 4.0 + 0.0j, 200.0 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    support_mass = jnp.asarray([0.25, 0.0], dtype=jnp.float32)
    # Shell 1 is a valid outer shell beyond the current model cutoff. Shell 2
    # is the drop-bin sentinel and must not enter either norm-correction sum.
    shell_indices = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    shells, per_image = _weighted_image_power_shells_and_per_image(
        processed_half,
        shell_indices,
        support_mass,
        shell_count=2,
        norm_unweighted_shell_cutoff=0,
    )

    np.testing.assert_array_equal(np.asarray(shells), np.asarray([0.25, 20.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(per_image), np.asarray([4.25, 16.0], dtype=np.float32))


def test_k1_relion_fine_mstep_prune_keeps_unweighted_high_shell_image_power(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")

    n_images = 4
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0)
    fine_parent = np.arange(16, dtype=np.int64)
    fine_translations = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32),
        np.asarray([0, 2, 4], dtype=np.int32),
        np.asarray([1, 3, 5], dtype=np.int32),
        np.asarray([0, 5], dtype=np.int32),
    ]

    common = dict(
        experiment_dataset=MockDataset(n_images=n_images, seed=901),
        volume=_hermitian_volume(VOLUME_SHAPE, seed=907),
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        translations=jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        significant_sample_indices=significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=2,
        return_stats=True,
        accumulate_noise=True,
        half_spectrum_scoring=True,
        group_ids=np.asarray([0, 1, 0, 1], dtype=np.int64),
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", str(1024**3))
    def prune_everything(probs, *, adaptive_fraction):
        del adaptive_fraction
        return jnp.zeros_like(probs), jnp.zeros(probs.shape[0], dtype=jnp.int32), jnp.zeros(probs.shape[0], dtype=jnp.int32)

    monkeypatch.setattr(bucketed_mod, "_relion_pass2_reconstruction_probs", prune_everything)
    pruned = compute_pass2_stats_sparse(
        **common,
        relion_fine_mstep_prune=True,
        adaptive_fraction=0.5,
    )
    pruned_noise = pruned[-1]

    assert pruned_noise.sumw == pytest.approx(0.0)
    assert pruned_noise.wsum_sigma2_offset == pytest.approx(0.0)
    np.testing.assert_allclose(np.asarray(pruned_noise.wsum_sigma2_noise), 0.0, rtol=0, atol=0)
    image_power = np.asarray(pruned_noise.wsum_img_power)
    np.testing.assert_allclose(image_power[:2], 0.0, rtol=0, atol=0)
    assert np.any(image_power[2:] > 0.0)
    assert np.any(np.asarray(pruned_noise.wsum_norm_correction) > 0.0)
    assert np.all(np.asarray(pruned_noise.wsum_norm_correction) >= 0.0)
    np.testing.assert_allclose(np.asarray(pruned_noise.wsum_scale_correction_xa), 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(pruned_noise.wsum_scale_correction_aa), 0.0, rtol=0, atol=0)


def test_compact_pair_score_only_normalization_ignores_padding():
    pair_scores = jnp.asarray(
        [
            [0.0, 2.0, 1000.0],
            [-5.0, 999.0, 1000.0],
            [1000.0, 1001.0, 1002.0],
        ],
        dtype=jnp.float32,
    )
    pair_mask = jnp.asarray(
        [
            [True, True, False],
            [True, False, False],
            [False, False, False],
        ],
        dtype=bool,
    )

    log_z, best_log_score, best_argmax, max_posterior = _normalize_pass2_pairs_score_only(
        pair_scores,
        pair_mask,
    )
    logsum = _logsumexp_pass2_pairs_score_only(pair_scores, pair_mask)

    np.testing.assert_allclose(np.asarray(log_z[:2]), np.asarray(logsum[:2]), rtol=1e-7, atol=1e-7)
    assert float(log_z[0]) == pytest.approx(float(np.logaddexp(0.0, 2.0)), abs=1e-6)
    assert float(log_z[1]) == pytest.approx(-5.0, abs=1e-6)
    assert float(log_z[2]) == pytest.approx(0.0, abs=0.0)
    assert np.isneginf(float(logsum[2]))
    np.testing.assert_allclose(np.asarray(best_log_score), np.asarray([2.0, -5.0, -np.inf]), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(best_argmax), np.asarray([1, 0, 0]))
    assert float(max_posterior[0]) == pytest.approx(float(np.exp(2.0 - np.logaddexp(0.0, 2.0))), abs=1e-6)
    assert float(max_posterior[1]) == pytest.approx(1.0, abs=1e-6)
    assert float(max_posterior[2]) == pytest.approx(0.0, abs=0.0)


def test_compact_pair_padding_cannot_be_selected_as_best():
    per_image_inputs = {
        "candidate_mask": [
            np.asarray(
                [
                    [False, True, False, False],
                    [False, False, False, True],
                ],
                dtype=bool,
            ),
        ],
        "oversampled_rot_indices": [np.asarray([21, 22], dtype=np.int64)],
        "log_prior": [np.asarray([0.5, 0.7], dtype=np.float32)],
    }
    compact = _prepare_per_image_compact_candidate_pairs(per_image_inputs)
    arrays = _build_compact_pair_bucket_arrays(
        {"pair_bucket_size": 4, "image_indices": np.asarray([0], dtype=np.int64)},
        compact,
    )

    scores = np.asarray([[5.0, 7.0, 1000.0, 2000.0]], dtype=np.float64)
    best = _best_compact_pair_from_scores(
        scores,
        arrays["pair_mask"],
        arrays["local_rotation_row"],
        arrays["translation_idx"],
        arrays["rotation_index"],
    )

    assert bool(best["has_valid"][0])
    assert int(best["pair_index"][0]) == 1
    assert int(best["local_rotation_row"][0]) == 1
    assert int(best["translation_idx"][0]) == 3
    assert int(best["rotation_index"][0]) == 22
    assert float(best["score"][0]) == pytest.approx(7.0)


def _make_late_iter_sparse_kclass_inputs(*, n_classes=4, n_images=8, n_rot=512, n_fine_trans=116):
    valid_pairs_per_image = 192
    per_image_inputs_by_class = []
    for class_index in range(n_classes):
        masks = []
        for image_idx in range(n_images):
            mask = np.zeros((n_rot, n_fine_trans), dtype=bool)
            flat = (np.arange(valid_pairs_per_image, dtype=np.int64) * 37 + image_idx * 11 + class_index * 17) % (
                n_rot * n_fine_trans
            )
            mask[flat // n_fine_trans, flat % n_fine_trans] = True
            masks.append(mask)
        per_image_inputs_by_class.append(
            {
                "oversampled_rots": [
                    np.broadcast_to(np.eye(3, dtype=np.float32), (n_rot, 3, 3)).copy()
                    for _ in range(n_images)
                ],
                "oversampled_rot_indices": [
                    np.arange(n_rot, dtype=np.int64) + class_index * 10_000
                    for _ in range(n_images)
                ],
                "log_prior": [
                    np.linspace(-1.0, 1.0, n_rot, dtype=np.float32)
                    for _ in range(n_images)
                ],
                "candidate_mask": masks,
            }
        )
    return per_image_inputs_by_class


def test_compact_pair_plan_reports_late_iter_candidate_reduction(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    n_classes = 4
    n_fine_trans = 116
    per_image_inputs_by_class = _make_late_iter_sparse_kclass_inputs(
        n_classes=n_classes,
        n_images=8,
        n_rot=512,
        n_fine_trans=n_fine_trans,
    )
    dense_buckets = _bucket_sparse_k_class_pass2_inputs(
        per_image_inputs_by_class,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    compact_inputs_by_class = [
        _prepare_per_image_compact_candidate_pairs(per_image_inputs)
        for per_image_inputs in per_image_inputs_by_class
    ]
    compact_buckets = _bucket_sparse_k_class_compact_pair_inputs(
        compact_inputs_by_class,
        max_pair_candidates_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    stats = _compact_k_class_pair_plan_stats(
        per_image_inputs_by_class,
        dense_buckets,
        n_fine_trans,
        max_pair_candidates_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )

    assert dense_buckets
    assert compact_buckets
    assert stats.valid_pair_candidates == n_classes * 8 * 192
    assert stats.padded_pair_candidates < stats.rectangular_candidates / 20
    assert stats.reduction_factor > 20
    assert stats.padded_reduction_factor > 20
    assert stats.median_valid_pairs_per_image == 192


def test_compact_pair_chunk_cap_defaults_to_dense_cap_unless_explicit(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", raising=False)
    assert _compact_pair_max_images_per_microbatch_for_pass(19) == 19

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "128")
    assert _compact_pair_max_images_per_microbatch_for_pass(19) == 128


def test_compact_pair_prepare_cap_can_tighten_but_not_raise_dense_cap():
    assert (
        _compact_pair_prepare_max_images_per_microbatch(
            dense_max_images_per_microbatch=993,
            compact_pair_max_images_per_microbatch=512,
        )
        == 512
    )
    assert (
        _compact_pair_prepare_max_images_per_microbatch(
            dense_max_images_per_microbatch=19,
            compact_pair_max_images_per_microbatch=128,
        )
        == 19
    )


def test_compact_pair_bucketing_can_coalesce_high_pair_tail(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    counts = [4096] * 20 + [8192] * 2 + [12288] * 3 + [16384] * 2
    compact_inputs_by_class = tuple(
        {"pair_counts": np.asarray(counts, dtype=np.int64)}
        for _ in range(4)
    )

    baseline = _bucket_sparse_k_class_compact_pair_inputs(
        compact_inputs_by_class,
        max_pair_candidates_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    coalesced = _bucket_sparse_k_class_compact_pair_inputs(
        compact_inputs_by_class,
        max_pair_candidates_per_microbatch=10**12,
        max_images_per_microbatch=1000,
        tail_bucket_coalesce_max_images=8,
        tail_bucket_coalesce_max_inflation=2.0,
        tail_bucket_coalesce_min_bucket_size=4096,
    )

    assert sorted({int(bucket["pair_bucket_size"]) for bucket in baseline}) == [
        4096,
        8192,
        12288,
        16384,
    ]
    assert sorted({int(bucket["pair_bucket_size"]) for bucket in coalesced}) == [
        4096,
        16384,
    ]
    assert len(coalesced) < len(baseline)
    assert sum(len(bucket["image_indices"]) for bucket in coalesced) == len(counts)


def test_compact_pair_tail_coalescing_keeps_executed_chunks_under_hypothesis_cap(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    counts = [
        282_624,
        286_720,
        290_816,
        294_912,
        299_008,
        303_104,
        307_200,
        311_296,
        315_392,
        323_584,
        331_776,
        339_968,
        348_160,
        356_352,
        417_792,
    ]
    n_classes = 4
    max_hypotheses = 10_045_744
    compact_inputs_by_class = tuple(
        {"pair_counts": np.asarray(counts, dtype=np.int64)}
        for _ in range(n_classes)
    )

    baseline = _bucket_sparse_k_class_compact_pair_inputs(
        compact_inputs_by_class,
        max_pair_candidates_per_microbatch=max_hypotheses,
        max_images_per_microbatch=19,
    )
    coalesced = _bucket_sparse_k_class_compact_pair_inputs(
        compact_inputs_by_class,
        max_pair_candidates_per_microbatch=max_hypotheses,
        max_images_per_microbatch=19,
        tail_bucket_coalesce_max_images=19,
        tail_bucket_coalesce_max_inflation=2.0,
        tail_bucket_coalesce_min_bucket_size=4096,
    )

    assert len(coalesced) < len(baseline)
    assert sum(len(bucket["image_indices"]) for bucket in coalesced) == len(counts)
    assert len({int(bucket["pair_bucket_size"]) for bucket in coalesced}) < len(
        {int(bucket["pair_bucket_size"]) for bucket in baseline}
    )
    for bucket in coalesced:
        n_images = len(bucket["image_indices"])
        pair_bucket_size = int(bucket["pair_bucket_size"])
        assert n_classes * n_images * pair_bucket_size <= max_hypotheses


def test_compact_pair_tail_coalescing_preserves_masked_image_partition(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    counts = np.asarray(
        [128, 4096, 8192, 12288, 16384, 256, 20480, 24576, 32768, 512],
        dtype=np.int64,
    )
    n_classes = 4
    pair_counts_by_class = tuple(counts + class_idx for class_idx in range(n_classes))
    image_mask = counts >= 4096
    max_hypotheses = 1_000_000

    baseline = _bucket_sparse_k_class_compact_pair_counts(
        pair_counts_by_class,
        max_pair_candidates_per_microbatch=max_hypotheses,
        max_images_per_microbatch=19,
        image_mask=image_mask,
    )
    coalesced = _bucket_sparse_k_class_compact_pair_counts(
        pair_counts_by_class,
        max_pair_candidates_per_microbatch=max_hypotheses,
        max_images_per_microbatch=19,
        image_mask=image_mask,
        tail_bucket_coalesce_max_images=19,
        tail_bucket_coalesce_max_inflation=2.0,
        tail_bucket_coalesce_min_bucket_size=4096,
    )

    def bucket_size_by_image(buckets):
        result = {}
        for bucket in buckets:
            pair_bucket_size = int(bucket["pair_bucket_size"])
            for image_idx in np.asarray(bucket["image_indices"], dtype=np.int64):
                assert int(image_idx) not in result
                result[int(image_idx)] = pair_bucket_size
            assert n_classes * len(bucket["image_indices"]) * pair_bucket_size <= max_hypotheses
        return result

    baseline_by_image = bucket_size_by_image(baseline)
    coalesced_by_image = bucket_size_by_image(coalesced)
    selected_images = set(np.nonzero(image_mask)[0].astype(int).tolist())
    assert set(baseline_by_image) == selected_images
    assert set(coalesced_by_image) == selected_images
    assert any(coalesced_by_image[idx] > baseline_by_image[idx] for idx in selected_images)
    for idx in selected_images:
        assert coalesced_by_image[idx] >= baseline_by_image[idx]
        assert coalesced_by_image[idx] >= max(int(pair_counts[idx]) for pair_counts in pair_counts_by_class)


def test_compact_pair_planner_can_decouple_from_dense_image_cap(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", "1")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", raising=False)

    n_classes = 4
    n_images = 64
    n_fine_trans = 16
    per_image_inputs_by_class = _make_late_iter_sparse_kclass_inputs(
        n_classes=n_classes,
        n_images=n_images,
        n_rot=64,
        n_fine_trans=n_fine_trans,
    )
    dense_buckets = _bucket_sparse_k_class_pass2_inputs(
        per_image_inputs_by_class,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=19,
    )

    dense_capped = _maybe_prepare_sparse_k_class_compact_pair_plan(
        per_image_inputs_by_class,
        dense_buckets,
        n_fine_trans,
        max_pair_candidates_per_microbatch=10**12,
        max_images_per_microbatch=19,
    )
    assert dense_capped is not None
    assert dense_capped.max_images_per_microbatch == 19
    assert max(len(bucket["image_indices"]) for bucket in dense_capped.buckets) == 19
    assert len(dense_capped.buckets) > 1

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "64")
    decoupled = _maybe_prepare_sparse_k_class_compact_pair_plan(
        per_image_inputs_by_class,
        dense_buckets,
        n_fine_trans,
        max_pair_candidates_per_microbatch=10**12,
        max_images_per_microbatch=19,
    )
    assert decoupled is not None
    assert decoupled.max_images_per_microbatch == 64
    assert max(len(bucket["image_indices"]) for bucket in decoupled.buckets) == n_images
    assert len(decoupled.buckets) < len(dense_capped.buckets)
    assert decoupled.valid_pair_candidates == dense_capped.valid_pair_candidates
    assert decoupled.padded_pair_candidates == dense_capped.padded_pair_candidates

    execution_buckets = _split_compact_pair_buckets_by_projection_gather_budget(
        decoupled.buckets,
        per_image_inputs_by_class,
        n_score_pixels=17,
        n_recon_pixels=19,
        projection_complex_dtype=np.complex64,
        max_gather_bytes=10**18,
        max_prepare_images_per_microbatch=19,
        rotation_block_size_for_quantization=5000,
    )
    assert len(execution_buckets) > len(decoupled.buckets)
    assert max(len(bucket["image_indices"]) for bucket in execution_buckets) == 19
    np.testing.assert_array_equal(
        np.sort(np.concatenate([bucket["image_indices"] for bucket in execution_buckets])),
        np.arange(n_images, dtype=np.int64),
    )


def test_compact_pair_execution_split_honors_dense_mstep_budget(monkeypatch):
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "64")

    n_classes = 4
    n_images = 64
    n_fine_trans = 16
    per_image_inputs_by_class = _make_late_iter_sparse_kclass_inputs(
        n_classes=n_classes,
        n_images=n_images,
        n_rot=64,
        n_fine_trans=n_fine_trans,
    )
    dense_buckets = _bucket_sparse_k_class_pass2_inputs(
        per_image_inputs_by_class,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=19,
    )
    compact_plan = _maybe_prepare_sparse_k_class_compact_pair_plan(
        per_image_inputs_by_class,
        dense_buckets,
        n_fine_trans,
        max_pair_candidates_per_microbatch=10**12,
        max_images_per_microbatch=19,
    )
    assert compact_plan is not None
    assert max(len(bucket["image_indices"]) for bucket in compact_plan.buckets) == n_images

    execution_buckets = _split_compact_pair_buckets_by_projection_gather_budget(
        compact_plan.buckets,
        per_image_inputs_by_class,
        n_score_pixels=17,
        n_recon_pixels=19,
        projection_complex_dtype=np.complex64,
        max_gather_bytes=None,
        max_dense_mstep_bytes=1,
        n_fine_trans=n_fine_trans,
        prob_dtype=np.float64,
        max_prepare_images_per_microbatch=None,
        rotation_block_size_for_quantization=5000,
    )

    assert len(execution_buckets) > len(compact_plan.buckets)
    assert max(len(bucket["image_indices"]) for bucket in execution_buckets) == 1
    np.testing.assert_array_equal(
        np.sort(np.concatenate([bucket["image_indices"] for bucket in execution_buckets])),
        np.arange(n_images, dtype=np.int64),
    )


def test_compact_pair_dense_mstep_budget_env_override(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DENSE_MSTEP_MAX_BYTES", raising=False)
    assert _compact_pair_dense_mstep_max_bytes_for_pass(None) > 0

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DENSE_MSTEP_MAX_BYTES", "12345")
    assert _compact_pair_dense_mstep_max_bytes_for_pass(None) == 12345


def test_compact_pair_execution_defaults_to_high_bucket_hybrid(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", raising=False)

    assert _compact_pair_execution_enabled_for_pass() is True
    assert _compact_pair_min_bucket_size_for_pass() == 512
    assert _compact_pair_min_bucket_size_for_pass(1) == 1

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "0")
    assert _compact_pair_execution_enabled_for_pass() is False

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1024")
    assert _compact_pair_execution_enabled_for_pass() is True
    assert _compact_pair_min_bucket_size_for_pass() == 1024
    assert _compact_pair_min_bucket_size_for_pass(1) == 1024


def test_compact_pair_execution_treats_blank_env_flags_as_unset(monkeypatch):
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", " ")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", "")

    assert _compact_pair_execution_enabled_for_pass() is True


def test_compact_pair_check_mode_disables_auto_execution(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", "1")

    assert _compact_pair_execution_enabled_for_pass() is False

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    assert _compact_pair_execution_enabled_for_pass() is True


def test_compact_pair_planner_is_opt_in_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", raising=False)
    per_image_inputs_by_class = _make_late_iter_sparse_kclass_inputs(
        n_classes=2,
        n_images=2,
        n_rot=64,
        n_fine_trans=16,
    )
    dense_buckets = _bucket_sparse_k_class_pass2_inputs(
        per_image_inputs_by_class,
        n_fine_trans=16,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    assert (
        _maybe_prepare_sparse_k_class_compact_pair_plan(
            per_image_inputs_by_class,
            dense_buckets,
            16,
            max_pair_candidates_per_microbatch=10**12,
            max_images_per_microbatch=1000,
        )
        is None
    )

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", "1")
    stats = _maybe_prepare_sparse_k_class_compact_pair_plan(
        per_image_inputs_by_class,
        dense_buckets,
        16,
        max_pair_candidates_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    assert stats is not None
    assert stats.valid_pair_candidates > 0
    assert stats.rectangular_candidates > stats.valid_pair_candidates

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", "0")
    assert (
        _maybe_prepare_sparse_k_class_compact_pair_plan(
            per_image_inputs_by_class,
            dense_buckets,
            16,
            max_pair_candidates_per_microbatch=10**12,
            max_images_per_microbatch=1000,
        )
        is None
    )


def test_sparse_pass2_projection_cap_chunks_projection_calls(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    calls = []

    def fake_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type
        calls.append(int(rotations_block.shape[0]))
        n_half = 5
        rotation_ids = jnp.asarray(rotations_block[:, 0, 0], dtype=jnp.float32)
        proj = rotation_ids[:, None] + jnp.arange(n_half, dtype=jnp.float32)[None, :]
        proj = proj.astype(jnp.complex64)
        return_abs2 = kwargs.get("return_abs2", None)
        return proj, None if return_abs2 is False else jnp.abs(proj) ** 2

    monkeypatch.setattr(bucketed_mod, "_compute_projections_block", fake_project)
    rotations = np.zeros((10, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.arange(10, dtype=np.float32)

    proj, abs2 = _compute_sparse_pass2_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        max_projected_rotations=4,
        return_abs2=False,
    )

    assert calls == [4, 4, 2]
    assert abs2 is None
    np.testing.assert_array_equal(np.asarray(proj[:, 0].real), np.arange(10, dtype=np.float32))


def test_sparse_pass2_windowed_projection_cap_keeps_only_requested_pixels(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    calls = []

    def fake_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type
        calls.append(int(rotations_block.shape[0]))
        n_half = 6
        rotation_ids = jnp.asarray(rotations_block[:, 0, 0], dtype=jnp.float32)
        proj = rotation_ids[:, None] * 10.0 + jnp.arange(n_half, dtype=jnp.float32)[None, :]
        return proj.astype(jnp.complex64), None

    monkeypatch.setattr(bucketed_mod, "_compute_projections_block", fake_project)
    rotations = np.zeros((7, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.arange(7, dtype=np.float32)

    score, recon, recon_abs2 = _compute_sparse_pass2_windowed_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        score_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        recon_indices=jnp.asarray([1, 5], dtype=jnp.int32),
        max_projected_rotations=3,
    )

    assert calls == [3, 3, 1]
    assert score.shape == (7, 2)
    assert recon.shape == (7, 2)
    assert recon_abs2.shape == (7, 2)
    np.testing.assert_array_equal(np.asarray(score[4].real), np.asarray([40.0, 42.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(recon[4].real), np.asarray([41.0, 45.0], dtype=np.float32))


def test_sparse_pass2_windowed_projection_uses_relion_projector_branch(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    calls = []
    relion_projector_half = jnp.ones((4, 4, 3), dtype=jnp.complex64)

    def fake_relion_projector(
        volume_relion_half,
        rotations_block,
        image_shape,
        *,
        r_max,
        padding_factor,
        return_abs2,
        centered_rows,
        dense_scale,
        relion_texture_interp,
        projector_output_size=None,
        mask_current_image_disk=True,
    ):
        del projector_output_size
        calls.append(
            {
                "n_rot": int(rotations_block.shape[0]),
                "image_shape": tuple(image_shape),
                "r_max": int(r_max),
                "padding_factor": int(padding_factor),
                "return_abs2": bool(return_abs2),
                "centered_rows": bool(centered_rows),
                "dense_scale": bool(dense_scale),
                "relion_texture_interp": relion_texture_interp,
                "mask_current_image_disk": bool(mask_current_image_disk),
                "projector_shape": tuple(volume_relion_half.shape),
            }
        )
        n_half = 6
        rotation_ids = jnp.asarray(rotations_block[:, 0, 0], dtype=jnp.float32)
        proj = rotation_ids[:, None] * 10.0 + jnp.arange(n_half, dtype=jnp.float32)[None, :]
        proj = proj.astype(jnp.complex64)
        return proj, None if not return_abs2 else jnp.abs(proj) ** 2

    monkeypatch.setattr(bucketed_mod, "_compute_relion_projector_projections_block", fake_relion_projector)
    rotations = np.zeros((7, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.arange(7, dtype=np.float32)

    score, recon, recon_abs2 = _compute_sparse_pass2_windowed_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        score_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        recon_indices=jnp.asarray([1, 5], dtype=jnp.int32),
        max_projected_rotations=3,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=3,
        projection_padding_factor=2,
        relion_texture_interp=False,
        mask_current_image_disk=False,
    )

    assert [call["n_rot"] for call in calls] == [3, 3, 1]
    for call in calls:
        assert call == {
            "n_rot": call["n_rot"],
            "image_shape": IMAGE_SHAPE,
            "r_max": 3,
            "padding_factor": 2,
            "return_abs2": False,
            "centered_rows": True,
            "dense_scale": True,
            "relion_texture_interp": False,
            "mask_current_image_disk": False,
            "projector_shape": (4, 4, 3),
        }
    assert score.shape == (7, 2)
    assert recon.shape == (7, 2)
    assert recon_abs2.shape == (7, 2)
    np.testing.assert_array_equal(np.asarray(score[4].real), np.asarray([40.0, 42.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(recon[4].real), np.asarray([41.0, 45.0], dtype=np.float32))


def test_sparse_pass2_windowed_projection_cap_casts_chunks_before_concat(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    def fake_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type, kwargs
        n_half = 6
        rotation_ids = jnp.asarray(rotations_block[:, 0, 0], dtype=jnp.float64)
        proj = rotation_ids[:, None] * 10.0 + jnp.arange(n_half, dtype=jnp.float64)[None, :]
        return proj.astype(jnp.complex128), None

    monkeypatch.setattr(bucketed_mod, "_compute_projections_block", fake_project)
    rotations = np.zeros((7, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.arange(7, dtype=np.float32)

    score, recon, recon_abs2 = _compute_sparse_pass2_windowed_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex128),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        score_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        recon_indices=jnp.asarray([1, 5], dtype=jnp.int32),
        max_projected_rotations=3,
        output_complex_dtype=jnp.complex64,
        output_abs2_dtype=jnp.float32,
    )

    assert score.dtype == jnp.complex64
    assert recon.dtype == jnp.complex64
    assert recon_abs2.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(recon[4].real), np.asarray([41.0, 45.0], dtype=np.float32))


def test_prepare_bucket_io_windowed_shifted_matches_full_half_slice(monkeypatch):
    """Opt-in windowed prepare must match full-half prepare followed by gather."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    ds = MockDataset(n_images=4, seed=612)
    batch_indices = np.asarray([0, 1, 2], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    ctf_params = jnp.asarray(ds.CTF_params[batch_indices])
    config = ForwardModelConfig.from_dataset(
        ds,
        disc_type="linear_interp",
        process_fn=ds.process_images,
    )
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    window_spec = make_fourier_window_spec(
        IMAGE_SHAPE,
        current_size=6,
        n_half=n_half,
        include_recon_window=True,
    )
    assert window_spec.use_window
    fine_translations = jnp.asarray(
        [
            [0.0, 0.0],
            [0.25, -0.5],
            [1.0, 0.75],
        ],
        dtype=jnp.float32,
    )
    noise_variance_half = jnp.linspace(0.8, 1.4, n_half, dtype=jnp.float32)
    image_corrections = np.asarray([1.0, 0.93, 1.11, 1.07], dtype=np.float32)
    scale_corrections = np.asarray([1.0, 1.08, 0.91, 1.03], dtype=np.float32)

    common_kwargs = dict(
        experiment_dataset=ds,
        batch=batch,
        ctf_params=ctf_params,
        image_indices=batch_indices,
        noise_variance_half=noise_variance_half,
        fine_translations=fine_translations,
        config=config,
        n_trans=int(fine_translations.shape[0]),
        score_with_masked_images=True,
        half_spectrum_scoring=True,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=None,
        use_float64_scoring=False,
        return_direct_scoring_io=True,
        score_only=False,
        score_mode="gaussian",
        window_indices=window_spec.score_indices,
        recon_window_indices=window_spec.recon_indices,
    )
    full = _prepare_bucket_io(**common_kwargs, return_windowed_shifted=False)

    def forbid_full_phase_table(*_args, **_kwargs):
        raise AssertionError("windowed prepare should not build a full half-spectrum phase table")

    monkeypatch.setattr(bucketed_mod, "half_translation_phase_table", forbid_full_phase_table)
    windowed = _prepare_bucket_io(**common_kwargs, return_windowed_shifted=True)

    score_indices = np.asarray(window_spec.score_indices_np, dtype=np.int32)
    recon_indices = np.asarray(window_spec.recon_indices_np, dtype=np.int32)
    np.testing.assert_allclose(
        np.asarray(windowed[0]),
        np.asarray(full[0])[:, score_indices],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(windowed[1]),
        np.asarray(full[1])[:, recon_indices],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(windowed[2]), np.asarray(full[2]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(windowed[3]),
        np.asarray(full[3])[:, score_indices],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(windowed[4]),
        np.asarray(full[4])[:, recon_indices],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(windowed[5]),
        np.asarray(full[5])[:, recon_indices],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(windowed[6]), np.asarray(full[6]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(windowed[7]),
        np.asarray(full[7])[:, score_indices],
        rtol=1e-6,
        atol=1e-6,
    )

    windowed_without_shifted_score = _prepare_bucket_io(
        **common_kwargs,
        return_windowed_shifted=True,
        return_shifted_score=False,
    )
    assert windowed_without_shifted_score[0] is None
    for idx in (1, 2, 3, 4, 5, 6, 7):
        np.testing.assert_allclose(
            np.asarray(windowed_without_shifted_score[idx]),
            np.asarray(windowed[idx]),
            rtol=1e-6,
            atol=1e-6,
        )

    score_phase = _half_translation_phase_table_for_indices(
        fine_translations,
        IMAGE_SHAPE,
        window_spec.score_indices,
    )
    recon_phase = _half_translation_phase_table_for_indices(
        fine_translations,
        IMAGE_SHAPE,
        window_spec.recon_indices,
    )

    def forbid_windowed_phase_table(*_args, **_kwargs):
        raise AssertionError("precomputed windowed phases should be reused")

    monkeypatch.setattr(bucketed_mod, "_half_translation_phase_table_for_indices", forbid_windowed_phase_table)
    precomputed = _prepare_bucket_io(
        **common_kwargs,
        score_translation_phases=score_phase,
        recon_translation_phases=recon_phase,
        return_windowed_shifted=True,
    )
    for actual, expected in zip(precomputed, windowed, strict=True):
        if actual is None and expected is None:
            continue
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_prepare_bucket_io_routes_direct_score_translation_through_relion_cuda(
    monkeypatch,
):
    from recovar import cuda_backproject

    ds = MockDataset(n_images=3, seed=20260727)
    batch_indices = np.asarray([0, 2], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    config = ForwardModelConfig.from_dataset(
        ds,
        disc_type="linear_interp",
        process_fn=ds.process_images,
    )
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    window_spec = make_fourier_window_spec(
        IMAGE_SHAPE,
        current_size=6,
        n_half=n_half,
        include_recon_window=True,
    )
    fine_translations = np.asarray(
        [[-0.75, 0.25], [0.25, -0.75]],
        dtype=np.float32,
    )
    translation_angles = jnp.asarray(
        _relion_translation_angles_f32(fine_translations, IMAGE_SHAPE),
        dtype=jnp.float32,
    )
    calls = []

    def fake_translate(images, angles, pixel_indices, image_shape):
        calls.append(
            {
                "images": np.asarray(images),
                "angles": np.asarray(angles),
                "pixel_indices": np.asarray(pixel_indices),
                "image_shape": image_shape,
            }
        )
        return jnp.full(
            (images.shape[0] * angles.shape[0], images.shape[1]),
            jnp.complex64(7.0 + 3.0j),
        )

    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f32",
        fake_translate,
    )
    result = _prepare_bucket_io(
        experiment_dataset=ds,
        batch=batch,
        ctf_params=jnp.asarray(ds.CTF_params[batch_indices]),
        image_indices=batch_indices,
        noise_variance_half=jnp.ones(n_half, dtype=jnp.float32),
        fine_translations=fine_translations,
        config=config,
        n_trans=fine_translations.shape[0],
        score_with_masked_images=True,
        half_spectrum_scoring=True,
        image_corrections=np.ones(ds.n_units, dtype=np.float32),
        scale_corrections=np.ones(ds.n_units, dtype=np.float32),
        image_pre_shifts=None,
        use_float64_scoring=False,
        return_direct_scoring_io=True,
        score_mode="gaussian",
        window_indices=window_spec.score_indices,
        recon_window_indices=window_spec.recon_indices,
        relion_score_translation_angles=translation_angles,
        return_windowed_shifted=True,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0]["pixel_indices"],
        np.asarray(window_spec.score_indices, dtype=np.int32),
    )
    np.testing.assert_array_equal(calls[0]["angles"], np.asarray(translation_angles))
    assert calls[0]["image_shape"] == IMAGE_SHAPE
    np.testing.assert_array_equal(
        np.asarray(result[7]),
        np.full(
            (
                batch_indices.size * fine_translations.shape[0],
                window_spec.score_indices.shape[0],
            ),
            np.complex64(7.0 + 3.0j),
        ),
    )


def test_prepare_bucket_io_exact_cc_keeps_relion_image_and_corr_operands_separate():
    ds = MockDataset(n_images=2, seed=20260809)
    batch_indices = np.asarray([0, 1], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    config = ForwardModelConfig.from_dataset(
        ds,
        disc_type="linear_interp",
        process_fn=ds.process_images,
    )
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    common = dict(
        experiment_dataset=ds,
        batch=batch,
        ctf_params=jnp.asarray(ds.CTF_params[batch_indices]),
        image_indices=batch_indices,
        noise_variance_half=jnp.ones(n_half, dtype=jnp.float32),
        fine_translations=jnp.zeros((1, 2), dtype=jnp.float32),
        config=config,
        n_trans=1,
        score_with_masked_images=False,
        half_spectrum_scoring=True,
        image_corrections=None,
        scale_corrections=None,
        image_pre_shifts=None,
        use_float64_scoring=False,
        return_direct_scoring_io=True,
        score_mode="normalized_cc",
    )

    folded = _prepare_bucket_io(**common)
    exact = _prepare_bucket_io(
        **common,
        relion_exact_normalized_cc_operands=True,
    )
    folded_image = np.asarray(folded[7]).reshape(batch_indices.size, 1, n_half)
    corrected_image = np.asarray(exact[7]).reshape(batch_indices.size, 1, n_half)
    corr_image = np.asarray(exact[3])[:, None, :]

    assert not np.array_equal(corrected_image, folded_image)
    np.testing.assert_allclose(
        corrected_image * corr_image,
        folded_image,
        rtol=2e-5,
        atol=2e-6,
    )


def test_prepare_bucket_io_routes_relion_cuda_operands_to_score_and_reconstruction():
    ds = MockDataset(n_images=2, seed=714)

    class _Backend:
        image_mask_mode = "relion_background_fill"
        relion_fourier_backend = "relion_cuda"

    ds.image_source.backend = _Backend()
    calls = []

    def capture_process(batch, apply_image_mask=False, **kwargs):
        calls.append((np.asarray(batch), apply_image_mask, kwargs))
        processed = _raw_real_process_half(batch, apply_image_mask=apply_image_mask)
        factors = jnp.asarray(kwargs["relion_normalization_factors"], dtype=processed.real.dtype)
        return processed * factors[:, None]

    ds.process_images_half = capture_process
    batch_indices = np.asarray([0, 1], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    config = ForwardModelConfig.from_dataset(ds, disc_type="linear_interp", process_fn=ds.process_images)
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    image_corrections = np.asarray([0.8, 1.5], dtype=np.float32)
    scale_corrections = np.asarray([2.0, 0.5], dtype=np.float32)
    image_pre_shifts = np.asarray([[1.0, -1.0], [-2.0, 1.0]], dtype=np.float32)

    result = _prepare_bucket_io(
        ds,
        batch,
        jnp.asarray(ds.CTF_params[batch_indices]),
        batch_indices,
        jnp.ones(n_half, dtype=jnp.float32),
        jnp.zeros((1, 2), dtype=jnp.float32),
        config,
        1,
        True,
        False,
        image_corrections,
        scale_corrections,
        image_pre_shifts,
        False,
    )

    assert [call[1] for call in calls] == [True, False]
    for raw_batch, _, kwargs in calls:
        np.testing.assert_array_equal(raw_batch, ds._images)
        np.testing.assert_array_equal(
            kwargs["relion_normalization_factors"],
            image_corrections / scale_corrections,
        )
        np.testing.assert_array_equal(kwargs["relion_integer_shifts"], image_pre_shifts.astype(np.int32))

    expected_recon = _raw_real_process_half(batch) * jnp.asarray(image_corrections)[:, None]
    np.testing.assert_allclose(np.asarray(result[1]), np.asarray(expected_recon), rtol=1e-6, atol=1e-6)


def test_prepare_bucket_io_windowed_reuses_unmasked_recon_shift_for_noise(monkeypatch):
    """Unmasked windowed prepare can reuse the recon-window shifted image."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    ds = MockDataset(n_images=4, seed=613)
    batch_indices = np.asarray([0, 1, 2], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    ctf_params = jnp.asarray(ds.CTF_params[batch_indices])
    config = ForwardModelConfig.from_dataset(
        ds,
        disc_type="linear_interp",
        process_fn=ds.process_images,
    )
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    window_spec = make_fourier_window_spec(
        IMAGE_SHAPE,
        current_size=6,
        n_half=n_half,
        include_recon_window=True,
    )
    assert window_spec.n_score != window_spec.n_recon
    fine_translations = jnp.asarray(
        [
            [0.0, 0.0],
            [0.25, -0.5],
            [1.0, 0.75],
        ],
        dtype=jnp.float32,
    )
    common_kwargs = dict(
        experiment_dataset=ds,
        batch=batch,
        ctf_params=ctf_params,
        image_indices=batch_indices,
        noise_variance_half=jnp.linspace(0.8, 1.4, n_half, dtype=jnp.float32),
        fine_translations=fine_translations,
        config=config,
        n_trans=int(fine_translations.shape[0]),
        half_spectrum_scoring=True,
        image_corrections=np.asarray([1.0, 0.93, 1.11, 1.07], dtype=np.float32),
        scale_corrections=np.asarray([1.0, 1.08, 0.91, 1.03], dtype=np.float32),
        image_pre_shifts=None,
        use_float64_scoring=False,
        return_direct_scoring_io=True,
        score_only=False,
        score_mode="gaussian",
        window_indices=window_spec.score_indices,
        recon_window_indices=window_spec.recon_indices,
        return_windowed_shifted=True,
    )
    original_apply = bucketed_mod.apply_half_translation_phases
    call_shapes = []

    def counting_apply(images, phases):
        call_shapes.append((tuple(images.shape), tuple(phases.shape)))
        return original_apply(images, phases)

    monkeypatch.setattr(bucketed_mod, "apply_half_translation_phases", counting_apply)
    unmasked = _prepare_bucket_io(**common_kwargs, score_with_masked_images=False)
    assert len(call_shapes) == 3
    np.testing.assert_allclose(np.asarray(unmasked[5]), np.asarray(unmasked[1]), rtol=0, atol=0)

    call_shapes.clear()
    _prepare_bucket_io(**common_kwargs, score_with_masked_images=True)
    assert len(call_shapes) == 4


@pytest.mark.parametrize(
    ("score_mode", "score_only"),
    [
        ("gaussian", True),
        ("normalized_cc", False),
    ],
)
def test_prepare_bucket_io_windowed_shifted_score_modes_match_full_half_slice(
    score_mode,
    score_only,
    monkeypatch,
):
    """Windowed prepare must preserve score-only and normalized-CC modes."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    ds = MockDataset(n_images=4, seed=713)
    batch_indices = np.asarray([0, 2, 3], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    ctf_params = jnp.asarray(ds.CTF_params[batch_indices])
    config = ForwardModelConfig.from_dataset(
        ds,
        disc_type="linear_interp",
        process_fn=ds.process_images,
    )
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    window_spec = make_fourier_window_spec(
        IMAGE_SHAPE,
        current_size=6,
        n_half=n_half,
        include_recon_window=True,
    )
    fine_translations = jnp.asarray(
        [
            [0.0, 0.0],
            [-0.5, 0.25],
        ],
        dtype=jnp.float32,
    )
    image_pre_shifts = np.asarray(
        [
            [0.0, 0.0],
            [0.25, -0.5],
            [-0.75, 0.5],
            [0.5, 0.25],
        ],
        dtype=np.float32,
    )
    common_kwargs = dict(
        experiment_dataset=ds,
        batch=batch,
        ctf_params=ctf_params,
        image_indices=batch_indices,
        noise_variance_half=jnp.linspace(0.9, 1.6, n_half, dtype=jnp.float32),
        fine_translations=fine_translations,
        config=config,
        n_trans=int(fine_translations.shape[0]),
        score_with_masked_images=False,
        half_spectrum_scoring=True,
        image_corrections=np.asarray([1.0, 0.97, 1.05, 1.12], dtype=np.float32),
        scale_corrections=np.asarray([1.0, 1.02, 0.94, 1.07], dtype=np.float32),
        image_pre_shifts=image_pre_shifts,
        use_float64_scoring=False,
        return_direct_scoring_io=True,
        score_only=score_only,
        score_mode=score_mode,
        window_indices=window_spec.score_indices,
        recon_window_indices=window_spec.recon_indices,
    )
    full = _prepare_bucket_io(**common_kwargs, return_windowed_shifted=False)

    def forbid_full_phase_table(*_args, **_kwargs):
        raise AssertionError("windowed prepare should not build a full half-spectrum phase table")

    monkeypatch.setattr(bucketed_mod, "half_translation_phase_table", forbid_full_phase_table)
    windowed = _prepare_bucket_io(**common_kwargs, return_windowed_shifted=True)

    score_indices = np.asarray(window_spec.score_indices_np, dtype=np.int32)
    recon_indices = np.asarray(window_spec.recon_indices_np, dtype=np.int32)
    if score_only:
        assert windowed[0] is None
        assert windowed[1] is None
        assert windowed[4] is None
        assert windowed[5] is None
        np.testing.assert_allclose(np.asarray(windowed[6]), np.asarray(full[6]), rtol=0, atol=0)
    else:
        np.testing.assert_allclose(
            np.asarray(windowed[0]),
            np.asarray(full[0])[:, score_indices],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(windowed[1]),
            np.asarray(full[1])[:, recon_indices],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(windowed[5]),
            np.asarray(full[5])[:, recon_indices],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(np.asarray(windowed[6]), np.asarray(full[6]), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(windowed[4]),
            np.asarray(full[4])[:, recon_indices],
            rtol=1e-6,
            atol=1e-6,
        )
        windowed_without_shifted_score = _prepare_bucket_io(
            **common_kwargs,
            return_windowed_shifted=True,
            return_shifted_score=False,
        )
        assert windowed_without_shifted_score[0] is None
        for idx in (1, 2, 3, 4, 5, 6, 7):
            np.testing.assert_allclose(
                np.asarray(windowed_without_shifted_score[idx]),
                np.asarray(windowed[idx]),
                rtol=1e-6,
                atol=1e-6,
            )
    np.testing.assert_allclose(np.asarray(windowed[2]), np.asarray(full[2]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(windowed[3]),
        np.asarray(full[3])[:, score_indices],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(windowed[7]),
        np.asarray(full[7])[:, score_indices],
        rtol=1e-6,
        atol=1e-6,
    )


def test_sparse_pass2_device_memory_probe_honors_visible_device():
    smi_output = "\n".join(
        [
            "0, GPU-a100, 40960",
            "1, GPU-h100, 81559",
        ],
    )

    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "1") == 81559 * 1024**2
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "GPU-h100") == 81559 * 1024**2
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "h100") == 81559 * 1024**2
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "-1") is None
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, None) == 40960 * 1024**2


def test_exact_raw_diff2_cache_budget_admission_and_fallback(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    gib = 1024**3
    mib = 1024**2

    assert _exact_raw_diff2_cache_limit_bytes(80 * gib, 40 * gib, 20 * gib) == 512 * mib
    assert _exact_raw_diff2_cache_limit_bytes(80 * gib, 1 * gib, 20 * gib) == 256 * mib
    assert _exact_raw_diff2_cache_limit_bytes(80 * gib, 40 * gib, 1 * gib) == 256 * mib
    assert _exact_raw_diff2_cache_limit_bytes(None, 40 * gib, 20 * gib) == 0
    assert _exact_raw_diff2_cache_limit_bytes(80 * gib, None, 20 * gib) == 0
    assert _exact_raw_diff2_cache_limit_bytes(80 * gib, 40 * gib, None) == 0
    assert _exact_raw_diff2_cache_limit_bytes(0, 40 * gib, 20 * gib) == 0
    assert _exact_raw_diff2_cache_limit_bytes(80 * gib, 0, 20 * gib) == 0
    assert _exact_raw_diff2_cache_limit_bytes(80 * gib, 40 * gib, 0) == 0
    assert _exact_raw_diff2_cache_limit_bytes(
        80 * gib,
        40 * gib,
        20 * gib,
        max_cache_bytes=128 * mib,
    ) == 128 * mib
    assert _exact_raw_diff2_cache_limit_bytes(
        80 * gib,
        40 * gib,
        20 * gib,
        max_cache_bytes=0,
    ) == 0

    estimated = _exact_raw_diff2_cache_estimated_bytes(2, 131_072, 116)
    assert estimated == 116 * mib
    assert _exact_raw_diff2_cache_fits_budget(estimated, estimated)
    assert not _exact_raw_diff2_cache_fits_budget(estimated + 4, estimated)
    assert not _exact_raw_diff2_cache_fits_budget(0, estimated)
    allocator_near_cap = estimated * 4 - 1
    near_cap_limit = _exact_raw_diff2_cache_limit_bytes(
        80 * gib,
        40 * gib,
        allocator_near_cap,
    )
    assert near_cap_limit < estimated
    assert not _exact_raw_diff2_cache_fits_budget(estimated, near_cap_limit)

    class FakeGpu:
        platform = "gpu"

        @staticmethod
        def memory_stats():
            return {"bytes_limit": 40 * gib, "bytes_in_use": 7 * gib}

    monkeypatch.setattr(bucketed_mod.jax, "devices", lambda: [FakeGpu()])
    assert bucketed_mod._jax_allocator_free_memory_bytes() == 33 * gib

    monkeypatch.setattr(FakeGpu, "memory_stats", staticmethod(lambda: {"bytes_limit": 40 * gib}))
    assert bucketed_mod._jax_allocator_free_memory_bytes() is None


def test_half_translation_phase_table_matches_generic_translate_images():
    rng = np.random.default_rng(13)
    image_shape = (16, 16)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    weighted_half = jnp.asarray(
        rng.normal(size=(3, n_half)).astype(np.float32)
        + 1j * rng.normal(size=(3, n_half)).astype(np.float32),
        dtype=jnp.complex64,
    )
    translations = jnp.asarray(rng.normal(size=(5, 2)).astype(np.float32))

    tiled_images = jnp.repeat(weighted_half[:, None, :], translations.shape[0], axis=1).reshape(
        weighted_half.shape[0] * translations.shape[0],
        -1,
    )
    tiled_translations = jnp.repeat(translations[None], weighted_half.shape[0], axis=0).reshape(
        weighted_half.shape[0] * translations.shape[0],
        -1,
    )
    generic = core.translate_images(tiled_images, tiled_translations, image_shape, half_image=True)
    phase_table = apply_half_translation_phases(
        weighted_half,
        half_translation_phase_table(translations, image_shape),
    )

    np.testing.assert_array_equal(np.asarray(phase_table), np.asarray(generic))


def test_indexed_half_translation_phase_table_matches_full_slice():
    translations = jnp.asarray(
        [
            [0.0, 0.0],
            [0.25, -0.5],
            [1.0, 0.75],
        ],
        dtype=jnp.float32,
    )
    pixel_indices = jnp.asarray([0, 2, 5, 7, 11], dtype=jnp.int32)
    full = half_translation_phase_table(translations, IMAGE_SHAPE)
    indexed = _half_translation_phase_table_for_indices(translations, IMAGE_SHAPE, pixel_indices)
    np.testing.assert_allclose(
        np.asarray(indexed),
        np.asarray(full)[:, np.asarray(pixel_indices)],
        rtol=1e-6,
        atol=1e-6,
    )


def test_em_translation_phase_dot_products_request_highest_precision():
    """Prevent A100 TF32 rounding from changing RELION translation phases."""

    translations = jnp.asarray(
        [
            [-2.0479863, 1.9520137],
            [1.9520137, -1.0479863],
        ],
        dtype=jnp.float32,
    )
    pixel_indices = jnp.asarray([0, 2, 5, 7, 11], dtype=jnp.int32)
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    images = jnp.ones((translations.shape[0], n_half), dtype=jnp.complex64)

    phase_jaxprs = (
        jax.make_jaxpr(lambda value: half_translation_phase_table(value, IMAGE_SHAPE))(translations),
        jax.make_jaxpr(
            lambda value: _half_translation_phase_table_for_indices(
                value,
                IMAGE_SHAPE,
                pixel_indices,
            )
        )(translations),
        jax.make_jaxpr(
            lambda image, value: core.translate_images(
                image,
                value,
                IMAGE_SHAPE,
                half_image=True,
            )
        )(images, translations),
    )

    for phase_jaxpr in phase_jaxprs:
        jaxpr_text = str(phase_jaxpr)
        assert "dot_general[" in jaxpr_text
        assert "precision=(Precision.HIGHEST, Precision.HIGHEST)" in jaxpr_text


def test_score_only_sparse_normalizer_matches_full_normalizer_stats():
    scores = jnp.asarray(
        [
            [[1.0, -2.0, -jnp.inf], [0.25, -0.5, -3.0]],
            [[-jnp.inf, -jnp.inf, -jnp.inf], [-jnp.inf, -jnp.inf, -jnp.inf]],
        ],
        dtype=jnp.float32,
    )

    full_log_z, _probs, full_best, full_argmax, full_pmax = _normalize_pass2_bucket(scores)
    score_log_z, score_best, score_argmax, score_pmax = _normalize_pass2_bucket_score_only(scores)

    np.testing.assert_allclose(np.asarray(score_log_z), np.asarray(full_log_z), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(score_best), np.asarray(full_best), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(score_argmax), np.asarray(full_argmax))
    logz_only = np.asarray(_logsumexp_pass2_bucket_score_only(scores))
    np.testing.assert_allclose(logz_only[0], np.asarray(full_log_z)[0], rtol=0, atol=0)
    assert np.isneginf(logz_only[1])
    np.testing.assert_allclose(np.asarray(score_pmax), np.asarray(full_pmax), rtol=1e-7, atol=1e-7)


def test_fine_rotation_override_preserves_fine_grid_order_and_parent_map():
    fine_rotations = np.arange(6 * 9, dtype=np.float32).reshape(6, 3, 3)
    fine_parent = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)

    per_image = _prepare_per_image_pass2_inputs(
        [np.array([0, 2], dtype=np.int32)],
        n_coarse_rot=3,
        n_coarse_trans=1,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=1,
        fine_translation_parent=np.array([0], dtype=np.int32),
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
    )

    np.testing.assert_array_equal(per_image["oversampled_rot_indices"][0], np.array([0, 2, 3, 5]))
    np.testing.assert_array_equal(per_image["parent_map"][0], np.array([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(per_image["oversampled_rots"][0], fine_rotations[[0, 2, 3, 5]])


def test_fine_rotation_override_can_follow_relion_parent_execution_order():
    fine_rotations = np.arange(6 * 9, dtype=np.float32).reshape(6, 3, 3)
    # Order-1 RECOVAR parent ids are psi-slow/direction-fast. Parent 48 is
    # direction 0, psi 1, so RELION executes it before parent 1.
    fine_parent = np.array([0, 0, 1, 1, 48, 48], dtype=np.int64)

    per_image = _prepare_per_image_pass2_inputs(
        [np.array([0, 1, 48], dtype=np.int32)],
        n_coarse_rot=576,
        n_coarse_trans=1,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=1,
        fine_translation_parent=np.array([0], dtype=np.int32),
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=True,
    )

    expected_indices = np.array([0, 1, 4, 5, 2, 3], dtype=np.int64)
    np.testing.assert_array_equal(per_image["oversampled_rot_indices"][0], expected_indices)
    np.testing.assert_array_equal(
        per_image["parent_map"][0],
        np.array([0, 0, 2, 2, 1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(per_image["oversampled_rots"][0], fine_rotations[expected_indices])


def test_exact_relion_fine_posterior_implies_relion_parent_execution_order(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.delenv("RECOVAR_RELION_FINE_ROTATION_EXECUTION_ORDER", raising=False)
    assert not bucketed_mod._relion_fine_parent_execution_order_enabled(
        use_relion_f32_fine_posterior=False,
    )
    assert bucketed_mod._relion_fine_parent_execution_order_enabled(
        use_relion_f32_fine_posterior=True,
    )

    monkeypatch.setenv("RECOVAR_RELION_FINE_ROTATION_EXECUTION_ORDER", "1")
    assert bucketed_mod._relion_fine_parent_execution_order_enabled(
        use_relion_f32_fine_posterior=False,
    )


def test_full_support_fine_rotation_override_reuses_shared_arrays():
    fine_rotations = np.arange(6 * 9, dtype=np.float32).reshape(6, 3, 3)
    fine_parent = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)
    rotation_log_prior = np.log(np.array([0.2, 0.3, 0.5], dtype=np.float32))

    per_image = _prepare_per_image_pass2_inputs(
        [None, None, None],
        n_coarse_rot=3,
        n_coarse_trans=1,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=2,
        fine_translation_parent=np.array([0, 0], dtype=np.int32),
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
    )

    for key in ("oversampled_rots", "parent_map", "oversampled_rot_indices", "unique_rot", "log_prior", "candidate_mask"):
        assert per_image[key][0] is per_image[key][1]
        assert per_image[key][1] is per_image[key][2]
    np.testing.assert_array_equal(per_image["oversampled_rot_indices"][0], np.arange(6, dtype=np.int64))
    np.testing.assert_array_equal(per_image["parent_map"][0], fine_parent.astype(np.int32))
    np.testing.assert_array_equal(per_image["oversampled_rots"][0], fine_rotations)
    np.testing.assert_allclose(per_image["log_prior"][0], rotation_log_prior[fine_parent], rtol=0, atol=0)
    assert isinstance(per_image["candidate_mask"][0], SparseCandidateMask)
    assert per_image["candidate_mask"][0].mode == "full"
    assert _candidate_mask_count(per_image["candidate_mask"][0]) == 12


def test_fine_grid_candidate_mask_uses_parented_translation_support():
    fine_rotations = np.arange(5 * 9, dtype=np.float32).reshape(5, 3, 3)
    fine_parent = np.array([0, 0, 1, 3, 3], dtype=np.int64)
    fine_translation_parent = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

    per_image = _prepare_per_image_pass2_inputs(
        [np.array([0, 2, 10], dtype=np.int32)],
        n_coarse_rot=4,
        n_coarse_trans=3,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=fine_translation_parent.size,
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
    )

    expected = np.array(
        [
            [True, False, True, True, False, True],
            [True, False, True, True, False, True],
            [False, True, False, False, True, False],
            [False, True, False, False, True, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(per_image["oversampled_rot_indices"][0], np.array([0, 1, 3, 4]))
    assert isinstance(per_image["candidate_mask"][0], SparseCandidateMask)
    assert _candidate_mask_count(per_image["candidate_mask"][0]) == int(expected.sum())
    np.testing.assert_array_equal(per_image["candidate_mask"][0], expected)


def test_sparse_pass2_projection_cache_reuses_fine_grid_projection_chunks(monkeypatch):
    """Fine-grid override projections are shared across sparse buckets."""

    n_images = 8
    n_coarse_rot = 48
    n_coarse_trans = 2
    fine_rotations = np.tile(np.eye(3, dtype=np.float32), (10, 1, 1))
    fine_parent = np.array([0, 0, 1, 1, 2, 3, 3, 4, 5, 7], dtype=np.int64)
    fine_translations = np.array(
        [[0.0, 0.0], [0.25, 0.0], [1.0, 0.0], [1.25, 0.0]],
        dtype=np.float32,
    )
    fine_translation_parent = np.array([0, 0, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([parent * n_coarse_trans for parent in parents], dtype=np.int32)
        for parents in ([0], [0, 1], [0, 1, 2], [3], [3, 4], [5], [7], [0, 3, 5])
    ]

    ds = MockDataset(n_images=n_images, seed=41)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=43)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", "16")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "4")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", str(1024**3))

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    projection_call_sizes = []

    def fake_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type
        projection_call_sizes.append(int(rotations_block.shape[0]))
        n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
        proj = jnp.zeros((rotations_block.shape[0], n_half), dtype=jnp.complex64)
        return_abs2 = kwargs.get("return_abs2", None)
        return proj, None if return_abs2 is False else jnp.zeros((rotations_block.shape[0], n_half), dtype=jnp.float32)

    monkeypatch.setattr(bucketed_mod, "_compute_projections_block", fake_project)

    compute_pass2_stats_sparse(
        ds,
        volume,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=False,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )

    assert projection_call_sizes == [4, 4, 2]
    assert sum(projection_call_sizes) == fine_rotations.shape[0]


def test_sparse_pass2_full_support_projection_cache_chunks_scores(monkeypatch):
    """Full-support cached buckets must stream score chunks without changing results."""

    n_images = 1
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0)
    fine_parent = np.arange(16, dtype=np.int64)
    fine_translations = np.asarray(
        [[0.0, 0.0], [0.25, 0.0], [1.0, 0.0], [1.25, 0.0]],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    translations = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    ds = MockDataset(n_images=n_images, seed=47)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=49)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)

    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", str(1024**3))
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", "1000000")

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    # Hold raw-diff2 cache admission constant across CPU-only and GPU runners.
    # This test counts the two full-score passes caused by fine M-step pruning;
    # cache fallback/recomputation has its own dedicated test below.
    ample_memory = 40 * 1024**3
    monkeypatch.setattr(bucketed_mod, "_device_memory_limit_bytes", lambda: ample_memory)
    monkeypatch.setattr(bucketed_mod, "_device_free_memory_bytes", lambda: ample_memory)
    monkeypatch.setattr(bucketed_mod, "_jax_allocator_free_memory_bytes", lambda: ample_memory)

    def fake_window_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type
        n_rot = int(rotations_block.shape[0])
        n_score = int(kwargs["score_indices"].shape[0])
        score = jnp.zeros((n_rot, n_score), dtype=jnp.complex64)
        recon_indices = kwargs.get("recon_indices")
        if recon_indices is None:
            return score, None, None
        n_recon = int(recon_indices.shape[0])
        recon = jnp.zeros((n_rot, n_recon), dtype=jnp.complex64)
        recon_abs2 = jnp.ones((n_rot, n_recon), dtype=jnp.float32)
        return score, recon, recon_abs2

    monkeypatch.setattr(bucketed_mod, "_compute_sparse_pass2_windowed_projections_block", fake_window_project)

    kwargs = dict(
        experiment_dataset=ds,
        volume=volume,
        mean_variance=mean_variance,
        noise_variance=noise_variance,
        translations=translations,
        significant_sample_indices=[None],
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=4,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=True,
        half_spectrum_scoring=True,
        relion_fine_mstep_prune=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK", "64")
    unchunked = compute_pass2_stats_sparse(**kwargs)

    score_chunk_sizes = []
    original_score = bucketed_mod._score_pass2_bucket_relion_gpu_diff2
    original_raw_score = bucketed_mod._score_pass2_bucket_relion_gpu_diff2_raw
    raw_score_refs = []
    raw_score_rotation_sizes = []
    max_live_raw_score_arrays = 0

    def counting_score(*args, **score_kwargs):
        score_chunk_sizes.append(int(args[2].shape[1]))
        return original_score(*args, **score_kwargs)

    def counting_raw_score(*args, **score_kwargs):
        nonlocal max_live_raw_score_arrays
        gc.collect()
        max_live_raw_score_arrays = max(
            max_live_raw_score_arrays,
            sum(ref() is not None for ref in raw_score_refs),
        )
        result = original_raw_score(*args, **score_kwargs)
        jax.block_until_ready(result)
        raw_score_rotation_sizes.append(int(args[2].shape[1]))
        raw_score_refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(bucketed_mod, "_score_pass2_bucket_relion_gpu_diff2", counting_score)
    monkeypatch.setattr(bucketed_mod, "_score_pass2_bucket_relion_gpu_diff2_raw", counting_raw_score)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK", "4")
    chunked = compute_pass2_stats_sparse(**kwargs)

    np.testing.assert_allclose(np.asarray(chunked[0]), np.asarray(unchunked[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(chunked[1]), np.asarray(unchunked[1]), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(chunked[2]), np.asarray(unchunked[2]))
    np.testing.assert_allclose(np.asarray(chunked[3]), np.asarray(unchunked[3]), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(chunked[4]), np.asarray(unchunked[4]), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(chunked[5]), np.asarray(unchunked[5]))
    _assert_relion_stats_close(chunked[6], unchunked[6], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(chunked[7]), np.asarray(unchunked[7]), rtol=1e-6, atol=1e-6)
    _assert_noise_stats_close((chunked[8],), (unchunked[8],), rtol=1e-5, atol=1e-5)
    # Exact-Gaussian scoring invokes the full scorer once per chunk to build
    # the fine M-step pruning support, then once per chunk again while
    # accumulating the final M-step/noise statistics.  The initial raw-diff2
    # minimum pass is counted separately below.
    assert score_chunk_sizes == [4, 4, 4, 4, 4, 4, 4, 4]
    assert len(raw_score_refs) >= 4
    assert max(raw_score_rotation_sizes) <= 4
    # JAX may retain a few completed result wrappers in its dispatch cache;
    # require forward progress rather than assuming Python weak-reference
    # lifetime is identical to device-buffer lifetime.
    assert max_live_raw_score_arrays < len(raw_score_refs)


def test_sparse_pass2_projection_cache_chunks_non_identity_indices(monkeypatch):
    """Cached sparse buckets must stream arbitrary rotation-index gathers."""

    n_images = 1
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0)
    fine_parent = np.arange(16, dtype=np.int64)
    fine_translations = np.asarray(
        [[0.0, 0.0], [0.25, 0.0], [1.0, 0.0], [1.25, 0.0]],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    translations = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    ds = MockDataset(n_images=n_images, seed=57)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=59)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    selected_rotations = np.asarray([1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15], dtype=np.int32)
    significant_samples = [
        np.asarray(
            [rot * 2 + trans for rot in selected_rotations.tolist() for trans in (0, 1)],
            dtype=np.int32,
        )
    ]

    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", str(1024**3))
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", "1000000")

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    def fake_window_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type
        n_rot = int(rotations_block.shape[0])
        n_score = int(kwargs["score_indices"].shape[0])
        score = jnp.zeros((n_rot, n_score), dtype=jnp.complex64)
        recon_indices = kwargs.get("recon_indices")
        if recon_indices is None:
            return score, None, None
        n_recon = int(recon_indices.shape[0])
        recon = jnp.zeros((n_rot, n_recon), dtype=jnp.complex64)
        recon_abs2 = jnp.ones((n_rot, n_recon), dtype=jnp.float32)
        return score, recon, recon_abs2

    monkeypatch.setattr(bucketed_mod, "_compute_sparse_pass2_windowed_projections_block", fake_window_project)

    kwargs = dict(
        experiment_dataset=ds,
        volume=volume,
        mean_variance=mean_variance,
        noise_variance=noise_variance,
        translations=translations,
        significant_sample_indices=significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=4,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=True,
        half_spectrum_scoring=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK", "64")
    unchunked = compute_pass2_stats_sparse(**kwargs)

    score_chunk_sizes = []
    original_score = bucketed_mod._score_pass2_bucket_relion_gpu_diff2

    def counting_score(*args, **score_kwargs):
        score_chunk_sizes.append(int(args[2].shape[1]))
        return original_score(*args, **score_kwargs)

    monkeypatch.setattr(bucketed_mod, "_score_pass2_bucket_relion_gpu_diff2", counting_score)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK", "4")
    chunked = compute_pass2_stats_sparse(**kwargs)

    np.testing.assert_allclose(np.asarray(chunked[0]), np.asarray(unchunked[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(chunked[1]), np.asarray(unchunked[1]), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(chunked[2]), np.asarray(unchunked[2]))
    np.testing.assert_allclose(np.asarray(chunked[3]), np.asarray(unchunked[3]), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(chunked[4]), np.asarray(unchunked[4]), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(chunked[5]), np.asarray(unchunked[5]))
    _assert_relion_stats_close(chunked[6], unchunked[6], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(chunked[7]), np.asarray(unchunked[7]), rtol=1e-6, atol=1e-6)
    _assert_noise_stats_close((chunked[8],), (unchunked[8],), rtol=1e-5, atol=1e-5)
    assert score_chunk_sizes
    assert max(score_chunk_sizes) <= 4
    assert sum(score_chunk_sizes) > len(score_chunk_sizes)


def test_score_log_z_only_matches_full_score_probe(monkeypatch):
    n_images = 6
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]], dtype=np.float32)
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([0, 1, 4, 7], dtype=np.int32),
        np.asarray([2, 3, 6], dtype=np.int32),
        np.asarray([8, 9], dtype=np.int32),
        np.asarray([0, 5, 10], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([2, 11], dtype=np.int32),
    ]

    ds = MockDataset(n_images=n_images, seed=51)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=53)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    full = compute_pass2_stats_sparse(
        ds,
        volume,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=False,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )
    log_evidence, score_log_z = compute_pass2_stats_sparse(
        ds,
        volume,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_score_log_z_only=True,
        accumulate_noise=False,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )

    np.testing.assert_allclose(np.asarray(log_evidence), np.asarray(full[6].log_evidence_per_image), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(score_log_z), np.asarray(full[7]), rtol=0, atol=0)
    # Gaussian score logZ is absolute (the common diff2 minimum has been
    # removed), so it is directly commensurate across independent calls.
    np.testing.assert_allclose(np.asarray(score_log_z), np.asarray(log_evidence), rtol=0, atol=0)
    assert np.all(np.asarray(full[6].best_log_score_per_image) <= np.asarray(log_evidence))

    cc_kwargs = dict(
        experiment_dataset=ds,
        volume=volume,
        mean_variance=mean_variance,
        noise_variance=noise_variance,
        translations=translations,
        significant_sample_indices=significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        accumulate_noise=False,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
        relion_firstiter_score_mode="normalized_cc",
    )
    cc_full = compute_pass2_stats_sparse(
        **cc_kwargs,
        return_stats=True,
        return_score_log_z=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
    )
    cc_log_evidence, cc_score_log_z = compute_pass2_stats_sparse(
        **cc_kwargs,
        return_score_log_z_only=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
    )
    np.testing.assert_allclose(
        np.asarray(cc_log_evidence),
        np.asarray(cc_full[6].log_evidence_per_image),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(np.asarray(cc_score_log_z), np.asarray(cc_full[7]), rtol=0, atol=0)
    assert np.any(np.asarray(cc_score_log_z) != np.asarray(cc_log_evidence))


def test_fused_other_class_log_z_matches_two_pass_normalization(monkeypatch):
    n_images = 5
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]], dtype=np.float32)
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([0, 1, 4, 7], dtype=np.int32),
        np.asarray([2, 3, 6], dtype=np.int32),
        np.asarray([8, 9], dtype=np.int32),
        np.asarray([0, 5, 10], dtype=np.int32),
        np.asarray([1, 2, 11], dtype=np.int32),
    ]

    ds = MockDataset(n_images=n_images, seed=71)
    volume_a = _hermitian_volume(VOLUME_SHAPE, seed=73)
    volume_b = _hermitian_volume(VOLUME_SHAPE, seed=79)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    common = dict(
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_stats=True,
        accumulate_noise=False,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
    )

    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    _, score_a = compute_pass2_stats_sparse(
        ds,
        volume_a,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        return_score_log_z_only=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        **common,
    )
    log_evidence_b, score_b = compute_pass2_stats_sparse(
        ds,
        volume_b,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        return_score_log_z_only=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        **common,
    )
    global_score_log_z = np.logaddexp(np.asarray(score_a, dtype=np.float64), np.asarray(score_b, dtype=np.float64))

    two_pass = compute_pass2_stats_sparse(
        ds,
        volume_b,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        normalization_log_z=global_score_log_z,
        normalization_score_mode="gaussian",
        **common,
    )
    fused = compute_pass2_stats_sparse(
        ds,
        volume_b,
        mean_variance,
        noise_variance,
        translations,
        significant_samples,
        normalization_other_score_log_z=score_a,
        normalization_score_mode="gaussian",
        return_score_log_z=True,
        **common,
    )

    np.testing.assert_allclose(np.asarray(fused[0]), np.asarray(two_pass[0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fused[1]), np.asarray(two_pass[1]), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(fused[2]), np.asarray(two_pass[2]))
    np.testing.assert_array_equal(np.asarray(fused[5]), np.asarray(two_pass[5]))
    np.testing.assert_allclose(
        np.asarray(fused[6].best_log_score_per_image),
        np.asarray(two_pass[6].best_log_score_per_image),
    )
    np.testing.assert_allclose(
        np.asarray(fused[6].max_posterior_per_image),
        np.asarray(two_pass[6].max_posterior_per_image),
    )
    np.testing.assert_allclose(
        np.asarray(fused[6].rotation_posterior_sums),
        np.asarray(two_pass[6].rotation_posterior_sums),
    )
    np.testing.assert_allclose(np.asarray(fused[6].log_evidence_per_image), np.asarray(log_evidence_b), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(fused[7]), np.asarray(score_b), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(score_b), np.asarray(log_evidence_b), rtol=0, atol=0)


@pytest.mark.parametrize("fine_prune", [False, True])
@pytest.mark.parametrize("winner_take_all", [False, True])
def test_sparse_pass2_rotation_chunking_matches_unchunked_windowed_path(
    monkeypatch, tmp_path, fine_prune, winner_take_all
):
    from recovar.em.dense_single_volume.helpers import compact_candidate_capture as capture_mod
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    monkeypatch.delenv(capture_mod.CAPTURE_DIR_ENV, raising=False)
    monkeypatch.delenv(capture_mod.CAPTURE_ITERATION_ENV, raising=False)
    monkeypatch.setattr(capture_mod, "_capture_counter", 0)
    monkeypatch.setattr(bucketed_mod, "_projection_cache_fits_budget", lambda *_args, **_kwargs: False)

    n_images = 4
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 8, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    fine_translations = np.asarray(
        [
            [0.0, 0.0],
            [0.25, 0.0],
            [0.5, 0.0],
            [0.75, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 1.0],
            [0.5, 1.0],
            [0.75, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([parent * 2 + (image_idx % 2) for parent in range(8)], dtype=np.int32)
        for image_idx in range(n_images)
    ]

    ds = MockDataset(n_images=n_images, seed=301)
    ds.dataset_indices = np.arange(n_images, dtype=np.int64)
    common = dict(
        experiment_dataset=ds,
        volume=_hermitian_volume(VOLUME_SHAPE, seed=303),
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        translations=jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        significant_sample_indices=significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=4,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=True,
        group_ids=np.asarray([0, 1, 0, 1], dtype=np.int64),
        scale_correction_group_count=5,
        half_spectrum_scoring=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
        relion_fine_mstep_prune=fine_prune,
        relion_firstiter_winner_take_all=winner_take_all,
    )

    disabled_unchunked = None
    if not fine_prune and not winner_take_all:
        original_capture = bucketed_mod.maybe_capture_k1_production_bucket

        def fail_if_disabled_capture_is_called(**_kwargs):
            raise AssertionError("disabled compact capture helper was called")

        monkeypatch.setattr(
            bucketed_mod,
            "maybe_capture_k1_production_bucket",
            fail_if_disabled_capture_is_called,
        )
        monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", str(1024**3))
        disabled_unchunked = compute_pass2_stats_sparse(**common)
        monkeypatch.setattr(
            bucketed_mod,
            "maybe_capture_k1_production_bucket",
            original_capture,
        )

    monkeypatch.setenv(capture_mod.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(capture_mod.CAPTURE_ITERATION_ENV, "3")
    try:
        bucketed_mod.set_bpref_contribution_dump_context(iteration=3, half=1)
        monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", str(1024**3))
        unchunked = compute_pass2_stats_sparse(**common)
        ds.dataset_indices = np.arange(n_images, dtype=np.int64) + n_images
        bucketed_mod.set_bpref_contribution_dump_context(iteration=3, half=2)
        monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", "512")
        chunked = compute_pass2_stats_sparse(**common)
    finally:
        bucketed_mod.clear_bpref_contribution_dump_context()

    marker = capture_mod.finalize_raw_capture_directory(
        tmp_path,
        expected_original_indices_by_half={
            1: np.arange(n_images, dtype=np.int64),
            2: np.arange(n_images, dtype=np.int64) + n_images,
        },
        expected_iteration=3,
    )
    assert marker["particle_count"] == 2 * n_images
    shards_by_half = {}
    for path in tmp_path.glob("raw_k1_*.npz"):
        with np.load(path, allow_pickle=False) as shard:
            shards_by_half[int(shard["half"])] = {
                name: np.asarray(shard[name]) for name in shard.files
            }
    assert set(shards_by_half) == {1, 2}
    unchunked_capture = shards_by_half[1]
    chunked_capture = shards_by_half[2]
    np.testing.assert_array_equal(
        chunked_capture["original_indices"] - n_images,
        unchunked_capture["original_indices"],
    )
    for name in (
        "candidate_offset",
        "candidate_local_rotation",
        "candidate_translation",
        "significant",
        "rotation_offset",
        "rotation_global_index",
        "rotation_parent_local",
        "rotation_parent_global",
    ):
        np.testing.assert_array_equal(chunked_capture[name], unchunked_capture[name])
    if not fine_prune and not winner_take_all:
        assert np.all(chunked_capture["significant"] == 1)
    np.testing.assert_allclose(
        chunked_capture["raw_combined_score"],
        unchunked_capture["raw_combined_score"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        chunked_capture["posterior"],
        unchunked_capture["posterior"],
        rtol=1e-6,
        atol=1e-6,
    )
    if winner_take_all:
        assert np.all(chunked_capture["pmax"] == 1)
        assert np.all(chunked_capture["significant_count"] == 1)
        assert np.all(chunked_capture["significant_threshold"] == 1)
        assert np.all(np.asarray(chunked[6].max_posterior_per_image) == 1)

    if disabled_unchunked is not None:
        np.testing.assert_array_equal(np.asarray(disabled_unchunked[0]), np.asarray(unchunked[0]))
        np.testing.assert_array_equal(np.asarray(disabled_unchunked[1]), np.asarray(unchunked[1]))
        np.testing.assert_array_equal(
            np.asarray(disabled_unchunked[6].max_posterior_per_image),
            np.asarray(unchunked[6].max_posterior_per_image),
        )

    np.testing.assert_allclose(np.asarray(chunked[0]), np.asarray(unchunked[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(chunked[1]), np.asarray(unchunked[1]), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(chunked[2]), np.asarray(unchunked[2]))
    np.testing.assert_array_equal(np.asarray(chunked[5]), np.asarray(unchunked[5]))
    np.testing.assert_allclose(
        np.asarray(chunked[6].log_evidence_per_image),
        np.asarray(unchunked[6].log_evidence_per_image),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(chunked[6].best_log_score_per_image),
        np.asarray(unchunked[6].best_log_score_per_image),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(chunked[6].max_posterior_per_image),
        np.asarray(unchunked[6].max_posterior_per_image),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(chunked[6].rotation_posterior_sums),
        np.asarray(unchunked[6].rotation_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(chunked[7]), np.asarray(unchunked[7]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(chunked[8].wsum_sigma2_noise),
        np.asarray(unchunked[8].wsum_sigma2_noise),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(chunked[8].wsum_img_power),
        np.asarray(unchunked[8].wsum_img_power),
        rtol=1e-6,
        atol=1e-6,
    )
    assert chunked[8].wsum_norm_correction is not None
    assert unchunked[8].wsum_norm_correction is not None
    np.testing.assert_allclose(
        np.asarray(chunked[8].wsum_norm_correction),
        np.asarray(unchunked[8].wsum_norm_correction),
        rtol=1e-5,
        atol=1e-5,
    )
    assert chunked[8].wsum_scale_correction_xa is not None
    assert chunked[8].wsum_scale_correction_aa is not None
    assert np.asarray(chunked[8].wsum_scale_correction_xa).shape == (5,)
    np.testing.assert_array_equal(np.asarray(chunked[8].wsum_scale_correction_xa)[2:], 0.0)
    np.testing.assert_array_equal(np.asarray(chunked[8].wsum_scale_correction_aa)[2:], 0.0)
    np.testing.assert_allclose(
        np.asarray(chunked[8].wsum_scale_correction_xa),
        np.asarray(unchunked[8].wsum_scale_correction_xa),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(chunked[8].wsum_scale_correction_aa),
        np.asarray(unchunked[8].wsum_scale_correction_aa),
        rtol=1e-5,
        atol=1e-5,
    )
    assert chunked[8].sumw == pytest.approx(unchunked[8].sumw, abs=1e-6)

    no_scale_shells = compute_pass2_stats_sparse(
        **{
            **common,
            "scale_correction_data_vs_prior": np.zeros(IMAGE_SHAPE[0] // 2 + 1, dtype=np.float32),
        }
    )
    np.testing.assert_array_equal(np.asarray(no_scale_shells[8].wsum_scale_correction_xa), 0.0)
    np.testing.assert_array_equal(np.asarray(no_scale_shells[8].wsum_scale_correction_aa), 0.0)
    np.testing.assert_allclose(
        np.asarray(no_scale_shells[8].wsum_norm_correction),
        np.asarray(chunked[8].wsum_norm_correction),
        rtol=1e-5,
        atol=1e-5,
    )

    common_pruned = dict(common)
    common_pruned["relion_fine_mstep_prune"] = True
    common_pruned["adaptive_fraction"] = 0.5
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", str(1024**3))
    unchunked_pruned = compute_pass2_stats_sparse(**common_pruned)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", "512")
    chunked_pruned = compute_pass2_stats_sparse(**common_pruned)
    _assert_noise_stats_close((chunked_pruned[8],), (unchunked_pruned[8],), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(chunked_pruned[6].rotation_posterior_sums),
        np.asarray(unchunked_pruned[6].rotation_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    pruned_rotation_mass = np.sum(
        np.asarray(unchunked_pruned[6].rotation_posterior_sums)
    )
    unpruned_rotation_mass = np.sum(
        np.asarray(unchunked[6].rotation_posterior_sums)
    )
    if winner_take_all:
        # Winner-take-all leaves one unit-weight candidate per image, so
        # subsequent significant-weight pruning cannot reduce total mass.
        assert pruned_rotation_mass == pytest.approx(unpruned_rotation_mass)
    else:
        assert pruned_rotation_mass < unpruned_rotation_mass

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", str(1024**3))
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", "0")
    full_prepare = compute_pass2_stats_sparse(**common)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", raising=False)
    windowed_prepare = compute_pass2_stats_sparse(**common)
    np.testing.assert_allclose(np.asarray(windowed_prepare[0]), np.asarray(full_prepare[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(windowed_prepare[1]), np.asarray(full_prepare[1]), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(windowed_prepare[2]), np.asarray(full_prepare[2]))
    np.testing.assert_array_equal(np.asarray(windowed_prepare[5]), np.asarray(full_prepare[5]))
    _assert_relion_stats_close(windowed_prepare[6], full_prepare[6], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(windowed_prepare[7]), np.asarray(full_prepare[7]), rtol=1e-6, atol=1e-6)
    _assert_noise_stats_close((windowed_prepare[8],), (full_prepare[8],), rtol=1e-5, atol=1e-5)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", raising=False)

    common_no_noise = dict(common)
    common_no_noise["accumulate_noise"] = False
    external_log_z = np.asarray(unchunked[7], dtype=np.float64) + 0.25
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", str(1024**3))
    unchunked_external = compute_pass2_stats_sparse(
        **common_no_noise,
        normalization_log_z=external_log_z,
        normalization_score_mode="gaussian",
    )
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", "512")
    chunked_external = compute_pass2_stats_sparse(
        **common_no_noise,
        normalization_log_z=external_log_z,
        normalization_score_mode="gaussian",
    )
    with pytest.raises(ValueError, match="external score normalization mode"):
        compute_pass2_stats_sparse(
            **common_no_noise,
            normalization_log_z=external_log_z,
            normalization_score_mode="normalized_cc",
        )
    with pytest.raises(ValueError, match="requires normalization_score_mode"):
        compute_pass2_stats_sparse(
            **common_no_noise,
            normalization_log_z=external_log_z,
        )

    np.testing.assert_allclose(
        np.asarray(chunked_external[0]),
        np.asarray(unchunked_external[0]),
        rtol=5e-4,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(chunked_external[1]),
        np.asarray(unchunked_external[1]),
        rtol=5e-4,
        atol=2e-3,
    )
    np.testing.assert_array_equal(np.asarray(chunked_external[2]), np.asarray(unchunked_external[2]))
    np.testing.assert_allclose(
        np.asarray(chunked_external[6].rotation_posterior_sums),
        np.asarray(unchunked_external[6].rotation_posterior_sums),
        rtol=1e-4,
        atol=1e-4,
    )


def test_exact_raw_diff2_cache_matches_fallback_bitwise_and_removes_recompute(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", "512")
    monkeypatch.delenv("RECOVAR_DISABLE_RELION_EXACT_FINE_GAUSSIAN", raising=False)
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    monkeypatch.setattr(bucketed_mod, "_projection_cache_fits_budget", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(bucketed_mod, "_device_memory_limit_bytes", lambda: 80 * 1024**3)
    monkeypatch.setattr(bucketed_mod, "_jax_allocator_free_memory_bytes", lambda: 20 * 1024**3)

    n_images = 2
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 8, axis=0)
    fine_translations = np.asarray(
        [[0.25 * x, float(y)] for y in range(2) for x in range(4)],
        dtype=np.float32,
    )
    common = dict(
        experiment_dataset=MockDataset(n_images=n_images, seed=1301),
        volume=_hermitian_volume(VOLUME_SHAPE, seed=1303),
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        translations=jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        significant_sample_indices=[
            np.asarray([parent * 2 + image_idx for parent in range(8)], dtype=np.int32)
            for image_idx in range(n_images)
        ],
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=4,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=False,
        half_spectrum_scoring=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=np.arange(8, dtype=np.int64),
        fine_translations_override=fine_translations,
        fine_translation_parent_override=np.repeat(np.arange(2, dtype=np.int32), 4),
    )

    combined_score_calls = 0
    original_combined_score = bucketed_mod._score_pass2_bucket_relion_gpu_diff2

    def count_combined_score_calls(*args, **kwargs):
        nonlocal combined_score_calls
        combined_score_calls += 1
        return original_combined_score(*args, **kwargs)

    monkeypatch.setattr(
        bucketed_mod,
        "_score_pass2_bucket_relion_gpu_diff2",
        count_combined_score_calls,
    )
    monkeypatch.setattr(bucketed_mod, "_device_free_memory_bytes", lambda: None)
    fallback = compute_pass2_stats_sparse(**common)
    fallback_combined_score_calls = combined_score_calls

    combined_score_calls = 0
    monkeypatch.setattr(bucketed_mod, "_device_free_memory_bytes", lambda: 40 * 1024**3)
    cached = compute_pass2_stats_sparse(**common)
    cached_combined_score_calls = combined_score_calls

    combined_score_calls = 0
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_EXACT_RAW_DIFF2_CACHE_MAX_BYTES", "0")
    disabled = compute_pass2_stats_sparse(**common)
    disabled_combined_score_calls = combined_score_calls

    assert cached_combined_score_calls < fallback_combined_score_calls
    assert disabled_combined_score_calls == fallback_combined_score_calls
    fallback_leaves = jax.tree_util.tree_leaves(fallback)
    cached_leaves = jax.tree_util.tree_leaves(cached)
    disabled_leaves = jax.tree_util.tree_leaves(disabled)
    assert len(cached_leaves) == len(fallback_leaves)
    assert len(disabled_leaves) == len(fallback_leaves)
    for cached_leaf, disabled_leaf, fallback_leaf in zip(
        cached_leaves,
        disabled_leaves,
        fallback_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(cached_leaf), np.asarray(fallback_leaf))
        np.testing.assert_array_equal(np.asarray(disabled_leaf), np.asarray(fallback_leaf))


@pytest.mark.parametrize("f32_fine_posterior", [False, True])
@pytest.mark.parametrize("shadow_only", [False, True])
def test_sparse_pass2_rotation_chunking_applies_to_relion_x_half_mstep_with_nonmatching_dump(
    monkeypatch,
    tmp_path,
    f32_fine_posterior,
    shadow_only,
):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "0")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_CURRENT_SIZE", "999")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", "512")
    if f32_fine_posterior:
        monkeypatch.setenv("RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR", "1")
    else:
        monkeypatch.delenv(
            "RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR",
            raising=False,
        )
    monkeypatch.setenv(
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR",
        str(tmp_path / "contributions"),
    )
    monkeypatch.setattr(bucketed_mod, "_projection_cache_fits_budget", lambda *_args, **_kwargs: False)
    adjoint_window_indices = []
    contribution_calls = []

    def capture_contribution(**kwargs):
        contribution_calls.append(kwargs)

    monkeypatch.setattr(
        bucketed_mod,
        "_maybe_dump_bpref_contribution_rows",
        capture_contribution,
    )
    if shadow_only:
        def resolve_shadow_modes(**kwargs):
            active = bool(kwargs["contribution_diagnostics_active"])
            return {
                "device_signature_requested": active,
                "contribution_diagnostics_active": active,
                "shadow_only": active,
                "high_precision_operand_bundle": False,
            }

        monkeypatch.setattr(
            bucketed_mod,
            "_resolve_bpref_bucket_diagnostic_modes",
            resolve_shadow_modes,
        )

    def fake_accumulate_adjoint(_flat_block, _flat_rotations, volume, **kwargs):
        assert kwargs["relion_x_half"] is True
        adjoint_window_indices.append(np.asarray(kwargs["window_indices"], dtype=np.int32))
        return volume

    monkeypatch.setattr(bucketed_mod, "_accumulate_adjoint_block_chunked", fake_accumulate_adjoint)

    projection_call_rows = []
    original_windowed_project = bucketed_mod._compute_sparse_pass2_windowed_projections_block

    def record_windowed_project(*args, **kwargs):
        projection_call_rows.append(int(args[1].shape[0]))
        return original_windowed_project(*args, **kwargs)

    monkeypatch.setattr(
        bucketed_mod,
        "_compute_sparse_pass2_windowed_projections_block",
        record_windowed_project,
    )

    n_images = 4
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 8, axis=0)
    fine_parent = np.arange(8, dtype=np.int64)
    fine_translations = np.asarray(
        [
            [0.0, 0.0],
            [0.25, 0.0],
            [0.5, 0.0],
            [0.75, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 1.0],
            [0.5, 1.0],
            [0.75, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([parent * 2 + (image_idx % 2) for parent in range(8)], dtype=np.int32)
        for image_idx in range(n_images)
    ]

    common = dict(
        experiment_dataset=MockDataset(n_images=n_images, seed=701),
        volume=_hermitian_volume(VOLUME_SHAPE, seed=703),
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        translations=jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        significant_sample_indices=significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=4,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=True,
        half_spectrum_scoring=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
        relion_x_half_mstep=True,
        relion_fine_mstep_prune=True,
        adaptive_fraction=0.5,
    )
    captured_result = compute_pass2_stats_sparse(**common)
    monkeypatch.delenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR")
    baseline_result = compute_pass2_stats_sparse(**common)

    assert projection_call_rows
    expected_recon_indices = make_fourier_window_spec(
        IMAGE_SHAPE,
        4,
        IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1),
        include_recon_window=True,
    ).recon_indices
    expected_xhalf_indices = bucketed_mod.centered_half_indices_to_fftw_half_indices(
        IMAGE_SHAPE,
        expected_recon_indices,
    )
    assert adjoint_window_indices
    for actual_indices in adjoint_window_indices:
        np.testing.assert_array_equal(actual_indices, np.asarray(expected_xhalf_indices, dtype=np.int32))
    assert len(contribution_calls) == 1
    contribution = contribution_calls[0]
    captured_rotation_count = np.asarray(contribution["rotations"]).shape[1]
    full_bucket_rows = int(n_images * captured_rotation_count)
    assert max(projection_call_rows) < full_bucket_rows
    expected_candidate_shape = (
        n_images,
        captured_rotation_count,
        fine_translations.shape[0],
    )
    assert np.asarray(contribution["scores"]).shape == expected_candidate_shape
    assert np.asarray(contribution["preprior_scores"]).shape == expected_candidate_shape
    assert np.asarray(contribution["probs"]).shape == expected_candidate_shape
    assert np.asarray(contribution["reconstruction_probs"]).shape == expected_candidate_shape
    assert np.asarray(contribution["reconstruction_mask"]).shape == expected_candidate_shape
    assert np.asarray(contribution["summed"]).shape[:2] == expected_candidate_shape[:2]
    assert np.asarray(contribution["ctf_probs"]).shape == np.asarray(
        contribution["summed"]
    ).shape
    assert np.asarray(contribution["rotations"]).shape == (
        n_images,
        captured_rotation_count,
        3,
        3,
    )
    assert np.asarray(contribution["reconstruction_sum_weight"]).shape == (n_images,)
    assert np.asarray(contribution["reconstruction_threshold"]).shape == (n_images,)
    assert contribution["shadow_only_mode"] is shadow_only
    assert contribution["shadow_score_bitwise_equal"] is shadow_only
    if shadow_only:
        assert contribution["shadow_reduction_agreement"] is not None
    else:
        assert contribution["shadow_reduction_agreement"] is None
    reconstruction_probs = np.asarray(contribution["reconstruction_probs"])
    reconstruction_mask = np.asarray(contribution["reconstruction_mask"])
    assert np.all(reconstruction_probs[~reconstruction_mask] == 0)
    captured_leaves = jax.tree_util.tree_leaves(captured_result)
    baseline_leaves = jax.tree_util.tree_leaves(baseline_result)
    assert len(captured_leaves) == len(baseline_leaves)
    for captured_leaf, baseline_leaf in zip(
        captured_leaves,
        baseline_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(captured_leaf),
            np.asarray(baseline_leaf),
        )


def test_sparse_pass2_chunked_fine_mstep_prune_is_uncapped(monkeypatch):
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES", "512")
    monkeypatch.setattr(bucketed_mod, "_projection_cache_fits_budget", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        bucketed_mod,
        "_accumulate_adjoint_block_chunked",
        lambda _flat_block, _flat_rotations, volume, **_kwargs: volume,
    )

    max_significant_args = []
    original_find_significant = bucketed_mod._find_significant_mask_full_sort

    def record_find_significant(weights_flat, adaptive_fraction=0.999, max_significants=500):
        max_significant_args.append(int(max_significants))
        return original_find_significant(
            weights_flat,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )

    monkeypatch.setattr(bucketed_mod, "_find_significant_mask_full_sort", record_find_significant)

    n_images = 4
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 8, axis=0)
    fine_parent = np.arange(8, dtype=np.int64)
    fine_translations = np.asarray(
        [
            [0.0, 0.0],
            [0.25, 0.0],
            [0.5, 0.0],
            [0.75, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 1.0],
            [0.5, 1.0],
            [0.75, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([parent * 2 + (image_idx % 2) for parent in range(8)], dtype=np.int32)
        for image_idx in range(n_images)
    ]

    compute_pass2_stats_sparse(
        experiment_dataset=MockDataset(n_images=n_images, seed=711),
        volume=_hermitian_volume(VOLUME_SHAPE, seed=719),
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        translations=jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        significant_sample_indices=significant_samples,
        nside_level=1,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=4,
        return_stats=True,
        return_score_log_z=True,
        accumulate_noise=True,
        half_spectrum_scoring=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
        relion_fine_mstep_prune=True,
        adaptive_fraction=0.5,
    )

    assert max_significant_args
    assert set(max_significant_args) == {-1}


def test_fused_sparse_k_class_pass2_matches_existing_two_pass_path(monkeypatch):
    """Default fused Class3D pass-2 must preserve the legacy sparse semantics."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod
    from recovar.em.sampling import rotation_grid_size

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")

    fused_score_results = []
    original_fused_pass2 = bucketed_mod.compute_k_class_pass2_stats_sparse_fused

    def capture_fused_score_result(*args, **kwargs):
        result = original_fused_pass2(*args, **kwargs)
        fused_score_results.append(result)
        return result

    monkeypatch.setattr(
        bucketed_mod,
        "compute_k_class_pass2_stats_sparse_fused",
        capture_fused_score_result,
    )

    n_images = 5
    n_classes = 2
    n_coarse_rot = rotation_grid_size(1)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray(
        [
            [0.0, 0.0],
            [0.25, 0.0],
            [0.5, 0.0],
            [0.75, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 1.0],
            [0.5, 1.0],
            [0.75, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    coarse_translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    significant_by_class = [
        [
            np.asarray([0, 1, 4, 7], dtype=np.int32),
            np.asarray([2, 3, 6], dtype=np.int32),
            np.asarray([8, 9], dtype=np.int32),
            np.asarray([0, 5, 10], dtype=np.int32),
            np.asarray([1, 2, 11], dtype=np.int32),
        ],
        [
            np.asarray([0, 2, 5], dtype=np.int32),
            np.asarray([1, 3, 7], dtype=np.int32),
            np.asarray([4, 8], dtype=np.int32),
            np.asarray([3, 9, 11], dtype=np.int32),
            np.asarray([0, 6], dtype=np.int32),
        ],
    ]
    ds = MockDataset(n_images=n_images, seed=91)
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=101),
            _hermitian_volume(VOLUME_SHAPE, seed=103),
        ],
    )
    kwargs = dict(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.asarray([0.45, 0.55], dtype=np.float64)),
        accumulate_noise=True,
        return_best_pose_details=True,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={
            "current_size": None,
            "relion_half_volume_mstep": False,
            # Numbered iterations after firstiter_cc explicitly forward the
            # inactive diagnostic scope through the fused K-class route.
            "bpref_device_signature_active": False,
            "group_ids": np.asarray([0, 1, 0, 1, 2], dtype=np.int64),
            "scale_correction_group_count": 7,
            "scale_corrections": np.asarray([1.0, 1.08, 0.91, 1.03, 0.97], dtype=np.float32),
            "scale_correction_data_vs_prior": np.stack(
                [
                    np.full(IMAGE_SHAPE[0] // 2 + 1, 4.0, dtype=np.float32),
                    np.zeros(IMAGE_SHAPE[0] // 2 + 1, dtype=np.float32),
                ],
            ),
        },
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "3")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "0")
    with pytest.raises(RuntimeError, match="requires fused scoring"):
        _run_sparse_k_class_adaptive_pass2(**kwargs)
    kwargs["engine_kwargs"]["relion_exact_fine_gaussian"] = False
    legacy = _run_sparse_k_class_adaptive_pass2(**kwargs)
    kwargs["engine_kwargs"].pop("relion_exact_fine_gaussian")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "0")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", raising=False)

    def fail_if_compact_pair_scorer_is_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("compact pair scorer must stay off when compact-pair execution is disabled")

    monkeypatch.setattr(
        bucketed_mod,
        "_score_pass2_pairs_relion_gpu_diff2_raw",
        fail_if_compact_pair_scorer_is_called,
    )
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED_NOISE_NORM", "0")

    def unsupported_fused_pass2(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError("forced unsupported fused route")

    monkeypatch.setattr(
        bucketed_mod,
        "compute_k_class_pass2_stats_sparse_fused",
        unsupported_fused_pass2,
    )
    with pytest.raises(RuntimeError, match="cannot fall back"):
        _run_sparse_k_class_adaptive_pass2(**kwargs)
    monkeypatch.setattr(
        bucketed_mod,
        "compute_k_class_pass2_stats_sparse_fused",
        capture_fused_score_result,
    )
    fused = _run_sparse_k_class_adaptive_pass2(**kwargs)
    assert fused_score_results
    np.testing.assert_allclose(
        np.asarray(fused_score_results[-1].class_score_log_z),
        np.asarray(fused_score_results[-1].class_log_evidence),
        rtol=0,
        atol=0,
    )
    assert fused.profile_summary["sparse_kclass_raw_host_staging_total_bytes"] > 0
    assert (
        fused.profile_summary["sparse_kclass_raw_host_staging_peak_bytes"]
        <= fused.profile_summary["sparse_kclass_raw_host_staging_max_bytes"]
    )
    assert fused.profile_summary["sparse_kclass_raw_host_staging_s"] >= 0.0
    assert fused.profile_summary["sparse_kclass_exact_relion_gaussian"] is True
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES", "1")
    with pytest.raises(MemoryError, match="raw diff2 host staging would exceed"):
        _run_sparse_k_class_adaptive_pass2(**kwargs)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES", raising=False)
    assert fused.profile_summary["sparse_kclass_rectangular_active_rows"] is True
    assert fused.profile_summary["sparse_kclass_rectangular_active_prematmul"] is False
    assert fused.profile_summary["sparse_kclass_rectangular_active_rows_min_bucket_size"] == 4096
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE", "1")
    active_fused = _run_sparse_k_class_adaptive_pass2(**kwargs)
    assert active_fused.profile_summary["sparse_kclass_rectangular_active_rows_min_bucket_size"] == 1
    assert active_fused.profile_summary["sparse_kclass_rectangular_mstep_active_rows"] > 0
    np.testing.assert_allclose(
        np.asarray(active_fused.Ft_y),
        np.asarray(legacy.Ft_y),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(active_fused.Ft_ctf),
        np.asarray(legacy.Ft_ctf),
        rtol=1e-5,
        atol=1e-5,
    )
    _assert_noise_residual_terms_close(active_fused.noise_stats, legacy.noise_stats, rtol=1e-5, atol=1e-5)
    _assert_k_class_noise_sumw_matches_class_mass(active_fused, rtol=1e-4, atol=1e-4)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO", "999")
    prematmul_active_fused = _run_sparse_k_class_adaptive_pass2(**kwargs)
    assert prematmul_active_fused.profile_summary["sparse_kclass_rectangular_active_prematmul"] is True
    assert prematmul_active_fused.profile_summary["sparse_kclass_rectangular_active_prematmul_attempts"] > 0
    assert prematmul_active_fused.profile_summary["sparse_kclass_rectangular_active_prematmul_used"] > 0
    assert (
        prematmul_active_fused.profile_summary["sparse_kclass_rectangular_active_prematmul_attempts"]
        == prematmul_active_fused.profile_summary["sparse_kclass_rectangular_active_prematmul_used"]
        + prematmul_active_fused.profile_summary["sparse_kclass_rectangular_active_prematmul_skipped"]
    )
    np.testing.assert_allclose(
        np.asarray(prematmul_active_fused.Ft_y),
        np.asarray(legacy.Ft_y),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(prematmul_active_fused.Ft_ctf),
        np.asarray(legacy.Ft_ctf),
        rtol=1e-5,
        atol=1e-5,
    )
    _assert_noise_residual_terms_close(
        prematmul_active_fused.noise_stats,
        legacy.noise_stats,
        rtol=1e-5,
        atol=1e-5,
    )
    _assert_k_class_noise_sumw_matches_class_mass(prematmul_active_fused, rtol=1e-4, atol=1e-4)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_FUSED_NOISE_NORM", raising=False)
    fused_noise_norm = _run_sparse_k_class_adaptive_pass2(**kwargs)
    np.testing.assert_allclose(np.asarray(fused_noise_norm.Ft_y), np.asarray(fused.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(fused_noise_norm.Ft_ctf),
        np.asarray(fused.Ft_ctf),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(fused_noise_norm.per_class_hard_assignments),
        np.asarray(fused.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(fused_noise_norm.class_assignments), np.asarray(fused.class_assignments))
    np.testing.assert_array_equal(np.asarray(fused_noise_norm.pose_assignments), np.asarray(fused.pose_assignments))
    np.testing.assert_allclose(
        np.asarray(fused_noise_norm.class_responsibilities),
        np.asarray(fused.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused_noise_norm.class_posterior_sums),
        np.asarray(fused.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_noise_stats_close(fused_noise_norm.noise_stats, fused.noise_stats, rtol=1e-5, atol=1e-5)
    _assert_k_class_extra_outputs_close(fused_noise_norm, fused)
    assert fused.profile_summary["sparse_kclass_fused_noise_norm"] is False
    assert fused_noise_norm.profile_summary["sparse_kclass_fused_noise_norm"] is True
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_FUSED_NOISE_NORM", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", "1")
    compact = _run_sparse_k_class_adaptive_pass2(**kwargs)

    np.testing.assert_allclose(np.asarray(fused.Ft_y), np.asarray(legacy.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(fused.Ft_ctf), np.asarray(legacy.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(fused.per_class_hard_assignments),
        np.asarray(legacy.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(fused.class_assignments), np.asarray(legacy.class_assignments))
    np.testing.assert_array_equal(np.asarray(fused.pose_assignments), np.asarray(legacy.pose_assignments))
    np.testing.assert_allclose(
        np.asarray(fused.class_responsibilities),
        np.asarray(legacy.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused.class_posterior_sums),
        np.asarray(legacy.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_noise_stats_close(fused.noise_stats, legacy.noise_stats, rtol=1e-5, atol=1e-5)
    assert fused.noise_stats[0].wsum_scale_correction_xa is not None
    assert fused.noise_stats[0].wsum_scale_correction_aa is not None
    assert np.asarray(fused.noise_stats[0].wsum_scale_correction_xa).shape == (7,)
    np.testing.assert_array_equal(np.asarray(fused.noise_stats[0].wsum_scale_correction_xa)[3:], 0.0)
    np.testing.assert_array_equal(np.asarray(fused.noise_stats[0].wsum_scale_correction_aa)[3:], 0.0)
    np.testing.assert_array_equal(np.asarray(fused.noise_stats[1].wsum_scale_correction_xa), 0.0)
    np.testing.assert_array_equal(np.asarray(fused.noise_stats[1].wsum_scale_correction_aa), 0.0)
    _assert_k_class_noise_sumw_matches_class_mass(fused, rtol=1e-4, atol=1e-4)

    window_kwargs = dict(kwargs)
    window_kwargs["engine_kwargs"] = {
        **kwargs["engine_kwargs"],
        "current_size": 2,
        # This block compares window-preparation mechanics, including the
        # intentionally non-fused legacy path. Exact reduced-size scoring is
        # covered separately and rejects this full-spectrum configuration.
        "relion_exact_fine_gaussian": False,
    }
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", "0")
    window_full_prepare = _run_sparse_k_class_adaptive_pass2(**window_kwargs)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", raising=False)
    windowed_prepare = _run_sparse_k_class_adaptive_pass2(**window_kwargs)
    np.testing.assert_allclose(
        np.asarray(windowed_prepare.Ft_y),
        np.asarray(window_full_prepare.Ft_y),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(windowed_prepare.Ft_ctf),
        np.asarray(window_full_prepare.Ft_ctf),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(windowed_prepare.per_class_hard_assignments),
        np.asarray(window_full_prepare.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(windowed_prepare.class_assignments),
        np.asarray(window_full_prepare.class_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(windowed_prepare.pose_assignments),
        np.asarray(window_full_prepare.pose_assignments),
    )
    np.testing.assert_allclose(
        np.asarray(windowed_prepare.class_responsibilities),
        np.asarray(window_full_prepare.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(windowed_prepare.class_posterior_sums),
        np.asarray(window_full_prepare.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_k_class_extra_outputs_close(windowed_prepare, window_full_prepare)
    assert window_full_prepare.profile_summary["sparse_kclass_windowed_prepare"] is False
    assert windowed_prepare.profile_summary["sparse_kclass_windowed_prepare"] is True
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", raising=False)

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "0")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", "0")
    legacy_window_full_prepare = _run_sparse_k_class_adaptive_pass2(**window_kwargs)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", raising=False)
    legacy_windowed_prepare = _run_sparse_k_class_adaptive_pass2(**window_kwargs)
    np.testing.assert_allclose(
        np.asarray(legacy_windowed_prepare.Ft_y),
        np.asarray(legacy_window_full_prepare.Ft_y),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(legacy_windowed_prepare.Ft_ctf),
        np.asarray(legacy_window_full_prepare.Ft_ctf),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(legacy_windowed_prepare.per_class_hard_assignments),
        np.asarray(legacy_window_full_prepare.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(legacy_windowed_prepare.class_assignments),
        np.asarray(legacy_window_full_prepare.class_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(legacy_windowed_prepare.pose_assignments),
        np.asarray(legacy_window_full_prepare.pose_assignments),
    )
    np.testing.assert_allclose(
        np.asarray(legacy_windowed_prepare.class_responsibilities),
        np.asarray(legacy_window_full_prepare.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(legacy_windowed_prepare.class_posterior_sums),
        np.asarray(legacy_window_full_prepare.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_k_class_extra_outputs_close(legacy_windowed_prepare, legacy_window_full_prepare)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", "1")

    np.testing.assert_allclose(np.asarray(compact.Ft_y), np.asarray(fused.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(compact.Ft_ctf), np.asarray(fused.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(compact.per_class_hard_assignments),
        np.asarray(fused.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(compact.class_assignments), np.asarray(fused.class_assignments))
    np.testing.assert_array_equal(np.asarray(compact.pose_assignments), np.asarray(fused.pose_assignments))
    np.testing.assert_allclose(
        np.asarray(compact.class_responsibilities),
        np.asarray(fused.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(compact.class_posterior_sums),
        np.asarray(fused.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_k_class_extra_outputs_close(compact, fused)
    assert compact.profile_summary["sparse_kclass_compact_buckets"] is True
    assert compact.profile_summary["sparse_kclass_compact_slot_ratio"] <= 1.0
    assert compact.profile_summary["sparse_kclass_valid_pair_candidates"] > 0
    assert compact.profile_summary["sparse_kclass_rectangular_pair_candidates"] > compact.profile_summary[
        "sparse_kclass_valid_pair_candidates"
    ]
    assert compact.profile_summary["sparse_kclass_valid_pair_reduction"] > 1.0

    monkeypatch.setattr(
        bucketed_mod,
        "_score_pass2_pairs_relion_gpu_diff2_raw",
        _score_pass2_pairs_relion_gpu_diff2_raw,
    )
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", raising=False)
    compact_pairs = _run_sparse_k_class_adaptive_pass2(**kwargs)
    np.testing.assert_allclose(np.asarray(compact_pairs.Ft_y), np.asarray(fused.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(compact_pairs.Ft_ctf), np.asarray(fused.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(compact_pairs.per_class_hard_assignments),
        np.asarray(fused.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(compact_pairs.class_assignments), np.asarray(fused.class_assignments))
    np.testing.assert_array_equal(np.asarray(compact_pairs.pose_assignments), np.asarray(fused.pose_assignments))
    np.testing.assert_allclose(
        np.asarray(compact_pairs.class_responsibilities),
        np.asarray(fused.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(compact_pairs.class_posterior_sums),
        np.asarray(fused.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_k_class_extra_outputs_close(compact_pairs, fused)
    assert compact_pairs.profile_summary["sparse_kclass_compact_pairs"] is True
    assert compact_pairs.profile_summary["sparse_kclass_compact_pair_buckets"] > 0
    assert compact_pairs.profile_summary["sparse_kclass_padded_pair_reduction"] > 1.0
    assert compact_pairs.profile_summary["sparse_kclass_compact_active_rows"] is True
    assert compact_pairs.profile_summary["sparse_kclass_compact_mstep_active_rows"] > 0
    assert (
        compact_pairs.profile_summary["sparse_kclass_compact_mstep_padded_active_rows"]
        >= compact_pairs.profile_summary["sparse_kclass_compact_mstep_active_rows"]
    )
    assert compact_pairs.profile_summary["sparse_kclass_compact_mstep_active_ratio"] <= 1.0
    assert compact_pairs.profile_summary["sparse_kclass_compact_mstep_padded_active_ratio"] <= 1.0

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS", "0")
    compact_pairs_no_active = _run_sparse_k_class_adaptive_pass2(**kwargs)
    np.testing.assert_allclose(
        np.asarray(compact_pairs_no_active.Ft_y),
        np.asarray(fused.Ft_y),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(compact_pairs_no_active.Ft_ctf),
        np.asarray(fused.Ft_ctf),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(compact_pairs_no_active.per_class_hard_assignments),
        np.asarray(fused.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(compact_pairs_no_active.class_assignments),
        np.asarray(fused.class_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(compact_pairs_no_active.pose_assignments),
        np.asarray(fused.pose_assignments),
    )
    np.testing.assert_allclose(
        np.asarray(compact_pairs_no_active.class_responsibilities),
        np.asarray(fused.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(compact_pairs_no_active.class_posterior_sums),
        np.asarray(fused.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_k_class_extra_outputs_close(compact_pairs_no_active, fused)
    assert compact_pairs_no_active.profile_summary["sparse_kclass_compact_pairs"] is True
    assert compact_pairs_no_active.profile_summary["sparse_kclass_compact_active_rows"] is False

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "0")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE", "1")
    rectangular_active = _run_sparse_k_class_adaptive_pass2(**kwargs)
    np.testing.assert_allclose(
        np.asarray(rectangular_active.Ft_y),
        np.asarray(fused.Ft_y),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(rectangular_active.Ft_ctf),
        np.asarray(fused.Ft_ctf),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(rectangular_active.per_class_hard_assignments),
        np.asarray(fused.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(rectangular_active.class_assignments),
        np.asarray(fused.class_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(rectangular_active.pose_assignments),
        np.asarray(fused.pose_assignments),
    )
    np.testing.assert_allclose(
        np.asarray(rectangular_active.class_responsibilities),
        np.asarray(fused.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(rectangular_active.class_posterior_sums),
        np.asarray(fused.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_k_class_extra_outputs_close(rectangular_active, fused)
    assert rectangular_active.profile_summary["sparse_kclass_rectangular_active_rows"] is True
    assert rectangular_active.profile_summary["sparse_kclass_rectangular_active_prematmul"] is False
    assert rectangular_active.profile_summary["sparse_kclass_rectangular_active_rows_min_bucket_size"] == 1
    assert rectangular_active.profile_summary["sparse_kclass_rectangular_mstep_active_rows"] > 0
    assert (
        rectangular_active.profile_summary["sparse_kclass_rectangular_mstep_padded_active_rows"]
        >= rectangular_active.profile_summary["sparse_kclass_rectangular_mstep_active_rows"]
    )
    assert rectangular_active.profile_summary["sparse_kclass_rectangular_mstep_active_ratio"] <= 1.0
    assert rectangular_active.profile_summary["sparse_kclass_rectangular_mstep_padded_active_ratio"] <= 1.0
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO", "999")
    rectangular_active_prematmul = _run_sparse_k_class_adaptive_pass2(**kwargs)
    np.testing.assert_allclose(
        np.asarray(rectangular_active_prematmul.Ft_y),
        np.asarray(fused.Ft_y),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(rectangular_active_prematmul.Ft_ctf),
        np.asarray(fused.Ft_ctf),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(rectangular_active_prematmul.per_class_hard_assignments),
        np.asarray(fused.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(rectangular_active_prematmul.class_assignments),
        np.asarray(fused.class_assignments),
    )
    np.testing.assert_array_equal(
        np.asarray(rectangular_active_prematmul.pose_assignments),
        np.asarray(fused.pose_assignments),
    )
    np.testing.assert_allclose(
        np.asarray(rectangular_active_prematmul.class_responsibilities),
        np.asarray(fused.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(rectangular_active_prematmul.class_posterior_sums),
        np.asarray(fused.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_k_class_extra_outputs_close(rectangular_active_prematmul, fused)
    assert rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_active_rows"] is True
    assert rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_active_prematmul"] is True
    assert rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_active_prematmul_attempts"] > 0
    assert rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_active_prematmul_used"] > 0
    assert (
        rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_active_prematmul_attempts"]
        == rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_active_prematmul_used"]
        + rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_active_prematmul_skipped"]
    )
    assert rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_active_rows_min_bucket_size"] == 1
    assert rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_mstep_active_rows"] > 0
    assert (
        rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_mstep_padded_active_rows"]
        >= rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_mstep_active_rows"]
    )
    assert rectangular_active_prematmul.profile_summary["sparse_kclass_rectangular_mstep_active_ratio"] <= 1.0
    assert rectangular_active_prematmul.profile_summary[
        "sparse_kclass_rectangular_mstep_padded_active_ratio"
    ] <= 1.0
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")

    scorer_calls = {"rectangular": 0, "compact_pair": 0}

    def counting_rectangular_score(*args, **kwargs):
        scorer_calls["rectangular"] += 1
        return _score_pass2_bucket_relion_gpu_diff2_raw(*args, **kwargs)

    def counting_compact_pair_score(*args, **kwargs):
        scorer_calls["compact_pair"] += 1
        return _score_pass2_pairs_relion_gpu_diff2_raw(*args, **kwargs)

    monkeypatch.setattr(
        bucketed_mod,
        "_score_pass2_bucket_relion_gpu_diff2_raw",
        counting_rectangular_score,
    )
    monkeypatch.setattr(
        bucketed_mod,
        "_score_pass2_pairs_relion_gpu_diff2_raw",
        counting_compact_pair_score,
    )
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "32")
    hybrid = _run_sparse_k_class_adaptive_pass2(**kwargs)
    np.testing.assert_allclose(np.asarray(hybrid.Ft_y), np.asarray(fused.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(hybrid.Ft_ctf), np.asarray(fused.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(hybrid.per_class_hard_assignments),
        np.asarray(fused.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(hybrid.class_assignments), np.asarray(fused.class_assignments))
    np.testing.assert_array_equal(np.asarray(hybrid.pose_assignments), np.asarray(fused.pose_assignments))
    np.testing.assert_allclose(
        np.asarray(hybrid.class_responsibilities),
        np.asarray(fused.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(hybrid.class_posterior_sums),
        np.asarray(fused.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_k_class_extra_outputs_close(hybrid, fused)
    assert scorer_calls["rectangular"] > 0
    assert scorer_calls["compact_pair"] > 0
    assert hybrid.profile_summary["sparse_kclass_compact_pairs"] is True
    assert hybrid.profile_summary["sparse_kclass_compact_pairs_min_bucket_size"] == 32
    assert hybrid.profile_summary["sparse_kclass_hybrid_compact_pair_images"] > 0
    assert hybrid.profile_summary["sparse_kclass_hybrid_rectangular_images"] > 0
    assert (
        hybrid.profile_summary["sparse_kclass_hybrid_compact_pair_images"]
        + hybrid.profile_summary["sparse_kclass_hybrid_rectangular_images"]
        == n_images
    )

    scorer_calls = {"rectangular": 0, "compact_pair": 0}
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS", "1")
    hybrid_active = _run_sparse_k_class_adaptive_pass2(**kwargs)
    np.testing.assert_allclose(np.asarray(hybrid_active.Ft_y), np.asarray(fused.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(hybrid_active.Ft_ctf), np.asarray(fused.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(hybrid_active.per_class_hard_assignments),
        np.asarray(fused.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(hybrid_active.class_assignments), np.asarray(fused.class_assignments))
    np.testing.assert_array_equal(np.asarray(hybrid_active.pose_assignments), np.asarray(fused.pose_assignments))
    np.testing.assert_allclose(
        np.asarray(hybrid_active.class_responsibilities),
        np.asarray(fused.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(hybrid_active.class_posterior_sums),
        np.asarray(fused.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_k_class_extra_outputs_close(hybrid_active, fused)
    assert scorer_calls["rectangular"] > 0
    assert scorer_calls["compact_pair"] > 0
    assert hybrid_active.profile_summary["sparse_kclass_compact_pairs"] is True
    assert hybrid_active.profile_summary["sparse_kclass_compact_active_rows"] is True
    assert hybrid_active.profile_summary["sparse_kclass_compact_pairs_min_bucket_size"] == 32
    assert hybrid_active.profile_summary["sparse_kclass_hybrid_compact_pair_images"] > 0
    assert hybrid_active.profile_summary["sparse_kclass_hybrid_rectangular_images"] > 0
    assert (
        hybrid_active.profile_summary["sparse_kclass_hybrid_compact_pair_images"]
        + hybrid_active.profile_summary["sparse_kclass_hybrid_rectangular_images"]
        == n_images
    )
    assert hybrid_active.profile_summary["sparse_kclass_compact_mstep_active_rows"] > 0
    assert hybrid_active.profile_summary["sparse_kclass_compact_mstep_active_ratio"] <= 1.0

    monkeypatch.setattr(
        bucketed_mod,
        "_score_pass2_bucket_relion_gpu_diff2",
        _score_pass2_bucket_relion_gpu_diff2,
    )
    monkeypatch.setattr(
        bucketed_mod,
        "_score_pass2_pairs_relion_gpu_diff2",
        _score_pass2_pairs_relion_gpu_diff2,
    )
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", "1")
    checked = _run_sparse_k_class_adaptive_pass2(**kwargs)
    assert checked.profile_summary["sparse_kclass_compact_pair_check_rows"] > 0
    np.testing.assert_allclose(np.asarray(checked.Ft_y), np.asarray(fused.Ft_y), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(checked.Ft_ctf), np.asarray(fused.Ft_ctf), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(checked.class_assignments), np.asarray(fused.class_assignments))
    np.testing.assert_array_equal(np.asarray(checked.pose_assignments), np.asarray(fused.pose_assignments))

    kwargs["engine_kwargs"]["relion_exact_fine_gaussian"] = False
    checked_algebraic = _run_sparse_k_class_adaptive_pass2(**kwargs)
    assert checked_algebraic.profile_summary["sparse_kclass_compact_pair_check_rows"] > 0
    kwargs["engine_kwargs"].pop("relion_exact_fine_gaussian")


def test_fused_sparse_k1_default_compact_pairs_matches_existing_sparse_path(monkeypatch):
    """Explicit K=1 fused sparse adaptive pass-2 should preserve the legacy sparse result."""

    from recovar.em.sampling import rotation_grid_size

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "3")

    n_images = 5
    n_classes = 1
    n_coarse_rot = rotation_grid_size(1)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray(
        [
            [0.0, 0.0],
            [0.25, 0.0],
            [0.5, 0.0],
            [0.75, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 1.0],
            [0.5, 1.0],
            [0.75, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    coarse_translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    significant_by_class = [
        [
            np.asarray([0, 1, 4, 7], dtype=np.int32),
            np.asarray([2, 3, 6], dtype=np.int32),
            np.asarray([8, 9], dtype=np.int32),
            np.asarray([0, 5, 10], dtype=np.int32),
            np.asarray([1, 2, 11], dtype=np.int32),
        ],
    ]
    ds = MockDataset(n_images=n_images, seed=91)
    volumes = jnp.asarray(_hermitian_volume(VOLUME_SHAPE, seed=101))[None, :]
    kwargs = dict(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.zeros(n_classes, dtype=np.float64),
        accumulate_noise=True,
        return_best_pose_details=True,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={
            "current_size": None,
            "relion_half_volume_mstep": False,
            "group_ids": np.asarray([0, 1, 0, 1, 2], dtype=np.int64),
            "scale_corrections": np.asarray([1.0, 1.08, 0.91, 1.03, 0.97], dtype=np.float32),
        },
    )

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "0")
    legacy = _run_sparse_k_class_adaptive_pass2(**kwargs)

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    fused = _run_sparse_k_class_adaptive_pass2(**kwargs)

    np.testing.assert_allclose(np.asarray(fused.Ft_y), np.asarray(legacy.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(fused.Ft_ctf), np.asarray(legacy.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(fused.per_class_hard_assignments),
        np.asarray(legacy.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(fused.class_assignments), np.asarray(legacy.class_assignments))
    np.testing.assert_array_equal(np.asarray(fused.pose_assignments), np.asarray(legacy.pose_assignments))
    np.testing.assert_allclose(
        np.asarray(fused.class_responsibilities),
        np.asarray(legacy.class_responsibilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused.class_posterior_sums),
        np.asarray(legacy.class_posterior_sums),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_noise_residual_terms_close(fused.noise_stats, legacy.noise_stats, rtol=1e-5, atol=1e-5)
    assert fused.noise_stats[0].wsum_scale_correction_xa is not None
    assert fused.noise_stats[0].wsum_scale_correction_aa is not None
    assert np.asarray(fused.noise_stats[0].wsum_scale_correction_xa).shape == (3,)
    np.testing.assert_allclose(
        np.asarray(fused.noise_stats[0].wsum_scale_correction_xa),
        np.asarray(legacy.noise_stats[0].wsum_scale_correction_xa),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(fused.noise_stats[0].wsum_scale_correction_aa),
        np.asarray(legacy.noise_stats[0].wsum_scale_correction_aa),
        rtol=1e-5,
        atol=1e-5,
    )
    _assert_k_class_noise_sumw_matches_class_mass(fused, rtol=1e-4, atol=1e-4)
    _assert_k_class_extra_outputs_close(fused, legacy)
    assert fused.profile_summary["sparse_kclass_compact_pairs"] is True
    assert fused.profile_summary["sparse_kclass_valid_pair_reduction"] > 1.0
    np.testing.assert_array_equal(np.asarray(fused.class_assignments), np.zeros(n_images, dtype=np.int32))


def test_fused_sparse_k_class_capture_requires_companion_contribution_dump(monkeypatch):
    """Fused K-class device capture fails closed without its operand bundle."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    signature = inspect.signature(bucketed_mod.compute_k_class_pass2_stats_sparse_fused)
    assert "bpref_device_signature_active" in signature.parameters

    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    monkeypatch.delenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", raising=False)
    with pytest.raises(RuntimeError, match="requires RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR"):
        bucketed_mod.compute_k_class_pass2_stats_sparse_fused(
            None,
            np.zeros((2, 1), dtype=np.complex64),
            np.ones(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            [[], []],
            rotation_log_priors_by_class=[None, None],
            nside_level=0,
            disc_type="linear_interp",
            oversampling_order=0,
            current_size=None,
            bpref_device_signature_active=True,
        )


def test_bpref_contribution_stop_requires_completed_target_files(monkeypatch, tmp_path):
    """The diagnostic sentinel fires only after requested files exist."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    contribution_path = tmp_path / "contribution.npz"
    device_path = tmp_path / "contribution.device.npz"
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_STOP_AFTER_TARGET", "1")
    monkeypatch.delenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", raising=False)

    with pytest.raises(RuntimeError, match="missing its contribution file"):
        bucketed_mod._maybe_stop_after_bpref_contribution_dump(
            contribution_path=contribution_path,
            device_signature_path=None,
        )

    contribution_path.write_bytes(b"contribution")
    with pytest.raises(bucketed_mod.BPrefContributionDumpComplete) as exc_info:
        bucketed_mod._maybe_stop_after_bpref_contribution_dump(
            contribution_path=contribution_path,
            device_signature_path=None,
        )
    assert exc_info.value.contribution_path == contribution_path
    assert exc_info.value.device_signature_path is None

    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="missing its requested device-signature file"):
        bucketed_mod._maybe_stop_after_bpref_contribution_dump(
            contribution_path=contribution_path,
            device_signature_path=device_path,
        )

    device_path.write_bytes(b"device")
    with pytest.raises(bucketed_mod.BPrefContributionDumpComplete) as exc_info:
        bucketed_mod._maybe_stop_after_bpref_contribution_dump(
            contribution_path=contribution_path,
            device_signature_path=device_path,
        )
    assert exc_info.value.contribution_path == contribution_path
    assert exc_info.value.device_signature_path == device_path


def test_fused_sparse_k_class_capture_is_observational(monkeypatch, tmp_path):
    """Selected fused-K capture rows must not change authoritative accumulators."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod
    from recovar.em.sampling import rotation_grid_size

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS", "0")
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", str(tmp_path / "device"))
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", str(tmp_path / "contributions"))
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES", "0")
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_CLASS", "2")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION", "1")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", "1")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS", "1")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")
    monkeypatch.setattr(bucketed_mod, "_require_bpref_device_soft_particle_arm", lambda **_kwargs: None)
    monkeypatch.setattr(
        bucketed_mod,
        "_accumulate_adjoint_block_chunked",
        lambda _flat_block, _flat_rotations, volume, **_kwargs: volume,
    )

    captures = []

    def capture_rows(**kwargs):
        captures.append(kwargs)

    monkeypatch.setattr(bucketed_mod, "_maybe_dump_bpref_contribution_rows", capture_rows)

    n_images = 2
    n_classes = 2
    n_coarse_rot = rotation_grid_size(0)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
    fine_parent = np.asarray([0, 1, 2], dtype=np.int64)
    fine_translations = np.asarray([[0.0, 0.0], [0.25, 0.0]], dtype=np.float32)
    fine_translation_parent = np.zeros(2, dtype=np.int32)
    significant_by_class = [
        [np.asarray([0, 1], dtype=np.int32), np.asarray([1, 2], dtype=np.int32)],
        [np.asarray([0, 2], dtype=np.int32), np.asarray([0, 1], dtype=np.int32)],
    ]
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=2027),
            _hermitian_volume(VOLUME_SHAPE, seed=2029),
        ]
    )
    common = dict(
        experiment_dataset=MockDataset(n_images=n_images, seed=2039),
        volumes=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        translations=np.asarray([[0.0, 0.0]], dtype=np.float32),
        significant_sample_indices_by_class=significant_by_class,
        rotation_log_priors_by_class=[None] * n_classes,
        nside_level=0,
        disc_type="linear_interp",
        oversampling_order=0,
        current_size=4,
        half_spectrum_scoring=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
        relion_x_half_mstep=True,
        relion_fine_mstep_prune_mode="joint",
        adaptive_fraction=0.9,
    )

    plain = bucketed_mod.compute_k_class_pass2_stats_sparse_fused(
        **common,
        bpref_device_signature_active=False,
    )
    instrumented = bucketed_mod.compute_k_class_pass2_stats_sparse_fused(
        **common,
        bpref_device_signature_active=True,
    )

    assert captures
    assert {capture["class_index"] for capture in captures} == {1}
    assert all(np.array_equal(capture["image_indices"], np.asarray([0])) for capture in captures)
    assert all(capture["shadow_only_mode"] is True for capture in captures)
    np.testing.assert_array_equal(np.asarray(instrumented.Ft_y), np.asarray(plain.Ft_y))
    np.testing.assert_array_equal(np.asarray(instrumented.Ft_ctf), np.asarray(plain.Ft_ctf))


def test_sparse_kclass_fused_default_keeps_k1_on_single_class_path(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_FUSED", raising=False)
    assert _use_fused_sparse_k_class_pass2(1) is False
    assert _use_fused_sparse_k_class_pass2(2) is True

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    assert _use_fused_sparse_k_class_pass2(1) is True

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "0")
    assert _use_fused_sparse_k_class_pass2(2) is False


def test_compact_pair_half_spectrum_reuses_mstep_sums_for_noise(monkeypatch):
    """Compact-pair noise can reuse M-step sums when score/recon images match."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod
    from recovar.em.sampling import rotation_grid_size

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "3")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", raising=False)

    n_images = 5
    n_classes = 2
    n_coarse_rot = rotation_grid_size(1)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    coarse_translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    significant_by_class = [
        [
            np.asarray([0, 1, 4, 7], dtype=np.int32),
            np.asarray([2, 3, 6], dtype=np.int32),
            np.asarray([8, 9], dtype=np.int32),
            np.asarray([0, 5, 10], dtype=np.int32),
            np.asarray([1, 2, 11], dtype=np.int32),
        ],
        [
            np.asarray([0, 2, 5], dtype=np.int32),
            np.asarray([1, 3, 7], dtype=np.int32),
            np.asarray([4, 8], dtype=np.int32),
            np.asarray([3, 9, 11], dtype=np.int32),
            np.asarray([0, 6], dtype=np.int32),
        ],
    ]
    ds = MockDataset(n_images=n_images, seed=91)
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=101),
            _hermitian_volume(VOLUME_SHAPE, seed=103),
        ],
    )
    kwargs = dict(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.asarray([0.45, 0.55], dtype=np.float64)),
        accumulate_noise=True,
        return_best_pose_details=True,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={
            "current_size": None,
            "relion_half_volume_mstep": False,
            "half_spectrum_scoring": True,
        },
    )

    original_weighted_sums = bucketed_mod._compact_pair_weighted_rotation_sums

    def run_with_reuse(enabled):
        calls = {"count": 0}

        def counting_weighted_sums(*args, **kwargs):
            calls["count"] += 1
            return original_weighted_sums(*args, **kwargs)

        monkeypatch.setattr(
            bucketed_mod,
            "_compact_pair_weighted_rotation_sums",
            counting_weighted_sums,
        )
        if enabled is None:
            monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS", raising=False)
        else:
            monkeypatch.setenv(
                "RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS",
                "1" if enabled else "0",
            )
        result = _run_sparse_k_class_adaptive_pass2(**kwargs)
        monkeypatch.setattr(
            bucketed_mod,
            "_compact_pair_weighted_rotation_sums",
            original_weighted_sums,
        )
        return result, calls["count"]

    reused, reused_calls = run_with_reuse(True)
    disabled, disabled_calls = run_with_reuse(False)
    defaulted, defaulted_calls = run_with_reuse(None)
    reuse_count = int(reused.profile_summary["sparse_kclass_compact_noise_sum_reuses"])

    assert reuse_count > 0
    assert int(disabled.profile_summary["sparse_kclass_compact_noise_sum_reuses"]) == 0
    assert int(defaulted.profile_summary["sparse_kclass_compact_noise_sum_reuses"]) == 0
    assert int(reused.profile_summary["sparse_kclass_compact_noise_fused_active_gathers"]) > 0
    assert int(disabled.profile_summary["sparse_kclass_compact_noise_fused_active_gathers"]) > 0
    assert int(defaulted.profile_summary["sparse_kclass_compact_noise_fused_active_gathers"]) > 0
    assert reused_calls + reuse_count == disabled_calls
    assert defaulted_calls == disabled_calls
    np.testing.assert_allclose(np.asarray(reused.Ft_y), np.asarray(disabled.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(reused.Ft_ctf), np.asarray(disabled.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(defaulted.Ft_y), np.asarray(disabled.Ft_y), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(defaulted.Ft_ctf), np.asarray(disabled.Ft_ctf), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(reused.class_assignments), np.asarray(disabled.class_assignments))
    np.testing.assert_array_equal(np.asarray(reused.pose_assignments), np.asarray(disabled.pose_assignments))
    np.testing.assert_array_equal(np.asarray(defaulted.class_assignments), np.asarray(disabled.class_assignments))
    np.testing.assert_array_equal(np.asarray(defaulted.pose_assignments), np.asarray(disabled.pose_assignments))
    _assert_noise_stats_close(reused.noise_stats, disabled.noise_stats, rtol=1e-5, atol=1e-5)
    _assert_noise_stats_close(defaulted.noise_stats, disabled.noise_stats, rtol=0, atol=0)
    _assert_k_class_extra_outputs_close(reused, disabled)
    _assert_k_class_extra_outputs_close(defaulted, disabled)

    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS", raising=False)


def test_compact_pair_tail_coalesced_execution_matches_uncoalesced(monkeypatch):
    """Tail coalescing may change only compact-pair padding and chunking."""

    from recovar.em.sampling import rotation_grid_size

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "16")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES", "1000000")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", raising=False)

    n_images = 4
    n_classes = 2
    n_coarse_rot = rotation_grid_size(1)
    n_coarse_trans = 4
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 64, axis=0)
    fine_parent = np.repeat(np.arange(8, dtype=np.int64), 8)
    fine_translations = np.asarray(
        [
            [float(dx), float(dy)]
            for dx in (0.0, 0.25, 0.5, 0.75)
            for dy in (0.0, 0.5, 1.0, 1.5)
        ],
        dtype=np.float32,
    )
    fine_translation_parent = np.repeat(np.arange(n_coarse_trans, dtype=np.int32), 4)
    coarse_translations = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
        dtype=np.float32,
    )

    def coarse_samples(n_rot, n_trans):
        return np.asarray(
            [rot * n_coarse_trans + trans for rot in range(n_rot) for trans in range(n_trans)],
            dtype=np.int32,
        )

    significant_by_class = [
        [
            coarse_samples(1, 1),
            coarse_samples(1, 2),
            coarse_samples(2, 2),
            coarse_samples(4, 2),
        ],
        [
            coarse_samples(1, 1),
            coarse_samples(2, 1),
            coarse_samples(2, 2),
            coarse_samples(4, 2),
        ],
    ]
    ds = MockDataset(n_images=n_images, seed=131)
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=151),
            _hermitian_volume(VOLUME_SHAPE, seed=157),
        ],
    )
    kwargs = dict(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.asarray([0.45, 0.55], dtype=np.float64)),
        accumulate_noise=True,
        return_best_pose_details=True,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={"current_size": None, "relion_half_volume_mstep": False},
    )

    def assert_same_result(actual, expected):
        np.testing.assert_allclose(np.asarray(actual.Ft_y), np.asarray(expected.Ft_y), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(actual.Ft_ctf), np.asarray(expected.Ft_ctf), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(
            np.asarray(actual.per_class_hard_assignments),
            np.asarray(expected.per_class_hard_assignments),
        )
        np.testing.assert_array_equal(np.asarray(actual.class_assignments), np.asarray(expected.class_assignments))
        np.testing.assert_array_equal(np.asarray(actual.pose_assignments), np.asarray(expected.pose_assignments))
        np.testing.assert_allclose(
            np.asarray(actual.class_responsibilities),
            np.asarray(expected.class_responsibilities),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(actual.class_posterior_sums),
            np.asarray(expected.class_posterior_sums),
            rtol=1e-6,
            atol=1e-6,
        )
        _assert_k_class_extra_outputs_close(actual, expected)

    for active_rows in ("1", "0"):
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS", active_rows)
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES", "0")
        uncoalesced = _run_sparse_k_class_adaptive_pass2(**kwargs)
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES", "4")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION", "2.0")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE", "16")
        coalesced = _run_sparse_k_class_adaptive_pass2(**kwargs)

        assert coalesced.profile_summary["sparse_kclass_compact_pair_buckets"] < uncoalesced.profile_summary[
            "sparse_kclass_compact_pair_buckets"
        ]
        assert uncoalesced.profile_summary["sparse_kclass_compact_pair_tail_coalesce_max_images"] == 0
        assert coalesced.profile_summary["sparse_kclass_compact_pair_tail_coalesce_max_images"] == 4
        assert coalesced.profile_summary["sparse_kclass_compact_pair_tail_coalesce_max_inflation"] == pytest.approx(
            2.0,
        )
        assert coalesced.profile_summary["sparse_kclass_compact_pair_tail_coalesce_min_bucket_size"] == 16
        assert_same_result(coalesced, uncoalesced)

    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE", raising=False)


def test_compact_pair_masked_scoring_reuses_noise_ctf_sums(monkeypatch):
    """Masked compact-pair noise recomputes image sums but reuses CTF/prob sums."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod
    from recovar.em.sampling import rotation_grid_size

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "3")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", raising=False)

    n_images = 5
    n_classes = 2
    n_coarse_rot = rotation_grid_size(1)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    coarse_translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    significant_by_class = [
        [
            np.asarray([0, 1, 4, 7], dtype=np.int32),
            np.asarray([2, 3, 6], dtype=np.int32),
            np.asarray([8, 9], dtype=np.int32),
            np.asarray([0, 5, 10], dtype=np.int32),
            np.asarray([1, 2, 11], dtype=np.int32),
        ],
        [
            np.asarray([0, 2, 5], dtype=np.int32),
            np.asarray([1, 3, 7], dtype=np.int32),
            np.asarray([4, 8], dtype=np.int32),
            np.asarray([3, 9, 11], dtype=np.int32),
            np.asarray([0, 6], dtype=np.int32),
        ],
    ]
    ds = MockDataset(n_images=n_images, seed=91)
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=101),
            _hermitian_volume(VOLUME_SHAPE, seed=103),
        ],
    )
    kwargs = dict(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.asarray([0.45, 0.55], dtype=np.float64)),
        accumulate_noise=True,
        return_best_pose_details=True,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={
            "current_size": None,
            "relion_half_volume_mstep": False,
            "half_spectrum_scoring": True,
            "score_with_masked_images": True,
        },
    )

    original_weighted_sums = bucketed_mod._compact_pair_weighted_rotation_sums
    original_image_sums = bucketed_mod._compact_pair_weighted_image_sums
    original_fused_sums = bucketed_mod._compact_pair_weighted_rotation_and_image_sums

    def fail_duplicate_count_scan(*args, **kwargs):
        del args, kwargs
        raise AssertionError("compact-pair execution should reuse precomputed candidate counts")

    monkeypatch.setattr(
        bucketed_mod,
        "_compact_pair_counts_from_candidate_masks",
        fail_duplicate_count_scan,
    )

    def run_with_reuse(enabled: bool):
        calls = {"weighted": 0, "image": 0, "fused": 0}

        def counting_weighted_sums(*args, **kwargs):
            calls["weighted"] += 1
            return original_weighted_sums(*args, **kwargs)

        def counting_image_sums(*args, **kwargs):
            calls["image"] += 1
            return original_image_sums(*args, **kwargs)

        def counting_fused_sums(*args, **kwargs):
            calls["fused"] += 1
            return original_fused_sums(*args, **kwargs)

        monkeypatch.setattr(
            bucketed_mod,
            "_compact_pair_weighted_rotation_sums",
            counting_weighted_sums,
        )
        monkeypatch.setattr(
            bucketed_mod,
            "_compact_pair_weighted_image_sums",
            counting_image_sums,
        )
        monkeypatch.setattr(
            bucketed_mod,
            "_compact_pair_weighted_rotation_and_image_sums",
            counting_fused_sums,
        )
        monkeypatch.setenv(
            "RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS",
            "1" if enabled else "0",
        )
        result = _run_sparse_k_class_adaptive_pass2(**kwargs)
        monkeypatch.setattr(
            bucketed_mod,
            "_compact_pair_weighted_rotation_sums",
            original_weighted_sums,
        )
        monkeypatch.setattr(
            bucketed_mod,
            "_compact_pair_weighted_image_sums",
            original_image_sums,
        )
        monkeypatch.setattr(
            bucketed_mod,
            "_compact_pair_weighted_rotation_and_image_sums",
            original_fused_sums,
        )
        return result, calls

    reused, reused_calls = run_with_reuse(True)
    disabled, disabled_calls = run_with_reuse(False)
    ctf_reuse_count = int(reused.profile_summary["sparse_kclass_compact_noise_ctf_sum_reuses"])

    assert int(reused.profile_summary["sparse_kclass_compact_noise_sum_reuses"]) == 0
    assert ctf_reuse_count > 0
    assert int(reused.profile_summary["sparse_kclass_compact_noise_image_sum_precomputes"]) == ctf_reuse_count
    assert int(reused.profile_summary["sparse_kclass_compact_noise_fused_active_gathers"]) > 0
    assert int(disabled.profile_summary["sparse_kclass_compact_noise_fused_active_gathers"]) > 0
    assert int(disabled.profile_summary["sparse_kclass_compact_noise_ctf_sum_reuses"]) == 0
    assert reused_calls["fused"] == ctf_reuse_count
    assert reused_calls["weighted"] == 0
    assert reused_calls["image"] == 0
    assert reused_calls["fused"] + ctf_reuse_count == disabled_calls["weighted"]
    assert disabled_calls["fused"] == 0
    assert disabled_calls["image"] == 0
    np.testing.assert_allclose(np.asarray(reused.Ft_y), np.asarray(disabled.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(reused.Ft_ctf), np.asarray(disabled.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(reused.class_assignments), np.asarray(disabled.class_assignments))
    np.testing.assert_array_equal(np.asarray(reused.pose_assignments), np.asarray(disabled.pose_assignments))
    _assert_noise_stats_close(reused.noise_stats, disabled.noise_stats, rtol=1e-5, atol=1e-5)
    _assert_k_class_extra_outputs_close(reused, disabled)

    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS", raising=False)


def test_fused_sparse_k_class_relion_half_mstep_keeps_half_accumulators(monkeypatch):
    """K-class sparse fused pass-2 should not expand RELION half accumulators on return."""

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod
    from recovar.em.sampling import rotation_grid_size

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK", raising=False)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "0")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS", raising=False)

    def fail_if_full_expansion_is_called(*_args, **_kwargs):
        raise AssertionError("sparse fused K-class should keep half-volume accumulators")

    monkeypatch.setattr(bucketed_mod, "half_volume_accumulators_to_full", fail_if_full_expansion_is_called)

    n_images = 2
    n_classes = 2
    n_coarse_rot = rotation_grid_size(0)
    coarse_translations = np.asarray([[0.0, 0.0]], dtype=np.float32)
    fine_translations = np.asarray([[0.0, 0.0]], dtype=np.float32)
    fine_translation_parent = np.asarray([0], dtype=np.int32)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    fine_parent = np.asarray([0, 1], dtype=np.int64)
    significant_by_class = [
        [np.asarray([0], dtype=np.int32) for _ in range(n_images)],
        [np.asarray([0], dtype=np.int32) for _ in range(n_images)],
    ]
    ds = MockDataset(n_images=n_images, seed=191)
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=193),
            _hermitian_volume(VOLUME_SHAPE, seed=197),
        ],
    )

    result = _run_sparse_k_class_adaptive_pass2(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.full(n_classes, 1.0 / n_classes, dtype=np.float64)),
        accumulate_noise=False,
        return_best_pose_details=False,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={"current_size": None, "relion_half_volume_mstep": True},
    )

    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(VOLUME_SHAPE)))
    assert np.asarray(result.Ft_y).shape == (n_classes, half_size)
    assert np.asarray(result.Ft_ctf).shape == (n_classes, half_size)
    assert result.profile_summary["sparse_kclass_fused_s"] >= 0.0
    assert isinstance(result.profile_summary["sparse_kclass_windowed_translation_tile_cap"], bool)


def test_fused_sparse_k_class_fine_mstep_prune_flag_exercises_compact_pairs(monkeypatch):
    """Diagnostic fine-pass M-step pruning should work without x-half layout."""

    from recovar.em.sampling import rotation_grid_size

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE", "1")

    n_images = 2
    n_classes = 2
    n_coarse_rot = rotation_grid_size(0)
    coarse_translations = np.asarray([[0.0, 0.0]], dtype=np.float32)
    fine_translations = np.asarray([[0.0, 0.0]], dtype=np.float32)
    fine_translation_parent = np.asarray([0], dtype=np.int32)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    fine_parent = np.asarray([0, 1], dtype=np.int64)
    significant_by_class = [
        [np.asarray([0], dtype=np.int32) for _ in range(n_images)],
        [np.asarray([0], dtype=np.int32) for _ in range(n_images)],
    ]
    ds = MockDataset(n_images=n_images, seed=211)
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=223),
            _hermitian_volume(VOLUME_SHAPE, seed=227),
        ],
    )

    result = _run_sparse_k_class_adaptive_pass2(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.full(n_classes, 1.0 / n_classes, dtype=np.float64)),
        accumulate_noise=True,
        return_best_pose_details=False,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={"current_size": None, "relion_half_volume_mstep": False},
    )

    assert result.profile_summary["sparse_kclass_compact_pairs"] is True
    assert result.profile_summary["sparse_kclass_relion_fine_mstep_prune"] is True
    assert result.profile_summary["sparse_kclass_relion_fine_mstep_prune_mode"] == "per_class"
    np.testing.assert_allclose(
        np.sum(np.asarray(result.class_posterior_sums)),
        float(n_images),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.class_mstep_posterior_sums),
        np.asarray(result.profile_summary["sparse_kclass_mstep_class_posterior_sums"]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert result.profile_summary["class_posterior_sums_used_override"] is True
    np.testing.assert_allclose(
        np.asarray(result.class_posterior_sums),
        np.asarray(result.profile_summary["class_posterior_sums_returned"]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.class_mstep_posterior_sums),
        np.asarray(result.profile_summary["class_mstep_posterior_sums_returned"]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.asarray(result.profile_summary["class_posterior_sums_full"]).shape == (n_classes,)
    _assert_k_class_noise_sumw_matches_class_mass(result, rtol=1e-4, atol=1e-4)


def test_fused_sparse_k_class_joint_fine_mstep_prune_flag(monkeypatch):
    """Joint fine-pass M-step pruning should be reachable for Class3D parity."""

    from recovar.em.sampling import rotation_grid_size

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE", "joint")

    n_images = 2
    n_classes = 2
    n_coarse_rot = rotation_grid_size(0)
    coarse_translations = np.asarray([[0.0, 0.0]], dtype=np.float32)
    fine_translations = np.asarray([[0.0, 0.0]], dtype=np.float32)
    fine_translation_parent = np.asarray([0], dtype=np.int32)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    fine_parent = np.asarray([0, 1], dtype=np.int64)
    significant_by_class = [
        [np.asarray([0], dtype=np.int32) for _ in range(n_images)],
        [np.asarray([0], dtype=np.int32) for _ in range(n_images)],
    ]
    ds = MockDataset(n_images=n_images, seed=229)
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=233),
            _hermitian_volume(VOLUME_SHAPE, seed=239),
        ],
    )

    result = _run_sparse_k_class_adaptive_pass2(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.full(n_classes, 1.0 / n_classes, dtype=np.float64)),
        accumulate_noise=True,
        return_best_pose_details=False,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={"current_size": None, "relion_half_volume_mstep": False},
    )

    assert result.profile_summary["sparse_kclass_compact_pairs"] is True
    assert result.profile_summary["sparse_kclass_relion_fine_mstep_prune"] is True
    assert result.profile_summary["sparse_kclass_relion_fine_mstep_prune_mode"] == "joint"
    np.testing.assert_allclose(
        np.sum(np.asarray(result.class_posterior_sums)),
        float(n_images),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.class_mstep_posterior_sums),
        np.asarray(result.profile_summary["sparse_kclass_mstep_class_posterior_sums"]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert result.profile_summary["class_posterior_sums_used_override"] is True
    np.testing.assert_allclose(
        np.asarray(result.class_posterior_sums),
        np.asarray(result.profile_summary["class_posterior_sums_returned"]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.class_mstep_posterior_sums),
        np.asarray(result.profile_summary["class_mstep_posterior_sums_returned"]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.asarray(result.profile_summary["class_posterior_sums_full"]).shape == (n_classes,)
    _assert_k_class_noise_sumw_matches_class_mass(result, rtol=1e-4, atol=1e-4)


def test_compact_significance_uses_complement_for_dense_masks():
    mask = np.ones(16, dtype=bool)
    mask[[3, 11]] = False

    compact = compact_significant_sample_indices_from_mask(mask)

    assert isinstance(compact, ComplementSignificantSampleIndices)
    np.testing.assert_array_equal(compact.excluded_indices, np.asarray([3, 11], dtype=np.int32))
    assert compact.total_size == 16
    assert significant_sample_count(compact, 16) == 14
    np.testing.assert_array_equal(significant_sample_ids(compact, 16), np.flatnonzero(mask))


def test_prepare_pass2_inputs_complement_matches_explicit_dense_support():
    n_coarse_rot = 4
    n_coarse_trans = 3
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 0, 1, 2, 2, 3], dtype=np.int64)
    fine_translation_parent = np.asarray([0, 0, 1, 2], dtype=np.int32)
    excluded = np.asarray([1, 8], dtype=np.int32)
    full_mask = np.ones(n_coarse_rot * n_coarse_trans, dtype=bool)
    full_mask[excluded] = False
    explicit = np.flatnonzero(full_mask).astype(np.int32)
    complement = ComplementSignificantSampleIndices(
        excluded_indices=excluded,
        total_size=int(full_mask.size),
    )

    common = dict(
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=0,
        oversampling_order=1,
        n_fine_trans=int(fine_translation_parent.size),
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=np.arange(n_coarse_rot, dtype=np.float32),
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
    )
    explicit_inputs = _prepare_per_image_pass2_inputs([explicit], **common)
    complement_inputs = _prepare_per_image_pass2_inputs([complement], **common)

    np.testing.assert_array_equal(
        complement_inputs["oversampled_rot_indices"][0],
        explicit_inputs["oversampled_rot_indices"][0],
    )
    np.testing.assert_array_equal(complement_inputs["parent_map"][0], explicit_inputs["parent_map"][0])
    np.testing.assert_array_equal(complement_inputs["log_prior"][0], explicit_inputs["log_prior"][0])
    np.testing.assert_array_equal(
        np.asarray(complement_inputs["candidate_mask"][0]),
        np.asarray(explicit_inputs["candidate_mask"][0]),
    )
    assert _candidate_mask_count(complement_inputs["candidate_mask"][0]) == _candidate_mask_count(
        explicit_inputs["candidate_mask"][0],
    )


def test_kclass_fine_mask_complement_matches_explicit_dense_support():
    n_rot_coarse = 3
    n_trans_coarse = 2
    n_rot_fine = 5
    n_trans_fine = 3
    rot_parent_map = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
    trans_parent_map = np.asarray([0, 1, 1], dtype=np.int64)
    excluded = np.asarray([1, 4], dtype=np.int32)
    full_mask = np.ones(n_rot_coarse * n_trans_coarse, dtype=bool)
    full_mask[excluded] = False
    explicit = np.flatnonzero(full_mask).astype(np.int32)
    complement = ComplementSignificantSampleIndices(
        excluded_indices=excluded,
        total_size=int(full_mask.size),
    )

    explicit_mask = _build_fine_grid_significance_mask(
        [explicit],
        n_rot_coarse,
        n_trans_coarse,
        n_rot_fine,
        n_trans_fine,
        rot_oversampling_factor=1,
        trans_oversampling_factor=1,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_images=1,
    )
    complement_mask = _build_fine_grid_significance_mask(
        [complement],
        n_rot_coarse,
        n_trans_coarse,
        n_rot_fine,
        n_trans_fine,
        rot_oversampling_factor=1,
        trans_oversampling_factor=1,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_images=1,
    )
    np.testing.assert_array_equal(complement_mask, explicit_mask)

    explicit_stats = _fine_support_stats(
        [[explicit]],
        n_rot_coarse=n_rot_coarse,
        n_trans_coarse=n_trans_coarse,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_rot_fine=n_rot_fine,
        n_trans_fine=n_trans_fine,
    )
    complement_stats = _fine_support_stats(
        [[complement]],
        n_rot_coarse=n_rot_coarse,
        n_trans_coarse=n_trans_coarse,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_rot_fine=n_rot_fine,
        n_trans_fine=n_trans_fine,
    )
    assert complement_stats == explicit_stats


def test_compact_pair_filter_routes_complement_masks_to_rectangular():
    mask = SparseCandidateMask(
        mode="coarse_exclude",
        n_rows=3,
        n_fine_trans=2,
        parent_map=np.asarray([0, 1, 2], dtype=np.int32),
        coarse_excluded=np.asarray([1], dtype=np.int32),
        fine_translation_parent=np.asarray([0, 1], dtype=np.int32),
        count=5,
    )
    filtered, excluded = _compact_pair_execution_mask_excluding_full_support(
        [{"candidate_mask": [mask]}],
        None,
    )

    assert excluded == 1
    np.testing.assert_array_equal(filtered, np.asarray([False]))


def test_compact_pair_xhalf_gpu_matches_rectangular_fused(monkeypatch):
    """GPU-only guard for compact-pair parity in RELION x-half M-step mode."""

    if os.environ.get("RECOVAR_RUN_CUDA_XHALF_TEST") != "1":
        pytest.skip("set RECOVAR_RUN_CUDA_XHALF_TEST=1 to run the CUDA x-half compact-pair guard")

    import jax

    import recovar.cuda_backproject as cb
    from recovar.em.sampling import rotation_grid_size

    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("CUDA x-half compact-pair guard requires a JAX GPU device")
    if not cb.cuda_available():
        pytest.skip(cb.cuda_unavailable_error())

    n_images = 5
    n_classes = 2
    n_coarse_rot = rotation_grid_size(1)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]],
        dtype=np.float32,
    )
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    coarse_translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    significant_by_class = [
        [
            np.asarray([0, 1, 4, 7], dtype=np.int32),
            np.asarray([2, 3, 6], dtype=np.int32),
            np.asarray([8, 9], dtype=np.int32),
            np.asarray([0, 5, 10], dtype=np.int32),
            np.asarray([1, 2, 11], dtype=np.int32),
        ],
        [
            np.asarray([0, 2, 5], dtype=np.int32),
            np.asarray([1, 3, 7], dtype=np.int32),
            np.asarray([4, 8], dtype=np.int32),
            np.asarray([3, 9, 11], dtype=np.int32),
            np.asarray([0, 6], dtype=np.int32),
        ],
    ]
    ds = MockDataset(n_images=n_images, seed=91)
    volumes = jnp.stack(
        [
            _hermitian_volume(VOLUME_SHAPE, seed=101),
            _hermitian_volume(VOLUME_SHAPE, seed=103),
        ],
    )
    kwargs = dict(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.asarray([0.45, 0.55], dtype=np.float64)),
        accumulate_noise=True,
        return_best_pose_details=True,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={
            "current_size": None,
            "relion_half_volume_mstep": False,
            "mstep_relion_x_half": True,
            "adaptive_fraction": 0.75,
        },
    )

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "3")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP", "pair_sparse")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "0")
    fused = _run_sparse_k_class_adaptive_pass2(**kwargs)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    compact_pairs = _run_sparse_k_class_adaptive_pass2(**kwargs)

    np.testing.assert_allclose(np.asarray(compact_pairs.Ft_y), np.asarray(fused.Ft_y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(compact_pairs.Ft_ctf), np.asarray(fused.Ft_ctf), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(compact_pairs.per_class_hard_assignments),
        np.asarray(fused.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(compact_pairs.class_assignments), np.asarray(fused.class_assignments))
    np.testing.assert_array_equal(np.asarray(compact_pairs.pose_assignments), np.asarray(fused.pose_assignments))
    _assert_k_class_extra_outputs_close(compact_pairs, fused)
    assert compact_pairs.profile_summary["sparse_kclass_compact_pairs"] is True
    assert compact_pairs.profile_summary["sparse_kclass_compact_pair_mstep_pair_sparse_requested"] is True
    assert compact_pairs.profile_summary["sparse_kclass_compact_pair_mstep_pair_sparse_effective"] is False
    assert compact_pairs.profile_summary["sparse_kclass_compact_pair_mstep_pair_sparse_xhalf_fallback"] is True


def test_bucketed_call_count_bounded_versus_perimage():
    """The bucketed path should make far fewer ``run_em``-style backend calls.

    We count the number of times ``_score_pass2_bucket_relion_gpu_diff2`` is invoked by the
    bucketed path: that should equal the number of buckets, much less than
    ``n_images`` (which is what the per-image reference invokes).
    """
    n_images = 24
    nside_level = 1
    rng = np.random.default_rng(13)
    n_coarse_rot = 48
    n_coarse_trans = 2
    counts = rng.integers(low=1, high=12, size=n_images)
    sig_indices = [
        (rng.choice(n_coarse_rot, size=int(c), replace=False).astype(np.int32) * n_coarse_trans).astype(np.int32)
        for c in counts
    ]

    ds = MockDataset(n_images=n_images, seed=29)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=37)
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    common_kwargs = dict(
        nside_level=nside_level,
        disc_type="linear_interp",
        oversampling_order=1,
        current_size=None,
        return_stats=True,
        accumulate_noise=False,
    )

    # Count score-bucket invocations
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod

    original_score = bucketed_mod._score_pass2_bucket_relion_gpu_diff2
    score_call_count = {"n": 0}

    def counting_score(*args, **kwargs):
        score_call_count["n"] += 1
        return original_score(*args, **kwargs)

    bucketed_mod._score_pass2_bucket_relion_gpu_diff2 = counting_score
    try:
        # Warm jit cache by running once
        compute_pass2_stats_sparse(
            ds, volume, mean_variance, noise_variance, translations, sig_indices, **common_kwargs
        )
        score_call_count["n"] = 0
        # Re-run to count the actual invocations
        compute_pass2_stats_sparse(
            ds, volume, mean_variance, noise_variance, translations, sig_indices, **common_kwargs
        )
    finally:
        bucketed_mod._score_pass2_bucket_relion_gpu_diff2 = original_score

    # The number of bucketed score calls is the number of buckets.  With
    # n_images=24 and counts in [1, 12], we expect at most ~5-6 buckets
    # (powers-of-two padding: 16, 32, 64, 128 round to a few sizes).
    # Definitely << n_images.
    assert score_call_count["n"] < n_images, (
        f"Bucketed score was called {score_call_count['n']} times for {n_images} images "
        "— expected fewer (one per bucket)."
    )
    print(f"Bucketed: {score_call_count['n']} score calls for {n_images} images")
