"""Focused CPU orchestration contracts for the certified K=1 score hybrid."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers import significance
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids,
    plan_coarse_gemm_certificate_topology,
)


def test_direct_and_hybrid_share_disk_masked_compact_projection_operands():
    source = Path(significance.__file__).read_text()
    helper_start = source.index("    def _project_relion_compact_score_rows(")
    helper_stop = source.index("\n    def _project_block(", helper_start)
    helper = source[helper_start:helper_stop]
    direct_stop = source.index("\n    coarse_selector_execution =", helper_stop)
    direct = source[helper_stop:direct_stop]
    cache_start = source.index("    def _project_coarse_gemm_rows(", direct_stop)
    cache_stop = source.index("\n    def _project_coarse_gemm_block_once(", cache_start)
    cache = source[cache_start:cache_stop]

    assert "mask_current_image_disk=True" in helper
    assert "mask_current_image_disk=False" not in helper
    assert "_project_relion_compact_score_rows(" in direct
    assert "_project_relion_compact_score_rows(" in cache


def test_coarse_gaussian_gemm_hybrid_is_default_off_and_fail_closed(monkeypatch):
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._coarse_gaussian_gemm_hybrid_enabled()
    assert significance._coarse_gaussian_gemm_hybrid_enabled(default=True)

    for disabled in ("0", "false", "no", "off"):
        monkeypatch.setenv(variable, disabled)
        assert not significance._coarse_gaussian_gemm_hybrid_enabled(default=True)
    for enabled in ("1", "true", "yes", "on"):
        monkeypatch.setenv(variable, enabled)
        assert significance._coarse_gaussian_gemm_hybrid_enabled()

    monkeypatch.setenv(variable, "automatic")
    with pytest.raises(ValueError, match=variable):
        significance._coarse_gaussian_gemm_hybrid_enabled()


def test_coarse_gaussian_gemm_compact_posterior_is_strict_default_off(
    monkeypatch,
):
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._coarse_gaussian_gemm_compact_posterior_enabled()
    assert significance._coarse_gaussian_gemm_compact_posterior_enabled(default=True)

    for disabled in ("0", "false", "no", "off"):
        monkeypatch.setenv(variable, disabled)
        assert not significance._coarse_gaussian_gemm_compact_posterior_enabled(
            default=True,
        )
    for enabled in ("1", "true", "yes", "on"):
        monkeypatch.setenv(variable, enabled)
        assert significance._coarse_gaussian_gemm_compact_posterior_enabled()

    monkeypatch.setenv(variable, "automatic")
    with pytest.raises(ValueError, match=variable):
        significance._coarse_gaussian_gemm_compact_posterior_enabled()


def test_compact_hybrid_image_batch_override_is_explicit_and_strict(monkeypatch):
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE"
    monkeypatch.delenv(variable, raising=False)
    assert significance._coarse_gaussian_gemm_hybrid_image_batch_size_request() is None

    monkeypatch.setenv(variable, "200")
    assert significance._coarse_gaussian_gemm_hybrid_image_batch_size_request() == 200

    for invalid in ("", "0", "-1", "1.5", "many"):
        monkeypatch.setenv(variable, invalid)
        with pytest.raises(ValueError, match=variable):
            significance._coarse_gaussian_gemm_hybrid_image_batch_size_request()


def test_compact_hybrid_image_batch_override_coalesces_streamed_matmul_rows():
    resolved = significance._resolve_coarse_gaussian_gemm_hybrid_image_batch_size(
        110,
        requested_batch_size=500,
        n_images=200,
        certificate_chunk_rows=4_608,
        n_translations=49,
        hybrid_enabled=True,
        compact_posterior_enabled=True,
        score_float_budget=200_000_000,
    )

    assert resolved == 200
    assert resolved * 4_608 * 49 == 45_158_400


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"hybrid_enabled": False}, "GEMM_HYBRID"),
        ({"compact_posterior_enabled": False}, "COMPACT_POSTERIOR"),
    ],
)
def test_compact_hybrid_image_batch_override_requires_compact_hybrid(
    updates,
    message,
):
    values = dict(
        requested_batch_size=8,
        n_images=8,
        certificate_chunk_rows=16,
        n_translations=3,
        hybrid_enabled=True,
        compact_posterior_enabled=True,
        score_float_budget=1_000,
    )
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        significance._resolve_coarse_gaussian_gemm_hybrid_image_batch_size(
            2,
            **values,
        )


def test_compact_hybrid_image_batch_override_fails_closed_on_streamed_tile_budget():
    with pytest.raises(MemoryError, match="240 > 239 floats"):
        significance._resolve_coarse_gaussian_gemm_hybrid_image_batch_size(
            2,
            requested_batch_size=5,
            n_images=5,
            certificate_chunk_rows=16,
            n_translations=3,
            hybrid_enabled=True,
            compact_posterior_enabled=True,
            score_float_budget=239,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"hybrid_enabled": False}, "GEMM_HYBRID"),
        ({"return_class_best": True}, "return_class_best=False"),
        ({"return_class_second": True}, "return_class_best=False"),
    ],
)
def test_compact_posterior_rejects_unsupported_runtime_contracts(updates, message):
    values = dict(
        hybrid_enabled=True,
        return_class_best=False,
        return_class_second=False,
    )
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        significance._validate_coarse_gaussian_gemm_compact_posterior_request(
            **values,
        )


def test_coarse_significance_support_audit_is_strict_default_off(monkeypatch):
    variable = "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._coarse_significance_support_audit_enabled()
    assert significance._coarse_significance_support_audit_enabled(default=True)

    for disabled in ("0", "false", "no", "off"):
        monkeypatch.setenv(variable, disabled)
        assert not significance._coarse_significance_support_audit_enabled(
            default=True,
        )
    for enabled in ("1", "true", "yes", "on"):
        monkeypatch.setenv(variable, enabled)
        assert significance._coarse_significance_support_audit_enabled()

    monkeypatch.setenv(variable, "automatic")
    with pytest.raises(ValueError, match=variable):
        significance._coarse_significance_support_audit_enabled()


def test_coarse_support_exact_ids_require_a_separate_opt_in(monkeypatch):
    variable = "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._coarse_significance_support_audit_ids_enabled()
    for enabled in ("1", "true", "yes", "on"):
        monkeypatch.setenv(variable, enabled)
        assert significance._coarse_significance_support_audit_ids_enabled()
    monkeypatch.setenv(variable, "automatic")
    with pytest.raises(ValueError, match=variable):
        significance._coarse_significance_support_audit_ids_enabled()


def test_coarse_significance_support_audit_is_exact_and_localizable():
    supports = [
        [
            np.asarray([1, 3], dtype=np.int32),
            significance.ComplementSignificantSampleIndices(
                excluded_indices=np.asarray([0, 2], dtype=np.int32),
                total_size=4,
            ),
        ],
        [
            None,
            np.zeros(0, dtype=np.int32),
        ],
    ]

    first = significance._build_coarse_significance_support_audit(
        supports,
        samples_per_class=4,
        include_ids=True,
    )
    repeated = significance._build_coarse_significance_support_audit(
        supports,
        samples_per_class=4,
        include_ids=True,
    )

    assert first == repeated
    assert first["schema"] == "recovar.coarse_significance_support_audit.v2"
    assert first["support_ids_included"] is True
    assert first["classification"] == "diagnostic_only"
    assert first["n_classes"] == 2
    assert first["n_images"] == 2
    assert first["samples_per_class"] == 4
    assert first["per_class_image_selected_counts"] == [[2, 2], [4, 0]]
    assert first["per_class_image_support_ids"] == [
        [[1, 3], [1, 3]],
        [[0, 1, 2, 3], []],
    ]
    assert first["selected_count_sum"] == 8
    assert first["selected_count_min"] == 0
    assert first["selected_count_max"] == 4
    assert len(first["aggregate_support_sha256"]) == 64
    assert all(
        len(digest) == 64 for class_digests in first["per_class_image_support_sha256"] for digest in class_digests
    )

    changed = significance._build_coarse_significance_support_audit(
        [[np.asarray([1, 2]), supports[0][1]], supports[1]],
        samples_per_class=4,
        include_ids=True,
    )
    assert changed["aggregate_support_sha256"] != first["aggregate_support_sha256"]
    assert changed["per_class_image_support_sha256"][0][1] == first["per_class_image_support_sha256"][0][1]

    hashes_only = significance._build_coarse_significance_support_audit(
        supports,
        samples_per_class=4,
    )
    assert hashes_only["support_ids_included"] is False
    assert "per_class_image_support_ids" not in hashes_only


@pytest.mark.parametrize(
    "supports",
    [
        [[np.asarray([2, 1], dtype=np.int32)]],
        [[np.asarray([1, 1], dtype=np.int32)]],
        [[np.asarray([-1], dtype=np.int32)]],
        [[np.asarray([4], dtype=np.int32)]],
    ],
)
def test_coarse_significance_support_audit_rejects_noncanonical_ids(supports):
    with pytest.raises(ValueError, match="strictly increasing in-range"):
        significance._build_coarse_significance_support_audit(
            supports,
            samples_per_class=4,
        )


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_coarse_gaussian_gemm_hybrid_capacity_requires_positive_integer(
    monkeypatch,
    value,
):
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY"
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        significance._coarse_gaussian_gemm_hybrid_block_capacity()


def _hybrid_request_kwargs(**updates):
    values = dict(
        macro_enabled=True,
        projection_cache_enabled=True,
        n_classes=1,
        n_rotations=32,
        score_mode="gaussian",
        coarse_gaussian_ffi_enabled=True,
        exact_coarse_operands_enabled=True,
        relion_f32_coarse_support_enabled=True,
        collect_significance=True,
        any_diagnostic_requested=False,
    )
    values.update(updates)
    return values


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"macro_enabled": False}, "GEMM_MACRO"),
        ({"projection_cache_enabled": False}, "PROJECTION_CACHE"),
        ({"n_classes": 2}, "K=1"),
        ({"n_rotations": 31}, "divisible by 16"),
        ({"score_mode": "normalized_cc"}, "score_mode='gaussian'"),
        ({"coarse_gaussian_ffi_enabled": False}, "exact RELION"),
        ({"exact_coarse_operands_enabled": False}, "EXACT_COARSE_OPERANDS"),
        ({"relion_f32_coarse_support_enabled": False}, "F32_COARSE_SUPPORT"),
        ({"collect_significance": False}, "collect_significance=True"),
        ({"any_diagnostic_requested": True}, "diagnostics"),
    ],
)
def test_coarse_gaussian_gemm_hybrid_rejects_incomplete_contracts(
    updates,
    message,
):
    with pytest.raises(ValueError, match=message):
        significance._validate_coarse_gaussian_gemm_hybrid_request(
            **_hybrid_request_kwargs(**updates),
        )


def _hybrid_operands():
    batch_size = 2
    n_rotations = 32
    n_translations = 3
    n_pixels = 4
    cache = np.zeros((1, n_rotations, n_pixels), dtype=np.complex64)
    shifted = np.zeros(
        (batch_size, n_translations, n_pixels),
        dtype=np.complex64,
    )
    weight = np.ones((batch_size, n_pixels), dtype=np.float32)
    initial = np.asarray([1.0, 2.0], dtype=np.float32)
    topology = plan_coarse_gemm_certificate_topology(
        np.arange(n_pixels, dtype=np.int32),
        compact_pixel_count=n_pixels,
        translation_count=n_translations,
    )
    return cache, shifted, weight, initial, topology


def _selected_diff2_from_ids(
    reference,
    shifted,
    weight,
    initial,
    block_ids,
    *,
    topology,
):
    del weight, topology
    ids = np.asarray(block_ids, dtype=np.int32)
    output = np.full(
        (shifted.shape[0], ids.shape[1], 16, shifted.shape[1]),
        np.inf,
        dtype=np.float32,
    )
    for image_index in range(shifted.shape[0]):
        for slot_index, block_id in enumerate(ids[image_index]):
            if block_id >= 0:
                assert block_id * 16 + 15 < reference.shape[0]
                output[image_index, slot_index] = np.float32(initial[image_index])
    return jnp.asarray(output)


def test_hybrid_publishes_only_selected_exact_source16_scores(monkeypatch):
    cache, shifted, weight, initial, topology = _hybrid_operands()
    monkeypatch.setattr(
        significance,
        "_relion_coarse_diff2_rotation_blocks_from_topology_f32",
        _selected_diff2_from_ids,
    )

    def reject_full_fallback(*_args, **_kwargs):
        raise AssertionError("eligible selected rescore must not run full fallback")

    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rectangular_f32",
        reject_full_fallback,
    )
    translation_prior = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)
    result = significance._compute_coarse_gaussian_gemm_hybrid_batch(
        jnp.asarray(cache),
        jnp.asarray(shifted),
        jnp.asarray(weight),
        jnp.asarray(initial),
        topology=topology,
        actual_image_count=1,
        class_log_prior=np.float64(0.25),
        translation_log_prior=translation_prior,
        certificate_chunk_rows=16,
        block_capacity=2,
    )

    assert result.used_selected_rescore
    assert result.scores_include_priors
    assert result.fallback_reason is None
    np.testing.assert_array_equal(result.selection.block_ids[0], [0, 1])
    np.testing.assert_array_equal(result.selection.block_ids[1], [-1, -1])
    expected_row = -jnp.float32(1.0) + jnp.float32(0.25) + jnp.asarray(translation_prior)[None, :]
    expected_row = np.broadcast_to(
        np.asarray(expected_row),
        (cache.shape[1], shifted.shape[1]),
    )
    np.testing.assert_array_equal(np.asarray(result.scores[0]), expected_row)
    assert np.all(np.isneginf(np.asarray(result.scores[1])))
    np.testing.assert_array_equal(
        np.asarray(result.raw_score_max),
        np.asarray([-1.0, 0.0], dtype=np.float32),
    )


def test_hybrid_compact_posterior_retains_ordered_ids_without_dense_scatter(
    monkeypatch,
):
    cache, shifted, weight, initial, topology = _hybrid_operands()
    monkeypatch.setattr(
        significance,
        "_relion_coarse_diff2_rotation_blocks_from_topology_f32",
        _selected_diff2_from_ids,
    )

    def reject_dense(*_args, **_kwargs):
        raise AssertionError("compact opt-in must not assemble a dense selected table")

    def reject_full_fallback(*_args, **_kwargs):
        raise AssertionError("eligible compact rescore must not run full fallback")

    monkeypatch.setattr(
        significance,
        "assemble_coarse_gemm_hybrid_dense_scores_f32",
        reject_dense,
    )
    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rectangular_f32",
        reject_full_fallback,
    )
    result = significance._compute_coarse_gaussian_gemm_hybrid_batch(
        jnp.asarray(cache),
        jnp.asarray(shifted),
        jnp.asarray(weight),
        jnp.asarray(initial),
        topology=topology,
        actual_image_count=1,
        class_log_prior=np.float32(0.25),
        certificate_chunk_rows=16,
        block_capacity=2,
        compact_posterior=True,
    )

    assert result.scores is None
    assert result.compact_scores is not None
    assert result.used_selected_rescore
    compact = result.compact_scores
    candidate_mask = np.isfinite(np.asarray(compact.posterior_scores_flat))
    pose_ids = map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(
        compact,
        candidate_mask,
    )
    np.testing.assert_array_equal(
        pose_ids[0],
        np.arange(cache.shape[1] * shifted.shape[1], dtype=np.int32),
    )
    assert not np.any(candidate_mask[1])
    assert pose_ids[1].size == 0
    assert compact.posterior_scores_flat.shape == (
        shifted.shape[0],
        2 * 16 * shifted.shape[1],
    )


def test_hybrid_capacity_overflow_uses_one_full_direct_batch(monkeypatch):
    cache, shifted, weight, initial, topology = _hybrid_operands()
    selected_calls = 0
    full_calls = 0

    def reject_selected(*_args, **_kwargs):
        nonlocal selected_calls
        selected_calls += 1
        raise AssertionError("ineligible selection must not run selected rescore")

    def full_direct(reference, shifted_image, score_weight, initial_diff2, mapping):
        nonlocal full_calls
        full_calls += 1
        assert reference.shape == cache.shape[1:]
        assert shifted_image.shape == shifted.shape
        assert score_weight.shape == weight.shape
        np.testing.assert_array_equal(np.asarray(initial_diff2), initial)
        np.testing.assert_array_equal(
            np.asarray(mapping),
            topology.full_to_compact,
        )
        values = np.arange(
            shifted.shape[0] * cache.shape[1] * shifted.shape[1],
            dtype=np.float32,
        ).reshape(shifted.shape[0], cache.shape[1], shifted.shape[1])
        return jnp.asarray(values + np.float32(1.0))

    monkeypatch.setattr(
        significance,
        "_relion_coarse_diff2_rotation_blocks_from_topology_f32",
        reject_selected,
    )
    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rectangular_f32",
        full_direct,
    )
    result = significance._compute_coarse_gaussian_gemm_hybrid_batch(
        jnp.asarray(cache),
        jnp.asarray(shifted),
        jnp.asarray(weight),
        jnp.asarray(initial),
        topology=topology,
        actual_image_count=2,
        class_log_prior=np.float32(0.0),
        certificate_chunk_rows=16,
        block_capacity=1,
    )

    assert not result.used_selected_rescore
    assert not result.scores_include_priors
    assert result.fallback_reason == "block_capacity_overflow"
    assert selected_calls == 0
    assert full_calls == 1
    np.testing.assert_array_equal(
        np.asarray(result.raw_score_max),
        np.asarray([-1.0, -97.0], dtype=np.float32),
    )


@pytest.mark.parametrize("compact_posterior", [False, True])
@pytest.mark.parametrize("invalid_value", [np.nan, np.float32(-1.0)])
def test_hybrid_invalid_selected_output_falls_back_for_whole_batch(
    monkeypatch,
    invalid_value,
    compact_posterior,
):
    cache, shifted, weight, initial, topology = _hybrid_operands()
    full_calls = 0

    def invalid_selected(*args, **kwargs):
        selected = np.array(
            _selected_diff2_from_ids(*args, **kwargs),
            copy=True,
        )
        selected[0, 0, 0, 0] = invalid_value
        return jnp.asarray(selected)

    def full_direct(reference, shifted_image, score_weight, initial_diff2, mapping):
        nonlocal full_calls
        del score_weight, initial_diff2, mapping
        full_calls += 1
        return jnp.full(
            (shifted_image.shape[0], reference.shape[0], shifted_image.shape[1]),
            np.float32(3.0),
        )

    monkeypatch.setattr(
        significance,
        "_relion_coarse_diff2_rotation_blocks_from_topology_f32",
        invalid_selected,
    )
    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rectangular_f32",
        full_direct,
    )
    result = significance._compute_coarse_gaussian_gemm_hybrid_batch(
        jnp.asarray(cache),
        jnp.asarray(shifted),
        jnp.asarray(weight),
        jnp.asarray(initial),
        topology=topology,
        actual_image_count=1,
        class_log_prior=np.float32(0.0),
        certificate_chunk_rows=16,
        block_capacity=2,
        compact_posterior=compact_posterior,
    )

    assert not result.used_selected_rescore
    assert result.compact_scores is None
    assert result.fallback_reason == "invalid_selected_exact_output"
    assert full_calls == 1
    np.testing.assert_array_equal(
        np.asarray(result.scores),
        np.full(result.scores.shape, -3.0, dtype=np.float32),
    )


@pytest.mark.parametrize("winning_block", [0, 1, 2])
def test_omitting_certified_zero_weight_blocks_preserves_f64_logsumexp_bits(
    winning_block,
):
    """The 138-score certificate margin is far below one float64 sum ULP."""

    block_size = 16
    translation_count = 3
    full = np.full(
        (1, 3 * block_size, translation_count),
        np.float32(-200.0),
        dtype=np.float32,
    )
    start = winning_block * block_size
    full[:, start : start + block_size] = np.linspace(
        -1.0,
        -20.0,
        block_size * translation_count,
        dtype=np.float32,
    ).reshape(1, block_size, translation_count)
    selected = full.copy()
    selected[full <= np.float32(-139.0)] = -np.inf

    def reduce_blocks(values):
        with significance.jax.enable_x64(True):
            maximum = jnp.full((1,), -jnp.inf, dtype=jnp.float32)
            total = jnp.zeros((1,), dtype=jnp.float64)
            for block_index in range(3):
                block_start = block_index * block_size
                maximum, total = significance._update_logsumexp(
                    maximum,
                    total,
                    jnp.asarray(
                        values[:, block_start : block_start + block_size],
                    ),
                )
            return np.asarray(maximum), np.asarray(total)

    full_maximum, full_total = reduce_blocks(full)
    selected_maximum, selected_total = reduce_blocks(selected)
    np.testing.assert_array_equal(selected_maximum.view(np.uint32), full_maximum.view(np.uint32))
    np.testing.assert_array_equal(selected_total.view(np.uint64), full_total.view(np.uint64))
