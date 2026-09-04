"""Focused CPU contracts for the shared coarse projection/GEMM macro."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import scoring, significance
from recovar.em.dense_single_volume.helpers.coarse_gemm_streaming import (
    COARSE_GEMM_STREAMING_SCHEMA,
)


def _macro_operands(*, real_dtype, n_images=4, n_trans=3, n_rotations=5, n_pixels=11):
    rng = np.random.default_rng(20260901)
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    projected = (
        rng.normal(size=(n_rotations, n_pixels))
        + 1j * rng.normal(size=(n_rotations, n_pixels))
    ).astype(complex_dtype)
    shifted = (
        rng.normal(size=(n_images, n_trans, n_pixels))
        + 1j * rng.normal(size=(n_images, n_trans, n_pixels))
    ).astype(complex_dtype)
    weight = rng.uniform(0.05, 2.0, size=(n_images, n_pixels)).astype(real_dtype)
    initial = rng.uniform(0.0, 4.0, size=n_images).astype(real_dtype)
    return projected, np.abs(projected) ** 2, shifted, weight, initial


def _direct_scores(projected, shifted, weight, initial):
    difference = projected[None, :, None, :] - shifted[:, None, :, :]
    return -initial[:, None, None] - 0.5 * np.sum(
        weight[:, None, None, :] * np.abs(difference) ** 2,
        axis=-1,
    )


def test_coarse_gaussian_gemm_macro_is_default_off_and_fail_closed(monkeypatch):
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._coarse_gaussian_gemm_macro_enabled()
    assert significance._coarse_gaussian_gemm_macro_enabled(default=True)

    for disabled in ("0", "false", "no", "off"):
        monkeypatch.setenv(variable, disabled)
        assert not significance._coarse_gaussian_gemm_macro_enabled(default=True)
    for enabled in ("1", "true", "yes", "on"):
        monkeypatch.setenv(variable, enabled)
        assert significance._coarse_gaussian_gemm_macro_enabled()

    monkeypatch.setenv(variable, "automatic")
    with pytest.raises(ValueError, match=variable):
        significance._coarse_gaussian_gemm_macro_enabled()


def test_exact_coarse_skip_generic_operands_is_default_off_and_fail_closed(
    monkeypatch,
):
    variable = "RECOVAR_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._k1_relion_exact_coarse_skip_generic_operands_enabled()
    assert significance._k1_relion_exact_coarse_skip_generic_operands_enabled(
        default=True,
    )

    for disabled in ("0", "false", "no", "off"):
        monkeypatch.setenv(variable, disabled)
        assert not (
            significance._k1_relion_exact_coarse_skip_generic_operands_enabled(
                default=True,
            )
        )
    for enabled in ("1", "true", "yes", "on"):
        monkeypatch.setenv(variable, enabled)
        assert significance._k1_relion_exact_coarse_skip_generic_operands_enabled()

    monkeypatch.setenv(variable, "automatic")
    with pytest.raises(ValueError, match=variable):
        significance._k1_relion_exact_coarse_skip_generic_operands_enabled()

    assert not significance._resolve_k1_relion_exact_coarse_skip_generic_operands(
        requested=False,
        exact_coarse_operands_enabled=False,
    )
    assert not significance._resolve_k1_relion_exact_coarse_skip_generic_operands(
        requested=False,
        exact_coarse_operands_enabled=True,
    )
    assert significance._resolve_k1_relion_exact_coarse_skip_generic_operands(
        requested=True,
        exact_coarse_operands_enabled=True,
    )
    with pytest.raises(ValueError, match=variable):
        significance._resolve_k1_relion_exact_coarse_skip_generic_operands(
            requested=True,
            exact_coarse_operands_enabled=False,
        )

    profile_variable = "RECOVAR_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE"
    monkeypatch.delenv(profile_variable, raising=False)
    assert not significance._k1_relion_exact_coarse_assembly_profile_enabled()
    monkeypatch.setenv(profile_variable, "1")
    assert significance._k1_relion_exact_coarse_assembly_profile_enabled()
    monkeypatch.setenv(profile_variable, "automatic")
    with pytest.raises(ValueError, match=profile_variable):
        significance._k1_relion_exact_coarse_assembly_profile_enabled()


def test_exact_compact_preprocess_is_default_off_and_fail_closed(monkeypatch):
    variable = "RECOVAR_K1_RELION_EXACT_COMPACT_PREPROCESS"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._k1_relion_exact_compact_preprocess_enabled()
    assert significance._k1_relion_exact_compact_preprocess_enabled(default=True)

    for disabled in ("0", "false", "no", "off"):
        monkeypatch.setenv(variable, disabled)
        assert not significance._k1_relion_exact_compact_preprocess_enabled(
            default=True,
        )
    for enabled in ("1", "true", "yes", "on"):
        monkeypatch.setenv(variable, enabled)
        assert significance._k1_relion_exact_compact_preprocess_enabled()

    monkeypatch.setenv(variable, "automatic")
    with pytest.raises(ValueError, match=variable):
        significance._k1_relion_exact_compact_preprocess_enabled()

    valid = {
        "exact_coarse_skip_generic_operands_enabled": True,
        "exact_coarse_operands_enabled": True,
        "coarse_gaussian_gemm_hybrid_requested": True,
        "coarse_gaussian_gemm_compact_posterior_requested": True,
        "score_mode": "gaussian",
        "any_diagnostic_requested": False,
    }
    assert not significance._resolve_k1_relion_exact_compact_preprocess(
        requested=False,
        **valid,
    )
    assert significance._resolve_k1_relion_exact_compact_preprocess(
        requested=True,
        **valid,
    )
    invalid = (
        ("exact_coarse_skip_generic_operands_enabled", False),
        ("exact_coarse_operands_enabled", False),
        ("coarse_gaussian_gemm_hybrid_requested", False),
        ("coarse_gaussian_gemm_compact_posterior_requested", False),
        ("score_mode", "normalized_cc"),
        ("any_diagnostic_requested", True),
    )
    for field, value in invalid:
        kwargs = dict(valid)
        kwargs[field] = value
        with pytest.raises(ValueError, match=variable):
            significance._resolve_k1_relion_exact_compact_preprocess(
                requested=True,
                **kwargs,
            )


def test_coarse_gaussian_gemm_projection_cache_is_default_off_and_fail_closed(
    monkeypatch,
):
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._coarse_gaussian_gemm_projection_cache_enabled()
    assert significance._coarse_gaussian_gemm_projection_cache_enabled(
        default=True,
    )

    for disabled in ("0", "false", "no", "off"):
        monkeypatch.setenv(variable, disabled)
        assert not significance._coarse_gaussian_gemm_projection_cache_enabled(
            default=True,
        )
    for enabled in ("1", "true", "yes", "on"):
        monkeypatch.setenv(variable, enabled)
        assert significance._coarse_gaussian_gemm_projection_cache_enabled()

    monkeypatch.setenv(variable, "automatic")
    with pytest.raises(ValueError, match=variable):
        significance._coarse_gaussian_gemm_projection_cache_enabled()


def _projection_cache_request_kwargs(**updates):
    values = dict(
        macro_enabled=True,
        n_classes=1,
        n_rotations=16,
        coarse_gaussian_ffi_enabled=True,
        exact_coarse_operands_enabled=True,
        use_relion_projector=True,
        relion_texture_interp_enabled=True,
        half_spectrum_scoring=True,
        use_float64_scoring=False,
        relion_projector_dtype=np.complex64,
    )
    values.update(updates)
    return values


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"macro_enabled": False}, "GEMM_MACRO"),
        ({"n_classes": 2}, "K=1"),
        ({"n_rotations": 17}, "divisible by 16"),
        ({"coarse_gaussian_ffi_enabled": False}, "exact RELION"),
        ({"exact_coarse_operands_enabled": False}, "EXACT_COARSE_OPERANDS"),
        ({"use_relion_projector": False}, "RELION texture projector"),
        ({"relion_texture_interp_enabled": False}, "RELION texture projector"),
        ({"half_spectrum_scoring": False}, "half-spectrum"),
        ({"use_float64_scoring": True}, "float32/complex64"),
        ({"relion_projector_dtype": np.complex128}, "complex64 RELION projector"),
    ],
)
def test_coarse_gaussian_gemm_projection_cache_rejects_unqualified_contracts(
    updates,
    message,
):
    with pytest.raises((ValueError, TypeError), match=message):
        significance._validate_coarse_gaussian_gemm_projection_cache_request(
            **_projection_cache_request_kwargs(**updates),
        )


def test_coarse_gaussian_gemm_projection_cache_plan_is_conservative_for_gf46(
    monkeypatch,
):
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB"
    monkeypatch.delenv(variable, raising=False)
    budget_bytes = significance._coarse_gaussian_gemm_projection_cache_budget_bytes()
    plan = significance._plan_coarse_gaussian_gemm_projection_cache(
        n_rotations=36_864,
        compact_pixel_count=5_100,
        image_shape=(128, 128),
        budget_bytes=budget_bytes,
    )

    assert budget_bytes == 4 * 1024**3
    assert plan.cache_shape == (1, 36_864, 5_100)
    assert plan.cache_dtype == np.dtype(np.complex64)
    assert plan.chunk_rows == 4_608
    assert plan.chunk_count_per_table == 8
    assert plan.retained_bytes == 1_504_051_200
    assert plan.additional_transient_bytes == 306_708_480
    assert plan.destination_copy_bytes == plan.retained_bytes
    assert plan.predicted_peak_bytes == 3_502_817_280
    assert not plan.destination_alias_proven
    assert plan.admitted
    stats = significance._coarse_gaussian_gemm_projection_cache_stats(
        plan,
        enabled=True,
    )
    assert stats["conservative_predicted_peak_bytes"] == 3_502_817_280
    assert stats["h100_alias_evidence_applies_to_plan"] is True
    assert stats["h100_observed_donated_insert_alias"] is True
    assert stats["h100_observed_alias_peak_bytes"] == 1_998_766_080
    assert stats["h100_alias_evidence_used_for_admission"] is False

    monkeypatch.setenv(variable, "3.0")
    rejected = significance._plan_coarse_gaussian_gemm_projection_cache(
        n_rotations=36_864,
        compact_pixel_count=5_100,
        image_shape=(128, 128),
        budget_bytes=(
            significance._coarse_gaussian_gemm_projection_cache_budget_bytes()
        ),
    )
    assert not rejected.admitted
    assert "exceeds budget" in rejected.admission_reason


def test_coarse_gaussian_gemm_projection_cache_reuses_exact_c64_blocks_bitwise():
    rng = np.random.default_rng(13332001)
    n_rotations = 16
    n_pixels = 7
    projected = (
        rng.normal(size=(n_rotations, n_pixels))
        + 1j * rng.normal(size=(n_rotations, n_pixels))
    ).astype(np.complex64)
    plan = significance._plan_coarse_gaussian_gemm_projection_cache(
        n_rotations=n_rotations,
        compact_pixel_count=n_pixels,
        image_shape=(4, 4),
        budget_bytes=1_000_000,
    )
    stats = significance._coarse_gaussian_gemm_projection_cache_stats(
        plan,
        enabled=True,
    )
    assert stats["h100_alias_evidence_applies_to_plan"] is False
    assert stats["h100_observed_donated_insert_alias"] is None
    assert stats["h100_observed_alias_peak_bytes"] is None
    build_calls = []

    def project_block(table_index, start, stop):
        build_calls.append((table_index, start, stop))
        return jnp.asarray(projected[start:stop])

    cache = significance._build_coarse_gaussian_gemm_projection_cache(
        plan,
        project_block,
    )
    assert build_calls == [(0, 0, 16)]

    rotations_block = np.zeros((6, 3, 3), dtype=np.float32)
    cached_projection, cached_abs2 = (
        significance._project_coarse_gaussian_gemm_projection_cache_block_once(
            cache,
            0,
            object(),
            rotations_block,
            rotation_start=3,
        )
    )
    expected_projection = projected[3:9]
    expected_abs2 = np.asarray(jnp.abs(jnp.asarray(expected_projection)) ** 2)
    np.testing.assert_array_equal(
        np.asarray(cached_projection).view(np.uint32),
        expected_projection.view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(cached_abs2).view(np.uint32),
        expected_abs2.view(np.uint32),
    )
    assert build_calls == [(0, 0, 16)]

    shifted = (
        rng.normal(size=(3, 2, n_pixels))
        + 1j * rng.normal(size=(3, 2, n_pixels))
    ).astype(np.complex64)
    pixel_weight = rng.uniform(0.1, 2.0, size=(3, n_pixels)).astype(np.float32)
    initial_diff2 = rng.uniform(0.0, 3.0, size=3).astype(np.float32)

    def uncached_projector(_class_index, _mean_for_proj, _rotations_block):
        return jnp.asarray(expected_projection), jnp.asarray(expected_abs2)

    def cached_projector(class_index, mean_for_proj, selected_rotations):
        return significance._project_coarse_gaussian_gemm_projection_cache_block_once(
            cache,
            class_index,
            mean_for_proj,
            selected_rotations,
            rotation_start=3,
        )

    score_arguments = (
        0,
        object(),
        rotations_block,
        jnp.asarray(shifted),
        jnp.asarray(pixel_weight),
        jnp.asarray(initial_diff2),
        3,
    )
    uncached_scores = np.asarray(
        significance._score_relion_coarse_gaussian_gemm_macro(
            uncached_projector,
            *score_arguments,
            image_shape=(4, 4),
            volume_shape=(4, 4, 4),
        )
    )
    cached_scores = np.asarray(
        significance._score_relion_coarse_gaussian_gemm_macro(
            cached_projector,
            *score_arguments,
            image_shape=(4, 4),
            volume_shape=(4, 4, 4),
        )
    )
    np.testing.assert_array_equal(
        cached_scores.view(np.uint32),
        uncached_scores.view(np.uint32),
    )
    assert build_calls == [(0, 0, 16)]

    tail_projection, tail_abs2 = (
        significance._project_coarse_gaussian_gemm_projection_cache_block_once(
            cache,
            0,
            object(),
            rotations_block,
            rotation_start=13,
        )
    )
    np.testing.assert_array_equal(np.asarray(tail_projection[:3]), projected[13:])
    np.testing.assert_array_equal(np.asarray(tail_projection[3:]), 0.0)
    np.testing.assert_array_equal(np.asarray(tail_abs2[3:]), 0.0)
    assert build_calls == [(0, 0, 16)]


def _backend_kwargs(**updates):
    values = dict(
        gemm_macro_requested=True,
        gemm_hybrid_requested=False,
        score_mode="gaussian",
        fused_projector_requested=False,
        fused_projector_enabled=False,
        canonical_reduction_requested=False,
        canonical_reduction_enabled=False,
        native_atomic_reduction_requested=False,
        native_atomic_reduction_enabled=False,
        single_lane_canonical_requested=False,
        single_lane_canonical_enabled=False,
        multistream_requested=False,
        multistream_enabled=False,
        native_texture_requested=False,
        native_texture_enabled=False,
    )
    values.update(updates)
    return values


def test_coarse_gaussian_gemm_backend_resolves_only_when_unambiguous():
    assert significance._resolve_coarse_gaussian_score_backend(
        **_backend_kwargs(),
    ) is significance._CoarseGaussianScoreBackend.GEMM_MACRO


def test_coarse_gaussian_gemm_backend_rejects_non_gaussian_score_mode():
    with pytest.raises(ValueError, match="score_mode='gaussian'"):
        significance._resolve_coarse_gaussian_score_backend(
            **_backend_kwargs(score_mode="normalized_cc"),
        )


@pytest.mark.parametrize(
    ("selector", "environment_name"),
    [
        ("fused_projector_requested", "RECOVAR_K1_COARSE_FUSED_PROJECTOR"),
        ("canonical_reduction_requested", "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION"),
        ("native_atomic_reduction_requested", "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION"),
        ("single_lane_canonical_requested", "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL"),
        ("multistream_requested", "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS"),
        ("native_texture_requested", "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE"),
    ],
)
def test_coarse_gaussian_gemm_backend_rejects_every_competing_selector(
    selector,
    environment_name,
):
    with pytest.raises(ValueError, match=environment_name):
        significance._resolve_coarse_gaussian_score_backend(
            **_backend_kwargs(**{selector: True}),
        )


@pytest.mark.parametrize(
    "selector",
    [
        "fused_projector_requested",
        "canonical_reduction_requested",
        "native_atomic_reduction_requested",
        "single_lane_canonical_requested",
        "multistream_requested",
    ],
)
def test_coarse_gaussian_gemm_hybrid_allows_fused_fallback_family(selector):
    assert significance._resolve_coarse_gaussian_score_backend(
        **_backend_kwargs(
            gemm_hybrid_requested=True,
            **{selector: True},
        ),
    ) is significance._CoarseGaussianScoreBackend.GEMM_MACRO


def test_coarse_gaussian_gemm_hybrid_still_rejects_native_texture():
    with pytest.raises(ValueError, match="RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE"):
        significance._resolve_coarse_gaussian_score_backend(
            **_backend_kwargs(
                gemm_hybrid_requested=True,
                native_texture_requested=True,
            ),
        )


def test_coarse_gaussian_gemm_resource_gate_records_full_transient_and_host_sync():
    resources = significance._coarse_gaussian_gemm_resources(
        rotation_block_size=17,
        image_shape=(128, 128),
        compact_pixel_count=64 * 33,
        budget_bytes=10**9,
    )
    assert resources.full_centered_projection_bytes == 17 * 128 * 65 * 8
    assert resources.compact_projection_bytes == 17 * 64 * 33 * 8
    assert resources.compact_projection_abs2_bytes == 17 * 64 * 33 * 4
    assert resources.predicted_peak_projection_bytes == (
        resources.full_centered_projection_bytes
        + resources.compact_projection_bytes
        + resources.compact_projection_abs2_bytes
    )
    assert resources.pixel_index_device_to_host_materializations == 0

    with pytest.raises(MemoryError, match="predicted projection transient"):
        significance._coarse_gaussian_gemm_resources(
            rotation_block_size=17,
            image_shape=(128, 128),
            compact_pixel_count=64 * 33,
            budget_bytes=resources.predicted_peak_projection_bytes - 1,
        )


def test_initial_model_multigroup_diagnostic_scopes_are_deterministic_and_unique():
    from recovar.em.initial_model import dense_adapter

    groups = [
        (0, np.asarray([0, 2], dtype=np.int64), None),
        (1, np.asarray([1], dtype=np.int64), None),
    ]
    first = dense_adapter._initial_model_coarse_gemm_diagnostic_scopes(
        groups,
        debug_iteration=3,
        current_size=8,
        n_classes=2,
    )
    second = dense_adapter._initial_model_coarse_gemm_diagnostic_scopes(
        groups,
        debug_iteration=3,
        current_size=8,
        n_classes=2,
    )

    assert first == second
    assert tuple(first) == (0, 1)
    assert first[0].run_id == first[1].run_id
    assert "_cs0008_" in first[0].run_id
    assert first[0].call_id != first[1].call_id
    assert "group0000_halfseth00" in first[0].call_id
    assert "group0001_halfseth01" in first[1].call_id
    assert first[0].expected_call_ids == first[1].expected_call_ids
    assert first[0].finalize is False
    assert first[1].finalize is True


def test_coarse_gemm_scope_manifests_fail_closed_on_collisions_and_duplicates(
    tmp_path,
):
    run_id = "collision_contract"
    call_ids = ("call0000_group0000_halfseth00", "call0001_group0001_halfseth01")
    first = significance.CoarseGaussianGemmDiagnosticScope(
        run_id=run_id,
        call_id=call_ids[0],
        expected_call_ids=call_ids,
        finalize=False,
    )
    significance._seal_coarse_gaussian_gemm_diagnostic_scope(
        str(tmp_path),
        scope=first,
        selection_policy="explicit_call_scope_intersection",
        requested_targets={0, 1},
        targets_in_scope={0},
        captured_target_counts={0: 1},
        artifact_paths=[str(tmp_path / "first.npz")],
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        significance._seal_coarse_gaussian_gemm_diagnostic_scope(
            str(tmp_path),
            scope=first,
            selection_policy="explicit_call_scope_intersection",
            requested_targets={0, 1},
            targets_in_scope={0},
            captured_target_counts={0: 1},
            artifact_paths=[str(tmp_path / "first.npz")],
        )

    duplicate_dir = tmp_path / "duplicate"
    duplicate_call_ids = ("call0000_a", "call0001_b")
    for call_index, call_id in enumerate(duplicate_call_ids):
        scope = significance.CoarseGaussianGemmDiagnosticScope(
            run_id="duplicate_target_contract",
            call_id=call_id,
            expected_call_ids=duplicate_call_ids,
            finalize=call_index == 1,
        )
        if call_index == 0:
            significance._seal_coarse_gaussian_gemm_diagnostic_scope(
                str(duplicate_dir),
                scope=scope,
                selection_policy="explicit_call_scope_intersection",
                requested_targets={0},
                targets_in_scope={0},
                captured_target_counts={0: 1},
                artifact_paths=[str(duplicate_dir / "a.npz")],
            )
        else:
            with pytest.raises(RuntimeError, match=r"duplicate=\[0\]"):
                significance._seal_coarse_gaussian_gemm_diagnostic_scope(
                    str(duplicate_dir),
                    scope=scope,
                    selection_policy="explicit_call_scope_intersection",
                    requested_targets={0},
                    targets_in_scope={0},
                    captured_target_counts={0: 1},
                    artifact_paths=[str(duplicate_dir / "b.npz")],
                )


def test_coarse_gemm_diagnostic_classifies_negative_implied_diff2_as_no_go(
    tmp_path,
):
    output_path = tmp_path / "negative_implied_diff2.npz"
    direct = np.zeros((1, 1, 1, 1), dtype=np.float32)
    macro = np.full((1, 1, 1, 1), np.float32(1.0e-3), dtype=np.float32)
    resources = significance._coarse_gaussian_gemm_resources(
        rotation_block_size=1,
        image_shape=(4, 4),
        compact_pixel_count=1,
        budget_bytes=10**6,
    )
    scope = significance.CoarseGaussianGemmDiagnosticScope(
        run_id="negative_implied_diff2",
        call_id="call0000_global",
        expected_call_ids=("call0000_global",),
        finalize=True,
    )
    significance._write_coarse_gaussian_gemm_diagnostic(
        str(output_path),
        direct_scores_pre_prior=direct,
        macro_scores_pre_prior=macro,
        direct_scores_with_prior=direct,
        macro_scores_with_prior=macro,
        direct_support=np.ones((1, 1), dtype=bool),
        macro_support=np.ones((1, 1), dtype=bool),
        original_indices=np.asarray([0]),
        local_indices=np.asarray([0]),
        actual_batch_size=1,
        padded_batch_size=1,
        adaptive_fraction=0.5,
        max_significants=1,
        resource_estimate=resources,
        diagnostic_scope=scope,
        diagnostic_selection_policy="strict_single_call",
        debug_iteration=0,
        current_size=4,
    )
    with np.load(output_path) as payload:
        assert str(payload["qualification_status"]).startswith("NO_GO")
        assert int(payload["macro_only_negative_implied_diff2_count"]) == 1
        reasons = set(payload["automatic_no_go_reasons"].tolist())
        assert "negative_implied_diff2" in reasons
        assert "exact-zero_cancellation_drift" in reasons


def test_coarse_gemm_qualification_allows_only_fully_qualified_stable_noise():
    stable = scoring._coarse_gaussian_qualification_decision(
        exact_arithmetic_equivalent=True,
        repeat_stable=True,
        unbiased_non_directional=True,
        bounded_non_growing=True,
        discrete_choices_equal=True,
        final_basin_quality_equal=True,
        material_runtime_win=True,
        scale_amplified=False,
        negative_implied_diff2=False,
        nonfinite_scores=False,
        exact_zero_cancellation_drift=False,
    )
    assert stable["status"] == (
        "GO_STABLE_BOUNDED_MATHEMATICALLY_EQUIVALENT_NOISE"
    )
    assert stable["requires_bitwise_score_identity"] is False
    assert stable["requires_exact_discrete_identity"] is True

    unqualified = scoring._coarse_gaussian_qualification_decision(
        exact_arithmetic_equivalent=True,
        repeat_stable=None,
        unbiased_non_directional=None,
        bounded_non_growing=None,
        discrete_choices_equal=True,
        final_basin_quality_equal=None,
        material_runtime_win=None,
        scale_amplified=None,
        negative_implied_diff2=False,
        nonfinite_scores=False,
        exact_zero_cancellation_drift=False,
    )
    assert unqualified["status"] == "NO_GO_UNQUALIFIED"
    assert "repeat_stability" in unqualified["pending_gates"]

    for bad_flag in ("scale_amplified", "negative_implied_diff2"):
        kwargs = dict(
            exact_arithmetic_equivalent=True,
            repeat_stable=True,
            unbiased_non_directional=True,
            bounded_non_growing=True,
            discrete_choices_equal=True,
            final_basin_quality_equal=True,
            material_runtime_win=True,
            scale_amplified=False,
            negative_implied_diff2=False,
            nonfinite_scores=False,
            exact_zero_cancellation_drift=False,
        )
        kwargs[bad_flag] = True
        rejected = scoring._coarse_gaussian_qualification_decision(**kwargs)
        assert rejected["status"] == "NO_GO"


def test_coarse_gaussian_gemm_scores_report_direct_objective_float32(record_property):
    """Report float32 drift without turning this sample into a tolerance."""

    operands = _macro_operands(real_dtype=np.float32)
    actual = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            *map(jnp.asarray, operands),
            operands[2].shape[0],
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
    )
    expected = _direct_scores(operands[0], operands[2], operands[3], operands[4])
    diagnostics = scoring._coarse_gaussian_direct_macro_diagnostics(
        expected[:, None, :, :],
        actual[:, None, :, :],
    )

    assert actual.shape == (4, 5, 3)
    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(
        np.argmax(actual.reshape(actual.shape[0], -1), axis=1),
        np.argmax(expected.reshape(expected.shape[0], -1), axis=1),
    )
    record_property(
        "float32_direct_macro_raw",
        json.dumps(
            {
                "precision_bits": int(diagnostics["score_precision_bits"]),
                "signed_mean_delta": diagnostics[
                    "signed_mean_delta_per_image"
                ].tolist(),
                "max_abs_delta": diagnostics["max_abs_delta_per_image"].tolist(),
                "max_ulp_delta": int(np.max(diagnostics["ulp_score_delta"])),
            },
            sort_keys=True,
        ),
    )


def test_coarse_gaussian_gemm_scores_report_direct_objective_float64(record_property):
    """Report the float64 companion without inventing a promotion epsilon."""

    operands = _macro_operands(real_dtype=np.float64)
    actual = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            *map(jnp.asarray, operands),
            operands[2].shape[0],
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
    )
    expected = _direct_scores(operands[0], operands[2], operands[3], operands[4])
    diagnostics = scoring._coarse_gaussian_direct_macro_diagnostics(
        expected[:, None, :, :],
        actual[:, None, :, :],
    )

    assert actual.dtype == np.float64
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(
        np.argmax(actual.reshape(actual.shape[0], -1), axis=1),
        np.argmax(expected.reshape(expected.shape[0], -1), axis=1),
    )
    record_property(
        "float64_direct_macro_raw",
        json.dumps(
            {
                "precision_bits": int(diagnostics["score_precision_bits"]),
                "signed_mean_delta": diagnostics[
                    "signed_mean_delta_per_image"
                ].tolist(),
                "max_abs_delta": diagnostics["max_abs_delta_per_image"].tolist(),
                "max_ulp_delta": int(np.max(diagnostics["ulp_score_delta"])),
            },
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
def test_coarse_gaussian_gemm_direct_square_cancellation_stress_equal_operands(
    real_dtype,
    record_property,
):
    """p == s records cancellation behavior but cannot qualify the macro."""

    if real_dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(13290001)
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    unit_projected = (
        rng.normal(size=(1, 4096)) + 1j * rng.normal(size=(1, 4096))
    ).astype(complex_dtype)
    unit_weight = rng.uniform(0.5, 1.5, size=(1, 4096)).astype(real_dtype)
    records = []
    score_deltas = []
    nonzero_macro_by_scale = []
    for scale in (1.0, 100.0, 1.0e4):
        projected = np.asarray(scale, dtype=real_dtype) * unit_projected
        shifted = projected[None, :, :].copy()
        initial = np.zeros(1, dtype=real_dtype)
        macro = np.asarray(
            scoring._relion_coarse_gaussian_gemm_scores(
                jnp.asarray(projected),
                jnp.asarray(np.abs(projected) ** 2, dtype=real_dtype),
                jnp.asarray(shifted),
                jnp.asarray(unit_weight),
                jnp.asarray(initial),
                1,
                image_shape=(128, 128),
                volume_shape=(128, 128, 128),
            )
        )
        direct = _direct_scores(projected, shifted, unit_weight, initial)
        diagnostics = scoring._coarse_gaussian_direct_macro_diagnostics(
            direct[:, None, :, :],
            macro[:, None, :, :],
        )
        np.testing.assert_array_equal(direct, np.zeros_like(direct))
        assert np.all(np.isfinite(macro))
        nonzero_macro = bool(np.any(macro != 0.0))
        nonzero_macro_by_scale.append(nonzero_macro)
        np.testing.assert_array_equal(
            diagnostics["exact_zero_direct_nonzero_macro_per_image"],
            nonzero_macro,
        )
        score_deltas.append(diagnostics["score_delta"])
        records.append(
            {
                "classification": (
                    "NO_GO_observed_cancellation_drift"
                    if nonzero_macro
                    else "NO_GO_inconclusive_exact_equal_fixture"
                ),
                "operand_scale": scale,
                "precision_bits": int(diagnostics["score_precision_bits"]),
                "signed_mean_score_delta": float(
                    diagnostics["signed_mean_delta_per_image"][0]
                ),
                "max_abs_score_delta": float(
                    diagnostics["max_abs_delta_per_image"][0]
                ),
                "max_ulp_score_delta": int(np.max(diagnostics["ulp_score_delta"])),
                "positive_delta_count": int(
                    diagnostics["positive_delta_count_per_image"][0]
                ),
                "negative_delta_count": int(
                    diagnostics["negative_delta_count_per_image"][0]
                ),
                "negative_implied_diff2_count": int(np.count_nonzero(macro > 0.0)),
            }
        )
    scale_panel = scoring._coarse_gaussian_scale_panel_diagnostics(
        [1.0, 100.0, 1.0e4],
        np.stack(score_deltas, axis=0),
        precision_bits=np.dtype(real_dtype).itemsize * 8,
    )
    # XLA may contract or simplify this exact-equality fixture so that every
    # expanded-square score is exactly zero.  That is mathematically valid but
    # supplies no qualification evidence.  When drift is observed, classify
    # the scale panel from the observations instead of requiring a particular
    # backend/lowering artifact.
    scale_amplified = bool(scale_panel["scale_amplified"])
    expected_status = (
        "NO_GO_scale-amplified_drift"
        if scale_amplified
        else "NO_GO_unqualified_non-growing_scale_panel"
    )
    assert str(scale_panel["qualification_status"]) == expected_status
    if not any(nonzero_macro_by_scale):
        np.testing.assert_array_equal(
            np.stack(score_deltas, axis=0),
            np.zeros_like(np.stack(score_deltas, axis=0)),
        )
    record_property("cancellation_records", json.dumps(records, sort_keys=True))
    assert all(record["classification"].startswith("NO_GO") for record in records)


@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
def test_coarse_gaussian_gemm_direct_square_cancellation_stress_nearby_operands(
    real_dtype,
    record_property,
):
    """p≈s reports raw drift and cannot establish a promotion tolerance."""

    if real_dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(13290002)
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    projected = (
        100.0
        * (
            rng.normal(size=(3, 4096))
            + 1j * rng.normal(size=(3, 4096))
        )
    ).astype(complex_dtype)
    perturbation_scale = 1.0e-4 if real_dtype == np.float32 else 1.0e-8
    shifted = (
        projected[None, :1, :]
        + perturbation_scale
        * 100.0
        * (
            rng.normal(size=(1, 2, 4096))
            + 1j * rng.normal(size=(1, 2, 4096))
        )
    ).astype(complex_dtype)
    weight = rng.uniform(0.5, 1.5, size=(1, 4096)).astype(real_dtype)
    initial = np.zeros(1, dtype=real_dtype)
    macro = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            jnp.asarray(projected),
            jnp.asarray(np.abs(projected) ** 2, dtype=real_dtype),
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(initial),
            1,
            image_shape=(128, 128),
            volume_shape=(128, 128, 128),
        )
    )
    direct = _direct_scores(projected, shifted, weight, initial)
    delta = macro - direct
    diagnostics = scoring._coarse_gaussian_direct_macro_diagnostics(
        direct[:, None, :, :],
        macro[:, None, :, :],
    )

    assert np.all(np.isfinite(macro))
    assert np.any(delta > 0.0) or np.any(delta < 0.0)
    record_property(
        "nearby_cancellation_record",
        json.dumps(
            {
                "classification": "NO_GO_pending_production_repeat_envelope",
                "operand_scale": 100.0,
                "precision_bits": int(diagnostics["score_precision_bits"]),
                "signed_mean_score_delta": float(
                    diagnostics["signed_mean_delta_per_image"][0]
                ),
                "max_abs_score_delta": float(
                    diagnostics["max_abs_delta_per_image"][0]
                ),
                "max_ulp_score_delta": int(np.max(diagnostics["ulp_score_delta"])),
                "positive_delta_count": int(
                    diagnostics["positive_delta_count_per_image"][0]
                ),
                "negative_delta_count": int(
                    diagnostics["negative_delta_count_per_image"][0]
                ),
            },
            sort_keys=True,
        ),
    )


def test_coarse_gaussian_direct_macro_diagnostics_preserve_layout_and_discretes():
    direct = np.array(
        [[[[3.0, 1.0], [2.0, 0.0]]], [[[1.0, 4.0], [3.0, 2.0]]]],
        dtype=np.float32,
    )
    macro = direct.copy()
    macro[1, 0, 1, 0] = 5.0
    direct_support = np.array([[True, False, True, False], [False, True, True, False]])
    macro_support = direct_support.copy()
    macro_support[1, 2] = False
    diagnostics = scoring._coarse_gaussian_direct_macro_diagnostics(
        direct,
        macro,
        direct_support=direct_support,
        macro_support=macro_support,
    )

    assert diagnostics["score_delta"].shape == direct.shape
    assert diagnostics["absolute_score_delta"].shape == direct.shape
    assert diagnostics["relative_score_delta_to_direct"].shape == direct.shape
    assert diagnostics["ulp_score_delta"].shape == direct.shape
    assert int(diagnostics["score_precision_bits"]) == 32
    np.testing.assert_array_equal(
        diagnostics["positive_delta_count_per_image"]
        + diagnostics["negative_delta_count_per_image"]
        + diagnostics["zero_delta_count_per_image"],
        np.full(2, 4),
    )
    np.testing.assert_array_equal(diagnostics["argmax_equal"], [True, False])
    np.testing.assert_array_equal(diagnostics["support_equal"], [True, False])
    np.testing.assert_array_equal(
        diagnostics["support_symmetric_difference_count"],
        [0, 1],
    )


def test_coarse_gaussian_diagnostics_report_exact_ulp_and_repeat_spread():
    direct = np.asarray([0.0, 1.0, -1.0], dtype=np.float32).reshape(1, 1, 1, 3)
    macro = np.asarray(
        [
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.nextafter(np.float32(1.0), np.float32(2.0)),
            np.nextafter(np.float32(-1.0), np.float32(-2.0)),
        ],
        dtype=np.float32,
    ).reshape(1, 1, 1, 3)
    diagnostics = scoring._coarse_gaussian_direct_macro_diagnostics(direct, macro)
    np.testing.assert_array_equal(diagnostics["ulp_score_delta"], 1)

    repeat_deltas = np.stack(
        [
            diagnostics["score_delta"],
            diagnostics["score_delta"] * 2.0,
            diagnostics["score_delta"] * -1.0,
        ],
        axis=0,
    )
    repeat = scoring._coarse_gaussian_repeat_spread_diagnostics(repeat_deltas)
    assert int(repeat["repeat_count"]) == 3
    np.testing.assert_array_equal(
        repeat["elementwise_delta_repeat_spread"],
        np.ptp(repeat_deltas, axis=0),
    )
    assert str(repeat["qualification_status"]).startswith("NO_GO")


def test_coarse_gaussian_gemm_scores_ignore_poisoned_tail_exactly():
    projected, projected_abs2, shifted, weight, initial = _macro_operands(
        real_dtype=np.float32,
        n_images=5,
    )
    actual_image_count = 3
    clean_shifted = shifted.copy()
    clean_weight = weight.copy()
    clean_initial = initial.copy()
    clean_shifted[actual_image_count:] = 0
    clean_weight[actual_image_count:] = 0
    clean_initial[actual_image_count:] = 0
    poisoned_shifted = clean_shifted.copy()
    poisoned_weight = clean_weight.copy()
    poisoned_initial = clean_initial.copy()
    poisoned_shifted[actual_image_count:] = np.complex64(np.nan + 1j * np.nan)
    poisoned_weight[actual_image_count:] = np.nan
    poisoned_initial[actual_image_count:] = np.nan

    common = (
        jnp.asarray(projected),
        jnp.asarray(projected_abs2),
    )
    clean = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            *common,
            jnp.asarray(clean_shifted),
            jnp.asarray(clean_weight),
            jnp.asarray(clean_initial),
            actual_image_count,
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
    )
    poisoned = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            *common,
            jnp.asarray(poisoned_shifted),
            jnp.asarray(poisoned_weight),
            jnp.asarray(poisoned_initial),
            actual_image_count,
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
    )

    np.testing.assert_array_equal(
        poisoned[:actual_image_count].view(np.uint32),
        clean[:actual_image_count].view(np.uint32),
    )
    np.testing.assert_array_equal(
        poisoned[actual_image_count:].view(np.uint32),
        np.zeros_like(poisoned[actual_image_count:]).view(np.uint32),
    )


def test_coarse_gaussian_gemm_macro_projects_once_and_binds_all_image_lanes(
    monkeypatch,
):
    projected, projected_abs2, shifted, weight, initial = _macro_operands(
        real_dtype=np.float32,
        n_images=6,
        n_rotations=7,
    )
    rotations = np.arange(7 * 3 * 3, dtype=np.float32).reshape(7, 3, 3)
    mean = object()
    projection_calls = []
    score_calls = []

    def project_once(class_index, mean_for_proj, rotations_block):
        projection_calls.append((class_index, mean_for_proj, np.asarray(rotations_block)))
        return jnp.asarray(projected), jnp.asarray(projected_abs2)

    sentinel = jnp.arange(6 * 7 * 3, dtype=jnp.float32).reshape(6, 7, 3)

    def capture_score(*args, **kwargs):
        score_calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(
        significance,
        "_relion_coarse_gaussian_gemm_scores",
        capture_score,
    )
    result = significance._score_relion_coarse_gaussian_gemm_macro(
        project_once,
        2,
        mean,
        rotations,
        jnp.asarray(shifted),
        jnp.asarray(weight),
        jnp.asarray(initial),
        4,
        image_shape=(8, 8),
        volume_shape=(8, 8, 8),
    )

    assert result is sentinel
    assert len(projection_calls) == 1
    assert projection_calls[0][0] == 2
    assert projection_calls[0][1] is mean
    np.testing.assert_array_equal(projection_calls[0][2], rotations)
    assert len(score_calls) == 1
    bound, static = score_calls[0]
    np.testing.assert_array_equal(np.asarray(bound[0]), projected)
    np.testing.assert_array_equal(np.asarray(bound[1]), projected_abs2)
    np.testing.assert_array_equal(np.asarray(bound[2]), shifted)
    np.testing.assert_array_equal(np.asarray(bound[3]), weight)
    np.testing.assert_array_equal(np.asarray(bound[4]), initial)
    assert bound[5] == 4
    assert static == {"image_shape": (8, 8), "volume_shape": (8, 8, 8)}


def test_coarse_gaussian_gemm_macro_is_shared_by_em_and_initial_model():
    from recovar.em.dense_single_volume import k_class
    from recovar.em.initial_model import dense_adapter

    assert (
        dense_adapter._compute_k_class_significance_batched
        is significance._compute_k_class_significance_batched
    )
    assert (
        "_compute_k_class_significance_batched"
        in k_class.run_dense_k_class_em_adaptive.__code__.co_names
    )
    assert (
        scoring._relion_coarse_gaussian_gemm_scores_jit._fun.__globals__[
            "_e_step_block_scores_windowed"
        ]
        is scoring._e_step_block_scores_windowed
    )


class _MacroIntegrationDataset:
    """Tiny strict-preprocess dataset for the live shared significance path."""

    image_shape = (4, 4)
    image_size = 16
    grid_size = 4
    padding = 0
    volume_shape = (4, 4, 4)
    volume_size = 64
    voxel_size = 1.0
    dtype = jnp.complex64
    premultiplied_ctf = False

    def __init__(self, original_indices=None):
        self._original_indices = np.asarray(
            [0, 1, 2] if original_indices is None else original_indices,
            dtype=np.int64,
        )
        self.n_images = int(self._original_indices.size)
        self.n_units = self.n_images
        self._images = np.stack(
            [
                np.full(self.image_shape, int(original_index) + 1, dtype=np.float32)
                for original_index in self._original_indices
            ]
        )
        self.CTF_params = np.zeros((self.n_units, 9), dtype=np.float32)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (self.n_units, 1, 1))
        self.translations = np.zeros((self.n_units, 2), dtype=np.float32)

        class _Backend:
            image_mask = np.ones((4, 4), dtype=np.float32)
            image_mask_mode = "relion_background_fill"
            relion_fourier_backend = "relion_cuda"

        class _ImageSource:
            backend = _Backend()

        self.image_source = _ImageSource()

    @staticmethod
    def ctf_evaluator(params, image_shape=None, voxel_size=None, *, half_image=False):
        del voxel_size
        if half_image:
            pixel_count = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
        else:
            pixel_count = int(image_shape[0]) * int(image_shape[1])
        return jnp.ones((params.shape[0], pixel_count), dtype=jnp.float32)

    @staticmethod
    def process_images(batch, apply_image_mask=False, **kwargs):
        del apply_image_mask, kwargs
        batch = jnp.asarray(batch)
        return jnp.repeat(
            batch[:, :1, :1].reshape(batch.shape[0], 1).astype(jnp.complex64),
            16,
            axis=1,
        )

    @staticmethod
    def process_images_half(batch, apply_image_mask=False, **kwargs):
        del apply_image_mask, kwargs
        batch = jnp.asarray(batch)
        return jnp.repeat(
            batch[:, :1, :1].reshape(batch.shape[0], 1).astype(jnp.complex64),
            12,
            axis=1,
        )

    @property
    def image_mask(self):
        return np.ones(self.image_shape, dtype=np.float32)

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        del by_image, kwargs
        if indices is None:
            indices = np.arange(self.n_units)
        indices = np.asarray(indices, dtype=np.int64)
        for start in range(0, indices.size, int(batch_size)):
            selected = indices[start : start + int(batch_size)]
            yield (
                jnp.asarray(self._images[selected]),
                self.rotation_matrices[selected],
                self.translations[selected],
                jnp.asarray(self.CTF_params[selected]),
                None,
                selected,
                selected,
            )

    def original_image_indices_from_local(self, indices):
        return self._original_indices[np.asarray(indices, dtype=np.int64)]

    def subset(self, indices):
        return type(self)(
            self._original_indices[np.asarray(indices, dtype=np.int64)],
        )


def test_coarse_gaussian_gemm_live_k1_cache_builds_once_outside_image_loop(
    monkeypatch,
):
    """The opt-in cache owns projections once and only serves later blocks."""

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers import projection as projection_helpers
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed

    for name, value in {
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE": "0",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB": "0.001",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB": "0.01",
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI": "1",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF": "1",
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR": "0",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "0",
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "0",
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE": "0",
        "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS": "1",
        "RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "0",
    }.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(significance.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "cuda_available", lambda: True)
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_exact_ctf_half_from_source_star",
        lambda _dataset, indices, image_shape: jnp.ones(
            (
                len(indices),
                int(image_shape[0]) * (int(image_shape[1]) // 2 + 1),
            ),
            dtype=jnp.float64,
        ),
    )
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_exact_ctf_half_from_source_star_host",
        lambda _dataset, indices, image_shape: np.ones(
            (
                len(indices),
                int(image_shape[0]) * (int(image_shape[1]) // 2 + 1),
            ),
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_cuda_powerclass_highres_xi2_half",
        lambda processed, **_kwargs: jnp.zeros(
            processed.shape[0],
            dtype=jnp.float32,
        ),
    )
    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f32",
        lambda images, translation_angles, pixel_indices, image_shape: jnp.repeat(
            images[:, None, :],
            int(translation_angles.shape[0]),
            axis=1,
        ).reshape(images.shape[0] * int(translation_angles.shape[0]), -1),
    )

    projection_calls = []

    def fake_projection(projector_half, rotations_block, image_shape, **kwargs):
        del projector_half, image_shape
        rotation_codes = np.asarray(rotations_block)[:, 0, 1].astype(np.float32)
        projection_calls.append(
            (
                rotation_codes.copy(),
                bool(kwargs.get("return_abs2", True)),
            )
        )
        projected = jnp.repeat(
            jnp.asarray(rotation_codes, dtype=jnp.complex64)[:, None],
            len(kwargs["pixel_indices"]),
            axis=1,
        )
        projected_abs2 = (
            jnp.abs(projected) ** 2
            if kwargs.get("return_abs2", True)
            else None
        )
        return projected, projected_abs2

    monkeypatch.setattr(
        projection_helpers,
        "compute_relion_projector_projections_block",
        fake_projection,
    )

    def controlled_scores(
        projected,
        projected_abs2,
        shifted,
        weight,
        initial,
        actual_image_count,
        **_kwargs,
    ):
        del projected_abs2, weight, initial, actual_image_count
        codes = jnp.asarray(projected.real[:, 0], dtype=jnp.float32)
        return jnp.broadcast_to(
            codes[None, :, None],
            (shifted.shape[0], projected.shape[0], shifted.shape[1]),
        )

    monkeypatch.setattr(
        significance,
        "_relion_coarse_gaussian_gemm_scores",
        controlled_scores,
    )

    dataset = _MacroIntegrationDataset()
    rotations = np.tile(np.eye(3, dtype=np.float32), (16, 1, 1))
    rotations[:, 0, 1] = np.arange(1, 17, dtype=np.float32)
    translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    relion_projector = jnp.ones((1, 3, 3, 2), dtype=jnp.complex64)
    common = dict(
        class_log_priors=np.zeros(1, dtype=np.float64),
        adaptive_fraction=0.5,
        max_significants=1,
        image_batch_size=2,
        rotation_block_size=6,
        current_size=4,
        half_spectrum_scoring=True,
        relion_projector_half=relion_projector,
        relion_projector_r_max=1,
        relion_projector_texture_interp=True,
        score_mode="gaussian",
        collect_significance=False,
        pad_final_image_batch=True,
    )

    def run():
        return significance._compute_k_class_significance_batched(
            dataset,
            jnp.zeros((1, dataset.volume_size), dtype=jnp.complex64),
            jnp.ones(dataset.image_size, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            **common,
        )

    uncached = run()
    uncached_calls = tuple(projection_calls)
    assert len(uncached_calls) == 6
    assert all(returned_abs2 for _codes, returned_abs2 in uncached_calls)

    projection_calls.clear()
    monkeypatch.setenv("RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE", "1")
    cached = run()
    assert len(projection_calls) == 1
    np.testing.assert_array_equal(projection_calls[0][0], np.arange(1, 17))
    assert projection_calls[0][1] is False

    for key in (
        "normalization_log_z",
        "normalization_log_evidence",
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "class_log_evidence_per_image",
        "class_assignments",
    ):
        np.testing.assert_array_equal(cached[5][key], uncached[5][key])
    np.testing.assert_array_equal(cached[2], uncached[2])
    np.testing.assert_array_equal(cached[3], uncached[3])
    cache_stats = cached[5]["coarse_gaussian_gemm_projection_cache"]
    assert cache_stats["enabled"] is True
    assert cache_stats["cache_shape"] == (1, 16, 12)
    assert cache_stats["stores_projection_abs2"] is False
    assert cache_stats["h100_alias_evidence_applies_to_plan"] is False
    assert cache_stats["h100_observed_donated_insert_alias"] is None


@pytest.mark.parametrize(
    (
        "compact_posterior",
        "hybrid_image_batch_size",
        "force_fallback",
        "fused_fallback",
    ),
    [
        (False, None, False, False),
        (True, None, False, False),
        (True, 3, False, False),
        (True, 3, True, False),
        (True, None, True, False),
        (True, None, False, True),
        (True, None, True, True),
    ],
)
def test_live_k1_hybrid_reuses_exact_scores_in_both_significance_passes(
    monkeypatch,
    compact_posterior,
    hybrid_image_batch_size,
    force_fallback,
    fused_fallback,
):
    """Selected and exact-full-direct hybrid scores are reused in both passes."""

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers import (
        oversampling,
        preprocessing,
        sparse_pass2_bucketed,
    )
    from recovar.em.dense_single_volume.helpers import projection as projection_helpers
    from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
        CoarseGemmHybridBlockSelection,
        CoarseGemmHybridCompactScores,
    )

    for name, value in {
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR": (
            "1" if compact_posterior else "0"
        ),
        "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB": "0.001",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB": "0.01",
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI": "1",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF": "1",
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR": "1" if fused_fallback else "0",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "0",
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "0",
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE": "0",
        "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS": "1",
        "RECOVAR_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS": "0",
        "RECOVAR_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE": "1",
        "RECOVAR_K1_RELION_EXACT_COMPACT_PREPROCESS": "0",
        "RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "1",
        "RECOVAR_SIGNIFICANCE_SCORE_CACHE": "0",
        "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(
        "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE",
        raising=False,
    )
    if hybrid_image_batch_size is not None:
        monkeypatch.setenv(
            "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE",
            str(hybrid_image_batch_size),
        )

    monkeypatch.setattr(significance.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "cuda_available", lambda: True)
    fused_calls = []

    def fake_fused_projector(
        projector_full,
        rotations_block,
        images,
        translation_angles,
        score_weight,
        initial_diff2,
        full_to_compact,
        **kwargs,
    ):
        fused_calls.append(
            {
                "rotation_count": int(rotations_block.shape[0]),
                "actual_image_count": int(images.shape[0]),
                "lookup": np.asarray(full_to_compact).copy(),
                "kwargs": dict(kwargs),
            }
        )
        assert projector_full.dtype == jnp.complex64
        assert images.dtype == jnp.complex64
        assert translation_angles.dtype == jnp.float32
        assert score_weight.dtype == jnp.float32
        assert initial_diff2.dtype == jnp.float32
        return jnp.zeros(
            (
                int(images.shape[0]),
                int(rotations_block.shape[0]),
                int(translation_angles.shape[0]),
            ),
            dtype=jnp.float32,
        )

    fake_fused_projector.__name__ = "relion_coarse_diff2_projector_f32"
    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_projector_f32",
        fake_fused_projector,
    )
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_exact_ctf_half_from_source_star",
        lambda _dataset, indices, image_shape: jnp.ones(
            (
                len(indices),
                int(image_shape[0]) * (int(image_shape[1]) // 2 + 1),
            ),
            dtype=jnp.float64,
        ),
    )
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_exact_ctf_half_from_source_star_host",
        lambda _dataset, indices, image_shape: np.ones(
            (
                len(indices),
                int(image_shape[0]) * (int(image_shape[1]) // 2 + 1),
            ),
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_cuda_powerclass_highres_xi2_half",
        lambda processed, **_kwargs: jnp.zeros(
            processed.shape[0],
            dtype=jnp.float32,
        ),
    )
    original_process_half_image = preprocessing.process_half_image
    process_calls = []

    def tracked_process_half_image(*args, **kwargs):
        processed = original_process_half_image(*args, **kwargs)
        relion_kwargs = kwargs.get("relion_preprocess_kwargs")
        process_calls.append(
            (
                np.asarray(args[1]).copy(),
                bool(
                    relion_kwargs is not None
                    and relion_kwargs.get("relion_fft_per_image", False)
                ),
                np.asarray(processed).copy(),
            )
        )
        return processed

    monkeypatch.setattr(
        preprocessing,
        "process_half_image",
        tracked_process_half_image,
    )
    translation_calls = []

    def fake_translate(images, translation_angles, pixel_indices, image_shape):
        translation_calls.append(
            (
                np.asarray(images).copy(),
                np.asarray(translation_angles).copy(),
                np.asarray(pixel_indices).copy(),
                tuple(image_shape),
            )
        )
        return jnp.repeat(
            images[:, None, :],
            int(translation_angles.shape[0]),
            axis=1,
        ).reshape(images.shape[0] * int(translation_angles.shape[0]), -1)

    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f32",
        fake_translate,
    )

    projection_calls = []

    def fake_projection(projector_half, rotations_block, image_shape, **kwargs):
        del projector_half, image_shape
        projection_calls.append(
            (len(rotations_block), bool(kwargs.get("return_abs2", True))),
        )
        projected = jnp.zeros(
            (len(rotations_block), len(kwargs["pixel_indices"])),
            dtype=jnp.complex64,
        )
        return projected, None

    monkeypatch.setattr(
        projection_helpers,
        "compute_relion_projector_projections_block",
        fake_projection,
    )

    helper_outputs = []
    operand_inputs = []
    static_overflow_requests = []
    compact_expected = compact_posterior

    def fake_hybrid(
        projection_cache,
        shifted_corrected,
        pixel_weight,
        initial_diff2,
        *,
        topology,
        actual_image_count,
        class_log_prior,
        rotation_log_prior,
        translation_log_prior,
        certificate_chunk_rows,
        block_capacity,
        compact_posterior: bool,
        force_static_dense_after_overflow: bool,
        logical_full_pixel_count,
        capture_selected_diff2: bool,
        full_dense_diff2_fn,
    ):
        batch_size = int(shifted_corrected.shape[0])
        static_overflow_requests.append(force_static_dense_after_overflow)
        operand_inputs.append(
            (
                np.asarray(shifted_corrected).copy(),
                np.asarray(pixel_weight).copy(),
                np.asarray(initial_diff2).copy(),
            )
        )
        assert projection_cache.shape == (1, 16, 12)
        assert topology.compact_pixel_count == 12
        assert topology.translation_count == 2
        assert actual_image_count in ((3,) if hybrid_image_batch_size else (1, 2))
        assert np.float32(class_log_prior) == np.float32(0.25)
        assert rotation_log_prior is None
        assert translation_log_prior.shape == (batch_size, 2)
        assert certificate_chunk_rows == 16
        assert block_capacity == 64
        assert compact_posterior is compact_expected
        assert logical_full_pixel_count is None
        assert capture_selected_diff2 is False
        assert (full_dense_diff2_fn is not None) is fused_fallback

        scores = np.full((batch_size, 16, 2), -100.0, dtype=np.float32)
        scores[:actual_image_count, 5, 1] = np.float32(3.0)
        scores[actual_image_count:] = -np.inf
        block_ids = np.full((batch_size, block_capacity), -1, dtype=np.int32)
        block_ids[:actual_image_count, 0] = 0
        counts = np.zeros(batch_size, dtype=np.int32)
        counts[:actual_image_count] = 1
        use_selected = not force_fallback and not force_static_dense_after_overflow
        if not use_selected and full_dense_diff2_fn is not None:
            full_diff2 = np.asarray(full_dense_diff2_fn())
            assert full_diff2.shape == scores.shape
            assert full_diff2.dtype == np.float32
        fallback_reason = (
            "prior_batch_block_capacity_overflow"
            if force_static_dense_after_overflow
            else "block_capacity_overflow"
            if force_fallback
            else None
        )
        selection = CoarseGemmHybridBlockSelection(
            eligible=use_selected,
            fallback_reason=fallback_reason,
            block_ids=block_ids,
            block_count=counts,
            posterior_block_count=counts.copy(),
            raw_max_block_count=counts.copy(),
        )
        compact = None
        published_scores = jnp.asarray(scores)
        if compact_posterior and use_selected:
            compact_width = block_capacity * 16 * 2
            compact_values = np.full(
                (batch_size, compact_width),
                -np.inf,
                dtype=np.float32,
            )
            active_width = 16 * 2
            compact_values[:actual_image_count, :active_width] = scores[
                :actual_image_count
            ].reshape(actual_image_count, -1)
            compact = CoarseGemmHybridCompactScores(
                posterior_scores_flat=jnp.asarray(compact_values),
                source_block_ids=jnp.asarray(block_ids),
                block_count=jnp.asarray(counts),
                raw_score_max=jnp.zeros(batch_size, dtype=jnp.float32),
                min_diff2_offsets=jnp.zeros(batch_size, dtype=jnp.float32),
                best_score=jnp.max(jnp.asarray(compact_values), axis=1),
                best_pose=jnp.where(
                    jnp.arange(batch_size) < actual_image_count,
                    jnp.int32(11),
                    jnp.int32(0),
                ),
                selected_output_valid=jnp.ones(batch_size, dtype=jnp.bool_),
            )
            published_scores = None
            helper_outputs.append(compact_values)
        else:
            helper_outputs.append(np.asarray(published_scores).reshape(batch_size, -1))
        result = significance.CoarseGaussianGemmHybridBatchResult(
            scores=published_scores,
            raw_score_max=jnp.zeros(batch_size, dtype=jnp.float32),
            scores_include_priors=use_selected,
            used_selected_rescore=use_selected,
            fallback_reason=fallback_reason,
            selection=selection,
            compact_scores=compact,
            score_representation=(
                "dense_full_direct_static_capacity"
                if force_static_dense_after_overflow
                else (
                    "dense_full_direct_dynamic_fallback"
                    if force_fallback
                    else (
                        "compact_selected_exact"
                        if compact_posterior
                        else "dense_selected_exact"
                    )
                )
            ),
            full_dense_backend=(
                None
                if use_selected
                else "fused_projector"
                if full_dense_diff2_fn is not None
                else "rectangular"
            ),
        )
        return result

    monkeypatch.setattr(
        significance,
        "_compute_coarse_gaussian_gemm_hybrid_batch",
        fake_hybrid,
    )

    posterior_inputs = []

    def fake_posterior(
        score_values,
        *,
        adaptive_fraction,
        max_significants,
        tie_score_ulps,
        min_diff2_offsets,
        filter_positive_before_sort=None,
    ):
        del adaptive_fraction, max_significants, tie_score_ulps
        assert filter_positive_before_sort is (
            False if compact_expected and not force_fallback else None
        )
        scores = jnp.asarray(score_values, dtype=jnp.float32)
        posterior_inputs.append(np.asarray(scores))
        np.testing.assert_array_equal(
            np.asarray(min_diff2_offsets),
            np.zeros(scores.shape[0], dtype=np.float32),
        )
        best = jnp.argmax(scores, axis=1)
        rows = jnp.arange(scores.shape[0], dtype=jnp.int32)
        mask = jnp.zeros(scores.shape, dtype=jnp.bool_).at[rows, best].set(True)
        has_mass = jnp.any(jnp.isfinite(scores), axis=1)
        mask &= has_mass[:, None]
        weights = mask.astype(jnp.float32)
        count = has_mass.astype(jnp.int32)
        total = has_mass.astype(jnp.float32)
        return weights, mask, count, count, total, total

    monkeypatch.setattr(
        oversampling,
        "relion_cuda_f32_coarse_posterior",
        fake_posterior,
    )

    dataset = _MacroIntegrationDataset()
    rotations = np.tile(np.eye(3, dtype=np.float32), (16, 1, 1))
    translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    translation_prior = np.asarray(
        [[0.0, 0.5], [0.25, -0.5], [-0.25, 0.75]],
        dtype=np.float32,
    )
    def run():
        return significance._compute_k_class_significance_batched(
            dataset,
            jnp.zeros((1, dataset.volume_size), dtype=jnp.complex64),
            jnp.ones(dataset.image_size, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            class_log_priors=np.asarray([0.25], dtype=np.float64),
            adaptive_fraction=0.5,
            max_significants=1,
            image_batch_size=2,
            rotation_block_size=6,
            current_size=4,
            translation_log_prior=translation_prior,
            half_spectrum_scoring=True,
            relion_projector_half=jnp.ones(
                (1, 3, 3, 2),
                dtype=jnp.complex64,
            ),
            relion_projector_r_max=1,
            relion_projector_texture_interp=True,
            score_mode="gaussian",
            collect_significance=True,
            pad_final_image_batch=True,
        )

    control = run()
    control_projection_calls = tuple(projection_calls)
    control_helper_outputs = tuple(value.copy() for value in helper_outputs)
    control_operand_inputs = tuple(
        tuple(value.copy() for value in operands) for operands in operand_inputs
    )
    control_posterior_inputs = tuple(value.copy() for value in posterior_inputs)
    control_translation_calls = tuple(translation_calls)
    control_process_calls = tuple(
        (batch.copy(), per_image, processed.copy())
        for batch, per_image, processed in process_calls
    )
    control_fused_calls = tuple(fused_calls)

    projection_calls.clear()
    helper_outputs.clear()
    operand_inputs.clear()
    static_overflow_requests.clear()
    posterior_inputs.clear()
    translation_calls.clear()
    process_calls.clear()
    fused_calls.clear()
    monkeypatch.setenv(
        "RECOVAR_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS",
        "1",
    )
    result = run()

    assert projection_calls == [(16, False)]
    assert control_projection_calls == ((16, False),)
    expected_batch_count = 1 if hybrid_image_batch_size else 2
    expected_physical_rows = (
        int(hybrid_image_batch_size)
        if hybrid_image_batch_size
        else 2 * expected_batch_count
    )
    assert len(helper_outputs) == len(posterior_inputs) == expected_batch_count
    assert len(operand_inputs) == expected_batch_count
    assert len(translation_calls) == expected_batch_count
    assert len(control_translation_calls) == 2 * expected_batch_count
    assert len(process_calls) == 2 * expected_batch_count
    assert len(control_process_calls) == 2 * expected_batch_count
    assert static_overflow_requests == (
        [False, True]
        if force_fallback and expected_batch_count == 2
        else [False] * expected_batch_count
    )
    expected_fused_calls = (
        expected_batch_count if fused_fallback and force_fallback else 0
    )
    assert len(control_fused_calls) == expected_fused_calls
    assert len(fused_calls) == expected_fused_calls
    for call in fused_calls:
        assert call["rotation_count"] == 16
        assert call["lookup"].shape == (12,)
        assert call["lookup"].dtype == np.int32
        assert call["kwargs"]["current_size"] == 4
        assert call["kwargs"]["physical_image_size"] == 4
    assert [call[1] for call in process_calls] == [False, True] * expected_batch_count
    assert [call[1] for call in control_process_calls] == [
        False,
        True,
    ] * expected_batch_count
    for candidate_call, control_call in zip(
        translation_calls,
        control_translation_calls[1::2],
    ):
        for candidate_value, control_value in zip(
            candidate_call[:3],
            control_call[:3],
        ):
            np.testing.assert_array_equal(candidate_value, control_value)
        assert candidate_call[3] == control_call[3]
    if not force_fallback:
        for helper_scores, posterior_scores in zip(helper_outputs, posterior_inputs):
            np.testing.assert_array_equal(posterior_scores, helper_scores)
    for candidate_scores, control_scores in zip(
        helper_outputs,
        control_helper_outputs,
    ):
        np.testing.assert_array_equal(candidate_scores, control_scores)
    for candidate_scores, control_scores in zip(
        posterior_inputs,
        control_posterior_inputs,
    ):
        np.testing.assert_array_equal(candidate_scores, control_scores)
    for candidate_operands, control_operands in zip(
        operand_inputs,
        control_operand_inputs,
    ):
        for candidate_value, control_value in zip(
            candidate_operands,
            control_operands,
        ):
            np.testing.assert_array_equal(candidate_value, control_value)

    for index in range(4):
        np.testing.assert_array_equal(result[index], control[index])
    assert len(result[4]) == len(control[4])
    for candidate_class, control_class in zip(result[4], control[4]):
        assert len(candidate_class) == len(control_class)
        for candidate_support, control_support in zip(
            candidate_class,
            control_class,
        ):
            np.testing.assert_array_equal(candidate_support, control_support)
    for key in (
        "normalization_log_z",
        "normalization_log_evidence",
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "class_log_evidence_per_image",
        "class_assignments",
        "significant_cutoff_counts",
    ):
        np.testing.assert_array_equal(result[5][key], control[5][key])

    control_assembly = control[5]["exact_coarse_operand_assembly"]
    assert control_assembly == {
        "skip_generic_default_enabled": False,
        "skip_generic_requested": False,
        "skip_generic_effective": False,
        "exact_coarse_operands_effective": True,
        "exact_compact_preprocess_default_enabled": False,
        "exact_compact_preprocess_requested": False,
        "exact_compact_preprocess_effective": False,
        "generic_score_preprocess_count": expected_batch_count,
        "exact_source_preprocess_count": expected_batch_count,
        "generic_ctf_evaluation_count": expected_batch_count,
        "generic_full_translation_count": expected_batch_count,
        "generic_assembly_count": expected_batch_count,
        "exact_assembly_count": expected_batch_count,
        "translate_score_call_site_count": 2 * expected_batch_count,
        "translate_score_call_count": 2 * expected_batch_count,
        "downstream_operand_source": "exact_source_star",
        "diagnostic_operand_source": "exact_source_star",
        "raw_score_capture_changed": False,
        "generic_fallback_policy": "available",
        "skipped_generic_outputs": [],
    }
    candidate_assembly = result[5]["exact_coarse_operand_assembly"]
    assert candidate_assembly == {
        "skip_generic_default_enabled": False,
        "skip_generic_requested": True,
        "skip_generic_effective": True,
        "exact_coarse_operands_effective": True,
        "exact_compact_preprocess_default_enabled": False,
        "exact_compact_preprocess_requested": False,
        "exact_compact_preprocess_effective": False,
        "generic_score_preprocess_count": expected_batch_count,
        "exact_source_preprocess_count": expected_batch_count,
        "generic_ctf_evaluation_count": expected_batch_count,
        "generic_full_translation_count": expected_batch_count,
        "generic_assembly_count": 0,
        "exact_assembly_count": expected_batch_count,
        "translate_score_call_site_count": expected_batch_count,
        "translate_score_call_count": expected_batch_count,
        "downstream_operand_source": "exact_source_star",
        "diagnostic_operand_source": "exact_source_star",
        "raw_score_capture_changed": False,
        "generic_fallback_policy": "available",
        "skipped_generic_outputs": [
            "coarse_gaussian_shifted_corrected",
            "coarse_gaussian_pixel_weight",
            "coarse_gaussian_unshifted_corrected",
        ],
    }
    np.testing.assert_array_equal(result[1], np.ones(3, dtype=np.int32))
    np.testing.assert_array_equal(result[2], np.full(3, 11, dtype=np.int32))
    np.testing.assert_array_equal(result[3], np.zeros(3, dtype=np.int32))
    hybrid_stats = result[5]["coarse_gaussian_gemm_hybrid"]
    assert hybrid_stats["compact_posterior_enabled"] is compact_posterior
    assert hybrid_stats["compact_posterior_default_enabled"] is False
    assert hybrid_stats["selected_score_layout"] == (
        "fixed_capacity_source16" if compact_posterior else "dense_global"
    )
    assert hybrid_stats["positive_only_scan_role"] == (
        "correctness_oracle_not_runtime" if compact_posterior else None
    )
    assert hybrid_stats["batch_count"] == expected_batch_count
    assert hybrid_stats["actual_image_batch_sizes"] == (
        [3] if hybrid_image_batch_size else [2, 1]
    )
    assert hybrid_stats["physical_image_batch_sizes"] == (
        [3] if hybrid_image_batch_size else [2, 2]
    )
    assert hybrid_stats["selected_rescore_batch_count"] == (
        0 if force_fallback else expected_batch_count
    )
    expected_fallback_batches = 1 if force_fallback else 0
    expected_static_batches = (
        expected_batch_count - 1 if force_fallback else 0
    )
    first_batch_images = 3 if hybrid_image_batch_size else 2
    assert hybrid_stats["fallback_batch_count"] == expected_fallback_batches
    assert hybrid_stats["static_dense_batch_count"] == expected_static_batches
    expected_full_dense_batches = expected_fallback_batches + expected_static_batches
    assert hybrid_stats["full_dense_batch_count"] == expected_full_dense_batches
    assert hybrid_stats["fused_full_fallback_batch_count"] == (
        expected_full_dense_batches if fused_fallback and force_fallback else 0
    )
    assert (
        hybrid_stats["rectangular_full_fallback_batch_count"]
        == (
            0
            if fused_fallback and force_fallback
            else expected_full_dense_batches
        )
    )
    expected_full_backend = "fused_projector" if fused_fallback else "rectangular"
    assert hybrid_stats["full_fallback_backend_requested"] == expected_full_backend
    assert hybrid_stats["full_fallback_backend_armed"] == expected_full_backend
    assert hybrid_stats["full_fallback_backend_effective"] == (
        expected_full_backend if expected_full_dense_batches else None
    )
    assert hybrid_stats["all_full_dense_batches_used_fused"] is (
        fused_fallback if expected_full_dense_batches else None
    )
    assert hybrid_stats["selected_rescore_image_count"] == (
        0 if force_fallback else 3
    )
    assert hybrid_stats["fallback_image_count"] == (
        first_batch_images if force_fallback else 0
    )
    assert hybrid_stats["static_dense_image_count"] == (
        3 - first_batch_images if force_fallback else 0
    )
    expected_full_dense_images = 3 if force_fallback else 0
    assert hybrid_stats["full_dense_image_count"] == expected_full_dense_images
    assert hybrid_stats["fused_full_fallback_image_count"] == (
        expected_full_dense_images if fused_fallback else 0
    )
    assert (
        hybrid_stats["rectangular_full_fallback_image_count"]
        == (0 if fused_fallback else expected_full_dense_images)
    )
    selector_audit = result[5]["coarse_selector_audit"]
    assert selector_audit["requested_fused"] is fused_fallback
    assert selector_audit["effective_fused"] is bool(expected_fused_calls)
    assert selector_audit["counts"]["fused_calls"] == expected_fused_calls
    assert selector_audit["counts"]["actual_rows"] == (
        3 if fused_fallback and force_fallback else 0
    )
    assert hybrid_stats["selected_source16_block_count"] == (
        0 if force_fallback else 3
    )
    assert hybrid_stats["selected_exact_candidate_fraction"] == (
        None if force_fallback else 1.0
    )
    assert hybrid_stats["input_image_batch_size"] == 2
    assert hybrid_stats["requested_hybrid_image_batch_size"] == hybrid_image_batch_size
    assert hybrid_stats["effective_image_batch_size"] == (
        hybrid_image_batch_size or 2
    )
    assert hybrid_stats[
        "streamed_certificate_candidate_count_at_effective_batch"
    ] == (hybrid_image_batch_size or 2) * 16 * 2
    expected_table_multiplier = 0 if force_fallback else expected_physical_rows
    assert hybrid_stats["selected_score_table_capacity_candidates"] == (
        expected_table_multiplier * 64 * 16 * 2
    )
    assert hybrid_stats["dense_global_score_table_capacity_candidates"] == (
        expected_table_multiplier * 16 * 2
    )
    assert hybrid_stats["selected_score_table_capacity_bytes_f32"] == (
        expected_table_multiplier * 64 * 16 * 2 * 4
    )
    assert hybrid_stats["dense_global_score_table_capacity_bytes_f32"] == (
        expected_table_multiplier * 16 * 2 * 4
    )
    assert hybrid_stats["selected_to_dense_score_table_capacity_fraction"] == (
        None if force_fallback else 64.0
    )
    assert hybrid_stats["fallback_reasons"] == (
        {"block_capacity_overflow": 1} if force_fallback else {}
    )
    assert hybrid_stats["overflow_latch_scope"] == (
        "current_significance_call_exact_geometry_and_capacity"
    )
    assert hybrid_stats["overflow_latch_active_at_return"] is force_fallback
    assert hybrid_stats["overflow_latch_activation_count"] == (
        1 if force_fallback else 0
    )
    assert hybrid_stats["overflow_latch_static_dense_batch_count"] == (
        expected_static_batches
    )
    assert hybrid_stats["overflow_latch_static_dense_image_count"] == (
        3 - first_batch_images if force_fallback else 0
    )
    support_audit = result[5]["coarse_significance_support_audit"]
    assert support_audit["n_classes"] == 1
    assert support_audit["n_images"] == 3
    assert support_audit["samples_per_class"] == 32
    assert support_audit["per_class_image_selected_counts"] == [[1, 1, 1]]
    assert len(support_audit["aggregate_support_sha256"]) == 64

    if not compact_posterior:
        return

    single_translate_helper_outputs = tuple(
        value.copy() for value in helper_outputs
    )
    single_translate_operand_inputs = tuple(
        tuple(value.copy() for value in operands) for operands in operand_inputs
    )
    single_translate_posterior_inputs = tuple(
        value.copy() for value in posterior_inputs
    )
    single_translate_translation_calls = tuple(translation_calls)
    single_translate_process_calls = tuple(
        (batch.copy(), per_image, processed.copy())
        for batch, per_image, processed in process_calls
    )

    projection_calls.clear()
    helper_outputs.clear()
    operand_inputs.clear()
    posterior_inputs.clear()
    translation_calls.clear()
    process_calls.clear()
    monkeypatch.setenv(
        "RECOVAR_K1_RELION_EXACT_COMPACT_PREPROCESS",
        "1",
    )
    specialized = run()

    assert projection_calls == [(16, False)]
    assert len(process_calls) == expected_batch_count
    assert all(call[1] for call in process_calls)
    single_translate_exact_process_calls = tuple(
        call for call in single_translate_process_calls if call[1]
    )
    assert len(single_translate_exact_process_calls) == expected_batch_count
    for specialized_call, single_translate_call in zip(
        process_calls,
        single_translate_exact_process_calls,
    ):
        np.testing.assert_array_equal(specialized_call[0], single_translate_call[0])
        np.testing.assert_array_equal(specialized_call[2], single_translate_call[2])

    assert len(translation_calls) == expected_batch_count
    for specialized_call, single_translate_call in zip(
        translation_calls,
        single_translate_translation_calls,
    ):
        for specialized_value, single_translate_value in zip(
            specialized_call[:3],
            single_translate_call[:3],
        ):
            np.testing.assert_array_equal(specialized_value, single_translate_value)
        assert specialized_call[3] == single_translate_call[3]
    for specialized_scores, single_translate_scores in zip(
        helper_outputs,
        single_translate_helper_outputs,
    ):
        np.testing.assert_array_equal(specialized_scores, single_translate_scores)
    for specialized_scores, single_translate_scores in zip(
        posterior_inputs,
        single_translate_posterior_inputs,
    ):
        np.testing.assert_array_equal(specialized_scores, single_translate_scores)
    for specialized_operands, single_translate_operands in zip(
        operand_inputs,
        single_translate_operand_inputs,
    ):
        for specialized_value, single_translate_value in zip(
            specialized_operands,
            single_translate_operands,
        ):
            np.testing.assert_array_equal(specialized_value, single_translate_value)

    for index in range(4):
        np.testing.assert_array_equal(specialized[index], result[index])
    assert len(specialized[4]) == len(result[4])
    for specialized_class, single_translate_class in zip(
        specialized[4],
        result[4],
    ):
        assert len(specialized_class) == len(single_translate_class)
        for specialized_support, single_translate_support in zip(
            specialized_class,
            single_translate_class,
        ):
            np.testing.assert_array_equal(
                specialized_support,
                single_translate_support,
            )
    for key in (
        "normalization_log_z",
        "normalization_log_evidence",
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "class_log_evidence_per_image",
        "class_assignments",
        "significant_cutoff_counts",
    ):
        np.testing.assert_array_equal(specialized[5][key], result[5][key])

    specialized_assembly = specialized[5]["exact_coarse_operand_assembly"]
    expected_specialized_assembly = dict(candidate_assembly)
    expected_specialized_assembly.update(
        {
            "exact_compact_preprocess_requested": True,
            "exact_compact_preprocess_effective": True,
            "generic_score_preprocess_count": 0,
            "generic_ctf_evaluation_count": 0,
            "generic_full_translation_count": 0,
            "generic_fallback_policy": "exact_full_direct_scores_available",
        }
    )
    assert specialized_assembly == expected_specialized_assembly


def test_coarse_gaussian_gemm_live_k2_priors_multigroup_and_poisoned_tails(
    monkeypatch,
    tmp_path,
):
    """Live K-class pass preserves layout, priors, support, and both tail masks."""

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers import projection as projection_helpers
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed

    for name, value in {
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE": "0",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB": "0.01",
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI": "1",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF": "1",
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR": "0",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "0",
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "0",
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE": "0",
        "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS": "1",
        "RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "0",
    }.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(significance.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "cuda_available", lambda: True)
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_exact_ctf_half_from_source_star",
        lambda _dataset, indices, image_shape: jnp.ones(
            (len(indices), int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)),
            dtype=jnp.float64,
        ),
    )
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_exact_ctf_half_from_source_star_host",
        lambda _dataset, indices, image_shape: np.ones(
            (len(indices), int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)),
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_cuda_powerclass_highres_xi2_half",
        lambda processed, **_kwargs: jnp.zeros(processed.shape[0], dtype=jnp.float32),
    )
    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f32",
        lambda images, translation_angles, pixel_indices, image_shape: jnp.repeat(
            images[:, None, :],
            int(translation_angles.shape[0]),
            axis=1,
        ).reshape(images.shape[0] * int(translation_angles.shape[0]), -1),
    )

    projection_calls = []

    def fake_projection(projector_half, rotations_block, image_shape, **kwargs):
        del image_shape
        assert isinstance(kwargs["pixel_indices"], np.ndarray)
        class_index = int(np.rint(np.asarray(projector_half)[0, 0, 0].real))
        rotations_np = np.asarray(rotations_block)
        rotation_markers = rotations_np[:, 0, 1]
        is_poisoned_tail = rotation_markers == 0.0
        rotation_ids = rotation_markers - 10.0
        codes = class_index * 100.0 + rotation_ids * 10.0
        codes = np.where(is_poisoned_tail, 9999.0, codes).astype(np.float32)
        projection_calls.append((class_index, codes.copy(), is_poisoned_tail.copy()))
        projected = jnp.repeat(
            jnp.asarray(codes, dtype=jnp.complex64)[:, None],
            len(kwargs["pixel_indices"]),
            axis=1,
        )
        return projected, jnp.abs(projected) ** 2

    monkeypatch.setattr(
        projection_helpers,
        "compute_relion_projector_projections_block",
        fake_projection,
    )

    score_calls = []

    def designed_scores(projected, shifted, actual_count):
        projected_codes = np.asarray(projected)[:, 0].real
        shifted_np = np.asarray(shifted)
        scores = np.full(
            (shifted_np.shape[0], projected_codes.size, shifted_np.shape[1]),
            -4.0,
            dtype=np.float32,
        )
        for image_lane in range(int(actual_count)):
            original_image_id = int(
                np.rint(np.max(shifted_np[image_lane].real))
            ) - 1
            for rotation_lane, code in enumerate(projected_codes):
                if code > 1000.0:
                    continue
                class_index = int(code // 100.0)
                rotation_index = int(np.rint((code - class_index * 100.0) / 10.0))
                # Every image starts at class 0 / rotation 0 / translation 0.
                if class_index == 0 and rotation_index == 0:
                    scores[image_lane, rotation_lane, 0] = 0.0
                # Each small runner-up gap is independently overcome by one
                # prior axis in the qualification calls below.
                if original_image_id == 0 and class_index == 1 and rotation_index == 0:
                    scores[image_lane, rotation_lane, 0] = -0.05
                if original_image_id == 1 and class_index == 0 and rotation_index == 1:
                    scores[image_lane, rotation_lane, 0] = -0.05
                    scores[image_lane, rotation_lane, 1] = -0.1
                if original_image_id == 1 and class_index == 0 and rotation_index == 0:
                    scores[image_lane, rotation_lane, 1] = -0.05
        return jnp.asarray(scores)

    def controlled_scores(
        projected,
        projected_abs2,
        shifted,
        weight,
        initial,
        actual_image_count,
        **_kwargs,
    ):
        del projected_abs2, weight, initial
        shifted_np = np.asarray(shifted)
        actual_count = int(np.asarray(actual_image_count))
        if poison_tail["enabled"] and actual_count < shifted_np.shape[0]:
            assert np.isnan(shifted_np[actual_count:]).all()
        scores = designed_scores(projected, shifted, actual_count)
        score_calls.append((actual_count, tuple(shifted.shape)))
        return scores

    monkeypatch.setattr(
        significance,
        "_relion_coarse_gaussian_gemm_scores",
        controlled_scores,
    )

    original_pad = significance._pad_significance_preprocess_inputs
    poison_tail = {"enabled": False}

    def maybe_poison_tail(*args, **kwargs):
        result = list(original_pad(*args, **kwargs))
        if poison_tail["enabled"] and np.asarray(result[0]).shape[0] > np.asarray(args[0]).shape[0]:
            batch = np.asarray(result[0]).copy()
            ctf = np.asarray(result[1]).copy()
            batch[-1] = np.nan
            ctf[-1] = np.nan
            result[0] = batch
            result[1] = ctf
        return tuple(result)

    monkeypatch.setattr(
        significance,
        "_pad_significance_preprocess_inputs",
        maybe_poison_tail,
    )

    dataset = _MacroIntegrationDataset()
    rotations = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
    rotations[:, 0, 1] = np.asarray([10.0, 11.0, 12.0], dtype=np.float32)
    translations = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    relion_projector = jnp.stack(
        [
            jnp.full((3, 3, 2), class_index, dtype=jnp.complex64)
            for class_index in range(2)
        ]
    )
    common = dict(
        adaptive_fraction=0.5,
        max_significants=1,
        image_batch_size=2,
        rotation_block_size=2,
        current_size=4,
        half_spectrum_scoring=True,
        relion_projector_half=relion_projector,
        relion_projector_r_max=1,
        relion_projector_texture_interp=True,
        score_mode="gaussian",
        collect_significance=True,
        return_class_best=True,
        pad_final_image_batch=True,
    )

    zero_class_prior = np.zeros(2, dtype=np.float64)
    active_class_prior = np.asarray([-0.1, 0.0], dtype=np.float64)
    zero_rotation_prior = np.zeros((2, 3), dtype=np.float32)
    active_rotation_prior = zero_rotation_prior.copy()
    active_rotation_prior[0, 1] = 0.1
    zero_translation_prior = np.zeros((3, 2), dtype=np.float32)
    active_translation_prior = zero_translation_prior.copy()
    active_translation_prior[1, 1] = 0.1

    def run_with_priors(
        selected_dataset,
        *,
        class_prior,
        rotation_prior,
        translation_prior,
        diagnostic_scope=None,
    ):
        selected_translation_prior = translation_prior[
            selected_dataset._original_indices
        ]
        return significance._compute_k_class_significance_batched(
            selected_dataset,
            jnp.zeros((2, selected_dataset.volume_size), dtype=jnp.complex64),
            jnp.ones(selected_dataset.image_size, dtype=jnp.float32),
            rotations,
            translations,
            "linear_interp",
            class_log_priors=class_prior,
            rotation_log_prior=rotation_prior,
            translation_log_prior=selected_translation_prior,
            coarse_gemm_diagnostic_scope=diagnostic_scope,
            **common,
        )

    zero_priors = run_with_priors(
        dataset,
        class_prior=zero_class_prior,
        rotation_prior=zero_rotation_prior,
        translation_prior=zero_translation_prior,
    )
    class_only = run_with_priors(
        dataset,
        class_prior=active_class_prior,
        rotation_prior=zero_rotation_prior,
        translation_prior=zero_translation_prior,
    )
    rotation_only = run_with_priors(
        dataset,
        class_prior=zero_class_prior,
        rotation_prior=active_rotation_prior,
        translation_prior=zero_translation_prior,
    )
    translation_only = run_with_priors(
        dataset,
        class_prior=zero_class_prior,
        rotation_prior=zero_rotation_prior,
        translation_prior=active_translation_prior,
    )
    clean = run_with_priors(
        dataset,
        class_prior=active_class_prior,
        rotation_prior=active_rotation_prior,
        translation_prior=active_translation_prior,
    )

    # Each prior axis independently crosses one 0.05 score gap.  Removing
    # priors therefore changes the exact winner/support contract; this is not
    # a decorative prior fixture with ten-point score margins.
    np.testing.assert_array_equal(zero_priors[2], [0, 0, 0])
    np.testing.assert_array_equal(zero_priors[3], [0, 0, 0])
    np.testing.assert_array_equal(class_only[2], [0, 0, 0])
    np.testing.assert_array_equal(class_only[3], [1, 0, 0])
    np.testing.assert_array_equal(rotation_only[2], [0, 2, 0])
    np.testing.assert_array_equal(rotation_only[3], [0, 0, 0])
    np.testing.assert_array_equal(translation_only[2], [0, 1, 0])
    np.testing.assert_array_equal(translation_only[3], [0, 0, 0])
    np.testing.assert_array_equal(clean[2], [0, 3, 0])
    np.testing.assert_array_equal(clean[3], [1, 0, 0])

    def unique_support_pairs(result):
        pairs = []
        for image_index in range(3):
            image_pairs = []
            for class_index in range(2):
                for pose_id in significance.significant_sample_ids(
                    result[4][class_index][image_index],
                    6,
                ):
                    image_pairs.append((class_index, int(pose_id)))
            assert len(image_pairs) == 1
            pairs.append(image_pairs[0])
        return tuple(pairs)

    assert unique_support_pairs(zero_priors) == ((0, 0), (0, 0), (0, 0))
    assert unique_support_pairs(class_only) == ((1, 0), (0, 0), (0, 0))
    assert unique_support_pairs(rotation_only) == ((0, 0), (0, 2), (0, 0))
    assert unique_support_pairs(translation_only) == ((0, 0), (0, 1), (0, 0))
    assert unique_support_pairs(clean) == ((1, 0), (0, 3), (0, 0))

    diagnostic_dir = tmp_path / "paired_scores"
    monkeypatch.setenv(
        "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR",
        str(diagnostic_dir),
    )
    monkeypatch.setenv(
        "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES",
        "0,1",
    )
    stream_diagnostic_dir = tmp_path / "streaming_scores"
    monkeypatch.setenv(
        "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR",
        str(stream_diagnostic_dir),
    )
    monkeypatch.setenv(
        "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_TOPK",
        "12",
    )

    def direct_square_stub(projected, shifted, weight, initial, full_to_compact):
        del initial, full_to_compact
        active = np.any(np.asarray(weight) != 0.0, axis=1)
        actual_count = int(np.count_nonzero(active))
        return -designed_scores(projected, shifted, actual_count)

    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rectangular_f32",
        direct_square_stub,
    )
    poison_tail["enabled"] = True
    group_datasets = (dataset.subset([0, 2]), dataset.subset([1]))
    run_id = "initial_model_it0001_k002_multigroup"
    call_ids = (
        "call0000_group0000_halfseth00",
        "call0001_group0001_halfseth01",
    )
    scopes = tuple(
        significance.CoarseGaussianGemmDiagnosticScope(
            run_id=run_id,
            call_id=call_id,
            expected_call_ids=call_ids,
            finalize=index == len(call_ids) - 1,
        )
        for index, call_id in enumerate(call_ids)
    )
    poisoned_groups = tuple(
        run_with_priors(
            group_dataset,
            class_prior=active_class_prior,
            rotation_prior=active_rotation_prior,
            translation_prior=active_translation_prior,
            diagnostic_scope=scopes[index],
        )
        for index, group_dataset in enumerate(group_datasets)
    )

    np.testing.assert_array_equal(
        poisoned_groups[0][0] | poisoned_groups[1][0],
        clean[0],
    )
    for result, original_indices in zip(poisoned_groups, ([0, 2], [1])):
        np.testing.assert_array_equal(result[1], clean[1][original_indices])
        np.testing.assert_array_equal(result[2], clean[2][original_indices])
        np.testing.assert_array_equal(result[3], clean[3][original_indices])
    np.testing.assert_array_equal(clean[1], np.ones(3, dtype=np.int32))
    expected_support = [
        [[], [3], [0]],
        [[0], [], []],
    ]
    for class_index in range(2):
        for image_index in range(3):
            np.testing.assert_array_equal(
                significance.significant_sample_ids(
                    clean[4][class_index][image_index],
                    6,
                ),
                np.asarray(expected_support[class_index][image_index], dtype=np.int64),
            )
    assert [call[0] for call in score_calls].count(1) > 0
    assert [call[0] for call in score_calls].count(2) > 0
    assert any(np.any(tail) for _, _, tail in projection_calls)
    resources = clean[5]["coarse_gaussian_gemm_resources"]
    assert resources["pixel_index_device_to_host_materializations"] == 0
    assert resources["predicted_peak_projection_bytes"] <= resources[
        "projected_transient_budget_bytes"
    ]
    clean_qualification = clean[5]["coarse_gaussian_gemm_qualification"]
    assert clean_qualification["paired_capture_requested"] is False
    assert clean_qualification["paired_capture_active"] is False
    assert clean_qualification["clean_timing_eligible"] is True
    assert clean_qualification["numerical_qualification_status"] == "NO_GO_UNQUALIFIED"
    assert clean_qualification["requires_bitwise_score_identity"] is False
    assert clean_qualification["requires_exact_discrete_identity"] is True
    for poisoned in poisoned_groups:
        captured_qualification = poisoned[5]["coarse_gaussian_gemm_qualification"]
        assert captured_qualification["paired_capture_requested"] is True
        assert captured_qualification["paired_capture_active"] is True
        assert captured_qualification["clean_timing_eligible"] is False
        assert "separate diagnostic-off timing arm" in captured_qualification["timing_policy"]
    diagnostic_paths = sorted(diagnostic_dir.glob("*.npz"))
    assert len(diagnostic_paths) == 2
    captured_original_indices = []
    for diagnostic_path in diagnostic_paths:
        with np.load(diagnostic_path) as payload:
            assert str(payload["layout"]) == "image,class,rotation,translation"
            assert bool(payload["paired_capture_active"]) is True
            assert bool(payload["clean_timing_eligible"]) is False
            assert str(payload["qualification_status"]).startswith("NO_GO")
            assert payload["automatic_no_go_reasons"].size == 0
            assert payload["pending_qualification_gates"].size > 0
            assert bool(payload["requires_bitwise_score_identity"]) is False
            assert bool(payload["requires_exact_discrete_identity"]) is True
            assert str(payload["repeat_spread_assessment"]).startswith("NO_GO")
            assert str(payload["scale_growth_assessment"]).startswith("NO_GO")
            assert "separate_diagnostic-off_timing_arm" in str(payload["timing_policy"])
            np.testing.assert_array_equal(
                payload["direct_scores_with_prior"],
                payload["macro_scores_with_prior"],
            )
            np.testing.assert_array_equal(payload["argmax_equal"], True)
            np.testing.assert_array_equal(payload["support_equal"], True)
            assert int(payload["score_precision_bits"]) == 32
            assert int(payload["resource_pixel_index_device_to_host_materializations"]) == 0
            captured_original_indices.extend(payload["original_indices"].tolist())
    assert sorted(captured_original_indices) == [0, 1]
    assert len({path.name for path in diagnostic_paths}) == 2
    assert call_ids[0] in diagnostic_paths[0].name
    assert call_ids[1] in diagnostic_paths[1].name

    scope_manifest_paths = sorted(diagnostic_dir.glob("coarse_gemm_scope_*.json"))
    assert len(scope_manifest_paths) == 2
    scope_records = [json.loads(path.read_text()) for path in scope_manifest_paths]
    assert {tuple(record["targets_in_scope"]) for record in scope_records} == {
        (0,),
        (1,),
    }
    assert {
        tuple(record["targets_explicitly_out_of_scope"])
        for record in scope_records
    } == {(0,), (1,)}
    aggregate_path = diagnostic_dir / f"coarse_gemm_manifest_{run_id}.json"
    aggregate = json.loads(aggregate_path.read_text())
    assert aggregate["all_requested_captured_exactly_once"] is True
    assert aggregate["captured_target_counts"] == {"0": 1, "1": 1}

    stream_paths = sorted(stream_diagnostic_dir.glob("coarse_gemm_rescore_*.npz"))
    assert len(stream_paths) == 2
    stream_original_indices = []
    for stream_path in stream_paths:
        with np.load(stream_path, allow_pickle=False) as payload:
            assert payload["schema"].item() == COARSE_GEMM_STREAMING_SCHEMA
            assert payload["retained_topk"].item() == 12
            assert payload["stores_score_cube"].item() is False
            assert payload["production_behavior_changed"].item() is False
            assert payload["winner_comparison_coverage"].all()
            assert payload["winner_equal"].all()
            assert payload["support_comparison_coverage"].all()
            np.testing.assert_array_equal(payload["support_false_negative_count"], 0)
            np.testing.assert_array_equal(payload["support_false_positive_count"], 0)
            np.testing.assert_array_equal(payload["all_candidate_max_abs_delta"], 0.0)
            assert payload[
                "relion_nonzero_surface_error_safe_superset_coverage"
            ].all()
            stream_original_indices.extend(payload["original_indices"].tolist())
    assert sorted(stream_original_indices) == [0, 1, 2]

    stream_scope_paths = sorted(
        stream_diagnostic_dir.glob("coarse_gemm_rescore_scope_*.json")
    )
    assert len(stream_scope_paths) == 2
    stream_aggregate_path = (
        stream_diagnostic_dir / f"coarse_gemm_rescore_manifest_{run_id}.json"
    )
    stream_aggregate = json.loads(stream_aggregate_path.read_text())
    assert stream_aggregate["particle_count"] == 3
    assert stream_aggregate["all_particles_captured_exactly_once"] is True
    assert stream_aggregate["stores_score_cube"] is False


@pytest.mark.parametrize(
    ("bad_operand", "message"),
    [
        ("projection_pixels", "projection and image pixels"),
        ("weight_shape", "pixel_weight"),
        ("tail_count", "actual_image_count"),
    ],
)
def test_coarse_gaussian_gemm_macro_rejects_ambiguous_bindings(
    bad_operand,
    message,
):
    projected, projected_abs2, shifted, weight, initial = _macro_operands(
        real_dtype=np.float32,
    )
    actual_count = shifted.shape[0]
    if bad_operand == "projection_pixels":
        projected = projected[:, :-1]
        projected_abs2 = projected_abs2[:, :-1]
    elif bad_operand == "weight_shape":
        weight = weight[:, :-1]
    else:
        actual_count += 1

    with pytest.raises((ValueError, TypeError), match=message):
        scoring._relion_coarse_gaussian_gemm_scores(
            *map(jnp.asarray, (projected, projected_abs2, shifted, weight, initial)),
            actual_count,
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
