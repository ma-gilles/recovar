from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.diagnostics.coarse_gaussian_diagnostics import (
    CoarseGaussianGemmDiagnosticScope,
    _coarse_gaussian_gemm_streaming_diagnostic_request,
    _seal_coarse_gaussian_gemm_streaming_scope,
)
from recovar.em.scoring.coarse_gemm_streaming import (
    COARSE_GEMM_STREAMING_SCHEMA,
    coarse_gemm_streaming_dual_state_bytes,
    coarse_gemm_streaming_state_bytes,
    initialize_coarse_gemm_streaming_state,
    summarize_coarse_gemm_streaming_state,
    update_coarse_gemm_streaming_state,
    write_coarse_gemm_streaming_summary,
)

pytestmark = pytest.mark.unit


def _stream(direct: np.ndarray, macro: np.ndarray, *, topk: int, block_size: int):
    state = initialize_coarse_gemm_streaming_state(
        direct.shape[0],
        topk,
        score_dtype=jnp.float32,
    )
    for start in range(0, direct.shape[1], block_size):
        stop = min(start + block_size, direct.shape[1])
        state = update_coarse_gemm_streaming_state(
            state,
            jnp.asarray(direct[:, start:stop], dtype=jnp.float32),
            jnp.asarray(macro[:, start:stop], dtype=jnp.float32),
            candidate_offset=start,
            actual_image_count=direct.shape[0],
        )
    return state


def _mask(candidate_count: int, ids_by_image: list[tuple[int, ...]]) -> np.ndarray:
    result = np.zeros((len(ids_by_image), candidate_count), dtype=bool)
    for row, candidate_ids in enumerate(ids_by_image):
        result[row, np.asarray(candidate_ids, dtype=np.int64)] = True
    return result


def test_streaming_dual_topk_matches_full_block_order_and_error_extrema() -> None:
    direct = np.asarray(
        [
            [12.0, 9.0, 8.0, 7.0, 11.0, 6.0, 5.0, 4.0, 10.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0],
        ],
        dtype=np.float32,
    )
    delta = np.asarray(
        [
            [0.125, 0.0, -0.125, 0.0, 0.125, 0.0, -0.125, 0.0, 0.125, 0.0, -0.125, 0.0],
            [0.0, -0.125, 0.0, 0.125, 0.0, -0.125, 0.0, 0.125, 0.0, -0.125, 0.0, 0.125],
        ],
        dtype=np.float32,
    )
    macro = direct + delta
    state = _stream(direct, macro, topk=5, block_size=4)

    direct_ids = np.asarray(state.direct_candidate_ids)[:, :5]
    macro_ids = np.asarray(state.macro_candidate_ids)[:, :5]
    expected_direct = np.argsort(-direct, axis=1, kind="stable")[:, :5]
    expected_macro = np.argsort(-macro, axis=1, kind="stable")[:, :5]

    assert np.array_equal(direct_ids, expected_direct)
    assert np.array_equal(macro_ids, expected_macro)
    assert np.array_equal(
        np.asarray(state.max_abs_delta),
        np.asarray([0.125, 0.125]),
    )
    assert np.array_equal(np.asarray(state.finite_pair_count), np.asarray([12, 12]))
    assert np.array_equal(np.asarray(state.nonfinite_pair_count), np.asarray([0, 0]))


def test_streamed_summary_reports_support_errors_and_conservative_supersets() -> None:
    direct = np.asarray([[10.0, 9.0, 8.0, 7.0, 1.0, 0.0]], dtype=np.float32)
    macro = np.asarray([[10.0, 8.0, 9.0, 7.0, 1.0, 0.0]], dtype=np.float32)
    state = _stream(direct, macro, topk=4, block_size=2)
    summary = summarize_coarse_gemm_streaming_state(
        state,
        state,
        _mask(6, [(0, 2)]),
        actual_image_count=1,
        adaptive_fraction=0.999,
        max_significants=2,
        band_widths=(0.0, 0.5, 1.0, 2.0),
    )

    assert summary["direct_support_coverage"].tolist() == [True]
    assert summary["macro_support_ranked_coverage"].tolist() == [True]
    assert summary["support_comparison_coverage"].tolist() == [True]
    assert summary["direct_support_count"].tolist() == [2]
    assert summary["macro_support_count"].tolist() == [2]
    assert summary["support_false_negative_count"].tolist() == [1]
    assert summary["support_false_positive_count"].tolist() == [1]
    assert summary["direct_cutoff_score"].tolist() == [9.0]
    assert summary["macro_cutoff_score"].tolist() == [9.0]
    assert summary["support_observed_minimum_band"].tolist() == [1.0]
    assert summary["support_global_error_safe_band"].tolist() == [1.0]
    assert summary["support_observed_superset_count"].tolist() == [3]
    assert summary["support_observed_superset_coverage"].tolist() == [True]
    assert summary["winner_equal"].tolist() == [True]
    assert summary["winner_comparison_coverage"].tolist() == [True]
    assert summary["winner_observed_minimum_band"].tolist() == [0.0]
    assert summary["winner_global_error_safe_band"].tolist() == [2.0]
    assert summary["winner_global_error_superset_count"].tolist() == [3]
    assert summary["near_macro_cutoff_count"].tolist() == [[1, 1, 3, 4]]
    assert summary["near_macro_cutoff_coverage"].tolist() == [[True, True, True, True]]
    assert summary["band_superset_count"].tolist() == [[2, 2, 3, 4]]
    assert summary["band_superset_coverage"].tolist() == [[True, True, True, True]]
    assert summary["macro_max_band_superset_count"].tolist() == [[1, 1, 2, 3]]
    assert summary["macro_max_band_superset_coverage"].tolist() == [[True, True, True, True]]
    assert summary["relion_nonzero_surface_error_safe_width_from_macro_max"].tolist() == [140.0]
    assert summary["relion_nonzero_surface_error_safe_superset_count"].tolist() == [4]
    assert summary["relion_nonzero_surface_error_safe_superset_coverage"].tolist() == [False]


def test_streamed_summary_fails_closed_when_support_or_band_overflows_topk() -> None:
    scores = np.asarray([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float32)

    support_state = _stream(scores, scores, topk=2, block_size=2)
    support_summary = summarize_coarse_gemm_streaming_state(
        support_state,
        support_state,
        _mask(5, [(0, 1, 2)]),
        actual_image_count=1,
        adaptive_fraction=0.999,
        max_significants=3,
        band_widths=(0.0,),
    )
    assert support_summary["direct_support_coverage"].tolist() == [False]
    assert support_summary["macro_support_ranked_coverage"].tolist() == [False]
    assert support_summary["support_comparison_coverage"].tolist() == [False]
    assert support_summary["support_false_negative_count"].tolist() == [-1]
    assert support_summary["support_false_positive_count"].tolist() == [-1]

    band_state = _stream(scores, scores, topk=3, block_size=2)
    band_summary = summarize_coarse_gemm_streaming_state(
        band_state,
        band_state,
        _mask(5, [(0,)]),
        actual_image_count=1,
        adaptive_fraction=0.5,
        max_significants=1,
        band_widths=(0.0, 1.0, 3.0),
    )
    assert band_summary["band_superset_count"].tolist() == [[1, 2, 3]]
    assert band_summary["band_superset_coverage"].tolist() == [[True, True, False]]
    assert band_summary["near_macro_cutoff_coverage"].tolist() == [[True, True, False]]


def test_streamed_summary_fails_closed_on_winner_and_cutoff_ties() -> None:
    scores = np.asarray([[5.0, 5.0, 4.0, 4.0]], dtype=np.float32)
    state = _stream(scores, scores, topk=3, block_size=2)
    summary = summarize_coarse_gemm_streaming_state(
        state,
        state,
        _mask(4, [(0, 1, 2, 3)]),
        actual_image_count=1,
        adaptive_fraction=0.999,
        max_significants=3,
        band_widths=(0.0, 1.0),
    )

    assert summary["direct_winner_unique"].tolist() == [False]
    assert summary["macro_winner_unique"].tolist() == [False]
    assert summary["winner_comparison_coverage"].tolist() == [False]
    assert summary["winner_equal"].tolist() == [False]
    assert summary["direct_support_coverage"].tolist() == [False]
    assert summary["macro_support_ranked_coverage"].tolist() == [False]
    assert summary["support_comparison_coverage"].tolist() == [False]


def test_relion_nonzero_surface_superset_is_certified_only_when_complete() -> None:
    direct = np.asarray([[10.0, 0.0, -128.0, -130.0, -131.0]], dtype=np.float32)
    macro = np.asarray([[10.5, 0.0, -127.5, -130.5, -131.5]], dtype=np.float32)
    state = _stream(direct, macro, topk=5, block_size=2)
    summary = summarize_coarse_gemm_streaming_state(
        state,
        state,
        _mask(5, [(0,)]),
        actual_image_count=1,
        adaptive_fraction=0.5,
        max_significants=1,
        band_widths=(136.0, 138.0, 140.0, 144.0, 160.0),
        n_rotations=5,
        n_translations=1,
        source_rotation_block_size=2,
        rotation_block_capacity=2,
    )

    assert summary["all_candidate_error_coverage"].tolist() == [True]
    assert summary["all_candidate_max_abs_delta"].tolist() == [0.5]
    assert summary["relion_nonzero_surface_error_safe_width_from_macro_max"].tolist() == [139.0]
    assert summary["relion_nonzero_surface_error_safe_superset_count"].tolist() == [3]
    assert summary["relion_nonzero_surface_error_safe_superset_coverage"].tolist() == [True]
    assert summary["macro_max_band_superset_count"].tolist() == [[2, 3, 3, 5, 5]]
    assert summary["macro_max_band_superset_coverage"].tolist() == [[True] * 5]
    assert summary["macro_max_band_source_rotation_block_count"].tolist() == [[1, 2, 2, 3, 3]]
    assert summary["macro_max_band_source_rotation_block_count_coverage"].tolist() == [[True] * 5]
    assert summary["relion_nonzero_surface_error_safe_source_rotation_block_count"].tolist() == [2]
    assert summary["relion_nonzero_surface_error_safe_source_rotation_block_ids"].tolist() == [[0, 1]]
    assert summary["relion_nonzero_surface_error_safe_source_rotation_block_overflow"].tolist() == [False]
    assert summary["relion_nonzero_surface_error_safe_source_rotation_block_list_coverage"].tolist() == [True]

    overflow = summarize_coarse_gemm_streaming_state(
        state,
        state,
        _mask(5, [(0,)]),
        actual_image_count=1,
        adaptive_fraction=0.5,
        max_significants=1,
        band_widths=(140.0,),
        n_rotations=5,
        n_translations=1,
        source_rotation_block_size=2,
        rotation_block_capacity=1,
    )
    assert overflow["relion_nonzero_surface_error_safe_source_rotation_block_count"].tolist() == [2]
    assert overflow["relion_nonzero_surface_error_safe_source_rotation_block_ids"].tolist() == [[0]]
    assert overflow["relion_nonzero_surface_error_safe_source_rotation_block_sentinel_id"].tolist() == [1]
    assert overflow["relion_nonzero_surface_error_safe_source_rotation_block_overflow"].tolist() == [True]
    assert overflow["relion_nonzero_surface_error_safe_source_rotation_block_list_coverage"].tolist() == [False]


def test_raw_winner_block_is_unioned_when_a_poor_prior_excludes_its_posterior_block() -> None:
    pre_prior_scores = np.asarray(
        [[10.0, 9.0, 8.0, 7.0, 6.0, 5.0]],
        dtype=np.float32,
    )
    posterior_scores = np.asarray(
        [[-190.0, -191.0, 9.0, 8.0, 7.0, 6.0]],
        dtype=np.float32,
    )
    pre_prior_state = _stream(
        pre_prior_scores,
        pre_prior_scores,
        topk=6,
        block_size=2,
    )
    posterior_state = _stream(
        posterior_scores,
        posterior_scores,
        topk=6,
        block_size=2,
    )

    summary = summarize_coarse_gemm_streaming_state(
        posterior_state,
        pre_prior_state,
        _mask(6, [(2,)]),
        actual_image_count=1,
        adaptive_fraction=0.5,
        max_significants=1,
        band_widths=(0.0,),
        n_rotations=6,
        n_translations=1,
        source_rotation_block_size=2,
        rotation_block_capacity=4,
    )

    assert np.asarray(pre_prior_state.direct_candidate_ids)[0, 0] == 0
    assert summary["relion_nonzero_surface_error_safe_source_rotation_block_ids"].tolist() == [
        [1, 2, -1, -1]
    ]
    assert summary["raw_max_error_safe_width_from_macro_raw_max"].tolist() == [0.0]
    assert summary["raw_max_error_safe_source_rotation_block_ids"].tolist() == [
        [0, -1, -1, -1]
    ]
    assert summary["relion_rescore_source_rotation_block_union_ids"].tolist() == [
        [0, 1, 2, -1]
    ]
    assert summary[
        "relion_rescore_source_rotation_block_union_added_by_raw_max_count"
    ].tolist() == [1]
    assert summary[
        "relion_rescore_source_rotation_block_union_list_coverage"
    ].tolist() == [True]


def test_k2_final_rotation_padding_and_union_overflow_use_global_class_block_ids() -> None:
    n_rotations = 5
    n_translations = 2
    n_classes = 2
    rotation_block_size = 4
    candidate_count = n_classes * n_rotations * n_translations
    posterior_scores = np.arange(
        candidate_count,
        0,
        -1,
        dtype=np.float32,
    )[None, :]
    pre_prior_scores = np.zeros((1, candidate_count), dtype=np.float32)
    pre_prior_scores[0, 18] = 100.0

    def stream_padded(scores: np.ndarray) -> object:
        state = initialize_coarse_gemm_streaming_state(
            1,
            candidate_count,
            score_dtype=jnp.float32,
        )
        candidates_per_class = n_rotations * n_translations
        for class_index in range(n_classes):
            class_offset = class_index * candidates_per_class
            for r0 in range(0, n_rotations, rotation_block_size):
                valid_rotations = min(rotation_block_size, n_rotations - r0)
                valid_candidates = valid_rotations * n_translations
                block = np.full(
                    (1, rotation_block_size * n_translations),
                    -np.inf,
                    dtype=np.float32,
                )
                source_start = class_offset + r0 * n_translations
                block[:, :valid_candidates] = scores[
                    :,
                    source_start : source_start + valid_candidates,
                ]
                state = update_coarse_gemm_streaming_state(
                    state,
                    jnp.asarray(block),
                    jnp.asarray(block),
                    candidate_offset=source_start,
                    actual_image_count=1,
                )
        return state

    posterior_state = stream_padded(posterior_scores)
    pre_prior_state = stream_padded(pre_prior_scores)
    kwargs = dict(
        actual_image_count=1,
        adaptive_fraction=0.5,
        max_significants=1,
        band_widths=(0.0,),
        n_rotations=n_rotations,
        n_translations=n_translations,
        source_rotation_block_size=2,
    )
    summary = summarize_coarse_gemm_streaming_state(
        posterior_state,
        pre_prior_state,
        _mask(candidate_count, [(0,)]),
        rotation_block_capacity=6,
        **kwargs,
    )

    assert summary["finite_pair_count"].tolist() == [candidate_count]
    assert summary["pre_prior_finite_pair_count"].tolist() == [candidate_count]
    assert summary["nonfinite_pair_count"].tolist() == [0]
    assert summary["pre_prior_nonfinite_pair_count"].tolist() == [0]
    assert summary["raw_max_error_safe_source_rotation_block_ids"].tolist() == [
        [5, -1, -1, -1, -1, -1]
    ]
    assert summary["relion_rescore_source_rotation_block_union_ids"].tolist() == [
        [0, 1, 2, 3, 4, 5]
    ]
    assert summary[
        "relion_rescore_source_rotation_block_union_list_coverage"
    ].tolist() == [True]

    overflow = summarize_coarse_gemm_streaming_state(
        posterior_state,
        pre_prior_state,
        _mask(candidate_count, [(0,)]),
        rotation_block_capacity=5,
        **kwargs,
    )
    assert overflow["relion_rescore_source_rotation_block_union_ids"].tolist() == [
        [0, 1, 2, 3, 4]
    ]
    assert overflow[
        "relion_rescore_source_rotation_block_union_sentinel_id"
    ].tolist() == [5]
    assert overflow["relion_rescore_source_rotation_block_union_overflow"].tolist() == [
        True
    ]
    assert overflow[
        "relion_rescore_source_rotation_block_union_list_coverage"
    ].tolist() == [False]


def test_streaming_reductions_remain_float64_when_global_jax_x64_is_disabled() -> None:
    with jax.enable_x64(False):
        state = initialize_coarse_gemm_streaming_state(1, 2, score_dtype=jnp.float32)
        state = update_coarse_gemm_streaming_state(
            state,
            jnp.asarray([[2.0, 1.0]], dtype=jnp.float32),
            jnp.asarray([[2.25, 0.75]], dtype=jnp.float32),
            candidate_offset=0,
            actual_image_count=1,
        )

        assert state.direct_logsumexp_sum.dtype == jnp.dtype(np.float64)
        assert state.max_abs_delta.dtype == jnp.dtype(np.float64)
        assert state.signed_delta_sum.dtype == jnp.dtype(np.float64)
        assert state.squared_delta_sum.dtype == jnp.dtype(np.float64)
        assert state.finite_pair_count.dtype == jnp.dtype(np.int64)
        assert state.nonfinite_pair_count.dtype == jnp.dtype(np.int64)


def test_streaming_state_is_megabytes_instead_of_the_gf46_score_cube() -> None:
    state_bytes_observed_batch = coarse_gemm_streaming_state_bytes(187, 2048)
    state_bytes_requested_batch = coarse_gemm_streaming_state_bytes(500, 2048)
    dual_bytes_observed_batch = coarse_gemm_streaming_dual_state_bytes(187, 2048)
    dual_bytes_requested_batch = coarse_gemm_streaming_dual_state_bytes(500, 2048)
    paired_gf46_cube_bytes = 2 * 1000 * 36_864 * 29 * np.dtype(np.float32).itemsize

    assert state_bytes_observed_batch == 9_205_636
    assert state_bytes_requested_batch == 24_614_000
    assert dual_bytes_observed_batch == 18_411_272
    assert dual_bytes_requested_batch == 49_228_000
    assert state_bytes_observed_batch < 10 * 1024**2
    assert state_bytes_requested_batch < 25 * 1024**2
    assert dual_bytes_requested_batch < 50 * 1024**2
    assert paired_gf46_cube_bytes == 8_552_448_000
    assert dual_bytes_requested_batch * 150 < paired_gf46_cube_bytes


def test_streaming_excludes_padded_rows_and_structural_negative_infinity() -> None:
    state = initialize_coarse_gemm_streaming_state(2, 3, score_dtype=jnp.float32)
    state = update_coarse_gemm_streaming_state(
        state,
        jnp.asarray([[5.0, 4.0, -jnp.inf], [9.0, 8.0, 7.0]], dtype=jnp.float32),
        jnp.asarray([[5.0, jnp.nan, -jnp.inf], [9.0, 8.0, 7.0]], dtype=jnp.float32),
        candidate_offset=0,
        actual_image_count=1,
    )

    assert np.asarray(state.finite_pair_count).tolist() == [1, 0]
    assert np.asarray(state.nonfinite_pair_count).tolist() == [1, 0]
    assert np.asarray(state.direct_candidate_ids)[0].tolist() == [0, -1, -1, -1]
    summary = summarize_coarse_gemm_streaming_state(
        state,
        state,
        _mask(3, [(0,)]),
        actual_image_count=1,
        adaptive_fraction=0.5,
        max_significants=1,
        band_widths=(0.0,),
    )
    assert summary["all_candidate_error_coverage"].tolist() == [False]
    assert summary["winner_comparison_coverage"].tolist() == [False]
    assert summary["support_comparison_coverage"].tolist() == [False]


def test_streaming_request_defaults_above_maxsig_and_rejects_orphan_topk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR",
        str(tmp_path),
    )
    directory, topk = _coarse_gaussian_gemm_streaming_diagnostic_request(
        max_significants=500,
    )
    assert directory == str(tmp_path.resolve())
    assert topk == 2048

    monkeypatch.delenv("RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR")
    monkeypatch.setenv("RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_TOPK", "2048")
    with pytest.raises(ValueError, match="requires"):
        _coarse_gaussian_gemm_streaming_diagnostic_request(max_significants=500)


def test_compact_summary_writer_is_immutable_and_omits_ranked_tables(tmp_path: Path) -> None:
    direct = np.asarray([[5.0, 4.0, 3.0, 2.0]], dtype=np.float32)
    macro = np.asarray([[5.0, 3.0, 4.0, 2.0]], dtype=np.float32)
    state = _stream(direct, macro, topk=3, block_size=2)
    output = tmp_path / "summary.npz"
    kwargs = {
        "original_indices": np.asarray([2160], dtype=np.int64),
        "local_indices": np.asarray([999], dtype=np.int64),
        "actual_image_count": 1,
        "padded_image_count": 1,
        "adaptive_fraction": 0.999,
        "max_significants": 2,
        "diagnostic_run_id": "run",
        "diagnostic_call_id": "call",
        "debug_iteration": 181,
        "current_size": 100,
        "band_widths": (0.0, 1.0),
    }
    write_coarse_gemm_streaming_summary(
        str(output),
        state,
        state,
        _mask(4, [(0, 2)]),
        **kwargs,
    )

    with np.load(output, allow_pickle=False) as artifact:
        assert artifact["schema"].item() == COARSE_GEMM_STREAMING_SCHEMA
        assert artifact["original_indices"].tolist() == [2160]
        assert artifact["stores_score_cube"].item() is False
        assert "direct_scores" not in artifact
        assert "macro_scores" not in artifact
        assert artifact["production_behavior_changed"].item() is False
        assert artifact["posterior_streaming_state_bytes"].item() == 148
        assert artifact["pre_prior_streaming_state_bytes"].item() == 148
        assert artifact["persistent_streaming_state_bytes"].item() == 296
    assert output.stat().st_size < 100_000
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_coarse_gemm_streaming_summary(
            str(output),
            state,
            state,
            _mask(4, [(0, 2)]),
            **kwargs,
        )


def test_streaming_scope_manifest_seals_every_particle_once(tmp_path: Path) -> None:
    expected_calls = ("call0", "call1")
    paths = []
    particle_ids = (17, 23)
    for call_index, (call_id, particle_id) in enumerate(zip(expected_calls, particle_ids, strict=True)):
        direct = np.asarray([[5.0, -200.0, -100.0]], dtype=np.float32)
        state = _stream(direct, direct, topk=2, block_size=2)
        path = tmp_path / f"{call_id}.npz"
        write_coarse_gemm_streaming_summary(
            str(path),
            state,
            state,
            _mask(3, [(0,)]),
            original_indices=np.asarray([particle_id], dtype=np.int64),
            local_indices=np.asarray([call_index], dtype=np.int64),
            actual_image_count=1,
            padded_image_count=1,
            adaptive_fraction=0.5,
            max_significants=1,
            diagnostic_run_id="run",
            diagnostic_call_id=call_id,
            debug_iteration=181,
            current_size=100,
            band_widths=(0.0,),
            n_rotations=3,
            n_translations=1,
            source_rotation_block_size=2,
            rotation_block_capacity=2,
        )
        paths.append(path)
        scope = CoarseGaussianGemmDiagnosticScope(
            run_id="run",
            call_id=call_id,
            expected_call_ids=expected_calls,
            finalize=call_index == 1,
        )
        scope_path, aggregate_path = _seal_coarse_gaussian_gemm_streaming_scope(
            str(tmp_path),
            scope=scope,
            retained_topk=2,
            artifact_paths=[str(path)],
            original_indices=[particle_id],
        )
        assert Path(scope_path).is_file()
        if call_index == 0:
            assert aggregate_path is None

    assert aggregate_path is not None
    aggregate = json.loads(Path(aggregate_path).read_text())
    assert aggregate["particle_count"] == 2
    assert aggregate["all_particles_captured_exactly_once"] is True
    assert aggregate["stores_score_cube"] is False
    assert [record["original_indices"] for record in aggregate["scope_records"]] == [[17], [23]]
    report = aggregate["summary"]
    assert report["particle_count"] == 2
    assert report["all_candidate"]["error_coverage_count"] == 2
    assert report["all_candidate"]["max_abs_delta"] == 0.0
    assert report["pre_prior_all_candidate"]["error_coverage_count"] == 2
    assert report["pre_prior_all_candidate"]["max_abs_delta"] == 0.0
    assert report["winner"]["comparison_coverage_count"] == 2
    assert report["winner"]["mismatch_count"] == 0
    assert report["support"]["comparison_coverage_count"] == 2
    assert report["support"]["false_negative_total"] == 0
    assert report["support"]["false_positive_total"] == 0
    assert report["relion_raw_max_rescore"]["safe_pair_coverage_count"] == 2
    assert report["relion_raw_max_rescore"][
        "safe_source_rotation_block_list_coverage_count"
    ] == 2
    assert report["relion_rescore_union"][
        "source_rotation_block_list_coverage_count"
    ] == 2
    assert report["relion_rescore_union"]["source_rotation_block_count"][
        "maximum"
    ] == 2
