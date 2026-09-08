"""CPU publication contracts; CUDA posterior arithmetic is qualified separately."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import coarse_publication as pub
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    CoarseGemmHybridBlockSelection,
    assemble_coarse_gemm_hybrid_compact_scores_f32,
)
from recovar.em.dense_single_volume.helpers.coarse_partition import CoarseRowResult
from recovar.em.dense_single_volume.helpers.significance import CoarseGaussianGemmHybridBatchResult

pytestmark = pytest.mark.unit


@pytest.fixture
def one_winner(monkeypatch):
    calls = []

    def posterior(values, **kwargs):
        calls.append(kwargs)
        finite = jnp.isfinite(values)
        mask = finite & (values == jnp.max(jnp.where(finite, values, -jnp.inf), axis=1)[:, None])
        count = jnp.sum(mask, axis=1, dtype=jnp.int32)
        return (
            mask.astype(jnp.float32),
            mask,
            count,
            count,
            jnp.ones(values.shape[0], jnp.float32),
            jnp.ones(values.shape[0], jnp.float32),
        )

    pub._posterior_statistics.clear_cache()
    monkeypatch.setattr(pub, "relion_cuda_f32_coarse_posterior", posterior)
    yield calls
    pub._posterior_statistics.clear_cache()


def mixed_groups():
    counts = np.array([1, 1, 0, 0], np.int32)
    selection = CoarseGemmHybridBlockSelection(
        True, None, np.array([[1], [0], [-1], [-1]], np.int32), counts, counts.copy(), counts.copy()
    )
    diff = np.full((4, 1, 16, 2), np.inf, np.float32)
    diff[:2] = 1000.0
    diff[0, 0, 1, 1] = 0.0
    diff[1, 0, 2, 0] = 0.0
    compact = assemble_coarse_gemm_hybrid_compact_scores_f32(
        diff, selection, actual_image_count=2, n_rotations=32, class_log_prior=0.0
    )
    selected = CoarseGaussianGemmHybridBatchResult(
        None, compact.raw_score_max, True, True, None, selection, compact_scores=compact
    )
    raw = np.full((3, 32, 2), np.nan, np.float32)
    raw[0] = -1000.0
    raw[0, 2, 1] = 0.0
    full = CoarseGaussianGemmHybridBatchResult(raw, np.zeros(3, np.float32), False, False, "overflow", None)
    return (
        CoarseRowResult(np.array([0, 2], np.int32), selected, None),
        CoarseRowResult(np.array([1], np.int32), full, None),
    )


def publish(groups, **kwargs):
    options = dict(
        actual_image_count=3,
        n_rotations=32,
        n_translations=2,
        class_log_prior=0.0,
        rotation_log_prior=None,
        rotation_chunk_rows=16,
        adaptive_fraction=0.999,
        max_significants=100,
        tie_score_ulps=0,
    )
    options.update(kwargs)
    return pub.publish_coarse_rows(groups, **options)


def test_mixed_publication_restores_source_ids_and_excludes_poisoned_padding(one_winner):
    result = publish(mixed_groups())
    for key in ("winner", "best_pose", "support_ids"):
        np.testing.assert_array_equal(result[key], [35, 5, 4])
    np.testing.assert_array_equal(result["support_offsets"], [0, 1, 2, 3])
    for key in ("n_significant", "cutoff_count", "pmax", "sum_weight", "threshold"):
        np.testing.assert_array_equal(result[key], [1, 1, 1])
    for key in ("best_score", "raw_max", "global_log_z"):
        np.testing.assert_array_equal(result[key], [0.0, 0.0, 0.0])
    assert len(result) == 12


def test_posterior_policy_is_forwarded_without_hardcoded_probe_values(one_winner):
    publish(mixed_groups(), adaptive_fraction=0.975, max_significants=7, tie_score_ulps=2)
    assert one_winner
    for call in one_winner:
        assert call["adaptive_fraction"] == 0.975
        assert call["max_significants"] == 7
        assert call["tie_score_ulps"] == 2
        assert call["filter_positive_before_sort"] is False


@pytest.mark.parametrize("rows", [[0, 0], [0], [0, 3], [-1, 2], [0.0, 2.0]])
def test_invalid_source_image_coverage_is_rejected_before_posterior(rows, one_winner):
    selected, full = mixed_groups()
    selected = selected._replace(image_indices=np.asarray(rows))
    with pytest.raises(ValueError, match="cover each original image"):
        publish((selected, full))
    assert not one_winner


def test_dense_scores_that_already_include_priors_are_rejected(one_winner):
    selected, full = mixed_groups()
    full = full._replace(result=full.result._replace(scores_include_priors=True))
    with pytest.raises(ValueError, match="precede prior addition"):
        publish((selected, full))


def test_nonfinite_actual_statistics_are_not_silently_published(one_winner):
    selected, full = mixed_groups()
    raw = np.asarray(full.result.scores).copy()
    raw[0] = np.nan
    full = full._replace(result=full.result._replace(scores=raw))
    with pytest.raises(ValueError, match="Nonfinite actual coarse statistic"):
        publish((selected, full))


@pytest.mark.parametrize("first_group", ["mixed", "all_overflow", "invalid_certificate"])
def test_actual_significance_engine_publishes_identical_complete_state(monkeypatch, one_winner, first_group):
    """Exercise the engine boundary on fixed scores, including all host outputs."""
    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers import coarse_partition, oversampling, projection, significance
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sparse
    from recovar.em.dense_single_volume.helpers.coarse_partition import CoarseRowGroup, CoarseRowPlan
    import runpy
    from pathlib import Path

    _MacroIntegrationDataset = runpy.run_path(str(Path(__file__).parents[1] / "test_coarse_gaussian_gemm_macro.py"))[
        "_MacroIntegrationDataset"
    ]

    for name, value in {
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY": "1",
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI": "1",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF": "1",
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR": "0",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "0",
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "0",
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE": "0",
        "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS": "1",
        "RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "1",
        "RECOVAR_COARSE_ROW_PARTITION": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(significance.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "cuda_available", lambda: True)
    monkeypatch.setattr(oversampling, "relion_cuda_f32_coarse_posterior", pub.relion_cuda_f32_coarse_posterior)
    for name in ("_relion_exact_ctf_half_from_source_star", "_relion_exact_ctf_half_from_source_star_host"):
        monkeypatch.setattr(
            sparse, name, lambda _dataset, indices, image_shape: np.ones((len(indices), 12), np.float64)
        )
    monkeypatch.setattr(
        sparse,
        "_relion_cuda_powerclass_highres_xi2_half",
        lambda processed, **kwargs: jnp.zeros(processed.shape[0], jnp.float32),
    )
    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f32",
        lambda images, angles, pixels, shape: jnp.repeat(images[:, None, :], len(angles), axis=1).reshape(
            len(images) * len(angles), -1
        ),
    )

    def project(_volume, rotations, _shape, **kwargs):
        values = jnp.ones((len(rotations), len(kwargs["pixel_indices"])), jnp.complex64)
        return values, jnp.ones(values.shape, jnp.float32) if kwargs.get("return_abs2", True) else None

    monkeypatch.setattr(projection, "compute_relion_projector_projections_block", project)
    selected, full = mixed_groups()
    full = full._replace(
        result=full.result._replace(
            full_dense_backend="rectangular",
            full_dense_kernel="rectangular",
            score_representation="dense_full_direct_static_capacity",
        )
    )
    groups = (selected, full)
    plan = CoarseRowPlan(
        tuple(CoarseRowGroup(g.image_indices, len(g.result.raw_score_max), g.result.selection) for g in groups),
        "block_capacity_overflow",
        True,
    )
    raw = np.full((3, 32, 2), -1000.0, np.float32)
    raw[0, 17, 1] = raw[1, 2, 1] = raw[2, 2, 0] = 0.0
    baseline = full.result._replace(scores=jnp.asarray(raw))
    monkeypatch.setattr(significance, "_compute_coarse_gaussian_gemm_hybrid_batch", lambda *args, **kwargs: baseline)
    partition_calls = []

    counts = np.ones(3, np.int32)
    all_selection = CoarseGemmHybridBlockSelection(
        True, None, np.array([[1], [0], [0]], np.int32), counts, counts.copy(), counts.copy()
    )
    selected_raw = np.stack([raw[0, 16:32], raw[1, :16], raw[2, :16]])[:, None]
    compact = assemble_coarse_gemm_hybrid_compact_scores_f32(
        -selected_raw, all_selection, actual_image_count=3, n_rotations=32, class_log_prior=0.0
    )
    all_selected = CoarseRowResult(
        np.arange(3, dtype=np.int32),
        selected.result._replace(raw_score_max=compact.raw_score_max, selection=all_selection, compact_scores=compact),
        None,
    )
    all_selected_plan = CoarseRowPlan((CoarseRowGroup(all_selected.image_indices, 3, all_selection),), None, False)
    full_group = CoarseRowResult(np.arange(3, dtype=np.int32), baseline, None)
    full_plan = CoarseRowPlan(
        (CoarseRowGroup(full_group.image_indices, 3, None),),
        "block_capacity_overflow" if first_group == "all_overflow" else "invalid_certificate",
        False,
    )

    def partition(*args, **kwargs):
        partition_calls.append(kwargs)
        assert kwargs["actual_image_count"] == 3
        np.testing.assert_array_equal(kwargs["translation_log_prior"], np.zeros((3, 2), np.float32))
        if len(partition_calls) > 1:
            return all_selected_plan, (all_selected,)
        return (plan, groups) if first_group == "mixed" else (full_plan, (full_group,))

    monkeypatch.setattr(coarse_partition, "compute_partitioned_coarse_batch", partition)
    dataset = _MacroIntegrationDataset(np.arange(6))

    def run():
        return significance._compute_k_class_significance_batched(
            dataset,
            jnp.zeros((1, 64), jnp.complex64),
            jnp.ones(16, jnp.float32),
            np.tile(np.eye(3, dtype=np.float32), (32, 1, 1)),
            jnp.zeros((2, 2), jnp.float32),
            "linear_interp",
            class_log_priors=np.zeros(1),
            translation_log_prior=jnp.zeros(2, jnp.float32),
            adaptive_fraction=0.999,
            max_significants=100,
            image_batch_size=3,
            rotation_block_size=16,
            current_size=4,
            half_spectrum_scoring=True,
            relion_projector_half=jnp.zeros((1, 3, 3, 2), jnp.complex64),
            relion_projector_r_max=1,
            relion_projector_texture_interp=True,
            collect_significance=True,
            pad_final_image_batch=True,
        )

    control = run()
    assert not partition_calls
    monkeypatch.setenv("RECOVAR_COARSE_ROW_PARTITION", "1")
    candidate = run()
    assert len(partition_calls) == (1 if first_group == "all_overflow" else 2)
    for i in range(4):
        np.testing.assert_array_equal(candidate[i], control[i])
    for actual, expected in zip(candidate[4][0], control[4][0], strict=True):
        np.testing.assert_array_equal(actual, expected)
    fields = [key for key, value in control[5].items() if isinstance(value, np.ndarray)]
    assert "relion_f32_sum_weight" in fields and "normalization_log_z" in fields
    for key in fields:
        np.testing.assert_array_equal(candidate[5][key], control[5][key])
    audit = candidate[5]["coarse_gaussian_gemm_hybrid"]
    assert audit["row_partition"]["mixed_input_batch_count"] == int(first_group == "mixed")
    assert audit["row_partition"]["input_batch_count"] == 2
    assert audit["row_partition"]["execution_group_count"] == (3 if first_group == "mixed" else 2)
    assert (
        audit["selected_rescore_image_count"] == {"mixed": 5, "all_overflow": 0, "invalid_certificate": 3}[first_group]
    )
    assert audit["full_dense_image_count"] == {"mixed": 1, "all_overflow": 6, "invalid_certificate": 3}[first_group]
    assert audit["overflow_latch_activation_count"] == int(first_group == "all_overflow")
