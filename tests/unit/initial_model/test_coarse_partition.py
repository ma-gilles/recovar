"""Exact CPU coverage of row partition contracts, independent of CUDA timing."""

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import coarse_partition as p
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    CoarseGemmHybridIntervalState,
    validate_coarse_gemm_hybrid_block_selection_for_rescore,
)

pytestmark = pytest.mark.unit


def certificate(counts, *, physical=None, blocks=80, translations=3):
    actual = len(counts)
    physical = actual if physical is None else physical
    lower = np.full((physical, blocks), -1000.0, np.float64)
    upper = lower.copy()
    for row, count in enumerate(counts):
        lower[row, 0] = 0.0
        upper[row, :count] = 0.0
    candidates = np.zeros(physical, np.int64)
    candidates[:actual] = blocks * 16 * translations
    return CoarseGemmHybridIntervalState(
        lower,
        upper,
        lower.copy(),
        upper.copy(),
        np.ones(blocks * 16, np.int32),
        candidates,
        np.zeros(physical, np.int64),
    )


def plan(state, actual):
    return p.plan_coarse_rows(state, actual_image_count=actual, n_rotations=80 * 16, n_translations=3)


def test_one_overflow_does_not_promote_other_images_or_padding():
    result = plan(certificate([1, 65, 64, 2], physical=7), 4)
    assert result.partitioned and result.fallback_reason == "block_capacity_overflow"
    selected, full = result.groups
    np.testing.assert_array_equal(selected.image_indices, [0, 2, 3])
    np.testing.assert_array_equal(full.image_indices, [1])
    assert selected.physical_count == full.physical_count == 32
    assert full.selection is None
    np.testing.assert_array_equal(selected.selection.block_count[:3], [1, 64, 2])
    for row, count in enumerate((1, 64, 2)):
        np.testing.assert_array_equal(selected.selection.block_ids[row, :count], np.arange(count))
        assert np.all(selected.selection.block_ids[row, count:] == -1)
    assert np.all(selected.selection.block_ids[3:] == -1)
    assert np.all(selected.selection.block_count[3:] == 0)
    validate_coarse_gemm_hybrid_block_selection_for_rescore(selected.selection, actual_image_count=3, n_rotations=1280)


@pytest.mark.parametrize("counts,selected", [([1, 64, 3], True), ([65, 80, 66], False)])
def test_homogeneous_batch_preserves_original_physical_shape(counts, selected):
    result = plan(certificate(counts, physical=5), 3)
    assert not result.partitioned and len(result.groups) == 1
    group = result.groups[0]
    assert group.physical_count == 5 and (group.selection is not None) == selected
    np.testing.assert_array_equal(group.image_indices, [0, 1, 2])


@pytest.mark.parametrize(
    "field", ["rotation_visit_count", "candidate_count", "invalid_candidate_count", "posterior_block_upper_max"]
)
def test_invalid_certificate_keeps_entire_batch_full(field):
    state = certificate([1, 65, 2])
    bad = np.array(getattr(state, field), copy=True)
    if field == "rotation_visit_count":
        bad[7] = 2
    elif field == "candidate_count":
        bad[0] -= 1
    elif field == "invalid_candidate_count":
        bad[0] = 1
    else:
        bad[0, 1] = np.nan
    result = plan(state._replace(**{field: bad}), 3)
    assert not result.partitioned and len(result.groups) == 1
    assert result.groups[0].selection is None
    assert result.fallback_reason != "block_capacity_overflow"
    np.testing.assert_array_equal(result.groups[0].image_indices, [0, 1, 2])


@pytest.mark.parametrize("kw", [{"actual_image_count": 0}, {"block_capacity": 0}, {"row_quantum": 0}])
def test_invalid_plan_configuration_rejected(kw):
    config = dict(actual_image_count=2, n_rotations=1280, n_translations=3)
    config.update(kw)
    with pytest.raises(ValueError):
        p.plan_coarse_rows(certificate([1, 65]), **config)


@pytest.mark.parametrize("has_prior", [False, True])
def test_device_gather_retains_image_order_and_zeroes_poisoned_padding(has_prior):
    shifted = np.arange(4 * 3 * 2, dtype=np.float32).astype(np.complex64).reshape(4, 3, 2)
    weight = np.arange(8, dtype=np.float32).reshape(4, 2)
    initial = np.arange(4, dtype=np.float32)
    prior = np.arange(12, dtype=np.float32).reshape(4, 3) if has_prior else None
    shifted[0] = np.nan
    weight[0] = np.inf
    initial[0] = np.nan
    if has_prior:
        prior[0] = np.nan
    rows = np.array([3, 1], np.int32)
    result = p.gather_coarse_rows(shifted, weight, initial, prior, rows, physical_count=5)
    for source, got in zip((shifted, weight, initial, prior), result, strict=True):
        if source is None:
            assert got is None
            continue
        np.testing.assert_array_equal(got[:2], source[rows])
        np.testing.assert_array_equal(got[2:], np.zeros_like(np.asarray(got[2:])))
        assert got.dtype == source.dtype


@pytest.mark.parametrize("invalid_selected", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_composition_certifies_once_and_only_full_scores_required_groups(monkeypatch, invalid_selected, capture):
    from recovar.em.dense_single_volume.helpers import significance as s
    from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import plan_coarse_gemm_certificate_topology

    state = certificate([1, 65, 64, 2], physical=5)
    topology = plan_coarse_gemm_certificate_topology(
        np.arange(2, dtype=np.int32), compact_pixel_count=2, translation_count=3
    )
    shifted = np.broadcast_to(np.arange(5, dtype=np.float32)[:, None, None], (5, 3, 2)).astype(np.complex64).copy()
    weight = np.ones((5, 2), np.float32)
    initial = np.arange(5, dtype=np.float32)
    prior = np.broadcast_to(initial[:, None], (5, 3)).copy()
    cache = np.zeros((1, 1280, 2), np.complex64)
    events = []
    prepared = object()

    def prepare(x, w, i, n):
        np.testing.assert_array_equal(x, shifted)
        assert n == 4
        events.append("prepare")
        return prepared

    def update(value, projected, images, **kwargs):
        assert value is state and images is prepared
        assert kwargs["real_cross"] is True and kwargs["rotation_offset"] == 0
        np.testing.assert_array_equal(kwargs["translation_log_prior"], prior)
        events.append("certificate")
        return state

    def selected(projected, x, w, i, ids, **kwargs):
        np.testing.assert_array_equal(x[:3], shifted[[0, 2, 3]])
        assert x.shape == (32, 3, 2) and kwargs["logical_full_pixel_count"] == 2
        events.append("selected")
        result = np.where(np.asarray(ids)[:, :, None, None] >= 0, 0.0, np.inf)
        result = np.broadcast_to(result, (32, 64, 16, 3)).astype(np.float32).copy()
        if invalid_selected:
            result[0, 0, 0, 0] = np.nan
        return result

    full_rows = []

    def full(projected, x, w, i, **kwargs):
        assert kwargs["force_static_dense_after_overflow"] is True
        n = kwargs["actual_image_count"]
        full_rows.append(np.asarray(i[:n]).tolist())
        np.testing.assert_array_equal(kwargs["translation_log_prior"][:n], prior[np.asarray(i[:n], np.int32)])
        return s.CoarseGaussianGemmHybridBatchResult(
            scores=np.zeros((32, 1280, 3), np.float32),
            raw_score_max=np.zeros(32, np.float32),
            scores_include_priors=False,
            used_selected_rescore=False,
            fallback_reason="sentinel",
            selection=None,
            score_representation="dense_full_direct_static_capacity",
        )

    monkeypatch.setattr(s, "_prepare_relion_coarse_gaussian_gemm_f64_image_batch", prepare)
    monkeypatch.setattr(s, "initialize_coarse_gemm_hybrid_interval_state", lambda *args: state)
    monkeypatch.setattr(s, "_relion_coarse_gaussian_gemm_update_certificate_state", update)
    monkeypatch.setattr(s, "_relion_coarse_diff2_rotation_blocks_from_topology_f32", selected)
    monkeypatch.setattr(s, "_compute_coarse_gaussian_gemm_hybrid_batch", full)
    result, groups = p.compute_partitioned_coarse_batch(
        cache,
        shifted,
        weight,
        initial,
        topology=topology,
        actual_image_count=4,
        translation_log_prior=prior,
        certificate_chunk_rows=1280,
        real_cross=True,
        logical_full_pixel_count=2,
        capture_selected_diff2=capture,
    )
    assert result.partitioned and events == ["prepare", "certificate", "selected"]
    assert full_rows == ([[0.0, 2.0, 3.0], [1.0]] if invalid_selected else [[1.0]])
    np.testing.assert_array_equal(groups[0].image_indices, [0, 2, 3])
    np.testing.assert_array_equal(groups[1].image_indices, [1])
    assert groups[0].result.used_selected_rescore is (not invalid_selected)
    if invalid_selected:
        assert groups[0].result.fallback_reason == "invalid_selected_exact_output"
    else:
        assert groups[0].result.scores is None
        assert (groups[0].result.diagnostic_selected_diff2 is not None) == capture
        np.testing.assert_array_equal(groups[0].result.compact_scores.block_count[:3], [1, 64, 2])
    assert groups[1].result.fallback_reason == "block_capacity_overflow"
