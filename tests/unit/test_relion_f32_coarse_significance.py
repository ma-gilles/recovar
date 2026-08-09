"""RELION CUDA float32 coarse-posterior significance tests."""

import numpy as np

from recovar.em.dense_single_volume.helpers.oversampling import (
    relion_cuda_f32_coarse_posterior,
)
from recovar.em.dense_single_volume.helpers.significance import (
    _K1_RELION_F32_COARSE_SUPPORT_ENV,
    _k1_relion_f32_coarse_support_enabled,
)


def _numpy_reference(scores, adaptive_fraction, max_significants):
    scores = np.asarray(scores, dtype=np.float32)
    probabilities = np.zeros_like(scores)
    masks = np.zeros_like(scores, dtype=bool)
    n_significant = np.zeros(scores.shape[0], dtype=np.int32)
    cutoff_count = np.zeros(scores.shape[0], dtype=np.int32)
    sums = np.zeros(scores.shape[0], dtype=np.float32)
    thresholds = np.zeros(scores.shape[0], dtype=np.float32)
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        best = np.max(row[finite])
        shifted = np.where(finite, row - best + np.float32(50.0), -np.inf).astype(np.float32)
        raw = np.where(shifted < np.float32(-88.0), np.float32(0.0), np.exp(shifted)).astype(
            np.float32,
        )
        raw = np.where(finite & np.isfinite(raw), raw, np.float32(0.0))
        positive = raw[raw > 0.0]
        ordered = np.sort(positive)
        cumulative = np.cumsum(ordered, dtype=np.float32)
        total = cumulative[-1]
        target = np.float32((1.0 - adaptive_fraction) * np.float64(total))
        threshold_index = int(np.searchsorted(cumulative, target, side="right"))
        if max_significants is not None and max_significants > 0:
            threshold_index = max(threshold_index, len(ordered) - max_significants)
        threshold = ordered[threshold_index]
        mask = (raw > 0.0) & (raw >= threshold)
        probabilities[row_index] = raw / total
        masks[row_index] = mask
        n_significant[row_index] = np.count_nonzero(mask)
        cutoff_count[row_index] = len(ordered) - threshold_index
        sums[row_index] = total
        thresholds[row_index] = threshold
    return probabilities, masks, n_significant, cutoff_count, sums, thresholds


def test_relion_cuda_f32_coarse_posterior_matches_numpy_reference():
    scores = np.array(
        [
            [4.0, 3.5, 3.5, -100.0, -np.inf, 1.0],
            [-7.0, -7.25, -8.0, -9.0, -10.0, -11.0],
        ],
        dtype=np.float32,
    )
    expected = _numpy_reference(scores, adaptive_fraction=0.8, max_significants=4)
    actual = tuple(
        np.asarray(value)
        for value in relion_cuda_f32_coarse_posterior(
            scores,
            adaptive_fraction=0.8,
            max_significants=4,
        )
    )
    for actual_value, expected_value in zip(actual[1:4], expected[1:4]):
        np.testing.assert_array_equal(actual_value, expected_value)
    for actual_value, expected_value in (
        (actual[0], expected[0]),
        (actual[4], expected[4]),
        (actual[5], expected[5]),
    ):
        np.testing.assert_allclose(
            actual_value,
            expected_value,
            rtol=np.finfo(np.float32).eps,
            atol=0.0,
        )


def test_relion_cuda_f32_coarse_posterior_expands_cutoff_ties_after_rank_cap():
    scores = np.array([[5.0, 4.0, 4.0, 4.0, 0.0]], dtype=np.float32)
    _, mask, n_significant, cutoff_count, _, _ = (
        relion_cuda_f32_coarse_posterior(
            scores,
            adaptive_fraction=0.999,
            max_significants=2,
        )
    )
    np.testing.assert_array_equal(np.asarray(mask), [[True, True, True, True, False]])
    np.testing.assert_array_equal(np.asarray(n_significant), [4])
    np.testing.assert_array_equal(np.asarray(cutoff_count), [2])


def test_relion_f32_coarse_support_gate_defaults_off(monkeypatch):
    monkeypatch.delenv(_K1_RELION_F32_COARSE_SUPPORT_ENV, raising=False)
    assert not _k1_relion_f32_coarse_support_enabled()
    monkeypatch.setenv(_K1_RELION_F32_COARSE_SUPPORT_ENV, "1")
    assert _k1_relion_f32_coarse_support_enabled()
