"""Focused tests for the opt-in RELION float32 fine-posterior diagnostic."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _RELION_X_HALF_F32_FINE_POSTERIOR_ENV,
    _relion_f32_fine_reconstruction_probs,
    _relion_pass2_reconstruction_probs,
    _relion_pass2_reconstruction_probs_for_mstep,
)

pytestmark = pytest.mark.unit


def _numpy_relion_f32_reference(scores, adaptive_fraction):
    scores = np.asarray(scores, dtype=np.float32)
    flat = scores.reshape(scores.shape[0], -1)
    output_probs = np.zeros_like(flat, dtype=np.float32)
    output_mask = np.zeros_like(flat, dtype=bool)
    counts = np.zeros(flat.shape[0], dtype=np.int32)
    sum_weights = np.zeros(flat.shape[0], dtype=np.float32)
    thresholds = np.zeros(flat.shape[0], dtype=np.float32)

    for row_idx, row in enumerate(flat):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        best = np.max(row[finite]).astype(np.float32)
        shifted = np.full(row.shape, -np.inf, dtype=np.float32)
        shifted[finite] = (row[finite] - best + np.float32(50.0)).astype(np.float32)
        raw = np.zeros_like(row, dtype=np.float32)
        exponentiated = finite & (shifted >= np.float32(-88.0))
        raw[exponentiated] = np.exp(shifted[exponentiated]).astype(np.float32)
        raw[~np.isfinite(raw)] = np.float32(0.0)

        cumulative = np.cumsum(np.sort(raw), dtype=np.float32)
        sum_weight = cumulative[-1]
        if not np.isfinite(sum_weight) or sum_weight <= np.float32(0.0):
            continue
        target = np.float32(np.float64(1.0 - adaptive_fraction) * np.float64(sum_weight))
        threshold_idx = min(int(np.searchsorted(cumulative, target, side="right")), row.size - 1)
        threshold = np.sort(raw)[threshold_idx]
        mask = finite & (raw >= threshold)

        output_mask[row_idx] = mask
        output_probs[row_idx, mask] = (raw[mask] / sum_weight).astype(np.float32)
        counts[row_idx] = np.int32(np.sum(mask))
        sum_weights[row_idx] = sum_weight
        thresholds[row_idx] = threshold

    return (
        output_probs.reshape(scores.shape),
        output_mask.reshape(scores.shape),
        counts,
        sum_weights,
        thresholds,
    )


def test_relion_f32_fine_posterior_matches_numpy_reference_with_cutoff_ties():
    scores = np.asarray(
        [
            [[0.0, -1.0, -1.0, -4.0], [-6.0, -np.inf, -9.0, -9.0]],
            [[0.0, 0.0, -2.0, np.nan], [-5.0, -5.0, -12.0, -140.0]],
            [[-np.inf, np.nan, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf, -np.inf]],
        ],
        dtype=np.float32,
    )
    adaptive_fraction = 0.75

    actual = tuple(
        np.asarray(value)
        for value in _relion_f32_fine_reconstruction_probs(
            jnp.asarray(scores),
            adaptive_fraction=adaptive_fraction,
        )
    )
    expected = _numpy_relion_f32_reference(scores, adaptive_fraction)

    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=0.0)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_allclose(actual[3], expected[3], rtol=2e-6, atol=0.0)
    np.testing.assert_allclose(actual[4], expected[4], rtol=2e-6, atol=0.0)
    # Equal raw weights at the significance boundary must be retained together.
    assert actual[1][0, 0, 1] == actual[1][0, 0, 2]
    # Invalid padded hypotheses must never be revived when a threshold is zero.
    assert not np.any(actual[1][2])


def test_relion_f32_fine_posterior_gate_off_preserves_default(monkeypatch):
    monkeypatch.delenv(_RELION_X_HALF_F32_FINE_POSTERIOR_ENV, raising=False)
    scores = jnp.asarray([[[0.0, -0.2, -1.7], [-2.1, -3.5, -np.inf]]], dtype=jnp.float32)
    probs = jnp.asarray([[[0.52, 0.31, 0.09], [0.05, 0.03, 0.0]]], dtype=jnp.float64)

    expected = _relion_pass2_reconstruction_probs(probs, adaptive_fraction=0.9)
    actual = _relion_pass2_reconstruction_probs_for_mstep(
        scores,
        probs,
        adaptive_fraction=0.9,
        use_relion_x_half_mstep=True,
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_value), np.asarray(expected_value))


def test_relion_f32_fine_posterior_gate_is_xhalf_only(monkeypatch):
    monkeypatch.setenv(_RELION_X_HALF_F32_FINE_POSTERIOR_ENV, "1")
    scores = jnp.asarray([[[0.0, -0.2, -1.7], [-2.1, -3.5, -np.inf]]], dtype=jnp.float32)
    probs = jnp.asarray([[[0.52, 0.31, 0.09], [0.05, 0.03, 0.0]]], dtype=jnp.float64)

    expected = _relion_pass2_reconstruction_probs(probs, adaptive_fraction=0.9)
    actual = _relion_pass2_reconstruction_probs_for_mstep(
        scores,
        probs,
        adaptive_fraction=0.9,
        use_relion_x_half_mstep=False,
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_value), np.asarray(expected_value))


def test_sparse_pass2_winner_take_all_excludes_f32_posterior_override(monkeypatch):
    monkeypatch.setenv(_RELION_X_HALF_F32_FINE_POSTERIOR_ENV, "1")
    scores = jnp.asarray([[[0.0, -0.2, -1.7], [-2.1, -3.5, -np.inf]]], dtype=jnp.float32)
    probs = jnp.asarray([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=jnp.float64)

    expected = _relion_pass2_reconstruction_probs(probs, adaptive_fraction=0.9)
    actual = _relion_pass2_reconstruction_probs_for_mstep(
        scores,
        probs,
        adaptive_fraction=0.9,
        use_relion_x_half_mstep=True,
        winner_take_all=True,
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_value), np.asarray(expected_value))
