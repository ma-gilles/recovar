"""Focused contracts for InitialModel local/compact pass-2 routing."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _compact_pair_weighted_rotation_sums,
    subtract_projected_reference_from_sparse_mstep_rotation_sums,
    subtract_projected_reference_from_sparse_mstep_sums,
)
from recovar.em.dense_single_volume.helpers.types import make_relion_stats
from recovar.em.dense_single_volume.local_backprojection import (
    compute_relion_f32_sequential_mstep_sums,
)
from recovar.em.initial_model.dense_adapter import (
    _collapse_compact_pass2_rotation_stats_to_directions,
    _compact_sparse_pass2_enabled,
)

pytestmark = pytest.mark.unit


def test_compact_sparse_pass2_auto_routes_k1_local_and_kclass_compact(monkeypatch):
    monkeypatch.delenv("RECOVAR_INITIAL_MODEL_COMPACT_SPARSE_PASS2", raising=False)
    assert _compact_sparse_pass2_enabled(1) is False
    assert _compact_sparse_pass2_enabled(2) is True
    assert _compact_sparse_pass2_enabled(4) is True

    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("RECOVAR_INITIAL_MODEL_COMPACT_SPARSE_PASS2", value)
        assert _compact_sparse_pass2_enabled(1) is True


def test_explicit_pass2_engine_overrides_legacy_environment(monkeypatch):
    monkeypatch.setenv("RECOVAR_INITIAL_MODEL_COMPACT_SPARSE_PASS2", "0")
    assert _compact_sparse_pass2_enabled(2, "compact") is True
    monkeypatch.setenv("RECOVAR_INITIAL_MODEL_COMPACT_SPARSE_PASS2", "1")
    assert _compact_sparse_pass2_enabled(2, "local") is False

    with pytest.raises(ValueError, match="pass2_engine"):
        _compact_sparse_pass2_enabled(2, "unknown")


def test_compact_sparse_pass2_is_not_k1_scoped():
    from inspect import getsource

    from recovar.em.initial_model import dense_adapter

    source = getsource(dense_adapter._run_sparse_pass2_initial_model_estep)
    assert "compact sparse pass 2 is currently qualified only for K=1" not in source


def test_sparse_residual_mstep_matches_vdam_formula():
    summed = jnp.asarray(
        [[[7.0 + 2.0j, -3.0 + 1.0j], [4.0 - 1.0j, 5.0 + 3.0j]]],
        dtype=jnp.complex64,
    )
    reconstruction_probs = jnp.asarray(
        [[[0.2, 0.3, 0.0], [0.1, 0.15, 0.25]]],
        dtype=jnp.float32,
    )
    projected_reference = jnp.asarray(
        [[[2.0 + 1.0j, -1.0 + 0.5j], [3.0 - 2.0j, 0.5 + 1.0j]]],
        dtype=jnp.complex64,
    )
    ctf2_over_noise = jnp.asarray([[4.0, 0.25]], dtype=jnp.float32)

    actual = subtract_projected_reference_from_sparse_mstep_sums(
        summed,
        reconstruction_probs,
        projected_reference,
        ctf2_over_noise,
    )
    posterior_mass = np.asarray(reconstruction_probs).sum(axis=-1)
    expected = np.asarray(summed) - (
        posterior_mass[..., None]
        * np.asarray(projected_reference)
        * np.asarray(ctf2_over_noise)[:, None, :]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    from_mass = subtract_projected_reference_from_sparse_mstep_rotation_sums(
        summed,
        posterior_mass,
        projected_reference,
        ctf2_over_noise,
    )
    np.testing.assert_allclose(from_mass, expected, rtol=0.0, atol=0.0)


def test_compact_mstep_can_preserve_relion_translation_reduction(monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION", "1")
    pair_probs = jnp.asarray([[0.2, 0.3, 0.1, 0.4]], dtype=jnp.float32)
    rotation_rows = jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32)
    translation_ids = jnp.asarray([[0, 1, 0, 1]], dtype=jnp.int32)
    pair_mask = jnp.ones_like(pair_probs, dtype=bool)
    shifted = jnp.asarray(
        [[[1.0 + 2.0j, -3.0 + 0.5j], [4.0 - 1.0j, 2.0 + 3.0j]]],
        dtype=jnp.complex64,
    )
    ctf2 = jnp.asarray([[2.0, 0.25]], dtype=jnp.float32)

    summed, weight, probs_sum_t, _translation_posterior = (
        _compact_pair_weighted_rotation_sums(
            pair_probs,
            rotation_rows,
            translation_ids,
            pair_mask,
            shifted,
            ctf2,
            n_rotation_rows=2,
            relion_x_half=True,
        )
    )
    dense_probs = jnp.asarray([[[0.2, 0.3], [0.1, 0.4]]], dtype=jnp.float32)
    expected_summed, expected_weight = compute_relion_f32_sequential_mstep_sums(
        dense_probs,
        shifted,
        ctf2,
    )
    np.testing.assert_array_equal(summed, expected_summed)
    np.testing.assert_array_equal(weight, expected_weight)
    np.testing.assert_array_equal(probs_sum_t, jnp.sum(dense_probs, axis=-1))


class _StatsResult(NamedTuple):
    stats: object
    per_class_stats: tuple[object, ...]


def _stats(rotation_sums):
    return make_relion_stats(
        log_evidence_per_image=np.zeros(1),
        best_log_score_per_image=np.zeros(1),
        max_posterior_per_image=np.ones(1),
        rotation_posterior_sums=np.asarray(rotation_sums, dtype=np.float64),
        rotation_dtype=jnp.float64,
    )


def test_compact_rotation_statistics_collapse_psi_into_vdam_directions():
    result = _StatsResult(
        stats=_stats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        per_class_stats=(_stats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),),
    )

    collapsed = _collapse_compact_pass2_rotation_stats_to_directions(result, n_psi=3)

    np.testing.assert_array_equal(collapsed.stats.rotation_posterior_sums, [6.0, 15.0])
    np.testing.assert_array_equal(
        collapsed.per_class_stats[0].rotation_posterior_sums,
        [6.0, 15.0],
    )


def test_compact_rotation_statistics_reject_incompatible_psi_count():
    result = _StatsResult(stats=_stats([1.0, 2.0, 3.0]), per_class_stats=(_stats([1.0, 2.0, 3.0]),))

    with pytest.raises(ValueError, match="not divisible"):
        _collapse_compact_pass2_rotation_stats_to_directions(result, n_psi=2)
