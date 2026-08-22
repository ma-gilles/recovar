"""Focused contracts for the opt-in shared compact InitialModel pass 2."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    subtract_projected_reference_from_sparse_mstep_sums,
)
from recovar.em.dense_single_volume.helpers.types import make_relion_stats
from recovar.em.initial_model.dense_adapter import (
    _collapse_compact_pass2_rotation_stats_to_directions,
    _compact_sparse_pass2_enabled,
)

pytestmark = pytest.mark.unit


def test_compact_sparse_pass2_is_opt_in(monkeypatch):
    monkeypatch.delenv("RECOVAR_INITIAL_MODEL_COMPACT_SPARSE_PASS2", raising=False)
    assert _compact_sparse_pass2_enabled() is False

    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("RECOVAR_INITIAL_MODEL_COMPACT_SPARSE_PASS2", value)
        assert _compact_sparse_pass2_enabled() is True


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
