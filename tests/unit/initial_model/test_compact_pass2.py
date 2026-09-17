"""Focused contracts for the single InitialModel pass-2 engine."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.mstep_reference import numpy_relion_f32_mstep_sums

from recovar.em.helpers.types import make_relion_stats
from recovar.em.sparse_pass2.sparse_pass2_compact_pair_sums import _compact_pair_weighted_rotation_sums
from recovar.em.sparse_pass2.sparse_pass2_window import (
    subtract_projected_reference_from_sparse_mstep_rotation_sums,
    subtract_projected_reference_from_sparse_mstep_sums,
)
from recovar.em.vdam.sparse_pass2_estep import _resolve_pass2_engine

pytestmark = pytest.mark.unit


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
    expected_summed, expected_weight = numpy_relion_f32_mstep_sums(
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




def test_one_engine_serves_k1_and_kclass():
    """The compact sparse pass-2 route is gone: every selector resolves to the exact-local
    engine, which scores all classes in one pass over class-segmented rows."""
    for token in ("auto", "local", "local_segmented", " AUTO "):
        assert _resolve_pass2_engine(token) in {"auto", "local", "local_segmented"}


def test_compact_engine_is_no_longer_selectable():
    with pytest.raises(ValueError, match="must be one of 'auto', 'local' or 'local_segmented'"):
        _resolve_pass2_engine("compact")
