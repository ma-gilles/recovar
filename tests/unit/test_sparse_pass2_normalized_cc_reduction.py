"""RELION firstiter-CC float32 reduction parity tests."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _score_pass2_bucket_normalized_cc,
    _score_pass2_bucket_normalized_cc_single_cached,
    _score_pass2_pairs_normalized_cc,
)

pytestmark = pytest.mark.unit


def _inputs():
    rng = np.random.default_rng(161552)
    batch, n_rot, n_trans, n_pix = 1, 3, 4, 17
    shifted = (
        rng.normal(size=(batch, n_trans, n_pix))
        + 1j * rng.normal(size=(batch, n_trans, n_pix))
    ).astype(np.complex64)
    projections = (
        rng.normal(size=(batch, n_rot, n_pix))
        + 1j * rng.normal(size=(batch, n_rot, n_pix))
    ).astype(np.complex64)
    score_weight = rng.uniform(0.1, 1.5, size=(batch, n_pix)).astype(np.float32)
    half_weight = rng.choice(np.asarray([1.0, 2.0], dtype=np.float32), size=n_pix)
    mask = np.ones((batch, n_rot, n_trans), dtype=bool)
    return shifted, score_weight, projections, half_weight, mask


def test_normalized_cc_bucket_cached_are_bitwise_equal_and_pair_order_matches():
    shifted, score_weight, projections, half_weight, mask = _inputs()
    bucket = np.asarray(
        _score_pass2_bucket_normalized_cc(
            jnp.asarray(shifted),
            jnp.asarray(score_weight),
            jnp.asarray(projections),
            jnp.asarray(half_weight),
            jnp.asarray(mask),
        )
    )[0]
    cached = np.asarray(
        _score_pass2_bucket_normalized_cc_single_cached(
            jnp.asarray(shifted[0]),
            jnp.asarray(score_weight[0]),
            jnp.asarray(projections[0]),
            jnp.asarray(half_weight),
            jnp.asarray(mask[0]),
        )
    )

    rotation_rows, translation_rows = np.meshgrid(
        np.arange(projections.shape[1], dtype=np.int32),
        np.arange(shifted.shape[1], dtype=np.int32),
        indexing="ij",
    )
    pair = np.asarray(
        _score_pass2_pairs_normalized_cc(
            jnp.asarray(shifted),
            jnp.asarray(score_weight),
            jnp.asarray(projections),
            jnp.asarray(half_weight),
            jnp.asarray(rotation_rows.reshape(1, -1)),
            jnp.asarray(translation_rows.reshape(1, -1)),
            jnp.ones((1, rotation_rows.size), dtype=bool),
        )
    ).reshape(bucket.shape)

    np.testing.assert_array_equal(cached, bucket)
    np.testing.assert_array_equal(np.argsort(pair, axis=None), np.argsort(bucket, axis=None))


@pytest.mark.parametrize(
    "scorer,args",
    [
        (
            _score_pass2_bucket_normalized_cc,
            lambda shifted, weight, proj, half, mask: (shifted, weight, proj, half, mask),
        ),
        (
            _score_pass2_bucket_normalized_cc_single_cached,
            lambda shifted, weight, proj, half, mask: (shifted[0], weight[0], proj[0], half, mask[0]),
        ),
    ],
)
def test_normalized_cc_cross_lowers_to_reduce_not_complex_dot(scorer, args):
    shifted, score_weight, projections, half_weight, mask = map(jnp.asarray, _inputs())
    jaxpr = str(jax.make_jaxpr(scorer)(*args(shifted, score_weight, projections, half_weight, mask)))

    # The real projection-norm contraction remains a dot_general; the cross
    # contraction must instead contain the explicit float32 reduce_sum.
    assert "reduce_sum" in jaxpr
    assert jaxpr.count("dot_general") == 1


def test_normalized_cc_pair_cross_lowers_to_reduce_not_complex_dot():
    shifted, score_weight, projections, half_weight, _mask = map(jnp.asarray, _inputs())
    rotation_rows, translation_rows = np.meshgrid(
        np.arange(projections.shape[1], dtype=np.int32),
        np.arange(shifted.shape[1], dtype=np.int32),
        indexing="ij",
    )
    pair_mask = jnp.ones((1, rotation_rows.size), dtype=bool)
    jaxpr = str(
        jax.make_jaxpr(_score_pass2_pairs_normalized_cc)(
            shifted,
            score_weight,
            projections,
            half_weight,
            jnp.asarray(rotation_rows.reshape(1, -1)),
            jnp.asarray(translation_rows.reshape(1, -1)),
            pair_mask,
        )
    )

    assert "reduce_sum" in jaxpr
    assert jaxpr.count("dot_general") == 1
