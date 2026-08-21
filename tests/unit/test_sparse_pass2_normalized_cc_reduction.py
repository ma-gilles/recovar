"""RELION firstiter-CC float32 reduction parity tests."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _score_pass2_bucket_normalized_cc,
    _score_pass2_bucket_relion_gpu_normalized_cc,
    _score_pass2_bucket_relion_gpu_normalized_cc_single_cached,
    _score_pass2_pairs_relion_gpu_normalized_cc,
)

pytestmark = pytest.mark.unit


def _inputs(n_pix=17):
    rng = np.random.default_rng(161552)
    batch, n_rot, n_trans = 1, 3, 4
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
        _score_pass2_bucket_relion_gpu_normalized_cc(
            jnp.asarray(shifted),
            jnp.asarray(score_weight),
            jnp.asarray(projections),
            jnp.asarray(half_weight),
            jnp.asarray(mask),
        )
    )[0]
    cached = np.asarray(
        _score_pass2_bucket_relion_gpu_normalized_cc_single_cached(
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
        _score_pass2_pairs_relion_gpu_normalized_cc(
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


def test_normalized_cc_restores_relion_pixel_order_before_256_lane_tree():
    shifted, score_weight, projections, half_weight, mask = _inputs(n_pix=521)
    baseline = np.asarray(
        _score_pass2_bucket_relion_gpu_normalized_cc(
            jnp.asarray(shifted),
            jnp.asarray(score_weight),
            jnp.asarray(projections),
            jnp.asarray(half_weight),
            jnp.asarray(mask),
        )
    )

    rng = np.random.default_rng(8173)
    compact_gather = rng.permutation(shifted.shape[-1])
    full_to_compact = np.argsort(compact_gather).astype(np.int32)
    restored = np.asarray(
        _score_pass2_bucket_relion_gpu_normalized_cc(
            jnp.asarray(shifted[..., compact_gather]),
            jnp.asarray(score_weight[..., compact_gather]),
            jnp.asarray(projections[..., compact_gather]),
            jnp.asarray(half_weight[compact_gather]),
            jnp.asarray(mask),
            jnp.asarray(full_to_compact),
        )
    )

    np.testing.assert_array_equal(restored, baseline)


@pytest.mark.parametrize(
    "scorer,args",
    [
        (
            _score_pass2_bucket_relion_gpu_normalized_cc,
            lambda shifted, weight, proj, half, mask: (shifted, weight, proj, half, mask),
        ),
        (
            _score_pass2_bucket_relion_gpu_normalized_cc_single_cached,
            lambda shifted, weight, proj, half, mask: (shifted[0], weight[0], proj[0], half, mask[0]),
        ),
    ],
)
def test_normalized_cc_lowers_to_relion_lane_scan_not_generic_reduction(scorer, args):
    shifted, score_weight, projections, half_weight, mask = map(jnp.asarray, _inputs())
    jaxpr = str(jax.make_jaxpr(scorer)(*args(shifted, score_weight, projections, half_weight, mask)))

    assert "scan" in jaxpr
    assert "reduce_sum" not in jaxpr
    assert "dot_general" not in jaxpr


def test_normalized_cc_pair_lowers_to_relion_lane_scan_not_generic_reduction():
    shifted, score_weight, projections, half_weight, _mask = map(jnp.asarray, _inputs())
    rotation_rows, translation_rows = np.meshgrid(
        np.arange(projections.shape[1], dtype=np.int32),
        np.arange(shifted.shape[1], dtype=np.int32),
        indexing="ij",
    )
    pair_mask = jnp.ones((1, rotation_rows.size), dtype=bool)
    jaxpr = str(
        jax.make_jaxpr(_score_pass2_pairs_relion_gpu_normalized_cc)(
            shifted,
            score_weight,
            projections,
            half_weight,
            jnp.asarray(rotation_rows.reshape(1, -1)),
            jnp.asarray(translation_rows.reshape(1, -1)),
            pair_mask,
        )
    )

    assert "scan" in jaxpr
    assert "reduce_sum" not in jaxpr
    assert "dot_general" not in jaxpr


def test_historical_normalized_cc_scorer_remains_the_generic_default():
    shifted, score_weight, projections, half_weight, mask = map(jnp.asarray, _inputs())
    jaxpr = str(
        jax.make_jaxpr(_score_pass2_bucket_normalized_cc)(
            shifted,
            score_weight,
            projections,
            half_weight,
            mask,
        )
    )

    assert "reduce_sum" in jaxpr
    assert jaxpr.count("dot_general") == 1
    assert "scan" not in jaxpr
