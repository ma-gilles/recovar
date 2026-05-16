"""Parity test for the R-chunked variant of score_local_pose_ppca_bucket.

The chunked path tiles the R (rotation) axis to cut peak working memory so
``image_batch_size > 1`` becomes feasible on HP6 top-p neighborhoods. The
one-shot kernel is the reference; the chunked path must match on the fields
it computes exactly (``logZ``, ``pmax``, ``best_*``, ``top_*``, ``best_log_score_per_image``).
``rotation_posterior_sums`` and ``n_significant_per_image`` are documented
to be approximate / zeroed in the chunked path and are not asserted here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.ppca_refinement.local_dataset import (
    _score_local_pose_ppca_bucket_rotation_chunked,
    score_local_pose_ppca_bucket,
)


def _synth_inputs(B: int, T: int, R: int, P: int, F: int, *, seed: int):
    rng = np.random.default_rng(seed)
    Y1 = (
        rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))
    ).astype(np.complex64)
    proj_aug = (
        rng.standard_normal((B, R, P, F)) + 1j * rng.standard_normal((B, R, P, F))
    ).astype(np.complex64)
    ctf2_over_noise = rng.uniform(0.01, 1.0, size=(B, F)).astype(np.float32)
    y_norm = rng.uniform(0.1, 5.0, size=(B,)).astype(np.float32)
    pose_log_prior = rng.standard_normal((B, R, T)).astype(np.float32) * 0.1
    return (
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
        jnp.asarray(pose_log_prior),
    )


@pytest.mark.parametrize(
    "B,T,R,P,F,chunk,top_k",
    [
        (2, 1, 32, 5, 17, 8, 1),
        (2, 1, 32, 5, 17, 8, 4),
        (3, 2, 64, 4, 21, 16, 4),
        (1, 1, 100, 3, 19, 32, 3),  # R not divisible by chunk → padding path
        (2, 1, 30, 5, 17, 7, 2),    # both non-divisible AND padding > 0
    ],
)
def test_chunked_matches_unchunked(B, T, R, P, F, chunk, top_k):
    Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior = _synth_inputs(B, T, R, P, F, seed=42)

    ref = score_local_pose_ppca_bucket(
        Y1,
        proj_aug,
        ctf2_over_noise,
        y_norm,
        pose_log_prior,
        top_pose_count=top_k,
    )
    chunked = _score_local_pose_ppca_bucket_rotation_chunked(
        Y1,
        proj_aug,
        ctf2_over_noise,
        y_norm,
        pose_log_prior,
        significance_threshold=1e-3,
        top_pose_count=top_k,
        rotation_chunk_size=chunk,
    )

    np.testing.assert_allclose(
        np.asarray(chunked.logZ), np.asarray(ref.logZ), rtol=1e-4, atol=1e-3,
        err_msg="logZ should match across chunked aggregation",
    )
    np.testing.assert_allclose(
        np.asarray(chunked.best_log_score_per_image),
        np.asarray(ref.best_log_score_per_image),
        rtol=1e-5, atol=1e-4,
        err_msg="best_log_score_per_image is exact max — chunked must match",
    )
    np.testing.assert_array_equal(
        np.asarray(chunked.best_rotation_idx), np.asarray(ref.best_rotation_idx),
    )
    np.testing.assert_array_equal(
        np.asarray(chunked.best_translation_idx), np.asarray(ref.best_translation_idx),
    )
    np.testing.assert_allclose(
        np.asarray(chunked.pmax), np.asarray(ref.pmax), rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(chunked.max_posterior_per_image),
        np.asarray(ref.max_posterior_per_image),
        rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(chunked.top_log_score_per_image),
        np.asarray(ref.top_log_score_per_image),
        rtol=1e-5, atol=1e-4,
        err_msg="top-K log scores are the largest K values — chunked aggregation must match",
    )
    np.testing.assert_array_equal(
        np.asarray(chunked.top_rotation_idx), np.asarray(ref.top_rotation_idx),
    )
    np.testing.assert_array_equal(
        np.asarray(chunked.top_translation_idx), np.asarray(ref.top_translation_idx),
    )
    np.testing.assert_allclose(
        np.asarray(chunked.top_posterior_per_image),
        np.asarray(ref.top_posterior_per_image),
        rtol=1e-4, atol=1e-4,
    )


def test_chunked_skips_when_chunk_size_exceeds_R():
    """When chunk >= R, the wrapper should delegate to the one-shot kernel."""

    Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior = _synth_inputs(
        2, 1, 16, 5, 17, seed=7,
    )
    ref = score_local_pose_ppca_bucket(
        Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior, top_pose_count=2,
    )
    delegated = _score_local_pose_ppca_bucket_rotation_chunked(
        Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior,
        significance_threshold=1e-3, top_pose_count=2, rotation_chunk_size=64,
    )
    # rotation_posterior_sums is zeroed in chunked path; on the delegated case
    # it should match the reference (one-shot kernel).
    np.testing.assert_allclose(
        np.asarray(delegated.rotation_posterior_sums),
        np.asarray(ref.rotation_posterior_sums),
        rtol=1e-5,
    )
