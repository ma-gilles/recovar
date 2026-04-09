"""Tests for the posterior helper at edge `q` values.

Existing tests use `q=2` to `q=4`. This file extends to:
  - `q=1` (single PC) — catches degenerate-shape bugs where a (q,)
    or (q, q) tensor would silently broadcast wrong.
  - `q=6` (larger q) — catches potential numerical stability issues
    with bigger H matrices (q×q Cholesky).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.posterior import (
    make_half_image_weights,
    score_from_half_image_projections,
)
from recovar.em.ppca_abinitio.types import PosteriorStats

pytestmark = pytest.mark.unit


IMAGE_SHAPE = (4, 4)
N_FULL = 16
N_HALF = 4 * 3  # H * (W//2 + 1)


def _real_derived_full(rng):
    real = rng.standard_normal(IMAGE_SHAPE).astype(np.float64)
    return jnp.asarray(ftu.get_dft2(jnp.asarray(real)).reshape(-1), dtype=jnp.complex128)


def _make_inputs(rng, n_rot, n_pc, n_img):
    mean = jnp.stack([_real_derived_full(rng) for _ in range(n_rot)])
    u_rows = jnp.stack([jnp.stack([_real_derived_full(rng) for _ in range(n_pc)]) for _ in range(n_rot)])
    u_rows = 0.1 * u_rows
    s = jnp.asarray(0.5 + rng.uniform(size=n_pc), dtype=jnp.float64)
    batch = jnp.stack([_real_derived_full(rng) for _ in range(n_img)])
    return mean, u_rows, s, batch


def _call_kernel(mean_proj_full, u_proj_full, s, batch_full, n_trans=1):
    n_img = batch_full.shape[0]
    n_rot = mean_proj_full.shape[0]
    q = u_proj_full.shape[1]
    weights_half = make_half_image_weights(IMAGE_SHAPE)

    mean_proj_half = ftu.full_image_to_half_image(jnp.asarray(mean_proj_full), IMAGE_SHAPE)
    u_proj_half = ftu.full_image_to_half_image(
        jnp.asarray(u_proj_full).reshape(n_rot * q, N_FULL), IMAGE_SHAPE
    ).reshape(n_rot, q, N_HALF)

    batch_half = ftu.full_image_to_half_image(jnp.asarray(batch_full), IMAGE_SHAPE)
    shifted_half = jnp.broadcast_to(batch_half[:, None, :], (n_img, n_trans, N_HALF))
    ctf2_over_nv_half = jnp.ones((n_img, N_HALF), dtype=jnp.float64)

    return score_from_half_image_projections(
        mean_proj_half=mean_proj_half.astype(jnp.complex128),
        u_proj_half=u_proj_half.astype(jnp.complex128),
        s=jnp.asarray(s, dtype=jnp.float64),
        shifted_half=shifted_half.astype(jnp.complex128),
        ctf2_over_nv_half=ctf2_over_nv_half,
        weights_half=weights_half,
    )


# ---------------------------------------------------------------------------
# q = 1 edge case
# ---------------------------------------------------------------------------


def test_kernel_q_equals_1_basic_shapes():
    rng = np.random.default_rng(0)
    n_rot, n_img = 2, 3
    mean, u, s, batch = _make_inputs(rng, n_rot, 1, n_img)
    stats = _call_kernel(mean, u, s, batch)

    assert isinstance(stats, PosteriorStats)
    assert stats.log_scores.shape == (n_img, n_rot, 1)
    assert stats.post_mean.shape == (n_img, n_rot, 1, 1)
    assert stats.post_Hinv.shape == (n_img, n_rot, 1, 1)
    assert jnp.all(jnp.isfinite(stats.log_scores))
    assert jnp.all(jnp.isfinite(stats.post_mean))
    assert jnp.all(jnp.isfinite(stats.post_Hinv))


def test_kernel_q_equals_1_post_Hinv_is_positive_scalar():
    """For q=1, H is a 1×1 matrix and Hinv = 1/H. Should be > 0."""
    rng = np.random.default_rng(1)
    mean, u, s, batch = _make_inputs(rng, n_rot=2, n_pc=1, n_img=2)
    stats = _call_kernel(mean, u, s, batch)
    Hinv = np.asarray(stats.post_Hinv)
    assert np.all(Hinv > 0), f"q=1 Hinv contains non-positive values: {Hinv}"


def test_kernel_q_equals_1_log_resp_normalizes():
    rng = np.random.default_rng(2)
    mean, u, s, batch = _make_inputs(rng, n_rot=4, n_pc=1, n_img=3)
    stats = _call_kernel(mean, u, s, batch)
    sums = np.exp(np.asarray(stats.log_resp).reshape(3, -1)).sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# q = 6 (larger q) edge case
# ---------------------------------------------------------------------------


def test_kernel_q_equals_6_basic_shapes_and_finiteness():
    rng = np.random.default_rng(3)
    n_rot, n_img = 3, 4
    mean, u, s, batch = _make_inputs(rng, n_rot, 6, n_img)
    stats = _call_kernel(mean, u, s, batch)

    assert stats.log_scores.shape == (n_img, n_rot, 1)
    assert stats.post_mean.shape == (n_img, n_rot, 1, 6)
    assert stats.post_Hinv.shape == (n_img, n_rot, 6, 6)
    assert jnp.all(jnp.isfinite(stats.log_scores))
    assert jnp.all(jnp.isfinite(stats.post_mean))
    assert jnp.all(jnp.isfinite(stats.post_Hinv))


def test_kernel_q_equals_6_post_Hinv_is_positive_definite():
    """The 6×6 H_inv must be PD at every (i, r). Catches potential
    Cholesky failures from larger q."""
    rng = np.random.default_rng(4)
    mean, u, s, batch = _make_inputs(rng, n_rot=2, n_pc=6, n_img=2)
    stats = _call_kernel(mean, u, s, batch)
    Hinv = np.asarray(stats.post_Hinv)
    for i in range(2):
        for r in range(2):
            eigs = np.linalg.eigvalsh(Hinv[i, r])
            assert eigs.min() > 0, f"q=6 Hinv at (i={i}, r={r}) not PD: eigs={eigs}"


def test_kernel_q_equals_6_log_resp_normalizes():
    rng = np.random.default_rng(5)
    mean, u, s, batch = _make_inputs(rng, n_rot=3, n_pc=6, n_img=2)
    stats = _call_kernel(mean, u, s, batch)
    sums = np.exp(np.asarray(stats.log_resp).reshape(2, -1)).sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-12)
