"""Phase 11 (M10) tests: halfset combiners + mean prior provider."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from recovar.em.ppca_refinement.halfset_combine import (  # noqa: E402
    low_resol_join_halfset_combine,
    make_halfset_combiner,
    mean_halfset_combine,
)

pytestmark = pytest.mark.unit


def test_mean_halfset_combine_3d():
    rng = np.random.default_rng(0)
    h0 = rng.standard_normal((8, 8, 8)).astype(np.float32)
    h1 = rng.standard_normal((8, 8, 8)).astype(np.float32)
    out = mean_halfset_combine(jnp.asarray(h0), jnp.asarray(h1), kind="mu")
    np.testing.assert_allclose(np.asarray(out), 0.5 * (h0 + h1), rtol=1e-6)


def test_mean_halfset_combine_4d_per_pc():
    rng = np.random.default_rng(0)
    h0 = rng.standard_normal((3, 6, 6, 6)).astype(np.float32)
    h1 = rng.standard_normal((3, 6, 6, 6)).astype(np.float32)
    out = mean_halfset_combine(jnp.asarray(h0), jnp.asarray(h1), kind="W")
    np.testing.assert_allclose(np.asarray(out), 0.5 * (h0 + h1), rtol=1e-6)


def test_low_resol_join_halfset_combine_runs_3d():
    rng = np.random.default_rng(0)
    h0 = rng.standard_normal((8, 8, 8)).astype(np.float32)
    h1 = rng.standard_normal((8, 8, 8)).astype(np.float32)
    out = low_resol_join_halfset_combine(
        h0,
        h1,
        voxel_size=4.25,
        low_resol_join_halves_angstrom=40.0,
    )
    out_np = np.asarray(out)
    assert out_np.shape == (8, 8, 8)
    # Currently equivalent to simple mean (per docstring caveat).
    np.testing.assert_allclose(out_np, 0.5 * (h0 + h1), rtol=1e-4, atol=1e-5)


def test_low_resol_join_halfset_combine_runs_4d_per_pc():
    rng = np.random.default_rng(0)
    h0 = rng.standard_normal((2, 8, 8, 8)).astype(np.float32)
    h1 = rng.standard_normal((2, 8, 8, 8)).astype(np.float32)
    out = low_resol_join_halfset_combine(h0, h1, voxel_size=4.25)
    assert out.shape == (2, 8, 8, 8)


def test_make_halfset_combiner_dispatches_method():
    rng = np.random.default_rng(0)
    h0 = jnp.asarray(rng.standard_normal((6, 6, 6)).astype(np.float32))
    h1 = jnp.asarray(rng.standard_normal((6, 6, 6)).astype(np.float32))

    mean_fn = make_halfset_combiner(method="mean")
    join_fn = make_halfset_combiner(method="low_resol_join", voxel_size=4.25)

    a = mean_fn(h0, h1, "mu")
    b = join_fn(h0, h1, "mu")
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-5)

    with pytest.raises(ValueError, match="unknown halfset combine"):
        make_halfset_combiner(method="bogus")
    with pytest.raises(ValueError, match="voxel_size is required"):
        make_halfset_combiner(method="low_resol_join")


def test_low_resol_join_rejects_shape_mismatch():
    h0 = np.zeros((4, 4, 4), dtype=np.float32)
    h1 = np.zeros((5, 5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        low_resol_join_halfset_combine(h0, h1, voxel_size=4.25)


def test_low_resol_join_rejects_unsupported_ndim():
    h0 = np.zeros((4, 4), dtype=np.float32)
    h1 = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 3D or 4D"):
        low_resol_join_halfset_combine(h0, h1, voxel_size=4.25)
