"""Tests for `iter_posterior_blocks` (the streaming variant of the
posterior helper).

Pin: the streaming iterator must yield blocks whose concatenation
equals the materialized `PosteriorStats` from
`score_and_posterior_moments_eqx`. The streaming variant exists to
support real workloads where the full `(n_img, n_rot, n_trans, q)`
tensor doesn't fit in memory; until this test was added, it was
implemented but never validated.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.posterior import (
    iter_posterior_blocks,
    score_and_posterior_moments_eqx,
)
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


VOLUME_SHAPE = (8, 8, 8)
IMAGE_SHAPE = (8, 8)


def _identity_ctf(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(batch, apply_image_mask=False):
    return batch


class _SyntheticConfig(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, ctf_params, *, half_image=False):
        full = _identity_ctf(ctf_params, self.image_shape, self.voxel_size)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, batch, apply_image_mask=False):
        return _identity_process(batch, apply_image_mask=apply_image_mask)


def _make_dataset_and_config(seed=0):
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=4,
        n_images_val=2,
        sigma_real=0.2,
        seed=seed,
    )
    cfg = _SyntheticConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    return ds, cfg


def test_iter_posterior_blocks_concatenation_equals_materialized():
    """The streaming iterator must produce identical results to the
    materialized helper, after concatenating across blocks."""
    ds, cfg = _make_dataset_and_config()

    materialized = score_and_posterior_moments_eqx(
        cfg,
        ds.mu_half_true,
        ds.U_half_true,
        ds.s_true,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
    )

    # Stream with several block sizes
    n_rot = ds.n_rot
    n_trans = ds.n_trans
    rot_block_size = max(1, n_rot // 4)  # 4 rotation blocks
    trans_block_size = max(1, n_trans // 2)  # 2 translation blocks

    blocks = list(
        iter_posterior_blocks(
            cfg,
            ds.mu_half_true,
            ds.U_half_true,
            ds.s_true,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
            rot_block_size=rot_block_size,
            trans_block_size=trans_block_size,
        )
    )

    # Reconstruct the full tensors from blocks
    n_img = ds.n_img
    q = int(ds.U_half_true.shape[0])

    log_scores_streamed = np.zeros((n_img, n_rot, n_trans), dtype=np.float64)
    post_mean_streamed = np.zeros((n_img, n_rot, n_trans, q), dtype=np.float64)
    post_Hinv_seen = np.zeros((n_img, n_rot, q, q), dtype=np.float64)
    post_Hinv_seen_count = np.zeros((n_img, n_rot), dtype=np.int32)

    for block in blocks:
        rs = block.rot_slice
        ts = block.trans_slice
        log_scores_streamed[:, rs, ts] = np.asarray(block.log_scores)
        post_mean_streamed[:, rs, ts, :] = np.asarray(block.post_mean)
        # post_Hinv is translation-independent — every trans-block writes
        # the same value, but we just record once.
        if post_Hinv_seen_count[0, rs.start] == 0:
            post_Hinv_seen[:, rs, :, :] = np.asarray(block.post_Hinv)
        post_Hinv_seen_count[:, rs] += 1

    np.testing.assert_allclose(
        log_scores_streamed,
        np.asarray(materialized.log_scores),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        post_mean_streamed,
        np.asarray(materialized.post_mean),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        post_Hinv_seen,
        np.asarray(materialized.post_Hinv),
        rtol=1e-10,
        atol=1e-12,
    )


def test_iter_posterior_blocks_handles_single_block():
    """rot_block_size = n_rot, trans_block_size = n_trans should
    yield exactly ONE block equal to the full tensor."""
    ds, cfg = _make_dataset_and_config()
    materialized = score_and_posterior_moments_eqx(
        cfg,
        ds.mu_half_true,
        ds.U_half_true,
        ds.s_true,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
    )
    blocks = list(
        iter_posterior_blocks(
            cfg,
            ds.mu_half_true,
            ds.U_half_true,
            ds.s_true,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
            rot_block_size=ds.n_rot,
            trans_block_size=ds.n_trans,
        )
    )
    assert len(blocks) == 1
    block = blocks[0]
    np.testing.assert_array_equal(
        np.asarray(block.log_scores),
        np.asarray(materialized.log_scores),
    )
    np.testing.assert_array_equal(
        np.asarray(block.post_mean),
        np.asarray(materialized.post_mean),
    )


def test_iter_posterior_blocks_handles_uneven_block_size():
    """rot_block_size that doesn't evenly divide n_rot should still
    cover all rotations exactly once."""
    ds, cfg = _make_dataset_and_config()
    n_rot = ds.n_rot
    rot_block_size = 7  # 72 / 7 = 10 blocks of 7 + 1 block of 2
    trans_block_size = 3

    blocks = list(
        iter_posterior_blocks(
            cfg,
            ds.mu_half_true,
            ds.U_half_true,
            ds.s_true,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
            rot_block_size=rot_block_size,
            trans_block_size=trans_block_size,
        )
    )

    # Verify rotation coverage
    seen_rot = set()
    for b in blocks:
        for r in range(b.rot_slice.start, b.rot_slice.stop):
            seen_rot.add(r)
    assert seen_rot == set(range(n_rot))

    # Verify translation coverage per rotation block
    n_trans = ds.n_trans
    seen_trans_per_rot_block = {}
    for b in blocks:
        key = (b.rot_slice.start, b.rot_slice.stop)
        if key not in seen_trans_per_rot_block:
            seen_trans_per_rot_block[key] = set()
        for t in range(b.trans_slice.start, b.trans_slice.stop):
            seen_trans_per_rot_block[key].add(t)
    for trans_set in seen_trans_per_rot_block.values():
        assert trans_set == set(range(n_trans))


def test_iter_posterior_blocks_block_shapes_are_consistent():
    ds, cfg = _make_dataset_and_config()
    blocks = list(
        iter_posterior_blocks(
            cfg,
            ds.mu_half_true,
            ds.U_half_true,
            ds.s_true,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
            rot_block_size=10,
            trans_block_size=2,
        )
    )
    n_img = ds.n_img
    q = int(ds.U_half_true.shape[0])
    for b in blocks:
        n_rot_block = b.rot_slice.stop - b.rot_slice.start
        n_trans_block = b.trans_slice.stop - b.trans_slice.start
        assert b.log_scores.shape == (n_img, n_rot_block, n_trans_block)
        assert b.post_mean.shape == (n_img, n_rot_block, n_trans_block, q)
        assert b.post_Hinv.shape == (n_img, n_rot_block, q, q)
