"""Phase 8 (M7) tests: local-pose driver via the sparse engine.

Verifies that ``run_pose_marginal_ppca_refine`` correctly orchestrates the
sparse engine (M6) end-to-end:

  * sparse_block_provider yields :class:`SparsePoseBlock` per iteration
  * sparse engine runs and produces per-image diagnostics
  * sparse_backprojector → augmented M-step → halfset combine
  * pose_estimates (hard poses) are updated from posterior best per image

Production wiring (LocalHypothesisLayout → SparseHypothesisLayout, real
dataset, real backprojection) lands at M10. Here we exercise the
orchestration against synthetic blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import recovar.core.fourier_transform_utils as ftu  # noqa: E402
from recovar.em.ppca_refinement.iterations import (  # noqa: E402
    IterationOpts,
    SparsePoseBlock,
    run_pose_marginal_ppca_refine,
)
from recovar.em.ppca_refinement.sparse_engine import (  # noqa: E402
    SparseHypothesisLayout,
)
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState  # noqa: E402
from recovar.ppca import AugmentedPPCAStats  # noqa: E402
from recovar.ppca.ppca import _tri_size  # noqa: E402

pytestmark = pytest.mark.unit


def _make_synthetic_state(rng, q, grid):
    vol_shape = (grid, grid, grid)
    half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
    half_vol = int(np.prod(half_vs))

    def zeros_vol():
        return jnp.zeros(vol_shape, dtype=jnp.float32)

    def zeros_W():
        return jnp.zeros((q,) + vol_shape, dtype=jnp.float32)

    return (
        PoseMarginalPPCAEMState(
            mu_half=(zeros_vol(), zeros_vol()),
            W_half=(zeros_W(), zeros_W()),
            mu_score=zeros_vol(),
            W_score=zeros_W(),
            W_prior=jnp.full((half_vol, q), 1e3, dtype=jnp.float32),
            mean_prior=jnp.full((half_vol,), 1e3, dtype=jnp.float32),
            z_prior_precision_diag=jnp.ones((q,), dtype=jnp.float32),
            noise_variance=jnp.ones((half_vol,), dtype=jnp.float32),
            contrast_params=None,
            masks=None,
            pose_estimates={},
            pose_priors=None,
            refinement_schedule_state=None,
            hyperparams=None,
        ),
        vol_shape,
        half_vs,
        half_vol,
    )


def _make_sparse_block(rng, *, n_images, n_hyp_per_image, F, q, halfset_idx):
    P = q + 1
    Nh = n_images * n_hyp_per_image
    image_id = np.repeat(np.arange(n_images, dtype=np.int32), n_hyp_per_image)
    Y1 = (rng.standard_normal((Nh, F)) + 1j * rng.standard_normal((Nh, F))).astype(np.complex64)
    proj_aug = (rng.standard_normal((Nh, P, F)) + 1j * rng.standard_normal((Nh, P, F))).astype(np.complex64)
    ctf2 = rng.uniform(0.1, 1.0, size=(Nh, F)).astype(np.float32)
    y_norm = rng.uniform(0.5, 2.0, size=(Nh,)).astype(np.float32)
    rotations_per_hyp = np.broadcast_to(np.eye(3, dtype=np.float32), (Nh, 3, 3)).copy()
    # Add a small per-hypothesis perturbation so different hypotheses
    # produce different best-pose records.
    perturb = rng.normal(scale=0.01, size=(Nh, 3, 3)).astype(np.float32)
    rotations_per_hyp = rotations_per_hyp + perturb
    translations_per_hyp = rng.normal(scale=0.5, size=(Nh, 2)).astype(np.float32)
    image_indices = np.arange(n_images, dtype=np.int32)
    layout = SparseHypothesisLayout(
        Y1=jnp.asarray(Y1),
        proj_aug=jnp.asarray(proj_aug),
        ctf2_over_noise=jnp.asarray(ctf2),
        y_norm=jnp.asarray(y_norm),
        pose_log_prior=None,
        image_id=jnp.asarray(image_id),
        n_images=n_images,
    )
    return SparsePoseBlock(
        layout=layout,
        halfset_idx=halfset_idx,
        rotations_per_hyp=jnp.asarray(rotations_per_hyp),
        translations_per_hyp=jnp.asarray(translations_per_hyp),
        image_indices=jnp.asarray(image_indices),
    )


def _make_dummy_sparse_backprojector(half_vol, q):
    P = q + 1

    def backproject(stats_blocks, halfset_idx):
        rng = np.random.default_rng(123 + halfset_idx)
        tri = _tri_size(P)
        lhs = np.zeros((half_vol, tri), dtype=np.float32)
        idx = 0
        for i in range(P):
            for j in range(i, P):
                lhs[:, idx] = (
                    (1.0 + rng.uniform(0, 0.1, size=half_vol).astype(np.float32))
                    if i == j
                    else rng.normal(0, 0.01, size=half_vol).astype(np.float32)
                )
                idx += 1
        rhs = (
            rng.standard_normal((half_vol, P)).astype(np.float32)
            + 1j * rng.standard_normal((half_vol, P)).astype(np.float32)
        ).astype(np.complex64)
        n_hyp_total = sum(b.layout.image_id.shape[0] for b, _, _ in stats_blocks)
        return AugmentedPPCAStats(
            rhs=jnp.asarray(rhs),
            lhs_tri=jnp.asarray(lhs),
            n_images=n_hyp_total,
        )

    return backproject


def test_local_pose_driver_orchestrates_one_iter():
    rng = np.random.default_rng(0)
    state, vol_shape, half_vs, half_vol = _make_synthetic_state(rng, q=2, grid=10)
    F = 6
    blocks = [
        _make_sparse_block(rng, n_images=3, n_hyp_per_image=4, F=F, q=2, halfset_idx=0),
        _make_sparse_block(rng, n_images=3, n_hyp_per_image=4, F=F, q=2, halfset_idx=1),
    ]

    def sparse_block_provider(theta_score, pose_estimates, iteration):
        return blocks

    sparse_backprojector = _make_dummy_sparse_backprojector(half_vol, 2)
    final_state, log = run_pose_marginal_ppca_refine(
        state,
        sparse_block_provider=sparse_block_provider,
        sparse_backprojector=sparse_backprojector,
        mask=jnp.ones(vol_shape, dtype=jnp.float32),
        opts=IterationOpts(EM_iter=1, pcg_maxiter=5),
    )
    assert len(log) == 1
    # Hard poses updated to a non-empty dict (one entry per (halfset, image_id)).
    assert len(final_state.pose_estimates) > 0
    assert float(jnp.linalg.norm(final_state.mu_half[0])) > 0
    assert float(jnp.linalg.norm(final_state.mu_half[1])) > 0


def test_local_pose_driver_iterates_and_pose_estimates_track():
    rng = np.random.default_rng(0)
    state, vol_shape, half_vs, half_vol = _make_synthetic_state(rng, q=1, grid=8)
    F = 4
    # Two iterations with different blocks ⇒ different best poses.
    blocks_iter0 = [
        _make_sparse_block(rng, n_images=2, n_hyp_per_image=3, F=F, q=1, halfset_idx=0),
        _make_sparse_block(rng, n_images=2, n_hyp_per_image=3, F=F, q=1, halfset_idx=1),
    ]
    blocks_iter1 = [
        _make_sparse_block(rng, n_images=2, n_hyp_per_image=3, F=F, q=1, halfset_idx=0),
        _make_sparse_block(rng, n_images=2, n_hyp_per_image=3, F=F, q=1, halfset_idx=1),
    ]
    iter_blocks = [blocks_iter0, blocks_iter1]
    seen_pose_estimates: list = []

    def sparse_block_provider(theta_score, pose_estimates, iteration):
        seen_pose_estimates.append(dict(pose_estimates) if pose_estimates else {})
        return iter_blocks[iteration]

    sparse_backprojector = _make_dummy_sparse_backprojector(half_vol, 1)
    final_state, log = run_pose_marginal_ppca_refine(
        state,
        sparse_block_provider=sparse_block_provider,
        sparse_backprojector=sparse_backprojector,
        mask=jnp.ones(vol_shape, dtype=jnp.float32),
        opts=IterationOpts(EM_iter=2, pcg_maxiter=5),
    )
    assert len(log) == 2
    # iter 0 sees empty pose_estimates (initial state); iter 1 sees populated.
    assert seen_pose_estimates[0] == {}
    assert len(seen_pose_estimates[1]) > 0


def test_local_pose_driver_rejects_missing_callbacks():
    rng = np.random.default_rng(0)
    state, vol_shape, _, _ = _make_synthetic_state(rng, q=1, grid=8)

    # Neither provider given.
    with pytest.raises(ValueError, match="Provide exactly one"):
        run_pose_marginal_ppca_refine(
            state,
            mask=jnp.ones(vol_shape, dtype=jnp.float32),
            opts=IterationOpts(EM_iter=1),
        )

    # Sparse provider without sparse backprojector.
    def sparse_provider(theta_score, pose_estimates, iteration):
        return []

    with pytest.raises(ValueError, match="sparse_block_provider requires sparse_backprojector"):
        run_pose_marginal_ppca_refine(
            state,
            sparse_block_provider=sparse_provider,
            mask=jnp.ones(vol_shape, dtype=jnp.float32),
            opts=IterationOpts(EM_iter=1),
        )

    # Both providers given.
    def dense_provider(theta_score, iteration):
        return []

    with pytest.raises(ValueError, match="Provide exactly one"):
        run_pose_marginal_ppca_refine(
            state,
            block_provider=dense_provider,
            sparse_block_provider=sparse_provider,
            mask=jnp.ones(vol_shape, dtype=jnp.float32),
            opts=IterationOpts(EM_iter=1),
        )


def test_local_pose_cli_accepts_local_mode():
    from recovar.em.ppca_refinement.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "particles.star",
            "--out",
            "/tmp/out",
            "--init-mean",
            "consensus.mrc",
            "--zdim",
            "6",
            "--pose-mode",
            "local",
            "--engine",
            "sparse",
            "--contrast",
            "none",
            "--reuse-kclass-pose-schedule",
        ]
    )
    assert ns.pose_mode == "local"
    assert ns.engine == "sparse"
    assert ns.reuse_kclass_pose_schedule is True
