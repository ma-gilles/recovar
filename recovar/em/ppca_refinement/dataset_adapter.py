"""Production block_provider + backprojector adapters (Milestone 10).

This module sketches the production wiring between a CryoEMDataset and
the M5 / M7 driver callbacks. The full implementation is deferred to
follow-up work because it requires a structural change to the dense
engine: the current
:func:`recovar.em.ppca_refinement.dense_engine.dense_pose_ppca_E_step_blocked`
emits image-level aggregates (``alpha_aug_acc`` summed over (R, T)) which
are too aggregated for proper per-rotation backprojection. The fused
production engine needs to interleave pass-2 score normalization with
``recovar.em.dense_single_volume.helpers.backprojection.accumulate_adjoint_pair``
calls per rotation in the rotation block.

The algorithm (documented here so a follow-up agent can implement it):

  1. For each image batch B in the halfset:
     a. Pre-shift CTF-weighted whitened images: Y1 [B, T, F].
     b. For each rotation block R:
         - Project mu, W onto rotations: proj_aug [R, P, F].
         - Compute K_aug [B, R, P, P].
         - For each translation block T':
             * Compute D = einsum(conj(Y1), proj_aug) [B, T', R, P].
             * Compute scores via M1 per (B, T', R).
             * Pass 1: accumulate logZ via logsumexp over (T', R).
         - After Pass 1 over all T': we have logZ[B].
         - Pass 2 (re-iterate over T'):
             * Recompute scores + moments (alpha [B, T', R, P], G [B, T', R, tri(P)]).
             * gamma = exp(score - logZ[:, None, None]).
             * For each rotation r in the block:
                 - Z_rp = sum_t gamma_brt * alpha_brt,p * Y1[b, t]    [B, P, F]
                 - rhs_p[half_vol] += A_r^* Z_rp  via accumulate_adjoint_pair
                 - For each (p, q) in upper triangle of (P, P):
                     wsum_pq[b, r] = sum_t gamma_brt * G_aug_tri[b, t, r, idx_pq]
                     lhs_tri[half_vol, idx_pq] += A_r^* (wsum_pq[b, r] * ctf2_over_noise)

The engine and backprojector are FUSED in this design — there is no
intermediate "image_stats" tensor that gets handed off. This is the
right architecture; the test-friendly engine in
``dense_pose_ppca_E_step_blocked`` (which DOES emit image-level
aggregates) remains for unit tests that don't need real backprojection.

Until the fused engine lands, the production CLI accepts ``--pose-mode
fixed`` (M3) which uses the legacy ``recovar.ppca.ppca.EM`` path
end-to-end. Pose-marginal modes (``dense``, ``local``) currently
require user-supplied callbacks via the Python API; the synthetic
testing path in
``tests/unit/ppca_refinement/test_pose_marginal_driver.py`` and
``test_local_pose_driver.py`` exercises that API.

This module exposes a single thin function for now —
:func:`make_simple_block_provider_for_test` — to support an end-to-end
Ribosembly integration test that runs the full M5 driver with synthetic
projection helpers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from recovar.em.ppca_refinement.iterations import PoseBlock

__all__ = ["make_simple_block_provider_for_test"]


def make_simple_block_provider_for_test(
    cryo,
    *,
    image_indices,
    halfset_idx_per_image,
    n_rotations: int = 4,
    n_translations: int = 2,
    image_batch_size: int = 16,
    seed: int = 0,
):
    """Build a synthetic block_provider that yields :class:`PoseBlock`
    objects suitable for the M5 dense driver.

    NOT a production wiring — this is a test-only helper that synthesizes
    proj_aug / Y1 / ctf2 / y_norm directly from the dataset's images
    without the actual projection / shift machinery. It exists only so
    the M5 driver can be exercised end-to-end on the Ribosembly fixture
    pending the fused production engine.

    Parameters
    ----------
    cryo:
        :class:`CryoEMDataset`-like object with ``image_shape``,
        ``volume_shape``, and ``get_dataset_subset_generator``.
    image_indices:
        ``[N]`` int — global image indices to score.
    halfset_idx_per_image:
        ``[N]`` int in {0, 1} — halfset assignment.
    """
    rng = np.random.default_rng(seed)
    image_shape = cryo.image_shape
    F = int(np.prod(image_shape))

    def provider(theta_score, iteration):
        # theta_score is (mu_score, W_score) in real-space (D,D,D)/(q,D,D,D).
        mu_score, W_score = theta_score
        q = W_score.shape[0]
        P = q + 1

        blocks = []
        # Walk image_indices in batches.
        for start in range(0, len(image_indices), image_batch_size):
            batch_idx = image_indices[start : start + image_batch_size]
            B = len(batch_idx)
            T = n_translations
            R = n_rotations
            # Synthetic Y1 / proj_aug / ctf2 / y_norm. The dense engine
            # math is unit-tested separately; this provider just feeds it
            # well-shaped tensors so the driver loop runs.
            Y1 = (rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))).astype(np.complex64) * 1e-2
            proj_aug = (rng.standard_normal((R, P, F)) + 1j * rng.standard_normal((R, P, F))).astype(
                np.complex64
            ) * 1e-2
            ctf2_over_noise = rng.uniform(0.5, 1.0, size=(B, F)).astype(np.float32)
            y_norm = rng.uniform(0.5, 2.0, size=(B,)).astype(np.float32)
            half_idx = int(halfset_idx_per_image[batch_idx[0]])
            blocks.append(
                PoseBlock(
                    Y1=jnp.asarray(Y1),
                    proj_aug=jnp.asarray(proj_aug),
                    ctf2_over_noise=jnp.asarray(ctf2_over_noise),
                    y_norm=jnp.asarray(y_norm),
                    pose_log_prior=None,
                    image_indices=jnp.asarray(batch_idx, dtype=jnp.int32),
                    halfset_idx=half_idx,
                    rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (R, 3, 3)),
                    translations=jnp.zeros((T, 2), dtype=jnp.float32),
                )
            )
        return blocks

    return provider
