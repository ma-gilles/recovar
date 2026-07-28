from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.output import metrics as output_metrics
from recovar.ppca import ppca

__all__ = ["compute_eigenvector_metrics", "compute_mean_relative_error"]


def compute_eigenvector_metrics(W_half, gt_data, volume_shape, use_est_rank = True) -> dict:

    U_est,s_est, _ = ppca._orthonormalize_W_to_basis(jnp.asarray(W_half), volume_shape)
    U_est = ftu.get_dft3(U_est).reshape(U_est.shape[0], -1).T

    U_gt = jnp.asarray(gt_data.get_u())
    s_gt = jnp.asarray(gt_data.get_s())

    if use_est_rank:
        U_gt = U_gt[:, :U_est.shape[1]]
        s_gt = s_gt[:U_est.shape[1]]

    _, rel_var, _ = output_metrics.get_all_variance_scores(U_est, U_gt, s_gt)
    sine_angles = output_metrics.subspace_angles(U_est, U_gt, max_rank=min(U_est.shape[1], U_gt.shape[1]))
    cosine_similarity = np.sqrt(np.maximum(1.0 - sine_angles**2, 0.0))

    fro_diff = output_metrics.fro_norm_diff_low_rank(
        jnp.asarray(U_est), jnp.asarray(s_est), jnp.asarray(U_gt), jnp.asarray(s_gt)
    ).real
    fro_gt = jnp.sqrt(jnp.sum(jnp.asarray(s_gt) ** 2))

    return {
        "gt_relative_variance": float(rel_var[-1]),
        "gt_relative_variance_curve": np.asarray(rel_var, dtype=np.float32).tolist(),
        "gt_cosine_similarity": float(np.mean(cosine_similarity)),
        "gt_fro_relative_error": float(fro_diff / fro_gt),
    }


