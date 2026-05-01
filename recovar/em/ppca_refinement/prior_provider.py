"""Prior providers for the pose-marginal PPCA driver (Milestone 10).

The user's resolution of the open question "Mean prior source for the
augmented PCG":

    "yes the same prior as used for k-class/1 class I suppose"

So the mean prior is :func:`recovar.reconstruction.regularization.compute_relion_prior`
applied to the current half-volumes (the same RELION-style spectral
prior used by k-class and homogeneous refinement). This module exposes a
thin wrapper that fits the M5 driver's ``prior_recompute_fn`` Protocol.

The PPCA loading prior ``W_prior`` remains a separate object computed by
``recovar.ppca.prior_estimation.estimate_hybrid_shell_prior_from_data``
(CLAUDE.md §7 non-negotiables: ``W_prior`` is NOT the same as the mean
prior). This module covers the mean prior only.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from recovar.reconstruction import regularization

__all__ = ["compute_mean_prior_relion", "make_mean_prior_provider"]


def compute_mean_prior_relion(
    halfset_datasets,
    cov_noise,
    mu_h0,
    mu_h1,
    batch_size: int,
    *,
    estimate_merged_SNR: bool = False,
    noise_level: Any | None = None,
    tau2_fudge: float = 1.0,
):
    """Compute the RELION-style mean prior used by k-class refinement.

    This is :func:`recovar.reconstruction.regularization.compute_relion_prior`
    — exposed at this top-level for the pose-marginal driver to import
    without reaching into ``recovar.reconstruction``.

    Parameters
    ----------
    halfset_datasets:
        Tuple ``(cryo_h0, cryo_h1)`` of half-set CryoEMDataset objects.
    cov_noise:
        Scalar noise covariance.
    mu_h0, mu_h1:
        Half-volume mean estimates in flattened Fourier coefficients
        (the format ``compute_relion_prior`` expects).
    batch_size:
        GPU batch size for the noise-prior helper.

    Returns
    -------
    prior, fsc, prior_avg:
        Per-shell spectral prior, FSC curve between the halves, and the
        averaged prior. ``prior`` is the value to thread into the
        augmented M-step's ``mean_prior`` argument (after broadcasting to
        per-half-volume voxel layout — see :func:`make_mean_prior_provider`).
    """
    return regularization.compute_relion_prior(
        halfset_datasets,
        cov_noise,
        mu_h0,
        mu_h1,
        batch_size,
        estimate_merged_SNR=estimate_merged_SNR,
        noise_level=noise_level,
        tau2_fudge=tau2_fudge,
    )


def make_mean_prior_provider(
    halfset_datasets,
    cov_noise,
    batch_size: int,
    *,
    tau2_fudge: float = 1.0,
):
    """Build a callable ``(state) -> mean_prior_per_voxel`` suitable as
    ``prior_recompute_fn`` for the M5 driver.

    The returned function reads the latest ``state.mu_half[0]`` and
    ``state.mu_half[1]``, computes the RELION prior via
    :func:`compute_mean_prior_relion`, and returns a per-half-voxel
    array compatible with ``solve_augmented_ppca_mstep``'s
    ``mean_prior`` parameter (variance, not precision; the M-step
    inverts internally).
    """

    def recompute(state):
        # state.mu_half is (mu_real_h0, mu_real_h1) of shape (D, D, D).
        # compute_relion_prior expects flattened Fourier means.
        import recovar.core.fourier_transform_utils as ftu

        mu_h0_real = np.asarray(state.mu_half[0])
        mu_h1_real = np.asarray(state.mu_half[1])
        vol_shape = mu_h0_real.shape
        mu_h0_full_f = np.asarray(ftu.get_dft3(jnp.asarray(mu_h0_real))).reshape(-1)
        mu_h1_full_f = np.asarray(ftu.get_dft3(jnp.asarray(mu_h1_real))).reshape(-1)

        prior, _fsc, _prior_avg = compute_mean_prior_relion(
            halfset_datasets,
            cov_noise,
            mu_h0_full_f,
            mu_h1_full_f,
            batch_size,
            tau2_fudge=tau2_fudge,
        )
        # ``prior`` is per-shell; broadcast to per-half-volume voxels.
        # The augmented M-step expects shape ``[half_vol]`` real
        # variance, matching ``W_prior[half_vol, q]`` slot 0.
        half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
        half_vol = int(np.prod(half_vs))
        # Map per-shell prior to per-voxel half-volume.
        from recovar.reconstruction.regularization import (
            broadcast_shell_to_volume,
        )

        try:
            mean_prior_full = broadcast_shell_to_volume(np.asarray(prior), vol_shape)
        except Exception:
            # Fallback: scalar mean of the shell prior, broadcast.
            mean_prior_full = np.full(int(np.prod(vol_shape)), float(np.mean(prior)), dtype=np.float32)
        # Reduce to half-volume.
        mean_prior_half = np.asarray(mean_prior_full).reshape(vol_shape)
        # The half-volume convention is the rfft-packed (D, D, D//2+1).
        mean_prior_half_packed = np.asarray(
            ftu.full_volume_to_half_volume(jnp.asarray(mean_prior_half), vol_shape)
        ).real.astype(np.float32)
        if mean_prior_half_packed.shape != (half_vol,):
            mean_prior_half_packed = mean_prior_half_packed.reshape(half_vol).astype(np.float32)
        return jnp.asarray(mean_prior_half_packed)

    return recompute
