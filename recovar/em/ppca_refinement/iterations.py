"""EM iteration drivers for ``recovar ppca-refine`` (Milestone 3+).

Currently only the M3 fixed-pose driver is wired up. M5 will add
``run_pose_marginal_ppca_refine`` for the dense / sparse pose-marginalized
path that uses the augmented [μ, W] PCG.

Design note: M3 deliberately keeps μ fixed and only updates W. This is
the boring + reliable mode and gives an early CLI parity sanity test
against the legacy ``recovar.ppca.ppca.EM(...)``. Mean refinement and
pose marginalization are stacked on in subsequent milestones via the
augmented PCG (already available from M2).
"""

from __future__ import annotations

from typing import Any

from recovar.ppca import EM as legacy_em

from .state import FixedPosePPCAState

__all__ = ["run_fixed_pose_ppca_refine"]


def run_fixed_pose_ppca_refine(
    experiment_dataset,
    state: FixedPosePPCAState,
    *,
    EM_iter: int = 20,
    pcg_maxiter: int = 20,
    update_noise: bool = False,
    noise_update_ema: float = 0.5,
    return_iteration_data: bool = False,
    return_posterior_info: bool = False,
    extra_em_kwargs: dict[str, Any] | None = None,
):
    """M3 fixed-pose driver — thin wrapper around legacy
    ``recovar.ppca.ppca.EM(...)``.

    Holds μ fixed (legacy EM's contract) and updates only W. Multi-mask is
    supported via ``state.masks`` + ``state.pc_mask_assignment``. Contrast
    is supported via ``state.contrast_mode`` ∈ {"none", "profile",
    "marginalize"}. The ``state.pc_prior_config`` is forwarded to the EM
    so its snapshot lands in ``iteration_data[0]`` when requested.

    Returns the same tuple ``recovar.ppca.ppca.EM`` returns. Caller can
    set ``return_iteration_data`` / ``return_posterior_info`` to extend
    the tuple.
    """
    extra = dict(extra_em_kwargs or {})
    return legacy_em(
        experiment_dataset,
        state.mean_estimate,
        state.W_initial,
        state.W_prior,
        EM_iter=EM_iter,
        pcg_maxiter=pcg_maxiter,
        contrast_mode=state.contrast_mode,
        contrast_grid=state.contrast_grid,
        contrast_weights=state.contrast_weights,
        contrast_mean=state.contrast_mean,
        contrast_variance=state.contrast_variance,
        volume_mask=state.volume_mask,
        dilated_volume_mask=state.dilated_volume_mask,
        masks=state.masks,
        pc_mask_assignment=state.pc_mask_assignment,
        update_noise=update_noise,
        noise_update_ema=noise_update_ema,
        return_iteration_data=return_iteration_data,
        return_posterior_info=return_posterior_info,
        pc_prior_config=state.pc_prior_config,
        **extra,
    )
