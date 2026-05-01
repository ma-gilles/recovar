"""EM iteration drivers for ``recovar ppca-refine``.

Two drivers:

* :func:`run_fixed_pose_ppca_refine` (M3) — fixed-pose, thin wrapper around
  legacy ``recovar.ppca.ppca.EM``.
* :func:`run_pose_marginal_ppca_refine` (M5+) — dense / sparse
  pose-marginalized augmented EM with halfset state, hard-pose updates
  from posterior maxima, and the §7.4 prior schedule.

The pose-marginal driver is callback-based on three integration boundaries:

* ``block_provider`` — projects ``theta_score`` onto rotation blocks and
  yields ``(Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior)`` per
  block. The implementation depends on the dataset backend
  (``recovar.data_io.cryoem_dataset`` for production; synthetic fixtures
  for tests). Lands in M10 alongside the dataset-loading concern.
* ``backprojector`` — converts image-level augmented moments to half-volume
  ``AugmentedPPCAStats`` via the existing
  ``recovar.em.dense_single_volume.helpers.backprojection`` helpers.
* ``halfset_combiner`` — combines ``mu_half`` and ``W_half`` into the
  filtered ``theta_score`` used by the next E-step
  (``recovar.em.dense_single_volume.helpers.convergence.RefinementState``
  pattern). Default in M5: simple average; production halfset combine
  lands in M10.

This separation lets the driver be tested in isolation without a real
``CryoEMDataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, NamedTuple, Protocol

import jax.numpy as jnp

from recovar.ppca import EM as legacy_em
from recovar.ppca import (
    AugmentedPPCAStats,
    PCPriorConfig,
    solve_augmented_ppca_mstep,
)

from .dense_engine import (
    DenseImageStats,
    PosteriorDiagnostics,
    dense_pose_ppca_E_step_blocked,
)
from .state import FixedPosePPCAState, PoseMarginalPPCAEMState

__all__ = [
    "run_fixed_pose_ppca_refine",
    "run_pose_marginal_ppca_refine",
    "IterationOpts",
    "PoseBlock",
    "BlockProvider",
    "Backprojector",
    "HalfsetCombiner",
]


# ---------------------------------------------------------------------------
# M3 driver (already implemented)
# ---------------------------------------------------------------------------


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
    ``recovar.ppca.ppca.EM(...)``. See :class:`FixedPosePPCAState` for the
    state contract.
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


# ---------------------------------------------------------------------------
# M5 driver: pose-marginal augmented EM
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IterationOpts:
    """Options for ``run_pose_marginal_ppca_refine``."""

    EM_iter: int = 20
    pcg_maxiter: int = 20
    pcg_tol: float = 1e-4
    significance_threshold: float = 1e-3
    update_noise: bool = False
    noise_update_ema: float = 0.5
    use_float64_scoring: bool = False
    pc_prior_config: PCPriorConfig | None = None


class PoseBlock(NamedTuple):
    """A single (image_batch × rotation × translation) block input.

    Constructed by the ``block_provider`` callback per EM iteration.
    """

    Y1: Any  # [B, T, F] complex64
    proj_aug: Any  # [R, P, F] complex64
    ctf2_over_noise: Any  # [B, F] real32
    y_norm: Any  # [B] real32
    pose_log_prior: Any | None  # [B, R, T] real32 or None
    image_indices: Any  # [B] int32 — global image indices for backprojection
    halfset_idx: int  # 0 or 1
    rotations: Any  # [R, 3, 3] real32 — per-block rotations (for backprojection)
    translations: Any  # [T, 2] real32 — per-block translation offsets


class BlockProvider(Protocol):
    """Yields blocks for one E-step pass."""

    def __call__(
        self,
        theta_score_for_half: tuple[Any, Any],  # (mu_score, W_score)
        iteration: int,
    ) -> Iterable[PoseBlock]: ...


class Backprojector(Protocol):
    """Converts image-level augmented moments + posterior diagnostics into
    half-volume ``AugmentedPPCAStats``.

    Implementation reuses
    ``recovar.em.dense_single_volume.helpers.backprojection.accumulate_adjoint_pair``
    per augmented component plus the half-volume Hermitian enforcement on
    the x=0 plane.
    """

    def __call__(
        self,
        image_stats: list[tuple[PoseBlock, DenseImageStats, PosteriorDiagnostics]],
        halfset_idx: int,
    ) -> AugmentedPPCAStats: ...


class HalfsetCombiner(Protocol):
    """Combines two halfset volumes (μ or W components) into a single
    filtered scoring volume. Production implementation uses the
    ``RefinementState`` filtering convention in
    ``recovar.em.dense_single_volume.helpers.convergence``.
    """

    def __call__(self, vol_h0: Any, vol_h1: Any, kind: str) -> Any: ...


def _default_halfset_combine(vol_h0, vol_h1, kind: str):
    """Simple arithmetic mean — placeholder for the M10 production combine
    via ``RefinementState`` filtered averaging."""
    return 0.5 * (jnp.asarray(vol_h0) + jnp.asarray(vol_h1))


def _split_blocks_by_halfset(blocks: Iterable[PoseBlock]):
    h0, h1 = [], []
    for b in blocks:
        (h0 if b.halfset_idx == 0 else h1).append(b)
    return h0, h1


def _should_recompute_prior(it: int, cfg: PCPriorConfig) -> bool:
    """§7.4 schedule: freeze for the first `prior_freeze_iters`; one-shot
    recompute at `recompute_once_after_iter` (only when allowed)."""
    if cfg.allow_every_iter_prior_update:
        return it >= cfg.prior_freeze_iters
    if cfg.recompute_once_after_iter is not None and it == cfg.recompute_once_after_iter:
        return True
    return False


def run_pose_marginal_ppca_refine(
    initial_state: PoseMarginalPPCAEMState,
    *,
    block_provider: BlockProvider,
    backprojector: Backprojector,
    halfset_combiner: HalfsetCombiner | None = None,
    mask: Any,
    masks: Any | None = None,
    pc_mask_assignment: Any | None = None,
    mean_mask_idx: int = 0,
    prior_recompute_fn: Callable[[PoseMarginalPPCAEMState], Any] | None = None,
    opts: IterationOpts = IterationOpts(),
) -> tuple[PoseMarginalPPCAEMState, list[dict]]:
    """Pose-marginalized augmented EM driver (M5).

    Parameters
    ----------
    initial_state:
        :class:`PoseMarginalPPCAEMState` carrying ``mu_half``, ``W_half``,
        ``mu_score``, ``W_score``, ``W_prior``, ``mean_prior``,
        ``noise_variance``, ``contrast_params``, etc.
    block_provider:
        Callback that yields (R, T, F) score blocks for the current
        ``theta_score``. Splits images into halfsets internally — each
        block carries its ``halfset_idx``.
    backprojector:
        Callback converting image-level moments + diagnostics into
        half-volume ``AugmentedPPCAStats``.
    halfset_combiner:
        Combines ``mu_half`` and ``W_half`` into ``mu_score`` and
        ``W_score`` for the next E-step. Default: simple mean.
    mask, masks, pc_mask_assignment, mean_mask_idx:
        Forwarded to :func:`recovar.ppca.solve_augmented_ppca_mstep`.
    prior_recompute_fn:
        Optional callable that recomputes ``W_prior`` from the current
        state when the schedule allows. Wired in M10 via
        ``estimate_hybrid_shell_prior_from_data``.

    Returns
    -------
    final_state:
        Updated :class:`PoseMarginalPPCAEMState` with refined μ, W, and
        diagnostics.
    iteration_log:
        Per-iteration list of diagnostic dicts: log_evidence, mean
        ``pmax``, mean ``n_significant``, prior penalty, etc.
    """
    if halfset_combiner is None:
        halfset_combiner = _default_halfset_combine

    state = initial_state
    iteration_log: list[dict] = []
    cfg = opts.pc_prior_config or PCPriorConfig()

    for it in range(opts.EM_iter):
        # ------------------------------ E-step --------------------------------
        all_blocks = list(block_provider((state.mu_score, state.W_score), it))
        h0_blocks, h1_blocks = _split_blocks_by_halfset(all_blocks)

        per_half_image_stats: dict[int, list] = {0: [], 1: []}
        all_logZ: list[float] = []
        all_pmax: list[float] = []
        all_nsig: list[int] = []

        for half_idx, half_blocks in [(0, h0_blocks), (1, h1_blocks)]:
            for block in half_blocks:
                img_stats, diag = dense_pose_ppca_E_step_blocked(
                    block.Y1,
                    block.proj_aug,
                    block.ctf2_over_noise,
                    block.y_norm,
                    block.pose_log_prior,
                    significance_threshold=opts.significance_threshold,
                )
                per_half_image_stats[half_idx].append((block, img_stats, diag))
                all_logZ.extend([float(x) for x in jnp.asarray(diag.logZ)])
                all_pmax.extend([float(x) for x in jnp.asarray(diag.pmax)])
                all_nsig.extend([int(x) for x in jnp.asarray(diag.n_significant_per_image)])

        # ------------------------------ M-step --------------------------------
        new_mu_half: list[Any] = []
        new_W_half: list[Any] = []
        for half_idx in (0, 1):
            stats: AugmentedPPCAStats = backprojector(per_half_image_stats[half_idx], half_idx)
            mu_init, W_init = state.mu_half[half_idx], state.W_half[half_idx]
            mu_real, W_real = solve_augmented_ppca_mstep(
                stats,
                mean_prior=state.mean_prior,
                W_prior=state.W_prior,
                mask=mask,
                masks=masks,
                pc_mask_assignment=pc_mask_assignment,
                mean_mask_idx=mean_mask_idx,
                maxiter=opts.pcg_maxiter,
                tol=opts.pcg_tol,
                theta_init=(mu_init, W_init) if mu_init is not None and W_init is not None else None,
            )
            new_mu_half.append(mu_real)
            new_W_half.append(W_real)

        # ----------------------- Halfset combine + state update --------------
        mu_score_new = halfset_combiner(new_mu_half[0], new_mu_half[1], "mu")
        W_score_new = halfset_combiner(new_W_half[0], new_W_half[1], "W")

        # Optional prior recompute per §7.4 schedule.
        new_W_prior = state.W_prior
        if prior_recompute_fn is not None and _should_recompute_prior(it, cfg):
            new_W_prior = prior_recompute_fn(state)

        state = state.replace(
            mu_half=(new_mu_half[0], new_mu_half[1]),
            W_half=(new_W_half[0], new_W_half[1]),
            mu_score=mu_score_new,
            W_score=W_score_new,
            W_prior=new_W_prior,
        )

        # ------------------------------ Diagnostics ---------------------------
        n_images = len(all_logZ) or 1
        iter_info = {
            "iteration": it,
            "log_evidence_total": float(sum(all_logZ)),
            "log_evidence_mean": float(sum(all_logZ) / n_images),
            "pmax_mean": float(sum(all_pmax) / n_images),
            "n_significant_mean": float(sum(all_nsig) / n_images),
            "prior_recomputed": new_W_prior is not state.W_prior,
        }
        if it == 0 and opts.pc_prior_config is not None:
            iter_info["pc_prior_config"] = opts.pc_prior_config.to_dict()
        iteration_log.append(iter_info)

    return state, iteration_log
