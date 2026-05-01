"""PPCA prior configuration for the pose-marginalized PPCA refinement project.

See ``recovar/em/ppca_refinement/CLAUDE.md`` §7 for the full prior contract:
the latent prior on ``z`` is identity in v1, and the loading-volume prior on
``W`` is a per-voxel per-PC variance ``W_prior[half_vol, q]`` with
regularizer ``1 / (W_prior + ε)``.

The default ``W_prior(ξ, k) = max(τ_floor, α_prior · d_ppca(shell(ξ)) / q_total)``
is computed by ``recovar/ppca/prior_estimation.py::estimate_hybrid_shell_prior_from_data``.
``PCPriorConfig`` is the lightweight, JSON-serializable bundle of knobs that
controls **how** that prior is estimated, refreshed, and (optionally) frozen
during EM. It is passed through the new pose-marginalized refinement driver
(``recovar.em.ppca_refinement``) and snapshotted into per-iteration
diagnostics so a future agent can recover the run's prior policy from
``iteration_data``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PCPriorConfig:
    """Configuration for the PPCA loading-volume prior ``W_prior``.

    Defaults match §7 of ``recovar/em/ppca_refinement/CLAUDE.md``. ``frozen``
    so it can be safely shared across JIT boundaries and snapshotted into
    diagnostics without aliasing.

    Attributes
    ----------
    latent_prior_mode:
        Always ``"identity"`` in v1 — ``z ~ N(0, I_q)``. PC eigenvalues live
        in ``W``, not in the latent prior. Future work may add other modes.
    pc_prior_mode:
        ``"hybrid_shell"`` uses ``estimate_hybrid_shell_prior_from_data`` to
        build a shell-averaged variance with a ``|μ|²`` repair on unreliable
        high-frequency shells. Other modes are reserved for ablation.
    prior_scale:
        ``α_prior`` in the formula
        ``W_prior(ξ, k) = max(τ_floor, α_prior · d_ppca(shell(ξ)) / q_total)``.
    variance_floor:
        ``τ_floor`` in the same formula. Avoids division-by-zero in
        ``W_reg_diag = 1 / (W_prior + ε)``.
    use_q_total_for_division:
        Always ``True`` for v1: the divisor is ``q_total = opts.zdim``, not
        ``q_active``. Staged activation must not weaken regularization on
        the first active PCs.
    smooth_shell_prior:
        Enable the smoothing pass inside ``estimate_hybrid_shell_prior_from_data``.
    prior_freeze_iters:
        Iterations 0..prior_freeze_iters−1 use the prior computed at iter 0
        without recomputation. Avoids coupling prior drift to pose / contrast
        / noise drift early in EM.
    recompute_once_after_iter:
        When ``allow_every_iter_prior_update`` is False (default), this is
        the single iteration after which ``W_prior`` may be refreshed once.
        ``None`` disables the one-shot recompute.
    allow_every_iter_prior_update:
        Off by default. Turning this on couples prior scale to every other
        moving piece of EM and makes diagnostics hard to read; only enable
        for explicit ablation runs.
    """

    latent_prior_mode: str = "identity"
    pc_prior_mode: str = "hybrid_shell"
    prior_scale: float = 1.0
    variance_floor: float = 1e-8
    use_q_total_for_division: bool = True
    smooth_shell_prior: bool = True
    prior_freeze_iters: int = 3
    recompute_once_after_iter: int | None = 5
    allow_every_iter_prior_update: bool = False

    def to_dict(self) -> dict:
        """JSON-serializable snapshot for diagnostics."""
        return asdict(self)
