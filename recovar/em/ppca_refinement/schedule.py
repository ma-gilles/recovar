"""PPCA-specific resolution gating around the K-class refinement schedule."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class PPCARefinementScheduleState:
    """Schedule state tracked by PPCA on top of K-class pose mechanics."""

    current_size: int
    healpix_order: int
    translation_stage: int = 0
    best_pose_indices: np.ndarray | None = None
    previous_best_pose_indices: np.ndarray | None = None
    pose_change_fraction: float = 1.0
    halfset_mean_fsc: np.ndarray | None = None
    halfset_means_aligned: bool = False
    halfset_resolution_supports: bool = False
    no_halfset_drift: bool = True
    kclass_schedule_allows: bool = False
    pmax_mean: float = 0.0
    logZ_mean: float = float("-inf")
    nsig_mean: float = 0.0
    W_subspace_agreement: float | None = None
    q: int = 0
    diagnostics: dict = field(default_factory=dict)

    def replace(self, **changes) -> "PPCARefinementScheduleState":
        return replace(self, **changes)


@dataclass(frozen=True)
class HalfsetResolutionGateDecision:
    """Decision from the gold-standard PPCA halfset/current-size gate."""

    allow_increase: bool
    reasons: tuple[str, ...]
    pose_change_fraction: float
    W_subspace_agreement: float | None = None


def compute_pose_change_fraction(current_best, previous_best) -> float:
    """Return the fraction of changed best-pose assignments."""
    if current_best is None or previous_best is None:
        return 1.0
    current = np.asarray(current_best)
    previous = np.asarray(previous_best)
    if current.shape != previous.shape:
        raise ValueError(f"best-pose shapes differ: {current.shape} vs {previous.shape}")
    if current.size == 0:
        return 0.0
    return float(np.mean(current != previous))


def loading_subspace_agreement(W_a, W_b, *, rtol: float = 1e-12) -> float:
    """Return a sign/rotation-invariant loading-subspace agreement in [0, 1]."""
    if W_a is None or W_b is None:
        return float("nan")
    A = np.asarray(W_a).reshape(np.asarray(W_a).shape[0], -1)
    B = np.asarray(W_b).reshape(np.asarray(W_b).shape[0], -1)
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"loading banks must have same q, got {A.shape[0]} and {B.shape[0]}")
    q = A.shape[0]
    if q == 0:
        return 1.0

    def _orthonormal_rows(X):
        norms = np.linalg.norm(X, axis=1)
        keep = norms > rtol
        if not np.any(keep):
            return np.zeros((0, X.shape[1]), dtype=np.float64)
        Q, _ = np.linalg.qr(X[keep].T)
        return Q.T

    Qa = _orthonormal_rows(A)
    Qb = _orthonormal_rows(B)
    if Qa.shape[0] == 0 and Qb.shape[0] == 0:
        return 1.0
    if Qa.shape[0] != Qb.shape[0]:
        return 0.0
    singular_values = np.linalg.svd(Qa @ Qb.T, compute_uv=False)
    if singular_values.size == 0:
        return 1.0
    return float(np.clip(np.min(singular_values), 0.0, 1.0))


def evaluate_halfset_resolution_gate(
    state: PPCARefinementScheduleState,
    *,
    pose_stability_threshold: float = 0.0,
    require_pmax_at_least: float | None = None,
) -> HalfsetResolutionGateDecision:
    """Evaluate PPCA resolution growth.

    Good pmax is optional supporting evidence. It cannot permit resolution
    growth without K-class schedule agreement, pose stability, aligned halfset
    means, and halfset mean-resolution support.
    """
    reasons: list[str] = []
    pose_change = compute_pose_change_fraction(state.best_pose_indices, state.previous_best_pose_indices)
    if pose_change > float(pose_stability_threshold):
        reasons.append("best poses changed")
    if not bool(state.kclass_schedule_allows):
        reasons.append("kclass schedule blocked")
    if not bool(state.halfset_means_aligned):
        reasons.append("halfset means not aligned")
    if not bool(state.halfset_resolution_supports):
        reasons.append("halfset mean comparison below requested resolution")
    if not bool(state.no_halfset_drift):
        reasons.append("halfset drift or frame mismatch detected")
    if require_pmax_at_least is not None and float(state.pmax_mean) < float(require_pmax_at_least):
        reasons.append("pmax below diagnostic threshold")
    return HalfsetResolutionGateDecision(
        allow_increase=not reasons,
        reasons=tuple(reasons),
        pose_change_fraction=pose_change,
        W_subspace_agreement=state.W_subspace_agreement,
    )
