"""EM-compatible pruning rules for cryoSPARC-style BnB.

Splits the rule into:
- ``prune_by_posterior_mass_upper_bound``: per-image score margin
  tau = -log(posterior_tail_tol). A candidate is kept iff its upper score is
  within tau of the per-image best upper score. This is the EM-correct rule:
  any candidate it drops carries at most exp(-tau) of the posterior mass.
- ``apply_orientation_and_shift_caps``: cryoSPARC's pathological-image
  guards — keep at most 12.5% of orientations and 25% of shifts per image
  (Suppl §"Approximations"), with floors ``min_orientations_per_image`` and
  ``min_shifts_per_image`` so a noise-only particle does not collapse to
  zero candidates.
- ``compute_omitted_mass_upper``: per-image upper bound on the omitted
  posterior mass after pruning, for the convergence-guard diagnostic.

The Phase-2 ``support._prune_by_tail_mass_and_caps`` is implemented as a
shim over these three primitives for backward compatibility.
"""

from __future__ import annotations

import numpy as np

from .options import BranchBoundOptions


def prune_by_score_margin(
    sample_mask: np.ndarray,
    upper_scores: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    """Keep candidates whose upper score is within ``tau`` of the per-image best.

    Inactive candidates (sample_mask == False) get score -inf so they cannot
    survive. Returns a new sample_mask.
    """
    n_images = sample_mask.shape[0]
    flat_upper = np.where(sample_mask, upper_scores, -np.inf).reshape(n_images, -1)
    best = np.max(flat_upper, axis=1)
    keep_flat = flat_upper >= (best[:, None] - float(tau))
    return keep_flat.reshape(sample_mask.shape)


def apply_orientation_cap(
    sample_mask: np.ndarray,
    upper_scores: np.ndarray,
    *,
    fraction: float,
    floor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most ``ceil(fraction * n_rot)`` rotations per image (with floor).

    A rotation is "kept" if any retained translation under it survives.
    Returns ``(new_sample_mask, cap_applied_per_image)`` where
    ``cap_applied_per_image[i] = True`` iff the cap actually fired (i.e.
    fewer rotations would have been kept by the margin rule alone).
    """
    n_images, n_rot, n_trans = sample_mask.shape
    rot_upper = np.where(sample_mask, upper_scores, -np.inf).max(axis=2)
    n_keep = max(int(floor), int(np.ceil(fraction * n_rot)))
    n_keep = min(n_keep, n_rot)
    cap_applied = np.zeros(n_images, dtype=bool)
    if n_keep < n_rot:
        thresh = np.partition(rot_upper, n_rot - n_keep, axis=1)[:, n_rot - n_keep]
        rot_keep = rot_upper >= thresh[:, None]
        cap_applied = sample_mask.any(axis=2).sum(axis=1) > n_keep
        new_mask = sample_mask & rot_keep[:, :, None]
    else:
        new_mask = sample_mask
    return new_mask, cap_applied


def apply_shift_cap(
    sample_mask: np.ndarray,
    upper_scores: np.ndarray,
    *,
    fraction: float,
    floor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most ``ceil(fraction * n_trans)`` shifts per image (with floor).

    Returns ``(new_sample_mask, cap_applied_per_image)``.
    """
    n_images, n_rot, n_trans = sample_mask.shape
    trans_upper = np.where(sample_mask, upper_scores, -np.inf).max(axis=1)
    n_keep = max(int(floor), int(np.ceil(fraction * n_trans)))
    n_keep = min(n_keep, n_trans)
    cap_applied = np.zeros(n_images, dtype=bool)
    if n_keep < n_trans:
        thresh = np.partition(trans_upper, n_trans - n_keep, axis=1)[:, n_trans - n_keep]
        trans_keep = trans_upper >= thresh[:, None]
        cap_applied = sample_mask.any(axis=1).sum(axis=1) > n_keep
        new_mask = sample_mask & trans_keep[:, None, :]
    else:
        new_mask = sample_mask
    return new_mask, cap_applied


def compute_omitted_mass_upper(
    pre_prune_mask: np.ndarray,
    post_prune_mask: np.ndarray,
    upper_scores: np.ndarray,
) -> np.ndarray:
    """Per-image upper bound on the omitted posterior mass.

    For image i, with U_i(q) the score upper bound:

        rho_i <= sum_{q in pruned} exp(U_i(q) - U_i^max) /
                 sum_{q in kept}   exp(U_i(q) - U_i^max)

    where U_i^max is the max over the kept set. If the kept set is empty,
    returns 1.0 for that image (the pruning rule must restore at least one
    candidate before this is meaningful).
    """
    n_images = pre_prune_mask.shape[0]
    rho = np.zeros(n_images, dtype=np.float32)
    for i in range(n_images):
        flat_u = upper_scores[i].reshape(-1)
        active = pre_prune_mask[i].reshape(-1)
        kept = post_prune_mask[i].reshape(-1)
        pruned = active & ~kept
        if not np.any(kept):
            rho[i] = 1.0
            continue
        u_max = float(np.max(flat_u[kept]))
        kept_sum = float(np.sum(np.exp(flat_u[kept] - u_max)))
        pruned_sum = (
            float(np.sum(np.exp(flat_u[pruned] - u_max))) if np.any(pruned) else 0.0
        )
        rho[i] = pruned_sum / max(kept_sum + pruned_sum, 1e-30)
    return rho


def prune_by_tail_mass_and_caps(
    sample_mask: np.ndarray,
    upper_scores: np.ndarray,
    options: BranchBoundOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply margin pruning + caps + floors. Returns
    ``(new_mask, omitted_mass_upper, cap_applied_per_image)``.
    """
    if options.score_margin is not None:
        tau = float(options.score_margin)
    else:
        tau = -np.log(max(options.posterior_tail_tol, 1e-300))

    pre = sample_mask.copy()
    masked_by_margin = prune_by_score_margin(pre, upper_scores, tau=tau)
    after_rot_cap, cap_rot = apply_orientation_cap(
        masked_by_margin, upper_scores,
        fraction=options.max_orientation_fraction,
        floor=options.min_orientations_per_image,
    )
    after_shift_cap, cap_shift = apply_shift_cap(
        after_rot_cap, upper_scores,
        fraction=options.max_shift_fraction,
        floor=options.min_shifts_per_image,
    )
    cap_applied = cap_rot | cap_shift

    # Floor on joint candidates per image: if an image has fewer than
    # min_joint_candidates_per_image survivors, restore the top-K overall.
    floor = int(options.min_joint_candidates_per_image)
    n_images, n_rot, n_trans = sample_mask.shape
    out = after_shift_cap.copy()
    n_kept = out.reshape(n_images, -1).sum(axis=1)
    below = n_kept < floor
    if np.any(below) and floor > 0:
        for i in np.where(below)[0]:
            flat_u = np.where(pre[i].reshape(-1), upper_scores[i].reshape(-1), -np.inf)
            k = min(floor, n_rot * n_trans)
            if k <= 0 or not np.any(np.isfinite(flat_u)):
                continue
            thresh = np.partition(flat_u, n_rot * n_trans - k)[n_rot * n_trans - k]
            keep_i = (flat_u >= thresh).reshape(n_rot, n_trans) & pre[i]
            out[i] = keep_i

    # Ceiling on joint candidates per image (rare).
    ceiling = int(options.max_joint_candidates_per_image)
    if ceiling > 0:
        n_kept = out.reshape(n_images, -1).sum(axis=1)
        above = n_kept > ceiling
        if np.any(above):
            for i in np.where(above)[0]:
                flat_u = np.where(out[i].reshape(-1), upper_scores[i].reshape(-1), -np.inf)
                k = ceiling
                thresh = np.partition(flat_u, n_rot * n_trans - k)[n_rot * n_trans - k]
                out[i] = (flat_u >= thresh).reshape(n_rot, n_trans)

    rho = compute_omitted_mass_upper(pre, out, upper_scores)
    return out, rho, cap_applied
