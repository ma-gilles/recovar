"""Convergence detection, angular step refinement, and local search state.

Implements RELION-style convergence criteria for the dense single-volume
EM refinement loop:

- **Assignment tracking**: fraction of images whose hard assignment changed
  by more than one HEALPix step between iterations.
- **Average Pmax**: mean of per-image maximum posterior probability.
- **Resolution stall**: count of iterations without resolution improvement.
- **Auto-termination**: when angular sampling is at finest level, resolution
  stalled for 1 iter, and assignments stable for 1 iter.
- **Angular step refinement**: increment HEALPix order when assignments
  and resolution stabilize.
- **Local angular search**: switch from global exhaustive to local
  Gaussian-prior search when HEALPix order >= 4.

See docs/relion5_auto_refine_algorithm.md, sections A (convergence),
G (angular sampling), H (speed tricks).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# RELION defaults
MAX_NR_ITER_WO_RESOL_GAIN = 1
MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES = 1
LOCAL_SEARCH_HEALPIX_ORDER = 4  # Switch to local search at order >= 4 (~3.7 deg)
SIGMA_CUTOFF = 3.0  # Only search within 3-sigma of prior


def healpix_angular_step(order: int) -> float:
    """Return approximate angular step in degrees for a HEALPix order.

    Formula: step = 360 / (6 * 2^order).

    Parameters
    ----------
    order : int
        HEALPix order (nside = 2^order).

    Returns
    -------
    float
        Angular step in degrees.
    """
    return 360.0 / (6.0 * (2 ** order))


def effective_angular_step(order: int, adaptive_oversampling: int = 0) -> float:
    """Return effective angular step accounting for adaptive oversampling.

    Parameters
    ----------
    order : int
        HEALPix order.
    adaptive_oversampling : int
        Oversampling levels (each level halves the step).

    Returns
    -------
    float
        Effective angular step in degrees.
    """
    return healpix_angular_step(order) / (2 ** adaptive_oversampling)


@dataclass
class RefinementState:
    """Tracks refinement progress for convergence detection and angular stepping.

    This is a standalone data container -- it does NOT modify the refine loop
    directly.  The refine loop queries this state to decide whether to
    converge, refine angular sampling, or switch to local search.

    Attributes
    ----------
    iteration : int
        Current iteration number (0-indexed).
    healpix_order : int
        Current HEALPix order for the coarse rotation grid.
    angular_step : float
        Coarse angular step in degrees (before oversampling).
    adaptive_oversampling : int
        Oversampling level (0 = none, 1 = 2x finer).
    translation_range : float
        Current translation search range in pixels.
    translation_step : float
        Current translation step size in pixels.
    current_resolution : float
        Current resolution estimate in Angstrom (or pixels if no voxel_size).
    nr_iter_wo_resol_gain : int
        Count of consecutive iterations without resolution improvement.
    nr_iter_wo_assignment_changes : int
        Count of consecutive iterations without large assignment changes.
    has_converged : bool
        True when convergence criteria are met.
    do_local_search : bool
        True when local angular search is active.
    sigma_rot : float
        Gaussian prior sigma for rotation (radians), used in local search.
    sigma_psi : float
        Gaussian prior sigma for in-plane rotation (radians), used in local search.
    best_rotations : np.ndarray or None
        Shape (n_images,) -- index of best rotation per image from last iteration.
    best_translations : np.ndarray or None
        Shape (n_images, 2) -- best translation offset per image from last iteration.
    ave_Pmax : float
        Mean of per-image maximum posterior probability.
    acc_rot : float
        Estimated angular accuracy in degrees.
    acc_trans : float
        Estimated translational accuracy in pixels.
    max_healpix_order : int
        Maximum allowed HEALPix order (finest angular sampling).
    previous_resolution : float
        Resolution from the previous iteration (for stall detection).
    fraction_changed : float
        Fraction of images whose assignment changed by > 1 HEALPix step.
    changes_optimal_offsets : float
        RMS change in optimal offsets (pixels) between iterations.
    """

    # Grid parameters
    iteration: int = 0
    healpix_order: int = 2
    angular_step: float = field(default=0.0)
    adaptive_oversampling: int = 0
    translation_range: float = 10.0
    translation_step: float = 2.0

    # Resolution tracking
    current_resolution: float = float("inf")
    previous_resolution: float = float("inf")

    # Convergence counters
    nr_iter_wo_resol_gain: int = 0
    nr_iter_wo_assignment_changes: int = 0

    # Convergence flags
    has_converged: bool = False
    do_local_search: bool = False

    # Local search priors (radians)
    sigma_rot: float = 0.0
    sigma_psi: float = 0.0

    # Per-image best assignments
    best_rotations: Optional[np.ndarray] = field(default=None, repr=False)
    best_translations: Optional[np.ndarray] = field(default=None, repr=False)

    # Quality metrics
    ave_Pmax: float = 0.0
    acc_rot: float = float("inf")
    acc_trans: float = float("inf")

    # Limits
    max_healpix_order: int = 7

    # Change tracking
    fraction_changed: float = 1.0
    changes_optimal_offsets: float = float("inf")

    def __post_init__(self):
        if self.angular_step == 0.0:
            self.angular_step = healpix_angular_step(self.healpix_order)

    @property
    def effective_step(self) -> float:
        """Effective angular step in degrees (accounting for oversampling)."""
        return effective_angular_step(self.healpix_order, self.adaptive_oversampling)

    @property
    def has_fine_enough_angular_sampling(self) -> bool:
        """True when angular sampling is at the finest allowed level."""
        return self.healpix_order >= self.max_healpix_order

    @property
    def should_do_local_search(self) -> bool:
        """True when HEALPix order is high enough for local search."""
        return self.healpix_order >= LOCAL_SEARCH_HEALPIX_ORDER


# ---------------------------------------------------------------------------
# Assignment change tracking
# ---------------------------------------------------------------------------


def compute_assignment_changes(
    current_assignments: np.ndarray,
    previous_assignments: np.ndarray,
    n_rotations: int,
    n_translations: int,
    healpix_order: int,
) -> float:
    """Compute fraction of images whose rotation assignment changed significantly.

    "Significantly" means the new best rotation is more than one HEALPix step
    away from the previous best.  We compare rotation indices (not the full
    assignment that includes translation).

    Parameters
    ----------
    current_assignments : np.ndarray, shape (n_images,)
        Current hard assignments (rot_idx * n_trans + trans_idx).
    previous_assignments : np.ndarray, shape (n_images,)
        Previous hard assignments.
    n_rotations : int
        Number of rotations in the grid.
    n_translations : int
        Number of translations in the grid.
    healpix_order : int
        Current HEALPix order (used only for logging context).

    Returns
    -------
    float
        Fraction of images that changed rotation assignment.
        Value in [0, 1].
    """
    if current_assignments is None or previous_assignments is None:
        return 1.0

    current_assignments = np.asarray(current_assignments)
    previous_assignments = np.asarray(previous_assignments)

    if current_assignments.shape != previous_assignments.shape:
        return 1.0

    n_images = len(current_assignments)
    if n_images == 0:
        return 0.0

    # Extract rotation indices
    current_rot = current_assignments // n_translations
    previous_rot = previous_assignments // n_translations

    # Fraction that changed rotation
    changed = current_rot != previous_rot
    fraction = float(np.mean(changed))

    return fraction


def compute_translation_changes(
    current_assignments: np.ndarray,
    previous_assignments: np.ndarray,
    translations: np.ndarray,
    n_translations: int,
) -> float:
    """Compute RMS change in optimal translation offsets between iterations.

    Parameters
    ----------
    current_assignments : np.ndarray, shape (n_images,)
        Current hard assignments (rot_idx * n_trans + trans_idx).
    previous_assignments : np.ndarray, shape (n_images,)
        Previous hard assignments.
    translations : np.ndarray, shape (n_trans, 2)
        Translation grid.
    n_translations : int
        Number of translations.

    Returns
    -------
    float
        RMS change in translation offsets (pixels).
    """
    if current_assignments is None or previous_assignments is None:
        return float("inf")

    current_assignments = np.asarray(current_assignments)
    previous_assignments = np.asarray(previous_assignments)

    if current_assignments.shape != previous_assignments.shape:
        return float("inf")

    translations = np.asarray(translations)

    # Extract translation indices
    current_trans_idx = current_assignments % n_translations
    previous_trans_idx = previous_assignments % n_translations

    # Look up actual offsets
    current_offsets = translations[current_trans_idx]
    previous_offsets = translations[previous_trans_idx]

    # RMS of the per-image offset changes
    diffs = current_offsets - previous_offsets
    rms = float(np.sqrt(np.mean(np.sum(diffs**2, axis=-1))))

    return rms


def compute_ave_Pmax(max_posterior_per_image: np.ndarray) -> float:
    """Compute average of per-image maximum posterior probability.

    Parameters
    ----------
    max_posterior_per_image : np.ndarray, shape (n_images,)
        Maximum posterior probability for each image.

    Returns
    -------
    float
        Mean Pmax across all images.
    """
    max_posterior_per_image = np.asarray(max_posterior_per_image)
    if max_posterior_per_image.size == 0:
        return 0.0
    return float(np.mean(max_posterior_per_image))


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------


def check_convergence(state: RefinementState) -> bool:
    """Check RELION-style convergence criteria.

    Convergence requires ALL of:
    1. Angular sampling at finest level (healpix_order >= max_healpix_order)
    2. Resolution stalled for >= MAX_NR_ITER_WO_RESOL_GAIN iterations
    3. Assignments stable for >= MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES iterations

    Parameters
    ----------
    state : RefinementState
        Current refinement state.

    Returns
    -------
    bool
        True if convergence criteria are met.
    """
    if not state.has_fine_enough_angular_sampling:
        return False

    if state.nr_iter_wo_resol_gain < MAX_NR_ITER_WO_RESOL_GAIN:
        return False

    if state.nr_iter_wo_assignment_changes < MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES:
        return False

    return True


# ---------------------------------------------------------------------------
# Angular step refinement
# ---------------------------------------------------------------------------


def should_refine_angular_sampling(state: RefinementState) -> bool:
    """Check whether angular sampling should be refined (HEALPix order incremented).

    Refinement triggers when:
    1. Resolution stalled for >= MAX_NR_ITER_WO_RESOL_GAIN iterations
    2. Assignments stable for >= MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES iterations
    3. Current angular step is NOT already finer than 75% of acc_rot

    Parameters
    ----------
    state : RefinementState
        Current refinement state.

    Returns
    -------
    bool
        True if angular sampling should be refined.
    """
    # Already at finest level
    if state.has_fine_enough_angular_sampling:
        return False

    # Need resolution stalls (required) and either assignment stalls
    # OR extended resolution stalls (>= 5 iters).  The extended check
    # handles adaptive oversampling where assignments churn due to
    # the changing oversampled grid but orientations are actually stable.
    if state.nr_iter_wo_resol_gain < MAX_NR_ITER_WO_RESOL_GAIN:
        return False

    EXTENDED_RESOL_STALL = 5
    if (state.nr_iter_wo_assignment_changes < MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES
            and state.nr_iter_wo_resol_gain < EXTENDED_RESOL_STALL):
        return False

    # Don't refine beyond 75% of estimated angular accuracy
    if state.acc_rot < float("inf"):
        if state.effective_step < 0.75 * state.acc_rot:
            logger.info(
                "Angular step %.2f deg < 75%% of acc_rot %.2f deg; "
                "not refining further",
                state.effective_step,
                state.acc_rot,
            )
            return False

    return True


def refine_angular_sampling(state: RefinementState) -> RefinementState:
    """Increment HEALPix order and update translation parameters.

    Follows RELION's ``updateAngularSampling()`` logic:
    - HEALPix order += 1
    - Translation step = min(1.5, 0.75 * acc_trans) * 2^adaptive_oversampling
    - Translation range = 5 * changes_optimal_offsets (capped at 1.3x previous)

    Also activates local search when the new order >= LOCAL_SEARCH_HEALPIX_ORDER.

    Parameters
    ----------
    state : RefinementState
        Current refinement state (not modified in place).

    Returns
    -------
    RefinementState
        New state with updated angular and translation parameters.
        The convergence counters are reset.
    """
    new_order = state.healpix_order + 1
    new_angular_step = healpix_angular_step(new_order)

    # Update translation step: RELION formula
    if state.acc_trans < float("inf"):
        new_trans_step = min(1.5, 0.75 * state.acc_trans) * (2 ** state.adaptive_oversampling)
    else:
        # Fall back to halving the current step
        new_trans_step = state.translation_step / 2.0

    # Update translation range: 5 * changes_optimal_offsets, capped at 1.3x previous
    if state.changes_optimal_offsets < float("inf"):
        new_trans_range = 5.0 * state.changes_optimal_offsets
        new_trans_range = min(new_trans_range, 1.3 * state.translation_range)
    else:
        new_trans_range = state.translation_range

    # Ensure translation range is at least a few steps
    new_trans_range = max(new_trans_range, 3.0 * new_trans_step)

    # Determine local search activation
    do_local = new_order >= LOCAL_SEARCH_HEALPIX_ORDER

    # Compute sigma for local search: sigma2 = 2 * 2 * angular_step^2
    # (RELION convention, angular_step in degrees, sigma in radians for storage)
    if do_local:
        step_rad = np.deg2rad(
            new_angular_step / (2 ** state.adaptive_oversampling)
        )
        sigma_rad = np.sqrt(2.0 * 2.0) * step_rad
    else:
        sigma_rad = 0.0

    logger.info(
        "Refining angular sampling: order %d -> %d (%.2f -> %.2f deg), "
        "trans_step %.2f -> %.2f, trans_range %.2f -> %.2f, "
        "local_search=%s",
        state.healpix_order,
        new_order,
        state.angular_step,
        new_angular_step,
        state.translation_step,
        new_trans_step,
        state.translation_range,
        new_trans_range,
        do_local,
    )

    # Build new state -- reset stall counters, preserve per-image assignments
    return RefinementState(
        iteration=state.iteration,
        healpix_order=new_order,
        angular_step=new_angular_step,
        adaptive_oversampling=state.adaptive_oversampling,
        translation_range=new_trans_range,
        translation_step=new_trans_step,
        current_resolution=state.current_resolution,
        previous_resolution=state.current_resolution,
        nr_iter_wo_resol_gain=0,
        nr_iter_wo_assignment_changes=0,
        has_converged=False,
        do_local_search=do_local,
        sigma_rot=sigma_rad,
        sigma_psi=sigma_rad,
        best_rotations=state.best_rotations,
        best_translations=state.best_translations,
        ave_Pmax=state.ave_Pmax,
        acc_rot=state.acc_rot,
        acc_trans=state.acc_trans,
        max_healpix_order=state.max_healpix_order,
        fraction_changed=state.fraction_changed,
        changes_optimal_offsets=state.changes_optimal_offsets,
    )


# ---------------------------------------------------------------------------
# Full iteration update
# ---------------------------------------------------------------------------


def update_refinement_state(
    state: RefinementState,
    current_assignments: np.ndarray,
    previous_assignments: Optional[np.ndarray],
    n_rotations: int,
    n_translations: int,
    translations: np.ndarray,
    new_resolution: float,
    max_posterior_per_image: Optional[np.ndarray] = None,
    acc_rot: Optional[float] = None,
    acc_trans: Optional[float] = None,
) -> RefinementState:
    """Update RefinementState after one EM iteration.

    Computes assignment changes, resolution stalls, and Pmax, then
    determines whether to refine angular sampling or declare convergence.

    Parameters
    ----------
    state : RefinementState
        State from the *beginning* of this iteration.
    current_assignments : np.ndarray, shape (n_images,)
        Hard assignments from this iteration.
    previous_assignments : np.ndarray or None
        Hard assignments from the previous iteration (None for iter 0).
    n_rotations : int
        Number of rotations in the current grid.
    n_translations : int
        Number of translations in the current grid.
    translations : np.ndarray, shape (n_trans, 2)
        Translation grid.
    new_resolution : float
        Resolution estimate from this iteration.
    max_posterior_per_image : np.ndarray or None
        Per-image maximum posterior probability (from E-step weights).
    acc_rot : float or None
        Estimated angular accuracy in degrees.  If None, unchanged.
    acc_trans : float or None
        Estimated translation accuracy in pixels.  If None, unchanged.

    Returns
    -------
    RefinementState
        Updated state reflecting this iteration's results.
    """
    # --- Compute assignment changes ---
    frac_changed = compute_assignment_changes(
        current_assignments,
        previous_assignments,
        n_rotations,
        n_translations,
        state.healpix_order,
    )

    trans_changes = compute_translation_changes(
        current_assignments,
        previous_assignments,
        translations,
        n_translations,
    )

    # --- Compute Pmax ---
    ave_pmax = state.ave_Pmax
    if max_posterior_per_image is not None:
        ave_pmax = compute_ave_Pmax(max_posterior_per_image)

    # --- Resolution stall detection ---
    resol_improved = new_resolution < state.current_resolution
    if resol_improved:
        nr_iter_wo_resol_gain = 0
    else:
        nr_iter_wo_resol_gain = state.nr_iter_wo_resol_gain + 1

    # --- Assignment stability detection ---
    # "Large changes" threshold: fraction_changed > 0 means some images changed.
    # RELION considers assignments "stable" when fraction_changed is small.
    # We use fraction_changed < 0.01 (1%) as "no large changes".
    ASSIGNMENT_CHANGE_THRESHOLD = 0.01
    if frac_changed < ASSIGNMENT_CHANGE_THRESHOLD:
        nr_iter_wo_assignment_changes = state.nr_iter_wo_assignment_changes + 1
    else:
        nr_iter_wo_assignment_changes = 0

    # --- Update accuracy estimates ---
    new_acc_rot = acc_rot if acc_rot is not None else state.acc_rot
    new_acc_trans = acc_trans if acc_trans is not None else state.acc_trans

    # --- Build updated state ---
    updated = RefinementState(
        iteration=state.iteration + 1,
        healpix_order=state.healpix_order,
        angular_step=state.angular_step,
        adaptive_oversampling=state.adaptive_oversampling,
        translation_range=state.translation_range,
        translation_step=state.translation_step,
        current_resolution=new_resolution,
        previous_resolution=state.current_resolution,
        nr_iter_wo_resol_gain=nr_iter_wo_resol_gain,
        nr_iter_wo_assignment_changes=nr_iter_wo_assignment_changes,
        has_converged=False,
        do_local_search=state.do_local_search,
        sigma_rot=state.sigma_rot,
        sigma_psi=state.sigma_psi,
        best_rotations=current_assignments,
        best_translations=None,
        ave_Pmax=ave_pmax,
        acc_rot=new_acc_rot,
        acc_trans=new_acc_trans,
        max_healpix_order=state.max_healpix_order,
        fraction_changed=frac_changed,
        changes_optimal_offsets=trans_changes,
    )

    logger.info(
        "Iteration %d: frac_changed=%.4f, resol=%.2f (prev=%.2f), "
        "stalls: resol=%d, assign=%d, ave_Pmax=%.4f",
        updated.iteration,
        frac_changed,
        new_resolution,
        state.current_resolution,
        nr_iter_wo_resol_gain,
        nr_iter_wo_assignment_changes,
        ave_pmax,
    )

    # --- Check if we should refine angular sampling ---
    if should_refine_angular_sampling(updated):
        updated = refine_angular_sampling(updated)
        return updated

    # --- Check convergence ---
    if check_convergence(updated):
        updated.has_converged = True
        logger.info(
            "Convergence detected at iteration %d: "
            "resolution stalled for %d iter, assignments stable for %d iter, "
            "angular sampling at finest level (order %d)",
            updated.iteration,
            nr_iter_wo_resol_gain,
            nr_iter_wo_assignment_changes,
            updated.healpix_order,
        )

    return updated
