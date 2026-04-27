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

# RELION's "smallest changes thus far" sentinel values (ml_optimiser.cpp:1042-1044)
SMALLEST_CHANGES_INIT_ORIENTATIONS = 999.0  # degrees
SMALLEST_CHANGES_INIT_OFFSETS = 999.0  # angstroms
SMALLEST_CHANGES_INIT_CLASSES = 9999999.0  # integer count


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
    return 360.0 / (6.0 * (2**order))


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
    return healpix_angular_step(order) / (2**adaptive_oversampling)


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
    # Particle diameter in Å (used for the resolution-based acc_rot proxy
    # in should_refine_angular_sampling). Set by the caller via
    # update_refinement_state(..., particle_diameter_angstrom=...).
    particle_diameter_angstrom: float = 0.0

    # Limits
    max_healpix_order: int = 7

    # Change tracking
    fraction_changed: float = 1.0
    changes_optimal_offsets: float = float("inf")

    # RELION-exact change tracking (B3 + B4):
    # current_changes_optimal_orientations is the mean RELION-style angular
    # distance (calculateAngularDistance, healpix_sampling.cpp:1969) between
    # this iteration's per-particle best rotation and the previous iteration's,
    # in degrees. current_changes_optimal_offsets is the RELION-style per-dim
    # RMS translation change in **angstroms** (NOT pixels), per
    # ml_optimiser.cpp:9252:
    #     current_changes_optimal_offsets = sqrt(sum_(dx^2+dy^2+dz^2) / (2 * N))
    # where the factor 2 is RELION's per-dim averaging convention (not 3,
    # even when z is included). Set on `update_refinement_state` when the
    # caller provides per-image rotation matrices and translations.
    current_changes_optimal_orientations: float = float("inf")
    current_changes_optimal_offsets_angstrom: float = float("inf")
    current_changes_optimal_classes: float = float("inf")
    # Sticky "smallest thus far" trackers (ml_optimiser.cpp:9282-9285).
    smallest_changes_optimal_orientations: float = SMALLEST_CHANGES_INIT_ORIENTATIONS
    smallest_changes_optimal_offsets_angstrom: float = SMALLEST_CHANGES_INIT_OFFSETS
    smallest_changes_optimal_classes: float = SMALLEST_CHANGES_INIT_CLASSES
    # Counter incremented when current changes meet the "small enough"
    # criterion (within 3% of smallest OR ratio < 0.4 of sampling step).
    # Replaces the older `nr_iter_wo_assignment_changes` for the
    # check_convergence path; the older field is kept for legacy logging.
    nr_iter_wo_large_hidden_variable_changes: int = 0

    def __post_init__(self):
        if self.angular_step == 0.0:
            self.angular_step = healpix_angular_step(self.healpix_order)
        if self.should_do_local_search:
            self.do_local_search = True

    @property
    def effective_step(self) -> float:
        """Effective angular step in degrees (accounting for oversampling)."""
        return effective_angular_step(self.healpix_order, self.adaptive_oversampling)

    @property
    def has_fine_enough_angular_sampling(self) -> bool:
        """True when angular sampling should not be refined further.

        Fires when the per-particle ``acc_rot`` estimate is populated and
        ``effective_step < 0.75 * acc_rot``.

        Mirrors RELION ``ml_optimiser.cpp:9817`` which sets
        ``has_fine_enough_angular_sampling = true`` when the old step is
        already finer than ``0.75 * acc_rot``, so the convergence check
        downstream (``ml_optimiser.cpp:10137``) can proceed.

        ``max_healpix_order`` is a RECOVAR runtime cap, not RELION's
        fine-enough criterion. Hitting that cap must stop further grid
        refinement but must not by itself trigger the final all-data
        iteration, otherwise RECOVAR can terminate earlier than RELION.
        """
        if self.acc_rot < float("inf"):
            if self.effective_step < 0.75 * self.acc_rot:
                return True
        return False

    @property
    def should_do_local_search(self) -> bool:
        """True when HEALPix order is high enough for local search."""
        return self.healpix_order >= LOCAL_SEARCH_HEALPIX_ORDER

    def crowther_angle_step_degrees(self) -> float:
        """Resolution-driven angular step from RELION's Crowther formula.

        Mirrors RELION ``ml_optimiser.cpp:9778``::

            int nr_ang_steps = CEIL(PI * particle_diameter * mymodel.current_resolution)
            myresol_angstep = 360. / nr_ang_steps

        ``mymodel.current_resolution`` is in 1/Å, so

            nr_ang_steps = ceil(pi * particle_diameter[Å] / resolution[Å])

        Returns ``inf`` when ``particle_diameter_angstrom`` or
        ``current_resolution`` are unset, so callers can fall back to the
        legacy "always allow refinement" behavior.

        Used as a per-iter proxy for ``acc_rot``: any angular step finer
        than ``0.75 * crowther_step`` is finer than the resolution-driven
        sampling requirement, so further bumping is unlikely to improve
        anything.  This matches RELION's "stop when old_step <
        0.75 * acc_rot" check (``ml_optimiser.cpp:9817``) for the cases
        where the proper per-particle perturbation acc_rot is dominated
        by the resolution limit (which is most cases at low/mid res).
        """
        pd = float(self.particle_diameter_angstrom)
        res = float(self.current_resolution)
        if pd <= 0.0 or not np.isfinite(res) or res <= 0.0:
            return float("inf")
        nr_ang_steps = int(np.ceil(np.pi * pd / res))
        if nr_ang_steps <= 0:
            return float("inf")
        return 360.0 / float(nr_ang_steps)


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


def relion_angular_distance_per_particle(M_current: np.ndarray, M_previous: np.ndarray) -> np.ndarray:
    """Per-particle RELION-style angular distance between two rotation matrices.

    Implements ``HealpixSampling::calculateAngularDistance`` (see
    ``relion/src/healpix_sampling.cpp:1969-2013``) for the no-symmetry case.
    The distance for one matrix pair is::

        axes_dist = (1/3) * sum_{i=0..2} ACOSD(dot(E1[i,:], E2[i,:]))

    i.e., the mean of the angles between corresponding ROWS of the two
    Euler matrices, in degrees. RELION minimizes over symmetry operators
    when the symmetry group is non-trivial; here we assume C1 (no
    symmetry), matching recovar's current behavior.

    Parameters
    ----------
    M_current, M_previous : np.ndarray, shape ``(N, 3, 3)``
        Per-particle rotation matrices for the current and previous
        iterations. Both must have the same shape.

    Returns
    -------
    np.ndarray, shape ``(N,)``
        Per-particle angular distance in degrees.
    """
    M_current = np.asarray(M_current, dtype=np.float64)
    M_previous = np.asarray(M_previous, dtype=np.float64)
    if M_current.shape != M_previous.shape:
        raise ValueError(
            f"M_current and M_previous must have the same shape, got {M_current.shape} vs {M_previous.shape}"
        )
    if M_current.ndim != 3 or M_current.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (N, 3, 3) rotation matrices, got {M_current.shape}")

    # Per-particle dot product of corresponding rows.
    # einsum 'nij,nij->ni' gives shape (N, 3) where entry [n, i] is the
    # dot product of M_current[n, i, :] and M_previous[n, i, :].
    cos_per_row = np.einsum("nij,nij->ni", M_current, M_previous)
    cos_per_row = np.clip(cos_per_row, -1.0, 1.0)
    angles_deg = np.rad2deg(np.arccos(cos_per_row))  # (N, 3)
    return angles_deg.mean(axis=-1)  # (N,)


def compute_relion_orientation_changes(
    current_rotations: Optional[np.ndarray],
    previous_rotations: Optional[np.ndarray],
) -> float:
    """Mean RELION-style angular distance across all particles, in degrees.

    Returns ``+inf`` when either input is None or shapes don't match,
    matching the convention used elsewhere in this module for "no prior
    iteration to compare against".

    Implements the aggregate part of ``monitorHiddenVariableChanges`` /
    ``updateOverallChangesInHiddenVariables`` (ml_optimiser.cpp:9230 and
    9251)::

        sum_changes_optimal_orientations += angular_distance(...)
        ...
        current_changes_optimal_orientations =
            sum_changes_optimal_orientations / sum_changes_count
    """
    if current_rotations is None or previous_rotations is None:
        return float("inf")
    current_rotations = np.asarray(current_rotations)
    previous_rotations = np.asarray(previous_rotations)
    if current_rotations.shape != previous_rotations.shape:
        return float("inf")
    if current_rotations.size == 0:
        return 0.0
    per_particle = relion_angular_distance_per_particle(current_rotations, previous_rotations)
    return float(np.mean(per_particle))


def compute_relion_offset_changes_angstrom(
    current_translations_pixel: Optional[np.ndarray],
    previous_translations_pixel: Optional[np.ndarray],
    voxel_size: float,
) -> float:
    """RELION-style per-particle offset change in **angstroms**.

    Implements ml_optimiser.cpp:9232 + 9252::

        sum_changes_optimal_offsets +=
            (xoff-old_xoff)^2 + (yoff-old_yoff)^2 + (zoff-old_zoff)^2
        ...
        current_changes_optimal_offsets =
            sqrt(sum_changes_optimal_offsets / (2 * sum_changes_count))

    where xoff/yoff are in **angstroms** (RELION multiplies the metadata
    pixel offsets by ``my_pixel_size`` at line 9222-9225). The factor 2 in
    the denominator is RELION's hard-coded per-dim averaging convention,
    NOT the data dimensionality (it stays 2 even with z != 0). The result
    has units of **angstroms** because the inputs are in angstroms.

    Parameters
    ----------
    current_translations_pixel, previous_translations_pixel : np.ndarray
        Per-particle translations in PIXEL units, shape ``(N, 2)``.
    voxel_size : float
        Pixel size in angstroms (e.g. 4.25 for the 5k benchmark).
    """
    if current_translations_pixel is None or previous_translations_pixel is None:
        return float("inf")
    current_translations_pixel = np.asarray(current_translations_pixel)
    previous_translations_pixel = np.asarray(previous_translations_pixel)
    if current_translations_pixel.shape != previous_translations_pixel.shape:
        return float("inf")
    if current_translations_pixel.size == 0:
        return 0.0
    diffs_pixel = current_translations_pixel - previous_translations_pixel
    diffs_ang = diffs_pixel * float(voxel_size)
    sum_sq_per_particle = np.sum(diffs_ang**2, axis=-1)  # (N,)
    n = sum_sq_per_particle.shape[0]
    return float(np.sqrt(sum_sq_per_particle.sum() / (2.0 * n)))


def calculate_expected_angular_errors(
    healpix_order: int,
    nr_significant_per_image: np.ndarray,
    n_translations: int = 1,
) -> tuple[float, float]:
    """Estimate angular and translational accuracy from the posterior.

    Port of RELION's ``calculateExpectedAngularErrors()``
    (``ml_optimiser.cpp:9534-9564``).  The angular precision is estimated
    from the number of significant orientations per image::

        sigma2_rot  = step^2 / (3 * n_sig)
        sigma2_tilt = step^2 / (3 * n_sig)
        sigma2_psi  = step^2 / n_sig
        acc_rot = RAD2DEG(sqrt(sigma2_rot + sigma2_tilt))

    where ``step = DEG2RAD(healpix_angular_step(order))``.

    Parameters
    ----------
    healpix_order : int
        Current HEALPix order (base, NOT oversampled).
    nr_significant_per_image : np.ndarray, shape (n_images,)
        Number of significant orientation samples per image.
    n_translations : int
        Number of translations in the grid (used to factor out
        translations from n_significant).

    Returns
    -------
    acc_rot : float
        Estimated angular accuracy in degrees.
    acc_trans : float
        Placeholder (returns inf — proper acc_trans requires
        translation-specific significant counts).
    """
    nr_sig = np.asarray(nr_significant_per_image, dtype=np.float64)
    nr_sig_rot = nr_sig / max(1, n_translations)
    nr_sig_rot = np.maximum(nr_sig_rot, 1.0)
    step_rad = np.deg2rad(healpix_angular_step(healpix_order))
    sigma2_per_image = 2.0 * step_rad**2 / (3.0 * nr_sig_rot)
    acc_rot_per_image = np.rad2deg(np.sqrt(sigma2_per_image))
    acc_rot = float(np.mean(acc_rot_per_image))
    return acc_rot, float("inf")


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

    Implements ml_optimiser.cpp:10135-10204 ``MlOptimiser::checkConvergence``.
    Convergence requires ALL of:
    1. RELION fine-enough angular sampling
    2. Resolution stalled for >= MAX_NR_ITER_WO_RESOL_GAIN iterations
    3. Hidden-variable changes stable for >=
       MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES iterations.

    The third condition uses the RELION-exact ``nr_iter_wo_large_hidden_variable_changes``
    counter (B4) when available; falls back to the legacy
    ``nr_iter_wo_assignment_changes`` if the per-particle change tracking
    has never been populated (e.g. very early iterations or callers that
    haven't passed rotation matrices to ``update_refinement_state``).

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

    # Prefer the RELION-exact counter when populated; fall back to the
    # legacy fraction-based counter for backwards compatibility.
    relion_changes_seen = state.smallest_changes_optimal_orientations < SMALLEST_CHANGES_INIT_ORIENTATIONS
    if relion_changes_seen:
        if state.nr_iter_wo_large_hidden_variable_changes < MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES:
            return False
    else:
        if state.nr_iter_wo_assignment_changes < MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES:
            return False

    return True


# ---------------------------------------------------------------------------
# Angular step refinement
# ---------------------------------------------------------------------------


def should_refine_angular_sampling(state: RefinementState) -> bool:
    """Check whether angular sampling should be refined (HEALPix order incremented).

    Mirrors RELION ``MlOptimiser::updateAngularSampling`` at
    ``ml_optimiser.cpp:9772-9790``:

    .. code-block:: cpp

        do_proceed_resolution = nr_iter_wo_resol_gain >= MAX_NR_ITER_WO_RESOL_GAIN;
        do_proceed_hidden_variables = nr_iter_wo_large_hidden_variable_changes
                                      >= MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES;
        if (do_proceed_resolution && do_proceed_hidden_variables) { bump }

    Refinement triggers when:
    1. Resolution stalled for >= MAX_NR_ITER_WO_RESOL_GAIN iterations
    2. The RELION-exact ``nr_iter_wo_large_hidden_variable_changes`` counter
       (B3+B4) >= MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES.  When that
       counter has not been populated yet (very early iterations or callers
       that don't pass rotation matrices to ``update_refinement_state``),
       fall back to the legacy ``nr_iter_wo_assignment_changes``.
    3. Current angular step is NOT already finer than 75% of acc_rot
    4. ``healpix_order`` is below the RECOVAR hard cap. The cap only prevents
       runaway grid growth; it is not a RELION convergence criterion.

    Parameters
    ----------
    state : RefinementState
        Current refinement state.

    Returns
    -------
    bool
        True if angular sampling should be refined.
    """
    # RELION fine-enough sampling: do not refine further. This may allow
    # convergence downstream once the stall counters are also satisfied.
    if state.has_fine_enough_angular_sampling:
        return False

    # RECOVAR runtime cap: do not refine beyond it, but do not report
    # fine-enough/converged just because the cap was reached.
    if state.healpix_order >= state.max_healpix_order:
        logger.info(
            "Angular sampling reached max_healpix_order=%d; not refining further "
            "(RELION fine-enough criterion not yet met)",
            state.max_healpix_order,
        )
        return False

    # Resolution stalls are required.  RELION uses the strict
    # `nr_iter_wo_resol_gain >= MAX_NR_ITER_WO_RESOL_GAIN` check; we keep
    # the same threshold (==1).
    if state.nr_iter_wo_resol_gain < MAX_NR_ITER_WO_RESOL_GAIN:
        return False

    # Hidden-variable stalls: prefer the RELION-exact B3+B4 counter
    # (`nr_iter_wo_large_hidden_variable_changes`) when available.  Fall
    # back to the legacy `nr_iter_wo_assignment_changes` plus the extended
    # resol-stall escape hatch only when B3+B4 hasn't been populated.
    relion_changes_seen = state.smallest_changes_optimal_orientations < SMALLEST_CHANGES_INIT_ORIENTATIONS
    if relion_changes_seen:
        if state.nr_iter_wo_large_hidden_variable_changes < MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES:
            return False
    else:
        EXTENDED_RESOL_STALL = 5
        if (
            state.nr_iter_wo_assignment_changes < MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES
            and state.nr_iter_wo_resol_gain < EXTENDED_RESOL_STALL
        ):
            return False

    # Don't refine beyond 75% of estimated angular accuracy.
    if state.acc_rot < float("inf"):
        if state.effective_step < 0.75 * state.acc_rot:
            logger.info(
                "Angular step %.2f deg < 75%% of acc_rot %.2f deg; not refining further",
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
        new_trans_step = min(1.5, 0.75 * state.acc_trans) * (2**state.adaptive_oversampling)
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
        step_rad = np.deg2rad(new_angular_step / (2**state.adaptive_oversampling))
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

    # Build new state -- reset stall counters AND the RELION-exact
    # smallest-changes trackers (matches ml_optimiser.cpp:9919-9922 in
    # `updateAngularSampling`: when sampling refines, RELION resets both
    # the stall counters and the sticky smallest_changes_optimal_*
    # baselines).
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
        particle_diameter_angstrom=state.particle_diameter_angstrom,
        max_healpix_order=state.max_healpix_order,
        fraction_changed=state.fraction_changed,
        changes_optimal_offsets=state.changes_optimal_offsets,
        # RELION-exact reset (B4):
        current_changes_optimal_orientations=float("inf"),
        current_changes_optimal_offsets_angstrom=float("inf"),
        current_changes_optimal_classes=float("inf"),
        smallest_changes_optimal_orientations=SMALLEST_CHANGES_INIT_ORIENTATIONS,
        smallest_changes_optimal_offsets_angstrom=SMALLEST_CHANGES_INIT_OFFSETS,
        smallest_changes_optimal_classes=SMALLEST_CHANGES_INIT_CLASSES,
        nr_iter_wo_large_hidden_variable_changes=0,
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
    *,
    current_rotation_matrices: Optional[np.ndarray] = None,
    previous_rotation_matrices: Optional[np.ndarray] = None,
    current_translations_pixel: Optional[np.ndarray] = None,
    previous_translations_pixel: Optional[np.ndarray] = None,
    voxel_size_angstrom: float = 1.0,
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
    current_rotation_matrices, previous_rotation_matrices : np.ndarray, optional
        Per-particle rotation matrices for the current and previous
        iterations, shape ``(n_images, 3, 3)``. When both provided, the
        RELION-exact ``current_changes_optimal_orientations`` is computed
        and used in the new check_convergence path. The legacy
        ``frac_changed`` is still computed from the integer assignments.
    current_translations_pixel, previous_translations_pixel : np.ndarray, optional
        Per-particle translation vectors in PIXEL units, shape ``(n_images, 2)``.
        When both provided, the RELION-exact
        ``current_changes_optimal_offsets_angstrom`` is computed.
    voxel_size_angstrom : float, default 1.0
        Pixel size in angstroms. Required for the offset metric.

    Returns
    -------
    RefinementState
        Updated state reflecting this iteration's results.
    """
    # --- Compute assignment changes (legacy path, still used by logging) ---
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

    # --- Compute RELION-exact change tracking (B3) ---
    current_changes_orientations = compute_relion_orientation_changes(
        current_rotation_matrices,
        previous_rotation_matrices,
    )
    current_changes_offsets_angstrom = compute_relion_offset_changes_angstrom(
        current_translations_pixel,
        previous_translations_pixel,
        voxel_size_angstrom,
    )
    # Single-class refine: classes never change.
    current_changes_classes = 0.0

    # --- Compute Pmax ---
    ave_pmax = state.ave_Pmax
    if max_posterior_per_image is not None:
        ave_pmax = compute_ave_Pmax(max_posterior_per_image)

    # --- Resolution stall detection ---
    # ``new_resolution`` and ``state.current_resolution`` are in Angstroms
    # (lower = better resolution).  Matches RELION
    # MlOptimiser::updateCurrentResolution at ml_optimiser.cpp:5658-5663:
    #
    #   if (newres <= mymodel.current_resolution+0.0001) // Å, lower is better
    #       nr_iter_wo_resol_gain_sum_bodies++;
    #   else
    #       nr_iter_wo_resol_gain = 0;
    resol_improved = new_resolution < state.current_resolution
    if resol_improved:
        nr_iter_wo_resol_gain = 0
    else:
        nr_iter_wo_resol_gain = state.nr_iter_wo_resol_gain + 1

    # --- Assignment stability detection (legacy) ---
    # "Large changes" threshold: fraction_changed > 0 means some images changed.
    # RELION considers assignments "stable" when fraction_changed is small.
    # We use fraction_changed < 0.01 (1%) as "no large changes".
    ASSIGNMENT_CHANGE_THRESHOLD = 0.01
    if frac_changed < ASSIGNMENT_CHANGE_THRESHOLD:
        nr_iter_wo_assignment_changes = state.nr_iter_wo_assignment_changes + 1
    else:
        nr_iter_wo_assignment_changes = 0

    # --- RELION-exact "smallest changes thus far" + counter (B4) ---
    # ml_optimiser.cpp:9267-9285. The counter increments when:
    #   (1) classes are within 3% of smallest (always true for single-class
    #       once smallest_classes = 0), AND
    #   (2) translations are EITHER (small relative to sampling step,
    #       ratio < 0.4) OR (within 3% of smallest), AND
    #   (3) orientations are EITHER (small relative to sampling step,
    #       ratio < 0.4) OR (within 3% of smallest).
    # When the inputs are missing (early iters before per-particle data is
    # collected), we conservatively skip the relion counter and fall back
    # to the legacy nr_iter_wo_assignment_changes path.
    nr_iter_wo_large_hidden_variable_changes = state.nr_iter_wo_large_hidden_variable_changes
    smallest_orient = state.smallest_changes_optimal_orientations
    smallest_offsets = state.smallest_changes_optimal_offsets_angstrom
    smallest_classes = state.smallest_changes_optimal_classes
    if np.isfinite(current_changes_orientations) and np.isfinite(current_changes_offsets_angstrom):
        # Sampling steps used as the "small enough" denominator. RELION uses
        # the ANGULAR SAMPLING STEP (in degrees, after oversampling) for the
        # orientation ratio and the TRANSLATION SAMPLING STEP (in pixels)
        # for the offset ratio. Convert offsets to pixels for the ratio.
        rot_step_deg = effective_angular_step(
            state.healpix_order,
            state.adaptive_oversampling,
        )
        # offset RMS is in angstroms; convert to pixels for the ratio
        # comparison against the translation sampling step (also in pixels).
        if voxel_size_angstrom > 0:
            offsets_pixels = current_changes_offsets_angstrom / voxel_size_angstrom
        else:
            offsets_pixels = current_changes_offsets_angstrom
        trans_step = state.translation_step
        ratio_orient_changes = current_changes_orientations / rot_step_deg if rot_step_deg > 0 else float("inf")
        ratio_trans_changes = offsets_pixels / trans_step if trans_step > 0 else float("inf")

        class_ok = 1.03 * current_changes_classes >= smallest_classes
        trans_ok = ratio_trans_changes < 0.40 or 1.03 * current_changes_offsets_angstrom >= smallest_offsets
        rot_ok = ratio_orient_changes < 0.40 or 1.03 * current_changes_orientations >= smallest_orient

        if class_ok and trans_ok and rot_ok:
            nr_iter_wo_large_hidden_variable_changes += 1
        else:
            nr_iter_wo_large_hidden_variable_changes = 0

        # Update sticky smallest trackers AFTER the counter increment.
        if current_changes_orientations < smallest_orient:
            smallest_orient = current_changes_orientations
        if current_changes_offsets_angstrom < smallest_offsets:
            smallest_offsets = current_changes_offsets_angstrom
        if current_changes_classes < smallest_classes:
            smallest_classes = round(current_changes_classes)

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
        particle_diameter_angstrom=state.particle_diameter_angstrom,
        max_healpix_order=state.max_healpix_order,
        fraction_changed=frac_changed,
        changes_optimal_offsets=trans_changes,
        current_changes_optimal_orientations=current_changes_orientations,
        current_changes_optimal_offsets_angstrom=current_changes_offsets_angstrom,
        current_changes_optimal_classes=current_changes_classes,
        smallest_changes_optimal_orientations=smallest_orient,
        smallest_changes_optimal_offsets_angstrom=smallest_offsets,
        smallest_changes_optimal_classes=smallest_classes,
        nr_iter_wo_large_hidden_variable_changes=nr_iter_wo_large_hidden_variable_changes,
    )

    logger.info(
        "Iteration %d: frac_changed=%.4f, resol=%.2f (prev=%.2f), "
        "stalls: resol=%d, assign=%d, hvc=%d, ave_Pmax=%.4f, "
        "Δrot=%.3f deg, Δtrans=%.3f Å",
        updated.iteration,
        frac_changed,
        new_resolution,
        state.current_resolution,
        nr_iter_wo_resol_gain,
        nr_iter_wo_assignment_changes,
        nr_iter_wo_large_hidden_variable_changes,
        ave_pmax,
        current_changes_orientations if np.isfinite(current_changes_orientations) else float("nan"),
        current_changes_offsets_angstrom if np.isfinite(current_changes_offsets_angstrom) else float("nan"),
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
            "resolution stalled for %d iter, hvc stable for %d iter, "
            "RELION fine-enough angular sampling reached (order %d)",
            updated.iteration,
            nr_iter_wo_resol_gain,
            nr_iter_wo_large_hidden_variable_changes,
            updated.healpix_order,
        )

    return updated
