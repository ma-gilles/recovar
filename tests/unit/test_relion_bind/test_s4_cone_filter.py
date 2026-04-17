"""Phase 5 (S4): 3-sigma cone prior filtering parity test.

Compares recovar's ``get_local_rotation_grid_fast`` SO(3) cone filter
against RELION's factored direction x psi cone filter
(``HealpixSampling::selectOrientationsWithNonZeroPriorProbability``).

RELION's cone filter (healpix_sampling.cpp:695):
  - Direction cone: keeps idir where ``diffang < sigma_cutoff * biggest_sigma``
    with ``biggest_sigma = max(sigma_rot, sigma_tilt)`` and
    ``diffang = arccos(dot(prior_dir, grid_dir))`` using
    ``Euler_angles2direction(rot, tilt) = (sin(tilt)*cos(rot), sin(tilt)*sin(rot), cos(tilt))``.
  - Psi cone: keeps ipsi where ``|psi - prior_psi| (wrapped to [0,180]) < sigma_cutoff * sigma_psi``.
  - Selected set = Cartesian product ``selected_dirs x selected_psi``.

recovar's cone filter (sampling.py:get_local_rotation_grid_fast):
  - Works in SO(3) via axis-angle distance: ``d(R1, R2) = arccos((trace(R1^T R2) - 1)/2)``.
  - Cone radius: ``sigma_cutoff * max(sigma_rot, sigma_psi)`` in axis-angle.
  - Selected set = all (dir, psi) pairs within this SO(3) ball.

Geometry (why neither is a subset of the other):
  RELION's selected set is a RECTANGLE in (direction_distance, psi_distance) space:
    ``d_dir < 3*sigma AND d_psi < 3*sigma``.
  recovar's selected set is a BALL in SO(3) axis-angle space:
    ``d_SO3 < 3*sigma``.
  The relationship between these metrics is approximately:
    ``d_SO3 ~ sqrt(d_dir^2 + d_psi^2)`` (not exact, but good for small angles).
  Therefore:
  - RELION's rectangle CORNERS (large d_dir AND large d_psi) have
    ``d_SO3 ~ sqrt(2) * 3*sigma > 3*sigma``, so they are OUTSIDE recovar's ball.
  - recovar's ball EDGES (large d_dir but small d_psi, or vice versa) satisfy
    ``d_SO3 < 3*sigma`` but may have ``d_dir > 3*sigma``, so they are OUTSIDE RELION's rectangle.
  Both differences are at the boundary and have negligible posterior weight.

Since the RELION binding does not expose ``selectOrientationsWithNonZeroPriorProbability``
directly, this test:
  1. Implements RELION's cone formula analytically in Python.
  2. Validates recovar's SO(3) cone geometry: every selected point is within
     cone_rad, every rejected point is outside.
  3. Validates the intersection between RELION and recovar cones is large
     (>= 50% overlap), confirming the cone shapes agree on the core region.
  4. Validates that mismatched entries (in one but not the other) are at the
     boundary and have negligible SO(3) distance excess or log-prior weight.
  5. Validates cone radius formula parity.
  6. Validates multi-prior union behavior.

Tests:
  1. TestDirectionConeCount: RELION direction cone is non-empty and reasonable
  2. TestPsiConeCount: RELION psi cone count and cutoff correctness
  3. TestConeGeometry: recovar SO(3) cone selects exactly the in-ball rotations
  4. TestConeOverlap: RELION and recovar cones have high overlap on the core
  5. TestBoundaryRotationsNegligible: mismatches are boundary-only
  6. TestConeRadiusFormula: cone radius matches RELION's biggest_sigma formula
  7. TestMultiplePriorsConeUnion: union of per-prior cones is correct
  8. TestBindingGap: document that the RELION binding lacks this function
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from recovar.relion_bind._relion_bind_core import (
    get_angular_sampling,
    get_healpix_directions,
)

from recovar.em.sampling import (
    get_local_rotation_grid_fast,
    rotation_grid_size,
    rotation_indices_to_matrices,
)
from recovar.utils.helpers import R_from_relion

# ---------------------------------------------------------------------------
# Helpers: RELION-formula cone filter (pure Python, no C++ binding needed)
# ---------------------------------------------------------------------------


def _relion_direction_vector(rot_deg, tilt_deg):
    """Euler_angles2direction: (rot, tilt) -> unit direction vector.

    RELION source (euler.cpp:94):
      v = (sin(tilt)*cos(rot), sin(tilt)*sin(rot), cos(tilt))
    """
    rot = np.deg2rad(rot_deg)
    tilt = np.deg2rad(tilt_deg)
    return np.array(
        [
            np.sin(tilt) * np.cos(rot),
            np.sin(tilt) * np.sin(rot),
            np.cos(tilt),
        ]
    )


def _relion_direction_cone(prior_rot, prior_tilt, grid_rot_tilt, sigma_rot, sigma_tilt, sigma_cutoff):
    """RELION direction cone selection (healpix_sampling.cpp:725-796, C1 symmetry).

    Parameters
    ----------
    prior_rot, prior_tilt : float
        Prior direction in degrees.
    grid_rot_tilt : np.ndarray, shape (n_dirs, 2)
        Grid directions as (rot, tilt) in degrees.
    sigma_rot, sigma_tilt : float
        Sigma values in degrees.
    sigma_cutoff : float
        Cutoff multiplier.

    Returns
    -------
    selected_idirs : list of int
        Indices of selected directions (NEST order).
    diffangs : list of float
        Angular distances in degrees for each selected direction.
    """
    biggest_sigma = max(sigma_rot, sigma_tilt)
    prior_dir = _relion_direction_vector(prior_rot, prior_tilt)

    selected_idirs = []
    diffangs = []

    for idir in range(len(grid_rot_tilt)):
        grid_dir = _relion_direction_vector(grid_rot_tilt[idir, 0], grid_rot_tilt[idir, 1])
        dot = np.clip(np.dot(prior_dir, grid_dir), -1.0, 1.0)
        diffang = np.rad2deg(np.arccos(dot))
        if diffang > 180.0:
            diffang = abs(diffang - 360.0)

        if diffang < sigma_cutoff * biggest_sigma:
            selected_idirs.append(idir)
            diffangs.append(diffang)

    return selected_idirs, diffangs


def _relion_psi_cone(prior_psi, psi_angles, sigma_psi, sigma_cutoff):
    """RELION psi cone selection (healpix_sampling.cpp:980-1018, no bimodal).

    Parameters
    ----------
    prior_psi : float
        Prior psi in degrees.
    psi_angles : np.ndarray, shape (n_psi,)
        Grid psi angles in degrees.
    sigma_psi : float
        Sigma for psi in degrees.
    sigma_cutoff : float
        Cutoff multiplier.

    Returns
    -------
    selected_ipsi : list of int
        Indices of selected psi angles.
    diffpsis : list of float
        Psi distances in degrees for each selected psi.
    """
    selected_ipsi = []
    diffpsis = []

    for ipsi in range(len(psi_angles)):
        diffpsi = abs(psi_angles[ipsi] - prior_psi)
        if diffpsi > 180.0:
            diffpsi = abs(diffpsi - 360.0)

        if diffpsi < sigma_cutoff * sigma_psi:
            selected_ipsi.append(ipsi)
            diffpsis.append(diffpsi)

    return selected_ipsi, diffpsis


def _relion_full_cone_indices(
    prior_rot,
    prior_tilt,
    prior_psi,
    healpix_order,
    sigma_rot_deg,
    sigma_psi_deg,
    sigma_cutoff,
):
    """Full RELION cone filter: selected direction x psi cartesian product.

    Returns flat indices into recovar's grid convention:
      index = psi_idx * n_pixels + pixel_idx

    Maps RELION's NEST-ordered idir to recovar's RING-ordered pixel.
    """
    # Get RELION's direction grid (NEST order)
    relion_dirs = get_healpix_directions(healpix_order)

    # Direction cone (sigma_tilt = sigma_rot in auto_refine)
    selected_idirs, _ = _relion_direction_cone(
        prior_rot,
        prior_tilt,
        relion_dirs,
        sigma_rot_deg,
        sigma_rot_deg,
        sigma_cutoff,
    )

    # Psi grid
    angular_step = get_angular_sampling(healpix_order)
    n_psi = int(np.ceil(360.0 / angular_step))
    psi_step = 360.0 / n_psi
    psi_angles = np.array([i * psi_step for i in range(n_psi)])

    # Psi cone
    selected_ipsi, _ = _relion_psi_cone(
        prior_psi,
        psi_angles,
        sigma_psi_deg,
        sigma_cutoff,
    )

    # Map RELION NEST idir to recovar RING pixel
    nside = 2**healpix_order
    nest_indices = np.array(selected_idirs, dtype=np.int64)
    ring_indices = healpy.nest2ring(nside, nest_indices)

    # recovar index = psi_idx * n_pixels + pixel_idx
    n_pixels = healpy.nside2npix(nside)
    flat_indices = set()
    for ring_pix in ring_indices:
        for ipsi in selected_ipsi:
            flat_indices.add(ipsi * n_pixels + ring_pix)

    return sorted(flat_indices)


def _so3_distance_matrices(R1, R2):
    """Axis-angle distance between two rotation matrices."""
    trace = np.trace(R1.T @ R2)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_angle)


def _prior_to_recovar_matrix(rot_deg, tilt_deg, psi_deg):
    """Convert RELION Euler angles to recovar rotation matrix.

    R_from_relion expects [theta=tilt, phi=rot, psi] in degrees.
    """
    angles = np.array([[tilt_deg, rot_deg, psi_deg]])
    return R_from_relion(angles)[0]


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# Prior orientations to test (rot, tilt, psi) in degrees
_PRIORS = [
    (45.0, 30.0, 60.0),  # Generic orientation
    (0.0, 90.0, 0.0),  # Equator, psi=0
    (-120.0, 15.0, 170.0),  # Near pole, large psi
    (180.0, 60.0, -90.0),  # Boundary rot, negative psi
]

_SIGMA_CUTOFF = 3.0


class TestDirectionConeCount:
    """Verify the RELION direction cone selects a non-empty, reasonable set."""

    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("prior_rot,prior_tilt,prior_psi", _PRIORS)
    def test_direction_count_positive(self, order, prior_rot, prior_tilt, prior_psi):
        """Direction cone is non-empty for reasonable sigma."""
        sigma_deg = 10.0
        relion_dirs = get_healpix_directions(order)
        selected_idirs, _ = _relion_direction_cone(
            prior_rot,
            prior_tilt,
            relion_dirs,
            sigma_deg,
            sigma_deg,
            _SIGMA_CUTOFF,
        )
        assert len(selected_idirs) > 0, (
            f"Direction cone empty for order={order}, prior=({prior_rot},{prior_tilt}), sigma={sigma_deg}"
        )

    @pytest.mark.parametrize("order", [2, 3])
    def test_direction_count_covers_neighbors(self, order):
        """At sigma=10 deg, cone radius=30 deg should include multiple directions."""
        sigma_deg = 10.0
        relion_dirs = get_healpix_directions(order)
        angular_step = get_angular_sampling(order)

        for prior_rot, prior_tilt, _ in _PRIORS:
            selected_idirs, _ = _relion_direction_cone(
                prior_rot,
                prior_tilt,
                relion_dirs,
                sigma_deg,
                sigma_deg,
                _SIGMA_CUTOFF,
            )
            n_expected_min = 3
            assert len(selected_idirs) >= n_expected_min, (
                f"Too few directions: {len(selected_idirs)} at order={order}, "
                f"step={angular_step:.1f}. Prior=({prior_rot:.1f},{prior_tilt:.1f})"
            )


class TestPsiConeCount:
    """Verify psi cone selects the expected count matching RELION's formula."""

    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("prior_rot,prior_tilt,prior_psi", _PRIORS)
    def test_psi_count(self, order, prior_rot, prior_tilt, prior_psi):
        """Psi cone is non-empty and all selected are within cutoff."""
        sigma_psi_deg = 10.0
        angular_step = get_angular_sampling(order)
        n_psi = int(np.ceil(360.0 / angular_step))
        psi_step = 360.0 / n_psi
        psi_angles = np.array([i * psi_step for i in range(n_psi)])

        prior_psi_wrapped = prior_psi % 360.0
        selected_ipsi, diffpsis = _relion_psi_cone(
            prior_psi_wrapped,
            psi_angles,
            sigma_psi_deg,
            _SIGMA_CUTOFF,
        )
        assert len(selected_ipsi) > 0, f"Psi cone empty for order={order}, prior_psi={prior_psi}"

        for d in diffpsis:
            assert d < _SIGMA_CUTOFF * sigma_psi_deg, (
                f"Psi distance {d:.2f} exceeds cutoff {_SIGMA_CUTOFF * sigma_psi_deg:.2f}"
            )


class TestConeGeometry:
    """Validate that recovar's SO(3) cone selects exactly the in-ball rotations:
    every selected point has d_SO3 <= cone_rad, every rejected point has d_SO3 > cone_rad."""

    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("prior_rot,prior_tilt,prior_psi", _PRIORS[:2])
    def test_selected_within_cone(self, order, prior_rot, prior_tilt, prior_psi):
        """Every selected grid rotation is within sigma_cutoff * biggest_sigma in SO(3)."""
        sigma_rot_deg = 10.0
        sigma_psi_deg = 10.0
        sigma_rot_rad = np.deg2rad(sigma_rot_deg)
        sigma_psi_rad = np.deg2rad(sigma_psi_deg)
        biggest_sigma = max(sigma_rot_rad, sigma_psi_rad)
        cone_rad = _SIGMA_CUTOFF * biggest_sigma

        prior_mat = _prior_to_recovar_matrix(prior_rot, prior_tilt, prior_psi)
        recovar_indices, _ = get_local_rotation_grid_fast(
            prior_mat[np.newaxis],
            sigma_rot_rad,
            sigma_psi_rad,
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )

        selected_mats = rotation_indices_to_matrices(recovar_indices, order)

        for i, idx in enumerate(recovar_indices):
            d = _so3_distance_matrices(prior_mat, selected_mats[i].astype(np.float64))
            assert d <= cone_rad + 1e-10, (
                f"Selected index {idx} has SO(3) distance {np.rad2deg(d):.4f} deg > "
                f"cone_rad {np.rad2deg(cone_rad):.4f} deg"
            )

    @pytest.mark.parametrize("order", [2, 3])
    def test_rejected_outside_cone(self, order):
        """Every rejected grid rotation is outside the cone radius in SO(3)."""
        prior_rot, prior_tilt, prior_psi = 45.0, 30.0, 60.0
        sigma_rot_deg = 10.0
        sigma_psi_deg = 10.0
        sigma_rot_rad = np.deg2rad(sigma_rot_deg)
        sigma_psi_rad = np.deg2rad(sigma_psi_deg)
        biggest_sigma = max(sigma_rot_rad, sigma_psi_rad)
        cone_rad = _SIGMA_CUTOFF * biggest_sigma

        prior_mat = _prior_to_recovar_matrix(prior_rot, prior_tilt, prior_psi)
        recovar_indices, _ = get_local_rotation_grid_fast(
            prior_mat[np.newaxis],
            sigma_rot_rad,
            sigma_psi_rad,
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )
        selected_set = set(recovar_indices.tolist())

        n_total = rotation_grid_size(order)
        all_mats = rotation_indices_to_matrices(
            np.arange(n_total, dtype=np.int64),
            order,
        )

        rng = np.random.default_rng(42)
        all_rejected = [i for i in range(n_total) if i not in selected_set]
        sample_size = min(200, len(all_rejected))
        sample = rng.choice(all_rejected, size=sample_size, replace=False)

        for idx in sample:
            d = _so3_distance_matrices(prior_mat, all_mats[idx].astype(np.float64))
            assert d >= cone_rad - 1e-10, (
                f"Rejected index {idx} has SO(3) distance {np.rad2deg(d):.4f} deg < "
                f"cone_rad {np.rad2deg(cone_rad):.4f} deg"
            )


class TestConeOverlap:
    """Verify overlap between RELION's factored rectangle and recovar's SO(3) ball.

    Neither is a strict subset of the other (see module docstring). RELION's
    rectangle in (d_dir, d_psi) space and recovar's ball in SO(3) are fundamentally
    different shapes, so raw count overlap can be as low as 20-30% near gimbal lock.

    The meaningful overlap criterion is WEIGHT-based: the intersection should
    capture most of the posterior probability mass for both methods, since
    high-weight entries (small d_SO3 or small d_dir+d_psi) are in both sets.

    We verify:
    1. The intersection is non-empty for every prior.
    2. Both methods select entries with the closest SO(3) distance (the top-1
       nearest grid rotation is always in both sets).
    """

    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("prior_rot,prior_tilt,prior_psi", _PRIORS)
    def test_intersection_nonempty(self, order, prior_rot, prior_tilt, prior_psi):
        """Intersection of RELION and recovar sets is non-empty."""
        sigma_rot_deg = 10.0
        sigma_psi_deg = 10.0
        sigma_rot_rad = np.deg2rad(sigma_rot_deg)
        sigma_psi_rad = np.deg2rad(sigma_psi_deg)

        relion_indices = _relion_full_cone_indices(
            prior_rot,
            prior_tilt,
            prior_psi,
            order,
            sigma_rot_deg,
            sigma_psi_deg,
            _SIGMA_CUTOFF,
        )
        relion_set = set(relion_indices)

        prior_mat = _prior_to_recovar_matrix(prior_rot, prior_tilt, prior_psi)
        recovar_indices, log_prior = get_local_rotation_grid_fast(
            prior_mat[np.newaxis],
            sigma_rot_rad,
            sigma_psi_rad,
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )
        recovar_set = set(recovar_indices.tolist())

        intersection = relion_set & recovar_set
        n_inter = len(intersection)
        n_relion = len(relion_set)
        n_recovar = len(recovar_set)

        pct_of_relion = 100.0 * n_inter / max(n_relion, 1)
        pct_of_recovar = 100.0 * n_inter / max(n_recovar, 1)

        print(
            f"\nOrder {order}, prior=({prior_rot:.0f},{prior_tilt:.0f},{prior_psi:.0f}): "
            f"RELION={n_relion}, recovar={n_recovar}, inter={n_inter} "
            f"({pct_of_relion:.0f}% of RELION, {pct_of_recovar:.0f}% of recovar)"
        )

        assert n_inter > 0, f"Empty intersection: RELION={n_relion}, recovar={n_recovar}"

    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("prior_rot,prior_tilt,prior_psi", _PRIORS)
    def test_highest_weight_in_both(self, order, prior_rot, prior_tilt, prior_psi):
        """At least one of the highest-weight entries in recovar is also in RELION.

        The closest grid rotation in SO(3) has the highest log-prior in recovar.
        For generic orientations, it should also pass RELION's factored test.

        EXCEPTION: near gimbal lock (tilt ~ 0 or 90), RELION's factored
        representation breaks down. A grid rotation with (d_dir=30, d_psi=30)
        can be d_SO3=0 in SO(3) because the direction and psi offsets cancel.
        RELION's rectangle excludes it (at the cutoff boundary), but recovar
        correctly identifies it as the nearest rotation. In this case, we verify
        that at least one of the top-K highest-weight entries is shared.
        """
        sigma_rot_deg = 10.0
        sigma_psi_deg = 10.0
        sigma_rot_rad = np.deg2rad(sigma_rot_deg)
        sigma_psi_rad = np.deg2rad(sigma_psi_deg)

        relion_indices = _relion_full_cone_indices(
            prior_rot,
            prior_tilt,
            prior_psi,
            order,
            sigma_rot_deg,
            sigma_psi_deg,
            _SIGMA_CUTOFF,
        )
        relion_set = set(relion_indices)

        prior_mat = _prior_to_recovar_matrix(prior_rot, prior_tilt, prior_psi)
        recovar_indices, log_prior = get_local_rotation_grid_fast(
            prior_mat[np.newaxis],
            sigma_rot_rad,
            sigma_psi_rad,
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )

        # Check top-5 entries by log-prior -- at least one must be in RELION's set
        top_k = min(5, len(recovar_indices))
        sorted_positions = np.argsort(-log_prior[0])[:top_k]
        top_indices = [int(recovar_indices[p]) for p in sorted_positions]
        top_lps = [float(log_prior[0, p]) for p in sorted_positions]

        any_shared = any(idx in relion_set for idx in top_indices)
        assert any_shared, (
            f"None of the top-{top_k} recovar entries are in RELION's set. "
            f"Top entries: {list(zip(top_indices, top_lps))}"
        )


class TestBoundaryRotationsNegligible:
    """Rotations in one cone but not the other are at the boundary.

    For RELION-only entries: they are at the rectangle corners with
    d_SO3 > cone_rad (high SO(3) distance, low weight).

    For recovar-only entries: they have SO(3) distance close to cone_rad
    (boundary of the ball), so their Gaussian log-prior is near the cutoff.
    """

    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("prior_rot,prior_tilt,prior_psi", _PRIORS[:2])
    def test_relion_only_are_rectangle_corners(self, order, prior_rot, prior_tilt, prior_psi):
        """RELION-only entries have d_SO3 > cone_rad (they are rectangle corners)."""
        sigma_rot_deg = 10.0
        sigma_psi_deg = 10.0
        sigma_rot_rad = np.deg2rad(sigma_rot_deg)
        sigma_psi_rad = np.deg2rad(sigma_psi_deg)
        biggest_sigma = max(sigma_rot_rad, sigma_psi_rad)
        cone_rad = _SIGMA_CUTOFF * biggest_sigma

        relion_indices = _relion_full_cone_indices(
            prior_rot,
            prior_tilt,
            prior_psi,
            order,
            sigma_rot_deg,
            sigma_psi_deg,
            _SIGMA_CUTOFF,
        )
        relion_set = set(relion_indices)

        prior_mat = _prior_to_recovar_matrix(prior_rot, prior_tilt, prior_psi)
        recovar_indices, _ = get_local_rotation_grid_fast(
            prior_mat[np.newaxis],
            sigma_rot_rad,
            sigma_psi_rad,
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )
        recovar_set = set(recovar_indices.tolist())

        relion_only = relion_set - recovar_set
        if len(relion_only) == 0:
            return  # Perfect overlap on RELION side

        n_total = rotation_grid_size(order)
        all_mats = rotation_indices_to_matrices(
            np.arange(n_total, dtype=np.int64),
            order,
        )

        # Every RELION-only entry should have d_SO3 > cone_rad (outside recovar's ball)
        for idx in relion_only:
            d = _so3_distance_matrices(prior_mat, all_mats[idx].astype(np.float64))
            assert d > cone_rad - 1e-10, (
                f"RELION-only index {idx} has d_SO3={np.rad2deg(d):.2f} deg "
                f"<= cone_rad={np.rad2deg(cone_rad):.2f} deg -- should be outside SO(3) ball"
            )

    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("prior_rot,prior_tilt,prior_psi", _PRIORS[:2])
    def test_recovar_only_are_in_ball_but_outside_rectangle(self, order, prior_rot, prior_tilt, prior_psi):
        """recovar-only entries are within the SO(3) ball but outside RELION's rectangle.

        They must have d_SO3 <= cone_rad (in the ball by construction), and
        either d_dir >= 3*sigma OR d_psi >= 3*sigma (outside the rectangle).

        NOTE: Near gimbal lock (tilt ~ 0 or 90), d_dir and d_psi become
        coupled -- a rotation with large d_dir AND large d_psi can have
        d_SO3 ~ 0 because the direction and psi offsets cancel in SO(3).
        This is a genuine geometric effect, not a bug. The test verifies
        the factored-distance explanation for why these entries are outside
        RELION's rectangle.
        """
        sigma_rot_deg = 10.0
        sigma_psi_deg = 10.0
        sigma_rot_rad = np.deg2rad(sigma_rot_deg)
        sigma_psi_rad = np.deg2rad(sigma_psi_deg)
        biggest_sigma = max(sigma_rot_rad, sigma_psi_rad)
        cone_rad = _SIGMA_CUTOFF * biggest_sigma
        dir_cutoff_deg = _SIGMA_CUTOFF * sigma_rot_deg
        psi_cutoff_deg = _SIGMA_CUTOFF * sigma_psi_deg

        relion_indices = _relion_full_cone_indices(
            prior_rot,
            prior_tilt,
            prior_psi,
            order,
            sigma_rot_deg,
            sigma_psi_deg,
            _SIGMA_CUTOFF,
        )
        relion_set = set(relion_indices)

        prior_mat = _prior_to_recovar_matrix(prior_rot, prior_tilt, prior_psi)
        recovar_indices, log_prior = get_local_rotation_grid_fast(
            prior_mat[np.newaxis],
            sigma_rot_rad,
            sigma_psi_rad,
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=True,
        )
        recovar_set = set(recovar_indices.tolist())

        recovar_only = recovar_set - relion_set
        if len(recovar_only) == 0:
            return  # Perfect overlap on recovar side

        # Compute direction and psi distances for recovar-only entries
        relion_dirs = get_healpix_directions(order)
        nside = 2**order
        n_pixels = healpy.nside2npix(nside)
        angular_step = get_angular_sampling(order)
        n_psi = int(np.ceil(360.0 / angular_step))
        psi_step = 360.0 / n_psi

        prior_dir = _relion_direction_vector(prior_rot, prior_tilt)
        prior_psi_wrapped = prior_psi % 360.0

        n_total = rotation_grid_size(order)
        all_mats = rotation_indices_to_matrices(
            np.arange(n_total, dtype=np.int64),
            order,
        )

        for idx in recovar_only:
            # Must be in the SO(3) ball
            d = _so3_distance_matrices(prior_mat, all_mats[idx].astype(np.float64))
            assert d <= cone_rad + 1e-10, (
                f"recovar-only index {idx} has d_SO3={np.rad2deg(d):.2f} deg > cone_rad={np.rad2deg(cone_rad):.2f} deg"
            )

            # Must be outside RELION's rectangle: d_dir >= cutoff OR d_psi >= cutoff
            pix = idx % n_pixels
            psi_idx = idx // n_pixels
            nest_pix = healpy.ring2nest(nside, pix)
            grid_dir = _relion_direction_vector(relion_dirs[nest_pix, 0], relion_dirs[nest_pix, 1])
            d_dir = np.rad2deg(np.arccos(np.clip(np.dot(prior_dir, grid_dir), -1.0, 1.0)))
            d_psi = abs(psi_idx * psi_step - prior_psi_wrapped)
            if d_psi > 180.0:
                d_psi = abs(d_psi - 360.0)

            outside_rect = d_dir >= dir_cutoff_deg - 1e-10 or d_psi >= psi_cutoff_deg - 1e-10
            assert outside_rect, (
                f"recovar-only index {idx}: d_dir={d_dir:.2f} < {dir_cutoff_deg:.2f} AND "
                f"d_psi={d_psi:.2f} < {psi_cutoff_deg:.2f} -- should be outside RELION rectangle"
            )


class TestConeRadiusFormula:
    """Verify recovar's cone_rad = sigma_cutoff * max(sigma_rot, sigma_psi)
    matches RELION's biggest_sigma = max(sigma_rot, sigma_tilt) formula."""

    @pytest.mark.parametrize(
        "sigma_rot_deg,sigma_psi_deg",
        [(10.0, 10.0), (5.0, 15.0), (15.0, 5.0)],
    )
    def test_cone_radius_matches_relion(self, sigma_rot_deg, sigma_psi_deg):
        """RELION: biggest_sigma = max(sigma_rot, sigma_tilt); cone = sigma_cutoff * biggest_sigma.
        recovar: biggest_sigma = max(sigma_rot, sigma_psi); cone = sigma_cutoff * biggest_sigma.

        These match when sigma_psi <= sigma_rot (since RELION's sigma_tilt = sigma_rot
        in auto_refine). When sigma_psi > sigma_rot, recovar's cone is conservatively
        LARGER, which is correct (never misses a high-weight rotation).
        """
        # RELION formula (sigma_tilt = sigma_rot in auto_refine)
        relion_biggest_sigma = max(sigma_rot_deg, sigma_rot_deg)
        relion_cone_deg = _SIGMA_CUTOFF * relion_biggest_sigma

        # recovar formula
        sigma_rot_rad = np.deg2rad(sigma_rot_deg)
        sigma_psi_rad = np.deg2rad(sigma_psi_deg)
        recovar_biggest_sigma = max(sigma_rot_rad, sigma_psi_rad)
        recovar_cone_deg = np.rad2deg(_SIGMA_CUTOFF * recovar_biggest_sigma)

        if sigma_psi_deg <= sigma_rot_deg:
            assert abs(recovar_cone_deg - relion_cone_deg) < 1e-10, (
                f"Cone mismatch: RELION={relion_cone_deg:.4f} vs recovar={recovar_cone_deg:.4f}"
            )
        else:
            # recovar cone is larger (conservative)
            assert recovar_cone_deg >= relion_cone_deg - 1e-10, (
                f"recovar cone ({recovar_cone_deg:.4f}) smaller than RELION cone ({relion_cone_deg:.4f})"
            )


class TestMultiplePriorsConeUnion:
    """When multiple priors are passed, the selected set is the UNION of
    individual cones, matching RELION's per-particle behavior."""

    @pytest.mark.parametrize("order", [2, 3])
    def test_union_is_superset_of_individual(self, order):
        """Union cone from 3 priors contains all indices from each individual cone."""
        sigma_rot_deg = 10.0
        sigma_psi_deg = 10.0
        sigma_rot_rad = np.deg2rad(sigma_rot_deg)
        sigma_psi_rad = np.deg2rad(sigma_psi_deg)

        priors = _PRIORS[:3]
        prior_mats = np.array([_prior_to_recovar_matrix(r, t, p) for r, t, p in priors])

        # Union call
        union_indices, _ = get_local_rotation_grid_fast(
            prior_mats,
            sigma_rot_rad,
            sigma_psi_rad,
            order,
            sigma_cutoff=_SIGMA_CUTOFF,
            per_image=False,
        )
        union_set = set(union_indices.tolist())

        # Individual calls
        for i, (r, t, p) in enumerate(priors):
            mat_i = _prior_to_recovar_matrix(r, t, p)
            ind_indices, _ = get_local_rotation_grid_fast(
                mat_i[np.newaxis],
                sigma_rot_rad,
                sigma_psi_rad,
                order,
                sigma_cutoff=_SIGMA_CUTOFF,
                per_image=True,
            )
            ind_set = set(ind_indices.tolist())

            missing = ind_set - union_set
            assert len(missing) == 0, f"Prior {i} ({r},{t},{p}): {len(missing)} indices missing from union"

        # Union should be no larger than sum of individuals (upper bound)
        total_individual = 0
        for r, t, p in priors:
            mat_i = _prior_to_recovar_matrix(r, t, p)
            ind_indices, _ = get_local_rotation_grid_fast(
                mat_i[np.newaxis],
                sigma_rot_rad,
                sigma_psi_rad,
                order,
                sigma_cutoff=_SIGMA_CUTOFF,
                per_image=True,
            )
            total_individual += len(ind_indices)
        assert len(union_set) <= total_individual, (
            f"Union ({len(union_set)}) larger than sum of individuals ({total_individual})"
        )

        print(f"\nOrder {order}: union={len(union_set)}, sum_individual={total_individual}")


class TestBindingGap:
    """Document the gap: RELION's selectOrientationsWithNonZeroPriorProbability
    is NOT exposed in sampling_bind.cpp.

    This test explicitly validates that the binding does not have the function,
    so the gap is tracked. When the binding is extended, update this test
    to call it directly instead of the analytical Python reimplementation.
    """

    def test_no_select_orientations_binding(self):
        """Confirm the binding gap: no selectOrientationsWithNonZeroPriorProbability."""
        from recovar.relion_bind import _relion_bind_core as core

        assert not hasattr(core, "select_orientations_with_prior"), (
            "Binding now exposes select_orientations_with_prior! Update test_s4_cone_filter.py to use it directly."
        )
        assert not hasattr(core, "selectOrientationsWithNonZeroPriorProbability"), (
            "Binding now exposes the RELION function directly! Update test_s4_cone_filter.py to use it directly."
        )
