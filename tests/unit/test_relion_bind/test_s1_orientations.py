"""Phase 5 (S1): Compare RELION's HEALPix orientation grid against recovar.

RELION: Healpix_Base in NEST order, pix2ang_z_phi → (rot=phi_deg, tilt=acos(z)_deg).
recovar: healpy in RING order, pix2ang → (theta, phi) → R_from_relion([theta_deg, phi_deg, psi]).

The pixel CENTERS are identical between NEST and RING — only the enumeration
order differs. We compare the two grids as unordered SETS of rotation matrices.

Tests:
1. Direction count matches for orders 0-4
2. Direction sets match (as rotation matrices) after sorting
3. Oversampled sub-grid: RELION's nest2xyf splitting matches recovar's grid
4. Perturbation: RELION vs recovar's apply_relion_rotation_perturbation
"""

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from recovar.relion_bind._relion_bind_core import (
    euler_angles_to_matrix,
    get_angular_sampling,
    get_coarse_orientations,
    get_healpix_directions,
    get_oversampled_orientations,
)


def _recovar_directions(order):
    """Get recovar's HEALPix directions as (rot, tilt) in degrees."""
    nside = 2**order
    npix = healpy.nside2npix(nside)
    theta, phi = healpy.pix2ang(nside, np.arange(npix))
    rot = np.rad2deg(phi)
    tilt = np.rad2deg(theta)
    return np.stack([rot, tilt], axis=-1)


def _direction_set_match(dirs_a, dirs_b, atol=1e-10):
    """Check if two direction sets match (as unordered sets on the sphere)."""
    n_a, n_b = len(dirs_a), len(dirs_b)
    if n_a != n_b:
        return False, f"Count mismatch: {n_a} vs {n_b}"

    cos_tilt_a = np.cos(np.deg2rad(dirs_a[:, 1]))
    cos_tilt_b = np.cos(np.deg2rad(dirs_b[:, 1]))
    sin_tilt_a = np.sin(np.deg2rad(dirs_a[:, 1]))
    sin_tilt_b = np.sin(np.deg2rad(dirs_b[:, 1]))

    xa = sin_tilt_a * np.cos(np.deg2rad(dirs_a[:, 0]))
    ya = sin_tilt_a * np.sin(np.deg2rad(dirs_a[:, 0]))
    za = cos_tilt_a
    xb = sin_tilt_b * np.cos(np.deg2rad(dirs_b[:, 0]))
    yb = sin_tilt_b * np.sin(np.deg2rad(dirs_b[:, 0]))
    zb = cos_tilt_b

    pts_a = np.stack([xa, ya, za], axis=-1)
    pts_b = np.stack([xb, yb, zb], axis=-1)

    dots = pts_a @ pts_b.T
    best_match = np.max(dots, axis=1)
    worst = np.min(best_match)
    if worst < 1.0 - atol:
        return False, f"Worst dot product: {worst:.15f}"
    return True, f"All matched (worst dot = {worst:.15f})"


class TestDirectionCount:
    """Verify RELION and recovar produce the same number of HEALPix directions."""

    @pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
    def test_direction_count(self, order):
        relion_dirs = get_healpix_directions(order)
        expected = healpy.nside2npix(2**order)
        assert relion_dirs.shape == (expected, 2), (
            f"Order {order}: RELION {relion_dirs.shape[0]} vs expected {expected}"
        )

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_psi_count(self, order):
        angular_step = get_angular_sampling(order)
        n_psi_relion = int(np.ceil(360.0 / angular_step))

        from recovar.em.sampling import rotation_grid_n_in_planes

        n_psi_recovar = rotation_grid_n_in_planes(order)
        assert n_psi_relion == n_psi_recovar, (
            f"Order {order}: psi count RELION={n_psi_relion} vs recovar={n_psi_recovar}"
        )


class TestDirectionSetMatch:
    """Verify RELION and recovar produce the same set of directions on the sphere."""

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_direction_sets_match(self, order):
        relion_dirs = get_healpix_directions(order)
        recovar_dirs = _recovar_directions(order)

        ok, msg = _direction_set_match(relion_dirs, recovar_dirs)
        print(f"\nOrder {order}: {msg}")
        assert ok, f"Direction set mismatch at order {order}: {msg}"


class TestRotationMatrices:
    """Compare RELION and recovar rotation matrices for the same Euler angles."""

    def test_euler_round_trip(self):
        """RELION Euler→matrix→Euler round-trip preserves angles."""
        from recovar.relion_bind._relion_bind_core import matrix_to_euler_angles

        for rot, tilt, psi in [(45, 90, 30), (0, 0, 0), (-120, 45, 180), (180, 90, -90)]:
            R = euler_angles_to_matrix(float(rot), float(tilt), float(psi))
            r2, t2, p2 = matrix_to_euler_angles(R)
            R2 = euler_angles_to_matrix(r2, t2, p2)
            diff = np.max(np.abs(R - R2))
            assert diff < 1e-12, f"Round-trip error for ({rot},{tilt},{psi}): {diff:.2e}"

    @pytest.mark.parametrize("order", [2, 3])
    def test_rotation_matrices_match_recovar(self, order):
        """RELION and recovar grids cover the same set of SO(3) orientations.

        RELION's Euler_angles2matrix(rot, tilt, psi) produces matrices in
        RELION's coordinate frame. recovar's R_from_relion applies a frame
        adjustment (volume axis flip + sign). To compare, we convert RELION's
        [rot, tilt, psi] → healpy [tilt, rot, psi] and pass through
        R_from_relion, then match as unordered sets.
        """
        from recovar.em.sampling import get_rotation_grid
        from recovar.utils.helpers import R_from_relion

        relion_coarse = get_coarse_orientations(order)
        n_total = relion_coarse.shape[0]

        # RELION returns [rot=phi, tilt=theta, psi]; R_from_relion expects
        # [theta, phi, psi] (healpy convention) — swap columns 0,1.
        healpy_angles = relion_coarse.copy()
        healpy_angles[:, 0] = relion_coarse[:, 1]  # tilt → theta
        healpy_angles[:, 1] = relion_coarse[:, 0]  # rot → phi
        relion_mats = R_from_relion(healpy_angles)

        recovar_mats = np.array(get_rotation_grid(order, matrices=True))

        assert relion_mats.shape[0] == recovar_mats.shape[0], (
            f"Grid size mismatch: RELION={relion_mats.shape[0]} vs recovar={recovar_mats.shape[0]}"
        )

        traces = np.einsum("nij,mij->nm", relion_mats, recovar_mats)
        cos_angle = np.clip((traces - 1.0) / 2.0, -1, 1)
        best_per_relion = np.max(cos_angle, axis=1)
        angles_deg = np.rad2deg(np.arccos(np.clip(best_per_relion, -1, 1)))

        worst_angle = np.max(angles_deg)
        mean_angle = np.mean(angles_deg)
        n_unmatched = np.sum(angles_deg > 0.1)

        print(
            f"\nOrder {order}: worst={worst_angle:.4f}°, mean={mean_angle:.6f}°, "
            f"unmatched(>0.1°)={n_unmatched}/{n_total}"
        )
        assert worst_angle < 0.1, f"Rotation set coverage mismatch: worst angular distance = {worst_angle:.4f}°"


class TestOversampledGrid:
    """Compare oversampled orientation sub-grids."""

    @pytest.mark.parametrize("oversampling_order", [1, 2])
    def test_oversampled_count(self, oversampling_order):
        """Oversampled grid has expected number of orientations."""
        order = 2
        ov = get_oversampled_orientations(order, oversampling_order, 0, 0, 0.0)
        n_dir_over = 4**oversampling_order
        n_psi_over = 2**oversampling_order
        expected = n_dir_over * n_psi_over
        assert ov.shape[0] == expected, f"OS={oversampling_order}: got {ov.shape[0]}, expected {expected}"

    def test_oversampled_os0_matches_coarse(self):
        """With oversampling=0, get_oversampled_orientations returns the coarse grid point."""
        order = 2
        coarse = get_coarse_orientations(order)
        n_psi = int(np.ceil(360.0 / get_angular_sampling(order)))
        n_dir = coarse.shape[0] // n_psi

        for idir in [0, 5, n_dir - 1]:
            for ipsi in [0, 3, n_psi - 1]:
                ov = get_oversampled_orientations(order, 0, idir, ipsi, 0.0)
                assert ov.shape[0] == 1
                coarse_idx = idir * n_psi + ipsi
                diff = np.max(np.abs(ov[0] - coarse[coarse_idx]))
                assert diff < 1e-12, f"OS=0 mismatch at idir={idir}, ipsi={ipsi}: diff={diff:.2e}"

    def test_oversampled_within_coarse_cell(self):
        """Oversampled directions lie within the coarse cell angular radius."""
        order = 2
        coarse = get_coarse_orientations(order)
        coarse_step = get_angular_sampling(order)
        n_psi = int(np.ceil(360.0 / coarse_step))

        idir, ipsi = 10, 5
        coarse_idx = idir * n_psi + ipsi
        rot_c, tilt_c, psi_c = coarse[coarse_idx]

        ov = get_oversampled_orientations(order, 1, idir, ipsi, 0.0)
        R_coarse = euler_angles_to_matrix(rot_c, tilt_c, psi_c)
        for i in range(ov.shape[0]):
            R_over = euler_angles_to_matrix(ov[i, 0], ov[i, 1], ov[i, 2])
            trace = np.trace(R_coarse.T @ R_over)
            cos_angle = (trace - 1.0) / 2.0
            angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
            assert angle_deg < coarse_step, (
                f"Oversampled point {i} at {angle_deg:.2f}° exceeds coarse step {coarse_step:.2f}°"
            )


class TestPerturbation:
    """Compare RELION perturbation against recovar's apply_relion_rotation_perturbation."""

    def test_perturbation_vs_recovar(self):
        from recovar.em.sampling import apply_relion_rotation_perturbation

        order = 2
        random_pert = 0.3
        coarse_step = get_angular_sampling(order)

        relion_pert = get_oversampled_orientations(order, 0, 10, 5, random_pert)
        relion_unpert = get_oversampled_orientations(order, 0, 10, 5, 0.0)

        R_unpert = euler_angles_to_matrix(relion_unpert[0, 0], relion_unpert[0, 1], relion_unpert[0, 2])
        R_pert_relion = euler_angles_to_matrix(relion_pert[0, 0], relion_pert[0, 1], relion_pert[0, 2])

        R_pert_recovar = apply_relion_rotation_perturbation(R_unpert[np.newaxis], random_pert, coarse_step)[0]

        diff = np.max(np.abs(R_pert_relion - R_pert_recovar))
        print(f"\nPerturbation parity: max |R_relion - R_recovar| = {diff:.2e}")
        assert diff < 1e-10, f"Perturbation mismatch: {diff:.2e}"
