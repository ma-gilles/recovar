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
    euler_angles_to_inverse_matrices,
    euler_angles_to_matrix,
    get_angular_sampling,
    get_coarse_orientations,
    get_healpix_directions,
    get_oversampled_orientations,
    get_oversampled_orientations_batch,
)


def test_mstep_rotation_helper_uses_exact_relion_host_inverse():
    from recovar.em.sampling import _relion_mstep_rotations_from_eulers

    angles = np.asarray(
        [
            [-82.34140109, 159.18206946, -123.87249525],
            [13.123456789, 42.987654321, -77.111111111],
        ],
        dtype=np.float64,
    )
    expected = np.swapaxes(euler_angles_to_inverse_matrices(angles), 1, 2)
    actual = _relion_mstep_rotations_from_eulers(angles, dtype=np.float64)
    np.testing.assert_array_equal(actual, expected)


def test_batched_oversampled_orientations_match_individual_calls_exactly():
    idirs = np.asarray([0, 17, 191], dtype=np.int64)
    ipsis = np.asarray([0, 5, 11], dtype=np.int64)
    perturbation = -0.261189430952
    expected = np.concatenate(
        [
            get_oversampled_orientations(2, 1, int(idir), int(ipsi), perturbation)
            for idir, ipsi in zip(idirs, ipsis, strict=True)
        ],
        axis=0,
    )
    actual = get_oversampled_orientations_batch(2, 1, idirs, ipsis, perturbation)
    np.testing.assert_array_equal(actual, expected)


def test_sampled_grid_uses_native_eulers_and_inverse_exactly():
    from recovar.em.sampling import get_oversampled_rotation_grid_from_samples

    order = 2
    n_pixels = 12 * (4**order)
    parents = np.asarray([17, 5 * n_pixels + 191], dtype=np.int64)
    perturbation = -0.261189430952
    eulers = get_oversampled_orientations_batch(
        order,
        1,
        parents % n_pixels,
        parents // n_pixels,
        perturbation,
    )
    expected = np.swapaxes(euler_angles_to_inverse_matrices(eulers), 1, 2)
    rotations, _, mstep_rotations = get_oversampled_rotation_grid_from_samples(
        parents,
        order,
        oversampling_order=1,
        random_perturbation=perturbation,
        return_mstep_rotations=True,
        dtype=np.float64,
    )
    np.testing.assert_array_equal(rotations, expected)
    np.testing.assert_array_equal(mstep_rotations, expected)


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

        After the RELION-grid convention fix, ``get_rotation_grid`` now
        emits Euler angles in the same ``[rot, tilt, psi]`` convention that
        ``R_from_relion`` expects.  Compare the two grids directly as
        unordered sets of matrices.
        """
        from recovar.em.sampling import get_rotation_grid
        from recovar.utils.helpers import R_from_relion

        relion_coarse = get_coarse_orientations(order)
        n_total = relion_coarse.shape[0]
        relion_mats = R_from_relion(relion_coarse)

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

    def test_full_precision_seeded_perturbation_matches_accelerated_matrix_dump(self):
        """STAR-rounded perturbation must not replace RELION's live seeded value."""
        from recovar.em.sampling import (
            get_oversampled_rotation_grid_from_samples,
            relion_sampling_perturbation_for_iteration,
        )

        parent_sample = np.asarray([47 * 768 + 149], dtype=np.int64)
        exact_perturbation = relion_sampling_perturbation_for_iteration(0.5, 20260712, 2)
        exact, _ = get_oversampled_rotation_grid_from_samples(
            parent_sample,
            parent_nside_level=3,
            oversampling_order=1,
            random_perturbation=exact_perturbation,
        )
        rounded, _ = get_oversampled_rotation_grid_from_samples(
            parent_sample,
            parent_nside_level=3,
            oversampling_order=1,
            random_perturbation=0.405200,
        )
        expected_effective = np.asarray(
            [
                [-0.19852252304553986, 0.9793077111244202, 0.03930952027440071],
                [-0.620071291923523, -0.09443635493516922, -0.7788410782814026],
                [-0.7590128183364868, -0.17899219691753387, 0.6259883046150208],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(exact[0].T, expected_effective)
        assert np.any(rounded[0] != exact[0])

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

    @pytest.mark.parametrize("oversampling_order", [1, 2])
    @pytest.mark.parametrize("random_perturbation", [0.0, 0.461207])
    def test_recovar_sampled_oversampling_matches_relion_binding_order(self, oversampling_order, random_perturbation):
        """RECOVAR's sampled oversampling path must preserve RELION child order."""
        from recovar.em.sampling import (
            _relion_euler_angles_to_matrix,
            get_oversampled_rotation_grid_from_samples,
            rotation_grid_n_in_planes,
        )

        order = 2
        n_pixels = healpy.nside2npix(2**order)
        parent_samples = [
            0,
            5,
            n_pixels + 17,
            3 * n_pixels + 191,
        ]
        matrices, parent_map, child_indices = get_oversampled_rotation_grid_from_samples(
            np.asarray(parent_samples, dtype=np.int64),
            order,
            oversampling_order=oversampling_order,
            random_perturbation=random_perturbation,
            return_rotation_indices=True,
            rotation_index_order="recovar",
        )

        expected_blocks = []
        expected_parent = []
        for parent_pos, sample in enumerate(parent_samples):
            idir = int(sample % n_pixels)
            ipsi = int(sample // n_pixels)
            expected_eulers = get_oversampled_orientations(
                order,
                oversampling_order,
                idir,
                ipsi,
                random_perturbation,
            )
            expected_blocks.append(_relion_euler_angles_to_matrix(expected_eulers).astype(np.float32))
            expected_parent.extend([parent_pos] * expected_eulers.shape[0])

        expected_matrices = np.concatenate(expected_blocks, axis=0)
        np.testing.assert_array_equal(parent_map, np.asarray(expected_parent, dtype=np.int64))
        np.testing.assert_allclose(matrices, expected_matrices, rtol=2e-5, atol=2e-5)

        fine_order = order + oversampling_order
        fine_n_pixels = healpy.nside2npix(2**fine_order)
        fine_n_psi = rotation_grid_n_in_planes(fine_order)
        assert child_indices.shape == parent_map.shape
        assert np.all(child_indices >= 0)
        assert np.all(child_indices < fine_n_pixels * fine_n_psi)


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
