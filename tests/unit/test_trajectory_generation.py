"""Unit tests for recovar.simulation.trajectory_generation."""

import os
import tempfile

import numpy as np
import pytest

from recovar.simulation.pdb_utils import AtomGroup
from recovar.simulation.trajectory_generation import (
    compute_bfactor_scaling,
    generate_conformation_2D,
    generate_trajectory_volumes,
    path_arm_only,
    path_asymmetric,
    path_head_only,
    path_symmetric,
    prepare_5nrl_subcomplexes,
    rigid_motion,
    split_atom_group_by_chains,
    stitched_path,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# split_atom_group_by_chains
# ---------------------------------------------------------------------------


class TestSplitAtomGroupByChains:
    def test_basic_split(self):
        atoms = AtomGroup()
        atoms.setCoords(np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=np.float64))
        atoms.setElements(np.array(["C", "N", "O", "C"]))
        atoms.setChids(np.array(["A", "A", "B", "C"]))

        groups = split_atom_group_by_chains(atoms, [["A"], ["B", "C"]])
        assert len(groups) == 2
        assert groups[0][0].shape == (2, 3)
        assert groups[1][0].shape == (2, 3)
        np.testing.assert_array_equal(groups[0][1], ["C", "N"])
        np.testing.assert_array_equal(groups[1][1], ["O", "C"])

    def test_empty_chain_group(self):
        atoms = AtomGroup()
        atoms.setCoords(np.array([[1, 0, 0]], dtype=np.float64))
        atoms.setElements(np.array(["C"]))
        atoms.setChids(np.array(["A"]))

        groups = split_atom_group_by_chains(atoms, [["A"], ["X"]])
        assert groups[0][0].shape == (1, 3)
        assert groups[1][0].shape == (0, 3)


# ---------------------------------------------------------------------------
# rigid_motion
# ---------------------------------------------------------------------------


class TestRigidMotion:
    def test_identity_rotation(self):
        from scipy.spatial.transform import Rotation

        coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        pivot = np.array([0, 0, 0], dtype=np.float64)
        result = rigid_motion(coords, pivot, Rotation.identity())
        np.testing.assert_allclose(result, coords, atol=1e-14)

    def test_rotation_around_pivot(self):
        from scipy.spatial.transform import Rotation

        # 180-degree rotation around z-axis: (1,0,0) -> (-1,0,0)
        coords = np.array([[2, 0, 0]], dtype=np.float64)
        pivot = np.array([1, 0, 0], dtype=np.float64)
        rot = Rotation.from_euler("z", 180, degrees=True)
        result = rigid_motion(coords, pivot, rot)
        np.testing.assert_allclose(result, [[0, 0, 0]], atol=1e-14)

    def test_preserves_distances(self):
        from scipy.spatial.transform import Rotation

        rng = np.random.default_rng(42)
        coords = rng.standard_normal((50, 3))
        pivot = rng.standard_normal(3)
        rot = Rotation.random(random_state=42)
        result = rigid_motion(coords, pivot, rot)

        # Pairwise distances should be preserved
        from scipy.spatial.distance import pdist

        np.testing.assert_allclose(pdist(result), pdist(coords), atol=1e-12)


# ---------------------------------------------------------------------------
# generate_conformation_2D and path functions
# ---------------------------------------------------------------------------


class TestPathFunctions:
    @pytest.fixture
    def simple_groups(self):
        """3 groups of 10 atoms each, plus a pivot."""
        rng = np.random.default_rng(123)
        groups = [rng.standard_normal((10, 3)) for _ in range(3)]
        pivot = np.zeros(3)
        return groups, pivot

    def test_path_symmetric_at_zero(self, simple_groups):
        groups, pivot = simple_groups
        result = path_symmetric(0, groups, pivot)
        # At t=0, rot_around_x gives -10 degrees, so groups[0] unchanged
        np.testing.assert_allclose(result[0], groups[0], atol=1e-14)

    def test_path_symmetric_modifies_arms(self, simple_groups):
        groups, pivot = simple_groups
        result = path_symmetric(15, groups, pivot)
        # B and Db should be rotated
        assert not np.allclose(result[1], groups[1])
        assert not np.allclose(result[2], groups[2])

    def test_path_asymmetric_fixes_B(self, simple_groups):
        groups, pivot = simple_groups
        result1 = path_asymmetric(0, groups, pivot)
        result2 = path_asymmetric(10, groups, pivot)
        # B should be the same (both at t_right_fixed=40)
        np.testing.assert_allclose(result1[1], result2[1], atol=1e-14)
        # Db should differ
        assert not np.allclose(result1[2], result2[2])

    def test_stitched_path_continuity(self, simple_groups):
        groups, pivot = simple_groups
        # At t=35, stitched should equal path_symmetric(35)
        result_stitch = stitched_path(35, groups, pivot)
        result_sym = path_symmetric(35, groups, pivot)
        for r1, r2 in zip(result_stitch, result_sym):
            np.testing.assert_allclose(r1, r2, atol=1e-14)

    def test_conformation_2D_ab_unchanged(self, simple_groups):
        groups, pivot = simple_groups
        result = generate_conformation_2D(groups, 5, 10, pivot)
        # Ab (index 0) should not be rotated
        np.testing.assert_allclose(result[0], groups[0], atol=1e-14)

    def test_path_arm_only_keeps_head_fixed(self, simple_groups):
        groups, pivot = simple_groups
        # Db (head) should be in the SAME pose at t=0 and t=20 (head fixed across the trajectory).
        # Note: rot_around_x(0) is not identity (notebook offset of -10 deg), so we compare two
        # different `t` values rather than to the input groups.
        result_a = path_arm_only(0, groups, pivot)
        result_b = path_arm_only(20, groups, pivot)
        np.testing.assert_allclose(result_a[0], result_b[0], atol=1e-14)  # Ab unchanged
        np.testing.assert_allclose(result_a[2], result_b[2], atol=1e-14)  # Db (head) unchanged
        assert not np.allclose(result_a[1], result_b[1])  # B (arm) varies with t

    def test_path_head_only_keeps_arm_fixed(self, simple_groups):
        groups, pivot = simple_groups
        result_a = path_head_only(0, groups, pivot)
        result_b = path_head_only(20, groups, pivot)
        np.testing.assert_allclose(result_a[0], result_b[0], atol=1e-14)  # Ab unchanged
        np.testing.assert_allclose(result_a[1], result_b[1], atol=1e-14)  # B (arm) unchanged
        assert not np.allclose(result_a[2], result_b[2])  # Db (head) varies with t


# ---------------------------------------------------------------------------
# compute_bfactor_scaling
# ---------------------------------------------------------------------------


class TestBfactorScaling:
    def test_shape(self):
        shape = (16, 16, 16)
        result = compute_bfactor_scaling(shape, voxel_size=4.25, Bfactor=80)
        assert result.shape == shape

    def test_max_is_one(self):
        shape = (16, 16, 16)
        result = compute_bfactor_scaling(shape, voxel_size=4.25, Bfactor=80)
        # The DC component (zero frequency) should give exp(0) = 1,
        # which is the maximum value in the array.
        assert abs(np.max(result) - 1.0) < 1e-10

    def test_bfactor_zero_is_all_ones(self):
        shape = (16, 16, 16)
        result = compute_bfactor_scaling(shape, voxel_size=4.25, Bfactor=0)
        np.testing.assert_allclose(result, 1.0, atol=1e-14)

    def test_higher_bfactor_more_dampening(self):
        shape = (16, 16, 16)
        low_b = compute_bfactor_scaling(shape, voxel_size=4.25, Bfactor=20)
        high_b = compute_bfactor_scaling(shape, voxel_size=4.25, Bfactor=200)
        # Higher B should dampen more (smaller values at high freq)
        assert np.sum(high_b) < np.sum(low_b)


# ---------------------------------------------------------------------------
# prepare_5nrl_subcomplexes (loading the shipped .npz asset)
# ---------------------------------------------------------------------------


class TestPrepare5nrl:
    def test_loads_and_returns_correct_structure(self):
        group_coords, group_elements, fixed_pt = prepare_5nrl_subcomplexes()
        assert len(group_coords) == 3
        assert len(group_elements) == 3
        assert fixed_pt.shape == (3,)

        # Total atoms should match 5nrl
        total = sum(c.shape[0] for c in group_coords)
        assert total > 90000  # 5nrl has ~100k atoms

        # Elements should be valid
        all_elems = set(np.concatenate(group_elements).tolist())
        assert "C" in all_elems
        assert "N" in all_elems

    def test_centered_bounding_box(self):
        group_coords, _, _ = prepare_5nrl_subcomplexes()
        all_coords = np.concatenate(group_coords)
        # Centering uses bounding-box midpoint, not mean
        bbox_center = (all_coords.max(axis=0) + all_coords.min(axis=0)) / 2
        np.testing.assert_allclose(bbox_center, 0, atol=1.0)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            prepare_5nrl_subcomplexes("/nonexistent/path.npz")


# ---------------------------------------------------------------------------
# generate_trajectory_volumes (tiny grid, fast)
# ---------------------------------------------------------------------------


class TestGenerateTrajectoryVolumes:
    def test_generates_mrc_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n_vols = 3
            grid_size = 16
            prefix = generate_trajectory_volumes(
                output_dir=tmpdir,
                grid_size=grid_size,
                n_volumes=n_vols,
                Bfactor=80,
                max_rotation_degrees=5.0,
            )
            for i in range(n_vols):
                path = f"{prefix}{i:04d}.mrc"
                assert os.path.isfile(path), f"Missing volume: {path}"

    def test_volumes_differ_along_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n_vols = 5
            grid_size = 16
            prefix = generate_trajectory_volumes(
                output_dir=tmpdir,
                grid_size=grid_size,
                n_volumes=n_vols,
                Bfactor=80,
                max_rotation_degrees=10.0,
            )
            from recovar import utils

            vol_first = utils.load_mrc(f"{prefix}0000.mrc")
            vol_last = utils.load_mrc(f"{prefix}{n_vols - 1:04d}.mrc")
            # Volumes at different trajectory points should differ
            assert not np.allclose(vol_first, vol_last, atol=1e-6)
