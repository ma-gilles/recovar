"""Trajectory generation from PDB/CIF structures via rigid-body motions.

Replicates the approach in ``make_trajectories.ipynb`` using PDB 5nrl
(pre-catalytic spliceosome, 58 chains).  Provides reusable functions for:

1. Splitting an AtomGroup into subcomplexes by chain ID lists.
2. Applying rigid-body rotations around a pivot (hinge) point.
3. Generating conformational trajectories (1-D and 2-D).
4. Writing trajectory volumes (scattering potentials with B-factor dampening).

Usage::

    from recovar.simulation.trajectory_generation import (
        split_atom_group_by_chains,
        rigid_motion,
        generate_conformation_2D,
        generate_trajectory_volumes,
    )
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 5nrl spliceosome structure constants (from make_trajectories.ipynb)
# ---------------------------------------------------------------------------

# Chain groupings: the 58 chains are partitioned into 3 subcomplexes.
# Ab = main body + extra chains from group C2
# B  = right arm (8 chains)
# Db = top arm minus the "head" chains (12 chains)
CHAIN_GROUPS_5NRL = {
    "Ab": [
        "A",
        "C",
        "D",
        "E",
        "F",
        "J",
        "K",
        "L",
        "M",
        "N",
        "X",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "4",
        "5",
        "6",
        "G",
        "H",
    ],
    "B": ["B", "k", "l", "m", "n", "p", "q", "r"],
    "Db": ["I", "O", "P", "Q", "S", "a", "z", "o", "j", "3", "7", "8"],
}

# Hinge point: atom index 59060 in the concatenated (Ab+B+Db) coordinate array
# after centering.  Taken directly from the notebook (``fixed_idx1 = 59060``).
HINGE_INDEX_5NRL = 59060

# Default location of the pre-extracted 5nrl atom data (coords, elements,
# chain IDs stored as a compressed npz — ~1.2 MB vs 14 MB for the full CIF).
DEFAULT_5NRL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "5nrl_atoms.npz",
)


# ---------------------------------------------------------------------------
# Atom-group utilities
# ---------------------------------------------------------------------------


def split_atom_group_by_chains(atoms, chain_groups):
    """Split an AtomGroup into sub-groups by chain ID lists.

    Parameters
    ----------
    atoms : AtomGroup
        Must have ``chain_ids`` populated (via ``getChids()``).
    chain_groups : list[list[str]]
        Each inner list gives the chain IDs for one subcomplex,
        e.g. ``[['A', 'C'], ['B'], ['D']]``.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        ``(coords, elements)`` for each subcomplex.
    """
    chain_ids = atoms.getChids()
    all_coords = atoms.getCoords()
    all_elements = atoms.getElements()
    groups = []
    for chains in chain_groups:
        mask = np.zeros(len(chain_ids), dtype=bool)
        for c in chains:
            mask |= chain_ids == c
        groups.append((all_coords[mask].copy(), all_elements[mask].copy()))
    return groups


# ---------------------------------------------------------------------------
# Rigid-body motion primitives
# ---------------------------------------------------------------------------


def rigid_motion(coords, pivot, rotation):
    """Apply a rigid-body rotation around *pivot*.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    pivot : np.ndarray, shape (3,)
    rotation : scipy.spatial.transform.Rotation

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    return rotation.apply(coords - pivot) + pivot


def rot_around_x(t_degrees):
    """Rotation of ``-10 + t`` degrees around the x-axis (notebook convention)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("x", -10 + t_degrees, degrees=True)


def rot_around_z(t_degrees):
    """Rotation of *t* degrees around the z-axis."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("z", t_degrees, degrees=True)


# ---------------------------------------------------------------------------
# Trajectory path functions (from make_trajectories.ipynb)
# ---------------------------------------------------------------------------


def generate_conformation_2D(group_coords, t_right, t_top, fixed_pt):
    """Generate a 2-D conformation by rotating subcomplexes B and Db.

    Parameters
    ----------
    group_coords : list[np.ndarray]
        ``[Ab_coords, B_coords, Db_coords]`` — centered coordinate arrays.
    t_right : float
        Rotation angle (degrees) for the right arm (B), via ``rot_around_x``.
    t_top : float
        Rotation angle (degrees) for the top arm (Db), via ``rot_around_x``.
    fixed_pt : np.ndarray, shape (3,)
        Hinge/pivot point.

    Returns
    -------
    list[np.ndarray]
        ``[Ab_coords, rotated_B_coords, rotated_Db_coords]``.
    """
    coords = [c.copy() for c in group_coords]
    coords[1] = rigid_motion(coords[1], fixed_pt, rot_around_x(t_right))
    coords[2] = rigid_motion(coords[2], fixed_pt, rot_around_x(t_top))
    return coords


def path_symmetric(t, group_coords, fixed_pt):
    """Path where both B and Db rotate by the same angle *t*.

    ``t`` in [0, ~40] degrees.  Corresponds to ``path1`` in the notebook.
    """
    return generate_conformation_2D(group_coords, t, t, fixed_pt)


def path_asymmetric(t, group_coords, fixed_pt, t_right_fixed=40):
    """Path where B stays at *t_right_fixed* and Db reverses.

    ``t`` in [0, ~35] degrees.  Corresponds to ``path2`` in the notebook
    (``generate_conformation_2D(coords, 40, 40-t, fixed_pt)``).
    """
    return generate_conformation_2D(group_coords, t_right_fixed, t_right_fixed - t, fixed_pt)


def path_left_right(t, group_coords, fixed_pt):
    """Left-right (z-axis rotation) path for B and Db.

    ``t`` in [0, ~10] degrees.  Corresponds to ``pathLR`` in the notebook.
    """
    coords = [c.copy() for c in group_coords]
    rot = rot_around_z(t)
    coords[1] = rigid_motion(coords[1], fixed_pt, rot)
    coords[2] = rigid_motion(coords[2], fixed_pt, rot)
    return coords


def path_arm_only(t, group_coords, fixed_pt):
    """Path where only the right arm (B) rotates; the head (Db) stays fixed.

    Useful when you want a smaller moving region with a larger rotation
    angle — exercises ``compute_state``'s focus mask on a localized swept
    volume rather than the combined B+Db region used by ``path_symmetric``.
    """
    return generate_conformation_2D(group_coords, t, 0, fixed_pt)


def path_head_only(t, group_coords, fixed_pt):
    """Path where only the head (Db) rotates; the right arm (B) stays fixed."""
    return generate_conformation_2D(group_coords, 0, t, fixed_pt)


def stitched_path(t, group_coords, fixed_pt):
    """Two-segment path: symmetric up to t=35, then asymmetric reversal.

    ``t`` in [0, ~60] degrees.  Corresponds to ``stitched_path`` in the notebook.
    """
    if t <= 35:
        return path_symmetric(t, group_coords, fixed_pt)
    else:
        return path_asymmetric(t - 35, group_coords, fixed_pt)


# ---------------------------------------------------------------------------
# Prepare 5nrl subcomplexes (centered, with hinge point)
# ---------------------------------------------------------------------------


def _load_5nrl_atoms(path):
    """Load 5nrl atom data from a ``.npz``, ``.cif``, or ``.pdb`` file."""
    from recovar.simulation.pdb_utils import AtomGroup

    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=False)
        atoms = AtomGroup()
        atoms.setCoords(data["coords"].astype(np.float64))
        atoms.setElements(data["elements"])
        atoms.setChids(data["chain_ids"])
        return atoms

    # Fall back to full CIF/PDB parsing
    from recovar.simulation.pdb_utils import parse_pdb

    return parse_pdb(path)


def prepare_5nrl_subcomplexes(pdb_path=None):
    """Parse 5nrl and return centered subcomplex coordinates + hinge point.

    Parameters
    ----------
    pdb_path : str or None
        Path to the 5nrl data file (``.npz``, ``.cif``, or ``.pdb``).
        Defaults to ``DEFAULT_5NRL_PATH`` (the compact npz).

    Returns
    -------
    group_coords : list[np.ndarray]
        ``[Ab_coords, B_coords, Db_coords]`` — centered.
    group_elements : list[np.ndarray]
        ``[Ab_elements, B_elements, Db_elements]``.
    fixed_pt : np.ndarray, shape (3,)
        Hinge point in centered coordinates.
    """
    from recovar.simulation import simulate_scattering_potential as ssp

    if pdb_path is None:
        pdb_path = DEFAULT_5NRL_PATH
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"5nrl data file not found: {pdb_path}.  Expected at {DEFAULT_5NRL_PATH}")

    atoms = _load_5nrl_atoms(pdb_path)
    logger.info("Loaded %s: %d atoms, %d chains", pdb_path, atoms.numAtoms(), len(set(atoms.getChids().tolist())))

    chain_lists = [CHAIN_GROUPS_5NRL["Ab"], CHAIN_GROUPS_5NRL["B"], CHAIN_GROUPS_5NRL["Db"]]
    groups = split_atom_group_by_chains(atoms, chain_lists)
    group_coords = [g[0] for g in groups]
    group_elements = [g[1] for g in groups]

    for name, coords in zip(["Ab", "B", "Db"], group_coords):
        logger.info("Subcomplex %s: %d atoms", name, len(coords))

    # Center
    all_coords_flat = np.concatenate(group_coords)
    center_offset = ssp.get_center_coord_offset(all_coords_flat)
    group_coords = [c - center_offset for c in group_coords]

    # Hinge point
    all_centered = np.concatenate(group_coords)
    if HINGE_INDEX_5NRL < len(all_centered):
        fixed_pt = all_centered[HINGE_INDEX_5NRL].copy()
    else:
        from scipy.spatial import cKDTree

        tree = cKDTree(group_coords[0])
        dists, _ = tree.query(group_coords[1], k=1)
        closest = np.argsort(dists)[:50]
        fixed_pt = np.mean(group_coords[1][closest], axis=0)
    logger.info("Hinge point (centered): %s", fixed_pt)

    return group_coords, group_elements, fixed_pt


# ---------------------------------------------------------------------------
# B-factor scaling (matching the notebook's apply_B_factor)
# ---------------------------------------------------------------------------


def compute_bfactor_scaling(volume_shape, voxel_size, Bfactor):
    """Compute frequency-dependent B-factor dampening in 3-D.

    Matches the notebook's ``apply_B_factor`` function:
    ``B_fac_scaling = exp(-B * |freq|^2 / 4)``.

    Returns an array of shape *volume_shape* (not flattened).
    """
    from recovar import core as core_module

    vol_idx = np.arange(np.prod(volume_shape))
    freqs = core_module.vec_indices_to_frequencies(vol_idx, volume_shape) / (volume_shape[0] * voxel_size)
    freq_norms = np.linalg.norm(freqs, axis=-1) ** 2
    return np.exp(-Bfactor * freq_norms / 4).reshape(volume_shape)


# ---------------------------------------------------------------------------
# Volume generation
# ---------------------------------------------------------------------------


def generate_trajectory_volumes(
    output_dir,
    grid_size=64,
    n_volumes=50,
    voxel_size=None,
    Bfactor=80,
    max_rotation_degrees=10.0,
    path_fn=None,
    pdb_path=None,
    prefix_name="vol",
    output_prefix=None,
):
    """Generate trajectory volumes from 5nrl using rigid-body subcomplex motions.

    Parameters
    ----------
    output_dir : str
        Base output directory.
    grid_size : int
        Volume grid size (default 64, matching the notebook).
    n_volumes : int
        Number of volumes along the trajectory.
    voxel_size : float or None
        Voxel size in Angstroms.  Default: ``4.25 * 128 / grid_size``.
    Bfactor : float
        B-factor for Fourier-space dampening.
    max_rotation_degrees : float
        Maximum rotation angle for the trajectory.
    path_fn : callable or None
        ``path_fn(t, group_coords, fixed_pt) -> [coords_Ab, coords_B, coords_Db]``.
        Default: ``path_symmetric`` (both B and Db rotate together).
    pdb_path : str or None
        Path to 5nrl CIF/PDB file.
    prefix_name : str
        Volume file prefix name (e.g. ``"vol"``).
    output_prefix : str or None
        Full prefix path override.

    Returns
    -------
    str
        Volume prefix path (e.g. ``<dir>/vol`` for files vol0000.mrc, ...).
    """
    import recovar.core.fourier_transform_utils as ftu
    from recovar import utils
    from recovar.output import output as output_module
    from recovar.simulation import simulate_scattering_potential as ssp
    from recovar.simulation.pdb_utils import AtomGroup

    if voxel_size is None:
        voxel_size = 4.25 * 128 / grid_size

    if path_fn is None:
        path_fn = path_symmetric

    if output_prefix is None:
        vols_dir = Path(output_dir) / "generated_volumes"
        output_module.mkdir_safe(str(vols_dir))
        volume_prefix = str(vols_dir / prefix_name)
    else:
        volume_prefix = str(output_prefix)
        output_module.mkdir_safe(str(Path(volume_prefix).parent))

    # Prepare structure
    group_coords, group_elements, fixed_pt = prepare_5nrl_subcomplexes(pdb_path)
    all_types = np.concatenate(group_elements)

    # B-factor scaling
    volume_shape = tuple(3 * [grid_size])
    B_fac_scaling = compute_bfactor_scaling(volume_shape, voxel_size, Bfactor)

    # Generate trajectory
    ts = np.linspace(0, max_rotation_degrees, n_volumes)
    for idx, t in enumerate(ts):
        coords_list = path_fn(t, group_coords, fixed_pt)
        combined_coords = np.concatenate(coords_list)

        ag = AtomGroup()
        ag.setCoords(combined_coords)
        ag.setElements(all_types)

        ft_mol = ssp.generate_molecule_spectrum_from_pdb_id(
            ag,
            voxel_size=voxel_size,
            grid_size=grid_size,
            force_symmetry=True,
            from_atom_group=True,
            do_center_atoms=False,
        )
        ft_mol = ft_mol.reshape(volume_shape) * B_fac_scaling
        vol = ftu.get_idft3(ft_mol.reshape(volume_shape)).real

        utils.write_mrc(
            f"{volume_prefix}{idx:04d}.mrc",
            vol.astype(np.float32),
            voxel_size=voxel_size,
        )
        if idx % 10 == 0:
            logger.info("Generated volume %d/%d (t=%.1f deg)", idx + 1, n_volumes, t)

    logger.info("Generated %d trajectory volumes at %s", n_volumes, volume_prefix)
    return volume_prefix
