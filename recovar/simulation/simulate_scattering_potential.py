#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:04:24 2021

@author: marcaurele
"""

import json
import logging
import os

import finufft

logger = logging.getLogger(__name__)

# Try prody first; fall back to built-in pdb_utils
try:
    import prody as _prody

    _HAS_PRODY = True
except ImportError:
    _HAS_PRODY = False

from recovar.simulation.pdb_utils import AtomGroup as _AtomGroup, parse_pdb as _parse_pdb, write_pdb as _write_pdb


def _parsePDB(path_or_id):
    """Parse PDB using prody if available, otherwise built-in parser."""
    if _HAS_PRODY:
        return _prody.parsePDB(path_or_id)
    return _parse_pdb(path_or_id)


def _writePDB(filepath, atoms):
    """Write PDB using prody if available, otherwise built-in writer."""
    if _HAS_PRODY:
        _prody.writePDB(filepath, atoms)
    else:
        _write_pdb(filepath, atoms)


def _newAtomGroup():
    """Create AtomGroup using prody if available, otherwise built-in."""
    if _HAS_PRODY:
        return _prody.AtomGroup()
    return _AtomGroup()


import numpy as np
from collections import defaultdict
import recovar.core.fourier_transform_utils as fourier_transform_utils

FINUFFT_EPS = 1e-8

# This code simulated the scattering potential of a molecule using a nufft, and the atomic positions, and the atomic shape function that was experimentally determined in a paper

## The real space grid is always defined
# [ -N/2,- N/2 +1 ......, -N/2 + N-1] * voxel_size
# =[ -N/2,- N/2 +1 ......, N/2 -1] * voxel_size
# And the frequency grid
# if k is even:
# [ -N/2,- N/2 +1 ......, N/2 -1] / (voxel_size * N )
# and if  k odd:
# [ -(N-1)/2 ,- N/2 +1 ......, (N-1)/2] / (voxel_size * N )

# These are the atomic scattering potential coefficients tabulated in:
# Peng, L-M., et al. "Robust parameterization of elastic and absorptive electron atomic scattering factors."
# Acta Crystallographica Section A: Foundations of Crystallography 52.2 (1996): 257-276.
atom_coeff_path = os.path.join(os.path.dirname(__file__), "..", "assets", "atom_coeffs_extended.json")
with open(atom_coeff_path, "r") as f:
    atom_coeffs = json.load(f)

atom_coeffs2 = {}
for key in atom_coeffs:
    atom_coeffs2[key.upper()] = atom_coeffs[key]
atom_coeffs = atom_coeffs2


def generate_bag_of_atom_projection(grid_size, radius, voxel_size, N_atoms, atom_shape_fn):
    ft_mol = generate_synthetic_spectrum_of_molecule(radius, grid_size, voxel_size, atom_shape_fn, N_atoms)
    image = np.real(
        fourier_transform_utils.get_inverse_fourier_transform(ft_mol[grid_size // 2], voxel_size=voxel_size)
    )
    return image


def get_fourier_transform_of_molecules_on_k_grid(
    atom_coords, weights, voxel_size, grid_size, eps=FINUFFT_EPS, jax_backend=False
):
    normalized_atom_coords = (atom_coords) / grid_size * (2 * np.pi) / voxel_size

    # Since x \in [ -pi, pi], this already does the same scaling that the DFT_to_FT_scaling does
    if jax_backend:
        import jax_finufft

        ft = jax_finufft.nufft1(
            (grid_size, grid_size, grid_size),
            weights,
            normalized_atom_coords[:, 0],
            normalized_atom_coords[:, 1],
            normalized_atom_coords[:, 2],
            iflag=-1,
            eps=eps,
        )
    else:
        ft = finufft.nufft3d1(
            normalized_atom_coords[:, 0],
            normalized_atom_coords[:, 1],
            normalized_atom_coords[:, 2],
            weights,
            (grid_size, grid_size, grid_size),
            isign=-1,
            eps=eps,
        )
    return ft


def get_fourier_transform_of_molecules_at_freq_coords(
    atom_coords, weights, voxel_size, freq_coords, eps=FINUFFT_EPS, jax_backend=False
):

    scale = np.max(np.abs(atom_coords))

    normalized_atom_coords = (atom_coords * 2 * np.pi / scale).astype(np.dtype("float64"))
    normalized_freqs = (freq_coords * scale).astype(np.dtype("float64"))
    weights = weights.astype(np.dtype("complex128"))
    out = np.zeros_like(normalized_freqs).astype(np.dtype("complex128"))

    x = normalized_atom_coords[:, 0].astype(np.dtype("float64"))
    y = normalized_atom_coords[:, 1].astype(np.dtype("float64"))
    z = normalized_atom_coords[:, 2].astype(np.dtype("float64"))
    c = weights.astype(np.dtype("complex128"))
    s = normalized_freqs[:, 0].astype(np.dtype("float64"))
    t = normalized_freqs[:, 1].astype(np.dtype("float64"))
    u = normalized_freqs[:, 2].astype(np.dtype("float64"))

    ft = finufft.nufft3d3(x, y, z, c, s, t, u, out=None, isign=-1, eps=eps)

    if np.isnan(np.sum(ft)):
        with open("nufft_nans", "wb") as f:
            import pickle

            params = {}
            params["coords"] = normalized_atom_coords
            params["freqs"] = normalized_freqs
            params["weights"] = weights
            params = pickle.dump(params, f)

    return ft


def get_fourier_transform_of_volume_at_freq_coords(volume, freq_coords, voxel_size, eps=FINUFFT_EPS, jax_backend=False):
    scaled_freq_coords = freq_coords * 2 * np.pi * voxel_size
    ft = finufft.nufft3d2(
        scaled_freq_coords[:, 0], scaled_freq_coords[:, 1], scaled_freq_coords[:, 2], volume, isign=-1, eps=eps
    )
    return ft


def get_exponent_and_constant_of_gaussian_FT(sigma, dim=3):
    # real space exp
    a = 1 / (2 * sigma**2)
    tau = np.pi**2 / a

    cst = ((1 / (sigma * np.sqrt(2 * np.pi))) * np.sqrt(np.pi / a)) ** dim
    return tau, cst


def gaussian_atom_shape_fn(psi, sigma):
    rs = np.linalg.norm(psi, axis=-1)

    expo, cst = get_exponent_and_constant_of_gaussian_FT(sigma, dim=3)

    return cst * np.exp(-(rs**2) * expo)


def compute_gaussian_on_k_grid(sigma, grid_size, voxel_size):
    rs = fourier_transform_utils.get_grid_of_radial_distances(3 * [grid_size], voxel_size=voxel_size, scaled=True)
    expo, cst = get_exponent_and_constant_of_gaussian_FT(sigma, dim=3)
    return cst * np.exp(-(rs**2) * expo)


def get_gaussian_fn_on_k(sigma):
    expo, cst = get_exponent_and_constant_of_gaussian_FT(sigma, dim=3)

    def gaussian_fn(x):
        return cst * np.exp(-(np.linalg.norm(x, axis=-1) ** 2) * expo)

    return gaussian_fn


def generate_synthetic_spectrum_of_molecule(radius, grid_size, voxel_size=1, atom_shape_fn=None, N_atoms=None):
    atom_shape_fn = (lambda x: gaussian_atom_shape_fn(x, 1)) if atom_shape_fn is None else atom_shape_fn
    N_atoms = choose_number_of_atoms(radius) if N_atoms is None else N_atoms

    if radius >= (grid_size / 2 * voxel_size):
        raise ValueError(f"radius ({radius}) must be < grid_size/2 * voxel_size ({grid_size / 2 * voxel_size})")

    atom_coords = get_random_points_in_unit_ball(N_atoms) * radius
    return generate_spectrum_of_molecule_from_atom_coords(atom_coords, grid_size, voxel_size, atom_shape_fn)


def generate_spectrum_of_molecule_from_atom_coords(
    atom_coords, voxel_size, grid_size, atom_shape_fn, jax_backend=False
):
    weights = np.ones(atom_coords.shape[0], dtype=atom_coords.dtype).astype(complex)
    fourier_transform = get_fourier_transform_of_molecules_on_k_grid(
        atom_coords, weights, voxel_size, grid_size, jax_backend=jax_backend
    )

    k_coords = fourier_transform_utils.get_k_coordinate_of_each_pixel(
        3 * [grid_size], voxel_size=voxel_size, scaled=True
    )
    weight = atom_shape_fn(k_coords).reshape(fourier_transform.shape)

    return fourier_transform * weight


def generate_spectrum_of_molecule_from_atom_coords_at_freq_coords(
    atom_coords, voxel_size, freq_coords, atom_shape_fn, jax_backend=False
):
    weights = np.ones(atom_coords.shape[0], dtype=atom_coords.dtype).astype(complex)
    fourier_transform = get_fourier_transform_of_molecules_at_freq_coords(
        atom_coords, weights, voxel_size, freq_coords=freq_coords, jax_backend=jax_backend
    )
    weight = atom_shape_fn(freq_coords).reshape(fourier_transform.shape)
    return fourier_transform * weight


def five_gaussian_atom_shape(psi, coeffs):
    a = coeffs[:5]
    b = coeffs[5:]
    if psi.ndim == 1:
        rs = psi
    else:
        rs = np.linalg.norm(psi, axis=-1)

    potential = np.zeros(psi.shape[0])
    for k in range(5):
        potential += a[k] * np.exp(-b[k] * rs**2)
    return potential


def generate_volume_from_pdb(molecule, voxel_size, grid_size):
    atoms = _parsePDB(molecule)
    return generate_volume_from_atoms(atoms, voxel_size=voxel_size, grid_size=grid_size)


def generate_potential_at_freqs_from_pdb(molecule, voxel_size, freq_coords):
    atoms = center_atoms(_parsePDB(molecule))
    return generate_volume_from_atoms(atoms, voxel_size=voxel_size, grid_size=None, freq_coords=freq_coords)


def generate_potential_at_freqs_from_atoms(atoms, voxel_size, freq_coords):
    return generate_volume_from_atoms(atoms, voxel_size=voxel_size, grid_size=None, freq_coords=freq_coords)


def generate_volume_from_atom_positions_and_types(
    atom_coords, atom_types, voxel_size, grid_size=None, freq_coords=None, jax_backend=False
):
    use_freq_coords = freq_coords is not None
    atom_coords = np.asarray(atom_coords)
    atom_types = np.asarray(atom_types).reshape(-1)
    if atom_coords.ndim != 2 or atom_coords.shape[1] != 3:
        raise ValueError(f"atom_coords must have shape (N, 3), got {atom_coords.shape}")
    if atom_coords.shape[0] != atom_types.shape[0]:
        raise ValueError(f"atom_coords and atom_types length mismatch: {atom_coords.shape[0]} vs {atom_types.shape[0]}")
    # Accept mixed/lowercase element names from synthetic inputs by canonicalizing
    # to the uppercase keys used in atom_coeffs.
    atom_types = np.char.upper(atom_types.astype(str))
    unknown = sorted({name for name in np.unique(atom_types) if name not in atom_coeffs})
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(f"Unknown atom types: {preview}. Expected element symbols present in recovar atom_coeffs.")
    out_real_dtype = np.result_type(atom_coords.dtype, np.float32)
    if use_freq_coords:
        out_real_dtype = np.result_type(out_real_dtype, np.asarray(freq_coords).dtype)
    if np.dtype(out_real_dtype).itemsize <= np.dtype(np.float32).itemsize:
        out_complex_dtype = np.complex64
    else:
        out_complex_dtype = np.complex128

    atom_indices = defaultdict(list)
    for k, atype in enumerate(atom_types):
        atom_indices[atype].append(k)

    atoms_grouped_by_elements = {}
    for atom_name in atom_indices:
        xx = np.array(atom_indices[atom_name])
        atoms_grouped_by_elements[atom_name] = atom_coords[xx]

    # Compute density for each kind of element
    output_shape = freq_coords.shape[0] if use_freq_coords else 3 * [grid_size]
    density = np.zeros(output_shape, dtype=out_complex_dtype)
    for atom_name in atoms_grouped_by_elements:
        atom_shape_fn = lambda x: five_gaussian_atom_shape(x, atom_coeffs[atom_name])
        if use_freq_coords:
            xx = generate_spectrum_of_molecule_from_atom_coords_at_freq_coords(
                atoms_grouped_by_elements[atom_name],
                voxel_size,
                freq_coords=freq_coords,
                atom_shape_fn=atom_shape_fn,
                jax_backend=jax_backend,
            )
            density = density + xx.astype(out_complex_dtype, copy=False)
        else:
            xx = generate_spectrum_of_molecule_from_atom_coords(
                atoms_grouped_by_elements[atom_name],
                voxel_size,
                grid_size=grid_size,
                atom_shape_fn=atom_shape_fn,
                jax_backend=jax_backend,
            )
            density = density + xx.astype(out_complex_dtype, copy=False)

        if np.isnan(np.sum(density)):
            raise ValueError(f"NaN in density after processing atom '{atom_name}'")
    return density


def one_fn(x):
    return np.ones_like(x[:, 0])


def generate_volume_from_atoms(atoms, voxel_size=None, grid_size=None, freq_coords=None, jax_backend=False):

    atom_coords = atoms.getCoords()
    if freq_coords is not None:
        atom_coords = atom_coords.astype(freq_coords.dtype)
    # Group atoms by elements
    atom_types = atoms.getData("element")
    return generate_volume_from_atom_positions_and_types(
        atom_coords,
        atom_types,
        voxel_size=voxel_size,
        grid_size=grid_size,
        freq_coords=freq_coords,
        jax_backend=jax_backend,
    )


def get_average_atom_shape_fn(atoms):

    atom_coords = atoms.getCoords()
    atom_coords = atom_coords - np.mean(atom_coords, axis=0)

    # Group atoms by elements
    atom_names = atoms.getData("element")
    atom_indices = defaultdict(list)
    for k, aname in enumerate(atom_names):
        atom_indices[aname].append(k)

    atom_shape_fns = {}
    atom_proportions = {}

    for atom_name in atom_indices:
        atom_shape_fns[atom_name] = get_atom_shape_fn(atom_name)
        atom_proportions[atom_name] = len(atom_indices[atom_name]) / atom_coords.shape[0]

    def average_atom_shape(psi):
        if psi.ndim == 1:
            rs = psi
        else:
            rs = np.linalg.norm(psi, axis=-1)

        density = np.zeros(rs.shape)
        for atom_name in atom_shape_fns:
            density += atom_proportions[atom_name] * atom_shape_fns[atom_name](rs)
        return density

    return average_atom_shape


def get_atom_shape_fn(atom_name):
    def shape_fn(x):
        return five_gaussian_atom_shape(x, atom_coeffs[atom_name])

    return shape_fn


def generate_synthetic_atomgroup(radius):
    N_atoms = choose_number_of_atoms(radius)
    atom_coor = get_random_points_in_unit_ball(N_atoms) * radius

    ATOM_TYPE = "C"
    atoms = _newAtomGroup()
    atoms.setCoords(atom_coor)
    atoms.setNames(np.array(N_atoms * [ATOM_TYPE], dtype="<U6"))
    atoms.setElements(np.array(N_atoms * [ATOM_TYPE], dtype="<U6"))

    # ProDy compatibility: set internal flags if using prody AtomGroup
    if _HAS_PRODY:
        ATOMGROUP_FLAGS = ["hetatm", "pdbter", "ca", "calpha", "protein", "aminoacid", "nucleic", "hetero"]
        atoms._flags = {}
        for key in ATOMGROUP_FLAGS:
            atoms._flags[key] = np.zeros(N_atoms, dtype=bool)
    return atoms


def generate_synthetic_pdb_dataset(directory, radius, n_samples=1):

    if not os.path.exists(directory):
        os.makedirs(directory)

    for k in range(n_samples):
        atoms = generate_synthetic_atomgroup(radius)
        filename = directory + "/mol" + str(k) + ".pdb"
        _writePDB(filename, atoms)


# From the paper
def choose_number_of_atoms(R):
    ro = 0.8  # average protein density
    atom_per_weight = 9.1 / 110  # For Carbon
    return int(np.round(4 * np.pi / 3 * R**3 * ro * atom_per_weight))


def get_random_points_in_unit_ball(N):
    p = np.random.normal(0, 1, (N, 3))
    norms = np.linalg.norm(p, axis=1)
    r = np.random.random(N) ** (1.0 / 3)
    random_in_sphere = p * r[:, np.newaxis] / norms[:, np.newaxis]
    return random_in_sphere


def simulate_scattering_potential_on_grid(radius, N, voxel_size=1):
    N_atoms = choose_number_of_atoms(radius)
    coords = get_random_points_in_unit_ball(N_atoms) * radius
    coords_on_grid = (np.round(coords / voxel_size) + N / 2).astype(int)
    grid = np.zeros((N, N, N))
    for coor in coords_on_grid:
        grid[coor[0], coor[1], coor[2]] += 1
    return grid


def get_center_coord_offset(coords):
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    offset = min_coords + (max_coords - min_coords) / 2
    return offset


def center_atoms(atoms):
    # Center coords
    coords = atoms.getCoords()
    offset = get_center_coord_offset(coords)
    coords = coords - offset
    atoms.setCoords(coords)
    return atoms


def generate_molecule_spectrum_from_pdb_id(
    molecule, voxel_size, grid_size, force_symmetry=True, verbosity="none", from_atom_group=False, do_center_atoms=None
):

    do_center_atoms = not from_atom_group if do_center_atoms is None else do_center_atoms

    if not from_atom_group:
        atoms = _parsePDB(molecule)
    else:
        atoms = molecule

    if do_center_atoms:
        atoms = center_atoms(atoms)

    ft_mol = generate_volume_from_atoms(atoms, grid_size=grid_size, voxel_size=voxel_size)

    if (grid_size % 2) == 0 and force_symmetry:
        ft_mol[0] *= 0
        ft_mol[:, 0] *= 0
        ft_mol[:, :, 0] *= 0

    return ft_mol


def voltage_to_wavelength(voltage):
    # Borrowed from ASPIRE https://github.com/ComputationalCryoEM/ASPIRE-Python
    """
    Convert from electron voltage to wavelength.
    :param voltage: float, The electron voltage in kV.
    :return: float, The electron wavelength in nm.
    """
    return 12.2643247 / np.sqrt(voltage * 1e3 + 0.978466 * voltage**2)
