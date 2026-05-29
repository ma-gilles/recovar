"""Render a morph-trajectory PDB stack into the walkthrough's 01_raw_volumes/ layout.

Takes a directory of PDB files (one per trajectory frame, glob ``morph_*.pdb``),
centers every frame on the global centroid across all frames (so internal
motion is preserved while the molecule sits at box center), and renders each
frame as a B-factored 3-D scattering potential MRC at the requested grid /
voxel size.

The resulting ``<output_dir>/01_raw_volumes/vol{idx:04d}.mrc`` layout is the
same one ``benchmark_kernel_bandwidth_1d.py`` produces in its first stage, so
running this script first lets the walkthrough's "01_raw_volumes" step
short-circuit and reuse these volumes.

Default PDB dir is the user's 100-frame spike-RBD morph set; pass --pdb-dir to
point at a different stack.

Usage:
    python -m recovar.commands.render_spike_morph_volumes <output_dir> \\
        [--pdb-dir DIR] [--grid-size 128] [--voxel-size 2.0] [--bfactor 60]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import recovar.jax_config  # noqa: F401
from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.simulation import simulate_scattering_potential as ssp
from recovar.simulation.pdb_utils import AtomGroup
from recovar.simulation.trajectory_generation import compute_bfactor_scaling

DEFAULT_PDB_DIR = Path("/home/mg6942/myscratch/spike_pdb_motion")
DEFAULT_GRID_SIZE = 128
DEFAULT_VOXEL_SIZE = 2.0
DEFAULT_BFACTOR = 60.0

log = logging.getLogger(__name__)


def parse_pdb_atoms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords: list[tuple[float, float, float]] = []
    elements: list[str] = []
    with path.open() as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            elem = line[76:78].strip() or line[12:16].strip()[0]
            elements.append(elem)
    return np.asarray(coords, dtype=np.float64), np.asarray(elements)


def render_stack(
    pdb_dir: Path,
    out_dir: Path,
    grid_size: int = DEFAULT_GRID_SIZE,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    bfactor: float = DEFAULT_BFACTOR,
    glob_pattern: str = "morph_*.pdb",
) -> Path:
    files = sorted(pdb_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"no PDB files in {pdb_dir} matching {glob_pattern!r}")
    log.info("Found %d PDB frames in %s", len(files), pdb_dir)

    coords_stack: list[np.ndarray] = []
    elements_ref: np.ndarray | None = None
    for f in files:
        c, e = parse_pdb_atoms(f)
        coords_stack.append(c)
        if elements_ref is None:
            elements_ref = e
        elif len(e) != len(elements_ref):
            raise ValueError(f"atom count mismatch: {f.name} has {len(e)} vs {len(elements_ref)}")
    assert elements_ref is not None
    coords_arr = np.stack(coords_stack)

    grand_mean = coords_arr.reshape(-1, 3).mean(axis=0)
    log.info("Grand-mean centroid (Å): %s", grand_mean)
    coords_arr = coords_arr - grand_mean

    raw_dir = out_dir / "01_raw_volumes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    volume_shape = (grid_size, grid_size, grid_size)
    bfac_scale = compute_bfactor_scaling(volume_shape, voxel_size, bfactor)

    for idx, coords in enumerate(coords_arr):
        ag = AtomGroup()
        ag.setCoords(coords.astype(np.float64))
        ag.setElements(elements_ref)

        ft_mol = ssp.generate_molecule_spectrum_from_pdb_id(
            ag,
            voxel_size=voxel_size,
            grid_size=grid_size,
            force_symmetry=True,
            from_atom_group=True,
            do_center_atoms=False,
        )
        ft_mol = ft_mol.reshape(volume_shape) * bfac_scale
        vol = ftu.get_idft3(ft_mol).real

        out_path = raw_dir / f"vol{idx:04d}.mrc"
        utils.write_mrc(str(out_path), vol.astype(np.float32), voxel_size=voxel_size)
        if idx % 10 == 0 or idx == len(coords_arr) - 1:
            log.info("wrote %s  (mean=%.4g std=%.4g)", out_path.name, float(vol.mean()), float(vol.std()))

    log.info(
        "Done. Rendered %d volumes at %.3f Å/voxel (B=%.1f) into %s", len(coords_arr), voxel_size, bfactor, raw_dir
    )
    return raw_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="Run dir; volumes go to <output_dir>/01_raw_volumes/")
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR, help="Directory of morph_*.pdb files")
    parser.add_argument("--glob", type=str, default="morph_*.pdb", help="PDB filename glob pattern")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE, help="Å per voxel")
    parser.add_argument("--bfactor", type=float, default=DEFAULT_BFACTOR, help="Rendering B-factor (Å²)")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    render_stack(
        pdb_dir=args.pdb_dir,
        out_dir=args.output_dir.resolve(),
        grid_size=args.grid_size,
        voxel_size=args.voxel_size,
        bfactor=args.bfactor,
        glob_pattern=args.glob,
    )


if __name__ == "__main__":
    main()
