"""Simulate a RELION 5 subtomogram (2D-stack) dataset for EM/VDAM development.

Writes a RELION project that ``relion_refine --ios optimisation_set.star`` reads
(see :mod:`recovar.simulation.relion_tomo`), plus ``particles_2d.star`` for
``recovar pipeline --tilt-series``. The atomic-volume EM-development preset
(solvent contrast plus B-factor) is on by default; pass
``--no-atomic-solvent-correction`` for experimental or already-corrected maps.

Usage::

    recovar make_relion_tomo_dataset vols/vol 4.25 1200 -o project --grid-size 128 --optics-groups 2
"""

import argparse
import logging

from recovar.simulation import relion_tomo, solvent_contrast


def add_args(parser):
    parser.add_argument("volumes_path_root", help="Volume prefix: <prefix>0000.mrc, <prefix>0001.mrc, ...")
    parser.add_argument("voxel_size", type=float, help="Voxel size (A) of the volumes at --grid-size")
    parser.add_argument("n_particles", type=int, help="Number of particles")
    parser.add_argument("-o", "--output", required=True, help="Output RELION project directory")
    parser.add_argument("--grid-size", type=int, default=128, help="Image box (px) of optics group 1")
    parser.add_argument("--n-tomograms", type=int, default=4)
    parser.add_argument(
        "--optics-groups",
        type=int,
        choices=range(1, len(relion_tomo.DEFAULT_OPTICS_GROUPS) + 1),
        default=len(relion_tomo.DEFAULT_OPTICS_GROUPS),
        help="Number of optics groups taken from relion_tomo.DEFAULT_OPTICS_GROUPS",
    )
    parser.add_argument("--max-tilt", type=float, default=60.0)
    parser.add_argument("--tilt-step", type=float, default=3.0)
    parser.add_argument("--dose-per-tilt", type=float, default=3.0, help="e/A^2 per tilt")
    parser.add_argument("--snr", type=float, default=0.05)
    parser.add_argument("--hidden-tilt-fraction", type=float, default=0.0)
    parser.add_argument(
        "--origin-std-angstrom", type=float, default=0.0, help="Standard deviation of the ground-truth 3D offsets (A)"
    )
    parser.add_argument("--premultiplied-ctf", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    solvent_contrast.add_cli_arguments(parser, enabled_by_default=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    relion_tomo.generate_relion5_tomo_dataset(
        args.output,
        args.volumes_path_root,
        args.voxel_size,
        args.n_particles,
        grid_size=args.grid_size,
        n_tomograms=args.n_tomograms,
        optics_groups=relion_tomo.DEFAULT_OPTICS_GROUPS[: args.optics_groups],
        max_tilt=args.max_tilt,
        tilt_step=args.tilt_step,
        dose_per_tilt=args.dose_per_tilt,
        snr=args.snr,
        hidden_tilt_fraction=args.hidden_tilt_fraction,
        origin_std_angstrom=args.origin_std_angstrom,
        premultiplied_ctf=args.premultiplied_ctf,
        seed=args.seed,
        **solvent_contrast.kwargs_from_cli_args(args),
    )


if __name__ == "__main__":
    main()
