import argparse
import logging
import os

import numpy as np

from recovar import utils

logger = logging.getLogger(__name__)
from recovar.heterogeneity import deconvolve_density
from recovar.output import output


def estimate_stable_states(density, latent_space_bounds, percent_top=1, n_local_maxs=3, file_path=None):
    file_path = os.path.normpath(file_path)
    output.mkdir_safe(file_path)
    density_dir = os.path.join(file_path, "density")
    output.mkdir_safe(density_dir)

    latent_pts_z, latent_pts_grid = deconvolve_density.find_local_maxs_of_density(
        density,
        latent_space_bounds,
        percent_top=percent_top,
        n_local_maxs=n_local_maxs,
        plot_folder=file_path,
    )
    output.plot_over_density(density, points=latent_pts_grid, annotate=True, plot_folder=density_dir, cmap="inferno")
    np.savetxt(os.path.join(file_path, "stable_state_all_coords.txt"), latent_pts_z)
    for i, pts in enumerate(latent_pts_z):
        np.savetxt(os.path.join(file_path, f"stable_state_{i}_coords.txt"), pts)


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate stable states from density.")
    parser.add_argument(
        "density", type=str, help="Path to the density file (.pkl), output by estimate_conformational_density.py."
    )
    parser.add_argument(
        "-o", "--output", dest="file_path", type=str, required=True, help="Path to save the output files."
    )
    parser.add_argument("--percent_top", type=float, default=1, help="Percentage of top density points to consider.")
    parser.add_argument(
        "--n_local_maxs",
        type=int,
        default=3,
        help="Number of local maxima to find. If <1, will use whatever HDBSCAN finds.",
    )
    from recovar.utils.parser_args import add_output_name_arg, add_project_arg

    add_project_arg(parser)
    add_output_name_arg(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    # job_context checks for args.output; this parser uses dest="file_path"
    args.output = args.file_path

    from recovar.project.job_context import job_context

    with job_context(args, "estimate_stable_states") as ctx:
        args.file_path = ctx.output_dir
        dens_pkl = utils.pickle_load(args.density)
        density = dens_pkl["density"]
        latent_space_bounds = dens_pkl["latent_space_bounds"]
        estimate_stable_states(density, latent_space_bounds, args.percent_top, args.n_local_maxs, args.file_path)


if __name__ == "__main__":
    main()
