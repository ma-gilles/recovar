from recovar import deconvolve_density, output, utils
import numpy as np
import argparse

def estimate_stable_states(density, latent_space_bounds, percent_top = 1, n_local_maxs = 3, file_path = None):
    file_path = file_path + '/' if file_path[-1] != '/' else file_path
    output.mkdir_safe(file_path)
    output.mkdir_safe(file_path + '/density/')
    # output.mkdir_safe(file_path + '/local_max_comp_viz/')

    latent_pts_z, latent_pts_grid = deconvolve_density.find_local_maxs_of_density(density, latent_space_bounds, percent_top = percent_top, n_local_maxs = n_local_maxs, plot_folder = file_path )
    output.plot_over_density(density, points = latent_pts_grid,  annotate=True, plot_folder = file_path + '/density/', cmap = 'inferno')
    np.savetxt(file_path + 'stable_state_all_coords.txt', latent_pts_z)
    for i in range(len(latent_pts_z)):
        np.savetxt(file_path + f'stable_state_{i}_coords.txt', latent_pts_z[i])


def parse_args():
    parser = argparse.ArgumentParser(description='Estimate stable states from density.')
    parser.add_argument('density', type=str, help='Path to the density file (.pkl), output by estimate_conformational_density.py.')
    parser.add_argument('-o', '--output', dest = "file_path", type=str, required=True, help='Path to save the output files.')
    parser.add_argument('--percent_top', type=float, default=1, help='Percentage of top density points to consider.')
    parser.add_argument('--n_local_maxs', type=int, default=3, help='Number of local maxima to find. If <1, will use whatever HDBSCAN finds.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dens_pkl = utils.pickle_load(args.density)
    density = dens_pkl['density']
    latent_space_bounds = dens_pkl['latent_space_bounds']
    estimate_stable_states(density, latent_space_bounds, args.percent_top, args.n_local_maxs, args.file_path)


