import recovar.config 
import logging
import numpy as np
from recovar import output as o
from recovar import deconvolve_density, utils, output
from scipy.spatial import distance_matrix
import os, argparse
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from recovar import parser_args

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate conformational density from recovar results")
    parser.add_argument("recovar_result_dir", type=str, help="Directory containing recovar results provided to pipeline.py")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the density estimation results. Default = recovar_result_dir/density/")
    parser.add_argument("--deconvolved_dim", type=int, default=4, help="Dimension of deconvolved space (default 4). This is the dimension of the PC space where the conformational density is estimated. The runtime increases exponentially with this number, so <=5 is recommended.")
    parser.add_argument("--z_dim_used", type=int, default=4, help="Dimension of latent variable used (default 4)")
    parser.add_argument("--percentile_reject", type=int, default=10, help="Percentile of data to reject b/c they have large covariance (default 10%)")
    parser.add_argument("--num_disc_points", type=int, default=50, help="Number of discretization points in each dimension for the grid density estimation (default 50) = 50^4 points")
    parser.add_argument("--alphas", type=float, nargs='*', default=None, help="List of alphas for regularization (default (1e-9, 1e-8, ..., 1e1)")
    parser.add_argument("--percentile_bound", type=int, default=1, help="Rejects zs with coordinates above this bound for deciding the bounds of the grid (default 1 =1%)")
    return parser.parse_args()



def estimate_conformational_density(recovar_result_dir, output_dir = None, deconvolved_dim=4, z_dim_used=4, percentile_reject=10, num_disc_points=50, alphas=None, percentile_bound=1):
    output_dir = recovar_result_dir + '/density/' if output_dir is None else output_dir
    output.mkdir_safe(output_dir )

    pipeline_output = o.PipelineOutput(recovar_result_dir + '/')
    percentile_reject = 10
    alphas = np.flip(np.logspace(-9, 1, 11)) if alphas is None else np.array(alphas)

    zdim = f"{z_dim_used}_noreg"
    lbfgsb_sols, alphas, cost, reg_cost, density, total_covar, grids, bounds = deconvolve_density.get_deconvolved_density(
        pipeline_output, zdim=zdim, pca_dim_max=deconvolved_dim, percentile_reject=percentile_reject, kernel_option='sampling', num_points=num_disc_points, alphas=alphas, percentile_bound=percentile_bound, save_to_file=None
    )

    deconvolve_density.plot_density(lbfgsb_sols, density, alphas)
    plt.savefig(output_dir + '/all_densities.png')


    from kneed import KneeLocator
    kn = KneeLocator(np.log10(1/alphas), np.log(cost), curve='convex', direction='decreasing')
    # kn.plot_knee()
    # plt.savefig(output_dir + 'knee_locator.png', transparent = True)
    logger.info(f"Knee point: {kn.knee}")
    knee_idx = np.argmin(np.abs(np.log10(1/alphas) - kn.knee))
    logger.info(f"Knee point: alpha = {10**(1/kn.knee)} at idx = {knee_idx}")

    output.mkdir_safe(output_dir + '/all_densities/' )
    for idx in range(len(lbfgsb_sols)):
        utils.pickle_dump({ 'density' : lbfgsb_sols[idx], 'latent_space_bounds' : bounds, 'alpha' : alphas[idx] }, output_dir + '/all_densities/' + f'deconv_density_{idx}.pkl')
    idx = knee_idx
    utils.pickle_dump({ 'density' : lbfgsb_sols[idx], 'latent_space_bounds' : bounds, 'alpha' : alphas[idx] }, output_dir + f'deconv_density_knee.pkl')
    plt.figure(figsize = (12,10))
    for i, (alpha, c) in enumerate(zip(alphas, cost)):
        plt.text(alpha, c, str(i), fontsize=18)
    plt.loglog(alphas, cost, '-o')
    plt.loglog(np.ones(2)*alphas[idx], [min(cost), max(cost)], '--', color='black')
    plt.text(alphas[idx], min(cost), f'knee point: {alphas[idx]:.2e}, idx ={idx}', rotation=90, verticalalignment='bottom')
    plt.ylabel('Cost')
    plt.xlabel('Lambda (regularization parameter)')

    plt.gca().invert_xaxis()
    plt.savefig(output_dir + 'Lcurve.png', transparent = True)


def main():
    args = parse_args()
    estimate_conformational_density(
        recovar_result_dir=args.recovar_result_dir,
        output_dir=args.output_dir,
        deconvolved_dim=args.deconvolved_dim,
        z_dim_used=args.z_dim_used,
        percentile_reject=args.percentile_reject,
        num_disc_points=args.num_disc_points,
        alphas=args.alphas,
        percentile_bound=args.percentile_bound
    )

if __name__ == "__main__":
    main()
