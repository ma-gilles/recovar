# If you want to extend and use recovar, you should import this first
import recovar.config 
import jax.numpy as jnp
import numpy as np

import os, argparse, time, logging
from recovar import output as o
from recovar import dataset, utils, noise
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save model",
    )

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser.add_argument('--zdim', type=list_of_ints, default=[1,2,4,10,20], help="Dimensions of latent variable. Default=1,2,4,10,20")

    # parser.add_argument(
    #     "--zdim", type=list, help="Dimension of latent variable"
    # )
    parser.add_argument(
        "--poses", type=os.path.abspath, required=True, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, required=True, help="CTF parameters (.pkl)"
    )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter particles by these indices",
    )

    group.add_argument(
        "--uninvert-data",
        dest="uninvert_data",
        default = "false",
        help="Invert data sign: options: true, false (default)",
    )


    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file",
    )
    group.add_argument(
        "--strip-prefix",
        help="Path prefix to strip from filenames in star file. Useful when star file contains longer paths than available on the system.",
    )
    group.add_argument(
            "--n-images",
            default = -1,
            dest="n_images",
            type=int,
            help="Number of images to use (should only use for quick run)",
        )
    
    group.add_argument(
            "--padding",
            type=int,
            default = 0,
            help="Real-space padding",
        )
    
    group.add_argument(
            "--halfsets",
            default = None,
            type=os.path.abspath,
            help="Path to a file with indices of split dataset (.pkl).",
        )


    group.add_argument(
            "--noise-model",
            dest = "noise_model",
            default = "radial",
            help="what noise model to use. Options are radial (default) computed from outside the masks, and white computed by power spectrum at high frequencies"
        )


    parser.add_argument(
        "--Bfactor",  type =float, default=0, help="0"
    )

    parser.add_argument(
        "--n-bins",  type =float, default=50, dest="n_bins",help="number of bins for kernel regression"
    )

    parser.add_argument(
        "--embedding", type=os.path.abspath, required=True, help="Image embeddings zs (.pkl), e.g. 00_cryodrgn256/z.24.pkl if you want to use a cryoDRGN embedding."
    )

    parser.add_argument(
        "--target", type=os.path.abspath, required=True, help="Target zs to evaluate the kernel regression (.txt)"
    )

    parser.add_argument(
        "--zdim1",  action="store_true", help="Whether dimension 1 embedding is used. This is an annoying corner case for np.loadtxt..."
    )

    parser.add_argument(
        "--tilt-series", action="store_true",  dest="tilt_series", help="Whether to use tilt_series."
    )

    return parser
    


def generate(args):
    st_time = time.time()

    o.mkdir_safe(args.outdir)
    logger.addHandler(logging.FileHandler(f"{args.outdir}/run.log"))
    logger.info(args)
    ind_split = dataset.figure_out_halfsets(args)

    dataset_loader_dict = dataset.make_dataset_loader_dict(args)

    cryos = dataset.get_split_datasets_from_dict(dataset_loader_dict, ind_split)

    # center = cryos[0].center    
    zs = utils.pickle_load(args.embedding)
    zs_split = [ zs[cryos[0].dataset_indices], zs[cryos[1].dataset_indices] ]
    zs = np.concatenate(zs_split)

    target = np.loadtxt(args.target)

    if args.zdim1:
        zdim =1
        target = target[:,None]
    else:
        zdim = target.shape[-1]
        if target.ndim ==1:
            # logger.warning("Did you mean to use --zdim1?")
            target = target[None]

    cov_zs = np.zeros((zs.shape[0], zs.shape[-1], zs.shape[-1]))
    cov_zs += np.eye(zs.shape[-1])
    # import pdb; pdb.set_trace()
    # noise_variance = np.ones(zs.shape[0])
    noise_variance, _ = noise.estimate_noise_variance(cryos[0], 100)
    noise_variance = np.ones(cryos[0].image_shape[0]//2-1) * noise_variance

    for cryo in cryos:
        cryo.noise = noise.RadialNoiseModel(noise_variance)
        
    output_folder = args.outdir
    o.compute_and_save_reweighted(cryos, target, zs, cov_zs, output_folder, args.Bfactor, args.n_bins)

    return 



def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    generate(args)

if __name__ == "__main__":
    main()