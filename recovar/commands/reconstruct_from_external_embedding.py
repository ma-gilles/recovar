import recovar.jax_config
import argparse
import logging
import os
import time

import numpy as np

from recovar import utils
from recovar.data_io import cryoem_dataset, halfsets
from recovar.output import output as o
from recovar.reconstruction import noise

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

    parser.add_argument('--zdim', type=list_of_ints, default=[1, 2, 4, 10, 20],
                        help="Dimensions of latent variable. Default=1,2,4,10,20")

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
        "--particle-ind",
        dest="tilt_ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter particles by these indices (only for tilt-series/cryo-ET)",
    )

    group.add_argument(
        "--uninvert-data",
        dest="uninvert_data",
        default="false",
        help="Invert data sign: options: true, false (default)",
    )

    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file",
    )
    group.add_argument(
        "--strip-prefix",
        help="Path prefix to strip from filenames in star file.",
    )
    group.add_argument(
        "--n-images",
        default=-1,
        dest="n_images",
        type=int,
        help="Number of images to use (should only use for quick run)",
    )

    group.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Real-space padding",
    )

    group.add_argument(
        "--halfsets",
        default=None,
        type=os.path.abspath,
        help="Halfset indices (.pkl). If omitted, reads _rlnRandomSubset from star file, or splits randomly",
    )

    group.add_argument(
        "--noise-model",
        dest="noise_model",
        default="radial",
        help="Noise model: radial (default) or white",
    )

    parser.add_argument(
        "--Bfactor", type=float, default=0, help="B-factor for sharpening"
    )

    parser.add_argument(
        "--n-bins", type=float, default=50, dest="n_bins",
        help="Number of bins for kernel regression",
    )

    parser.add_argument(
        "--embedding", type=os.path.abspath, required=True,
        help="Image embeddings (.pkl), e.g. z.24.pkl from cryoDRGN",
    )

    parser.add_argument(
        "--target", type=os.path.abspath, required=True,
        help="Target latent points to evaluate kernel regression (.txt)",
    )

    parser.add_argument(
        "--zdim1", action="store_true",
        help="Whether dimension-1 embedding is used",
    )

    parser.add_argument(
        "--tilt-series", action="store_true", dest="tilt_series",
        help="Whether to use tilt series",
    )

    group.add_argument(
        "--ntilts", default=None, type=int,
        help="Number of tilts to use per tilt series (default: all)",
    )

    group.add_argument(
        "--tilt-series-ctf", default=None,
        help="CTF mode for tilt series: cryoem, relion5, warp (default: auto)",
    )

    group.add_argument(
        "--angle-per-tilt", type=float, default=3.0,
        help="Angle per tilt in degrees (default: 3.0)",
    )

    group.add_argument(
        "--dose-per-tilt", type=float, default=2.9,
        help="Dose per tilt in e-/A^2 (default: 2.9)",
    )

    group.add_argument(
        "--premultiplied-ctf", action="store_true",
        help="Whether CTF is premultiplied in the data",
    )

    return parser

##TODO: I would like to make this function much easier to use. There should be a "basic interface"
## That any problem cna use, but I also would like a few specialized ones to make it easy to run on the output of
## cryodrgn, cryosparc + 3DVA, cryosparc + 3DFLex, RELION + (whatever their 3Dflex is called)
def generate(args):
    st_time = time.time()

    o.mkdir_safe(args.outdir)
    from recovar.utils.helpers import RobustFileHandler
    logger.addHandler(RobustFileHandler(f"{args.outdir}/run.log"))
    logger.info(args)

    if args.tilt_series_ctf is None:
        args.tilt_series_ctf = 'relion5' if args.tilt_series else 'cryoem'
        logger.info("Setting tilt_series_ctf to %s", args.tilt_series_ctf)

    ind_split = halfsets.resolve_halfset_indices(args)
    dataset_spec = halfsets.HalfsetDatasetSpec.from_args(args)
    ds = halfsets.load_halfset_dataset(dataset_spec, ind_split=ind_split)

    zs = utils.pickle_load(args.embedding)
    zs_split = [zs[ds.halfset_local_image_indices(0)], zs[ds.halfset_local_image_indices(1)]]
    zs = np.concatenate(zs_split)

    target = np.loadtxt(args.target)

    if args.zdim1:
        target = target[:, None]
    else:
        if target.ndim == 1:
            target = target[None]

    cov_zs = np.tile(np.eye(zs.shape[-1], dtype=zs.dtype), (zs.shape[0], 1, 1))

    half0_ds = ds.get_halfset_dataset(0, independent=False)
    noise_variance, _ = noise.estimate_noise_variance(half0_ds, 100)
    noise_variance = np.full(ds.image_shape[0] // 2 - 1, noise_variance)

    ds.set_radial_noise_model(noise_variance)

    o.compute_and_save_reweighted(ds, target, zs, cov_zs, args.outdir,
                                  args.Bfactor, args.n_bins)


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
