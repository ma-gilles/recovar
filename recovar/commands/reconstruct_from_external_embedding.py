import argparse
import logging
import os
import time
import warnings

import numpy as np

from recovar import utils
from recovar.data_io import halfsets
from recovar.output import output as o
from recovar.reconstruction import noise

logger = logging.getLogger(__name__)


def _load_external_embeddings(path):
    """Load external latent coordinates from pickle or NumPy-native formats."""
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    if suffix not in (".pkl", ".npy", ".npz"):
        raise ValueError("Embedding should be a .pkl, .npy, or .npz file")
    zs = utils.load_serialized_payload(
        path,
        name="embedding",
        npz_keys=("latent_coords", "embedding", "embeddings", "zs"),
    )

    zs = np.asarray(zs)
    if not np.issubdtype(zs.dtype, np.number):
        raise ValueError("Embedding array must be numeric.")
    try:
        zs = zs.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding array must be numeric.") from exc
    if zs.size == 0:
        raise ValueError("Embedding array is empty.")
    if not np.all(np.isfinite(zs)):
        raise ValueError("Embedding array contains non-finite values (NaN/Inf).")
    if zs.ndim == 1:
        zs = zs[:, None]
    if zs.ndim != 2:
        raise ValueError(f"Embedding array must have shape (n_images, zdim); got {zs.shape}")
    return zs


def _load_target_points(path):
    """Load target latent points from text, pickle, or NumPy-native formats."""
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    if suffix not in (".txt", ".pkl", ".npy", ".npz"):
        raise ValueError("Target latent points should be a .txt, .pkl, .npy, or .npz file")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        target = utils.load_serialized_payload(
            path,
            name="target latent points",
            allow_text=True,
            npz_keys=("target", "targets", "latent_points", "target_zs", "zs", "points"),
        )

    target = np.asarray(target)
    try:
        target = target.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Target latent points must be numeric.") from exc
    if target.ndim == 0:
        target = target.reshape(1)
    if target.size == 0:
        raise ValueError("Target latent points array is empty.")
    if not np.all(np.isfinite(target)):
        raise ValueError("Target latent points contain non-finite values (NaN/Inf).")
    if target.ndim > 2:
        raise ValueError(f"Target latent points must be 1D or 2D; got {target.shape}")
    return target


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
        return list(map(int, arg.split(",")))

    parser.add_argument(
        "--zdim",
        type=list_of_ints,
        default=[1, 2, 4, 10, 20],
        help="Dimensions of latent variable. Default=1,2,4,10,20",
    )

    parser.add_argument("--poses", type=os.path.abspath, required=True, help="Image poses (.pkl/.npy/.npz)")
    parser.add_argument(
        "--ctf",
        metavar="pkl|npy|npz",
        type=os.path.abspath,
        required=True,
        help="CTF parameters (.pkl/.npy/.npz)",
    )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--ind",
        type=os.path.abspath,
        metavar="FILE",
        help="Filter particles by these indices (.pkl/.npy/.npz/.txt)",
    )

    group.add_argument(
        "--particle-ind",
        dest="tilt_ind",
        type=os.path.abspath,
        metavar="FILE",
        help="Filter particles by these indices (.pkl/.npy/.npz/.txt; only for tilt-series/cryo-ET)",
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
        help="Halfset indices (.pkl/.npy/.npz/.txt). If omitted, reads _rlnRandomSubset from star file, or splits randomly",
    )

    group.add_argument(
        "--noise-model",
        dest="noise_model",
        default="radial",
        help="Noise model: radial (default) or white",
    )

    parser.add_argument("--Bfactor", type=float, default=0, help="B-factor for sharpening")

    parser.add_argument(
        "--n-bins",
        type=float,
        default=50,
        dest="n_bins",
        help="Number of bins for kernel regression",
    )

    parser.add_argument(
        "--embedding",
        type=os.path.abspath,
        required=True,
        help="Image embeddings (.pkl/.npy/.npz), e.g. z.24.pkl from cryoDRGN or model/zdim_N/latent_coords.npy",
    )

    parser.add_argument(
        "--target",
        type=os.path.abspath,
        required=True,
        help="Target latent points to evaluate kernel regression (.txt/.pkl/.npy/.npz)",
    )

    parser.add_argument(
        "--zdim1",
        action="store_true",
        help="Whether dimension-1 embedding is used",
    )

    parser.add_argument(
        "--tilt-series",
        action="store_true",
        dest="tilt_series",
        help="Whether to use tilt series",
    )

    group.add_argument(
        "--ntilts",
        default=None,
        type=int,
        help="Number of tilts to use per tilt series (default: all)",
    )

    group.add_argument(
        "--tilt-series-ctf",
        default=None,
        help="CTF mode for tilt series: cryoem, relion5, warp (default: auto)",
    )

    group.add_argument(
        "--angle-per-tilt",
        type=float,
        default=3.0,
        help="Angle per tilt in degrees (default: 3.0)",
    )

    group.add_argument(
        "--dose-per-tilt",
        type=float,
        default=2.9,
        help="Dose per tilt in e-/A^2 (default: 2.9)",
    )

    group.add_argument(
        "--premultiplied-ctf",
        action="store_true",
        help="Whether CTF is premultiplied in the data",
    )

    from recovar.utils.parser_args import add_output_name_arg, add_project_arg

    add_project_arg(parser)
    add_output_name_arg(parser)

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
        args.tilt_series_ctf = "relion5" if args.tilt_series else "cryoem"
        logger.info("Setting tilt_series_ctf to %s", args.tilt_series_ctf)

    ind_split = halfsets.resolve_halfset_indices(args)
    dataset_spec = halfsets.HalfsetDatasetSpec.from_args(args)
    ds = halfsets.load_halfset_dataset(dataset_spec, ind_split=ind_split)

    # External embeddings are expected in dataset-local (original) order.
    zs = _load_external_embeddings(args.embedding)

    target = _load_target_points(args.target)

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

    o.compute_and_save_reweighted(ds, target, zs, cov_zs, args.outdir, args.Bfactor, args.n_bins)


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    from recovar.project.job_context import job_context

    with job_context(args, "reconstruct_from_external_embedding") as ctx:
        args.outdir = ctx.output_dir
        generate(args)


if __name__ == "__main__":
    main()
