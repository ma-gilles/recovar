import recovar.jax_config
import argparse
import logging
import os
import time

import numpy as np

from recovar.output import output as o
from recovar import utils
from recovar.data_io import halfsets
from recovar.heterogeneity import embedding

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "result_dir",
        type=os.path.abspath,
        help="result dir (output dir of pipeline)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=False,
        help="Output directory to save model",
    )
    parser.add_argument("--zdim", type=int, help="Dimension of latent variable (a single int, not a list)")

    parser.add_argument("--n-clusters", metavar=int, default=40, help="Number of k-means clusters")

    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=6,
        dest="n_trajectories",
        help="how many trajectories to compute between k-means clusters",
    )

    parser.add_argument(
        "--skip-umap",
        dest="skip_umap",
        action="store_true",
        help="whether to skip u-map embedding (can be slow for large dataset)",
    )

    parser.add_argument("--q", metavar=float, default=None, help="quantile used for reweighting (default = 0.95)")

    parser.add_argument(
        "--n-std",
        metavar=float,
        default=None,
        help="number of standard deviations to use for reweighting (don't set q and this parameter, only one of them)",
    )

    return parser


def compute_embedding(recovar_result_dir):
    results = o.load_results_new(recovar_result_dir)
    ds = halfsets.load_halfset_dataset_from_args(results["input_args"])
    options = utils.make_algorithm_options(results["input_args"])

    gpu_memory = utils.get_gpu_memory_total()
    latent_coords = {}
    latent_precision = {}
    est_contrasts = {}
    coords = results["latent_coords"]
    zdims = sorted(coords.keys())
    if not zdims:
        input_zdim = getattr(results.get("input_args", None), "zdim", None)
        if input_zdim is None:
            raise ValueError(
                "Could not determine latent dimensions to embed (missing results['latent_coords'] and input_args.zdim)"
            )
        zdims = list(input_zdim) if isinstance(input_zdim, (list, tuple, np.ndarray)) else [int(input_zdim)]

    for zdim in zdims:
        z_time = time.time()
        latent_coords[zdim], latent_precision[zdim], est_contrasts[zdim] = embedding.get_per_image_embedding(
            results["means"].combined,
            results["u"]["rescaled"],
            results["s"]["rescaled"],
            zdim,
            results["cov_noise"],
            ds,
            results["volume_mask"],
            gpu_memory,
            "linear_interp",
            contrast_grid=None,
            contrast_option=options.contrast,
        )
        logger.info("embedding time for zdim=%s: %s", zdim, time.time() - z_time)

    return latent_coords, latent_precision, est_contrasts


def compute_embedding_and_save(recovar_result_dir):
    latent_coords, latent_precision, est_contrasts = compute_embedding(recovar_result_dir)


def main():
    raise NotImplementedError("This script is not ready yet")


if __name__ == "__main__":
    main()
