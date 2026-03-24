import argparse
import logging
import os

import numpy as np

from recovar import utils

logger = logging.getLogger(__name__)


def extract_image_subset_from_kmeans(path_to_centers, kmeans_indices, inverse, output_path):
    if not os.path.exists(path_to_centers):
        raise FileNotFoundError(f"Path to centers {path_to_centers} does not exist")
    if not path_to_centers.endswith('.pkl'):
        raise ValueError("path_to_centers must be a .pkl file")
    if not output_path.endswith('.pkl'):
        raise ValueError("output_path must be a .pkl file")

    centers = utils.pickle_load(path_to_centers)
    raw_labels = centers['labels']
    labels = np.where(np.isnan(raw_labels), -1, raw_labels).astype(int)
    good_indices = np.zeros(labels.size, dtype=bool)
    for kmean_index in kmeans_indices:
        good_indices |= (labels == kmean_index)
    if inverse:
        good_indices = ~good_indices
        good_indices[np.isnan(raw_labels)] = False
    indices = np.where(good_indices)[0]

    utils.pickle_dump(indices, output_path)
    n_valid = np.sum(~np.isnan(centers['labels']))
    logger.info("Saved %d out of %d indices of subset of images to %s. "
                "Total dataset has %d images (or very close to that).",
                indices.size, n_valid, path_to_centers, labels.size)


def main():
    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description='Extract a subset of images based on the kmeans indices')
    parser.add_argument('path_to_centers', type=os.path.abspath, help='Path to the centers.pkl file')
    parser.add_argument('output_path', type=os.path.abspath, help='Path to the output .pkl file containing the indices of subset of images')
    parser.add_argument('kmeans_indices', type=list_of_ints, help='List of kmeans indices to keep. E.g. 20,30,50')
    parser.add_argument('-i', '--inverse', action='store_true', help='If provided, keep the images that correspond to kmeans centers that are not in list of kmeans indices')
    from recovar.utils.parser_args import add_project_arg
    add_project_arg(parser)

    args = parser.parse_args()
    # job_context checks for args.output; this parser uses output_path
    args.output = args.output_path

    from recovar.project.job_context import job_context
    with job_context(args, "extract_image_subset_from_kmeans") as ctx:
        args.output_path = ctx.output_dir

        log_dir = os.path.dirname(args.output_path)
        log_file = os.path.join(log_dir, "extract_subset_run.log")

        from recovar.utils.helpers import RobustFileHandler, RobustStreamHandler
        logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            level=logging.INFO,
                            force=True,
                            handlers=[
            RobustFileHandler(log_file),
            RobustStreamHandler()])

        extract_image_subset_from_kmeans(args.path_to_centers, args.kmeans_indices, args.inverse, args.output_path)


if __name__ == '__main__':
    main()
