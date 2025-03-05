## Make a p
import argparse
from recovar import heterogeneity_volume, utils, locres
import os
import numpy as np
import logging
logger = logging.getLogger(__name__)
## To figure out which point to sample 




def extract_image_subset_from_kmeans(path_to_centers, kmeans_indices, inverse, output_path):
    assert os.path.exists(path_to_centers), f"Path to centers {path_to_centers} does not exist"
    assert path_to_centers.endswith('.pkl'), "path_to_centers must be a .pkl file"
    assert output_path.endswith('.pkl'), "output_path must be a .pkl file"

    centers = utils.pickle_load(path_to_centers)
    labels = centers['labels'].astype(int)
    labels[np.isnan(centers['labels'])] = -1
    good_indices = np.zeros(labels.size, dtype = bool)
    for kmean_index in kmeans_indices:
        good_indices += (labels == kmean_index)
    if inverse:
        good_indices = ~good_indices
        good_indices[np.isnan(centers['labels'])]= 0
    indices = np.where(good_indices)[0]

    utils.pickle_dump(indices, output_path)
    logger.info(f"Saved {indices.size} out of {np.sum(~np.isnan(centers['labels']))} indices of subset of images to {path_to_centers}. Total dataset has {labels.size} images (or very close to that).")



def main():


    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description='Extract a subset of images based on the kmeans indices')
    parser.add_argument('path_to_centers', type=os.path.abspath, help='Path to the centers.pkl file')
    parser.add_argument('output_path', type=os.path.abspath, help='Path to the output .pkl file containing the indices of subset of images')
    parser.add_argument('kmeans_indices', type=list_of_ints,  help='List of kmeans indices to keep. E.g. 20,30,50')
    parser.add_argument('-i', '--inverse', action='store_true', help='If provided, keep the images that correspond to kmeans centesr that are not in list of kmeans indices')

    args = parser.parse_args()

    log_dir = os.path.dirname(args.output_path)
    log_file = os.path.join(log_dir, "extract_subset_run.log")

    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        level=logging.INFO,
                        force = True, 
                        handlers=[
        logging.FileHandler(f"{log_file}"),
        logging.StreamHandler()])

    # Check that either subvol_idx, or mask or coordinate are provided
    extract_image_subset_from_kmeans(args.path_to_centers, args.kmeans_indices, args.inverse, args.output_path)




if __name__ == '__main__':
    main()
