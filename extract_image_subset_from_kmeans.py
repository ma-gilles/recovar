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
    good_indices = np.zeros(labels.size, dtype = bool)
    for kmean_index in kmeans_indices:
        good_indices += (labels == kmean_index)
    if inverse:
        good_indices = ~good_indices

    indices = np.where(good_indices)[0]

    utils.pickle_dump(indices, output_path)




if __name__ == '__main__':

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description='Extract a subset of images based on the kmeans indices')
    parser.add_argument('path_to_centers', type=os.path.abspath, help='Path to the centers.pkl file')
    parser.add_argument('output_path', type=os.path.abspath, help='Path to the output .pkl file containing the indices of subset of images')
    parser.add_argument('kmeans_indices', type=list_of_ints,  help='List of kmeans indices to keep. E.g. 20,30,50')
    parser.add_argument('-i', '--inverse', action='store_true', help='If provided, keep the images that correspond to kmeans centesr that are not in list of kmeans indices')

    args = parser.parse_args()
    # Check that either subvol_idx, or mask or coordinate are provided
    extract_image_subset_from_kmeans(args.path_to_centers, args.kmeans_indices, args.inverse, args.output_path)



