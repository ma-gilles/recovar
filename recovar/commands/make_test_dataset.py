import argparse
import logging
import os

import numpy as np
import recovar.jax_config

from recovar.output import output
from recovar.simulation import simulator

logger = logging.getLogger(__name__)


def make_test_dataset(output_dir, image_size=64, noise_level=0.1, n_images=None,
                      create_nested_structure=False, nested_prefix="Extract/job193",
                      tilt_series=False, outlier_file_input=None, percent_outliers=0.0,
                      percent_tilt_series_outliers=0.0, seed=None, *,
                      grid_size=None, volume_input=None, n_tilts=None,
                      premultiplied_ctf=False):
    """Generate a synthetic test dataset used by integration tests and examples.

    Parameters keep backward compatibility with older callers while also
    supporting newer CLI aliases:
    - ``grid_size``: alias of ``image_size`` (takes precedence when provided)
    - ``volume_input``: volume prefix root (default: bundled assets)
    - ``n_tilts``: number of tilts for ``tilt_series=True`` (default: 27)
    """
    if seed is not None:
        np.random.seed(seed)
    if grid_size is None:
        grid_size = image_size
    else:
        grid_size = int(grid_size)

    this_dir = os.path.dirname(__file__)
    volume_folder_input = volume_input if volume_input is not None else os.path.join(this_dir, '..', 'assets', 'vol')

    output_folder = os.path.join(output_dir, 'test_dataset')
    output.mkdir_safe(output_folder)
    n_images = 1000 if n_images is None else int(n_images)
    # Voxel size scales with grid size to keep the same physical extent as the 128-px assets.
    voxel_size = 4.25 * 128 / grid_size

    # Historical default for bundled 3-volume assets; for custom volume sets,
    # use uniform distribution over however many volumes are provided.
    volume_distribution = np.array([1 / 4, 1 / 4, 1 / 2]) if volume_input is None else None

    if tilt_series:
        n_tilts = 27 if n_tilts is None else int(n_tilts)
        n_particles = max(1, n_images // max(n_tilts, 1))
        logger.info("Generating tilt series with %d particles and %d tilts per particle",
                     n_particles, n_tilts)

        image_stack, sim_info = simulator.generate_synthetic_dataset(
            output_folder, voxel_size, volume_folder_input, n_images,
            outlier_file_input=outlier_file_input, grid_size=grid_size,
            volume_distribution=volume_distribution, dataset_params_option="uniform",
            noise_level=noise_level, noise_model="radial1", put_extra_particles=False,
            percent_outliers=percent_outliers, volume_radius=0.7,
            trailing_zero_format_in_vol_name=True,
            noise_scale_std=0, contrast_std=0.1, disc_type='linear_interp',
            create_nested_structure=create_nested_structure, nested_prefix=nested_prefix,
            n_tilts=n_tilts, dose_per_tilt=3, angle_per_tilt=3,
            percent_tilt_series_outliers=percent_tilt_series_outliers,
            premultiplied_ctf=premultiplied_ctf,
        )
    else:
        image_stack, sim_info = simulator.generate_synthetic_dataset(
            output_folder, voxel_size, volume_folder_input, n_images,
            outlier_file_input=outlier_file_input, grid_size=grid_size,
            volume_distribution=volume_distribution, dataset_params_option="uniform",
            noise_level=noise_level, noise_model="radial1", put_extra_particles=False,
            percent_outliers=percent_outliers, volume_radius=0.7,
            trailing_zero_format_in_vol_name=True,
            noise_scale_std=0, contrast_std=0.1, disc_type='linear_interp',
            create_nested_structure=create_nested_structure, nested_prefix=nested_prefix,
            percent_tilt_series_outliers=percent_tilt_series_outliers,
            premultiplied_ctf=premultiplied_ctf,
        )

    logger.info("Finished generating dataset %s", output_folder)
    if create_nested_structure:
        logger.info("Created nested structure with prefix: %s", nested_prefix)
        logger.info("Use --strip-prefix %s when loading the dataset", nested_prefix)
    if tilt_series:
        logger.info("Generated tilt series dataset with %d tilts per particle", n_tilts)
    if percent_outliers > 0:
        logger.info("Generated dataset with %.1f%% outliers", percent_outliers * 100)
        if outlier_file_input:
            logger.info("Used outlier volume: %s", outlier_file_input)
    if tilt_series and percent_tilt_series_outliers > 0:
        logger.info("Generated tilt series dataset with %.1f%% tilt outliers",
                     percent_tilt_series_outliers * 100)


def main():
    parser = argparse.ArgumentParser(description="Generate a test dataset for recovar")
    parser.add_argument("output_dir", nargs='?', default=os.getcwd(),
                        help="Output directory for the test dataset")
    parser.add_argument("--noise-level", type=float, default=0.1,
                        help="Noise level for the dataset")
    parser.add_argument("--n-images", type=int, help="Number of images to generate")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Image size (default: 64 for 64x64 images)")
    parser.add_argument("--grid-size", type=int, default=None,
                        help="Alias for --image-size (takes precedence when set)")
    parser.add_argument("--volume-input", default=None,
                        help="Optional input volume prefix (e.g. /path/to/vol for vol0000.mrc, ...)")
    parser.add_argument("--n-tilts", type=int, default=None,
                        help="Number of tilts when --tilt-series is enabled (default: 27)")
    parser.add_argument("--create-nested-structure", action="store_true",
                        help="Create a nested folder structure to test strip_prefix functionality")
    parser.add_argument("--nested-prefix", default="Extract/job193",
                        help="Prefix path for nested structure (default: Extract/job193)")
    parser.add_argument("--tilt-series", action="store_true",
                        help="Generate tilt series dataset instead of single particle dataset")
    parser.add_argument("--outlier-file-input",
                        help="Path to outlier volume file")
    parser.add_argument("--percent-outliers", default=0.0, type=float,
                        help="Percentage of outliers in the dataset")
    parser.add_argument("--percent-tilt-series-outliers", type=float, default=0.0,
                        help="Percentage of tilt outliers in tilt series dataset")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible dataset generation")
    parser.add_argument("--premultiplied-ctf", action="store_true",
                        help="Generate dataset with premultiplied CTF")

    args = parser.parse_args()

    make_test_dataset(
        args.output_dir, args.image_size, args.noise_level, args.n_images,
        args.create_nested_structure, args.nested_prefix, args.tilt_series,
        args.outlier_file_input, args.percent_outliers,
        args.percent_tilt_series_outliers, args.seed,
        grid_size=args.grid_size, volume_input=args.volume_input, n_tilts=args.n_tilts,
        premultiplied_ctf=args.premultiplied_ctf,
    )


if __name__ == '__main__':
    main()
