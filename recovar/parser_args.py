import os, argparse


def standard_downstream_args(parser: argparse.ArgumentParser, analyze= False):

    parser.add_argument(
        "result_dir",
        type=os.path.abspath,
        help="output directory provided to pipeline.py using the --o option. Note that this function has its own --o option!. For analyze, it will be by default result_dir/output/analysis_[zdim]. For other functions, it is a required argument.",
    )

    if analyze:
        parser.add_argument(
            "-o",
            "--outdir",
            type=os.path.abspath,
            help="Output directory to save all outputs. For analyze, it will be by default result_dir/output/analysis_[zdim]. For other functions, it is a required argument.",
        )
    else:
        parser.add_argument(
            "-o",
            "--outdir",
            type=os.path.abspath,
            required=True,
            help="Output directory to save all outputs. For analyze, it will be by default result_dir/output/analysis_[zdim]. For other functions, it is a required argument.",
        )



    parser.add_argument(
        "--Bfactor",  type =float, default=0, help="B-factor sharpening. The B-factor of the consensus reconstruction is probably a good guess. Default is 0, which means no sharpening."
    )

    parser.add_argument(
        "--n-bins",  type =float, default=50, dest="n_bins",help="number of bins for kernel regression. Default is 50 and works well for most cases. E.g., it was used to generate all figures in the paper"
    )   

    parser.add_argument(
        "--maskrad-fraction",  type =float, default=20, dest="maskrad_fraction",help="Radius of mask used in kernel regression. Default = 20, which means radius = grid_size/20 pixels, or grid_size * voxel_size / 20 angstrom. Default works well for most cases. E.g., it was used to generate all figures in the paper. If you are using cryo-ET or very noisy (or very not noisy data), you might want to decrease (increase) this value. If you are using low resolution data (say less than 128x128 images), you might want to increase this value. If you are using very high resolution data (say more than 512x512 images), you might want to decrease this value. I have little experience with these cases."
    )

    parser.add_argument(
        "--n-min-images",  type =int, default=None, dest="n_min_images",help="minimum number of images to compute kernel regression. Default = 100 for SPA, and 10 particles for tilt series. Default works well for most cases. E.g., it was used to generate all figures in the paper. If you are using cryo-ET or very noisy (or very not noisy data), you might want to increase (decrease) this value."
    )

    parser.add_argument(
        "--zdim1",  action="store_true", help="Whether dimension 1 is used. This is an annoying corner case for np.loadtxt..."
    )

    parser.add_argument(
        "--no-z-regularization",  action="store_true", dest="no_z_regularization", help="Whether to use z regularization"
    )

    parser.add_argument(
        "--lazy",  action="store_true", help="Whether to use lazy loading")

    parser.add_argument(
        "--particles",  default=None, help="Particle stack dataset. If you don't pass an argument, the same stack as provided to pipeline.py will be used. You should use this option in case you want to use a higher resolution stack.")

    parser.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file. Same as the --datadir option in pipeline.py. If you don't pass an argument, the same stack as provided to pipeline.py will be used. You should use this option in case you want to use a higher resolution stack.",
    )

    return parser

