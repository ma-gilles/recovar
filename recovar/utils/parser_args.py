import argparse
import os


def add_project_arg(parser: argparse.ArgumentParser):
    """Add the ``--project`` argument to any command parser."""
    parser.add_argument(
        "--project",
        type=os.path.abspath,
        default=None,
        help="Path to a recovar project directory (containing project.json). "
        "If omitted, auto-detects by walking up from the current directory. "
        "When a project is active, output directories are auto-generated "
        "(e.g. ComputeState/job_0001/), and downstream commands default to the latest completed Pipeline job when result_dir is omitted.",
    )
    return parser


def add_output_name_arg(parser: argparse.ArgumentParser):
    """Add a human-readable project job label without changing the job dir name."""
    parser.add_argument(
        "--output-name",
        default=None,
        help="Human-readable job label stored in project metadata and shown in the GUI. "
        "Does not change the on-disk job directory name.",
    )
    return parser


def add_gpu_memory_arg(parser: argparse.ArgumentParser):
    """Add ``--gpu-memory`` to any command that does heavy GPU work.

    The flag overrides the auto-detected GPU memory used by recovar's
    auto-batch-size formulas. Lower it if the heterogeneity / kernel-
    regression / backproject step OOMs on your GPU — particularly when
    running with ``RECOVAR_DISABLE_CUDA=1`` (the JAX-native fallback uses
    ~3x more memory per image than the custom CUDA kernel).
    """
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=None,
        dest="gpu_memory",
        help=(
            "GPU memory budget in GB used by the auto-batch-size formula. "
            "Default: detect via JAX. Lower than your physical VRAM if you "
            "OOM (especially under RECOVAR_DISABLE_CUDA=1, where the "
            "JAX-native fallback path needs ~3x more memory per image)."
        ),
    )
    return parser


def apply_gpu_memory_arg(args, logger=None) -> None:
    """If ``--gpu-memory N`` was given, propagate to ``set_gpu_memory_limit``."""
    if getattr(args, "gpu_memory", None) is not None:
        from recovar.utils import helpers as _utils

        _utils.set_gpu_memory_limit(args.gpu_memory)
        if logger is not None:
            logger.info("GPU memory budget set to %.1f GB via --gpu-memory", args.gpu_memory)


def standard_downstream_args(parser: argparse.ArgumentParser, analyze=False):

    parser.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        type=os.path.abspath,
        help="Pipeline job directory (e.g. Pipeline/job_0001 or an absolute path). "
        "When omitted in project mode, the latest completed Pipeline job is used.",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        help="Output directory. If omitted and a project is active, "
        "auto-generates a numbered directory (e.g. ComputeState/job_0001/).",
    )

    add_project_arg(parser)
    add_output_name_arg(parser)

    parser.add_argument(
        "--Bfactor",
        type=float,
        default=0,
        help="B-factor sharpening. The B-factor of the consensus reconstruction is probably a good guess. Default is 0, which means no sharpening.",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        dest="n_bins",
        help="number of bins for kernel regression. Default is 50 and works well for most cases. E.g., it was used to generate all figures in the paper",
    )

    parser.add_argument(
        "--maskrad-fraction",
        type=float,
        default=20,
        dest="maskrad_fraction",
        help="Radius of mask used in kernel regression. Default = 20, which means radius = grid_size/20 pixels, or grid_size * voxel_size / 20 angstrom. Default works well for most cases. E.g., it was used to generate all figures in the paper. If you are using cryo-ET or very noisy (or very not noisy data), you might want to decrease (increase) this value. If you are using low resolution data (say less than 128x128 images), you might want to increase this value. If you are using very high resolution data (say more than 512x512 images), you might want to decrease this value. I have little experience with these cases.",
    )

    parser.add_argument(
        "--n-min-particles",
        type=int,
        default=None,
        dest="n_min_particles",
        help="minimum number of particles to compute kernel regression. Default = 100. Default works well for most cases. E.g., it was used to generate all figures in the paper. If you are using cryo-ET or very noisy (or very not noisy data), you might want to increase (decrease) this value.",
    )

    parser.add_argument(
        "--zdim1",
        action="store_true",
        help="Whether dimension 1 is used. This is an annoying corner case for np.loadtxt...",
    )

    parser.add_argument(
        "--no-z-regularization", action="store_true", dest="no_z_regularization", help="Whether to use z regularization"
    )

    parser.add_argument("--lazy", action="store_true", help="Whether to use lazy loading")

    parser.add_argument(
        "--particles",
        default=None,
        help="Particle stack dataset. If you don't pass an argument, the same stack as provided to pipeline.py will be used. You should use this option in case you want to use a higher resolution stack.",
    )

    parser.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file. If not specified, uses the directory of the star file.",
    )

    parser.add_argument(
        "--strip-prefix",
        help="Path prefix to strip from filenames in star file (using in starfile input ONLY). Useful when star file contains longer paths than available on the system. By default, it strips the full path (except the filename). E.g, if you starfile path is Extract/job193/Subtomograms/XXX/XXX.mrcs, and your directory looks like /your/path/to/Subtomograms, then you can use --strip-prefix Extract/job193 --datadir /your/path/to/.",
    )

    parser.add_argument(
        "--apply-global-filtering",
        action="store_true",
        help="Apply global FSC filtering to generated halfmaps and save them with _filtered.mrc suffix. Uses the pipeline's volume_mask for FSC estimation.",
    )

    parser.add_argument(
        "--fsc-mask-radius",
        type=float,
        default=None,
        help="Radius of spherical mask for FSC estimation (in Angstroms). If None, uses pipeline output volume_mask. Overrides the pipeline mask if specified.",
    )

    parser.add_argument(
        "--fsc-mask-edgewidth",
        type=float,
        default=None,
        help="Edge width of FSC mask (in Angstroms). If None, uses 10%% of fsc-mask-radius. Only used if fsc-mask-radius is specified.",
    )

    add_gpu_memory_arg(parser)

    return parser
