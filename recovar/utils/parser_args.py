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
    """Add ``--gpu-gb`` to any heavy-GPU command.

    Lowers the GPU budget used by recovar's auto-batch-size formulas
    AND caps JAX's actual allocation via
    ``XLA_PYTHON_CLIENT_MEM_FRACTION`` (the latter happens in
    ``recovar.command_line._apply_gpu_memory_cap`` before jax
    initializes). Lower it if the heterogeneity / kernel-regression /
    backproject step OOMs on your GPU.
    """
    parser.add_argument(
        "--gpu-gb",
        type=float,
        default=None,
        dest="gpu_memory",
        help=(
            "GPU memory budget in GB. Caps JAX allocation AND the "
            "auto-batch-size formula. Default: full physical VRAM via "
            "JAX. Lower this on a constrained or shared GPU."
        ),
    )
    return parser


def apply_gpu_memory_arg(args, logger=None) -> None:
    """If ``--gpu-gb`` was given, propagate to ``set_gpu_memory_limit``."""
    if getattr(args, "gpu_memory", None) is not None:
        from recovar.utils import helpers as _utils

        _utils.set_gpu_memory_limit(args.gpu_memory)
        if logger is not None:
            logger.info(
                "GPU memory budget set to %.1f GB via --gpu-gb",
                args.gpu_memory,
            )


def add_memory_planning_args(parser: argparse.ArgumentParser):
    """Add the full GPU-memory flag set for any heavy-GPU command.

    Adds:
      - ``--gpu-gb``                        budget (caps JAX + batch formulas)
      - ``--low-memory-option``             halve batch sizes
      - ``--very-low-memory-option``        quarter batch sizes
      - ``--adaptive-n-pcs``                pick n_pcs to fit the budget
                                            (reproducible: same flags +
                                            dataset = same n_pcs)
      - ``--memory-diagnostics``            write memory_trace.jsonl per phase
      - ``--fail-on-memory-exceed``         test-only end-of-run assertion
      - ``--memory-safety-fraction``        advanced multiplier (testing)

    Hidden / test-only:
      - ``--debug-fail-on-memory-exceed``   alias of --fail-on-memory-exceed
        (reserved for the unit-test override env var)
    """
    if not any(a.dest == "gpu_memory" for a in parser._actions):
        add_gpu_memory_arg(parser)

    # Idempotency: callers may stack add_args (pipeline.add_args + a
    # wrapper command, or standard_downstream_args + a custom helper).
    # If we've already wired the planning flags, just return.
    if any(a.dest == "low_memory_option" for a in parser._actions):
        return parser

    group = parser.add_argument_group("Memory planning")
    group.add_argument(
        "--low-memory-option",
        dest="low_memory_option",
        action="store_true",
        help="Halve image/volume/column batch sizes to stretch a tight budget.",
    )
    group.add_argument(
        "--very-low-memory-option",
        dest="very_low_memory_option",
        action="store_true",
        help="Quarter image/volume/column batch sizes (more aggressive).",
    )
    group.add_argument(
        "--adaptive-n-pcs",
        dest="adaptive_memory",
        action="store_true",
        help=(
            "Adaptively reduce the number of principal components to fit "
            "the GPU memory budget. By default, 200 PCs are used regardless "
            "of GPU size for reproducibility. With this flag, the planner "
            "consults the calibrated peak-memory table and picks the "
            "largest n_pcs whose predicted peak fits the budget. "
            "Reproducible: same flags + same dataset = same n_pcs."
        ),
    )
    group.add_argument(
        "--memory-diagnostics",
        dest="memory_diagnostics",
        action="store_true",
        help=(
            "Write memory_trace.jsonl with per-phase JAX peak-memory rows "
            "to the output directory. memory_plan.json is written "
            "unconditionally (it's tiny)."
        ),
    )
    group.add_argument(
        "--fail-on-memory-exceed",
        dest="fail_on_memory_exceed",
        action="store_true",
        help=(
            "Test-only: at end of run, assert max(jax_peak_gb) <= --gpu-gb * 1.05; "
            "exit non-zero with a structured failure summary if violated. "
            "Does NOT abort mid-run."
        ),
    )
    group.add_argument(
        "--memory-safety-fraction",
        dest="memory_safety_fraction",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def apply_memory_planning_args(
    args,
    *,
    command,
    grid_size,
    n_images,
    outdir=None,
    logger=None,
    desired_n_pcs=200,
):
    """Build a MemoryPlan, write memory_plan.json, and return both helpers.

    Returns ``(plan, trace)`` where:
      - ``plan`` is a ``recovar.utils.memory_planner.MemoryPlan``
      - ``trace`` is a ``MemoryTraceWriter`` (no-op if --memory-diagnostics
        is not set)

    Heavy-GPU commands then read ``plan.image_batch_size`` etc. directly
    instead of calling the legacy ``utils.get_*_batch_size`` formulas.
    """
    from recovar.utils import memory_planner

    plan = memory_planner.make_memory_plan(
        command=command,
        grid_size=grid_size,
        n_images=n_images,
        requested_gpu_gb=getattr(args, "gpu_memory", None),
        low_memory=bool(getattr(args, "low_memory_option", False)),
        very_low_memory=bool(getattr(args, "very_low_memory_option", False)),
        adaptive_n_pcs=bool(getattr(args, "adaptive_memory", False)),
        desired_n_pcs=desired_n_pcs,
    )

    if logger is not None:
        budget = plan.budget
        logger.info(
            "GPU memory plan: backend=%s, requested=%s GB, effective=%.2f GB "
            "(source=%s), n_pcs=%d (calibration_status=%s)",
            budget.backend,
            f"{budget.requested_gb:.1f}" if budget.requested_gb else "auto",
            budget.effective_budget_gb,
            budget.source,
            plan.n_pcs_to_compute,
            plan.calibration_status,
        )
        for warn in budget.warnings:
            logger.warning("memory plan warning: %s", warn)

    if outdir is not None:
        try:
            memory_planner.write_memory_plan_json(plan, outdir)
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not write memory_plan.json: %s", exc)

    trace = None
    if outdir is not None:
        trace = memory_planner.MemoryTraceWriter(
            outdir,
            enabled=bool(getattr(args, "memory_diagnostics", False)),
        )

    return plan, trace


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

    # Full memory-planning surface (includes --gpu-gb / --gpu-memory and
    # all the new diagnostic / safety flags) so every downstream command
    # exposes the same UX as ``recovar pipeline``.
    add_memory_planning_args(parser)

    return parser
