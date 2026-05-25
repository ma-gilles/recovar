import argparse
import math
import os


def positive_finite_gb(raw: str) -> float:
    """argparse ``type`` for ``--gpu-budget-gb``: reject NaN / inf / non-positive.

    Shared with the pre-import scanner in ``recovar.command_line`` so a
    malformed budget fails the same way before AND after jax has been
    imported. Without this, ``float("NaN")`` and ``float("0")`` slip
    through, produce nonsensical XLA env values, and burn a Slurm hour.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--gpu-budget-gb={raw!r} is not a number")
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"--gpu-budget-gb={raw!r} is not finite (NaN or inf)")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"--gpu-budget-gb={raw!r} must be > 0")
    return value


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
    """Add ``--gpu-budget-gb`` to any heavy-GPU command.

    A SOFT budget for RECOVAR's auto-batch-size formulas. It does not
    enforce anything against JAX directly — JAX's allocation behavior
    is controlled separately by ``XLA_PYTHON_CLIENT_MEM_FRACTION`` and
    ``XLA_PYTHON_CLIENT_PREALLOCATE``. Pass it when you want recovar
    to plan smaller batches than your physical VRAM (e.g. on a shared
    GPU or to leave headroom for another process).
    """
    parser.add_argument(
        "--gpu-budget-gb",
        type=positive_finite_gb,
        default=None,
        dest="gpu_memory",
        help=(
            "Soft GPU memory budget in GB used by RECOVAR's auto-batch-size "
            "formula (positive finite). Default: full physical VRAM as "
            "reported by JAX. Lower this on a constrained or shared GPU. "
            "This is NOT a JAX-level cap — see XLA_PYTHON_CLIENT_MEM_FRACTION "
            "and XLA_PYTHON_CLIENT_PREALLOCATE for that."
        ),
    )
    return parser


def apply_gpu_memory_arg(args, logger=None) -> None:
    """If ``--gpu-budget-gb`` was given, propagate to ``set_gpu_memory_limit``."""
    if getattr(args, "gpu_memory", None) is not None:
        from recovar.utils import helpers as _utils

        _utils.set_gpu_memory_limit(args.gpu_memory)
        if logger is not None:
            logger.info(
                "GPU batch-size budget set to %.1f GB via --gpu-budget-gb",
                args.gpu_memory,
            )


def add_memory_planning_args(parser: argparse.ArgumentParser):
    """Add the full GPU-memory flag set for any heavy-GPU command.

    Adds:
      - ``--gpu-budget-gb``                        budget (caps JAX + batch formulas)
      - ``--low-memory-option``             halve batch sizes
      - ``--very-low-memory-option``        quarter batch sizes
      - ``--adaptive-n-pcs``                pick n_pcs to fit the budget
                                            (reproducible: same flags +
                                            dataset = same n_pcs)
      - ``--memory-profile``                heavyweight jax.profiler captures
                                            (always-on diagnostics live in
                                            ``<outdir>/_diagnostics/``)
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Adaptively reduce the number of principal components to fit "
            "the GPU memory budget. ON BY DEFAULT (was opt-in pre-2026-05-15). "
            "The planner predicts peak memory for the (grid, n_pcs) cell and "
            "walks n_pcs down from 200 until the prediction fits the budget. "
            "Reproducible: same flags + same dataset = same n_pcs. Use "
            "--no-adaptive-n-pcs to force the legacy fixed-200-PCs behavior."
        ),
    )
    # NOTE: ``--memory-diagnostics`` was removed. Diagnostics
    # (memory_plan.json, memory_trace.jsonl, allocator_env.json,
    # args.json) are ALWAYS written into ``<outdir>/_diagnostics/``.
    # The cost is < 1 second per run and the diagnostic value is
    # high enough that opt-in didn't make sense.
    group.add_argument(
        "--memory-profile",
        dest="memory_profile",
        action="store_true",
        help=(
            "HEAVYWEIGHT: capture jax.profiler.save_device_memory_profile "
            "snapshots at each phase boundary (~50-200 ms per phase plus "
            "~5-50 MB profile files). Used by the validation sweep and "
            "manual debugging; not needed for production runs."
        ),
    )
    group.add_argument(
        "--fail-on-memory-exceed",
        dest="fail_on_memory_exceed",
        action="store_true",
        help=(
            "Test-only: at end of run, assert max(jax_peak_gb) <= --gpu-budget-gb * 1.05; "
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


def write_run_metadata(args, outdir, logger=None):
    """Write args.json and allocator_env.json into ``outdir/_diagnostics/``.

    Records: serialized argparse Namespace, the JAX/XLA env vars in
    effect, the canonical CUDA env var, and the recovar git head if
    available. Captured at planner-construction time so two runs with
    identical CLI flags but different shell env can be distinguished.
    """
    import json as _json
    import os as _os
    import subprocess as _subprocess

    from recovar.utils.memory_planner import diagnostics_dir

    diag = diagnostics_dir(outdir)

    # args.json
    try:
        args_dict = {k: _serialize_value(v) for k, v in vars(args).items()}
        # Resolve git head best-effort
        try:
            head = _subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            git_head = head.stdout.strip() if head.returncode == 0 else None
        except Exception:
            git_head = None
        args_dict["__git_head__"] = git_head
        (diag / "args.json").write_text(_json.dumps(args_dict, indent=2, default=str))
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not write args.json: %s", exc)

    # allocator_env.json
    try:
        env_keys = [
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "TF_GPU_ALLOCATOR",
            "CUDA_VISIBLE_DEVICES",
            "RECOVAR_DISABLE_CUDA",
            "RECOVAR_CUDA_DISABLE",
            "JAX_PLATFORMS",
        ]
        env_record = {k: _os.environ.get(k) for k in env_keys}
        (diag / "allocator_env.json").write_text(_json.dumps(env_record, indent=2))
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not write allocator_env.json: %s", exc)


def _serialize_value(v):
    """Best-effort JSON-serializable representation."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _serialize_value(x) for k, x in v.items()}
    return str(v)


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
      - ``trace`` is a ``MemoryTraceWriter`` (always-on; writes to
        ``<outdir>/_diagnostics/memory_trace.jsonl``)

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

    # Backend-aware budget for legacy ``get_*_batch_size`` formulas.
    #
    # Saturation sweep (A100 80GB, slurm 8020210, 17 cells) showed the
    # legacy formulas OVERSHOOT the budget under ``jax_fallback``:
    #
    #   - g=64 jax_fallback, --gpu-budget-gb 12 → observed peak 17 GB
    #     (1.42× overshoot — would have OOMed under --fail-on-memory-exceed)
    #   - g=128 jax_fallback at every tested budget (12, 24, 40, 76 GB)
    #     → OOM (peak > budget always; formula refuses to shrink)
    #   - g=256 jax_fallback → OOM at full 76 GB
    #
    # custom_cuda fits comfortably (ratios 0.09-0.72 across grids). So
    # the fix is to deflate the budget HANDED TO LEGACY only when
    # backend == jax_fallback. The user's stated budget is unchanged;
    # only the value the batch-size formulas see is reduced.
    #
    # Divisor = 3.0 chosen conservatively: at g=64 the observed overshoot
    # was 1.42, so /1.5 would suffice there; at g=128/256 the formula
    # OOMs even at /3.2, but we cap at 3.0 because going further would
    # make small-budget jax_fallback runs run pathologically small
    # batches. Users on tight jax_fallback configs should consider
    # ``--low-memory-option`` for additional headroom.
    _BACKEND_DIVISOR = {
        "custom_cuda": 1.0,
        "jax_fallback": 3.0,
        "cpu": 1.0,
    }
    divisor = _BACKEND_DIVISOR.get(plan.budget.backend, 1.0)
    # ONLY override the global limit when the divisor is non-trivial.
    # Calling ``set_gpu_memory_limit`` unconditionally — even with the
    # planner's effective_budget — shifts batch sizes from "whatever
    # JAX reports as bytes_limit" to "effective_budget", and the
    # noise-floor cryo-ET outlier metrics flip on that small a delta
    # (slurm 8025670 captured it). For custom_cuda (divisor=1.0) we
    # leave the global alone so production paths are bit-identical to
    # dev.
    if divisor != 1.0:
        legacy_budget = plan.budget.effective_budget_gb / divisor
        if logger is not None:
            logger.info(
                "Deflating budget for legacy batch-size formulas: %.2f GB / %.2f = "
                "%.2f GB (backend=%s; sweep 8020210 showed legacy overshoots otherwise)",
                plan.budget.effective_budget_gb,
                divisor,
                legacy_budget,
                plan.budget.backend,
            )
        try:
            from recovar.utils import helpers as _helpers

            _helpers.set_gpu_memory_limit(legacy_budget)
        except Exception:
            pass

    if outdir is not None:
        try:
            memory_planner.write_memory_plan_json(plan, outdir)
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not write memory_plan.json: %s", exc)

        # Run metadata is opt-in via --memory-profile so we don't fork
        # `git rev-parse HEAD` on every nested pipeline call (each
        # pipeline_with_outliers round calls apply_memory_planning_args
        # afresh; subprocess calls per-round can perturb noisy outlier
        # metrics — slurm 7985775 / 7990236 / 8000338 vs 7998510).
        if getattr(args, "memory_profile", False):
            try:
                write_run_metadata(args, outdir, logger=logger)
            except Exception as exc:
                if logger is not None:
                    logger.warning("Could not write run metadata: %s", exc)

    # Memory trace is opt-in via --memory-profile. Always-on tracing
    # injected JAX memory_stats() + nvidia-smi subprocess calls at
    # every phase boundary; that turned out to perturb noisy outlier
    # regression metrics enough to fail the cryo-ET long test even
    # though the trace itself doesn't change pipeline math. External
    # sweep harnesses that want the trace can set ``--memory-profile``
    # explicitly per cell.
    trace = None
    if outdir is not None:
        trace = memory_planner.MemoryTraceWriter(outdir, enabled=bool(getattr(args, "memory_profile", False)))

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

    # Full memory-planning surface (includes --gpu-budget-gb / --gpu-memory and
    # all the new diagnostic / safety flags) so every downstream command
    # exposes the same UX as ``recovar pipeline``.
    add_memory_planning_args(parser)

    return parser
