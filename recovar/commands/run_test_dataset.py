"""Smoke-test the recovar install end-to-end on a tiny synthetic dataset.

Three design contracts:

1. Arguments are forwarded to inner ``recovar pipeline ...`` etc. calls
   as ``argv`` lists, never shell strings. This makes logging / quoting
   reliable and lets us cleanly capture stderr+stdout for the
   error-hints classifier on failure.

2. ``--adaptive-n-pcs`` is ALWAYS spliced into every inner pipeline /
   downstream call by default. This wrapper exists to answer "is your
   install correct?", not to act as a science test — it has to finish
   even on a small or shared GPU. Pass ``--full-memory-test`` to opt
   out and exercise the fixed 200-PC, non-adaptive configuration.

3. The set of subcommands that accept memory-planning flags is the
   single source of truth (``_COMMANDS_WITH_MEMORY_ARGS``). The
   ``_recovar_argv`` helper consults that set to decide whether to
   splice memory flags, and is the only way this module builds
   subprocess invocations. Adding a new heavy-GPU command anywhere
   in the codebase only requires updating that set here.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import shutil
import subprocess
import sys

import jax

from recovar.output import output

logger = logging.getLogger(__name__)


# Subcommands that accept the full memory-planning flag surface
# (``--gpu-budget-gb``, ``--low-memory-option``, ``--adaptive-n-pcs``, …).
# Single source of truth: tests and the splicer below both read this.
_COMMANDS_WITH_MEMORY_ARGS = frozenset(
    {
        "pipeline",
        "pipeline_with_outliers",
        "analyze",
        "compute_state",
        "compute_trajectory",
        "reconstruct_from_external_embedding",
        "junk_particle_detection",
        "outlier_detection",
    }
)

# Subcommands that accept ``--accept-cpu``. Today only the pipeline
# entry points; everything else honors JAX_PLATFORMS=cpu via env var.
_COMMANDS_WITH_ACCEPT_CPU = frozenset({"pipeline", "pipeline_with_outliers"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run integration tests for recovar")
    parser.add_argument("--output-dir", "-o", default="/tmp/")
    parser.add_argument("--n-images", type=int, default=None, help="Number of synthetic images to generate")
    parser.add_argument("--image-size", type=int, default=64, help="Synthetic image size/grid size")
    parser.add_argument("--all-tests", action="store_true", help="Run all tests")
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the broader default smoke (currently: also a second pipeline "
            "variant). Without this, the default scope is minimal: one "
            "pipeline + analyze + estimate_conformational_density. See "
            "issue #143."
        ),
    )
    parser.add_argument("--tilt-series-only", action="store_true", help="Run only tilt series tests")
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Do not delete the test dataset directory after successful tests",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help=(
            "Force CPU-only execution. Sets JAX_PLATFORMS=cpu so JAX ignores any "
            "visible GPUs, AND passes --accept-cpu to the inner pipeline so it "
            "doesn't bail on the no-GPU check."
        ),
    )
    parser.add_argument(
        "--full-memory-test",
        action="store_true",
        help=(
            "Disable the auto-add of --adaptive-n-pcs for inner pipeline calls. "
            "Use this when you want the smoke test to exercise the fixed 200-PC, "
            "non-adaptive configuration even on a constrained --gpu-budget-gb budget."
        ),
    )

    # Memory-planning flag set: --gpu-budget-gb, --low-memory-option,
    # --very-low-memory-option, --adaptive-n-pcs, --memory-profile,
    # --fail-on-memory-exceed, --memory-safety-fraction.
    from recovar.utils.parser_args import add_memory_planning_args

    add_memory_planning_args(parser)
    return parser


def _build_forward_argv(args: argparse.Namespace) -> list[str]:
    """Return the list of argv tokens spliced into memory-aware subcommands.

    Since 2026-05-15 ``--adaptive-n-pcs`` is ON BY DEFAULT in the
    downstream pipeline (BooleanOptionalAction). The smoke test always
    splices a deterministic ``--adaptive-n-pcs`` or ``--no-adaptive-n-pcs``
    so the inner subprocess gets an explicit choice regardless of how
    its own defaults evolve. ``--full-memory-test`` forces the opt-out
    (fixed 200 PCs) for users who want to stress-test the unshrunk
    config on a constrained budget.
    """
    fwd: list[str] = []
    if args.gpu_memory is not None:
        fwd += ["--gpu-budget-gb", str(args.gpu_memory)]
    if args.low_memory_option:
        fwd.append("--low-memory-option")
    if args.very_low_memory_option:
        fwd.append("--very-low-memory-option")

    if args.full_memory_test:
        fwd.append("--no-adaptive-n-pcs")
        if args.adaptive_memory:
            logger.warning(
                "Both --adaptive-n-pcs and --full-memory-test passed; "
                "--full-memory-test wins (splicing --no-adaptive-n-pcs)."
            )
    elif args.adaptive_memory:
        fwd.append("--adaptive-n-pcs")
    else:
        fwd.append("--no-adaptive-n-pcs")

    # `--memory-diagnostics` was removed. The new
    # `--memory-profile` is heavyweight; only forward if explicitly set.
    if getattr(args, "memory_profile", False):
        fwd.append("--memory-profile")
    if args.fail_on_memory_exceed:
        fwd.append("--fail-on-memory-exceed")
    if args.memory_safety_fraction is not None:
        fwd += ["--memory-safety-fraction", str(args.memory_safety_fraction)]
    return fwd


def main():
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    do_all_tests = args.all_tests
    tilt_series_only = args.tilt_series_only
    delete_everything = not args.no_delete
    run_on_cpu = args.cpu
    dataset_dir = args.output_dir
    forward_argv = _build_forward_argv(args)
    image_size = args.image_size
    n_images = args.n_images

    if image_size <= 0:
        parser.error(f"--image-size must be positive, got {image_size}")
    if n_images is not None and n_images <= 0:
        parser.error(f"--n-images must be positive, got {n_images}")

    if run_on_cpu:
        # Force CPU-only mode in spawned subprocesses. This wrapper itself
        # has already imported jax, so the env var doesn't affect THIS
        # process — but it propagates to the child pipelines.
        os.environ["JAX_PLATFORMS"] = "cpu"

    base_argv = [sys.executable, "-m", "recovar.command_line"]

    passed_functions: list[str] = []
    failed_functions: list[str] = []
    cleanup_paths: list[str] = []

    def check_gpu():
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                logger.info("GPU devices found: %s", gpu_devices)
            else:
                _gpu_error()
        except Exception as e:
            logger.error("Error checking for GPU devices: %s", e)
            _gpu_error()

    def _gpu_error():
        logger.error(
            "No GPU devices found by JAX. Please ensure that JAX is properly "
            "configured with CUDA and a compatible GPU.\n"
            "Driver version must be >= 525.60.13 for CUDA 12 on Linux.\n"
            "Reinstall jax: pip install -U 'jax[cuda12]'\n"
            "To run on CPU, use the --cpu flag."
        )
        sys.exit(1)

    if not run_on_cpu:
        check_gpu()
        try:
            from recovar.cuda_backproject import cuda_available

            if cuda_available():
                logger.info("CUDA backproject/project kernels: OK")
            else:
                logger.warning(
                    "Custom CUDA kernels not available — falling back to JAX GPU "
                    "path (~2x slower). Run 'recovar build_custom_cuda' to build "
                    "them, or see https://github.com/ma-gilles/recovar/issues/131"
                )
        except RuntimeError as e:
            logger.error("%s", e)
            sys.exit(1)

    def run_command(argv: list[str], description: str, function_name: str) -> bool:
        """Invoke ``argv`` as a subprocess. Returns True on success.

        On failure, captured stdout+stderr is run through the error-hints
        classifier so the caller still gets actionable advice even though
        the inner ``recovar`` invocation exits non-zero from inside its
        own error wrapper.

        The child env is built through ``recovar_subprocess_env`` so
        parent XLA/JAX settings propagate consistently to every inner
        recovar command (issue #143).
        """
        from recovar.utils.subprocess_helpers import recovar_subprocess_env

        logger.info("Running: %s", description)
        logger.info("Command: %s", " ".join(argv))
        result = subprocess.run(argv, check=False, capture_output=True, text=True, env=recovar_subprocess_env())
        # Echo the child's output so the user sees it (capture_output silences
        # the stream by default).
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        if result.returncode == 0:
            logger.info("Success: %s", description)
            passed_functions.append(function_name)
            return True

        logger.error("Failed: %s (exit %d)", description, result.returncode)
        failed_functions.append(function_name)
        # If the inner ``recovar`` subprocess already printed a formatted
        # hint (which it does via ``command_line._run_with_error_hints``),
        # don't print a second one — the wrapper would just push the
        # actionable advice further from the tail of the Slurm log.
        # Detect by looking for the ``═``*N delimiter that
        # ``error_hints.format_error_hint`` emits.
        from recovar.utils import error_hints

        captured_stderr = result.stderr or ""
        already_printed = error_hints._DELIMITER in captured_stderr
        if already_printed:
            logger.debug(
                "Skipping wrapper-level error hint for %s; the inner subprocess already printed one.",
                function_name,
            )
            return False

        try:
            ctx = error_hints.collect_context()
            hint = error_hints.classify_subprocess_failure(captured_stderr, result.stdout or "", ctx)
            if hint is not None:
                sys.stderr.write("\n")
                sys.stderr.write(error_hints.format_error_hint(hint))
                sys.stderr.flush()
        except Exception as exc:
            logger.debug("Could not classify subprocess failure: %s", exc)
        return False

    def _p(*parts):
        return os.path.join(dataset_dir, *parts)

    def _recovar_argv(cmd: str, *tokens: str) -> list[str]:
        """Build a ``recovar <cmd> <tokens>`` argv with memory-flag
        forwarding driven by ``_COMMANDS_WITH_MEMORY_ARGS``.

        Centralizing this is the answer to #135 reliability: any new
        heavy-GPU command is wired up by adding its name to that set,
        and every subprocess call here goes through this single entry
        point so we cannot forget to splice the flags.
        """
        argv = [*base_argv, cmd, *tokens]
        if cmd in _COMMANDS_WITH_MEMORY_ARGS:
            argv.extend(forward_argv)
        if run_on_cpu and cmd in _COMMANDS_WITH_ACCEPT_CPU:
            argv.append("--accept-cpu")
        return argv

    def pipeline_argv(*pos: str, output_path: str, extras: list[str]) -> list[str]:
        # Thin wrapper that injects ``-o <output_path>`` so the legacy
        # call sites below stay readable.
        return _recovar_argv("pipeline", *pos, "-o", output_path, *extras)

    if tilt_series_only:
        logger.info("Running tilt series tests only...")
        cleanup_paths.append(os.path.join(dataset_dir, "tilt_test"))

        run_command(
            _recovar_argv(
                "make_test_dataset",
                _p("tilt_test"),
                "--n-images",
                str(n_images if n_images is not None else 10000),
                "--image-size",
                str(image_size),
                "--tilt-series",
            ),
            "Generate a test dataset for tilt series",
            "make_test_dataset_tilt",
        )

        run_command(
            pipeline_argv(
                _p("tilt_test", "test_dataset", "particles.star"),
                "--poses",
                _p("tilt_test", "test_dataset", "poses.pkl"),
                "--ctf",
                _p("tilt_test", "test_dataset", "ctf.pkl"),
                "--tilt-series",
                "--tilt-series-ctf=relion5",
                "--correct-contrast",
                "--mask=from_halfmaps",
                "--lazy",
                "--ignore-zero-frequency",
                output_path=_p("tilt_test", "test_dataset", "pipeline_tilt_output"),
                extras=[],
            ),
            "Run pipeline with tilt series",
            "pipeline_tilt",
        )

        run_command(
            _recovar_argv(
                "analyze",
                _p("tilt_test", "test_dataset", "pipeline_tilt_output"),
                "--zdim=2",
                "--no-z-regularization",
                "--n-clusters=3",
                "--n-trajectories=0",
            ),
            "Run analyze with tilt series",
            "analyze_tilt",
        )

        target_path = _p("tilt_test", "test_dataset", "target.txt")
        target_parent = os.path.dirname(target_path)
        if os.path.isdir(target_parent):
            with open(target_path, "w", encoding="utf-8") as fh:
                fh.write("0.0 0.0\n")
            passed_functions.append("create_target_tilt")
        else:
            # Don't promote this to a failed_function: if the parent
            # directory is missing, an earlier subprocess (most likely
            # make_test_dataset_tilt) already failed and is recorded
            # there. Just log and move on.
            logger.warning(
                "Skipping create_target_tilt: parent directory %s does not exist.",
                target_parent,
            )

        run_command(
            _recovar_argv(
                "reconstruct_from_external_embedding",
                _p("tilt_test", "test_dataset", "particles.star"),
                "--poses",
                _p("tilt_test", "test_dataset", "poses.pkl"),
                "--ctf",
                _p("tilt_test", "test_dataset", "ctf.pkl"),
                "--tilt-series",
                "--embedding",
                _p("tilt_test", "test_dataset", "pipeline_tilt_output", "embeddings.pkl"),
                "--target",
                target_path,
                "-o",
                _p("tilt_test", "test_dataset", "reconstruct_tilt_output"),
            ),
            "Test reconstruct_from_external_embedding with tilt series",
            "reconstruct_tilt",
        )

    else:
        cleanup_paths.append(os.path.join(dataset_dir, "test_dataset"))
        make_dataset_args = [
            "make_test_dataset",
            dataset_dir,
            "--image-size",
            str(image_size),
        ]
        if n_images is not None:
            make_dataset_args.extend(["--n-images", str(n_images)])
        run_command(
            _recovar_argv(*make_dataset_args),
            "Generate a small test dataset",
            "make_test_dataset",
        )

        common_pos = [
            _p("test_dataset", f"particles.{image_size}.mrcs"),
            "--poses",
            _p("test_dataset", "poses.pkl"),
            "--ctf",
            _p("test_dataset", "ctf.pkl"),
        ]

        run_command(
            pipeline_argv(
                *common_pos,
                "--correct-contrast",
                "--mask=from_halfmaps",
                "--lazy",
                "--ignore-zero-frequency",
                output_path=_p("test_dataset", "pipeline_output"),
                extras=[],
            ),
            "Run pipeline (variant 1)",
            "pipeline",
        )

        # Variant 2 is a near-duplicate of variant 1 (no --ignore-zero-frequency)
        # that doubles the wall-clock cost of the default smoke. Gate it
        # behind --full and --all-tests so the default smoke completes in a
        # few minutes (issue #143).
        if args.full or do_all_tests:
            run_command(
                pipeline_argv(
                    *common_pos,
                    "--correct-contrast",
                    "--mask=from_halfmaps",
                    "--lazy",
                    output_path=_p("test_dataset", "pipeline_output"),
                    extras=[],
                ),
                "Run pipeline (variant 2)",
                "pipeline",
            )

        run_command(
            _recovar_argv(
                "analyze",
                _p("test_dataset", "pipeline_output"),
                "--zdim=2",
                "--no-z-regularization",
                "--n-clusters=3",
                "--n-trajectories=0",
            ),
            "Run analyze",
            "analyze",
        )

        run_command(
            _recovar_argv(
                "estimate_conformational_density",
                _p("test_dataset", "pipeline_output"),
                "--pca_dim",
                "2",
            ),
            "Estimate conformational density",
            "estimate_conformational_density",
        )

        if do_all_tests:
            K = 2

            run_command(
                _recovar_argv(
                    "pipeline_with_outliers",
                    *common_pos,
                    "--correct-contrast",
                    "-o",
                    _p("test_dataset", "pipeline_with_outliers_output"),
                    "--mask=from_halfmaps",
                    "--lazy",
                    "--zdim",
                    "4",
                    "--k-rounds",
                    str(K),
                ),
                f"Run pipeline_with_outliers for {K} rounds",
                "pipeline_with_outliers",
            )

            run_command(
                _recovar_argv(
                    "analyze",
                    _p("test_dataset", "pipeline_output"),
                    "--zdim=2",
                    "--no-z-regularization",
                    "--n-clusters=3",
                    "--n-trajectories=1",
                    "--density",
                    _p("test_dataset", "pipeline_output", "density", "data", "deconv_density_knee.pkl"),
                    "--skip-centers",
                ),
                "Run analyze with density",
                "analyze",
            )

            run_command(
                _recovar_argv(
                    "compute_trajectory",
                    _p("test_dataset", "pipeline_output"),
                    "-o",
                    _p("test_dataset", "pipeline_output", "trajectory1"),
                    "--endpts",
                    _p("test_dataset", "pipeline_output", "analysis_2_noreg", "kmeans", "centers.txt"),
                    "--ind=0,1",
                    "--density",
                    _p("test_dataset", "pipeline_output", "density", "data", "deconv_density_knee.pkl"),
                    "--zdim=2",
                    "--n-vols-along-path=3",
                ),
                "Compute trajectory (option 1)",
                "compute_trajectory (option 1)",
            )

            run_command(
                _recovar_argv(
                    "compute_trajectory",
                    _p("test_dataset", "pipeline_output"),
                    "-o",
                    _p("test_dataset", "pipeline_output", "trajectory2"),
                    "--z_st",
                    _p(
                        "test_dataset",
                        "pipeline_output",
                        "analysis_2_noreg",
                        "kmeans",
                        "diagnostics",
                        "center000",
                        "latent_coords.txt",
                    ),
                    "--z_end",
                    _p(
                        "test_dataset",
                        "pipeline_output",
                        "analysis_2_noreg",
                        "kmeans",
                        "diagnostics",
                        "center002",
                        "latent_coords.txt",
                    ),
                    "--density",
                    _p("test_dataset", "pipeline_output", "density", "data", "deconv_density_knee.pkl"),
                    "--zdim=2",
                    "--n-vols-along-path=0",
                ),
                "Compute trajectory (option 2)",
                "compute_trajectory (option 2)",
            )

            run_command(
                _recovar_argv(
                    "estimate_stable_states",
                    _p(
                        "test_dataset",
                        "pipeline_output",
                        "density",
                        "data",
                        "all_densities",
                        "deconv_density_1.pkl",
                    ),
                    "--percent_top=10",
                    "--n_local_maxs=-1",
                    "-o",
                    _p("test_dataset", "pipeline_output", "stable_states"),
                ),
                "Estimate stable states",
                "estimate_stable_states",
            )

            target_path = _p("test_dataset", "target.txt")
            target_parent = os.path.dirname(target_path)
            if os.path.isdir(target_parent):
                with open(target_path, "w", encoding="utf-8") as fh:
                    fh.write("0.0 0.0\n")
                passed_functions.append("create_target")
            else:
                logger.warning(
                    "Skipping create_target: parent directory %s does not exist.",
                    target_parent,
                )

            pipeline_output_path = _p("test_dataset", "pipeline_output")
            embedding_2_path = _p("test_dataset", "embedding_2.pkl")
            if not os.path.exists(pipeline_output_path):
                logger.error("Failed: prepare embedding for reconstruction (missing %s)", pipeline_output_path)
                failed_functions.append("prepare_embedding_for_reconstruct")
            else:
                try:
                    po_tmp = output.PipelineOutput(pipeline_output_path)
                    latent_coords_2 = po_tmp.get_embedding_component("latent_coords", 2)
                    with open(embedding_2_path, "wb") as f:
                        pickle.dump(latent_coords_2, f)
                    del po_tmp
                except Exception as e:
                    logger.error("Failed: prepare embedding for reconstruction (%s)", e)
                    failed_functions.append("prepare_embedding_for_reconstruct")
                else:
                    run_command(
                        _recovar_argv(
                            "reconstruct_from_external_embedding",
                            _p("test_dataset", "particles.64.mrcs"),
                            "--poses",
                            _p("test_dataset", "poses.pkl"),
                            "--ctf",
                            _p("test_dataset", "ctf.pkl"),
                            "--embedding",
                            embedding_2_path,
                            "--target",
                            target_path,
                            "-o",
                            _p("test_dataset", "reconstruct_output"),
                        ),
                        "Test reconstruct_from_external_embedding",
                        "reconstruct",
                    )

    if failed_functions:
        logger.error("The following functions failed:")
        for func in failed_functions:
            logger.error("  - %s", func)
        logger.error("Please check the output above for details.")
        sys.exit(1)
    else:
        logger.info("All functions completed successfully!")
        if delete_everything:
            for path in cleanup_paths:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    logger.info("Deleted test directory: %s", path)


if __name__ == "__main__":
    main()
