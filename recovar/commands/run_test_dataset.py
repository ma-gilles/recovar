"""Smoke-test the recovar install end-to-end on a tiny synthetic dataset.

Two design contracts:

1. Arguments are forwarded to inner ``recovar pipeline ...`` etc. calls
   as ``argv`` lists, never shell strings. This makes logging / quoting
   reliable and lets us cleanly capture stderr+stdout for the
   error-hints classifier on failure.

2. When the user passes ``--gpu-gb`` to constrain the smoke test (e.g.
   to verify the install on a smaller GPU), we automatically splice
   ``--adaptive-n-pcs`` into every inner pipeline call so the run
   finishes. This wrapper exists to answer "is your install correct?",
   not to act as a science test — robustness wins. Pass
   ``--full-memory-test`` to force the default 200-PC configuration.
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


# Flags this wrapper consumes; everything else is forwarded.
_WRAPPER_OPTS = {
    "--output-dir",
    "-o",
    "--all-tests",
    "--tilt-series-only",
    "--no-delete",
    "--cpu",
    "--full-memory-test",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run integration tests for recovar")
    parser.add_argument("--output-dir", "-o", default="/tmp/")
    parser.add_argument("--all-tests", action="store_true", help="Run all tests")
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
            "Use this when you want the smoke test to exercise the default 200-PC "
            "configuration even on a constrained --gpu-gb budget."
        ),
    )

    # Memory-planning surface: --gpu-gb / --gpu-memory + low/very-low-memory +
    # adaptive aliases + diagnostics + fail-on-memory-exceed +
    # memory-safety-fraction + hard-gpu-memory-limit.
    from recovar.utils.parser_args import add_memory_planning_args

    add_memory_planning_args(parser)
    return parser


def _build_forward_argv(args: argparse.Namespace) -> list[str]:
    """Return the list of argv tokens to splice into every inner pipeline call.

    Handles the auto-adaptive default: when --gpu-gb is set and
    --full-memory-test is NOT, splice in --adaptive-n-pcs so the smoke
    test always finishes.
    """
    fwd: list[str] = []
    if args.gpu_memory is not None:
        fwd += ["--gpu-gb", str(args.gpu_memory)]
    if args.low_memory_option:
        fwd.append("--low-memory-option")
    if args.very_low_memory_option:
        fwd.append("--very-low-memory-option")

    auto_adaptive = args.gpu_memory is not None and not args.full_memory_test and not args.adaptive_memory
    if args.adaptive_memory or auto_adaptive:
        fwd.append("--adaptive-n-pcs")
        if auto_adaptive:
            logger.info(
                "run_test_dataset: enabling --adaptive-n-pcs for inner pipeline "
                "calls because --gpu-gb was supplied. Pass --full-memory-test to "
                "force the default 200-PC configuration."
            )

    if args.memory_diagnostics:
        fwd.append("--memory-diagnostics")
    if args.fail_on_memory_exceed:
        fwd.append("--fail-on-memory-exceed")
    if args.memory_safety_fraction is not None:
        fwd += ["--memory-safety-fraction", str(args.memory_safety_fraction)]
    if args.hard_gpu_memory_limit:
        fwd.append("--hard-gpu-memory-limit")
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

    if run_on_cpu:
        # Force CPU-only mode in spawned subprocesses. This wrapper itself
        # has already imported jax, so the env var doesn't affect THIS
        # process — but it propagates to the child pipelines.
        os.environ["JAX_PLATFORMS"] = "cpu"

    base_argv = [sys.executable, "-m", "recovar.command_line"]
    cpu_extra = ["--accept-cpu"] if run_on_cpu else []

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
        """
        logger.info("Running: %s", description)
        logger.info("Command: %s", " ".join(argv))
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
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
        try:
            from recovar.utils import error_hints

            ctx = error_hints.collect_context()
            hint = error_hints.classify_subprocess_failure(result.stderr or "", result.stdout or "", ctx)
            if hint is not None:
                sys.stderr.write("\n")
                sys.stderr.write(error_hints.format_error_hint(hint))
                sys.stderr.flush()
        except Exception as exc:
            logger.debug("Could not classify subprocess failure: %s", exc)
        return False

    def _p(*parts):
        return os.path.join(dataset_dir, *parts)

    def pipeline_argv(*pos: str, output_path: str, extras: list[str]) -> list[str]:
        return [
            *base_argv,
            "pipeline",
            *pos,
            "-o",
            output_path,
            *extras,
            *cpu_extra,
            *forward_argv,
        ]

    if tilt_series_only:
        logger.info("Running tilt series tests only...")
        cleanup_paths.append(os.path.join(dataset_dir, "tilt_test"))

        run_command(
            [*base_argv, "make_test_dataset", _p("tilt_test"), "--n-images", "10000", "--tilt-series"],
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
            [
                *base_argv,
                "analyze",
                _p("tilt_test", "test_dataset", "pipeline_tilt_output"),
                "--zdim=2",
                "--no-z-regularization",
                "--n-clusters=3",
                "--n-trajectories=0",
            ],
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
            [
                *base_argv,
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
                *forward_argv,
            ],
            "Test reconstruct_from_external_embedding with tilt series",
            "reconstruct_tilt",
        )

    else:
        cleanup_paths.append(os.path.join(dataset_dir, "test_dataset"))
        run_command(
            [*base_argv, "make_test_dataset", dataset_dir],
            "Generate a small test dataset",
            "make_test_dataset",
        )

        common_pos = [
            _p("test_dataset", "particles.64.mrcs"),
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
            [
                *base_argv,
                "analyze",
                _p("test_dataset", "pipeline_output"),
                "--zdim=2",
                "--no-z-regularization",
                "--n-clusters=3",
                "--n-trajectories=0",
                *forward_argv,
            ],
            "Run analyze",
            "analyze",
        )

        run_command(
            [
                *base_argv,
                "estimate_conformational_density",
                _p("test_dataset", "pipeline_output"),
                "--pca_dim",
                "2",
            ],
            "Estimate conformational density",
            "estimate_conformational_density",
        )

        if do_all_tests:
            K = 2

            run_command(
                [
                    *base_argv,
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
                    *forward_argv,
                ],
                f"Run pipeline_with_outliers for {K} rounds",
                "pipeline_with_outliers",
            )

            run_command(
                [
                    *base_argv,
                    "analyze",
                    _p("test_dataset", "pipeline_output"),
                    "--zdim=2",
                    "--no-z-regularization",
                    "--n-clusters=3",
                    "--n-trajectories=1",
                    "--density",
                    _p("test_dataset", "pipeline_output", "density", "data", "deconv_density_knee.pkl"),
                    "--skip-centers",
                    *forward_argv,
                ],
                "Run analyze with density",
                "analyze",
            )

            run_command(
                [
                    *base_argv,
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
                    *forward_argv,
                ],
                "Compute trajectory (option 1)",
                "compute_trajectory (option 1)",
            )

            run_command(
                [
                    *base_argv,
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
                    *forward_argv,
                ],
                "Compute trajectory (option 2)",
                "compute_trajectory (option 2)",
            )

            run_command(
                [
                    *base_argv,
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
                ],
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
                        [
                            *base_argv,
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
                            *forward_argv,
                        ],
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
