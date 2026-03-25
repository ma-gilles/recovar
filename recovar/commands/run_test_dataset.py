import argparse
import logging
import os
import pickle
import shlex
import shutil
import subprocess
import sys

import jax

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run integration tests for recovar")
    parser.add_argument("--output-dir", "-o", default="/tmp/")
    parser.add_argument("--all-tests", action="store_true", help="Run all tests")
    parser.add_argument("--tilt-series-only", action="store_true", help="Run only tilt series tests")
    parser.add_argument(
        "--no-delete", action="store_true", help="Do not delete the test dataset directory after successful tests"
    )
    parser.add_argument("--cpu", action="store_true", help="Run on CPU only (skip GPU check)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    do_all_tests = args.all_tests
    tilt_series_only = args.tilt_series_only
    delete_everything = not args.no_delete
    run_on_cpu = args.cpu
    dataset_dir = args.output_dir

    BASE_CMD = "recovar"

    passed_functions = []
    failed_functions = []
    cleanup_paths = []

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

    def run_command(command, description, function_name):
        logger.info("Running: %s", description)
        logger.info("Command: %s", command)
        result = subprocess.run(command, shell=True)
        if result.returncode == 0:
            logger.info("Success: %s", description)
            passed_functions.append(function_name)
        else:
            logger.error("Failed: %s", description)
            failed_functions.append(function_name)

    cpu_string = " --accept-cpu" if run_on_cpu else ""

    def _p(*parts):
        return os.path.join(dataset_dir, *parts)

    def _pq(*parts):
        return shlex.quote(_p(*parts))

    dataset_dir_q = shlex.quote(dataset_dir)

    if tilt_series_only:
        logger.info("Running tilt series tests only...")
        cleanup_paths.append(os.path.join(dataset_dir, "tilt_test"))

        run_command(
            f"{BASE_CMD} make_test_dataset {_pq('tilt_test')} --n-images 10000 --tilt-series",
            "Generate a test dataset for tilt series",
            "make_test_dataset_tilt",
        )

        run_command(
            f"{BASE_CMD} pipeline {_pq('tilt_test', 'test_dataset', 'particles.star')} --poses {_pq('tilt_test', 'test_dataset', 'poses.pkl')} --ctf {_pq('tilt_test', 'test_dataset', 'ctf.pkl')} --tilt-series --tilt-series-ctf=relion5 --correct-contrast -o {_pq('tilt_test', 'test_dataset', 'pipeline_tilt_output')} --mask=from_halfmaps --lazy --ignore-zero-frequency {cpu_string}",
            "Run pipeline with tilt series",
            "pipeline_tilt",
        )

        run_command(
            f"{BASE_CMD} analyze {_pq('tilt_test', 'test_dataset', 'pipeline_tilt_output')} --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0",
            "Run analyze with tilt series",
            "analyze_tilt",
        )

        run_command(
            f'echo "0.0 0.0" > {_pq("tilt_test", "test_dataset", "target.txt")}',
            "Create target file for tilt series reconstruction",
            "create_target_tilt",
        )

        run_command(
            f"{BASE_CMD} reconstruct_from_external_embedding {_pq('tilt_test', 'test_dataset', 'particles.star')} --poses {_pq('tilt_test', 'test_dataset', 'poses.pkl')} --ctf {_pq('tilt_test', 'test_dataset', 'ctf.pkl')} --tilt-series --embedding {_pq('tilt_test', 'test_dataset', 'pipeline_tilt_output', 'embeddings.pkl')} --target {_pq('tilt_test', 'test_dataset', 'target.txt')} -o {_pq('tilt_test', 'test_dataset', 'reconstruct_tilt_output')}",
            "Test reconstruct_from_external_embedding with tilt series",
            "reconstruct_tilt",
        )

    else:
        cleanup_paths.append(os.path.join(dataset_dir, "test_dataset"))
        run_command(
            f"{BASE_CMD} make_test_dataset {dataset_dir_q}",
            "Generate a small test dataset",
            "make_test_dataset",
        )

        run_command(
            f"{BASE_CMD} pipeline {_pq('test_dataset', 'particles.64.mrcs')} --poses {_pq('test_dataset', 'poses.pkl')} --ctf {_pq('test_dataset', 'ctf.pkl')} --correct-contrast -o {_pq('test_dataset', 'pipeline_output')} --mask=from_halfmaps --lazy --ignore-zero-frequency {cpu_string}",
            "Run pipeline (variant 1)",
            "pipeline",
        )

        run_command(
            f"{BASE_CMD} pipeline {_pq('test_dataset', 'particles.64.mrcs')} --poses {_pq('test_dataset', 'poses.pkl')} --ctf {_pq('test_dataset', 'ctf.pkl')} --correct-contrast -o {_pq('test_dataset', 'pipeline_output')} --mask=from_halfmaps --lazy {cpu_string}",
            "Run pipeline (variant 2)",
            "pipeline",
        )

        run_command(
            f"{BASE_CMD} analyze {_pq('test_dataset', 'pipeline_output')} --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0",
            "Run analyze",
            "analyze",
        )

        run_command(
            f"{BASE_CMD} estimate_conformational_density {_pq('test_dataset', 'pipeline_output')} --pca_dim 2",
            "Estimate conformational density",
            "estimate_conformational_density",
        )

        if do_all_tests:
            K = 2

            run_command(
                f"{BASE_CMD} pipeline_with_outliers {_pq('test_dataset', 'particles.64.mrcs')} --poses {_pq('test_dataset', 'poses.pkl')} --ctf {_pq('test_dataset', 'ctf.pkl')} --correct-contrast -o {_pq('test_dataset', 'pipeline_with_outliers_output')} --mask=from_halfmaps --lazy --zdim 4 --k-rounds {K}",
                f"Run pipeline_with_outliers for {K} rounds",
                "pipeline_with_outliers",
            )

            run_command(
                f"{BASE_CMD} analyze {_pq('test_dataset', 'pipeline_output')} --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=1 --density {_pq('test_dataset', 'pipeline_output', 'density', 'data', 'deconv_density_knee.pkl')} --skip-centers",
                "Run analyze with density",
                "analyze",
            )

            run_command(
                f"{BASE_CMD} compute_trajectory {_pq('test_dataset', 'pipeline_output')} -o {_pq('test_dataset', 'pipeline_output', 'trajectory1')} --endpts {_pq('test_dataset', 'pipeline_output', 'analysis_2_noreg', 'kmeans', 'centers.txt')} --ind=0,1 --density {_pq('test_dataset', 'pipeline_output', 'density', 'data', 'deconv_density_knee.pkl')} --zdim=2 --n-vols-along-path=3",
                "Compute trajectory (option 1)",
                "compute_trajectory (option 1)",
            )

            run_command(
                f"{BASE_CMD} compute_trajectory {_pq('test_dataset', 'pipeline_output')} -o {_pq('test_dataset', 'pipeline_output', 'trajectory2')} --z_st {_pq('test_dataset', 'pipeline_output', 'analysis_2_noreg', 'kmeans', 'diagnostics', 'center000', 'latent_coords.txt')} --z_end {_pq('test_dataset', 'pipeline_output', 'analysis_2_noreg', 'kmeans', 'diagnostics', 'center002', 'latent_coords.txt')} --density {_pq('test_dataset', 'pipeline_output', 'density', 'data', 'deconv_density_knee.pkl')} --zdim=2 --n-vols-along-path=0",
                "Compute trajectory (option 2)",
                "compute_trajectory (option 2)",
            )

            run_command(
                f"{BASE_CMD} estimate_stable_states {_pq('test_dataset', 'pipeline_output', 'density', 'data', 'all_densities', 'deconv_density_1.pkl')} --percent_top=10 --n_local_maxs=-1 -o {_pq('test_dataset', 'pipeline_output', 'stable_states')}",
                "Estimate stable states",
                "estimate_stable_states",
            )

            run_command(
                f'echo "0.0 0.0" > {_pq("test_dataset", "target.txt")}',
                "Create target file for reconstruction",
                "create_target",
            )

            embedding_model_path = _p("test_dataset", "pipeline_output", "model", "embeddings.pkl")
            embedding_2_path = _p("test_dataset", "embedding_2.pkl")
            if not os.path.exists(embedding_model_path):
                logger.error("Failed: prepare embedding for reconstruction (missing %s)", embedding_model_path)
                failed_functions.append("prepare_embedding_for_reconstruct")
            else:
                try:
                    with open(embedding_model_path, "rb") as f:
                        embeddings = pickle.load(f)
                    with open(embedding_2_path, "wb") as f:
                        pickle.dump(embeddings["latent_coords"][2], f)
                except Exception as e:
                    logger.error("Failed: prepare embedding for reconstruction (%s)", e)
                    failed_functions.append("prepare_embedding_for_reconstruct")
                else:
                    run_command(
                        f"{BASE_CMD} reconstruct_from_external_embedding {_pq('test_dataset', 'particles.64.mrcs')} --poses {_pq('test_dataset', 'poses.pkl')} --ctf {_pq('test_dataset', 'ctf.pkl')} --embedding {shlex.quote(embedding_2_path)} --target {_pq('test_dataset', 'target.txt')} -o {_pq('test_dataset', 'reconstruct_output')}",
                        "Test reconstruct_from_external_embedding",
                        "reconstruct",
                    )

    if failed_functions:
        logger.error("The following functions failed:")
        for func in failed_functions:
            logger.error("  - %s", func)
        logger.error("Please check the output above for details.")
    else:
        logger.info("All functions completed successfully!")
        if delete_everything:
            for path in cleanup_paths:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    logger.info("Deleted test directory: %s", path)


if __name__ == "__main__":
    main()
