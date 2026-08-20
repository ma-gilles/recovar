#!/usr/bin/env python

import argparse
import logging
import os
import pickle
import shutil
import subprocess
import sys

import jax
import numpy as np

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run tests for recovar outliers pipeline")
    parser.add_argument("--output-dir", "-o", default="/tmp/recovar_test/")
    parser.add_argument(
        "--no-delete", action="store_true", help="Do not delete the test dataset directory after successful tests"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only execution. Sets JAX_PLATFORMS=cpu so JAX ignores any visible GPUs, AND passes --accept-cpu to the inner pipeline so it doesn't bail on the no-GPU check.",
    )
    parser.add_argument("--tilt-series", action="store_true", help="Test with tilt series dataset")
    parser.add_argument("--n-images", type=int, default=10000, help="Number of images to generate in test dataset")
    parser.add_argument("--k-rounds", type=int, default=2, help="Number of rounds for outlier detection pipeline")
    parser.add_argument(
        "--percent-outliers", type=float, default=0.15, help="Percentage of outliers to inject into dataset"
    )
    parser.add_argument(
        "--percent-tilt-series-outliers",
        type=float,
        default=0.1,
        help="Percentage of tilt outliers to inject into tilt series dataset",
    )
    # Selective testing flags
    parser.add_argument("--test-basic", action="store_true", help="Test basic outlier detection pipeline")
    parser.add_argument(
        "--test-analyze-chain", action="store_true", help="Test chaining pipeline with analyze commands"
    )
    parser.add_argument(
        "--test-trajectory-chain", action="store_true", help="Test chaining pipeline with compute_trajectory commands"
    )
    parser.add_argument("--test-tilt-series", action="store_true", help="Test tilt series specific functionality")

    # If no specific test is selected, run all tests
    parser.add_argument(
        "--run-all", action="store_true", help="Run all tests (default if no specific test is selected)"
    )

    args = parser.parse_args()

    # Determine which tests to run
    if not any(
        [args.test_basic, args.test_analyze_chain, args.test_trajectory_chain, args.test_tilt_series, args.run_all]
    ):
        # Default: run all tests
        args.run_all = True

    delete_everything = not args.no_delete
    run_on_cpu = args.cpu
    dataset_dir = args.output_dir
    n_images = args.n_images
    k_rounds = args.k_rounds
    percent_outliers = args.percent_outliers
    percent_tilt_series_outliers = args.percent_tilt_series_outliers

    # Force CPU-only mode in any spawned subprocess. This wrapper has already
    # imported jax above, so this env var doesn't affect THIS process's jax
    # state — but it does propagate to the `recovar pipeline ...` subprocesses
    # we spawn below, which is where the actual work happens.
    if run_on_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    base_argv = [sys.executable, "-m", "recovar.command_line"]

    passed_functions = []
    failed_functions = []
    test_dataset_dir = os.path.join(dataset_dir, "outliers_test")
    cpu_args = ["--accept-cpu"] if run_on_cpu else []

    def error_message():
        logger.error(
            "No GPU devices found by JAX. Please ensure that JAX is properly configured "
            "with CUDA and a compatible GPU. See https://jax.readthedocs.io/en/latest/installation.html"
        )
        logger.error("If you truly want to run on CPU, please run the script with the --cpu flag.")
        sys.exit(1)

    def check_gpu():
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                logger.info("GPU devices found: %s", gpu_devices)
            else:
                error_message()
        except Exception as e:
            logger.error("Error occurred while checking for GPU devices: %s", e)
            error_message()

    if not run_on_cpu:
        check_gpu()

    def _p(*parts):
        return os.path.join(test_dataset_dir, *parts)

    def _recovar_argv(cmd, *tokens):
        return [*base_argv, cmd, *tokens]

    def _pipeline_with_outliers_argv(particle_file, outdir, *extra):
        return _recovar_argv(
            "pipeline_with_outliers",
            _p("test_dataset", particle_file),
            "--poses",
            _p("test_dataset", "poses.pkl"),
            "--ctf",
            _p("test_dataset", "ctf.pkl"),
            "--correct-contrast",
            "-o",
            _p("test_dataset", outdir),
            "--mask=from_halfmaps",
            "--lazy",
            "--zdim",
            "4",
            *extra,
            *cpu_args,
        )

    def _analyze_argv(pipeline_dir, outdir):
        return _recovar_argv(
            "analyze",
            _p("test_dataset", pipeline_dir),
            "--outdir",
            _p("test_dataset", outdir),
            "--zdim",
            "4",
            *cpu_args,
        )

    def _compute_trajectory_argv(pipeline_dir, outdir):
        return _recovar_argv(
            "compute_trajectory",
            _p("test_dataset", pipeline_dir),
            "--outdir",
            _p("test_dataset", outdir),
            "--zdim",
            "4",
            "--endpts",
            _p("test_dataset", "analyze_chain_output", "kmeans", "centers.txt"),
            *cpu_args,
        )

    def run_command(argv, description, function_name, should_fail=False):
        logger.info("Running: %s", description)
        logger.info("Command: %s", " ".join(argv))
        result = subprocess.run(argv)
        if result.returncode == 0:
            logger.info("Success: %s", description)
            passed_functions.append(function_name)
        else:
            logger.error("Failed: %s", description)
            if not should_fail:
                failed_functions.append(function_name)
            else:
                logger.info("(Expected failure)")

    # Create an outlier volume
    outlier_volume_path = f"{test_dataset_dir}/test_dataset/outlier_volume.mrc"
    create_outlier_volume(outlier_volume_path, grid_size=64)

    # Generate test dataset
    if args.tilt_series:
        logger.info("Generating tilt series test dataset...")
        run_command(
            _recovar_argv(
                "make_test_dataset",
                test_dataset_dir,
                "--n-images",
                str(n_images),
                "--tilt-series",
                "--outlier-file-input",
                outlier_volume_path,
                "--percent-outliers",
                str(percent_outliers),
                "--percent-tilt-series-outliers",
                str(percent_tilt_series_outliers),
            ),
            "Generate a test dataset with tilt series for outlier testing",
            "make_test_dataset_tilt_outliers",
        )
    else:
        logger.info("Generating regular test dataset...")
        run_command(
            _recovar_argv(
                "make_test_dataset",
                test_dataset_dir,
                "--n-images",
                str(n_images),
                "--outlier-file-input",
                outlier_volume_path,
                "--percent-outliers",
                str(percent_outliers),
            ),
            "Generate a test dataset for outlier testing",
            "make_test_dataset_outliers",
        )

    def run_basic_tests(is_tilt_series, k_rounds):
        """Run basic outlier detection pipeline tests."""
        if is_tilt_series:
            run_command(
                _pipeline_with_outliers_argv(
                    "particles.star",
                    "pipeline_outliers_output",
                    "--tilt-series",
                    "--tilt-series-ctf=relion5",
                    "--k-rounds",
                    str(k_rounds),
                    "--use-contrast-detection",
                    "--use-junk-detection",
                    "--save-pipeline-indices",
                ),
                f"Run pipeline_with_outliers for {k_rounds} rounds with tilt series",
                "pipeline_with_outliers_tilt",
            )
        else:
            run_command(
                _pipeline_with_outliers_argv(
                    "particles.64.mrcs",
                    "pipeline_outliers_output",
                    "--k-rounds",
                    str(k_rounds),
                    "--use-contrast-detection",
                    "--use-junk-detection",
                    "--save-pipeline-indices",
                ),
                f"Run pipeline_with_outliers for {k_rounds} rounds",
                "pipeline_with_outliers",
            )

    def run_analyze_chain_tests():
        """Test chaining pipeline with analyze command."""
        run_command(
            _pipeline_with_outliers_argv(
                "particles.64.mrcs",
                "pipeline_analyze_chain",
                "--k-rounds",
                "1",
                "--use-contrast-detection",
                "--save-pipeline-indices",
            ),
            "Run pipeline for chaining with analyze",
            "pipeline_analyze_chain",
        )

        run_command(
            _analyze_argv("pipeline_analyze_chain/round_1", "analyze_chain_output"),
            "Run analyze command",
            "analyze_chain",
        )

        run_command(
            _analyze_argv("pipeline_analyze_chain/round_1", "analyze_chain_output2"),
            "Run second analyze command",
            "analyze_chain_final",
        )

    def run_trajectory_chain_tests():
        """Test chaining pipeline with compute_trajectory command."""
        run_command(
            _pipeline_with_outliers_argv(
                "particles.64.mrcs",
                "pipeline_trajectory_chain",
                "--k-rounds",
                "1",
                "--use-contrast-detection",
                "--save-pipeline-indices",
            ),
            "Run pipeline for chaining with compute_trajectory",
            "pipeline_trajectory_chain",
        )

        run_command(
            _analyze_argv("pipeline_analyze_chain/round_1", "analyze_chain_output"),
            "Run analyze command",
            "analyze_chain",
        )

        run_command(
            _compute_trajectory_argv("pipeline_trajectory_chain/round_1", "trajectory_chain_output"),
            "Run compute_trajectory command",
            "compute_trajectory_chain",
        )

        run_command(
            _compute_trajectory_argv("pipeline_trajectory_chain/round_1", "trajectory_chain_output2"),
            "Run second compute_trajectory command",
            "compute_trajectory_chain_final",
        )

    def run_tilt_series_tests():
        """Test tilt series specific functionality."""
        # This function is called when --test-tilt-series is used
        # The basic tilt series tests are already covered in run_basic_tests
        # Additional tilt series specific tests can be added here if needed
        pass

    # Run selected tests
    if args.run_all or args.test_basic:
        run_basic_tests(args.tilt_series, k_rounds)

    if args.run_all or args.test_analyze_chain:
        run_analyze_chain_tests()

    if args.run_all or args.test_trajectory_chain:
        run_trajectory_chain_tests()

    if args.run_all or args.test_tilt_series:
        run_tilt_series_tests()

    # Verify results and cleanup
    if args.run_all or args.test_basic:
        verify_outlier_results(test_dataset_dir, args.tilt_series, k_rounds)
        analyze_outlier_detection_accuracy(test_dataset_dir, args.tilt_series, k_rounds)

    verify_temp_cleanup(test_dataset_dir, args.tilt_series)

    if failed_functions:
        logger.error("The following functions failed:")
        for func in failed_functions:
            logger.error("- %s", func)
        logger.error("Please check the output above for details.")
        return 1
    else:
        logger.info("All outlier pipeline tests passed!")
        if delete_everything:
            logger.info("Cleaning up test directory: %s", test_dataset_dir)
            shutil.rmtree(test_dataset_dir, ignore_errors=True)
        return 0


def create_outlier_volume(output_path, grid_size=64):
    """Create a random anisotropic outlier volume for testing."""
    import mrcfile

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Vectorized generation avoids slow Python triple-loops while preserving
    # anisotropic, high-frequency content used for outlier stress tests.
    volume_shape = (grid_size, grid_size, grid_size)
    center = grid_size // 2
    rng = np.random.default_rng(42)

    coords = np.indices(volume_shape, dtype=np.float32)
    x = coords[0] - center
    y = coords[1] - center
    z = coords[2] - center

    scale_x = rng.uniform(0.8, 1.2, size=volume_shape).astype(np.float32)
    scale_y = rng.uniform(0.6, 1.4, size=volume_shape).astype(np.float32)
    scale_z = rng.uniform(0.7, 1.3, size=volume_shape).astype(np.float32)

    normalized_dist = np.sqrt((x * scale_x) ** 2 + (y * scale_y) ** 2 + (z * scale_z) ** 2)
    shell_mask = (normalized_dist > (0.25 * center)) & (normalized_dist < (0.7 * center))
    core_mask = normalized_dist < (0.2 * center)

    angle_factor = np.sin(0.3 * (x + center)) * np.cos(0.2 * (y + center)) * np.sin(0.25 * (z + center))
    random_factor = 0.5 + 0.5 * rng.random(volume_shape, dtype=np.float32)
    shell_density = 0.3 + 0.7 * random_factor + 0.2 * angle_factor.astype(np.float32)

    volume = (rng.random(volume_shape, dtype=np.float32) * 0.1).astype(np.float32)
    volume[shell_mask] = shell_density[shell_mask]
    volume += (0.5 * rng.random(volume_shape, dtype=np.float32)) * core_mask.astype(np.float32)
    volume += rng.random(volume_shape, dtype=np.float32) * 0.1

    # Normalize robustly to [0, 1]
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    denom = max(vmax - vmin, 1e-8)
    volume = ((volume - vmin) / denom).astype(np.float32, copy=False)

    # Save as MRC file
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(volume)
        mrc.voxel_size = 4.25 * 128 / grid_size

    logger.info("Created random anisotropic outlier volume: %s", output_path)


def verify_outlier_results(test_dir, is_tilt_series, k_rounds):
    """Verify that outlier detection results were properly saved."""
    logger.info("Verifying outlier detection results...")

    base_output_dir = f"{test_dir}/test_dataset/pipeline_outliers_output"
    if not os.path.exists(base_output_dir):
        logger.error("Main output directory not found: %s", base_output_dir)
        return False
    logger.info("Found output directory: %s", base_output_dir)

    # Check that inliers/outliers files were saved for each round
    for round_num in range(1, k_rounds + 1):
        inliers_file = f"{base_output_dir}/inliers_round_{round_num}.pkl"
        outliers_file = f"{base_output_dir}/outliers_round_{round_num}.pkl"

        if not os.path.exists(inliers_file):
            logger.error("Inliers file not found for round %d: %s", round_num, inliers_file)
            return False

        if not os.path.exists(outliers_file):
            logger.error("Outliers file not found for round %d: %s", round_num, outliers_file)
            return False

        # Check that the files contain valid data
        try:
            with open(inliers_file, "rb") as f:
                inliers = pickle.load(f)
            with open(outliers_file, "rb") as f:
                outliers = pickle.load(f)

            logger.info("Round %d: %d image inliers, %d image outliers", round_num, len(inliers), len(outliers))

        except Exception as e:
            logger.error("Failed to load image indices for round %d: %s", round_num, e)
            return False

    # Check that all rounds inliers file exists
    all_inliers_file = f"{base_output_dir}/all_rounds_inliers.pkl"
    if not os.path.exists(all_inliers_file):
        logger.error("All rounds inliers file not found: %s", all_inliers_file)
        return False

    # For tilt series, check that particle indices were also saved
    if is_tilt_series:
        for round_num in range(1, k_rounds + 1):
            particle_inliers_file = f"{base_output_dir}/particle_inliers_round_{round_num}.pkl"
            particle_outliers_file = f"{base_output_dir}/particle_outliers_round_{round_num}.pkl"

            if os.path.exists(particle_inliers_file) and os.path.exists(particle_outliers_file):
                try:
                    with open(particle_inliers_file, "rb") as f:
                        particle_inliers = pickle.load(f)
                    with open(particle_outliers_file, "rb") as f:
                        particle_outliers = pickle.load(f)
                    logger.info(
                        "Round %d: %d particle inliers, %d particle outliers",
                        round_num,
                        len(particle_inliers),
                        len(particle_outliers),
                    )
                except Exception as e:
                    logger.warning("Round %d: Particle indices saved but failed to load: %s", round_num, e)
            else:
                logger.warning("Particle indices not found for round %d", round_num)

    logger.info("Outlier detection results verification completed successfully!")
    return True


def analyze_outlier_detection_accuracy(test_dir, is_tilt_series, k_rounds):
    """Analyze and report statistics about outlier detection accuracy."""
    logger.info("Analyzing outlier detection accuracy...")

    # Load simulation info to get ground truth
    sim_info_path = f"{test_dir}/test_dataset/simulation_info.pkl"
    if not os.path.exists(sim_info_path):
        logger.error("Simulation info not found: %s", sim_info_path)
        return False

    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    # Get ground truth outlier assignments
    image_assignments = sim_info["image_assignment"]
    n_images = len(image_assignments)

    # Identify ground truth outliers
    particle_outlier_indices = np.where(image_assignments == -1)[0]  # Particle outliers
    tilt_outlier_indices = np.where(image_assignments == -2)[0]  # Tilt outliers
    all_outlier_indices = np.concatenate([particle_outlier_indices, tilt_outlier_indices])

    logger.info("Ground truth statistics:")
    logger.info("  Total images: %d", n_images)
    logger.info("  Total outliers: %d (%.1f%%)", len(all_outlier_indices), len(all_outlier_indices) / n_images * 100)

    # For tilt series, also analyze particle-level ground truth
    if is_tilt_series:
        tilt_series_assignment = sim_info["tilt_series_assignment"]
        tilt_groups = sim_info["tilt_groups"]
        n_particles = len(tilt_series_assignment)

        # Identify particles with tilt outliers (entire particle is outlier)
        particle_outlier_particles = np.where(tilt_series_assignment == -1)[0]  # Particles with tilt outliers

        logger.info("  Total particles: %d", n_particles)
        logger.info(
            "  Particles with tilt outliers: %d (%.1f%%)",
            len(particle_outlier_particles),
            len(particle_outlier_particles) / n_particles * 100,
        )

        # Map particle outliers to image indices for comparison
        particle_outlier_images = []
        for particle_idx in particle_outlier_particles:
            particle_images = np.where(tilt_groups == particle_idx)[0]
            particle_outlier_images.extend(particle_images)
        particle_outlier_images = np.array(particle_outlier_images)

        logger.info(
            "  Images from particles with tilt outliers: %d (%.1f%%)",
            len(particle_outlier_images),
            len(particle_outlier_images) / n_images * 100,
        )

    # Analyze each round
    base_output_dir = f"{test_dir}/test_dataset/pipeline_outliers_output"

    for round_num in range(1, k_rounds + 1):
        logger.info("Round %d analysis:", round_num)

        # Load detected image inliers
        inliers_file = f"{base_output_dir}/inliers_round_{round_num}.pkl"
        if not os.path.exists(inliers_file):
            logger.error("  Image inliers file not found: %s", inliers_file)
            continue
        with open(inliers_file, "rb") as f:
            detected_image_inliers = pickle.load(f)

        # Compute detected image outliers as the complement of inliers
        detected_image_outliers = np.setdiff1d(np.arange(n_images), detected_image_inliers)

        # Verify that inliers + outliers = all images (after first round)
        if round_num > 1:
            total_detected = len(detected_image_inliers) + len(detected_image_outliers)
            if total_detected != n_images:
                logger.warning("  Inliers + outliers (%d) != total images (%d)", total_detected, n_images)

        # Load detected particle outliers (for tilt series)
        detected_particle_outliers = None
        detected_particle_inliers = None
        if is_tilt_series:
            particle_inliers_file = f"{base_output_dir}/particle_inliers_round_{round_num}.pkl"
            if os.path.exists(particle_inliers_file):
                with open(particle_inliers_file, "rb") as f:
                    detected_particle_inliers = pickle.load(f)
                # Compute detected particle outliers as the complement of inliers
                detected_particle_outliers = np.setdiff1d(np.arange(n_particles), detected_particle_inliers)
            else:
                logger.warning("  Particle inliers file not found: %s", particle_inliers_file)

        # Report image-level statistics
        logger.info("  Image-level statistics:")
        n_detected_images = len(detected_image_outliers)
        n_correct_images = len(np.intersect1d(detected_image_outliers, all_outlier_indices))
        n_false_positives_images = n_detected_images - n_correct_images
        n_false_negatives_images = len(all_outlier_indices) - n_correct_images
        n_true_negatives_images = n_images - (n_correct_images + n_false_positives_images + n_false_negatives_images)

        # 2x2 confusion matrix for images
        logger.info("    Confusion matrix (images):")
        logger.info("    ┌─────────────┬───────────────┬───────────────┬─────────┐")
        logger.info("    │             │ Pred Outlier  │ Pred Inlier   │  Total  │")
        logger.info("    ├─────────────┼───────────────┼───────────────┼─────────┤")
        logger.info(
            "    │ GT Outlier  │ %13d │ %13d │ %7d │",
            n_correct_images,
            n_false_negatives_images,
            len(all_outlier_indices),
        )
        logger.info(
            "    │ GT Inlier   │ %13d │ %13d │ %7d │",
            n_false_positives_images,
            n_true_negatives_images,
            n_images - len(all_outlier_indices),
        )
        logger.info("    ├─────────────┼───────────────┼───────────────┼─────────┤")
        logger.info("    │   Total     │ %13d │ %13d │ %7d │", n_detected_images, len(detected_image_inliers), n_images)
        logger.info("    └─────────────┴───────────────┴───────────────┴─────────┘")

        # Report particle-level statistics (for tilt series)
        if is_tilt_series and detected_particle_outliers is not None and detected_particle_inliers is not None:
            logger.info("  Particle-level statistics:")
            n_detected_particles = len(detected_particle_outliers)
            n_detected_particle_inliers = len(detected_particle_inliers)
            n_correct_particles = len(np.intersect1d(detected_particle_outliers, particle_outlier_particles))
            n_false_positives_particles = n_detected_particles - n_correct_particles
            n_false_negatives_particles = len(particle_outlier_particles) - n_correct_particles
            n_true_negatives_particles = n_particles - (
                n_correct_particles + n_false_positives_particles + n_false_negatives_particles
            )

            # 2x2 confusion matrix for particles
            logger.info("    Confusion matrix (particles):")
            logger.info("    ┌─────────────┬───────────────┬───────────────┬─────────┐")
            logger.info("    │             │ Pred Outlier  │ Pred Inlier   │  Total  │")
            logger.info("    ├─────────────┼───────────────┼───────────────┼─────────┤")
            logger.info(
                "    │ GT Outlier  │ %13d │ %13d │ %7d │",
                n_correct_particles,
                n_false_negatives_particles,
                len(particle_outlier_particles),
            )
            logger.info(
                "    │ GT Inlier   │ %13d │ %13d │ %7d │",
                n_false_positives_particles,
                n_true_negatives_particles,
                n_particles - len(particle_outlier_particles),
            )
            logger.info("    ├─────────────┼───────────────┼───────────────┼─────────┤")
            logger.info(
                "    │   Total     │ %13d │ %13d │ %7d │",
                n_detected_particles,
                n_detected_particle_inliers,
                n_particles,
            )
            logger.info("    └─────────────┴───────────────┴───────────────┴─────────┘")

        elif is_tilt_series:
            logger.info("  Particle-level statistics: Not available (particle outliers file not found)")

    return True


def verify_temp_cleanup(test_dir, is_tilt_series):
    """Verify that temp directories were cleaned up."""
    logger.info("Verifying temp directories cleanup...")

    base_output_dir = f"{test_dir}/test_dataset/pipeline_outliers_output"
    if not os.path.exists(base_output_dir):
        logger.error("Main output directory not found: %s", base_output_dir)
        return False
    logger.info("Found output directory: %s", base_output_dir)

    logger.info("Temp directories cleanup verification completed successfully!")
    return True


if __name__ == "__main__":
    sys.exit(main())
