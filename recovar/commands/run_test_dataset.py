import subprocess
import shutil
import os
import jax
import sys
import argparse

def main():

    parser = argparse.ArgumentParser(description="Run tests for recovar")
    parser.add_argument('--output-dir', '-o', default = '/tmp/')
    parser.add_argument('--all-tests', action='store_true', help='Run all tests')
    parser.add_argument('--no-delete', action='store_true', help='Do not delete the test dataset directory after successful tests')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU only (skip GPU check)')
    args = parser.parse_args()

    do_all_tests = args.all_tests
    delete_everything = not args.no_delete
    run_on_cpu = args.cpu
    dataset_dir = args.output_dir

    # Base command is now "recovar" (which dispatches to the appropriate subcommand)
    BASE_CMD = "recovar"

    passed_functions = []
    failed_functions = []

    def error_message():
        print("--------------------------------------------")
        print("--------------------------------------------")
        print("No GPU devices found by JAX. Please ensure that JAX is properly configured with CUDA and a compatible GPU. Some info from the JAX website (https://jax.readthedocs.io/en/latest/installation.html):\n"
              "You must first install the NVIDIA driver. It is recommended to install the newest driver available from NVIDIA, but the driver version must be >= 525.60.13 for CUDA 12 on Linux. Then reinstall jax as follows:\n"
              "pip uninstall jax jaxlib; \n pip install -U \"jax[cuda12]\"==0.5.0")
        print("If you truly want to run on CPU, please run the script with the --cpu flag. Note that while this test will run, a real dataset will be extremely slow on CPU.")
        print("--------------------------------------------")
        print("--------------------------------------------")
        exit(1)

    def check_gpu():
        try:
            gpu_devices = jax.devices('gpu')
            if gpu_devices:
                print("GPU devices found:", gpu_devices)
            else:
                error_message()
        except Exception as e:
            print("Error occurred while checking for GPU devices:", e)
            error_message()

    if not run_on_cpu:
        check_gpu()

    def run_command(command, description, function_name):
        print(f"Running: {description}")
        print(f"Command: {command}\n")
        result = subprocess.run(command, shell=True)
        if result.returncode == 0:
            print(f"Success: {description}\n")
            passed_functions.append(function_name)
        else:
            print(f"Failed: {description}\n")
            failed_functions.append(function_name)

    cpu_string = " --accept-cpu" if run_on_cpu else ""

    

    # Generate a small test dataset - should take about 30 sec
    run_command(
        f'{BASE_CMD} make_test_dataset {dataset_dir}',
        'Generate a small test dataset',
        'make_test_dataset'
    )

    # Run pipeline, first variant - should take about 2 min
    run_command(
        f'{BASE_CMD} pipeline {dataset_dir}/test_dataset/particles.64.mrcs --poses {dataset_dir}/test_dataset/poses.pkl --ctf {dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {dataset_dir}/test_dataset/pipeline_output --mask=from_halfmaps --lazy --ignore-zero-frequency {cpu_string}',
        'Run pipeline (variant 1)',
        'pipeline'
    )

    # Run pipeline, second variant - should take about 2 min
    run_command(
        f'{BASE_CMD} pipeline {dataset_dir}/test_dataset/particles.64.mrcs --poses {dataset_dir}/test_dataset/poses.pkl --ctf {dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {dataset_dir}/test_dataset/pipeline_output --mask=from_halfmaps --lazy {cpu_string}',
        'Run pipeline (variant 2)',
        'pipeline'
    )

    # Run analyze with 2D embedding and no z-regularization on latent space (better for density estimation) - should take about 5 min
    run_command(
        f'{BASE_CMD} analyze {dataset_dir}/test_dataset/pipeline_output --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0',
        'Run analyze',
        'analyze'
    )

    # Estimate conformational density
    run_command(
        f'{BASE_CMD} estimate_conformational_density {dataset_dir}/test_dataset/pipeline_output --pca_dim 2',
        'Estimate conformational density',
        'estimate_conformational_density'
    )

    if do_all_tests:
        # Set the number of rounds K for the outlier detection pipeline
        K = 2  # Adjust K as needed

        # Generate a test dataset with nested structure for strip_prefix testing
        run_command(
            f'{BASE_CMD} make_test_dataset {dataset_dir} --create-nested-structure --nested-prefix Extract/job193',
            'Generate a test dataset with nested structure',
            'make_test_dataset_nested'
        )

        # Test pipeline with strip_prefix functionality
        run_command(
            f'{BASE_CMD} pipeline {dataset_dir}/test_dataset/particles.star --poses {dataset_dir}/test_dataset/poses.pkl --ctf {dataset_dir}/test_dataset/ctf.pkl --strip-prefix Extract/job193 --correct-contrast -o {dataset_dir}/test_dataset/pipeline_strip_prefix_output --mask=from_halfmaps --lazy --ignore-zero-frequency {cpu_string}',
            'Run pipeline with strip_prefix functionality',
            'pipeline_strip_prefix'
        )

        # Run analyze with strip_prefix functionality
        run_command(
            f'{BASE_CMD} analyze {dataset_dir}/test_dataset/pipeline_strip_prefix_output --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0',
            'Run analyze with strip_prefix',
            'analyze_strip_prefix'
        )

        # Generate a test dataset with nested structure for tilt series testing
        run_command(
            f'{BASE_CMD} make_test_dataset {dataset_dir} --create-nested-structure --nested-prefix Extract/job193 --n-images 100',
            'Generate a test dataset with nested structure for tilt series',
            'make_test_dataset_nested_tilt'
        )

        # Test pipeline with strip_prefix and tilt series functionality
        run_command(
            f'{BASE_CMD} pipeline {dataset_dir}/test_dataset/particles.star --poses {dataset_dir}/test_dataset/poses.pkl --ctf {dataset_dir}/test_dataset/ctf.pkl --strip-prefix Extract/job193 --tilt-series --tilt-series-ctf=relion5 --correct-contrast -o {dataset_dir}/test_dataset/pipeline_strip_prefix_tilt_output --mask=from_halfmaps --lazy --ignore-zero-frequency {cpu_string}',
            'Run pipeline with strip_prefix and tilt series functionality',
            'pipeline_strip_prefix_tilt'
        )

        # Run analyze with strip_prefix and tilt series functionality
        run_command(
            f'{BASE_CMD} analyze {dataset_dir}/test_dataset/pipeline_strip_prefix_tilt_output --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0',
            'Run analyze with strip_prefix and tilt series',
            'analyze_strip_prefix_tilt'
        )

        # Run pipeline_with_outliers with K rounds
        run_command(
            f'{BASE_CMD} pipeline_with_outliers {dataset_dir}/test_dataset/particles.star --poses {dataset_dir}/test_dataset/poses.pkl --ctf {dataset_dir}/test_dataset/ctf.pkl --strip-prefix Extract/job193 --correct-contrast -o {dataset_dir}/test_dataset/pipeline_with_outliers_output --mask=from_halfmaps --lazy --zdim 4 --k-rounds {K}',
            f'Run pipeline_with_outliers for {K} rounds',
            'pipeline_with_outliers'
        )

        # Run analyze with density and trajectory estimation - should take about 5 min
        run_command(
            f'{BASE_CMD} analyze {dataset_dir}/test_dataset/pipeline_output --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=1 --density {dataset_dir}/test_dataset/pipeline_output/density/deconv_density_knee.pkl --skip-centers',
            'Run analyze with density',
            'analyze'
        )

        # Compute trajectory - option 1
        run_command(
            f'{BASE_CMD} compute_trajectory {dataset_dir}/test_dataset/pipeline_output -o {dataset_dir}/test_dataset/pipeline_output/trajectory1 --endpts {dataset_dir}/test_dataset/pipeline_output/analysis_2_noreg/kmeans_center_coords.txt --ind=0,1 --density {dataset_dir}/test_dataset/pipeline_output/density/deconv_density_knee.pkl --zdim=2 --n-vols-along-path=3',
            'Compute trajectory (option 1)',
            'compute_trajectory (option 1)'
        )

        # Compute trajectory - option 2
        run_command(
            f'{BASE_CMD} compute_trajectory {dataset_dir}/test_dataset/pipeline_output -o {dataset_dir}/test_dataset/pipeline_output/trajectory2 --z_st {dataset_dir}/test_dataset/pipeline_output/analysis_2_noreg/kmeans_center_volumes/vol0000/latent_coords.txt --z_end {dataset_dir}/test_dataset/pipeline_output/analysis_2_noreg/kmeans_center_volumes/vol0002/latent_coords.txt --density {dataset_dir}/test_dataset/pipeline_output/density/deconv_density_knee.pkl --zdim=2 --n-vols-along-path=0',
            'Compute trajectory (option 2)',
            'compute_trajectory (option 2)'
        )

        # Run estimate_stable_states
        run_command(
            f'{BASE_CMD} estimate_stable_states {dataset_dir}/test_dataset/pipeline_output/density/all_densities/deconv_density_1.pkl --percent_top=10 --n_local_maxs=-1 -o {dataset_dir}/test_dataset/pipeline_output/stable_states',
            'Estimate stable states',
            'estimate_stable_states'
        )

        # Test reconstruct_from_external_embedding with strip_prefix
        run_command(
            f'{BASE_CMD} reconstruct_from_external_embedding {dataset_dir}/test_dataset/particles.star --poses {dataset_dir}/test_dataset/poses.pkl --ctf {dataset_dir}/test_dataset/ctf.pkl --strip-prefix Extract/job193 --embedding {dataset_dir}/test_dataset/pipeline_strip_prefix_output/embeddings.pkl --output {dataset_dir}/test_dataset/reconstruct_strip_prefix_output --mask=from_halfmaps --lazy --ignore-zero-frequency {cpu_string}',
            'Test reconstruct_from_external_embedding with strip_prefix',
            'reconstruct_strip_prefix'
        )

        # Test reconstruct_from_external_embedding with strip_prefix and tilt series
        run_command(
            f'{BASE_CMD} reconstruct_from_external_embedding {dataset_dir}/test_dataset/particles.star --poses {dataset_dir}/test_dataset/poses.pkl --ctf {dataset_dir}/test_dataset/ctf.pkl --strip-prefix Extract/job193 --tilt-series --tilt-series-ctf=relion5 --correct-contrast --embedding {dataset_dir}/test_dataset/pipeline_strip_prefix_tilt_output/embeddings.pkl --output {dataset_dir}/test_dataset/reconstruct_strip_prefix_tilt_output --mask=from_halfmaps --lazy --ignore-zero-frequency {cpu_string}',
            'Test reconstruct_from_external_embedding with strip_prefix and tilt series',
            'reconstruct_strip_prefix_tilt'
        )

    if failed_functions:
        print("The following functions failed:")
        for func in failed_functions:
            print(f"- {func}")
        print("\nPlease check the output above for details.")
    else:
        print("All functions completed successfully!")
        # Delete the test_dataset directory since all steps passed
        if delete_everything and os.path.exists('test_dataset'):
            shutil.rmtree('test_dataset')
            print("Test dataset directory 'test_dataset' has been deleted.")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=__doc__)
    # args = parser.parse_args()
    main()
