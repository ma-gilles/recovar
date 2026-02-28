import sys
import subprocess
import shutil
import os
import shlex
import jax
import argparse
import pickle

def main():

    parser = argparse.ArgumentParser(description="Run tests for recovar")
    parser.add_argument('--output-dir', '-o', default = '/tmp/')
    parser.add_argument('--all-tests', action='store_true', help='Run all tests')
    parser.add_argument('--tilt-series-only', action='store_true', help='Run only tilt series tests')
    parser.add_argument('--no-delete', action='store_true', help='Do not delete the test dataset directory after successful tests')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU only (skip GPU check)')
    args = parser.parse_args()

    do_all_tests = args.all_tests
    tilt_series_only = args.tilt_series_only
    delete_everything = not args.no_delete
    run_on_cpu = args.cpu
    dataset_dir = args.output_dir

    # Base command is now "recovar" (which dispatches to the appropriate subcommand)
    BASE_CMD = "recovar"

    passed_functions = []
    failed_functions = []
    cleanup_paths = []

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

    def _p(*parts):
        return os.path.join(dataset_dir, *parts)

    def _pq(*parts):
        return shlex.quote(_p(*parts))

    dataset_dir_q = shlex.quote(dataset_dir)

    if tilt_series_only:
        # Run only tilt series tests
        print("Running tilt series tests only...")
        cleanup_paths.append(os.path.join(dataset_dir, "tilt_test"))
        
        # Generate a test dataset for tilt series testing (without nested structure)
        run_command(
            f'{BASE_CMD} make_test_dataset {_pq("tilt_test")} --n-images 10000 --tilt-series',
            'Generate a test dataset for tilt series',
            'make_test_dataset_tilt'
        )

        # Test pipeline with tilt series functionality
        run_command(
            f'{BASE_CMD} pipeline {_pq("tilt_test", "test_dataset", "particles.star")} --poses {_pq("tilt_test", "test_dataset", "poses.pkl")} --ctf {_pq("tilt_test", "test_dataset", "ctf.pkl")} --tilt-series --tilt-series-ctf=relion5 --correct-contrast -o {_pq("tilt_test", "test_dataset", "pipeline_tilt_output")} --mask=from_halfmaps --lazy --ignore-zero-frequency {cpu_string}',
            'Run pipeline with tilt series functionality',
            'pipeline_tilt'
        )

        # Run analyze with tilt series functionality
        run_command(
            f'{BASE_CMD} analyze {_pq("tilt_test", "test_dataset", "pipeline_tilt_output")} --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0',
            'Run analyze with tilt series',
            'analyze_tilt'
        )

        # Create a simple target file for tilt series reconstruction testing
        run_command(
            f'echo "0.0 0.0" > {_pq("tilt_test", "test_dataset", "target.txt")}',
            'Create target file for tilt series reconstruction',
            'create_target_tilt'
        )

        # Test reconstruct_from_external_embedding with tilt series
        run_command(
            f'{BASE_CMD} reconstruct_from_external_embedding {_pq("tilt_test", "test_dataset", "particles.star")} --poses {_pq("tilt_test", "test_dataset", "poses.pkl")} --ctf {_pq("tilt_test", "test_dataset", "ctf.pkl")} --tilt-series --embedding {_pq("tilt_test", "test_dataset", "pipeline_tilt_output", "embeddings.pkl")} --target {_pq("tilt_test", "test_dataset", "target.txt")} -o {_pq("tilt_test", "test_dataset", "reconstruct_tilt_output")}',
            'Test reconstruct_from_external_embedding with tilt series',
            'reconstruct_tilt'
        )
        
    else:
        # Generate a small test dataset - should take about 30 sec
        cleanup_paths.append(os.path.join(dataset_dir, "test_dataset"))
        run_command(
            f'{BASE_CMD} make_test_dataset {dataset_dir_q}',
            'Generate a small test dataset',
            'make_test_dataset'
        )

        # Run pipeline, first variant - should take about 2 min
        run_command(
            f'{BASE_CMD} pipeline {_pq("test_dataset", "particles.64.mrcs")} --poses {_pq("test_dataset", "poses.pkl")} --ctf {_pq("test_dataset", "ctf.pkl")} --correct-contrast -o {_pq("test_dataset", "pipeline_output")} --mask=from_halfmaps --lazy --ignore-zero-frequency {cpu_string}',
            'Run pipeline (variant 1)',
            'pipeline'
        )

        # Run pipeline, second variant - should take about 2 min
        run_command(
            f'{BASE_CMD} pipeline {_pq("test_dataset", "particles.64.mrcs")} --poses {_pq("test_dataset", "poses.pkl")} --ctf {_pq("test_dataset", "ctf.pkl")} --correct-contrast -o {_pq("test_dataset", "pipeline_output")} --mask=from_halfmaps --lazy {cpu_string}',
            'Run pipeline (variant 2)',
            'pipeline'
        )

        # Run analyze with 2D embedding and no z-regularization on latent space (better for density estimation) - should take about 5 min
        run_command(
            f'{BASE_CMD} analyze {_pq("test_dataset", "pipeline_output")} --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0',
            'Run analyze',
            'analyze'
        )

        # Estimate conformational density
        run_command(
            f'{BASE_CMD} estimate_conformational_density {_pq("test_dataset", "pipeline_output")} --pca_dim 2',
            'Estimate conformational density',
            'estimate_conformational_density'
        )

        if do_all_tests:
            # Set the number of rounds K for the outlier detection pipeline
            K = 2  # Adjust K as needed

            # Run pipeline_with_outliers with K rounds
            run_command(
                f'{BASE_CMD} pipeline_with_outliers {_pq("test_dataset", "particles.64.mrcs")} --poses {_pq("test_dataset", "poses.pkl")} --ctf {_pq("test_dataset", "ctf.pkl")} --correct-contrast -o {_pq("test_dataset", "pipeline_with_outliers_output")} --mask=from_halfmaps --lazy --zdim 4 --k-rounds {K}',
                f'Run pipeline_with_outliers for {K} rounds',
                'pipeline_with_outliers'
            )

            # Run analyze with density and trajectory estimation - should take about 5 min
            run_command(
                f'{BASE_CMD} analyze {_pq("test_dataset", "pipeline_output")} --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=1 --density {_pq("test_dataset", "pipeline_output", "density", "deconv_density_knee.pkl")} --skip-centers',
                'Run analyze with density',
                'analyze'
            )

            # Compute trajectory - option 1
            run_command(
                f'{BASE_CMD} compute_trajectory {_pq("test_dataset", "pipeline_output")} -o {_pq("test_dataset", "pipeline_output", "trajectory1")} --endpts {_pq("test_dataset", "pipeline_output", "analysis_2_noreg", "kmeans", "centers.txt")} --ind=0,1 --density {_pq("test_dataset", "pipeline_output", "density", "deconv_density_knee.pkl")} --zdim=2 --n-vols-along-path=3',
                'Compute trajectory (option 1)',
                'compute_trajectory (option 1)'
            )

            # Compute trajectory - option 2
            run_command(
                f'{BASE_CMD} compute_trajectory {_pq("test_dataset", "pipeline_output")} -o {_pq("test_dataset", "pipeline_output", "trajectory2")} --z_st {_pq("test_dataset", "pipeline_output", "analysis_2_noreg", "kmeans", "diagnostics", "center000", "latent_coords.txt")} --z_end {_pq("test_dataset", "pipeline_output", "analysis_2_noreg", "kmeans", "diagnostics", "center002", "latent_coords.txt")} --density {_pq("test_dataset", "pipeline_output", "density", "deconv_density_knee.pkl")} --zdim=2 --n-vols-along-path=0',
                'Compute trajectory (option 2)',
                'compute_trajectory (option 2)'
            )

            # Run estimate_stable_states
            run_command(
                f'{BASE_CMD} estimate_stable_states {_pq("test_dataset", "pipeline_output", "density", "all_densities", "deconv_density_1.pkl")} --percent_top=10 --n_local_maxs=-1 -o {_pq("test_dataset", "pipeline_output", "stable_states")}',
                'Estimate stable states',
                'estimate_stable_states'
            )

            # Create a simple target file for reconstruction testing
            run_command(
                f'echo "0.0 0.0" > {_pq("test_dataset", "target.txt")}',
                'Create target file for reconstruction',
                'create_target'
            )
            # Test reconstruct_from_external_embedding
            embedding_model_path = _p("test_dataset", "pipeline_output", "model", "embeddings.pkl")
            embedding_2_path = _p("test_dataset", "embedding_2.pkl")
            if not os.path.exists(embedding_model_path):
                print(f"Failed: prepare embedding for reconstruction (missing {embedding_model_path})\n")
                failed_functions.append('prepare_embedding_for_reconstruct')
            else:
                try:
                    with open(embedding_model_path, 'rb') as f:
                        embeddings = pickle.load(f)
                    with open(embedding_2_path, 'wb') as f:
                        pickle.dump(embeddings['zs'][2], f)
                except Exception as e:
                    print(f"Failed: prepare embedding for reconstruction ({e})\n")
                    failed_functions.append('prepare_embedding_for_reconstruct')
                else:
                    run_command(
                        f'{BASE_CMD} reconstruct_from_external_embedding {_pq("test_dataset", "particles.64.mrcs")} --poses {_pq("test_dataset", "poses.pkl")} --ctf {_pq("test_dataset", "ctf.pkl")} --embedding {shlex.quote(embedding_2_path)} --target {_pq("test_dataset", "target.txt")} -o {_pq("test_dataset", "reconstruct_output")}',
                        'Test reconstruct_from_external_embedding',
                        'reconstruct'
                    )
            

    if failed_functions:
        print("The following functions failed:")
        for func in failed_functions:
            print(f"- {func}")
        print("\nPlease check the output above for details.")
    else:
        print("All functions completed successfully!")
        # Delete the test_dataset directory since all steps passed
        if delete_everything:
            deleted_any = False
            for path in cleanup_paths:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"Test dataset directory '{path}' has been deleted.")
                    deleted_any = True
            if not deleted_any:
                print("No generated test dataset directories found to delete.")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=__doc__)
    # args = parser.parse_args()
    main()
