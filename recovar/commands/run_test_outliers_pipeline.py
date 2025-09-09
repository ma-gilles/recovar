#!/usr/bin/env python

import subprocess
import shutil
import os
import jax
import sys
import argparse
import numpy as np
import pickle
import glob

def main():
    parser = argparse.ArgumentParser(description="Run tests for recovar outliers pipeline")
    parser.add_argument('--output-dir', '-o', default='/tmp/recovar_test/')
    parser.add_argument('--no-delete', action='store_true', help='Do not delete the test dataset directory after successful tests')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU only (skip GPU check)')
    parser.add_argument('--tilt-series', action='store_true', help='Test with tilt series dataset')
    parser.add_argument('--n-images', type=int, default=10000, help='Number of images to generate in test dataset')
    parser.add_argument('--k-rounds', type=int, default=2, help='Number of rounds for outlier detection pipeline')
    parser.add_argument('--percent-outliers', type=float, default=0.15, help='Percentage of outliers to inject into dataset')
    parser.add_argument('--percent-tilt-series-outliers', type=float, default=0.1, help='Percentage of tilt outliers to inject into tilt series dataset')
    parser.add_argument('--copy-to-folder-path', default='/tmp/recovar_tmp', help='Path for copy-to-folder functionality')
    
    # Selective testing flags
    parser.add_argument('--test-basic', action='store_true', help='Test basic outlier detection pipeline')
    parser.add_argument('--test-copy-to-folder', action='store_true', help='Test copy-to-folder functionality')
    parser.add_argument('--test-chaining', action='store_true', help='Test command chaining with --no-cleanup')
    parser.add_argument('--test-analyze-chain', action='store_true', help='Test chaining pipeline with analyze commands')
    parser.add_argument('--test-trajectory-chain', action='store_true', help='Test chaining pipeline with compute_trajectory commands')
    parser.add_argument('--test-tilt-series', action='store_true', help='Test tilt series specific functionality')
    
    # If no specific test is selected, run all tests
    parser.add_argument('--run-all', action='store_true', help='Run all tests (default if no specific test is selected)')
    
    args = parser.parse_args()

    # Determine which tests to run
    if not any([args.test_basic, args.test_copy_to_folder, args.test_chaining, 
                args.test_analyze_chain, args.test_trajectory_chain, args.test_tilt_series, args.run_all]):
        # Default: run all tests
        args.run_all = True
    
    delete_everything = not args.no_delete
    run_on_cpu = args.cpu
    dataset_dir = args.output_dir
    n_images = args.n_images
    k_rounds = args.k_rounds
    percent_outliers = args.percent_outliers
    percent_tilt_series_outliers = args.percent_tilt_series_outliers
    copy_to_folder_path = args.copy_to_folder_path

    # Base command is now "recovar" (which dispatches to the appropriate subcommand)
    BASE_CMD = "python -m recovar.command_line"

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

    def run_command(command, description, function_name, should_fail=False):
        print(f"Running: {description}")
        print(f"Command: {command}\n")
        result = subprocess.run(command, shell=True)
        if result.returncode == 0:
            print(f"Success: {description}\n")
            passed_functions.append(function_name)
        else:
            print(f"Failed: {description}\n")
            if not should_fail:
                failed_functions.append(function_name)
            else:
                print(f"But it was expected to fail!")

    cpu_string = " --accept-cpu" if run_on_cpu else ""

    # Create a test dataset with outliers
    test_dataset_dir = os.path.join(dataset_dir, "outliers_test")
    
    # Create an outlier volume
    outlier_volume_path = f"{test_dataset_dir}/test_dataset/outlier_volume.mrc"
    create_outlier_volume(outlier_volume_path, grid_size=64)
    
    # Generate test dataset
    if args.tilt_series:
        print("Generating tilt series test dataset...")
        run_command(
            f'{BASE_CMD} make_test_dataset {test_dataset_dir} --n-images {n_images} --tilt-series --outlier-file-input {outlier_volume_path} --percent-outliers {percent_outliers} --percent-tilt-series-outliers {percent_tilt_series_outliers}',
            'Generate a test dataset with tilt series for outlier testing',
            'make_test_dataset_tilt_outliers'
        )
    else:
        print("Generating regular test dataset...")
        run_command(
            f'{BASE_CMD} make_test_dataset {test_dataset_dir} --n-images {n_images} --outlier-file-input {outlier_volume_path} --percent-outliers {percent_outliers}',
            'Generate a test dataset for outlier testing',
            'make_test_dataset_outliers'
        )

    def run_basic_tests(test_dataset_dir, is_tilt_series, k_rounds, cpu_string):
        """Run basic outlier detection pipeline tests."""
        if is_tilt_series:
            run_command(
                f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.star --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --tilt-series --tilt-series-ctf=relion5 --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_outliers_output --mask=from_halfmaps --lazy --zdim 4 --k-rounds {k_rounds} --use-contrast-detection --use-junk-detection --save-pipeline-indices {cpu_string}',
                f'Run pipeline_with_outliers for {k_rounds} rounds with tilt series',
                'pipeline_with_outliers_tilt'
            )
        else:
            run_command(
                f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.64.mrcs --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_outliers_output --mask=from_halfmaps --lazy --zdim 4 --k-rounds {k_rounds} --use-contrast-detection --use-junk-detection --save-pipeline-indices {cpu_string}',
                f'Run pipeline_with_outliers for {k_rounds} rounds',
                'pipeline_with_outliers'
            )

    def run_copy_to_folder_tests(test_dataset_dir, is_tilt_series, cpu_string, copy_to_folder_path):
        """Test copy-to-folder functionality."""
        if is_tilt_series:
            copy_test_dir_tilt = os.path.join(test_dataset_dir, "copy_test_tilt")
            os.makedirs(copy_test_dir_tilt, exist_ok=True)
            run_command(
                f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.star --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --tilt-series --tilt-series-ctf=relion5 --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_outliers_copy_test_tilt --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --use-junk-detection --save-pipeline-indices --copy-to-folder {copy_test_dir_tilt} --datadir {test_dataset_dir}/test_dataset {cpu_string}',
                'Run pipeline_with_outliers with copy-to-folder option for tilt series',
                'pipeline_with_outliers_copy_test_tilt'
            )
            
            run_command(
                f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.star --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --tilt-series --tilt-series-ctf=relion5 --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_outliers_copy_test_tilt_no_cleanup --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --use-junk-detection --save-pipeline-indices --copy-to-folder {copy_test_dir_tilt} --no-cleanup --datadir {test_dataset_dir}/test_dataset {cpu_string}',
                'Run pipeline_with_outliers with copy-to-folder and no-cleanup options for tilt series (for chaining)',
                'pipeline_with_outliers_copy_test_tilt_no_cleanup'
            )
            # Add a test for missing file referenced in star file
            # Create a fake star file referencing a missing file
            missing_star = os.path.join(copy_test_dir_tilt, 'missing_file.star')
            with open(missing_star, 'w') as f:
                f.write('data_\nloop_\n_rlnImageName #1\n000001@missing_file.mrcs\n')
            try:
                run_command(
                    f'{BASE_CMD} pipeline_with_outliers {missing_star} --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --tilt-series --tilt-series-ctf=relion5 --correct-contrast -o {copy_test_dir_tilt}/fail_output --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --use-junk-detection --save-pipeline-indices --copy-to-folder {copy_test_dir_tilt} {cpu_string}',
                    'Expect failure: missing file referenced in star file with copy-to-folder',
                    'pipeline_with_outliers_copy_test_tilt_missing_file'
                )
            except Exception as e:
                print('Expected failure for missing file in star file:', e)
                passed_functions.append('pipeline_with_outliers_copy_test_tilt_missing_file')
        else:
            run_command(
                f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.64.mrcs --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_outliers_copy_test --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --use-junk-detection --save-pipeline-indices --copy-to-folder {copy_to_folder_path} {cpu_string}',
                'Run pipeline_with_outliers with copy-to-folder option',
                'pipeline_with_outliers_copy_test'
            )
            
            run_command(
                f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.64.mrcs --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_outliers_copy_test_no_cleanup --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --use-junk-detection --save-pipeline-indices --copy-to-folder {copy_to_folder_path} --no-cleanup {cpu_string}',
                'Run pipeline_with_outliers with copy-to-folder and no-cleanup options (for chaining)',
                'pipeline_with_outliers_copy_test_no_cleanup'
            )
            # Add a test for missing file referenced in star file
            missing_star = os.path.join(test_dataset_dir, 'missing_file.star')
            with open(missing_star, 'w') as f:
                f.write('data_\nloop_\n_rlnImageName #1\n000001@missing_file.mrcs\n')
            run_command(
                f'{BASE_CMD} pipeline_with_outliers {missing_star} --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {test_dataset_dir}/fail_output --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --use-junk-detection --save-pipeline-indices --copy-to-folder {copy_to_folder_path} {cpu_string}',
                'Expect failure: missing file referenced in star file with copy-to-folder',
                'pipeline_with_outliers_copy_test_missing_file',
                should_fail=True
            )

    def run_chaining_tests(test_dataset_dir, cpu_string, copy_to_folder_path):
        """Test chaining multiple pipeline calls with --no-cleanup."""
        run_command(
            f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.64.mrcs --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_chain_test1 --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --save-pipeline-indices --copy-to-folder {copy_to_folder_path} --no-cleanup {cpu_string}',
            'Run first pipeline call in chain (with --no-cleanup)',
            'pipeline_chain_test1'
        )
        
        run_command(
            f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.64.mrcs --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_chain_test2 --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --save-pipeline-indices --copy-to-folder {copy_to_folder_path} --no-cleanup {cpu_string}',
            'Run second pipeline call in chain (with --no-cleanup)',
            'pipeline_chain_test2'
        )
        
        run_command(
            f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.64.mrcs --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_chain_test3 --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --save-pipeline-indices --copy-to-folder {copy_to_folder_path} {cpu_string}',
            'Run third pipeline call in chain (final call, cleanup happens)',
            'pipeline_chain_test3'
        )

    def run_analyze_chain_tests(test_dataset_dir, cpu_string, copy_to_folder_path):
        """Test chaining pipeline with analyze command."""
        run_command(
            f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.64.mrcs --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_analyze_chain --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --save-pipeline-indices --copy-to-folder {copy_to_folder_path} --no-cleanup {cpu_string}',
            'Run pipeline with --no-cleanup for chaining with analyze',
            'pipeline_analyze_chain'
        )
        
        run_command(
            f'{BASE_CMD} analyze {test_dataset_dir}/test_dataset/pipeline_analyze_chain/round_1 --outdir {test_dataset_dir}/test_dataset/analyze_chain_output --zdim 4 --copy-to-folder {copy_to_folder_path} --no-cleanup {cpu_string}',
            'Run analyze command in chain (with --no-cleanup)',
            'analyze_chain'
        )
        
        run_command(
            f'{BASE_CMD} analyze {test_dataset_dir}/test_dataset/pipeline_analyze_chain/round_1 --outdir {test_dataset_dir}/test_dataset/analyze_chain_output2 --zdim 4 --copy-to-folder {copy_to_folder_path} {cpu_string}',
            'Run second analyze command in chain (final call, cleanup happens)',
            'analyze_chain_final'
        )

    def run_trajectory_chain_tests(test_dataset_dir, cpu_string, copy_to_folder_path):
        """Test chaining pipeline with compute_trajectory command."""
        run_command(
            f'{BASE_CMD} pipeline_with_outliers {test_dataset_dir}/test_dataset/particles.64.mrcs --poses {test_dataset_dir}/test_dataset/poses.pkl --ctf {test_dataset_dir}/test_dataset/ctf.pkl --correct-contrast -o {test_dataset_dir}/test_dataset/pipeline_trajectory_chain --mask=from_halfmaps --lazy --zdim 4 --k-rounds 1 --use-contrast-detection --save-pipeline-indices --copy-to-folder {copy_to_folder_path} --no-cleanup {cpu_string}',
            'Run pipeline with --no-cleanup for chaining with compute_trajectory',
            'pipeline_trajectory_chain'
        )
        
        run_command(
            f'{BASE_CMD} analyze {test_dataset_dir}/test_dataset/pipeline_analyze_chain/round_1 --outdir {test_dataset_dir}/test_dataset/analyze_chain_output --zdim 4 --copy-to-folder {copy_to_folder_path} --no-cleanup {cpu_string}',
            'Run analyze command in chain (with --no-cleanup)',
            'analyze_chain'
        )

        run_command(
            f'{BASE_CMD} compute_trajectory {test_dataset_dir}/test_dataset/pipeline_trajectory_chain/round_1 --outdir {test_dataset_dir}/test_dataset/trajectory_chain_output --zdim 4 --endpts {test_dataset_dir}/test_dataset/analyze_chain_output/kmeans_center_coords.txt --copy-to-folder {copy_to_folder_path} --no-cleanup {cpu_string}',
            'Run compute_trajectory command in chain (with --no-cleanup)',
            'compute_trajectory_chain'
        )
        
        run_command(
            f'{BASE_CMD} compute_trajectory {test_dataset_dir}/test_dataset/pipeline_trajectory_chain/round_1 --outdir {test_dataset_dir}/test_dataset/trajectory_chain_output2 --zdim 4 --endpts {test_dataset_dir}/test_dataset/analyze_chain_output/kmeans_center_coords.txt --copy-to-folder {copy_to_folder_path} {cpu_string}',
            'Run second compute_trajectory command in chain (final call, cleanup happens)',
            'compute_trajectory_chain_final'
        )

    def run_tilt_series_tests(test_dataset_dir, k_rounds, cpu_string):
        """Test tilt series specific functionality."""
        # This function is called when --test-tilt-series is used
        # The basic tilt series tests are already covered in run_basic_tests
        # Additional tilt series specific tests can be added here if needed
        pass


    # Run selected tests
    if args.run_all or args.test_basic:
        run_basic_tests(test_dataset_dir, args.tilt_series, k_rounds, cpu_string)
    
    if args.run_all or args.test_copy_to_folder:
        run_copy_to_folder_tests(test_dataset_dir, args.tilt_series, cpu_string, copy_to_folder_path)
    
    if args.run_all or args.test_chaining:
        run_chaining_tests(test_dataset_dir, cpu_string, copy_to_folder_path)
    
    if args.run_all or args.test_analyze_chain:
        run_analyze_chain_tests(test_dataset_dir, cpu_string, copy_to_folder_path)
    
    if args.run_all or args.test_trajectory_chain:
        run_trajectory_chain_tests(test_dataset_dir, cpu_string, copy_to_folder_path)
    
    if args.run_all or args.test_tilt_series:
        run_tilt_series_tests(test_dataset_dir, k_rounds, cpu_string)

    # Verify results and cleanup
    if args.run_all or args.test_basic:
        verify_outlier_results(test_dataset_dir, args.tilt_series, k_rounds)
        analyze_outlier_detection_accuracy(test_dataset_dir, args.tilt_series, k_rounds)
    
    verify_temp_cleanup(test_dataset_dir, args.tilt_series)

    if failed_functions:
        print("The following functions failed:")
        for func in failed_functions:
            print(f"- {func}")
        print("\nPlease check the output above for details.")
        return 1
    else:
        print("All outlier pipeline tests passed!")
        if delete_everything:
            print(f"Cleaning up test directory: {test_dataset_dir}")
            shutil.rmtree(test_dataset_dir, ignore_errors=True)
        return 0

def create_outlier_volume(output_path, grid_size=64):
    """Create a random anisotropic outlier volume for testing."""
    import mrcfile
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a random anisotropic volume with different characteristics
    volume_shape = (grid_size, grid_size, grid_size)
    center = grid_size // 2
    
    # Set random seed for reproducibility but different from normal volumes
    np.random.seed(42)  # Different seed from normal volumes
    
    # Create base volume with random noise
    volume = np.random.rand(*volume_shape).astype(np.float32) * 0.1
    
    # Add anisotropic features that look different from different directions
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # Distance from center
                dist = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                
                # Create anisotropic structure - ellipsoid with random orientation
                # Scale factors for anisotropy
                scale_x = 0.8 + 0.4 * np.random.rand()
                scale_y = 0.6 + 0.8 * np.random.rand() 
                scale_z = 0.7 + 0.6 * np.random.rand()
                
                # Ellipsoid equation with random orientation
                normalized_dist = np.sqrt(
                    ((i - center) * scale_x)**2 + 
                    ((j - center) * scale_y)**2 + 
                    ((k - center) * scale_z)**2
                )
                
                # Add some random protrusions and indentations
                angle_factor = np.sin(0.3 * i) * np.cos(0.2 * j) * np.sin(0.25 * k)
                random_factor = 0.5 + 0.5 * np.random.rand()
                
                # Create irregular surface
                if 0.25 * center < normalized_dist < 0.7 * center:
                    # Add varying density based on position
                    density = 0.3 + 0.7 * random_factor + 0.2 * angle_factor
                    volume[i, j, k] = density
                
                # Add some random internal structures
                if normalized_dist < 0.2 * center:
                    volume[i, j, k] += 0.5 * np.random.rand()
    
    # Add some high-frequency noise to make it more irregular
    noise = np.random.rand(*volume_shape).astype(np.float32) * 0.1
    volume += noise
    
    # Normalize the volume
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    
    # Save as MRC file
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(volume)
        mrc.voxel_size = 4.25 * 128 / grid_size
    
    print(f"Created random anisotropic outlier volume: {output_path}")

def verify_outlier_results(test_dir, is_tilt_series, k_rounds):
    """Verify that outlier detection results were properly saved."""
    print("\nVerifying outlier detection results...")
    
    # Check both possible output dirs
    base_output_dirs = [f"{test_dir}/test_dataset/pipeline_outliers_output"]
    if is_tilt_series:
        base_output_dirs.append(f"{test_dir}/test_dataset/pipeline_outliers_copy_test_tilt")
        base_output_dirs.append(f"{test_dir}/test_dataset/pipeline_outliers_copy_test_tilt_no_cleanup")
    else:
        base_output_dirs.append(f"{test_dir}/test_dataset/pipeline_outliers_copy_test")
        base_output_dirs.append(f"{test_dir}/test_dataset/pipeline_outliers_copy_test_no_cleanup")
    found = False
    for base_output_dir in base_output_dirs:
        if os.path.exists(base_output_dir):
            print(f"Found output directory: {base_output_dir}")
            found = True
            break
    if not found:
        print(f"ERROR: None of the main output directories found: {base_output_dirs}")
        return False
    
    # Check that inliers/outliers files were saved for each round
    for round_num in range(1, k_rounds + 1):
        inliers_file = f"{base_output_dir}/inliers_round_{round_num}.pkl"
        outliers_file = f"{base_output_dir}/outliers_round_{round_num}.pkl"
        
        if not os.path.exists(inliers_file):
            print(f"ERROR: Inliers file not found for round {round_num}: {inliers_file}")
            return False
        
        if not os.path.exists(outliers_file):
            print(f"ERROR: Outliers file not found for round {round_num}: {outliers_file}")
            return False
        
        # Check that the files contain valid data
        try:
            with open(inliers_file, 'rb') as f:
                inliers = pickle.load(f)
            with open(outliers_file, 'rb') as f:
                outliers = pickle.load(f)
            
            print(f"Round {round_num}: {len(inliers)} image inliers, {len(outliers)} image outliers")
            
        except Exception as e:
            print(f"ERROR: Failed to load image indices for round {round_num}: {e}")
            return False
    
    # Check that all rounds inliers file exists
    all_inliers_file = f"{base_output_dir}/all_rounds_inliers.pkl"
    if not os.path.exists(all_inliers_file):
        print(f"ERROR: All rounds inliers file not found: {all_inliers_file}")
        return False
    
    # For tilt series, check that particle indices were also saved
    if is_tilt_series:
        for round_num in range(1, k_rounds + 1):
            particle_inliers_file = f"{base_output_dir}/particle_inliers_round_{round_num}.pkl"
            particle_outliers_file = f"{base_output_dir}/particle_outliers_round_{round_num}.pkl"
            
            if os.path.exists(particle_inliers_file) and os.path.exists(particle_outliers_file):
                try:
                    with open(particle_inliers_file, 'rb') as f:
                        particle_inliers = pickle.load(f)
                    with open(particle_outliers_file, 'rb') as f:
                        particle_outliers = pickle.load(f)
                    print(f"Round {round_num}: {len(particle_inliers)} particle inliers, {len(particle_outliers)} particle outliers")
                except Exception as e:
                    print(f"Round {round_num}: Particle indices saved but failed to load: {e}")
            else:
                print(f"WARNING: Particle indices not found for round {round_num}")
    
    print("Outlier detection results verification completed successfully!")
    return True

def analyze_outlier_detection_accuracy(test_dir, is_tilt_series, k_rounds):
    """Analyze and report statistics about outlier detection accuracy."""
    print("\nAnalyzing outlier detection accuracy...")
    
    # Load simulation info to get ground truth
    sim_info_path = f"{test_dir}/test_dataset/simulation_info.pkl"
    if not os.path.exists(sim_info_path):
        print(f"ERROR: Simulation info not found: {sim_info_path}")
        return False
    
    with open(sim_info_path, 'rb') as f:
        sim_info = pickle.load(f)
    
    # Get ground truth outlier assignments
    image_assignments = sim_info['image_assignment']
    n_images = len(image_assignments)
    
    # Identify ground truth outliers
    particle_outlier_indices = np.where(image_assignments == -1)[0]  # Particle outliers
    tilt_outlier_indices = np.where(image_assignments == -2)[0]      # Tilt outliers
    all_outlier_indices = np.concatenate([particle_outlier_indices, tilt_outlier_indices])
    
    print(f"Ground truth statistics:")
    print(f"  Total images: {n_images}")
    # print(f"  Particle outliers: {len(particle_outlier_indices)} ({len(particle_outlier_indices)/n_images*100:.1f}%)")
    # print(f"  Tilt outliers: {len(tilt_outlier_indices)} ({len(tilt_outlier_indices)/n_images*100:.1f}%)")
    print(f"  Total outliers: {len(all_outlier_indices)} ({len(all_outlier_indices)/n_images*100:.1f}%)")
    
    # For tilt series, also analyze particle-level ground truth
    if is_tilt_series:
        tilt_series_assignment = sim_info['tilt_series_assignment']
        tilt_groups = sim_info['tilt_groups']
        n_particles = len(tilt_series_assignment)
        
        # Identify particles with tilt outliers (entire particle is outlier)
        particle_outlier_particles = np.where(tilt_series_assignment == -1)[0]  # Particles with tilt outliers
        
        print(f"  Total particles: {n_particles}")
        print(f"  Particles with tilt outliers: {len(particle_outlier_particles)} ({len(particle_outlier_particles)/n_particles*100:.1f}%)")
        
        # Map particle outliers to image indices for comparison
        particle_outlier_images = []
        for particle_idx in particle_outlier_particles:
            particle_images = np.where(tilt_groups == particle_idx)[0]
            particle_outlier_images.extend(particle_images)
        particle_outlier_images = np.array(particle_outlier_images)
        
        print(f"  Images from particles with tilt outliers: {len(particle_outlier_images)} ({len(particle_outlier_images)/n_images*100:.1f}%)")
    
    # Analyze each round
    base_output_dir = f"{test_dir}/test_dataset/pipeline_outliers_output"
    
    for round_num in range(1, k_rounds + 1):
        print(f"\nRound {round_num} analysis:")
        
        # Load detected image inliers
        inliers_file = f"{base_output_dir}/inliers_round_{round_num}.pkl"
        if not os.path.exists(inliers_file):
            print(f"  ERROR: Image inliers file not found: {inliers_file}")
            continue
        with open(inliers_file, 'rb') as f:
            detected_image_inliers = pickle.load(f)

        # Compute detected image outliers as the complement of inliers
        detected_image_outliers = np.setdiff1d(np.arange(n_images), detected_image_inliers)

        # Verify that inliers + outliers = all images (after first round)
        if round_num > 1:
            total_detected = len(detected_image_inliers) + len(detected_image_outliers)
            if total_detected != n_images:
                print(f"  WARNING: Inliers + outliers ({total_detected}) != total images ({n_images})")
        
        # Load detected particle outliers (for tilt series)
        detected_particle_outliers = None
        detected_particle_inliers = None
        if is_tilt_series:
            particle_inliers_file = f"{base_output_dir}/particle_inliers_round_{round_num}.pkl"
            if os.path.exists(particle_inliers_file):
                with open(particle_inliers_file, 'rb') as f:
                    detected_particle_inliers = pickle.load(f)
                # Compute detected particle outliers as the complement of inliers
                detected_particle_outliers = np.setdiff1d(np.arange(n_particles), detected_particle_inliers)
            else:
                print(f"  WARNING: Particle inliers file not found: {particle_inliers_file}")
        
        # Report image-level statistics
        print(f"  Image-level statistics:")
        n_detected_images = len(detected_image_outliers)
        n_correct_images = len(np.intersect1d(detected_image_outliers, all_outlier_indices))
        n_false_positives_images = n_detected_images - n_correct_images
        n_false_negatives_images = len(all_outlier_indices) - n_correct_images
        n_true_negatives_images = n_images - (n_correct_images + n_false_positives_images + n_false_negatives_images)

        # 2x2 confusion matrix for images (pretty print)
        print("    Confusion matrix (images):")
        print("    ┌─────────────┬───────────────┬───────────────┬─────────┐")
        print("    │             │ Pred Outlier  │ Pred Inlier   │  Total  │")
        print("    ├─────────────┼───────────────┼───────────────┼─────────┤")
        print(f"    │ GT Outlier  │ {n_correct_images:13d} │ {n_false_negatives_images:13d} │ {len(all_outlier_indices):7d} │")
        print(f"    │ GT Inlier   │ {n_false_positives_images:13d} │ {n_true_negatives_images:13d} │ {n_images - len(all_outlier_indices):7d} │")
        print("    ├─────────────┼───────────────┼───────────────┼─────────┤")
        print(f"    │   Total     │ {n_detected_images:13d} │ {len(detected_image_inliers):13d} │ {n_images:7d} │")
        print("    └─────────────┴───────────────┴───────────────┴─────────┘\n")

        # Report particle-level statistics (for tilt series)
        if is_tilt_series and detected_particle_outliers is not None and detected_particle_inliers is not None:
            print(f"  Particle-level statistics:")
            n_detected_particles = len(detected_particle_outliers)
            n_detected_particle_inliers = len(detected_particle_inliers)
            n_correct_particles = len(np.intersect1d(detected_particle_outliers, particle_outlier_particles))
            n_false_positives_particles = n_detected_particles - n_correct_particles
            n_false_negatives_particles = len(particle_outlier_particles) - n_correct_particles
            n_true_negatives_particles = n_particles - (n_correct_particles + n_false_positives_particles + n_false_negatives_particles)

            # 2x2 confusion matrix for particles (pretty print)
            print("    Confusion matrix (particles):")
            print("    ┌─────────────┬───────────────┬───────────────┬─────────┐")
            print("    │             │ Pred Outlier  │ Pred Inlier   │  Total  │")
            print("    ├─────────────┼───────────────┼───────────────┼─────────┤")
            print(f"    │ GT Outlier  │ {n_correct_particles:13d} │ {n_false_negatives_particles:13d} │ {len(particle_outlier_particles):7d} │")
            print(f"    │ GT Inlier   │ {n_false_positives_particles:13d} │ {n_true_negatives_particles:13d} │ {n_particles - len(particle_outlier_particles):7d} │")
            print("    ├─────────────┼───────────────┼───────────────┼─────────┤")
            print(f"    │   Total     │ {n_detected_particles:13d} │ {n_detected_particle_inliers:13d} │ {n_particles:7d} │")
            print("    └─────────────┴───────────────┴───────────────┴─────────┘\n")

            # # Analyze by outlier type
            # print(f"  Outlier type analysis:")
            
            # # Particle outliers (individual images with wrong structure)
            # if len(particle_outlier_indices) > 0:
            #     particle_outlier_correct = len(np.intersect1d(detected_image_outliers, particle_outlier_indices))
            #     particle_outlier_recall = particle_outlier_correct / len(particle_outlier_indices)
            #     print(f"    Particle outlier recall: {particle_outlier_recall:.3f} ({particle_outlier_correct}/{len(particle_outlier_indices)})")
            
            # # Tilt outliers (entire particles with wrong structure)
            # if len(tilt_outlier_indices) > 0:
            #     tilt_outlier_correct = len(np.intersect1d(detected_image_outliers, tilt_outlier_indices))
            #     tilt_outlier_recall = tilt_outlier_correct / len(tilt_outlier_indices)
            #     print(f"    Tilt outlier recall: {tilt_outlier_recall:.3f} ({tilt_outlier_correct}/{len(tilt_outlier_indices)})")
                
            #     # Also check particle-level detection of tilt outliers
            #     if len(particle_outlier_particles) > 0:
            #         tilt_particle_correct = len(np.intersect1d(detected_particle_outliers, particle_outlier_particles))
            #         tilt_particle_recall = tilt_particle_correct / len(particle_outlier_particles)
            #         print(f"    Tilt outlier particle recall: {tilt_particle_recall:.3f} ({tilt_particle_correct}/{len(particle_outlier_particles)})")
        elif is_tilt_series:
            print(f"  Particle-level statistics: Not available (particle outliers file not found)")

    # from recovar.tilt_dataset import tilt_series_indices_to_image_indices
    # from recovar import tilt_dataset
    # starfile = f"{test_dir}/test_dataset/particles.star"
    # particle_to_tilts, tilts_to_particle = tilt_dataset.TiltSeriesData.parse_particle_tilt(starfile)
    # bad_images = tilt_series_indices_to_image_indices(particle_outlier_particles, starfile)
    # bad_tilts = [tilts_to_particle[i] for i in all_outlier_indices]
    # print(np.unique(bad_tilts))
    return True

def verify_temp_cleanup(test_dir, is_tilt_series):
    """Verify that temp directories were cleaned up."""
    print("\nVerifying temp directories cleanup...")
    
    # Check both possible output dirs
    base_output_dirs = [f"{test_dir}/test_dataset/pipeline_outliers_output"]
    if is_tilt_series:
        base_output_dirs.append(f"{test_dir}/test_dataset/pipeline_outliers_copy_test_tilt")
        base_output_dirs.append(f"{test_dir}/test_dataset/pipeline_outliers_copy_test_tilt_no_cleanup")
    else:
        base_output_dirs.append(f"{test_dir}/test_dataset/pipeline_outliers_copy_test")
        base_output_dirs.append(f"{test_dir}/test_dataset/pipeline_outliers_copy_test_no_cleanup")
    found = False
    for base_output_dir in base_output_dirs:
        if os.path.exists(base_output_dir):
            print(f"Found output directory: {base_output_dir}")
            found = True
            break
    if not found:
        print(f"ERROR: None of the main output directories found: {base_output_dirs}")
        return False
    
    # Check for any remaining temp directories that might not have been cleaned up
    # Look for directories in /tmp that might be from our pipeline
    temp_dirs = glob.glob("/tmp/recovar_*")
    if temp_dirs:
        print(f"Found {len(temp_dirs)} potential temp directories:")
        for temp_dir in temp_dirs:
            print(f"  {temp_dir}")
        
        # Check if any of these are from our no-cleanup test
        no_cleanup_dirs = [d for d in temp_dirs if "no_cleanup" in d or "test" in d]
        if no_cleanup_dirs:
            print("  These appear to be from no-cleanup tests (expected to remain)")
        else:
            print("  WARNING: These temp directories should have been cleaned up")
    else:
        print("No remaining temp directories found - cleanup appears successful")
    
    print("Temp directories cleanup verification completed successfully!")
    return True


if __name__ == "__main__":
    sys.exit(main()) 