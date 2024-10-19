import subprocess
import shutil
import os
import jax

RECOVAR_PATH = './'

passed_functions = []
failed_functions = []

# Check if JAX can find a GPU device
def check_gpu():
    try:
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            print("GPU devices found:", gpu_devices)
        else:
            print("--------------------------------------------")
            print("--------------------------------------------")
            print("No GPU devices found. Please ensure that JAX is properly configured with CUDA and a compatible GPU. Typically, the problem was during the installation of JAX which could not find the CUDA libraries. This can be fixed by setting the correct paths to the CUDA libraries in the environment variables, or module load depending on your system. Note that you may have to reinstall JAX after setting the correct paths.  E.g. run the following:\n pip uninstall jax jaxlib; \n pip install -U \"jax[cuda12_pip]\"==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
            print("--------------------------------------------")
            print("--------------------------------------------")

            exit(1)
    
    except Exception as e:
        print("Error occurred while checking for GPU devices:", e)
        print("--------------------------------------------")
        print("--------------------------------------------")
        print("No GPU devices found. Please ensure that JAX is properly configured with CUDA and a compatible GPU. Typically, the problem was during the installation of JAX which could not find the CUDA libraries. This can be fixed by setting the correct paths to the CUDA libraries in the environment variables, or module load depending on your system. Note that you may have to reinstall JAX after setting the correct paths.  E.g. run the following:\n pip uninstall jax jaxlib; \n pip install -U \"jax[cuda12_pip]\"==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        print("--------------------------------------------")
        print("--------------------------------------------")

        exit(1)


# Check for GPU availability
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

# Generate a small test dataset - should take about 30 sec
run_command(
    f'python {RECOVAR_PATH}/make_test_dataset.py',
    'Generate a small test dataset',
    'make_test_dataset.py'
)

# Run pipeline, should take about 2 min
run_command(
    f'python {RECOVAR_PATH}/pipeline.py test_dataset/particles.64.mrcs --poses test_dataset/poses.pkl --ctf test_dataset/ctf.pkl --correct-contrast -o test_dataset/pipeline_output --mask=from_halfmaps --lazy',
    'Run pipeline',
    'pipeline.py'
)

# Run analyze.py with 2D embedding and no regularization on latent space (better for density estimation)
# Should take about 5 min
run_command(
    f'python {RECOVAR_PATH}/analyze.py test_dataset/pipeline_output --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0',
    'Run analyze.py',
    'analyze.py'
)

# Estimate conformational density
run_command(
    f'python {RECOVAR_PATH}/estimate_conformational_density.py test_dataset/pipeline_output --pca_dim 2',
    'Estimate conformational density',
    'estimate_conformational_density.py'
)

# Run analyze.py with 2D embedding and no regularization on latent space (better for density estimation) and trajectory estimation
# Should take about 5 min
run_command(
    f'python {RECOVAR_PATH}/analyze.py test_dataset/pipeline_output --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=1 --density test_dataset/pipeline_output/density/deconv_density_knee.pkl --skip-centers',
    'Run analyze.py with density',
    'analyze.py'
)


# Compute trajectory - option 1
run_command(
    f'python {RECOVAR_PATH}/compute_trajectory.py test_dataset/pipeline_output -o test_dataset/pipeline_output/trajectory1 --endpts test_dataset/pipeline_output/analysis_2_noreg/kmeans_center_coords.txt --density test_dataset/pipeline_output/density/deconv_density_knee.pkl --zdim=2 --n-vols-along-path=3',
    'Compute trajectory - option 1',
    'compute_trajectory.py (option 1)'
)

# Compute trajectory - option 2
run_command(
    f'python {RECOVAR_PATH}/compute_trajectory.py test_dataset/pipeline_output -o test_dataset/pipeline_output/trajectory2 --endpts test_dataset/pipeline_output/analysis_2_noreg/kmeans_result.pkl --density test_dataset/pipeline_output/density/deconv_density_knee.pkl --kmeans-ind=0,1 --zdim=2 --n-vols-along-path=0',
    'Compute trajectory - option 2',
    'compute_trajectory.py (option 2)'
)

print(f"\nTotal steps passed: {len(passed_functions)}")
print(f"Total steps failed: {len(failed_functions)}\n")

if failed_functions:
    print("The following functions failed:")
    for func in failed_functions:
        print(f"- {func}")
    print("\nPlease check the output above for details.")
else:
    print("All functions completed successfully!")

    # Delete the test_dataset directory since all steps passed
    if os.path.exists('test_dataset'):
        shutil.rmtree('test_dataset')
        print("Test dataset directory 'test_dataset' has been deleted.")

# One way to make sure everything went well is that the states in test_dataset/pipeline_output/output/analysis_2_noreg/kmeans_center_volumes/all_volumes
# should be similar to the simulated ones in recovar/data/vol*.mrc (the order doesn't matter, though).
