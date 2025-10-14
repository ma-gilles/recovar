#!/bin/bash
set -e  # Exit on any error

# Clone the code
cd ~/recovar_test

# Remove existing directory if it exists
if [ -d "recovar" ]; then
    echo "Removing existing recovar directory..."
    rm -rf recovar
fi

git clone https://github.com/ma-gilles/recovar.git

cd recovar
# Optionally: choose your branch
# git checkout <branch>

# Initialize conda for this shell session
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

# Create a fresh environment
echo "Creating conda environment..."
conda create --name recovar_dev_test python=3.11 -y

# Enable the environment
echo "Activating conda environment..."
conda activate recovar_dev_test

# Verify environment is active
if [[ "$CONDA_DEFAULT_ENV" != "recovar_dev_test" ]]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

# Install key dependencies that don't play well with pip
echo "Installing dependencies..."
pip install git+https://github.com/scikit-fmm/scikit-fmm.git
pip install -f https://download.pytorch.org/whl/torch_stable.html torch==2.3.1+cpu
pip install "jax[cuda12]"==0.5.0    # or use "jax[cpu]"==0.5.0 if no NVIDIA GPU

# Install recovar from the checked-out code in editable mode
echo "Installing recovar..."
pip install -e ".[dev]"
python -m ipykernel install --user --name=recovar_dev_test

# Create test dataset directory

DATASET_DIR="~/mytigress/test_dataset/"
# Generate test dataset - Should take about 30 sec
echo "Generating test dataset..."
recovar make_test_dataset $DATASET_DIR --image-size=128 --n-images=100000

cd $DATASET_DIR/test_dataset/

echo "Running test..."
recovar run_test_dataset --o ~/mytigress/test_dataset/ 

# Run pipeline - Should take about 30 min - This the step we want to profile first!
echo "Running pipeline..."
recovar pipeline particles.128.mrcs --ctf ctf.pkl --poses poses.pkl --mask=from_halfmaps -o pipeline_output


# # Generate a more realistic dataset - Should take about 3 min
# echo "Generating test dataset..."
# recovar make_test_dataset /tmp/ --image-size=256 --n-images=300000

# cd /tmp/test_dataset/

# # Run pipeline - Should take about 6 hours
# echo "Running pipeline..."
# recovar pipeline particles.256.mrcs --ctf ctf.pkl --poses poses.pkl --mask=from_halfmaps -o pipeline_output




echo "Script completed successfully!"