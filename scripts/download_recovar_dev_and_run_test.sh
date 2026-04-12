#!/bin/bash
set -e  # Exit on any error

# Clone the code
mkdir -p ~/recovar_test
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

# Install recovar plus pinned CUDA-enabled JAX wheels. This does not install
# nvcc; make sure a local CUDA toolkit is available if you want the custom
# RECOVAR CUDA extension.
echo "Installing dependencies..."
pip install -e ".[cuda,dev]"

# Run built-in test
echo "Running test..."
recovar run_test_dataset

echo "Script completed successfully!"
