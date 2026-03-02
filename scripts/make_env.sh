#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT=recovar_dev_3

# Source conda's shell integration so `conda activate` works in scripts
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create a fresh environment
conda create --name $ENVIRONMENT python=3.11 -y

# Enable the environment
conda activate $ENVIRONMENT

# Keep installs isolated to this env (avoid ~/.local package leakage)
export PYTHONNOUSERSITE=1
unset PIP_USER

# Upgrade packaging tooling in-env
python -m pip install -U pip setuptools wheel

# Install key dependencies that don't play well with pip
python -m pip install git+https://github.com/scikit-fmm/scikit-fmm.git
python -m pip install "jax[cuda12]"==0.9.0.1    # or use "jax[cpu]"==0.9.0.1 if no NVIDIA GPU

# Required by current command/profiling paths
python -m pip install nvtx

# Install recovar from the checked-out code in editable mode
# Use the extras your project defines; if none, drop [dev].
python -m pip install -e ".[dev]"

# Verify environment consistency
python -m pip check
python -c "import jax, grain, ml_dtypes, matplotlib, nvtx; print('env ok')"

python -m ipykernel install --user --name=$ENVIRONMENT 
