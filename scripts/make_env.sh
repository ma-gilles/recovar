#!/bin/bash
# Create a conda development environment for RECOVAR.
# Usage: source scripts/make_env.sh

ENVIRONMENT=recovar_dev

# Initialize conda
eval "$(conda shell.bash hook)"

# Create a fresh environment
conda create --name $ENVIRONMENT python=3.11 -y

# Enable the environment
conda activate $ENVIRONMENT

# Keep installs isolated to this env (avoid ~/.local package leakage)
export PYTHONNOUSERSITE=1
unset PIP_USER

# Upgrade packaging tooling in-env
python -m pip install -U pip setuptools wheel

# Install recovar plus pinned CUDA-enabled JAX wheels. This does not install
# nvcc; make sure a local CUDA toolkit is available if you want the custom
# RECOVAR CUDA extension.
python -m pip install -e ".[cuda,dev]"

# Verify environment consistency
python -m pip check
python -c "import jax, matplotlib; print('env ok')"

python -m ipykernel install --user --name=$ENVIRONMENT
