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

# Install key dependencies that don't play well with pip
python -m pip install git+https://github.com/scikit-fmm/scikit-fmm.git
python -m pip install "jax[cuda12]"==0.9.0.1    # or use "jax[cpu]"==0.9.0.1 if no NVIDIA GPU

# Install recovar from the checked-out code in editable mode
python -m pip install -e ".[dev]"

# Verify environment consistency
python -m pip check
python -c "import jax, matplotlib; print('env ok')"

python -m ipykernel install --user --name=$ENVIRONMENT
