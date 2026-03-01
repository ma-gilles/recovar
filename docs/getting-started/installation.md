# Installation

RECOVAR requires CUDA and [JAX](https://jax.readthedocs.io/en/latest/index.html). JAX is installed automatically, but you need CUDA drivers on your system (typically already present on GPU clusters).

Installation takes less than 5 minutes.

## Quick install (pip)

```bash
conda create --name recovar python=3.11 -y
conda activate recovar
pip install git+https://github.com/scikit-fmm/scikit-fmm.git "jax[cuda12]"==0.9.0.1 recovar
```

Verify:

```bash
recovar run_test_dataset
```

## Development install

For the latest version or contributing:

```bash
# Clone
git clone https://github.com/ma-gilles/recovar.git
cd recovar

# Create environment
conda create --name recovar_dev python=3.11 -y
conda activate recovar_dev

# Isolate installs to this env
export PYTHONNOUSERSITE=1
unset PIP_USER

# Install dependencies
python -m pip install -U pip setuptools wheel
python -m pip install git+https://github.com/scikit-fmm/scikit-fmm.git
python -m pip install "jax[cuda12]"==0.9.0.1    # or "jax[cpu]" for CPU-only
python -m pip install nvtx

# Install recovar in editable mode
python -m pip install -e ".[dev]"

# Verify
python -m pip check
python -c "import jax, matplotlib, nvtx; print('env ok')"
```

## CPU-only install

For testing without a GPU:

```bash
conda create --name recovar python=3.11 -y
conda activate recovar
pip install git+https://github.com/scikit-fmm/scikit-fmm.git "jax[cpu]"==0.9.0.1 recovar
```

!!! warning
    CPU-only mode is useful for testing but not practical for real datasets. The pipeline requires GPU acceleration for reasonable performance.

## Jupyter kernel

To use RECOVAR from Jupyter notebooks:

```bash
conda activate recovar
python -m ipykernel install --user --name=recovar
```

## Pixi (alternative)

If you use [pixi](https://prefix.dev/):

```bash
git clone https://github.com/ma-gilles/recovar.git
cd recovar
pixi install
pixi run test
```

## Docker & HPC containers

For reproducible GPU environments, especially on HPC clusters, RECOVAR provides
Docker and Apptainer/Singularity container definitions. See the
[Docker & Containers guide](docker.md) for building images, running containers,
and submitting jobs with Slurm.
