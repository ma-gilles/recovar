# Installation

RECOVAR requires Python 3.11+ and [JAX](https://jax.readthedocs.io/en/latest/index.html). A CUDA GPU is required for practical use (CPU-only mode is available for testing). Installation takes less than 5 minutes.

## Quick install (pip)

```bash
conda create --name recovar python=3.11 -y
conda activate recovar
pip install "recovar[cuda]"
```

Verify:

```bash
recovar run_test_dataset
```

## Reproducible install (pixi)

For an exact reproducible environment with pinned dependencies:

```bash
git clone https://github.com/ma-gilles/recovar.git
cd recovar && git checkout dev
pixi install
pixi run install-recovar
pixi run smoke-import-recovar
```

## Development install

For the latest version or contributing:

```bash
git clone https://github.com/ma-gilles/recovar.git
cd recovar && git checkout dev

conda create --name recovar_dev python=3.11 -y
conda activate recovar_dev
pip install -e ".[cuda,dev]"

# Verify
python -c "import jax; print(jax.devices())"
recovar run_test_dataset
```

Or install the dev branch directly without cloning:

```bash
pip install "recovar[cuda] @ git+https://github.com/ma-gilles/recovar.git@dev"
```

## CPU-only install

For testing without a GPU:

```bash
pip install recovar
```

!!! warning
    CPU-only mode is useful for testing but not practical for real datasets. The pipeline requires GPU acceleration for reasonable performance.

## Jupyter kernel

To use RECOVAR from Jupyter notebooks:

```bash
conda activate recovar
pip install ipykernel
python -m ipykernel install --user --name=recovar
```

## Pixi (fully reproducible)

[Pixi](https://prefix.dev/) gives you a hermetic environment with exact pinned dependencies via a lock file. This is the recommended setup for development:

```bash
git clone https://github.com/ma-gilles/recovar.git
cd recovar && git checkout dev
pixi install
pixi run install-recovar
pixi run smoke-import-recovar
```

Run tests with pixi:

```bash
pixi run test-fast        # Unit tests (no GPU)
pixi run test-full        # Full suite (requires GPU)
```

## Docker

Docker gives you a reproducible GPU environment without managing conda/pip
dependencies on your host. You need Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### 1. Build the image

```bash
git clone https://github.com/ma-gilles/recovar.git
cd recovar
bash scripts/build_container.sh
```

This creates `recovar:latest` with CUDA 12.6, pixi, and Nsight Systems.
Your host user ID is baked in so bind-mounted files have correct ownership.

### 2. Run RECOVAR

```bash
docker run --rm --gpus all \
    -v $(pwd):/workspace -w /workspace \
    --user $(id -u):$(id -g) \
    recovar:latest -c "
        pixi install
        pixi run install-recovar
        recovar run_test_dataset
    "
```

Or start an interactive shell:

```bash
docker run --rm -it --gpus all \
    -v $(pwd):/workspace -w /workspace \
    --user $(id -u):$(id -g) \
    recovar:latest
```

Then inside the container:

```bash
pixi install
pixi run install-recovar
recovar pipeline particles.star -o output --mask mask.mrc
```

### 3. HPC clusters (Apptainer/Singularity)

HPC compute nodes typically cannot pull Docker images. Convert to `.sif` first:

```bash
bash scripts/build_recovar_sif.sh
```

Then submit jobs with the workload script:

```bash
./scripts/crun_recovar_workload_della.sh smoke-recovar
```

See the [Docker & Containers guide](docker.md) for the full reference
on environment variables, Slurm configuration, available actions, and
troubleshooting.
