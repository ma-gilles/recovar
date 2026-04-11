# Installation

RECOVAR requires Python 3.11+ and [JAX](https://jax.readthedocs.io/en/latest/index.html). A CUDA GPU is required for practical use, and a CPU-only path is available for testing.

## Quick install

Install the published package with CUDA-enabled JAX wheels:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install "recovar[gpu]"
```

Verify:

```bash
python -c "import jax; print(jax.devices())"
recovar run_test_dataset
```

All dependencies are pinned to exact versions for reliability. You should not encounter version conflicts.

If you are installing from a local checkout instead of PyPI, use `pip install ".[gpu]"`.

`recovar[cuda]` remains available as a compatibility alias for `recovar[gpu]`.

## Pixi (fully reproducible)

[Pixi](https://prefix.dev/) gives you a hermetic environment with every dependency locked via `pixi.lock`. This is the most reproducible option:

```bash
git clone https://github.com/ma-gilles/recovar.git
cd recovar
pixi install
pixi run install-recovar
pixi run smoke-import-recovar
```

Run tests with pixi:

```bash
pixi run test-fast        # Unit tests (no GPU)
pixi run test-full        # Full suite (requires GPU)
```

## Flexible install (for developers)

If you need to reconcile recovar with other packages in your environment (e.g., you already have a JAX version installed), use the `flexible` extra which uses minimum version bounds instead of exact pins:

```bash
git clone https://github.com/ma-gilles/recovar.git
cd recovar

conda create --name recovar_dev python=3.11 -y
conda activate recovar_dev
pip install -e ".[flexible,gpu-flexible,dev]"

# Verify
python -c "import jax; print(jax.devices())"
recovar run_test_dataset
```

## Native extensions

RECOVAR has two compiled extensions with different install behavior:

- The fast-marching C++ extension is bundled in published Linux and macOS wheels for supported builds. Source and editable installs build it locally when a C++ compiler is available. If that build fails, installation still succeeds and RECOVAR uses the pure-Python fallback instead. No separate `scikit-fmm` install is required.
- The CUDA backproject/project extension is optional and disabled by default. Installing `recovar[gpu]` or `.[gpu]` provides the CUDA-enabled JAX wheels, but not the CUDA compiler. To use RECOVAR's custom CUDA kernels, make sure `nvcc` is available via `NVCC`, `CUDACXX`, `PATH`, `LOCAL_CUDA_PATH`, `CUDA_HOME`, or `CUDA_PATH`, then run `recovar build_custom_cuda` and set `RECOVAR_ENABLE_CUSTOM_CUDA=1`.

## CPU-only install

For testing without a GPU:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install recovar
recovar run_test_dataset --cpu
```

!!! warning
    CPU-only mode is useful for testing but not practical for real datasets. The pipeline requires GPU acceleration for reasonable performance.

## Jupyter kernel

To use RECOVAR from Jupyter notebooks:

```bash
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=recovar
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
