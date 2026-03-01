# Docker & Containers

RECOVAR provides Docker and Apptainer/Singularity container definitions for
reproducible GPU-accelerated runs, especially on HPC clusters.

## Prerequisites

- NVIDIA GPU with CUDA 12.x drivers
- Docker (for local use) or Apptainer/Singularity (for HPC)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`nvidia-docker2` or `nvidia-container-toolkit`)

## Architecture

The container image provides the **runtime environment** (CUDA toolkit, pixi
package manager, Nsight Systems profiler) but does **not** include RECOVAR
source code or the pixi environment. These are bind-mounted and installed at
runtime, so the same image works with different code versions.

```
Container image (recovar:latest)
├── NVIDIA CUDA 12.6 dev tools (Ubuntu 22.04)
├── pixi package manager
└── NVIDIA Nsight Systems profiler

At runtime:
├── /workspace  ← bind-mounted from host (your repo checkout)
└── pixi install + pip install -e .  ← runs inside the container
```

## Building the Docker image

```bash
bash scripts/build_container.sh
```

This builds `recovar:latest` with your host user ID/group ID baked in (avoids
permission issues with bind-mounted volumes). Under the hood it runs:

```bash
docker build . -t recovar:latest --network host \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    --build-arg USERNAME=$(whoami)
```

## Running locally with Docker

### Basic run (GPU required)

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

### Interactive shell

```bash
docker run --rm -it --gpus all \
    -v $(pwd):/workspace -w /workspace \
    --user $(id -u):$(id -g) \
    recovar:latest
```

### Key flags explained

| Flag | Purpose |
|------|---------|
| `--gpus all` | Expose all GPUs (or `--gpus "device=0,1"` for specific ones) |
| `-v $(pwd):/workspace` | Bind-mount your code into the container |
| `--user $(id -u):$(id -g)` | Run as your host user (avoids root-owned files) |
| `--rm` | Remove the container when it exits |
| `--net host --ipc=host` | Share host network and IPC (needed for multi-GPU NCCL) |

## Building Apptainer/Singularity images for HPC

HPC compute nodes typically cannot pull Docker images from registries.
Convert to a `.sif` file on a node that has both Docker and Apptainer:

### Option 1: Helper script (recommended)

```bash
bash scripts/build_recovar_sif.sh [output_path.sif]
```

This builds the Docker image if needed, then converts it to `.sif`.

### Option 2: Manual conversion

```bash
# Build Docker image first
bash scripts/build_container.sh

# Convert to .sif
apptainer build recovar.sif docker-daemon://recovar:latest
```

### Option 3: Build directly from definition file (no Docker required)

```bash
apptainer build recovar.sif recovar.def
```

!!! note
    The Docker image and the Apptainer definition (`recovar.def`) both use
    NVIDIA CUDA 12.6 as the base and include pixi + Nsight Systems. They
    produce equivalent runtime environments.

## Running on HPC clusters with Slurm

The `scripts/crun_recovar_workload_della.sh` script handles container execution,
Slurm job submission, and environment setup automatically. It:

- Auto-detects the container runtime (Docker or Apptainer/Singularity)
- Auto-detects the scheduler (Slurm or local execution)
- Sets up scratch directories for pixi environments (avoids home quota issues)
- Handles Nsight Systems profiling integration

### Quick start

```bash
# Smoke test: validate container runtime + pixi
./scripts/crun_recovar_workload_della.sh smoke-container

# Full smoke test: pixi env + RECOVAR install + import check
./scripts/crun_recovar_workload_della.sh smoke-recovar

# Run RECOVAR's built-in test dataset
./scripts/crun_recovar_workload_della.sh test-recovar

# Run with 1 GPU
./scripts/crun_recovar_workload_della.sh test-1gpu
```

### Configuration via environment variables

All settings can be overridden:

```bash
# Use a specific .sif image
CONTAINER_IMAGE=/path/to/recovar.sif ./scripts/crun_recovar_workload_della.sh test-1gpu

# Force local execution (no Slurm)
SCHEDULER=local ./scripts/crun_recovar_workload_della.sh smoke-recovar

# Use a specific Slurm partition and account
SLURM_PARTITION=gpu SLURM_ACCOUNT=myaccount ./scripts/crun_recovar_workload_della.sh test-1gpu

# Force Docker instead of auto-detected Apptainer
CONTAINER_TOOL=docker ./scripts/crun_recovar_workload_della.sh test-1gpu
```

### Runtime environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER` | `auto` | `slurm`, `local`, or `auto` (auto-detects) |
| `CONTAINER_TOOL` | `auto` | `docker`, `apptainer`, or `auto` |
| `CONTAINER_IMAGE` | `recovar:latest` | Docker tag or `.sif` path |
| `CONTAINER_SIF` | (empty) | Explicit `.sif` path when image is a Docker tag |
| `RECOVAR_WORK_BASE` | auto-detected from `/scratch` | Base directory for job scripts, output, pixi envs |
| `WORKDIR_IN_CONTAINER` | `/workspace` | Mount point for source code |
| `DATA_BASE` | `$RECOVAR_WORK_BASE/data` | Path to dataset directory |
| `NSYS_BIN` | (empty) | Path to host Nsight Systems binary (overrides container version) |
| `SKIP_IMAGE_BUILD` | `0` | Set `1` to skip Docker image existence check |
| `SKIP_PIXI_ENV_INSTALL` | `0` | Set `1` to skip `pixi install` |
| `SKIP_RECOVAR_INSTALL` | `0` | Set `1` to skip `pixi run install-recovar` |

### Slurm configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SLURM_PARTITION` | (empty) | Partition name |
| `SLURM_ACCOUNT` | (empty) | Account/allocation |
| `SLURM_QOS` | (empty) | Quality of service |
| `SLURM_CONSTRAINT` | (empty) | Node constraint (e.g., `a100`) |
| `SLURM_DEPENDENCY` | (empty) | Job dependency (e.g., `afterok:123456`) |
| `SLURM_GPU_REQUEST_STYLE` | `gpus` | `gpus` (`--gpus=N`) or `gres` (`--gres=gpu:N`) |
| `SLURM_CPUS_PER_TASK` | (empty) | CPUs per task |
| `SLURM_MEM` | (empty) | Memory request (e.g., `500G`) |
| `SLURM_NODES` | `1` | Number of nodes |
| `SLURM_NTASKS` | `1` | Number of tasks |

### Available actions

Run `./scripts/crun_recovar_workload_della.sh` with no arguments to see all options.

| Action | GPUs | Time | Description |
|--------|------|------|-------------|
| `smoke-container` | 1 | 5m | Validate container + pixi presence |
| `smoke-recovar` | 1 | 15m | Full env + install + import test |
| `test-recovar` | 1 | 30m | Run RECOVAR's built-in test dataset |
| `test-1gpu` | 1 | 30m | Test with 1 GPU |
| `test-2gpu` | 2 | 30m | Test with 2 GPUs |
| `test-4gpu` | 4 | 30m | Test with 4 GPUs |
| `test-8gpu` | 8 | 30m | Test with 8 GPUs |
| `profile-1gpu` | 1 | 2h | Profile 128-100k dataset, 1 GPU |
| `profile-2gpu` | 2 | 45m | Profile 128-100k dataset, 2 GPUs |
| `profile-4gpu` | 4 | 30m | Profile 128-100k dataset, 4 GPUs |
| `profile-*gpu-256` | 1-4 | 3-12h | Profile 256-300k dataset |
| `create-small` | 1 | 1h | Create 128-100k synthetic dataset |
| `create-large` | 1 | 2h | Create 256-300k synthetic dataset |
| `pipeline-small` | 1 | 1h | Run pipeline on 128-100k dataset |
| `pipeline-large` | 1 | 2h | Run pipeline on 256-300k dataset |
| `pipeline-large-lazy` | 1 | 12h | Pipeline with lazy loading (saves memory) |
| `pipeline-large-nolazy` | 1 | 12h | Pipeline without lazy loading (needs ~800G RAM) |

## Common workflows

### First-time setup on an HPC cluster

1. Clone the repo on a login node:
   ```bash
   git clone https://github.com/ma-gilles/recovar.git
   cd recovar
   ```

2. Build the container image:
   ```bash
   # If Docker is available on the login node:
   bash scripts/build_container.sh
   bash scripts/build_recovar_sif.sh

   # Or build directly with Apptainer:
   apptainer build recovar.sif recovar.def
   ```

3. Set the container image path:
   ```bash
   export CONTAINER_IMAGE=$(pwd)/recovar.sif
   ```

4. Run a smoke test:
   ```bash
   ./scripts/crun_recovar_workload_della.sh smoke-recovar
   ```

### Profiling with Nsight Systems

```bash
# Profile with 4 GPUs on 128-100k dataset
./scripts/crun_recovar_workload_della.sh profile-4gpu

# Use a specific Nsight Systems version (controls .nsys-rep format)
NSYS_BIN=/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys \
    ./scripts/crun_recovar_workload_della.sh profile-4gpu
```

Profile reports (`.nsys-rep` files) are written to the working directory and
can be opened with Nsight Systems on your local machine.

### Skipping env install for repeated runs

After the first run installs the pixi environment and RECOVAR, you can skip
these steps to speed up subsequent runs:

```bash
SKIP_PIXI_ENV_INSTALL=1 SKIP_RECOVAR_INSTALL=1 \
    ./scripts/crun_recovar_workload_della.sh test-1gpu
```

## File reference

| File | Description |
|------|-------------|
| `Dockerfile` | Docker image definition (CUDA 12.6 + pixi + Nsight Systems) |
| `recovar.def` | Apptainer/Singularity definition (equivalent to Dockerfile) |
| `scripts/build_container.sh` | Build the Docker image with host user matching |
| `scripts/build_recovar_sif.sh` | Convert Docker image to Apptainer `.sif` |
| `scripts/crun_recovar_workload_della.sh` | HPC workload submission (Slurm + container) |
| `.dockerignore` | Excludes large files from Docker build context |

## Troubleshooting

**"Container image not found"**
:   Run `bash scripts/build_container.sh` to build the Docker image.

**"compute nodes cannot pull from Docker"**
:   Build a `.sif` first: `bash scripts/build_recovar_sif.sh`. Then set
    `CONTAINER_IMAGE=/path/to/recovar.sif`.

**Permission errors on bind-mounted files**
:   The Docker image embeds your user ID at build time. Rebuild with
    `bash scripts/build_container.sh` if your user ID changed. For Apptainer,
    this is handled automatically (Apptainer runs as the calling user).

**Home directory quota exceeded**
:   Set `RECOVAR_WORK_BASE` to a scratch filesystem. The workload script
    is auto-detected from your writable scratch directory (e.g.,
    `/scratch/gpfs/<group>/<user>/recovar_profiling`).

**pixi not found inside the container**
:   The workload script will auto-install pixi to scratch if it's missing from
    the image. Ensure the container has `curl` or `wget`.

**CUDA/GPU not detected**
:   Verify the NVIDIA Container Toolkit is installed: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`.
    For Apptainer, use the `--nv` flag (handled automatically by the workload script).
