#!/bin/bash

# Script to submit RECOVAR jobs to a cluster scheduler (Slurm by default) or run locally.
# Usage: ./crun_recovar_workload.sh <action>

set -e

# Configuration (paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPTS_DIR="$SCRIPT_DIR/scripts/job_scripts"
OUTPUT_DIR="$SCRIPT_DIR/scripts/output"

# Configuration (scheduler + container)
# - Scheduler: slurm|local|auto (auto picks slurm if sbatch exists, else local)
# - Container tool: docker|apptainer|auto (auto prefers apptainer if present, else docker)
SCHEDULER="${SCHEDULER:-auto}"
CONTAINER_TOOL="${CONTAINER_TOOL:-auto}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-recovar:latest}"

# Nsight Systems override (to control report version compatibility with your nsys-ui).
# If set, this should be a full path to an `nsys` binary on the host (e.g.
# /opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys).
NSYS_BIN="${NSYS_BIN:-}"

# Optional Slurm settings (env overrides)
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-}"   # e.g. "a100|h100"
SLURM_NODES="${SLURM_NODES:-1}"
SLURM_NTASKS="${SLURM_NTASKS:-1}"
# GPU request style:
# - gpus: uses "#SBATCH --gpus=N" (newer Slurm)
# - gres: uses "#SBATCH --gres=gpu:N" (common on many clusters)
SLURM_GPU_REQUEST_STYLE="${SLURM_GPU_REQUEST_STYLE:-gpus}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-}"
SLURM_MEM="${SLURM_MEM:-}"

# Optional runtime settings (env overrides)
WORKDIR_IN_CONTAINER="${WORKDIR_IN_CONTAINER:-/workspace}"
SKIP_IMAGE_BUILD="${SKIP_IMAGE_BUILD:-0}"
SKIP_PIXI_ENV_INSTALL="${SKIP_PIXI_ENV_INSTALL:-0}"
SKIP_RECOVAR_INSTALL="${SKIP_RECOVAR_INSTALL:-0}"

# Pixi storage overrides (strongly recommended on HPC to avoid repo/home quotas)
# These are passed into the container and (if /scratch exists) we bind-mount /scratch.
PIXI_HOME="${PIXI_HOME:-}"
RATTLER_CACHE_DIR="${RATTLER_CACHE_DIR:-}"
PIXI_DETACHED_ENVIRONMENTS="${PIXI_DETACHED_ENVIRONMENTS:-1}"

# Create directories if they don't exist
mkdir -p "$JOB_SCRIPTS_DIR"
mkdir -p "$OUTPUT_DIR"

# ---------- helpers ----------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

detect_scheduler() {
    if [ "$SCHEDULER" != "auto" ]; then
        echo "$SCHEDULER"
        return 0
    fi
    if have_cmd sbatch; then
        echo "slurm"
        return 0
    fi
    echo "local"
}

escape_sed_replacement() {
    # Escape '&', '\', and delimiter '|' for sed replacement.
    echo "$1" | sed -e 's/[&|\\]/\\&/g'
}

slurm_directives() {
    local num_gpus="$1"
    local time_limit="$2"
    local job_name="$3"

    # Slurm job names cannot contain whitespace; sanitize to a safe token.
    local safe_job_name="${job_name//[^A-Za-z0-9._-]/_}"
    echo "#SBATCH --job-name=${safe_job_name}"
    echo "#SBATCH --output=${OUTPUT_DIR}/slurm-%j.out"
    echo "#SBATCH --error=${OUTPUT_DIR}/slurm-%j.err"
    echo "#SBATCH --nodes=${SLURM_NODES}"
    echo "#SBATCH --ntasks=${SLURM_NTASKS}"
    echo "#SBATCH --time=${time_limit}"
    if [ "$SLURM_GPU_REQUEST_STYLE" = "gres" ]; then
        echo "#SBATCH --gres=gpu:${num_gpus}"
    else
        echo "#SBATCH --gpus=${num_gpus}"
    fi

    if [ -n "$SLURM_PARTITION" ]; then echo "#SBATCH --partition=${SLURM_PARTITION}"; fi
    if [ -n "$SLURM_ACCOUNT" ]; then echo "#SBATCH --account=${SLURM_ACCOUNT}"; fi
    if [ -n "$SLURM_QOS" ]; then echo "#SBATCH --qos=${SLURM_QOS}"; fi
    if [ -n "$SLURM_CONSTRAINT" ]; then echo "#SBATCH --constraint=${SLURM_CONSTRAINT}"; fi
    if [ -n "$SLURM_CPUS_PER_TASK" ]; then echo "#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}"; fi
    if [ -n "$SLURM_MEM" ]; then echo "#SBATCH --mem=${SLURM_MEM}"; fi
}

# Function to generate a batch script
generate_batch_script() {
    local task_name=$1
    local task_cmd=$2
    local num_gpus=$3
    local time_limit=$4
    local scheduler=$5
    local pid=$$
    local batch_script="$JOB_SCRIPTS_DIR/recovar_batch_${pid}.sh"
    
    {
        echo "#!/bin/bash"
        if [ "$scheduler" = "slurm" ]; then
            slurm_directives "$num_gpus" "$time_limit" "$task_name"
        fi

        cat <<'EOF'
set -e

echo "=========================================="
echo "RECOVAR Batch Job Starting"
echo "Node: $(hostname)"
echo "Date: $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPUs: $(nvidia-smi -L)"
fi
echo "=========================================="

# Configuration (can be overridden by env vars at submission time)
SCRIPT_DIR="${SCRIPT_DIR:-__SCRIPT_DIR__}"
OUTPUT_DIR="${OUTPUT_DIR:-__OUTPUT_DIR__}"
CONTAINER_TOOL="${CONTAINER_TOOL:-__CONTAINER_TOOL__}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-__CONTAINER_IMAGE__}"
WORKDIR_IN_CONTAINER="${WORKDIR_IN_CONTAINER:-__WORKDIR_IN_CONTAINER__}"
SKIP_IMAGE_BUILD="${SKIP_IMAGE_BUILD:-__SKIP_IMAGE_BUILD__}"
SKIP_PIXI_ENV_INSTALL="${SKIP_PIXI_ENV_INSTALL:-__SKIP_PIXI_ENV_INSTALL__}"
SKIP_RECOVAR_INSTALL="${SKIP_RECOVAR_INSTALL:-__SKIP_RECOVAR_INSTALL__}"
NSYS_BIN="${NSYS_BIN:-__NSYS_BIN__}"
PIXI_HOME="${PIXI_HOME:-__PIXI_HOME__}"
RATTLER_CACHE_DIR="${RATTLER_CACHE_DIR:-__RATTLER_CACHE_DIR__}"
PIXI_DETACHED_ENVIRONMENTS="${PIXI_DETACHED_ENVIRONMENTS:-__PIXI_DETACHED_ENVIRONMENTS__}"
TASK_CMD="TASK_CMD_PLACEHOLDER"

cd "$SCRIPT_DIR"

echo "Task command: $TASK_CMD"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Container tool: $CONTAINER_TOOL"
echo "Container image: $CONTAINER_IMAGE"
if [ -n "${NSYS_BIN:-}" ]; then
  echo "NSYS_BIN override: $NSYS_BIN"
fi

run_in_container() {
  local task_cmd="$1"

  local tool="$CONTAINER_TOOL"
  if [ "$tool" = "auto" ]; then
    if command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1; then
      tool="apptainer"
    elif command -v docker >/dev/null 2>&1; then
      tool="docker"
    else
      tool="none"
    fi
  fi

  if [ "$tool" = "none" ]; then
    echo "Error: No container runtime found (docker/apptainer/singularity)." >&2
    exit 1
  fi

  if [ "$tool" = "docker" ]; then
    if [ "$SKIP_IMAGE_BUILD" != "1" ]; then
      if ! docker image inspect "$CONTAINER_IMAGE" >/dev/null 2>&1; then
        echo "Container image not found ($CONTAINER_IMAGE). Building..."
        bash "$SCRIPT_DIR/scripts/build_container.sh"
      else
        echo "Container image found: $CONTAINER_IMAGE"
      fi
    else
      echo "Skipping container image build check (SKIP_IMAGE_BUILD=1)."
    fi

    local gpu_args=()
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      gpu_args=(--gpus "device=${CUDA_VISIBLE_DEVICES}")
    else
      gpu_args=(--gpus all)
    fi

    local scratch_mount=()
    if [ -d /scratch ]; then
      scratch_mount=(-v /scratch:/scratch)
    fi

    docker run --rm --net host --ipc=host \
      --runtime=nvidia "${gpu_args[@]}" \
      -v "$SCRIPT_DIR":"$WORKDIR_IN_CONTAINER" \
      "${scratch_mount[@]}" \
      -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
      -w "$WORKDIR_IN_CONTAINER" \
      --user "$(id -u):$(id -g)" \
      "$CONTAINER_IMAGE" \
      bash -lc "
        set -e
        unset PYTHONPATH PYTHONHOME
        export PYTHONNOUSERSITE=1
        # If requested, force a specific host-provided Nsight Systems version (controls .nsys-rep format).
        if [ -n \"\$NSYS_BIN\" ]; then
          if [ -x \"\$NSYS_BIN\" ]; then
            export PATH=\"\$(dirname \"\$NSYS_BIN\"):\$PATH\"
          else
            echo \"Warning: NSYS_BIN is set but not executable: \$NSYS_BIN\" >&2
          fi
        fi
        if command -v nsys >/dev/null 2>&1; then nsys --version || true; fi
        # Place pixi envs/caches on /scratch if available (avoids quota issues).
        if [ -z \"\$PIXI_HOME\" ]; then
          if [ -d /scratch ]; then
            USER_NAME=\"\${USER:-\$(id -un)}\"
            export PIXI_HOME=\"/scratch/\$USER_NAME/pixi\"
          fi
        fi
        if [ -n \"\$PIXI_HOME\" ]; then
          mkdir -p \"\$PIXI_HOME\"
          export PATH=\"\$PIXI_HOME/bin:\$PATH\"
        fi
        if [ -z \"\$RATTLER_CACHE_DIR\" ] && [ -n \"\$PIXI_HOME\" ]; then
          export RATTLER_CACHE_DIR=\"\$PIXI_HOME/rattler-cache\"
        fi
        if [ -n \"\$RATTLER_CACHE_DIR\" ]; then
          mkdir -p \"\$RATTLER_CACHE_DIR\"
        fi

        if ! command -v pixi >/dev/null 2>&1; then
          echo 'pixi not found in image; installing to \$HOME/.pixi...'
          PIXI_HOME=\"\${PIXI_HOME:-\$HOME/.pixi}\"
          if command -v curl >/dev/null 2>&1; then
            curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=\"\$PIXI_HOME\" PIXI_NO_PATH_UPDATE=1 bash
          elif command -v wget >/dev/null 2>&1; then
            wget -qO- https://pixi.sh/install.sh | PIXI_HOME=\"\$PIXI_HOME\" PIXI_NO_PATH_UPDATE=1 bash
          else
            echo 'Error: need curl or wget in the container to install pixi.' >&2
            exit 1
          fi
          export PATH=\"\$PIXI_HOME/bin:\$PATH\"
        fi

        # Ensure workspace envs are detached from the repo (prevents writing to /workspace/.pixi).
        if [ \"\$PIXI_DETACHED_ENVIRONMENTS\" = \"1\" ]; then
          pixi config set --global detached-environments true >/dev/null 2>&1 || true
        fi

        if [ \"$SKIP_PIXI_ENV_INSTALL\" != \"1\" ]; then
          echo 'Installing dependencies (pixi install)...'
          pixi install
        else
          echo 'Skipping pixi environment install (SKIP_PIXI_ENV_INSTALL=1).'
        fi

        if [ \"$SKIP_RECOVAR_INSTALL\" != \"1\" ]; then
          echo 'Installing RECOVAR...'
          pixi run install-recovar
        else
          echo 'Skipping RECOVAR install (SKIP_RECOVAR_INSTALL=1).'
        fi

        export PYTHONNOUSERSITE=1
        echo 'Running task: $task_cmd'
        $task_cmd
      "
    return 0
  fi

  # apptainer/singularity
  local apptainer_bin="apptainer"
  if ! command -v apptainer >/dev/null 2>&1 && command -v singularity >/dev/null 2>&1; then
    apptainer_bin="singularity"
  fi

  # For apptainer, CONTAINER_IMAGE should typically be a .sif path or URI like docker://...
  if [ ! -f "$CONTAINER_IMAGE" ] && [[ "$CONTAINER_IMAGE" != *"://"* ]]; then
    echo "Error: CONTAINER_IMAGE='$CONTAINER_IMAGE' does not exist as a file." >&2
    echo "For Apptainer/Singularity, set CONTAINER_IMAGE to a .sif path or a URI like 'docker://nvidia/cuda:12.6.0-devel-ubuntu22.04'." >&2
    exit 1
  fi

  local scratch_bind=()
  if [ -d /scratch ]; then
    scratch_bind=(-B /scratch:/scratch)
  fi

  "$apptainer_bin" exec --nv \
    -B "$SCRIPT_DIR":"$WORKDIR_IN_CONTAINER" \
    "${scratch_bind[@]}" \
    -B /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems \
    --pwd "$WORKDIR_IN_CONTAINER" \
    "$CONTAINER_IMAGE" \
    bash -lc "
      set -e
      unset PYTHONPATH PYTHONHOME
      export PYTHONNOUSERSITE=1
      # If requested, force a specific host-provided Nsight Systems version (controls .nsys-rep format).
      if [ -n \"\$NSYS_BIN\" ]; then
        if [ -x \"\$NSYS_BIN\" ]; then
          export PATH=\"\$(dirname \"\$NSYS_BIN\"):\$PATH\"
        else
          echo \"Warning: NSYS_BIN is set but not executable: \$NSYS_BIN\" >&2
        fi
      fi
      if command -v nsys >/dev/null 2>&1; then nsys --version || true; fi
      # Place pixi envs/caches on /scratch if available (avoids quota issues).
      if [ -z \"\$PIXI_HOME\" ]; then
        if [ -d /scratch ]; then
          USER_NAME=\"\${USER:-\$(id -un)}\"
          export PIXI_HOME=\"/scratch/\$USER_NAME/pixi\"
        fi
      fi
      if [ -n \"\$PIXI_HOME\" ]; then
        mkdir -p \"\$PIXI_HOME\"
        export PATH=\"\$PIXI_HOME/bin:\$PATH\"
      fi
      if [ -z \"\$RATTLER_CACHE_DIR\" ] && [ -n \"\$PIXI_HOME\" ]; then
        export RATTLER_CACHE_DIR=\"\$PIXI_HOME/rattler-cache\"
      fi
      if [ -n \"\$RATTLER_CACHE_DIR\" ]; then
        mkdir -p \"\$RATTLER_CACHE_DIR\"
      fi

      if ! command -v pixi >/dev/null 2>&1; then
        echo 'pixi not found in image; installing to \$HOME/.pixi...'
        PIXI_HOME=\"\${PIXI_HOME:-\$HOME/.pixi}\"
        if command -v curl >/dev/null 2>&1; then
          curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=\"\$PIXI_HOME\" PIXI_NO_PATH_UPDATE=1 bash
        elif command -v wget >/dev/null 2>&1; then
          wget -qO- https://pixi.sh/install.sh | PIXI_HOME=\"\$PIXI_HOME\" PIXI_NO_PATH_UPDATE=1 bash
        else
          echo 'Error: need curl or wget in the container to install pixi.' >&2
          exit 1
        fi
        export PATH=\"\$PIXI_HOME/bin:\$PATH\"
      fi

      # Ensure workspace envs are detached from the repo (prevents writing to /workspace/.pixi).
      if [ \"\$PIXI_DETACHED_ENVIRONMENTS\" = \"1\" ]; then
        pixi config set --global detached-environments true >/dev/null 2>&1 || true
      fi

      if [ \"$SKIP_PIXI_ENV_INSTALL\" != \"1\" ]; then
        echo 'Installing dependencies (pixi install)...'
        pixi install
      else
        echo 'Skipping pixi environment install (SKIP_PIXI_ENV_INSTALL=1).'
      fi

      if [ \"$SKIP_RECOVAR_INSTALL\" != \"1\" ]; then
        echo 'Installing RECOVAR...'
        pixi run install-recovar
      else
        echo 'Skipping RECOVAR install (SKIP_RECOVAR_INSTALL=1).'
      fi

      echo 'Running task: $task_cmd'
      $task_cmd
    "
}

run_in_container "$TASK_CMD"

echo "=========================================="
echo "RECOVAR Batch Job Completed"
echo "Date: $(date)"
echo "=========================================="
EOF
    } > "$batch_script"

    # Replace the task command placeholder
    local escaped_task_cmd
    escaped_task_cmd="$(escape_sed_replacement "$task_cmd")"
    sed -i "s|TASK_CMD_PLACEHOLDER|$escaped_task_cmd|g" "$batch_script"

    local escaped_script_dir escaped_output_dir escaped_container_tool escaped_container_image escaped_workdir escaped_skip
    escaped_script_dir="$(escape_sed_replacement "$SCRIPT_DIR")"
    escaped_output_dir="$(escape_sed_replacement "$OUTPUT_DIR")"
    escaped_container_tool="$(escape_sed_replacement "$CONTAINER_TOOL")"
    escaped_container_image="$(escape_sed_replacement "$CONTAINER_IMAGE")"
    escaped_workdir="$(escape_sed_replacement "$WORKDIR_IN_CONTAINER")"
    escaped_skip="$(escape_sed_replacement "$SKIP_IMAGE_BUILD")"
    local escaped_skip_pixi_env escaped_skip_recovar
    escaped_skip_pixi_env="$(escape_sed_replacement "$SKIP_PIXI_ENV_INSTALL")"
    escaped_skip_recovar="$(escape_sed_replacement "$SKIP_RECOVAR_INSTALL")"
    local escaped_pixi_home escaped_rattler_cache escaped_detached_envs
    escaped_pixi_home="$(escape_sed_replacement "$PIXI_HOME")"
    escaped_rattler_cache="$(escape_sed_replacement "$RATTLER_CACHE_DIR")"
    escaped_detached_envs="$(escape_sed_replacement "$PIXI_DETACHED_ENVIRONMENTS")"
    local escaped_nsys_bin
    escaped_nsys_bin="$(escape_sed_replacement "$NSYS_BIN")"

    sed -i "s|__SCRIPT_DIR__|$escaped_script_dir|g" "$batch_script"
    sed -i "s|__OUTPUT_DIR__|$escaped_output_dir|g" "$batch_script"
    sed -i "s|__CONTAINER_TOOL__|$escaped_container_tool|g" "$batch_script"
    sed -i "s|__CONTAINER_IMAGE__|$escaped_container_image|g" "$batch_script"
    sed -i "s|__WORKDIR_IN_CONTAINER__|$escaped_workdir|g" "$batch_script"
    sed -i "s|__SKIP_IMAGE_BUILD__|$escaped_skip|g" "$batch_script"
    sed -i "s|__SKIP_PIXI_ENV_INSTALL__|$escaped_skip_pixi_env|g" "$batch_script"
    sed -i "s|__SKIP_RECOVAR_INSTALL__|$escaped_skip_recovar|g" "$batch_script"
    sed -i "s|__NSYS_BIN__|$escaped_nsys_bin|g" "$batch_script"
    sed -i "s|__PIXI_HOME__|$escaped_pixi_home|g" "$batch_script"
    sed -i "s|__RATTLER_CACHE_DIR__|$escaped_rattler_cache|g" "$batch_script"
    sed -i "s|__PIXI_DETACHED_ENVIRONMENTS__|$escaped_detached_envs|g" "$batch_script"
    
    # Make the script executable
    chmod +x "$batch_script"
    
    echo "$batch_script"
}

# Function to submit a job
submit_job() {
    local task_name=$1
    local task_cmd=$2
    local num_gpus=$3
    local time_limit=$4

    local scheduler
    scheduler="$(detect_scheduler)"
    
    echo "========================================"
    echo "Submitting job: $task_name"
    echo "Task command: $task_cmd"
    echo "Number of GPUs: $num_gpus"
    echo "Time limit: $time_limit"
    echo "Scheduler: $scheduler"
    echo "========================================"

    if [ "$scheduler" = "local" ]; then
        echo "Running locally (no scheduler detected)."
        local batch_script
        batch_script=$(generate_batch_script "$task_name" "$task_cmd" "$num_gpus" "$time_limit" "local")
        bash "$batch_script"
        echo ""
        return 0
    fi

    if [ "$scheduler" != "slurm" ]; then
        echo "Error: Unsupported scheduler '$scheduler'. Supported: slurm, local" >&2
        echo "Tip: set SCHEDULER=local to run without a scheduler." >&2
        exit 1
    fi

    # Generate the batch script
    local batch_script
    batch_script=$(generate_batch_script "$task_name" "$task_cmd" "$num_gpus" "$time_limit" "slurm")
    echo "Generated batch script: $batch_script"

    sbatch "$batch_script"
    echo "Submitted via sbatch."
    echo ""
}

# Main action handler
ACTION=$1

if [ -z "$ACTION" ]; then
    echo "Usage: $0 <action>"
    echo ""
    echo "Environment overrides:"
    echo "  SCHEDULER=slurm|local|auto            (default: auto)"
    echo "  CONTAINER_TOOL=docker|apptainer|auto  (default: auto)"
    echo "  CONTAINER_IMAGE=<image>               (docker tag or apptainer .sif/URI; default: recovar:latest)"
    echo "  WORKDIR_IN_CONTAINER=/workspace       (default: /workspace)"
    echo "  NSYS_BIN=/path/to/nsys                (force a specific Nsight Systems CLI; controls .nsys-rep version)"
    echo "  SKIP_IMAGE_BUILD=1                    (skip docker image check/build)"
    echo "  SKIP_PIXI_ENV_INSTALL=1               (skip 'pixi install' inside the container)"
    echo "  SKIP_RECOVAR_INSTALL=1                (skip 'pixi run install-recovar' inside the container)"
    echo ""
    echo "Slurm env overrides (optional):"
    echo "  SLURM_PARTITION=...  SLURM_ACCOUNT=...  SLURM_QOS=...  SLURM_CONSTRAINT=...  SLURM_NODES=...  SLURM_NTASKS=...  SLURM_GPU_REQUEST_STYLE=gpus|gres  SLURM_CPUS_PER_TASK=...  SLURM_MEM=..."
    echo ""
    echo "Available actions:"
    echo "  Smoke tests:"
    echo "    smoke-container  - Validate container runtime + pixi presence quickly (5m)"
    echo "    smoke-recovar    - Validate pixi env + recovar install + CLI import (15m)"
    echo ""
    echo "  Test runs (128-100k dataset):"
    echo "    test-1gpu       - Test with 1 GPU (30m)"
    echo "    test-2gpu       - Test with 2 GPUs (30m)"
    echo "    test-4gpu       - Test with 4 GPUs (30m)"
    echo "    test-8gpu       - Test with 8 GPUs (30m)"
    echo ""
    echo "  Comparison:"
    echo "    compare-all     - Compare outputs from all multi-GPU runs (10m)"
    echo ""
    echo "  Profiling (128-100k dataset, 50k images):"
    echo "    profile-1gpu    - Profile 1 GPU run (1h)"
    echo "    profile-2gpu    - Profile 2 GPU run (45m)"
    echo "    profile-4gpu    - Profile 4 GPU run (30m)"
    echo ""
    echo "  Profiling (256-300k dataset):"
    echo "    profile-1gpu-256 - Profile 1 GPU run (3h)"
    echo "    profile-2gpu-256 - Profile 2 GPU run (2h)"
    echo "    profile-4gpu-256 - Profile 4 GPU run (3h)"
    echo ""
    echo "  Dataset creation:"
    echo "    create-small    - Create 128-100k dataset (1h)"
    echo "    create-large    - Create 256-300k dataset (2h)"
    echo ""
    echo "  Pipeline runs:"
    echo "    pipeline-small  - Run pipeline on 128-100k dataset (1h)"
    echo "    pipeline-large  - Run pipeline on 256-300k dataset (2h)"
    echo ""
    exit 1
fi

case $ACTION in
    smoke-container)
        # Keep this fast: don't resolve the full pixi env and don't install recovar.
        SKIP_PIXI_ENV_INSTALL=1
        SKIP_RECOVAR_INSTALL=1
        submit_job "Smoke Container" "pixi --version" 1 "00:05:00"
        ;;
    smoke-recovar)
        # Full path smoke test: env solve + editable install + a cheap CLI command.
        submit_job "Smoke Recovar" "pixi run smoke-import-recovar" 1 "00:15:00"
        ;;
    test-recovar)
        # Real functional test (small): runs RECOVAR's bundled test dataset pipeline.
        submit_job "Test Recovar" "pixi run test-recovar" 1 "00:30:00"
        ;;
    # Test runs
    test-1gpu)
        submit_job "Test 1 GPU" "pixi run test-1gpu" 1 "00:30:00"
        ;;
    test-2gpu)
        submit_job "Test 2 GPUs" "pixi run test-2gpu" 2 "00:30:00"
        ;;
    test-4gpu)
        submit_job "Test 4 GPUs" "pixi run test-4gpu" 4 "00:30:00"
        ;;
    test-8gpu)
        submit_job "Test 8 GPUs" "pixi run test-8gpu" 8 "00:30:00"
        ;;
    
    # Comparison
    compare-all)
        submit_job "Compare Multi-GPU Outputs" "pixi run compare-all-multigpu" 1 "00:10:00"
        ;;
    
    # Profiling (128-100k dataset)
    profile-1gpu)
        submit_job "Profile 1 GPU (128-100k)" "pixi run --frozen --locked --no-install profile-1gpu" 1 "01:00:00"
        ;;
    profile-2gpu)
        submit_job "Profile 2 GPUs (128-100k)" "pixi run --frozen --locked --no-install profile-2gpu" 2 "00:45:00"
        ;;
    profile-4gpu)
        submit_job "Profile 4 GPUs (128-100k)" "pixi run --frozen --locked --no-install profile-4gpu" 4 "00:30:00"
        ;;
    
    # Profiling (256-300k dataset)
    profile-1gpu-256)
        submit_job "Profile 1 GPU (256-300k)" "pixi run --frozen --locked --no-install profile-1gpu-256" 1 "03:00:00"
        ;;
    profile-2gpu-256)
        submit_job "Profile 2 GPUs (256-300k)" "pixi run --frozen --locked --no-install profile-2gpu-256" 2 "02:00:00"
        ;;
    profile-4gpu-256)
        submit_job "Profile 4 GPUs (256-300k)" "pixi run --frozen --locked --no-install profile-4gpu-256" 4 "03:00:00"
        ;;
    
    # Dataset creation
    create-small)
        submit_job "Create Small Dataset" "pixi run create-dataset-small" 1 "01:00:00"
        ;;
    create-large)
        submit_job "Create Large Dataset" "pixi run create-dataset-large" 1 "02:00:00"
        ;;
    
    # Pipeline runs
    pipeline-small)
        submit_job "Pipeline Small" "pixi run pipeline-small" 1 "01:00:00"
        ;;
    pipeline-large)
        submit_job "Pipeline Large" "pixi run pipeline-large" 1 "02:00:00"
        ;;
    
    *)
        echo "Error: Unknown action '$ACTION'"
        echo "Run '$0' without arguments to see available actions."
        exit 1
        ;;
esac

