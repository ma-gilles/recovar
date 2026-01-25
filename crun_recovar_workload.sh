#!/bin/bash

# Script to submit RECOVAR jobs to the cluster using crun
# Usage: ./crun_recovar_workload.sh <action>

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPTS_DIR="$SCRIPT_DIR/scripts/job_scripts"
OUTPUT_DIR="$SCRIPT_DIR/scripts/output"

# GPU query - ensure we only use A100 GPUs (40GB or 80GB) for consistency
GPU_QUERY="gpu.chip=ga100 and cpu.arch=x86_64"

# Create directories if they don't exist
mkdir -p "$JOB_SCRIPTS_DIR"
mkdir -p "$OUTPUT_DIR"

# Function to generate a batch script
generate_batch_script() {
    local task_cmd=$1
    local pid=$$
    local batch_script="$JOB_SCRIPTS_DIR/recovar_batch_${pid}.sh"
    
    cat > "$batch_script" <<'EOF'
#!/bin/bash
#SBATCH --output=/home/scratch.dleshchev_other/recovar/scripts/output/slurm-%j.out
set -e

echo "=========================================="
echo "RECOVAR Batch Job Starting"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPUs: $(nvidia-smi -L)"
echo "=========================================="

# Configuration
SCRIPT_DIR="/home/scratch.dleshchev_other/recovar"
CONTAINER_IMAGE="recovar:latest"
TASK_CMD="TASK_CMD_PLACEHOLDER"

cd "$SCRIPT_DIR"

# Check if container image exists
if ! docker images | grep -q "recovar.*latest"; then
    echo "Container image not found. Building..."
    bash scripts/build_container.sh
else
    echo "Container image found: $CONTAINER_IMAGE"
fi

# Run the container with the task
echo "Starting container and running task..."
echo "Task command: $TASK_CMD"
echo "GPU device: $NV_GPU"

docker run --rm --net host --ipc=host \
    --runtime=nvidia --gpus \"device=$NV_GPU\" \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    --user 143984:30 \
    "$CONTAINER_IMAGE" \
    -c "
        set -e
        echo 'Installing dependencies...'
        pixi install
        
        echo 'Installing RECOVAR...'
        pixi run install-recovar
        
        echo 'Running pixi task: $TASK_CMD'
        $TASK_CMD
        
        echo 'Task completed successfully!'
    "

echo "=========================================="
echo "RECOVAR Batch Job Completed"
echo "Date: $(date)"
echo "=========================================="
EOF
    
    # Replace the task command placeholder
    sed -i "s|TASK_CMD_PLACEHOLDER|$task_cmd|g" "$batch_script"
    
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
    
    echo "========================================"
    echo "Submitting job: $task_name"
    echo "Task command: $task_cmd"
    echo "Number of GPUs: $num_gpus"
    echo "Time limit: $time_limit"
    echo "========================================"
    
    # Generate the batch script
    batch_script=$(generate_batch_script "$task_cmd")
    echo "Generated batch script: $batch_script"
    
    # Submit using crun
    crun -q "$GPU_QUERY" --gpus="$num_gpus" -t "$time_limit" --cpu-arch-agnostic -b "$batch_script"
    
    echo "Job submitted successfully!"
    echo ""
}

# Main action handler
ACTION=$1

if [ -z "$ACTION" ]; then
    echo "Usage: $0 <action>"
    echo ""
    echo "Available actions:"
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
    echo "    profile-4gpu-256 - Profile 4 GPU run (1.5h)"
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
        submit_job "Profile 1 GPU (128-100k)" "pixi run profile-1gpu" 1 "01:00:00"
        ;;
    profile-2gpu)
        submit_job "Profile 2 GPUs (128-100k)" "pixi run profile-2gpu" 2 "00:45:00"
        ;;
    profile-4gpu)
        submit_job "Profile 4 GPUs (128-100k)" "pixi run profile-4gpu" 4 "00:30:00"
        ;;
    
    # Profiling (256-300k dataset)
    profile-1gpu-256)
        submit_job "Profile 1 GPU (256-300k)" "pixi run profile-1gpu-256" 1 "03:00:00"
        ;;
    profile-2gpu-256)
        submit_job "Profile 2 GPUs (256-300k)" "pixi run profile-2gpu-256" 2 "02:00:00"
        ;;
    profile-4gpu-256)
        submit_job "Profile 4 GPUs (256-300k)" "pixi run profile-4gpu-256" 4 "01:30:00"
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

