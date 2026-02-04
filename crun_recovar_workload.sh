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
#SBATCH --output=/home/scratch.dleshchev_other/heterogeneity_dev/scripts/output/slurm-%j.out
set -e

echo "=========================================="
echo "RECOVAR Batch Job Starting"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPUs: $(nvidia-smi -L)"
echo "=========================================="

# Configuration
SCRIPT_DIR="/home/scratch.dleshchev_other/heterogeneity_dev"
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
    
    # Submit using crun and capture output
    # Note: crun controls output location and places files in the working directory
    # The SLURM --output directive in the batch script is ignored by crun
    local crun_output=$(crun -q "$GPU_QUERY" --gpus="$num_gpus" -t "$time_limit" --cpu-arch-agnostic \
        -b "$batch_script" 2>&1)
    
    echo "$crun_output"
    
    # Extract job ID from crun output
    local job_id=$(echo "$crun_output" | grep "Job Id" | awk '{print $NF}')
    
    if [ -n "$job_id" ]; then
        echo ""
        echo "Job ID: $job_id"
        echo "Output file: slurm-$job_id.out"
        echo ""
        echo "To monitor: tail -f slurm-$job_id.out"
        echo "To organize outputs later, run: ./crun_recovar_workload.sh organize-outputs"
    fi
    
    echo "Job submitted successfully!"
    echo ""
}

# Function to submit a multi-node job
submit_multinode_job() {
    local task_name=$1
    local script_path=$2
    local num_nodes=$3
    local num_gpus=$4
    local time_limit=$5
    
    echo "========================================"
    echo "Submitting MULTI-NODE job: $task_name"
    echo "Script: $script_path"
    echo "Number of nodes: $num_nodes"
    echo "GPUs per node: $num_gpus"
    echo "Time limit: $time_limit"
    echo "GPU query: $GPU_QUERY"
    echo "========================================"
    
    # For multi-node jobs, we use crun's multi-node support
    # crun will handle the SLURM submission with proper node allocation
    crun -q "$GPU_QUERY" --nodes="$num_nodes" --gpus="$num_gpus" -t "$time_limit" --cpu-arch-agnostic \
        -b "$script_path"
    
    echo "Multi-node job submitted successfully!"
    echo "Monitor with: tail -f $SCRIPT_DIR/scripts/output/slurm-*.out"
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
    echo "  Multi-node frequency-parallel (2 GPU testing):"
    echo "    freq-parallel-2node  - Frequency-parallel with 2 nodes, 2 GPUs each (1h)"
    echo ""
    echo "  Speedup testing:"
    echo "    test-baseline-2gpu   - Single-node baseline with 2 GPUs (20k images, 1h)"
    echo "    test-2node-freq      - 2-node frequency-parallel, 2 GPUs each (20k images, 1h)"
    echo ""
    echo "  Utilities:"
    echo "    organize-outputs     - Move all slurm-*.out files to scripts/output/"
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
    
    # Profiling (128-100k dataset, full 100k images)
    profile-1gpu)
        submit_job "Profile 1 GPU (128-100k)" "pixi run profile-1gpu" 1 "01:00:00"
        ;;
    profile-2gpu)
        submit_job "Profile 2 GPUs (128-100k)" "pixi run profile-2gpu" 2 "01:30:00"
        ;;
    profile-4gpu)
        submit_job "Profile 4 GPUs (128-100k)" "pixi run profile-4gpu" 4 "00:45:00"
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
    
    # Multi-node frequency-parallel
    freq-parallel-2node)
        submit_multinode_job "Freq-Parallel 2 Nodes" "$SCRIPT_DIR/scripts/run_frequency_parallel_multigpu.sh" 2 2 "01:00:00"
        ;;
    
    # Speedup testing
    test-baseline-2gpu)
        submit_job "Speedup-Baseline-2GPU-20k" "pixi run speedup-baseline-2gpu" 2 "01:00:00"
        ;;
    
    test-2node-freq)
        echo "========================================"
        echo "IMPORTANT: Multi-node jobs need custom SLURM script"
        echo "========================================"
        echo "For 2-node testing, use:"
        echo "  sbatch scripts/submit_2node_speedup_test.sh"
        echo ""
        echo "This is because crun's multi-node support requires special handling."
        exit 1
        ;;
    
    organize-outputs)
        echo "========================================"
        echo "Organizing output files"
        echo "========================================"
        # Find all slurm output files in the root directory
        shopt -s nullglob
        files=("$SCRIPT_DIR"/slurm-*.out)
        if [ ${#files[@]} -eq 0 ]; then
            echo "No slurm output files found in root directory"
        else
            echo "Moving ${#files[@]} files to scripts/output/"
            for file in "${files[@]}"; do
                filename=$(basename "$file")
                echo "  Moving $filename"
                mv "$file" "$OUTPUT_DIR/"
            done
            echo "Done! All output files moved to scripts/output/"
        fi
        ;;
    
    *)
        echo "Error: Unknown action '$ACTION'"
        echo "Run '$0' without arguments to see available actions."
        exit 1
        ;;
esac

