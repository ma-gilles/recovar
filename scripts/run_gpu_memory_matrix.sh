#!/usr/bin/env bash
# Submit the GPU integration matrix as one Slurm job per (budget, backend) cell.
#
# Two flavors:
#   FAST set:   BUDGETS="16 40 75" CUDA_MODES="custom_cuda"
#   FULL set:   BUDGETS="8 12 16 24 40 60 75" CUDA_MODES="custom_cuda jax_fallback"
#
# Each cell runs `recovar run_test_dataset --gpu-budget-gb $N
# --fail-on-memory-exceed --memory-profile --no-delete` against a
# small-but-realistic synthetic dataset, then asserts:
#   - memory_plan.json exists and reports effective_budget_gb <= N
#   - memory_trace.jsonl exists because --memory-profile was passed, and
#     max(jax_peak_gb) <= N * 1.05 (production fail-on-memory-exceed slack)
#
# Aggregated results land in $OUT_ROOT/summary.txt after the summary job
# fires.

set -euo pipefail

WORKDIR="${WORKDIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
OUT_ROOT="${OUT_ROOT:-/scratch/gpfs/GILLES/mg6942/_agent_scratch/gpu_memory_matrix_$(date +%Y%m%d_%H%M%S)}"
BUDGETS="${BUDGETS:-16 40 75}"
CUDA_MODES="${CUDA_MODES:-custom_cuda}"
N_IMAGES="${N_IMAGES:-2000}"
GRID_SIZE="${GRID_SIZE:-128}"
TIME_LIMIT="${TIME_LIMIT:-1:00:00}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-gilles}"
SLURM_PARTITION="${SLURM_PARTITION:-cryoem}"
SERIAL="${SERIAL:-0}"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/sbatch" "$OUT_ROOT/cells"

job_ids=()
prev_job_id=""
for mode in $CUDA_MODES; do
  for gb in $BUDGETS; do
    cell_dir="$OUT_ROOT/cells/${mode}_${gb}gb"
    sbatch_script="$OUT_ROOT/sbatch/${mode}_${gb}gb.sh"
    dependency_line=""
    if [ "$SERIAL" = "1" ] && [ -n "$prev_job_id" ]; then
      dependency_line="#SBATCH --dependency=afterok:${prev_job_id}"
    fi

    if [ "$mode" = "jax_fallback" ]; then
      env_setup='export RECOVAR_DISABLE_CUDA=1'
    else
      env_setup='unset RECOVAR_DISABLE_CUDA'
    fi

    cat > "$sbatch_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=recovar-mem-${mode}-${gb}gb
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=$OUT_ROOT/logs/${mode}-${gb}gb-%j.out
#SBATCH --exclusive
${dependency_line}

set -euo pipefail
cd "${WORKDIR}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/memmatrix_\${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/memmatrix_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/memmatrix_\${SLURM_JOB_ID}"
export RECOVAR_CUDA_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/recovar_cuda_cache/memmatrix_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR" "\$PIXI_HOME" "\$RATTLER_CACHE_DIR" "\$RECOVAR_CUDA_CACHE_DIR"
if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export CUDA_PATH=/usr/local/cuda-12.8
  export NVCC=/usr/local/cuda-12.8/bin/nvcc
  export CUDACXX="\$NVCC"
  export PATH="/usr/local/cuda-12.8/bin:\$PATH"
elif ! command -v nvcc >/dev/null 2>&1; then
  echo "FAIL: nvcc not found; set NVCC/CUDACXX or install cudatoolkit/12.8"
  exit 1
fi
${env_setup}

archive_dir="${cell_dir}/_memory_artifacts"
archive_diagnostics() {
  mkdir -p "\$archive_dir"
  find "${cell_dir}" -type f \\
    \\( -name memory_plan.json -o -name memory_trace.jsonl -o -name run.log -o -name job.json -o -name command.txt \\) \\
    -print0 2>/dev/null | while IFS= read -r -d '' path; do
      rel="\${path#${cell_dir}/}"
      safe="\${rel//\\//__}"
      cp -f "\$path" "\$archive_dir/\$safe" 2>/dev/null || true
    done
}
cleanup_heavy_outputs() {
  rm -rf "${cell_dir}/test_dataset"
  rm -rf "\$TMPDIR" "\$PIXI_HOME" "\$RATTLER_CACHE_DIR" "\$RECOVAR_CUDA_CACHE_DIR"
}
trap cleanup_heavy_outputs EXIT

if [ "${mode}" = "custom_cuda" ]; then
  pixi run python -c "from recovar.cuda_backproject import cuda_available; raise SystemExit(0 if cuda_available() else 'custom CUDA unavailable')"
fi

mkdir -p "${cell_dir}"
run_rc=0
pixi run python -m recovar.command_line run_test_dataset \\
    --output-dir "${cell_dir}" \\
    --n-images ${N_IMAGES} \\
    --image-size ${GRID_SIZE} \\
    --gpu-budget-gb ${gb} \\
    --fail-on-memory-exceed \\
    --memory-profile \\
    --no-delete || run_rc=\$?
archive_diagnostics || true
if [ "\$run_rc" -ne 0 ]; then
  exit "\$run_rc"
fi

# Verify diagnostic outputs exist somewhere under the cell directory
# (memory_plan is always written; memory_trace is written because this
# harness passes --memory-profile).
plan_count=\$(find "${cell_dir}" -name memory_plan.json | wc -l)
trace_count=\$(find "${cell_dir}" -name memory_trace.jsonl | wc -l)
echo "memory_plan.json instances: \$plan_count"
echo "memory_trace.jsonl instances: \$trace_count"
if [ "\$plan_count" -lt 1 ]; then
  echo "FAIL: no memory_plan.json was written"
  exit 1
fi
if [ "\$trace_count" -lt 1 ]; then
  echo "FAIL: no memory_trace.jsonl was written"
  exit 1
fi

# Contract assertions: the planner must not inflate the user's stated
# budget, and actual peak memory must remain within production
# --fail-on-memory-exceed slack.
effective_gb=\$(pixi run python -c "
import json
from pathlib import Path
values = []
for path in Path('${cell_dir}').rglob('memory_plan.json'):
    row = json.loads(path.read_text())
    budget = row.get('budget') or {}
    if 'effective_budget_gb' in budget:
        values.append(float(budget['effective_budget_gb']))
print(max(values) if values else 0.0)
")
echo "max effective planner budget = \${effective_gb} GB"
pixi run python -c "
effective = float('\${effective_gb}')
budget = float('${gb}')
if effective > budget + 1e-3:
    raise SystemExit(f'CONTRACT VIOLATION: effective planner budget {effective} GB > requested budget {budget} GB')
print('CONTRACT_OK: effective planner budget does not exceed request')
"

SLACK=\${SLACK:-1.05}
budget=${gb}
threshold=\$(pixi run python -c "print(${gb} * \$SLACK)")
peak_gb=\$(find "${cell_dir}" -name memory_trace.jsonl -exec cat {} \\; \\
  | pixi run python -c "
import json, sys
peaks = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except Exception:
        continue
    if row.get('jax_memory_stats_available') and 'jax_peak_gb' in row:
        peaks.append(float(row['jax_peak_gb']))
print(max(peaks) if peaks else 0.0)
")
echo "observed peak = \${peak_gb} GB; budget = \${budget} GB; threshold = \${threshold} GB"
pixi run python -c "
import sys
peak = float('\${peak_gb}')
threshold = float('\${threshold}')
if peak > threshold:
    sys.exit(f'CONTRACT VIOLATION: peak {peak} GB > budget * \$SLACK = {threshold} GB')
print('CONTRACT_OK: peak fits within budget * \$SLACK')
"
EOF
    chmod +x "$sbatch_script"
    job_id=$(sbatch --parsable "$sbatch_script")
    echo "submitted: ${mode}-${gb}gb (job $job_id)"
    job_ids+=("$job_id")
    prev_job_id="$job_id"
  done
done

# Summary job: depends on all cells.
deps=$(IFS=:; echo "${job_ids[*]}")
summary_script="$OUT_ROOT/sbatch/summary.sh"
cat > "$summary_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=recovar-mem-summary
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=00:10:00
#SBATCH --output=$OUT_ROOT/logs/summary-%j.out
#SBATCH --dependency=afterany:$deps

set -euo pipefail
echo "=== GPU memory matrix summary ===" > $OUT_ROOT/summary.txt
for log in $OUT_ROOT/logs/*.out; do
  echo "--- \$log" >> $OUT_ROOT/summary.txt
  tail -20 "\$log" >> $OUT_ROOT/summary.txt
done
echo "Wrote $OUT_ROOT/summary.txt"
EOF
chmod +x "$summary_script"
summary_id=$(sbatch --parsable "$summary_script")

echo
echo "Submitted ${#job_ids[@]} matrix cells + 1 summary (job $summary_id)."
echo "OUT_ROOT=$OUT_ROOT"
echo "Watch: squeue -u \$USER --name=recovar-mem-"
echo "Summary lands at: $OUT_ROOT/summary.txt"
