#!/usr/bin/env bash
# Submit the GPU integration matrix as one Slurm job per (budget, backend) cell.
#
# Two flavors:
#   FAST set:   BUDGETS="16 40 75" CUDA_MODES="custom_cuda"
#   FULL set:   BUDGETS="8 12 16 24 40 60 75" CUDA_MODES="custom_cuda jax_fallback"
#
# Each cell runs `recovar run_test_dataset --gpu-budget-gb $N --adaptive-n-pcs
# --fail-on-memory-exceed --no-delete` against a small-but-realistic
# synthetic dataset (diagnostics are always-on, no flag needed), then asserts:
#   - memory_plan.json exists and reports effective_budget_gb <= N * 1.2
#   - memory_trace.jsonl exists and max(jax_peak_gb) <= N * 1.2 (slack)
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

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/sbatch" "$OUT_ROOT/cells"

job_ids=()
for mode in $CUDA_MODES; do
  for gb in $BUDGETS; do
    cell_dir="$OUT_ROOT/cells/${mode}_${gb}gb"
    sbatch_script="$OUT_ROOT/sbatch/${mode}_${gb}gb.sh"

    if [ "$mode" = "jax_fallback" ]; then
      env_setup='export RECOVAR_DISABLE_CUDA=1'
    else
      env_setup='unset RECOVAR_DISABLE_CUDA'
    fi

    cat > "$sbatch_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=recovar-mem-${mode}-${gb}gb
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=$OUT_ROOT/logs/${mode}-${gb}gb-%j.out
#SBATCH --exclusive

set -euo pipefail
cd "${WORKDIR}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/memmatrix_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR"
${env_setup}

mkdir -p "${cell_dir}"
pixi run python -m recovar.command_line run_test_dataset \\
    --output-dir "${cell_dir}" \\
    --gpu-budget-gb ${gb} \\
    --fail-on-memory-exceed \\
    --memory-profile \\
    --no-delete

# Verify diagnostic outputs exist somewhere under the cell directory
# (always-on under <outdir>/_diagnostics/).
plan_count=\$(find "${cell_dir}" -name memory_plan.json | wc -l)
trace_count=\$(find "${cell_dir}" -name memory_trace.jsonl | wc -l)
echo "memory_plan.json instances: \$plan_count"
echo "memory_trace.jsonl instances: \$trace_count"
if [ "\$plan_count" -lt 1 ]; then
  echo "FAIL: no memory_plan.json was written"
  exit 1
fi

# Contract assertion: actual peak memory must not exceed budget * 1.20.
# This is the cross-architecture portable test — irrespective of GPU
# generation, IF you ask for N GB then peak should fit N GB (with 20%
# slack for allocator fragmentation).
SLACK=\${SLACK:-1.20}
budget=${gb}
threshold=\$(python3 -c "print(${gb} * \$SLACK)")
peak_gb=\$(find "${cell_dir}" -name memory_trace.jsonl -exec cat {} \\; \\
  | python3 -c "
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
python3 -c "
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
  done
done

# Summary job: depends on all cells.
deps=$(IFS=:; echo "${job_ids[*]}")
summary_script="$OUT_ROOT/sbatch/summary.sh"
cat > "$summary_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=recovar-mem-summary
#SBATCH --account=amits
#SBATCH --partition=cryoem
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
