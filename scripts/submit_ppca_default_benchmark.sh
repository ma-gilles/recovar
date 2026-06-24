#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_ppca_bench_dev2}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/CRYOEM/singerlab/mg6942}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/ppca_default_benchmark_${STAMP}}"
MANIFEST="${MANIFEST:-$RESULTS_ROOT/manifest.json}"
PPCA_BENCH_SCRATCH_ROOT="${PPCA_BENCH_SCRATCH_ROOT:-$PROJECT_ROOT/ppca_default_benchmark_scratch}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$PROJECT_ROOT/slurmo}"
GPU_GB="${GPU_GB:-70}"
GRID_SIZE="${GRID_SIZE:-256}"
MAX_ARRAY_CONCURRENCY="${MAX_ARRAY_CONCURRENCY:-64}"

cd "$REPO_ROOT"
mkdir -p "$RESULTS_ROOT"
mkdir -p "$SLURM_LOG_DIR"

PYTHON="$REPO_ROOT/.pixi/envs/default/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Missing pixi python at $PYTHON" >&2
  exit 1
fi

"$PYTHON" scripts/ppca_default_benchmark.py write-manifest \
  --repo-root "$REPO_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --output "$MANIFEST" \
  --grid-size "$GRID_SIZE"

N_CASES="$("$PYTHON" - "$MANIFEST" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f)["n_cases"])
PY
)"

if [ "$N_CASES" -le 0 ]; then
  echo "Manifest contains no cases: $MANIFEST" >&2
  exit 1
fi

ARRAY_SPEC="0-$((N_CASES - 1))%${MAX_ARRAY_CONCURRENCY}"
array_jid="$(
  sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --output="$SLURM_LOG_DIR/%x-%A_%a.out" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",MANIFEST="$MANIFEST",GPU_GB="$GPU_GB",PPCA_BENCH_SCRATCH_ROOT="$PPCA_BENCH_SCRATCH_ROOT" \
    scripts/ppca_default_benchmark_array.sbatch
)"

aggregate_jid="$(
  sbatch --parsable \
    --job-name="ppca-bench-aggregate" \
    --dependency="afterany:${array_jid}" \
    --account=gilles \
    --partition=cryoem \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=32G \
    --time=01:00:00 \
    --output="$SLURM_LOG_DIR/%x-%j.out" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",MANIFEST="$MANIFEST",PPCA_BENCH_SCRATCH_ROOT="$PPCA_BENCH_SCRATCH_ROOT" \
    --wrap="cd '$REPO_ROOT' && export PYTHONNOUSERSITE=1 TMPDIR='$PPCA_BENCH_SCRATCH_ROOT/tmp/\${SLURM_JOB_ID}' PIXI_HOME='$PPCA_BENCH_SCRATCH_ROOT/pixi_home/\${SLURM_JOB_ID}' RATTLER_CACHE_DIR='$PPCA_BENCH_SCRATCH_ROOT/rattler_cache/\${SLURM_JOB_ID}' && mkdir -p \"\$TMPDIR\" \"\$PIXI_HOME\" \"\$RATTLER_CACHE_DIR\" && unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV && '$PYTHON' scripts/ppca_default_benchmark.py summarize --manifest '$MANIFEST'"
)"

{
  echo "results_root=$RESULTS_ROOT"
  echo "manifest=$MANIFEST"
  echo "ppca_bench_scratch_root=$PPCA_BENCH_SCRATCH_ROOT"
  echo "slurm_log_dir=$SLURM_LOG_DIR"
  echo "array_job=$array_jid"
  echo "aggregate_job=$aggregate_jid"
  echo "array_spec=$ARRAY_SPEC"
  echo "grid_size=$GRID_SIZE"
} | tee "$RESULTS_ROOT/submission.txt"
