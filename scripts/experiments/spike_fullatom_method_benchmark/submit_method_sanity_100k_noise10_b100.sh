#!/usr/bin/env bash
# Submit the fixed-100k method sanity jobs: cryoDRGN, CryoSPARC 3DFlex, and RECOVAR pipeline.

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529}"
mkdir -p "$BENCH_ROOT/slurm"

cd "$WORKDIR"

drgn_job="$(sbatch --parsable scripts/experiments/spike_fullatom_method_benchmark/submit_cryodrgn_100k_noise10_b100_sanity.sbatch)"
three_df_job="$(sbatch --parsable scripts/experiments/spike_fullatom_method_benchmark/submit_cryosparc_3dflex_100k_noise10_b100_sanity.sbatch)"
recovar_job="$(sbatch --parsable scripts/experiments/spike_fullatom_method_benchmark/submit_recovar_pipeline_100k_noise10_b100_param_sanity.sbatch)"

manifest="$BENCH_ROOT/slurm/submitted_jobs_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "submitted_at=$(date -Is)"
  echo "bench_root=$BENCH_ROOT"
  echo "cryodrgn_job=$drgn_job"
  echo "cryosparc_3dflex_submit_job=$three_df_job"
  echo "recovar_pipeline_param_job=$recovar_job"
} | tee "$manifest"

echo "MANIFEST=$manifest"
