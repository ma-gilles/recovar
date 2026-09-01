#!/usr/bin/env bash
#SBATCH --job-name=k1c05-r38594-h20
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row38594_native_fine_h20g1_20260822T1740ET/logs/science-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row38594_native_fine_h20g1_20260822T1740ET/logs/science-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --constraint=h100
#SBATCH --nodelist=della-h20g1
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

# Resolve the fully audited same-GPU launcher with only the physical H100 UUID
# and launcher identity changed. Slurm directives in the resolved stream are
# comments; this wrapper owns the h20g1 allocation above.
set -euo pipefail

SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
BASE_LAUNCHER=${SOURCE_ROOT}/scripts/run_k1_case05_row38594_native_fine.sbatch
WRAPPER=${SOURCE_ROOT}/scripts/run_k1_case05_row38594_native_fine_h20g1.sbatch
EXPECTED_BASE_SHA256=85315ae78477fa56b6dbb574a9ca45ccf23f2e31277880c48a7363e90c6d9b3f
EXPECTED_H20G1_GPU_UUID=GPU-b9385536-02b1-27fd-5e5f-99ad67950492

test "$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_BASE_SHA256}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256:?set wrapper hash}"

# These diagnostics are passive outputs of the same production score kernel.
export RELION_ACC_DUMP_CC_COMPONENTS=1
export RELION_DUMP_PASS1_DIFF2=1

sed \
  -e "s|EXPECTED_CAPTURE_GPU_UUID=GPU-0d7b80c7-fef8-e346-6332-de36ae1af518|EXPECTED_CAPTURE_GPU_UUID=${EXPECTED_H20G1_GPU_UUID}|" \
  -e "s|LAUNCHER=\${SOURCE_ROOT}/scripts/run_k1_case05_row38594_native_fine.sbatch|LAUNCHER=${WRAPPER}|" \
  "${BASE_LAUNCHER}" | bash
