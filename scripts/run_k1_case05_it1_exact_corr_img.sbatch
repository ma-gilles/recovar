#!/usr/bin/env bash
#SBATCH --job-name=k1c05-it1-corr
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_exact_corr_img_20260822T1850ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_exact_corr_img_20260822T1850ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:30:00

# Complete both halves of iteration 1 with the source-row diagnostic captures
# retained, but without stopping after the target particle is written.
set -euo pipefail

SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
BASE_LAUNCHER=${SOURCE_ROOT}/scripts/run_k1_case05_it1_translation_contribution.sbatch
WRAPPER=${SOURCE_ROOT}/scripts/run_k1_case05_it1_exact_corr_img.sbatch
EXPECTED_BASE_SHA256=38019dde67cea573b87d44ea4b48a1b31891d6145ae518ea572139a0de8c5097

test "$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_BASE_SHA256}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = \
  "${EXPECTED_LAUNCHER_SHA256:?set wrapper hash}"

sed \
  -e "s|LAUNCHER=\${REPO}/scripts/run_k1_case05_it1_translation_contribution.sbatch|LAUNCHER=${WRAPPER}|" \
  -e '/export RECOVAR_BPREF_CONTRIBUTION_STOP_AFTER_TARGET=1/d' \
  -e "/rg -q 'BPref contribution dump completed; stopping at the requested pass-2 boundary'/d" \
  "${BASE_LAUNCHER}" | bash
