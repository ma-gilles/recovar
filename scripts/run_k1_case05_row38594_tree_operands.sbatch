#!/usr/bin/env bash
#SBATCH --job-name=k1c05-r38594-tree
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row38594_tree_operands_20260822T1755ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row38594_tree_operands_20260822T1755ET/logs/run-%j.err
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

# Reuse the audited row-38594 pass-2/contribution launcher, adding only the
# existing passive tree-rescore operand dump at the same first-iteration row.
set -euo pipefail

SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
BASE_LAUNCHER=${SOURCE_ROOT}/scripts/run_k1_case05_it1_translation_contribution.sbatch
WRAPPER=${SOURCE_ROOT}/scripts/run_k1_case05_row38594_tree_operands.sbatch
EXPECTED_BASE_SHA256=38019dde67cea573b87d44ea4b48a1b31891d6145ae518ea572139a0de8c5097

test "$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_BASE_SHA256}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = \
  "${EXPECTED_LAUNCHER_SHA256:?set wrapper hash}"

sed \
  -e "s|LAUNCHER=\${REPO}/scripts/run_k1_case05_it1_translation_contribution.sbatch|LAUNCHER=${WRAPPER}|" \
  -e '/export RECOVAR_PASS2_DUMP_ITERATION=1/a\export RECOVAR_SIGNIFICANCE_DUMP_DIR=${ROOT}/tree_rescore\nexport RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=38594\nexport RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE=56\nexport RECOVAR_SIGNIFICANCE_DUMP_ITERATION=1' \
  "${BASE_LAUNCHER}" | bash
