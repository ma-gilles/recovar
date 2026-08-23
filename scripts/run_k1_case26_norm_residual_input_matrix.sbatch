#!/usr/bin/env bash
#SBATCH --job-name=k1c26-norm-input
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_norm_residual_inputs_20260810T2245ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_norm_residual_inputs_20260810T2245ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=00:08:00

set -euo pipefail
: "${EXPECTED_MATRIX_SHA256:?submission must pin matrix launcher hash}"
: "${EXPECTED_CHILD_SHA256:?submission must pin child launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"
: "${EXPECTED_REPO_DIFF_SHA256:?submission must pin repository diff}"

ROOT=${RUN_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_norm_residual_inputs_20260810T2245ET}
RUNTIME=${RUNTIME_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_norm_residual_inputs_20260810T2245ET}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
MATRIX=${REPO}/scripts/run_k1_case26_norm_residual_input_matrix.sbatch
CHILD=${REPO}/scripts/run_k1_case26_norm_state_capture.sbatch

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${MATRIX}" | awk '{print $1}')" = "${EXPECTED_MATRIX_SHA256}"
test "$(sha256sum "${CHILD}" | awk '{print $1}')" = "${EXPECTED_CHILD_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
mkdir -p "${ROOT}/logs" "${ROOT}/provenance"

run_arm() {
    local label=$1
    local cache_mode=$2
    local arm_root=${ROOT}/${label}
    local arm_runtime=${RUNTIME}/${label}
    mkdir -p "${arm_root}" "${arm_runtime}"
    touch "${arm_root}/SAFE_TO_DELETE" "${arm_runtime}/SAFE_TO_DELETE"
    EXPECTED_LAUNCHER_SHA256=${EXPECTED_CHILD_SHA256} \
    EXPECTED_REPO_HEAD=${EXPECTED_REPO_HEAD} \
    EXPECTED_REPO_DIFF_SHA256=${EXPECTED_REPO_DIFF_SHA256} \
    RUN_ROOT=${arm_root} RUNTIME_ROOT=${arm_runtime} \
    PASS2_CACHE_MODE=${cache_mode} PASS2_DUMP_DIR=${arm_root}/pass2 \
    PASS2_DUMP_NORM_RESIDUAL_INPUTS=1 \
    bash "${CHILD}"
}

run_arm dump_cache_on on
run_arm dump_cache_off off

sha256sum "${ROOT}"/*/pass2/norm_residual_orig000066_half2_cs056.npz \
    > "${ROOT}/provenance/norm_input_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
