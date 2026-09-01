#!/usr/bin/env bash
#SBATCH --job-name=k1c10-5pose-cf
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CAPTURE_ROOT:?CAPTURE_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_PPREF_ANALYZER_SHA256:?EXPECTED_PPREF_ANALYZER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
BASELINE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_it1_scale_boundary_20260823T055027ET/output
NATIVE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_native_fresh_verbose_fine_row12334_it2_20260822T2320ET/relion
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_exact_it15_final_20260823T0230ET/cuda/libcuda_backproject.so
ANALYZER=${REPO}/scripts/analyze_k1_pose_winner_map_counterfactual.py
PPREF_ANALYZER=${REPO}/scripts/analyze_k1_ppref_map_provenance.py
PPREF=${NATIVE}/../analysis/native_live_it2_half1_ppref_cs056.schema-v1.bin
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -n "$(find "${CAPTURE_ROOT}" -maxdepth 1 -name 'SUCCESS_*' -print -quit)"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu scripts/run_full_refinement.py scripts/run_multi_iter_parity.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${PPREF_ANALYZER}" | awk '{print $1}')" = "${EXPECTED_PPREF_ANALYZER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
test -s "${PPREF}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${RUN_ROOT}/logs" "${RUN_ROOT}/analysis" "${RUN_ROOT}/maps" "${RUN_ROOT}/provenance"

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
export CUDA_VISIBLE_DEVICES=0 RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src

cd "${REPO}"
git status --short --branch > "${RUN_ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu scripts/run_full_refinement.py scripts/run_multi_iter_parity.py \
    > "${RUN_ROOT}/provenance/source_diff_${SLURM_JOB_ID}.patch"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"

command=("${PYTHON}" -m scripts.analyze_k1_pose_winner_map_counterfactual
    --contribution-dir "${CAPTURE_ROOT}/contributions"
    --pose-comparison "${BASELINE}/pose_comparison_iter000.npz"
    --intermediates "${BASELINE}/intermediates"
    --native-relion-dir "${NATIVE}"
    --output-dir "${RUN_ROOT}/maps"
    --source-rows 10692,25973,47531,83842,89358
    --output-json "${RUN_ROOT}/analysis/K1_CASE10_IT1_FIVE_POSE_COUNTERFACTUAL.json")
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
"${command[@]}" > "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1

ppref_command=("${PYTHON}" -m scripts.analyze_k1_ppref_map_provenance
    --ppref "${PPREF}"
    --map "saved_recovar:${BASELINE}/intermediates/it000_half1_reg.mrc:recovar"
    --map "raw_counterfactual:${RUN_ROOT}/maps/counterfactual_half1.mrc:recovar"
    --map "delta_adjusted_counterfactual:${RUN_ROOT}/maps/delta_adjusted_saved_half1.mrc:recovar"
    --map "native_relion:${NATIVE}/run_it001_half1_class001.mrc:relion"
    --output-json "${RUN_ROOT}/analysis/K1_CASE10_IT1_FIVE_POSE_PPREF.json")
printf '%q ' "${ppref_command[@]}" > "${RUN_ROOT}/provenance/ppref_command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/ppref_command_${SLURM_JOB_ID}.sh"
"${ppref_command[@]}" > "${RUN_ROOT}/logs/ppref_${SLURM_JOB_ID}.log" 2>&1

test -s "${RUN_ROOT}/analysis/K1_CASE10_IT1_FIVE_POSE_COUNTERFACTUAL.json"
test -s "${RUN_ROOT}/analysis/K1_CASE10_IT1_FIVE_POSE_PPREF.json"
! rg -q 'RESOURCE_EXHAUSTED|Out of memory|Traceback' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.log" "${RUN_ROOT}/logs/ppref_${SLURM_JOB_ID}.log"
find "${RUN_ROOT}/analysis" "${RUN_ROOT}/maps" -type f -print0 \
    | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
