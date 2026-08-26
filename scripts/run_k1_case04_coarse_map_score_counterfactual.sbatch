#!/usr/bin/env bash
#SBATCH --job-name=k1c04-mapscore
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=${K1_CASE04_MAP_SCORE_CUDA_LIB:-${REPO}/recovar/cuda/libcuda_backproject.so}
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared
LAUNCHER=${REPO}/scripts/run_k1_case04_coarse_map_score_counterfactual.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_coarse_map_score_counterfactual.py
NATIVE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_native_coarse_stack68695_20260821T1010ET
RECOVAR_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_recovar_fresh_coarse_stack68695_20260821T1035ET
NATIVE_CAPTURE=${K1_CASE04_MAP_SCORE_NATIVE_CAPTURE:-${NATIVE_ROOT}/capture/part65218_stack68695.coarse-v1.bin}
RECOVAR_CAPTURE=${K1_CASE04_MAP_SCORE_RECOVAR_CAPTURE:-${RECOVAR_ROOT}/coarse/significance_orig068694_it002_cs100.npz}
NATIVE_MAP=${K1_CASE04_MAP_SCORE_NATIVE_MAP:-${NATIVE_ROOT}/relion/run_it001_half2_class001.mrc}
RECOVAR_MAP=${K1_CASE04_MAP_SCORE_RECOVAR_MAP:-${RECOVAR_ROOT}/output/intermediates/it000_half2_reg.mrc}
NATIVE_MAP_CONVENTION=${K1_CASE04_MAP_SCORE_NATIVE_MAP_CONVENTION:-relion}
RECOVAR_MAP_CONVENTION=${K1_CASE04_MAP_SCORE_RECOVAR_MAP_CONVENTION:-recovar}
OUTPUT=${K1_CASE04_MAP_SCORE_OUTPUT:-${ROOT}/analysis/COARSE_MAP_SCORE_COUNTERFACTUAL.json}
WINNER=${K1_MAP_SCORE_WINNER:-26325,19}
TARGET=${K1_MAP_SCORE_TARGET:-27095,14}
NATIVE_DIRECTIONS=${K1_MAP_SCORE_NATIVE_DIRECTIONS:-768}
NATIVE_PSI=${K1_MAP_SCORE_NATIVE_PSI:-48}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test -s "${CUDA_LIB}"
for artifact in "${NATIVE_CAPTURE}" "${RECOVAR_CAPTURE}" "${NATIVE_MAP}" "${RECOVAR_MAP}"; do
    test -s "${artifact}"
done
test ! -e "${OUTPUT}"
[[ "${WINNER}" =~ ^[0-9]+,[0-9]+$ ]]
[[ "${TARGET}" =~ ^[0-9]+,[0-9]+$ ]]
[[ "${NATIVE_DIRECTIONS}" =~ ^[1-9][0-9]*$ ]]
[[ "${NATIVE_PSI}" =~ ^[1-9][0-9]*$ ]]

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
export CUDA_VISIBLE_DEVICES=0 RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${ANALYZER}" "${CUDA_LIB}" \
    "${NATIVE_CAPTURE}" "${RECOVAR_CAPTURE}" "${NATIVE_MAP}" "${RECOVAR_MAP}" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=("${PYTHON}" -m scripts.analyze_k1_coarse_map_score_counterfactual
  --native-capture "${NATIVE_CAPTURE}" --recovar-capture "${RECOVAR_CAPTURE}"
  --native-map "${NATIVE_MAP}" --recovar-map "${RECOVAR_MAP}"
  --native-map-convention "${NATIVE_MAP_CONVENTION}"
  --recovar-map-convention "${RECOVAR_MAP_CONVENTION}"
  --winner "${WINNER}" --target "${TARGET}"
  --native-directions "${NATIVE_DIRECTIONS}" --native-psi "${NATIVE_PSI}"
  --output-json "${OUTPUT}")
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
cd "${REPO}"
start_epoch=$(date +%s)
"${command[@]}" > "${ROOT}/analysis/run_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
wall_s=$(( $(date +%s) - start_epoch ))
test -s "${OUTPUT}"
printf '{"job_id":"%s","wall_s":%d}\n' "${SLURM_JOB_ID}" "${wall_s}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${OUTPUT}" > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
