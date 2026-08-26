#!/usr/bin/env bash
#SBATCH --job-name=k1-ppref-coarse-v1
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${PPREF:?PPREF is required}"
: "${NATIVE_COARSE:?NATIVE_COARSE is required}"
: "${RECOVAR_CAPTURE:?RECOVAR_CAPTURE is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_CUDA_SHA256:?EXPECTED_CUDA_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/analyze_k1_exact_ppref_coarse_v1_boundary.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_exact_ppref_coarse_v1_boundary.py
CUDA_LIB=${CUDA_LIB:-${REPO}/recovar/cuda/libcuda_backproject.so}
RELION_BIND=${RELION_BIND:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared}
PHYSICAL_IMAGE_SIZE=${PHYSICAL_IMAGE_SIZE:-384}
NATIVE_DIRECTIONS=${NATIVE_DIRECTIONS:-768}
NATIVE_PSI=${NATIVE_PSI:-48}
OUTPUT=${OUTPUT:-${RUN_ROOT}/analysis/K1_EXACT_PPREF_COARSE_V1_BOUNDARY.json}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
for artifact in "${PPREF}" "${NATIVE_COARSE}" "${RECOVAR_CAPTURE}"; do test -s "${artifact}"; done
test ! -e "${OUTPUT}"
[[ "${PHYSICAL_IMAGE_SIZE}" =~ ^[1-9][0-9]*$ ]]
[[ "${NATIVE_DIRECTIONS}" =~ ^[1-9][0-9]*$ ]]
[[ "${NATIVE_PSI}" =~ ^[1-9][0-9]*$ ]]

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" "${RUN_ROOT}/logs"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
  "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/jax_cache" \
  "${RUNTIME_ROOT}/pycache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

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

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader \
  > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; repo=pathlib.Path('${REPO}').resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + '/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert jax.default_backend() == 'gpu'; assert cuda_backproject.cuda_available()"

sha256sum "${LAUNCHER}" "${ANALYZER}" "${CUDA_LIB}" "${PPREF}" \
  "${NATIVE_COARSE}" "${RECOVAR_CAPTURE}" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
git -C "${REPO}" status --short --branch \
  > "${RUN_ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"

command=("${PYTHON}" -m scripts.analyze_k1_exact_ppref_coarse_v1_boundary
  --ppref "${PPREF}" --native-coarse "${NATIVE_COARSE}"
  --recovar "${RECOVAR_CAPTURE}" --physical-image-size "${PHYSICAL_IMAGE_SIZE}"
  --native-directions "${NATIVE_DIRECTIONS}" --native-psi "${NATIVE_PSI}"
  --output-json "${OUTPUT}")
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"

cd "${REPO}"
start_epoch=$(date +%s)
"${command[@]}" > "${RUN_ROOT}/analysis/run_${SLURM_JOB_ID}.stdout" \
  2> "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
wall_s=$(( $(date +%s) - start_epoch ))
test -s "${OUTPUT}"
sha256sum "${OUTPUT}" > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d}\n' "${SLURM_JOB_ID}" "${wall_s}" \
  > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
