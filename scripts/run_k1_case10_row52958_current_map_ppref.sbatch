#!/usr/bin/env bash
#SBATCH --job-name=k1c10-r52958-map
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
: "${NATIVE_MAP:?NATIVE_MAP is required}"
: "${RECOVAR_MAP:?RECOVAR_MAP is required}"
: "${RECOVAR_CAPTURE:?RECOVAR_CAPTURE is required}"
: "${NATIVE_COARSE:?NATIVE_COARSE is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_BUILDER_SHA256:?EXPECTED_BUILDER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_CUDA_SHA256:?EXPECTED_CUDA_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case10_row52958_current_map_ppref.sbatch
BUILDER=${REPO}/scripts/build_k1_ppref_capture_from_map.py
ANALYZER=${REPO}/scripts/analyze_k1_exact_ppref_coarse_v1_boundary.py
CUDA_LIB=${CUDA_LIB:-${REPO}/recovar/cuda/libcuda_backproject.so}
RELION_BIND=${RELION_BIND:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared}
CURRENT_SIZE=${CURRENT_SIZE:-56}
PHYSICAL_IMAGE_SIZE=${PHYSICAL_IMAGE_SIZE:-384}
NATIVE_DIRECTIONS=${NATIVE_DIRECTIONS:-768}
NATIVE_PSI=${NATIVE_PSI:-48}
NATIVE_PPREF=${RUN_ROOT}/ppref/native_map_ppref.bin
RECOVAR_PPREF=${RUN_ROOT}/ppref/recovar_map_ppref.bin
NATIVE_REPORT=${RUN_ROOT}/analysis/native_map_boundary.json
RECOVAR_REPORT=${RUN_ROOT}/analysis/recovar_map_boundary.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${BUILDER}" | awk '{print $1}')" = "${EXPECTED_BUILDER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
for artifact in "${NATIVE_MAP}" "${RECOVAR_MAP}" "${RECOVAR_CAPTURE}" "${NATIVE_COARSE}"; do
  test -s "${artifact}"
done
for output in "${NATIVE_PPREF}" "${RECOVAR_PPREF}" "${NATIVE_REPORT}" "${RECOVAR_REPORT}"; do
  test ! -e "${output}"
done
for value in "${CURRENT_SIZE}" "${PHYSICAL_IMAGE_SIZE}" "${NATIVE_DIRECTIONS}" "${NATIVE_PSI}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]]
done

mkdir -p "${RUN_ROOT}/ppref" "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" "${RUN_ROOT}/logs"
mkdir -p "${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp" \
  "${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home" \
  "${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache" \
  "${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache" \
  "${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache

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
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; repo=pathlib.Path('${REPO}').resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + '/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert jax.default_backend() == 'gpu'; assert cuda_backproject.cuda_available(); from recovar.relion_bind import _relion_bind_core"

sha256sum "${LAUNCHER}" "${BUILDER}" "${ANALYZER}" "${CUDA_LIB}" \
  "${NATIVE_MAP}" "${RECOVAR_MAP}" "${RECOVAR_CAPTURE}" "${NATIVE_COARSE}" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
git -C "${REPO}" status --short --branch \
  > "${RUN_ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"

cd "${REPO}"
"${PYTHON}" -m scripts.build_k1_ppref_capture_from_map \
  --map "${NATIVE_MAP}" --map-convention relion \
  --current-size "${CURRENT_SIZE}" --padding-factor 2 --iteration 2 --rank 1 \
  --output "${NATIVE_PPREF}"
"${PYTHON}" -m scripts.build_k1_ppref_capture_from_map \
  --map "${RECOVAR_MAP}" --map-convention recovar \
  --current-size "${CURRENT_SIZE}" --padding-factor 2 --iteration 2 --rank 1 \
  --output "${RECOVAR_PPREF}"

"${PYTHON}" -m scripts.analyze_k1_exact_ppref_coarse_v1_boundary \
  --ppref "${NATIVE_PPREF}" --recovar "${RECOVAR_CAPTURE}" \
  --native-coarse "${NATIVE_COARSE}" --physical-image-size "${PHYSICAL_IMAGE_SIZE}" \
  --native-directions "${NATIVE_DIRECTIONS}" --native-psi "${NATIVE_PSI}" \
  --output-json "${NATIVE_REPORT}" \
  > "${RUN_ROOT}/analysis/native_map_${SLURM_JOB_ID}.stdout"
"${PYTHON}" -m scripts.analyze_k1_exact_ppref_coarse_v1_boundary \
  --ppref "${RECOVAR_PPREF}" --recovar "${RECOVAR_CAPTURE}" \
  --native-coarse "${NATIVE_COARSE}" --physical-image-size "${PHYSICAL_IMAGE_SIZE}" \
  --native-directions "${NATIVE_DIRECTIONS}" --native-psi "${NATIVE_PSI}" \
  --output-json "${RECOVAR_REPORT}" \
  > "${RUN_ROOT}/analysis/recovar_map_${SLURM_JOB_ID}.stdout"

sha256sum "${NATIVE_PPREF}" "${RECOVAR_PPREF}" "${NATIVE_REPORT}" "${RECOVAR_REPORT}" \
  > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
