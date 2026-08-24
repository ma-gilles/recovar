#!/usr/bin/env bash
#SBATCH --job-name=k1c05-map-ppref
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
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_BUILDER_SHA256:?EXPECTED_BUILDER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_CUDA_SHA256:?EXPECTED_CUDA_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case05_map_ppref_coarse_boundary.sbatch
BUILDER=${REPO}/scripts/build_k1_ppref_capture_from_map.py
ANALYZER=${REPO}/scripts/analyze_k1_exact_ppref_coarse_boundary.py
CUDA_LIB=${CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_stage_split_20260821T1600ET/cuda/libcuda_backproject.so}
RELION_BIND=${RELION_BIND:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared}
NATIVE_COMPONENTS=${NATIVE_COMPONENTS:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row94083_native_fresh_winner_lane_it2_20260822T1220ET/capture/part93429_stack94084.p1-v2.bin}
RECOVAR_CAPTURE=${RECOVAR_CAPTURE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row94083_recovar_projection_winner_it2_20260822T1227ET/pass1/significance_orig094083_it002_cs056.npz}
NATIVE_MAP=${NATIVE_MAP:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row94083_native_fresh_winner_lane_it2_20260822T1220ET/relion/run_it001_half1_class001.mrc}
RECOVAR_MAP=${RECOVAR_MAP:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row94083_recovar_projection_winner_it2_20260822T1227ET/output/intermediates/it000_half1_reg.mrc}
RECOVAR_MAP_CONVENTION=${K1_MAP_PPREF_CANDIDATE_CONVENTION:-recovar}
CURRENT_SIZE=${K1_MAP_PPREF_CURRENT_SIZE:-56}
NATIVE_PPREF=${RUN_ROOT}/ppref/native_numbered_map_ppref.bin
RECOVAR_PPREF=${RUN_ROOT}/ppref/recovar_numbered_map_ppref.bin
NATIVE_REPORT=${RUN_ROOT}/analysis/K1_NATIVE_NUMBERED_MAP_PPREF_COARSE_BOUNDARY.json
RECOVAR_REPORT=${RUN_ROOT}/analysis/K1_RECOVAR_NUMBERED_MAP_PPREF_COARSE_BOUNDARY.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
case "${CURRENT_SIZE}" in ''|*[!0-9]*) echo "K1_MAP_PPREF_CURRENT_SIZE must be a positive integer" >&2; exit 2 ;; esac
test "${CURRENT_SIZE}" -gt 0
case "${RECOVAR_MAP_CONVENTION}" in relion|recovar) ;; *) echo "K1_MAP_PPREF_CANDIDATE_CONVENTION must be relion or recovar" >&2; exit 2 ;; esac
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${BUILDER}" | awk '{print $1}')" = "${EXPECTED_BUILDER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
for artifact in "${NATIVE_COMPONENTS}" "${RECOVAR_CAPTURE}" "${NATIVE_MAP}" "${RECOVAR_MAP}"; do test -s "${artifact}"; done
for output in "${NATIVE_PPREF}" "${RECOVAR_PPREF}" "${NATIVE_REPORT}" "${RECOVAR_REPORT}"; do test ! -e "${output}"; done

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${RUN_ROOT}/ppref" "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" "${RUN_ROOT}/logs"

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

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; repo=pathlib.Path('${REPO}').resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + '/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert jax.default_backend() == 'gpu'; assert cuda_backproject.cuda_available(); from recovar.relion_bind import _relion_bind_core"

cd "${REPO}"
"${PYTHON}" -m scripts.build_k1_ppref_capture_from_map \
  --map "${NATIVE_MAP}" --map-convention relion \
  --current-size "${CURRENT_SIZE}" --padding-factor 2 --iteration 2 --rank 1 \
  --output "${NATIVE_PPREF}"
"${PYTHON}" -m scripts.build_k1_ppref_capture_from_map \
  --map "${RECOVAR_MAP}" --map-convention "${RECOVAR_MAP_CONVENTION}" \
  --current-size "${CURRENT_SIZE}" --padding-factor 2 --iteration 2 --rank 1 \
  --output "${RECOVAR_PPREF}"

"${PYTHON}" -m scripts.analyze_k1_exact_ppref_coarse_boundary \
  --ppref "${NATIVE_PPREF}" --recovar "${RECOVAR_CAPTURE}" \
  --native-components "${NATIVE_COMPONENTS}" --physical-image-size 256 \
  --output-json "${NATIVE_REPORT}" \
  > "${RUN_ROOT}/analysis/native_numbered_map_${SLURM_JOB_ID}.stdout"
"${PYTHON}" -m scripts.analyze_k1_exact_ppref_coarse_boundary \
  --ppref "${RECOVAR_PPREF}" --recovar "${RECOVAR_CAPTURE}" \
  --native-components "${NATIVE_COMPONENTS}" --physical-image-size 256 \
  --output-json "${RECOVAR_REPORT}" \
  > "${RUN_ROOT}/analysis/recovar_numbered_map_${SLURM_JOB_ID}.stdout"

sha256sum "${NATIVE_COMPONENTS}" "${RECOVAR_CAPTURE}" "${NATIVE_MAP}" "${RECOVAR_MAP}" \
  "${NATIVE_PPREF}" "${RECOVAR_PPREF}" "${NATIVE_REPORT}" "${RECOVAR_REPORT}" \
  > "${RUN_ROOT}/provenance/map_ppref_boundary_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
