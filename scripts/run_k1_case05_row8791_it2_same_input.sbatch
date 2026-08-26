#!/usr/bin/env bash
#SBATCH --job-name=k1c05-r8791-state
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:45:00

set -euo pipefail
trap 'status=$?; printf "ERROR status=%d line=%d command=%q\n" "${status}" "${LINENO}" "${BASH_COMMAND}" >&2' ERR

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_REPO_DIFF_SHA256:?EXPECTED_REPO_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_SHA256:?EXPECTED_CUDA_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
SOURCE_ROOT=${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET}
CASE_DIR=${CASE_DIR:-${SOURCE_ROOT}/cases/5_very_high_noise_100k_g256_white_noise10_bf80}
NATIVE_STATE_DIR=${RELION_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row08791_fresh_verbose_it2_07d3102_20260826T1525ET/relion}
CUDA_LIB=${CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_cuda_sealed_aca8_20260826T0605ET/libcuda_backproject.so}
TARGET_SOURCE_ROW=${TARGET_SOURCE_ROW:-8791}
TARGET_CURRENT_SIZE=${TARGET_CURRENT_SIZE:-56}
EXPECTED_COARSE_SIGNIFICANT=${EXPECTED_COARSE_SIGNIFICANT:-13}
EXPECTED_FINE_CANDIDATES=${EXPECTED_FINE_CANDIDATES:-416}
EXPECTED_FINE_SUPPORT=${EXPECTED_FINE_SUPPORT:-84}
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
RELION_BIND=${SOURCE_ROOT}/relion_bind_build/shared
LAUNCHER=$(realpath "$0")
OUTPUT=${RUN_ROOT}/output
PASS2=${RUN_ROOT}/pass2
COARSE=${RUN_ROOT}/coarse

for value_name in TARGET_SOURCE_ROW TARGET_CURRENT_SIZE EXPECTED_COARSE_SIGNIFICANT EXPECTED_FINE_CANDIDATES EXPECTED_FINE_SUPPORT; do
    value=${!value_name}
    case "${value}" in ''|*[!0-9]*) echo "${value_name} must be a non-negative integer" >&2; exit 2 ;; esac
done
test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${CASE_DIR}/data/particles.star"
test -s "${NATIVE_STATE_DIR}/run_it001_optimiser.star"
test -s "${NATIVE_STATE_DIR}/run_it001_data.star"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
mkdir -p "${OUTPUT}" "${PASS2}" "${COARSE}"
test -z "$(find "${OUTPUT}" "${PASS2}" "${COARSE}" -mindepth 1 -print -quit)"
mkdir -p "${OUTPUT}/intermediates" "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

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
export CUDA_VISIBLE_DEVICES=0

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_NATIVE_FINE_SCORE_UNITS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${CASE_DIR}/data/particles.star
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
export RECOVAR_SIGNIFICANCE_DUMP_DIR=${COARSE}
export RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=${TARGET_SOURCE_ROW}
export RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE=${TARGET_CURRENT_SIZE}
export RECOVAR_SIGNIFICANCE_DUMP_ITERATION=2
export RECOVAR_SIGNIFICANCE_DUMP_PASSIVE_CACHE=1
export RECOVAR_PASS2_DUMP_DIR=${PASS2}
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${TARGET_SOURCE_ROW}
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=${TARGET_CURRENT_SIZE}
export RECOVAR_PASS2_DUMP_ITERATION=1
export RECOVAR_PASS2_DUMP_CLASS=1
export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
export RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${RUN_ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" "${NATIVE_STATE_DIR}/run_it001_optimiser.star" "${NATIVE_STATE_DIR}/run_it001_data.star" > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert '.pixi/envs/default/' in str(Path(jax.__file__).resolve())
assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
PY

command=("${PYTHON}" -m scripts.run_multi_iter_parity
    --relion_dir "${NATIVE_STATE_DIR}"
    --data_star "${CASE_DIR}/data/particles.star"
    --iter 1 --max_iter 1
    --continuous-relion-noise-state
    --skip_final_iteration
    --image-fourier-backend relion_cuda
    --image_batch_size 187
    --rotation_block_size 8192
    --output_dir "${OUTPUT}"
    --save_intermediates_dir "${OUTPUT}/intermediates")
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
set +e
"${command[@]}" > "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1
command_status=$?
set -e
wall_s=$(( $(date +%s) - start_epoch ))

printf -v source_padded '%06d' "${TARGET_SOURCE_ROW}"
printf -v size_padded '%03d' "${TARGET_CURRENT_SIZE}"
capture=${PASS2}/pass2_orig${source_padded}_cs${size_padded}.npz
coarse_capture=${COARSE}/significance_orig${source_padded}_it002_cs${size_padded}.npz
test "${command_status}" -ne 0
test -s "${capture}" -a -s "${coarse_capture}"
! rg -q 'RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.log"
rg -q 'Pass2DumpComplete: requested RECOVAR pass-2 dump target set was written' "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.log"

"${PYTHON}" - "${capture}" "${coarse_capture}" "${RUN_ROOT}/analysis/CAPTURE_VALIDATION.json" \
    "${TARGET_SOURCE_ROW}" "${TARGET_CURRENT_SIZE}" "${EXPECTED_COARSE_SIGNIFICANT}" \
    "${EXPECTED_FINE_CANDIDATES}" "${EXPECTED_FINE_SUPPORT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

fine_path, coarse_path, output_path = map(Path, sys.argv[1:4])
source, size, coarse_count, fine_count, support_count = map(int, sys.argv[4:])
with np.load(fine_path, allow_pickle=False) as fine, np.load(coarse_path, allow_pickle=False) as coarse:
    observed = {
        'source_row_zero_based': int(fine['original_index']),
        'current_size': int(fine['current_size']),
        'coarse_significant_count': int(np.count_nonzero(coarse['significant_mask'])),
        'fine_candidate_count': int(np.count_nonzero(fine['candidate_mask'])),
        'fine_support_count': int(np.count_nonzero(fine['reconstruction_mask'])),
        'winner': [int(v) for v in np.unravel_index(int(np.argmax(fine['probs'])), fine['probs'].shape)],
    }
assert observed['source_row_zero_based'] == source
assert observed['current_size'] == size
assert observed['coarse_significant_count'] == coarse_count
assert observed['fine_candidate_count'] == fine_count
assert observed['fine_support_count'] == support_count
output_path.write_text(json.dumps({'status': 'complete', **observed}, indent=2, sort_keys=True) + '\n')
PY

sha256sum "${capture}" "${coarse_capture}" "${RUN_ROOT}/analysis/CAPTURE_VALIDATION.json" > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":2,"same_input":"RELION iteration-1 state"}\n' "${SLURM_JOB_ID}" "${wall_s}" > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
