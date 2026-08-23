#!/usr/bin/env bash
#SBATCH --job-name=k1c04-rec-coarse
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_GIT_DIFF_SHA256:?EXPECTED_GIT_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_SHA256:?EXPECTED_CUDA_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
RELION=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
RELION_BIND=${SOURCE_ROOT}/relion_bind_build/shared
CUDA_LIB=${REPO}/recovar/cuda/libcuda_backproject.so
LAUNCHER=${REPO}/scripts/run_k1_case04_recovar_coarse_boundary.sbatch
OUTPUT=${ROOT}/output
COARSE=${ROOT}/coarse

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_GIT_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 0a90714711cdf0a1820d46fa82460d828b87e1ecbd2e49de645170863546ac98
test -z "$(find "${OUTPUT}" "${COARSE}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
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
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
export CUDA_VISIBLE_DEVICES=0

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_SIGNIFICANCE_DUMP_DIR=${COARSE}
export RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=68695
export RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE=100
export RECOVAR_SIGNIFICANCE_DUMP_ITERATION=2
export RECOVAR_SIGNIFICANCE_DUMP_TARGET_HALF=2
export RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

cd "${REPO}"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; repo=pathlib.Path.cwd().resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(repo); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper(); assert cuda_backproject.cuda_available(); print(jax.devices())"

command=("${PYTHON}" -m scripts.run_multi_iter_parity
  --relion_dir "${RELION}"
  --data_star "${DATA}/particles.star"
  --iter 1 --max_iter 1
  --continuous-relion-noise-state
  --skip_final_iteration
  --image_batch_size 2
  --rotation_block_size 8192
  --image-fourier-backend relion_cuda
  --keep_stack_indices 1,68695
  --output_dir "${OUTPUT}"
  --save_intermediates_dir "${OUTPUT}/intermediates")
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
set +e
"${command[@]}" > "${ROOT}/run.log" 2>&1
command_status=$?
set -e
wall_s=$(( $(date +%s) - start_epoch ))

dump=${COARSE}/significance_orig068695_it002_cs100.npz
test -s "${dump}"
test "${command_status}" -ne 0
rg -q 'requested RECOVAR coarse-significance target was written' "${ROOT}/run.log"
! rg -q 'RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${ROOT}/run.log"
"${PYTHON}" - "${dump}" "${ROOT}/analysis/RECOVAR_COARSE_CAPTURE_VALIDATION.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
output = Path(sys.argv[2])
with np.load(path, allow_pickle=False) as capture:
    assert int(capture["original_index"]) == 68695
    assert int(capture["debug_iteration"]) == 2
    assert int(capture["current_size"]) == 100
    assert int(capture["n_rot"]) == 36864
    assert int(capture["n_trans"]) == 29
    significant = np.asarray(capture["significant_mask"], dtype=bool)
    output.write_text(json.dumps({
        "status": "complete",
        "path": str(path.resolve()),
        "diagnostic_dataset_index": int(capture["original_index"]),
        "diagnostic_local_iteration": int(capture["debug_iteration"]),
        "relion_consumer_iteration": 2,
        "current_size": int(capture["current_size"]),
        "rotation_count": int(capture["n_rot"]),
        "translation_count": int(capture["n_trans"]),
        "significant_count": int(np.count_nonzero(significant)),
    }, indent=2, sort_keys=True) + "\n")
PY
sha256sum "${dump}" "${ROOT}/analysis/RECOVAR_COARSE_CAPTURE_VALIDATION.json" \
  > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"stack_index_one_based":68695,"relion_consumer_iteration":2}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
