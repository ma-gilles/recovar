#!/usr/bin/env bash
#SBATCH --job-name=k1c10-fresh-it1
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
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
FIXED_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${FIXED_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE}/data
RELION=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=${CUDA_LIB_OVERRIDE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_exact_it15_final_20260823T0230ET/cuda/libcuda_backproject.so}
RELION_BIND=${FIXED_ROOT}/relion_bind_build/shared
LAUNCHER=${REPO}/scripts/run_k1_case10_fresh_it1_boundary.sbatch
OUTPUT=${ROOT}/output
TARGET_ROWS=${TARGET_ROWS:-10692,25973,47531,83842,89358}
TARGET_RAW_ROW=${TARGET_RAW_ROW:-10692}
SCORE_CURRENT_SIZE=${SCORE_CURRENT_SIZE:-58}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -z "$(find "${OUTPUT}" -mindepth 1 ! -name SAFE_TO_DELETE -print -quit)"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu recovar/cuda_backproject.py scripts/run_full_refinement.py scripts/run_multi_iter_parity.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${OUTPUT}" "${OUTPUT}/intermediates" "${ROOT}/logs" "${ROOT}/provenance" \
    "${ROOT}/pass2" "${ROOT}/projector" "${ROOT}/contributions"

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
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign exactly one visible GPU}"

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
export RECOVAR_RELION_PROJECTOR_DUMP_DIR=${ROOT}/projector
export RECOVAR_PASS2_DUMP_DIR=${ROOT}/pass2
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${TARGET_RAW_ROW}
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=${SCORE_CURRENT_SIZE}
export RECOVAR_PASS2_DUMP_ITERATION=1
export RECOVAR_PASS2_DUMP_CLASS=1
export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
export RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR=${ROOT}/contributions
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION=1
export RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE=${SCORE_CURRENT_SIZE}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES=${TARGET_ROWS}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID=case10-fresh-it1-boundary
export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY=${ROOT}/provenance/image_names.npy
export RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256=6abfc7829307ed63cc1b8b24ba99ec1afca25e5467876ef056034a9dd8779585
export RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
git status --short --branch > "${ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu recovar/cuda_backproject.py scripts/run_full_refinement.py scripts/run_multi_iter_parity.py \
    > "${ROOT}/provenance/source_diff_${SLURM_JOB_ID}.patch"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"

"${PYTHON}" - <<'PY'
from pathlib import Path
import os
import numpy as np
import starfile

star_path = Path(os.environ["RECOVAR_K1_RELION_EXACT_CTF_STAR"])
tables = starfile.read(str(star_path))
particles = tables["particles"] if isinstance(tables, dict) else tables
identities = []
for value in particles["rlnImageName"].astype(str):
    stack_index, stack_path = value.split("@", 1)
    resolved = Path(stack_path)
    if not resolved.is_absolute():
        resolved = (star_path.parent / resolved).resolve()
    identities.append(f"{int(stack_index)}@{resolved}")
np.save(os.environ["RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY"], np.asarray(identities, dtype=str))
PY

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
print(jax.devices())
PY

command=("${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter 1 --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed 1710 --perturb_seed 1710
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json"
    --timing_dir "${OUTPUT}/timing"
    --save_intermediates_dir "${OUTPUT}/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${RELION}/run_it000_data.star"
    --relion_optimiser "${RELION}/run_it000_optimiser.star"
    --relion_init_dir "${RELION}")
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
"${command[@]}" > "${ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1

test -n "$(find "${ROOT}/projector" -maxdepth 1 -type f -name 'iter*_half0_relion_projector_half.npz' -print -quit)"
printf -v target_raw_tag '%06d' "${TARGET_RAW_ROW}"
printf -v score_current_tag '%03d' "${SCORE_CURRENT_SIZE}"
test -s "${ROOT}/pass2/pass2_orig${target_raw_tag}_cs${score_current_tag}.npz"
test -s "${OUTPUT}/intermediates/it000_ha_half1.npy"
! rg -q 'RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent|Traceback' \
    "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
find "${ROOT}/projector" "${ROOT}/pass2" "${ROOT}/contributions" "${OUTPUT}/intermediates" \
    -type f -print0 | sort -z | xargs -0 sha256sum > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
