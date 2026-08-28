#!/usr/bin/env bash
#SBATCH --job-name=k1-clean-final-boundary
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${REPO:?REPO is required}"
: "${CASE_ROOT:?CASE_ROOT is required}"
: "${GRID_SIZE:?GRID_SIZE is required}"
: "${HEALPIX_ORDER:?HEALPIX_ORDER is required}"
: "${REFINEMENT_SEED:?REFINEMENT_SEED is required}"
: "${CUDA_LIB:?CUDA_LIB is required}"
: "${BIND_BUILD_DIR:?BIND_BUILD_DIR is required}"
: "${RELION_SRC_DIR:?RELION_SRC_DIR is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"
: "${EXPECTED_INPUTS_SHA256:?EXPECTED_INPUTS_SHA256 is required}"

DATA=${CASE_ROOT}/data
RELION=${CASE_ROOT}/relion_ref
LAUNCHER=${REPO}/scripts/run_k1_clean_final_boundary.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
OUTPUT=${RUN_ROOT}/output
ACCUM=${RUN_ROOT}/final_bpref_accum
PROVENANCE=${RUN_ROOT}/provenance
RUN_CUDA_DIR=${RUN_ROOT}/cuda
RUN_CUDA_LIB=${RUN_CUDA_DIR}/libcuda_backproject.so
RELION_SOURCE=${RELION_SRC_DIR}

input_digest() {
    sha256sum \
        "${DATA}/particles.star" \
        "${DATA}/particles.${GRID_SIZE}.mrcs" \
        "${DATA}/reference_init_relion.mrc" \
        | sha256sum | awk '{print $1}'
}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${RELION}/run_it000_data.star"
test -s "${RELION}/run_it000_optimiser.star"
test -s "${DATA}/particles.star"
test -s "${DATA}/particles.${GRID_SIZE}.mrcs"
test -s "${DATA}/reference_init_relion.mrc"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em scripts/run_full_refinement.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
test "$(input_digest)" = "${EXPECTED_INPUTS_SHA256}"
test -z "$(
    find "${RUN_ROOT}" -mindepth 1 \
        ! -name SAFE_TO_DELETE \
        ! -name 'slurm-*.out' \
        -print -quit
)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${OUTPUT}" "${OUTPUT}/intermediates" "${OUTPUT}/timing" \
    "${ACCUM}" "${PROVENANCE}" "${RUN_ROOT}/logs" "${RUN_CUDA_DIR}"
test ! -e "${RUN_CUDA_LIB}"
install -m 0555 "${CUDA_LIB}" "${RUN_CUDA_LIB}"

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_ROOT=/usr/local/cuda-12.8
export PATH=${CUDA_ROOT}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_ROOT}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign exactly one visible H100}"

export RECOVAR_CUDA_LIB=${RUN_CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${BIND_BUILD_DIR}
export RELION_SRC_DIR=${RELION_SOURCE}
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
export RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE=1
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR=${ACCUM}
export RECOVAR_SAVE_INTERMEDIATES_DIR=1
export RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED=1
unset RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER RECOVAR_FINAL_ALL_DATA_GRID_CORRECT

cd "${REPO}"
git status --short --branch > "${PROVENANCE}/git_status_${SLURM_JOB_ID}.txt"
git diff --binary -- recovar/em scripts/run_full_refinement.py \
    > "${PROVENANCE}/source_diff_${SLURM_JOB_ID}.patch"
env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" "${RUN_CUDA_LIB}" \
    "${DATA}/particles.star" "${DATA}/particles.${GRID_SIZE}.mrcs" \
    "${DATA}/reference_init_relion.mrc" "${RELION}/run_it000_data.star" \
    "${RELION}/run_it000_optimiser.star" \
    > "${PROVENANCE}/inputs_${SLURM_JOB_ID}.sha256"

"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert ".pixi/envs/default/" in str(Path(jax.__file__).resolve())
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices())
PY

command=("${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter 999
    --healpix_order "${HEALPIX_ORDER}" --auto_local_healpix_order 4
    --offset_range 3.0 --offset_step 1.0 --adaptive_oversampling 1
    --init_resolution 30.0 --image_batch_size 187 --rotation_block_size 8192
    --seed "${REFINEMENT_SEED}" --perturb_seed "${REFINEMENT_SEED}"
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json"
    --timing_dir "${OUTPUT}/timing"
    --relion_half_sets "${RELION}/run_it000_data.star"
    --relion_optimiser "${RELION}/run_it000_optimiser.star"
    --relion_init_dir "${RELION}"
    --save_intermediates_dir "${OUTPUT}/intermediates"
    --save_intermediates_skip_unregularized)
printf '%q ' "${command[@]}" > "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
start_seconds=$(date +%s)
"${command[@]}" > "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1
end_seconds=$(date +%s)
printf '{"elapsed_seconds":%d,"job_id":"%s"}\n' \
    "$((end_seconds - start_seconds))" "${SLURM_JOB_ID}" \
    > "${PROVENANCE}/walltime_${SLURM_JOB_ID}.json"

test -s "${OUTPUT}/refinement_results.npz"
test -s "${OUTPUT}/final_merged.mrc"
test -s "${OUTPUT}/intermediates/manifest_final_half0.npz"
test -s "${OUTPUT}/intermediates/manifest_final_half1.npz"
test -s "${ACCUM}/recovar_final_bpref_accum.npz"
! rg -q 'RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent|Traceback' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.log"
OUTPUT_RESULTS=${OUTPUT}/refinement_results.npz "${PYTHON}" - <<'PY'
import os
import numpy as np

with np.load(os.environ["OUTPUT_RESULTS"], allow_pickle=True) as results:
    assert int(np.asarray(results["git_dirty_count"]).item()) == 0
    assert bool(np.asarray(results["final_all_data_ran"]).item())
PY
find "${OUTPUT}" "${ACCUM}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
