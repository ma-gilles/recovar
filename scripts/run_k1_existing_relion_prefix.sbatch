#!/usr/bin/env bash
#SBATCH --job-name=k1-relion-prefix
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=04:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CASE_SOURCE:?CASE_SOURCE is required}"
: "${CASE_NAME:?CASE_NAME is required}"
: "${SEED:?SEED is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_norm_matrix_fb3995ca_20260821}
PYTHON=${PYTHON:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_normspectrum_fb3995ca_20260821T2025ET/venv/bin/python}
CUDA_LIB=${CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_normspectrum_fb3995ca_20260821T2025ET/cuda/libcuda_backproject.so}
RELION_BIND=${RELION_BIND:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_normspectrum_fb3995ca_20260821T2025ET/relion_bind_build/shared}
EXPECTED_HEAD=${EXPECTED_HEAD:-fb3995ca08192fe31956fc853a305eb94f1e0739}
MAX_ITER=${MAX_ITER:-2}

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
DATA=${CASE_SOURCE}/data
RELION=${CASE_SOURCE}/relion_ref
OUTPUT=${ROOT}/output
ANALYSIS=${ROOT}/analysis

test -f "${ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME}/SAFE_TO_DELETE"
test -x "${PYTHON}"
test -f "${CUDA_LIB}"
test -s "${DATA}/particles.star"
test -s "${DATA}/reference_gt.mrc"
test -s "${RELION}/run_it000_data.star"
test -s "${RELION}/run_it000_optimiser.star"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test -z "$(git -C "${REPO}" status --porcelain=v1)"

mkdir -p "${ROOT}/logs" "${ROOT}/provenance" "${OUTPUT}/intermediates" "${ANALYSIS}"
mkdir -p "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache" \
  "${RUNTIME}/pycache" "${RUNTIME}/jax_cache" "${RUNTIME}/cuda_cache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
  case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export RECOVAR_CUDA_CACHE_DIR=${RUNTIME}/cuda_cache

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
PYTHON_REAL=$(readlink -f "${PYTHON}")
PIXI_ENV=$(cd "$(dirname "${PYTHON_REAL}")/.." && pwd)
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}

# Match the fixed terminal matrix. The production source-faithful spectrum
# normalization and exact BPref defaults are intentionally not overridden.
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
unset RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM
unset RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
git rev-parse HEAD > "${ROOT}/provenance/repo_head_${SLURM_JOB_ID}.txt"
git status --short > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
sha256sum "${CUDA_LIB}" "${DATA}/particles.star" \
  "${RELION}/run_it000_data.star" "${RELION}/run_it000_optimiser.star" \
  > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
  > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"

"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices())
PY

command=(
  "${PYTHON}" -m scripts.run_full_refinement
  --data_dir "${DATA}" --output "${OUTPUT}" --max_iter "${MAX_ITER}" --skip_final_iteration
  --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
  --adaptive_oversampling 1 --init_resolution 30.0
  --image_batch_size 187 --rotation_block_size 8192
  --seed "${SEED}" --perturb_seed "${SEED}" --particle_diameter_ang 200
  --tau2_fudge 1.0 --max_significants -1 --firstiter_cc --apply-initial-lowpass
  --image-fourier-backend relion_cuda
  --save_intermediates_dir "${OUTPUT}/intermediates" --save_intermediates_skip_unregularized
  --relion_half_sets "${RELION}/run_it000_data.star"
  --relion_optimiser "${RELION}/run_it000_optimiser.star"
  --relion_init_dir "${RELION}"
)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
"${command[@]}" > "${ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1
wall_s=$(( $(date +%s) - start_epoch ))

! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
test -s "${OUTPUT}/refinement_results.npz"

"${PYTHON}" -m scripts.audit_k1_fsc_trajectory \
  --case-root "${CASE_SOURCE}" \
  --recovar-dir "${OUTPUT}" \
  --relion-dir "${RELION}" \
  --gt-volume "${DATA}/reference_gt.mrc" \
  --numbered-only --allow-incomplete \
  --output-json "${ANALYSIS}/prefix_fsc.json" \
  --output-markdown "${ANALYSIS}/prefix_fsc.md" \
  --output-shellwise-npz "${ANALYSIS}/prefix_fsc_shellwise.npz"

"${PYTHON}" -m scripts.audit_k1_intermediate_trajectory \
  --case-root "${CASE_SOURCE}" \
  --recovar-dir "${OUTPUT}" \
  --relion-dir "${RELION}" \
  --output-json "${ANALYSIS}/prefix_intermediates.json" \
  --output-markdown "${ANALYSIS}/prefix_intermediates.md"

sha256sum "${OUTPUT}/refinement_results.npz" "${ANALYSIS}/prefix_fsc.json" \
  "${ANALYSIS}/prefix_fsc_shellwise.npz" "${ANALYSIS}/prefix_intermediates.json" \
  > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"case":"%s","job_id":"%s","wall_s":%d,"max_iter":%d,"mode":"matrix_faithful_existing_relion_prefix"}\n' \
  "${CASE_NAME}" "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" \
  > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
