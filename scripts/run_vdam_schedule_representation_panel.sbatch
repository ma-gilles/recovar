#!/usr/bin/env bash
#SBATCH --job-name=vdam-repr-panel
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-repr-panel-%j.out

set -euo pipefail

: "${VDAM_REPRESENTATION_PANEL_ROOT:?set a new disposable output root}"
: "${VDAM_REPRESENTATION_PANEL_CHECKPOINT:?set the exact optimiser checkpoint}"
: "${VDAM_REPRESENTATION_PANEL_DATA_STAR:?set the exact input data STAR}"
: "${VDAM_REPRESENTATION_PANEL_DATA_DIR:?set the particle-stack directory}"
: "${EXPECTED_REPO_HEAD:?pin the tracked RECOVAR source head}"
: "${RECOVAR_CUDA_LIB:?reuse a qualified custom CUDA library}"
: "${RECOVAR_RELION_BIND_BUILD_DIR:?reuse a qualified RELION binding directory}"

ROOT=${VDAM_REPRESENTATION_PANEL_ROOT}
CHECKPOINT=${VDAM_REPRESENTATION_PANEL_CHECKPOINT}
DATA_STAR=${VDAM_REPRESENTATION_PANEL_DATA_STAR}
DATA_DIR=${VDAM_REPRESENTATION_PANEL_DATA_DIR}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CHECKPOINT_ITERATION=${CHECKPOINT_ITERATION:-25}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
RELION_CUDA_TOOLKIT=${RELION_CUDA_TOOLKIT:-/usr/local/cuda-12.6}
CUDA_TOOLKIT=${CUDA_TOOLKIT:-/usr/local/cuda-13.1}
CUSPARSE_LIBRARY=${CUSPARSE_LIBRARY:-${RELION_CUDA_TOOLKIT}/lib64/libcusparse.so.12}
RELION_RUNTIME=${RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
MPI_ROOT=${MPI_ROOT:-/usr/local/openmpi/cuda-12.6/4.1.6/gcc}
CUDA_LIB_SOURCE=${RECOVAR_CUDA_LIB}
PARTICLE_STACK=${VDAM_REPRESENTATION_PANEL_PARTICLE_STACK:-${DATA_DIR}/particles.128.mrcs}
TARGET_GPU_UUID=${TARGET_GPU_UUID:-}
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
PANEL=${ROOT}/panel
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so

on_error() {
  local status=$?
  local message="VDAM representation panel failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
  printf '%s\n' "${message}" >&2
  if [[ -d "${PROVENANCE}" ]]; then
    printf '%s\n' "${message}" > "${PROVENANCE}/failure.txt"
  fi
  exit "${status}"
}
trap on_error ERR

test ! -e "${ROOT}"
test -x "${PIXI_PY}"
test -s "${CHECKPOINT}"
test -s "${DATA_STAR}"
test -d "${DATA_DIR}"
test -s "${PARTICLE_STACK}"
test -s "${CUDA_LIB_SOURCE}"
test -d "${RECOVAR_RELION_BIND_BUILD_DIR}"
test -s "${CUSPARSE_LIBRARY}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"

mapfile -t PROFILE_ENVIRONMENT < <(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" - <<'PY'
from scripts.run_vdam_late_iteration_profile import _all_optimized_q32_environment

for name, value in sorted(_all_optimized_q32_environment().items()):
    print(f"{name}={value}")
PY
)
test "${#PROFILE_ENVIRONMENT[@]}" -gt 0
for assignment in "${PROFILE_ENVIRONMENT[@]}"; do
  export "${assignment}"
done

source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}

mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RUNTIME}/jax_cache" "${RUNTIME}/tmp" \
  "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache"
touch "${ROOT}/SAFE_TO_DELETE" "${RUNTIME}/jax_cache/SAFE_TO_DELETE"
cp --reflink=auto "${CUDA_LIB_SOURCE}" "${CUDA_BINARY}"
test -s "${CUDA_BINARY}"
git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no \
  > "${PROVENANCE}/repo_status.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"
sha256sum "${CHECKPOINT}" "${DATA_STAR}" "${PARTICLE_STACK}" \
  "${CUDA_LIB_SOURCE}" "${CUDA_BINARY}" "${CUSPARSE_LIBRARY}" \
  "${REPO_ROOT}/scripts/run_vdam_schedule_representation_panel.py" \
  "${REPO_ROOT}/scripts/analyze_vdam_schedule_representation_gate.py" \
  "${REPO_ROOT}/scripts/run_vdam_schedule_representation_panel.sbatch" \
  > "${PROVENANCE}/static_inputs.sha256"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_SKIP_CUDA_CONSTRAINTS_CHECK
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export CUDA_LAUNCH_BLOCKING=0
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PATH=${CUDA_TOOLKIT}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${RELION_CUDA_TOOLKIT}/lib64:${RELION_CUDA_TOOLKIT}/targets/x86_64-linux/lib:${CUDA_TOOLKIT}/lib64:${CUDA_TOOLKIT}/targets/x86_64-linux/lib

(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" "${PIXI_PY}" - <<'PY'
import jax
import recovar
import recovar.cuda_backproject as cuda_backproject
from recovar.relion_bind import _relion_bind_core as relion_bind_core

assert recovar.__file__ is not None
assert any(device.platform in {"gpu", "cuda"} for device in jax.devices())
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert callable(getattr(relion_bind_core, "vdam_rnd_unif_sequence", None))
print("VDAM representation panel runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

PANEL_COMMAND=(
  env "LD_PRELOAD=${CUSPARSE_LIBRARY}"
  "${PIXI_PY}" -m scripts.run_vdam_schedule_representation_panel
  --checkpoint-optimiser "${CHECKPOINT}"
  --input-star "${DATA_STAR}"
  --data-dir "${DATA_DIR}"
  --output-root "${PANEL}"
  --checkpoint-iteration "${CHECKPOINT_ITERATION}"
  --nr-iter "${NR_ITER_SCHEDULE}"
  --random-seed 29
  --image-batch-size 500
  --image-size 128
  --exact-local-bucket-radix 4
  --exact-local-physical-order-chunk-size 0
  --stable-fourier-window-shapes
)
printf '%q ' "${PANEL_COMMAND[@]}" > "${PROVENANCE}/panel_command.sh"
printf '\n' >> "${PROVENANCE}/panel_command.sh"
(
  cd "${REPO_ROOT}"
  "${PANEL_COMMAND[@]}"
) > "${PROVENANCE}/panel.stdout" 2> "${PROVENANCE}/panel.stderr"
test -s "${PANEL}/panel.json"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m scripts.analyze_vdam_schedule_representation_gate \
    --panel-root "${PANEL}" \
    --output "${PROVENANCE}/schedule_representation_gate.json"
) > "${PROVENANCE}/gate.stdout" 2> "${PROVENANCE}/gate.stderr"
test -s "${PROVENANCE}/schedule_representation_gate.json"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
touch "${ROOT}/COMPLETED"
