#!/usr/bin/env bash
#SBATCH --job-name=k1-fine-cont
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${CONTROL_DIR:?CONTROL_DIR is required}"
: "${CONTINUE_STAR:?CONTINUE_STAR is required}"
: "${TARGET_STACK_ONE_BASED:?TARGET_STACK_ONE_BASED is required}"
: "${TARGET_ITERATION:?TARGET_ITERATION is required}"
: "${PHYSICAL_ORDER_SEED:?PHYSICAL_ORDER_SEED is required}"
: "${SAMPLING_PERTURBATION_OVERRIDE:?SAMPLING_PERTURBATION_OVERRIDE is required}"
: "${EXPECTED_CONTINUE_SHA256:?EXPECTED_CONTINUE_SHA256 is required}"
: "${EXPECTED_PARTICLES_SHA256:?EXPECTED_PARTICLES_SHA256 is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/run_k1_native_fine_continuation.sbatch
AUDIT_REPO=${AUDIT_REPO:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout}
RELION_SOURCE=${RELION_SOURCE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/source}
RELION_BINARY=${RELION_BINARY:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/build/bin/relion_refine_mpi}
RELION_RUNTIME=${RELION_RUNTIME:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717}
EXPECTED_RELION_BINARY_SHA256=${EXPECTED_RELION_BINARY_SHA256:-1c3f67f52a7663f29c3cc0c8aacc995eba27517961e24ee4a1be01b6413c0b1d}
EXPECTED_RELION_SOURCE_HEAD=${EXPECTED_RELION_SOURCE_HEAD:-f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3}
EXPECTED_RELION_SOURCE_TREE=${EXPECTED_RELION_SOURCE_TREE:-3111173edb150f98bc2446efc54229ff736b40a1}
EXPECTED_RELION_SOURCE_DIFF_SHA256=${EXPECTED_RELION_SOURCE_DIFF_SHA256:-15e9cceb38815cc85ed946e1bb5ee8a9b42f89a530cf8d5af2d76e83228be3f9}
EXPECTED_AUDIT_REPO_HEAD=${EXPECTED_AUDIT_REPO_HEAD:-187a9f60cdea19363d49d46099b052f23288f05a}
FINE_OPERAND_ROTATION_LOCAL=${FINE_OPERAND_ROTATION_LOCAL:-}
FINE_OPERAND_TRANSLATIONS=${FINE_OPERAND_TRANSLATIONS:-}
FINE_OPERAND_TRANSLATIONS=${FINE_OPERAND_TRANSLATIONS//:/,}

CAPTURE=${RUN_ROOT}/capture
RELION_OUT=${RUN_ROOT}/relion

test "${TARGET_STACK_ONE_BASED}" -gt 0
test "${TARGET_ITERATION}" -gt 0
if [[ -n "${FINE_OPERAND_ROTATION_LOCAL}" || -n "${FINE_OPERAND_TRANSLATIONS}" ]]; then
  [[ "${FINE_OPERAND_ROTATION_LOCAL}" =~ ^[0-9]+$ ]]
  IFS=',' read -r -a fine_operand_translation_values <<< "${FINE_OPERAND_TRANSLATIONS}"
  test "${#fine_operand_translation_values[@]}" -gt 0
  for translation in "${fine_operand_translation_values[@]}"; do
    [[ "${translation}" =~ ^[0-9]+$ ]]
  done
  test "$(printf '%s\n' "${fine_operand_translation_values[@]}" | sort -u | wc -l)" \
    -eq "${#fine_operand_translation_values[@]}"
fi
test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${PYTHON}"
test -s "${DATA_DIR}/particles.star"
test -s "${CONTINUE_STAR}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${CONTINUE_STAR}" | awk '{print $1}')" = "${EXPECTED_CONTINUE_SHA256}"
test "$(sha256sum "${DATA_DIR}/particles.star" | awk '{print $1}')" = "${EXPECTED_PARTICLES_SHA256}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_SOURCE_HEAD}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD^{tree})" = "${EXPECTED_RELION_SOURCE_TREE}"
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = "${EXPECTED_AUDIT_REPO_HEAD}"
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
mkdir -p "${CAPTURE}" "${RELION_OUT}"
test -z "$(find "${CAPTURE}" "${RELION_OUT}" -mindepth 1 -print -quit)"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" "${RUN_ROOT}/logs"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
  "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
  case "${variable_name}" in
    RECOVAR_*|RELION_BPRE_*|RELION_FINE_*|RELION_PREITER_*|RELION_SAMPLING_PERTURBATION_*)
      unset "${variable_name}"
      ;;
  esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

export RELION_BPRE_CAPTURE_DIR=${CAPTURE}
export RELION_BPRE_CAPTURE_SCHEMA=2
export RELION_BPRE_CAPTURE_ITER=${TARGET_ITERATION}
export RELION_BPRE_CAPTURE_CLASS=1
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=1
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=2000000000
export RELION_BPRE_CAPTURE_STACKS=${TARGET_STACK_ONE_BASED}
export RELION_BPRE_CAPTURE_GEOMETRY_ONLY=1
export RELION_PREITER_PARTICLE_ORDER_SEED=${PHYSICAL_ORDER_SEED}
export RELION_PREITER_PARTICLE_ORDER_ITER=${TARGET_ITERATION}
export RELION_SAMPLING_PERTURBATION_OVERRIDE=${SAMPLING_PERTURBATION_OVERRIDE}
export RELION_SAMPLING_PERTURBATION_OVERRIDE_ITER=${TARGET_ITERATION}
if [[ -n "${FINE_OPERAND_ROTATION_LOCAL}" ]]; then
  export RELION_FINE_OPERAND_CAPTURE=1
  export RELION_FINE_OPERAND_CAPTURE_ROTATION_LOCAL=${FINE_OPERAND_ROTATION_LOCAL}
  export RELION_FINE_OPERAND_CAPTURE_TRANSLATIONS=${FINE_OPERAND_TRANSLATIONS}
  export RELION_FINE_OPERAND_CAPTURE_EXPECTED_CANDIDATES=${#fine_operand_translation_values[@]}
  export RELION_FINE_OPERAND_CAPTURE_MAX_BYTES=2000000000
fi

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
  --continue "${CONTINUE_STAR}" --o "${RELION_OUT}/run"
  --auto_iter_max "${TARGET_ITERATION}" --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${CONTINUE_STAR}" \
  "${REPO}/scripts/analyze_k1_fine_score_stages.py" \
  "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start_epoch=$(date +%s)
(
  cd "${DATA_DIR}"
  "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

fine_score=$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-score-v1.bin')
factor=$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.bpre-v2.bin')
test -n "${fine_score}" -a -n "${factor}"
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-score-v1.bin' | wc -l)" -eq 1
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.bpre-v2.bin' | wc -l)" -eq 1
if [[ -n "${FINE_OPERAND_ROTATION_LOCAL}" ]]; then
  test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-operand-v1.bin' | wc -l)" -eq 1
else
  test -z "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-operand-v1.bin' -print -quit)"
fi
test -s "$(printf '%s/run_it%03d_data.star' "${RELION_OUT}" "${TARGET_ITERATION}")"
"${PYTHON}" - "${fine_score}" "${factor}" "${TARGET_STACK_ONE_BASED}" "${TARGET_ITERATION}" <<'PY'
import sys
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture

fine = load_fine_score_capture(sys.argv[1])
factor = load_factor_capture(sys.argv[2])
expected_stack = int(sys.argv[3])
expected_iteration = int(sys.argv[4])
assert fine.stack_index == factor.stack_index == expected_stack
assert fine.header[4] == factor.header[9] == expected_iteration
assert factor.geometry_only
assert (fine.candidates["flags"] & ACTIVE).any()
PY
rg -q "\[RELION_PREITER_PARTICLE_ORDER\] iteration ${TARGET_ITERATION} effective_seed ${PHYSICAL_ORDER_SEED} particles 100000" \
  "${RELION_OUT}/runner.stderr"
rg -q "\[RELION_SAMPLING_PERTURBATION_OVERRIDE\] iter ${TARGET_ITERATION} requested" \
  "${RELION_OUT}/runner.stdout"
! rg -qi 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${RELION_OUT}/runner.stderr"

"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  --control-dir "${CONTROL_DIR}" --candidate-dir "${RELION_OUT}" --iterations "${TARGET_ITERATION}" \
  --map-fsc-auc-threshold 0.999999 \
  --output-json "${RUN_ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
  --output-markdown "${RUN_ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${RUN_ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

find "${CAPTURE}" "${RELION_OUT}" "${RUN_ROOT}/analysis" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":%d,"stack_index_one_based":%d,"continued_from":"%s","geometry_only":true}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" "${TARGET_ITERATION}" "${TARGET_STACK_ONE_BASED}" "${CONTINUE_STAR}" \
  > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
