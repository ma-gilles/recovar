#!/usr/bin/env bash
#SBATCH --job-name=k1-fresh-fine
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=03:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${CONTROL_DIR:?CONTROL_DIR is required}"
: "${TARGET_STACK_ONE_BASED:?TARGET_STACK_ONE_BASED is required}"
: "${RANDOM_SEED:?RANDOM_SEED is required}"
: "${EXPECTED_PARTICLES_SHA256:?EXPECTED_PARTICLES_SHA256 is required}"
: "${EXPECTED_REFERENCE_SHA256:?EXPECTED_REFERENCE_SHA256 is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/run_k1_native_fresh_fine_capture.sbatch
AUDIT_REPO=${AUDIT_REPO:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout}
RELION_SOURCE=${RELION_SOURCE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/source}
RELION_BINARY=${RELION_BINARY:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/build/bin/relion_refine_mpi}
RELION_RUNTIME=${RELION_RUNTIME:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717}
EXPECTED_RELION_BINARY_SHA256=${EXPECTED_RELION_BINARY_SHA256:-1c3f67f52a7663f29c3cc0c8aacc995eba27517961e24ee4a1be01b6413c0b1d}
EXPECTED_RELION_SOURCE_HEAD=${EXPECTED_RELION_SOURCE_HEAD:-f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3}
EXPECTED_RELION_SOURCE_TREE=${EXPECTED_RELION_SOURCE_TREE:-3111173edb150f98bc2446efc54229ff736b40a1}
EXPECTED_RELION_SOURCE_DIFF_SHA256=${EXPECTED_RELION_SOURCE_DIFF_SHA256:-15e9cceb38815cc85ed946e1bb5ee8a9b42f89a530cf8d5af2d76e83228be3f9}
EXPECTED_AUDIT_REPO_HEAD=${EXPECTED_AUDIT_REPO_HEAD:-187a9f60cdea19363d49d46099b052f23288f05a}
TARGET_ITERATION=${TARGET_ITERATION:-2}
TARGET_STACKS_ONE_BASED=${TARGET_STACKS_ONE_BASED:-${TARGET_STACK_ONE_BASED}}
TARGET_STACKS_ONE_BASED=${TARGET_STACKS_ONE_BASED//:/,}
PARTICLE_DIAMETER=${PARTICLE_DIAMETER:-200}
INITIAL_HIGH=${INITIAL_HIGH:-30}
HEALPIX_ORDER=${HEALPIX_ORDER:-3}
OFFSET_RANGE=${OFFSET_RANGE:-3.0}
OFFSET_STEP=${OFFSET_STEP:-1.0}
OVERSAMPLING=${OVERSAMPLING:-1}
FINE_OPERAND_ROTATION_LOCAL=${FINE_OPERAND_ROTATION_LOCAL:-}
FINE_OPERAND_TRANSLATIONS=${FINE_OPERAND_TRANSLATIONS:-}
FINE_OPERAND_TRANSLATIONS=${FINE_OPERAND_TRANSLATIONS//:/,}

CAPTURE=${RUN_ROOT}/capture
RELION_OUT=${RUN_ROOT}/relion

IFS=',' read -r -a target_stack_values <<< "${TARGET_STACKS_ONE_BASED}"
for target_stack in "${target_stack_values[@]}"; do
  if [[ ! "${target_stack}" =~ ^[0-9]+$ ]] || (( target_stack < 1 )); then
    echo "TARGET_STACKS_ONE_BASED must contain unique positive integers" >&2
    exit 2
  fi
done
TARGET_COUNT=${#target_stack_values[@]}
test "${TARGET_COUNT}" -gt 0
test "$(printf '%s\n' "${target_stack_values[@]}" | sort -u | wc -l)" -eq "${TARGET_COUNT}"
[[ "${TARGET_ITERATION}" =~ ^[1-9][0-9]*$ ]]
if [[ -n "${FINE_OPERAND_ROTATION_LOCAL}" || -n "${FINE_OPERAND_TRANSLATIONS}" ]]; then
  test "${TARGET_COUNT}" -eq 1
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
test -s "${DATA_DIR}/reference_init_relion.mrc"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${DATA_DIR}/particles.star" | awk '{print $1}')" = "${EXPECTED_PARTICLES_SHA256}"
test "$(sha256sum "${DATA_DIR}/reference_init_relion.mrc" | awk '{print $1}')" = "${EXPECTED_REFERENCE_SHA256}"
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
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=${TARGET_COUNT}
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=${TARGET_COUNT}
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=2000000000
export RELION_BPRE_CAPTURE_STACKS=${TARGET_STACKS_ONE_BASED}
export RELION_BPRE_CAPTURE_GEOMETRY_ONLY=1
if [[ -n "${FINE_OPERAND_ROTATION_LOCAL}" ]]; then
  export RELION_FINE_OPERAND_CAPTURE=1
  export RELION_FINE_OPERAND_CAPTURE_ROTATION_LOCAL=${FINE_OPERAND_ROTATION_LOCAL}
  export RELION_FINE_OPERAND_CAPTURE_TRANSLATIONS=${FINE_OPERAND_TRANSLATIONS}
  export RELION_FINE_OPERAND_CAPTURE_EXPECTED_CANDIDATES=${#fine_operand_translation_values[@]}
  export RELION_FINE_OPERAND_CAPTURE_MAX_BYTES=2000000000
fi

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
  --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
  --auto_refine --split_random_halves --particle_diameter "${PARTICLE_DIAMETER}"
  --ini_high "${INITIAL_HIGH}" --firstiter_cc --ctf --flatten_solvent --zero_mask
  --low_resol_join_halves 40 --norm --scale --healpix_order "${HEALPIX_ORDER}"
  --offset_range "${OFFSET_RANGE}" --offset_step "${OFFSET_STEP}"
  --oversampling "${OVERSAMPLING}" --pad 2 --random_seed "${RANDOM_SEED}"
  --auto_iter_max "${TARGET_ITERATION}" --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${DATA_DIR}/particles.star" \
  "${DATA_DIR}/reference_init_relion.mrc" \
  "${REPO}/scripts/analyze_k1_fine_score_stages.py" \
  "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start_epoch=$(date +%s)
(
  cd "${DATA_DIR}"
  "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-score-v1.bin' | wc -l)" -eq "${TARGET_COUNT}"
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.bpre-v2.bin' | wc -l)" -eq "${TARGET_COUNT}"
if [[ -n "${FINE_OPERAND_ROTATION_LOCAL}" ]]; then
  test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-operand-v1.bin' | wc -l)" -eq 1
else
  test -z "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-operand-v1.bin' -print -quit)"
fi
printf -v target_iteration_label '%03d' "${TARGET_ITERATION}"
test -s "${RELION_OUT}/run_it${target_iteration_label}_data.star"
"${PYTHON}" - "${CAPTURE}" "${TARGET_STACKS_ONE_BASED}" "${TARGET_ITERATION}" <<'PY'
import sys
from pathlib import Path

from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture

capture = Path(sys.argv[1])
expected_stacks = {int(value) for value in sys.argv[2].split(",")}
expected_iteration = int(sys.argv[3])
fine = [load_fine_score_capture(path) for path in sorted(capture.glob("*.fine-score-v1.bin"))]
factor = [load_factor_capture(path) for path in sorted(capture.glob("*.bpre-v2.bin"))]
assert {item.stack_index for item in fine} == expected_stacks
assert {item.stack_index for item in factor} == expected_stacks
assert {item.header[4] for item in fine} == {expected_iteration}
assert {item.header[9] for item in factor} == {expected_iteration}
assert all(item.geometry_only for item in factor)
assert all((item.candidates["flags"] & ACTIVE).any() for item in fine)
PY
! rg -qi 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${RELION_OUT}/runner.stderr"

mapfile -t audit_iterations < <(seq 1 "${TARGET_ITERATION}")
"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  --control-dir "${CONTROL_DIR}" --candidate-dir "${RELION_OUT}" \
  --iterations "${audit_iterations[@]}" \
  --map-fsc-auc-threshold 0.999999 \
  --output-json "${RUN_ROOT}/analysis/RELION_FRESH_CAPTURE_INERTNESS.json" \
  --output-markdown "${RUN_ROOT}/analysis/RELION_FRESH_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${RUN_ROOT}/analysis/RELION_FRESH_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

find "${CAPTURE}" "${RELION_OUT}" "${RUN_ROOT}/analysis" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
iteration_csv=$(seq -s, 1 "${TARGET_ITERATION}")
printf '{"job_id":"%s","wall_s":%d,"iterations":[%s],"stack_indices_one_based":"%s","fresh_uninterrupted":true}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" "${iteration_csv}" "${TARGET_STACKS_ONE_BASED}" \
  > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
