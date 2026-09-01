#!/usr/bin/env bash
#SBATCH --job-name=k1-fresh-coarse
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
: "${TARGET_PART_ID:?TARGET_PART_ID is required}"
: "${TARGET_STACK_ONE_BASED:?TARGET_STACK_ONE_BASED is required}"
: "${OPERAND_ROTATION_KEYS:?OPERAND_ROTATION_KEYS is required}"
: "${RANDOM_SEED:?RANDOM_SEED is required}"
: "${EXPECTED_PARTICLES_SHA256:?EXPECTED_PARTICLES_SHA256 is required}"
: "${EXPECTED_REFERENCE_SHA256:?EXPECTED_REFERENCE_SHA256 is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/run_k1_native_fresh_coarse_capture.sbatch
BUILD_ROOT=${BUILD_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case07_live_coarse_operands_20260815T0358ET}
RELION_SOURCE=${RELION_SOURCE:-${BUILD_ROOT}/relion_source}
RELION_BINARY=${RELION_BINARY:-${BUILD_ROOT}/build_login/bin/relion_refine_mpi}
RELION_RUNTIME=${RELION_RUNTIME:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717}
AUDIT_REPO=${AUDIT_REPO:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout}
EXPECTED_RELION_SOURCE_HEAD=${EXPECTED_RELION_SOURCE_HEAD:-8ba3e0a86fef0ad16d83206be4a219fe74cf28df}
EXPECTED_RELION_SOURCE_DIFF_SHA256=${EXPECTED_RELION_SOURCE_DIFF_SHA256:-3cf6026238fbfb0d3d8442f5546a0d041eccbed7e1cea1431d96accbc997205f}
EXPECTED_RELION_BINARY_SHA256=${EXPECTED_RELION_BINARY_SHA256:-c1620145f7c3529d701303d0354b62ac17611bf812f6273192e4a45a7a8704d7}
EXPECTED_AUDIT_REPO_HEAD=${EXPECTED_AUDIT_REPO_HEAD:-187a9f60cdea19363d49d46099b052f23288f05a}
CONTINUE_STAR=${CONTINUE_STAR:-}
TARGET_ITERATION=${TARGET_ITERATION:-2}
PHYSICAL_ORDER_SEED=${PHYSICAL_ORDER_SEED:-${RANDOM_SEED}}
SAMPLING_PERTURBATION_OVERRIDE=${SAMPLING_PERTURBATION_OVERRIDE:-}

CAPTURE=${RUN_ROOT}/capture
RELION_OUT=${RUN_ROOT}/relion

test "${TARGET_PART_ID}" -ge 0
test "${TARGET_STACK_ONE_BASED}" -gt 0
test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${PYTHON}" -a -x "${RELION_BINARY}"
test -s "${DATA_DIR}/particles.star" -a -s "${DATA_DIR}/reference_init_relion.mrc"
if [[ -n "${CONTINUE_STAR}" ]]; then
  test -s "${CONTINUE_STAR}"
  test -n "${SAMPLING_PERTURBATION_OVERRIDE}"
fi
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${DATA_DIR}/particles.star" | awk '{print $1}')" = "${EXPECTED_PARTICLES_SHA256}"
test "$(sha256sum "${DATA_DIR}/reference_init_relion.mrc" | awk '{print $1}')" = "${EXPECTED_REFERENCE_SHA256}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_SOURCE_HEAD}"
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = "${EXPECTED_AUDIT_REPO_HEAD}"
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
mkdir -p "${CAPTURE}" "${RELION_OUT}" "${RUN_ROOT}/analysis" \
  "${RUN_ROOT}/provenance" "${RUN_ROOT}/logs"
test -z "$(find "${CAPTURE}" "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
while IFS='=' read -r variable_name _; do
  case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

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

export RELION_P1_CAPTURE_DIR=${CAPTURE}
export RELION_P1_CAPTURE_SCHEMA=3
export RELION_P1_CAPTURE_ITER=${TARGET_ITERATION}
export RELION_P1_CAPTURE_EXPECTED_PARTICLES=1
export RELION_P1_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_P1_CAPTURE_EXPECTED_FOLLOWERS=1
export RELION_P1_CAPTURE_MAX_BYTES=1000000000
export RELION_P1_CAPTURE_PART_IDS=${TARGET_PART_ID}
export RELION_P1_CAPTURE_OPERAND_ROTATION_KEYS=${OPERAND_ROTATION_KEYS}
if [[ -n "${CONTINUE_STAR}" ]]; then
  export RELION_PREITER_PARTICLE_ORDER_SEED=${PHYSICAL_ORDER_SEED}
  export RELION_PREITER_PARTICLE_ORDER_ITER=${TARGET_ITERATION}
  export RELION_SAMPLING_PERTURBATION_OVERRIDE=${SAMPLING_PERTURBATION_OVERRIDE}
  export RELION_SAMPLING_PERTURBATION_OVERRIDE_ITER=${TARGET_ITERATION}
fi

env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${RELION_SOURCE}" diff HEAD --binary > "${RUN_ROOT}/provenance/relion_source.patch"
sha256sum "${RELION_BINARY}" "${LAUNCHER}" "${DATA_DIR}/particles.star" \
  "${DATA_DIR}/reference_init_relion.mrc" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

if [[ -n "${CONTINUE_STAR}" ]]; then
  command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --continue "${CONTINUE_STAR}" --o "${RELION_OUT}/run"
    --auto_iter_max "${TARGET_ITERATION}" --pool 3 --gpu 0:0:0 --j 4)
else
  command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed "${RANDOM_SEED}" --auto_iter_max 2
    --pool 3 --gpu 0:0:0 --j 4)
fi
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
  cd "${DATA_DIR}"
  "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

prefix=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK_ONE_BASED}
for suffix in p1-v2 p1-op-v2 p1-lane-v1 p1-live-v1; do
  test -s "${prefix}.${suffix}.bin"
done
test -z "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.tmp.*' -print -quit)"

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_coarse_live_operands \
  --live "${prefix}.p1-live-v1.bin" \
  --operands "${prefix}.p1-op-v2.bin" \
  --lanes "${prefix}.p1-lane-v1.bin" \
  --components "${prefix}.p1-v2.bin" \
  --output-json "${RUN_ROOT}/analysis/K1_FRESH_COARSE_LIVE_OPERANDS.json"

"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  --control-dir "${CONTROL_DIR}" --candidate-dir "${RELION_OUT}" --iterations $([[ -n "${CONTINUE_STAR}" ]] && printf '%s' "${TARGET_ITERATION}" || printf '1 2') \
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
printf '{"job_id":"%s","wall_s":%d,"iterations":"%s","part_id":%d,"stack_index_one_based":%d,"rotation_keys":"%s","continued":%s}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" "$([[ -n "${CONTINUE_STAR}" ]] && printf '%s' "${TARGET_ITERATION}" || printf '1,2')" \
  "${TARGET_PART_ID}" "${TARGET_STACK_ONE_BASED}" "${OPERAND_ROTATION_KEYS}" \
  "$([[ -n "${CONTINUE_STAR}" ]] && printf true || printf false)" \
  > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
