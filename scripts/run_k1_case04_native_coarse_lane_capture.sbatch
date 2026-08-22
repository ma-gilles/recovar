#!/usr/bin/env bash
#SBATCH --job-name=k1c04-lanes
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
LAUNCHER=${REPO}/scripts/run_k1_case04_native_coarse_lane_capture.sbatch
SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_coarse_lane_capture_4cf1015c_20260809T0805ET/relion_source
BUILD=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_coarse_lane_capture_4cf1015c_20260809T0805ET/relion_build_retry8
RELION_BINARY=${BUILD}/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
CAPTURE=${ROOT}/capture
RELION_OUT=${ROOT}/relion
PYTHON=${REPO}/.pixi/envs/default/bin/python
TARGET_ITERATION=2
TARGET_PART_ID=65218
TARGET_STACK=68695
ROTATION_KEYS=10258,10355

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = 8ba3e0a86fef0ad16d83206be4a219fe74cf28df
test "$(git -C "${SOURCE}" rev-parse HEAD^{tree})" = 61e8f9cce38fcac3fea64278ca884c8048566b75
test -z "$(git -C "${SOURCE}" status --porcelain=v1)"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 53a6bd57df57d3c36711f1658bc6947a3a68592406b2a5b12123e8cd02110a3e
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 0a90714711cdf0a1820d46fa82460d828b87e1ecbd2e49de645170863546ac98
test -x "${PYTHON}" -a -x "${RELION_BINARY}"
test -z "$(find "${CAPTURE}" "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
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

export RELION_P1_CAPTURE_DIR=${CAPTURE}
export RELION_P1_CAPTURE_SCHEMA=3
export RELION_P1_CAPTURE_ITER=${TARGET_ITERATION}
export RELION_P1_CAPTURE_EXPECTED_PARTICLES=1
export RELION_P1_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_P1_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_P1_CAPTURE_MAX_BYTES=200000000
export RELION_P1_CAPTURE_PART_IDS=${TARGET_PART_ID}
export RELION_P1_CAPTURE_OPERAND_ROTATION_KEYS=${ROTATION_KEYS}
export SLURM_EXPORT_ENV=ALL

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" \
    "${REPO}/scripts/validate_relion_coarse_lane_capture.py" \
    "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
  --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
  --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
  --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
  --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
  --oversampling 1 --pad 2 --random_seed 1704 --auto_iter_max 2
  --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
  cd "${DATA}"
  "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

components=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK}.p1-v2.bin
operands=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK}.p1-op-v2.bin
lanes=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK}.p1-lane-v1.bin
test -s "${components}" -a -s "${operands}" -a -s "${lanes}"
test -s "${RELION_OUT}/run_it002_data.star"
cd "${REPO}"
set +e
"${PYTHON}" -m scripts.validate_relion_coarse_lane_capture \
    "${CAPTURE}" --output-json "${ROOT}/analysis/RELION_LANE_VALIDATION.json"
validation_status=$?
set -e
test "${validation_status}" -le 1
"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" --iterations 1 2 \
  --map-fsc-auc-threshold 0.999999 \
  --output-json "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
  --output-markdown "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

printf '{"job_id":"%s","wall_s":%d,"part_id":%d,"stack_index_one_based":%d,"rotation_keys":"%s","validation_status":%d}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" "${TARGET_PART_ID}" "${TARGET_STACK}" \
  "${ROTATION_KEYS}" "${validation_status}" \
  > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
find "${CAPTURE}" "${RELION_OUT}" "${ROOT}/analysis" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
