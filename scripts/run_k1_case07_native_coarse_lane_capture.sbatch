#!/usr/bin/env bash
#SBATCH --job-name=k1c07-lanes
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
: "${RUN_ROOT:?set an empty disposable run root}"
: "${RUNTIME_ROOT:?set an empty disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
LAUNCHER=${REPO}/scripts/run_k1_case07_native_coarse_lane_capture.sbatch
SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_coarse_lane_capture_4cf1015c_20260809T0805ET/relion_source
BUILD=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_coarse_lane_capture_4cf1015c_20260809T0805ET/relion_build_retry8
RELION_BINARY=${BUILD}/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/cases/7_anisotropic_100k_g256_white_noise1_bf80
DATA=${CASE}/data
CAPTURE=${ROOT}/capture
RELION_OUT=${ROOT}/relion
TARGET_ITERATION=2
TARGET_PART_ID=${K1_CASE07_LANE_PART_ID:-22883}
TARGET_STACK=${K1_CASE07_LANE_STACK_ONE_BASED:-30592}
ROTATION_KEYS=${K1_CASE07_LANE_ROTATION_KEYS:-20197}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = 8ba3e0a86fef0ad16d83206be4a219fe74cf28df
test "$(git -C "${SOURCE}" rev-parse HEAD^{tree})" = 61e8f9cce38fcac3fea64278ca884c8048566b75
test -z "$(git -C "${SOURCE}" status --porcelain=v1)"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 53a6bd57df57d3c36711f1658bc6947a3a68592406b2a5b12123e8cd02110a3e
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 12186134dba0deda4a04a131658f03850d5987e9cd8fd7f8730fb556b5d14316
test "$(sha256sum "${DATA}/particles.256.mrcs" | awk '{print $1}')" = a799a90687b26dceba5832a2de2729fe92d8718ca795c89d65795dc2cb9ea25c
test -z "$(find "${CAPTURE}" "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${ROOT}/analysis" "${ROOT}/provenance"

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
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign exactly one visible GPU}"
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)" in
    *H100*) ;;
    *) exit 2 ;;
esac
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

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

command=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1707 --auto_iter_max 2
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"

components=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK}.p1-v2.bin
operands=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK}.p1-op-v2.bin
lanes=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK}.p1-lane-v1.bin
test -s "${components}" -a -s "${operands}" -a -s "${lanes}"
test -s "${RELION_OUT}/run_it002_data.star"
cd "${REPO}"
set +e
.pixi/envs/default/bin/python -m scripts.validate_relion_coarse_lane_capture \
    "${CAPTURE}" --output-json "${ROOT}/analysis/RELION_LANE_VALIDATION.json"
validation_status=$?
set -e
test "${validation_status}" -le 1
printf '{"job_id":"%s","wall_s":%d,"part_id":%d,"stack_index_one_based":%d,"validation_status":%d}\n' \
    "${SLURM_JOB_ID}" "$(( $(date +%s) - start ))" "${TARGET_PART_ID}" \
    "${TARGET_STACK}" "${validation_status}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${components}" "${operands}" \
    "${lanes}" "${ROOT}/analysis/RELION_LANE_VALIDATION.json" \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
