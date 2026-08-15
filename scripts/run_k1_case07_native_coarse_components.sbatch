#!/usr/bin/env bash
#SBATCH --job-name=k1c07-components
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:30:00

set -euo pipefail
: "${RUN_ROOT:?set an empty run root}"
: "${RUNTIME_ROOT:?set an empty runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher}"
: "${EXPECTED_REPO_HEAD:?pin the RECOVAR checkout}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
LAUNCHER=${REPO}/scripts/run_k1_case07_native_coarse_components.sbatch
SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_components_source_b571275_20260812/source
BUILD=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_components_source_b571275_20260812/build
RELION_BINARY=${BUILD}/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/cases/7_anisotropic_100k_g256_white_noise1_bf80
DATA=${CASE}/data
CAPTURE=${ROOT}/capture
RELION_OUT=${ROOT}/relion
TARGET_PART_ID=${K1_CASE07_COMPONENT_PART_ID:-51690}
TARGET_STACK=${K1_CASE07_COMPONENT_STACK_ONE_BASED:-56519}
ROTATION_KEYS=${K1_CASE07_COMPONENT_ROTATION_KEYS:-28056,28059}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${SOURCE}" diff -- src/acc/acc_ml_optimiser_impl.h | sha256sum | awk '{print $1}')" = e98ecf36c38d6a28d6e00485152e0949f3de5a00d348583e4ff7fa46c0c7e615
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 7f91065a92781c23e82df318f38a9ffa83fce10fd75728e8723b98b10eb73b69
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
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

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
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0 CUDA_VISIBLE_DEVICES=0
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)" in
    *H100*) ;;
    *) exit 2 ;;
esac

export RELION_P1_CAPTURE_DIR=${CAPTURE}
export RELION_P1_CAPTURE_SCHEMA=2
export RELION_P1_CAPTURE_ITER=2
export RELION_P1_CAPTURE_EXPECTED_PARTICLES=1
export RELION_P1_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_P1_CAPTURE_EXPECTED_FOLLOWERS=1
export RELION_P1_CAPTURE_MAX_BYTES=100000000
export RELION_P1_CAPTURE_PART_IDS=${TARGET_PART_ID}
export RELION_P1_CAPTURE_OPERAND_ROTATION_KEYS=${ROTATION_KEYS}

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
{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "target_part_id=${TARGET_PART_ID}"
    echo "target_stack_one_based=${TARGET_STACK}"
    echo "rotation_keys=${ROTATION_KEYS}"
    echo "repo_head=${EXPECTED_REPO_HEAD}"
    sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${DATA}/particles.star"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"
start=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"

components=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK}.p1-v2.bin
operands=${CAPTURE}/part${TARGET_PART_ID}_stack${TARGET_STACK}.p1-op-v2.bin
test -s "${components}" -a -s "${operands}"
test -s "${RELION_OUT}/run_it002_data.star"
printf '{"job_id":"%s","wall_s":%d,"part_id":%d,"stack_index_one_based":%d}\n' \
    "${SLURM_JOB_ID}" "$(( $(date +%s) - start ))" "${TARGET_PART_ID}" "${TARGET_STACK}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
find "${CAPTURE}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/capture_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
