#!/usr/bin/env bash
#SBATCH --job-name=relion-k1-xaaa-fresh
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
: "${TARGET_PART_IDS:?TARGET_PART_IDS is required}"
: "${EXPECTED_BINARY_SHA256:?EXPECTED_BINARY_SHA256 is required}"
: "${EXPECTED_INPUT_STAR_SHA256:?EXPECTED_INPUT_STAR_SHA256 is required}"
: "${EXPECTED_REFERENCE_SHA256:?EXPECTED_REFERENCE_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
FIXED_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${FIXED_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE}/data
BUILD_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_scale_totals_d476e6_20260823T0310ET
BINARY=${BUILD_ROOT}/build_mpi_gcc/bin/relion_refine_mpi
INPUT_STAR=${DATA}/particles.star
REFERENCE=${DATA}/reference_init_relion.mrc
LAUNCHER=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820/scripts/run_relion_k1_scale_totals_fresh_it2.sbatch
OUTPUT=${ROOT}/relion_ref
DUMP=${ROOT}/scale_totals

require_equal() {
    local label=$1 actual=$2 expected=$3
    if [[ "${actual}" != "${expected}" ]]; then
        printf 'provenance mismatch: %s actual=%q expected=%q\n' "${label}" "${actual}" "${expected}" >&2
        exit 2
    fi
}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -x "${BINARY}" -a -s "${INPUT_STAR}" -a -s "${REFERENCE}"
test -z "$(find "${OUTPUT}" "${DUMP}" -mindepth 1 -print -quit)"
require_equal binary "$(sha256sum "${BINARY}" | awk '{print $1}')" "${EXPECTED_BINARY_SHA256}"
require_equal input_star "$(sha256sum "${INPUT_STAR}" | awk '{print $1}')" "${EXPECTED_INPUT_STAR_SHA256}"
require_equal reference "$(sha256sum "${REFERENCE}" | awk '{print $1}')" "${EXPECTED_REFERENCE_SHA256}"
require_equal launcher "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" "${EXPECTED_LAUNCHER_SHA256}"
require_equal source_head "$(git -C "${BUILD_ROOT}/source" rev-parse HEAD)" d476e6f6a4f1f37627c06ace5227fc374c0c2b05
require_equal source_diff "$(git -C "${BUILD_ROOT}/source" diff --binary | sha256sum | awk '{print $1}')" 2a77d17c04e08466f28b3ea724f2cf127bab716ac0248de98f684bc6f2d30ae7

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${ROOT}/logs" "${ROOT}/provenance" "${OUTPUT}" "${DUMP}"

source /etc/profile.d/modules.sh
set +u
module purge
module load gcc/11 openmpi/gcc/4.1.6 cudatoolkit/12.6
set -u
export CUDA_VISIBLE_DEVICES=0
export RELION_DUMP_K1_SCALE_TOTALS_DIR=${DUMP}
export RELION_DUMP_K1_SCALE_TOTAL_PART_IDS=${TARGET_PART_IDS}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}

{
    printf 'binary=%s\n' "${BINARY}"
    printf 'binary_sha256=%s\n' "${EXPECTED_BINARY_SHA256}"
    printf 'input_star=%s\n' "${INPUT_STAR}"
    printf 'reference=%s\n' "${REFERENCE}"
    printf 'target_part_ids=%s\n' "${TARGET_PART_IDS}"
} > "${ROOT}/provenance/command_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"

cd "${DATA}"
mpirun -n 3 "${BINARY}" \
    --i particles.star \
    --ref reference_init_relion.mrc \
    --o "${OUTPUT}/run" \
    --auto_refine \
    --split_random_halves \
    --particle_diameter 200 \
    --ini_high 30 \
    --firstiter_cc \
    --ctf \
    --flatten_solvent \
    --zero_mask \
    --low_resol_join_halves 40 \
    --norm \
    --scale \
    --healpix_order 3 \
    --offset_range 3.0 \
    --offset_step 1.0 \
    --oversampling 1 \
    --pad 2 \
    --random_seed 1710 \
    --auto_iter_max 2 \
    --pool 3 \
    --gpu 0 \
    --j 4 \
    > "${ROOT}/logs/relion_${SLURM_JOB_ID}.log" 2>&1

test -s "${OUTPUT}/run_it002_data.star"
test -s "${OUTPUT}/run_it002_half1_model.star"
test -s "${OUTPUT}/run_it002_half2_model.star"
test -s "${DUMP}/scale_totals_half1.tsv"
test -s "${DUMP}/scale_totals_half2.tsv"
test "$(awk -F '\t' '$2 == "iter=2" {count++} END {print count + 0}' "${DUMP}"/scale_totals_half*.tsv)" -eq 40
sha256sum "${OUTPUT}"/run_it002_*.star "${DUMP}"/scale_totals_half*.tsv \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
