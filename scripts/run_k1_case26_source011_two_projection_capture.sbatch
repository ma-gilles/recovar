#!/usr/bin/env bash
#SBATCH --job-name=k1c26-2proj
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_source011_two_projection_20260811T1339ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_source011_two_projection_20260811T1339ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:10:00

set -euo pipefail
: "${EXPECTED_REPO_HEAD:?submission must pin RECOVAR repository head}"
: "${EXPECTED_RELION_BINARY_SHA256:?submission must pin diagnostic RELION binary}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?submission must pin diagnostic RELION source diff}"

ROOT=${K1_CASE26_CAPTURE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_source011_two_projection_20260811T1339ET}
RUNTIME=${K1_CASE26_CAPTURE_RUNTIME:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_source011_two_projection_20260811T1339ET}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_source_20260810
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_build_20260810/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
OUTPUT=${ROOT}/relion
CAPTURE=${ROOT}/capture
CAPTURE_ITERATION=${K1_CASE26_CAPTURE_ITERATION:-2}
MAX_ITER=${K1_CASE26_MAX_ITER:-${CAPTURE_ITERATION}}
PART_IDS=${K1_CASE26_RELION_PART_IDS:-24}
IFS=, read -r -a capture_part_ids <<< "${PART_IDS}"
EXPECTED_PARTICLES=${#capture_part_ids[@]}
MAX_PARTICLES_PER_RANK=${K1_CASE26_MAX_PARTICLES_PER_RANK:-${EXPECTED_PARTICLES}}
ROTATION_KEYS=${K1_CASE26_RELION_ROTATION_KEYS-24289,24018}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${RELION_SOURCE}" diff -- src/acc/acc_ml_optimiser_impl.h src/backprojector.cpp | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 34016f7f367587125956de8f00895c27b14225511c8ef0f517179b890685e139
test -z "$(find "${OUTPUT}" "${CAPTURE}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_ACC_DUMP_*|RELION_DUMP_*|RELION_P1_*) unset "${variable_name}" ;; esac
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
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

# Defaults capture source row 11, which is RELION part_id 24 after its
# one-time physical dispatch.  The environment overrides make the same
# launcher usable for later focused particle/iteration boundaries.
export RELION_P1_CAPTURE_DIR=${CAPTURE}
export RELION_P1_CAPTURE_SCHEMA=2
export RELION_P1_CAPTURE_ITER=${CAPTURE_ITERATION}
export RELION_P1_CAPTURE_EXPECTED_PARTICLES=${EXPECTED_PARTICLES}
export RELION_P1_CAPTURE_MAX_PARTICLES_PER_RANK=${MAX_PARTICLES_PER_RANK}
export RELION_P1_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_P1_CAPTURE_MAX_BYTES=${K1_CASE26_CAPTURE_MAX_BYTES:-536870912}
export RELION_P1_CAPTURE_PART_IDS=${PART_IDS}
if [[ -n "${ROTATION_KEYS}" ]]; then
    export RELION_P1_CAPTURE_OPERAND_ROTATION_KEYS=${ROTATION_KEYS}
else
    unset RELION_P1_CAPTURE_OPERAND_ROTATION_KEYS
fi

COMMAND=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${OUTPUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1726 --auto_iter_max "${MAX_ITER}"
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"

start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${OUTPUT}/runner.stdout" 2> "${OUTPUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test -s "$(printf '%s/run_it%03d_data.star' "${OUTPUT}" "${MAX_ITER}")"
test "$(find "${CAPTURE}" -maxdepth 1 -name '*.p1-v2.bin' | wc -l)" -eq "${EXPECTED_PARTICLES}"
if [[ -n "${ROTATION_KEYS}" ]]; then
    test "$(find "${CAPTURE}" -maxdepth 1 -name '*.p1-op-v2.bin' | wc -l)" -eq "${EXPECTED_PARTICLES}"
else
    test "$(find "${CAPTURE}" -maxdepth 1 -name '*.p1-op-v2.bin' | wc -l)" -eq 0
fi
! grep -Eiq 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${OUTPUT}/runner.stderr"
find "${CAPTURE}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/capture_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":%d,"part_ids":"%s","relion_rotation_keys":"%s"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${CAPTURE_ITERATION}" "${PART_IDS}" "${ROTATION_KEYS}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
