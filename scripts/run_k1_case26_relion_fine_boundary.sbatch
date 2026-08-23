#!/usr/bin/env bash
#SBATCH --job-name=k1c26-rel-fine
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:15:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_RELION_HEAD:?submission must pin RELION source head}"
: "${EXPECTED_RELION_DIFF_SHA256:?submission must pin RELION source diff}"
: "${EXPECTED_RELION_BINARY_SHA256:?submission must pin RELION binary}"

ROOT=${K1_CASE26_FINE_ROOT:?set a unique run root}
RUNTIME=${K1_CASE26_FINE_RUNTIME:?set a unique runtime root}
PART_ID=${K1_CASE26_RELION_PART_ID:-24}
CAPTURE_ITERATION=${K1_CASE26_CAPTURE_ITERATION:-2}
MAX_ITER=${K1_CASE26_MAX_ITER:-${CAPTURE_ITERATION}}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
LAUNCHER=${REPO}/scripts/run_k1_case26_relion_fine_boundary.sbatch
RELION_SOURCE=/scratch/gpfs/GILLES/mg6942/relion
RELION_BINARY=${RELION_SOURCE}/build_patched/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${CASE}/data
OUTPUT=${ROOT}/relion
CAPTURE=${ROOT}/capture

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_HEAD}"
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_DIFF_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test -z "$(find "${OUTPUT}" "${CAPTURE}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_ACC_DUMP_*|RELION_DUMP_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${ROOT}/provenance"

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

export RELION_ACC_DUMP_DIR=${CAPTURE}
export RELION_ACC_DUMP_PART_ID=${PART_ID}
export RELION_ACC_DUMP_ITER=${CAPTURE_ITERATION}
export RELION_ACC_DUMP_VERBOSE=1

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
for required in \
    pass1_exp_Mweight_posterior.bin \
    pass1_candidate_weight_normalized.bin \
    pass1_candidate_in_reconstruction_set.bin \
    pass1_acc_rot_id.bin \
    pass1_acc_rot_idx.bin \
    pass1_acc_trans_idx.bin \
    pass1_exp_sum_weight.bin \
    pass1_exp_significant_weight.bin; do
    test -s "${CAPTURE}/${required}"
done
! grep -Eiq 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${OUTPUT}/runner.stderr"
find "${CAPTURE}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/capture_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":%d,"part_id":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${CAPTURE_ITERATION}" "${PART_ID}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
