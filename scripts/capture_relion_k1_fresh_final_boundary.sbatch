#!/usr/bin/env bash
#SBATCH --job-name=k1-fresh-final-capture
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=10:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CASE_ROOT:?CASE_ROOT is required}"
: "${GRID_SIZE:?GRID_SIZE is required}"
: "${RANDOM_SEED:?RANDOM_SEED is required}"
: "${FINAL_RELION_ITERATION:?FINAL_RELION_ITERATION is required}"
: "${RELION_BINARY:?RELION_BINARY is required}"
: "${EXPECTED_BINARY_SHA256:?EXPECTED_BINARY_SHA256 is required}"
: "${EXPECTED_INPUTS_SHA256:?EXPECTED_INPUTS_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_pr167_postintegration_debug_20260827
LAUNCHER=${REPO}/scripts/capture_relion_k1_fresh_final_boundary.sbatch
DATA=${CASE_ROOT}/data
BINARY=${RELION_BINARY}
OUTPUT=${RUN_ROOT}/relion
DUMP=${RUN_ROOT}/bpref
PROVENANCE=${RUN_ROOT}/provenance
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
MPI_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_mpi_dev_cuda126_openmpi416_gcc_20260827

input_digest() {
    sha256sum \
        "${DATA}/particles.star" \
        "${DATA}/particles.${GRID_SIZE}.mrcs" \
        "${DATA}/reference_init_relion.mrc" \
        | sha256sum | awk '{print $1}'
}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${DATA}/particles.star"
test -s "${DATA}/particles.${GRID_SIZE}.mrcs"
test -s "${DATA}/reference_init_relion.mrc"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = "${EXPECTED_BINARY_SHA256}"
test "$(strings "${BINARY}" | rg -c RELION_DUMP_BPREF)" -ge 1
test "$(strings "${BINARY}" | rg -c pre_reconstruct)" -ge 1
test "$(input_digest)" = "${EXPECTED_INPUTS_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test -z "$(
    find "${RUN_ROOT}" -mindepth 1 \
        ! -name SAFE_TO_DELETE \
        ! -name 'slurm-*.out' \
        -print -quit
)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${OUTPUT}" "${DUMP}" "${PROVENANCE}" "${RUN_ROOT}/logs"

source /etc/profile.d/modules.sh
set +u
module purge
module load gcc/11 openmpi/cuda-12.6/gcc/4.1.6 cudatoolkit/12.6
set -u
CUDA_ROOT=/usr/local/cuda-12.6
export PATH=${MPI_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_ROOT}/lib64
test -r "${MPI_ROOT}/lib64/libmpi.so"
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign one visible H100}"

export RELION_DUMP_DIR=${DUMP}
export RELION_DUMP_BPREF=1
export RELION_DUMP_ITER=${FINAL_RELION_ITERATION}

env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
sha256sum "${BINARY}" "${LAUNCHER}" \
    > "${PROVENANCE}/static_${SLURM_JOB_ID}.sha256"
sha256sum "${DATA}/particles.star" "${DATA}/particles.${GRID_SIZE}.mrcs" \
    "${DATA}/reference_init_relion.mrc" \
    > "${PROVENANCE}/inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
    --i particles.star
    --ref reference_init_relion.mrc
    --o "${OUTPUT}/run"
    --auto_refine
    --split_random_halves
    --particle_diameter 200
    --ini_high 30
    --firstiter_cc
    --ctf
    --flatten_solvent
    --zero_mask
    --low_resol_join_halves 40
    --norm
    --scale
    --healpix_order 3
    --offset_range 3.0
    --offset_step 1.0
    --oversampling 1
    --pad 2
    --random_seed "${RANDOM_SEED}"
    --auto_iter_max 999
    --pool 3
    --gpu 0:0:0
    --j 4)
printf 'cd %q\n' "${DATA}" > "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '%q ' "${command[@]}" >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
start_seconds=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
end_seconds=$(date +%s)
printf '{"elapsed_seconds":%d,"job_id":"%s"}\n' \
    "$((end_seconds - start_seconds))" "${SLURM_JOB_ID}" \
    > "${PROVENANCE}/walltime_${SLURM_JOB_ID}.json"

PREFIX=${DUMP}/mstep_it$(printf '%03d' "${FINAL_RELION_ITERATION}")_rank1_half0_c0_pre_reconstruct
for suffix in bpref_data bpref_weight tau2 sigma2 data_vs_prior fsc fourier_coverage; do
    test -s "${PREFIX}_${suffix}.bin"
done
test -s "${PREFIX}_bpref_meta.txt"
test -s "${OUTPUT}/run_class001.mrc"
rg -q "^iter=${FINAL_RELION_ITERATION}$" "${PREFIX}_bpref_meta.txt"
! rg -q 'ERROR|Error|Traceback|segmentation fault|Out of memory' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stdout" \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
find "${OUTPUT}" "${DUMP}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
