#!/usr/bin/env bash
#SBATCH --job-name=k1-final-recon-stage
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RELION_BINARY:?RELION_BINARY is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_BINARY_SHA256:?EXPECTED_BINARY_SHA256 is required}"
: "${EXPECTED_INPUTS_SHA256:?EXPECTED_INPUTS_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

BINARY=${RELION_BINARY}
REPO=${REPO_ROOT}
LAUNCHER=${REPO}/scripts/capture_relion_k1_case10_final_reconstruct_stages.sbatch
FIXED_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE_ROOT=${FIXED_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE_ROOT}/data
CONTROL=${CASE_ROOT}/relion_ref
CONTINUE=${CONTROL}/run_it015_optimiser.star
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
OUTPUT=${RUN_ROOT}/relion
BPREF_DUMP=${RUN_ROOT}/bpref
STAGE_DUMP=${RUN_ROOT}/reconstruct
PROVENANCE=${RUN_ROOT}/provenance

input_digest() {
    sha256sum \
        "${CONTROL}/run_it015_optimiser.star" \
        "${CONTROL}/run_it015_data.star" \
        "${CONTROL}/run_it015_half1_model.star" \
        "${CONTROL}/run_it015_half2_model.star" \
        "${CONTROL}/run_it015_sampling.star" \
        "${CONTROL}/run_it015_half1_class001.mrc" \
        "${CONTROL}/run_it015_half2_class001.mrc" \
        "${CONTROL}/run_class001.mrc" \
        | sha256sum | awk '{print $1}'
}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${DATA}/particles.384.mrcs"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = "${EXPECTED_BINARY_SHA256}"
test "$(strings "${BINARY}" | rg -c RELION_DUMP_BPREF)" -ge 1
test "$(strings "${BINARY}" | rg -c RELION_RESTORE_FRESH_PHYSICAL_ORDER)" -ge 1
test "$(strings "${BINARY}" | rg -c RELION_RECON_DUMP_DIR)" -ge 1
test "$(strings "${BINARY}" | rg -c volume_after_gridding)" -ge 1
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${OUTPUT}" "${BPREF_DUMP}" "${STAGE_DUMP}" "${PROVENANCE}" "${RUN_ROOT}/logs"

source /etc/profile.d/modules.sh
set +u
module purge
module load gcc/11 openmpi/cuda-12.6/gcc/4.1.6 cudatoolkit/12.6
set -u
MPI_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_mpi_dev_cuda126_openmpi416_gcc_20260827
CUDA_ROOT=/usr/local/cuda-12.6
export PATH=${MPI_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_ROOT}/lib64
test -r "${MPI_ROOT}/lib64/libmpi.so"
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign one visible H100}"

# The sealed iteration-15 continuation has exactly one filtered joined
# reconstruction on rank 1. Restore the fresh-run physical order so the BPref
# and every dumped reconstruction stage describe the same call.
export RELION_RESTORE_FRESH_PHYSICAL_ORDER=1
export RELION_DUMP_DIR=${BPREF_DUMP}
export RELION_DUMP_BPREF=1
export RELION_DUMP_ITER=16
export RELION_RECON_DUMP_DIR=${STAGE_DUMP}
export RELION_RECON_DUMP_CALL=0

env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
sha256sum "${BINARY}" "${LAUNCHER}" \
    > "${PROVENANCE}/static_${SLURM_JOB_ID}.sha256"
sha256sum \
    "${CONTROL}/run_it015_optimiser.star" \
    "${CONTROL}/run_it015_data.star" \
    "${CONTROL}/run_it015_half1_model.star" \
    "${CONTROL}/run_it015_half2_model.star" \
    "${CONTROL}/run_it015_sampling.star" \
    "${CONTROL}/run_it015_half1_class001.mrc" \
    "${CONTROL}/run_it015_half2_class001.mrc" \
    "${CONTROL}/run_class001.mrc" \
    > "${PROVENANCE}/inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
    --continue "${CONTINUE}" --o "${OUTPUT}/run" --force_converge
    --auto_iter_max 999 --pool 3 --gpu 0:0:0 --j 4)
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

PREFIX=${BPREF_DUMP}/mstep_it016_rank1_half0_c0_pre_reconstruct
for suffix in bpref_data bpref_weight tau2 sigma2 data_vs_prior fsc fourier_coverage; do
    test -s "${PREFIX}_${suffix}.bin"
done
test -s "${PREFIX}_bpref_meta.txt"
for stage in tau2 fweight_decentered fweight_regularized radavg_weight \
    fconv_divided volume_before_gridding volume_after_gridding; do
    test "$(find "${STAGE_DUMP}" -maxdepth 1 -type f -name "reconstruct_rank01_pid*_call0000_${stage}.bin" | wc -l)" -eq 1
done
test "$(find "${STAGE_DUMP}" -maxdepth 1 -type f -name 'reconstruct_rank01_pid*_call0000_metadata.txt' | wc -l)" -eq 1
RANK1_METADATA=$(find "${STAGE_DUMP}" -maxdepth 1 -type f -name 'reconstruct_rank01_pid*_call0000_metadata.txt')
rg -q '^do_map=1$' "${RANK1_METADATA}"
rg -q '^skip_gridding=1$' "${RANK1_METADATA}"
rg -q '^pad_size=771$' "${RANK1_METADATA}"
rg -q '^r_max=192$' "${RANK1_METADATA}"
test -s "${OUTPUT}/run_class001.mrc"
! rg -q 'ERROR|Error|Traceback|segmentation fault|Out of memory' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stdout" \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
find "${OUTPUT}" "${BPREF_DUMP}" "${STAGE_DUMP}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
