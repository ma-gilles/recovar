#!/usr/bin/env bash
#SBATCH --job-name=vdam-relion-cont
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:15:00

set -euo pipefail

: "${VDAM_RELION_CONT_ROOT:?set a unique disposable output root}"
: "${VDAM_RELION_CONT_INPUT_DIR:?set the materialized fixture data directory}"
: "${VDAM_RELION_CONT_OPTIMISER:?set the exact incoming optimiser.star}"

ROOT=${VDAM_RELION_CONT_ROOT}
INPUT=${VDAM_RELION_CONT_INPUT_DIR}
CONTINUE=${VDAM_RELION_CONT_OPTIMISER}
ITERATION=${VDAM_RELION_CONT_ITERATION:-3}
PART_ID=${VDAM_RELION_CONT_PART_ID:-9}
STACK_INDEX=${VDAM_RELION_CONT_STACK_INDEX:-}
CAPTURE_ENABLED=${VDAM_RELION_CONT_CAPTURE:-0}
PERTURBATION=${VDAM_RELION_CONT_PERTURBATION:-}
RELION_BINARY=${VDAM_RELION_BINARY:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}
RELION_RUNTIME=${VDAM_RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
OUTPUT=${ROOT}/relion
CAPTURE=${ROOT}/capture

test -f "${ROOT}/SAFE_TO_DELETE"
test -s "${INPUT}/particles.star"
test -s "${CONTINUE}"
test -x "${RELION_BINARY}"
case "${CAPTURE_ENABLED}" in 0|1) ;; *) echo "VDAM_RELION_CONT_CAPTURE must be 0 or 1" >&2; exit 2 ;; esac
mkdir -p "${OUTPUT}" "${CAPTURE}" "${ROOT}/provenance"
test -z "$(find "${OUTPUT}" "${CAPTURE}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_ACC_DUMP_*|RELION_DUMP_*) unset "${variable_name}" ;; esac
done < <(env)

export TMPDIR=${ROOT}/runtime/${SLURM_JOB_ID}/tmp
mkdir -p "${TMPDIR}"

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

if [[ -n "${PERTURBATION}" ]]; then
    export RELION_FORCE_SAMPLING_PERTURB=${PERTURBATION}
fi

if [[ "${CAPTURE_ENABLED}" == 1 ]]; then
    export RELION_ACC_DUMP_DIR=${CAPTURE}
    export RELION_ACC_DUMP_PART_ID=${PART_ID}
    if [[ -n "${STACK_INDEX}" ]]; then
        export RELION_ACC_DUMP_STACK_INDEX=${STACK_INDEX}
    fi
    export RELION_ACC_DUMP_ITER=${ITERATION}
    export RELION_ACC_DUMP_VERBOSE=1
fi

COMMAND=(
    srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none "${RELION_BINARY}"
    --continue "${CONTINUE}" --o "${OUTPUT}/run"
    --pool 3 --gpu 0 --j 8
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
sha256sum "${RELION_BINARY}" "${CONTINUE}" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

start_epoch=$(date +%s)
(
    cd "${INPUT}"
    "${COMMAND[@]}"
) > "${OUTPUT}/runner.stdout" 2> "${OUTPUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test -s "${OUTPUT}/run_it$(printf '%03d' "${ITERATION}")_data.star"
if [[ "${CAPTURE_ENABLED}" == 1 ]]; then
    test -s "${CAPTURE}/Pmax.bin"
    test -s "${CAPTURE}/pass0_coarse_raw_diff2.bin"
    test -s "${CAPTURE}/pass1_exp_Mweight_posterior.bin"
fi
! rg -qi 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${OUTPUT}/runner.stderr"
printf '{"job_id":"%s","wall_s":%d,"iteration":%d,"part_id":%d,"stack_index":"%s","capture":%s,"continued_from":"%s","forced_perturbation":"%s"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${ITERATION}" "${PART_ID}" "${STACK_INDEX}" "${CAPTURE_ENABLED}" "${CONTINUE}" "${PERTURBATION}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
