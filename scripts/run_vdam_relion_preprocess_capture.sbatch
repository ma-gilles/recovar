#!/usr/bin/env bash
#SBATCH --job-name=vdam-preprocess
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

: "${VDAM_PREPROCESS_ROOT:?set a unique scratch output root}"
: "${VDAM_PREPROCESS_INPUT_DIR:?set the materialized VDAM fixture data directory}"

ROOT=${VDAM_PREPROCESS_ROOT}
INPUT=${VDAM_PREPROCESS_INPUT_DIR}
PART_ID=${VDAM_PREPROCESS_PART_ID:-69}
PART_IDS=${VDAM_PREPROCESS_PART_IDS:-}
EXPECTED_PART_COUNT=${VDAM_PREPROCESS_EXPECTED_PART_COUNT:-}
CAPTURE_ITERATION=${VDAM_PREPROCESS_ITERATION:-1}
NR_ITER_SCHEDULE=${VDAM_PREPROCESS_NR_ITER_SCHEDULE:-${CAPTURE_ITERATION}}
THREADS=${VDAM_PREPROCESS_THREADS:-8}
CAPTURE_MSTEP=${VDAM_PREPROCESS_CAPTURE_MSTEP:-0}
CAPTURE_PARTICLE_BPREF=${VDAM_PREPROCESS_CAPTURE_PARTICLE_BPREF:-0}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
LAUNCHER=${REPO_ROOT}/scripts/run_vdam_relion_preprocess_capture.sbatch
RELION_SOURCE=${VDAM_RELION_SOURCE:-/scratch/gpfs/GILLES/mg6942/relion}
: "${VDAM_RELION_BINARY:?set the isolated non-MPI RELION capture binary}"
RELION_BINARY=${VDAM_RELION_BINARY}
RELION_RUNTIME=${VDAM_RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
OUTPUT=${ROOT}/relion
CAPTURE=${ROOT}/capture

test -f "${ROOT}/SAFE_TO_DELETE"
test -s "${INPUT}/particles.star"
test -x "${RELION_BINARY}"
mkdir -p "${OUTPUT}" "${CAPTURE}"
test -z "$(find "${OUTPUT}" "${CAPTURE}" -mindepth 1 -print -quit)"
test "${NR_ITER_SCHEDULE}" -ge "${CAPTURE_ITERATION}"
test "${THREADS}" -ge 1
case "${CAPTURE_MSTEP}" in 0|1) ;; *) echo "VDAM_PREPROCESS_CAPTURE_MSTEP must be 0 or 1" >&2; exit 2 ;; esac
case "${CAPTURE_PARTICLE_BPREF}" in 0|1) ;; *) echo "VDAM_PREPROCESS_CAPTURE_PARTICLE_BPREF must be 0 or 1" >&2; exit 2 ;; esac
if [[ "${CAPTURE_MSTEP}" = 1 ]]; then
    test "${CAPTURE_ITERATION}" -eq 1
fi
if [[ -n "${PART_IDS}" ]]; then
    test "${PART_IDS}" = "$(printf '%s' "${PART_IDS}" | tr -cd '0-9,')"
fi
if [[ -n "${EXPECTED_PART_COUNT}" ]]; then
    test "${EXPECTED_PART_COUNT}" -ge 1
    actual_part_count=$(tr ',' '\n' <<<"${PART_IDS:-${PART_ID}}" | wc -l)
    test "${actual_part_count}" -eq "${EXPECTED_PART_COUNT}"
fi

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_ACC_DUMP_*|RELION_DUMP_*) unset "${variable_name}" ;; esac
done < <(env)

export TMPDIR=${ROOT}/runtime/${SLURM_JOB_ID}/tmp
mkdir -p "${TMPDIR}" "${ROOT}/provenance"

source /etc/profile.d/modules.sh
set +u
module purge
set +e
module load relion/5.0.0/gcc-11.5.0
module_status=$?
set -e
if [[ "${module_status}" -ne 0 ]]; then
    echo "RELION module returned ${module_status}; using the pinned runtime overlay" >&2
fi
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
if [[ -n "${PART_IDS}" ]]; then
    export RELION_ACC_DUMP_PART_IDS=${PART_IDS}
    export RELION_ACC_DUMP_VERBOSE=0
else
    export RELION_ACC_DUMP_VERBOSE=1
fi
export RELION_DUMP_DIR=${CAPTURE}
export RELION_DUMP_PART_ID=${PART_ID}
export RELION_DUMP_ITER=${CAPTURE_ITERATION}
if [[ "${CAPTURE_PARTICLE_BPREF}" = 1 ]]; then
    export RELION_ACC_DUMP_PARTICLE_BPREF=1
fi
if [[ "${CAPTURE_MSTEP}" = 1 ]]; then
    export RECOVAR_DEBUG_DUMP_DIR=${CAPTURE}
    export RECOVAR_DEBUG_DUMP_MSTEP_ITER=${CAPTURE_ITERATION}
fi

COMMAND=(
    srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none "${RELION_BINARY}"
    --o "${OUTPUT}/run" --iter "${NR_ITER_SCHEDULE}" --grad --denovo_3dref
    --grad_write_iter 1 --i particles.star --ctf --K 1 --sym C1
    --flatten_solvent --zero_mask --dont_combine_weights_via_disc --pool 3 --pad 1
    --particle_diameter 200 --oversampling 1 --healpix_order 1 --offset_range 6.0
    --offset_step 2.0 --auto_sampling --tau2_fudge 4.0 --random_seed 0 --j "${THREADS}" --gpu 0
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
git -C "${RELION_SOURCE}" rev-parse HEAD > "${ROOT}/provenance/relion_head.txt"
git -C "${RELION_SOURCE}" diff --binary | sha256sum > "${ROOT}/provenance/relion_diff.sha256"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" > "${ROOT}/provenance/relion_binary.sha256"

start_epoch=$(date +%s)
if [[ "${NR_ITER_SCHEDULE}" -gt "${CAPTURE_ITERATION}" ]]; then
    set +e
    (
        cd "${INPUT}"
        "${COMMAND[@]}"
    ) > "${OUTPUT}/runner.stdout" 2> "${OUTPUT}/runner.stderr" &
    runner_pid=$!
    set -e
    target_tag=$(printf '%03d' "${CAPTURE_ITERATION}")
    target_complete=0
    for _poll_index in $(seq 1 600); do
        if [[ -s "${OUTPUT}/run_it${target_tag}_sampling.star" ]]; then
            target_complete=1
            break
        fi
        if ! kill -0 "${runner_pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    test "${target_complete}" -eq 1
    active_step=$(
        squeue --steps -j "${SLURM_JOB_ID}" -h -o '%i' \
            | awk -v job="${SLURM_JOB_ID}" '$0 ~ "^" job "\\.[0-9]+$" {step=$0} END {print step}'
    )
    if [[ -n "${active_step}" ]]; then
        scancel "${active_step}"
    else
        kill "${runner_pid}" 2>/dev/null || true
    fi
    set +e
    wait "${runner_pid}"
    set -e
    touch "${ROOT}/CAPTURE_STOPPED_AFTER_IT${target_tag}"
else
    (
        cd "${INPUT}"
        "${COMMAND[@]}"
    ) > "${OUTPUT}/runner.stdout" 2> "${OUTPUT}/runner.stderr"
fi
wall_s=$(( $(date +%s) - start_epoch ))

if [[ -z "${PART_IDS}" ]]; then
    for required in normalized_shifted_real masked_real masked_fourier_pre_optics_real masked_fourier_pre_optics_imag; do
        test -s "${CAPTURE}/preprocess_img0_${required}.bin"
    done
    test -s "${CAPTURE}/preprocess_img0_softmask_background.bin"
    PART_IDS=${PART_ID}
fi
for capture_part_id in ${PART_IDS//,/ }; do
    for suffix in sorted_weights sum_weight significant_weight eulers trans_xyz ctfs Fimg_unweighted_nomask; do
        test -s "${CAPTURE}/img0_part${capture_part_id}_storeWavg_${suffix}.bin"
    done
    if [[ "${CAPTURE_PARTICLE_BPREF}" = 1 ]]; then
        for suffix in bp_dims bp_real bp_imag bp_weight; do
            test -s "${CAPTURE}/img0_part${capture_part_id}_backproject_${suffix}.bin"
        done
    fi
done
for required in Minvsigma2 sigma2_noise sigma2_fudge; do
    test -s "${CAPTURE}/${required}.bin"
done
if [[ "${CAPTURE_MSTEP}" = 1 ]]; then
    for required in \
        pipe_it1_c0_bp_data_pre_reweight.bin pipe_it1_c0_bp_weight.bin \
        pipe_it1_c0_bp_data_h_pre_reweight.bin pipe_it1_c0_bp_weight_h.bin; do
        test -s "${CAPTURE}/${required}"
    done
fi
printf '{"job_id":"%s","wall_s":%d,"iteration":%d,"nr_iter_schedule":%d,"part_id":%d,"part_ids":"%s","part_count":%d,"threads":%d,"capture_mstep":%d,"capture_particle_bpref":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${CAPTURE_ITERATION}" "${NR_ITER_SCHEDULE}" "${PART_ID}" "${PART_IDS}" "$(tr ',' '\n' <<<"${PART_IDS}" | wc -l)" "${THREADS}" "${CAPTURE_MSTEP}" "${CAPTURE_PARTICLE_BPREF}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
