#!/usr/bin/env bash
#SBATCH --job-name=vdam-native-bpref-prefix
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-native-bpref-prefix-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a unique disposable output root}"
: "${DATA_DIR:?set the materialized VDAM fixture data directory}"
: "${VDAM_RELION_BPREF_BINARY:?set the pinned RELION BPref-prefix binary}"
: "${VDAM_RELION_BPREF_SOURCE:?set the corresponding RELION source worktree}"
: "${VDAM_NATIVE_PART_IDS:?set colon-separated native part IDs}"
: "${EXPECTED_RELION_BINARY_SHA256:?pin the RELION binary hash}"
: "${EXPECTED_RELION_SOURCE_HEAD:?pin the RELION source head}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?pin the RELION source diff hash}"

TARGET_ITERATION=${TARGET_ITERATION:-1}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
PART_IDS=${VDAM_NATIVE_PART_IDS//:/,}
RELION_RUNTIME=${VDAM_RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
RELION_OUTPUT=${OUTPUT_ROOT}/relion
CAPTURE=${OUTPUT_ROOT}/native_bpref_prefix
PROVENANCE=${OUTPUT_ROOT}/provenance

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -s "${DATA_DIR}/particles.star"
test -x "${VDAM_RELION_BPREF_BINARY}"
test "${PART_IDS}" = "$(printf '%s' "${PART_IDS}" | tr -cd '0-9,')"
test "${TARGET_ITERATION}" -eq 1
test "${NR_ITER_SCHEDULE}" -gt "${TARGET_ITERATION}"
test "$(sha256sum "${VDAM_RELION_BPREF_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(git -C "${VDAM_RELION_BPREF_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_SOURCE_HEAD}"
test "$(git -C "${VDAM_RELION_BPREF_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
mkdir -p "${RELION_OUTPUT}" "${CAPTURE}" "${PROVENANCE}" "${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/tmp"
for directory in "${RELION_OUTPUT}" "${CAPTURE}" "${PROVENANCE}"; do
    test -z "$(find "${directory}" -mindepth 1 -print -quit)"
done

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export TMPDIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/tmp
export RELION_DUMP_K1_BPREF_PREFIX_DIR=${CAPTURE}
export RELION_DUMP_K1_BPREF_PREFIX_PART_IDS=${PART_IDS}
export RELION_DUMP_K1_BPREF_PREFIX_ITER=${TARGET_ITERATION}

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

COMMAND=(
    srun --mpi=pmix --ntasks=1 --cpus-per-task=2 --cpu-bind=none "${VDAM_RELION_BPREF_BINARY}"
    --o "${RELION_OUTPUT}/run" --iter "${NR_ITER_SCHEDULE}" --grad --denovo_3dref
    --grad_write_iter 1 --i particles.star --ctf --K 1 --sym C1
    --flatten_solvent --zero_mask --dont_combine_weights_via_disc --pool 3 --pad 1
    --particle_diameter 200 --oversampling 1 --healpix_order 1 --offset_range 6.0
    --offset_step 2.0 --auto_sampling --tau2_fudge 4.0 --random_seed 0 --j 1 --gpu 0
)
printf '%q ' "${COMMAND[@]}" > "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
git -C "${VDAM_RELION_BPREF_SOURCE}" diff --binary | sha256sum > "${PROVENANCE}/relion_source_diff.sha256"
git -C "${VDAM_RELION_BPREF_SOURCE}" rev-parse HEAD > "${PROVENANCE}/relion_source_head.txt"
sha256sum "${VDAM_RELION_BPREF_BINARY}" > "${PROVENANCE}/relion_binary.sha256"

start_epoch=$(date +%s)
set +e
(
    cd "${DATA_DIR}"
    "${COMMAND[@]}"
) > "${RELION_OUTPUT}/runner.stdout" 2> "${RELION_OUTPUT}/runner.stderr" &
runner_pid=$!
set -e
target_tag=$(printf '%03d' "${TARGET_ITERATION}")
target_complete=0
for _poll_index in $(seq 1 600); do
    if [[ -s "${RELION_OUTPUT}/run_it${target_tag}_sampling.star" ]]; then
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
    scancel "${active_step}" 2>/dev/null || kill "${runner_pid}" 2>/dev/null || true
else
    kill "${runner_pid}" 2>/dev/null || true
fi
set +e
wait "${runner_pid}"
set -e
wall_s=$(( $(date +%s) - start_epoch ))

expected_count=$(tr ',' '\n' <<< "${PART_IDS}" | wc -l)
actual_count=$(find "${CAPTURE}" -maxdepth 1 -type f -name '*_bpref_prefix_metadata.bin' | wc -l)
test "${actual_count}" -eq "${expected_count}"
for suffix in real imag weight operand_image_real operand_image_imag operand_translation_x \
    operand_translation_y operand_translation_z operand_weights operand_minvsigma2 \
    operand_ctf operand_eulers operand_controls metadata; do
    test "$(find "${CAPTURE}" -maxdepth 1 -type f -name "*_bpref_prefix_${suffix}.bin" | wc -l)" \
        -eq "${expected_count}"
done
printf '{"job_id":"%s","wall_s":%d,"iteration":%d,"nr_iter_schedule":%d,"part_ids":"%s","part_count":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${TARGET_ITERATION}" "${NR_ITER_SCHEDULE}" \
    "${PART_IDS}" "${expected_count}" > "${PROVENANCE}/run.json"
touch "${OUTPUT_ROOT}/CAPTURE_STOPPED_AFTER_IT${target_tag}"
touch "${OUTPUT_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
