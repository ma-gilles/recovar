#!/usr/bin/env bash
#SBATCH --job-name=vdam-native-particle-panel
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-native-particle-panel-%j.out

set -euo pipefail

: "${VDAM_NATIVE_PARTICLE_PANEL_ROOT:?set a unique disposable panel root}"
: "${VDAM_PREPROCESS_INPUT_DIR:?set the materialized fixture data directory}"
: "${VDAM_RELION_BINARY:?set the pinned RELION capture binary}"
: "${VDAM_NATIVE_PARTICLE_PANEL_IDS:?set colon-separated native part IDs}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${SLURM_ARRAY_TASK_ID:?submit one isolated native capture per Slurm array task}"

PANEL_ROOT=${VDAM_NATIVE_PARTICLE_PANEL_ROOT}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
RUNNER=${REPO_ROOT}/scripts/run_vdam_relion_preprocess_capture.sbatch
CAPTURE_ITERATION=${VDAM_PREPROCESS_ITERATION:-2}
NR_ITER_SCHEDULE=${VDAM_PREPROCESS_NR_ITER_SCHEDULE:-200}
PART_IDS=${VDAM_NATIVE_PARTICLE_PANEL_IDS//:/,}
ORIGINAL_IDS=${VDAM_NATIVE_PARTICLE_PANEL_ORIGINAL_IDS:-}

test -f "${PANEL_ROOT}/SAFE_TO_DELETE"
if [[ ! "${PART_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "VDAM_NATIVE_PARTICLE_PANEL_IDS must be a colon-separated integer list" >&2
    exit 2
fi
if [[ -n "${ORIGINAL_IDS}" ]] && [[ ! "${ORIGINAL_IDS}" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
    echo "VDAM_NATIVE_PARTICLE_PANEL_ORIGINAL_IDS must be a colon-separated integer list" >&2
    exit 2
fi
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test -s "${VDAM_PREPROCESS_INPUT_DIR}/particles.star"
test -x "${VDAM_RELION_BINARY}"
test -f "${RUNNER}"
mkdir -p "${PANEL_ROOT}/capture" "${PANEL_ROOT}/provenance"

IFS=',' read -r -a capture_part_ids <<< "${PART_IDS}"
task_index=${SLURM_ARRAY_TASK_ID}
if (( task_index < 0 || task_index >= ${#capture_part_ids[@]} )); then
    echo "Slurm array index ${task_index} is outside [0, $((${#capture_part_ids[@]} - 1))]" >&2
    exit 2
fi
part_id=${capture_part_ids[task_index]}
original_index=""
if [[ -n "${ORIGINAL_IDS}" ]]; then
    IFS=':' read -r -a original_indices <<< "${ORIGINAL_IDS}"
    if (( ${#original_indices[@]} != ${#capture_part_ids[@]} )); then
        echo "native part-ID and RECOVAR original-index lists must have equal length" >&2
        exit 2
    fi
    original_index=${original_indices[task_index]}
fi

target_root=$(printf '%s/part-%03d' "${PANEL_ROOT}" "${part_id}")
mkdir -p "${target_root}"
touch "${target_root}/SAFE_TO_DELETE"
test -z "$(find "${target_root}" -mindepth 1 ! -name SAFE_TO_DELETE -print -quit)"
sha256sum "${REPO_ROOT}/scripts/run_vdam_native_particle_capture_panel.sbatch" \
    "${RUNNER}" "${VDAM_RELION_BINARY}" "${VDAM_PREPROCESS_INPUT_DIR}/particles.star" \
    > "${PANEL_ROOT}/provenance/static_inputs_task${task_index}.sha256"

VDAM_PREPROCESS_ROOT=${target_root} \
    VDAM_PREPROCESS_PART_ID=${part_id} \
    VDAM_PREPROCESS_ITERATION=${CAPTURE_ITERATION} \
    VDAM_PREPROCESS_NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE} \
    bash "${RUNNER}"
test -f "${target_root}/RUN_SUCCESS_${SLURM_JOB_ID}"
for source_path in "${target_root}"/capture/img0_part"${part_id}"_storeWavg_*.bin; do
    test -s "${source_path}"
    ln -s "${source_path}" "${PANEL_ROOT}/capture/$(basename "${source_path}")"
done

printf '{"array_job_id":"%s","array_task_id":%d,"job_id":"%s","git_head":"%s","iteration":%d,"nr_iter_schedule":%d,"native_part_id":%d,"recovar_original_index":"%s"}\n' \
    "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" "${task_index}" "${SLURM_JOB_ID}" \
    "${EXPECTED_REPO_HEAD}" "${CAPTURE_ITERATION}" "${NR_ITER_SCHEDULE}" \
    "${part_id}" "${original_index}" \
    > "${PANEL_ROOT}/provenance/run_task${task_index}.json"
touch "${PANEL_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}_${task_index}"
