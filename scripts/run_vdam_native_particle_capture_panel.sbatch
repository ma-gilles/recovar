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

PANEL_ROOT=${VDAM_NATIVE_PARTICLE_PANEL_ROOT}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
RUNNER=${REPO_ROOT}/scripts/run_vdam_relion_preprocess_capture.sbatch
CAPTURE_ITERATION=${VDAM_PREPROCESS_ITERATION:-2}
NR_ITER_SCHEDULE=${VDAM_PREPROCESS_NR_ITER_SCHEDULE:-200}
PART_IDS=${VDAM_NATIVE_PARTICLE_PANEL_IDS//:/,}

test -f "${PANEL_ROOT}/SAFE_TO_DELETE"
if [[ ! "${PART_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "VDAM_NATIVE_PARTICLE_PANEL_IDS must be a colon-separated integer list" >&2
    exit 2
fi
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test -s "${VDAM_PREPROCESS_INPUT_DIR}/particles.star"
test -x "${VDAM_RELION_BINARY}"
test -f "${RUNNER}"
mkdir -p "${PANEL_ROOT}/capture" "${PANEL_ROOT}/provenance"
test -z "$(find "${PANEL_ROOT}/capture" "${PANEL_ROOT}/provenance" -mindepth 1 -print -quit)"

sha256sum "${REPO_ROOT}/scripts/run_vdam_native_particle_capture_panel.sbatch" \
    "${RUNNER}" "${VDAM_RELION_BINARY}" "${VDAM_PREPROCESS_INPUT_DIR}/particles.star" \
    > "${PANEL_ROOT}/provenance/static_inputs.sha256"

IFS=',' read -r -a capture_part_ids <<< "${PART_IDS}"
for part_id in "${capture_part_ids[@]}"; do
    target_root=$(printf '%s/part-%03d' "${PANEL_ROOT}" "${part_id}")
    mkdir -p "${target_root}"
    touch "${target_root}/SAFE_TO_DELETE"
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
done

printf '{"job_id":"%s","git_head":"%s","iteration":%d,"nr_iter_schedule":%d,"part_ids":"%s"}\n' \
    "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${CAPTURE_ITERATION}" \
    "${NR_ITER_SCHEDULE}" "${PART_IDS}" > "${PANEL_ROOT}/provenance/run.json"
touch "${PANEL_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
