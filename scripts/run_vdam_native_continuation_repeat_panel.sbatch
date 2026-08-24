#!/usr/bin/env bash
#SBATCH --job-name=vdam-native-repeat
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

set -euo pipefail

: "${VDAM_RELION_REPEAT_ROOT:?set a unique disposable panel root}"
: "${VDAM_RELION_CONT_INPUT_DIR:?set the materialized fixture data directory}"
: "${VDAM_RELION_CONT_OPTIMISER:?set the exact incoming optimiser.star}"

PANEL_ROOT=${VDAM_RELION_REPEAT_ROOT}
REPEAT_COUNT=${VDAM_RELION_REPEAT_COUNT:-8}
RUNNER=${VDAM_RELION_REPEAT_RUNNER:-scripts/run_vdam_relion_continuation_capture.sbatch}

test -f "${PANEL_ROOT}/SAFE_TO_DELETE"
test "${REPEAT_COUNT}" -ge 1
test -f "${RUNNER}"

for repeat_index in $(seq 1 "${REPEAT_COUNT}"); do
    repeat_root=$(printf '%s/repeat-%02d' "${PANEL_ROOT}" "${repeat_index}")
    mkdir -p "${repeat_root}"
    touch "${repeat_root}/SAFE_TO_DELETE"
    VDAM_RELION_CONT_ROOT=${repeat_root} \
        VDAM_RELION_CONT_CAPTURE=1 \
        bash "${RUNNER}"
done

touch "${PANEL_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
