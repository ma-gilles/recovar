#!/usr/bin/env bash
#SBATCH --job-name=vdam-bpref-chunk
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-bpref-chunk-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE panel root}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPEATS=${REPEATS:-4}
VDAM_BPREF_PARTICLE_CHUNK_ARMS=${VDAM_BPREF_PARTICLE_CHUNK_ARMS:-default,3}
PANEL_RUNNER=${REPO_ROOT}/scripts/run_vdam_expected_accuracy_ab_panel.sbatch

case "${REPEATS}" in
  ''|*[!0-9]*) echo "REPEATS must be a positive integer" >&2; exit 2 ;;
esac
test "${REPEATS}" -ge 1
test ! -e "${OUTPUT_ROOT}"
test -x "${PANEL_RUNNER}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"

IFS=',' read -r -a requested_arms <<< "${VDAM_BPREF_PARTICLE_CHUNK_ARMS}"
if (( ${#requested_arms[@]} < 1 )); then
  echo "VDAM_BPREF_PARTICLE_CHUNK_ARMS must select default or positive integers" >&2
  exit 2
fi
seen_arms=()
for arm in "${requested_arms[@]}"; do
  if [[ "${arm}" != default ]] && [[ ! "${arm}" =~ ^[1-9][0-9]*$ ]]; then
    echo "unsupported BPref particle chunk arm: ${arm}" >&2
    exit 2
  fi
  if [[ " ${seen_arms[*]} " == *" ${arm} "* ]]; then
    echo "duplicate BPref particle chunk arm: ${arm}" >&2
    exit 2
  fi
  seen_arms+=("${arm}")
done

mkdir -p "${OUTPUT_ROOT}/provenance"
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"
printf '%s\n' "${VDAM_BPREF_PARTICLE_CHUNK_ARMS}" \
  > "${OUTPUT_ROOT}/provenance/requested_arms.csv"

for repeat in $(seq 1 "${REPEATS}"); do
  if (( ${#requested_arms[@]} > 1 && repeat % 2 == 0 )); then
    arms=()
    for ((arm_index=${#requested_arms[@]}-1; arm_index>=0; arm_index--)); do
      arms+=("${requested_arms[arm_index]}")
    done
  else
    arms=("${requested_arms[@]}")
  fi
  for arm in "${arms[@]}"; do
    run_root=$(printf '%s/%s/repeat-%02d' "${OUTPUT_ROOT}" "${arm}" "${repeat}")
    printf '%02d\t%s\n' "${repeat}" "${arm}" \
      >> "${OUTPUT_ROOT}/provenance/execution_order.tsv"
    (
      if [[ "${arm}" == default ]]; then
        unset RECOVAR_EXACT_LOCAL_SOURCE_BPREF_PARTICLE_CHUNK_SIZE
      else
        export RECOVAR_EXACT_LOCAL_SOURCE_BPREF_PARTICLE_CHUNK_SIZE=${arm}
      fi
      OUTPUT_ROOT=${run_root} \
      REPEATS=1 \
      VDAM_EXPECTED_ACCURACY_MODES=baseline \
      VDAM_CAPTURE_LOCAL_SCORE=0 \
      VDAM_CAPTURE_FUSED_SCORES=0 \
      VDAM_CAPTURE_COARSE_SCORE=0 \
        bash "${PANEL_RUNNER}"
    )
  done
done

touch "${OUTPUT_ROOT}/COMPLETED"
