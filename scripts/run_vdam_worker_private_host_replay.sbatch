#!/bin/bash
#SBATCH --job-name=vdam-worker-private
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${VDAM_REPO_ROOT:?set the immutable science worktree}"
: "${VDAM_EXPECTED_HEAD:?set the immutable science commit}"
: "${VDAM_INPUT_DIRECTORY:?set the sealed host-replay input directory}"
: "${VDAM_OUTPUT_DIRECTORY:?set a fresh output directory}"
: "${VDAM_LIBRARY:?set the qualified CUDA replay library}"
: "${VDAM_NATIVE_DIRECTORY:?set the matched native M-step directory}"
: "${VDAM_RECOVAR_DIRECTORY:?set the matched RECOVAR M-step directory}"
: "${VDAM_TARGET_GPU_UUID:?pin the physical H100}"
: "${RECOVAR_VDAM_EXACT_NATIVE_PTX:?set the exact RELION PTX}"
VDAM_TOPOLOGY=${VDAM_TOPOLOGY:-}
VDAM_TOPOLOGY_DATA_STAR=${VDAM_TOPOLOGY_DATA_STAR:-}
VDAM_NATIVE_PANEL_DIRECTORY=${VDAM_NATIVE_PANEL_DIRECTORY:-}
VDAM_MERGE_CALLBACKS=${VDAM_MERGE_CALLBACKS:-0}
case "${VDAM_MERGE_CALLBACKS}" in
  0|1) ;;
  *) echo "VDAM_MERGE_CALLBACKS must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -n "${VDAM_TOPOLOGY}" || -n "${VDAM_TOPOLOGY_DATA_STAR}" ]]; then
  test -s "${VDAM_TOPOLOGY}"
  test -s "${VDAM_TOPOLOGY_DATA_STAR}"
fi
if [[ -n "${VDAM_NATIVE_PANEL_DIRECTORY}" ]]; then
  test -d "${VDAM_NATIVE_PANEL_DIRECTORY}"
  test -n "${VDAM_TOPOLOGY}"
fi

cd "${VDAM_REPO_ROOT}"
test "$(git rev-parse HEAD)" = "${VDAM_EXPECTED_HEAD}"
test -f "${VDAM_LIBRARY}"
test -f "${RECOVAR_VDAM_EXACT_NATIVE_PTX}"

visible_gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
if [[ "${visible_gpu_uuid}" != "${VDAM_TARGET_GPU_UUID}" ]]; then
  echo "VDAM_TARGET_GPU_MISS expected=${VDAM_TARGET_GPU_UUID} observed=${visible_gpu_uuid}" >&2
  exit 75
fi

mkdir -p "${VDAM_OUTPUT_DIRECTORY}"
touch "${VDAM_OUTPUT_DIRECTORY}/SAFE_TO_DELETE"
runtime_root="${VDAM_OUTPUT_DIRECTORY}/runtime/${SLURM_JOB_ID}"
mkdir -p "${runtime_root}/tmp" "${runtime_root}/cache"
export TMPDIR="${runtime_root}/tmp"
export CUDA_CACHE_PATH="${runtime_root}/cache"

start_s=$(date +%s)
command=(pixi run python -m scripts.run_vdam_worker_private_host_replay \
  --input-directory "${VDAM_INPUT_DIRECTORY}" \
  --output-directory "${VDAM_OUTPUT_DIRECTORY}/replay" \
  --library "${VDAM_LIBRARY}" \
  --native-directory "${VDAM_NATIVE_DIRECTORY}" \
  --recovar-directory "${VDAM_RECOVAR_DIRECTORY}" \
  --worker-count 8 \
  --iteration 1)
if [[ -n "${VDAM_TOPOLOGY}" ]]; then
  command+=(--topology "${VDAM_TOPOLOGY}" \
    --topology-data-star "${VDAM_TOPOLOGY_DATA_STAR}")
fi
if [[ -n "${VDAM_NATIVE_PANEL_DIRECTORY}" ]]; then
  command+=(--native-panel-directory "${VDAM_NATIVE_PANEL_DIRECTORY}")
fi
if [[ "${VDAM_MERGE_CALLBACKS}" == 1 ]]; then
  command+=(--merge-callbacks)
fi
"${command[@]}" \
  > "${VDAM_OUTPUT_DIRECTORY}/runner.stdout" \
  2> "${VDAM_OUTPUT_DIRECTORY}/runner.stderr"
end_s=$(date +%s)

python - "${VDAM_OUTPUT_DIRECTORY}/provenance.json" \
  "${SLURM_JOB_ID}" "${visible_gpu_uuid}" "${VDAM_EXPECTED_HEAD}" \
  "$((end_s - start_s))" "${VDAM_TOPOLOGY}" "${VDAM_TOPOLOGY_DATA_STAR}" \
  "${VDAM_NATIVE_PANEL_DIRECTORY}" "${VDAM_MERGE_CALLBACKS}" <<'PY'
import json
import sys
from pathlib import Path

output, job_id, gpu_uuid, head, wall_s, topology, data_star, panels, merged = sys.argv[1:]
Path(output).write_text(
    json.dumps(
        {
            "job_id": job_id,
            "gpu_uuid": gpu_uuid,
            "science_head": head,
            "wall_seconds": int(wall_s),
            "native_topology": topology or None,
            "native_topology_data_star": data_star or None,
            "native_panel_directory": panels or None,
            "callbacks_merged_in_native_launch_order": merged == "1",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
touch "${VDAM_OUTPUT_DIRECTORY}/RUN_SUCCESS_${SLURM_JOB_ID}"
