#!/usr/bin/env bash
#SBATCH --job-name=vdam-matched-bpref
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:00:00

set -euo pipefail

: "${VDAM_PANEL_ROOT:?set a fresh SAFE_TO_DELETE panel root}"
: "${VDAM_REPO_ROOT:?set the immutable science worktree}"
: "${VDAM_EXPECTED_HEAD:?pin the immutable science commit}"
: "${VDAM_DATA_DIR:?set the frozen materialized VDAM fixture directory}"
: "${VDAM_RELION_BINARY:?set the isolated instrumented RELION binary}"
: "${VDAM_CUDA_LIBRARY:?set the qualified CUDA replay library}"
: "${VDAM_EXPECTED_CUDA_SHA256:?pin the qualified CUDA replay library}"
: "${VDAM_EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding}"
: "${VDAM_INPUT_DIRECTORY:?set the sealed host-replay input directory}"
: "${VDAM_NATIVE_DIRECTORY:?set the matched native M-step directory}"
: "${VDAM_RECOVAR_DIRECTORY:?set the matched RECOVAR M-step directory}"
: "${VDAM_TOPOLOGY:?set the native worker topology capture}"
: "${VDAM_TOPOLOGY_DATA_STAR:?set the native topology data STAR}"
: "${VDAM_NATIVE_PANEL_DIRECTORY:?set the native Euler/weight panels}"
: "${RECOVAR_VDAM_EXACT_NATIVE_PTX:?set the exact RELION PTX}"
: "${VDAM_EXPECTED_EXACT_PTX_SHA256:?pin the exact RELION PTX}"

REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}
test "${REPEAT_COUNT}" -ge 2
test -f "${VDAM_PANEL_ROOT}/SAFE_TO_DELETE"
test -z "$(find "${VDAM_PANEL_ROOT}" -mindepth 1 -maxdepth 1 \
  ! -name SAFE_TO_DELETE \
  ! -name "slurm-${SLURM_JOB_ID}.out" \
  ! -name "slurm-${SLURM_JOB_ID}.err" \
  -print -quit)"
cd "${VDAM_REPO_ROOT}"
test "$(git rev-parse HEAD)" = "${VDAM_EXPECTED_HEAD}"
test -z "$(git status --porcelain=v1 --untracked-files=no)"
test -s "${VDAM_DATA_DIR}/particles.star"
test -x "${VDAM_RELION_BINARY}"
test -f "${VDAM_CUDA_LIBRARY}"
test "$(sha256sum "${VDAM_CUDA_LIBRARY}" | awk '{print $1}')" = \
  "${VDAM_EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RECOVAR_VDAM_EXACT_NATIVE_PTX}" | awk '{print $1}')" = \
  "${VDAM_EXPECTED_EXACT_PTX_SHA256}"
test -s "${VDAM_TOPOLOGY}"
test -s "${VDAM_TOPOLOGY_DATA_STAR}"
test -d "${VDAM_NATIVE_PANEL_DIRECTORY}"

physical_gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
case "${physical_gpu_uuid}" in
  GPU-*) ;;
  *) echo "failed to resolve one physical GPU UUID: ${physical_gpu_uuid}" >&2; exit 2 ;;
esac

for repeat in $(seq 1 "${REPEAT_COUNT}"); do
  repeat_tag=$(printf '%02d' "${repeat}")
  output_root=${VDAM_PANEL_ROOT}/repeat-${repeat_tag}
  mkdir -p "${output_root}"
  touch "${output_root}/SAFE_TO_DELETE"
  env \
    OUTPUT_ROOT="${output_root}" \
    DATA_DIR="${VDAM_DATA_DIR}" \
    EXPECTED_REPO_HEAD="${VDAM_EXPECTED_HEAD}" \
    EXPECTED_CUDA_SHA256="${VDAM_EXPECTED_CUDA_SHA256}" \
    EXPECTED_RELION_BIND_SHA256="${VDAM_EXPECTED_RELION_BIND_SHA256}" \
    VDAM_RELION_BINARY="${VDAM_RELION_BINARY}" \
    RECOVAR_CUDA_LIB_SOURCE="${VDAM_CUDA_LIBRARY}" \
    REPO_ROOT="${VDAM_REPO_ROOT}" \
    TARGET_ITERATION=1 \
    NR_ITER_SCHEDULE=200 \
    RELION_THREADS=8 \
    RANDOM_SEED=0 \
    TARGET_GPU_UUID="${physical_gpu_uuid}" \
    VDAM_NATIVE_ONLY=1 \
    bash scripts/run_vdam_fullschedule_mstep_boundary.sbatch \
      > "${output_root}/driver.stdout" \
      2> "${output_root}/driver.stderr"
done

python - "${VDAM_PANEL_ROOT}/submission_provenance.json" \
  "${VDAM_EXPECTED_HEAD}" "${physical_gpu_uuid}" \
  "${VDAM_EXPECTED_CUDA_SHA256}" "${RECOVAR_VDAM_EXACT_NATIVE_PTX}" \
  "${VDAM_EXPECTED_EXACT_PTX_SHA256}" "${REPEAT_COUNT}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

output, head, gpu_uuid, cuda_sha, ptx_path, ptx_sha, repeats = sys.argv[1:]
Path(output).write_text(
    json.dumps(
        {
            "schema": "recovar.vdam_replay_repeat_panel_submission.v2",
            "submitted_at": datetime.now().astimezone().isoformat(),
            "science_head": head,
            "physical_gpu_uuid": gpu_uuid,
            "execution_policy": "one allocation, one physical H100, sequential native and replay arms",
            "cuda_library_sha256": cuda_sha,
            "exact_native_ptx": ptx_path,
            "exact_native_ptx_sha256": ptx_sha,
            "repeat_count_per_topology": int(repeats),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

arm_number=0
for kind in private shared; do
  shared_accumulators=0
  if [[ "${kind}" = shared ]]; then
    shared_accumulators=1
  fi
  for repeat in $(seq 1 "${REPEAT_COUNT}"); do
    arm_number=$((arm_number + 1))
    arm_tag=$(printf '%02d' "${arm_number}")
    output_root=${VDAM_PANEL_ROOT}/arm-${arm_tag}-${kind}
    env \
      VDAM_REPO_ROOT="${VDAM_REPO_ROOT}" \
      VDAM_EXPECTED_HEAD="${VDAM_EXPECTED_HEAD}" \
      VDAM_INPUT_DIRECTORY="${VDAM_INPUT_DIRECTORY}" \
      VDAM_OUTPUT_DIRECTORY="${output_root}" \
      VDAM_LIBRARY="${VDAM_CUDA_LIBRARY}" \
      VDAM_NATIVE_DIRECTORY="${VDAM_NATIVE_DIRECTORY}" \
      VDAM_RECOVAR_DIRECTORY="${VDAM_RECOVAR_DIRECTORY}" \
      VDAM_TARGET_GPU_UUID="${physical_gpu_uuid}" \
      RECOVAR_VDAM_EXACT_NATIVE_PTX="${RECOVAR_VDAM_EXACT_NATIVE_PTX}" \
      VDAM_TOPOLOGY="${VDAM_TOPOLOGY}" \
      VDAM_TOPOLOGY_DATA_STAR="${VDAM_TOPOLOGY_DATA_STAR}" \
      VDAM_NATIVE_PANEL_DIRECTORY="${VDAM_NATIVE_PANEL_DIRECTORY}" \
      VDAM_MERGE_CALLBACKS=1 \
      VDAM_SHARED_ACCUMULATORS="${shared_accumulators}" \
      bash scripts/run_vdam_worker_private_host_replay.sbatch \
        > "${VDAM_PANEL_ROOT}/arm-${arm_tag}-${kind}.driver.stdout" \
        2> "${VDAM_PANEL_ROOT}/arm-${arm_tag}-${kind}.driver.stderr"
  done
done

mkdir -p "${VDAM_PANEL_ROOT}/analysis"
pixi run python -m scripts.audit_vdam_bpref_replay_repeat_panel \
  --panel-root "${VDAM_PANEL_ROOT}" \
  --repeat-count "${REPEAT_COUNT}" \
  --expected-gpu-uuid "${physical_gpu_uuid}" \
  --output "${VDAM_PANEL_ROOT}/analysis/replay_repeat_distribution.json" \
  > "${VDAM_PANEL_ROOT}/analysis/replay_repeat_distribution.stdout"
pixi run python -m scripts.audit_vdam_native_bpref_repeat_panel \
  --native-panel-root "${VDAM_PANEL_ROOT}" \
  --replay-panel-root "${VDAM_PANEL_ROOT}" \
  --repeat-count "${REPEAT_COUNT}" \
  --iteration 1 \
  --output "${VDAM_PANEL_ROOT}/analysis/native_vs_replay_bpref_repeat.json" \
  > "${VDAM_PANEL_ROOT}/analysis/native_vs_replay_bpref_repeat.stdout"

sha256sum \
  scripts/run_vdam_matched_native_replay_panel.sbatch \
  scripts/run_vdam_fullschedule_mstep_boundary.sbatch \
  scripts/run_vdam_worker_private_host_replay.sbatch \
  "${VDAM_PANEL_ROOT}/submission_provenance.json" \
  "${VDAM_PANEL_ROOT}/analysis/replay_repeat_distribution.json" \
  "${VDAM_PANEL_ROOT}/analysis/native_vs_replay_bpref_repeat.json" \
  > "${VDAM_PANEL_ROOT}/matched_panel_evidence.sha256"
touch "${VDAM_PANEL_ROOT}/MATCHED_PANEL_SUCCESS"
