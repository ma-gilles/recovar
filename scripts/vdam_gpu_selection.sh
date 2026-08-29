#!/usr/bin/env bash
# Shared fail-closed physical-GPU selection for VDAM Slurm runners.

vdam_visible_gpu_uuids() {
  nvidia-smi --query-gpu=uuid --format=csv,noheader |
    sed 's/[[:space:]]//g; /^$/d'
}

vdam_select_target_gpu() {
  local target_uuid=${1:-}
  local miss_hold_seconds=${2:-0}
  local uuid
  local found=0
  local -a visible_gpu_uuids=()

  mapfile -t visible_gpu_uuids < <(vdam_visible_gpu_uuids)
  VDAM_VISIBLE_GPU_UUIDS_CSV=$(IFS=,; printf '%s' "${visible_gpu_uuids[*]}")

  if [[ -n "${target_uuid}" ]]; then
    for uuid in "${visible_gpu_uuids[@]}"; do
      if [[ "${uuid}" == "${target_uuid}" ]]; then
        found=1
        break
      fi
    done
    if [[ "${found}" != 1 ]]; then
      echo "VDAM_TARGET_GPU_MISS expected=${target_uuid} observed=${VDAM_VISIBLE_GPU_UUIDS_CSV:-none}" >&2
      if [[ "${miss_hold_seconds}" -gt 0 ]]; then
        sleep "${miss_hold_seconds}"
      fi
      return 75
    fi
    VDAM_SELECTED_GPU_UUID=${target_uuid}
  else
    if [[ "${#visible_gpu_uuids[@]}" != 1 ]]; then
      echo "VDAM requires one visible GPU when TARGET_GPU_UUID is unset; observed=${VDAM_VISIBLE_GPU_UUIDS_CSV:-none}" >&2
      return 2
    fi
    VDAM_SELECTED_GPU_UUID=${visible_gpu_uuids[0]}
  fi

  case "${VDAM_SELECTED_GPU_UUID}" in
    GPU-*) ;;
    *) echo "invalid GPU UUID: ${VDAM_SELECTED_GPU_UUID}" >&2; return 2 ;;
  esac
  export CUDA_VISIBLE_DEVICES=${VDAM_SELECTED_GPU_UUID}
  export RECOVAR_SELECTED_GPU_UUID=${VDAM_SELECTED_GPU_UUID}
  export VDAM_SELECTED_GPU_UUID VDAM_VISIBLE_GPU_UUIDS_CSV
}

vdam_verify_selected_gpu() {
  local selected_uuid=${1:?selected GPU UUID is required}
  local uuid

  while IFS= read -r uuid; do
    if [[ "${uuid}" == "${selected_uuid}" ]]; then
      return 0
    fi
  done < <(vdam_visible_gpu_uuids)
  echo "VDAM selected GPU disappeared: ${selected_uuid}" >&2
  return 2
}
