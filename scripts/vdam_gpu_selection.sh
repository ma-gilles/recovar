#!/usr/bin/env bash
# Shared fail-closed physical-GPU selection for VDAM Slurm runners.

vdam_visible_gpu_uuids() {
  nvidia-smi --query-gpu=uuid --format=csv,noheader |
    sed 's/[[:space:]]//g; /^$/d'
}

vdam_assert_target_gpu_allocated() {
  local target_uuid=${1:?target GPU UUID is required}
  local allocation_spec=${2:-}
  local query_output
  local token
  local uuid
  local found=0
  local used_cgroup_visible_fallback=0
  local -a allocation_tokens=()
  local -a resolved_uuids=()
  local -a token_uuids=()
  local -a visible_uuids=()

  if [[ -z "${allocation_spec}" ]]; then
    allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
  fi
  allocation_spec=${allocation_spec//[[:space:]]/}
  if [[ -z "${allocation_spec}" ]]; then
    echo "VDAM cannot prove the target GPU belongs to the Slurm allocation: no allocation spec" >&2
    return 76
  fi
  if [[ "${allocation_spec}" == ,* || "${allocation_spec}" == *, || "${allocation_spec}" == *,,* ]]; then
    echo "VDAM allocation spec contains an empty GPU selector: ${allocation_spec}" >&2
    return 76
  fi

  IFS=',' read -r -a allocation_tokens <<< "${allocation_spec}"
  for token in "${allocation_tokens[@]}"; do
    token=${token#gpu:}
    if [[ -z "${token}" ]]; then
      echo "VDAM allocation spec contains an empty GPU selector: ${allocation_spec}" >&2
      return 76
    fi
    if ! query_output=$(nvidia-smi -i "${token}" --query-gpu=uuid --format=csv,noheader); then
      # On some Slurm/cgroup configurations SLURM_JOB_GPUS retains the
      # node-global numeric ordinal while NVML exposes the sole allocated GPU
      # re-indexed as device zero.  A numeric selector is then unresolvable,
      # even though the cgroup-visible UUID still proves the allocation.  Only
      # accept that narrow case: one allocation token, one visible GPU, and an
      # exact match to the pinned target UUID.
      mapfile -t visible_uuids < <(vdam_visible_gpu_uuids)
      if [[ "${#allocation_tokens[@]}" -eq 1 \
        && "${token}" =~ ^[0-9]+$ \
        && "${#visible_uuids[@]}" -eq 1 \
        && "${visible_uuids[0]}" == "${target_uuid}" ]]; then
        query_output=${visible_uuids[0]}
        used_cgroup_visible_fallback=1
      else
        echo "VDAM cannot resolve allocated GPU selector ${token}: ${allocation_spec}" >&2
        return 76
      fi
    fi
    mapfile -t token_uuids < <(
      printf '%s\n' "${query_output}" | sed 's/[[:space:]]//g; /^$/d'
    )
    if [[ "${#token_uuids[@]}" -ne 1 ]]; then
      echo "VDAM allocated GPU selector ${token} did not resolve to one UUID" >&2
      return 76
    fi
    uuid=${token_uuids[0]}
    case "${uuid}" in
      GPU-*) ;;
      *) echo "VDAM allocated GPU selector ${token} resolved to invalid UUID ${uuid}" >&2; return 76 ;;
    esac
    resolved_uuids+=("${uuid}")
    if [[ "${uuid}" == "${target_uuid}" ]]; then
      found=1
    fi
  done

  VDAM_ALLOCATED_GPU_UUIDS_CSV=$(IFS=,; printf '%s' "${resolved_uuids[*]}")
  VDAM_ALLOCATION_SELECTOR_RESOLUTION=$(
    if [[ "${used_cgroup_visible_fallback}" == 1 ]]; then
      printf '%s' 'cgroup_single_visible_uuid'
    else
      printf '%s' 'direct_selector_query'
    fi
  )
  export VDAM_ALLOCATED_GPU_UUIDS_CSV VDAM_ALLOCATION_SELECTOR_RESOLUTION
  if [[ "${found}" != 1 ]]; then
    echo "VDAM_TARGET_GPU_NOT_ALLOCATED expected=${target_uuid} allocation_spec=${allocation_spec} resolved=${VDAM_ALLOCATED_GPU_UUIDS_CSV:-none}" >&2
    return 76
  fi
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
