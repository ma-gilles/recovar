#!/usr/bin/env bash

set -euo pipefail

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_execution_order_local_a100_20260810T1640ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_half1_execution_order_local_a100_20260810T1640ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
RELION=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion
CUDA_LIB=${REPO}/recovar/cuda/libcuda_backproject.so
ORDER_FILE=${ROOT}/provenance/native_half1_local_execution_order.txt

mkdir -p \
  "${ROOT}/control/accum" "${ROOT}/control/prejoin" "${ROOT}/control/output" \
  "${ROOT}/native_order/accum" "${ROOT}/native_order/prejoin" "${ROOT}/native_order/output" \
  "${ROOT}/analysis" "${ROOT}/provenance" \
  "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache" \
  "${RUNTIME}/jax_cache"
touch "${ROOT}/SAFE_TO_DELETE" "${RUNTIME}/SAFE_TO_DELETE"
test -z "$(find "${ROOT}/control/accum" "${ROOT}/control/prejoin" "${ROOT}/control/output" \
  "${ROOT}/native_order/accum" "${ROOT}/native_order/prejoin" "${ROOT}/native_order/output" \
  -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/relion_bind_build/shared
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER
unset RECOVAR_K1_COARSE_GAUSSIAN_FFI RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF
unset RECOVAR_K1_RELION_F32_COARSE_SUPPORT

cd "${REPO}"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; root=pathlib.Path.cwd().resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(root); assert pathlib.Path(jax.__file__).resolve().is_relative_to(root/'.pixi/envs/default'); assert len(jax.devices('gpu')) == 1 and 'A100' in jax.devices('gpu')[0].device_kind.upper(); assert cuda_backproject.cuda_available(); print(jax.devices())"

KEEP_STACK_INDICES=$("${PYTHON}" - "${RELION}/run_it001_data.star" <<'PY'
import sys
import starfile

particles = starfile.read(sys.argv[1])["particles"]
indices = [
    int(str(image_name).split("@", 1)[0]) - 1
    for image_name, subset in zip(
        particles["rlnImageName"], particles["rlnRandomSubset"], strict=True
    )
    if int(subset) == 1
]
assert len(indices) == 1490 and len(set(indices)) == len(indices)
print(",".join(map(str, indices)))
PY
)

"${PYTHON}" - "${RELION}/run_it001_data.star" "${ORDER_FILE}" <<'PY'
import sys
import numpy as np
import starfile
from recovar.relion_bind import _relion_bind_core as bind

particles = starfile.read(sys.argv[1])["particles"]
base_stack_indices = np.asarray(
    [
        int(str(image_name).split("@", 1)[0])
        for image_name, subset in zip(
            particles["rlnImageName"], particles["rlnRandomSubset"], strict=True
        )
        if int(subset) == 1
    ],
    dtype=np.int64,
)
assert base_stack_indices.size == 1490
assert np.unique(np.asarray(particles["rlnOpticsGroup"], dtype=np.int64)).size == 1
shuffle = np.asarray(
    bind.auto_refine_randomise_half_order(base_stack_indices.size, 1723),
    dtype=np.int64,
)
native_stack_order = base_stack_indices[shuffle]
recovar_local_stack_order = np.sort(base_stack_indices)
stack_to_local = {
    int(stack): int(local)
    for local, stack in enumerate(recovar_local_stack_order.tolist())
}
native_local_order = np.asarray(
    [stack_to_local[int(stack)] for stack in native_stack_order], dtype=np.int64
)
assert np.array_equal(np.sort(native_local_order), np.arange(1490, dtype=np.int64))
np.savetxt(sys.argv[2], native_local_order, fmt="%d")
PY
test "$(wc -l < "${ORDER_FILE}")" -eq 1490

run_arm() {
  local arm=$1
  local run_root=${ROOT}/${arm}
  export RECOVAR_BPREF_ACCUM_DUMP_DIR=${run_root}/accum
  export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${run_root}/prejoin
  export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case22_physical_it2_local_it1_half1_exact_operands_size60_${arm}_a100
  if [[ ${arm} == native_order ]]; then
    export RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE=${ORDER_FILE}
  else
    unset RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE
  fi
  local command=(
    "${PYTHON}" -m scripts.run_multi_iter_parity
    --relion_dir "${RELION}" --data_star "${DATA}/particles.star"
    --iter 1 --max_iter 1 --continuous-relion-noise-state --skip_final_iteration
    --image_batch_size 187 --rotation_block_size 8192 --image-fourier-backend relion_cuda
    --keep_stack_indices "${KEEP_STACK_INDICES}"
    --output_dir "${run_root}/output" --save_intermediates_dir "${run_root}/output/intermediates"
  )
  printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${arm}.sh"
  printf '\n' >> "${ROOT}/provenance/command_${arm}.sh"
  local start
  start=$(date +%s)
  "${command[@]}" > "${run_root}/output/run.log" 2>&1
  local wall_s=$(( $(date +%s) - start ))
  test -s "${run_root}/accum/recovar_bpref_accum_it001.npz"
  test -s "${run_root}/prejoin/recovar_bpref_prejoin_it001.npz"
  if [[ ${arm} == native_order ]]; then
    rg -q 'STRICT-PARITY diagnostic: executing K=1 BPref particles in the explicit local order' "${run_root}/output/run.log"
  else
    ! rg -q 'STRICT-PARITY diagnostic: executing K=1 BPref particles in the explicit local order' "${run_root}/output/run.log"
  fi
  ! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${run_root}/output/run.log"
  printf '{"arm":"%s","wall_s":%d,"physical_iteration":2,"half":1,"gpu_model":"A100","exact_bpref_operands":true}\n' \
    "${arm}" "${wall_s}" > "${ROOT}/provenance/walltime_${arm}.json"
}

git rev-parse HEAD > "${ROOT}/provenance/repo_head.txt"
git diff --binary | sha256sum > "${ROOT}/provenance/repo_diff.sha256"
sha256sum "${CUDA_LIB}" "${RELION}/run_it001_optimiser.star" "${ORDER_FILE}" > "${ROOT}/provenance/static_inputs.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi.txt"

run_arm control
run_arm native_order

"${PYTHON}" - "${ROOT}" <<'PY'
import json
import pathlib
import sys
import numpy as np

root = pathlib.Path(sys.argv[1])
paths = {
    arm: root / arm / "accum" / "recovar_bpref_accum_it001.npz"
    for arm in ("control", "native_order")
}
payloads = {arm: np.load(path, allow_pickle=False) for arm, path in paths.items()}

def rel_l2(a, b):
    aa = np.asarray(a)
    bb = np.asarray(b)
    denom = np.linalg.norm(bb.ravel())
    return float(np.linalg.norm((aa - bb).ravel()) / denom) if denom else 0.0

report = {
    "schema": "recovar-k1-bpref-execution-order-a100-ab-v1",
    "physical_iteration": 2,
    "half": 1,
    "intervention": "particle_execution_order_only",
    "gpu_model": "A100",
    "metrics": {},
}
for key in ("Ft_y_0", "Ft_ctf_0"):
    control = payloads["control"][key]
    candidate = payloads["native_order"][key]
    report["metrics"][key] = {
        "candidate_vs_control_relative_l2": rel_l2(candidate, control),
        "candidate_vs_control_max_abs": float(np.max(np.abs(candidate - control))),
        "bitwise_equal": bool(np.array_equal(candidate, control)),
    }
for payload in payloads.values():
    payload.close()
out = root / "analysis" / "K1_CASE22_IT2_HALF1_EXECUTION_ORDER_A100_AB.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

sha256sum \
  "${ROOT}/control/accum/recovar_bpref_accum_it001.npz" \
  "${ROOT}/native_order/accum/recovar_bpref_accum_it001.npz" \
  "${ROOT}/analysis/K1_CASE22_IT2_HALF1_EXECUTION_ORDER_A100_AB.json" \
  > "${ROOT}/provenance/outputs.sha256"
touch "${ROOT}/RUN_SUCCESS"
