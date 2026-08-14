#!/usr/bin/env bash
#SBATCH --job-name=k1-c22-h1-accum
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_native_exec_order_20260810T1745ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_native_exec_order_20260810T1745ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:30:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"
: "${EXPECTED_REPO_DIFF_SHA256:?submission must pin repository diff hash}"
: "${EXPECTED_CUDA_SHA256:?submission must pin CUDA library hash}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_native_exec_order_20260810T1745ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_half1_native_exec_order_20260810T1745ET
ROOT=${RECOVAR_K1_BPREF_RUN_ROOT:-${ROOT}}
RUNTIME=${RECOVAR_K1_BPREF_RUNTIME_ROOT:-${RUNTIME}}
ORDER_MODE=${RECOVAR_K1_BPREF_EXECUTION_ORDER_MODE:-native}
ORDER_CHUNK_SIZE=${RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE:-1}
if [[ ${ORDER_MODE} != native && ${ORDER_MODE} != native_grouped && ${ORDER_MODE} != default_single && ${ORDER_MODE} != standard_double ]]; then
  echo "RECOVAR_K1_BPREF_EXECUTION_ORDER_MODE must be native, native_grouped, default_single, or standard_double" >&2
  exit 2
fi
if ! [[ ${ORDER_CHUNK_SIZE} =~ ^[1-9][0-9]*$ ]]; then
  echo "RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE must be a positive integer" >&2
  exit 2
fi
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
RELION=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion
CUDA_LIB=${RECOVAR_K1_BPREF_CUDA_LIB:-${REPO}/recovar/cuda/libcuda_backproject.so}
LAUNCHER=${REPO}/scripts/run_k1_case22_half1_exact_accum.sbatch
ORDER_FILE=${ROOT}/provenance/native_half1_local_execution_order.txt

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -z "$(find "${ROOT}/accum" "${ROOT}/prejoin" "${ROOT}/output" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

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
if [[ ${ORDER_MODE} == standard_double ]]; then
  unset RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE RECOVAR_K1_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE
  export RECOVAR_RELION_X_HALF_MSTEP_DOUBLE=1
else
  export RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE=${ORDER_FILE}
  export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=${ORDER_CHUNK_SIZE}
  if [[ ${ORDER_MODE} == native_grouped ]]; then
    export RECOVAR_K1_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE=1
  else
    unset RECOVAR_K1_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE
  fi
  unset RECOVAR_RELION_X_HALF_MSTEP_DOUBLE
fi
export RECOVAR_BPREF_ACCUM_DUMP_DIR=${ROOT}/accum
export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${ROOT}/prejoin
# The replay starts from RELION iteration 1, but run_multi_iter_parity invokes
# the single physical-iteration-2 update at local loop index 0.  The dump API
# labels that update as local iteration 1 (iteration + 1).
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=1
export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case22_physical_it2_local_it1_half1_exact_operands_size60
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER
unset RECOVAR_K1_COARSE_GAUSSIAN_FFI RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF
unset RECOVAR_K1_RELION_F32_COARSE_SUPPORT

cd "${REPO}"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; root=pathlib.Path.cwd().resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(root); assert pathlib.Path(jax.__file__).resolve().is_relative_to(root/'.pixi/envs/default'); assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper(); assert cuda_backproject.cuda_available(); print(jax.devices())"

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

"${PYTHON}" - "${RELION}/run_it001_data.star" "${ORDER_FILE}" "${ORDER_MODE}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import starfile

from recovar.relion_bind import _relion_bind_core as bind

particles = starfile.read(sys.argv[1])["particles"]
base_stack_indices = np.asarray(
    [
        int(str(image_name).split("@", 1)[0])
        for image_name, subset in zip(
            particles["rlnImageName"],
            particles["rlnRandomSubset"],
            strict=True,
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
recovar_local_stack_order = np.sort(base_stack_indices)
if sys.argv[3] in {"native", "native_grouped"}:
    native_stack_order = base_stack_indices[shuffle]
    stack_to_local = {
        int(stack): int(local)
        for local, stack in enumerate(recovar_local_stack_order.tolist())
    }
    execution_order = np.asarray(
        [stack_to_local[int(stack)] for stack in native_stack_order],
        dtype=np.int64,
    )
elif sys.argv[3] == "default_single":
    membership_dir = Path(
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "em_k1_case22_rotation_mass_all_20260810T1600ET/membership"
    )
    shards = sorted(membership_dir.glob("bpref_membership_*.npz"))
    assert len(shards) == 1411
    local_index_chunks = []
    for path in shards:
        with np.load(path, allow_pickle=False) as archive:
            local_index_chunks.append(
                np.asarray(archive["local_indices"], dtype=np.int64)
            )
    execution_order = np.concatenate(local_index_chunks)
else:
    execution_order = np.arange(base_stack_indices.size, dtype=np.int64)
assert np.array_equal(np.sort(execution_order), np.arange(1490, dtype=np.int64))
np.savetxt(sys.argv[2], execution_order, fmt="%d")
PY
test "$(wc -l < "${ORDER_FILE}")" -eq 1490

COMMAND=(
  "${PYTHON}" -m scripts.run_multi_iter_parity
  --relion_dir "${RELION}" --data_star "${DATA}/particles.star"
  --iter 1 --max_iter 1 --continuous-relion-noise-state --skip_final_iteration
  --image_batch_size 187 --rotation_block_size 8192 --image-fourier-backend relion_cuda
  --keep_stack_indices "${KEEP_STACK_INDICES}"
  --output_dir "${ROOT}/output" --save_intermediates_dir "${ROOT}/output/intermediates"
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" "${RELION}/run_it001_optimiser.star" "${ORDER_FILE}" > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

start=$(date +%s)
"${COMMAND[@]}" > "${ROOT}/output/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))

ACCUM=${ROOT}/accum/recovar_bpref_accum_it001.npz
PREJOIN=${ROOT}/prejoin/recovar_bpref_prejoin_it001.npz
test -s "${ACCUM}" -a -s "${PREJOIN}"
"${PYTHON}" - "${PREJOIN}" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as payload:
    assert int(payload["iteration"]) == 1 and int(payload["current_size"]) == 60
    assert np.count_nonzero(payload["Ft_y_0"]) > 0
    assert np.count_nonzero(payload["Ft_ctf_0"]) > 0
    assert np.count_nonzero(payload["Ft_y_1"]) == 0
    assert np.count_nonzero(payload["Ft_ctf_1"]) == 0
PY
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${ROOT}/output/run.log"
if [[ ${ORDER_MODE} == standard_double ]]; then
  ! rg -q 'STRICT-PARITY diagnostic: executing K=1 BPref particles in the explicit local order' "${ROOT}/output/run.log"
else
  rg -q 'STRICT-PARITY diagnostic: executing K=1 BPref particles in the explicit local order' "${ROOT}/output/run.log"
fi
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":2,"half":1,"particles":1490,"exact_bpref_operands":true,"execution_order_mode":"%s","particles_per_bucket_call":%d,"rotation_chunk_size":8192}\n' "${SLURM_JOB_ID}" "${wall_s}" "${ORDER_MODE}" "${ORDER_CHUNK_SIZE}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${ACCUM}" "${PREJOIN}" > "${ROOT}/provenance/accum_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
