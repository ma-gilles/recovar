#!/usr/bin/env bash
#SBATCH --job-name=k1c10-it2-scale
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"

START_ITER=${START_ITER:-1}
PHYSICAL_ITER=$((START_ITER + 1))
CAPTURE_ORIGINAL_INDICES=${CAPTURE_ORIGINAL_INDICES:-}
CAPTURE_RAW_SCORE_OPERANDS=${CAPTURE_RAW_SCORE_OPERANDS:-0}
case "${CAPTURE_RAW_SCORE_OPERANDS}" in
    0|1) ;;
    *) echo "CAPTURE_RAW_SCORE_OPERANDS must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${CAPTURE_RAW_SCORE_OPERANDS}" = 1 && -z "${CAPTURE_ORIGINAL_INDICES}" ]]; then
    echo "CAPTURE_RAW_SCORE_OPERANDS=1 requires CAPTURE_ORIGINAL_INDICES" >&2
    exit 2
fi

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
FIXED_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${FIXED_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE}/data
RELION=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=${CUDA_LIB_OVERRIDE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_exact_it15_final_20260823T0230ET/cuda/libcuda_backproject.so}
RELION_BIND=${FIXED_ROOT}/relion_bind_build/shared
LAUNCHER=${REPO}/scripts/run_k1_case10_it2_scale_boundary.sbatch
OUTPUT=${ROOT}/output
DUMP=${ROOT}/parity_dump

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -z "$(find "${OUTPUT}" "${DUMP}" -mindepth 1 -print -quit)"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu scripts/run_full_refinement.py scripts/run_multi_iter_parity.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
RELION_BIND_LIB=$(find "${RELION_BIND}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit)
test "$(sha256sum "${RELION_BIND_LIB}" | awk '{print $1}')" = 56acdddab6d25bd83aac5e338162c4819f85ec26f68511458618f143c2b604b7
printf -v start_tag '%03d' "${START_ITER}"
printf -v physical_tag '%03d' "${PHYSICAL_ITER}"
test -s "${RELION}/run_it${start_tag}_optimiser.star" \
    -a -s "${RELION}/run_it${physical_tag}_data.star"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${OUTPUT}/intermediates" "${DUMP}" "${ROOT}/logs" "${ROOT}/provenance"

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
export CUDA_VISIBLE_DEVICES=0

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_PARITY_DUMP_DIR=${DUMP}
export RECOVAR_PARITY_DUMP_VOLUME_DOWNSAMPLE=16
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

if [[ -n "${CAPTURE_ORIGINAL_INDICES}" ]]; then
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR=${ROOT}/contributions
    # The one-step continuation reports START_ITER to the contribution hook;
    # PHYSICAL_ITER is used only for the saved parity/output iteration tag.
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION=${START_ITER}
    # Case 10 firstiter scores at 58 but BPref scatters at model size 56.
    # The hook receives the score size and records the exact M-step radius.
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE=58
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES=${CAPTURE_ORIGINAL_INDICES}
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID=case10-it1-five-pose-boundary
    export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY=${ROOT}/provenance/image_names.npy
    export RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256=6abfc7829307ed63cc1b8b24ba99ec1afca25e5467876ef056034a9dd8779585
    export RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE=1
    mkdir -p "${RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR}"
fi
if [[ "${CAPTURE_RAW_SCORE_OPERANDS}" = 1 ]]; then
    export RECOVAR_PASS2_DUMP_DIR=${ROOT}/pass2
    export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${CAPTURE_ORIGINAL_INDICES}
    export RECOVAR_PASS2_DUMP_CURRENT_SIZE=58
    # run_multi_iter_parity exposes the one-step continuation's local loop
    # index to the pass-2 capture hook; START_ITER is also that local index.
    export RECOVAR_PASS2_DUMP_ITERATION=${START_ITER}
    export RECOVAR_PASS2_DUMP_CLASS=1
    export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
    mkdir -p "${RECOVAR_PASS2_DUMP_DIR}"
fi

cd "${REPO}"
git status --short --branch > "${ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu scripts/run_full_refinement.py scripts/run_multi_iter_parity.py \
    > "${ROOT}/provenance/source_diff_${SLURM_JOB_ID}.patch"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"

if [[ -n "${CAPTURE_ORIGINAL_INDICES}" ]]; then
    "${PYTHON}" - <<'PY'
from pathlib import Path
import os

import numpy as np
import starfile

data_dir = Path(os.environ["RECOVAR_K1_RELION_EXACT_CTF_STAR"]).resolve().parent
tables = starfile.read(str(data_dir / "particles.star"))
particles = tables["particles"] if isinstance(tables, dict) else tables
identities = []
for value in particles["rlnImageName"].astype(str):
    stack_index, stack_path = value.split("@", 1)
    resolved = Path(stack_path)
    if not resolved.is_absolute():
        resolved = (data_dir / resolved).resolve()
    identities.append(f"{int(stack_index)}@{resolved}")
np.save(
    os.environ["RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY"],
    np.asarray(identities, dtype=str),
)
PY
fi

"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert '.pixi/envs/default/' in str(Path(jax.__file__).resolve())
assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices())
PY

command=("${PYTHON}" -m scripts.run_multi_iter_parity
    --relion_dir "${RELION}"
    --data_star "${DATA}/particles.star"
    --iter "${START_ITER}" --max_iter 1
    --continuous-relion-noise-state
    --skip_final_iteration
    --diagnostic-retain-terminal-group-scale-state
    --image-fourier-backend relion_cuda
    --image_batch_size 187
    --rotation_block_size 8192
    --output_dir "${OUTPUT}"
    --save_intermediates_dir "${OUTPUT}/intermediates")
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
"${command[@]}" > "${ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1

dump_file=${DUMP}/iter_${physical_tag}.npz
test -s "${dump_file}"
test -s "${OUTPUT}/refinement_results.npz"
! rg -q 'RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent|Traceback' \
    "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
sha256sum "${dump_file}" "${OUTPUT}/refinement_results.npz" \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
if [[ -n "${CAPTURE_ORIGINAL_INDICES}" ]]; then
    "${PYTHON}" - "${RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR}" \
        "${CAPTURE_ORIGINAL_INDICES}" "${ROOT}/provenance/contribution_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
expected = {int(value) for value in sys.argv[2].split(",") if value}
observed = set()
records = []
for path in sorted(root.glob("bpref_contribution_rows_it*_h*_*.npz")):
    with np.load(path, allow_pickle=False) as archive:
        rows = [int(value) for value in archive["original_indices"]]
        observed.update(rows)
        records.append({
            "path": str(path.resolve()),
            "half": int(archive["half"]),
            "current_size": int(archive["current_size"]),
            "original_indices": rows,
        })
if observed != expected:
    raise RuntimeError(f"captured source rows {sorted(observed)} != expected {sorted(expected)}")
Path(sys.argv[3]).write_text(json.dumps({
    "schema": "recovar.em.k1_case10_it1_five_pose_capture.v1",
    "expected_original_indices": sorted(expected),
    "observed_original_indices": sorted(observed),
    "records": records,
}, indent=2, sort_keys=True) + "\n")
PY
    find "${RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR}" -type f -print0 \
        | sort -z | xargs -0 sha256sum >> "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
fi
if [[ "${CAPTURE_RAW_SCORE_OPERANDS}" = 1 ]]; then
    "${PYTHON}" - "${RECOVAR_PASS2_DUMP_DIR}" \
        "${CAPTURE_ORIGINAL_INDICES}" <<'PY'
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
expected = {int(value) for value in sys.argv[2].split(",") if value}
observed = set()
for path in sorted(root.glob("pass2_orig*_cs058.npz")):
    with np.load(path, allow_pickle=False) as archive:
        if "raw_operand_schema" not in archive:
            raise RuntimeError(f"raw score operands missing from {path}")
        observed.add(int(archive["original_index"]))
if observed != expected:
    raise RuntimeError(
        f"raw-score source rows {sorted(observed)} != expected {sorted(expected)}"
    )
PY
    find "${RECOVAR_PASS2_DUMP_DIR}" -type f -print0 \
        | sort -z | xargs -0 sha256sum >> "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
fi
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
