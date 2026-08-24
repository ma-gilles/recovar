#!/usr/bin/env bash
#SBATCH --job-name=k1c10-r12334
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

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"

STAGE=${K1_ROW12334_STAGE:-fine}
case "${STAGE}" in fine|coarse) ;; *) echo "K1_ROW12334_STAGE must be fine or coarse" >&2; exit 2 ;; esac
TARGET_SOURCE_ROW=${K1_CASE10_TARGET_SOURCE_ROW:-12334}
case "${TARGET_SOURCE_ROW}" in
  ''|*[!0-9]*) echo "K1_CASE10_TARGET_SOURCE_ROW must be a non-negative integer" >&2; exit 2 ;;
esac
TARGET_STACK_INDEX=$((TARGET_SOURCE_ROW + 1))
INITIAL_HALF1_MRC=${K1_CASE10_INITIAL_HALF1_MRC:-}
INITIAL_HALF2_MRC=${K1_CASE10_INITIAL_HALF2_MRC:-}
if [[ -n "${INITIAL_HALF1_MRC}" || -n "${INITIAL_HALF2_MRC}" ]]; then
    if [[ -z "${INITIAL_HALF1_MRC}" || -z "${INITIAL_HALF2_MRC}" ]]; then
        echo "K1_CASE10_INITIAL_HALF1_MRC and K1_CASE10_INITIAL_HALF2_MRC must be supplied together" >&2
        exit 2
    fi
    test "${INITIAL_HALF1_MRC:0:1}" = / -a -s "${INITIAL_HALF1_MRC}"
    test "${INITIAL_HALF2_MRC:0:1}" = / -a -s "${INITIAL_HALF2_MRC}"
fi
if [[ "${STAGE}" = coarse && "${TARGET_SOURCE_ROW}" != 12334 ]]; then
    echo "The coarse diagnostic's fixed target tuple is only defined for source row 12334" >&2
    exit 2
fi
ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
FIXED_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${FIXED_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE}/data
RELION=${K1_ROW12334_RELION_DIR:-${CASE}/relion_ref}
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=${K1_ROW12334_CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_stage_split_20260821T1600ET/cuda/libcuda_backproject.so}
RELION_BIND=${FIXED_ROOT}/relion_bind_build/shared
NATIVE_ROOT=${K1_CASE10_NATIVE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_native_fine_stack12335_it2_20260821T2200ET}
NATIVE_FACTOR=$(find "${NATIVE_ROOT}/capture" -maxdepth 1 -type f \
    -name "*stack${TARGET_STACK_INDEX}_img0_class1.bpre-v2.bin" -print -quit)
NATIVE_FINE=$(find "${NATIVE_ROOT}/capture" -maxdepth 1 -type f \
    -name "*stack${TARGET_STACK_INDEX}_class1.fine-score-v1.bin" -print -quit)
LAUNCHER=${REPO}/scripts/run_k1_case10_row12334_it2_same_input.sbatch
OUTPUT=${ROOT}/output
PASS2=${ROOT}/pass2
COARSE=${ROOT}/coarse
ANALYSIS=${ROOT}/analysis

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -z "$(find "${OUTPUT}" "${PASS2}" "${COARSE}" "${ANALYSIS}" -mindepth 1 -print -quit)"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu scripts/run_full_refinement.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
RELION_BIND_LIB=$(find "${RELION_BIND}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit)
test "$(sha256sum "${RELION_BIND_LIB}" | awk '{print $1}')" = 56acdddab6d25bd83aac5e338162c4819f85ec26f68511458618f143c2b604b7
test -s "${NATIVE_FACTOR}" -a -s "${NATIVE_FINE}"
test -s "${RELION}/run_it001_optimiser.star" -a -s "${RELION}/run_it002_data.star"

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
    "${OUTPUT}/intermediates" "${PASS2}" "${COARSE}" "${ANALYSIS}" "${ROOT}/logs" "${ROOT}/provenance"

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
# Continuation replays intentionally disable the fresh-order guard. Pin the
# guarded fresh-K=1 production coarse scorer explicitly so both the coarse
# and downstream fine dump measure the uninterrupted trajectory path.
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
if [[ "${STAGE}" = fine ]]; then
    export RECOVAR_PASS2_DUMP_DIR=${PASS2}
    export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${TARGET_SOURCE_ROW}
    export RECOVAR_PASS2_DUMP_CURRENT_SIZE=58
    export RECOVAR_PASS2_DUMP_ITERATION=1
    export RECOVAR_PASS2_DUMP_CLASS=1
    export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
    export RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET=1
else
    export RECOVAR_SIGNIFICANCE_DUMP_DIR=${COARSE}
    export RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=12334
    export RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE=58
    export RECOVAR_SIGNIFICANCE_DUMP_ITERATION=2
    export RECOVAR_SIGNIFICANCE_DUMP_TARGET_HALF=1
    export RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET=1
    export RECOVAR_SIGNIFICANCE_DUMP_PROJECTION_ROTATIONS=28071
fi
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
git status --short --branch > "${ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu scripts/run_full_refinement.py > "${ROOT}/provenance/source_diff_${SLURM_JOB_ID}.patch"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"

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
    --iter 1 --max_iter 1
    --continuous-relion-noise-state
    --skip_final_iteration
    --image-fourier-backend relion_cuda
    --image_batch_size 187
    --rotation_block_size 8192
    --output_dir "${OUTPUT}"
    --save_intermediates_dir "${OUTPUT}/intermediates")
if [[ -n "${INITIAL_HALF1_MRC}" ]]; then
    command+=(
        --initial-half1-mrc "${INITIAL_HALF1_MRC}"
        --initial-half2-mrc "${INITIAL_HALF2_MRC}"
    )
fi
if [[ -n "${K1_ROW12334_NORMALIZATION_OVERRIDE_SPEC:-}" ]]; then
    command+=(
        --normalization-factor-override-at-iteration
        "${K1_ROW12334_NORMALIZATION_OVERRIDE_SPEC}"
    )
fi
if [[ -n "${K1_ROW12334_INITIAL_NORMALIZATION_OVERRIDE_SPEC:-}" ]]; then
    command+=(
        --normalization-factor-override
        "${K1_ROW12334_INITIAL_NORMALIZATION_OVERRIDE_SPEC}"
    )
fi
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
set +e
"${command[@]}" > "${ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1
command_status=$?
set -e
wall_s=$(( $(date +%s) - start_epoch ))

! rg -q 'RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' \
    "${ROOT}/logs/run_${SLURM_JOB_ID}.log"

if [[ "${STAGE}" = fine ]]; then
    test "${command_status}" -ne 0
    printf -v target_source_padded '%06d' "${TARGET_SOURCE_ROW}"
    capture=${PASS2}/pass2_orig${target_source_padded}_cs058.npz
    report=${ANALYSIS}/K1_CASE10_ROW${target_source_padded}_IT2_FINE_STAGES.json
    test -s "${capture}"
    test "$(find "${PASS2}" -maxdepth 1 -type f -name 'pass2_orig*_cs058.npz' | wc -l)" -eq 1
    rg -q 'Sparse K=1 pass-2 stop-after-dump requested .*target-set progress 1/1' \
        "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
    rg -q 'Pass2DumpComplete: requested RECOVAR pass-2 dump target set was written' \
        "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
    "${PYTHON}" -m scripts.analyze_k1_fine_score_stages \
        --native-factor "${NATIVE_FACTOR}" \
        --native-fine-score "${NATIVE_FINE}" \
        --recovar-capture "${capture}" \
        --physical-image-size 384 \
        --top-count 30 \
        --output-json "${report}" \
        > "${ROOT}/logs/analyze_${SLURM_JOB_ID}.log" 2>&1
    sha256sum "${capture}" "${NATIVE_FACTOR}" "${NATIVE_FINE}" "${report}" \
        > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
    if [[ -n "${INITIAL_HALF1_MRC}" ]]; then
        sha256sum "${INITIAL_HALF1_MRC}" "${INITIAL_HALF2_MRC}" \
            >> "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
    fi
else
    test "${command_status}" -eq 0
    capture=${COARSE}/significance_orig012334_it002_cs058.npz
    report=${ANALYSIS}/K1_CASE10_ROW12334_IT2_COARSE_TARGET.json
    test -s "${capture}"
    test "$(find "${COARSE}" -maxdepth 1 -type f -name 'significance_orig*_it002_cs058.npz' | wc -l)" -eq 1
    rg -q 'coarse-significance dump completed; stopping before pass-2/M-step work' \
        "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
    "${PYTHON}" - "${capture}" "${report}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

capture_path = Path(sys.argv[1]).resolve()
report_path = Path(sys.argv[2]).resolve()
target_rotation = 28071
target_translation = 25
with np.load(capture_path, allow_pickle=False) as payload:
    n_rot = int(payload["n_rot"])
    n_trans = int(payload["n_trans"])
    assert n_rot == 36864 and n_trans == 29
    flat_index = target_rotation * n_trans + target_translation
    weights = np.asarray(payload["weights_full"], dtype=np.float64).reshape(-1)
    significant = np.asarray(payload["significant_mask"], dtype=bool).reshape(-1)
    raw = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64).reshape(-1)
    combined = np.asarray(payload["scores_with_prior_per_class"], dtype=np.float64).reshape(-1)
    assert weights.size == significant.size == raw.size == combined.size == n_rot * n_trans
    order = np.argsort(-weights, kind="stable")
    rank_by_flat = np.empty(order.size, dtype=np.int64)
    rank_by_flat[order] = np.arange(1, order.size + 1, dtype=np.int64)
    selected_weights = weights[significant]
    report = {
        "status": "complete",
        "capture": str(capture_path),
        "source_row": int(payload["original_index"]),
        "local_index": int(payload["local_index"]),
        "diagnostic_iteration": int(payload["debug_iteration"]),
        "physical_iteration": 2,
        "current_size": int(payload["current_size"]),
        "coarse_rotation": target_rotation,
        "coarse_translation": target_translation,
        "flat_index": flat_index,
        "target_selected": bool(significant[flat_index]),
        "target_rank": int(rank_by_flat[flat_index]),
        "target_weight": float(weights[flat_index]),
        "target_raw_log_score": float(raw[flat_index]),
        "target_combined_log_score": float(combined[flat_index]),
        "normalization_log_z": float(payload["normalization_log_z"]),
        "best_score": float(payload["best_score"]),
        "max_posterior": float(payload["max_posterior"]),
        "significant_count": int(np.count_nonzero(significant)),
        "least_selected_weight": float(np.min(selected_weights)),
        "target_over_least_selected_weight": float(weights[flat_index] / np.min(selected_weights)),
        "target_cumulative_mass_inclusive": float(np.sum(weights[order[: rank_by_flat[flat_index]]], dtype=np.float64)),
    }
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
    sha256sum "${capture}" "${report}" > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
fi
printf '{"job_id":"%s","wall_s":%d,"stage":"%s","source_row":12334,"physical_iteration":2,"same_input":"RELION iteration-1 state"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${STAGE}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
