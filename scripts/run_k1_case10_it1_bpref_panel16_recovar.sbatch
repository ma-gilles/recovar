#!/usr/bin/env bash
#SBATCH --job-name=k1-bpref-prefix-r
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:35:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"
: "${ENABLE_LIVE_INITIAL_NOISE:?ENABLE_LIVE_INITIAL_NOISE is required}"
: "${CASE_SOURCE:?CASE_SOURCE is required}"
: "${CASE_NAME:?CASE_NAME is required}"
: "${PREFIX_SELECTION_JSON:?PREFIX_SELECTION_JSON is required}"
: "${EXPECTED_SELECTION_SHA256:?EXPECTED_SELECTION_SHA256 is required}"
: "${EXPECTED_DATA_STAR_SHA256:?EXPECTED_DATA_STAR_SHA256 is required}"
: "${PARTICLE_STACK:?PARTICLE_STACK is required}"
: "${EXPECTED_STACK_SIZE:?EXPECTED_STACK_SIZE is required}"
: "${CURRENT_SIZE:?CURRENT_SIZE is required}"
: "${SEED:?SEED is required}"

case "${ENABLE_LIVE_INITIAL_NOISE}" in
    0|1) ;;
    *) echo "ENABLE_LIVE_INITIAL_NOISE must be 0 or 1" >&2; exit 2 ;;
esac
[[ "${CURRENT_SIZE}" =~ ^[0-9]+$ ]]
[[ "${SEED}" =~ ^[0-9]+$ ]]
[[ "${EXPECTED_STACK_SIZE}" =~ ^[0-9]+$ ]]
CAPTURE_HIGH_PRECISION_CONTRIBUTIONS=${CAPTURE_HIGH_PRECISION_CONTRIBUTIONS:-0}
case "${CAPTURE_HIGH_PRECISION_CONTRIBUTIONS}" in
    0|1) ;;
    *) echo "CAPTURE_HIGH_PRECISION_CONTRIBUTIONS must be 0 or 1" >&2; exit 2 ;;
esac
CAPTURE_BPREF_BOUNDARY=${CAPTURE_BPREF_BOUNDARY:-0}
case "${CAPTURE_BPREF_BOUNDARY}" in
    0|1) ;;
    *) echo "CAPTURE_BPREF_BOUNDARY must be 0 or 1" >&2; exit 2 ;;
esac
CAPTURE_ACCUMULATOR_DELTA=${CAPTURE_ACCUMULATOR_DELTA:-0}
case "${CAPTURE_ACCUMULATOR_DELTA}" in
    0|1) ;;
    *) echo "CAPTURE_ACCUMULATOR_DELTA must be 0 or 1" >&2; exit 2 ;;
esac
ACCUMULATOR_DELTA_HALF=${ACCUMULATOR_DELTA_HALF:-0}
if [[ "${CAPTURE_ACCUMULATOR_DELTA}" == 1 && \
      "${ACCUMULATOR_DELTA_HALF}" != 1 && "${ACCUMULATOR_DELTA_HALF}" != 2 ]]; then
    echo "ACCUMULATOR_DELTA_HALF must be 1 or 2 when capture is enabled" >&2
    exit 2
fi
OFFSET_RANGE_PIXELS=${OFFSET_RANGE_PIXELS:-3.0}
OFFSET_STEP_PIXELS=${OFFSET_STEP_PIXELS:-1.0}
TRANSLATION_ANGLE_SCALE=${TRANSLATION_ANGLE_SCALE:-}

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
CASE=${CASE_SOURCE}
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case10_it1_bpref_panel16_recovar.sbatch
CUDA_LIB=${CUDA_LIB_OVERRIDE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_bpref_fma_44d770de3_20260823T1938ET/cuda/libcuda_backproject.so}
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared
CONTRIBUTIONS=${ROOT}/recovar/contributions
ACCUM=${ROOT}/recovar/accum
PREJOIN=${ROOT}/recovar/prejoin
ACCUMULATOR_DELTA=${ROOT}/recovar/accumulator_delta
IMAGE_NAMES=${ROOT}/provenance/image_names.npy

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -s "${PREFIX_SELECTION_JSON}"
test "$(sha256sum "${PREFIX_SELECTION_JSON}" | awk '{print $1}')" = "${EXPECTED_SELECTION_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu recovar/cuda_backproject.py scripts/run_full_refinement.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = "${EXPECTED_DATA_STAR_SHA256}"
test -s "${PARTICLE_STACK}"
test "$(stat -Lc %s "${PARTICLE_STACK}")" = "${EXPECTED_STACK_SIZE}"
readarray -t selection_values < <("${PYTHON}" - "${PREFIX_SELECTION_JSON}" "${CASE_NAME}" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    selection = json.load(stream)
assert selection["schema"] == "recovar-k1-bpref-prefix-selection-v2"
assert selection["case"] == sys.argv[2]
assert selection["physical_iteration"] == 1
half1 = selection["half1"]["original_indices"]
half2 = selection["half2"]["original_indices"]
assert len(half1) > 0 and len(half1) == len(half2)
assert len(set(half1 + half2)) == len(half1) + len(half2)
print(",".join(str(value) for value in half1 + half2))
print(len(half1) + len(half2))
PY
)
ORIGINAL_INDICES=${selection_values[0]}
SELECTION_COUNT=${selection_values[1]}

mkdir -p "${ROOT}/recovar/output" "${ROOT}/recovar/pass2" "${ROOT}/logs" \
    "${ROOT}/provenance"
for directory in "${ROOT}/recovar/output" "${ROOT}/recovar/pass2"; do
    test -z "$(find "${directory}" -mindepth 1 -print -quit)"
done
mkdir -p "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache" \
    "${RUNTIME}/pycache" "${RUNTIME}/jax_cache" "${RUNTIME}/cuda_cache"
cp "${PREFIX_SELECTION_JSON}" "${ROOT}/provenance/prefix_selection.json"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
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
export CUDA_VISIBLE_DEVICES=0
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_CUDA_CACHE_DIR=${RUNTIME}/cuda_cache
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=${ENABLE_LIVE_INITIAL_NOISE}
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF=${FIRSTITER_FUSED_BPREF:-1}
if [[ -n "${TRANSLATION_ANGLE_SCALE}" ]]; then
    export RECOVAR_K1_RELION_TRANSLATION_ANGLE_SCALE=${TRANSLATION_ANGLE_SCALE}
fi
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_PASS2_DUMP_DIR=${ROOT}/recovar/pass2
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${ORIGINAL_INDICES}
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=${CURRENT_SIZE}
export RECOVAR_PASS2_DUMP_CLASS=1
unset RECOVAR_PASS2_DUMP_RAW_OPERANDS
if [[ "${CAPTURE_HIGH_PRECISION_CONTRIBUTIONS}" == 1 ]]; then
    mkdir -p "${CONTRIBUTIONS}"
    test -z "$(find "${CONTRIBUTIONS}" -mindepth 1 -print -quit)"
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR=${CONTRIBUTIONS}
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION=1
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE=${CURRENT_SIZE}
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES=${ORIGINAL_INDICES}
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID=${CASE_NAME}-it1-prefix
    export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY=${IMAGE_NAMES}
    export RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256=$(sha256sum "${PARTICLE_STACK}" | awk '{print $1}')
    export RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE=1
fi
if [[ "${CAPTURE_BPREF_BOUNDARY}" == 1 ]]; then
    mkdir -p "${ACCUM}" "${PREJOIN}"
    test -z "$(find "${ACCUM}" "${PREJOIN}" -mindepth 1 -print -quit)"
    export RECOVAR_BPREF_ACCUM_DUMP_DIR=${ACCUM}
    export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${PREJOIN}
    export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=1
    export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=${CASE_NAME}-it1-prefix
fi
if [[ "${CAPTURE_ACCUMULATOR_DELTA}" == 1 ]]; then
    mkdir -p "${ACCUMULATOR_DELTA}"
    test -z "$(find "${ACCUMULATOR_DELTA}" -mindepth 1 -print -quit)"
    if [[ -n "${ACCUMULATOR_DELTA_SELECTION_JSON:-}" ]]; then
        test -s "${ACCUMULATOR_DELTA_SELECTION_JSON}"
        ACCUMULATOR_DELTA_INDICES=$("${PYTHON}" - "${ACCUMULATOR_DELTA_SELECTION_JSON}" "${ACCUMULATOR_DELTA_HALF}" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    selection = json.load(stream)
assert selection["schema"] == "recovar-k1-bpref-prefix-selection-v2"
print(",".join(str(value) for value in selection[f"half{sys.argv[2]}"]["original_indices"]))
PY
)
    elif [[ -z "${ACCUMULATOR_DELTA_INDICES:-}" ]]; then
        if [[ "${ACCUMULATOR_DELTA_HALF}" == 1 ]]; then
            ACCUMULATOR_DELTA_INDICES=0,14275,28628,42698,57062,71399,85827,99997
        else
            ACCUMULATOR_DELTA_INDICES=3,14301,28522,43039,57231,71452,85593,99999
        fi
    fi
    ACCUMULATOR_DELTA_COUNT=$(awk -F, '{print NF}' <<< "${ACCUMULATOR_DELTA_INDICES}")
    export RECOVAR_BPREF_ACCUMULATOR_DELTA_DUMP_DIR=${ACCUMULATOR_DELTA}
    export RECOVAR_BPREF_ACCUMULATOR_DELTA_ORIGINAL_INDICES=${ACCUMULATOR_DELTA_INDICES}
    export RECOVAR_BPREF_ACCUMULATOR_DELTA_ITERATION=1
    export RECOVAR_BPREF_ACCUMULATOR_DELTA_HALF=${ACCUMULATOR_DELTA_HALF}
    export RECOVAR_BPREF_ACCUMULATOR_DELTA_MAX_PARTICLES=${ACCUMULATOR_DELTA_COUNT}
    export RECOVAR_BPREF_ACCUMULATOR_DELTA_MAX_BYTES=${ACCUMULATOR_DELTA_MAX_BYTES:-805306368}
    export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY=${IMAGE_NAMES}
fi
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
git -C "${REPO}" status --short --branch > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu recovar/cuda_backproject.py scripts/run_full_refinement.py \
    > "${ROOT}/provenance/recovar_source_diff_${SLURM_JOB_ID}.patch"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" "${DATA}/particles.star" \
    "${ROOT}/provenance/prefix_selection.json" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
if [[ "${CAPTURE_HIGH_PRECISION_CONTRIBUTIONS}" == 1 || \
      "${CAPTURE_ACCUMULATOR_DELTA}" == 1 ]]; then
    "${PYTHON}" - "${DATA}/particles.star" "${DATA}" "${IMAGE_NAMES}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import starfile

particles = starfile.read(sys.argv[1])
particles = particles["particles"] if isinstance(particles, dict) else particles
data_root = Path(sys.argv[2]).resolve()
image_names = []
for value in particles["rlnImageName"]:
    stack_index, stack_path = str(value).split("@", 1)
    path = Path(stack_path)
    if not path.is_absolute():
        path = (data_root / path).resolve()
    image_names.append(f"{stack_index}@{path}")
np.save(sys.argv[3], np.asarray(image_names, dtype=str))
PY
fi
"${PYTHON}" - <<'PY'
from pathlib import Path

import jax
import recovar
from recovar import cuda_backproject

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert Path(jax.__file__).resolve().is_relative_to(root / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
PY

recovar_command=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${ROOT}/recovar/output" --max_iter 1 --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4
    --offset_range "${OFFSET_RANGE_PIXELS}" --offset_step "${OFFSET_STEP_PIXELS}"
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed "${SEED}" --perturb_seed "${SEED}"
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --save_intermediates_dir "${ROOT}/recovar/output/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${CONTROL}/run_it000_data.star"
    --relion_optimiser "${CONTROL}/run_it000_optimiser.star"
    --relion_init_dir "${CONTROL}"
)
printf '%q ' "${recovar_command[@]}" > "${ROOT}/provenance/command_recovar_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_recovar_${SLURM_JOB_ID}.sh"
recovar_start=$(date +%s)
"${recovar_command[@]}" > "${ROOT}/recovar/output/run.log" 2>&1
printf '{"wall_s":%d}\n' "$(( $(date +%s) - recovar_start ))" \
    > "${ROOT}/provenance/walltime_recovar_${SLURM_JOB_ID}.json"
test "$(find "${ROOT}/recovar/pass2" -maxdepth 1 -name '*.npz' | wc -l)" -eq "${SELECTION_COUNT}"
if [[ "${CAPTURE_HIGH_PRECISION_CONTRIBUTIONS}" == 1 ]]; then
    test "$(find "${CONTRIBUTIONS}" -maxdepth 1 -name 'bpref_contribution_rows_it001_h*_*.npz' | wc -l)" -ge 1
    "${PYTHON}" - "${CONTRIBUTIONS}" "${ORIGINAL_INDICES}" <<'PY'
import sys
from pathlib import Path

import numpy as np

expected = {int(value) for value in sys.argv[2].split(",")}
seen = set()
for path in Path(sys.argv[1]).glob("bpref_contribution_rows_it001_h*_*.npz"):
    with np.load(path, allow_pickle=False) as values:
        assert bool(values["high_precision_operand_bundle"])
        seen.update(np.asarray(values["original_indices"], dtype=np.int64).tolist())
assert seen == expected, (seen, expected)
PY
fi
if [[ "${CAPTURE_BPREF_BOUNDARY}" == 1 ]]; then
    test -s "${ACCUM}/recovar_bpref_accum_it001.npz"
    test -s "${PREJOIN}/recovar_bpref_prejoin_it001.npz"
fi
if [[ "${CAPTURE_ACCUMULATOR_DELTA}" == 1 ]]; then
    test "$(find "${ACCUMULATOR_DELTA}" -maxdepth 1 -name 'bpref_accumulator_delta_it001_h*.npz' | wc -l)" -eq "${ACCUMULATOR_DELTA_COUNT}"
fi
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' \
    "${ROOT}/recovar/output/run.log"

find "${ROOT}/recovar" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","case":"%s","particle_count":%s,"current_size":%s,"seed":%s,"live_initial_noise":%s,"high_precision_contributions":%s,"bpref_boundary":%s,"accumulator_delta":%s,"accumulator_delta_half":%s,"offset_range_pixels":%s,"offset_step_pixels":%s,"translation_angle_scale":"%s"}\n' \
    "${SLURM_JOB_ID}" "${CASE_NAME}" "${SELECTION_COUNT}" "${CURRENT_SIZE}" "${SEED}" "${ENABLE_LIVE_INITIAL_NOISE}" \
    "${CAPTURE_HIGH_PRECISION_CONTRIBUTIONS}" "${CAPTURE_BPREF_BOUNDARY}" \
    "${CAPTURE_ACCUMULATOR_DELTA}" "${ACCUMULATOR_DELTA_HALF}" \
    "${OFFSET_RANGE_PIXELS}" "${OFFSET_STEP_PIXELS}" "${TRANSLATION_ANGLE_SCALE}" \
    > "${ROOT}/provenance/completion_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
