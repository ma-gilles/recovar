#!/usr/bin/env bash
#SBATCH --job-name=k1c04-rec-it1-ops
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:00:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
RELION=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=${SOURCE_ROOT}/cuda/libcuda_backproject.so
RELION_BIND=${SOURCE_ROOT}/relion_bind_build/shared
LAUNCHER=${REPO}/scripts/run_k1_case04_recovar_it1_operand_capture.sbatch
OUTPUT=${ROOT}/output
CONTRIBUTIONS=${ROOT}/contributions
ACCUM=${ROOT}/accum
PREJOIN=${ROOT}/prejoin
IMAGE_NAMES=${ROOT}/provenance/image_names.npy
CAPTURE_INDICES=${CAPTURE_ORIGINAL_INDICES:-99886,71881,89328,77986,60268,16799,88741,18493,55303,80124,32732,63798,28614,92534,20635,29514,58368,75859}
DEVICE_SIGNATURE_DIR=${BPREF_DEVICE_SIGNATURE_DIR:-}
STOP_AFTER_TARGET=${BPREF_STOP_AFTER_TARGET:-0}

if [[ "${STOP_AFTER_TARGET}" != 0 && "${STOP_AFTER_TARGET}" != 1 ]]; then
    echo "BPREF_STOP_AFTER_TARGET must be 0 or 1" >&2
    exit 2
fi

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO}" status --porcelain=v1)"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = 3855570600674a87995ddf1be75b5f69c8d30d84f11880ff5ff9ae9699c14d85
RELION_BIND_LIB=$(find "${RELION_BIND}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit)
test "$(sha256sum "${RELION_BIND_LIB}" | awk '{print $1}')" = 56acdddab6d25bd83aac5e338162c4819f85ec26f68511458618f143c2b604b7
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 0a90714711cdf0a1820d46fa82460d828b87e1ecbd2e49de645170863546ac98
test "$(stat -Lc %s "${DATA}/particles.256.mrcs")" = 26214401024
test -s "${RELION}/run_it000_data.star" -a -s "${RELION}/run_it000_optimiser.star"
test -z "$(find "${OUTPUT}" "${CONTRIBUTIONS}" "${ACCUM}" "${PREJOIN}" -mindepth 1 -print -quit)"
if [[ -n "${DEVICE_SIGNATURE_DIR}" ]]; then
    mkdir -p "${DEVICE_SIGNATURE_DIR}"
    test -z "$(find "${DEVICE_SIGNATURE_DIR}" -mindepth 1 -print -quit)"
fi

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

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
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR=${CONTRIBUTIONS}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION=1
export RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE=56
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES=${CAPTURE_INDICES}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID=case04-it1-physical-panel
export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY=${IMAGE_NAMES}
export RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256=b0f1430d5884e71b9d09e9ff955b19502c2df84ee0b411ab40ffa4f8cc090ef2
export RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE=1
export RECOVAR_BPREF_ACCUM_DUMP_DIR=${ACCUM}
export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${PREJOIN}
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=1
export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case04-it1-physical-panel
if [[ -n "${DEVICE_SIGNATURE_DIR}" ]]; then
    export RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR=${DEVICE_SIGNATURE_DIR}
fi
if [[ "${STOP_AFTER_TARGET}" == 1 ]]; then
    export RECOVAR_BPREF_CONTRIBUTION_STOP_AFTER_TARGET=1
fi
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import numpy as np
import os
import recovar
import starfile
from recovar import cuda_backproject

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert Path(jax.__file__).resolve().is_relative_to(root / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
particles = starfile.read(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80/data/particles.star"
)["particles"]
data_root = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80/data"
)
image_names = []
for value in particles["rlnImageName"]:
    stack_index, stack_path = str(value).split("@", 1)
    path = Path(stack_path)
    if not path.is_absolute():
        path = (data_root / path).resolve()
    image_names.append(f"{stack_index}@{path}")
np.save(
    os.environ["RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY"],
    np.asarray(image_names, dtype=str),
)
print(jax.devices())
PY

command=("${PYTHON}" -m scripts.run_full_refinement
  --data_dir "${DATA}" --output "${OUTPUT}" --max_iter 1 --skip_final_iteration
  --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
  --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
  --rotation_block_size 8192 --seed 1704 --perturb_seed 1704
  --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
  --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
  --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json"
  --timing_dir "${OUTPUT}/timing"
  --save_intermediates_dir "${OUTPUT}/intermediates"
  --save_intermediates_skip_unregularized
  --relion_half_sets "${RELION}/run_it000_data.star"
  --relion_optimiser "${RELION}/run_it000_optimiser.star"
  --relion_init_dir "${RELION}")
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
"${command[@]}" > "${OUTPUT}/run.log" 2>&1
wall_s=$(( $(date +%s) - start_epoch ))

rg -q 'Applied RELION fresh paired AutoRefine particle order with effective seed 1705' "${OUTPUT}/run.log"
rg -q 'preserving fresh RELION physical BPref particle order' "${OUTPUT}/run.log"
if [[ "${STOP_AFTER_TARGET}" == 1 ]]; then
    rg -q 'BPref contribution dump completed; stopping at the requested pass-2 boundary' "${OUTPUT}/run.log"
    test "$(find "${CONTRIBUTIONS}" -maxdepth 1 -name 'bpref_contribution_rows_it001_h*_*.npz' | wc -l)" -eq 1
    test -n "${DEVICE_SIGNATURE_DIR}"
    test "$(find "${DEVICE_SIGNATURE_DIR}" -maxdepth 1 -name '*.device.npz' | wc -l)" -eq 1
else
    test -s "${ACCUM}/recovar_bpref_accum_it001.npz"
    test -s "${PREJOIN}/recovar_bpref_prejoin_it001.npz"
fi
test "$(find "${CONTRIBUTIONS}" -maxdepth 1 -name 'bpref_contribution_rows_it001_h*_*.npz' | wc -l)" -ge 1
"${PYTHON}" - "${CONTRIBUTIONS}" <<'PY'
from pathlib import Path
import os
import sys
import numpy as np

expected = {
    int(value)
    for value in os.environ["RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES"].split(",")
}
seen = set()
for path in Path(sys.argv[1]).glob("bpref_contribution_rows_it001_h*_*.npz"):
    with np.load(path, allow_pickle=False) as values:
        assert bool(values["high_precision_operand_bundle"])
        seen.update(np.asarray(values["original_indices"], dtype=np.int64).tolist())
assert seen == expected, (seen, expected)
PY
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run.log"
result_roots=("${OUTPUT}" "${CONTRIBUTIONS}" "${ACCUM}" "${PREJOIN}")
if [[ -n "${DEVICE_SIGNATURE_DIR}" ]]; then
    result_roots+=("${DEVICE_SIGNATURE_DIR}")
fi
find "${result_roots[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
particle_count=$(awk -F, '{n += NF} END {print n}' <<< "${CAPTURE_INDICES}")
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":1,"particle_count":%d,"stop_after_target":%s}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${particle_count}" "${STOP_AFTER_TARGET}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
