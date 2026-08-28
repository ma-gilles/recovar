#!/usr/bin/env bash
#SBATCH --job-name=k1c10-rec-panel
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=04:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${TARGET_HALF:?TARGET_HALF is required}"
: "${TARGET_ORIGINAL_INDICES:?TARGET_ORIGINAL_INDICES is required}"
: "${EXPECTED_TARGET_COUNT:?EXPECTED_TARGET_COUNT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_PRODUCTION_DIFF_SHA256:?EXPECTED_PRODUCTION_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"

case "${TARGET_HALF}" in
    1|2) ;;
    *) echo "TARGET_HALF must be 1 or 2" >&2; exit 2 ;;
esac
[[ "${EXPECTED_TARGET_COUNT}" =~ ^[1-9][0-9]*$ ]]
[[ "${TARGET_ORIGINAL_INDICES}" =~ ^[0-9]+(:[0-9]+)*$ ]]
TARGET_ORIGINAL_INDICES_CSV=${TARGET_ORIGINAL_INDICES//:/,}

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_pr167_postintegration_debug_20260827
LAUNCHER=${REPO}/scripts/capture_recovar_k1_case10_final_panel.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=${REPO}/recovar/cuda/libcuda_backproject.so
FIXED_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE_ROOT=${FIXED_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE_ROOT}/data
RELION=${CASE_ROOT}/relion_ref
SOURCE_RUN=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_terminal_current_b39_20260827T0732ET
SOURCE_MANIFESTS=${SOURCE_RUN}/output/intermediates
SOURCE_RESULTS=${SOURCE_RUN}/output/refinement_results.npz
RELION_BIND=${FIXED_ROOT}/relion_bind_build/shared
OUTPUT=${RUN_ROOT}/output
CONTRIBUTIONS=${RUN_ROOT}/contributions
PASS2=${RUN_ROOT}/pass2
PROVENANCE=${RUN_ROOT}/provenance
LOGS=${RUN_ROOT}/logs
IMAGE_NAMES=${PROVENANCE}/image_names.npy

test -f "${RUN_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${PYTHON}"
test -x "${CUDA_LIB}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu recovar/cuda_backproject.py scripts/run_full_refinement.py | sha256sum | awk '{print $1}')" = "${EXPECTED_PRODUCTION_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
test -s "${SOURCE_MANIFESTS}/manifest_final_half0.npz"
test -s "${SOURCE_MANIFESTS}/manifest_final_half1.npz"
test -s "${SOURCE_RESULTS}"
test -z "$(find "${RUN_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE ! -name 'slurm-*.out' -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
export RECOVAR_CACHE_DIR=
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${OUTPUT}" "${OUTPUT}/intermediates" "${OUTPUT}/timing" \
    "${CONTRIBUTIONS}" "${PASS2}" "${PROVENANCE}" "${LOGS}"

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
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign one visible H100}"

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
export RECOVAR_K1_RELION_NATIVE_FINE_SCORE_UNITS=1
export RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE=1
export RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=1
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT

export RECOVAR_PASS2_DUMP_DIR=${PASS2}
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${TARGET_ORIGINAL_INDICES_CSV}
export RECOVAR_PASS2_DUMP_ITERATION=16
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=384
export RECOVAR_PASS2_DUMP_CLASS=1
export RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR=${CONTRIBUTIONS}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION=16
export RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE=384
export RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF=${TARGET_HALF}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES=${TARGET_ORIGINAL_INDICES_CSV}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID=case10-final-panel-half${TARGET_HALF}
export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY=${IMAGE_NAMES}
export RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256=6abfc7829307ed63cc1b8b24ba99ec1afca25e5467876ef056034a9dd8779585
export RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE=1
unset RECOVAR_BPREF_CONTRIBUTION_TARGET_ONLY RECOVAR_BPREF_CONTRIBUTION_STOP_AFTER_TARGET

cd "${REPO}"
git status --short --branch > "${PROVENANCE}/git_status_${SLURM_JOB_ID}.txt"
git diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu recovar/cuda_backproject.py scripts/run_full_refinement.py \
    > "${PROVENANCE}/production_source_diff_${SLURM_JOB_ID}.patch"
env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
rg -q H100 "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
sha256sum "${SOURCE_MANIFESTS}/manifest_final_half0.npz" \
    "${SOURCE_MANIFESTS}/manifest_final_half1.npz" "${SOURCE_RESULTS}" \
    "${DATA}/particles.star" "${DATA}/particles.384.mrcs" \
    "${LAUNCHER}" "${CUDA_LIB}" > "${PROVENANCE}/static_${SLURM_JOB_ID}.sha256"

DATA_STAR=${DATA}/particles.star HALF_STAR=${RELION}/run_it000_data.star \
IMAGE_NAMES=${IMAGE_NAMES} \
TARGET_HALF=${TARGET_HALF} TARGET_ORIGINAL_INDICES=${TARGET_ORIGINAL_INDICES} \
EXPECTED_TARGET_COUNT=${EXPECTED_TARGET_COUNT} "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import numpy as np
import starfile

star_path = Path(os.environ["DATA_STAR"]).resolve()
document = starfile.read(star_path)
table = document["particles"] if isinstance(document, dict) else document
half_document = starfile.read(os.environ["HALF_STAR"])
half_table = half_document["particles"] if isinstance(half_document, dict) else half_document
identities = []
for value in table["rlnImageName"].astype(str):
    stack_index, stack_path = value.split("@", 1)
    path = Path(stack_path)
    if not path.is_absolute():
        path = (star_path.parent / path).resolve()
    identities.append(f"{int(stack_index)}@{path}")
targets = [int(value) for value in os.environ["TARGET_ORIGINAL_INDICES"].split(":")]
assert len(targets) == int(os.environ["EXPECTED_TARGET_COUNT"])
assert len(set(targets)) == len(targets)
assert min(targets) >= 0 and max(targets) < len(table)
half_by_stack = {
    int(str(image_name).split("@", 1)[0]): int(random_subset)
    for image_name, random_subset in zip(
        half_table["rlnImageName"], half_table["rlnRandomSubset"], strict=True
    )
}
halves = np.asarray(
    [half_by_stack[int(identity.split("@", 1)[0])] for identity in identities],
    dtype=np.int64,
)
assert np.all(halves[targets] == int(os.environ["TARGET_HALF"]))
assert [int(identities[index].split("@", 1)[0]) for index in targets] == [index + 1 for index in targets]
width = max(map(len, identities))
np.save(os.environ["IMAGE_NAMES"], np.asarray(identities, dtype=f"<U{width}"), allow_pickle=False)
PY

"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert Path(jax.__file__).resolve().is_relative_to(repo / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices())
PY

command=("${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter 0
    --init_relion_iteration 15 --healpix_order 5 --auto_local_healpix_order 4
    --offset_range 3.0 --offset_step 1.0 --adaptive_oversampling 1
    --init_resolution 30.0 --image_batch_size 187 --rotation_block_size 8192
    --seed 1710 --perturb_seed 1710 --particle_diameter_ang 200
    --tau2_fudge 1.0 --max_significants -1 --firstiter_cc
    --image-fourier-backend relion_cuda
    --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json"
    --timing_dir "${OUTPUT}/timing"
    --relion_half_sets "${RELION}/run_it000_data.star"
    --relion_optimiser "${RELION}/run_it000_optimiser.star"
    --save_intermediates_dir "${OUTPUT}/intermediates"
    --save_intermediates_skip_unregularized
    --diagnostic-final-manifest-dir "${SOURCE_MANIFESTS}"
    --diagnostic-final-source-results "${SOURCE_RESULTS}")
printf '%q ' "${command[@]}" > "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
started=$(date +%s)
"${command[@]}" > "${LOGS}/run_${SLURM_JOB_ID}.log" 2>&1
printf '{"elapsed_seconds":%d,"job_id":"%s","half":%d}\n' \
    "$(( $(date +%s) - started ))" "${SLURM_JOB_ID}" "${TARGET_HALF}" \
    > "${PROVENANCE}/walltime_${SLURM_JOB_ID}.json"

"${PYTHON}" - "${CONTRIBUTIONS}" "${PASS2}" "${TARGET_ORIGINAL_INDICES_CSV}" "${TARGET_HALF}" <<'PY'
import sys
from pathlib import Path

import numpy as np

contributions = Path(sys.argv[1])
pass2 = Path(sys.argv[2])
expected = {int(value) for value in sys.argv[3].split(",")}
half = int(sys.argv[4])
seen = set()
files = sorted(contributions.glob(f"bpref_contribution_rows_it016_h{half}_*.npz"))
assert files
for path in files:
    with np.load(path, allow_pickle=False) as values:
        assert bool(values["high_precision_operand_bundle"])
        assert int(values["half"]) == half
        seen.update(np.asarray(values["original_indices"], dtype=np.int64).tolist())
assert seen == expected, (seen, expected)
pass2_seen = set()
for path in pass2.glob("pass2_orig*_cs384.npz"):
    with np.load(path, allow_pickle=False) as values:
        pass2_seen.add(int(np.asarray(values["original_index"]).item()))
assert pass2_seen == expected, (pass2_seen, expected)
PY

rg -q 'RECOVAR BPref contribution dump completed' "${LOGS}/run_${SLURM_JOB_ID}.log"
! rg -q 'RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent|Traceback' \
    "${LOGS}/run_${SLURM_JOB_ID}.log"
find "${OUTPUT}" "${CONTRIBUTIONS}" "${PASS2}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
