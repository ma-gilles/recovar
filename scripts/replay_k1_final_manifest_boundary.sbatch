#!/usr/bin/env bash
#SBATCH --job-name=k1-final-replay
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=03:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${REPO:?REPO is required}"
: "${CASE_ROOT:?CASE_ROOT is required}"
: "${SOURCE_RUN:?SOURCE_RUN is required}"
: "${CUDA_LIB:?CUDA_LIB is required}"
: "${BIND_BUILD_DIR:?BIND_BUILD_DIR is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"

DATA=${CASE_ROOT}/data
RELION=${CASE_ROOT}/relion_ref
SOURCE_MANIFESTS=${SOURCE_RUN}/output/intermediates
SOURCE_RESULTS=${SOURCE_RUN}/output/refinement_results.npz
LAUNCHER=${REPO}/scripts/replay_k1_final_manifest_boundary.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
OUTPUT=${RUN_ROOT}/output
ACCUM=${RUN_ROOT}/final_bpref_accum
ANALYSIS=${RUN_ROOT}/analysis
RUN_CUDA_DIR=${RUN_ROOT}/cuda
RUN_CUDA_LIB=${RUN_CUDA_DIR}/libcuda_backproject.so

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -z "$(find "${OUTPUT}" -mindepth 1 ! -name SAFE_TO_DELETE -print -quit)"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em scripts/run_full_refinement.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
test -s "${SOURCE_MANIFESTS}/manifest_final_half0.npz"
test -s "${SOURCE_MANIFESTS}/manifest_final_half1.npz"
test -s "${SOURCE_RESULTS}"

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
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${OUTPUT}" "${OUTPUT}/intermediates" "${OUTPUT}/timing" \
    "${ACCUM}" "${ANALYSIS}" "${RUN_CUDA_DIR}" \
    "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
test ! -e "${RUN_CUDA_LIB}"
install -m 0555 "${CUDA_LIB}" "${RUN_CUDA_LIB}"

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
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign exactly one visible GPU}"

export RECOVAR_CUDA_LIB=${RUN_CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${BIND_BUILD_DIR}
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
export RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR=${ACCUM}
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT

cd "${REPO}"
git status --short --branch > "${RUN_ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git diff --binary -- recovar/em scripts/run_full_refinement.py \
    > "${RUN_ROOT}/provenance/source_diff_${SLURM_JOB_ID}.patch"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
sha256sum "${SOURCE_MANIFESTS}/manifest_final_half0.npz" \
    "${SOURCE_MANIFESTS}/manifest_final_half1.npz" "${SOURCE_RESULTS}" \
    > "${RUN_ROOT}/provenance/source_boundary_${SLURM_JOB_ID}.sha256"
sha256sum "${CUDA_LIB}" "${RUN_CUDA_LIB}" \
    > "${RUN_ROOT}/provenance/cuda_before_import_${SLURM_JOB_ID}.sha256"

"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert ".pixi/envs/default/" in str(Path(jax.__file__).resolve())
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices())
PY
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
sha256sum "${CUDA_LIB}" "${RUN_CUDA_LIB}" \
    > "${RUN_ROOT}/provenance/cuda_after_import_${SLURM_JOB_ID}.sha256"

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
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_seconds=$(date +%s)
"${command[@]}" > "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1
end_seconds=$(date +%s)
printf '{"elapsed_seconds":%d,"job_id":"%s"}\n' \
    "$((end_seconds - start_seconds))" "${SLURM_JOB_ID}" \
    > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"

test -s "${OUTPUT}/refinement_results.npz"
test -s "${OUTPUT}/intermediates/manifest_final_half0.npz"
test -s "${OUTPUT}/intermediates/manifest_final_half1.npz"
test -s "${ACCUM}/recovar_final_bpref_accum.npz"
! rg -q 'RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent|Traceback' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.log"

"${PYTHON}" -m scripts.analyze_k1_final_manifest_ab \
    --control-half1 "${SOURCE_MANIFESTS}/manifest_final_half0.npz" \
    --control-half2 "${SOURCE_MANIFESTS}/manifest_final_half1.npz" \
    --candidate-half1 "${OUTPUT}/intermediates/manifest_final_half0.npz" \
    --candidate-half2 "${OUTPUT}/intermediates/manifest_final_half1.npz" \
    --output-json "${ANALYSIS}/final_manifest_replay_ab.json"

SOURCE_OUTPUT=${SOURCE_RUN}/output CANDIDATE_OUTPUT=${OUTPUT} \
OUTPUT_JSON=${ANALYSIS}/final_map_replay_inertness.json \
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from scripts.audit_em_k1_membership_capture_inertness import _load_map, _metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


source = Path(os.environ["SOURCE_OUTPUT"]).resolve()
candidate = Path(os.environ["CANDIDATE_OUTPUT"]).resolve()
threshold = 0.999999
comparisons = {}
artifacts = {}
for name in ("final_half1.mrc", "final_half2.mrc", "final_merged.mrc"):
    source_path = source / name
    candidate_path = candidate / name
    row = _metrics(_load_map(source_path), _load_map(candidate_path))
    row["passed"] = bool(row["fsc_auc_non_dc"] >= threshold)
    comparisons[name] = row
    artifacts[f"source:{name}"] = {
        "path": str(source_path),
        "sha256": sha256(source_path),
    }
    artifacts[f"candidate:{name}"] = {
        "path": str(candidate_path),
        "sha256": sha256(candidate_path),
    }
report = {
    "schema": "recovar.em.k1_final_map_replay_inertness.v1",
    "metric_policy": "signed shellwise FSC and normalized non-DC FSC-AUC; no correlation",
    "fsc_auc_non_dc_threshold": threshold,
    "comparisons": comparisons,
    "all_inertness_gates_passed": all(row["passed"] for row in comparisons.values()),
    "artifacts": artifacts,
}
output = Path(os.environ["OUTPUT_JSON"])
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

find "${OUTPUT}" "${ACCUM}" "${ANALYSIS}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
sha256sum "${CUDA_LIB}" "${RUN_CUDA_LIB}" \
    > "${RUN_ROOT}/provenance/cuda_after_run_${SLURM_JOB_ID}.sha256"
