#!/usr/bin/env bash
#SBATCH --job-name=k1c05-it1-primitives
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00

set -euo pipefail

: "${RUN_ROOT:?set an empty absolute analysis root}"
: "${RUNTIME_ROOT:?set an empty absolute runtime root}"
: "${NATIVE_CAPTURE_ROOT:?set the validated native capture root}"
: "${EXPECTED_CAPTURE_JOB_ID:?pin the native capture Slurm job}"
: "${EXPECTED_REPO_HEAD:?pin repository HEAD}"
: "${EXPECTED_LAUNCHER_SHA256:?pin this launcher}"
: "${EXPECTED_ANALYZER_SHA256:?pin the primitive analyzer}"
: "${EXPECTED_SELECTION_SHA256:?pin the particle selection}"
: "${EXPECTED_SOURCE_STAR_SHA256:?pin the source STAR}"
: "${EXPECTED_SIGMA2_SHA256:?pin the sealed RELION noise spectrum}"
: "${EXPECTED_PASS2_SHA256:?pin the RECOVAR pass-2 particle dump}"
: "${EXPECTED_CONTRIBUTION_SHA256:?pin the RECOVAR contribution bundle}"
: "${EXPECTED_CUDA_LIB_SHA256:?pin the selected CUDA library}"

# A geometry-only firstiter-CC capture can finish and validate its requested
# BPref artifact even when the generic producer wrapper rejects the absence of
# an unrelated fine-score artifact.  In that case, point this launcher at an
# independently sealed qualification directory instead of manufacturing a
# producer success marker.
CAPTURE_QUALIFICATION_DIR=${CAPTURE_QUALIFICATION_DIR:-}

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/5_very_high_noise_100k_g256_white_noise10_bf80
SOURCE_STAR=${CASE}/data/particles.star
RECOVAR_CAPTURE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_exact_corr_img_20260822T1850ET
PASS2=${RECOVAR_CAPTURE}/pass2/pass2_orig038594_cs056.npz
CONTRIBUTION=${RECOVAR_CAPTURE}/contributions/bpref_contribution_rows_it001_h1_call000152_dump000000_cs056.npz
SIGMA2=${LIVE_INITIAL_SIGMA2:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row38594_native_fine_h20g1_20260822T1740ET/relion_dump/sigma2_noise.bin}
SELECTION=${RUN_ROOT}/selection.json
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case05_it1_bpref_primitives.sbatch
ANALYZER=${REPO}/scripts/compare_k1_relion_recovar_bpref_primitives.py
CUDA_LIB=${CUDA_LIB_OVERRIDE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_stage_split_20260821T1600ET/cuda/libcuda_backproject.so}
RELION_BIND=${SOURCE_ROOT}/relion_bind_build/shared
OUTPUT=${RUN_ROOT}/analysis/K1_CASE05_IT1_BPREF_PRIMITIVES.json
DUMPS=${RUN_ROOT}/analysis/primitive_dumps
PROVENANCE=${RUN_ROOT}/provenance
CAPTURE_SUCCESS=${NATIVE_CAPTURE_ROOT}/SUCCESS_${EXPECTED_CAPTURE_JOB_ID}
CAPTURE_INERTNESS=${NATIVE_CAPTURE_ROOT}/analysis/RELION_FRESH_CAPTURE_INERTNESS.json
CAPTURE_RESULTS=${NATIVE_CAPTURE_ROOT}/provenance/results_${EXPECTED_CAPTURE_JOB_ID}.sha256
CAPTURE_FACTOR_VALIDATION=
if [[ -n "${CAPTURE_QUALIFICATION_DIR}" ]]; then
    CAPTURE_INERTNESS=${CAPTURE_QUALIFICATION_DIR}/inertness.json
    CAPTURE_RESULTS=${CAPTURE_QUALIFICATION_DIR}/results.sha256
    CAPTURE_FACTOR_VALIDATION=${CAPTURE_QUALIFICATION_DIR}/factor_validation.json
fi

test "${RUN_ROOT#/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/}" != "${RUN_ROOT}"
test "${RUNTIME_ROOT#/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/}" != "${RUNTIME_ROOT}"
test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
if [[ -n "${CAPTURE_QUALIFICATION_DIR}" ]]; then
    case "${CAPTURE_QUALIFICATION_DIR}" in
        "${NATIVE_CAPTURE_ROOT}"/analysis/*) ;;
        *) echo "qualification directory must be inside the native capture analysis root" >&2; exit 2 ;;
    esac
    test -s "${CAPTURE_FACTOR_VALIDATION}" -a -s "${CAPTURE_INERTNESS}" -a -s "${CAPTURE_RESULTS}"
else
    test -f "${CAPTURE_SUCCESS}" -a -s "${CAPTURE_INERTNESS}" -a -s "${CAPTURE_RESULTS}"
fi
test -s "${SELECTION}" -a -s "${SOURCE_STAR}" -a -s "${SIGMA2}"
test -s "${PASS2}" -a -s "${CONTRIBUTION}"
test -x "${PYTHON}" -a -d "${RELION_BIND}"
test ! -e "${OUTPUT}" -a ! -e "${DUMPS}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${SELECTION}" | awk '{print $1}')" = "${EXPECTED_SELECTION_SHA256}"
test "$(sha256sum "${SOURCE_STAR}" | awk '{print $1}')" = "${EXPECTED_SOURCE_STAR_SHA256}"
test "$(sha256sum "${SIGMA2}" | awk '{print $1}')" = "${EXPECTED_SIGMA2_SHA256}"
test "$(sha256sum "${PASS2}" | awk '{print $1}')" = "${EXPECTED_PASS2_SHA256}"
test "$(sha256sum "${CONTRIBUTION}" | awk '{print $1}')" = "${EXPECTED_CONTRIBUTION_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
RELION_BIND_LIB=$(find "${RELION_BIND}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit)
test "$(sha256sum "${RELION_BIND_LIB}" | awk '{print $1}')" = 56acdddab6d25bd83aac5e338162c4819f85ec26f68511458618f143c2b604b7

"${PYTHON}" - "${CAPTURE_INERTNESS}" "${CAPTURE_FACTOR_VALIDATION}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["iterations"]
assert all(row["map_stable"] for row in report["iterations"])
assert all(row["selected_topology_exact"] for row in report["iterations"])
if sys.argv[2]:
    factor = json.load(open(sys.argv[2]))
    assert factor["capture_ready"] is True
    assert factor["particle_count"] == 1
PY

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${PROVENANCE}"

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
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
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

command=("${PYTHON}" -m scripts.compare_k1_relion_recovar_bpref_primitives
    --factor-directory "${NATIVE_CAPTURE_ROOT}/capture"
    --pass2-directory "${RECOVAR_CAPTURE}/pass2"
    --contribution-directory "${RECOVAR_CAPTURE}/contributions"
    --selection-json "${SELECTION}"
    --output-json "${OUTPUT}"
    --dump-directory "${DUMPS}"
    --source-star "${SOURCE_STAR}"
    --relion-bind-directory "${RELION_BIND}"
    --live-initial-sigma2 "${SIGMA2}")
printf '%q ' "${command[@]}" > "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
start=$(date +%s)
"${command[@]}" > "${RUN_ROOT}/logs/analyzer_${SLURM_JOB_ID}.out" 2> "${RUN_ROOT}/logs/analyzer_${SLURM_JOB_ID}.err"
wall_s=$(( $(date +%s) - start ))

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["status"] == "complete"
assert report["particle_count"] == 1
assert report["particles"][0]["stack_index_one_based"] == 38595
assert report["particles"][0]["first_exact_unequal_boundary"]
PY

printf '{"job_id":"%s","wall_s":%d,"capture_job_id":"%s"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${EXPECTED_CAPTURE_JOB_ID}" \
    > "${PROVENANCE}/walltime_${SLURM_JOB_ID}.json"
result_inputs=("${OUTPUT}" "${DUMPS}"/*.npz "${CAPTURE_INERTNESS}" "${CAPTURE_RESULTS}")
if [[ -n "${CAPTURE_FACTOR_VALIDATION}" ]]; then
    result_inputs+=("${CAPTURE_FACTOR_VALIDATION}")
fi
sha256sum "${result_inputs[@]}" > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
