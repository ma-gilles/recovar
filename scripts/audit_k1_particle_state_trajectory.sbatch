#!/usr/bin/env bash
#SBATCH --job-name=k1-state-trajectory
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

trap 'status=$?; printf "ERROR status=%d line=%d command=%q\n" "${status}" "${LINENO}" "${BASH_COMMAND}" >&2' ERR

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CASE_SOURCE:?CASE_SOURCE is required}"
: "${RELION_LAST_ITERATION:?RELION_LAST_ITERATION is required}"
: "${ANALYSIS_TAG:?ANALYSIS_TAG is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
PYTHON=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_particle_state_trajectory.sbatch
ANALYZER=${REPO}/scripts/audit_em_particle_state_distribution.py
OUTPUT=${RUN_ROOT}/output
DATA=${CASE_SOURCE}/data
RELION=${CASE_SOURCE}/relion_ref
ANALYSIS=${RUN_ROOT}/analysis/particle_state_trajectory_${ANALYSIS_TAG}

[[ "${RELION_LAST_ITERATION}" =~ ^[1-9][0-9]*$ ]]
[[ "${ANALYSIS_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]
test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${OUTPUT}/refinement_results.npz"
test -s "${DATA}/particles.star"
test -s "${RELION}/run_data.star"
test -x "${PYTHON}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test ! -e "${ANALYSIS}/particle_state.json"

relion_args=()
relion_inputs=()
for ((iteration = 1; iteration <= RELION_LAST_ITERATION; ++iteration)); do
    printf -v label '%03d' "${iteration}"
    star=${RELION}/run_it${label}_data.star
    test -s "${star}"
    relion_args+=(--relion-star "${iteration}:${star}")
    relion_inputs+=("${star}")
done

mkdir -p "${ANALYSIS}" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache

cd "${REPO}"
command=("${PYTHON}" -m scripts.audit_em_particle_state_distribution
    --recovar-results "${OUTPUT}/refinement_results.npz"
    --recovar-particles-star "${DATA}/particles.star"
    "${relion_args[@]}"
    --relion-iteration-offset 1
    --relion-final-star "${RELION}/run_data.star"
    --output-json "${ANALYSIS}/particle_state.json"
    --output-npz "${ANALYSIS}/particle_state_arrays.npz"
    --output-hash-manifest "${ANALYSIS}/particle_state.sha256")

printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/particle_state_trajectory_command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/particle_state_trajectory_command_${SLURM_JOB_ID}.sh"
env | sort > "${RUN_ROOT}/provenance/particle_state_trajectory_environment_${SLURM_JOB_ID}.txt"
"${command[@]}"

test -s "${ANALYSIS}/particle_state.json"
test -s "${ANALYSIS}/particle_state_arrays.npz"
"${PYTHON}" - "${ANALYSIS}/particle_state.json" "${RELION_LAST_ITERATION}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
assert report["status"] == "complete"
assert len(report["iterations"]) == expected
assert [item["relion_iteration"] for item in report["iterations"]] == list(range(1, expected + 1))
PY

sha256sum "${LAUNCHER}" "${ANALYZER}" "${OUTPUT}/refinement_results.npz" \
    "${RELION}/run_data.star" "${relion_inputs[@]}" \
    "${ANALYSIS}/particle_state.json" "${ANALYSIS}/particle_state_arrays.npz" \
    > "${RUN_ROOT}/provenance/particle_state_trajectory_${SLURM_JOB_ID}.sha256"
touch "${ANALYSIS}/SUCCESS_${SLURM_JOB_ID}"
