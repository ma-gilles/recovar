#!/usr/bin/env bash
#SBATCH --job-name=k1c10-delta-compare
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case10_it1_bpref_particle_delta_compare.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_bpref_particle_deltas.py
NATIVE=${NATIVE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_native_it1_bpref_particle_delta_20260824T0635ET}
RECOVAR_H1=${RECOVAR_H1_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_it1_bpref_particle_delta_recovar_h1_4a3a34bd7_20260824T0615ET}
RECOVAR_H2=${RECOVAR_H2_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_it1_bpref_particle_delta_recovar_h2_4a3a34bd7_20260824T0615ET}
OUTPUT=${ROOT}/analysis/K1_CASE10_BPREF_PARTICLE_DELTA.json

test -f "${ROOT}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test -n "$(find "${NATIVE}" -maxdepth 1 -name 'SUCCESS_*' -print -quit)"
test -n "$(find "${RECOVAR_H1}" -maxdepth 1 -name 'SUCCESS_*' -print -quit)"
test -n "$(find "${RECOVAR_H2}" -maxdepth 1 -name 'SUCCESS_*' -print -quit)"
mkdir -p "${ROOT}/analysis" "${ROOT}/provenance"
test -z "$(find "${ROOT}/analysis" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_bpref_particle_deltas \
    --native-directory "${NATIVE}/bpref_particle_delta" \
    --recovar-directory-half1 "${RECOVAR_H1}/recovar/accumulator_delta" \
    --recovar-directory-half2 "${RECOVAR_H2}/recovar/accumulator_delta" \
    --grid-size 384 --delta-relative-l2-bound 2e-7 \
    --output-json "${OUTPUT}" \
    | tee "${ROOT}/analysis/summary.txt"
"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    report = json.load(stream)
assert report["fixed_panel_score"]["total"] == 16
assert len(report["particles"]) == 16
assert {item["half"] for item in report["particles"]} == {1, 2}
PY
git status --short --branch > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${ANALYZER}" "${OUTPUT}" \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
