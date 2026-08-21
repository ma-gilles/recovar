#!/usr/bin/env bash
#SBATCH --job-name=k1-c22-mass-cmp
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rotation_mass_compare_20260810T1715ET/logs/analyze-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rotation_mass_compare_20260810T1715ET/logs/analyze-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=01:00:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher}"
: "${EXPECTED_REPO_DIFF_SHA256:?submission must pin repository diff}"
ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rotation_mass_compare_20260810T1715ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_rotation_mass_compare_20260810T1715ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_case22_rotation_mass_all.sbatch
RECOVAR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rotation_mass_all_20260810T1600ET/membership
RELION=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_rotation_mass_full_20260810T1730ET/capture
REPORT=${ROOT}/analysis/K1_CASE22_IT2_HALF1_ROTATION_MASS_ALL.json

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test -f /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rotation_mass_all_20260810T1600ET/RUN_SUCCESS_12221729
test -f /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_rotation_mass_full_20260810T1730ET/RUN_SUCCESS_12222630
test ! -e "${REPORT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

cd "${REPO}"
"${PYTHON}" -c 'import pathlib, recovar, jax; repo=pathlib.Path.cwd().resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/"); assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())'
sha256sum "${LAUNCHER}" scripts/analyze_k1_bpref_membership_all.py \
  scripts/validate_relion_bpref_rotation_mass.py > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
git diff --binary > "${ROOT}/provenance/repository.patch"

start=$(date +%s)
"${PYTHON}" -m scripts.analyze_k1_bpref_membership_all \
  --relion-directory "${RELION}" \
  --recovar-directory "${RECOVAR}" \
  --mpi-rank 1 \
  --expected-relion-particles 3000 \
  --expected-recovar-particles 1490 \
  --rotation-tolerance 1e-6 \
  --output-json "${REPORT}"
wall_s=$(( $(date +%s) - start ))

test -s "${REPORT}"
jq -e '.physical_iteration == 2 and .half == 1 and .fixed_metric.denominator == 1490' "${REPORT}" >/dev/null
printf '{"job_id":"%s","wall_s":%d,"particle_count":1490,"physical_iteration":2,"half":1}\n' "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/ANALYSIS_SUCCESS_${SLURM_JOB_ID}"
