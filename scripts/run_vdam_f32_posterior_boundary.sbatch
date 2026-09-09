#!/usr/bin/env bash
#SBATCH --job-name=vdam-f32-posterior
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-f32-posterior-%j.out

set -euo pipefail

: "${VDAM_F32_POSTERIOR_ROOT:?set a unique disposable output root}"
: "${VDAM_F32_POSTERIOR_NATIVE_DIR:?set one native verbose capture directory}"
: "${VDAM_F32_POSTERIOR_CONTRIBUTION:?set one RECOVAR BPref contribution NPZ}"
: "${VDAM_F32_POSTERIOR_ORIGINAL_INDEX:?set the RECOVAR original image index}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"

ROOT=${VDAM_F32_POSTERIOR_ROOT}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
ANALYZER=${REPO_ROOT}/scripts/analyze_vdam_f32_posterior_boundary.py

test -f "${ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test -s "${VDAM_F32_POSTERIOR_NATIVE_DIR}/pass1_exp_Mweight_posterior.bin"
test -s "${VDAM_F32_POSTERIOR_CONTRIBUTION}"
test -f "${ANALYZER}"
mkdir -p "${ROOT}/provenance"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}
export PIXI_HOME=/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}
export RATTLER_CACHE_DIR=/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

cd "${REPO_ROOT}"
PIXI_PY=$(pixi run which python)
"${PIXI_PY}" -c 'import pathlib,recovar,jax; repo=pathlib.Path.cwd().resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+"/"); assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())'
pixi run python -m pytest \
    tests/unit/initial_model/test_vdam_f32_posterior_boundary.py \
    tests/unit/test_sparse_pass2_relion_f32_posterior.py \
    tests/unit/test_cuda_relion_f32_posterior_ops.py \
    --run-gpu -v
pixi run python "${ANALYZER}" \
    --native-directory "${VDAM_F32_POSTERIOR_NATIVE_DIR}" \
    --contribution "${VDAM_F32_POSTERIOR_CONTRIBUTION}" \
    --original-index "${VDAM_F32_POSTERIOR_ORIGINAL_INDEX}" \
    --output-json "${ROOT}/posterior_boundary.json"

sha256sum "${ANALYZER}" "${VDAM_F32_POSTERIOR_CONTRIBUTION}" \
    "${VDAM_F32_POSTERIOR_NATIVE_DIR}/pass1_exp_Mweight_posterior.bin" \
    > "${ROOT}/provenance/static_inputs.sha256"
printf '{"job_id":"%s","git_head":"%s","original_index":%d}\n' \
    "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${VDAM_F32_POSTERIOR_ORIGINAL_INDEX}" \
    > "${ROOT}/provenance/run.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
