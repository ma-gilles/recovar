#!/usr/bin/env bash
#SBATCH --job-name=k1c10-ppref-src
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:20:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
FIXED_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${FIXED_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
NATIVE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_native_it1_cc_row10692_20260823T0701ET/capture
PYTHON=${REPO}/.pixi/envs/default/bin/python
RELION_BIND=${FIXED_ROOT}/relion_bind_build/shared

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

cd "${REPO}"
git status --short --branch > "${RUN_ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
sha256sum \
    scripts/analyze_k1_firstiter_ppref_sources.py \
    scripts/run_k1_case10_firstiter_ppref_sources.sbatch \
    "${CASE}/data/reference_init.mrc" \
    "${NATIVE}/pass1_class0_ppref_real.bin" \
    "${NATIVE}/pass1_class0_ppref_imag.bin" \
    > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

"${PYTHON}" -m scripts.analyze_k1_firstiter_ppref_sources \
    --native-directory "${NATIVE}" \
    --native-prefix pass1_class0_ppref_ \
    --fresh-reference-mrc "${CASE}/data/reference_init.mrc" \
    --serialized-it000-mrc "${CASE}/relion_ref/run_it000_half1_class001.mrc" \
    --pixel-size 1.4166666666666667 \
    --ini-high 30 \
    --current-size 56 \
    --output-json "${RUN_ROOT}/analysis/K1_CASE10_FIRSTITER_PPREF_SOURCES.json" \
    > "${RUN_ROOT}/logs/analyze_${SLURM_JOB_ID}.log" 2>&1
sha256sum "${RUN_ROOT}/analysis/K1_CASE10_FIRSTITER_PPREF_SOURCES.json" \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
