#!/usr/bin/env bash
#SBATCH --job-name=k1c22-it1-primitives
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/logs/primitive-analysis-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/logs/primitive-analysis-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it1_first_boundary_20260810T0945ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default/bin/python
PIXI_ENV=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case22_iteration1_primitive_analysis.sbatch
OUTPUT=${ROOT}/analysis/CASE22_IT1_BPREF_PRIMITIVE_BOUNDARY_V9.json
DUMPS=${ROOT}/analysis/it1_bpref_primitive_dumps_v9
SOURCE_STAR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_guigrid_localhighshell_full34_autonomous_ac5177d2_20260719T174000Z/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data/particles.star
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/relion_bind_build/shared
LIVE_SIGMA2=${ROOT}/analysis/live_initial_sigma2_recomputed_f64.npy

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = ea8f2b21bd3041c71ac747a9d4b7aee96882a61e
test -f "${ROOT}/OPERAND_BUNDLE_SUCCESS_12209989"
test -f "${SOURCE_STAR}" -a -d "${RELION_BIND}" -a -f "${LIVE_SIGMA2}"
test ! -e "${OUTPUT}" -a ! -e "${DUMPS}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
export RECOVAR_CUDA_LIB=${ROOT}/cuda_ptx_sm80_12_6/libcuda_backproject.so
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

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

cd "${REPO}"
"${PYTHON}" -m scripts.compare_k1_relion_recovar_bpref_primitives \
    --factor-directory "${ROOT}/capture" \
    --pass2-directory "${ROOT}/recovar/pass2" \
    --contribution-directory "${ROOT}/recovar/bpref_factors" \
    --selection-json "${ROOT}/provenance/SELECTION_IT1_CS056.json" \
    --output-json "${OUTPUT}" \
    --dump-directory "${DUMPS}" \
    --source-star "${SOURCE_STAR}" \
    --relion-bind-directory "${RELION_BIND}" \
    --live-initial-sigma2 "${LIVE_SIGMA2}"

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
from pathlib import Path
import sys
report = json.loads(Path(sys.argv[1]).read_text())
assert report["particle_count"] == 3
assert all(row["first_exact_unequal_boundary"] for row in report["particles"])
assert all(
    row["comparisons"]["relion_ctf_replayed_from_source_star"]["exact_equal"]
    and row["comparisons"]["relion_inverse_noise_replayed_from_float64_sigma2"]["exact_equal"]
    for row in report["particles"]
)
PY

{
    echo "job_id=${SLURM_JOB_ID}"
    nvidia-smi --query-gpu=uuid,name --format=csv,noheader
    sha256sum "${OUTPUT}"
    sha256sum "${RECOVAR_CUDA_LIB}"
    find "${DUMPS}" -maxdepth 1 -type f -name '*.npz' -print0 | sort -z | xargs -0 sha256sum
} > "${ROOT}/provenance/PRIMITIVE_ANALYSIS_${SLURM_JOB_ID}.txt"
touch "${ROOT}/PRIMITIVE_ANALYSIS_SUCCESS_${SLURM_JOB_ID}"
