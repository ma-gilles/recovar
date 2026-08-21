#!/usr/bin/env bash
#SBATCH --job-name=k1c22-it1-operands
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/logs/operand-bundle-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/logs/operand-bundle-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:10:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"

ROOT=${K1_OPERAND_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET}
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it1_first_boundary_20260810T0945ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_bpref_factor_capture_20260810
DOCS_REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default/bin/python
PIXI_ENV=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default
LAUNCHER=${DOCS_REPO}/scripts/run_k1_case22_iteration1_operand_bundle.sbatch
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
SEALED_CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_control/relion
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so
METADATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/provenance/bpref_metadata
OUTPUT=${ROOT}/recovar/bpref_factor_output
CONTRIBUTIONS=${ROOT}/recovar/bpref_factors
IMAGE_FOURIER_BACKEND=${K1_IMAGE_FOURIER_BACKEND:-host_numpy}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = 0dbd53b779d4c57226417f011d71a8f68ee98a0a
test -z "$(git -C "${REPO}" status --porcelain=v1)"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = 50a0b3b86478a31bd64e8202ed71a6bf70e28429262a957231cf19c50ceab569
test "$(sha256sum "${DATA}/particles.128.mrcs" | awk '{print $1}')" = 804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9
test "$(sha256sum "${METADATA}/image_names.npy" | awk '{print $1}')" = 0ba4e2dd5d6287acfba8455e08d29e05161b2a1a9c228395fd0761bc5c539621
test -f "${CONTRIBUTIONS}/SAFE_TO_DELETE"
test -z "$(find "${CONTRIBUTIONS}" -type f ! -name SAFE_TO_DELETE -print -quit)"
test ! -e "${OUTPUT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_BPRE_*|RELION_FINE_*|RELION_DISPATCH_LOG) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" "${JAX_COMPILATION_CACHE_DIR}"

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

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_CUDA_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/cuda_cache
export RECOVAR_RELION_BIND_BUILD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/relion_bind_build/shared
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR=${CONTRIBUTIONS}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION=1
export RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE=56
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES=251,590,2123
export RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID=case22-it1-fixed-panel
export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY=${METADATA}/image_names.npy
export RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256=804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9
export RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject
root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
PY

start=$(date +%s)
"${PYTHON}" -m scripts.run_full_refinement \
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter 1 \
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0 \
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187 \
    --rotation_block_size 8192 --seed 1722 --perturb_seed 1722 \
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1 \
    --firstiter_cc --apply-initial-lowpass \
    --image-fourier-backend "${IMAGE_FOURIER_BACKEND}" \
    --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json" \
    --timing_dir "${OUTPUT}/timing" \
    --save_intermediates_dir "${OUTPUT}/intermediates" \
    --save_intermediates_skip_unregularized \
    --relion_half_sets "${SEALED_CONTROL}/run_it000_data.star" \
    --relion_optimiser "${SEALED_CONTROL}/run_it000_optimiser.star" \
    --relion_init_dir "${SEALED_CONTROL}" \
    > "${OUTPUT}.log" 2>&1
wall_s=$(( $(date +%s) - start ))

test "$(find "${CONTRIBUTIONS}" -maxdepth 1 -name 'bpref_contribution_rows_it001_h*_*.npz' | wc -l)" -ge 1
"${PYTHON}" - "${CONTRIBUTIONS}" <<'PY'
from pathlib import Path
import sys
import numpy as np
root = Path(sys.argv[1])
seen = set()
for path in root.glob("bpref_contribution_rows_it001_h*_*.npz"):
    with np.load(path, allow_pickle=False) as values:
        assert bool(values["high_precision_operand_bundle"])
        seen.update(np.asarray(values["original_indices"], dtype=np.int64).tolist())
assert seen == {251, 590, 2123}, seen
PY

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "wall_s=${wall_s}"
    echo "gpu_uuid=${GPU_UUID}"
    echo "gpu_name=${GPU_NAME}"
    echo "repo_head=$(git -C "${REPO}" rev-parse HEAD)"
    echo "image_fourier_backend=${IMAGE_FOURIER_BACKEND}"
    echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset"
    echo "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset"
    find "${CONTRIBUTIONS}" -maxdepth 1 -type f -name '*.npz' -print0 | sort -z | xargs -0 sha256sum
} > "${ROOT}/provenance/OPERAND_BUNDLE_${SLURM_JOB_ID}.txt"
touch "${ROOT}/OPERAND_BUNDLE_SUCCESS_${SLURM_JOB_ID}"
