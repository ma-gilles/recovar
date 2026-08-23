#!/usr/bin/env bash
#SBATCH --job-name=k1c22-preproc
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fine_operand_first_mismatch_20260810T0850ET/logs/preprocess-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fine_operand_first_mismatch_20260810T0850ET/logs/preprocess-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:08:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fine_operand_first_mismatch_20260810T0850ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_preprocess_replay_20260810T0915ET
DOCS_REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
SOURCE_REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_bpref_factor_capture_20260810
PYTHON=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default/bin/python
PIXI_ENV=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_followup_20260809/.pixi/envs/default
LAUNCHER=${DOCS_REPO}/scripts/run_k1_case22_preprocess_replay.sbatch
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE_REPO}" rev-parse HEAD)" = 0dbd53b779d4c57226417f011d71a8f68ee98a0a
test -z "$(git -C "${SOURCE_REPO}" status --porcelain=v1)"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = 50a0b3b86478a31bd64e8202ed71a6bf70e28429262a957231cf19c50ceab569

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export RECOVAR_CUDA_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/cuda_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${RECOVAR_CUDA_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"
export RECOVAR_CUDA_LIB=${CUDA_LIB}

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

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac

cd "${SOURCE_REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert len(jax.devices("gpu")) == 1
assert "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
PY

{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "gpu_uuid=${GPU_UUID}"
    echo "gpu_name=${GPU_NAME}"
    echo "source_repo_head=$(git rev-parse HEAD)"
    echo "cuda_lib_sha256=$(sha256sum "${CUDA_LIB}" | awk '{print $1}')"
} > "${ROOT}/provenance/PREPROCESS_REPLAY_${SLURM_JOB_ID}.txt"

"${PYTHON}" -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1), run_name="__main__")' \
    "${DOCS_REPO}/scripts" \
    "${DOCS_REPO}/scripts/replay_k1_relion_cuda_preprocess_boundary.py" \
    --capture-dir "${ROOT}/capture" \
    --recovar-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/recovar_capture/pass2 \
    --particles-mrcs /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data/particles.128.mrcs \
    --source-state-star "${ROOT}/relion/run_it001_data.star" \
    --source-model-star "${ROOT}/relion/run_it001_half1_model.star" \
    --native-preprocess-capture-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_shared_checkpoint_fp64_1101a2e_20260731T2030ET/normal_a/capture \
    --particle 252:251 --particle 591:590 --particle 2124:2123 \
    --physical-image-size 128 --pixel-size 4.25 --particle-diameter 200 \
    --output-json "${ROOT}/analysis/K1_RELION_CUDA_PREPROCESS_REPLAY_H100.json" \
    --output-npz "${ROOT}/analysis/K1_RELION_CUDA_PREPROCESS_REPLAY_H100.npz" \
    > "${ROOT}/logs/replay_relion_cuda_preprocess_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/replay_relion_cuda_preprocess_${SLURM_JOB_ID}.stderr"

test -s "${ROOT}/analysis/K1_RELION_CUDA_PREPROCESS_REPLAY_H100.json"
test -s "${ROOT}/analysis/K1_RELION_CUDA_PREPROCESS_REPLAY_H100.npz"
touch "${ROOT}/PREPROCESS_REPLAY_SUCCESS_${SLURM_JOB_ID}"
