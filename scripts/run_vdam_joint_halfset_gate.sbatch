#!/usr/bin/env bash
#SBATCH --job-name=vdam-joint-halfset
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
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-joint-halfset-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CUDA_BINARY=${REPO_ROOT}/recovar/cuda/libcuda_backproject.so
NVCC_BIN=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}

test -x "${PIXI_PY}"
test -x "${NVCC_BIN}"
mkdir -p "$(dirname "${CUDA_BINARY}")"

make -B -C "${REPO_ROOT}/recovar/cuda" \
  PYTHON="${PIXI_PY}" \
  NVCC="${NVCC_BIN}" \
  LIB="${CUDA_BINARY}"

export RECOVAR_CUDA_LIB=${CUDA_BINARY}
unset RECOVAR_DISABLE_CUDA

"${PIXI_PY}" -m pytest -q \
  "${REPO_ROOT}/tests/unit/test_cuda_relion_translation.py" \
  "${REPO_ROOT}/tests/unit/test_fourier_window.py::TestFourierWindowSpec::test_initial_model_rounded_recon_window_keeps_cutoff_shell_rim" \
  "${REPO_ROOT}/tests/unit/test_k1_scale_aa_pixels.py::test_relion_wavg_rectangle_accepts_complete_rounded_shell_support" \
  "${REPO_ROOT}/tests/unit/initial_model/test_dense_adapter.py::test_dense_initial_model_estep_sparse_pass2_uses_coarse_parent_prior" \
  "${REPO_ROOT}/tests/unit/initial_model/test_dense_adapter.py::test_arrays_to_accumulators_splits_grouped_halfsets" \
  "${REPO_ROOT}/tests/unit/initial_model/test_dense_adapter.py::test_exact_k1_sparse_pass2_preserves_joint_halfset_particle_stream" \
  "${REPO_ROOT}/tests/unit/test_refine_relion_mode.py::test_bucket_local_hypothesis_layout_aligns_preserved_chunks_to_pool_three"

sha256sum "${CUDA_BINARY}"
