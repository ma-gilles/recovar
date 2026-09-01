#!/usr/bin/env bash
#SBATCH --job-name=k1c04-expf
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CAPTURE:?CAPTURE is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=${REPO}/recovar/cuda/libcuda_backproject.so

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -s "${CAPTURE}" -a -x "${PYTHON}" -a -s "${CUDA_LIB}"
test -z "$(find "${ROOT}/analysis" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
export CUDA_VISIBLE_DEVICES=0 RECOVAR_CUDA_LIB=${CUDA_LIB}

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
sha256sum "${CAPTURE}" "${CUDA_LIB}" "$0" > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

cd "${REPO}"
"${PYTHON}" - "${CAPTURE}" "${ROOT}/analysis/COARSE_EXPF_COUNTERFACTUAL.json" <<'PY'
import json
import sys
from pathlib import Path

import jax
import numpy as np

from recovar.em.dense_single_volume.helpers.oversampling import (
    relion_cuda_f32_coarse_posterior,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_f32_fine_reconstruction_probs,
)

capture_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
with np.load(capture_path, allow_pickle=False) as capture:
    n_rot = int(capture["n_rot"])
    n_trans = int(capture["n_trans"])
    scores = np.asarray(capture["scores_with_prior_per_class"], dtype=np.float32).reshape(1, -1)
    recorded_mask = np.asarray(capture["significant_mask"], dtype=bool).reshape(1, -1)
    adaptive_fraction = float(capture["adaptive_fraction"])
    recorded = {
        "selected_count": int(capture["n_significant"]),
        "cutoff_count": int(capture["relion_f32_cutoff_count"]),
        "sum_weight": float(np.float32(capture["relion_f32_sum_weight"])),
        "threshold": float(np.float32(capture["relion_f32_significant_weight"])),
    }

current = tuple(
    np.asarray(value)
    for value in relion_cuda_f32_coarse_posterior(
        jax.device_put(scores),
        adaptive_fraction=adaptive_fraction,
        max_significants=-1,
    )
)
native_expf = tuple(
    np.asarray(value)
    for value in _relion_f32_fine_reconstruction_probs(
        jax.device_put(scores.reshape(1, 1, -1)),
        adaptive_fraction=adaptive_fraction,
    )
)
current_prob, current_mask, current_n, current_cutoff, current_sum, current_threshold = current
_, native_mask_3d, native_n, native_sum, native_threshold = native_expf
native_mask = native_mask_3d.reshape(1, -1)

expected_native_pairs = {
    (26325, 18), (26325, 19), (26325, 20),
    (27095, 20),
    (27101, 13), (27101, 14), (27101, 19), (27101, 20), (27101, 21),
    (27862, 13), (27862, 14), (27862, 15), (27862, 19), (27862, 20),
    (27868, 13), (27868, 14), (27868, 19), (27868, 20),
}
expected_native_mask = np.zeros((1, n_rot * n_trans), dtype=bool)
for rotation, translation in expected_native_pairs:
    expected_native_mask[0, rotation * n_trans + translation] = True

def pairs(mask):
    indices = np.flatnonzero(np.asarray(mask).reshape(-1))
    return [[int(index // n_trans), int(index % n_trans)] for index in indices]

target_index = 27095 * n_trans + 14
report = {
    "schema": "recovar.em.k1_coarse_expf_counterfactual.v1",
    "device": str(jax.devices()[0]),
    "capture": str(capture_path.resolve()),
    "recorded": recorded,
    "current_jax_exp_replay": {
        "mask_exact": bool(np.array_equal(current_mask, recorded_mask)),
        "selected_count": int(current_n[0]),
        "cutoff_count": int(current_cutoff[0]),
        "sum_weight": float(current_sum[0]),
        "threshold": float(current_threshold[0]),
        "target_27095_14_selected": bool(current_mask[0, target_index]),
    },
    "native_cuda_expf_counterfactual": {
        "selected_count": int(native_n[0]),
        "sum_weight": float(native_sum[0]),
        "threshold": float(native_threshold[0]),
        "target_27095_14_selected": bool(native_mask[0, target_index]),
        "matches_known_native_support_exactly": bool(
            np.array_equal(native_mask, expected_native_mask)
        ),
        "selected_pairs": pairs(native_mask),
    },
    "membership_changes_jax_exp_to_native_expf": pairs(current_mask != native_mask),
}
assert report["current_jax_exp_replay"]["mask_exact"]
assert report["current_jax_exp_replay"]["selected_count"] == recorded["selected_count"]
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

sha256sum "${ROOT}/analysis/COARSE_EXPF_COUNTERFACTUAL.json" > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
