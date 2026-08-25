#!/usr/bin/env bash
#SBATCH --job-name=vdam-ref-repeat
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-ref-repeat-%j.out

set -euo pipefail

: "${CANDIDATE_MAP:?set the candidate iteration-1 map}"
: "${NATIVE_A_MAP:?set the first native iteration-1 map}"
: "${NATIVE_B_MAP:?set the independent native iteration-1 map}"
: "${SCORE_DIR:?set the candidate iteration-2 score-dump directory}"
: "${OUTPUT_JSON:?set a new output JSON path}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CUDA_BINARY=${REPO_ROOT}/recovar/cuda/libcuda_backproject.so
SCORE_IMAGE_IDS=${SCORE_IMAGE_IDS:-1001,1021,1022,1056,1084,1108,1121,1155}
DECOMPOSITION_REPORT=${DECOMPOSITION_REPORT:-}

test -x "${PIXI_PY}"
test -s "${CUDA_BINARY}"
test ! -e "${OUTPUT_JSON}"

IFS=, read -r -a image_ids <<< "${SCORE_IMAGE_IDS}"
score_args=()
for image_id in "${image_ids[@]}"; do
  score_path=${SCORE_DIR}/local_score_it002_image_${image_id}_single_class.npz
  test -s "${score_path}"
  score_args+=(--score-dump "${score_path}")
done
decomposition_args=()
if [[ -n "${DECOMPOSITION_REPORT}" ]]; then
  test -s "${DECOMPOSITION_REPORT}"
  decomposition_args+=(--decomposition-report "${DECOMPOSITION_REPORT}")
fi

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORMS JAX_PLATFORM_NAME
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}

"${PIXI_PY}" -m scripts.analyze_vdam_reference_repeat_projection \
  --map "candidate=${CANDIDATE_MAP}" \
  --map "native_a=${NATIVE_A_MAP}" \
  --map "native_b=${NATIVE_B_MAP}" \
  "${score_args[@]}" \
  "${decomposition_args[@]}" \
  --output "${OUTPUT_JSON}"

sha256sum "${OUTPUT_JSON}"
