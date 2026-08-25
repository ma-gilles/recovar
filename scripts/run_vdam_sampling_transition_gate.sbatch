#!/usr/bin/env bash
#SBATCH --job-name=vdam-sampling-gate
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-sampling-gate-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${BASELINE_ROOT:?set the completed paired baseline case root}"
: "${EXPECTED_REPO_HEAD:?pin the experimental source head}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified RECOVAR CUDA binary}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
STOP_ITERATION=${STOP_ITERATION:-92}
STOP_TAG=$(printf '%03d' "${STOP_ITERATION}")
CUDA_BINARY=${REPO_ROOT}/recovar/cuda/libcuda_backproject.so
DATA_DIR=${BASELINE_ROOT}/data
RELION_DIR=${BASELINE_ROOT}/relion
RECOVAR_DIR=${OUTPUT_ROOT}/recovar
ANALYSIS_DIR=${OUTPUT_ROOT}/analysis
PROVENANCE_DIR=${OUTPUT_ROOT}/provenance

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -s "${DATA_DIR}/particles.star"
test -s "${RELION_DIR}/run_it${STOP_TAG}_model.star"
test -x "${PIXI_PY}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
mkdir -p "${RECOVAR_DIR}" "${ANALYSIS_DIR}" "${PROVENANCE_DIR}"
test -z "$(find "${RECOVAR_DIR}" -mindepth 1 -print -quit)"

unset PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA RECOVAR_CUDA_DISABLE
export PYTHONPATH=${REPO_ROOT}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_DUMP_DIR=${OUTPUT_ROOT}/accuracy_inputs
export RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_DUMP_ITERATIONS=90
export JAX_COMPILATION_CACHE_DIR=${BASELINE_ROOT%/*}/jax_compilation_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.01
export TMPDIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" \
  "${RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_DUMP_DIR}"

git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/repo_head.txt"
sha256sum "${CUDA_BINARY}" "${DATA_DIR}/particles.star" > "${PROVENANCE_DIR}/input_sha256.txt"
nvidia-smi -q > "${PROVENANCE_DIR}/nvidia_smi.txt"

COMMAND=(
  "${PIXI_PY}" -m scripts.run_ab_initio
  --i "${DATA_DIR}/particles.star"
  --o "${RECOVAR_DIR}/run"
  --nr_iter 200
  --grad_write_iter 1
  --K 1
  --tau2_fudge 4
  --sym C1
  --do_run_C1 1
  --particle_diameter 200.0
  --random_seed 0
  --healpix_order 1
  --oversampling 1
  --offset_range 6
  --offset_step 2
  --padding_factor 1
  --image_batch_size 500
  --datadir "${DATA_DIR}"
  --gpu 0
  --require_custom_cuda
  --diagnostic_stop_after_iteration "${STOP_ITERATION}"
)
printf '%q ' "${COMMAND[@]}" > "${PROVENANCE_DIR}/recovar_command.sh"
printf '\n' >> "${PROVENANCE_DIR}/recovar_command.sh"

/usr/bin/time -v srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --cpu-bind=none \
  "${COMMAND[@]}" > "${RECOVAR_DIR}/runner.stdout" 2> "${RECOVAR_DIR}/runner.stderr"

"${PIXI_PY}" "${REPO_ROOT}/scripts/audit_vdam_expected_accuracy.py" \
  --input-star "${DATA_DIR}/particles.star" \
  --previous-prefix "${RECOVAR_DIR}/run_it089" \
  --target-prefix "${RELION_DIR}/run_it090" \
  --candidate-meta "${RECOVAR_DIR}/run_it090_recovar_meta.json" \
  --live-inputs "${RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_DUMP_DIR}/iter090_expected_accuracy_inputs.npz" \
  --output "${ANALYSIS_DIR}/expected_accuracy_it090.json"

map_args=()
for iteration in 69 70 79 80 85 86 87 88 89 90 91 92; do
  map_args+=(--checkpoint "${iteration}")
done
"${PIXI_PY}" "${REPO_ROOT}/scripts/audit_vdam_kclass_trajectory.py" \
  --candidate-dir "${RECOVAR_DIR}" --reference-dir "${RELION_DIR}" --K 1 \
  "${map_args[@]}" --minimum-fsc-auc 0.999 --minimum-assignment-accuracy 0.999 \
  --output-json "${ANALYSIS_DIR}/sampling_transition_maps.json" \
  --output-shells-npz "${ANALYSIS_DIR}/sampling_transition_shells.npz"

"${PIXI_PY}" "${REPO_ROOT}/scripts/audit_vdam_particle_state_trajectory.py" \
  --recovar-dir "${RECOVAR_DIR}" --relion-dir "${RELION_DIR}" \
  --iterations 82 83 84 85 86 87 88 89 90 91 92 \
  --output "${ANALYSIS_DIR}/sampling_transition_particles.json"

sha256sum "${ANALYSIS_DIR}"/* > "${PROVENANCE_DIR}/analysis_sha256.txt"
touch "${OUTPUT_ROOT}/COMPLETED"
