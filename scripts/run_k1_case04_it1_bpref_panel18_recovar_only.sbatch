#!/usr/bin/env bash
#SBATCH --job-name=k1c04-bpref18-r
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${NATIVE_RUN_ROOT:?NATIVE_RUN_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_REPO_DIFF_SHA256:?EXPECTED_REPO_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_SELECTION_SHA256:?EXPECTED_SELECTION_SHA256 is required}"
: "${EXPECTED_FACTOR_VALIDATION_SHA256:?EXPECTED_FACTOR_VALIDATION_SHA256 is required}"
: "${EXPECTED_INERTNESS_SHA256:?EXPECTED_INERTNESS_SHA256 is required}"
: "${ENABLE_LIVE_INITIAL_NOISE:?ENABLE_LIVE_INITIAL_NOISE is required}"

case "${ENABLE_LIVE_INITIAL_NOISE}" in
    0|1) ;;
    *) echo "ENABLE_LIVE_INITIAL_NOISE must be 0 or 1" >&2; exit 2 ;;
esac

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
NATIVE_ROOT=${NATIVE_RUN_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case04_it1_bpref_panel18_recovar_only.sbatch
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_stage_split_20260821T1600ET/cuda/libcuda_backproject.so
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared
ORIGINAL_INDICES=99886,71881,89328,77986,60268,16799,88741,18493,55303,80124,32732,63798,28614,92534,20635,29514,58368,75859

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -f "${NATIVE_ROOT}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${NATIVE_ROOT}/provenance/SELECTION.json" | awk '{print $1}')" = "${EXPECTED_SELECTION_SHA256}"
test "$(sha256sum "${NATIVE_ROOT}/analysis/RELION_FACTOR_VALIDATION.json" | awk '{print $1}')" = "${EXPECTED_FACTOR_VALIDATION_SHA256}"
test "$(sha256sum "${NATIVE_ROOT}/analysis/RELION_FACTOR_CAPTURE_INERTNESS.json" | awk '{print $1}')" = "${EXPECTED_INERTNESS_SHA256}"
test "$(find "${NATIVE_ROOT}/capture" -maxdepth 1 -name '*.bpre-v2.bin' | wc -l)" -eq 18
"${PYTHON}" - "${NATIVE_ROOT}/analysis/RELION_FACTOR_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["qualified"] is True
assert report["selected_topology_exact"] is True
assert report["minimum_signed_normalized_non_dc_fsc_auc"] >= 0.999999
PY

mkdir -p "${ROOT}/recovar/output" "${ROOT}/recovar/pass2" "${ROOT}/analysis" \
    "${ROOT}/logs" "${ROOT}/provenance"
for directory in "${ROOT}/recovar/output" "${ROOT}/recovar/pass2" "${ROOT}/analysis"; do
    test -z "$(find "${directory}" -mindepth 1 -print -quit)"
done
mkdir -p "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache" \
    "${RUNTIME}/pycache" "${RUNTIME}/jax_cache" "${RUNTIME}/cuda_cache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_CUDA_CACHE_DIR=${RUNTIME}/cuda_cache
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=${ENABLE_LIVE_INITIAL_NOISE}
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_PASS2_DUMP_DIR=${ROOT}/recovar/pass2
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${ORIGINAL_INDICES}
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=56
export RECOVAR_PASS2_DUMP_CLASS=1
unset RECOVAR_PASS2_DUMP_RAW_OPERANDS
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
git -C "${REPO}" status --short --branch > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
git -C "${REPO}" diff --binary | sha256sum > "${ROOT}/provenance/repo_diff_${SLURM_JOB_ID}.sha256"
sha256sum "${NATIVE_ROOT}/provenance/SELECTION.json" \
    "${NATIVE_ROOT}/analysis/RELION_FACTOR_VALIDATION.json" \
    "${NATIVE_ROOT}/analysis/RELION_FACTOR_CAPTURE_INERTNESS.json" \
    > "${ROOT}/provenance/native_inputs_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
"${PYTHON}" - <<'PY'
from pathlib import Path

import jax
import recovar
from recovar import cuda_backproject

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert Path(jax.__file__).resolve().is_relative_to(root / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
PY

recovar_command=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${ROOT}/recovar/output" --max_iter 1 --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed 1704 --perturb_seed 1704
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --save_intermediates_dir "${ROOT}/recovar/output/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${CONTROL}/run_it000_data.star"
    --relion_optimiser "${CONTROL}/run_it000_optimiser.star"
    --relion_init_dir "${CONTROL}"
)
printf '%q ' "${recovar_command[@]}" > "${ROOT}/provenance/command_recovar_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_recovar_${SLURM_JOB_ID}.sh"
recovar_start=$(date +%s)
"${recovar_command[@]}" > "${ROOT}/recovar/output/run.log" 2>&1
printf '{"wall_s":%d}\n' "$(( $(date +%s) - recovar_start ))" \
    > "${ROOT}/provenance/walltime_recovar_${SLURM_JOB_ID}.json"
test "$(find "${ROOT}/recovar/pass2" -maxdepth 1 -name '*.npz' | wc -l)" -eq 18
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' \
    "${ROOT}/recovar/output/run.log"

"${PYTHON}" -m scripts.analyze_k1_bpref_factor_boundary \
    --factor-directory "${NATIVE_ROOT}/capture" \
    --recovar-capture-directory "${ROOT}/recovar/pass2" \
    --selection-json "${NATIVE_ROOT}/provenance/SELECTION.json" \
    --capture-inertness-json "${NATIVE_ROOT}/analysis/RELION_FACTOR_CAPTURE_INERTNESS.json" \
    --output-json "${ROOT}/analysis/K1_CASE04_IT1_BPREF_PANEL18.json" \
    > "${ROOT}/logs/analyze_factor_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/analyze_factor_${SLURM_JOB_ID}.stderr"

find "${ROOT}/recovar" "${ROOT}/analysis" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","native_run_root":"%s","particle_count":18,"live_initial_noise":%s}\n' \
    "${SLURM_JOB_ID}" "${NATIVE_ROOT}" "${ENABLE_LIVE_INITIAL_NOISE}" \
    > "${ROOT}/provenance/completion_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
