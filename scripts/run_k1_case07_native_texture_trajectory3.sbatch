#!/usr/bin/env bash
#SBATCH --job-name=k1c07-native-it3
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/slurm/k1c07-native-it3-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/slurm/k1c07-native-it3-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=06:00:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?pin launcher hash}"
: "${EXPECTED_REPO_HEAD:?pin repository head}"
: "${EXPECTED_REPO_DIFF_SHA256:?pin repository diff}"
: "${EXPECTED_RELION_BIND_SHA256:?pin RELION binding hash}"
: "${EXPECTED_CUDA_SHA256:?pin CUDA library hash}"
: "${EXPECTED_BUILDER_SHA256:?pin audit builder hash}"
: "${EXPECTED_ANALYZER_SHA256:?pin movement analyzer hash}"
: "${EXPERIMENT_ROOT:?set experiment root}"
: "${EXPERIMENT_RUNTIME:?set runtime root}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case07_native_texture_trajectory3.sbatch
BUILDER=${REPO}/scripts/build_em_particle_audit_archive.py
ANALYZER=${REPO}/scripts/analyze_k1_autonomous_boundary_movement.py
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/cases/7_anisotropic_100k_g256_white_noise1_bf80
DATA=${CASE}/data
RELION=${CASE}/relion_ref
BASELINE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case07_combined_it2_20260810T1920ET
CUDA_LIB=${K1_CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_cuda126_compute80_ptx_e791e875_20260814T0948ET/cuda128_sm90_control/build/libcuda_backproject_cuda128_sm90.so}
RELION_BIND=${K1_RELION_BIND_BUILD_DIR:-${REPO}/recovar/relion_bind/build}
OUTPUT=${EXPERIMENT_ROOT}/output
ANALYSIS=${EXPERIMENT_ROOT}/analysis

test -f "${EXPERIMENT_ROOT}/SAFE_TO_DELETE" -a -f "${EXPERIMENT_RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${RELION_BIND}"/_relion_bind_core*.so | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${BUILDER}" | awk '{print $1}')" = "${EXPECTED_BUILDER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test -s "${DATA}/particles.star" -a -s "${RELION}/run_it003_data.star"
test -s "${BASELINE}/analysis/particle_autonomous_it1_it2_arrays.npz"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT}" "${ANALYSIS}" "${EXPERIMENT_ROOT}/provenance"
git -C "${REPO}" status --short --branch \
    > "${EXPERIMENT_ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git -C "${REPO}" diff --binary \
    > "${EXPERIMENT_ROOT}/provenance/repo_diff_${SLURM_JOB_ID}.patch"
test "$(sha256sum "${EXPERIMENT_ROOT}/provenance/repo_diff_${SLURM_JOB_ID}.patch" | awk '{print $1}')" \
    = "${EXPECTED_REPO_DIFF_SHA256}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/pycache
export JAX_COMPILATION_CACHE_DIR=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/jax_cache
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

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_TRANSLATION_GRID=1
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=1
if [[ -n "${K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE:-}" ]]; then
    [[ "${K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]]
    export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=${K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE}
fi
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER
unset RECOVAR_BPREF_ACCUM_DUMP_DIR RECOVAR_BPREF_PREJOIN_DUMP_DIR

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject
from recovar.relion_bind import _relion_bind_core as bind

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert Path(jax.__file__).resolve().is_relative_to(root / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert hasattr(bind, "auto_refine_randomise_half_orders")
print(jax.devices())
PY

COMMAND=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter 3 --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed 1707 --perturb_seed 1707
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json"
    --timing_dir "${OUTPUT}/timing"
    --save_intermediates_dir "${OUTPUT}/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${RELION}/run_it000_data.star"
    --relion_optimiser "${RELION}/run_it000_optimiser.star"
    --relion_init_dir "${RELION}"
)
printf '%q ' "${COMMAND[@]}" > "${EXPERIMENT_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${EXPERIMENT_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
env | rg '^(K1_|RECOVAR_|RELION_)' | sort \
    > "${EXPERIMENT_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv \
    > "${EXPERIMENT_ROOT}/provenance/gpu_${SLURM_JOB_ID}.csv"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" "${RELION_BIND}"/_relion_bind_core*.so \
    "${BUILDER}" "${ANALYZER}" "${DATA}/particles.star" \
    > "${EXPERIMENT_ROOT}/provenance/static_${SLURM_JOB_ID}.sha256"

start=$(date +%s)
"${COMMAND[@]}" > "${OUTPUT}/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))
test -s "${OUTPUT}/refinement_results.npz"
test -s "${OUTPUT}/intermediates/it002_half1_reg.mrc"
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run.log"
rg -q 'Applied RELION fresh paired AutoRefine particle order with effective seed 1708' "${OUTPUT}/run.log"
test "$(rg -c 'one particle per full orientation grid' "${OUTPUT}/run.log")" -ge 4

"${PYTHON}" "${BUILDER}" --output-dir "${OUTPUT}" \
    --destination "${ANALYSIS}/refinement_results_audit_source_order.npz"
"${PYTHON}" -m scripts.audit_em_particle_state_distribution \
    --recovar-results "${ANALYSIS}/refinement_results_audit_source_order.npz" \
    --recovar-particles-star "${DATA}/particles.star" \
    --relion-star "1:${RELION}/run_it001_data.star" \
    --relion-star "2:${RELION}/run_it002_data.star" \
    --relion-star "3:${RELION}/run_it003_data.star" \
    --recovar-iteration 0 --recovar-iteration 1 --recovar-iteration 2 \
    --relion-iteration-offset 1 \
    --output-json "${ANALYSIS}/particle_autonomous_it1_it3.json" \
    --output-npz "${ANALYSIS}/particle_autonomous_it1_it3_arrays.npz" \
    --output-hash-manifest "${ANALYSIS}/particle_autonomous_it1_it3.json.sha256"
"${PYTHON}" -m scripts.audit_k1_fsc_trajectory \
    --case-root "${CASE}" --recovar-dir "${OUTPUT}" --relion-dir "${RELION}" \
    --numbered-only --allow-incomplete \
    --min-cross-merged-fsc-auc 0.995 --min-merged-gt-delta -0.002 \
    --output-json "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_FSC.json" \
    --output-markdown "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_FSC.md" \
    --output-shellwise-npz "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_FSC_shellwise.npz"
"${PYTHON}" -m scripts.audit_k1_intermediate_trajectory \
    --case-root "${CASE}" --recovar-dir "${OUTPUT}" --relion-dir "${RELION}" \
    --output-json "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_STATE.json" \
    --output-markdown "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_STATE.md"
"${PYTHON}" -m scripts.analyze_k1_autonomous_boundary_movement \
    --baseline-particles "${BASELINE}/analysis/particle_autonomous_it1_it2_arrays.npz" \
    --candidate-particles "${ANALYSIS}/particle_autonomous_it1_it3_arrays.npz" \
    --baseline-fsc "${BASELINE}/analysis/case07_combined_fsc_it2.json" \
    --candidate-fsc "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_FSC.json" \
    --output-json "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_MOVEMENT.json" \
    --output-markdown "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_MOVEMENT.md"

printf '{"job_id":"%s","wall_s":%d,"max_physical_iteration":3,"final_all_data":false,"coarse_arm":"native_texture"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" > "${EXPERIMENT_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${OUTPUT}/refinement_results.npz" \
    "${ANALYSIS}/particle_autonomous_it1_it3.json" \
    "${ANALYSIS}/particle_autonomous_it1_it3_arrays.npz" \
    "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_FSC.json" \
    "${ANALYSIS}/K1_CASE07_NATIVE_TEXTURE_IT3_MOVEMENT.json" \
    > "${EXPERIMENT_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${EXPERIMENT_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
