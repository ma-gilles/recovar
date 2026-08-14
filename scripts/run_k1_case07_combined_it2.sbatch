#!/usr/bin/env bash
#SBATCH --job-name=k1c07-combined-it2
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/slurm/k1c07-combined-it2-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/slurm/k1c07-combined-it2-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:30:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"
: "${EXPECTED_REPO_DIFF_SHA256:?submission must pin repository diff}"
: "${EXPECTED_RELION_BIND_SHA256:?submission must pin RELION binding hash}"
: "${EXPECTED_CUDA_SHA256:?submission must pin CUDA library hash}"
: "${EXPERIMENT_ROOT:?submission must set EXPERIMENT_ROOT}"
: "${EXPERIMENT_RUNTIME:?submission must set EXPERIMENT_RUNTIME}"

ROOT=${EXPERIMENT_ROOT}
RUNTIME=${EXPERIMENT_RUNTIME}
REPO=${K1_REPO_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810}
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case07_combined_it2.sbatch
SOURCE_CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/cases/7_anisotropic_100k_g256_white_noise1_bf80
DATA=${SOURCE_CASE}/data
RELION=${SOURCE_CASE}/relion_ref
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_cuda126_compute80_ptx_e791e875_20260814T0948ET/cuda128_sm90_control/build/libcuda_backproject_cuda128_sm90.so
RELION_BIND=${K1_RELION_BIND_BUILD_DIR:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810/recovar/relion_bind/build}
OUTPUT=${ROOT}/output
ANALYSIS=${ROOT}/analysis
ACCUM=${ROOT}/accum
PREJOIN=${ROOT}/prejoin

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${RELION_BIND}"/_relion_bind_core*.so | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -s "${RELION}/run_it000_data.star" -a -s "${RELION}/run_it002_optimiser.star"
test -s "${DATA}/particles.star"
test ! -e "${OUTPUT}"
mkdir -p "${ROOT}/provenance" "${OUTPUT}" "${ANALYSIS}" "${ACCUM}" "${PREJOIN}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
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
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_BPREF_ACCUM_DUMP_DIR=${ACCUM}
export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${PREJOIN}
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=2
export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case07_combined_physical_it2
unset RECOVAR_K1_COARSE_GAUSSIAN_FFI
unset RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF
unset RECOVAR_K1_RELION_PER_IMAGE_COARSE_FFT
unset RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS
unset RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject
from recovar.relion_bind import _relion_bind_core as bind

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert ".pixi/envs/default/" in str(Path(jax.__file__).resolve())
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert hasattr(bind, "auto_refine_randomise_half_orders")
print(jax.devices())
print(bind.__file__)
PY

COMMAND=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter 2 --skip_final_iteration
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
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.csv"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" "${RELION_BIND}"/_relion_bind_core*.so \
    "${DATA}/particles.star" "${RELION}/run_it000_optimiser.star" \
    > "${ROOT}/provenance/static_${SLURM_JOB_ID}.sha256"

start=$(date +%s)
"${COMMAND[@]}" > "${OUTPUT}/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))
test -s "${OUTPUT}/refinement_results.npz"
test -s "${OUTPUT}/intermediates/it001_half1_reg.mrc"
test -s "${ACCUM}/recovar_bpref_accum_it002.npz"
test -s "${PREJOIN}/recovar_bpref_prejoin_it002.npz"
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run.log"
rg -q 'Applied RELION fresh paired AutoRefine particle order with effective seed 1708' "${OUTPUT}/run.log"
rg -q 'K=1 RELION coarse Gaussian FFI enabled (guarded fresh-K=1 default)' "${OUTPUT}/run.log"
rg -q 'K=1 RELION coarse CUDA sincosf translation enabled (guarded fresh-K=1 default)' "${OUTPUT}/run.log"
rg -q 'K=1 RELION per-image coarse FFT enabled (guarded fresh-K=1 default)' "${OUTPUT}/run.log"
! rg -q 'RELION exact coarse operands enabled' "${OUTPUT}/run.log"
rg -q 'global; batching only consecutive equal-size supports' "${OUTPUT}/run.log"
rg -q 'one adjoint launch per particle' "${OUTPUT}/run.log"

"${PYTHON}" -m scripts.audit_k1_fsc_trajectory \
    --case-root "${SOURCE_CASE}" --recovar-dir "${OUTPUT}" --relion-dir "${RELION}" \
    --numbered-only \
    --output-json "${ANALYSIS}/case07_combined_fsc_it2.json" \
    --output-markdown "${ANALYSIS}/case07_combined_fsc_it2.md" \
    --output-shellwise-npz "${ANALYSIS}/case07_combined_fsc_it2_shellwise.npz" \
    --min-cross-merged-fsc-auc 0 --min-merged-gt-delta -1
"${PYTHON}" -m scripts.audit_k1_intermediate_trajectory \
    --case-root "${SOURCE_CASE}" --recovar-dir "${OUTPUT}" --relion-dir "${RELION}" \
    --output-json "${ANALYSIS}/case07_combined_state_it2.json" \
    --output-markdown "${ANALYSIS}/case07_combined_state_it2.md"

printf '{"job_id":"%s","wall_s":%d,"max_physical_iteration":2,"intervention":"guarded_per_image_coarse_fft_plus_live_noise_plus_exact_bpref_operands_plus_global_relion_particle_accumulation"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${ANALYSIS}/case07_combined_fsc_it2.json" \
    "${ANALYSIS}/case07_combined_fsc_it2_shellwise.npz" \
    "${ANALYSIS}/case07_combined_state_it2.json" \
    "${ACCUM}/recovar_bpref_accum_it002.npz" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
