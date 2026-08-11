#!/usr/bin/env bash
#SBATCH --job-name=k1c26-order-it2
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_ordered_exact_it2_20260810T1925ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_ordered_exact_it2_20260810T1925ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=01:00:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"
: "${EXPECTED_REPO_DIFF_SHA256:?submission must pin repository diff}"

ROOT=${K1_CASE26_RUN_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_ordered_exact_it2_20260810T1925ET}
RUNTIME=${K1_CASE26_RUNTIME_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_ordered_exact_it2_20260810T1925ET}
MAX_ITER=${K1_CASE26_MAX_ITER:-2}
case "${MAX_ITER}" in
    2|3) ;;
    *) echo "K1_CASE26_MAX_ITER must be 2 or 3" >&2; exit 2 ;;
esac
REPORT_STEM=case26_ordered_exact_it${MAX_ITER}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case26_ordered_exact_it2.sbatch
SOURCE_CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${SOURCE_CASE}/data
RELION=${SOURCE_CASE}/relion_ref
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_bpref_order_fix_20260810/relion_bind
OUTPUT=${ROOT}/output
ANALYSIS=${ROOT}/analysis
ACCUM=${ROOT}/accum
PREJOIN=${ROOT}/prejoin
PASS2_DUMP_DIR=${K1_CASE26_PASS2_DUMP_DIR:-}
PASS2_DUMP_IDS=${K1_CASE26_PASS2_DUMP_IDS:-23,66,305,459,497}
PASS2_DUMP_CURRENT_SIZE=${K1_CASE26_PASS2_DUMP_CURRENT_SIZE:-56}
STATE_SWAP_VARIANT=${K1_CASE26_STATE_SWAP_VARIANT:-}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${RELION_BIND}"/_relion_bind_core*.so | awk '{print $1}')" = 793919bdb09875304d630727f9930ca729e1d261f6a354cecce7221af728dced
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = 50a0b3b86478a31bd64e8202ed71a6bf70e28429262a957231cf19c50ceab569
test -s "${RELION}/run_it000_data.star" -a -s "${RELION}/run_it002_optimiser.star"
test -s "${DATA}/particles.star"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT}" "${ANALYSIS}" "${ACCUM}" "${PREJOIN}" "${ROOT}/provenance"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
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
if [[ -z "${STATE_SWAP_VARIANT}" ]]; then
    export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
fi
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_BPREF_ACCUM_DUMP_DIR=${ACCUM}
export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${PREJOIN}
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=2
export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case26_ordered_exact_physical_it2
if [[ -n "${PASS2_DUMP_DIR}" ]]; then
    mkdir -p "${PASS2_DUMP_DIR}"
    export RECOVAR_PASS2_DUMP_DIR=${PASS2_DUMP_DIR}
    export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${PASS2_DUMP_IDS}
    export RECOVAR_PASS2_DUMP_CURRENT_SIZE=${PASS2_DUMP_CURRENT_SIZE}
    export RECOVAR_PASS2_DUMP_ITERATION=${MAX_ITER}
fi
unset RECOVAR_K1_COARSE_GAUSSIAN_FFI RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE
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
assert Path(jax.__file__).resolve().is_relative_to(root / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert hasattr(bind, "auto_refine_randomise_half_orders")
print(jax.devices())
PY

COMMAND=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter "${MAX_ITER}" --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image-fourier-backend relion_cuda
    --image_batch_size 187 --rotation_block_size 8192 --seed 1726 --perturb_seed 1726
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass
    --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json"
    --timing_dir "${OUTPUT}/timing"
    --save_intermediates_dir "${OUTPUT}/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${RELION}/run_it000_data.star"
    --relion_optimiser "${RELION}/run_it000_optimiser.star"
    --relion_init_dir "${RELION}"
)
if [[ -n "${STATE_SWAP_VARIANT}" ]]; then
    COMMAND+=(
        --perturb_replay_relion_dir "${RELION}"
        --state-swap-target-relion-iteration 2
        --state-swap-variant "${STATE_SWAP_VARIANT}"
        --state-swap-replay-relion-references
    )
fi
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.csv"
start=$(date +%s)
"${COMMAND[@]}" > "${OUTPUT}/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))

test -s "${OUTPUT}/refinement_results.npz"
test -s "${OUTPUT}/intermediates/it001_half1_reg.mrc"
test -s "${ACCUM}/recovar_bpref_accum_it002.npz"
test -s "${PREJOIN}/recovar_bpref_prejoin_it002.npz"
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run.log"
if [[ -n "${STATE_SWAP_VARIANT}" ]]; then
    rg -q "STATE-SWAP diagnostic: iteration=2 variant=${STATE_SWAP_VARIANT}" "${OUTPUT}/run.log"
else
    rg -q 'Applied RELION fresh paired AutoRefine particle order with effective seed 1727' "${OUTPUT}/run.log"
    rg -q 'global; batching only consecutive equal-size supports' "${OUTPUT}/run.log"
    rg -q 'one adjoint launch per particle' "${OUTPUT}/run.log"
fi
if [[ -n "${PASS2_DUMP_DIR}" ]]; then
    test "$(find "${PASS2_DUMP_DIR}" -maxdepth 1 -type f -name 'pass2_orig*_cs056.npz' | wc -l)" -eq 5
fi

fsc_audit_status=0
"${PYTHON}" -m scripts.audit_k1_fsc_trajectory \
    --case-root "${SOURCE_CASE}" --recovar-dir "${OUTPUT}" --relion-dir "${RELION}" \
    --numbered-only \
    --output-json "${ANALYSIS}/${REPORT_STEM}_fsc.json" \
    --output-markdown "${ANALYSIS}/${REPORT_STEM}_fsc.md" \
    --output-shellwise-npz "${ANALYSIS}/${REPORT_STEM}_fsc_shellwise.npz" \
    --min-cross-merged-fsc-auc 0 --min-merged-gt-delta -1 || fsc_audit_status=$?
# This is deliberately a short prefix discriminator against an 11-iteration
# reference.  The trajectory auditor returns 2 for that expected topology
# mismatch after writing the requested scientific comparison.
test "${fsc_audit_status}" -eq 0 -o "${fsc_audit_status}" -eq 2
"${PYTHON}" -m scripts.audit_k1_intermediate_trajectory \
    --case-root "${SOURCE_CASE}" --recovar-dir "${OUTPUT}" --relion-dir "${RELION}" \
    --output-json "${ANALYSIS}/${REPORT_STEM}_state.json" \
    --output-markdown "${ANALYSIS}/${REPORT_STEM}_state.md"

printf '{"job_id":"%s","wall_s":%d,"max_physical_iteration":%d,"intervention":"live_noise_plus_exact_bpref_operands_plus_global_relion_particle_accumulation","state_swap_variant":"%s"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" "${STATE_SWAP_VARIANT}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${ANALYSIS}/${REPORT_STEM}_fsc.json" \
    "${ANALYSIS}/${REPORT_STEM}_fsc_shellwise.npz" \
    "${ANALYSIS}/${REPORT_STEM}_state.json" \
    "${ACCUM}/recovar_bpref_accum_it002.npz" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
