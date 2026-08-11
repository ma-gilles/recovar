#!/usr/bin/env bash
#SBATCH --job-name=k1c26-it1-scatter
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_it1_scatter_ab_20260811T0120ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_it1_scatter_ab_20260811T0120ET/logs/run-%j.err
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
: "${EXPECTED_REPO_DIFF_SHA256:?submission must pin tracked repository diff}"

ROOT=${RUN_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_it1_scatter_ab_20260811T0120ET}
RUNTIME=${RUNTIME_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_it1_scatter_ab_20260811T0120ET}
WIENER_DUMP_DIR=${K1_WIENER_BOUNDARY_DUMP_DIR:-}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case26_it1_scatter_ab.sbatch
SOURCE_CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${SOURCE_CASE}/data
RELION=${SOURCE_CASE}/relion_ref
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_bpref_order_fix_20260810/relion_bind

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${RELION_BIND}"/_relion_bind_core*.so | awk '{print $1}')" = 793919bdb09875304d630727f9930ca729e1d261f6a354cecce7221af728dced
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = 50a0b3b86478a31bd64e8202ed71a6bf70e28429262a957231cf19c50ceab569
test -s "${RELION}/run_it000_data.star" -a -s "${RELION}/run_it000_optimiser.star"
test -s "${DATA}/particles.star"

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
    "${PYTHONPYCACHEPREFIX}" "${JAX_COMPILATION_CACHE_DIR}" "${ROOT}/provenance"

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
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
if [[ -n "${WIENER_DUMP_DIR}" ]]; then
    mkdir -p "${WIENER_DUMP_DIR}"
    export RECOVAR_RELION_WIENER_BOUNDARY_DUMP_DIR=${WIENER_DUMP_DIR}
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

run_arm() {
    arm=$1
    block_topology=$2
    fused=$3
    diagnostic=$4
    arm_root=${ROOT}/${arm}
    output=${arm_root}/output
    prejoin=${arm_root}/prejoin
    accum=${arm_root}/accum
    test ! -e "${output}"
    mkdir -p "${output}" "${prejoin}" "${accum}"
    export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${prejoin}
    export RECOVAR_BPREF_ACCUM_DUMP_DIR=${accum}
    export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=1
    export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case26_it1_scatter_${arm}
    if test "${diagnostic}" = 1; then
        export RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH=1
        export RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION=1
        export RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS=${fused}
        export RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=${block_topology}
    else
        unset RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH
        unset RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION
        unset RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS
        unset RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY
    fi
    command=(
        "${PYTHON}" -m scripts.run_full_refinement
        --data_dir "${DATA}" --output "${output}" --max_iter 1 --skip_final_iteration
        --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
        --adaptive_oversampling 1 --init_resolution 30.0 --image-fourier-backend relion_cuda
        --image_batch_size 187 --rotation_block_size 8192 --seed 1726 --perturb_seed 1726
        --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
        --firstiter_cc --apply-initial-lowpass
        --benchmark_ledger_json "${output}/benchmark_ledger.json"
        --timing_dir "${output}/timing"
        --save_intermediates_dir "${output}/intermediates"
        --save_intermediates_skip_unregularized
        --relion_half_sets "${RELION}/run_it000_data.star"
        --relion_optimiser "${RELION}/run_it000_optimiser.star"
        --relion_init_dir "${RELION}"
    )
    printf '%q ' "${command[@]}" > "${arm_root}/command.sh"
    printf '\n' >> "${arm_root}/command.sh"
    start=$(date +%s)
    "${command[@]}" > "${output}/run.log" 2>&1
    wall_s=$(( $(date +%s) - start ))
    test -s "${prejoin}/recovar_bpref_prejoin_it001.npz"
    test -s "${accum}/recovar_bpref_accum_it001.npz"
    ! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${output}/run.log"
    rg -q 'Applied RELION fresh paired AutoRefine particle order with effective seed 1727' "${output}/run.log"
    rg -q 'one adjoint launch per particle' "${output}/run.log"
    if test "${arm}" = production; then
        rg -q 'fresh K=1 firstiter-CC uses RELION fused' "${output}/run.log"
    elif test "${fused}" = 1; then
        rg -q 'fused real/imaginary/weight atomics enabled' "${output}/run.log"
    fi
    printf '{"arm":"%s","diagnostic":%s,"block_topology":%s,"fused_atomics":%s,"wall_s":%d}\n' \
        "${arm}" "${diagnostic}" "${block_topology}" "${fused}" "${wall_s}" \
        > "${arm_root}/walltime.json"
}

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.csv"
IFS=, read -r -a scatter_arms <<< "${SCATTER_ARMS:-control,fused_block}"
for arm in "${scatter_arms[@]}"; do
    case "${arm}" in
        production) run_arm production 0 0 0 ;;
        control) run_arm control 0 0 1 ;;
        block_only) run_arm block_only 1 0 1 ;;
        fused_block) run_arm fused_block 1 1 1 ;;
        *) echo "unknown scatter arm: ${arm}" >&2; exit 2 ;;
    esac
done
if [[ -n "${WIENER_DUMP_DIR}" ]]; then
    test "$(find "${WIENER_DUMP_DIR}" -maxdepth 1 -type f -name 'recovar_wiener_boundary_*.npz' | wc -l)" -eq 2
fi
sha256sum "${ROOT}"/*/prejoin/recovar_bpref_prejoin_it001.npz \
    "${ROOT}"/*/accum/recovar_bpref_accum_it001.npz \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
