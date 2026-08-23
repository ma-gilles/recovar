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
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case07_combined_it2.sbatch
SOURCE_CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/cases/7_anisotropic_100k_g256_white_noise1_bf80
DATA=${SOURCE_CASE}/data
RELION=${SOURCE_CASE}/relion_ref
CUDA_LIB=${K1_CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_cuda126_compute80_ptx_e791e875_20260814T0948ET/cuda128_sm90_control/build/libcuda_backproject_cuda128_sm90.so}
RELION_BIND=${REPO}/recovar/relion_bind/build
OUTPUT=${ROOT}/output
ANALYSIS=${ROOT}/analysis
ACCUM=${ROOT}/accum
PREJOIN=${ROOT}/prejoin
MAX_ITER=${K1_CASE07_MAX_ITER:-2}
BPREF_BOUNDARY_ITERATION=${K1_CASE07_BPREF_BOUNDARY_ITERATION:-2}
SKIP_POST_ANALYSIS=${K1_CASE07_SKIP_POST_ANALYSIS:-0}
COARSE_ARM=${K1_CASE07_COARSE_ARM:-default}
PREMASK_DIR=${K1_CASE07_PREMASK_DIR:-}
CONTRIBUTION_DIR=${K1_CASE07_CONTRIBUTION_DIR:-}
CONTRIBUTION_DEVICE_DIR=${K1_CASE07_CONTRIBUTION_DEVICE_DIR:-}
CONTRIBUTION_HALF=${K1_CASE07_CONTRIBUTION_HALF:-}
CONTRIBUTION_INDICES=${K1_CASE07_CONTRIBUTION_INDICES:-}
CONTRIBUTION_IMAGE_NAMES_NPY=${K1_CASE07_CONTRIBUTION_IMAGE_NAMES_NPY:-}
CONTRIBUTION_STACK_SHA256=${K1_CASE07_CONTRIBUTION_STACK_SHA256:-}
BPREF_PARTICLE_POOL_SIZE=${K1_CASE07_BPREF_PARTICLE_POOL_SIZE:-}
BPREF_EXECUTION_ORDER_CHUNK_SIZE=${K1_CASE07_BPREF_EXECUTION_ORDER_CHUNK_SIZE:-}
BPREF_BLOCK_TOPOLOGY=${K1_CASE07_BPREF_BLOCK_TOPOLOGY:-0}
BPREF_REVERSE_PHYSICAL_ORDER=${K1_CASE07_BPREF_REVERSE_PHYSICAL_ORDER:-0}
case "${MAX_ITER}:${BPREF_BOUNDARY_ITERATION}:${SKIP_POST_ANALYSIS}" in
    1:1:0|1:1:1|2:2:0|2:2:1) ;;
    *) echo "unsupported case-7 max/boundary/skip tuple" >&2; exit 2 ;;
esac
case "${COARSE_ARM}" in
    default|native_texture) ;;
    *) echo "unsupported K1_CASE07_COARSE_ARM=${COARSE_ARM}" >&2; exit 2 ;;
esac
if [[ "${COARSE_ARM}" = native_texture && "${MAX_ITER}" -lt 2 ]]; then
    echo "native_texture requires physical iteration 2 Gaussian scoring" >&2
    exit 2
fi
if [[ -n "${CONTRIBUTION_DIR}" ]]; then
    case "${CONTRIBUTION_HALF}" in 1|2) ;; *) echo "contribution half must be 1 or 2" >&2; exit 2 ;; esac
    test -n "${CONTRIBUTION_INDICES}"
    test -f "${CONTRIBUTION_IMAGE_NAMES_NPY}"
    test "${#CONTRIBUTION_STACK_SHA256}" -eq 64
    if [[ -n "${CONTRIBUTION_DEVICE_DIR}" ]]; then
        test "${CONTRIBUTION_DEVICE_DIR}" != "${CONTRIBUTION_DIR}"
    fi
elif [[ -n "${CONTRIBUTION_DEVICE_DIR}" ]]; then
    echo "contribution device capture requires a contribution directory" >&2
    exit 2
fi
if [[ -n "${BPREF_PARTICLE_POOL_SIZE}" ]]; then
    case "${BPREF_PARTICLE_POOL_SIZE}:${BPREF_EXECUTION_ORDER_CHUNK_SIZE}:${BPREF_BLOCK_TOPOLOGY}" in
        3:3:1) ;;
        *) echo "particle-pool diagnostic requires pool size 3, execution chunk size 3, and block topology 1" >&2; exit 2 ;;
    esac
fi
case "${BPREF_REVERSE_PHYSICAL_ORDER}" in 0|1) ;; *) echo "reverse physical order must be 0 or 1" >&2; exit 2 ;; esac
if [[ -n "${BPREF_PARTICLE_POOL_SIZE}" && "${BPREF_REVERSE_PHYSICAL_ORDER}" = 1 ]]; then
    echo "particle-pool and reverse-order diagnostics must run as separate interventions" >&2
    exit 2
fi

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
export RECOVAR_BPREF_ACCUM_DUMP_DIR=${ACCUM}
export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${PREJOIN}
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=${BPREF_BOUNDARY_ITERATION}
export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case07_combined_physical_it${BPREF_BOUNDARY_ITERATION}
if [[ -n "${BPREF_PARTICLE_POOL_SIZE}" ]]; then
    export RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE=${BPREF_PARTICLE_POOL_SIZE}
    export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=${BPREF_EXECUTION_ORDER_CHUNK_SIZE}
    export RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=${BPREF_BLOCK_TOPOLOGY}
else
    export RECOVAR_K1_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT=1
fi
if [[ "${BPREF_REVERSE_PHYSICAL_ORDER}" = 1 ]]; then
    export RECOVAR_K1_BPREF_REVERSE_PHYSICAL_ORDER=1
fi
if [[ -n "${CONTRIBUTION_DIR}" ]]; then
    mkdir -p "${CONTRIBUTION_DIR}"
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR=${CONTRIBUTION_DIR}
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION=1
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF=${CONTRIBUTION_HALF}
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE=56
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES=${CONTRIBUTION_INDICES}
    export RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID=case07-it1-half${CONTRIBUTION_HALF}-panel
    export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY=${CONTRIBUTION_IMAGE_NAMES_NPY}
    export RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256=${CONTRIBUTION_STACK_SHA256}
    export RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE=1
    if [[ -n "${CONTRIBUTION_DEVICE_DIR}" ]]; then
        mkdir -p "${CONTRIBUTION_DEVICE_DIR}"
        export RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR=${CONTRIBUTION_DEVICE_DIR}
        export RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH=1
        export RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION=1
        export RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS=1
        export RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=1
    fi
fi
if [[ -n "${PREMASK_DIR}" ]]; then
    export RECOVAR_PREMASK_DUMP_DIR=${PREMASK_DIR}
    export RECOVAR_PREMASK_DUMP_PRESERVE_DTYPE=1
fi
unset RECOVAR_K1_COARSE_GAUSSIAN_FFI
unset RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF
unset RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS
unset RECOVAR_K1_RELION_EXACT_CTF_STAR
unset RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER
if [[ "${COARSE_ARM}" = native_texture ]]; then
    export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
    export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
    export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
    export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
    export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=1
fi

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
print(bind.__file__)
PY

COMMAND=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter "${MAX_ITER}" --skip_final_iteration
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
if [[ -n "${CONTRIBUTION_DIR}" ]]; then
    {
        echo "contribution_half=${CONTRIBUTION_HALF}"
        echo "contribution_original_indices=${CONTRIBUTION_INDICES}"
        echo "contribution_original_indices_sha256=$(printf '%s' "${CONTRIBUTION_INDICES}" | sha256sum | awk '{print $1}')"
        echo "contribution_image_names_npy=${CONTRIBUTION_IMAGE_NAMES_NPY}"
        echo "contribution_image_names_sha256=$(sha256sum "${CONTRIBUTION_IMAGE_NAMES_NPY}" | awk '{print $1}')"
        echo "contribution_stack_sha256=${CONTRIBUTION_STACK_SHA256}"
        echo "contribution_device_dir=${CONTRIBUTION_DEVICE_DIR}"
    } > "${ROOT}/provenance/contribution_selection_${SLURM_JOB_ID}.txt"
fi

start=$(date +%s)
"${COMMAND[@]}" > "${OUTPUT}/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))
test -s "${OUTPUT}/refinement_results.npz"
last_local_iteration=$(printf '%03d' "$(( MAX_ITER - 1 ))")
boundary_label=$(printf '%03d' "${BPREF_BOUNDARY_ITERATION}")
test -s "${OUTPUT}/intermediates/it${last_local_iteration}_half1_reg.mrc"
test -s "${ACCUM}/recovar_bpref_accum_it${boundary_label}.npz"
test -s "${PREJOIN}/recovar_bpref_prejoin_it${boundary_label}.npz"
if [[ -n "${CONTRIBUTION_DIR}" ]]; then
    test "$(find "${CONTRIBUTION_DIR}" -maxdepth 1 -name 'bpref_contribution_rows_it001_h*_*.npz' | wc -l)" -ge 1
    "${PYTHON}" - "${CONTRIBUTION_DIR}" "${CONTRIBUTION_INDICES}" <<'PY'
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])
expected = {int(value) for value in sys.argv[2].split(",")}
seen = set()
for path in root.glob("bpref_contribution_rows_it001_h*_*.npz"):
    with np.load(path, allow_pickle=False) as values:
        assert bool(values["high_precision_operand_bundle"])
        seen.update(np.asarray(values["original_indices"], dtype=np.int64).tolist())
assert seen == expected, (seen, expected)
PY
    if [[ -n "${CONTRIBUTION_DEVICE_DIR}" ]]; then
        test "$(find "${CONTRIBUTION_DEVICE_DIR}" -maxdepth 1 -name '*.device.npz' | wc -l)" -ge 1
    fi
fi
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run.log"
rg -q 'Applied RELION fresh paired AutoRefine particle order with effective seed 1708' "${OUTPUT}/run.log"
if (( MAX_ITER > 1 )); then
    rg -q 'K=1 RELION coarse Gaussian FFI enabled (guarded fresh-K=1 default)' "${OUTPUT}/run.log"
    rg -q 'K=1 RELION coarse CUDA sincosf translation enabled (guarded fresh-K=1 default)' "${OUTPUT}/run.log"
    rg -q 'K=1 RELION exact coarse operands enabled (guarded fresh-K=1 default)' "${OUTPUT}/run.log"
fi
if [[ "${COARSE_ARM}" = native_texture ]]; then
    rg -q 'K=1 RELION native texture coarse diagnostic enabled' "${OUTPUT}/run.log"
    rg -q 'one particle per full orientation grid' "${OUTPUT}/run.log"
else
    ! rg -q 'K=1 RELION native texture coarse diagnostic enabled' "${OUTPUT}/run.log"
fi
if [[ "${BPREF_REVERSE_PHYSICAL_ORDER}" = 1 ]]; then
    rg -q 'reversing fresh RELION physical BPref particle order' "${OUTPUT}/run.log"
else
    rg -q 'global; batching only consecutive equal-size supports' "${OUTPUT}/run.log"
fi
if [[ -n "${BPREF_PARTICLE_POOL_SIZE}" ]]; then
    rg -q 'grouping consecutive fresh K=1 BPref particles into RELION-sized launch pools' "${OUTPUT}/run.log"
else
    rg -q 'one adjoint launch per particle' "${OUTPUT}/run.log"
fi

if [[ "${SKIP_POST_ANALYSIS}" = 0 ]]; then
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
fi

printf '{"job_id":"%s","wall_s":%d,"max_physical_iteration":%d,"bpref_boundary_iteration":%d,"intervention":"exact_fresh_k1","coarse_arm":"%s","bpref_particle_pool_size":"%s","bpref_execution_order_chunk_size":"%s","bpref_block_topology":"%s","bpref_reverse_physical_order":"%s"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" "${BPREF_BOUNDARY_ITERATION}" \
    "${COARSE_ARM}" \
    "${BPREF_PARTICLE_POOL_SIZE}" "${BPREF_EXECUTION_ORDER_CHUNK_SIZE}" "${BPREF_BLOCK_TOPOLOGY}" "${BPREF_REVERSE_PHYSICAL_ORDER}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${ACCUM}/recovar_bpref_accum_it${boundary_label}.npz" \
    "${PREJOIN}/recovar_bpref_prejoin_it${boundary_label}.npz" \
    "${OUTPUT}/intermediates/it${last_local_iteration}_half1_reg.mrc" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
