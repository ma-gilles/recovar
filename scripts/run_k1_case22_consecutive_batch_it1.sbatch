#!/usr/bin/env bash
#SBATCH --job-name=k1c22-batch-it1
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_consecutive_batch_it1_20260810T1945ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_consecutive_batch_it1_20260810T1945ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"
: "${EXPECTED_REPO_DIFF_SHA256:?submission must pin repository diff}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_consecutive_batch_it1_20260810T1945ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_consecutive_batch_it1_20260810T1945ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case22_consecutive_batch_it1.sbatch
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
SEALED_CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_control/relion
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_bpref_order_fix_20260810/relion_bind
SINGLETON_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fresh_order_production_it1_v2_20260810T1735ET
OUTPUT=${ROOT}/output
PREJOIN=${ROOT}/prejoin
ACCUM=${ROOT}/accum
ANALYSIS=${ROOT}/analysis

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${RELION_BIND}"/_relion_bind_core*.so | awk '{print $1}')" = 793919bdb09875304d630727f9930ca729e1d261f6a354cecce7221af728dced
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = 50a0b3b86478a31bd64e8202ed71a6bf70e28429262a957231cf19c50ceab569
test -s "${SINGLETON_ROOT}/accum/recovar_bpref_accum_it001.npz"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT}" "${PREJOIN}" "${ACCUM}" "${ANALYSIS}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*) unset "${variable_name}" ;; esac
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
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_BPREF_ACCUM_DUMP_DIR=${ACCUM}
export RECOVAR_BPREF_PREJOIN_DUMP_DIR=${PREJOIN}
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=1
export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case22_physical_it1_consecutive_equal_size_batch
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
assert Path(jax.__file__).resolve().is_relative_to(root / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert hasattr(bind, "auto_refine_randomise_half_orders")
print(jax.devices())
print(bind.__file__)
PY

COMMAND=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter 1 --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed 1722 --perturb_seed 1722
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json"
    --timing_dir "${OUTPUT}/timing"
    --save_intermediates_dir "${OUTPUT}/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${SEALED_CONTROL}/run_it000_data.star"
    --relion_optimiser "${SEALED_CONTROL}/run_it000_optimiser.star"
    --relion_init_dir "${SEALED_CONTROL}"
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

start=$(date +%s)
"${COMMAND[@]}" > "${OUTPUT}/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))

ACCUM_FILE=${ACCUM}/recovar_bpref_accum_it001.npz
SINGLETON_FILE=${SINGLETON_ROOT}/accum/recovar_bpref_accum_it001.npz
test -s "${ACCUM_FILE}"
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run.log"
rg -q 'global; batching only consecutive equal-size supports' "${OUTPUT}/run.log"
rg -q 'one adjoint launch per particle' "${OUTPUT}/run.log"
rg -q '1490 images -> 2 buckets' "${OUTPUT}/run.log"

"${PYTHON}" - "${ACCUM_FILE}" "${SINGLETON_FILE}" "${ANALYSIS}/K1_CASE22_IT1_BATCH_VS_SINGLETON.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

candidate_path, singleton_path, output_path = map(Path, sys.argv[1:])
report = {
    "schema": "recovar-k1-case22-it1-consecutive-batch-vs-singleton-v1",
    "physical_iteration": 1,
    "candidate": str(candidate_path),
    "singleton_control": str(singleton_path),
    "arrays": {},
}
with np.load(candidate_path, allow_pickle=False) as candidate, np.load(singleton_path, allow_pickle=False) as control:
    assert set(candidate.files) == set(control.files)
    for key in candidate.files:
        left = np.asarray(candidate[key])
        right = np.asarray(control[key])
        if left.dtype.kind not in "biufc" or left.shape != right.shape:
            continue
        difference = np.abs(left.astype(np.complex128) - right.astype(np.complex128))
        denominator = max(float(np.linalg.norm(right.astype(np.complex128).ravel())), np.finfo(np.float64).tiny)
        report["arrays"][key] = {
            "bitwise_equal": bool(np.array_equal(left, right)),
            "max_abs": float(np.max(difference, initial=0.0)),
            "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
        }
report["all_numeric_arrays_bitwise_equal"] = all(
    value["bitwise_equal"] for value in report["arrays"].values()
)
report["candidate_sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
report["singleton_sha256"] = hashlib.sha256(singleton_path.read_bytes()).hexdigest()
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

printf '{"job_id":"%s","wall_s":%d,"physical_iteration":1,"mode":"consecutive_equal_size_scoring_batches_with_sequential_particle_bpref"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" "${RELION_BIND}"/_relion_bind_core*.so \
    "${ACCUM_FILE}" "${ANALYSIS}/K1_CASE22_IT1_BATCH_VS_SINGLETON.json" \
    > "${ROOT}/provenance/outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
