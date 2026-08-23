#!/usr/bin/env bash
#SBATCH --job-name=k1c05-bpref-pool3
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?set an empty absolute run root}"
: "${RUNTIME_ROOT:?set an empty absolute runtime root}"
: "${EXPECTED_REPO_HEAD:?pin repository HEAD}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?pin RECOVAR source diff}"
: "${EXPECTED_LAUNCHER_SHA256:?pin this launcher}"
: "${EXPECTED_ANALYZER_SHA256:?pin the raw-accumulator analyzer}"
: "${PARTICLE_POOL_SIZE:?set the positive RELION-style particle launch pool size}"
test "${PARTICLE_POOL_SIZE}" -gt 0

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/5_very_high_noise_100k_g256_white_noise10_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
NATIVE_DUMP=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_bpref_boundary_retry_20260822T1515ET/native/raw_bpref
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_stage_split_20260821T1600ET/cuda/libcuda_backproject.so
RELION_BIND=${SOURCE_ROOT}/relion_bind_build/shared
LAUNCHER=${REPO}/scripts/run_k1_case05_it1_bpref_pool3.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_half1_raw_accumulator.py
RECOVAR_OUT=${RUN_ROOT}/output
RECOVAR_ACCUM=${RUN_ROOT}/accum
ANALYSIS=${RUN_ROOT}/analysis
PROVENANCE=${RUN_ROOT}/provenance

test "${RUN_ROOT#/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/}" != "${RUN_ROOT}"
test "${RUNTIME_ROOT#/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/}" != "${RUNTIME_ROOT}"
test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -z "$(find "${RECOVAR_OUT}" "${RECOVAR_ACCUM}" "${ANALYSIS}" -mindepth 1 -print -quit)"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em scripts/run_full_refinement.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = b5178b902aafaa945bf7271421edab28bd75f0291484006833e0da0ebc91494b
RELION_BIND_LIB=$(find "${RELION_BIND}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit)
test "$(sha256sum "${RELION_BIND_LIB}" | awk '{print $1}')" = 56acdddab6d25bd83aac5e338162c4819f85ec26f68511458618f143c2b604b7

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

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
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
export RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE=${PARTICLE_POOL_SIZE}
export RECOVAR_BPREF_ACCUM_DUMP_DIR=${RECOVAR_ACCUM}
export RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION=1
export RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID=case05-it1-pool${PARTICLE_POOL_SIZE}
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${PROVENANCE}/git_status_${SLURM_JOB_ID}.txt"
git -C "${REPO}" diff --binary -- recovar/em scripts/run_full_refinement.py \
    > "${PROVENANCE}/recovar_source_diff_${SLURM_JOB_ID}.patch"
sha256sum "${CUDA_LIB}" "${RELION_BIND_LIB}" "${LAUNCHER}" "${ANALYZER}" \
    "${DATA}/particles.star" > "${PROVENANCE}/static_inputs_${SLURM_JOB_ID}.sha256"

cd "${REPO}"
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
print(recovar.__file__)
print(jax.__file__)
print(jax.devices())
PY

command=("${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${RECOVAR_OUT}" --max_iter 1 --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed 1705 --perturb_seed 1705
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --benchmark_ledger_json "${RECOVAR_OUT}/benchmark_ledger.json"
    --timing_dir "${RECOVAR_OUT}/timing"
    --save_intermediates_dir "${RECOVAR_OUT}/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${CONTROL}/run_it000_data.star"
    --relion_optimiser "${CONTROL}/run_it000_optimiser.star"
    --relion_init_dir "${CONTROL}")
printf '%q ' "${command[@]}" > "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
start=$(date +%s)
"${command[@]}" > "${RECOVAR_OUT}/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))

if test "${PARTICLE_POOL_SIZE}" -gt 1; then
    rg -q "grouping consecutive fresh K=1 BPref particles.*pool_size=${PARTICLE_POOL_SIZE}" \
        "${RECOVAR_OUT}/run.log"
else
    rg -q 'one adjoint launch per particle' "${RECOVAR_OUT}/run.log"
fi
test -s "${RECOVAR_ACCUM}/recovar_bpref_accum_it001.npz"
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${RECOVAR_OUT}/run.log"

for half in 1 2; do
    "${PYTHON}" -m scripts.analyze_k1_half1_raw_accumulator \
        --recovar-bpref "${RECOVAR_ACCUM}/recovar_bpref_accum_it001.npz" \
        --native-data "${NATIVE_DUMP}/raw_bpref_data_rank0${half}_call0000.bin" \
        --native-weight "${NATIVE_DUMP}/raw_bpref_weight_rank0${half}_call0000.bin" \
        --half "${half}" --expected-local-iteration 1 --physical-iteration 1 --first-shell 1 \
        --output-json "${ANALYSIS}/K1_CASE05_IT1_BPREF_POOL3_HALF${half}.json"
done

printf '{"job_id":"%s","wall_s":%d,"pool_size":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${PARTICLE_POOL_SIZE}" \
    > "${PROVENANCE}/walltime_${SLURM_JOB_ID}.json"
sha256sum "${RECOVAR_ACCUM}/recovar_bpref_accum_it001.npz" "${ANALYSIS}"/*.json \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
