#!/usr/bin/env bash
#SBATCH --job-name=k1c26-exact-it3
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_exact_it3_dump_20260810T1950ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_exact_it3_dump_20260810T1950ET/logs/run-%j.err
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

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_exact_it3_dump_20260810T1950ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_exact_it3_dump_20260810T1950ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case26_exact_it3_dump.sbatch
SOURCE_CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${SOURCE_CASE}/data
RELION=${SOURCE_CASE}/relion_ref
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_bpref_order_fix_20260810/relion_bind
OUTPUT=${ROOT}/output
ANALYSIS=${ROOT}/analysis
PASS2=${ROOT}/pass2

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${RELION_BIND}"/_relion_bind_core*.so | awk '{print $1}')" = 793919bdb09875304d630727f9930ca729e1d261f6a354cecce7221af728dced
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = 50a0b3b86478a31bd64e8202ed71a6bf70e28429262a957231cf19c50ceab569
test -s "${RELION}/run_it002_optimiser.star" -a -s "${RELION}/run_it003_data.star"
test -s "${DATA}/particles.star"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT}" "${ANALYSIS}" "${PASS2}" "${ROOT}/provenance"

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
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_PASS2_DUMP_DIR=${PASS2}
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=31,362,512,882,974
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=66
unset RECOVAR_K1_COARSE_GAUSSIAN_FFI RECOVAR_K1_RELION_LIVE_INITIAL_NOISE
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

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
print(jax.devices())
PY

COMMAND=(
    "${PYTHON}" -m scripts.run_multi_iter_parity
    --relion_dir "${RELION}"
    --data_star "${DATA}/particles.star"
    --iter 2 --max_iter 1
    --continuous-relion-noise-state
    --skip_final_iteration
    --image_batch_size 187
    --rotation_block_size 8192
    --output_dir "${OUTPUT}"
    --save_intermediates_dir "${OUTPUT}/intermediates"
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.csv"
"${COMMAND[@]}" > "${OUTPUT}/run.log" 2>&1

test -s "${OUTPUT}/refinement_results.npz"
test -s "${OUTPUT}/pmax_comparison_iter000.npz"
test "$(find "${PASS2}" -maxdepth 1 -type f -name 'pass2_orig*_cs066.npz' | wc -l)" -eq 5
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run.log"

"${PYTHON}" -m scripts.audit_em_particle_state_distribution \
    --recovar-results "${OUTPUT}/refinement_results.npz" \
    --recovar-particles-star "${DATA}/particles.star" \
    --relion-star "3:${RELION}/run_it003_data.star" \
    --recovar-iteration 0 --relion-iteration-offset 3 \
    --output-json "${ANALYSIS}/case26_exact_it3_particle_state.json" \
    --output-npz "${ANALYSIS}/case26_exact_it3_particle_state.npz" \
    --output-hash-manifest "${ANALYSIS}/case26_exact_it3_particle_state.sha256"

sha256sum "${ANALYSIS}/case26_exact_it3_particle_state.json" \
    "${OUTPUT}/pmax_comparison_iter000.npz" \
    "${PASS2}"/pass2_orig*_cs066.npz \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
