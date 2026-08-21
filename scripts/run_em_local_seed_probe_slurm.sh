#!/usr/bin/env bash
# Submit a focused K=1 local-search probe seeded from an existing RECOVAR
# refinement_results.npz. This is for EM speed debugging: by default it starts
# directly in the local exact branch and exits before tau/FSC/final output
# work. The default is the fast pose/score diagnostic path: score-only local
# search with profiling off. Set EM_LOCAL_PROBE_STOP_AFTER_LOCAL_SEARCH_SCORE_ONLY=0
# to exercise the full local M-step path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="em_local_seed_probe_${TIMESTAMP}_${RANDOM}"
OUTPUT_ROOT="${EM_LOCAL_PROBE_OUTPUT_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/${RUN_ID}}"
JAX_CACHE_DIR="${EM_LOCAL_PROBE_JAX_CACHE_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/jax_cache/em_local_seed_probe}"
PROJECTOR_CACHE_DIR="${EM_LOCAL_PROBE_PROJECTOR_CACHE_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/projector_cache/em_local_seed_probe}"
INITIAL_NOISE_CACHE_DIR="${EM_LOCAL_PROBE_INITIAL_NOISE_CACHE_DIR-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/noise_cache/em_local_seed_probe}"
REPO_KEY="$(printf '%s' "${REPO_ROOT}" | sha1sum | awk '{print substr($1, 1, 12)}')"
NATIVE_BUILD_ROOT="${EM_LOCAL_PROBE_NATIVE_BUILD_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/native_build/em_local_seed_probe/${REPO_KEY}}"
CUDA_LIB="${EM_LOCAL_PROBE_CUDA_LIB:-${NATIVE_BUILD_ROOT}/cuda/libcuda_backproject.so}"
RELION_BIND_BUILD_DIR="${EM_LOCAL_PROBE_RELION_BIND_BUILD_DIR:-${NATIVE_BUILD_ROOT}/relion_bind}"
FORCE_NATIVE_REBUILD="${EM_LOCAL_PROBE_FORCE_NATIVE_REBUILD:-0}"
FORCE_INSTALL="${EM_LOCAL_PROBE_FORCE_INSTALL:-0}"
DATA_DIR="${EM_LOCAL_PROBE_DATA_DIR:?set EM_LOCAL_PROBE_DATA_DIR}"
SEED_NPZ="${EM_LOCAL_PROBE_SEED_NPZ:?set EM_LOCAL_PROBE_SEED_NPZ}"
POSE_ITER="${EM_LOCAL_PROBE_POSE_ITER:-last}"
INIT_NOISE_FROM_SEED_NPZ="${EM_LOCAL_PROBE_INIT_NOISE_FROM_SEED_NPZ:-0}"
INIT_NOISE_ITER="${EM_LOCAL_PROBE_INIT_NOISE_ITER:-${POSE_ITER}}"
PROFILE_NAME="${EM_LOCAL_PROBE_PROFILE_NAME:-default}"
ACCOUNT="${SBATCH_ACCOUNT:-gilles}"
PARTITION="${SBATCH_PARTITION:-cryoem}"
CONSTRAINT="${SBATCH_CONSTRAINT:-}"
TIME_LIMIT="${EM_LOCAL_PROBE_TIME_LIMIT:-03:00:00}"
MEM="${EM_LOCAL_PROBE_MEM:-256G}"
CPUS="${EM_LOCAL_PROBE_CPUS:-8}"
CUDA_MODULE="${CUDA_MODULE:-cudatoolkit/12.8}"
K1_IMAGE_BATCH_SIZE="${K1_IMAGE_BATCH_SIZE:-64}"
K1_ROTATION_BLOCK_SIZE="${K1_ROTATION_BLOCK_SIZE:-8192}"
RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION="${RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION:-0.20}"
RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET="${RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET:-}"
MAX_ITER="${EM_LOCAL_PROBE_MAX_ITER:-1}"
HEALPIX_ORDER="${EM_LOCAL_PROBE_HEALPIX_ORDER:-4}"
MAX_HEALPIX_ORDER="${EM_LOCAL_PROBE_MAX_HEALPIX_ORDER:-4}"
AUTO_LOCAL_HEALPIX_ORDER="${EM_LOCAL_PROBE_AUTO_LOCAL_HEALPIX_ORDER:-4}"
RELION_CURRENT_SIZES="${EM_LOCAL_PROBE_RELION_CURRENT_SIZES:-256}"
SEED="${EM_LOCAL_PROBE_SEED:-1729}"
SAVE_INTERMEDIATES="${EM_LOCAL_PROBE_SAVE_INTERMEDIATES:-0}"
SAVE_INTERMEDIATES_SKIP_UNREGULARIZED="${EM_LOCAL_PROBE_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED:-1}"
STOP_AFTER_SCORE_ONLY="${EM_LOCAL_PROBE_STOP_AFTER_LOCAL_SEARCH_SCORE_ONLY:-1}"
if [[ -n "${EM_LOCAL_PROBE_STOP_AFTER_PROFILE+x}" ]]; then
  STOP_AFTER_PROFILE="${EM_LOCAL_PROBE_STOP_AFTER_PROFILE}"
elif [[ "${STOP_AFTER_SCORE_ONLY}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  STOP_AFTER_PROFILE=0
else
  STOP_AFTER_PROFILE=1
fi
STOP_AFTER_LOCAL_SEARCH="${EM_LOCAL_PROBE_STOP_AFTER_LOCAL_SEARCH:-${STOP_AFTER_PROFILE}}"
DIAGNOSTIC_SINGLE_HALF="${EM_LOCAL_PROBE_DIAGNOSTIC_SINGLE_HALF:-0}"
SKIP_LARGE_OUTPUTS="${EM_LOCAL_PROBE_SKIP_LARGE_OUTPUTS:-1}"
EXTRA_ARGS="${EM_LOCAL_PROBE_EXTRA_ARGS:-}"

if [[ -n "${EM_LOCAL_PROBE_LOCAL_SEARCH_PROFILE+x}" ]]; then
  LOCAL_SEARCH_PROFILE="${EM_LOCAL_PROBE_LOCAL_SEARCH_PROFILE}"
elif [[ "${STOP_AFTER_SCORE_ONLY}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ || "${DIAGNOSTIC_SINGLE_HALF}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  LOCAL_SEARCH_PROFILE="off"
else
  LOCAL_SEARCH_PROFILE="on"
fi

case "${LOCAL_SEARCH_PROFILE}" in
  auto|on|off) ;;
  *)
    echo "EM_LOCAL_PROBE_LOCAL_SEARCH_PROFILE must be one of auto, on, off; got ${LOCAL_SEARCH_PROFILE}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}/jobs" "${OUTPUT_ROOT}/logs"
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"

CONSTRAINT_DIRECTIVE=""
if [[ -n "${CONSTRAINT}" ]]; then
  CONSTRAINT_DIRECTIVE="#SBATCH --constraint=${CONSTRAINT}"
fi

JOB_SCRIPT="${OUTPUT_ROOT}/jobs/${PROFILE_NAME}.sbatch"
cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_local_${PROFILE_NAME:0:18}
#SBATCH --output=${OUTPUT_ROOT}/logs/${PROFILE_NAME}.out
#SBATCH --error=${OUTPUT_ROOT}/logs/${PROFILE_NAME}.err
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
${CONSTRAINT_DIRECTIVE}
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}

set -euo pipefail

cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL
export PYTHONNOUSERSITE=1
export PIXI_FROZEN=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="\${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export TMPDIR="${OUTPUT_ROOT}/tmp/${PROFILE_NAME}_\${SLURM_JOB_ID}"
export PIXI_HOME="${OUTPUT_ROOT}/pixi_home/${PROFILE_NAME}_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${OUTPUT_ROOT}/rattler_cache/${PROFILE_NAME}_\${SLURM_JOB_ID}"
export RECOVAR_JAX_CACHE_DIR="${JAX_CACHE_DIR}"
export RECOVAR_RELION_PROJECTOR_CACHE_DIR="${PROJECTOR_CACHE_DIR}"
export RECOVAR_INITIAL_NOISE_CACHE_DIR="${INITIAL_NOISE_CACHE_DIR}"
export JAX_COMPILATION_CACHE_DIR="\${RECOVAR_JAX_CACHE_DIR}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-1}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:-0}"
export RECOVAR_EM_LOCAL_NATIVE_BUILD_ROOT="${NATIVE_BUILD_ROOT}"
export RECOVAR_CUDA_LIB="${CUDA_LIB}"
export RECOVAR_RELION_BIND_BUILD_DIR="${RELION_BIND_BUILD_DIR}"
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION="${RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION}"
if [[ -n "${RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET}" ]]; then
  export RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET="${RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET}"
fi
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}" "\${JAX_COMPILATION_CACHE_DIR}" "\${RECOVAR_RELION_PROJECTOR_CACHE_DIR}" "\${RECOVAR_EM_LOCAL_NATIVE_BUILD_ROOT}" "\$(dirname "\${RECOVAR_CUDA_LIB}")" "\${RECOVAR_RELION_BIND_BUILD_DIR}"
if [[ -n "\${RECOVAR_INITIAL_NOISE_CACHE_DIR}" ]]; then
  mkdir -p "\${RECOVAR_INITIAL_NOISE_CACHE_DIR}"
  touch "\${RECOVAR_INITIAL_NOISE_CACHE_DIR}/SAFE_TO_DELETE" || true
fi
touch "\${JAX_COMPILATION_CACHE_DIR}/SAFE_TO_DELETE" || true
touch "\${RECOVAR_RELION_PROJECTOR_CACHE_DIR}/SAFE_TO_DELETE" || true
touch "\${RECOVAR_EM_LOCAL_NATIVE_BUILD_ROOT}/SAFE_TO_DELETE" || true

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
  module load "${CUDA_MODULE}"
fi

GPU_NAME="\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || true)"
if [[ -z "\${CUDA_ARCH:-}" ]]; then
  if [[ "\${GPU_NAME}" == *H100* || "\${GPU_NAME}" == *H200* ]]; then
    export CUDA_ARCH="-gencode arch=compute_90,code=sm_90"
  elif [[ "\${GPU_NAME}" == *A100* || "\${GPU_NAME}" == *A30* ]]; then
    export CUDA_ARCH="-gencode arch=compute_80,code=sm_80"
  fi
fi

echo "Repo: ${REPO_ROOT}"
echo "HEAD: \$(git rev-parse HEAD)"
echo "Branch: \$(git symbolic-ref --short HEAD || true)"
echo "Profile: ${PROFILE_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "RECOVAR_JAX_CACHE_DIR=\${RECOVAR_JAX_CACHE_DIR}"
echo "RECOVAR_RELION_PROJECTOR_CACHE_DIR=\${RECOVAR_RELION_PROJECTOR_CACHE_DIR}"
echo "RECOVAR_INITIAL_NOISE_CACHE_DIR=\${RECOVAR_INITIAL_NOISE_CACHE_DIR:-<disabled>}"
echo "JAX_COMPILATION_CACHE_DIR=\${JAX_COMPILATION_CACHE_DIR}"
echo "JAX_ENABLE_COMPILATION_CACHE=\${JAX_ENABLE_COMPILATION_CACHE}"
echo "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=\${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS}"
echo "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=\${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES}"
echo "RECOVAR_EM_LOCAL_NATIVE_BUILD_ROOT=\${RECOVAR_EM_LOCAL_NATIVE_BUILD_ROOT}"
echo "RECOVAR_CUDA_LIB=\${RECOVAR_CUDA_LIB}"
echo "RECOVAR_RELION_BIND_BUILD_DIR=\${RECOVAR_RELION_BIND_BUILD_DIR}"
echo "EM_LOCAL_PROBE_FORCE_NATIVE_REBUILD=${FORCE_NATIVE_REBUILD}"
echo "EM_LOCAL_PROBE_FORCE_INSTALL=${FORCE_INSTALL}"
echo "Data dir: ${DATA_DIR}"
echo "Seed NPZ: ${SEED_NPZ}"
echo "Pose iter: ${POSE_ITER}"
echo "Init noise from seed NPZ: ${INIT_NOISE_FROM_SEED_NPZ}"
echo "Init noise iter: ${INIT_NOISE_ITER}"
echo "Local search profile mode: ${LOCAL_SEARCH_PROFILE}"
echo "Stop after profile: ${STOP_AFTER_PROFILE}"
echo "Stop after local search: ${STOP_AFTER_LOCAL_SEARCH}"
echo "Stop after local search score-only: ${STOP_AFTER_SCORE_ONLY}"
echo "Diagnostic single half: ${DIAGNOSTIC_SINGLE_HALF}"
echo "Skip large outputs: ${SKIP_LARGE_OUTPUTS}"
echo "Save intermediates skip unregularized: ${SAVE_INTERMEDIATES_SKIP_UNREGULARIZED}"
echo "RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS=\${RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS:-<unset>}"
echo "RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB=\${RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB:-<unset>}"
echo "RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST=${RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST:-<unset>}"
export RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST="${RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST:-}"
echo "RECOVAR_EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP=\${RECOVAR_EXACT_LOCAL_BIG_JIT_DEFER_PACKED_MSTEP:-<unset>}"
echo "RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS=\${RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS:-<unset>}"
echo "RECOVAR_EXACT_LOCAL_SPARSE_ADJOINT_TARGET_ROWS=\${RECOVAR_EXACT_LOCAL_SPARSE_ADJOINT_TARGET_ROWS:-<unset>}"
echo "RECOVAR_EXACT_LOCAL_BUCKET_QUANTUM=\${RECOVAR_EXACT_LOCAL_BUCKET_QUANTUM:-<unset>}"
echo "RECOVAR_LOCAL_BUCKET_QUANTUM=\${RECOVAR_LOCAL_BUCKET_QUANTUM:-<unset>}"
echo "RECOVAR_LOCAL_XHALF_BATCH_GUARD=\${RECOVAR_LOCAL_XHALF_BATCH_GUARD:-<unset>}"
echo "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=\${RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT:-<unset>}"
echo "RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY=\${RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY:-<unset>}"
echo "RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT=\${RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT:-<unset>}"
echo "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB=\${RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB:-<unset>}"
echo "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS=\${RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GROUPS:-<unset>}"
echo "RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS=\${RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_TARGET_ROW_PIXELS:-<unset>}"
echo "RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=\${RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION:-<unset>}"
echo "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET=\${RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET:-<unset>}"
echo "CUDA_ARCH=\${CUDA_ARCH:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

INSTALL_LOCK="${REPO_ROOT}/.pixi/install-recovar.lock"
mkdir -p "$(dirname "\${INSTALL_LOCK}")"
PIXI_PY="\$(pixi run --frozen which python)"
NEED_INSTALL=1
if ! [[ "${FORCE_INSTALL}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  if "\${PIXI_PY}" - <<'PY' >/dev/null 2>&1
import pathlib
import recovar

repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
PY
  then
    NEED_INSTALL=0
  fi
fi
if [[ "\${NEED_INSTALL}" == "1" ]]; then
  (
    flock 9
    pixi run --frozen install-recovar
  ) 9>"\${INSTALL_LOCK}"
else
  echo "Reusing editable RECOVAR install bound to this checkout"
fi
PIXI_ENV_ROOT="\$(cd "\$(dirname "\${PIXI_PY}")/.." && pwd)"
PIXI_NVIDIA_ROOT="\${PIXI_ENV_ROOT}/lib/python3.11/site-packages/nvidia"
if [[ -d "\${PIXI_NVIDIA_ROOT}" ]]; then
  PIXI_NVIDIA_LIB_DIRS="\$(find "\${PIXI_NVIDIA_ROOT}" -type d -name lib 2>/dev/null | paste -sd: -)"
else
  PIXI_NVIDIA_LIB_DIRS=""
fi
CUDA_TARGET_LIB_DIR=""
if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN_DIR="\$(dirname "\$(command -v nvcc)")"
  if [[ -d "\${CUDA_BIN_DIR}/../targets/x86_64-linux/lib" ]]; then
    CUDA_TARGET_LIB_DIR="\$(cd "\${CUDA_BIN_DIR}/../targets/x86_64-linux/lib" && pwd)"
  elif [[ -d "\${CUDA_BIN_DIR}/../lib64" ]]; then
    CUDA_TARGET_LIB_DIR="\$(cd "\${CUDA_BIN_DIR}/../lib64" && pwd)"
  fi
fi
if [[ -n "\${PIXI_NVIDIA_LIB_DIRS}" || -n "\${CUDA_TARGET_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="\${PIXI_NVIDIA_LIB_DIRS:+\${PIXI_NVIDIA_LIB_DIRS}:}\${CUDA_TARGET_LIB_DIR:+\${CUDA_TARGET_LIB_DIR}:}\${LD_LIBRARY_PATH:-}"
fi
echo "PIXI_ENV_ROOT=\${PIXI_ENV_ROOT}"
echo "PIXI_NVIDIA_ROOT=\${PIXI_NVIDIA_ROOT}"
echo "CUDA_TARGET_LIB_DIR=\${CUDA_TARGET_LIB_DIR:-<unset>}"
NATIVE_BUILD_LOCK="\${RECOVAR_EM_LOCAL_NATIVE_BUILD_ROOT}/build.lock"
(
  flock 8
  RELION_BIND_SO="\$(find "\${RECOVAR_RELION_BIND_BUILD_DIR}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit 2>/dev/null || true)"
  NEED_RELION_BIND_BUILD=0
  if [[ "${FORCE_NATIVE_REBUILD}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ || -z "\${RELION_BIND_SO}" ]]; then
    NEED_RELION_BIND_BUILD=1
  elif find recovar/relion_bind -maxdepth 1 -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name 'CMakeLists.txt' \) -newer "\${RELION_BIND_SO}" -print -quit | grep -q .; then
    NEED_RELION_BIND_BUILD=1
  fi
  if [[ "\${NEED_RELION_BIND_BUILD}" == "1" ]]; then
    "\${PIXI_PY}" recovar/relion_bind/build.py
  else
    echo "Reusing RELION binding: \${RELION_BIND_SO}"
  fi
  if [[ "${FORCE_NATIVE_REBUILD}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    env PYTHON="\${PIXI_PY}" make -C recovar/cuda LIB="\${RECOVAR_CUDA_LIB}" clean
  fi
  env PYTHON="\${PIXI_PY}" make -C recovar/cuda LIB="\${RECOVAR_CUDA_LIB}" all
) 8>"\${NATIVE_BUILD_LOCK}"

"\${PIXI_PY}" - <<'PY'
import pathlib
import jax
import recovar
import recovar.cuda_backproject as cb
from recovar.relion_bind import _relion_bind_core as relion_bind

repo = pathlib.Path.cwd().resolve()
print("recovar.__file__ =", pathlib.Path(recovar.__file__).resolve())
print("relion_bind.__file__ =", pathlib.Path(relion_bind.__file__).resolve())
print("jax.__file__ =", pathlib.Path(jax.__file__).resolve())
print("jax.devices() =", jax.devices())
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
assert any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in jax.devices())
assert cb.cuda_available(), cb.cuda_unavailable_error()
PY

OUT_DIR="${OUTPUT_ROOT}/runs/${PROFILE_NAME}"
mkdir -p "\${OUT_DIR}"
EXTRA_REFINEMENT_ARGS=()
if [[ "${SAVE_INTERMEDIATES}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  mkdir -p "\${OUT_DIR}/intermediates"
  EXTRA_REFINEMENT_ARGS+=(--save_intermediates_dir "\${OUT_DIR}/intermediates")
  if [[ "${SAVE_INTERMEDIATES_SKIP_UNREGULARIZED}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    EXTRA_REFINEMENT_ARGS+=(--save_intermediates_skip_unregularized)
  fi
fi
if [[ "${INIT_NOISE_FROM_SEED_NPZ}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  EXTRA_REFINEMENT_ARGS+=(--init_noise_from_npz "${SEED_NPZ}" --init_noise_iter "${INIT_NOISE_ITER}")
elif [[ -n "\${RECOVAR_INITIAL_NOISE_CACHE_DIR}" ]]; then
  EXTRA_REFINEMENT_ARGS+=(--initial_noise_cache_dir "\${RECOVAR_INITIAL_NOISE_CACHE_DIR}")
fi
EXTRA_REFINEMENT_ARGS+=(--local_search_profile "${LOCAL_SEARCH_PROFILE}")
if [[ "${STOP_AFTER_SCORE_ONLY}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search_score_only)
elif [[ "${STOP_AFTER_PROFILE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  if [[ "${LOCAL_SEARCH_PROFILE}" == "off" ]]; then
    EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search)
  else
    EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search_profile)
  fi
elif [[ "${STOP_AFTER_LOCAL_SEARCH}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  EXTRA_REFINEMENT_ARGS+=(--stop_after_local_search)
fi
if [[ "${DIAGNOSTIC_SINGLE_HALF}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  EXTRA_REFINEMENT_ARGS+=(--diagnostic_single_half)
fi
if [[ "${SKIP_LARGE_OUTPUTS}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  EXTRA_REFINEMENT_ARGS+=(--skip-large-outputs)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_REFINEMENT_ARGS+=(${EXTRA_ARGS})
fi
START_EPOCH="\$(date +%s)"
set +e
"\${PIXI_PY}" scripts/run_full_refinement.py \\
  --data_dir "${DATA_DIR}" \\
  --output "\${OUT_DIR}" \\
  --max_iter "${MAX_ITER}" \\
  --healpix_order "${HEALPIX_ORDER}" \\
  --max_healpix_order "${MAX_HEALPIX_ORDER}" \\
  --auto_local_healpix_order "${AUTO_LOCAL_HEALPIX_ORDER}" \\
  --relion_current_sizes "${RELION_CURRENT_SIZES}" \\
  --offset_range 3.0 \\
  --offset_step 1.0 \\
  --adaptive_oversampling 1 \\
  --init_resolution 30.0 \\
  --image_batch_size "${K1_IMAGE_BATCH_SIZE}" \\
  --rotation_block_size "${K1_ROTATION_BLOCK_SIZE}" \\
  --seed "${SEED}" \\
  --perturb_seed "${SEED}" \\
  --particle_diameter_ang 200 \\
  --tau2_fudge 1.0 \\
  --max_significants -1 \\
  --apply-initial-lowpass \\
  --init_previous_best_poses_npz "${SEED_NPZ}" \\
  --init_previous_best_poses_iter "${POSE_ITER}" \\
  --skip_final_iteration \\
  --benchmark_ledger_json "\${OUT_DIR}/benchmark_ledger.json" \\
  --timing_dir "\${OUT_DIR}/timing" \\
  "\${EXTRA_REFINEMENT_ARGS[@]}" \\
  2>&1 | tee "\${OUT_DIR}/run_full_refinement.log"
STATUS="\${PIPESTATUS[0]}"
set -e
END_EPOCH="\$(date +%s)"
cat > "\${OUT_DIR}/slurm_walltime.json" <<JSON
{"slurm_job_id":"\${SLURM_JOB_ID}","profile":"${PROFILE_NAME}","start_epoch":\${START_EPOCH},"end_epoch":\${END_EPOCH},"external_wall_s":\$((END_EPOCH - START_EPOCH)),"exit_status":\${STATUS}}
JSON
exit "\${STATUS}"
EOF

chmod +x "${JOB_SCRIPT}"
JOB_ID="$(sbatch --parsable "${JOB_SCRIPT}")"
echo "${JOB_ID}" > "${OUTPUT_ROOT}/jobs/${PROFILE_NAME}.jobid"
echo "Submitted ${PROFILE_NAME}: ${JOB_ID}"
echo "Output root: ${OUTPUT_ROOT}"
