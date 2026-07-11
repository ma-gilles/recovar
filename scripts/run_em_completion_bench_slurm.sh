#!/usr/bin/env bash
# EM completion benchmark Slurm launcher.
#
# This is the heavyweight EM-only evidence run required by recovar/em/AGENTS.md:
# K=1 and K=4, both 100k particles at 256x256, compared against stored RELION
# outputs for accuracy and speed. The scientific parameters follow the stored
# RELION GUI-launched jobs and accepted completion baselines; batch sizes are
# implementation/performance controls. This is intentionally separate from the
# repo-wide RECOVAR long-test suite and from the 50k EM-long pytest tier.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="em_completion_bench_${TIMESTAMP}_${RANDOM}"
SCRATCH_DIR="${EM_COMPLETION_SCRATCH_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/${RUN_ID}}"
ACCOUNT="${SBATCH_ACCOUNT:-gilles}"
PARTITION="${SBATCH_PARTITION:-cryoem}"
CONSTRAINT="${SBATCH_CONSTRAINT:-}"
# The setup job only builds/import-checks shared artifacts. Keep it off the GPU
# benchmark partition by default so timing probes do not queue behind GPU QOS.
SETUP_PARTITION="${EM_COMPLETION_SETUP_PARTITION:-cpu}"
SETUP_CONSTRAINT="${EM_COMPLETION_SETUP_CONSTRAINT:-}"
SETUP_GRES="${EM_COMPLETION_SETUP_GRES:-}"
SUMMARY_PARTITION="${EM_COMPLETION_SUMMARY_PARTITION:-cpu}"
SUMMARY_CONSTRAINT="${EM_COMPLETION_SUMMARY_CONSTRAINT:-}"
EXCLUSIVE="${EM_COMPLETION_EXCLUSIVE:-1}"
SINGLE_VISIBLE_GPU="${EM_COMPLETION_SINGLE_VISIBLE_GPU:-1}"
CUDA_MODULE="${CUDA_MODULE:-cudatoolkit/12.8}"
SBATCH_CONSTRAINT_DIRECTIVE=""
if [[ -n "${CONSTRAINT}" ]]; then
  SBATCH_CONSTRAINT_DIRECTIVE="#SBATCH --constraint=${CONSTRAINT}"
fi
SBATCH_SETUP_CONSTRAINT_DIRECTIVE=""
if [[ -n "${SETUP_CONSTRAINT}" ]]; then
  SBATCH_SETUP_CONSTRAINT_DIRECTIVE="#SBATCH --constraint=${SETUP_CONSTRAINT}"
fi
SBATCH_SUMMARY_CONSTRAINT_DIRECTIVE=""
if [[ -n "${SUMMARY_CONSTRAINT}" ]]; then
  SBATCH_SUMMARY_CONSTRAINT_DIRECTIVE="#SBATCH --constraint=${SUMMARY_CONSTRAINT}"
fi
SBATCH_SETUP_GRES_DIRECTIVE=""
if [[ -n "${SETUP_GRES}" ]]; then
  SBATCH_SETUP_GRES_DIRECTIVE="#SBATCH --gres=${SETUP_GRES}"
fi
SBATCH_EXCLUSIVE_DIRECTIVE=""
if [[ "${EXCLUSIVE}" != "0" ]]; then
  SBATCH_EXCLUSIVE_DIRECTIVE="#SBATCH --exclusive"
fi

K1_DATA_DIR="${K1_DATA_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/pdb_k1_g256_n100000_noise1_bf80_20260516}"
K1_RELION_DIR="${K1_RELION_DIR:-${K1_DATA_DIR}/relion_autorefine_k1_it015_os1}"
K4_DATA_DIR="${K4_DATA_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/ribosembly_k4_g256_n100000_completion_20260512_171123}"
K4_RELION_DIR="${K4_RELION_DIR:-${K4_DATA_DIR}/relion_class3d_k4_it015_clean9d9}"

K1_IMAGE_BATCH_SIZE="${K1_IMAGE_BATCH_SIZE:-187}"
K1_ROTATION_BLOCK_SIZE="${K1_ROTATION_BLOCK_SIZE:-8192}"
K4_IMAGE_BATCH_SIZE="${K4_IMAGE_BATCH_SIZE:-50}"
K4_ROTATION_BLOCK_SIZE="${K4_ROTATION_BLOCK_SIZE:-2000}"
K1_MAX_ITER="${K1_MAX_ITER:-17}"
K4_MAX_ITER="${K4_MAX_ITER:-15}"
K1_MEM="${K1_MEM:-500G}"
K4_MEM="${K4_MEM:-500G}"
K1_TIME_LIMIT="${K1_TIME_LIMIT:-15:00:00}"
K4_TIME_LIMIT="${K4_TIME_LIMIT:-15:00:00}"

RUN_K1=1
RUN_K4=1
RUN_FAST_TIER=0
RUN_K4_FUSED_SPARSE_PASS2=1
RECOVAR_SPARSE_KCLASS_GROUP_TIMING="${RECOVAR_SPARSE_KCLASS_GROUP_TIMING:-1}"
EM_COMPLETION_TIMING_PROBE="${EM_COMPLETION_TIMING_PROBE:-0}"
RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION="${RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION:-0.40}"
RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET="${RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET:-}"
RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE="${RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE:-1}"
RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT="${RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT:-1}"
RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS="${RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS:-}"
RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB="${RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB:-}"
RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS="${RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS:-}"
RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS="${RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS:-}"
WATCH=0
DRY_RUN=0

usage() {
  cat <<USAGE
Usage: $0 [--watch] [--dry-run] [--k1-only] [--k4-only] [--fast-tier] [--fast-tier-only] [--k4-fused-sparse-pass2] [--k4-legacy-sparse-pass2]

Environment overrides:
  EM_COMPLETION_SCRATCH_DIR  Scratch/log root (default: ${SCRATCH_DIR})
  SBATCH_ACCOUNT             Slurm account (default: ${ACCOUNT})
  SBATCH_PARTITION           Slurm partition (default: ${PARTITION})
  SBATCH_CONSTRAINT          Optional Slurm constraint, e.g. h100
  EM_COMPLETION_SETUP_PARTITION
                             Optional setup/build job partition (default: ${SETUP_PARTITION})
  EM_COMPLETION_SETUP_CONSTRAINT
                             Optional setup/build job constraint (default: none; benchmark jobs use SBATCH_CONSTRAINT)
  EM_COMPLETION_SETUP_GRES   Optional Slurm gres for setup/build job (default: none)
  EM_COMPLETION_SUMMARY_PARTITION
                             Optional CPU summary job partition (default: ${SUMMARY_PARTITION})
  EM_COMPLETION_SUMMARY_CONSTRAINT
                             Optional summary job constraint (default: none)
  EM_COMPLETION_EXCLUSIVE    Use exclusive GPU nodes for benchmark jobs (default: 1)
  EM_COMPLETION_SINGLE_VISIBLE_GPU
                             Expose only the first allocated GPU to CUDA/JAX for single-GPU timing (default: 1)
  CUDA_MODULE                Module loaded for nvcc (default: ${CUDA_MODULE})
  K1_DATA_DIR                K=1 fixture directory
  K1_RELION_DIR              K=1 RELION output directory
  K4_DATA_DIR                K=4 fixture directory
  K4_RELION_DIR              K=4 RELION output directory
  K1_IMAGE_BATCH_SIZE        K=1 image batch size (default: ${K1_IMAGE_BATCH_SIZE})
  K1_ROTATION_BLOCK_SIZE     K=1 rotation block size (default: ${K1_ROTATION_BLOCK_SIZE})
  K1_MAX_ITER                K=1 max iteration cap (default: ${K1_MAX_ITER}; high enough for stored RELION final pass)
  K1_MEM                     K=1 Slurm memory request (default: ${K1_MEM})
  K1_TIME_LIMIT              K=1 Slurm time limit (default: ${K1_TIME_LIMIT})
  K4_IMAGE_BATCH_SIZE        K=4 image batch size (default: ${K4_IMAGE_BATCH_SIZE})
  K4_ROTATION_BLOCK_SIZE     K=4 rotation block size (default: ${K4_ROTATION_BLOCK_SIZE})
  K4_MAX_ITER                K=4 max iteration cap (default: ${K4_MAX_ITER}; matches stored RELION Class3D fixture)
  K4_MEM                     K=4 Slurm memory request (default: ${K4_MEM})
  K4_TIME_LIMIT              K=4 Slurm time limit (default: ${K4_TIME_LIMIT})
  RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES
                             Optional sparse pass-2 tile cap for speed experiments
  RECOVAR_SPARSE_PASS2_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES
                             Optional fused K-class small-bucket tile cap for hybrid speed experiments
  RECOVAR_SPARSE_PASS2_SMALL_BUCKET_THRESHOLD
                             Optional fused K-class bucket-size threshold for the small-bucket cap
  RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES
                             Optional sparse pass-2 hypothesis cap for speed experiments
  RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS
                             Optional fused K-class projection-block cap for speed experiments
  RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES
                             Optional sparse pass-2 projection cache cap
  RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES
                             Optional sparse pass-2 projection gather cap
  RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES
                             Optional sparse pass-2 noise-stat block cap
  RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES
                             Optional sparse pass-2 adjoint block cap
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES
                             Optional high-tail bucket coalescing image cap; set 0 to disable
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION
                             Optional high-tail bucket coalescing padded-row inflation cap
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
                             Optional minimum bucket size for high-tail coalescing
  RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS
                             Optional fused K-class compact-bucket experiment flag
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS
                             Optional fused K-class compact-pair execution experiment flag
  RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS
                             Optional compact-pair noise-sum reuse flag; set 0 for A/B checks
  RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS
                             Optional compact-pair weighted image-sum fusion flag; set 0 for A/B checks
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP
                             Optional compact-pair M-step mode; set pair_sparse for opt-in sparse pair reductions
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH
                             Optional compact-pair chunk image cap for K=4 speed experiments
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES
                             Optional compact-pair high-tail coalescing image cap; set 0 to disable
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION
                             Optional compact-pair high-tail coalescing padded-row inflation cap
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE
                             Optional compact-pair minimum bucket size for high-tail coalescing
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE
                             Optional compact-pair hybrid threshold; lower pair buckets stay rectangular
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT
                             Optional compact-pair hybrid threshold report list, e.g. "8192,16384,65536"; set 0 to suppress
  RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS
                             Optional compact-pair active-row backprojection experiment flag
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS
                             Optional compact-pair diagnostic planner flag; set 0 for speed A/B runs
  RECOVAR_SPARSE_KCLASS_GROUP_TIMING
                             Coarse fused K-class bucket-group timing diagnostics (default: ${RECOVAR_SPARSE_KCLASS_GROUP_TIMING})
  RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP
                             Optional fused K-class tile sizing against active score/recon windows
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS
                             Optional rectangular high-bucket active-row M-step/noise pruning flag
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE
                             Optional minimum rectangular bucket size for active-row pruning
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL
                             Optional compute active rectangular M-step/noise rows before dense matmul
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO
                             Optional grouped/dense row ratio guard for rectangular prematmul
  RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE
                             Optional active-row gather padding multiple for stable JIT shapes
  RECOVAR_K_CLASS_DENSE_PASS2_SUPPORT_FRACTION
                             Optional K-class dense-pass2 fallback median-support threshold
  RECOVAR_K_CLASS_DENSE_PASS2_MEAN_SUPPORT_FRACTION
                             Optional K-class dense-pass2 fallback mean-support threshold
  RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_IMAGES
                             Optional small-dataset image-count threshold for dense-pass2 fallback
  RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_MEAN_SUPPORT_FRACTION
                             Optional small-dataset mean-support threshold for dense-pass2 fallback
  RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION
                             Optional dense EM projection/pose-pixel budget fraction for K=1 pass-1 A/B tests
  RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET
                             Optional firstiter_cc reconstruction-tile complex-value cap for 80GB debug speed probes
  RECOVAR_PASS1_FUSED        Optional pass-1 fused significance validation flag
  RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE
                             Replay the last numbered RELION state for final all-data scoring (default: ${RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE})
  RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT
                             Full parent support for adaptive local pass-2; default 1 prioritizes quality parity (current: ${RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT})
  RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS
                             Optional exact-local bucket row-pixel cap for speed probes
  RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB
                             Optional exact-local M-step row-output cap in GB for speed probes
  RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS
                             Exact-local bucket progress interval in completed chunks
  RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS
                             Exact-local bucket progress interval in seconds
  TF_GPU_ALLOCATOR           Optional JAX allocator setting, e.g. cuda_malloc_async
  EM_COMPLETION_TIMING_PROBE Set to 1 for shortened timing probes so the summary
                             records partial timing without requiring final FSC-AUC products
USAGE
}

for arg in "$@"; do
  case "${arg}" in
    --watch) WATCH=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --k1-only) RUN_K1=1; RUN_K4=0 ;;
    --k4-only) RUN_K1=0; RUN_K4=1 ;;
    --fast-tier) RUN_FAST_TIER=1 ;;
    --fast-tier-only) RUN_FAST_TIER=1; RUN_K1=0; RUN_K4=0 ;;
    --k4-fused-sparse-pass2) RUN_K4_FUSED_SPARSE_PASS2=1 ;;
    --k4-legacy-sparse-pass2) RUN_K4_FUSED_SPARSE_PASS2=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: ${arg}" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "${SCRATCH_DIR}/tmp" "${SCRATCH_DIR}/jobs"
touch "${SCRATCH_DIR}/SAFE_TO_DELETE"
CUDA_LIB="${SCRATCH_DIR}/cuda/libcuda_backproject.so"
INSTALL_LOCK="${REPO_ROOT}/.pixi/install-recovar.lock"

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "Required file is missing or empty: ${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Required directory is missing: ${path}" >&2
    exit 2
  fi
}

write_optional_env_exports() {
  local env_name
  local env_value
  for env_name in "$@"; do
    env_value="${!env_name-}"
    if [[ -n "${env_value}" ]]; then
      printf 'export %s=%q\n' "${env_name}" "${env_value}"
    fi
  done
}

require_dir "${REPO_ROOT}/recovar/em"
require_dir "${K1_DATA_DIR}"
require_dir "${K1_RELION_DIR}"
require_file "${K1_DATA_DIR}/particles.star"
require_file "${K1_DATA_DIR}/reference_gt.mrc"
require_file "${K1_RELION_DIR}/run_it000_data.star"
require_file "${K1_RELION_DIR}/run_it000_half1_model.star"
require_file "${K1_RELION_DIR}/run_it000_half2_model.star"
require_file "${K1_RELION_DIR}/run_it000_optimiser.star"
require_file "${K1_RELION_DIR}/run_it015_half1_class001.mrc"
require_file "${K1_RELION_DIR}/run_it015_half2_class001.mrc"
if (( K1_MAX_ITER >= 17 )); then
  require_file "${K1_RELION_DIR}/run_it016_data.star"
  require_file "${K1_RELION_DIR}/run_it016_half1_model.star"
  require_file "${K1_RELION_DIR}/run_it016_half2_model.star"
  require_file "${K1_RELION_DIR}/run_it016_optimiser.star"
  require_file "${K1_RELION_DIR}/run_sampling.star"
  require_file "${K1_RELION_DIR}/run_optimiser.star"
  require_file "${K1_RELION_DIR}/run_it016_half1_class001.mrc"
  require_file "${K1_RELION_DIR}/run_it016_half2_class001.mrc"
fi

require_dir "${K4_DATA_DIR}"
require_dir "${K4_RELION_DIR}"
require_file "${K4_DATA_DIR}/particles.star"
for class_idx in 001 002 003 004; do
  require_file "${K4_DATA_DIR}/reference_gt_class${class_idx}.mrc"
  require_file "${K4_DATA_DIR}/reference_init_class${class_idx}.mrc"
  require_file "${K4_RELION_DIR}/run_it015_class${class_idx}.mrc"
done
require_file "${K4_RELION_DIR}/run_it000_model.star"
require_file "${K4_RELION_DIR}/run_it001_optimiser.star"
require_file "${K4_RELION_DIR}/run_it015_optimiser.star"

# Guard against accidentally benchmarking tuned parameter variants. These are
# the stored RELION GUI-default command shapes accepted in the EM completion
# ledger.
grep -q -- "--auto_refine" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--firstiter_cc" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--split_random_halves" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--ini_high 30" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--healpix_order 3" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--offset_range 3.0" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--offset_step 1.0" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--oversampling 1" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -Eq "_rlnTau2FudgeArg[[:space:]]+-1(\\.0*)?" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -Eq "_rlnTau2FudgeFactor[[:space:]]+1\\.0*" "${K1_RELION_DIR}/run_it000_half1_model.star"
grep -Eq "_rlnDoSplitRandomHalves[[:space:]]+1" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -Eq "_rlnRandomSeed[[:space:]]+1775735620" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -Eq "_rlnParticleDiameter[[:space:]]+200\\.0*" "${K1_RELION_DIR}/run_it000_optimiser.star"
grep -q "_rlnRandomSubset" "${K1_RELION_DIR}/run_it000_data.star"
grep -q -- "--K 4" "${K4_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--tau2_fudge 4" "${K4_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--healpix_order 1" "${K4_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--offset_range 6" "${K4_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--offset_step 2" "${K4_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--oversampling 1" "${K4_RELION_DIR}/run_it000_optimiser.star"
grep -q -- "--dont_combine_weights_via_disc" "${K4_RELION_DIR}/run_it000_optimiser.star"
if grep -q -- "--firstiter_cc" "${K4_RELION_DIR}/run_it000_optimiser.star"; then
  echo "K=4 completion RELION fixture unexpectedly used --firstiter_cc; RECOVAR K=4 command must be updated before benchmarking." >&2
  exit 2
fi
if grep -q -- "--ini_high" "${K4_RELION_DIR}/run_it000_optimiser.star"; then
  echo "K=4 completion RELION fixture unexpectedly used --ini_high; RECOVAR K=4 command must be updated before benchmarking." >&2
  exit 2
fi
grep -Eq "_rlnTau2FudgeFactor[[:space:]]+4\\.0*" "${K4_RELION_DIR}/run_it000_model.star"
grep -Eq "_rlnDoSplitRandomHalves[[:space:]]+0" "${K4_RELION_DIR}/run_it000_optimiser.star"
grep -Eq "_rlnRandomSeed[[:space:]]+1778628798" "${K4_RELION_DIR}/run_it000_optimiser.star"
grep -Eq "_rlnParticleDiameter[[:space:]]+380\\.0*" "${K4_RELION_DIR}/run_it000_optimiser.star"

write_job_preamble() {
  local job_name="$1"
  cat <<EOF
set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL
# Submit shells often run CPU-only local tests. GPU Slurm jobs must not inherit
# those overrides or the CUDA provenance gate will correctly fail.
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PIXI_FROZEN=true
export TMPDIR="${SCRATCH_DIR}/tmp/${job_name}_\${SLURM_JOB_ID}"
export PIXI_HOME="${SCRATCH_DIR}/pixi_home/${job_name}_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${SCRATCH_DIR}/rattler_cache/${job_name}_\${SLURM_JOB_ID}"
export RECOVAR_JAX_CACHE_DIR="${SCRATCH_DIR}/jax_cache"
export JAX_COMPILATION_CACHE_DIR="\${RECOVAR_JAX_CACHE_DIR}"
# The benchmark TMPDIR is GPFS scratch, not node-local storage. Leave particle
# stack staging disabled unless the submitter explicitly points RECOVAR_CACHE_DIR
# at a fast local path such as /dev/shm or node-local NVMe.
export RECOVAR_CACHE_DIR="\${RECOVAR_CACHE_DIR-}"
export RECOVAR_CUDA_LIB="${CUDA_LIB}"
export RECOVAR_CUDA_CACHE_DIR="${SCRATCH_DIR}/cuda_cache/${job_name}_\${SLURM_JOB_ID}"
export RECOVAR_RELION_BIND_BUILD_DIR="${SCRATCH_DIR}/relion_bind_build/shared"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}" "\${RECOVAR_JAX_CACHE_DIR}" "\${RECOVAR_CUDA_CACHE_DIR}" "\${RECOVAR_RELION_BIND_BUILD_DIR}" "\$(dirname "\${RECOVAR_CUDA_LIB}")"
mkdir -p "${REPO_ROOT}/.pixi"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if ! module load "${CUDA_MODULE}"; then
  echo "WARNING: failed to load CUDA module ${CUDA_MODULE}; falling back to CUDA_HOME if available" >&2
fi
CUDA_HOME="\${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_HOME
if [[ -d "\${CUDA_HOME}/bin" ]]; then
  export PATH="\${CUDA_HOME}/bin:\${PATH}"
fi
CUDA_TARGET_LIB_DIR="\${CUDA_HOME}/targets/x86_64-linux/lib"
PIXI_NVIDIA_ROOT="${REPO_ROOT}/.pixi/envs/default/lib/python3.11/site-packages/nvidia"
if [[ -d "\${PIXI_NVIDIA_ROOT}" ]]; then
  PIXI_NVIDIA_LIB_DIRS="\$(find "\${PIXI_NVIDIA_ROOT}" -type d -name lib 2>/dev/null | paste -sd: -)"
else
  PIXI_NVIDIA_LIB_DIRS=""
fi
if [[ -n "\${PIXI_NVIDIA_LIB_DIRS}" ]]; then
  export LD_LIBRARY_PATH="\${PIXI_NVIDIA_LIB_DIRS}:\${CUDA_TARGET_LIB_DIR}:\${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="\${CUDA_TARGET_LIB_DIR}:\${LD_LIBRARY_PATH:-}"
fi

if [[ "${SINGLE_VISIBLE_GPU}" != "0" ]]; then
  SLURM_VISIBLE_GPUS="\${SLURM_STEP_GPUS:-\${SLURM_JOB_GPUS:-}}"
  CUDA_FIRST_GPU="\${CUDA_VISIBLE_DEVICES:-}"
  CUDA_FIRST_GPU="\${CUDA_FIRST_GPU%%,*}"
  if [[ -z "\${CUDA_FIRST_GPU}" ]]; then
    CUDA_FIRST_GPU="\${SLURM_VISIBLE_GPUS%%,*}"
  fi
  if [[ -n "\${CUDA_FIRST_GPU}" ]]; then
    export CUDA_VISIBLE_DEVICES="\${CUDA_FIRST_GPU}"
  fi
elif [[ -z "\${CUDA_VISIBLE_DEVICES:-}" ]]; then
  SLURM_VISIBLE_GPUS="\${SLURM_STEP_GPUS:-\${SLURM_JOB_GPUS:-}}"
  CUDA_FIRST_GPU="\${SLURM_VISIBLE_GPUS%%,*}"
  if [[ -n "\${CUDA_FIRST_GPU}" ]]; then
    export CUDA_VISIBLE_DEVICES="\${CUDA_FIRST_GPU}"
  fi
fi

echo "=== ${job_name} ==="
echo "Repo: ${REPO_ROOT}"
echo "HEAD: \$(git rev-parse HEAD)"
echo "Branch: \$(git symbolic-ref --short HEAD || echo '<detached>')"
echo "Dirty status:"
git status --short
echo "Slurm job: \${SLURM_JOB_ID}"
echo "Host: \$(hostname)"
echo "SLURM_JOB_GPUS=\${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=\${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
echo "TMPDIR=\${TMPDIR}"
echo "RECOVAR_CUDA_LIB=\${RECOVAR_CUDA_LIB}"
echo "RECOVAR_RELION_BIND_BUILD_DIR=\${RECOVAR_RELION_BIND_BUILD_DIR}"
echo "CUDA_HOME=\${CUDA_HOME}"
echo "LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}"
$(write_optional_env_exports \
  RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION \
  RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET \
  RECOVAR_PASS1_FUSED \
  RECOVAR_DISABLE_LOCAL_BIG_JIT \
  RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES \
  RECOVAR_SPARSE_PASS2_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES \
  RECOVAR_SPARSE_PASS2_SMALL_BUCKET_THRESHOLD \
  RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES \
  RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES \
  RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS \
  RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES \
  RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES \
  RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES \
  RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES \
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES \
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION \
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS \
  RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS \
  RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT \
  RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP \
  RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS \
  RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS \
  RECOVAR_SPARSE_KCLASS_GROUP_TIMING \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO \
  RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE \
  RECOVAR_K_CLASS_DENSE_PASS2_SUPPORT_FRACTION \
  RECOVAR_K_CLASS_DENSE_PASS2_MEAN_SUPPORT_FRACTION \
  RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_IMAGES \
  RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_MEAN_SUPPORT_FRACTION \
  RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE \
  RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT \
  RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS \
  RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB \
  RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS \
  RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS \
  TF_GPU_ALLOCATOR)
for env_name in \
  RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION \
  RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET \
  RECOVAR_PASS1_FUSED \
  RECOVAR_SPARSE_KCLASS_FUSED \
  RECOVAR_DISABLE_LOCAL_BIG_JIT \
  RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES \
  RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES \
  RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS \
  RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS \
  RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT \
  RECOVAR_SPARSE_KCLASS_GROUP_TIMING \
  RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO \
  RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE \
  RECOVAR_K_CLASS_DENSE_PASS2_SUPPORT_FRACTION \
  RECOVAR_K_CLASS_DENSE_PASS2_MEAN_SUPPORT_FRACTION \
  RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_IMAGES \
  RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_MEAN_SUPPORT_FRACTION \
  RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE \
  RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT \
  RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS \
  RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB \
  RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS \
  RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS; do
  env_value="\${!env_name-}"
  if [[ -n "\${env_value}" ]]; then
    echo "\${env_name}=\${env_value}"
  fi
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo
EOF
}

write_refresh_pixi_cuda_libs() {
  cat <<EOF
PIXI_NVIDIA_ROOT="${REPO_ROOT}/.pixi/envs/default/lib/python3.11/site-packages/nvidia"
if [[ -d "\${PIXI_NVIDIA_ROOT}" ]]; then
  PIXI_NVIDIA_LIB_DIRS="\$(find "\${PIXI_NVIDIA_ROOT}" -type d -name lib 2>/dev/null | paste -sd: -)"
else
  PIXI_NVIDIA_LIB_DIRS=""
fi
if [[ -n "\${PIXI_NVIDIA_LIB_DIRS}" ]]; then
  export LD_LIBRARY_PATH="\${PIXI_NVIDIA_LIB_DIRS}:\${CUDA_TARGET_LIB_DIR}:\${LD_LIBRARY_PATH:-}"
fi
EOF
}

write_build_cuda_lib() {
  cat <<EOF
mkdir -p "\$(dirname "\${RECOVAR_CUDA_LIB}")"
flock "${SCRATCH_DIR}/cuda/build.lock" \\
  env PYTHON="\${PIXI_PY}" make -C recovar/cuda LIB="\${RECOVAR_CUDA_LIB}" all
EOF
}

write_setup_script() {
  local script_path="${SCRATCH_DIR}/jobs/em_completion_setup.sh"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_completion_setup
#SBATCH --output=${SCRATCH_DIR}/em_completion_setup.out
#SBATCH --error=${SCRATCH_DIR}/em_completion_setup.err
#SBATCH --partition=${SETUP_PARTITION}
#SBATCH --account=${ACCOUNT}
${SBATCH_SETUP_CONSTRAINT_DIRECTIVE}
${SBATCH_SETUP_GRES_DIRECTIVE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

$(write_job_preamble "em_completion_setup")

flock "${INSTALL_LOCK}" bash -lc '
set -euo pipefail
rm -rf "\${RECOVAR_RELION_BIND_BUILD_DIR:?}"
mkdir -p "\${RECOVAR_RELION_BIND_BUILD_DIR}"
pixi run --frozen install-recovar
PIXI_PY="\$(pixi run --frozen which python)"
if ! "\${PIXI_PY}" -c "import pybind11" >/dev/null 2>&1; then
  echo "ERROR: pybind11 is missing from the pixi environment; run pixi install before submitting EM jobs." >&2
  exit 1
fi
pixi run --frozen python recovar/relion_bind/build.py
'
PIXI_PY="\$(pixi run --frozen which python)"
$(write_refresh_pixi_cuda_libs)

"\${PIXI_PY}" - <<'PY'
import os
import pathlib

import jax
import recovar
from recovar.relion_bind import _relion_bind_core as relion_bind

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
relion_bind_file = pathlib.Path(relion_bind.__file__).resolve()
jax_file = pathlib.Path(jax.__file__).resolve()
external_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")
external_bind_root = pathlib.Path(external_bind_dir).resolve() if external_bind_dir else None
print("recovar.__file__ =", recovar_file)
print("relion_bind.__file__ =", relion_bind_file)
print("jax.__file__ =", jax_file)
print("jax.devices() =", jax.devices())
assert str(recovar_file).startswith(str(repo) + "/"), recovar_file
assert str(relion_bind_file).startswith(str(repo) + "/") or (
    external_bind_root is not None
    and str(relion_bind_file).startswith(str(external_bind_root) + "/")
), relion_bind_file
assert ".pixi/envs/default/" in str(jax_file), jax_file
print("setup artifact gate ok")
PY
EOF
  chmod +x "${script_path}"
  printf '%s\n' "${script_path}"
}

write_fast_tier_script() {
  local script_path="${SCRATCH_DIR}/jobs/em_completion_fast_tier.sh"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_completion_fast_tier
#SBATCH --output=${SCRATCH_DIR}/em_completion_fast_tier.out
#SBATCH --error=${SCRATCH_DIR}/em_completion_fast_tier.err
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
${SBATCH_CONSTRAINT_DIRECTIVE}
#SBATCH --gres=gpu:1
${SBATCH_EXCLUSIVE_DIRECTIVE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=01:00:00

$(write_job_preamble "em_completion_fast_tier")

flock "${INSTALL_LOCK}" bash -lc 'pixi run --frozen install-recovar'
PIXI_PY="\$(pixi run --frozen which python)"
$(write_refresh_pixi_cuda_libs)
$(write_build_cuda_lib)
"\${PIXI_PY}" - <<'PY'
import os
import pathlib
import jax
import recovar
import recovar.cuda_backproject as cb
from recovar.relion_bind import _relion_bind_core as relion_bind
repo = pathlib.Path.cwd().resolve()
relion_bind_file = pathlib.Path(relion_bind.__file__).resolve()
external_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")
external_bind_root = pathlib.Path(external_bind_dir).resolve() if external_bind_dir else None
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert str(relion_bind_file).startswith(str(repo) + "/") or (
    external_bind_root is not None
    and str(relion_bind_file).startswith(str(external_bind_root) + "/")
)
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
print("jax.devices() =", jax.devices())
assert any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in jax.devices())
assert cb.cuda_available(), cb.cuda_unavailable_error()
print("provenance/cuda gate ok")
PY

pixi run --frozen test-em-parity-fast
EOF
  chmod +x "${script_path}"
  printf '%s\n' "${script_path}"
}

write_k1_script() {
  local output_dir="${SCRATCH_DIR}/k1_100k256_recovar"
  local script_path="${SCRATCH_DIR}/jobs/em_completion_k1_100k256.sh"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_completion_k1_100k
#SBATCH --output=${SCRATCH_DIR}/em_completion_k1_100k256.out
#SBATCH --error=${SCRATCH_DIR}/em_completion_k1_100k256.err
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
${SBATCH_CONSTRAINT_DIRECTIVE}
#SBATCH --gres=gpu:1
${SBATCH_EXCLUSIVE_DIRECTIVE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${K1_MEM}
#SBATCH --time=${K1_TIME_LIMIT}

$(write_job_preamble "em_completion_k1_100k256")

OUTPUT_DIR="${output_dir}"
mkdir -p "\${OUTPUT_DIR}"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 60 > "\${OUTPUT_DIR}/gpu_monitor.csv" &
MONITOR_PID="\$!"
trap 'kill "\${MONITOR_PID}" 2>/dev/null || true' EXIT

flock "${INSTALL_LOCK}" bash -lc 'pixi run --frozen install-recovar'
PIXI_PY="\$(pixi run --frozen which python)"
$(write_refresh_pixi_cuda_libs)
$(write_build_cuda_lib)
"\${PIXI_PY}" - <<'PY'
import os
import pathlib
import jax
import recovar
import recovar.cuda_backproject as cb
from recovar.relion_bind import _relion_bind_core as relion_bind
repo = pathlib.Path.cwd().resolve()
relion_bind_file = pathlib.Path(relion_bind.__file__).resolve()
external_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")
external_bind_root = pathlib.Path(external_bind_dir).resolve() if external_bind_dir else None
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert str(relion_bind_file).startswith(str(repo) + "/") or (
    external_bind_root is not None
    and str(relion_bind_file).startswith(str(external_bind_root) + "/")
)
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
print("jax.devices() =", jax.devices())
assert any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in jax.devices())
assert cb.cuda_available(), cb.cuda_unavailable_error()
print("provenance/cuda gate ok")
PY

START_EPOCH="\$(date +%s)"
REFINEMENT_EXTRA_ARGS=()
if [[ "${EM_COMPLETION_TIMING_PROBE}" == "1" ]]; then
  REFINEMENT_EXTRA_ARGS+=(--skip-large-outputs)
fi
set +e
"\${PIXI_PY}" scripts/run_full_refinement.py \\
  --data_dir "${K1_DATA_DIR}" \\
  --output "\${OUTPUT_DIR}" \\
  --max_iter "${K1_MAX_ITER}" \\
  --healpix_order 3 \\
  --offset_range 3.0 \\
  --offset_step 1.0 \\
  --adaptive_oversampling 1 \\
  --init_resolution 30.0 \\
  --image_batch_size "${K1_IMAGE_BATCH_SIZE}" \\
  --rotation_block_size "${K1_ROTATION_BLOCK_SIZE}" \\
  --seed 1775735620 \\
  --perturb_seed 1775735620 \\
  --relion_half_sets "${K1_RELION_DIR}/run_it000_data.star" \\
  --relion_optimiser "${K1_RELION_DIR}/run_it000_optimiser.star" \\
  --perturb_replay_relion_dir "${K1_RELION_DIR}" \\
  --particle_diameter_ang 200 \\
  --tau2_fudge 1.0 \\
  --firstiter_cc \\
  --apply-initial-lowpass \\
  --benchmark_ledger_json "\${OUTPUT_DIR}/benchmark_ledger.json" \\
  --timing_dir "\${OUTPUT_DIR}/timing" \\
  "\${REFINEMENT_EXTRA_ARGS[@]}" \\
  2>&1 | tee "\${OUTPUT_DIR}/run_full_refinement.log"
STATUS="\${PIPESTATUS[0]}"
set -e
END_EPOCH="\$(date +%s)"
cat > "\${OUTPUT_DIR}/slurm_walltime.json" <<JSON
{"slurm_job_id":"\${SLURM_JOB_ID}","start_epoch":\${START_EPOCH},"end_epoch":\${END_EPOCH},"external_wall_s":\$((END_EPOCH - START_EPOCH)),"exit_status":\${STATUS}}
JSON
exit "\${STATUS}"
EOF
  chmod +x "${script_path}"
  printf '%s\n' "${script_path}"
}

write_k4_script() {
  local output_dir="${SCRATCH_DIR}/k4_100k256_recovar"
  local script_path="${SCRATCH_DIR}/jobs/em_completion_k4_100k256.sh"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_completion_k4_100k
#SBATCH --output=${SCRATCH_DIR}/em_completion_k4_100k256.out
#SBATCH --error=${SCRATCH_DIR}/em_completion_k4_100k256.err
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
${SBATCH_CONSTRAINT_DIRECTIVE}
#SBATCH --gres=gpu:1
${SBATCH_EXCLUSIVE_DIRECTIVE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${K4_MEM}
#SBATCH --time=${K4_TIME_LIMIT}

$(write_job_preamble "em_completion_k4_100k256")

OUTPUT_DIR="${output_dir}"
mkdir -p "\${OUTPUT_DIR}"
if [[ "${RUN_K4_FUSED_SPARSE_PASS2}" -eq 1 ]]; then
  export RECOVAR_SPARSE_KCLASS_FUSED=1
else
  export RECOVAR_SPARSE_KCLASS_FUSED=0
fi
echo "RECOVAR_SPARSE_KCLASS_FUSED=\${RECOVAR_SPARSE_KCLASS_FUSED}"
for env_name in \
  RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES \
  RECOVAR_SPARSE_PASS2_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES \
  RECOVAR_SPARSE_PASS2_SMALL_BUCKET_THRESHOLD \
  RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES \
  RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES \
  RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS \
  RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES \
  RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES \
  RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES \
  RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES \
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES \
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION \
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS \
  RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS \
  RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT \
  RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS \
  RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS \
  RECOVAR_SPARSE_KCLASS_GROUP_TIMING \
  RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL \
  RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO \
  RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE \
  RECOVAR_K_CLASS_DENSE_PASS2_SUPPORT_FRACTION \
  RECOVAR_K_CLASS_DENSE_PASS2_MEAN_SUPPORT_FRACTION \
  RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_IMAGES \
  RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_MEAN_SUPPORT_FRACTION \
  RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET \
  TF_GPU_ALLOCATOR; do
  env_value="\${!env_name-}"
  if [[ -n "\${env_value}" ]]; then
    echo "\${env_name}=\${env_value}"
  fi
done
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 60 > "\${OUTPUT_DIR}/gpu_monitor.csv" &
MONITOR_PID="\$!"
trap 'kill "\${MONITOR_PID}" 2>/dev/null || true' EXIT

flock "${INSTALL_LOCK}" bash -lc 'pixi run --frozen install-recovar'
PIXI_PY="\$(pixi run --frozen which python)"
$(write_refresh_pixi_cuda_libs)
$(write_build_cuda_lib)
"\${PIXI_PY}" - <<'PY'
import os
import pathlib
import jax
import recovar
import recovar.cuda_backproject as cb
from recovar.relion_bind import _relion_bind_core as relion_bind
repo = pathlib.Path.cwd().resolve()
relion_bind_file = pathlib.Path(relion_bind.__file__).resolve()
external_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")
external_bind_root = pathlib.Path(external_bind_dir).resolve() if external_bind_dir else None
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert str(relion_bind_file).startswith(str(repo) + "/") or (
    external_bind_root is not None
    and str(relion_bind_file).startswith(str(external_bind_root) + "/")
)
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
print("jax.devices() =", jax.devices())
assert any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in jax.devices())
assert cb.cuda_available(), cb.cuda_unavailable_error()
print("provenance/cuda gate ok")
PY

START_EPOCH="\$(date +%s)"
REFINEMENT_EXTRA_ARGS=()
if [[ "${EM_COMPLETION_TIMING_PROBE}" == "1" ]]; then
  REFINEMENT_EXTRA_ARGS+=(--skip-large-outputs)
fi
set +e
"\${PIXI_PY}" scripts/run_full_refinement.py \\
  --data_dir "${K4_DATA_DIR}" \\
  --output "\${OUTPUT_DIR}" \\
  --max_iter "${K4_MAX_ITER}" \\
  --n_classes 4 \\
  --healpix_order 1 \\
  --offset_range 6 \\
  --offset_step 2 \\
  --adaptive_oversampling 1 \\
  --init_resolution 30.0 \\
  --image_batch_size "${K4_IMAGE_BATCH_SIZE}" \\
  --rotation_block_size "${K4_ROTATION_BLOCK_SIZE}" \\
  --seed 1778628798 \\
  --relion_optimiser "${K4_RELION_DIR}/run_it015_optimiser.star" \\
  --relion_init_dir "${K4_RELION_DIR}" \\
  --perturb_replay_relion_dir "${K4_RELION_DIR}" \\
  --particle_diameter_ang 380 \\
  --tau2_fudge 4.0 \\
  --benchmark_ledger_json "\${OUTPUT_DIR}/benchmark_ledger.json" \\
  --timing_dir "\${OUTPUT_DIR}/timing" \\
  "\${REFINEMENT_EXTRA_ARGS[@]}" \\
  2>&1 | tee "\${OUTPUT_DIR}/run_full_refinement.log"
STATUS="\${PIPESTATUS[0]}"
set -e
END_EPOCH="\$(date +%s)"
cat > "\${OUTPUT_DIR}/slurm_walltime.json" <<JSON
{"slurm_job_id":"\${SLURM_JOB_ID}","start_epoch":\${START_EPOCH},"end_epoch":\${END_EPOCH},"external_wall_s":\$((END_EPOCH - START_EPOCH)),"exit_status":\${STATUS}}
JSON
exit "\${STATUS}"
EOF
  chmod +x "${script_path}"
  printf '%s\n' "${script_path}"
}

write_summary_script() {
  local dependency="$1"
  local tracked_jobs="$2"
  local script_path="${SCRATCH_DIR}/jobs/em_completion_summary.sh"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_completion_summary
#SBATCH --output=${SCRATCH_DIR}/em_completion_summary.out
#SBATCH --error=${SCRATCH_DIR}/em_completion_summary.err
#SBATCH --partition=${SUMMARY_PARTITION}
#SBATCH --account=${ACCOUNT}
${SBATCH_SUMMARY_CONSTRAINT_DIRECTIVE}
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --dependency=${dependency}

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export RECOVAR_DISABLE_CUDA=1
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export PIXI_FROZEN=true
export TMPDIR="${SCRATCH_DIR}/tmp/em_completion_summary_\${SLURM_JOB_ID}"
export PIXI_HOME="${SCRATCH_DIR}/pixi_home/em_completion_summary_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${SCRATCH_DIR}/rattler_cache/em_completion_summary_\${SLURM_JOB_ID}"
export RECOVAR_JAX_CACHE_DIR="${SCRATCH_DIR}/jax_cache"
export JAX_COMPILATION_CACHE_DIR="\${RECOVAR_JAX_CACHE_DIR}"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}" "\${RECOVAR_JAX_CACHE_DIR}"

echo "=== EM completion benchmark summary ==="
echo "Repo: ${REPO_ROOT}"
echo "HEAD: \$(git rev-parse HEAD)"
echo "Branch: \$(git symbolic-ref --short HEAD || echo '<detached>')"
echo "Scratch: ${SCRATCH_DIR}"
echo "Timing probe: ${EM_COMPLETION_TIMING_PROBE}"
echo

for job_id in ${tracked_jobs}; do
  if [[ -n "\${job_id}" ]]; then
    sacct -j "\${job_id}" -X -o JobID,JobName%30,State,Elapsed,MaxRSS,ReqMem,AllocTRES || true
  fi
done
echo

for log_name in em_completion_setup em_completion_fast_tier em_completion_k1_100k256 em_completion_k4_100k256; do
  if [[ -s "${SCRATCH_DIR}/\${log_name}.out" ]]; then
    echo "--- \${log_name}.out tail ---"
    tail -80 "${SCRATCH_DIR}/\${log_name}.out"
    echo
  fi
  if [[ -s "${SCRATCH_DIR}/\${log_name}.err" ]]; then
    echo "--- \${log_name}.err tail ---"
    tail -40 "${SCRATCH_DIR}/\${log_name}.err"
    echo
  fi
done

if [[ -f scripts/summarize_em_completion_bench.py ]]; then
  REQUIRE_CASE_ARGS=()
  if [[ "${EM_COMPLETION_TIMING_PROBE}" != "1" && "${RUN_K1}" -eq 1 ]]; then
    REQUIRE_CASE_ARGS+=(--require-k1)
  fi
  if [[ "${EM_COMPLETION_TIMING_PROBE}" != "1" && "${RUN_K4}" -eq 1 ]]; then
    REQUIRE_CASE_ARGS+=(--require-k4)
  fi
  pixi run --frozen python scripts/summarize_em_completion_bench.py \\
    --k1-recovar-dir "${SCRATCH_DIR}/k1_100k256_recovar" \\
    --k1-relion-dir "${K1_RELION_DIR}" \\
    --k1-fixture-dir "${K1_DATA_DIR}" \\
    --k4-recovar-dir "${SCRATCH_DIR}/k4_100k256_recovar" \\
    --k4-relion-dir "${K4_RELION_DIR}" \\
    --k4-fixture-dir "${K4_DATA_DIR}" \\
    --output-json "${SCRATCH_DIR}/summary_metrics.json" \\
    --output-markdown "${SCRATCH_DIR}/summary.md" \\
    "\${REQUIRE_CASE_ARGS[@]}"
  echo
  echo "Summary JSON: ${SCRATCH_DIR}/summary_metrics.json"
  echo "Summary Markdown: ${SCRATCH_DIR}/summary.md"
  [[ -s "${SCRATCH_DIR}/summary.md" ]] && cat "${SCRATCH_DIR}/summary.md"
else
  echo "scripts/summarize_em_completion_bench.py is not present yet; skipping metric aggregation." >&2
fi
EOF
  chmod +x "${script_path}"
  printf '%s\n' "${script_path}"
}

submit_or_print() {
  local script_path="$1"
  shift
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY-RUN sbatch $* ${script_path}" >&2
    basename "${script_path}"
  else
    # Slurm treats SBATCH_* environment variables as implicit command-line
    # options.  The launcher uses SBATCH_CONSTRAINT/SBATCH_GRES as user-facing
    # inputs for generated GPU job directives, so strip them here; otherwise
    # CPU-only setup/summary jobs inherit the GPU constraint too.
    env -u SBATCH_ACCOUNT -u SBATCH_PARTITION -u SBATCH_CONSTRAINT \
      -u SBATCH_GRES -u SBATCH_GPUS -u SBATCH_GPUS_PER_NODE \
      sbatch --parsable "$@" "${script_path}"
  fi
}

echo "EM completion benchmark launcher"
echo "Repo: ${REPO_ROOT}"
echo "HEAD: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Branch: $(git -C "${REPO_ROOT}" symbolic-ref --short HEAD || echo '<detached>')"
echo "Scratch: ${SCRATCH_DIR}"
echo "Partition/account: ${PARTITION}/${ACCOUNT}"
echo "Constraint: ${CONSTRAINT:-<none>}"
echo "Setup partition: ${SETUP_PARTITION}"
echo "Setup constraint: ${SETUP_CONSTRAINT:-<none>}"
echo "Setup gres: ${SETUP_GRES:-<none>}"
echo "Summary partition: ${SUMMARY_PARTITION}"
echo "Summary constraint: ${SUMMARY_CONSTRAINT:-<none>}"
echo "Exclusive GPU jobs: ${EXCLUSIVE}"
echo "CUDA module: ${CUDA_MODULE}"
echo "K=1 fixture: ${K1_DATA_DIR}"
echo "K=1 RELION:  ${K1_RELION_DIR}"
echo "K=1 max iter: ${K1_MAX_ITER}"
echo "K=1 Slurm mem/time: ${K1_MEM}/${K1_TIME_LIMIT}"
echo "K=4 fixture: ${K4_DATA_DIR}"
echo "K=4 RELION:  ${K4_RELION_DIR}"
echo "K=4 max iter: ${K4_MAX_ITER}"
echo "K=4 Slurm mem/time: ${K4_MEM}/${K4_TIME_LIMIT}"
echo "K=4 fused sparse pass2: ${RUN_K4_FUSED_SPARSE_PASS2}"
echo "Timing probe: ${EM_COMPLETION_TIMING_PROBE}"
echo

SETUP_SCRIPT="$(write_setup_script)"
SETUP_JOB_ID="$(submit_or_print "${SETUP_SCRIPT}")"
echo "Setup job: ${SETUP_JOB_ID}"

DEPENDENCY_JOBS=("${SETUP_JOB_ID}")

if [[ "${RUN_FAST_TIER}" -eq 1 ]]; then
  FAST_SCRIPT="$(write_fast_tier_script)"
  FAST_TIER_JOB_ID="$(submit_or_print "${FAST_SCRIPT}" --dependency=afterok:"${SETUP_JOB_ID}")"
  export FAST_TIER_JOB_ID
  DEPENDENCY_JOBS+=("${FAST_TIER_JOB_ID}")
  echo "Fast EM parity tier job: ${FAST_TIER_JOB_ID}"
fi

if [[ "${RUN_K1}" -eq 1 ]]; then
  K1_SCRIPT="$(write_k1_script)"
  K1_JOB_ID="$(submit_or_print "${K1_SCRIPT}" --dependency=afterok:"${SETUP_JOB_ID}")"
  export K1_JOB_ID
  DEPENDENCY_JOBS+=("${K1_JOB_ID}")
  echo "K=1 100k/256 completion job: ${K1_JOB_ID}"
fi

if [[ "${RUN_K4}" -eq 1 ]]; then
  K4_SCRIPT="$(write_k4_script)"
  K4_JOB_ID="$(submit_or_print "${K4_SCRIPT}" --dependency=afterok:"${SETUP_JOB_ID}")"
  export K4_JOB_ID
  DEPENDENCY_JOBS+=("${K4_JOB_ID}")
  echo "K=4 100k/256 completion job: ${K4_JOB_ID}"
fi

SUMMARY_DEPENDENCY="afterany:$(IFS=:; echo "${DEPENDENCY_JOBS[*]}")"
TRACKED_JOB_IDS="$(IFS=' '; echo "${DEPENDENCY_JOBS[*]}")"
SUMMARY_SCRIPT="$(write_summary_script "${SUMMARY_DEPENDENCY}" "${TRACKED_JOB_IDS}")"
SUMMARY_JOB_ID="$(submit_or_print "${SUMMARY_SCRIPT}")"
echo "Summary job: ${SUMMARY_JOB_ID}"
echo
echo "Logs and outputs: ${SCRATCH_DIR}"

cat > "${SCRATCH_DIR}/submission.env" <<EOF
REPO_ROOT=${REPO_ROOT}
HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD)
BRANCH=$(git -C "${REPO_ROOT}" symbolic-ref --short HEAD || echo '<detached>')
SCRATCH_DIR=${SCRATCH_DIR}
SBATCH_PARTITION=${PARTITION}
SBATCH_ACCOUNT=${ACCOUNT}
SBATCH_CONSTRAINT=${CONSTRAINT}
EM_COMPLETION_SETUP_PARTITION=${SETUP_PARTITION}
EM_COMPLETION_SETUP_CONSTRAINT=${SETUP_CONSTRAINT}
EM_COMPLETION_SETUP_GRES=${SETUP_GRES}
EM_COMPLETION_SUMMARY_PARTITION=${SUMMARY_PARTITION}
EM_COMPLETION_SUMMARY_CONSTRAINT=${SUMMARY_CONSTRAINT}
EM_COMPLETION_EXCLUSIVE=${EXCLUSIVE}
EM_COMPLETION_SINGLE_VISIBLE_GPU=${SINGLE_VISIBLE_GPU}
SETUP_JOB_ID=${SETUP_JOB_ID}
FAST_TIER_JOB_ID=${FAST_TIER_JOB_ID:-}
K1_JOB_ID=${K1_JOB_ID:-}
K4_JOB_ID=${K4_JOB_ID:-}
SUMMARY_JOB_ID=${SUMMARY_JOB_ID}
K1_DATA_DIR=${K1_DATA_DIR}
K1_RELION_DIR=${K1_RELION_DIR}
K1_IMAGE_BATCH_SIZE=${K1_IMAGE_BATCH_SIZE}
K1_ROTATION_BLOCK_SIZE=${K1_ROTATION_BLOCK_SIZE}
K1_MAX_ITER=${K1_MAX_ITER}
K1_MEM=${K1_MEM}
K1_TIME_LIMIT=${K1_TIME_LIMIT}
K4_DATA_DIR=${K4_DATA_DIR}
K4_RELION_DIR=${K4_RELION_DIR}
K4_IMAGE_BATCH_SIZE=${K4_IMAGE_BATCH_SIZE}
K4_ROTATION_BLOCK_SIZE=${K4_ROTATION_BLOCK_SIZE}
K4_MAX_ITER=${K4_MAX_ITER}
K4_MEM=${K4_MEM}
K4_TIME_LIMIT=${K4_TIME_LIMIT}
CUDA_LIB=${CUDA_LIB}
RUN_K4_FUSED_SPARSE_PASS2=${RUN_K4_FUSED_SPARSE_PASS2}
EM_COMPLETION_TIMING_PROBE=${EM_COMPLETION_TIMING_PROBE}
RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES=${RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES:-}
RECOVAR_SPARSE_PASS2_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES=${RECOVAR_SPARSE_PASS2_SMALL_BUCKET_MAX_TRANSLATION_TILE_BYTES:-}
RECOVAR_SPARSE_PASS2_SMALL_BUCKET_THRESHOLD=${RECOVAR_SPARSE_PASS2_SMALL_BUCKET_THRESHOLD:-}
RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES=${RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES:-}
RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES=${RECOVAR_SPARSE_PASS2_SCORE_ONLY_MAX_HYPOTHESES:-}
RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS=${RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS:-}
RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES=${RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES:-}
RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES=${RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES:-}
RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES=${RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES:-}
RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES=${RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES:-}
RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES=${RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES:-}
RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION=${RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION:-}
RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE=${RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE:-}
RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS=${RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS:-}
RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS=${RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS:-}
RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS=${RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS:-}
RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS=${RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS:-}
RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP=${RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP:-}
RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH=${RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH:-}
RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES=${RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES:-}
RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION=${RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION:-}
RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE=${RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE:-}
RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE=${RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE:-}
RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT=${RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_THRESHOLD_REPORT:-}
RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS=${RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS:-}
RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS=${RECOVAR_SPARSE_KCLASS_COMPACT_BUCKETS:-}
RECOVAR_SPARSE_KCLASS_GROUP_TIMING=${RECOVAR_SPARSE_KCLASS_GROUP_TIMING:-}
RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP=${RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP:-}
RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS=${RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS:-}
RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE=${RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS_MIN_BUCKET_SIZE:-}
RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL=${RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL:-}
RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO=${RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO:-}
RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE=${RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE:-}
RECOVAR_K_CLASS_DENSE_PASS2_SUPPORT_FRACTION=${RECOVAR_K_CLASS_DENSE_PASS2_SUPPORT_FRACTION:-}
RECOVAR_K_CLASS_DENSE_PASS2_MEAN_SUPPORT_FRACTION=${RECOVAR_K_CLASS_DENSE_PASS2_MEAN_SUPPORT_FRACTION:-}
RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_IMAGES=${RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_IMAGES:-}
RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_MEAN_SUPPORT_FRACTION=${RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_MEAN_SUPPORT_FRACTION:-}
RECOVAR_SPARSE_KCLASS_FUSED=${RECOVAR_SPARSE_KCLASS_FUSED:-}
RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=${RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION:-}
RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET=${RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET:-}
RECOVAR_PASS1_FUSED=${RECOVAR_PASS1_FUSED:-}
RECOVAR_DISABLE_LOCAL_BIG_JIT=${RECOVAR_DISABLE_LOCAL_BIG_JIT:-}
RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=${RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE:-}
RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=${RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT:-}
RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS=${RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS:-}
RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB=${RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB:-}
RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS=${RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS:-}
RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS=${RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS:-}
TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-}
EOF

if [[ "${WATCH}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
  echo "Watching jobs. Press Ctrl-C to stop watching; jobs remain queued/running."
  while true; do
    date
    squeue -j "$(IFS=,; echo "${DEPENDENCY_JOBS[*]},${SUMMARY_JOB_ID}")" || true
    if [[ -z "$(squeue -h -j "${SUMMARY_JOB_ID}" 2>/dev/null || true)" ]]; then
      break
    fi
    sleep 60
  done
  echo "--- summary tail ---"
  tail -120 "${SCRATCH_DIR}/em_completion_summary.out" 2>/dev/null || true
fi
