#!/usr/bin/env bash
# High-resolution K=1 EM robustness matrix launcher.
#
# Generates synthetic single-class PDB datasets, runs RECOVAR's full EM
# refinement with GUI-like K=1 AutoRefine defaults, and summarizes FSC/AUC
# against the ground-truth volume. RELION baselines are intentionally optional:
# set EM_K1_MATRIX_RUN_RELION=1 for selected cases when head-to-head speed
# comparison is needed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="em_k1_robustness_${TIMESTAMP}_${RANDOM}"
SCRATCH_DIR="${EM_K1_MATRIX_SCRATCH_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/${RUN_ID}}"
RUNTIME_ROOT="${EM_K1_MATRIX_RUNTIME_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}}"
ACCOUNT="${SBATCH_ACCOUNT:-gilles}"
PARTITION="${SBATCH_PARTITION:-cryoem}"
SUMMARY_PARTITION="${EM_K1_MATRIX_SUMMARY_PARTITION:-cpu}"
CONSTRAINT="${SBATCH_CONSTRAINT:-}"
SETUP_PARTITION="${EM_K1_MATRIX_SETUP_PARTITION:-cpu}"
SETUP_CONSTRAINT="${EM_K1_MATRIX_SETUP_CONSTRAINT:-}"
SETUP_GRES="${EM_K1_MATRIX_SETUP_GRES:-}"
SUMMARY_CONSTRAINT="${EM_K1_MATRIX_SUMMARY_CONSTRAINT:-}"
SUMMARY_GRES="${EM_K1_MATRIX_SUMMARY_GRES:-}"
CUDA_MODULE="${CUDA_MODULE:-cudatoolkit/12.8}"
RELION_MODULE="${RELION_MODULE:-relion/5.0.1/gcc-11.5.0-gpu}"
RELION_REFINE_MPI="${RELION_REFINE_MPI:-relion_refine_mpi}"
RELION_EXTRA_LD_LIBRARY_PATH="${RELION_EXTRA_LD_LIBRARY_PATH:-}"
RELION_SRC_DIR="${RELION_SRC_DIR:-}"
EXCLUSIVE="${EM_K1_MATRIX_EXCLUSIVE:-0}"
SINGLE_VISIBLE_GPU="${EM_K1_MATRIX_SINGLE_VISIBLE_GPU:-1}"
RUN_RELION="${EM_K1_MATRIX_RUN_RELION:-0}"
RELION_MPI_RANKS="${RELION_MPI_RANKS:-3}"
RELION_POOL="${EM_K1_MATRIX_RELION_POOL:-3}"
NOCTF_RELION_USE_CTF="${EM_K1_NOCTF_RELION_USE_CTF:-1}"
MAX_ITER="${EM_K1_MATRIX_MAX_ITER:-999}"
TIME_LIMIT_OVERRIDE="${EM_K1_MATRIX_TIME_LIMIT:-}"
K1_IMAGE_BATCH_SIZE="${K1_IMAGE_BATCH_SIZE:-187}"
K1_ROTATION_BLOCK_SIZE="${K1_ROTATION_BLOCK_SIZE:-8192}"
STREAMING_CHUNK_SIZE="${EM_K1_MATRIX_STREAMING_CHUNK_SIZE:-1000}"
NOISE_RNG_BATCH_SIZE="${EM_K1_MATRIX_NOISE_RNG_BATCH_SIZE:-}"
RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION="${RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION:-0.40}"
RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET="${RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET:-}"
RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE="${RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE:-1}"
RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE="${RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE:-}"
RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE="${RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE:-}"
RECOVAR_FINAL_ALL_DATA_GRID_CORRECT="${RECOVAR_FINAL_ALL_DATA_GRID_CORRECT:-}"
RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT="${RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT:-0}"
RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY="${RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY:-}"
RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT="${RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT:-}"
RECOVAR_BPREF_ACCUM_DUMP_DIR="${RECOVAR_BPREF_ACCUM_DUMP_DIR:-}"
RECOVAR_PASS2_DUMP_DIR="${RECOVAR_PASS2_DUMP_DIR:-}"
RECOVAR_PASS2_DUMP_ORIGINAL_INDICES="${RECOVAR_PASS2_DUMP_ORIGINAL_INDICES:-}"
RECOVAR_PASS2_DUMP_CURRENT_SIZE="${RECOVAR_PASS2_DUMP_CURRENT_SIZE:-}"
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR="${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR:-}"
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES="${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES:-}"
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE="${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE:-}"
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION="${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION:-}"
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL="${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL:-}"
RECOVAR_LOCAL_SCORE_DUMP_DIR="${RECOVAR_LOCAL_SCORE_DUMP_DIR:-}"
RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES="${RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES:-}"
RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE="${RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE:-}"
RECOVAR_LOCAL_SCORE_DUMP_ITERATION="${RECOVAR_LOCAL_SCORE_DUMP_ITERATION:-}"
RECOVAR_LOCAL_SCORE_DUMP_LABEL="${RECOVAR_LOCAL_SCORE_DUMP_LABEL:-}"
RECOVAR_LOCAL_SCORE_DUMP_OPERANDS="${RECOVAR_LOCAL_SCORE_DUMP_OPERANDS:-}"
RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS="${RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS:-}"
RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB="${RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB:-}"
RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST="${RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST:-}"
RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS="${RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS:-}"
RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS="${RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS:-}"
RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS="${RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS:-}"
RECOVAR_RELION_PROJECTOR_DUMP_DIR="${RECOVAR_RELION_PROJECTOR_DUMP_DIR:-}"
RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR="${RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR:-}"
RECOVAR_MSTEP_DUMP_DIR="${RECOVAR_MSTEP_DUMP_DIR:-}"
RECOVAR_MSTEP_DUMP_MAX_CALLS="${RECOVAR_MSTEP_DUMP_MAX_CALLS:-}"
RECOVAR_MSTEP_DUMP_RAW="${RECOVAR_MSTEP_DUMP_RAW:-}"
RECOVAR_SAVE_INTERMEDIATES_DIR="${RECOVAR_SAVE_INTERMEDIATES_DIR:-}"
RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED="${RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED:-1}"
RELION_DUMP_DIR="${RELION_DUMP_DIR:-}"
RELION_DUMP_STACK_INDEX="${RELION_DUMP_STACK_INDEX:-}"
RELION_DUMP_PART_ID="${RELION_DUMP_PART_ID:-}"
RELION_DUMP_PART="${RELION_DUMP_PART:-}"
RELION_DUMP_ITER="${RELION_DUMP_ITER:-}"
WATCH=0
DRY_RUN=0
SELECTED_CASES=()

SBATCH_CONSTRAINT_DIRECTIVE=""
if [[ -n "${CONSTRAINT}" ]]; then
  SBATCH_CONSTRAINT_DIRECTIVE="#SBATCH --constraint=${CONSTRAINT}"
fi
SBATCH_SETUP_GRES_DIRECTIVE=""
if [[ -n "${SETUP_GRES}" ]]; then
  SBATCH_SETUP_GRES_DIRECTIVE="#SBATCH --gres=${SETUP_GRES}"
fi
SBATCH_SETUP_CONSTRAINT_DIRECTIVE=""
if [[ -n "${SETUP_CONSTRAINT}" ]]; then
  SBATCH_SETUP_CONSTRAINT_DIRECTIVE="#SBATCH --constraint=${SETUP_CONSTRAINT}"
fi
SBATCH_EXCLUSIVE_DIRECTIVE=""
if [[ "${EXCLUSIVE}" != "0" ]]; then
  SBATCH_EXCLUSIVE_DIRECTIVE="#SBATCH --exclusive"
fi
SBATCH_SUMMARY_CONSTRAINT_DIRECTIVE=""
if [[ -n "${SUMMARY_CONSTRAINT}" ]]; then
  SBATCH_SUMMARY_CONSTRAINT_DIRECTIVE="#SBATCH --constraint=${SUMMARY_CONSTRAINT}"
fi
SBATCH_SUMMARY_GRES_DIRECTIVE=""
if [[ -n "${SUMMARY_GRES}" ]]; then
  SBATCH_SUMMARY_GRES_DIRECTIVE="#SBATCH --gres=${SUMMARY_GRES}"
fi

usage() {
  cat <<USAGE
Usage: $0 [--watch] [--dry-run] [--case CASE_OR_INDEX] [--with-relion] [--recovar-only]

Runs a high-resolution K=1 end-to-end robustness matrix. By default all
configured cases are submitted. Limit cases with repeated --case or with
EM_K1_MATRIX_CASES, e.g.:

  EM_K1_MATRIX_CASES=1,4,anisotropic_high_noise $0

Environment overrides:
  EM_K1_MATRIX_SCRATCH_DIR          Scratch/log root (default: ${SCRATCH_DIR})
  EM_K1_MATRIX_RUNTIME_ROOT         Runtime tmp/pixi/rattler root (default: ${RUNTIME_ROOT})
  EM_K1_MATRIX_CASES                Comma-separated case names or 1-based indices
  EM_K1_MATRIX_RUN_RELION           Run RELION AutoRefine too (default: ${RUN_RELION})
  EM_K1_MATRIX_RELION_POOL          RELION --pool for strict-parity RELION baselines (default: ${RELION_POOL})
  EM_K1_MATRIX_MAX_ITER             RECOVAR max iterations and RELION --auto_iter_max (default: ${MAX_ITER})
  EM_K1_MATRIX_TIME_LIMIT           Override Slurm time limit for selected case jobs (default: per-case matrix value)
  EM_K1_MATRIX_STREAMING_CHUNK_SIZE Streaming MRC write chunk size (default: ${STREAMING_CHUNK_SIZE})
  EM_K1_MATRIX_NOISE_RNG_BATCH_SIZE Fixed simulator noise RNG batch size (default: generator batch)
  SBATCH_ACCOUNT                    Slurm account (default: ${ACCOUNT})
  SBATCH_PARTITION                  Slurm partition (default: ${PARTITION})
  EM_K1_MATRIX_SUMMARY_PARTITION    Slurm partition for summaries (default: ${SUMMARY_PARTITION})
  EM_K1_MATRIX_SUMMARY_CONSTRAINT   Optional Slurm constraint for summaries (default: ${SUMMARY_CONSTRAINT:-<none>})
  EM_K1_MATRIX_SUMMARY_GRES         Optional Slurm gres for summaries (default: ${SUMMARY_GRES:-<none>})
  EM_K1_MATRIX_SETUP_PARTITION      Optional Slurm partition for setup/build job (default: ${SETUP_PARTITION})
  EM_K1_MATRIX_SETUP_CONSTRAINT     Optional Slurm constraint for setup/build job (default: ${SETUP_CONSTRAINT:-<none>})
  EM_K1_MATRIX_SETUP_GRES           Optional Slurm gres for setup/build job (default: ${SETUP_GRES:-<none>})
  SBATCH_CONSTRAINT                 Optional Slurm constraint, e.g. h100
  EM_K1_MATRIX_EXCLUSIVE            Exclusive GPU nodes for case jobs (default: ${EXCLUSIVE}; set 1 for strict speed runs)
  EM_K1_MATRIX_SINGLE_VISIBLE_GPU   Restrict each case job to one visible GPU (default: ${SINGLE_VISIBLE_GPU})
  CUDA_MODULE                       CUDA module for nvcc (default: ${CUDA_MODULE})
  RELION_MODULE                     RELION module for optional baselines (default: ${RELION_MODULE})
  RELION_REFINE_MPI                 RELION executable for optional baselines (default: ${RELION_REFINE_MPI})
  RELION_EXTRA_LD_LIBRARY_PATH      Extra LD_LIBRARY_PATH prefix for RELION_REFINE_MPI
                                    (default: ${RELION_EXTRA_LD_LIBRARY_PATH:-<unset>})
  RELION_SRC_DIR                    RELION src directory used to build the RECOVAR
                                    parity binding (must contain projector.h)
  EM_K1_NOCTF_RELION_USE_CTF        For simulator no-CTF cases, run RELION --ctf using a sanitized
                                    RELION-only constant-CTF STAR (default: ${NOCTF_RELION_USE_CTF}).
                                    Set to 0 only for diagnostics; RELION's no-CTF path can reconstruct
                                    the simulator no-CTF convention with opposite map sign.
  K1_IMAGE_BATCH_SIZE               RECOVAR image batch size (default: ${K1_IMAGE_BATCH_SIZE})
  K1_ROTATION_BLOCK_SIZE            RECOVAR rotation block size (default: ${K1_ROTATION_BLOCK_SIZE})
  RECOVAR_LOCAL_BUCKET_QUANTUM      Override large-support bucket quantum for sparse pass-2 tails
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES
                                    Optional high-tail bucket coalescing image cap; set 0 to disable
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION
                                    Optional high-tail bucket coalescing padded-row inflation cap
  RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
                                    Optional minimum bucket size for high-tail coalescing
  RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES
                                    Override sparse pass-2 translation tile memory cap
  RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES
                                    Override sparse pass-2 projection/reconstruction gather cap
  RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES
                                    Override sparse pass-2 noise block memory cap
  RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES
                                    Override sparse pass-2 adjoint block memory cap
  RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES
                                    Override sparse pass-2 projection cache cap
  RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS
                                    Override sparse pass-2 rotations per projection call
  RECOVAR_K1_DENSE_PASS2            Diagnostic only: disable sparse adaptive K=1 pass-2
  RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING
                                    Diagnostic only: evaluate the full K=1 adaptive fine grid
                                    after pass-1; default unset keeps RELION significance support.
  RECOVAR_K1_RELION_X_HALF_MSTEP    Diagnostic only: set 0 to use the old native K=1 half-volume M-step
                                    instead of the default RELION x-half BPref-layout M-step.
  RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE
                                    Replay the last numbered RELION state for final all-data scoring (default: ${RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE})
  RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE
                                    Diagnostic only: use merged K=1 reference for final all-data scoring (default: ${RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE:-<unset>})
  RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE
                                    Diagnostic only: disable automatic final last-numbered RELION replay (default: ${RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE:-<unset>})
  RECOVAR_FINAL_ALL_DATA_GRID_CORRECT
                                    Set 1 to enable RELION-style final output gridding correction.
                                    Default is unset/off, matching the current GUI-quality path.
  RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT
                                    Diagnostic full-parent expansion for adaptive local pass-2; default 0 matches RELION's pruned parent support (current: ${RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT})
  RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY
                                    Diagnostic only: keep pass-1 significant parent rotations but expand all parent translations.
  RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT
                                    Diagnostic only: use broader local pass-2 support for normalization
                                    denominator while keeping the reconstruction support pruned.
                                    Accepted values: rotation_only, full_parent.
  RECOVAR_BPREF_ACCUM_DUMP_DIR      Diagnostic only: base directory for per-case RECOVAR per-iteration
                                    BPref accumulator dumps (default: ${RECOVAR_BPREF_ACCUM_DUMP_DIR:-<unset>})
  RECOVAR_PASS2_DUMP_DIR            Diagnostic only: base directory for per-case RECOVAR pass-2 operand dumps
                                    (default: ${RECOVAR_PASS2_DUMP_DIR:-<unset>})
  RECOVAR_PASS2_DUMP_ORIGINAL_INDICES
                                    Comma-separated original particle indices for RECOVAR_PASS2_DUMP_DIR
                                    (default: ${RECOVAR_PASS2_DUMP_ORIGINAL_INDICES:-<unset>})
  RECOVAR_PASS2_DUMP_CURRENT_SIZE   Current size filter for RECOVAR_PASS2_DUMP_DIR
                                    (default: ${RECOVAR_PASS2_DUMP_CURRENT_SIZE:-<unset>})
  RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR
                                    Diagnostic only: base directory for per-case production fused-path
                                    local posterior dumps (default: ${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR:-<unset>})
  RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES
                                    Comma-separated original particle indices for fused-path posterior dumps
                                    (default: ${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES:-<unset>})
  RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE
                                    Current size filter for fused-path posterior dumps
                                    (default: ${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE:-<unset>})
  RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION
                                    Iteration filter for fused-path posterior dumps
                                    (default: ${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION:-<unset>})
  RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL
                                    Optional filename label for fused-path posterior dumps
                                    (default: ${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL:-<unset>})
  RECOVAR_LOCAL_SCORE_DUMP_DIR      Diagnostic only: base directory for materialized local score dumps.
                                    This forces the non-fused local path for targeted particles
                                    (default: ${RECOVAR_LOCAL_SCORE_DUMP_DIR:-<unset>})
  RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES
                                    Comma-separated original particle indices for local score dumps
                                    (default: ${RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES:-<unset>})
  RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE
                                    Current size filter for local score dumps; use -1 for final full-size
                                    current_size=None (default: ${RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE:-<unset>})
  RECOVAR_LOCAL_SCORE_DUMP_ITERATION
                                    Iteration filter for local score dumps (default: ${RECOVAR_LOCAL_SCORE_DUMP_ITERATION:-<unset>})
  RECOVAR_LOCAL_SCORE_DUMP_LABEL    Optional filename label for local score dumps
                                    (default: ${RECOVAR_LOCAL_SCORE_DUMP_LABEL:-<unset>})
  RECOVAR_LOCAL_SCORE_DUMP_OPERANDS Set 1 to include large score/reconstruction operands in local score dumps
                                    (default: ${RECOVAR_LOCAL_SCORE_DUMP_OPERANDS:-<unset>})
  RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS
                                    Exact-local bucket row-pixel cap for speed probes.
  RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB
                                    Exact-local M-step row-output cap in GB for speed probes.
  RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST
                                    Multiplier for exact-local automatic microbatch boosting when
                                    row caps are not explicitly overridden.
  RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS
                                    Memory guard for exact local packed-noise projection rows.
                                    Lower this for OOM retries (default: RECOVAR internal default).
  RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS
                                    Exact-local bucket progress interval in completed chunks.
  RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS
                                    Exact-local bucket progress interval in seconds.
  RECOVAR_RELION_PROJECTOR_DUMP_DIR Diagnostic only: base directory for per-case RELION Projector::data
                                    slabs built by RECOVAR for scoring (default: ${RECOVAR_RELION_PROJECTOR_DUMP_DIR:-<unset>})
  RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR
                                    Diagnostic only: base directory for per-case final all-data BPref accumulator dumps
                                    (default: ${RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR:-<unset>})
  RECOVAR_MSTEP_DUMP_DIR            Diagnostic only: base directory for per-case RELION BPref/updateSSNR dumps
                                    from the patched RELION build (default: ${RECOVAR_MSTEP_DUMP_DIR:-<unset>})
  RECOVAR_MSTEP_DUMP_MAX_CALLS      Max RELION BPref dump calls per process (default: RELION patch default)
  RECOVAR_MSTEP_DUMP_RAW            Set 1 to include raw RELION BPref data/weight binary dumps
  RECOVAR_SAVE_INTERMEDIATES_DIR    Diagnostic only: pass --save_intermediates_dir to RECOVAR.
                                    Set to auto/1 for <recovar>/intermediates or to a base path for
                                    per-case subdirectories (default: ${RECOVAR_SAVE_INTERMEDIATES_DIR:-<unset>})
  RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED
                                    When intermediates are enabled, skip diagnostic unregularized
                                    maps by default to keep FSC debugging fast. Set to 0 to save
                                    the unregularized maps too (default: ${RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED})
  RELION_DUMP_DIR                   Diagnostic only: base directory for patched RELION E-step dumps
                                    (default: ${RELION_DUMP_DIR:-<unset>})
  RELION_DUMP_STACK_INDEX           Patched RELION 1-based stack index to dump
                                    (default: ${RELION_DUMP_STACK_INDEX:-<unset>})
  RELION_DUMP_PART_ID               Patched RELION original part_id to dump
                                    (default: ${RELION_DUMP_PART_ID:-<unset>})
  RELION_DUMP_PART                  Patched RELION sorted particle index to dump
                                    (default: ${RELION_DUMP_PART:-<unset>})
  RELION_DUMP_ITER                  Patched RELION iteration to dump; unset dumps all target hits
                                    (default: ${RELION_DUMP_ITER:-<unset>})
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --watch) WATCH=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --with-relion) RUN_RELION=1; shift ;;
    --recovar-only) RUN_RELION=0; shift ;;
    --case)
      if [[ "$#" -lt 2 ]]; then
        echo "--case requires an argument" >&2
        exit 2
      fi
      SELECTED_CASES+=("$2")
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "${EM_K1_MATRIX_CASES:-}" ]]; then
  IFS=',' read -r -a ENV_CASES <<< "${EM_K1_MATRIX_CASES}"
  SELECTED_CASES+=("${ENV_CASES[@]}")
fi

if [[ -z "${RELION_SRC_DIR}" || ! -f "${RELION_SRC_DIR}/projector.h" ]]; then
  echo "RELION_SRC_DIR must name a RELION src directory containing projector.h" >&2
  exit 2
fi

# Fields:
# index|name|n_images|grid|noise_level|noise_model|dataset_params_option|seed|pdb_bfactor|noise_scale_std|contrast_std|volume_radius|relion_bg_radius_px|time_limit|mem|streaming_chunk|streaming_mmap|percent_outliers|put_extra_particles|image_offset_n_std
CASES=(
  "1|baseline_100k_g256_white_noise1_bf80|100000|256|1.0|white|uniform|1701|80.0|0.0|0.0|0.7|-|15:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "2|more_images_200k_g256_white_noise1_bf80|200000|256|1.0|white|uniform|1702|80.0|0.0|0.0|0.7|-|24:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "3|more_images_300k_g256_white_noise1_bf80|300000|256|1.0|white|uniform|1703|80.0|0.0|0.0|0.7|-|24:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "4|high_noise_100k_g256_white_noise3_bf80|100000|256|3.0|white|uniform|1704|80.0|0.0|0.0|0.7|-|18:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "5|very_high_noise_100k_g256_white_noise10_bf80|100000|256|10.0|white|uniform|1705|80.0|0.0|0.0|0.7|-|18:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "6|noctf_control_100k_g256_white_noise3_bf80|100000|256|3.0|white|noctf|1706|80.0|0.0|0.0|0.7|-|18:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "7|anisotropic_100k_g256_white_noise1_bf80|100000|256|1.0|white|nonuniform|1707|80.0|0.0|0.0|0.7|-|18:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "8|anisotropic_high_noise_100k_g256_white_noise3_bf80|100000|256|3.0|white|nonuniform|1708|80.0|0.0|0.0|0.7|-|18:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "9|high_res_near_nyquist_100k_g384_white_noise1_bf0|100000|384|1.0|white|uniform|1709|0.0|0.0|0.0|0.7|-|24:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "10|high_res_anisotropic_100k_g384_radial_noise3_bf0|100000|384|3.0|radial1|nonuniform|1710|0.0|0.0|0.0|0.7|-|24:00:00|500G|${STREAMING_CHUNK_SIZE}|1|0.0|0|0.0"
  "11|small_baseline_3k_g128_white_noise1_bf80|3000|128|1.0|white|uniform|1711|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|0|0.0"
  "12|small_very_high_noise_3k_g128_white_noise10_bf80|3000|128|10.0|white|uniform|1712|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|0|0.0"
  "13|small_anisotropic_3k_g128_white_noise3_bf80|3000|128|3.0|white|nonuniform|1713|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|0|0.0"
  "14|small_noctf_3k_g128_white_noise3_bf80|3000|128|3.0|white|noctf|1714|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|0|0.0"
  "15|small_outliers_3k_g128_pct20_noise1_bf80|3000|128|1.0|white|uniform|1715|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.20|0|0.0"
  "16|small_anisotropic_outliers_3k_g128_pct25_noise3_bf80|3000|128|3.0|white|nonuniform|1716|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.25|0|0.0"
  "17|small_extra_particles_3k_g128_noise1_bf80|3000|128|1.0|white|uniform|1717|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|1|0.0"
  "18|small_contrast_noise_scale_3k_g128_noise1_bf80|3000|128|1.0|white|uniform|1718|80.0|0.5|0.5|0.7|-|03:00:00|128G|500|0|0.0|0|0.0"
  "19|small_image_offset_3k_g128_noise1_bf80|3000|128|1.0|white|uniform|1719|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|0|1.0"
  "20|small_high_res_radial_3k_g256_noise3_bf0|3000|256|3.0|radial1|uniform|1720|0.0|0.0|0.0|0.7|-|04:00:00|256G|500|0|0.0|0|0.0"
  "21|small_kent_angles_3k_g128_white_noise3_bf80|3000|128|3.0|white|kent|1721|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|0|0.0"
  "22|small_severe_outliers_3k_g128_radial_noise5_bf80|3000|128|5.0|radial1|nonuniform|1722|80.0|0.7|0.7|0.7|-|03:00:00|128G|500|0|0.50|0|1.5"
  "23|small_noctf_radial_3k_g128_noise3_bf80|3000|128|3.0|radial1|noctf|1723|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|0|0.0"
  "24|small_kent_outliers_3k_g128_pct20_noise3_bf80|3000|128|3.0|white|kent|1724|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.20|0|0.0"
  "25|tiny_baseline_1k_g128_white_noise3_bf80|1000|128|3.0|white|uniform|1725|80.0|0.0|0.0|0.7|-|02:00:00|96G|250|0|0.0|0|0.0"
  "26|tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80|1000|128|5.0|radial1|nonuniform|1726|80.0|0.5|0.5|0.7|-|02:00:00|96G|250|0|0.30|0|1.0"
  "27|small_extreme_outliers_3k_g128_pct70_noise1_bf80|3000|128|1.0|white|uniform|1727|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.70|0|0.0"
  "28|small_kent_extra_offset_3k_g128_noise3_bf80|3000|128|3.0|white|kent|1728|80.0|0.3|0.3|0.7|-|03:00:00|128G|500|0|0.0|1|0.5"
  "29|small_low_noise_3k_g128_white_noise0p2_bf80|3000|128|0.2|white|uniform|1729|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|0|0.0"
  "30|small_low_noise_kent_3k_g128_white_noise0p2_bf80|3000|128|0.2|white|kent|1730|80.0|0.0|0.0|0.7|-|03:00:00|128G|500|0|0.0|0|0.0"
  "31|mid_10k_g128_white_noise1_bf80|10000|128|1.0|white|uniform|1731|80.0|0.0|0.0|0.7|-|04:00:00|160G|500|0|0.0|0|0.0"
  "32|mid_10k_kent_g128_radial_noise3_bf80|10000|128|3.0|radial1|kent|1732|80.0|0.0|0.0|0.7|-|04:00:00|160G|500|0|0.0|0|0.0"
  "33|max_images_400k_g128_white_noise1_bf80|400000|128|1.0|white|uniform|1733|80.0|0.0|0.0|0.7|-|36:00:00|500G|2000|1|0.0|0|0.0"
  "34|max_images_400k_g128_radial_noise3_nonuniform_bf80|400000|128|3.0|radial1|nonuniform|1734|80.0|0.0|0.0|0.7|-|36:00:00|500G|2000|1|0.0|0|0.0"
)

mkdir -p "${SCRATCH_DIR}/jobs" "${SCRATCH_DIR}/summaries" "${RUNTIME_ROOT}"
touch "${SCRATCH_DIR}/SAFE_TO_DELETE" "${RUNTIME_ROOT}/SAFE_TO_DELETE"
CUDA_LIB="${SCRATCH_DIR}/cuda/libcuda_backproject.so"
INSTALL_LOCK="${REPO_ROOT}/.pixi/install-recovar.lock"

case_selected() {
  local idx="$1"
  local name="$2"
  local selected
  if [[ "${#SELECTED_CASES[@]}" -eq 0 ]]; then
    return 0
  fi
  for selected in "${SELECTED_CASES[@]}"; do
    selected="${selected//[[:space:]]/}"
    if [[ -z "${selected}" ]]; then
      continue
    fi
    if [[ "${selected}" == "${idx}" || "${selected}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

apply_case_overrides() {
  local row="$1"
  local idx name n_images grid noise_level noise_model dataset_params_option seed pdb_bfactor noise_scale_std contrast_std volume_radius relion_bg_radius time_limit mem streaming_chunk streaming_mmap percent_outliers put_extra_particles image_offset_n_std
  IFS='|' read -r idx name n_images grid noise_level noise_model dataset_params_option seed pdb_bfactor noise_scale_std contrast_std volume_radius relion_bg_radius time_limit mem streaming_chunk streaming_mmap percent_outliers put_extra_particles image_offset_n_std <<< "${row}"
  if [[ -n "${TIME_LIMIT_OVERRIDE}" ]]; then
    time_limit="${TIME_LIMIT_OVERRIDE}"
  fi
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${idx}" "${name}" "${n_images}" "${grid}" "${noise_level}" "${noise_model}" \
    "${dataset_params_option}" "${seed}" "${pdb_bfactor}" "${noise_scale_std}" \
    "${contrast_std}" "${volume_radius}" "${relion_bg_radius}" "${time_limit}" \
    "${mem}" "${streaming_chunk}" "${streaming_mmap}" "${percent_outliers}" \
    "${put_extra_particles}" "${image_offset_n_std}"
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

capture_git_provenance_snapshot() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  (
    cd "${REPO_ROOT}"
    git rev-parse HEAD > "${out_dir}/git_head.txt" 2>/dev/null || true
    git symbolic-ref --short HEAD > "${out_dir}/git_branch.txt" 2>/dev/null || echo "<detached>" > "${out_dir}/git_branch.txt"
    git status --porcelain=v1 > "${out_dir}/git_status_porcelain.txt" 2>/dev/null || true
    git status --short > "${out_dir}/git_status_short.txt" 2>/dev/null || true
    git diff --binary --no-ext-diff HEAD -- > "${out_dir}/git_diff.patch" 2>/dev/null || true
    sha256sum "${out_dir}/git_diff.patch" > "${out_dir}/git_diff.sha256" 2>/dev/null || true
    git ls-files --others --exclude-standard -z > "${out_dir}/git_untracked_files.zlist" 2>/dev/null || true
    : > "${out_dir}/git_untracked_file_hashes.tsv"
    if [[ -s "${out_dir}/git_untracked_files.zlist" ]]; then
      while IFS= read -r -d '' relpath; do
        if [[ -f "${relpath}" ]]; then
          sha256sum "${relpath}" >> "${out_dir}/git_untracked_file_hashes.tsv" 2>/dev/null || printf '%s\tunreadable\n' "${relpath}" >> "${out_dir}/git_untracked_file_hashes.tsv"
        else
          printf '%s\tnonfile-or-missing\n' "${relpath}" >> "${out_dir}/git_untracked_file_hashes.tsv"
        fi
      done < "${out_dir}/git_untracked_files.zlist"
      tar --null --files-from="${out_dir}/git_untracked_files.zlist" -cf "${out_dir}/git_untracked_files.tar" 2> "${out_dir}/git_untracked_files.tar.err" || true
    fi
    {
      sha256sum "${out_dir}/git_status_porcelain.txt" 2>/dev/null | awk '{print $1}' || true
      sha256sum "${out_dir}/git_diff.patch" 2>/dev/null | awk '{print $1}' || true
      sha256sum "${out_dir}/git_untracked_file_hashes.tsv" 2>/dev/null | awk '{print $1}' || true
    } > "${out_dir}/git_component_sha256.txt"
    sha256sum "${out_dir}/git_component_sha256.txt" | awk '{print $1}' > "${out_dir}/git_worktree_fingerprint.sha256"
  )
}

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
export RECOVAR_EXPECTED_REPO_ROOT="${REPO_ROOT}"
export PYTHONFAULTHANDLER="\${PYTHONFAULTHANDLER:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PIXI_FROZEN=true
export TMPDIR="${RUNTIME_ROOT}/${job_name}_\${SLURM_JOB_ID}/tmp"
export PIXI_HOME="${RUNTIME_ROOT}/${job_name}_\${SLURM_JOB_ID}/pixi_home"
export RATTLER_CACHE_DIR="${RUNTIME_ROOT}/${job_name}_\${SLURM_JOB_ID}/rattler_cache"
export RECOVAR_JAX_CACHE_DIR="${SCRATCH_DIR}/jax_cache"
export JAX_COMPILATION_CACHE_DIR="\${RECOVAR_JAX_CACHE_DIR}"
# The matrix TMPDIR is GPFS scratch, not node-local storage. Leave particle
# stack staging disabled unless the submitter explicitly points RECOVAR_CACHE_DIR
# at fast local storage.
export RECOVAR_CACHE_DIR="\${RECOVAR_CACHE_DIR-}"
export RECOVAR_CUDA_LIB="${CUDA_LIB}"
export RECOVAR_CUDA_CACHE_DIR="${SCRATCH_DIR}/cuda_cache/${job_name}_\${SLURM_JOB_ID}"
export RECOVAR_RELION_BIND_BUILD_DIR="${SCRATCH_DIR}/relion_bind_build/shared"
export RELION_SRC_DIR="${RELION_SRC_DIR}"
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
JOB_GIT_PROVENANCE_DIR="${SCRATCH_DIR}/job_provenance/${job_name}_\${SLURM_JOB_ID}"
mkdir -p "\${JOB_GIT_PROVENANCE_DIR}"
git rev-parse HEAD > "\${JOB_GIT_PROVENANCE_DIR}/git_head.txt" 2>/dev/null || true
git symbolic-ref --short HEAD > "\${JOB_GIT_PROVENANCE_DIR}/git_branch.txt" 2>/dev/null || echo "<detached>" > "\${JOB_GIT_PROVENANCE_DIR}/git_branch.txt"
git status --porcelain=v1 > "\${JOB_GIT_PROVENANCE_DIR}/git_status_porcelain.txt" 2>/dev/null || true
git status --short > "\${JOB_GIT_PROVENANCE_DIR}/git_status_short.txt" 2>/dev/null || true
git diff --binary --no-ext-diff HEAD -- > "\${JOB_GIT_PROVENANCE_DIR}/git_diff.patch" 2>/dev/null || true
sha256sum "\${JOB_GIT_PROVENANCE_DIR}/git_diff.patch" > "\${JOB_GIT_PROVENANCE_DIR}/git_diff.sha256" 2>/dev/null || true
git ls-files --others --exclude-standard -z > "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_files.zlist" 2>/dev/null || true
: > "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_file_hashes.tsv"
if [[ -s "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_files.zlist" ]]; then
  while IFS= read -r -d '' relpath; do
    if [[ -f "\${relpath}" ]]; then
      sha256sum "\${relpath}" >> "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_file_hashes.tsv" 2>/dev/null || printf '%s\tunreadable\n' "\${relpath}" >> "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_file_hashes.tsv"
    else
      printf '%s\tnonfile-or-missing\n' "\${relpath}" >> "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_file_hashes.tsv"
    fi
  done < "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_files.zlist"
  tar --null --files-from="\${JOB_GIT_PROVENANCE_DIR}/git_untracked_files.zlist" -cf "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_files.tar" 2> "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_files.tar.err" || true
fi
{
  sha256sum "\${JOB_GIT_PROVENANCE_DIR}/git_status_porcelain.txt" 2>/dev/null | awk '{print \$1}' || true
  sha256sum "\${JOB_GIT_PROVENANCE_DIR}/git_diff.patch" 2>/dev/null | awk '{print \$1}' || true
  sha256sum "\${JOB_GIT_PROVENANCE_DIR}/git_untracked_file_hashes.tsv" 2>/dev/null | awk '{print \$1}' || true
} > "\${JOB_GIT_PROVENANCE_DIR}/git_component_sha256.txt"
sha256sum "\${JOB_GIT_PROVENANCE_DIR}/git_component_sha256.txt" | awk '{print \$1}' > "\${JOB_GIT_PROVENANCE_DIR}/git_worktree_fingerprint.sha256"
echo "Git provenance dir: \${JOB_GIT_PROVENANCE_DIR}"
echo "Git diff SHA256: \$(awk '{print \$1}' "\${JOB_GIT_PROVENANCE_DIR}/git_diff.sha256" 2>/dev/null || true)"
echo "Git worktree fingerprint SHA256: \$(cat "\${JOB_GIT_PROVENANCE_DIR}/git_worktree_fingerprint.sha256" 2>/dev/null || true)"
EXPECTED_GIT_HEAD="${SUBMISSION_GIT_HEAD}"
EXPECTED_GIT_WORKTREE_FINGERPRINT_SHA256="${SUBMISSION_GIT_WORKTREE_FINGERPRINT_SHA256}"
ACTUAL_GIT_HEAD="\$(cat "\${JOB_GIT_PROVENANCE_DIR}/git_head.txt")"
ACTUAL_GIT_WORKTREE_FINGERPRINT_SHA256="\$(cat "\${JOB_GIT_PROVENANCE_DIR}/git_worktree_fingerprint.sha256")"
if [[ "\${ACTUAL_GIT_HEAD}" != "\${EXPECTED_GIT_HEAD}" ]]; then
  echo "ERROR: queued-job Git HEAD drift: expected \${EXPECTED_GIT_HEAD}, got \${ACTUAL_GIT_HEAD}" >&2
  exit 2
fi
if [[ "\${ACTUAL_GIT_WORKTREE_FINGERPRINT_SHA256}" != "\${EXPECTED_GIT_WORKTREE_FINGERPRINT_SHA256}" ]]; then
  echo "ERROR: queued-job worktree fingerprint drift: expected \${EXPECTED_GIT_WORKTREE_FINGERPRINT_SHA256}, got \${ACTUAL_GIT_WORKTREE_FINGERPRINT_SHA256}" >&2
  exit 2
fi
echo "Queued-job Git provenance gate ok"
echo "Slurm job: \${SLURM_JOB_ID}"
echo "Host: \$(hostname)"
echo "SLURM_JOB_GPUS=\${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=\${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
echo "TMPDIR=\${TMPDIR}"
echo "RECOVAR_CACHE_DIR=\${RECOVAR_CACHE_DIR:-<disabled>}"
echo "PYTHONFAULTHANDLER=\${PYTHONFAULTHANDLER}"
echo "RECOVAR_CUDA_LIB=\${RECOVAR_CUDA_LIB}"
echo "RECOVAR_RELION_BIND_BUILD_DIR=\${RECOVAR_RELION_BIND_BUILD_DIR}"
echo "RELION_SRC_DIR=\${RELION_SRC_DIR:-<unset>}"
echo "CUDA_HOME=\${CUDA_HOME}"
echo "LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}"
$(write_optional_env_exports TF_GPU_ALLOCATOR RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET RECOVAR_PASS1_FUSED RECOVAR_DISABLE_LOCAL_BIG_JIT RECOVAR_LOCAL_BUCKET_QUANTUM RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS RECOVAR_K1_DENSE_PASS2 RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING RECOVAR_K1_RELION_X_HALF_MSTEP RECOVAR_EM_LOW_PMAX_REFINE_GUARD RECOVAR_EM_LOW_PMAX_REFINE_MAX_AVE_PMAX RECOVAR_EM_LOW_PMAX_REFINE_MIN_RES_STALL RECOVAR_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT RECOVAR_BPREF_ACCUM_DUMP_DIR RECOVAR_PASS2_DUMP_DIR RECOVAR_PASS2_DUMP_ORIGINAL_INDICES RECOVAR_PASS2_DUMP_CURRENT_SIZE RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL RECOVAR_LOCAL_SCORE_DUMP_DIR RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE RECOVAR_LOCAL_SCORE_DUMP_ITERATION RECOVAR_LOCAL_SCORE_DUMP_LABEL RECOVAR_LOCAL_SCORE_DUMP_OPERANDS RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS RECOVAR_RELION_PROJECTOR_DUMP_DIR RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR RECOVAR_MSTEP_DUMP_DIR RECOVAR_MSTEP_DUMP_MAX_CALLS RECOVAR_MSTEP_DUMP_RAW RECOVAR_SAVE_INTERMEDIATES_DIR RELION_DUMP_DIR RELION_DUMP_STACK_INDEX RELION_DUMP_PART_ID RELION_DUMP_PART RELION_DUMP_ITER)
export RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED="${RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED}"
echo "RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED=${RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED}"
$(write_optional_env_exports RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB)
for env_name in TF_GPU_ALLOCATOR RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET RECOVAR_PASS1_FUSED RECOVAR_DISABLE_LOCAL_BIG_JIT RECOVAR_LOCAL_BUCKET_QUANTUM RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS RECOVAR_K1_DENSE_PASS2 RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING RECOVAR_K1_RELION_X_HALF_MSTEP RECOVAR_EM_LOW_PMAX_REFINE_GUARD RECOVAR_EM_LOW_PMAX_REFINE_MAX_AVE_PMAX RECOVAR_EM_LOW_PMAX_REFINE_MIN_RES_STALL RECOVAR_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT RECOVAR_BPREF_ACCUM_DUMP_DIR RECOVAR_PASS2_DUMP_DIR RECOVAR_PASS2_DUMP_ORIGINAL_INDICES RECOVAR_PASS2_DUMP_CURRENT_SIZE RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL RECOVAR_LOCAL_SCORE_DUMP_DIR RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE RECOVAR_LOCAL_SCORE_DUMP_ITERATION RECOVAR_LOCAL_SCORE_DUMP_LABEL RECOVAR_LOCAL_SCORE_DUMP_OPERANDS RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS RECOVAR_RELION_PROJECTOR_DUMP_DIR RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR RECOVAR_MSTEP_DUMP_DIR RECOVAR_MSTEP_DUMP_MAX_CALLS RECOVAR_MSTEP_DUMP_RAW RECOVAR_SAVE_INTERMEDIATES_DIR RELION_DUMP_DIR RELION_DUMP_STACK_INDEX RELION_DUMP_PART_ID RELION_DUMP_PART RELION_DUMP_ITER; do
  env_value="\${!env_name-}"
  if [[ -n "\${env_value}" ]]; then
    echo "\${env_name}=\${env_value}"
  fi
done
for env_name in RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB; do
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
CUDA_LIB_TMP="\${RECOVAR_CUDA_LIB}.\${SLURM_JOB_ID:-\$\$}.tmp"
export CUDA_LIB_TMP PIXI_PY
flock "${SCRATCH_DIR}/cuda/build.lock" bash -lc '
  set -euo pipefail
  if [[ -s "\${RECOVAR_CUDA_LIB}" ]]; then
    echo "Reusing shared CUDA library \${RECOVAR_CUDA_LIB}"
    exit 0
  fi
  rm -f "\${CUDA_LIB_TMP}"
  env PYTHON="\${PIXI_PY}" make -C recovar/cuda LIB="\${CUDA_LIB_TMP}" all
  mv -f "\${CUDA_LIB_TMP}" "\${RECOVAR_CUDA_LIB}"
'
EOF
}

write_setup_script() {
  local script_path="${SCRATCH_DIR}/jobs/em_k1_matrix_setup.sh"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_k1_matrix_setup
#SBATCH --output=${SCRATCH_DIR}/em_k1_matrix_setup.out
#SBATCH --error=${SCRATCH_DIR}/em_k1_matrix_setup.err
#SBATCH --partition=${SETUP_PARTITION}
#SBATCH --account=${ACCOUNT}
${SBATCH_SETUP_CONSTRAINT_DIRECTIVE}
${SBATCH_SETUP_GRES_DIRECTIVE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

$(write_job_preamble "em_k1_matrix_setup")

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
# The default setup partition is CPU-only.  Install RECOVAR and build the
# host RELION binding here; each GPU case builds/reuses the shared custom CUDA
# library under cuda/build.lock after CUDA is actually available.

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

write_case_script() {
  local row="$1"
  local idx name n_images grid noise_level noise_model dataset_params_option seed pdb_bfactor noise_scale_std contrast_std volume_radius relion_bg_radius time_limit mem streaming_chunk streaming_mmap percent_outliers put_extra_particles image_offset_n_std
  IFS='|' read -r idx name n_images grid noise_level noise_model dataset_params_option seed pdb_bfactor noise_scale_std contrast_std volume_radius relion_bg_radius time_limit mem streaming_chunk streaming_mmap percent_outliers put_extra_particles image_offset_n_std <<< "${row}"

  local case_root="${SCRATCH_DIR}/cases/${idx}_${name}"
  local data_dir="${case_root}/data"
  local recovar_dir="${case_root}/recovar"
  local relion_dir="${case_root}/relion_ref"
  local script_path="${SCRATCH_DIR}/jobs/em_k1_matrix_${idx}_${name}.sh"

  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_k1_${idx}_${name:0:18}
#SBATCH --output=${SCRATCH_DIR}/em_k1_matrix_${idx}_${name}.out
#SBATCH --error=${SCRATCH_DIR}/em_k1_matrix_${idx}_${name}.err
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
${SBATCH_CONSTRAINT_DIRECTIVE}
#SBATCH --gres=gpu:1
${SBATCH_EXCLUSIVE_DIRECTIVE}
#SBATCH --nodes=1
#SBATCH --ntasks=${RELION_MPI_RANKS}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${time_limit}

$(write_job_preamble "em_k1_matrix_${idx}_${name}")

CASE_ROOT="${case_root}"
DATA_DIR="${data_dir}"
RECOVAR_DIR="${recovar_dir}"
RELION_DIR="${relion_dir}"
mkdir -p "\${CASE_ROOT}" "\${DATA_DIR}" "\${RECOVAR_DIR}" "\${RELION_DIR}"
if [[ -n "\${RECOVAR_BPREF_ACCUM_DUMP_DIR:-}" ]]; then
  export RECOVAR_BPREF_ACCUM_DUMP_DIR="\${RECOVAR_BPREF_ACCUM_DUMP_DIR%/}/${idx}_${name}"
  mkdir -p "\${RECOVAR_BPREF_ACCUM_DUMP_DIR}"
  echo "RECOVAR_BPREF_ACCUM_DUMP_DIR=\${RECOVAR_BPREF_ACCUM_DUMP_DIR}"
fi
if [[ -n "\${RECOVAR_PASS2_DUMP_DIR:-}" ]]; then
  export RECOVAR_PASS2_DUMP_DIR="\${RECOVAR_PASS2_DUMP_DIR%/}/${idx}_${name}"
  mkdir -p "\${RECOVAR_PASS2_DUMP_DIR}"
  echo "RECOVAR_PASS2_DUMP_DIR=\${RECOVAR_PASS2_DUMP_DIR}"
fi
if [[ -n "\${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR:-}" ]]; then
  export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR="\${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR%/}/${idx}_${name}"
  mkdir -p "\${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR}"
  echo "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR=\${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR}"
fi
if [[ -n "\${RECOVAR_LOCAL_SCORE_DUMP_DIR:-}" ]]; then
  export RECOVAR_LOCAL_SCORE_DUMP_DIR="\${RECOVAR_LOCAL_SCORE_DUMP_DIR%/}/${idx}_${name}"
  mkdir -p "\${RECOVAR_LOCAL_SCORE_DUMP_DIR}"
  echo "RECOVAR_LOCAL_SCORE_DUMP_DIR=\${RECOVAR_LOCAL_SCORE_DUMP_DIR}"
fi
if [[ -n "\${RECOVAR_RELION_PROJECTOR_DUMP_DIR:-}" ]]; then
  export RECOVAR_RELION_PROJECTOR_DUMP_DIR="\${RECOVAR_RELION_PROJECTOR_DUMP_DIR%/}/${idx}_${name}"
  mkdir -p "\${RECOVAR_RELION_PROJECTOR_DUMP_DIR}"
  echo "RECOVAR_RELION_PROJECTOR_DUMP_DIR=\${RECOVAR_RELION_PROJECTOR_DUMP_DIR}"
fi
if [[ -n "\${RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR:-}" ]]; then
  export RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR="\${RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR%/}/${idx}_${name}"
  mkdir -p "\${RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR}"
  echo "RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR=\${RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR}"
fi
if [[ -n "\${RECOVAR_MSTEP_DUMP_DIR:-}" ]]; then
  export RECOVAR_MSTEP_DUMP_DIR="\${RECOVAR_MSTEP_DUMP_DIR%/}/${idx}_${name}"
  mkdir -p "\${RECOVAR_MSTEP_DUMP_DIR}"
  echo "RECOVAR_MSTEP_DUMP_DIR=\${RECOVAR_MSTEP_DUMP_DIR}"
fi
if [[ -n "\${RELION_DUMP_DIR:-}" ]]; then
  export RELION_DUMP_DIR="\${RELION_DUMP_DIR%/}/${idx}_${name}"
  mkdir -p "\${RELION_DUMP_DIR}"
  echo "RELION_DUMP_DIR=\${RELION_DUMP_DIR}"
fi
RECOVAR_EXTRA_ARGS=()
if [[ -n "\${RECOVAR_SAVE_INTERMEDIATES_DIR:-}" ]]; then
  if [[ "\${RECOVAR_SAVE_INTERMEDIATES_DIR}" == "1" || "\${RECOVAR_SAVE_INTERMEDIATES_DIR}" == "auto" ]]; then
    RECOVAR_INTERMEDIATES_DIR="\${RECOVAR_DIR}/intermediates"
  else
    RECOVAR_INTERMEDIATES_DIR="\${RECOVAR_SAVE_INTERMEDIATES_DIR%/}/${idx}_${name}"
  fi
  mkdir -p "\${RECOVAR_INTERMEDIATES_DIR}"
  echo "RECOVAR_INTERMEDIATES_DIR=\${RECOVAR_INTERMEDIATES_DIR}"
  RECOVAR_EXTRA_ARGS+=(--save_intermediates_dir "\${RECOVAR_INTERMEDIATES_DIR}")
  if [[ "\${RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    RECOVAR_EXTRA_ARGS+=(--save_intermediates_skip_unregularized)
  fi
fi

cat > "\${CASE_ROOT}/case_config.json" <<JSON
{
  "index": ${idx},
  "name": "${name}",
  "n_images": ${n_images},
  "grid_size": ${grid},
  "noise_level": ${noise_level},
  "noise_model": "${noise_model}",
  "dataset_params_option": "${dataset_params_option}",
  "seed": ${seed},
  "pdb_bfactor": ${pdb_bfactor},
  "noise_scale_std": ${noise_scale_std},
  "contrast_std": ${contrast_std},
  "volume_radius": ${volume_radius},
  "relion_bg_radius_px": "${relion_bg_radius}",
  "streaming_mmap": ${streaming_mmap},
  "percent_outliers": ${percent_outliers},
  "put_extra_particles": ${put_extra_particles},
  "image_offset_n_std": ${image_offset_n_std},
  "max_iter": ${MAX_ITER},
  "run_relion": ${RUN_RELION}
}
JSON

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 60 > "\${CASE_ROOT}/gpu_monitor.csv" &
MONITOR_PID="\$!"
trap 'kill "\${MONITOR_PID}" 2>/dev/null || true' EXIT

PIXI_PY="\$(pixi run --frozen which python)"
if ! "\${PIXI_PY}" - <<'PY'
import pathlib
import recovar

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
assert str(recovar_file).startswith(str(repo) + "/"), recovar_file
PY
then
  echo "RECOVAR editable install failed provenance check; reinstalling under lock"
  flock "${INSTALL_LOCK}" bash -lc 'pixi run --frozen install-recovar'
  PIXI_PY="\$(pixi run --frozen which python)"
fi
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
print("case provenance/cuda gate ok")
PY

PREPARE_ARGS=(
  --output-dir "\${DATA_DIR}"
  --n-images "${n_images}"
  --grid-size "${grid}"
  --noise-level "${noise_level}"
  --noise-model "${noise_model}"
  --dataset-params-option "${dataset_params_option}"
  --pdb-bfactor "${pdb_bfactor}"
  --noise-scale-std "${noise_scale_std}"
  --contrast-std "${contrast_std}"
  --volume-radius "${volume_radius}"
  --percent-outliers "${percent_outliers}"
  --image-offset-n-std "${image_offset_n_std}"
  --init-resolution-ang 30.0
  --seed "${seed}"
  --relion-normalize
  --streaming-chunk-size "${streaming_chunk}"
  --disc-type cubic
)
if [[ "${streaming_mmap}" -eq 1 ]]; then
  PREPARE_ARGS+=(--streaming-mmap)
else
  PREPARE_ARGS+=(--no-streaming-mmap)
fi
if [[ "${put_extra_particles}" -eq 1 ]]; then
  PREPARE_ARGS+=(--put-extra-particles)
fi
if [[ "${relion_bg_radius}" != "-" ]]; then
  PREPARE_ARGS+=(--relion-bg-radius-px "${relion_bg_radius}")
fi
if [[ -n "${NOISE_RNG_BATCH_SIZE}" ]]; then
  PREPARE_ARGS+=(--noise-rng-batch-size "${NOISE_RNG_BATCH_SIZE}")
fi

echo "=== Prepare ${name} ==="
PREP_START="\$(date +%s)"
"\${PIXI_PY}" -m scripts.prepare_pdb_k1_relion_sanity_benchmark "\${PREPARE_ARGS[@]}" 2>&1 | tee "\${CASE_ROOT}/prepare.log"
PREP_STATUS="\${PIPESTATUS[0]}"
PREP_END="\$(date +%s)"
cat > "\${CASE_ROOT}/prepare_walltime.json" <<JSON
{"slurm_job_id":"\${SLURM_JOB_ID}","start_epoch":\${PREP_START},"end_epoch":\${PREP_END},"external_wall_s":\$((PREP_END - PREP_START)),"exit_status":\${PREP_STATUS}}
JSON
if [[ "\${PREP_STATUS}" -ne 0 ]]; then
  exit "\${PREP_STATUS}"
fi

if [[ "${RUN_RELION}" -eq 1 ]]; then
  echo "=== Run RELION AutoRefine ${name} ==="
  RELION_START="\$(date +%s)"
  (
    unset LD_LIBRARY_PATH
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    fi
    export PS1="${PS1:-}"
    set +u
    module load "${RELION_MODULE}"
    set -u
    if [[ -n "${RELION_EXTRA_LD_LIBRARY_PATH}" ]]; then
      export LD_LIBRARY_PATH="${RELION_EXTRA_LD_LIBRARY_PATH}:\${LD_LIBRARY_PATH:-}"
    fi
    RELION_REFINE_MPI_BIN="${RELION_REFINE_MPI}"
    if [[ "\${RELION_REFINE_MPI_BIN}" == */* ]]; then
      if [[ ! -x "\${RELION_REFINE_MPI_BIN}" ]]; then
        echo "ERROR: RELION_REFINE_MPI is not executable: \${RELION_REFINE_MPI_BIN}" >&2
        exit 2
      fi
      echo "RELION_REFINE_MPI=\${RELION_REFINE_MPI_BIN}"
    else
      RELION_REFINE_MPI_BIN="\$(command -v "\${RELION_REFINE_MPI_BIN}")"
    fi
    echo "RELION_REFINE_MPI_RESOLVED=\${RELION_REFINE_MPI_BIN}"
    echo "RELION_REFINE_MPI_SHA256=\$(sha256sum "\${RELION_REFINE_MPI_BIN}" | awk '{print \$1}')"
    command -v mpirun
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    export CUDA_VISIBLE_DEVICES=0
    RELION_TMPDIR="\${SLURM_TMPDIR:-/tmp/\${USER:-mg6942}/relion_\${SLURM_JOB_ID:-manual}_${idx}_${name}}"
    mkdir -p "\${RELION_TMPDIR}"
    export TMPDIR="\${RELION_TMPDIR}"
    export TMP="\${RELION_TMPDIR}"
    export TEMP="\${RELION_TMPDIR}"
    export OMPI_MCA_orte_tmpdir_base="\${RELION_TMPDIR}"
    export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
    echo "RELION_TMPDIR=\${RELION_TMPDIR}"
    cd "\${DATA_DIR}"
    RELION_CTF_ARGS=(--ctf)
    RELION_INPUT_STAR="particles.star"
    if [[ "${dataset_params_option}" == "noctf" ]]; then
      if [[ "${NOCTF_RELION_USE_CTF}" == "1" ]]; then
        RELION_INPUT_STAR="particles_relion_identity_ctf.star"
        if [[ ! -s "\${RELION_INPUT_STAR}" || "\${RELION_INPUT_STAR}" -ot "particles.star" ]]; then
          "\${PIXI_PY}" -m scripts.make_relion_identity_ctf_star \\
            --input-star particles.star \\
            --output-star "\${RELION_INPUT_STAR}" \\
            --manifest particles_relion_identity_ctf.json \\
            --phase-shift-deg 180.0
        fi
      else
        RELION_CTF_ARGS=()
      fi
    fi
    echo "RELION_CTF_ARGS=\${RELION_CTF_ARGS[*]:-<none>}"
    echo "RELION_INPUT_STAR=\${RELION_INPUT_STAR}"
    RELION_ITER_PADDED="\$(printf "%03d" "${MAX_ITER}")"
    if [[ ! -s "\${RELION_DIR}/run_it\${RELION_ITER_PADDED}_half1_class001.mrc" || ! -s "\${RELION_DIR}/run_it\${RELION_ITER_PADDED}_half2_class001.mrc" ]]; then
      mpirun -n "${RELION_MPI_RANKS}" "\${RELION_REFINE_MPI_BIN}" \\
        --i "\${RELION_INPUT_STAR}" \\
        --ref reference_init_relion.mrc \\
        --o "\${RELION_DIR}/run" \\
        --auto_refine \\
        --split_random_halves \\
        --particle_diameter 200 \\
        --ini_high 30 \\
        --firstiter_cc \\
        "\${RELION_CTF_ARGS[@]}" \\
        --flatten_solvent \\
        --zero_mask \\
        --low_resol_join_halves 40 \\
        --norm \\
        --scale \\
        --healpix_order 3 \\
        --offset_range 3.0 \\
        --offset_step 1.0 \\
        --oversampling 1 \\
        --pad 2 \\
        --random_seed "${seed}" \\
        --auto_iter_max "${MAX_ITER}" \\
        --pool "${RELION_POOL}" \\
        --gpu 0 \\
        --j 4
    else
      echo "Reusing RELION output in \${RELION_DIR}"
    fi
  ) 2>&1 | tee "\${CASE_ROOT}/relion_autorefine.log"
  RELION_STATUS="\${PIPESTATUS[0]}"
  RELION_END="\$(date +%s)"
  cat > "\${RELION_DIR}/slurm_walltime.json" <<JSON
{"slurm_job_id":"\${SLURM_JOB_ID}","start_epoch":\${RELION_START},"end_epoch":\${RELION_END},"external_wall_s":\$((RELION_END - RELION_START)),"exit_status":\${RELION_STATUS}}
JSON
  if [[ "\${RELION_STATUS}" -ne 0 ]]; then
    exit "\${RELION_STATUS}"
  fi
fi

echo "=== Run RECOVAR K=1 EM ${name} ==="
RECOVAR_RELION_REPLAY_ARGS=()
RECOVAR_MAX_ITER="${MAX_ITER}"
if [[ "${RUN_RELION}" -eq 1 ]]; then
  RELION_ITER_PADDED="\$(printf "%03d" "${MAX_ITER}")"
  RELION_HALF_SET_STAR=""
  RELION_OPTIMISER_STAR=""

  try_relion_replay_pair() {
    local data_star="\$1"
    local optimiser_star="\$2"
    if [[ -s "\${data_star}" && -s "\${optimiser_star}" ]]; then
      RELION_HALF_SET_STAR="\${data_star}"
      RELION_OPTIMISER_STAR="\${optimiser_star}"
      return 0
    fi
    return 1
  }

  try_relion_replay_pair "\${RELION_DIR}/run_data.star" "\${RELION_DIR}/run_optimiser.star" || \\
    try_relion_replay_pair "\${RELION_DIR}/run_it\${RELION_ITER_PADDED}_data.star" "\${RELION_DIR}/run_it\${RELION_ITER_PADDED}_optimiser.star" || true
  if [[ -z "\${RELION_HALF_SET_STAR}" ]]; then
    LATEST_RELION_DATA_ITER="\$(find "\${RELION_DIR}" -maxdepth 1 -type f -name 'run_it[0-9][0-9][0-9]_data.star' -printf '%f\n' \\
      | sed -E 's/^run_it([0-9]{3})_data\\.star$/\\1/' \\
      | sort -n \\
      | tail -1)"
    if [[ -n "\${LATEST_RELION_DATA_ITER}" ]]; then
      LATEST_RELION_DATA_ITER="\$(printf "%03d" "\$((10#\${LATEST_RELION_DATA_ITER}))")"
      try_relion_replay_pair "\${RELION_DIR}/run_it\${LATEST_RELION_DATA_ITER}_data.star" "\${RELION_DIR}/run_it\${LATEST_RELION_DATA_ITER}_optimiser.star" || true
    fi
  fi
  if [[ -z "\${RELION_HALF_SET_STAR}" ]]; then
    try_relion_replay_pair "\${RELION_DIR}/run_it000_data.star" "\${RELION_DIR}/run_it000_optimiser.star" || true
  fi
  if [[ -z "\${RELION_HALF_SET_STAR}" || -z "\${RELION_OPTIMISER_STAR}" ]]; then
    echo "ERROR: RELION was requested but no matched data/optimiser STAR pair was found in \${RELION_DIR}" >&2
    exit 3
  fi
  RECOVAR_RELION_REPLAY_ARGS=(
    --relion_half_sets "\${RELION_HALF_SET_STAR}"
    --relion_optimiser "\${RELION_OPTIMISER_STAR}"
    --relion_init_dir "\${RELION_DIR}"
    --perturb_replay_relion_dir "\${RELION_DIR}"
  )
  LATEST_RELION_SAMPLING_ITER="\$(find "\${RELION_DIR}" -maxdepth 1 -type f -name 'run_it[0-9][0-9][0-9]_sampling.star' -printf '%f\n' \\
    | sed -E 's/^run_it([0-9]{3})_sampling\\.star$/\\1/' \\
    | sort -n \\
    | tail -1)"
  if [[ -n "\${LATEST_RELION_SAMPLING_ITER}" ]]; then
    LATEST_RELION_SAMPLING_ITER="\$((10#\${LATEST_RELION_SAMPLING_ITER}))"
    if [[ "\${LATEST_RELION_SAMPLING_ITER}" -gt 0 && "\${LATEST_RELION_SAMPLING_ITER}" -lt "\${RECOVAR_MAX_ITER}" ]]; then
      echo "Strict RELION replay: capping RECOVAR max_iter \${RECOVAR_MAX_ITER} -> \${LATEST_RELION_SAMPLING_ITER} (last sampling.star)"
      RECOVAR_MAX_ITER="\${LATEST_RELION_SAMPLING_ITER}"
    fi
  fi
  echo "RECOVAR strict RELION replay args: \${RECOVAR_RELION_REPLAY_ARGS[*]}"
fi
START_EPOCH="\$(date +%s)"
set +e
"\${PIXI_PY}" -m scripts.run_full_refinement \\
  --data_dir "\${DATA_DIR}" \\
  --output "\${RECOVAR_DIR}" \\
  --max_iter "\${RECOVAR_MAX_ITER}" \\
  --healpix_order 3 \\
  --auto_local_healpix_order 4 \\
  --offset_range 3.0 \\
  --offset_step 1.0 \\
  --adaptive_oversampling 1 \\
  --init_resolution 30.0 \\
  --image_batch_size "${K1_IMAGE_BATCH_SIZE}" \\
  --rotation_block_size "${K1_ROTATION_BLOCK_SIZE}" \\
  --seed "${seed}" \\
  --perturb_seed "${seed}" \\
  --particle_diameter_ang 200 \\
  --tau2_fudge 1.0 \\
  --max_significants -1 \\
  --firstiter_cc \\
  --apply-initial-lowpass \\
  --benchmark_ledger_json "\${RECOVAR_DIR}/benchmark_ledger.json" \\
  --timing_dir "\${RECOVAR_DIR}/timing" \\
  "\${RECOVAR_EXTRA_ARGS[@]}" \\
  "\${RECOVAR_RELION_REPLAY_ARGS[@]}" \\
  2>&1 | tee "\${RECOVAR_DIR}/run_full_refinement.log"
STATUS="\${PIPESTATUS[0]}"
set -e
END_EPOCH="\$(date +%s)"
cat > "\${RECOVAR_DIR}/slurm_walltime.json" <<JSON
{"slurm_job_id":"\${SLURM_JOB_ID}","start_epoch":\${START_EPOCH},"end_epoch":\${END_EPOCH},"external_wall_s":\$((END_EPOCH - START_EPOCH)),"exit_status":\${STATUS}}
JSON
if [[ "\${STATUS}" -eq 0 ]]; then
  echo "=== Summarize RECOVAR K=1 EM ${name} ==="
  SUMMARY_ARGS=()
  if [[ "${RUN_RELION}" -eq 1 ]]; then
    SUMMARY_ARGS=(--k1-relion-dir "\${RELION_DIR}")
  fi
  set +e
  pixi run --frozen python -m scripts.summarize_em_completion_bench \\
    --k1-recovar-dir "\${RECOVAR_DIR}" \\
    --k1-fixture-dir "\${DATA_DIR}" \\
    --output-json "\${CASE_ROOT}/summary_metrics.json" \\
    --output-markdown "\${CASE_ROOT}/summary.md" \\
    --require-k1 \\
    "\${SUMMARY_ARGS[@]}"
  SUMMARY_STATUS="\${PIPESTATUS[0]}"
  set -e
  echo "case_summary_status=\${SUMMARY_STATUS}"
  if [[ -s "\${CASE_ROOT}/summary.md" ]]; then
    tail -80 "\${CASE_ROOT}/summary.md" || true
  fi
  if [[ "\${SUMMARY_STATUS}" -ne 0 ]]; then
    STATUS="\${SUMMARY_STATUS}"
  fi
fi
exit "\${STATUS}"
EOF
  chmod +x "${script_path}"
  printf '%s\n' "${script_path}"
}

write_summary_script() {
  local dependency="$1"
  local tracked_jobs="$2"
  local case_table="$3"
  local script_path="${SCRATCH_DIR}/jobs/em_k1_matrix_summary.sh"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_k1_matrix_summary
#SBATCH --output=${SCRATCH_DIR}/em_k1_matrix_summary.out
#SBATCH --error=${SCRATCH_DIR}/em_k1_matrix_summary.err
#SBATCH --partition=${SUMMARY_PARTITION}
#SBATCH --account=${ACCOUNT}
${SBATCH_SUMMARY_CONSTRAINT_DIRECTIVE}
${SBATCH_SUMMARY_GRES_DIRECTIVE}
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --dependency=${dependency}

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL
export PYTHONNOUSERSITE=1
export RECOVAR_DISABLE_CUDA=1
export CUDA_VISIBLE_DEVICES=""
export JAX_CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PIXI_FROZEN=true
export TMPDIR="${RUNTIME_ROOT}/em_k1_matrix_summary_\${SLURM_JOB_ID}/tmp"
export PIXI_HOME="${RUNTIME_ROOT}/em_k1_matrix_summary_\${SLURM_JOB_ID}/pixi_home"
export RATTLER_CACHE_DIR="${RUNTIME_ROOT}/em_k1_matrix_summary_\${SLURM_JOB_ID}/rattler_cache"
export RECOVAR_JAX_CACHE_DIR="${SCRATCH_DIR}/jax_cache"
export JAX_COMPILATION_CACHE_DIR="\${RECOVAR_JAX_CACHE_DIR}"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}" "\${RECOVAR_JAX_CACHE_DIR}" "${SCRATCH_DIR}/summaries"

EXPECTED_GIT_HEAD="${SUBMISSION_GIT_HEAD}"
ACTUAL_GIT_HEAD="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_GIT_HEAD}" != "\${EXPECTED_GIT_HEAD}" ]]; then
  echo "ERROR: queued-summary Git HEAD drift: expected \${EXPECTED_GIT_HEAD}, got \${ACTUAL_GIT_HEAD}" >&2
  exit 2
fi
if [[ -n "\$(git status --porcelain=v1)" ]]; then
  echo "ERROR: queued-summary worktree is dirty" >&2
  git status --short >&2
  exit 2
fi
echo "Queued-summary Git provenance gate ok"

echo "=== EM K=1 robustness matrix summary ==="
echo "Repo: ${REPO_ROOT}"
echo "HEAD: \$(git rev-parse HEAD)"
echo "Branch: \$(git symbolic-ref --short HEAD || echo '<detached>')"
echo "Scratch: ${SCRATCH_DIR}"
echo

for job_id in ${tracked_jobs}; do
  if [[ -n "\${job_id}" ]]; then
    sacct -j "\${job_id}" -X -o JobID,JobName%38,State,Elapsed,MaxRSS,ReqMem,AllocTRES || true
  fi
done
echo

MATRIX_SUMMARY_STATUS=0
while IFS='|' read -r idx name n_images grid noise_level noise_model dataset_params_option seed pdb_bfactor noise_scale_std contrast_std volume_radius relion_bg_radius time_limit mem streaming_chunk streaming_mmap percent_outliers put_extra_particles image_offset_n_std case_root case_job_id; do
  [[ -z "\${idx}" || "\${idx}" == "index" ]] && continue
  recovar_dir="\${case_root}/recovar"
  relion_dir="\${case_root}/relion_ref"
  data_dir="\${case_root}/data"
  summary_json="\${case_root}/summary_metrics.json"
  summary_md="\${case_root}/summary.md"
  echo "=== summarize \${idx} \${name} ==="
  RELION_SUMMARY_ARGS=()
  if [[ "${RUN_RELION}" -eq 1 ]]; then
    RELION_SUMMARY_ARGS=(--k1-relion-dir "\${relion_dir}")
  fi
  set +e
  pixi run --frozen python -m scripts.summarize_em_completion_bench \\
    --k1-recovar-dir "\${recovar_dir}" \\
    --k1-fixture-dir "\${data_dir}" \\
    --output-json "\${summary_json}" \\
    --output-markdown "\${summary_md}" \\
    --require-k1 \\
    "\${RELION_SUMMARY_ARGS[@]}"
  summary_status="\$?"
  set -e
  echo "summary_status=\${summary_status}"
  if [[ "\${summary_status}" -ne 0 ]]; then
    MATRIX_SUMMARY_STATUS="\${summary_status}"
  fi
  if [[ -s "\${summary_md}" ]]; then
    tail -80 "\${summary_md}" || true
  fi
  echo
done < "${case_table}"

pixi run --frozen python - <<'PY'
from __future__ import annotations

import json
import math
from pathlib import Path

scratch = Path("${SCRATCH_DIR}")
case_table = Path("${case_table}")
rows = []
for line in case_table.read_text().splitlines():
    if not line.strip() or line.startswith("index|"):
        continue
    parts = line.split("|")
    (
        idx,
        name,
        n_images,
        grid,
        noise_level,
        noise_model,
        dataset_params_option,
        seed,
        pdb_bfactor,
        noise_scale_std,
        contrast_std,
        volume_radius,
        relion_bg_radius,
        time_limit,
        mem,
        streaming_chunk,
        streaming_mmap,
        percent_outliers,
        put_extra_particles,
        image_offset_n_std,
        case_root,
        case_job_id,
    ) = parts
    case_root_path = Path(case_root)
    summary_path = case_root_path / "summary_metrics.json"
    recovar_wall_s = None
    recovar_exit = None
    fsc_auc = None
    corr = None
    relion_fsc_auc = None
    status = "missing"
    notes = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        k1 = summary.get("k1", {})
        status = k1.get("status", "missing")
        notes = k1.get("notes") or []
        metrics = k1.get("metrics") or {}
        rec_gt = metrics.get("recovar_merged_vs_gt") or {}
        rel_gt = metrics.get("relion_merged_vs_gt") or {}
        fsc_auc = rec_gt.get("fsc_auc")
        corr = rec_gt.get("corr")
        relion_fsc_auc = rel_gt.get("fsc_auc")
    wall_path = case_root_path / "recovar" / "slurm_walltime.json"
    if wall_path.exists():
        wall = json.loads(wall_path.read_text())
        recovar_wall_s = wall.get("external_wall_s")
        recovar_exit = wall.get("exit_status")
    rows.append(
        {
            "idx": idx,
            "name": name,
            "n_images": int(n_images),
            "grid": int(grid),
            "noise_level": float(noise_level),
            "noise_model": noise_model,
            "poses": dataset_params_option,
            "noise_scale_std": float(noise_scale_std),
            "contrast_std": float(contrast_std),
            "streaming_mmap": bool(int(streaming_mmap)),
            "percent_outliers": float(percent_outliers),
            "put_extra_particles": bool(int(put_extra_particles)),
            "image_offset_n_std": float(image_offset_n_std),
            "job_id": case_job_id,
            "status": status,
            "recovar_exit": recovar_exit,
            "recovar_wall_s": recovar_wall_s,
            "recovar_vs_gt_fsc_auc": fsc_auc,
            "recovar_vs_gt_corr": corr,
            "relion_vs_gt_fsc_auc": relion_fsc_auc,
            "notes": notes,
            "case_root": str(case_root_path),
            "summary_json": str(summary_path),
        }
    )

def fmt(value, digits=6):
    if value is None:
        return "missing"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}g}"
    return str(value)

out_json = scratch / "k1_robustness_matrix_summary.json"
out_json.write_text(json.dumps({"schema": "em_k1_robustness_matrix_v1", "rows": rows}, indent=2) + "\n")

lines = [
    "# EM K=1 Robustness Matrix",
    "",
    "| # | Case | N | Grid | Noise | Poses | Stress | Job | Status | RECOVAR wall s | RECOVAR GT FSC AUC | RECOVAR GT corr | RELION GT FSC AUC |",
    "|---:|---|---:|---:|---|---|---|---:|---|---:|---:|---:|---:|",
]
for row in rows:
    noise = f"{row['noise_model']} {row['noise_level']:.4g}"
    stress = []
    if row["percent_outliers"]:
        stress.append(f"outliers={row['percent_outliers']:.2g}")
    if row["put_extra_particles"]:
        stress.append("extra")
    if row["noise_scale_std"]:
        stress.append(f"noise_scale_std={row['noise_scale_std']:.2g}")
    if row["contrast_std"]:
        stress.append(f"contrast_std={row['contrast_std']:.2g}")
    if row["image_offset_n_std"]:
        stress.append(f"offset={row['image_offset_n_std']:.2g}")
    stress_text = ",".join(stress) if stress else "-"
    lines.append(
        "| "
        + " | ".join(
            [
                row["idx"],
                row["name"],
                str(row["n_images"]),
                str(row["grid"]),
                noise,
                row["poses"],
                stress_text,
                fmt(row["job_id"]),
                row["status"],
                fmt(row["recovar_wall_s"]),
                fmt(row["recovar_vs_gt_fsc_auc"]),
                fmt(row["recovar_vs_gt_corr"]),
                fmt(row["relion_vs_gt_fsc_auc"]),
            ]
        )
        + " |"
    )
lines.extend(["", f"JSON: {out_json}", ""])
for row in rows:
    if row["notes"]:
        lines.append(f"- {row['idx']} {row['name']} notes: " + "; ".join(str(n) for n in row["notes"][:4]))
out_md = scratch / "k1_robustness_matrix_summary.md"
out_md.write_text("\n".join(lines).rstrip() + "\n")
print(out_md.read_text())
PY

echo "Matrix JSON: ${SCRATCH_DIR}/k1_robustness_matrix_summary.json"
echo "Matrix Markdown: ${SCRATCH_DIR}/k1_robustness_matrix_summary.md"
exit "\${MATRIX_SUMMARY_STATUS}"
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
    # options. The launcher uses these as configuration inputs, so strip them
    # before submission and let each generated script's directives take effect.
    env -u SBATCH_ACCOUNT -u SBATCH_PARTITION -u SBATCH_CONSTRAINT \
      -u SBATCH_GRES -u SBATCH_GPUS -u SBATCH_GPUS_PER_NODE \
      sbatch --parsable "$@" "${script_path}"
  fi
}

SUBMISSION_GIT_PROVENANCE_DIR="${SCRATCH_DIR}/provenance/submission"
capture_git_provenance_snapshot "${SUBMISSION_GIT_PROVENANCE_DIR}"
SUBMISSION_GIT_HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SUBMISSION_GIT_DIFF_SHA256="$(awk '{print $1}' "${SUBMISSION_GIT_PROVENANCE_DIR}/git_diff.sha256" 2>/dev/null || true)"
SUBMISSION_GIT_WORKTREE_FINGERPRINT_SHA256="$(cat "${SUBMISSION_GIT_PROVENANCE_DIR}/git_worktree_fingerprint.sha256" 2>/dev/null || true)"

echo "EM K=1 robustness matrix launcher"
echo "Repo: ${REPO_ROOT}"
echo "HEAD: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Branch: $(git -C "${REPO_ROOT}" symbolic-ref --short HEAD || echo '<detached>')"
echo "Submission git provenance: ${SUBMISSION_GIT_PROVENANCE_DIR}"
echo "Submission git diff SHA256: ${SUBMISSION_GIT_DIFF_SHA256:-<unavailable>}"
echo "Submission git worktree fingerprint SHA256: ${SUBMISSION_GIT_WORKTREE_FINGERPRINT_SHA256:-<unavailable>}"
echo "Scratch: ${SCRATCH_DIR}"
echo "Runtime root: ${RUNTIME_ROOT}"
echo "Partition/account: ${PARTITION}/${ACCOUNT}"
echo "Setup partition: ${SETUP_PARTITION}"
echo "Setup constraint: ${SETUP_CONSTRAINT:-<none>}"
echo "Summary partition: ${SUMMARY_PARTITION}"
echo "Summary constraint: ${SUMMARY_CONSTRAINT:-<none>}"
echo "Summary gres: ${SUMMARY_GRES:-<none>}"
echo "Constraint: ${CONSTRAINT:-<none>}"
echo "Setup gres: ${SETUP_GRES:-<none>}"
echo "Exclusive GPU jobs: ${EXCLUSIVE}"
echo "CUDA module: ${CUDA_MODULE}"
echo "Run RELION baselines: ${RUN_RELION}"
echo "RELION binding source: ${RELION_SRC_DIR:-<unset>}"
echo "Case time-limit override: ${TIME_LIMIT_OVERRIDE:-<none>}"
echo

CASE_TABLE="${SCRATCH_DIR}/selected_cases.tsv"
printf 'index|name|n_images|grid|noise_level|noise_model|dataset_params_option|seed|pdb_bfactor|noise_scale_std|contrast_std|volume_radius|relion_bg_radius_px|time_limit|mem|streaming_chunk|streaming_mmap|percent_outliers|put_extra_particles|image_offset_n_std|case_root|case_job_id\n' > "${CASE_TABLE}"

SETUP_SCRIPT="$(write_setup_script)"
SETUP_JOB_ID="$(submit_or_print "${SETUP_SCRIPT}")"
echo "Setup job: ${SETUP_JOB_ID}"

TRACKED_JOB_IDS=("${SETUP_JOB_ID}")
CASE_JOB_IDS=()
SELECTED_COUNT=0

for row in "${CASES[@]}"; do
  IFS='|' read -r idx name _rest <<< "${row}"
  if ! case_selected "${idx}" "${name}"; then
    continue
  fi
  row="$(apply_case_overrides "${row}")"
  SELECTED_COUNT=$((SELECTED_COUNT + 1))
  CASE_SCRIPT="$(write_case_script "${row}")"
  CASE_JOB_ID="$(submit_or_print "${CASE_SCRIPT}" --dependency=afterok:"${SETUP_JOB_ID}")"
  CASE_JOB_IDS+=("${CASE_JOB_ID}")
  TRACKED_JOB_IDS+=("${CASE_JOB_ID}")
  case_root="${SCRATCH_DIR}/cases/${idx}_${name}"
  printf '%s|%s|%s\n' "${row}" "${case_root}" "${CASE_JOB_ID}" >> "${CASE_TABLE}"
  echo "Case ${idx} ${name}: ${CASE_JOB_ID}"
done

if [[ "${SELECTED_COUNT}" -eq 0 ]]; then
  echo "No cases selected. Check --case or EM_K1_MATRIX_CASES." >&2
  exit 2
fi

SUMMARY_DEPENDENCY="afterany:$(IFS=:; echo "${TRACKED_JOB_IDS[*]}")"
TRACKED_JOB_TEXT="$(IFS=' '; echo "${TRACKED_JOB_IDS[*]}")"
SUMMARY_SCRIPT="$(write_summary_script "${SUMMARY_DEPENDENCY}" "${TRACKED_JOB_TEXT}" "${CASE_TABLE}")"
SUMMARY_JOB_ID="$(submit_or_print "${SUMMARY_SCRIPT}")"
echo "Summary job: ${SUMMARY_JOB_ID}"
echo "Logs and outputs: ${SCRATCH_DIR}"

cat > "${SCRATCH_DIR}/submission.env" <<EOF
REPO_ROOT=${REPO_ROOT}
HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD)
BRANCH=$(git -C "${REPO_ROOT}" symbolic-ref --short HEAD || echo '<detached>')
SUBMISSION_GIT_PROVENANCE_DIR=${SUBMISSION_GIT_PROVENANCE_DIR}
SUBMISSION_GIT_HEAD=${SUBMISSION_GIT_HEAD}
SUBMISSION_GIT_DIFF_SHA256=${SUBMISSION_GIT_DIFF_SHA256}
SUBMISSION_GIT_WORKTREE_FINGERPRINT_SHA256=${SUBMISSION_GIT_WORKTREE_FINGERPRINT_SHA256}
SCRATCH_DIR=${SCRATCH_DIR}
RUNTIME_ROOT=${RUNTIME_ROOT}
SBATCH_PARTITION=${PARTITION}
EM_K1_MATRIX_SETUP_PARTITION=${SETUP_PARTITION}
EM_K1_MATRIX_SETUP_CONSTRAINT=${SETUP_CONSTRAINT}
EM_K1_MATRIX_SUMMARY_PARTITION=${SUMMARY_PARTITION}
EM_K1_MATRIX_SUMMARY_CONSTRAINT=${SUMMARY_CONSTRAINT}
EM_K1_MATRIX_SUMMARY_GRES=${SUMMARY_GRES}
EM_K1_MATRIX_SETUP_GRES=${SETUP_GRES}
SBATCH_ACCOUNT=${ACCOUNT}
SBATCH_CONSTRAINT=${CONSTRAINT}
EM_K1_MATRIX_EXCLUSIVE=${EXCLUSIVE}
EM_K1_MATRIX_SINGLE_VISIBLE_GPU=${SINGLE_VISIBLE_GPU}
EM_K1_MATRIX_RUN_RELION=${RUN_RELION}
EM_K1_MATRIX_MAX_ITER=${MAX_ITER}
EM_K1_MATRIX_TIME_LIMIT=${TIME_LIMIT_OVERRIDE}
K1_IMAGE_BATCH_SIZE=${K1_IMAGE_BATCH_SIZE}
K1_ROTATION_BLOCK_SIZE=${K1_ROTATION_BLOCK_SIZE}
STREAMING_CHUNK_SIZE=${STREAMING_CHUNK_SIZE}
NOISE_RNG_BATCH_SIZE=${NOISE_RNG_BATCH_SIZE}
SETUP_JOB_ID=${SETUP_JOB_ID}
CASE_JOB_IDS='${CASE_JOB_IDS[*]}'
SUMMARY_JOB_ID=${SUMMARY_JOB_ID}
CASE_TABLE=${CASE_TABLE}
CUDA_LIB=${CUDA_LIB}
CUDA_MODULE=${CUDA_MODULE}
RELION_MODULE=${RELION_MODULE}
RELION_REFINE_MPI=${RELION_REFINE_MPI}
RELION_EXTRA_LD_LIBRARY_PATH=${RELION_EXTRA_LD_LIBRARY_PATH}
RELION_SRC_DIR=${RELION_SRC_DIR}
TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-}
RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=${RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION:-}
RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET=${RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET:-}
RECOVAR_PASS1_FUSED=${RECOVAR_PASS1_FUSED:-}
RECOVAR_DISABLE_LOCAL_BIG_JIT=${RECOVAR_DISABLE_LOCAL_BIG_JIT:-}
RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES=${RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES:-}
RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION=${RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION:-}
RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE=${RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE:-}
RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES=${RECOVAR_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES:-}
RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES=${RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES:-}
RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES=${RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES:-}
RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES=${RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES:-}
RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES=${RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES:-}
RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS=${RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS:-}
RECOVAR_K1_DENSE_PASS2=${RECOVAR_K1_DENSE_PASS2:-}
RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING=${RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING:-}
RECOVAR_K1_RELION_X_HALF_MSTEP=${RECOVAR_K1_RELION_X_HALF_MSTEP:-}
RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=${RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE:-}
RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE=${RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE:-}
RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE=${RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE:-}
RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=${RECOVAR_FINAL_ALL_DATA_GRID_CORRECT:-}
RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=${RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT:-}
RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY=${RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY:-}
RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT=${RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT:-}
RECOVAR_BPREF_ACCUM_DUMP_DIR=${RECOVAR_BPREF_ACCUM_DUMP_DIR:-}
RECOVAR_PASS2_DUMP_DIR=${RECOVAR_PASS2_DUMP_DIR:-}
RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${RECOVAR_PASS2_DUMP_ORIGINAL_INDICES:-}
RECOVAR_PASS2_DUMP_CURRENT_SIZE=${RECOVAR_PASS2_DUMP_CURRENT_SIZE:-}
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR=${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR:-}
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES=${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES:-}
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE=${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE:-}
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION=${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION:-}
RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL=${RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL:-}
RECOVAR_LOCAL_SCORE_DUMP_DIR=${RECOVAR_LOCAL_SCORE_DUMP_DIR:-}
RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES=${RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES:-}
RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE=${RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE:-}
RECOVAR_LOCAL_SCORE_DUMP_ITERATION=${RECOVAR_LOCAL_SCORE_DUMP_ITERATION:-}
RECOVAR_LOCAL_SCORE_DUMP_LABEL=${RECOVAR_LOCAL_SCORE_DUMP_LABEL:-}
RECOVAR_LOCAL_SCORE_DUMP_OPERANDS=${RECOVAR_LOCAL_SCORE_DUMP_OPERANDS:-}
RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS=${RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS:-}
RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB=${RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB:-}
RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST=${RECOVAR_EXACT_LOCAL_AUTO_MICROBATCH_BOOST:-}
RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS=${RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS:-}
RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS=${RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS:-}
RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS=${RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS:-}
RECOVAR_RELION_PROJECTOR_DUMP_DIR=${RECOVAR_RELION_PROJECTOR_DUMP_DIR:-}
RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR=${RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR:-}
RECOVAR_MSTEP_DUMP_DIR=${RECOVAR_MSTEP_DUMP_DIR:-}
RECOVAR_MSTEP_DUMP_MAX_CALLS=${RECOVAR_MSTEP_DUMP_MAX_CALLS:-}
RECOVAR_MSTEP_DUMP_RAW=${RECOVAR_MSTEP_DUMP_RAW:-}
RECOVAR_SAVE_INTERMEDIATES_DIR=${RECOVAR_SAVE_INTERMEDIATES_DIR:-}
RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED=${RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED:-}
RELION_DUMP_DIR=${RELION_DUMP_DIR:-}
RELION_DUMP_STACK_INDEX=${RELION_DUMP_STACK_INDEX:-}
RELION_DUMP_PART_ID=${RELION_DUMP_PART_ID:-}
RELION_DUMP_PART=${RELION_DUMP_PART:-}
RELION_DUMP_ITER=${RELION_DUMP_ITER:-}
EOF

if [[ "${WATCH}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
  echo "Watching jobs. Press Ctrl-C to stop watching; jobs remain queued/running."
  while true; do
    date
    squeue -j "$(IFS=,; echo "${TRACKED_JOB_IDS[*]},${SUMMARY_JOB_ID}")" || true
    if [[ -z "$(squeue -h -j "${SUMMARY_JOB_ID}" 2>/dev/null || true)" ]]; then
      break
    fi
    sleep 60
  done
  echo "--- summary tail ---"
  tail -160 "${SCRATCH_DIR}/em_k1_matrix_summary.out" 2>/dev/null || true
fi
