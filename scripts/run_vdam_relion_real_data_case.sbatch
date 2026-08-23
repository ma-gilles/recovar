#!/usr/bin/env bash
#SBATCH --job-name=vdam-k1-real
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-k1-real-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?submit with OUTPUT_ROOT=/scratch/...}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
SCORECARD="${VDAM_REAL_SCORECARD:-${REPO_ROOT}/docs/math/vdam_k1_real_data_suite_v1.json}"
CASE_ID="${CASE_ID:-vdam-r01}"
RELION_REFINE="${RELION_REFINE:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}"
PIXI_PY="${REPO_ROOT}/.pixi/envs/default/bin/python"
cd "${REPO_ROOT}"
SOURCE_GIT_HEAD="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "VDAM real-data qualification requires a clean tracked worktree" >&2
  exit 2
fi

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA RECOVAR_CUDA_DISABLE
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ "${VDAM_ALLOW_ASYNC_CUDA:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]] &&
   [[ "${VDAM_DETERMINISTIC_CUDA:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "VDAM_ALLOW_ASYNC_CUDA and VDAM_DETERMINISTIC_CUDA are mutually exclusive" >&2
  exit 2
fi
RECOVAR_CUDA_MODE_ARGS=()
if [[ "${VDAM_DETERMINISTIC_CUDA:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  export CUDA_LAUNCH_BLOCKING=1
  RECOVAR_CUDA_MODE_ARGS+=(--deterministic_cuda)
else
  export CUDA_LAUNCH_BLOCKING=0
fi
RUNTIME_ROOT="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${SLURM_JOB_ID:-manual}"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export PIXI_HOME="${RUNTIME_ROOT}/pixi_home"
export RATTLER_CACHE_DIR="${RUNTIME_ROOT}/rattler_cache"
# This runner's TMPDIR is shared scratch, not node-local storage.  RECOVAR's
# generic loader otherwise stages the entire source stack there before a
# bootstrap run that touches only a small random subset (80 GB for the frozen
# EMPIAR-10180 case).  Match the EM benchmark runner: disable implicit staging
# unless the submitter explicitly supplies a cache location.
export RECOVAR_CACHE_DIR="${RECOVAR_CACHE_DIR-}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${OUTPUT_ROOT}/jax_cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export RECOVAR_CUDA_LIB="${REPO_ROOT}/recovar/cuda/libcuda_backproject.so"
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" \
  "${OUTPUT_ROOT}/relion" "${OUTPUT_ROOT}/recovar" /scratch/gpfs/GILLES/mg6942/slurmo
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"

eval "$("${PIXI_PY}" - "${SCORECARD}" "${CASE_ID}" <<'PY'
import json
import hashlib
import shlex
import sys
from pathlib import Path

scorecard = json.loads(Path(sys.argv[1]).read_text())
case = next(row for row in scorecard["cases"] if row["id"] == sys.argv[2])
input_star = Path(case["input_star"])
if hashlib.sha256(input_star.read_bytes()).hexdigest() != case["input_star_sha256"]:
    raise SystemExit(f"input STAR digest mismatch: {input_star}")
source_indices = input_star.with_name("source_indices.npy")
if hashlib.sha256(source_indices.read_bytes()).hexdigest() != case["source_indices_sha256"]:
    raise SystemExit(f"source-index digest mismatch: {source_indices}")
definition = case["definition"]
values = {
    "INPUT_STAR": case["input_star"],
    "DATA_DIR": case["data_dir"],
    "NR_ITER": definition["nr_iter"],
    "K": definition["nr_classes"],
    "TAU2_FUDGE": definition["tau2_fudge"],
    "HEALPIX_ORDER": definition["healpix_order"],
    "OVERSAMPLING": definition["oversampling"],
    "OFFSET_RANGE": definition["offset_range_px"],
    "OFFSET_STEP": definition["offset_step_px"],
    "PAD": definition["padding_factor"],
    "SYM": "C1" if definition.get("do_run_C1", True) else definition.get("symmetry", "C1"),
    "PARTICLE_DIAMETER": definition["particle_diameter_angstrom"],
    "RANDOM_SEED": definition["random_seed"],
    "IMAGE_BATCH_SIZE": definition["image_batch_size"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

test -x "${PIXI_PY}"
test -x "${RELION_REFINE}"
test -f "${INPUT_STAR}"
test -d "${DATA_DIR}"

"${PIXI_PY}" - <<'PY'
import jax
import recovar.cuda_backproject as cuda_backproject

assert any(device.platform in {"gpu", "cuda"} for device in jax.devices())
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print("VDAM real-data GPU/CUDA-FFI gate passed", jax.devices())
PY

RELION_CMD=(
  "${RELION_REFINE}"
  --o "${OUTPUT_ROOT}/relion/run"
  --iter "${NR_ITER}"
  --grad --denovo_3dref --grad_write_iter 1
  --i "${INPUT_STAR}" --ctf --K "${K}" --sym "${SYM}"
  --flatten_solvent --zero_mask --dont_combine_weights_via_disc
  --pool 3 --pad "${PAD}" --particle_diameter "${PARTICLE_DIAMETER}"
  --oversampling "${OVERSAMPLING}" --healpix_order "${HEALPIX_ORDER}"
  --offset_range "${OFFSET_RANGE}" --offset_step "${OFFSET_STEP}"
  --auto_sampling --tau2_fudge "${TAU2_FUDGE}" --random_seed "${RANDOM_SEED}"
  --j "${SLURM_CPUS_PER_TASK:-8}" --gpu 0
)

RECOVAR_CMD=(
  "${PIXI_PY}" scripts/run_ab_initio.py
  --i "${INPUT_STAR}" --o "${OUTPUT_ROOT}/recovar/run"
  --nr_iter "${NR_ITER}" --grad_write_iter 1 --K "${K}" --tau2_fudge "${TAU2_FUDGE}"
  --sym "${SYM}" --do_run_C1 1 --particle_diameter "${PARTICLE_DIAMETER}"
  --random_seed "${RANDOM_SEED}" --healpix_order "${HEALPIX_ORDER}"
  --oversampling "${OVERSAMPLING}" --offset_range "${OFFSET_RANGE}"
  --offset_step "${OFFSET_STEP}" --padding_factor "${PAD}"
  --image_batch_size "${IMAGE_BATCH_SIZE}" --datadir "${DATA_DIR}"
  --gpu 0 --require_custom_cuda
  "${RECOVAR_CUDA_MODE_ARGS[@]}"
)

"${PIXI_PY}" - "${OUTPUT_ROOT}" "${RELION_CMD[@]}" __RECOVAR_COMMAND__ "${RECOVAR_CMD[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
split = sys.argv.index("__RECOVAR_COMMAND__")
(root / "relion" / "relion_command.json").write_text(
    json.dumps({"argv": sys.argv[2:split]}, indent=2, sort_keys=True) + "\n"
)
(root / "recovar" / "recovar_command.json").write_text(
    json.dumps({"argv": sys.argv[split + 1:]}, indent=2, sort_keys=True) + "\n"
)
PY

GPU_UUID_BEFORE="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
SECONDS=0
(cd "${DATA_DIR}" && "${RELION_CMD[@]}") >"${OUTPUT_ROOT}/relion/relion.log" 2>&1
RELION_WALL_S="${SECONDS}"
GPU_UUID_AFTER_RELION="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
SECONDS=0
"${RECOVAR_CMD[@]}" >"${OUTPUT_ROOT}/recovar/recovar.log" 2>&1
RECOVAR_WALL_S="${SECONDS}"
GPU_UUID_AFTER_RECOVAR="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
test "${GPU_UUID_BEFORE}" = "${GPU_UUID_AFTER_RELION}"
test "${GPU_UUID_BEFORE}" = "${GPU_UUID_AFTER_RECOVAR}"

if [[ "$(git rev-parse HEAD)" != "${SOURCE_GIT_HEAD}" ]] ||
   [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "VDAM real-data qualification worktree changed during execution" >&2
  exit 2
fi

"${PIXI_PY}" - "${OUTPUT_ROOT}" "${SCORECARD}" "${CASE_ID}" "${RELION_WALL_S}" "${RECOVAR_WALL_S}" "${GPU_UUID_BEFORE}" "${RELION_REFINE}" "${SOURCE_GIT_HEAD}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root, scorecard, case_id = map(Path, sys.argv[1:4])
relion_refine = Path(sys.argv[7]).resolve(strict=True)
relion_reference = {
    "executable": str(relion_refine),
    "executable_sha256": hashlib.sha256(relion_refine.read_bytes()).hexdigest(),
    "executable_size_bytes": relion_refine.stat().st_size,
}
source = subprocess.run(
    ["git", "-C", str(relion_refine.parent), "rev-parse", "--show-toplevel"],
    check=False,
    capture_output=True,
    text=True,
)
if source.returncode == 0:
    source_root = Path(source.stdout.strip()).resolve()
    relion_reference.update(
        source_git_root=str(source_root),
        source_git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        source_git_tracked_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=source_root,
                text=True,
            ).strip()
        ),
    )
report = {
    "schema": "recovar.vdam_relion_real_data_run.v1",
    "case_id": case_id.name,
    "scorecard": str(scorecard),
    "scorecard_sha256": hashlib.sha256(scorecard.read_bytes()).hexdigest(),
    "git_head": sys.argv[8],
    "git_head_at_provenance": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "tracked_worktree_clean_at_provenance": not bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    ),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "physical_gpu_uuid": sys.argv[6],
    "relion_reference": relion_reference,
    "relion_wall_s": int(sys.argv[4]),
    "recovar_wall_s": int(sys.argv[5]),
    "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING") == "1",
    "cuda_launch_blocking_value": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    "recovar_cache_dir": os.environ.get("RECOVAR_CACHE_DIR"),
    "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
    "relion_x_half_f32_fine_posterior": os.environ.get(
        "RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR"
    ),
}
(root / "run_provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

"${PIXI_PY}" scripts/audit_vdam_real_trajectory.py \
  --scorecard "${SCORECARD}" --case-id "${CASE_ID}" \
  --recovar-dir "${OUTPUT_ROOT}/recovar" --relion-dir "${OUTPUT_ROOT}/relion" \
  --provenance "${OUTPUT_ROOT}/run_provenance.json" \
  --output-json "${OUTPUT_ROOT}/trajectory_audit.json" \
  --output-shells-npz "${OUTPUT_ROOT}/trajectory_shellwise_fsc.npz"

echo "VDAM real-data pair complete: ${OUTPUT_ROOT}"
