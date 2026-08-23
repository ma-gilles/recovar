#!/usr/bin/env bash
#SBATCH --job-name=k1c04-fine-cont
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
TARGET_STACK_ONE_BASED=${TARGET_STACK_ONE_BASED:-61418}
CAPTURE_COARSE=${CAPTURE_COARSE:-0}
EXPECTED_COARSE_SIGNIFICANT_COUNT=${EXPECTED_COARSE_SIGNIFICANT_COUNT:-}
TARGET_ITERATION=2
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case04_native_fine_continuation.sbatch
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
CONTINUE=${CONTROL}/run_it001_optimiser.star
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/source
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/build/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
CAPTURE=${ROOT}/capture
RELION_OUT=${ROOT}/relion

test "${TARGET_STACK_ONE_BASED}" -gt 0
case "${CAPTURE_COARSE}" in 0|1) ;; *) echo "CAPTURE_COARSE must be 0 or 1" >&2; exit 2 ;; esac
if test "${CAPTURE_COARSE}" = 1; then
  test -n "${EXPECTED_COARSE_SIGNIFICANT_COUNT}"
fi
test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 1c3f67f52a7663f29c3cc0c8aacc995eba27517961e24ee4a1be01b6413c0b1d
test "$(sha256sum "${CONTINUE}" | awk '{print $1}')" = 881de25817c531ecc6df6879171d942152e27f8cfebfaf3db29c2052b0463b91
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 0a90714711cdf0a1820d46fa82460d828b87e1ecbd2e49de645170863546ac98
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD^{tree})" = 3111173edb150f98bc2446efc54229ff736b40a1
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = 15e9cceb38815cc85ed946e1bb5ee8a9b42f89a530cf8d5af2d76e83228be3f9
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = 187a9f60cdea19363d49d46099b052f23288f05a
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
mkdir -p "${CAPTURE}" "${RELION_OUT}"
test -z "$(find "${CAPTURE}" "${RELION_OUT}" -mindepth 1 -print -quit)"

mkdir -p "${ROOT}/analysis" "${ROOT}/provenance" "${ROOT}/logs"
mkdir -p "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache" \
  "${RUNTIME}/pycache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
  case "${variable_name}" in
    RECOVAR_*|RELION_BPRE_*|RELION_FINE_*|RELION_PREITER_*|RELION_SAMPLING_PERTURBATION_*)
      unset "${variable_name}"
      ;;
  esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

export RELION_BPRE_CAPTURE_DIR=${CAPTURE}
export RELION_BPRE_CAPTURE_SCHEMA=2
export RELION_BPRE_CAPTURE_ITER=${TARGET_ITERATION}
export RELION_BPRE_CAPTURE_CLASS=1
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=1
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=2000000000
export RELION_BPRE_CAPTURE_STACKS=${TARGET_STACK_ONE_BASED}
export RELION_BPRE_CAPTURE_GEOMETRY_ONLY=1

if test "${CAPTURE_COARSE}" = 1; then
  export RELION_COARSE_CAPTURE_DIR=${CAPTURE}
  export RELION_COARSE_CAPTURE_ITER=${TARGET_ITERATION}
  export RELION_COARSE_CAPTURE_STACK=${TARGET_STACK_ONE_BASED}
fi

# A continuation reconstructs sorted_idx and sampling state. Restore the
# uninterrupted fresh-run physical order and exact iteration-2 perturbation.
export RELION_PREITER_PARTICLE_ORDER_SEED=1705
export RELION_PREITER_PARTICLE_ORDER_ITER=${TARGET_ITERATION}
export RELION_SAMPLING_PERTURBATION_OVERRIDE=0.18417978286743164
export RELION_SAMPLING_PERTURBATION_OVERRIDE_ITER=${TARGET_ITERATION}

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
  --continue "${CONTINUE}" --o "${RELION_OUT}/run"
  --auto_iter_max "${TARGET_ITERATION}" --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${CONTINUE}" \
  "${REPO}/scripts/analyze_k1_fine_score_stages.py" \
  "${REPO}/scripts/analyze_k1_native_coarse_boundary.py" \
  "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start_epoch=$(date +%s)
(
  cd "${DATA}"
  "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

fine_score=$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-score-v1.bin')
factor=$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.bpre-v2.bin')
test -n "${fine_score}" -a -n "${factor}"
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.fine-score-v1.bin' | wc -l)" -eq 1
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.bpre-v2.bin' | wc -l)" -eq 1
test -s "${RELION_OUT}/run_it002_data.star"
"${PYTHON}" - "${fine_score}" "${factor}" "${TARGET_STACK_ONE_BASED}" <<'PY'
import sys
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture

fine = load_fine_score_capture(sys.argv[1])
factor = load_factor_capture(sys.argv[2])
expected_stack = int(sys.argv[3])
assert fine.stack_index == factor.stack_index == expected_stack
assert fine.header[4] == factor.header[9] == 2
assert factor.geometry_only
assert (fine.candidates["flags"] & ACTIVE).any()
PY
if test "${CAPTURE_COARSE}" = 1; then
  coarse=$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.coarse-v1.bin')
  test -n "${coarse}"
  test "$(printf '%s\n' "${coarse}" | wc -l)" -eq 1
  "${PYTHON}" - "${coarse}" "${TARGET_STACK_ONE_BASED}" \
    "${EXPECTED_COARSE_SIGNIFICANT_COUNT}" "${ROOT}/analysis/COARSE_CAPTURE_VALIDATION.json" <<'PY'
import json
import sys
from pathlib import Path

from scripts.analyze_k1_native_coarse_boundary import load_native_coarse_capture

path = Path(sys.argv[1])
stack = int(sys.argv[2])
expected_significant = int(sys.argv[3])
output = Path(sys.argv[4])
capture = load_native_coarse_capture(path)
header = capture.header
assert int(header[1]) == 2
assert int(header[2]) == stack
assert int(header[4]) == 36864
assert int(header[5]) == 29
assert int(header[9]) == expected_significant
output.write_text(json.dumps({
    "status": "complete",
    "path": str(path.resolve()),
    "iteration": int(header[1]),
    "stack_index_one_based": int(header[2]),
    "part_id": int(header[3]),
    "rotation_count": int(header[4]),
    "translation_count": int(header[5]),
    "significant_count": int(header[9]),
    "threshold_index": int(header[8]),
}, indent=2, sort_keys=True) + "\n")
PY
fi
rg -q '\[RELION_PREITER_PARTICLE_ORDER\] iteration 2 effective_seed 1705 particles 100000' \
  "${RELION_OUT}/runner.stderr"
rg -q '\[RELION_SAMPLING_PERTURBATION_OVERRIDE\] iter 2 requested 0\.18417978286743164 applied 0\.18417978286743164' \
  "${RELION_OUT}/runner.stdout"
! rg -qi 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${RELION_OUT}/runner.stderr"

"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" --iterations 2 \
  --map-fsc-auc-threshold 0.999999 \
  --output-json "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
  --output-markdown "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

find "${CAPTURE}" "${RELION_OUT}" "${ROOT}/analysis" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":2,"stack_index_one_based":%d,"continued_from":"%s","geometry_only":true}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" "${TARGET_STACK_ONE_BASED}" "${CONTINUE}" \
  > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
