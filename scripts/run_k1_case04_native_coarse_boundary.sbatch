#!/usr/bin/env bash
#SBATCH --job-name=k1c04-coarse
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
CAPTURE_STACK=${CAPTURE_STACK:-68695}
CAPTURE_ITERATION=${CAPTURE_ITERATION:-2}
EXPECTED_SIGNIFICANT_COUNT=${EXPECTED_SIGNIFICANT_COUNT:-18}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/source
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/build/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case04_native_coarse_boundary.sbatch
CAPTURE=${ROOT}/capture
RELION_OUT=${ROOT}/relion

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD^{tree})" = 3111173edb150f98bc2446efc54229ff736b40a1
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = 15e9cceb38815cc85ed946e1bb5ee8a9b42f89a530cf8d5af2d76e83228be3f9
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 1c3f67f52a7663f29c3cc0c8aacc995eba27517961e24ee4a1be01b6413c0b1d
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 0a90714711cdf0a1820d46fa82460d828b87e1ecbd2e49de645170863546ac98
test "$(sha256sum "${DATA}/reference_init_relion.mrc" | awk '{print $1}')" = beffccd2b5b9e4dba765751b2e8e7933229deb70e65061de97a486264066dec9
test -x "${PYTHON}" -a -x "${RELION_BINARY}"
test -z "$(find "${CAPTURE}" "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

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

export RELION_COARSE_CAPTURE_DIR=${CAPTURE}
export RELION_COARSE_CAPTURE_ITER=${CAPTURE_ITERATION}
export RELION_COARSE_CAPTURE_STACK=${CAPTURE_STACK}

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
git -C "${RELION_SOURCE}" diff --binary > "${ROOT}/provenance/relion_capture.patch"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" \
    "${REPO}/scripts/analyze_k1_native_coarse_boundary.py" \
    "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
  --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
  --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
  --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
  --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
  --oversampling 1 --pad 2 --random_seed 1704 --auto_iter_max 2
  --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
  cd "${DATA}"
  "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.coarse-v1.bin' | wc -l)" -eq 1
test -z "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.tmp.*' -print -quit)"
"${PYTHON}" - "${CAPTURE}" "${CAPTURE_STACK}" "${CAPTURE_ITERATION}" \
  "${EXPECTED_SIGNIFICANT_COUNT}" "${ROOT}/analysis/COARSE_CAPTURE_VALIDATION.json" <<'PY'
import json
import sys
from pathlib import Path

from scripts.analyze_k1_native_coarse_boundary import load_native_coarse_capture

capture_dir = Path(sys.argv[1])
stack = int(sys.argv[2])
iteration = int(sys.argv[3])
expected_significant = int(sys.argv[4])
output = Path(sys.argv[5])
paths = list(capture_dir.glob("*.coarse-v1.bin"))
assert len(paths) == 1
capture = load_native_coarse_capture(paths[0])
header = capture.header
assert int(header[1]) == iteration
assert int(header[2]) == stack
assert int(header[4]) == 36864
assert int(header[5]) == 29
assert int(header[9]) == expected_significant
assert int((capture.postexponent >= capture.postexponent[capture.postexponent > 0].min()).sum()) > 0
output.write_text(json.dumps({
    "status": "complete",
    "path": str(paths[0].resolve()),
    "iteration": int(header[1]),
    "stack_index_one_based": int(header[2]),
    "part_id": int(header[3]),
    "rotation_count": int(header[4]),
    "translation_count": int(header[5]),
    "significant_count": int(header[9]),
    "threshold_index": int(header[8]),
}, indent=2, sort_keys=True) + "\n")
PY
"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
  --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" --iterations 1 2 \
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
printf '{"job_id":"%s","wall_s":%d,"stack_index_one_based":%d,"iteration":%d}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" "${CAPTURE_STACK}" "${CAPTURE_ITERATION}" \
  > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
