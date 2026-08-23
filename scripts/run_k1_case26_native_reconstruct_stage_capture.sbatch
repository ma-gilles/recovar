#!/usr/bin/env bash
#SBATCH --job-name=k1c26-recon-stage
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_reconstruct_stage_20260811T1545ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_reconstruct_stage_20260811T1545ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:10:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"
: "${EXPECTED_RELION_BINARY_SHA256:?submission must pin diagnostic binary hash}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?submission must pin diagnostic source diff hash}"

ROOT=${RUN_ROOT:?RUN_ROOT is required}
RUNTIME=${RUNTIME_ROOT:?RUNTIME_ROOT is required}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case26_native_reconstruct_stage_capture.sbatch
RELION_SOURCE=${K1_RELION_SOURCE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_source_20260810}
RELION_BINARY=${K1_RELION_BINARY:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_build_20260810/bin/relion_refine_mpi}
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
RELION_OUT=${ROOT}/relion
DUMP_DIR=${ROOT}/reconstruct

require_sha() {
    test "$(sha256sum "$2" | awk '{print $1}')" = "$1"
}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
require_sha "${EXPECTED_LAUNCHER_SHA256}" "${LAUNCHER}"
require_sha "${EXPECTED_RELION_BINARY_SHA256}" "${RELION_BINARY}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${RELION_SOURCE}" diff -- src/acc/acc_ml_optimiser_impl.h src/acc/cuda/cuda_ml_optimiser.cu src/backprojector.cpp | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
require_sha 34016f7f367587125956de8f00895c27b14225511c8ef0f517179b890685e139 "${DATA}/particles.star"
require_sha 4cd77e4ddeb731f719adb0cfd661e4f00221e07d08e15e1a5dc229737d9c848c "${DATA}/particles.128.mrcs"
require_sha 7aafc4f04e482f7410ff37686ba46db14578334ca77909f04bff30dbba2789d1 "${DATA}/reference_init_relion.mrc"
test -z "$(find "${DUMP_DIR}" -mindepth 1 -print -quit)"
test -z "$(find "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

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
export RELION_RECON_DUMP_DIR=${DUMP_DIR}
export RELION_RECON_DUMP_CALL=0

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "gpu_name=${GPU_NAME}"
    echo "repo_head=${EXPECTED_REPO_HEAD}"
    echo "relion_source_head=$(git -C "${RELION_SOURCE}" rev-parse HEAD)"
    echo "relion_source_diff_sha256=${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
    echo "reconstruct_dump_call=0"
    echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset"
    echo "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset"
    sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${DATA}/particles.star" \
        "${DATA}/particles.128.mrcs" "${DATA}/reference_init_relion.mrc"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"
git -C "${RELION_SOURCE}" diff -- src/acc/acc_ml_optimiser_impl.h src/acc/cuda/cuda_ml_optimiser.cu src/backprojector.cpp \
    > "${ROOT}/provenance/relion_reconstruct_capture.patch"

COMMAND=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1726 --auto_iter_max 2
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

for stage in tau2 fweight_decentered fweight_regularized radavg_weight \
    fconv_divided volume_before_gridding volume_after_gridding; do
    test "$(find "${DUMP_DIR}" -maxdepth 1 -type f -name "reconstruct_rank*_pid*_call0000_${stage}.bin" | wc -l)" -eq 2
done
test "$(find "${DUMP_DIR}" -maxdepth 1 -type f -name 'reconstruct_rank*_pid*_call0000_metadata.txt' | wc -l)" -eq 2
test "$(find "${DUMP_DIR}" -maxdepth 1 -type f | wc -l)" -eq 16
"${PYTHON}" - <<'PY' "${DUMP_DIR}" "${ROOT}/analysis/RECONSTRUCT_CAPTURE_VALIDATION.json"
import hashlib
import json
import struct
import sys
from pathlib import Path

dump_dir = Path(sys.argv[1])
records = []
for path in sorted(dump_dir.glob("*.bin")):
    raw = path.read_bytes()
    assert len(raw) >= 64
    header = struct.unpack_from("<8q", raw)
    z, y, x = header[:3]
    assert z * y * x > 0
    records.append({
        "name": path.name,
        "shape_zyx": [z, y, x],
        "header": list(header),
        "payload_bytes": len(raw) - 64,
        "sha256": hashlib.sha256(raw).hexdigest(),
    })
assert len(records) == 14
Path(sys.argv[2]).write_text(json.dumps({"schema": "relion.reconstruct.stages.v1", "records": records}, indent=2) + "\n")
PY
"${PYTHON}" /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout/scripts/audit_relion_control_reproducibility.py \
    --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" --iterations 1 2 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - <<'PY' "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json"
import json
import sys
report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY
sha256sum "${DUMP_DIR}"/* "${ROOT}/analysis/RECONSTRUCT_CAPTURE_VALIDATION.json" \
    "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d}\n' "${SLURM_JOB_ID}" "${wall_s}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
