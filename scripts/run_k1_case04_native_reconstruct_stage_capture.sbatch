#!/usr/bin/env bash
#SBATCH --job-name=k1c04-recon-stage
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:00:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case04_native_reconstruct_stage_capture.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_source_20260810
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_build_20260810/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
RELION_OUT=${ROOT}/relion
DUMP_DIR=${ROOT}/reconstruct
ANALYSIS=${ROOT}/analysis

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = 60ba0c44004a323f231a59e051a5f444e6d94641bc628ce6b094cb1357d043ea
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 235052e48ccf843df48392c8e2b4f0d634b7dcedf7be27b88eb1cff746c07201
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 0a90714711cdf0a1820d46fa82460d828b87e1ecbd2e49de645170863546ac98
test "$(sha256sum "${DATA}/reference_init_relion.mrc" | awk '{print $1}')" = beffccd2b5b9e4dba765751b2e8e7933229deb70e65061de97a486264066dec9
test "$(stat -Lc %s "${DATA}/particles.256.mrcs")" = 26214401024
test -z "$(find "${DUMP_DIR}" "${RELION_OUT}" -mindepth 1 -print -quit)"

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
export RELION_RECON_DUMP_DIR=${DUMP_DIR}
export RELION_RECON_DUMP_CALL=0

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${RELION_SOURCE}" diff --binary > "${ROOT}/provenance/relion_capture.patch"
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

for stage in tau2 fweight_decentered fweight_regularized radavg_weight \
    fconv_divided volume_before_gridding volume_after_gridding; do
    test "$(find "${DUMP_DIR}" -maxdepth 1 -type f -name "reconstruct_rank*_pid*_call0000_${stage}.bin" | wc -l)" -eq 2
done
test "$(find "${DUMP_DIR}" -maxdepth 1 -type f -name 'reconstruct_rank*_pid*_call0000_metadata.txt' | wc -l)" -eq 2
test "$(find "${DUMP_DIR}" -maxdepth 1 -type f | wc -l)" -eq 16
"${PYTHON}" - <<'PY' "${DUMP_DIR}" "${ANALYSIS}/RECONSTRUCT_CAPTURE_VALIDATION.json"
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
    --output-json "${ANALYSIS}/RELION_CAPTURE_INERTNESS.json" \
    --output-markdown "${ANALYSIS}/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${ANALYSIS}/RELION_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY
find "${DUMP_DIR}" "${RELION_OUT}" "${ANALYSIS}" -type f -print0 | sort -z | xargs -0 sha256sum > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"reconstruct_call":0}\n' "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
