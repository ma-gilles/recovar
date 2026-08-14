#!/usr/bin/env bash
#SBATCH --job-name=k1c22-native-it3-recon
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RECOVAR_K1_NATIVE_IT3_RECON_ROOT:?set the empty run root}"
: "${RECOVAR_K1_NATIVE_IT3_RECON_RUNTIME_ROOT:?set the empty runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin the launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin the repository head}"

ROOT=${RECOVAR_K1_NATIVE_IT3_RECON_ROOT}
RUNTIME=${RECOVAR_K1_NATIVE_IT3_RECON_RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case22_native_it3_reconstruct_stage.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_source_20260810
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_build_20260810/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
RELION_OUT=${ROOT}/relion
DUMP_DIR=${ROOT}/reconstruct

require_sha() {
    test "$(sha256sum "$2" | awk '{print $1}')" = "$1"
}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
require_sha "${EXPECTED_LAUNCHER_SHA256}" "${LAUNCHER}"
require_sha 235052e48ccf843df48392c8e2b4f0d634b7dcedf7be27b88eb1cff746c07201 "${RELION_BINARY}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${RELION_SOURCE}" diff -- src/acc/acc_ml_optimiser_impl.h src/acc/cuda/cuda_ml_optimiser.cu src/backprojector.cpp | sha256sum | awk '{print $1}')" = 58c5751a1ba01b5a9e28d3dde357587676405aa7f020cabdb0395dc7d362df61
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(find "${DUMP_DIR}" "${RELION_OUT}" -mindepth 1 -print -quit)"

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
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0 CUDA_VISIBLE_DEVICES=0
export RELION_RECON_DUMP_DIR=${DUMP_DIR}
export RELION_RECON_DUMP_CALL=2

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "gpu_name=${GPU_NAME}"
    echo "repo_head=${EXPECTED_REPO_HEAD}"
    echo "relion_source_head=$(git -C "${RELION_SOURCE}" rev-parse HEAD)"
    echo "reconstruct_dump_call=2"
    echo "physical_iteration=3"
    echo "fresh_uninterrupted_auto_refine=true"
    echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset"
    echo "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset"
    sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${DATA}/particles.star" \
        "${DATA}/reference_init_relion.mrc"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"
git -C "${RELION_SOURCE}" diff -- src/acc/acc_ml_optimiser_impl.h src/acc/cuda/cuda_ml_optimiser.cu src/backprojector.cpp \
    > "${ROOT}/provenance/relion_reconstruct_capture.patch"

COMMAND=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1722 --auto_iter_max 3 --pool 3
    --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test -s "${RELION_OUT}/run_it003_data.star"
for stage in tau2 fweight_decentered fweight_regularized radavg_weight \
    fconv_divided volume_before_gridding volume_after_gridding; do
    test "$(find "${DUMP_DIR}" -maxdepth 1 -type f -name "reconstruct_rank*_pid*_call0002_${stage}.bin" | wc -l)" -eq 2
done
test "$(find "${DUMP_DIR}" -maxdepth 1 -type f -name 'reconstruct_rank*_pid*_call0002_metadata.txt' | wc -l)" -eq 2
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
    shape = header[:3]
    assert shape[0] * shape[1] * shape[2] > 0
    records.append({
        "name": path.name,
        "shape_zyx": list(shape),
        "payload_bytes": len(raw) - 64,
        "sha256": hashlib.sha256(raw).hexdigest(),
    })
assert len(records) == 14
Path(sys.argv[2]).write_text(json.dumps({"schema": "relion.reconstruct.stages.v1", "records": records}, indent=2) + "\n")
PY
sha256sum "${DUMP_DIR}"/* "${ROOT}/analysis/RECONSTRUCT_CAPTURE_VALIDATION.json" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":3,"dump_call":2,"fresh_uninterrupted_auto_refine":true}\n' "${SLURM_JOB_ID}" "${wall_s}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
