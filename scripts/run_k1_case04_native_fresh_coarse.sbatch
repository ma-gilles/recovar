#!/usr/bin/env bash
#SBATCH --job-name=k1c04-fresh-coarse
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

: "${K1_FRESH_COARSE_ROOT:?set disposable run root}"
: "${K1_FRESH_COARSE_RUNTIME:?set disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin launcher}"

ROOT=${K1_FRESH_COARSE_ROOT}
RUNTIME=${K1_FRESH_COARSE_RUNTIME}
BUILD_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case07_live_coarse_operands_20260815T0358ET
SOURCE=${BUILD_ROOT}/relion_source
BINARY=${BUILD_ROOT}/build_login/bin/relion_refine_mpi
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
LAUNCHER=${REPO}/scripts/run_k1_case04_native_fresh_coarse.sbatch

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = f4541486d6c11d5872e97b29954e9fd56d66217b
test "$(git -C "${SOURCE}" rev-parse HEAD)" = 8ba3e0a86fef0ad16d83206be4a219fe74cf28df
test "$(git -C "${SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = 3cf6026238fbfb0d3d8442f5546a0d041eccbed7e1cea1431d96accbc997205f
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = c1620145f7c3529d701303d0354b62ac17611bf812f6273192e4a45a7a8704d7
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test -z "$(find "${ROOT}/capture" "${ROOT}/relion" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
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

export RELION_P1_CAPTURE_DIR=${ROOT}/capture
export RELION_P1_CAPTURE_SCHEMA=3
export RELION_P1_CAPTURE_ITER=2
export RELION_P1_CAPTURE_EXPECTED_PARTICLES=1
export RELION_P1_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_P1_CAPTURE_EXPECTED_FOLLOWERS=1
export RELION_P1_CAPTURE_MAX_BYTES=1000000000
export RELION_P1_CAPTURE_PART_IDS=80015
export RELION_P1_CAPTURE_OPERAND_ROTATION_KEYS=35343
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${SOURCE}" diff HEAD --binary > "${ROOT}/provenance/relion_source.patch"
sha256sum "${BINARY}" "${LAUNCHER}" "${DATA}/particles.star" \
    "${DATA}/reference_init_relion.mrc" > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${ROOT}/relion/run"
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
) > "${ROOT}/relion/runner.stdout" 2> "${ROOT}/relion/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

for suffix in p1-v2 p1-op-v2 p1-lane-v1 p1-live-v1; do
    test -s "${ROOT}/capture/part80015_stack82010.${suffix}.bin"
done
test -z "$(find "${ROOT}/capture" -maxdepth 1 -type f -name '*.tmp.*' -print -quit)"

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_coarse_live_operands \
    --live "${ROOT}/capture/part80015_stack82010.p1-live-v1.bin" \
    --operands "${ROOT}/capture/part80015_stack82010.p1-op-v2.bin" \
    --lanes "${ROOT}/capture/part80015_stack82010.p1-lane-v1.bin" \
    --components "${ROOT}/capture/part80015_stack82010.p1-v2.bin" \
    --output-json "${ROOT}/analysis/K1_FRESH_COARSE_LIVE_OPERANDS.json" \
    > "${ROOT}/logs/live_operand_analysis.stdout"

cd "${AUDIT_REPO}"
"${PYTHON}" -m scripts.audit_relion_control_reproducibility \
    --control-dir "${CONTROL}" --candidate-dir "${ROOT}/relion" --iterations 1 2 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_FRESH_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_FRESH_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${ROOT}/analysis/RELION_FRESH_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

find "${ROOT}/capture" "${ROOT}/relion" "${ROOT}/analysis" -type f -print0 \
    | sort -z | xargs -0 sha256sum > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"iterations":[1,2],"capture":"fresh_uninterrupted_coarse_operands"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
