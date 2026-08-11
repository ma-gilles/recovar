#!/usr/bin/env bash
#SBATCH --job-name=k1c26-rfloat
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_corr_stages_rfloat_20260811T0159ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_corr_stages_rfloat_20260811T0159ET/logs/run-%j.err
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

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_corr_stages_rfloat_20260811T0159ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_native_it2_corr_stages_rfloat_20260811T0159ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case26_native_it2_corr_stages_rfloat.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_source_20260810
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_build_20260810/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
OUTPUT=${ROOT}/relion
DUMP=${ROOT}/corr_stages

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${RELION_SOURCE}" diff -- src/acc/acc_ml_optimiser_impl.h | sha256sum | awk '{print $1}')" = 5f027ae872fca578ef8b8f9667a634f9fd3a6b77f53c95a81a57731ac836d863
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 7ac817e5c35d736f20c5a517f7df0804eb36863086eaf810daa2bb1cd5e7c542
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 34016f7f367587125956de8f00895c27b14225511c8ef0f517179b890685e139
test -x "${PYTHON}" -a -x "${RELION_BINARY}"
test -z "$(find "${OUTPUT}" "${DUMP}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_ACC_DUMP_*|RELION_DUMP_*) unset "${variable_name}" ;; esac
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

export RELION_ACC_DUMP_DIR=${DUMP}
export RELION_ACC_DUMP_PART_IDS=634
export RELION_ACC_DUMP_ITER=2
export RELION_ACC_DUMP_CORR_STAGES=1

COMMAND=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${OUTPUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1726 --auto_iter_max 2
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
{
    echo "job_id=${SLURM_JOB_ID}"
    echo "physical_iteration=2"
    echo "part_id=634"
    sha256sum "${LAUNCHER}" "${RELION_BINARY}"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"

start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${OUTPUT}/runner.stdout" 2> "${OUTPUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test -s "${OUTPUT}/run_it002_data.star"
for name in minvsigma2 fctf fctf_squared fctf_rfloat fctf_squared_rfloat \
    after_ctf after_ctf_source_semantics after_scale corr_img scale scale_squared \
    do_ctf_correction refs_are_ctf_corrected do_scale_correction; do
    test -s "${DUMP}/img0_part634_fineCorr_${name}.bin"
done
! grep -Eiq 'Traceback|Out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${OUTPUT}/runner.stderr"

"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    --control-dir "${CONTROL}" --candidate-dir "${OUTPUT}" --iterations 1 2 \
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

find "${DUMP}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/corr_stages_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":2,"part_id":634}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
