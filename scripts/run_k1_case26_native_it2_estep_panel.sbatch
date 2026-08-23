#!/usr/bin/env bash
#SBATCH --job-name=k1c26-native-i2
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_estep_20260810T2034ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_estep_20260810T2034ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=00:20:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_estep_20260810T2034ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_native_it2_estep_20260810T2034ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case26_native_it2_estep_panel.sbatch
BINARY=/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
OUTPUT=${ROOT}/relion
ESTEP=${ROOT}/capture/estep
ACC=${ROOT}/capture/acc
PART_IDS=157,634,231,402,443

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = 471d305ec5a7a238a62b0f3a3aac22a2cbd0502a017a7cdb768e8ce5db006b45
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = 187a9f60cdea19363d49d46099b052f23288f05a
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
test -s "${DATA}/particles.star" -a -s "${DATA}/particles.128.mrcs"
test -s "${DATA}/reference_init_relion.mrc"
test -z "$(find "${OUTPUT}" "${ESTEP}" "${ACC}" -mindepth 1 -print -quit)"

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
export LD_LIBRARY_PATH=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

# Native, pre-STAR E-step capture.  The five part_ids map by rlnImageName to
# source indices 23, 66, 305, 459, and 497; all are random-subset 2.
export RECOVAR_DEBUG_ESTEP_DIR=${ESTEP}
export RECOVAR_DEBUG_ESTEP_PARTIDS=${PART_IDS}
export RECOVAR_DEBUG_ESTEP_ITER=2
export RELION_ACC_DUMP_DIR=${ACC}
export RELION_ACC_DUMP_PART_IDS=${PART_IDS}
export RELION_ACC_DUMP_ITER=2
export RELION_ACC_DUMP_VERBOSE=1
export RELION_ACC_DUMP_PASS1_DIFF2=1

COMMAND=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${OUTPUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1726 --auto_iter_max 2
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
sha256sum "${LAUNCHER}" "${BINARY}" "${DATA}/particles.star" \
    "${DATA}/particles.128.mrcs" "${DATA}/reference_init_relion.mrc" \
    "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.csv"

start=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${OUTPUT}/runner.stdout" 2> "${OUTPUT}/runner.stderr"
wall_s=$(( $(date +%s) - start ))

test -s "${OUTPUT}/run_it002_data.star"
for part_id in 157 634 231 402 443; do
    test -s "${ESTEP}/p${part_id}_pass0_diff2_raw.bin"
    test -s "${ESTEP}/p${part_id}_pass1_diff2_raw.bin"
    test -s "${ESTEP}/p${part_id}_pass1_exp_Mweight_postconvert.bin"
    test -s "${ESTEP}/p${part_id}_estep_meta.txt"
    test -s "${ESTEP}/p${part_id}_oversampled_rotations.bin"
    test -s "${ESTEP}/p${part_id}_oversampled_translations.bin"
    test -n "$(find "${ACC}" -maxdepth 1 -type f -name "*part${part_id}_pass1_class0_pass1_diff2_weights.bin" -print -quit)"
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

find "${ESTEP}" "${ACC}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/capture_artifacts_${SLURM_JOB_ID}.sha256"
sha256sum "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":2,"part_ids":[157,634,231,402,443]}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
