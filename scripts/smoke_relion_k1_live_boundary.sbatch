#!/usr/bin/env bash
#SBATCH --job-name=k1-live-boundary-smoke
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${RELION_BINARY:?RELION_BINARY is required}"
: "${EXPECTED_BINARY_SHA256:?EXPECTED_BINARY_SHA256 is required}"
: "${EXPECTED_INPUTS_SHA256:?EXPECTED_INPUTS_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_VALIDATOR_SHA256:?EXPECTED_VALIDATOR_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_pr167_postintegration_debug_20260827
LAUNCHER=${REPO}/scripts/smoke_relion_k1_live_boundary.sbatch
VALIDATOR=${REPO}/scripts/validate_relion_live_metadata.py
PYTHON=${REPO}/.pixi/envs/default/bin/python
BINARY=${RELION_BINARY}
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_guigrid_localhighshell_full34_autonomous_ac5177d2_20260719T174000Z/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80
DATA=${CASE}/data
MPI_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_mpi_dev_cuda126_openmpi416_gcc_20260827
MPIRUN=/usr/local/openmpi/cuda-12.6/4.1.6/gcc/bin/mpirun
CUDA_OVERLAY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/system_dev_overlay
AUDIT_SCRIPT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout/scripts/audit_relion_control_reproducibility.py

input_digest() {
    sha256sum "${DATA}/particles.star" "${DATA}/particles.128.mrcs" \
        "${DATA}/reference_init_relion.mrc" | sha256sum | awk '{print $1}'
}

test -f "${RUN_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${BINARY}"
test -x "${PYTHON}"
test -f "${AUDIT_SCRIPT}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = "${EXPECTED_BINARY_SHA256}"
test "$(input_digest)" = "${EXPECTED_INPUTS_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${VALIDATOR}" | awk '{print $1}')" = "${EXPECTED_VALIDATOR_SHA256}"
test -z "$(find "${RUN_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE ! -name 'slurm-*.out' -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${RUN_ROOT}/control" "${RUN_ROOT}/capture" "${RUN_ROOT}/live" \
    "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"

source /etc/profile.d/modules.sh
set +u
module purge
module load gcc/11 cudatoolkit/12.6
set -u
CUDA_ROOT=/usr/local/cuda-12.6
export PATH=${MPI_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_OVERLAY}/lib:${MPI_ROOT}/lib64:${CUDA_ROOT}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign one visible H100}"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
rg -q H100 "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
test -x "${MPIRUN}"

run_arm() {
    local arm=$1
    local output=${RUN_ROOT}/${arm}
    local command=("${MPIRUN}" -n 3 --bind-to none "${BINARY}"
        --i particles.star --ref reference_init_relion.mrc --o "${output}/run"
        --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
        --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
        --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
        --oversampling 1 --pad 2 --random_seed 1722 --auto_iter_max 1 --pool 3
        --gpu 0:0:0 --j 4)
    printf 'cd %q\n' "${DATA}" > "${RUN_ROOT}/provenance/command_${arm}_${SLURM_JOB_ID}.sh"
    printf '%q ' "${command[@]}" >> "${RUN_ROOT}/provenance/command_${arm}_${SLURM_JOB_ID}.sh"
    printf '\n' >> "${RUN_ROOT}/provenance/command_${arm}_${SLURM_JOB_ID}.sh"
    local started
    started=$(date +%s)
    (
        cd "${DATA}"
        "${command[@]}"
    ) > "${RUN_ROOT}/logs/${arm}_${SLURM_JOB_ID}.stdout" \
        2> "${RUN_ROOT}/logs/${arm}_${SLURM_JOB_ID}.stderr"
    printf '{"arm":"%s","wall_s":%d}\n' "${arm}" "$(( $(date +%s) - started ))" \
        > "${RUN_ROOT}/provenance/walltime_${arm}_${SLURM_JOB_ID}.json"
    test -s "${output}/run_it001_data.star"
    test -s "${output}/run_it001_half1_class001.mrc"
    test -s "${output}/run_it001_half2_class001.mrc"
}

run_arm control
export RELION_LIVE_METADATA_DIR=${RUN_ROOT}/live
export RELION_LIVE_METADATA_ITER=1
run_arm capture
unset RELION_LIVE_METADATA_DIR RELION_LIVE_METADATA_ITER

test -s "${RUN_ROOT}/live/live_metadata_it001_input.rlm-v1.bin"
test -s "${RUN_ROOT}/live/live_metadata_it001_output.rlm-v1.bin"
test "$(find "${RUN_ROOT}/live" -maxdepth 1 -name 'live_model_it001_rank*_state.txt' | wc -l)" -eq 2
"${PYTHON}" "${VALIDATOR}" \
    --input "${RUN_ROOT}/live/live_metadata_it001_input.rlm-v1.bin" \
    --output "${RUN_ROOT}/live/live_metadata_it001_output.rlm-v1.bin" \
    --expected-particles 3000 \
    --report-json "${RUN_ROOT}/analysis/live_metadata_validation.json" \
    --normalized-npz "${RUN_ROOT}/analysis/live_metadata_normalized.npz" \
    > "${RUN_ROOT}/logs/validate_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/validate_${SLURM_JOB_ID}.stderr"

"${PYTHON}" "${AUDIT_SCRIPT}" \
    --control-dir "${RUN_ROOT}/control" \
    --candidate-dir "${RUN_ROOT}/capture" \
    --iterations 1 --map-fsc-auc-threshold 0.999999 \
    --output-json "${RUN_ROOT}/analysis/capture_inertness.json" \
    --output-markdown "${RUN_ROOT}/analysis/capture_inertness.md" \
    > "${RUN_ROOT}/logs/inertness_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/inertness_${SLURM_JOB_ID}.stderr"

rg -q '\[RELION_LIVE_METADATA\] begin iteration=1 particles=3000 metadata_width=31' \
    "${RUN_ROOT}/logs/capture_${SLURM_JOB_ID}.stderr"
rg -q '\[RELION_LIVE_METADATA\] sealed iteration=1 particles=3000' \
    "${RUN_ROOT}/logs/capture_${SLURM_JOB_ID}.stderr"
! rg -q 'ERROR|Error|segmentation fault|Out of memory' \
    "${RUN_ROOT}/logs/control_${SLURM_JOB_ID}.stdout" \
    "${RUN_ROOT}/logs/control_${SLURM_JOB_ID}.stderr" \
    "${RUN_ROOT}/logs/capture_${SLURM_JOB_ID}.stdout" \
    "${RUN_ROOT}/logs/capture_${SLURM_JOB_ID}.stderr"

env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${BINARY}" "${LAUNCHER}" "${VALIDATOR}" \
    > "${RUN_ROOT}/provenance/static_${SLURM_JOB_ID}.sha256"
find "${RUN_ROOT}/control" "${RUN_ROOT}/capture" "${RUN_ROOT}/live" \
    "${RUN_ROOT}/analysis" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
