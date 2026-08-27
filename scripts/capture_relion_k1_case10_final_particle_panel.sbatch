#!/usr/bin/env bash
#SBATCH --job-name=k1-final-particle-panel
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
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
: "${RELION_BINARY:?RELION_BINARY is required}"
: "${EXPECTED_BINARY_SHA256:?EXPECTED_BINARY_SHA256 is required}"
: "${EXPECTED_INPUTS_SHA256:?EXPECTED_INPUTS_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

BINARY=${RELION_BINARY}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
LAUNCHER=${REPO}/scripts/capture_relion_k1_case10_final_particle_panel.sbatch
FIXED_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE_ROOT=${FIXED_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE_ROOT}/data
CONTROL=${CASE_ROOT}/relion_ref
FORCED_CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case10_forced_final_boundary_20260827T1650ET/relion
CONTINUE=${CONTROL}/run_it015_optimiser.star
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
OUTPUT=${RUN_ROOT}/relion
DUMP=${RUN_ROOT}/particle_bpref
ANALYSIS=${RUN_ROOT}/analysis
PROVENANCE=${RUN_ROOT}/provenance
PART_IDS=${K1_FINAL_PANEL_PART_IDS:-0,33334}
EXPECTED_PARTICLES=${K1_FINAL_PANEL_EXPECTED_PARTICLES:-2}

input_digest() {
    sha256sum \
        "${CONTROL}/run_it015_optimiser.star" \
        "${CONTROL}/run_it015_data.star" \
        "${CONTROL}/run_it015_half1_model.star" \
        "${CONTROL}/run_it015_half2_model.star" \
        "${CONTROL}/run_it015_sampling.star" \
        "${CONTROL}/run_it015_half1_class001.mrc" \
        "${CONTROL}/run_it015_half2_class001.mrc" \
        "${CONTROL}/run_class001.mrc" \
        | sha256sum | awk '{print $1}'
}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${DATA}/particles.384.mrcs"
test -s "${FORCED_CONTROL}/run_class001.mrc"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = "${EXPECTED_BINARY_SHA256}"
test "$(input_digest)" = "${EXPECTED_INPUTS_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(awk -F, '{print NF}' <<< "${PART_IDS}")" -eq "${EXPECTED_PARTICLES}"
test -z "$(
    find "${RUN_ROOT}" -mindepth 1 \
        ! -name SAFE_TO_DELETE \
        ! -name 'slurm-*.out' \
        -print -quit
)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${OUTPUT}" "${DUMP}" "${ANALYSIS}" "${PROVENANCE}" "${RUN_ROOT}/logs"

source /etc/profile.d/modules.sh
set +u
module purge
module load gcc/11 openmpi/cuda-12.6/gcc/4.1.6 cudatoolkit/12.6
set -u
MPI_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_mpi_dev_cuda126_openmpi416_gcc_20260827
CUDA_ROOT=/usr/local/cuda-12.6
export PATH=${MPI_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_ROOT}/lib64
test -r "${MPI_ROOT}/lib64/libmpi.so"
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign one visible H100}"

# Re-enter the sealed final pass without changing its original physical order.
export RELION_RESTORE_FRESH_PHYSICAL_ORDER=1
export RELION_BPRE_CAPTURE_DIR=${DUMP}
export RELION_BPRE_CAPTURE_SCHEMA=1
export RELION_BPRE_CAPTURE_ITER=16
export RELION_BPRE_CAPTURE_CLASS=1
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=${EXPECTED_PARTICLES}
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=${EXPECTED_PARTICLES}
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=64000000000
export RELION_BPRE_CAPTURE_DEVICE_BYTES=33554432
export RELION_BPRE_CAPTURE_PART_IDS=${PART_IDS}

env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
sha256sum "${BINARY}" "${LAUNCHER}" \
    > "${PROVENANCE}/static_${SLURM_JOB_ID}.sha256"
sha256sum \
    "${CONTROL}/run_it015_optimiser.star" \
    "${CONTROL}/run_it015_data.star" \
    "${CONTROL}/run_it015_half1_model.star" \
    "${CONTROL}/run_it015_half2_model.star" \
    "${CONTROL}/run_it015_sampling.star" \
    "${CONTROL}/run_it015_half1_class001.mrc" \
    "${CONTROL}/run_it015_half2_class001.mrc" \
    "${CONTROL}/run_class001.mrc" \
    > "${PROVENANCE}/inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
    --continue "${CONTINUE}" --o "${OUTPUT}/run" --force_converge
    --auto_iter_max 999 --pool 3 --gpu 0:0:0 --j 4)
printf 'cd %q\n' "${DATA}" > "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '%q ' "${command[@]}" >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
start_seconds=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
end_seconds=$(date +%s)
printf '{"elapsed_seconds":%d,"job_id":"%s","part_ids":"%s"}\n' \
    "$((end_seconds - start_seconds))" "${SLURM_JOB_ID}" "${PART_IDS}" \
    > "${PROVENANCE}/walltime_${SLURM_JOB_ID}.json"

test "$(find "${DUMP}" -maxdepth 1 -type f -name '*.bpre-v1.bin' | wc -l)" \
    -eq "${EXPECTED_PARTICLES}"
test "$(find "${DUMP}" -maxdepth 1 -type f -name '*_stack1_*.bpre-v1.bin' | wc -l)" -eq 1
test "$(find "${DUMP}" -maxdepth 1 -type f -name '*_stack4_*.bpre-v1.bin' | wc -l)" -eq 1
rg -q '\[RELION_RESTORE_FRESH_PHYSICAL_ORDER\] seed=1711 final_iter=16' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
test -s "${OUTPUT}/run_class001.mrc"
! rg -q 'ERROR|Error|Traceback|segmentation fault|Out of memory' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stdout" \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"

# The sealed map is the scientific order A/B.  Instrumentation inertness must
# instead use a continuation with the same forced-order intervention.
CONTROL_MAP=${FORCED_CONTROL}/run_class001.mrc \
CAPTURE_MAP=${OUTPUT}/run_class001.mrc \
OUTPUT_JSON=${ANALYSIS}/capture_inertness.json \
"${REPO}/.pixi/envs/default/bin/python" - <<'PY'
import json
import os
from pathlib import Path

from scripts.audit_em_k1_membership_capture_inertness import _load_map, _metrics

control = Path(os.environ["CONTROL_MAP"]).resolve()
capture = Path(os.environ["CAPTURE_MAP"]).resolve()
metrics = _metrics(_load_map(control), _load_map(capture))
threshold = 0.999999
metrics["passed"] = metrics["fsc_auc_non_dc"] >= threshold
report = {
    "schema": "em-k1-case10-final-particle-panel-inertness-v1",
    "metric_policy": "signed shellwise FSC and non-DC FSC-AUC; no correlation",
    "fsc_auc_non_dc_threshold": threshold,
    "comparison": metrics,
    "capture_inertness_qualified": metrics["passed"],
}
Path(os.environ["OUTPUT_JSON"]).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
if not metrics["passed"]:
    raise SystemExit("particle-panel instrumentation changed the final RELION map")
PY

find "${OUTPUT}" "${DUMP}" "${ANALYSIS}" -type f -print0 \
    | sort -z | xargs -0 sha256sum \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
