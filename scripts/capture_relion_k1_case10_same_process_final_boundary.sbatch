#!/usr/bin/env bash
#SBATCH --job-name=k1c10-live-final
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=10:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${RELION_BINARY:?RELION_BINARY is required}"
: "${EXPECTED_BINARY_SHA256:?EXPECTED_BINARY_SHA256 is required}"
: "${EXPECTED_INPUTS_SHA256:?EXPECTED_INPUTS_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_VALIDATOR_SHA256:?EXPECTED_VALIDATOR_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_pr167_postintegration_debug_20260827
LAUNCHER=${REPO}/scripts/capture_relion_k1_case10_same_process_final_boundary.sbatch
LIVE_VALIDATOR=${REPO}/scripts/validate_relion_live_metadata.py
FACTOR_VALIDATOR=${REPO}/scripts/validate_relion_bpref_factor_capture.py
SCORE_VALIDATOR=${REPO}/scripts/validate_relion_fine_score_capture.py
PYTHON=${REPO}/.pixi/envs/default/bin/python
BINARY=${RELION_BINARY}
CASE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE_ROOT}/data
OUTPUT=${RUN_ROOT}/relion
LIVE=${RUN_ROOT}/live_boundary
FACTORS=${RUN_ROOT}/particle_factors
BPREF=${RUN_ROOT}/bpref
ANALYSIS=${RUN_ROOT}/analysis
PROVENANCE=${RUN_ROOT}/provenance
MPI_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_mpi_dev_cuda126_openmpi416_gcc_20260827
MPIRUN=/usr/local/openmpi/cuda-12.6/4.1.6/gcc/bin/mpirun
CUDA_OVERLAY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/system_dev_overlay
FINAL_ITERATION=16
STACKS=60349,4707,17854,31108,93504,22158,78907,51591,73868,70322,39731,84610,65141,47676,10234,29834
EXPECTED_PANEL_PARTICLES=16

input_digest() {
    sha256sum "${DATA}/particles.star" "${DATA}/particles.384.mrcs" \
        "${DATA}/reference_init_relion.mrc" | sha256sum | awk '{print $1}'
}

test -f "${RUN_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${BINARY}"
test -x "${PYTHON}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = "${EXPECTED_BINARY_SHA256}"
test "$(input_digest)" = "${EXPECTED_INPUTS_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${LIVE_VALIDATOR}" | awk '{print $1}')" = "${EXPECTED_VALIDATOR_SHA256}"
for marker in RELION_LIVE_METADATA_DIR RELION_BPRE_CAPTURE_SCHEMA RELION_DUMP_BPREF pre_reconstruct; do
    marker_count=$(strings "${BINARY}" | rg -c "${marker}" || true)
    test "${marker_count:-0}" -ge 1
done
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
    "${OUTPUT}" "${LIVE}" "${FACTORS}" "${BPREF}" "${ANALYSIS}" \
    "${PROVENANCE}" "${RUN_ROOT}/logs"

"${PYTHON}" - "${PROVENANCE}/SELECTION.json" <<'PY'
import json
import sys

half1 = [60349, 4707, 17854, 31108, 93504, 22158, 78907, 51591]
half2 = [73868, 70322, 39731, 84610, 65141, 47676, 10234, 29834]
targets = [
    {
        "stack_index_one_based": stack,
        "expected_mpi_rank": half,
        "role": "fixed_two_half_extreme_pmax_or_translation_residual_panel",
    }
    for half, stacks in ((1, half1), (2, half2))
    for stack in stacks
]
report = {
    "schema": "recovar.em.k1_same_process_final_boundary_panel.v1",
    "case_id": 10,
    "physical_iteration": 16,
    "selection_policy": (
        "fixed 16-particle panel declared before execution: four largest absolute "
        "Pmax residuals and four largest translation residual norms per half in "
        "the sealed RECOVAR-versus-RELION case-10 final boundary"
    ),
    "selection_source_relion": (
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/"
        "10_high_res_anisotropic_100k_g384_radial_noise3_bf0/relion_ref/run_data.star"
    ),
    "selection_source_recovar": (
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/"
        "10_high_res_anisotropic_100k_g384_radial_noise3_bf0/recovar/refinement_results.npz"
    ),
    "capture_stack_indices_one_based": [row["stack_index_one_based"] for row in targets],
    "targets": targets,
}
with open(sys.argv[1], "w") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

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
    > "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
rg -q H100 "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
test -x "${MPIRUN}"

export RELION_LIVE_METADATA_DIR=${LIVE}
export RELION_LIVE_METADATA_ITER=${FINAL_ITERATION}
export RELION_BPRE_CAPTURE_DIR=${FACTORS}
export RELION_BPRE_CAPTURE_SCHEMA=2
export RELION_BPRE_CAPTURE_ITER=${FINAL_ITERATION}
export RELION_BPRE_CAPTURE_CLASS=1
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=${EXPECTED_PANEL_PARTICLES}
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=${EXPECTED_PANEL_PARTICLES}
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=64000000000
export RELION_BPRE_CAPTURE_STACKS=${STACKS}
export RELION_DUMP_DIR=${BPREF}
export RELION_DUMP_BPREF=1
export RELION_DUMP_ITER=${FINAL_ITERATION}

env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
sha256sum "${BINARY}" "${LAUNCHER}" "${LIVE_VALIDATOR}" \
    "${FACTOR_VALIDATOR}" "${SCORE_VALIDATOR}" \
    > "${PROVENANCE}/static_${SLURM_JOB_ID}.sha256"
sha256sum "${DATA}/particles.star" "${DATA}/particles.384.mrcs" \
    "${DATA}/reference_init_relion.mrc" \
    > "${PROVENANCE}/inputs_${SLURM_JOB_ID}.sha256"

command=("${MPIRUN}" -n 3 --bind-to none "${BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${OUTPUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1710 --auto_iter_max 999 --pool 3
    --gpu 0:0:0 --j 4)
printf 'cd %q\n' "${DATA}" > "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '%q ' "${command[@]}" >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${PROVENANCE}/command_${SLURM_JOB_ID}.sh"
started=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
printf '{"elapsed_seconds":%d,"job_id":"%s"}\n' \
    "$(( $(date +%s) - started ))" "${SLURM_JOB_ID}" \
    > "${PROVENANCE}/walltime_${SLURM_JOB_ID}.json"

LIVE_INPUT=${LIVE}/live_metadata_it016_input.rlm-v1.bin
LIVE_OUTPUT=${LIVE}/live_metadata_it016_output.rlm-v1.bin
test -s "${LIVE_INPUT}"
test -s "${LIVE_OUTPUT}"
test "$(find "${LIVE}" -maxdepth 1 -name 'live_model_it016_rank*_state.txt' | wc -l)" -eq 2
"${PYTHON}" "${LIVE_VALIDATOR}" \
    --input "${LIVE_INPUT}" --output "${LIVE_OUTPUT}" --expected-particles 100000 \
    --report-json "${ANALYSIS}/live_metadata_validation.json" \
    --normalized-npz "${ANALYSIS}/live_metadata_normalized.npz" \
    > "${RUN_ROOT}/logs/validate_live_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/validate_live_${SLURM_JOB_ID}.stderr"

test "$(find "${FACTORS}" -maxdepth 1 -name '*.bpre-v2.bin' | wc -l)" \
    -eq "${EXPECTED_PANEL_PARTICLES}"
test "$(find "${FACTORS}" -maxdepth 1 -name '*.fine-score-v1.bin' | wc -l)" \
    -eq "${EXPECTED_PANEL_PARTICLES}"
"${PYTHON}" "${FACTOR_VALIDATOR}" "${FACTORS}" \
    --selection-json "${PROVENANCE}/SELECTION.json" \
    --output-json "${ANALYSIS}/factor_validation.json" \
    > "${RUN_ROOT}/logs/validate_factors_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/validate_factors_${SLURM_JOB_ID}.stderr"
"${PYTHON}" "${SCORE_VALIDATOR}" --capture-dir "${FACTORS}" \
    --selection-json "${PROVENANCE}/SELECTION.json" \
    --output "${ANALYSIS}/fine_score_validation.json" \
    > "${RUN_ROOT}/logs/validate_scores_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/validate_scores_${SLURM_JOB_ID}.stderr"

PREFIX=${BPREF}/mstep_it016_rank1_half0_c0_pre_reconstruct
for suffix in bpref_data bpref_weight tau2 sigma2 data_vs_prior fsc fourier_coverage; do
    test -s "${PREFIX}_${suffix}.bin"
done
test -s "${PREFIX}_bpref_meta.txt"
test -s "${OUTPUT}/run_class001.mrc"
test -s "${OUTPUT}/run_half1_class001_unfil.mrc"
test -s "${OUTPUT}/run_half2_class001_unfil.mrc"
rg -q '\[RELION_LIVE_METADATA\] begin iteration=16 particles=100000 metadata_width=31' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
rg -q '\[RELION_LIVE_METADATA\] sealed iteration=16 particles=100000' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
rg -q '\[RELION_DUMP_BPREF\] wrote .*pre_reconstruct' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"
! rg -q 'ERROR|Error|Traceback|segmentation fault|Out of memory' \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stdout" \
    "${RUN_ROOT}/logs/run_${SLURM_JOB_ID}.stderr"

find "${OUTPUT}" "${LIVE}" "${FACTORS}" "${BPREF}" "${ANALYSIS}" \
    -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
