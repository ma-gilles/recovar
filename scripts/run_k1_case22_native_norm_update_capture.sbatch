#!/usr/bin/env bash
#SBATCH --job-name=k1-c22-norm-update
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:20:00

set -euo pipefail

: "${K1_NATIVE_NORM_ROOT:?set the disposable capture root}"
: "${K1_NATIVE_NORM_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_RELION_HEAD:?pin the RELION source head}"
: "${EXPECTED_RELION_DIFF_SHA256:?pin the RELION source diff}"
: "${EXPECTED_RELION_FILE_SHA256:?pin the instrumented source file}"
: "${EXPECTED_RELION_BINARY_SHA256:?pin the RELION binary}"

ROOT=${K1_NATIVE_NORM_ROOT}
RUNTIME=${K1_NATIVE_NORM_RUNTIME}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_components_source_b571275_20260812/source
BUILD=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_components_source_b571275_20260812/build
BIN=${BUILD}/bin/relion_refine_mpi
LAUNCHER=${REPO}/scripts/run_k1_case22_native_norm_update_capture.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion
RELION_OUT=${ROOT}/relion
CAPTURE_DIR=${ROOT}/capture

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_HEAD}"
test "$(git -C "${SOURCE}" diff --no-ext-diff --binary HEAD -- src/acc/acc_ml_optimiser_impl.h | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_DIFF_SHA256}"
test "$(sha256sum "${SOURCE}/src/acc/acc_ml_optimiser_impl.h" | awk '{print $1}')" = "${EXPECTED_RELION_FILE_SHA256}"
test "$(sha256sum "${BIN}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test -z "$(find "${CAPTURE_DIR}" "${RELION_OUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${ROOT}/analysis" "${ROOT}/logs" "${ROOT}/provenance"

export PATH=/usr/local/openmpi/cuda-12.6/4.1.6/gcc/bin:/usr/local/cuda-12.6/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/openmpi/cuda-12.6/4.1.6/gcc/lib64:/usr/local/cuda-12.6/lib64:${SOURCE}/external/fftw/lib:${LD_LIBRARY_PATH:-}
if ldd "${BIN}" | grep -q 'not found'; then
    ldd "${BIN}" >&2
    exit 2
fi

# Selecting the iteration-2 preprocess capture also activates passive update
# operand logging for this particle in iterations 1 and 2.
export RELION_P1_PREPROCESS_CAPTURE_DIR=${CAPTURE_DIR}
export RELION_P1_PREPROCESS_CAPTURE_SCHEMA=1
export RELION_P1_PREPROCESS_CAPTURE_ITER=2
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_PARTICLES=1
export RELION_P1_PREPROCESS_CAPTURE_MAX_PARTICLES_PER_RANK=1
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_P1_PREPROCESS_CAPTURE_MAX_BYTES=1073741824
export RELION_P1_PREPROCESS_CAPTURE_PART_IDS=515

nvidia-smi --query-gpu=timestamp,index,name,uuid,memory.total,driver_version --format=csv \
    > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${BIN}" "${SOURCE}/src/acc/acc_ml_optimiser_impl.h" \
    "${BUILD}/CMakeCache.txt" "${DATA}/particles.star" \
    "${DATA}/particles.128.mrcs" "${DATA}/reference_init_relion.mrc" \
    > "${ROOT}/provenance/inputs_${SLURM_JOB_ID}.sha256"

cd "${DATA}"
mpirun -n 3 "${BIN}" \
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run" \
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30 \
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40 \
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0 \
    --oversampling 1 --pad 2 --random_seed 1722 --auto_iter_max 2 \
    --pool 3 --gpu 0:0:0 --j 4 \
    > "${RELION_OUT}/runner_${SLURM_JOB_ID}.stdout" \
    2> "${RELION_OUT}/runner_${SLURM_JOB_ID}.stderr"

test -s "${CAPTURE_DIR}/part515_stack1462.preprocess-v1.bin"
test -s "${RELION_OUT}/run_it002_optimiser.star"
test "$(rg -c 'RELION_P1_NORMALIZATION_OPERANDS_V1 part_id=515 ' "${RELION_OUT}/runner_${SLURM_JOB_ID}.stderr")" -eq 1
test "$(rg -c 'RELION_P1_NORM_UPDATE_OPERANDS_V1 iter=1 part_id=515 ' "${RELION_OUT}/runner_${SLURM_JOB_ID}.stderr")" -eq 1

cd "${AUDIT_REPO}"
"${PYTHON}" -m scripts.audit_relion_control_reproducibility \
    --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" --iterations 1 2 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_NORM_UPDATE_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_NORM_UPDATE_CAPTURE_INERTNESS.md" \
    > "${ROOT}/logs/inertness_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/inertness_${SLURM_JOB_ID}.stderr"

"${PYTHON}" - "${ROOT}/analysis/RELION_NORM_UPDATE_CAPTURE_INERTNESS.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert len(report["iterations"]) == 2
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

find "${CAPTURE_DIR}" "${RELION_OUT}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
