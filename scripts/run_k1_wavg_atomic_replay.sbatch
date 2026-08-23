#!/usr/bin/env bash
#SBATCH --job-name=k1-wavg-atomic
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail
: "${K1_WAVG_ATOMIC_ROOT:?set the disposable run root}"
: "${K1_WAVG_ATOMIC_RUNTIME:?set the disposable runtime root}"
: "${K1_WAVG_ATOMIC_CUDA_LIB:?set the pinned CUDA library}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher}"
: "${EXPECTED_SCRIPT_SHA256:?pin the replay script}"
: "${EXPECTED_CUDA_SHA256:?pin the CUDA library}"

ROOT=${K1_WAVG_ATOMIC_ROOT}
RUNTIME=${K1_WAVG_ATOMIC_RUNTIME}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
LAUNCHER=${REPO}/scripts/run_k1_wavg_atomic_replay.sbatch
SCRIPT=${REPO}/scripts/replay_k1_wavg_atomic.py
PYTHON=${REPO}/.pixi/envs/default/bin/python
NATIVE_DIR=${K1_WAVG_ATOMIC_NATIVE_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_project_panel_it2_20260812T2030ET/capture}
NATIVE_PIXELS=${K1_WAVG_ATOMIC_NATIVE_PIXELS:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_scale_aa_pixels_it2_20260812T1924ET/capture/scale_aa_pixels.tsv}
RECOVAR_CAPTURE=${K1_WAVG_ATOMIC_RECOVAR_CAPTURE:-}
NATIVE_PREFIX=${K1_WAVG_ATOMIC_NATIVE_PREFIX:-img0_part109_storeWavg_}
TARGET_ITERATION=${K1_WAVG_ATOMIC_ITERATION:-2}
TARGET_HALF=${K1_WAVG_ATOMIC_HALF:-1}
TARGET_PART_ID=${K1_WAVG_ATOMIC_PART_ID:-109}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "${EXPECTED_SCRIPT_SHA256}"
test "$(sha256sum "${K1_WAVG_ATOMIC_CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -s "${NATIVE_PIXELS}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export RECOVAR_CUDA_LIB=${K1_WAVG_ATOMIC_CUDA_LIB}
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${ROOT}/analysis" "${ROOT}/provenance"

nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
"${PYTHON}" - <<'PY' > "${ROOT}/provenance/imports_${SLURM_JOB_ID}.txt"
import jax, recovar
print(recovar.__file__)
print(jax.__file__)
print(jax.devices())
PY

COMMAND=(
    "${PYTHON}" -m scripts.replay_k1_wavg_atomic
    --native-directory "${NATIVE_DIR}"
    --native-pixels "${NATIVE_PIXELS}"
    --native-prefix "${NATIVE_PREFIX}"
    --iteration "${TARGET_ITERATION}"
    --half "${TARGET_HALF}"
    --part-id "${TARGET_PART_ID}"
    --output-json "${ROOT}/analysis/wavg_atomic_replay.json"
)
if [[ -n "${RECOVAR_CAPTURE}" ]]; then
    test -s "${RECOVAR_CAPTURE}"
    COMMAND+=(--recovar-capture "${RECOVAR_CAPTURE}")
fi

start=$(date +%s)
"${COMMAND[@]}" \
    > "${ROOT}/analysis/wavg_atomic_replay.stdout"
printf '{"job_id":"%s","wall_s":%d,"iteration":%d,"half":%d,"part_id":%d}\n' \
    "${SLURM_JOB_ID}" "$(( $(date +%s) - start ))" \
    "${TARGET_ITERATION}" "${TARGET_HALF}" "${TARGET_PART_ID}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
sha256sum "${LAUNCHER}" "${SCRIPT}" "${K1_WAVG_ATOMIC_CUDA_LIB}" \
    "${ROOT}/analysis/wavg_atomic_replay.json" \
    > "${ROOT}/provenance/static_and_output_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
