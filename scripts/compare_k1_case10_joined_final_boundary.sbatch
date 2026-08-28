#!/usr/bin/env bash
#SBATCH --job-name=k1-final-boundary-compare
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${REPO:?REPO is required}"
: "${CASE_ROOT:?CASE_ROOT is required}"
: "${SOURCE_RECOVAR:?SOURCE_RECOVAR is required}"
: "${FINAL_RELION_ITERATION:?FINAL_RELION_ITERATION is required}"
: "${RECOVAR_CAPTURE_ROOT:?RECOVAR_CAPTURE_ROOT is required}"
: "${RELION_CAPTURE_ROOT:?RELION_CAPTURE_ROOT is required}"
: "${RECOVAR_CAPTURE_JOB_ID:?RECOVAR_CAPTURE_JOB_ID is required}"
: "${RELION_CAPTURE_JOB_ID:?RELION_CAPTURE_JOB_ID is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/compare_k1_case10_joined_final_boundary.sbatch
ANALYZER=${REPO}/scripts/replay_final_bpref_dump.py
DATA=${CASE_ROOT}/data
CONTROL=${CASE_ROOT}/relion_ref
REC_DUMP=${RECOVAR_CAPTURE_ROOT}/final_bpref_accum/recovar_final_bpref_accum.npz
NATIVE_ITERATION_TAG=$(printf '%03d' "${FINAL_RELION_ITERATION}")
NATIVE_PREFIX=${RELION_CAPTURE_ROOT}/bpref/mstep_it${NATIVE_ITERATION_TAG}_rank1_half0_c0_pre_reconstruct
ANALYSIS=${RUN_ROOT}/analysis
PROVENANCE=${RUN_ROOT}/provenance

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${RELION_CAPTURE_ROOT}/SUCCESS_${RELION_CAPTURE_JOB_ID}"
test -f "${RECOVAR_CAPTURE_ROOT}/SUCCESS_${RECOVAR_CAPTURE_JOB_ID}"
test -s "${REC_DUMP}"
for suffix in bpref_data bpref_weight tau2 sigma2 data_vs_prior fsc fourier_coverage; do
    test -s "${NATIVE_PREFIX}_${suffix}.bin"
done
test -s "${NATIVE_PREFIX}_bpref_meta.txt"
rg -q "^iter=${FINAL_RELION_ITERATION}$" "${NATIVE_PREFIX}_bpref_meta.txt"
rg -q '^rank=1$' "${NATIVE_PREFIX}_bpref_meta.txt"
rg -q '^halfset=0$' "${NATIVE_PREFIX}_bpref_meta.txt"
test -s "${RELION_CAPTURE_ROOT}/relion/run_class001.mrc"
test -s "${CONTROL}/run_class001.mrc"
test -s "${SOURCE_RECOVAR}/final_merged.mrc"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test -z "$(find "${RUN_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE ! -name 'slurm-*.out' -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${ANALYSIS}" "${PROVENANCE}"

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_ROOT=/usr/local/cuda-12.8
export PATH=${CUDA_ROOT}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_ROOT}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign exactly one visible H100}"
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT

cd "${REPO}"
env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
git status --short --branch > "${PROVENANCE}/git_status_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
sha256sum "${ANALYZER}" "${LAUNCHER}" "${REC_DUMP}" \
    "${NATIVE_PREFIX}_bpref_data.bin" "${NATIVE_PREFIX}_bpref_weight.bin" \
    "${NATIVE_PREFIX}_tau2.bin" "${NATIVE_PREFIX}_sigma2.bin" \
    "${NATIVE_PREFIX}_data_vs_prior.bin" "${NATIVE_PREFIX}_fsc.bin" \
    "${NATIVE_PREFIX}_fourier_coverage.bin" \
    "${RELION_CAPTURE_ROOT}/relion/run_class001.mrc" \
    "${CONTROL}/run_class001.mrc" "${SOURCE_RECOVAR}/final_merged.mrc" \
    > "${PROVENANCE}/inputs_${SLURM_JOB_ID}.sha256"

"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert ".pixi/envs/default/" in str(Path(jax.__file__).resolve())
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
print(jax.devices())
PY

"${PYTHON}" -m scripts.replay_final_bpref_dump \
    --dump "${REC_DUMP}" \
    --gt-mrc "${DATA}/reference_gt.mrc" \
    --relion-mrc "${CONTROL}/run_class001.mrc" \
    --output-json "${ANALYSIS}/recovar_accumulator_replay.json" \
    --output-mrc "${ANALYSIS}/recovar_accumulator_replay.mrc" \
    > "${ANALYSIS}/recovar_accumulator_replay.stdout.json"

"${PYTHON}" -m scripts.replay_final_bpref_dump \
    --dump "${REC_DUMP}" \
    --relion-bpref-prefix "${NATIVE_PREFIX}" \
    --compare-relion-boundary-spectra \
    --relion-tau2-bin "${NATIVE_PREFIX}_tau2.bin" \
    --gt-mrc "${DATA}/reference_gt.mrc" \
    --relion-mrc "${CONTROL}/run_class001.mrc" \
    --output-json "${ANALYSIS}/native_accumulator_replay.json" \
    --output-mrc "${ANALYSIS}/native_accumulator_replay.mrc" \
    > "${ANALYSIS}/native_accumulator_replay.stdout.json"

CONTROL_RELION=${CONTROL}/run_class001.mrc \
CANDIDATE_RELION=${RELION_CAPTURE_ROOT}/relion/run_class001.mrc \
SOURCE_RECOVAR_MAP=${SOURCE_RECOVAR}/final_merged.mrc \
REPLAY_RECOVAR_MAP=${ANALYSIS}/recovar_accumulator_replay.mrc \
OUTPUT_JSON=${ANALYSIS}/capture_and_replay_inertness.json \
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from scripts.audit_em_k1_membership_capture_inertness import _load_map, _metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


paths = {
    name: Path(os.environ[name]).resolve()
    for name in (
        "CONTROL_RELION",
        "CANDIDATE_RELION",
        "SOURCE_RECOVAR_MAP",
        "REPLAY_RECOVAR_MAP",
    )
}
comparisons = {
    "native_relion_capture_vs_sealed_relion": _metrics(
        _load_map(paths["CONTROL_RELION"]),
        _load_map(paths["CANDIDATE_RELION"]),
    ),
    "recovar_boundary_replay_vs_source_recovar": _metrics(
        _load_map(paths["SOURCE_RECOVAR_MAP"]),
        _load_map(paths["REPLAY_RECOVAR_MAP"]),
    ),
}
threshold = 0.999999
for row in comparisons.values():
    row["passed"] = row["fsc_auc_non_dc"] >= threshold
report = {
    "schema": "em-k1-final-boundary-inertness-v2",
    "metric_policy": "signed shellwise FSC and non-DC FSC-AUC; no correlation",
    "fsc_auc_non_dc_threshold": threshold,
    "comparisons": comparisons,
    "all_inertness_gates_passed": all(row["passed"] for row in comparisons.values()),
    "artifact_sha256": {str(path): sha256(path) for path in paths.values()},
}
output = Path(os.environ["OUTPUT_JSON"])
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

find "${ANALYSIS}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
