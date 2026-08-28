#!/usr/bin/env bash
#SBATCH --job-name=k1-relion-tau-oracle
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=recovar-test
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RECOVAR_DUMP_PATH:?RECOVAR_DUMP_PATH is required}"
: "${RELION_MODEL_STAR:?RELION_MODEL_STAR is required}"
: "${RELION_FINAL_MAP:?RELION_FINAL_MAP is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_REPLAY_SHA256:?EXPECTED_REPLAY_SHA256 is required}"

REPO=${REPO_ROOT}
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
BOUNDARY_DUMP=${RECOVAR_DUMP_PATH}
MODEL_STAR=${RELION_MODEL_STAR}
FINAL_MAP=${RELION_FINAL_MAP}
LAUNCHER=${REPO}/scripts/replay_k1_final_with_relion_model_tau.sbatch
REPLAY=${REPO}/scripts/replay_final_bpref_dump.py
ANALYSIS=${RUN_ROOT}/analysis
PROVENANCE=${RUN_ROOT}/provenance
TAU_BIN=${ANALYSIS}/sealed_relion_model_tau2.bin
REPORT=${ANALYSIS}/recovar_accumulator_relion_tau_replay.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${BOUNDARY_DUMP}" -a -s "${MODEL_STAR}" -a -s "${FINAL_MAP}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${REPLAY}" | awk '{print $1}')" = "${EXPECTED_REPLAY_SHA256}"
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

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert Path(jax.__file__).resolve().is_relative_to(repo / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
print(jax.devices())
PY

env | sort > "${PROVENANCE}/environment_${SLURM_JOB_ID}.txt"
git status --short --branch > "${PROVENANCE}/git_status_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${PROVENANCE}/gpu_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${REPLAY}" "${BOUNDARY_DUMP}" "${MODEL_STAR}" "${FINAL_MAP}" \
    > "${PROVENANCE}/inputs_${SLURM_JOB_ID}.sha256"

"${PYTHON}" - "${MODEL_STAR}" "${TAU_BIN}" <<'PY'
import struct
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
rows = []
inside = False
for line in source.read_text().splitlines():
    if line.strip() == "_rlnSpectralOrientabilityContribution #9":
        inside = True
        continue
    if not inside:
        continue
    fields = line.split()
    if len(fields) != 9:
        if rows:
            break
        continue
    rows.append(float(fields[7]))
if not rows:
    raise RuntimeError(f"no model-class tau2 rows found in {source}")
with target.open("wb") as stream:
    stream.write(struct.pack("<q", len(rows)))
    for value in rows:
        stream.write(struct.pack("<d", value))
print(f"wrote {len(rows)} sealed RELION tau2 shells to {target}")
PY

"${PYTHON}" -m scripts.replay_final_bpref_dump \
    --dump "${BOUNDARY_DUMP}" \
    --relion-mrc "${FINAL_MAP}" \
    --relion-tau2-bin "${TAU_BIN}" \
    --grid-correct off \
    --tau2-fsc-mode whole \
    --output-json "${REPORT}" \
    --output-mrc "${ANALYSIS}/recovar_accumulator_relion_tau_replay.mrc" \
    > "${ANALYSIS}/recovar_accumulator_relion_tau_replay.stdout.json"

"${PYTHON}" - "${REPORT}" "${ANALYSIS}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
metrics = report["metrics_vs_relion"]
summary = {
    "fsc_auc_vs_relion": metrics["fsc_auc"],
    "correlation_vs_relion": metrics.get("corr"),
    "relative_l2_vs_relion": metrics.get("relative_l2"),
    "tau2_source": report["tau2_source"],
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

sha256sum "${TAU_BIN}" "${REPORT}" "${ANALYSIS}/summary.json" \
    "${ANALYSIS}/recovar_accumulator_relion_tau_replay.mrc" \
    > "${PROVENANCE}/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
