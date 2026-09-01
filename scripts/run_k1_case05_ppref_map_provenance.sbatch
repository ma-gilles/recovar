#!/usr/bin/env bash
#SBATCH --job-name=k1c05-ppref-map
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case05_ppref_map_provenance.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_ppref_map_provenance.py
RELION_BIND=${RELION_BIND:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared}
PPREF=${PPREF:-${RUN_ROOT}/ppref/ppref_iter002_rank001_model000.bin}
NATIVE_MAP=${NATIVE_MAP:-${RUN_ROOT}/relion/run_it001_half1_class001.mrc}
RECOVAR_MAP=${RECOVAR_MAP:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row94083_recovar_projection_winner_it2_20260822T1227ET/output/intermediates/it000_half1_reg.mrc}
OUTPUT=${OUTPUT:-${RUN_ROOT}/analysis/K1_PPREF_MAP_PROVENANCE.json}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
for artifact in "${PPREF}" "${NATIVE_MAP}" "${RECOVAR_MAP}"; do test -s "${artifact}"; done
test ! -e "${OUTPUT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "$(dirname "${OUTPUT}")" "${RUN_ROOT}/provenance"

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
export CUDA_VISIBLE_DEVICES=0 RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
"${PYTHON}" -c "import pathlib,recovar,jax; repo=pathlib.Path('${REPO}').resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + '/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert jax.default_backend() == 'gpu'; from recovar.relion_bind import _relion_bind_core"

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_ppref_map_provenance \
  --ppref "${PPREF}" \
  --map "native_numbered:${NATIVE_MAP}:relion" \
  --map "recovar_numbered:${RECOVAR_MAP}:recovar" \
  --output-json "${OUTPUT}" \
  > "${RUN_ROOT}/analysis/ppref_map_provenance_${SLURM_JOB_ID}.stdout"

sha256sum "${PPREF}" "${NATIVE_MAP}" "${RECOVAR_MAP}" "${OUTPUT}" \
  > "${RUN_ROOT}/provenance/ppref_map_provenance_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_PPREF_MAP_${SLURM_JOB_ID}"
