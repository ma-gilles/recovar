#!/usr/bin/env bash
#SBATCH --job-name=k1-ppref-build
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
: "${K1_PPREF_BUILD_ROOT:?set the disposable build-audit root}"
: "${K1_PPREF_BUILD_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?pin the capture-only source diff}"
: "${EXPECTED_OLD_RELION_BINARY_SHA256:?pin the pre-build executable}"

ROOT=${K1_PPREF_BUILD_ROOT}
RUNTIME=${K1_PPREF_BUILD_RUNTIME}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
LAUNCHER=${REPO}/scripts/build_relion_k1_ppref_capture.sbatch
RELION_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET
SOURCE=${RELION_ROOT}/source
BUILD=${RELION_ROOT}/build
BINARY=${BUILD}/bin/relion_refine_mpi
OLD_BINARY_SHA256=${EXPECTED_OLD_RELION_BINARY_SHA256}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = "${OLD_BINARY_SHA256}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3
test "$(git -C "${SOURCE}" diff --name-only)" = src/acc/cuda/cuda_ml_optimiser.cu
test "$(git -C "${SOURCE}" diff -- src/acc/cuda/cuda_ml_optimiser.cu | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${ROOT}/provenance"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
export PATH=/usr/local/cuda-12.6/bin:${PATH}

cp --preserve=mode,timestamps "${BINARY}" "${ROOT}/relion_refine_mpi.pre_ppref"
sha256sum "${LAUNCHER}" "${BINARY}" "${SOURCE}/src/acc/cuda/cuda_ml_optimiser.cu" \
    > "${ROOT}/provenance/prebuild_${SLURM_JOB_ID}.sha256"
git -C "${SOURCE}" diff -- src/acc/cuda/cuda_ml_optimiser.cu \
    > "${ROOT}/provenance/relion_ppref_source.diff"

cmake --build "${BUILD}" --target refine_mpi -- -j4 \
    > "${ROOT}/build.stdout" 2> "${ROOT}/build.stderr"

test -x "${BINARY}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" != "${OLD_BINARY_SHA256}"
test "$(strings "${BINARY}" | grep -c RELION_PPREF_CAPTURE_DIR)" -eq 1
test "$(strings "${BINARY}" | grep -c 'ppref_iter%03d_rank%03d_model%03d.bin')" -eq 1
test "$(strings "${BINARY}" | grep -c RELION_SCORING_NOISE_CAPTURE)" -eq 1
test "$(strings "${BINARY}" | grep -c 'scoring_noise_iter%03d_rank%03d_optics%03zu.bin')" -eq 1
sha256sum "${BINARY}" > "${ROOT}/provenance/postbuild_binary_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
