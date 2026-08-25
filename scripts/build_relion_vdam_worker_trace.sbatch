#!/usr/bin/env bash
#SBATCH --job-name=vdam-worker-trace-build
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-worker-trace-build-%j.out

set -euo pipefail

: "${RELION_SOURCE:?set the isolated RELION worker-trace source worktree}"
: "${RELION_BUILD_ROOT:?set a disposable build/provenance root}"
: "${EXPECTED_RELION_HEAD:?pin the worker-trace source commit}"

test -f "${RELION_BUILD_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_HEAD}"
test -z "$(git -C "${RELION_SOURCE}" status --porcelain=v1 --untracked-files=no)"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.1/gcc-11.5.0-gpu
set -u

BUILD=${RELION_BUILD_ROOT}/build
PROVENANCE=${RELION_BUILD_ROOT}/provenance
mkdir -p "${BUILD}" "${PROVENANCE}"

git -C "${RELION_SOURCE}" show --stat --oneline HEAD \
    > "${PROVENANCE}/source_head.txt"
git -C "${RELION_SOURCE}" diff HEAD^ -- src/ml_optimiser.cpp src/ml_optimiser.h \
    > "${PROVENANCE}/worker_trace.patch"

/usr/bin/cmake --fresh -S "${RELION_SOURCE}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DMPI_C_COMPILER="$(command -v mpicc)" \
    -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
    -DCUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
    -DCUDA_ARCH=80 -DGUI=OFF -DBUILD_TESTS=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
/usr/bin/cmake --build "${BUILD}" --target refine --parallel "${SLURM_CPUS_PER_TASK}"

BINARY=${BUILD}/bin/relion_refine
test -x "${BINARY}"
strings "${BINARY}" | grep -Fx RELION_VDAM_WORKER_LOG_SCHEMA_V1 \
    > "${PROVENANCE}/schema_marker.txt"
sha256sum "${BINARY}" "${PROVENANCE}/worker_trace.patch" \
    > "${PROVENANCE}/build.sha256"
printf '{"job_id":"%s","source_head":"%s"}\n' \
    "${SLURM_JOB_ID}" "${EXPECTED_RELION_HEAD}" \
    > "${PROVENANCE}/build_job.json"
