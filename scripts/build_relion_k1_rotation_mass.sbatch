#!/usr/bin/env bash
#SBATCH --job-name=k1-rotmass-build
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_rotation_mass_rank1_build_20260810T1650ET/logs/build-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_rotation_mass_rank1_build_20260810T1650ET/logs/build-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?submission must pin RELION source diff hash}"
ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_rotation_mass_rank1_build_20260810T1650ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/relion_k1_rotation_mass_rank1_build_20260810T1650ET
SOURCE=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k1_case22_operand_capture_ac12ca86_20260729
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
LAUNCHER=${REPO}/scripts/build_relion_k1_rotation_mass.sbatch
BUILD=${ROOT}/build

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test ! -e "${BUILD}"
mkdir -p "${ROOT}/logs" "${ROOT}/provenance" "${RUNTIME}/${SLURM_JOB_ID}/tmp"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
PROVEN_OVERLAY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
MPI_INCLUDE=${PROVEN_OVERLAY}/openmpi_dev_overlay/include
GPU_LIB=${PROVEN_OVERLAY}/system_dev_overlay/lib
CUDA_INCLUDE=${PROVEN_OVERLAY}/system_dev_overlay/cuda_include
FFTW=/scratch/gpfs/GILLES/mg6942/relion/external/fftw
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${GPU_LIB}:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64

git -C "${SOURCE}" diff --binary > "${ROOT}/provenance/source.patch"
git -C "${SOURCE}" status --short > "${ROOT}/provenance/source.status"
cmake -S "${SOURCE}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_PATH="${GPU_LIB}" \
  -DCMAKE_CXX_FLAGS="-I${CUDA_INCLUDE}" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  -DCUDA_cufft_LIBRARY:FILEPATH="${GPU_LIB}/libcufft.so" \
  -DCUDA_curand_LIBRARY:FILEPATH="${GPU_LIB}/libcurand.so" \
  -DMPI_C_COMPILER_INCLUDE_DIRS="${MPI_INCLUDE}" \
  -DMPI_CXX_COMPILER_INCLUDE_DIRS="${MPI_INCLUDE}" \
  -DMPI_C_HEADER_DIR="${MPI_INCLUDE}" \
  -DMPI_CXX_HEADER_DIR="${MPI_INCLUDE}" \
  -DMPI_C_LIB_NAMES=mpi \
  -DMPI_CXX_LIB_NAMES='mpi_cxx;mpi' \
  -DMPI_mpi_LIBRARY="${MPI_ROOT}/lib64/libmpi.so" \
  -DMPI_mpi_cxx_LIBRARY="${MPI_ROOT}/lib64/libmpi_cxx.so" \
  -DOWN_FFTW_INCLUDES:PATH="${FFTW}/include" \
  -DOWN_FFTW_DOUBLE:FILEPATH="${FFTW}/lib/libfftw3.so" \
  -DOWN_FFTW_SINGLE:FILEPATH="${FFTW}/lib/libfftw3f.so" \
  -DFETCH_WEIGHTS=OFF \
  -DCUDA_ARCH=80 \
  -DGUI=OFF \
  -DFORCE_OWN_FFTW=OFF
cmake --build "${BUILD}" --target refine_mpi --parallel 8
test -x "${BUILD}/bin/relion_refine_mpi"
sha256sum "${BUILD}/bin/relion_refine_mpi" "${ROOT}/provenance/source.patch" > "${ROOT}/provenance/build_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/BUILD_SUCCESS_${SLURM_JOB_ID}"
