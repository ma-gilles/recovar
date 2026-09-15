#!/usr/bin/env bash
#SBATCH --job-name=vdam-native-img
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-native-img-%j.out

set -euo pipefail

: "${DATA_DIR:?submit with DATA_DIR=/scratch/.../vdam-08/data}"
: "${OUTPUT_DIR:?submit with OUTPUT_DIR=/scratch/...}"
: "${CAPTURE_DIR:?submit with CAPTURE_DIR=/scratch/...}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${CAPTURE_PART_IDS:=0}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LAUNCHER="${REPO_ROOT}/scripts/run_vdam_native_unmasked_capture.sbatch"
RELION_REFINE=/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine
RUN_ID="${SLURM_JOB_ID:-manual}"

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test -x "${RELION_REFINE}"
test -s "${DATA_DIR}/particles.star"
test -z "$(find "${OUTPUT_DIR}" "${CAPTURE_DIR}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORMS JAX_PLATFORM_NAME
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/tmp"
export PIXI_HOME="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/pixi_home"
export RATTLER_CACHE_DIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/rattler_cache"
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${OUTPUT_DIR}" "${CAPTURE_DIR}"

export RECOVAR_DEBUG_ESTEP_DIR="${CAPTURE_DIR}"
export RECOVAR_DEBUG_ESTEP_PARTIDS="${CAPTURE_PART_IDS}"
export RECOVAR_DEBUG_ESTEP_ITER=1

COMMAND=(
    "${RELION_REFINE}"
    --o "${OUTPUT_DIR}/run"
    --iter 1
    --grad
    --denovo_3dref
    --grad_write_iter 1
    --i "${DATA_DIR}/particles.star"
    --ctf
    --K 1
    --sym C1
    --flatten_solvent
    --zero_mask
    --dont_combine_weights_via_disc
    --pool 3
    --pad 1
    --particle_diameter 200
    --oversampling 1
    --healpix_order 1
    --offset_range 6
    --offset_step 2
    --auto_sampling
    --tau2_fudge 4
    --random_seed 0
    --j 8
)

printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/command.sh"
printf '\n' >> "${OUTPUT_DIR}/command.sh"
sha256sum "${LAUNCHER}" "${RELION_REFINE}" "${DATA_DIR}/particles.star" \
    "${DATA_DIR}/particles.128.mrcs" > "${OUTPUT_DIR}/inputs.sha256"

(
    cd "${DATA_DIR}"
    "${COMMAND[@]}"
) > "${OUTPUT_DIR}/runner.stdout" 2> "${OUTPUT_DIR}/runner.stderr"

IFS=',' read -r -a capture_part_ids <<< "${CAPTURE_PART_IDS}"
for part_id in "${capture_part_ids[@]}"; do
    test -s "${CAPTURE_DIR}/p${part_id}_Fimg_nomask.bin"
done
test -s "${OUTPUT_DIR}/run_it001_data.star"
sha256sum "${CAPTURE_DIR}"/* > "${OUTPUT_DIR}/capture.sha256"
touch "${OUTPUT_DIR}/RUN_SUCCESS_${SLURM_JOB_ID}"
