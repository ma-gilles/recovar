#!/usr/bin/env bash
#SBATCH --job-name=k1-native-prefix
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=01:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${RELION_SOURCE:?RELION_SOURCE is required}"
: "${RELION_BINARY:?RELION_BINARY is required}"
: "${CASE_SOURCE:?CASE_SOURCE is required}"
: "${CASE_NAME:?CASE_NAME is required}"
: "${PREFIX_SELECTION_JSON:?PREFIX_SELECTION_JSON is required}"
: "${EXPECTED_SELECTION_SHA256:?EXPECTED_SELECTION_SHA256 is required}"
: "${SEED:?SEED is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_RELION_BINARY_SHA256:?EXPECTED_RELION_BINARY_SHA256 is required}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?EXPECTED_RELION_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_DATA_STAR_SHA256:?EXPECTED_DATA_STAR_SHA256 is required}"
: "${EXPECTED_REFERENCE_SHA256:?EXPECTED_REFERENCE_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
CASE=${CASE_SOURCE}
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
SOURCE=${RELION_SOURCE}
BINARY=${RELION_BINARY}
LAUNCHER=${REPO}/scripts/run_k1_case10_native_bpref_prefix.sbatch
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
PYTHON=${REPO}/.pixi/envs/default/bin/python
CAPTURE=${RUN_ROOT}/native_prefix
OUTPUT=${RUN_ROOT}/relion
ANALYSIS=${RUN_ROOT}/analysis
EXPECTED_FILES_PER_PARTICLE=${EXPECTED_FILES_PER_PARTICLE:-4}
case "${EXPECTED_FILES_PER_PARTICLE}" in
    4|14) ;;
    *) echo "EXPECTED_FILES_PER_PARTICLE must be 4 or 14" >&2; exit 2 ;;
esac

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${PREFIX_SELECTION_JSON}"
test "$(sha256sum "${PREFIX_SELECTION_JSON}" | awk '{print $1}')" = "${EXPECTED_SELECTION_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = d476e6f6a4f1f37627c06ace5227fc374c0c2b05
test "$(git -C "${SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = "${EXPECTED_DATA_STAR_SHA256}"
test "$(sha256sum "${DATA}/reference_init_relion.mrc" | awk '{print $1}')" = "${EXPECTED_REFERENCE_SHA256}"

readarray -t selection_values < <("${PYTHON}" - "${PREFIX_SELECTION_JSON}" "${CASE_NAME}" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    selection = json.load(stream)
assert selection["schema"] == "recovar-k1-bpref-prefix-selection-v2"
assert selection["case"] == sys.argv[2]
assert selection["physical_iteration"] == 1
half1 = selection["half1"]["native_internal_indices"]
half2 = selection["half2"]["native_internal_indices"]
assert len(half1) > 0 and len(half1) == len(half2)
assert len(set(half1 + half2)) == len(half1) + len(half2)
print(",".join(str(value) for value in half1 + half2))
print(len(half1) + len(half2))
PY
)
PART_IDS=${selection_values[0]}
PARTICLE_COUNT=${selection_values[1]}

mkdir -p "${CAPTURE}" "${OUTPUT}" "${ANALYSIS}" "${RUN_ROOT}/logs" \
    "${RUN_ROOT}/provenance" "${RUNTIME_ROOT}/tmp" \
    "${RUNTIME_ROOT}/pixi_home" "${RUNTIME_ROOT}/rattler_cache" \
    "${RUNTIME_ROOT}/pycache"
test -z "$(find "${CAPTURE}" "${OUTPUT}" "${ANALYSIS}" -mindepth 1 -print -quit)"
cp "${PREFIX_SELECTION_JSON}" "${RUN_ROOT}/prefix_selection.json"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

export RELION_DUMP_K1_BPREF_PREFIX_DIR=${CAPTURE}
export RELION_DUMP_K1_BPREF_PREFIX_ITER=1
export RELION_DUMP_K1_BPREF_PREFIX_PART_IDS=${PART_IDS}

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader \
    > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${SOURCE}" diff --binary > "${RUN_ROOT}/provenance/relion_capture.patch"
sha256sum "${LAUNCHER}" "${BINARY}" "${DATA}/particles.star" \
    "${DATA}/reference_init_relion.mrc" "${RUN_ROOT}/prefix_selection.json" \
    > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
  --i particles.star --ref reference_init_relion.mrc --o "${OUTPUT}/run"
  --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
  --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
  --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
  --oversampling 1 --pad 2 --random_seed "${SEED}" --auto_iter_max 1
  --pool 12 --gpu 0:0:0 --j 1)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${OUTPUT}/runner.stdout" 2> "${OUTPUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test "$(find "${CAPTURE}" -maxdepth 1 -name '*.bin' | wc -l)" \
    -eq "$((EXPECTED_FILES_PER_PARTICLE * PARTICLE_COUNT))"
"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    --control-dir "${CONTROL}" --candidate-dir "${OUTPUT}" --iterations 1 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ANALYSIS}/RELION_CAPTURE_INERTNESS.json" \
    --output-markdown "${ANALYSIS}/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${ANALYSIS}/RELION_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    report = json.load(stream)
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY
find "${CAPTURE}" "${OUTPUT}" "${ANALYSIS}" -type f -print0 \
    | sort -z | xargs -0 sha256sum \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","case":"%s","wall_s":%d,"physical_iteration":1,"particle_count":%d,"random_seed":%d,"threads":1,"pool":12}\n' \
    "${SLURM_JOB_ID}" "${CASE_NAME}" "${wall_s}" "${PARTICLE_COUNT}" "${SEED}" \
    > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
