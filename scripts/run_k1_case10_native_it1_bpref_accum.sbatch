#!/usr/bin/env bash
#SBATCH --job-name=k1c10-native-bpref
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:30:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${SOURCE_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
LAUNCHER=${REPO}/scripts/run_k1_case10_native_it1_bpref_accum.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_source_20260810
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_build_20260810/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
PYTHON=${REPO}/.pixi/envs/default/bin/python
BPREF_ACCUM=${ROOT}/bpref_accum
RELION_OUT=${ROOT}/relion
ANALYSIS=${ROOT}/analysis
N_THREADS=${N_THREADS:-4}
PARTICLE_POOL=${PARTICLE_POOL:-3}

case "${N_THREADS}" in
    1|4) ;;
    *) echo "N_THREADS must be 1 or 4" >&2; exit 2 ;;
esac
case "${PARTICLE_POOL}" in
    3|12) ;;
    *) echo "PARTICLE_POOL must be 3 or 12" >&2; exit 2 ;;
esac

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = 60ba0c44004a323f231a59e051a5f444e6d94641bc628ce6b094cb1357d043ea
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 235052e48ccf843df48392c8e2b4f0d634b7dcedf7be27b88eb1cff746c07201
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 3be82df3627e0d6c3c12d1eb42e1ab0c7feb09117bb66a38cc53c0621e9f8436
test "$(sha256sum "${DATA}/reference_init_relion.mrc" | awk '{print $1}')" = d1d52afee6161d295f907a7dc633bb2cc3b7c7d2d575f6347297567375af78c2
test "$(stat -Lc %s "${DATA}/particles.384.mrcs")" = 58982401024

mkdir -p "${BPREF_ACCUM}" "${RELION_OUT}" "${ANALYSIS}" "${ROOT}/logs" \
    "${ROOT}/provenance" "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" \
    "${RUNTIME}/rattler_cache" "${RUNTIME}/pycache"
test -z "$(find "${BPREF_ACCUM}" "${RELION_OUT}" "${ANALYSIS}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0

export RELION_DUMP_DIR=${BPREF_ACCUM}
export RELION_DUMP_BPREF=1
export RELION_DUMP_ITER=1

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${RELION_SOURCE}" diff --binary > "${ROOT}/provenance/relion_capture.patch"
sha256sum "${LAUNCHER}" "${RELION_BINARY}" "${DATA}/particles.star" \
    "${DATA}/reference_init_relion.mrc" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
  --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
  --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
  --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
  --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
  --oversampling 1 --pad 2 --random_seed 1710 --auto_iter_max 2
  --pool "${PARTICLE_POOL}" --gpu 0:0:0 --j "${N_THREADS}")
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

for half_rank in 1:1 2:2; do
    half=${half_rank%%:*}
    rank=${half_rank##*:}
    for kind in data weight; do
        test -s "${BPREF_ACCUM}/mstep_it001_rank${rank}_half${half}_c0_pre_lowres_join_bpref_${kind}.bin"
    done
done
"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" --iterations 1 \
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
find "${BPREF_ACCUM}" "${RELION_OUT}" "${ANALYSIS}" -type f -print0 \
    | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":1,"threads":%d,"particle_pool":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${N_THREADS}" "${PARTICLE_POOL}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
