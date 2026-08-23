#!/usr/bin/env bash
#SBATCH --job-name=k1c04-native-avg
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

: "${K1_NATIVE_AVG_ROOT:?set the disposable capture root}"
: "${K1_NATIVE_AVG_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"

ROOT=${K1_NATIVE_AVG_ROOT}
RUNTIME=${K1_NATIVE_AVG_RUNTIME}
BUILD_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_native_avg_build_20260822T0450ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
BINARY=${BUILD_ROOT}/build/bin/relion_refine_mpi
SOURCE=${BUILD_ROOT}/source
LAUNCHER=${REPO}/scripts/run_k1_case04_native_avg_boundary.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python
RECOVAR_STATE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_norm64_target82009_it2_6d855156_20260822T0345ET/noise_debug/recovar_noise_update_it001.npz
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${SOURCE}" diff HEAD --binary | sha256sum | awk '{print $1}')" = 80a6036450dc3b20f2548aae91a67b52a93608fd56837563f041860ed1777326
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = c2f8d9d7245d96914187a6f9b0d47c9f1dc93beb88204cdbf2ec70ae5644efae
test -f "${RECOVAR_STATE}"
test -z "$(find "${ROOT}/relion" "${ROOT}/analysis" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

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
export RELION_K1_NORM_AVG_OPERANDS=1

nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${SOURCE}" diff HEAD --binary > "${ROOT}/provenance/relion_source.patch"
sha256sum "${LAUNCHER}" "${BINARY}" "${SOURCE}/src/ml_optimiser.cpp" \
    "${DATA}/particles.star" "${DATA}/reference_init_relion.mrc" "${RECOVAR_STATE}" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=(srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${ROOT}/relion/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1704 --auto_iter_max 1
    --pool 3 --gpu 0:0:0 --j 4)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${command[@]}"
) > "${ROOT}/relion/runner.stdout" 2> "${ROOT}/relion/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test "$(rg -c 'RELION_K1_NORM_AVG_OPERANDS_V1 iter=1 ' "${ROOT}/relion/runner.stderr")" -eq 2

cd "${AUDIT_REPO}"
"${PYTHON}" -m scripts.audit_relion_control_reproducibility \
    --control-dir "${CONTROL}" --candidate-dir "${ROOT}/relion" --iterations 1 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_AVG_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_AVG_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${ROOT}/analysis/RELION_AVG_CAPTURE_INERTNESS.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

"${PYTHON}" - "${ROOT}/relion/runner.stderr" "${RECOVAR_STATE}" \
    "${ROOT}/analysis/K1_CASE04_ITER1_NORM_AVG_BOUNDARY.json" <<'PY'
import json
import re
import sys

import numpy as np

pattern = re.compile(
    r"RELION_K1_NORM_AVG_OPERANDS_V1 iter=1 previous_avg=(\S+) mu=(\S+) "
    r"numerator=(\S+) denominator=(\S+) new_avg=(\S+)"
)
records = []
for line in open(sys.argv[1], errors="replace"):
    match = pattern.search(line)
    if match:
        previous, mu, numerator, denominator, new_avg = (
            float.fromhex(value) for value in match.groups()
        )
        records.append(
            {
                "previous_avg": previous,
                "mu": mu,
                "numerator": numerator,
                "denominator": denominator,
                "new_avg": new_avg,
            }
        )
assert len(records) == 2, records
with np.load(sys.argv[2], allow_pickle=False) as state:
    comparisons = []
    unused = set(range(len(records)))
    for half in (1, 2):
        wsum = np.asarray(state[f"half{half}_wsum_norm_correction"], dtype=np.float64)
        denominator = float(np.asarray(state[f"half{half}_sumw"]).item())
        norms = np.sqrt(2.0 * wsum / float(256**4))
        numerator = float(np.sum(norms, dtype=np.float64))
        record_index = min(unused, key=lambda index: abs(records[index]["denominator"] - denominator))
        unused.remove(record_index)
        native = records[record_index]
        comparisons.append(
            {
                "half": half,
                "native": native,
                "recovar": {
                    "numerator": numerator,
                    "denominator": denominator,
                    "new_avg": numerator / denominator,
                },
                "delta": {
                    "numerator": numerator - native["numerator"],
                    "denominator": denominator - native["denominator"],
                    "new_avg": numerator / denominator - native["new_avg"],
                },
            }
        )
assert not unused
report = {
    "schema": "recovar.em.k1_native_norm_average_boundary.v1",
    "status": "complete",
    "iteration": 1,
    "comparisons": comparisons,
    "metric_policy": "exact RFLOAT normalization operands; FSC reserved for inertness",
}
with open(sys.argv[3], "w") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

find "${ROOT}/relion" "${ROOT}/analysis" -type f -print0 \
    | sort -z | xargs -0 sha256sum > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"iterations":[1],"capture":"norm_average_numerator_denominator"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
