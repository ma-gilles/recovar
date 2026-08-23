#!/usr/bin/env bash
#SBATCH --job-name=k1c04-norm-panel
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

: "${K1_NATIVE_NORM_PANEL_ROOT:?set the disposable capture root}"
: "${K1_NATIVE_NORM_PANEL_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"

ROOT=${K1_NATIVE_NORM_PANEL_ROOT}
RUNTIME=${K1_NATIVE_NORM_PANEL_RUNTIME}
PANEL_COUNT=${K1_NATIVE_NORM_PANEL_COUNT:-64}
BUILD_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_native_norm_split_build_20260821T1810ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
CONTROL_DATA=${CONTROL}/run_it001_data.star
OUTLIERS=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_norm_residual_it1_h1_stack82010_retry2_6d855156_20260822T0435ET/analysis/K1_CASE04_NORM_ROUNDING_OUTLIER_PANEL.json
BINARY=${BUILD_ROOT}/build/bin/relion_refine_mpi
SOURCE=${BUILD_ROOT}/source
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
LAUNCHER=${REPO}/scripts/run_k1_case04_native_norm_panel.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python

if [[ ! ${PANEL_COUNT} =~ ^[1-9][0-9]*$ ]] || (( PANEL_COUNT > 376 )); then
    echo "K1_NATIVE_NORM_PANEL_COUNT must be in [1, 376], got ${PANEL_COUNT}" >&2
    exit 2
fi
test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = f4541486d6c11d5872e97b29954e9fd56d66217b
test "$(git -C "${SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${SOURCE}" diff HEAD --binary | sha256sum | awk '{print $1}')" = 883265481bb1363e46a81820a9c660155f098076906ee0f6acddca2412dab482
test "$(sha256sum "${SOURCE}/src/acc/acc_ml_optimiser_impl.h" | awk '{print $1}')" = e3201af255a6a2d7f94f6e3f9f3446628ac60c12a15965a6211b7be42d52f6bc
test "$(sha256sum "${BINARY}" | awk '{print $1}')" = 53e6f3f502e9938ac64a8e55ce85727fcc865912dc0f5ece6fdde95d7aa2d644
test "$(sha256sum "${OUTLIERS}" | awk '{print $1}')" = 75991d4f14beddaf8b43436a601421a606ec0dd42f1bed3ec1a8b0473410c74a
test "$(sha256sum "${CONTROL_DATA}" | awk '{print $1}')" = 33e126519689bb3258fcce27004485dbdaa6dce69eaa0ab695fd4395f9075776
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

"${PYTHON}" - "${OUTLIERS}" "${CONTROL_DATA}" "${PANEL_COUNT}" \
    "${ROOT}/provenance/selection_${SLURM_JOB_ID}.json" <<'PY'
import json
import sys

import starfile

outliers_path, control_path, count_text, output_path = sys.argv[1:]
outliers = json.load(open(outliers_path))
control = starfile.read(control_path)
particles = control["particles"] if isinstance(control, dict) else control
part_by_image = {
    str(image_name): int(index)
    for index, image_name in enumerate(particles["rlnImageName"].astype(str).tolist())
}
selected = []
for item in outliers["top"][: int(count_text)]:
    record = dict(item)
    record["native_part_id"] = part_by_image[record["image_name"]]
    selected.append(record)
assert len(selected) == int(count_text)
assert len({item["native_part_id"] for item in selected}) == len(selected)
report = {
    "schema": "recovar.em.k1_native_norm_panel_selection.v1",
    "status": "complete",
    "count": len(selected),
    "selected": selected,
}
with open(output_path, "w") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
SELECTION=${ROOT}/provenance/selection_${SLURM_JOB_ID}.json
PART_IDS=$("${PYTHON}" -c 'import json,sys; print(",".join(str(x["native_part_id"]) for x in json.load(open(sys.argv[1]))["selected"]))' "${SELECTION}")
test "$(awk -F, '{print NF}' <<< "${PART_IDS}")" -eq "${PANEL_COUNT}"

# Selecting particles activates only the passive iteration-1 norm logger.  The
# binary capture is configured for iteration 2, which this job never reaches.
export RELION_P1_PREPROCESS_CAPTURE_DIR=${ROOT}/capture
export RELION_P1_PREPROCESS_CAPTURE_SCHEMA=1
export RELION_P1_PREPROCESS_CAPTURE_ITER=2
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_PARTICLES=${PANEL_COUNT}
export RELION_P1_PREPROCESS_CAPTURE_MAX_PARTICLES_PER_RANK=${PANEL_COUNT}
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_P1_PREPROCESS_CAPTURE_MAX_BYTES=1073741824
export RELION_P1_PREPROCESS_CAPTURE_PART_IDS=${PART_IDS}

nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
    > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${SOURCE}" diff HEAD --binary > "${ROOT}/provenance/relion_source.patch"
sha256sum "${LAUNCHER}" "${BINARY}" "${SOURCE}/src/acc/acc_ml_optimiser_impl.h" \
    "${DATA}/particles.star" "${DATA}/reference_init_relion.mrc" "${OUTLIERS}" \
    "${CONTROL_DATA}" "${SELECTION}" > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

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

test "$(rg -c 'RELION_P1_NORM_UPDATE_OPERANDS_V1 iter=1 ' "${ROOT}/relion/runner.stderr")" -eq "${PANEL_COUNT}"
test "$(rg -c 'RELION_P1_NORM_SPLIT_OPERANDS_V1 iter=1 ' "${ROOT}/relion/runner.stderr")" -eq "${PANEL_COUNT}"

cd "${AUDIT_REPO}"
"${PYTHON}" -m scripts.audit_relion_control_reproducibility \
    --control-dir "${CONTROL}" --candidate-dir "${ROOT}/relion" --iterations 1 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_NORM_PANEL_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_NORM_PANEL_INERTNESS.md"

"${PYTHON}" - "${ROOT}/relion/runner.stderr" "${SELECTION}" \
    "${ROOT}/analysis/K1_CASE04_ITER1_NATIVE_NORM_PANEL.json" <<'PY'
import json
import re
import sys

update_pattern = re.compile(
    r"RELION_P1_NORM_UPDATE_OPERANDS_V1 iter=1 part_id=(\d+) .* "
    r"wsum_norm=(\S+) sqrt_2_wsum=(\S+) new_norm=(\S+)"
)
split_pattern = re.compile(
    r"RELION_P1_NORM_SPLIT_OPERANDS_V1 iter=1 part_id=(\d+) "
    r"current_size=(\S+) high_shell=(\S+) total=(\S+)"
)
updates = {}
splits = {}
for line in open(sys.argv[1], errors="replace"):
    match = update_pattern.search(line)
    if match:
        part_id = int(match.group(1))
        updates[part_id] = {
            "wsum_norm": float.fromhex(match.group(2)),
            "sqrt_2_wsum": float.fromhex(match.group(3)),
            "new_norm": float.fromhex(match.group(4)),
        }
    match = split_pattern.search(line)
    if match:
        part_id = int(match.group(1))
        splits[part_id] = {
            "current_size": float.fromhex(match.group(2)),
            "high_shell": float.fromhex(match.group(3)),
            "total": float.fromhex(match.group(4)),
        }
selection = json.load(open(sys.argv[2]))
records = []
for selected in selection["selected"]:
    part_id = int(selected["native_part_id"])
    assert part_id in updates and part_id in splits
    record = dict(selected)
    record["native"] = {**updates[part_id], **splits[part_id]}
    record["delta"] = {
        "recovar_minus_native_norm": float(selected["recovar_norm"]) - updates[part_id]["new_norm"],
        "recovar_minus_native_total": 0.5 * float(selected["recovar_norm"]) ** 2 - splits[part_id]["total"],
    }
    records.append(record)
deltas = [item["delta"]["recovar_minus_native_norm"] for item in records]
report = {
    "schema": "recovar.em.k1_native_norm_panel.v1",
    "status": "complete",
    "iteration": 1,
    "count": len(records),
    "summary": {
        "mean_recovar_minus_native_norm": sum(deltas) / len(deltas),
        "min_recovar_minus_native_norm": min(deltas),
        "max_recovar_minus_native_norm": max(deltas),
        "positive_count": sum(delta > 0 for delta in deltas),
        "negative_count": sum(delta < 0 for delta in deltas),
    },
    "records": records,
}
with open(sys.argv[3], "w") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(report["summary"], indent=2, sort_keys=True))
PY

"${PYTHON}" - "${ROOT}/analysis/RELION_NORM_PANEL_INERTNESS.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY
find "${ROOT}/relion" "${ROOT}/analysis" -type f -print0 \
    | sort -z | xargs -0 sha256sum > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"iterations":[1],"particle_count":%d,"capture":"norm_operand_panel"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${PANEL_COUNT}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
