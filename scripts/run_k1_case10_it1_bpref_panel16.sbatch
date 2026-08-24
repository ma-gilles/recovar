#!/usr/bin/env bash
#SBATCH --job-name=k1c10-bpref16
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_REPO_DIFF_SHA256:?EXPECTED_REPO_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case10_it1_bpref_panel16.sbatch
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_bpref_fma_44d770de3_20260823T1938ET/cuda/libcuda_backproject.so
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/source
RELION_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_k1_preiter_order_replay_6982c77_20260803T0215ET/build/bin/relion_refine_mpi
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
ORIGINAL_INDICES=0,14275,28628,42698,57062,71399,85827,99997,3,14301,28522,43039,57231,71452,85593,99999
STACK_INDICES=1,14276,28629,42699,57063,71400,85828,99998,4,14302,28523,43040,57232,71453,85594,100000

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = 1c3f67f52a7663f29c3cc0c8aacc995eba27517961e24ee4a1be01b6413c0b1d
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = f8355ba9ef87ee28f3ae7bfe937c3f2589ab7bd3
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD^{tree})" = 3111173edb150f98bc2446efc54229ff736b40a1
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = 15e9cceb38815cc85ed946e1bb5ee8a9b42f89a530cf8d5af2d76e83228be3f9
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = 2141e70e4ec32be930b4512a29de8991ee04e19ff337687b0001167acb8fb9f9
test "$(sha256sum "${DATA}/particles.star" | awk '{print $1}')" = 3be82df3627e0d6c3c12d1eb42e1ab0c7feb09117bb66a38cc53c0621e9f8436
test "$(stat -Lc %s "${DATA}/particles.384.mrcs")" = 58982401024

mkdir -p "${ROOT}/capture" "${ROOT}/relion" "${ROOT}/recovar/output" \
    "${ROOT}/recovar/pass2" "${ROOT}/analysis" "${ROOT}/logs" "${ROOT}/provenance"
for directory in "${ROOT}/capture" "${ROOT}/relion" "${ROOT}/recovar/output" \
    "${ROOT}/recovar/pass2" "${ROOT}/analysis"; do
    test -z "$(find "${directory}" -mindepth 1 -print -quit)"
done
mkdir -p "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache" \
    "${RUNTIME}/pycache" "${RUNTIME}/jax_cache" "${RUNTIME}/cuda_cache"

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
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache

"${PYTHON}" - "${DATA}/particles.star" "${CONTROL}/run_it000_data.star" \
    "${ROOT}/provenance/SELECTION.json" "${ORIGINAL_INDICES}" <<'PY'
import json
import sys

import starfile

source = starfile.read(sys.argv[1])
source = source["particles"] if isinstance(source, dict) else source
ordered = starfile.read(sys.argv[2])
ordered = ordered["particles"] if isinstance(ordered, dict) else ordered
subset_by_identity = {
    str(identity): int(subset)
    for identity, subset in zip(ordered["rlnImageName"], ordered["rlnRandomSubset"])
}
indices = [int(value) for value in sys.argv[4].split(",")]
targets = []
for original_index in indices:
    identity = str(source.iloc[original_index]["rlnImageName"])
    stack_index = int(identity.split("@", 1)[0])
    targets.append(
        {
            "original_index_zero_based": original_index,
            "stack_index_one_based": stack_index,
            "expected_mpi_rank": subset_by_identity[identity],
            "image_identity": identity,
            "baseline_support_delta": 0,
            "role": "two_half_source_order_quantile_panel",
        }
    )
report = {
    "schema": "recovar.em.k1_bpref_factor_panel.v1",
    "case_id": 10,
    "physical_iteration": 1,
    "physical_image_size": 384,
    "current_size": 58,
    "selection_policy": (
        "fixed 16-particle two-half source-order quantile panel, "
        "declared before the current exact-BPref factor comparison"
    ),
    "capture_stack_indices_one_based": [row["stack_index_one_based"] for row in targets],
    "targets": targets,
}
with open(sys.argv[3], "w") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.0/gcc-11.5.0
set -u
MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
RELION_LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_HOME}/lib64
export LD_LIBRARY_PATH=${RELION_LD_LIBRARY_PATH}
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export CUDA_VISIBLE_DEVICES=0
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac

export RELION_BPRE_CAPTURE_DIR=${ROOT}/capture
export RELION_BPRE_CAPTURE_SCHEMA=2
export RELION_BPRE_CAPTURE_ITER=1
export RELION_BPRE_CAPTURE_CLASS=1
export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=16
export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=16
export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_BPRE_CAPTURE_MAX_BYTES=4000000000
export RELION_BPRE_CAPTURE_STACKS=${STACK_INDICES}

relion_command=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${ROOT}/relion/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1710 --auto_iter_max 1 --pool 3
    --gpu 0:0:0 --j 4
)
printf '%q ' "${relion_command[@]}" > "${ROOT}/provenance/command_relion_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_relion_${SLURM_JOB_ID}.sh"
relion_start=$(date +%s)
(
    cd "${DATA}"
    "${relion_command[@]}"
) > "${ROOT}/relion/runner.stdout" 2> "${ROOT}/relion/runner.stderr"
printf '{"wall_s":%d}\n' "$(( $(date +%s) - relion_start ))" \
    > "${ROOT}/provenance/walltime_relion_${SLURM_JOB_ID}.json"
test "$(find "${ROOT}/capture" -maxdepth 1 -name '*.bpre-v2.bin' | wc -l)" -eq 16

cd "${REPO}"
"${PYTHON}" -m scripts.validate_relion_bpref_factor_capture \
    "${ROOT}/capture" --selection-json "${ROOT}/provenance/SELECTION.json" \
    --output-json "${ROOT}/analysis/RELION_FACTOR_VALIDATION.json" \
    > "${ROOT}/logs/validate_factor_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/validate_factor_${SLURM_JOB_ID}.stderr"
"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    --control-dir "${CONTROL}" --candidate-dir "${ROOT}/relion" --iterations 1 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    "${ROOT}/analysis/RELION_FACTOR_CAPTURE_INERTNESS.json" "${ROOT}/provenance/SELECTION.json" <<'PY'
import json
import sys

source = json.load(open(sys.argv[1]))
selection = json.load(open(sys.argv[3]))
row = source["iterations"][0]
minimum_auc = min(product["fsc_auc"] for product in row["maps"].values())
qualified = bool(row["map_stable"] and row["selected_topology_exact"] and minimum_auc >= 0.999999)
result = {
    "schema": "recovar.em.k1_bpref_factor_capture_inertness.v1",
    "status": "qualified" if qualified else "rejected",
    "qualified": qualified,
    "case_id": 10,
    "physical_iteration": 1,
    "target_stack_indices_one_based": [
        int(item["stack_index_one_based"]) for item in selection["targets"]
    ],
    "minimum_signed_normalized_non_dc_fsc_auc": minimum_auc,
    "selected_topology_exact": bool(row["selected_topology_exact"]),
    "full_inertness_report": sys.argv[1],
    "metric_policy": "signed shellwise FSC/FSC-AUC; exact topology; no correlation",
}
with open(sys.argv[2], "w") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
if not qualified:
    raise SystemExit("native factor capture failed inertness qualification")
PY

unset RELION_BPRE_CAPTURE_DIR RELION_BPRE_CAPTURE_SCHEMA RELION_BPRE_CAPTURE_ITER \
    RELION_BPRE_CAPTURE_CLASS RELION_BPRE_CAPTURE_EXPECTED_PARTICLES \
    RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS \
    RELION_BPRE_CAPTURE_MAX_BYTES RELION_BPRE_CAPTURE_STACKS
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib:${RELION_LD_LIBRARY_PATH}
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_CUDA_CACHE_DIR=${RUNTIME}/cuda_cache
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_PASS2_DUMP_DIR=${ROOT}/recovar/pass2
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${ORIGINAL_INDICES}
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=58
export RECOVAR_PASS2_DUMP_CLASS=1
unset RECOVAR_PASS2_DUMP_RAW_OPERANDS
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

git -C "${REPO}" status --short --branch > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
git -C "${REPO}" diff --binary | sha256sum > "${ROOT}/provenance/repo_diff_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
"${PYTHON}" - <<'PY'
from pathlib import Path

import jax
import recovar
from recovar import cuda_backproject

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert Path(jax.__file__).resolve().is_relative_to(root / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
PY

recovar_command=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${ROOT}/recovar/output" --max_iter 1 --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed 1710 --perturb_seed 1710
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --save_intermediates_dir "${ROOT}/recovar/output/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${CONTROL}/run_it000_data.star"
    --relion_optimiser "${CONTROL}/run_it000_optimiser.star"
    --relion_init_dir "${CONTROL}"
)
printf '%q ' "${recovar_command[@]}" > "${ROOT}/provenance/command_recovar_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_recovar_${SLURM_JOB_ID}.sh"
recovar_start=$(date +%s)
"${recovar_command[@]}" > "${ROOT}/recovar/output/run.log" 2>&1
printf '{"wall_s":%d}\n' "$(( $(date +%s) - recovar_start ))" \
    > "${ROOT}/provenance/walltime_recovar_${SLURM_JOB_ID}.json"
test "$(find "${ROOT}/recovar/pass2" -maxdepth 1 -name '*.npz' | wc -l)" -eq 16
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' \
    "${ROOT}/recovar/output/run.log"

"${PYTHON}" -m scripts.analyze_k1_bpref_factor_boundary \
    --factor-directory "${ROOT}/capture" \
    --recovar-capture-directory "${ROOT}/recovar/pass2" \
    --selection-json "${ROOT}/provenance/SELECTION.json" \
    --capture-inertness-json "${ROOT}/analysis/RELION_FACTOR_CAPTURE_INERTNESS.json" \
    --output-json "${ROOT}/analysis/K1_CASE10_IT1_BPREF_PANEL16.json" \
    > "${ROOT}/logs/analyze_factor_${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/analyze_factor_${SLURM_JOB_ID}.stderr"

find "${ROOT}/capture" "${ROOT}/relion" "${ROOT}/recovar" "${ROOT}/analysis" \
    -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","gpu_uuid":"%s","gpu_name":"%s","particle_count":16}\n' \
    "${SLURM_JOB_ID}" "${GPU_UUID}" "${GPU_NAME}" \
    > "${ROOT}/provenance/completion_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
