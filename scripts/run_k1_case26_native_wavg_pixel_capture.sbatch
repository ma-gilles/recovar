#!/usr/bin/env bash
#SBATCH --job-name=k1c26-wavg-pixel
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_wavg_pixels_20260810T2330ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_wavg_pixels_20260810T2330ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:10:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"
: "${EXPECTED_RELION_BINARY_SHA256:?submission must pin diagnostic binary hash}"
: "${EXPECTED_RELION_SOURCE_DIFF_SHA256:?submission must pin diagnostic source diff hash}"

ROOT=${RUN_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_wavg_sameprocess_20260811T0032ET}
RUNTIME=${RUNTIME_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_native_wavg_sameprocess_20260811T0032ET}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
AUDIT_REPO=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_factor_recovar_187a9f60_20260808T1100ET/checkout
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case26_native_wavg_pixel_capture.sbatch
RELION_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_source_20260810
RELION_BINARY=${RELION_BINARY_OVERRIDE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case26_wavg_pixel_build_20260810/bin/relion_refine_mpi}
RELION_RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
DATA=${CASE}/data
CONTROL=${CASE}/relion_ref
CAPTURE_DIR=${ROOT}/capture
BPREF_CAPTURE_DIR=${ROOT}/bpref_capture
BPREF_ACCUM_DIR=${ROOT}/bpref_accum
RELION_OUT=${ROOT}/relion
CAPTURE_PART_IDS=${CAPTURE_PART_IDS:-0,131,150,1,284,285,410,445,567,573,634,698,731,843,862,998,999}
CAPTURE_STACK_INDICES=${CAPTURE_STACK_INDICES:-1,216,233,10,354,355,468,5,609,614,67,727,757,858,875,998,999}
CAPTURE_EXPECTED_PARTICLES=${CAPTURE_EXPECTED_PARTICLES:-17}
CAPTURE_BPREF_PRESCATTER=${CAPTURE_BPREF_PRESCATTER:-0}
CAPTURE_BPREF_ACCUM=${CAPTURE_BPREF_ACCUM:-0}
CAPTURE_PPREF=${CAPTURE_PPREF:-0}
CAPTURE_WAVG_REFERENCE=${CAPTURE_WAVG_REFERENCE:-0}
PPREF_CAPTURE_DIR=${ROOT}/ppref

require_sha() {
    test "$(sha256sum "$2" | awk '{print $1}')" = "$1"
}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
require_sha "${EXPECTED_LAUNCHER_SHA256}" "${LAUNCHER}"
require_sha "${EXPECTED_RELION_BINARY_SHA256}" "${RELION_BINARY}"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = ed53c60d83125902c456b7fc5461c78c3966b306
test "$(git -C "${RELION_SOURCE}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
require_sha 34016f7f367587125956de8f00895c27b14225511c8ef0f517179b890685e139 "${DATA}/particles.star"
require_sha 4cd77e4ddeb731f719adb0cfd661e4f00221e07d08e15e1a5dc229737d9c848c "${DATA}/particles.128.mrcs"
require_sha 7aafc4f04e482f7410ff37686ba46db14578334ca77909f04bff30dbba2789d1 "${DATA}/reference_init_relion.mrc"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${AUDIT_REPO}" rev-parse HEAD)" = 187a9f60cdea19363d49d46099b052f23288f05a
test -z "$(git -C "${AUDIT_REPO}" status --porcelain=v1)"
test -x "${PYTHON}" -a -x "${RELION_BINARY}"
test -z "$(find "${CAPTURE_DIR}" -mindepth 1 -print -quit)"
test -z "$(find "${RELION_OUT}" -mindepth 1 -print -quit)"
if test "${CAPTURE_BPREF_PRESCATTER}" = 1; then
    test -d "${BPREF_CAPTURE_DIR}"
    test -z "$(find "${BPREF_CAPTURE_DIR}" -mindepth 1 -print -quit)"
fi
if test "${CAPTURE_BPREF_ACCUM}" = 1; then
    test -d "${BPREF_ACCUM_DIR}"
    test -z "$(find "${BPREF_ACCUM_DIR}" -mindepth 1 -print -quit)"
fi
if test "${CAPTURE_PPREF}" = 1; then
    mkdir -p "${PPREF_CAPTURE_DIR}"
    test -z "$(find "${PPREF_CAPTURE_DIR}" -mindepth 1 -print -quit)"
fi

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
if test "${CAPTURE_PPREF}" = 1; then
    export RELION_PPREF_CAPTURE_DIR=${PPREF_CAPTURE_DIR}
    export RELION_PPREF_CAPTURE_ITER=1
fi
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

export RELION_ACC_DUMP_DIR=${CAPTURE_DIR}
export RELION_ACC_DUMP_ITER=1
export RELION_ACC_DUMP_PART_IDS=${CAPTURE_PART_IDS}
export RELION_ACC_DUMP_WAVG_PIXELS=1
if test "${CAPTURE_WAVG_REFERENCE}" = 1; then
    export RELION_ACC_DUMP_WAVG_REFERENCE=1
fi
export RELION_P1_PREPROCESS_CAPTURE_DIR=${CAPTURE_DIR}
export RELION_P1_PREPROCESS_CAPTURE_SCHEMA=1
export RELION_P1_PREPROCESS_CAPTURE_ITER=1
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_PARTICLES=${CAPTURE_EXPECTED_PARTICLES}
export RELION_P1_PREPROCESS_CAPTURE_MAX_PARTICLES_PER_RANK=${CAPTURE_EXPECTED_PARTICLES}
export RELION_P1_PREPROCESS_CAPTURE_EXPECTED_FOLLOWERS=2
export RELION_P1_PREPROCESS_CAPTURE_MAX_BYTES=20000000
export RELION_P1_PREPROCESS_CAPTURE_PART_IDS=${CAPTURE_PART_IDS}
if test "${CAPTURE_BPREF_PRESCATTER}" = 1; then
    export RELION_BPRE_CAPTURE_DIR=${BPREF_CAPTURE_DIR}
    export RELION_BPRE_CAPTURE_SCHEMA=1
    export RELION_BPRE_CAPTURE_ITER=1
    export RELION_BPRE_CAPTURE_CLASS=1
    export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES=${CAPTURE_EXPECTED_PARTICLES}
    export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=${CAPTURE_EXPECTED_PARTICLES}
    export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=2
    export RELION_BPRE_CAPTURE_MAX_BYTES=200000000
    export RELION_BPRE_CAPTURE_DEVICE_BYTES=33554432
    export RELION_BPRE_CAPTURE_PART_IDS=${CAPTURE_PART_IDS}
fi
if test "${CAPTURE_BPREF_ACCUM}" = 1; then
    export RELION_DUMP_DIR=${BPREF_ACCUM_DIR}
    export RELION_DUMP_BPREF=1
    export RELION_DUMP_ITER=1
fi

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "gpu_uuid=${GPU_UUID}"
    echo "gpu_name=${GPU_NAME}"
    echo "repo_head=${EXPECTED_REPO_HEAD}"
    echo "relion_source_head=$(git -C "${RELION_SOURCE}" rev-parse HEAD)"
    echo "relion_source_diff_sha256=${EXPECTED_RELION_SOURCE_DIFF_SHA256}"
    echo "capture_part_ids=${CAPTURE_PART_IDS}"
    echo "capture_stack_indices=${CAPTURE_STACK_INDICES}"
    echo "capture_iteration=1"
    echo "capture_bpref_prescatter=${CAPTURE_BPREF_PRESCATTER}"
    echo "capture_bpref_accum=${CAPTURE_BPREF_ACCUM}"
    echo "capture_ppref=${CAPTURE_PPREF}"
    echo "capture_wavg_reference=${CAPTURE_WAVG_REFERENCE}"
    echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset"
    echo "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset"
    sha256sum "${LAUNCHER}" "${RELION_BINARY}" \
        "${DATA}/particles.star" "${DATA}/particles.128.mrcs" \
        "${DATA}/reference_init_relion.mrc"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"
git -C "${RELION_SOURCE}" diff --binary \
    > "${ROOT}/provenance/relion_wavg_capture.patch"

COMMAND=(
    srun --ntasks=3 --cpus-per-task=4 --cpu-bind=none "${RELION_BINARY}"
    --i particles.star --ref reference_init_relion.mrc --o "${RELION_OUT}/run"
    --auto_refine --split_random_halves --particle_diameter 200 --ini_high 30
    --firstiter_cc --ctf --flatten_solvent --zero_mask --low_resol_join_halves 40
    --norm --scale --healpix_order 3 --offset_range 3.0 --offset_step 1.0
    --oversampling 1 --pad 2 --random_seed 1726 --auto_iter_max 2
    --pool 3 --gpu 0:0:0 --j 4
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
(
    cd "${DATA}"
    "${COMMAND[@]}"
) > "${RELION_OUT}/runner.stdout" 2> "${RELION_OUT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

for suffix in \
    wdiff2_pixels aa_pixels xa_pixels mresol \
    fimg_real fimg_imag ctf trans_x trans_y weights eulers \
    valid_wdiff2_sum image_size current_size orientation_num translation_num; do
    test "$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name "*_storeWavg_${suffix}.bin" | wc -l)" -eq "${CAPTURE_EXPECTED_PARTICLES}"
done
if test "${CAPTURE_WAVG_REFERENCE}" = 1; then
    for suffix in reference_real reference_imag; do
        test "$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name "*_storeWavg_${suffix}.bin" | wc -l)" -eq "${CAPTURE_EXPECTED_PARTICLES}"
    done
fi
test "$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name '*.preprocess-v1.bin' | wc -l)" -eq "${CAPTURE_EXPECTED_PARTICLES}"
test -z "$(find "${CAPTURE_DIR}" -maxdepth 1 -type f -name '*.tmp.*' -print -quit)"
if test "${CAPTURE_BPREF_PRESCATTER}" = 1; then
    test "$(find "${BPREF_CAPTURE_DIR}" -maxdepth 1 -type f -name '*.bpre-v1.bin' | wc -l)" \
        -eq "${CAPTURE_EXPECTED_PARTICLES}"
    "${PYTHON}" "${REPO}/scripts/validate_relion_bpref_prescatter.py" \
        "${BPREF_CAPTURE_DIR}" --expected-particles "${CAPTURE_EXPECTED_PARTICLES}" \
        --output-json "${ROOT}/analysis/RELION_BPREF_PRESCATTER_VALIDATION.json"
fi
if test "${CAPTURE_BPREF_ACCUM}" = 1; then
    for half_rank in 1:1 2:2; do
        half=${half_rank%%:*}
        rank=${half_rank##*:}
        for kind in data weight; do
            test -s "${BPREF_ACCUM_DIR}/mstep_it001_rank${rank}_half${half}_c0_pre_lowres_join_bpref_${kind}.bin"
        done
        test -s "${BPREF_ACCUM_DIR}/mstep_it001_rank${rank}_half${half}_c0_pre_lowres_join_bpref_meta.txt"
    done
fi
if test "${CAPTURE_PPREF}" = 1; then
    # Both half projectors are constructed serially by the MPI master and use
    # rank -1/model 0.  The second (half-2) construction intentionally
    # overwrites the first; this focused capture therefore retains one file.
    test "$(find "${PPREF_CAPTURE_DIR}" -maxdepth 1 -type f -name 'ppref_iter001_*.bin' | wc -l)" -eq 1
fi
"${PYTHON}" "${REPO}/scripts/validate_relion_preprocess_capture.py" \
    "${CAPTURE_DIR}" --expected-particles "${CAPTURE_EXPECTED_PARTICLES}" \
    --expected-part-ids "${CAPTURE_PART_IDS}" \
    --expected-stack-indices "${CAPTURE_STACK_INDICES}" --expected-iteration 1 \
    --output "${ROOT}/analysis/RELION_PREPROCESS_CAPTURE_V1.json"
"${PYTHON}" - <<'PY' "${CAPTURE_DIR}" "${ROOT}/analysis/RELION_WAVG_CAPTURE_VALIDATION.json" "${CAPTURE_EXPECTED_PARTICLES}"
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

capture_dir = Path(sys.argv[1])
expected_particles = int(sys.argv[3])
records = []
for path in sorted(capture_dir.glob("*_storeWavg_wdiff2_pixels.bin")):
    match = re.fullmatch(r"img(\d+)_part(\d+)_storeWavg_wdiff2_pixels\.bin", path.name)
    assert match
    prefix = path.name.removesuffix("wdiff2_pixels.bin")
    arrays = {}
    for name, item_size in (
        ("wdiff2_pixels", 8),
        ("aa_pixels", 8),
        ("xa_pixels", 8),
        ("mresol", 4),
        ("fimg_real", 4),
        ("fimg_imag", 4),
        ("ctf", 4),
        ("trans_x", 4),
        ("trans_y", 4),
        ("weights", 4),
        ("eulers", 4),
    ):
        raw = (capture_dir / f"{prefix}{name}.bin").read_bytes()
        count = struct.unpack_from("<Q", raw)[0]
        assert len(raw) == 8 + count * item_size
        arrays[name] = count
    scalars = {}
    for name in (
        "valid_wdiff2_sum",
        "image_size",
        "current_size",
        "orientation_num",
        "translation_num",
    ):
        raw = (capture_dir / f"{prefix}{name}.bin").read_bytes()
        assert len(raw) == 16 and struct.unpack_from("<Q", raw)[0] == 1
        scalars[name] = struct.unpack_from("<d", raw, 8)[0]
    assert arrays["wdiff2_pixels"] == arrays["aa_pixels"] == arrays["xa_pixels"] == arrays["mresol"]
    assert scalars["image_size"] == arrays["wdiff2_pixels"]
    assert scalars["current_size"] == 56
    assert arrays["fimg_real"] == arrays["fimg_imag"] == arrays["ctf"] == scalars["image_size"]
    assert arrays["trans_x"] == arrays["trans_y"] == scalars["translation_num"]
    assert arrays["weights"] == scalars["orientation_num"] * scalars["translation_num"]
    assert arrays["eulers"] == scalars["orientation_num"] * 9
    records.append({
        "img_id": int(match.group(1)),
        "part_id": int(match.group(2)),
        "pixel_count": arrays["wdiff2_pixels"],
        "valid_wdiff2_sum": scalars["valid_wdiff2_sum"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
assert len(records) == expected_particles
Path(sys.argv[2]).write_text(json.dumps({"schema": "relion.acc.wavg_inputs.v2", "records": records}, indent=2) + "\n")
PY

"${PYTHON}" "${AUDIT_REPO}/scripts/audit_relion_control_reproducibility.py" \
    --control-dir "${CONTROL}" --candidate-dir "${RELION_OUT}" --iterations 1 2 \
    --map-fsc-auc-threshold 0.999999 \
    --output-json "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    --output-markdown "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.md"
"${PYTHON}" - <<'PY' "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json"
import json
import sys

report = json.load(open(sys.argv[1]))
assert all(item["map_stable"] for item in report["iterations"])
assert all(item["selected_topology_exact"] for item in report["iterations"])
PY

sha256sum "${CAPTURE_DIR}"/*.bin \
    "${CAPTURE_DIR}"/*.preprocess-v1.bin \
    "${ROOT}/analysis/RELION_PREPROCESS_CAPTURE_V1.json" \
    "${ROOT}/analysis/RELION_WAVG_CAPTURE_VALIDATION.json" \
    "${ROOT}/analysis/RELION_CAPTURE_INERTNESS.json" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
if test "${CAPTURE_BPREF_PRESCATTER}" = 1; then
    sha256sum "${BPREF_CAPTURE_DIR}"/*.bpre-v1.bin \
        "${ROOT}/analysis/RELION_BPREF_PRESCATTER_VALIDATION.json" \
        >> "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
fi
if test "${CAPTURE_BPREF_ACCUM}" = 1; then
    sha256sum "${BPREF_ACCUM_DIR}"/*_bpref_data.bin \
        "${BPREF_ACCUM_DIR}"/*_bpref_weight.bin \
        "${BPREF_ACCUM_DIR}"/*_bpref_meta.txt \
        >> "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
fi
if test "${CAPTURE_PPREF}" = 1; then
    sha256sum "${PPREF_CAPTURE_DIR}"/ppref_iter001_*.bin \
        >> "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
fi
printf '{"job_id":"%s","wall_s":%d,"particle_count":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${CAPTURE_EXPECTED_PARTICLES}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
