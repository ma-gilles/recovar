#!/usr/bin/env bash
#SBATCH --job-name=vdam-cutoff-panel
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-cutoff-panel-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${DATA_DIR:?set the frozen materialized fixture data directory}"
: "${RELION_DATA_STAR:?set the matching native RELION data STAR}"
: "${NATIVE_COMPONENTS_TSV:?set the native sigma2_noise_components.tsv}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified RECOVAR CUDA binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
TARGET_ITERATION=${TARGET_ITERATION:-1}
TARGET_HALF=${TARGET_HALF:--1}
IMAGE_SIZE=${IMAGE_SIZE:-128}
EXPECTED_PARTICLE_COUNT=${EXPECTED_PARTICLE_COUNT:-200}
RECOVAR_ROOT=${OUTPUT_ROOT}/recovar_prefix
SCORE_ROOT=${OUTPUT_ROOT}/recovar_score
PROVENANCE=${OUTPUT_ROOT}/provenance
ANALYSIS=${OUTPUT_ROOT}/analysis
PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
CUDA_BINARY=${REPO_ROOT}/recovar/cuda/libcuda_backproject.so
RELION_BIND_BINARY=${REPO_ROOT}/recovar/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -s "${DATA_DIR}/particles.star"
test -s "${RELION_DATA_STAR}"
test -s "${NATIVE_COMPONENTS_TSV}"
test -x "${PIXI_PY}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"
mkdir -p "${RECOVAR_ROOT}" "${SCORE_ROOT}" "${PROVENANCE}" "${ANALYSIS}"
test -z "$(find "${RECOVAR_ROOT}" "${SCORE_ROOT}" "${PROVENANCE}" "${ANALYSIS}" -mindepth 1 -print -quit)"

"${PIXI_PY}" - "${NATIVE_COMPONENTS_TSV}" "${RELION_DATA_STAR}" \
  "${TARGET_ITERATION}" "${TARGET_HALF}" "${EXPECTED_PARTICLE_COUNT}" \
  "${PROVENANCE}/particle_panel.tsv" <<'PY'
import sys
from pathlib import Path

import starfile

components, data_star, iteration, half, expected, output = sys.argv[1:]
prefix = f"acc_components\titer={int(iteration)}\t"
half_token = f"\thalfset={int(half)}\t"
part_ids = set()
for line in Path(components).read_text().splitlines():
    if not line.startswith(prefix) or half_token not in line:
        continue
    fields = dict(item.split("=", 1) for item in line.split("\t")[1:])
    if int(fields["optics_group"]) == 0:
        part_ids.add(int(fields["part_id"]))
part_ids = sorted(part_ids)
if len(part_ids) != int(expected):
    raise SystemExit(f"native particle count {len(part_ids)} != expected {expected}")
star = starfile.read(data_star)
particles = star["particles"] if isinstance(star, dict) else star
rows = []
for part_id in part_ids:
    image_name = str(particles.iloc[part_id]["rlnImageName"])
    image_number, separator, _stack = image_name.partition("@")
    if separator != "@" or int(image_number) <= 0:
        raise SystemExit(f"invalid image identity for part {part_id}: {image_name}")
    rows.append((part_id, int(image_number) - 1, image_name))
Path(output).write_text("part_id\toriginal_index\timage_name\n" + "".join(
    f"{part_id}\t{original_index}\t{image_name}\n"
    for part_id, original_index, image_name in rows
))
PY

original_indices=$(tail -n +2 "${PROVENANCE}/particle_panel.tsv" | cut -f2 | paste -sd, -)
test "$(awk 'END {print NR - 1}' "${PROVENANCE}/particle_panel.tsv")" -eq "${EXPECTED_PARTICLE_COUNT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORMS JAX_PLATFORM_NAME
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_LAUNCH_BLOCKING=0
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export JAX_COMPILATION_CACHE_DIR=${OUTPUT_ROOT}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${OUTPUT_ROOT}/runtime/${SLURM_JOB_ID}/rattler_cache
export RECOVAR_LOCAL_SCORE_DUMP_DIR=${SCORE_ROOT}
export RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES=${original_indices}
export RECOVAR_LOCAL_SCORE_DUMP_ITERATION=${TARGET_ITERATION}
export RECOVAR_LOCAL_SCORE_DUMP_OPERANDS=1
unset RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT RECOVAR_LOCAL_SCORE_DUMP_TARGET_ONLY
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

COMMAND=(
  "${PIXI_PY}" -m scripts.run_ab_initio
  --i "${DATA_DIR}/particles.star"
  --o "${RECOVAR_ROOT}/run"
  --nr_iter 200
  --grad_write_iter 1
  --K 1
  --tau2_fudge 4
  --sym C1
  --do_run_C1 1
  --particle_diameter 200
  --random_seed 0
  --healpix_order 1
  --oversampling 1
  --offset_range 6
  --offset_step 2
  --padding_factor 1
  --image_batch_size 500
  --datadir "${DATA_DIR}"
  --gpu 0
  --require_custom_cuda
  --diagnostic_stop_after_iteration "${TARGET_ITERATION}"
)
printf '%q ' "${COMMAND[@]}" > "${PROVENANCE}/recovar_command.sh"
printf '\n' >> "${PROVENANCE}/recovar_command.sh"
sha256sum "${REPO_ROOT}/scripts/run_vdam_cutoff_particle_panel.sbatch" \
  "${REPO_ROOT}/scripts/analyze_vdam_cutoff_particle_panel.py" \
  "${DATA_DIR}/particles.star" "${RELION_DATA_STAR}" "${NATIVE_COMPONENTS_TSV}" \
  "${CUDA_BINARY}" "${RELION_BIND_BINARY}" > "${PROVENANCE}/static_inputs.sha256"
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"

start_epoch=$(date +%s)
(
  cd "${REPO_ROOT}"
  "${COMMAND[@]}"
) > "${RECOVAR_ROOT}/runner.stdout" 2> "${RECOVAR_ROOT}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

iteration_tag=$(printf '%03d' "${TARGET_ITERATION}")
test -s "${RECOVAR_ROOT}/run_it${iteration_tag}_class001.mrc"
test "$(find "${SCORE_ROOT}" -maxdepth 1 -type f -name 'local_score_*.npz' | wc -l)" \
  -eq "${EXPECTED_PARTICLE_COUNT}"
(
cd "${REPO_ROOT}"
"${PIXI_PY}" -m scripts.analyze_vdam_cutoff_particle_panel \
  --native-components-tsv "${NATIVE_COMPONENTS_TSV}" \
  --score-directory "${SCORE_ROOT}" \
  --relion-data-star "${RELION_DATA_STAR}" \
  --iteration "${TARGET_ITERATION}" --half "${TARGET_HALF}" \
  --image-size "${IMAGE_SIZE}" --expected-particle-count "${EXPECTED_PARTICLE_COUNT}" \
  --output "${ANALYSIS}/cutoff_particle_panel.json"
)
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
printf '{"job_id":"%s","git_head":"%s","iteration":%d,"particle_count":%d,"recovar_wall_s":%d}\n' \
  "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${TARGET_ITERATION}" \
  "${EXPECTED_PARTICLE_COUNT}" "${wall_s}" > "${PROVENANCE}/run.json"
touch "${OUTPUT_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
