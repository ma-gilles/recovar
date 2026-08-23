#!/usr/bin/env bash
#SBATCH --job-name=k1c05-r38594-fine
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row38594_native_fine_20260822T1710ET/logs/science-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_row38594_native_fine_20260822T1710ET/logs/science-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --constraint=h100
#SBATCH --nodelist=della-h19g1
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail

RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to the immutable diagnostic output root}
RUNTIME_ROOT=${RUNTIME_ROOT:?set RUNTIME_ROOT to the job runtime root}
EXPECTED_LAUNCHER_SHA256=${EXPECTED_LAUNCHER_SHA256:?set launcher hash}
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
CASE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/5_very_high_noise_100k_g256_white_noise10_bf80
DATA_DIR=${CASE_ROOT}/data
CONTROL_RELION=${CASE_ROOT}/relion_ref
RECOVAR_NPZ=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_finepass_targets_20260724T143000Z/pass2/pass2_orig038594_cs056.npz
PATCHED_RELION=/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi
EXPECTED_PATCHED_RELION_SHA256=a9a961340af621d1cd581ccea2e96274f978b53213f9606c9f0e30d271903b8c
EXPECTED_SOURCE_HEAD=f4541486d6c11d5872e97b29954e9fd56d66217b
EXPECTED_CAPTURE_GPU_UUID=GPU-0d7b80c7-fef8-e346-6332-de36ae1af518
PYTHON=${SOURCE_ROOT}/.pixi/envs/default/bin/python
CONTROL_ROOT=${RUN_ROOT}/control_root
CAPTURE_ROOT=${RUN_ROOT}/capture_root
RELION_OUT=${CAPTURE_ROOT}/relion
RELION_DUMP=${RUN_ROOT}/relion_dump
INERTNESS_JSON=${RUN_ROOT}/analysis/ROW38594_CAPTURE_INERTNESS.json
COMPARISON_JSON=${RUN_ROOT}/analysis/ROW38594_IT1_FINE_VS_RELION.json
LAUNCHER=${SOURCE_ROOT}/scripts/run_k1_case05_row38594_native_fine.sbatch

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
unset RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER RECOVAR_FINAL_ALL_DATA_GRID_CORRECT
unset RECOVAR_USE_FLOAT64_SCORING RECOVAR_USE_FLOAT64_PROJECTIONS
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
mkdir -p \
  "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
  "${RELION_DUMP}" "${RELION_OUT}" "${RUN_ROOT}/analysis" \
  "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance" "${CONTROL_ROOT}"
touch "${RUN_ROOT}/SAFE_TO_DELETE" "${RUNTIME_ROOT}/SAFE_TO_DELETE"
ln -s "${CONTROL_RELION}" "${CONTROL_ROOT}/relion"

test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "${EXPECTED_SOURCE_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${PATCHED_RELION}" | awk '{print $1}')" = "${EXPECTED_PATCHED_RELION_SHA256}"
grep -aFq '[DUMP] enabled: dir=' "${PATCHED_RELION}"
test -x "${PYTHON}"
test "078363915af39e644c0540d6e1ad020e41d94c3c7ee298e28cf39d4788cd384c" = \
  "$(sha256sum "${DATA_DIR}/particles.star" | awk '{print $1}')"
test "fbb7bbcba5ad804a6d4d78bac87f024f3a876c82f21f9397c5278961b30eda47" = \
  "$(sha256sum "${DATA_DIR}/reference_init_relion.mrc" | awk '{print $1}')"
test "c024a27a8b2f8071a1015e845ed28a938e6d7b3ece309a8789d07b702fddbeb6" = \
  "$(sha256sum "${RECOVAR_NPZ}" | awk '{print $1}')"

mapfile -t visible_gpu_rows < <(
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
)
test "${#visible_gpu_rows[@]}" -eq 4
selected_gpu_index=""
for gpu_row in "${visible_gpu_rows[@]}"; do
  gpu_index=${gpu_row%%,*}
  gpu_uuid=${gpu_row#*, }
  if [[ "${gpu_uuid}" = "${EXPECTED_CAPTURE_GPU_UUID}" ]]; then
    selected_gpu_index=${gpu_index}
  fi
done
test -n "${selected_gpu_index}"
export CUDA_VISIBLE_DEVICES=${selected_gpu_index}
test "$(nvidia-smi -i "${selected_gpu_index}" --query-gpu=uuid --format=csv,noheader)" = \
  "${EXPECTED_CAPTURE_GPU_UUID}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv \
  > "${RUN_ROOT}/provenance/physical_gpu_inventory.csv"
{
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "source_root=${SOURCE_ROOT}"
  echo "source_head=${EXPECTED_SOURCE_HEAD}"
  echo "case_root=${CASE_ROOT}"
  echo "control_relion=${CONTROL_RELION}"
  echo "recovar_capture=${RECOVAR_NPZ}"
  echo "target_original_index_zero_based=38594"
  echo "target_stack_index_one_based=38595"
  echo "target_image_identity=38595@particles.256.mrcs"
  echo "target_iteration_one_based=1"
  echo "target_current_size=56"
  echo "capture_gpu_uuid=${EXPECTED_CAPTURE_GPU_UUID}"
  echo "selected_physical_gpu_index=${selected_gpu_index}"
  echo "patched_relion=${PATCHED_RELION}"
  echo "patched_relion_sha256=${EXPECTED_PATCHED_RELION_SHA256}"
  echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=<unset>"
  git -C "${SOURCE_ROOT}" status --short
  sha256sum "${LAUNCHER}" \
    "${SOURCE_ROOT}/scripts/analyze_relion_control_capture_inertness.py" \
    "${SOURCE_ROOT}/scripts/compare_relion_recovar_estep_dump.py"
} > "${RUN_ROOT}/provenance/preflight.txt"

(
unset LD_LIBRARY_PATH
source /etc/profile.d/modules.sh
export PS1=${PS1:-}
set +u
module load relion/5.0.1/gcc-11.5.0-gpu
set -u
export CUDA_VISIBLE_DEVICES=${selected_gpu_index}
export RELION_DUMP_DIR=${RELION_DUMP}
export RELION_DUMP_STACK_INDEX=38595
export RELION_DUMP_ITER=1
export RELION_ACC_DUMP_VERBOSE=1
export TMPDIR=${RUNTIME_ROOT}/relion_tmp
export TMP=${TMPDIR}
export TEMP=${TMPDIR}
export OMPI_MCA_orte_tmpdir_base=${TMPDIR}
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
mkdir -p "${TMPDIR}"
cd "${DATA_DIR}"
mpirun -n 3 "${PATCHED_RELION}" \
  --i particles.star \
  --ref reference_init_relion.mrc \
  --o "${RELION_OUT}/run" \
  --auto_refine \
  --split_random_halves \
  --particle_diameter 200 \
  --ini_high 30 \
  --firstiter_cc \
  --ctf \
  --flatten_solvent \
  --zero_mask \
  --low_resol_join_halves 40 \
  --norm \
  --scale \
  --healpix_order 3 \
  --offset_range 3.0 \
  --offset_step 1.0 \
  --oversampling 1 \
  --pad 2 \
  --random_seed 1705 \
  --auto_iter_max 1 \
  --pool 3 \
  --gpu 0 \
  --j 4
) 2>&1 | tee "${RUN_ROOT}/logs/relion.log"

test -s "${RELION_DUMP}/pass1_firstiter_cc_exp_Mweight_raw_preonehot.bin"
test -s "${RELION_DUMP}/pass1_firstiter_cc_raw_rot_idx.bin"
test -s "${RELION_DUMP}/pass1_firstiter_cc_raw_trans_idx.bin"
test -s "${RELION_DUMP}/pass1_firstiter_cc_raw_ihidden_overs.bin"
test -s "${RELION_OUT}/run_it001_data.star"
find "${RELION_DUMP}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${RUN_ROOT}/provenance/relion_dump.sha256"

cd "${SOURCE_ROOT}"
set +e
"${PYTHON}" scripts/analyze_relion_control_capture_inertness.py \
  --control-root "${CONTROL_ROOT}" \
  --capture-root "${CAPTURE_ROOT}" \
  --target-original-index 38594 \
  --target-image-identity '38595@particles.256.mrcs' \
  --expected-particle-count 100000 \
  --gpu-uuid "${EXPECTED_CAPTURE_GPU_UUID}" \
  --output-json "${INERTNESS_JSON}" \
  > "${RUN_ROOT}/logs/inertness.log" 2>&1
inertness_rc=$?
set -e
test -s "${INERTNESS_JSON}"
"${PYTHON}" - "${INERTNESS_JSON}" "${inertness_rc}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["particle_count"] == 100000
assert report["target"]["mismatch_fields"] == []
assert all(row["fsc_auc"] >= 0.999999 for row in report["half_map_comparison"])
assert int(sys.argv[2]) in (0, 1)
print(
    "capture qualification:",
    "global_status=", report["status"],
    "target_exact=True",
    "half_map_fsc_auc=", [row["fsc_auc"] for row in report["half_map_comparison"]],
)
PY

"${PYTHON}" scripts/compare_relion_recovar_estep_dump.py \
  --relion-dump-dir "${RELION_DUMP}" \
  --recovar-pass2-npz "${RECOVAR_NPZ}" \
  --recovar-class-index 0 \
  --relion-firstiter-pass pass1 \
  --match-mode matrix \
  --output-json "${COMPARISON_JSON}"

"${PYTHON}" - "${COMPARISON_JSON}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
assert result["match_mode"] == "matrix"
assert result["recovar_current_size"] == 56
assert result["recovar_original_index"] == 38594
assert result["recovar_candidate_count"] > 0
assert result["relion_candidate_count"] > 0
assert result["common_candidate_count"] > 0
print(
    "complete firstiter fine comparison:",
    "recovar_top_key=", result["recovar_top_key"],
    "relion_top_key=", result["relion_top_key"],
    "cross_top=", result["cross_top_candidate_details"],
)
PY

test "$(nvidia-smi -i "${selected_gpu_index}" --query-gpu=uuid --format=csv,noheader)" = \
  "${EXPECTED_CAPTURE_GPU_UUID}"
sha256sum \
  "${COMPARISON_JSON}" "${RECOVAR_NPZ}" "${INERTNESS_JSON}" \
  "${RUN_ROOT}/provenance/relion_dump.sha256" \
  > "${RUN_ROOT}/analysis/ROW38594_IT1_FINE_VS_RELION.sha256"
touch "${RUN_ROOT}/SCIENCE_COMPLETE"
echo "case-5 iteration-1 particle-38594 same-H100 fine-score diagnostic complete"
