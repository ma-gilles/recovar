#!/usr/bin/env bash
#SBATCH --job-name=k1c7-it2-current
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail
: "${EXPERIMENT_ROOT:?set EXPERIMENT_ROOT}"
: "${EXPERIMENT_RUNTIME:?set EXPERIMENT_RUNTIME}"
: "${EXPECTED_HEAD:?set EXPECTED_HEAD}"
: "${EXPECTED_CUDA_SHA256:?set EXPECTED_CUDA_SHA256}"
: "${EXPECTED_LAUNCHER_SHA256:?set EXPECTED_LAUNCHER_SHA256}"
: "${EXPECTED_REPO_DIFF_SHA256:?set EXPECTED_REPO_DIFF_SHA256}"
ARM=${ARM:-ffi}
case "${ARM}" in ffi|sincosf|per_image_fft|exact_operands) ;; *) exit 2 ;; esac
TARGET_SOURCE_ROWS=${K1_CASE07_TARGET_SOURCE_ROWS:-}
EXACT_COARSE_COMPONENTS=${K1_EXACT_COARSE_COMPONENTS:-}

REPO=${K1_REPO_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810}
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/cases/7_anisotropic_100k_g256_white_noise1_bf80
CUDA_LIB=${K1_CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_cuda126_compute80_ptx_e791e875_20260814T0948ET/cuda128_sm90_control/build/libcuda_backproject_cuda128_sm90.so}
LAUNCHER=${K1_LAUNCHER_PATH:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810/scripts/run_k1_case07_current_it2_boundary.sbatch}
AUDIT_BUILDER=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case07_exact_boundary_coarseffi_ab_dd6acb5e_20260810T0440ET/jobs/make_audit_archive.py

test -f "${EXPERIMENT_ROOT}/SAFE_TO_DELETE"
test -f "${EXPERIMENT_RUNTIME}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test -s "${AUDIT_BUILDER}"
mkdir -p "${EXPERIMENT_ROOT}"/{analysis,logs,output,provenance} \
  "${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}"/{tmp,pixi_home,rattler_cache,jax_cache,pycache}
touch "${EXPERIMENT_ROOT}/output/SAFE_TO_DELETE"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
  case "${variable_name}" in RECOVAR_*|RELION_BPRE_*|RELION_FINE_*|RELION_DISPATCH_LOG) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/pycache

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
export CUDA_VISIBLE_DEVICES=0

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/relion_bind_build/shared
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_K1_RELION_EXACT_TRANSLATION_GRID=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${CASE}/data/particles.star
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
if [[ "${ARM}" != ffi ]]; then export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1; fi
if [[ "${ARM}" = per_image_fft ]]; then export RECOVAR_K1_RELION_PER_IMAGE_COARSE_FFT=1; fi
if [[ "${ARM}" = exact_operands ]]; then export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1; fi
if [[ -n "${EXACT_COARSE_COMPONENTS}" ]]; then
  export RECOVAR_K1_RELION_EXACT_COARSE_COMPONENTS=${EXACT_COARSE_COMPONENTS}
fi
if [[ -n "${TARGET_SOURCE_ROWS}" ]]; then
  mkdir -p "${EXPERIMENT_ROOT}/pass1" "${EXPERIMENT_ROOT}/pass2"
  export RECOVAR_SIGNIFICANCE_DUMP_DIR=${EXPERIMENT_ROOT}/pass1
  export RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=${TARGET_SOURCE_ROWS}
  export RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE=100
  export RECOVAR_PASS2_DUMP_DIR=${EXPERIMENT_ROOT}/pass2
  export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${TARGET_SOURCE_ROWS}
  export RECOVAR_PASS2_DUMP_CURRENT_SIZE=100
  export RECOVAR_PASS2_DUMP_CLASS=1
  export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
  export RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET=1
fi
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

nvidia-smi -q > "${EXPERIMENT_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${EXPERIMENT_ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
env | sort > "${EXPERIMENT_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
cd "${REPO}"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; root=pathlib.Path.cwd().resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(root); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper(); assert cuda_backproject.cuda_available(); print(jax.devices())"

command=("${PYTHON}" -m scripts.run_multi_iter_parity
  --relion_dir "${CASE}/relion_ref" --data_star "${CASE}/data/particles.star"
  --iter 1 --max_iter 1 --continuous-relion-noise-state --skip_final_iteration
  --image_batch_size 187 --rotation_block_size 8192 --image-fourier-backend relion_cuda
  --output_dir "${EXPERIMENT_ROOT}/output"
  --save_intermediates_dir "${EXPERIMENT_ROOT}/output/intermediates")
printf '%q ' "${command[@]}" > "${EXPERIMENT_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${EXPERIMENT_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
set +e
"${command[@]}" > "${EXPERIMENT_ROOT}/output/run.log" 2>&1
command_status=$?
set -e
if [[ -n "${TARGET_SOURCE_ROWS}" ]]; then
  IFS=',' read -r -a target_rows <<< "${TARGET_SOURCE_ROWS}"
  expected_count=${#target_rows[@]}
  test "$(find "${EXPERIMENT_ROOT}/pass1" -maxdepth 1 -name 'significance_*.npz' | wc -l)" -eq "${expected_count}"
  test "$(find "${EXPERIMENT_ROOT}/pass2" -maxdepth 1 -name 'pass2_*.npz' | wc -l)" -eq "${expected_count}"
  test "${command_status}" -ne 0
  rg -q 'requested RECOVAR pass-2 dump target set was written' "${EXPERIMENT_ROOT}/output/run.log"
  sha256sum "${EXPERIMENT_ROOT}"/pass1/*.npz "${EXPERIMENT_ROOT}"/pass2/*.npz \
    > "${EXPERIMENT_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
  touch "${EXPERIMENT_ROOT}/SUCCESS_${SLURM_JOB_ID}"
  exit 0
fi
test "${command_status}" -eq 0
"${PYTHON}" "${AUDIT_BUILDER}" --output-dir "${EXPERIMENT_ROOT}/output"
"${PYTHON}" -m scripts.audit_em_particle_state_distribution \
  --recovar-results "${EXPERIMENT_ROOT}/analysis/refinement_results_audit_source_order.npz" \
  --recovar-particles-star "${CASE}/data/particles.star" \
  --relion-star "2:${CASE}/relion_ref/run_it002_data.star" \
  --recovar-iteration 0 --relion-iteration-offset 2 \
  --output-json "${EXPERIMENT_ROOT}/analysis/particle.json" \
  --output-npz "${EXPERIMENT_ROOT}/analysis/particle_arrays.npz" \
  --output-hash-manifest "${EXPERIMENT_ROOT}/analysis/particle.json.sha256"
sha256sum "${EXPERIMENT_ROOT}/output/refinement_results.npz" \
  "${EXPERIMENT_ROOT}/analysis/particle.json" \
  "${EXPERIMENT_ROOT}/analysis/particle_arrays.npz" \
  > "${EXPERIMENT_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${EXPERIMENT_ROOT}/SUCCESS_${SLURM_JOB_ID}"
