#!/bin/bash -l
#SBATCH --job-name=sp128-2m-deconv-report
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_2m_deconv_report_resume_batched_%j.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_2m_deconv_report_resume_batched_%j.err

set -euo pipefail
set -x

REPO="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar"
RUN_DIR="/scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_noctf_noise0p01_n2m_raw_index_embsigma100_20260514"
PIPELINE="${RUN_DIR}/06_pipeline_index_scaled_meansigma_seed20260513"
STANDARD_ROOT="${RUN_DIR}/07_compute_state_standard_index_gtnoise_lazy"
DECONV_ROOT="${RUN_DIR}/07_compute_state_deconvolved_index_gtnoise_lam0p2_20_lazy_batched_job${SLURM_JOB_ID}"
REPORT_OUT="${RUN_DIR}/08_kernel_report_mask_crafted_level0p0126_cosine3_lazy_2m_batched_job${SLURM_JOB_ID}"
TARGET_POINT="${RUN_DIR}/target_latent_point_index_state0050.txt"
TARGET_VOL="${RUN_DIR}/04_ground_truth/gt_vol0050.mrc"
MASK="${RUN_DIR}/05_masks/mask_crafted_level0p0126_cosine3_128.mrc"
LOG="${RUN_DIR}/run_deconv_report_resume_batched_job${SLURM_JOB_ID}.log"
LAMBDA_GRID="0.2,0.21970823,0.24135853,0.26514227,0.2912697,0.31997174,0.35150212,0.38613955,0.42419018,0.46599036,0.51190958,0.56235374,0.61776872,0.67864435,0.74551874,0.81898301,0.89968653,0.98834267,1.0857351,1.1927247,1.3102571,1.4393713,1.5812086,1.7370227,1.908191,2.0962263,2.3027908,2.5297104,2.778991,3.0528359,3.3536659,3.6841399,4.0471793,4.445993,4.8841062,5.3653916,5.8941034,6.4749151,7.1129606,7.8138799,8.5838685,9.4297327,10.358949,11.379732,12.501104,13.732977,15.08624,16.572855,18.205964,20"

mkdir -p "$RUN_DIR" /scratch/gpfs/GILLES/mg6942/slurmo
cd "$REPO"

export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu
export RECOVAR_CACHE_DIR=

{
  echo "START $(date -Is)"
  echo "host=$(hostname)"
  echo "repo=$REPO"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "git_status_short=$(git status --short -- recovar/data_io/image_backends.py recovar/commands/compute_state.py recovar/commands/spike_kernel_report.py recovar/simulation/simulator.py recovar/utils/helpers.py | tr '\n' ';')"
  echo "run_dir=$RUN_DIR"
  echo "pipeline=$PIPELINE"
  echo "standard_root=$STANDARD_ROOT"
  echo "deconv_root=$DECONV_ROOT"
  echo "report_out=$REPORT_OUT"
  echo "target_point=$TARGET_POINT"
  echo "target_vol=$TARGET_VOL"
  echo "mask=$MASK"
  echo "n_images=2000000"
  echo "grid_size=128"
  echo "no_ctf=true"
  echo "image_noise_level=0.01"
  echo "embedding_noise_std=100"
  echo "lambda_grid=$LAMBDA_GRID"
  nvidia-smi || true
} | tee "$LOG"

module load cudatoolkit/12.8 || true
pixi install 2>&1 | tee -a "$LOG"
pixi run install-recovar 2>&1 | tee -a "$LOG"
PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip uninstall -y recovar 2>&1 | tee -a "$LOG" || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed 2>&1 | tee -a "$LOG"

PY_SITE="$("$PIXI_PY" -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$PY_SITE/nvidia/cuda_runtime/lib:$PY_SITE/nvidia/cuda_nvrtc/lib:$PY_SITE/nvidia/cublas/lib:$PY_SITE/nvidia/cusolver/lib:$PY_SITE/nvidia/cusparse/lib:$PY_SITE/nvidia/cudnn/lib:$PY_SITE/nvidia/cufft/lib:$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
PYTHON="$PIXI_PY" make -C recovar/cuda clean all 2>&1 | tee -a "$LOG"
export RECOVAR_CUDA_LIB="$REPO/recovar/cuda/libcuda_backproject.so"

"$PIXI_PY" - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
import jax
import recovar

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
jax_file = pathlib.Path(jax.__file__).resolve()
print("recovar.__file__=", recovar_file)
print("jax.__file__=", jax_file)
print("jax.devices()=", jax.devices())
print("jax.default_backend()=", jax.default_backend())
assert str(recovar_file).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(jax_file)
assert jax.default_backend() == "gpu"
PY

"$PIXI_PY" -m py_compile \
  recovar/data_io/image_backends.py \
  recovar/commands/compute_state.py \
  recovar/commands/spike_kernel_report.py

STD_COUNT="$(find "$STANDARD_ROOT/diagnostics/state000/estimates_filt" -maxdepth 1 -name '*.mrc' | wc -l)"
echo "standard_estimates_filt_count=$STD_COUNT" | tee -a "$LOG"
test "$STD_COUNT" -eq 50

/usr/bin/time -v "$PIXI_PY" -m recovar.commands.compute_state "$PIPELINE" \
  -o "$DECONV_ROOT" \
  --latent-points "$TARGET_POINT" \
  --zdim1 \
  --lazy \
  --save-all-estimates \
  --kernel-regression-mode deconvolved \
  --deconv-lambda-grid "$LAMBDA_GRID" \
  2>&1 | tee -a "$LOG"

DECONV_COUNT="$(find "$DECONV_ROOT/diagnostics/state000/estimates_filt" -maxdepth 1 -name '*.mrc' | wc -l)"
echo "deconv_estimates_filt_count=$DECONV_COUNT" | tee -a "$LOG"
test "$DECONV_COUNT" -eq 50

/usr/bin/time -v "$PIXI_PY" -m recovar.commands.spike_kernel_report \
  --standard-root "$STANDARD_ROOT" \
  --deconvolved-root "$DECONV_ROOT" \
  --target-volume "$TARGET_VOL" \
  --mask "$MASK" \
  --out-dir "$REPORT_OUT" \
  --pipeline-root "$PIPELINE" \
  --target-point "$TARGET_POINT" \
  --report-title "2M spike no CTF noise0.01 | raw index embedding + sigma100 | batched loader" \
  --expected-candidates 50 \
  --score-frequency-max 0.35 \
  --plot-frequency-max 0.40 \
  --selected-count 10 \
  2>&1 | tee -a "$LOG"

{
  echo "DONE $(date -Is)"
  echo "DECONV_ROOT=$DECONV_ROOT"
  echo "REPORT_OUT=$REPORT_OUT"
  echo "TARGET_VOL=$TARGET_VOL"
  echo "MASK=$MASK"
  echo "LOG=$LOG"
} | tee -a "$LOG"
