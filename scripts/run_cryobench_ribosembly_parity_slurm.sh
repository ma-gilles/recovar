#!/bin/bash
#SBATCH --job-name=ribo-k16-parity
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/ribo-k16-parity-%j.out

set -euo pipefail

SNR="${SNR:?Set SNR=1 or SNR=10 before sbatch}"
GRID_SIZE="${GRID_SIZE:-256}"
N_IMAGES="${N_IMAGES:-100000}"
PDB_DIR="${PDB_DIR:-/home/mg6942/mytigress/cryobench2/Ribosembly/pdbs}"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/GILLES/mg6942/em_relion_proj}"
DISC_TYPE="${DISC_TYPE:-cubic}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/ribosembly_allk_g${GRID_SIZE}_n${N_IMAGES}_snr${SNR}_${DISC_TYPE}}"
RELION_REFINE="${RELION_REFINE:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}"
RELION_REFINE_MPI="${RELION_REFINE_MPI:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi}"
RELION_MODULE="${RELION_MODULE:-relion/5.0.1/gcc-11.5.0-gpu}"
PARTICLE_DIAMETER="${PARTICLE_DIAMETER:-380}"
CLASS3D_HEALPIX_ORDER="${CLASS3D_HEALPIX_ORDER:-2}"
CLASS3D_OFFSET_RANGE="${CLASS3D_OFFSET_RANGE:-3}"
CLASS3D_OFFSET_STEP="${CLASS3D_OFFSET_STEP:-1}"
CLASS3D_ITER="${CLASS3D_ITER:-1}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-50}"
ROTATION_BLOCK_SIZE="${ROTATION_BLOCK_SIZE:-1000}"
MEAN_CORR_GATE="${MEAN_CORR_GATE:-0.999}"
CLASS_ACC_GATE="${CLASS_ACC_GATE:-0.999}"
PMAX_ABS_MEAN_GATE="${PMAX_ABS_MEAN_GATE:-0.002}"

AGENT_ID="ribo_k16_snr${SNR}_${SLURM_JOB_ID}"
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}"
ENV_SETUP_LOCK="/scratch/gpfs/GILLES/mg6942/locks/recovar_relion_parity_env_setup.lock"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" /scratch/gpfs/GILLES/mg6942/slurmo "$DATA_DIR" "$(dirname "$ENV_SETUP_LOCK")"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"

echo "=== Repo provenance ==="
pwd
whoami
git rev-parse HEAD
git status --short

echo "=== Environment setup ==="
(
  flock 9
  pixi install
  PIXI_PY="$(pixi run which python)"
  "$PIXI_PY" -m pip uninstall -y recovar || true
  "$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed
  PYTHON="$PIXI_PY" make -C recovar/cuda clean all
  pixi run smoke-import-recovar
) 9>"$ENV_SETUP_LOCK"
PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -c "import pathlib,recovar,jax; repo=pathlib.Path.cwd().resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); print(jax.devices())"

echo "=== Prepare Ribosembly PDB multiclass benchmark ==="
"$PIXI_PY" scripts/prepare_cryobench_pdb_multiclass_relion_parity_benchmark.py \
  --pdb-dir "$PDB_DIR" \
  --output-dir "$DATA_DIR" \
  --n-images "$N_IMAGES" \
  --grid-size "$GRID_SIZE" \
  --snr "$SNR" \
  --disc-type "$DISC_TYPE" \
  --relion-normalize \
  --streaming-mmap

N_CLASSES="$("$PIXI_PY" - <<PY
import json
from pathlib import Path
manifest = json.loads((Path("$DATA_DIR") / "class_manifest.json").read_text())
print(len(manifest))
PY
)"
echo "N_CLASSES=$N_CLASSES"

echo "=== RELION Class3D one-iteration K-class reference ==="
CLASS3D_DIR="$DATA_DIR/relion_class3d_k${N_CLASSES}_os0_ref"
if [ ! -f "$CLASS3D_DIR/run_it001_model.star" ]; then
  mkdir -p "$CLASS3D_DIR"
  (
    export PS1="${PS1-}"
    module load "$RELION_MODULE"
    command -v mpirun
    cd "$DATA_DIR"
    mpirun -n 3 "$RELION_REFINE_MPI" \
      --i particles.star \
      --ref reference_init_classes_relion.star \
      --o "$CLASS3D_DIR/run" \
      --iter "$CLASS3D_ITER" \
      --tau2_fudge 4 \
      --particle_diameter "$PARTICLE_DIAMETER" \
      --K "$N_CLASSES" \
      --flatten_solvent \
      --zero_mask \
      --ctf \
      --norm \
      --scale \
      --sym C1 \
      --oversampling 0 \
      --healpix_order "$CLASS3D_HEALPIX_ORDER" \
      --offset_range "$CLASS3D_OFFSET_RANGE" \
      --offset_step "$CLASS3D_OFFSET_STEP" \
      --pad 2 \
      --pool 3 \
      --dont_combine_weights_via_disc \
      --gpu 0:0:0 \
      --j 4 \
      2>&1 | tee "$CLASS3D_DIR/run.log"
  )
else
  echo "Reusing existing RELION Class3D output at $CLASS3D_DIR"
fi

echo "=== RECOVAR K-class replay against RELION Class3D ==="
REPLAY_DIR="$DATA_DIR/recovar_k${N_CLASSES}_class3d_replay"
if [ ! -f "$REPLAY_DIR/summary.json" ]; then
  mkdir -p "$REPLAY_DIR"
  "$PIXI_PY" scripts/run_k_class_parity.py \
    --relion-dir "$CLASS3D_DIR" \
    --data-star "$DATA_DIR/particles.star" \
    --prev-iter 0 \
    --target-iter 1 \
    --output-dir "$REPLAY_DIR" \
    --image-batch-size "$IMAGE_BATCH_SIZE" \
    --rotation-block-size "$ROTATION_BLOCK_SIZE" \
    2>&1 | tee "$REPLAY_DIR/run.log"
else
  echo "Reusing existing RECOVAR replay summary at $REPLAY_DIR/summary.json"
fi

echo "=== K-class parity gates ==="
"$PIXI_PY" - <<PY | tee "$REPLAY_DIR/parity_gates.txt"
import json
import math
from pathlib import Path

summary = json.loads((Path("$REPLAY_DIR") / "summary.json").read_text())
mean_corr_gate = float("$MEAN_CORR_GATE")
class_acc_gate = float("$CLASS_ACC_GATE")
pmax_abs_mean_gate = float("$PMAX_ABS_MEAN_GATE")
expected_n_classes = int("$N_CLASSES")
expected_n_images = int("$N_IMAGES")
mean_corr = float(summary["best_permutation"]["mean_corr"])
class_acc = float(summary["class_assignment_accuracy_after_permutation"])
pmax_abs_mean = float(summary["pmax"]["abs_mean"])
n_classes = int(summary["n_classes"])
n_images = int(summary["n_images"])
nonfinite_corr_count = int(summary["best_permutation"].get("nonfinite_corr_count", 0))
chosen_nonfinite_corr_count = int(summary["best_permutation"].get("chosen_nonfinite_corr_count", 0))
print(f"n_classes={n_classes} n_images={n_images}")
print(f"mean_corr={mean_corr:.9f} gate={mean_corr_gate:.9f}")
print(f"class_acc={class_acc:.9f} gate={class_acc_gate:.9f}")
print(f"pmax_abs_mean={pmax_abs_mean:.9g} gate={pmax_abs_mean_gate:.9g}")
print(f"nonfinite_corr_count={nonfinite_corr_count}")
print(f"chosen_nonfinite_corr_count={chosen_nonfinite_corr_count}")
if n_classes != expected_n_classes:
    raise SystemExit(f"n_classes mismatch: {n_classes} != {expected_n_classes}")
if n_images != expected_n_images:
    raise SystemExit(f"n_images mismatch: {n_images} != {expected_n_images}")
if nonfinite_corr_count:
    raise SystemExit(f"nonfinite correlation entries present: {nonfinite_corr_count}")
if chosen_nonfinite_corr_count:
    raise SystemExit(f"chosen class permutation contains nonfinite correlations: {chosen_nonfinite_corr_count}")
if not math.isfinite(mean_corr):
    raise SystemExit("mean map correlation is not finite")
if not math.isfinite(class_acc):
    raise SystemExit("class assignment accuracy is not finite")
if not math.isfinite(pmax_abs_mean):
    raise SystemExit("Pmax abs mean is not finite")
if mean_corr < mean_corr_gate:
    raise SystemExit("mean map correlation gate failed")
if class_acc < class_acc_gate:
    raise SystemExit("class assignment accuracy gate failed")
if pmax_abs_mean > pmax_abs_mean_gate:
    raise SystemExit("Pmax abs mean gate failed")
PY

echo "=== RELION InitialModel K-class reference run ==="
IM_DIR="$DATA_DIR/relion_initialmodel_k${N_CLASSES}_it001"
if [ ! -f "$IM_DIR/run_it001_model.star" ]; then
  mkdir -p "$IM_DIR"
  (
    export PS1="${PS1-}"
    module load "$RELION_MODULE"
    cd "$DATA_DIR"
    "$RELION_REFINE" \
      --o "$IM_DIR/run" \
      --iter 1 \
      --grad \
      --denovo_3dref \
      --i particles.star \
      --ctf \
      --K "$N_CLASSES" \
      --sym C1 \
      --flatten_solvent \
      --zero_mask \
      --dont_combine_weights_via_disc \
      --pool 3 \
      --pad 1 \
      --particle_diameter "$PARTICLE_DIAMETER" \
      --oversampling 1 \
      --healpix_order 1 \
      --offset_range 6 \
      --offset_step 2 \
      --auto_sampling \
      --tau2_fudge 4 \
      --j 4 \
      --gpu 0 \
      2>&1 | tee "$IM_DIR/run.log"
  )
else
  echo "Reusing existing RELION InitialModel output at $IM_DIR"
fi

"$PIXI_PY" - <<PY
import json
from pathlib import Path
status = {
    "relion_initial_model_dir": "$IM_DIR",
    "recovar_initial_model_k16_parity": "unsupported",
    "reason": (
        "recovar.em.initial_model.gpu_pipeline.run_iter_gpu_vdam currently "
        "hardcodes K=1 in the VDAM M-step/momenta path; K=16 InitialModel "
        "parity needs a separate multi-class VDAM implementation."
    ),
}
Path("$DATA_DIR/initial_model_recovar_status.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status, indent=2))
PY

echo "=== Completed Ribosembly parity job ==="
echo "DATA_DIR=$DATA_DIR"
echo "CLASS3D_DIR=$CLASS3D_DIR"
echo "REPLAY_DIR=$REPLAY_DIR"
echo "IM_DIR=$IM_DIR"
