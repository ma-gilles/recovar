#!/usr/bin/env bash
# Long trajectory parity Slurm launcher.
#
# Submits 6 parallel GPU Slurm jobs:
#   recovar K=1 / K=2 / K=4 at nr_iter=N on 5k/128 fixtures
#   RELION  K=1 / K=2 / K=4 at nr_iter=N on the same fixtures
# Plus a summary job that diffs maps iter-by-iter.
#
# Defaults to nr_iter=200 (RELION's InitialModel default). Override with
# RUN_NR_ITER=N before invocation.
#
# Usage:
#   ./scripts/run_long_traj_parity_slurm.sh
#   ./scripts/run_long_traj_parity_slurm.sh --watch
#   RUN_NR_ITER=50 ./scripts/run_long_traj_parity_slurm.sh
#
# Outputs:
#   /scratch/gpfs/GILLES/mg6942/_agent_scratch/long_traj_parity_<TS>/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="long_traj_parity_${TIMESTAMP}_${RANDOM}"
SCRATCH_DIR="/scratch/gpfs/GILLES/mg6942/_agent_scratch/${RUN_ID}"
ACCOUNT="${SBATCH_ACCOUNT:-gilles}"
NR_ITER="${RUN_NR_ITER:-200}"

K2_DATA_DIR="/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k2_5k_128"
K4_DATA_DIR="/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k4_5k_128"
RELION_BIN="/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine"

mkdir -p "${SCRATCH_DIR}"
touch "${SCRATCH_DIR}/SAFE_TO_DELETE"

WATCH=0
for arg in "$@"; do
  case "$arg" in
    --watch) WATCH=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ------------------------------------------------------------------------
# RECOVAR job template
# ------------------------------------------------------------------------
make_recovar_script() {
    local job_name="$1" K="$2" data_dir="$3"
    local script_path="${SCRATCH_DIR}/${job_name}.sh"
    cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${SCRATCH_DIR}/${job_name}.out
#SBATCH --error=${SCRATCH_DIR}/${job_name}.err
#SBATCH --partition=cryoem
#SBATCH --account=${ACCOUNT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=23:59:00

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export TMPDIR="${SCRATCH_DIR}/tmp/${job_name}_\${SLURM_JOB_ID}"
export PIXI_HOME="${SCRATCH_DIR}/pixi_home/${job_name}_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${SCRATCH_DIR}/rattler_cache/${job_name}_\${SLURM_JOB_ID}"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}"

echo "=== ${job_name} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
git rev-parse HEAD
git symbolic-ref --short HEAD || echo '<detached>'

flock "${REPO_ROOT}/.pixi/install-recovar.lock" pixi run install-recovar
flock "${SCRATCH_DIR}/relion_bind_build.lock" bash -c 'rm -rf recovar/relion_bind/build && pixi run python recovar/relion_bind/build.py'

OUT_DIR="${SCRATCH_DIR}/recovar_K${K}"
mkdir -p "\${OUT_DIR}"

pixi run python scripts/run_ab_initio.py \\
  --i ${data_dir}/particles.star \\
  --datadir ${data_dir} \\
  --o "\${OUT_DIR}/run" \\
  --nr_iter ${NR_ITER} --K ${K} --sym C1 \\
  --particle_diameter 200 --tau2_fudge 4 --j 4 --random_seed 0 \\
  --healpix_order 1 --oversampling 1 \\
  --offset_range 6 --offset_step 2 \\
  --bootstrap_min_particles 1000 --sigma2_min_particles 1000 \\
  --padding_factor 1 --eager_images \\
  --image_batch_size 250 \\
  --rotation_block_size 5000

echo "TIME_END=\$(date -Is)"
ls -la "\${OUT_DIR}" | head -20
EOF
    chmod +x "${script_path}"
    echo "${script_path}"
}

# ------------------------------------------------------------------------
# RELION job template
# ------------------------------------------------------------------------
make_relion_script() {
    local job_name="$1" K="$2" data_dir="$3"
    local script_path="${SCRATCH_DIR}/${job_name}.sh"
    cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${SCRATCH_DIR}/${job_name}.out
#SBATCH --error=${SCRATCH_DIR}/${job_name}.err
#SBATCH --partition=cryoem
#SBATCH --account=${ACCOUNT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail
cd ${data_dir}
echo "=== ${job_name} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

OUT_DIR="${SCRATCH_DIR}/relion_K${K}"
mkdir -p "\${OUT_DIR}"

${RELION_BIN} --o "\${OUT_DIR}/run" \\
  --iter ${NR_ITER} --grad --denovo_3dref \\
  --i particles.star \\
  --ctf --K ${K} --sym C1 --flatten_solvent --zero_mask \\
  --dont_combine_weights_via_disc --pool 3 --pad 1 \\
  --particle_diameter 200 --oversampling 1 --healpix_order 1 \\
  --offset_range 6 --offset_step 2 --auto_sampling --tau2_fudge 4 \\
  --j 4 --gpu 0 --random_seed 0 --grad_write_iter 1

echo "TIME_END=\$(date -Is)"
ls -la "\${OUT_DIR}" | head -10
EOF
    chmod +x "${script_path}"
    echo "${script_path}"
}

# Submit recovar jobs
RECOVAR_K1_SCRIPT=$(make_recovar_script "longtraj_recovar_K1" 1 "${K2_DATA_DIR}")
RECOVAR_K2_SCRIPT=$(make_recovar_script "longtraj_recovar_K2" 2 "${K2_DATA_DIR}")
RECOVAR_K4_SCRIPT=$(make_recovar_script "longtraj_recovar_K4" 4 "${K4_DATA_DIR}")

# Submit relion jobs
RELION_K1_SCRIPT=$(make_relion_script "longtraj_relion_K1" 1 "${K2_DATA_DIR}")
RELION_K2_SCRIPT=$(make_relion_script "longtraj_relion_K2" 2 "${K2_DATA_DIR}")
RELION_K4_SCRIPT=$(make_relion_script "longtraj_relion_K4" 4 "${K4_DATA_DIR}")

REC_K1=$(sbatch --parsable "${RECOVAR_K1_SCRIPT}")
REC_K2=$(sbatch --parsable "${RECOVAR_K2_SCRIPT}")
REC_K4=$(sbatch --parsable "${RECOVAR_K4_SCRIPT}")
REL_K1=$(sbatch --parsable "${RELION_K1_SCRIPT}")
REL_K2=$(sbatch --parsable "${RELION_K2_SCRIPT}")
REL_K4=$(sbatch --parsable "${RELION_K4_SCRIPT}")

# ------------------------------------------------------------------------
# Summary job (depends on all 6)
# ------------------------------------------------------------------------
SUMMARY_SCRIPT="${SCRATCH_DIR}/longtraj_summary.sh"
cat > "${SUMMARY_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=longtraj_summary
#SBATCH --output=${SCRATCH_DIR}/summary.out
#SBATCH --error=${SCRATCH_DIR}/summary.err
#SBATCH --partition=cryoem
#SBATCH --account=${ACCOUNT}
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterany:${REC_K1}:${REC_K2}:${REC_K4}:${REL_K1}:${REL_K2}:${REL_K4}

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export TMPDIR="${SCRATCH_DIR}/tmp/summary_\${SLURM_JOB_ID}"
export PIXI_HOME="${SCRATCH_DIR}/pixi_home/summary_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${SCRATCH_DIR}/rattler_cache/summary_\${SLURM_JOB_ID}"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}"

echo "=== Long trajectory parity summary ==="
echo "Scratch: ${SCRATCH_DIR}"

for K in 1 2 4; do
  echo "--- K=\${K} ---"
  REC_DIR="${SCRATCH_DIR}/recovar_K\${K}"
  REL_DIR="${SCRATCH_DIR}/relion_K\${K}"
  echo "recovar iter MRCs: \$(ls "\${REC_DIR}"/run_it*_class*.mrc 2>/dev/null | wc -l)"
  echo "relion iter MRCs:  \$(ls "\${REL_DIR}"/run_it*_class*.mrc 2>/dev/null | wc -l)"
done

# Iter-by-iter map correlation
pixi run python - <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

SCRATCH_DIR = Path("${SCRATCH_DIR}")
NR_ITER = ${NR_ITER}

try:
    from recovar.utils import helpers as _h
    load_relion = _h.load_relion_volume
except Exception:
    import mrcfile
    def load_relion(p):
        with mrcfile.open(str(p), permissive=True) as m:
            return np.asarray(m.data, dtype=np.float64)

summary = {}
for K in (1, 2, 4):
    rec_root = SCRATCH_DIR / f"recovar_K{K}"
    rel_root = SCRATCH_DIR / f"relion_K{K}"
    if not rec_root.exists() or not rel_root.exists():
        summary[f"K{K}"] = {"missing": True}
        continue
    iters_done = []
    for it in range(1, NR_ITER + 1):
        rec_files = sorted(rec_root.glob(f"run_it{it:03d}_class*.mrc"))
        rel_files = sorted(rel_root.glob(f"run_it{it:03d}_class*.mrc"))
        if not rec_files or not rel_files:
            continue
        # Pair classes by Hungarian on first iter, then preserve permutation.
        rec_vols = [load_relion(f).reshape(-1) for f in rec_files]
        rel_vols = [load_relion(f).reshape(-1) for f in rel_files]
        # Compute corr matrix (n_classes x n_classes)
        n = min(len(rec_vols), len(rel_vols))
        ccm = np.zeros((n, n))
        for i, rv in enumerate(rec_vols[:n]):
            rv_n = rv - rv.mean()
            rv_norm = np.linalg.norm(rv_n) + 1e-30
            for j, lv in enumerate(rel_vols[:n]):
                lv_n = lv - lv.mean()
                ccm[i, j] = float(np.dot(rv_n, lv_n) / (rv_norm * (np.linalg.norm(lv_n) + 1e-30)))
        # Greedy diagonal match (good enough for trajectory plot)
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-ccm)
        matched_corrs = ccm[row, col]
        iters_done.append({
            "iter": it,
            "mean_corr": float(matched_corrs.mean()),
            "min_corr": float(matched_corrs.min()),
            "max_corr": float(matched_corrs.max()),
            "perm_recovar_to_relion": [int(c) for c in col],
        })
    summary[f"K{K}"] = {
        "n_iters_compared": len(iters_done),
        "iters": iters_done,
        "final_mean_corr": iters_done[-1]["mean_corr"] if iters_done else None,
        "final_min_corr": iters_done[-1]["min_corr"] if iters_done else None,
    }

out = SCRATCH_DIR / "longtraj_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"wrote {out}")

# Print compact table
for K, info in summary.items():
    if info.get("missing"):
        print(f"{K}: MISSING")
        continue
    print(f"\\n=== {K} (n_compared={info['n_iters_compared']}) ===")
    print(f"  final mean corr: {info['final_mean_corr']}")
    print(f"  final min corr:  {info['final_min_corr']}")
    iters = info["iters"]
    if iters:
        marks = [1, 5, 10, 25, 50, 100, 150, 200]
        for it in iters:
            if it["iter"] in marks or it["iter"] == iters[-1]["iter"]:
                print(f"  iter {it['iter']:3d}: mean={it['mean_corr']:.6f} min={it['min_corr']:.6f}")
PY

ls -la "${SCRATCH_DIR}"
EOF
chmod +x "${SUMMARY_SCRIPT}"
SUMMARY_JOB=$(sbatch --parsable "${SUMMARY_SCRIPT}")

echo "=== Long trajectory parity submitted ==="
echo "Scratch: ${SCRATCH_DIR}"
echo "  recovar K=1: ${REC_K1}"
echo "  recovar K=2: ${REC_K2}"
echo "  recovar K=4: ${REC_K4}"
echo "  relion  K=1: ${REL_K1}"
echo "  relion  K=2: ${REL_K2}"
echo "  relion  K=4: ${REL_K4}"
echo "  summary:     ${SUMMARY_JOB}"
echo "  nr_iter: ${NR_ITER}"

if [[ "$WATCH" == "1" ]]; then
    while squeue -h -j ${REC_K1},${REC_K2},${REC_K4},${REL_K1},${REL_K2},${REL_K4},${SUMMARY_JOB} > /dev/null 2>&1; do
        date
        squeue -j ${REC_K1},${REC_K2},${REC_K4},${REL_K1},${REL_K2},${REL_K4},${SUMMARY_JOB} 2>&1 | head -10
        sleep 60
    done
    cat "${SCRATCH_DIR}/summary.out"
fi
