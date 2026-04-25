#!/usr/bin/env bash
# Phase 3.3 vol=64 / healpix_order=2 scaling smoke test.
#
# Single Ribosembly q=4 run at vol=64, n=1024, order=2, weighted SVD
# warmstart, no anneal, 30 iters. Acceptance:
#   - completes without OOM on H100
#   - hun >= 0.70 (gracefully degraded vs vol=32 reference)
#   - wall time <= 60 min
#
# Output: cells/result.json + slurm log + memory peak from Slurm

set -euo pipefail

WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${WORKDIR}/_agent_scratch/ppca_phase3_vol64_smoke_${TS}"
mkdir -p "$OUT_ROOT/cells"

git -C "$WORKDIR" rev-parse HEAD > "${OUT_ROOT}/git_commit.txt"
echo "$(date)" > "${OUT_ROOT}/started_at.txt"

# Print the predicted memory budget into the run record for provenance.
"$WORKDIR/.pixi/envs/default/bin/python3.11" -c "
from recovar.em.ppca_abinitio.memory_model import format_memory_report
print(format_memory_report(
    n_img=1024, volume_shape=(64, 64, 64), image_shape=(64, 64),
    n_rot=1944, n_trans=5, q=4,
))
" > "${OUT_ROOT}/predicted_memory.txt" 2>&1 || echo "(memory model preview not available)" > "${OUT_ROOT}/predicted_memory.txt"

SCRIPT="${OUT_ROOT}/smoke.sbatch"
cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ppca-ph3-vol64-smoke
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=01:30:00
#SBATCH --output=${OUT_ROOT}/smoke-%j.out

set -euo pipefail

cd ${WORKDIR}

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${WORKDIR}/.tmp/slurm_\${SLURM_JOB_ID}"
export PIXI_HOME="${WORKDIR}/.tmp/pixi_home_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${WORKDIR}/.tmp/rattler_cache_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR" "\$PIXI_HOME" "\$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

PIXI_PY="\$(pixi run which python)"
export PATH="\$(dirname "\$PIXI_PY"):\$PATH"

"\$PIXI_PY" -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
print('ENV_OK — devices:', jax.devices())
"

echo '=== predicted memory budget ==='
cat ${OUT_ROOT}/predicted_memory.txt
echo

t0=\$(date +%s)
pixi run python scripts/ppca_abinitio/run_cryobench.py \\
    --dataset Ribosembly --vol 64 --n-images 1024 --healpix-order 2 \\
    --u-init svd --svd-warmstart weighted --mu-init perturbed \\
    --external-mode discrete_volumes --n-burnin 0 \\
    --q 4 --n-joint 30 --sigma 0.01 \\
    --s-init flat --ridge-mode scalar --anneal-schedule none \\
    --seed 0 --init-seed 0 \\
    --save-results ${OUT_ROOT}/cells/result.json
t1=\$(date +%s)
echo "wall-clock seconds: \$((t1 - t0))" > ${OUT_ROOT}/wall_time_seconds.txt
EOF

chmod +x "$SCRIPT"
JID=$(sbatch --parsable "$SCRIPT")
echo "${JID}  vol64-smoke" > "${OUT_ROOT}/job_id.txt"
echo "submitted job ${JID}"
echo "${OUT_ROOT}" > "${WORKDIR}/_agent_scratch/last_phase3_smoke_root.txt"
