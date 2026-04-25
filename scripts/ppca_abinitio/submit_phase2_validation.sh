#!/usr/bin/env bash
# Phase 2 validation: post-EM ProjCov refit calibration check across
# Ribosembly q=4, IgG-1D q=2, IgG-RL q=2 at vol=32, n=1024.
# Submits one Slurm job that runs all 3 cases sequentially.

set -euo pipefail

WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${WORKDIR}/_agent_scratch/ppca_phase2_validation_${TS}"
mkdir -p "$OUT_ROOT"

git -C "$WORKDIR" rev-parse HEAD > "${OUT_ROOT}/git_commit.txt"
echo "$(date)" > "${OUT_ROOT}/started_at.txt"

SCRIPT="${OUT_ROOT}/validate.sbatch"
cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ppca-ph2-validate
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=${OUT_ROOT}/validate-%j.out

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

pixi run python scripts/ppca_abinitio/validate_eigenvalue_refit.py \\
    --out ${OUT_ROOT}/calibration.json \\
    --seed 0
EOF

chmod +x "$SCRIPT"
JID=$(sbatch --parsable "$SCRIPT")
echo "${JID}  ph2-validate" > "${OUT_ROOT}/job_id.txt"
echo "submitted job ${JID}; results will land at ${OUT_ROOT}/calibration.json"
echo "${OUT_ROOT}" > "${WORKDIR}/_agent_scratch/last_phase2_validation_root.txt"
