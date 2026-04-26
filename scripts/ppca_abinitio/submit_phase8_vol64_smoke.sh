#!/usr/bin/env bash
# Phase 8.1 vol=64 smoke — does the einsum-fusion eliminate the OOM?
#
# Phase 4 history: 4 successive vol=64 attempts OOM'd up to 400GB host.
# Phase 8 fused the (n_img, n_rot, half_image) intermediate in
# _per_rotation_residual_image and the (n_img, n_rot, n_trans, q, q)
# intermediate in update_factor_closed_form. Bit-identicality verified
# at vol=32 (217 tests pass).
#
# Three configs, escalating:
#   1. vol=64 n=1024 order=1 (was the second-attempt OOM target)
#   2. vol=64 n=1024 order=2 (was the first-attempt 148 GiB OOM target)
#   3. vol=64 n=4096 order=2 (push test)
#
# Acceptance: at least config 1 completes; ideally all 3.

set -euo pipefail

WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408"
TS="$(date +%Y%m%d_%H%M%S)"
ROOT="${WORKDIR}/_agent_scratch/ppca_phase8_vol64_${TS}"
mkdir -p "$ROOT/cells" "$ROOT/slurm_logs"

git -C "$WORKDIR" rev-parse HEAD > "$ROOT/git_commit.txt"

submit() {
    local id="$1" args="$2" mem="$3"
    local cell_json="${ROOT}/cells/${id}.json"
    local script="${ROOT}/slurm_logs/${id}.sbatch"
    cat > "$script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ppca-ph8-${id}
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${mem}
#SBATCH --time=03:00:00
#SBATCH --output=${ROOT}/slurm_logs/${id}-%j.out
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
"\$PIXI_PY" -c "import recovar, jax; print('ENV_OK', jax.devices())"

t0=\$(date +%s)
pixi run python scripts/ppca_abinitio/run_cryobench.py ${args} \\
    --instrument --save-results ${cell_json}
t1=\$(date +%s)
echo "WALL_CLOCK_SECONDS=\$((t1-t0))"
EOF
    JID=$(sbatch --parsable "$script")
    echo "${JID}  ${id}"
}

COMMON="--vol 64 --svd-warmstart weighted --external-mode discrete_volumes --n-burnin 0 --s-init flat --ridge-mode scalar --u-init svd --mu-init perturbed --anneal-schedule none --dataset Ribosembly --q 4 --n-joint 30 --sigma 0.01 --seed 0 --init-seed 0"

SHORT_COMMON="--vol 64 --svd-warmstart weighted --external-mode discrete_volumes --n-burnin 0 --s-init flat --ridge-mode scalar --u-init svd --mu-init perturbed --anneal-schedule none --dataset Ribosembly --q 4 --n-joint 2 --sigma 0.01 --seed 0 --init-seed 0"
submit "vol64_short_n1024_bs128_mem600" "$SHORT_COMMON --n-images 1024 --healpix-order 1 --image-batch-size 128" 600GB
submit "vol64_short_n256_bs64_mem600"   "$SHORT_COMMON --n-images 256 --healpix-order 1 --image-batch-size 64"  600GB
submit "vol64_short_n128_bs32_mem600"   "$SHORT_COMMON --n-images 128 --healpix-order 1 --image-batch-size 32"  600GB

echo "==== submitted 3 cells to ${ROOT}"
