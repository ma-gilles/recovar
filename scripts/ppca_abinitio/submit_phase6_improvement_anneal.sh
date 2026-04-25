#!/usr/bin/env bash
# Phase 6 targeted improvement: does factor-only annealing rescue the
# Phase 6 failure modes?
#
# Failure modes from Phase 6:
#   - Cold-μ: zero-μ + any U init collapses (hun 0.07-0.29) at all σ tested.
#   - High-σ: Ribo q=8 σ=1.0 hun=0.261 (collapse).
#   - Very-low-σ: Ribo q=8 σ=0.001 hun=0.621 (basin-narrowness, expected).
#
# Phase 1 already showed factor-only-log1000 lifts Ribo q=8 from 0.774
# (no anneal) → 0.946 (anneal). Hypothesis: annealing also rescues the
# cold-μ and high-σ regimes.
#
# A/B cells (each anneal=none vs anneal=factor_only_log1000):
#   1. Cold-μ Ribo q=4 σ=0.01 (svd-U)        — A1n / A1f
#   2. Cold-μ Ribo q=8 σ=0.01 (svd-U)        — A2n / A2f
#   3. Cold-μ IgG-RL q=2 σ=0.01 (svd-U)       — A3n / A3f
#   4. High-σ Ribo q=4 σ=1.0 perturbed-μ     — B1n / B1f
#   5. High-σ Ribo q=8 σ=1.0 perturbed-μ     — B2n / B2f
#   6. Very-low-σ Ribo q=8 σ=0.001 perturbed-μ — C1n / C1f
#
# Each × 2 seeds × {none, factor_only} = 24 cells. ~3-50 GPU min/cell.

set -euo pipefail

WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408"
TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${WORKDIR}/_agent_scratch/ppca_phase6_improvement_${TS}"
SLURMO_DIR="${SWEEP_ROOT}/slurm_logs"
JSON_DIR="${SWEEP_ROOT}/cells"
JOBS_FILE="${SWEEP_ROOT}/job_ids.txt"
mkdir -p "$SLURMO_DIR" "$JSON_DIR"

git -C "$WORKDIR" rev-parse HEAD > "${SWEEP_ROOT}/git_commit.txt"
echo "$(date)" > "${SWEEP_ROOT}/started_at.txt"

gen_sbatch() {
    local cell_id="$1" cell_args="$2" cell_json="$3"
    local script="${SLURMO_DIR}/cell_${cell_id}.sbatch"
    cat > "$script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ppca-ph6i-${cell_id}
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:30:00
#SBATCH --output=${SLURMO_DIR}/cell_${cell_id}-%j.out

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

pixi run python scripts/ppca_abinitio/run_cryobench.py ${cell_args} --save-results ${cell_json}
EOF
    echo "$script"
}

submit() {
    local id="$1" args="$2"
    local cell_json="${JSON_DIR}/${id}.json"
    SCRIPT="$(gen_sbatch "${id}" "${args}" "${cell_json}")"
    JID=$(sbatch --parsable "$SCRIPT")
    echo "${JID}  ${id}" | tee -a "${JOBS_FILE}"
}

COMMON="--vol 32 --healpix-order 1 --svd-warmstart weighted --external-mode discrete_volumes --n-burnin 0 --s-init flat --ridge-mode scalar --u-init svd --n-images 1024"

n_joint_for() { if [ "$1" -ge 8 ]; then echo 100; else echo 30; fi; }
ANNEAL_F="--anneal-schedule log1000 --anneal-iters \$nj --anneal-factor-only"
ANNEAL_N="--anneal-schedule none"

N=0

# Group A — Cold-μ (zero-μ) tests
A_CELLS=("A1:Ribosembly:4:0.01" "A2:Ribosembly:8:0.01" "A3:IgG-RL:2:0.01")
for cell in "${A_CELLS[@]}"; do
    IFS=: read -r tag ds q sigma <<< "$cell"
    nj=$(n_joint_for "$q")
    for seed in 0 1; do
        # No anneal
        submit "${tag}_n_seed${seed}" "--dataset ${ds} ${COMMON} --mu-init zero --q ${q} --n-joint ${nj} --sigma ${sigma} --anneal-schedule none --seed ${seed} --init-seed ${seed}"
        N=$((N+1))
        # Factor-only anneal
        submit "${tag}_f_seed${seed}" "--dataset ${ds} ${COMMON} --mu-init zero --q ${q} --n-joint ${nj} --sigma ${sigma} --anneal-schedule log1000 --anneal-iters ${nj} --anneal-factor-only --seed ${seed} --init-seed ${seed}"
        N=$((N+1))
    done
done

# Group B — High-σ rescue (perturbed-μ, σ=1.0)
B_CELLS=("B1:Ribosembly:4:1.0" "B2:Ribosembly:8:1.0")
for cell in "${B_CELLS[@]}"; do
    IFS=: read -r tag ds q sigma <<< "$cell"
    nj=$(n_joint_for "$q")
    for seed in 0 1; do
        submit "${tag}_n_seed${seed}" "--dataset ${ds} ${COMMON} --mu-init perturbed --q ${q} --n-joint ${nj} --sigma ${sigma} --anneal-schedule none --seed ${seed} --init-seed ${seed}"
        N=$((N+1))
        submit "${tag}_f_seed${seed}" "--dataset ${ds} ${COMMON} --mu-init perturbed --q ${q} --n-joint ${nj} --sigma ${sigma} --anneal-schedule log1000 --anneal-iters ${nj} --anneal-factor-only --seed ${seed} --init-seed ${seed}"
        N=$((N+1))
    done
done

# Group C — Very-low-σ (perturbed-μ, σ=0.001)
C_CELLS=("C1:Ribosembly:8:0.001")
for cell in "${C_CELLS[@]}"; do
    IFS=: read -r tag ds q sigma <<< "$cell"
    nj=$(n_joint_for "$q")
    for seed in 0 1; do
        submit "${tag}_n_seed${seed}" "--dataset ${ds} ${COMMON} --mu-init perturbed --q ${q} --n-joint ${nj} --sigma ${sigma} --anneal-schedule none --seed ${seed} --init-seed ${seed}"
        N=$((N+1))
        submit "${tag}_f_seed${seed}" "--dataset ${ds} ${COMMON} --mu-init perturbed --q ${q} --n-joint ${nj} --sigma ${sigma} --anneal-schedule log1000 --anneal-iters ${nj} --anneal-factor-only --seed ${seed} --init-seed ${seed}"
        N=$((N+1))
    done
done

echo "==== submitted ${N} cells to ${SWEEP_ROOT}"
echo "${SWEEP_ROOT}" > "${WORKDIR}/_agent_scratch/last_phase6_improvement_root.txt"
