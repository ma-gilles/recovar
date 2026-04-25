#!/usr/bin/env bash
# Phase 7c — multiclass μ-init cold-rescue A/B
#
# Phase 6 baseline (cold μ, std warmstart):
#   Ribo q=4 σ=0.01: hun=0.293
#   Ribo q=8 σ=0.01: hun=0.217
#   IgG-RL q=2 σ=0.01: hun=0.171
#
# A/B = {mu-init zero}  vs  {mu-init multiclass}
# Test on 4 dataset×q rows × 2 seeds × 2 mu-init = 16 cells.
# Plus reference cells with --mu-init perturbed for Phase 6 calibration.
#
# Verdict criterion: lifts cold-μ Ribo q=4 from 0.29 → ≥ 0.65;
# doesn't regress the perturbed-μ baseline.

set -euo pipefail

WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408"
TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${WORKDIR}/_agent_scratch/ppca_phase7c_multiclass_${TS}"
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
#SBATCH --job-name=ppca-ph7c-${cell_id}
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
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

COMMON="--vol 32 --healpix-order 1 --svd-warmstart weighted --external-mode discrete_volumes --n-burnin 0 --s-init flat --ridge-mode scalar --u-init svd --n-images 1024 --anneal-schedule none"

n_joint_for() { if [ "$1" -ge 8 ]; then echo 100; else echo 30; fi; }

# Datasets: (dataset:q:sigma)
ROWS=("Ribosembly:4:0.01" "Ribosembly:8:0.01" "IgG-1D:2:0.1" "IgG-RL:2:0.1")

N=0
for row in "${ROWS[@]}"; do
    IFS=: read -r ds q sigma <<< "$row"
    nj=$(n_joint_for "$q")
    for seed in 0 1; do
        # Group A: mu-init zero (reproduces Phase 6 cold-μ baseline)
        submit "A_${ds//-/_}_q${q}_sig${sigma}_zero_seed${seed}" \
            "--dataset ${ds} ${COMMON} --q ${q} --n-joint ${nj} --sigma ${sigma} --mu-init zero --seed ${seed} --init-seed ${seed}"
        N=$((N+1))

        # Group B: mu-init multiclass (Phase 7c rescue)
        submit "B_${ds//-/_}_q${q}_sig${sigma}_multiclass_seed${seed}" \
            "--dataset ${ds} ${COMMON} --q ${q} --n-joint ${nj} --sigma ${sigma} --mu-init multiclass --multiclass-K 5 --multiclass-iters 50 --seed ${seed} --init-seed ${seed}"
        N=$((N+1))

        # Group C: mu-init perturbed (Phase 6 reference, no-regression check)
        submit "C_${ds//-/_}_q${q}_sig${sigma}_perturbed_seed${seed}" \
            "--dataset ${ds} ${COMMON} --q ${q} --n-joint ${nj} --sigma ${sigma} --mu-init perturbed --seed ${seed} --init-seed ${seed}"
        N=$((N+1))
    done
done

echo "==== submitted ${N} cells to ${SWEEP_ROOT}"
echo "${SWEEP_ROOT}" > "${WORKDIR}/_agent_scratch/last_phase7c_root.txt"
