#!/usr/bin/env bash
# Phase 1 ablation sweep: 12 cells × 3 seeds × 3 dataset/q rows = 108 jobs.
#
# Factors:
#   - s_init   ∈ {flat, truth}
#   - ridge    ∈ {scalar, w_prior}
#   - anneal   ∈ {none, factor-only-log1000}
#
# Rows:
#   - Ribosembly q=4   (n_joint=30, sigma=0.01)
#   - Ribosembly q=8   (n_joint=100, sigma=0.01)
#   - IgG-RL q=2       (n_joint=30, sigma=0.1)
#
# Each cell × seed runs run_cryobench.py with --save-results <cell.json>.
# Output goes under _agent_scratch/ppca_phase1_sweep_<TS>/ (auto-cleanup OK).

set -euo pipefail

WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408"
TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${WORKDIR}/_agent_scratch/ppca_phase1_sweep_${TS}"
SLURMO_DIR="${SWEEP_ROOT}/slurm_logs"
JSON_DIR="${SWEEP_ROOT}/cells"
JOBS_FILE="${SWEEP_ROOT}/job_ids.txt"
mkdir -p "$SLURMO_DIR" "$JSON_DIR"

# Provenance for the sweep root
git -C "$WORKDIR" rev-parse HEAD > "${SWEEP_ROOT}/git_commit.txt"
git -C "$WORKDIR" status --short > "${SWEEP_ROOT}/git_status.txt"
echo "$(date)" > "${SWEEP_ROOT}/started_at.txt"

# Per-cell sbatch template (templated via heredoc per call)
gen_sbatch() {
    local cell_id="$1"
    local cell_args="$2"
    local cell_json="$3"
    local script="${SLURMO_DIR}/cell_${cell_id}.sbatch"

    cat > "$script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ppca-ph1-${cell_id}
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
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

# Provenance gate
"\$PIXI_PY" -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
print('ENV_OK — devices:', jax.devices())
"

pixi run python scripts/ppca_abinitio/run_cryobench.py ${cell_args} --save-results ${cell_json}
EOF

    echo "$script"
}

# -------------------------------------------------------------------
# Cell config grid
# -------------------------------------------------------------------

# Rows: dataset / q / n_joint / sigma / external_mode
ROW_NAMES=("ribo_q4" "ribo_q8" "iggrl_q2")
ROW_DATASET=("Ribosembly" "Ribosembly" "IgG-RL")
ROW_Q=(4 8 2)
ROW_NJOINT=(30 100 30)
ROW_SIGMA=(0.01 0.01 0.1)

# Factors
S_INIT_VALS=("flat" "truth")
RIDGE_VALS=("scalar" "w_prior")
ANNEAL_VALS=("none" "factor_only")  # factor_only ⇒ --anneal-schedule log1000 --anneal-factor-only

SEEDS=(0 1 2)

COMMON="--vol 32 --n-images 1024 --healpix-order 1 --u-init svd --svd-warmstart weighted --mu-init perturbed --external-mode discrete_volumes --n-burnin 0"

# -------------------------------------------------------------------
# Submission loop
# -------------------------------------------------------------------
N_CELLS=0
for r_idx in "${!ROW_NAMES[@]}"; do
    row="${ROW_NAMES[$r_idx]}"
    dataset="${ROW_DATASET[$r_idx]}"
    q="${ROW_Q[$r_idx]}"
    n_joint="${ROW_NJOINT[$r_idx]}"
    sigma="${ROW_SIGMA[$r_idx]}"

    for s_init in "${S_INIT_VALS[@]}"; do
        for ridge in "${RIDGE_VALS[@]}"; do
            for anneal in "${ANNEAL_VALS[@]}"; do
                if [[ "$anneal" == "none" ]]; then
                    anneal_flags="--anneal-schedule none"
                else
                    anneal_flags="--anneal-schedule log1000 --anneal-iters ${n_joint} --anneal-factor-only"
                fi

                for seed in "${SEEDS[@]}"; do
                    cell_id="${row}_sinit-${s_init}_ridge-${ridge}_anneal-${anneal}_seed-${seed}"
                    cell_json="${JSON_DIR}/${cell_id}.json"
                    cell_args="--dataset ${dataset} ${COMMON} --q ${q} --n-joint ${n_joint} --sigma ${sigma} \
                               --s-init ${s_init} --ridge-mode ${ridge} ${anneal_flags} --seed ${seed} --init-seed ${seed}"

                    SCRIPT="$(gen_sbatch "${cell_id}" "${cell_args}" "${cell_json}")"
                    JID=$(sbatch --parsable "$SCRIPT")
                    echo "${JID}  ${cell_id}" | tee -a "${JOBS_FILE}"
                    N_CELLS=$((N_CELLS + 1))
                done
            done
        done
    done
done

echo "==== submitted ${N_CELLS} cells to ${SWEEP_ROOT}"
echo "results JSONs will appear in ${JSON_DIR}"
echo "monitor with: squeue -u \$USER -n ppca-ph1"
echo "${SWEEP_ROOT}" > "${WORKDIR}/_agent_scratch/last_phase1_sweep_root.txt"
