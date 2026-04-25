#!/usr/bin/env bash
# Phase 6 stress sweep: probe v0 limits along 4 axes from a fixed
# reference cell. NOT Cartesian. Total ≈ 124 cells × 1 seed pair = 248
# (we do 2 seeds per cell). Cells are per-axis fan-outs.
#
# Reference cell:
#   flat / scalar / no-anneal / σ=0.01 / perturbed-μ / svd-U / n=1024
#
# Axis A — SNR sweep:        σ ∈ {0.001, 0.01, 0.03, 0.1, 0.3, 1.0}
#                            on 4 datasets, 2 seeds. 6×4×2 = 48 cells.
# Axis B — cold init combos: zero-μ × {svd, random, zero}-U
#                            on 4 datasets × σ ∈ {0.01, 0.1}, 2 seeds.
#                            3×4×2×2 = 48 cells.
# Axis C — small-N:          n_img ∈ {64, 128, 256, 512}
#                            on Ribo q=4 + IgG-RL q=2, σ=0.01, 2 seeds.
#                            4×2×2 = 16 cells.
# Axis D — Tomotwin-100:     q ∈ {4, 8, 16}, σ ∈ {0.01, 0.1}, 2 seeds.
#                            3×2×2 = 12 cells.
#
# All cells: vol=32, healpix_order=1, weighted SVD warmstart (where
# applicable), 30 iters (q≤4) or 100 iters (q≥8).

set -euo pipefail

WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408"
TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${WORKDIR}/_agent_scratch/ppca_phase6_stress_${TS}"
SLURMO_DIR="${SWEEP_ROOT}/slurm_logs"
JSON_DIR="${SWEEP_ROOT}/cells"
JOBS_FILE="${SWEEP_ROOT}/job_ids.txt"
mkdir -p "$SLURMO_DIR" "$JSON_DIR"

git -C "$WORKDIR" rev-parse HEAD > "${SWEEP_ROOT}/git_commit.txt"
git -C "$WORKDIR" status --short > "${SWEEP_ROOT}/git_status.txt"
echo "$(date)" > "${SWEEP_ROOT}/started_at.txt"

# Per-cell sbatch generator
gen_sbatch() {
    local cell_id="$1"
    local cell_args="$2"
    local cell_json="$3"
    local script="${SLURMO_DIR}/cell_${cell_id}.sbatch"

    cat > "$script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ppca-ph6-${cell_id}
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

"\$PIXI_PY" -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
print('ENV_OK — devices:', jax.devices())
"

pixi run python scripts/ppca_abinitio/run_cryobench.py ${cell_args} --save-results ${cell_json}
EOF

    echo "$script"
}

# Common defaults
COMMON="--vol 32 --healpix-order 1 --svd-warmstart weighted --external-mode discrete_volumes --n-burnin 0 --s-init flat --ridge-mode scalar --anneal-schedule none"

# n_joint by q
n_joint_for() {
    if [ "$1" -ge 8 ]; then echo 100; else echo 30; fi
}

submit_cell() {
    local id="$1"
    local args="$2"
    local cell_json="${JSON_DIR}/${id}.json"
    SCRIPT="$(gen_sbatch "${id}" "${args}" "${cell_json}")"
    JID=$(sbatch --parsable "$SCRIPT")
    echo "${JID}  ${id}" | tee -a "${JOBS_FILE}"
}

N=0

# ============================================================
# AXIS A — SNR SWEEP
# ============================================================
DATASETS_AND_Q=("Ribosembly:4" "Ribosembly:8" "IgG-1D:2" "IgG-RL:2")
for ds_q in "${DATASETS_AND_Q[@]}"; do
    ds="${ds_q%:*}"; q="${ds_q#*:}"
    nj=$(n_joint_for "$q")
    for sigma in 0.001 0.01 0.03 0.1 0.3 1.0; do
        for seed in 0 1; do
            id="A_${ds//-/_}_q${q}_sig${sigma}_seed${seed}"
            args="--dataset ${ds} ${COMMON} --n-images 1024 --mu-init perturbed --u-init svd --q ${q} --n-joint ${nj} --sigma ${sigma} --seed ${seed} --init-seed ${seed}"
            submit_cell "$id" "$args"
            N=$((N+1))
        done
    done
done

# ============================================================
# AXIS B — COLD INIT COMBOS
# ============================================================
for ds_q in "${DATASETS_AND_Q[@]}"; do
    ds="${ds_q%:*}"; q="${ds_q#*:}"
    nj=$(n_joint_for "$q")
    for sigma in 0.01 0.1; do
        for u_init in svd random zero; do
            for seed in 0 1; do
                id="B_${ds//-/_}_q${q}_sig${sigma}_u${u_init}_seed${seed}"
                args="--dataset ${ds} ${COMMON} --n-images 1024 --mu-init zero --u-init ${u_init} --q ${q} --n-joint ${nj} --sigma ${sigma} --seed ${seed} --init-seed ${seed}"
                submit_cell "$id" "$args"
                N=$((N+1))
            done
        done
    done
done

# ============================================================
# AXIS C — SMALL-N
# ============================================================
for ds_q in "Ribosembly:4" "IgG-RL:2"; do
    ds="${ds_q%:*}"; q="${ds_q#*:}"
    nj=$(n_joint_for "$q")
    for n_img in 64 128 256 512; do
        for seed in 0 1; do
            id="C_${ds//-/_}_q${q}_n${n_img}_seed${seed}"
            args="--dataset ${ds} ${COMMON} --n-images ${n_img} --mu-init perturbed --u-init svd --q ${q} --n-joint ${nj} --sigma 0.01 --seed ${seed} --init-seed ${seed}"
            submit_cell "$id" "$args"
            N=$((N+1))
        done
    done
done

# ============================================================
# AXIS D — TOMOTWIN-100
# ============================================================
for q in 4 8 16; do
    nj=$(n_joint_for "$q")
    for sigma in 0.01 0.1; do
        for seed in 0 1; do
            id="D_Tomotwin_100_q${q}_sig${sigma}_seed${seed}"
            args="--dataset Tomotwin-100 ${COMMON} --n-images 1024 --mu-init perturbed --u-init svd --q ${q} --n-joint ${nj} --sigma ${sigma} --seed ${seed} --init-seed ${seed}"
            submit_cell "$id" "$args"
            N=$((N+1))
        done
    done
done

echo "==== submitted ${N} cells to ${SWEEP_ROOT}"
echo "${SWEEP_ROOT}" > "${WORKDIR}/_agent_scratch/last_phase6_stress_root.txt"
