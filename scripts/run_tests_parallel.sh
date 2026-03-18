#!/usr/bin/env bash
set -euo pipefail

# Submit test groups as parallel Slurm jobs, then a summary job that waits
# for all of them.  Each group gets its own sbatch script so WORKDIR is
# baked in at generation time (no variable sharing at runtime).
#
# The groups are chosen so that the heaviest regression tests run in parallel
# (each is independent — own synthetic data, own tmp_path).  This reduces
# wall-clock time from 6-12h (serial) to ~2-3h (parallel).
#
# Usage:
#   ./scripts/run_tests_parallel.sh full        # unit + smoke + gpu + tiny-metrics + downstream
#   ./scripts/run_tests_parallel.sh long-test   # all of the above + 6 long regression groups

MODE="${1:-full}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
SLURMO_DIR="/scratch/gpfs/GILLES/mg6942/slurmo"
RESULTS_DIR="${WORKDIR}/.test_results"
mkdir -p "$SLURMO_DIR" "$RESULTS_DIR"

TAG="parallel_$(date +%Y%m%d_%H%M%S)_${RANDOM}"

# ---------------------------------------------------------------------------
# Define test groups using parallel arrays (avoids IFS/delimiter issues).
#   G_NAMES:  group name (used in job name and XML file)
#   G_MEM:    Slurm --mem
#   G_TIME:   Slurm --time
#   G_ARGS:   pytest arguments
# ---------------------------------------------------------------------------
G_NAMES=()
G_MEM=()
G_TIME=()
G_ARGS=()

# -- Groups that always run --------------------------------------------------

# Unit tests (including GPU unit tests)
G_NAMES+=(unit)
G_MEM+=(300GB)
G_TIME+=(01:00:00)
G_ARGS+=("tests/ --ignore=tests/unit/test_gui_app.py --ignore=tests/integration --run-gpu -v")

# Smoke + tiny-metrics integration tests
G_NAMES+=(smoke-tiny)
G_MEM+=(300GB)
G_TIME+=(02:00:00)
G_ARGS+=("tests/integration/test_pipeline_smoke.py tests/integration/test_run_test_all_metrics_tiny_integration.py tests/integration/test_run_test_all_metrics_tiny_regression_baseline.py tests/integration/test_run_test_outliers_pipeline_tiny_integration.py tests/integration/test_run_test_outliers_pipeline_tiny_regression_baseline.py --run-integration --run-gpu --run-slow --run-tiny-metrics -v")

# Downstream commands (module-scoped fixtures — must stay together)
G_NAMES+=(downstream)
G_MEM+=(300GB)
G_TIME+=(02:00:00)
G_ARGS+=("tests/integration/test_downstream_commands_gpu.py tests/integration/test_downstream_commands_regression.py --run-integration --run-gpu --run-slow -v")

# -- Long-test groups (only with long-test mode) ----------------------------

if [[ "$MODE" == "long-test" ]]; then

    # Full metrics regression — SPA (50k images, 128³)
    G_NAMES+=(metrics-spa)
    G_MEM+=(500GB)
    G_TIME+=(04:00:00)
    G_ARGS+=("tests/integration/test_run_test_all_metrics_regression_long.py::test_run_test_all_metrics_regression_against_baseline --long-test -v")

    # Full metrics regression — cryo-ET
    G_NAMES+=(metrics-et)
    G_MEM+=(500GB)
    G_TIME+=(04:00:00)
    G_ARGS+=("tests/integration/test_run_test_all_metrics_regression_long.py::test_run_test_all_metrics_cryo_et_subsampling_regression_against_baseline --long-test -v")

    # Outliers pipeline regression (SPA + ET long tests)
    G_NAMES+=(outliers-long)
    G_MEM+=(500GB)
    G_TIME+=(06:00:00)
    G_ARGS+=("tests/integration/test_run_test_outliers_pipeline_regression.py --long-test --run-tiny-metrics -v")

    # PDB trajectory regression
    G_NAMES+=(pdb-traj)
    G_MEM+=(500GB)
    G_TIME+=(04:00:00)
    G_ARGS+=("tests/integration/test_pdb_trajectory_regression_long.py --long-test -v")

    # Pipeline with user-supplied indices (SPA + ET)
    G_NAMES+=(indices)
    G_MEM+=(500GB)
    G_TIME+=(06:00:00)
    G_ARGS+=("tests/integration/test_pipeline_with_indices_long.py --long-test -v")

    # GPU memory stress tests (~30 min)
    G_NAMES+=(stress)
    G_MEM+=(500GB)
    G_TIME+=(01:00:00)
    G_ARGS+=("tests/integration/test_gpu_memory_stress.py tests/integration/test_compute_state_gpu_stress.py --long-test -v")

    # Isolated pipeline function tests (~1.5h)
    G_NAMES+=(isolated-funcs)
    G_MEM+=(500GB)
    G_TIME+=(03:00:00)
    G_ARGS+=("tests/integration/test_pipeline_functions_isolated.py --long-test -v")

fi

# ---------------------------------------------------------------------------
# Submit each group as a separate Slurm job
# ---------------------------------------------------------------------------
JOB_IDS=()
echo "===== SUBMITTING ${#G_NAMES[@]} TEST GROUPS (mode=${MODE}) ====="
for i in "${!G_NAMES[@]}"; do
    name="${G_NAMES[$i]}"
    mem="${G_MEM[$i]}"
    walltime="${G_TIME[$i]}"
    pytest_args="${G_ARGS[$i]}"

    EXCLUSIVE_LINE=""
    if [[ "$mem" == "500GB" ]]; then
        EXCLUSIVE_LINE="#SBATCH --exclusive"
    fi

    SCRIPT="${SLURMO_DIR}/job_${TAG}_${name}.sh"
    XML_OUT="${RESULTS_DIR}/${TAG}_${name}.xml"

    cat > "$SCRIPT" << EOF
#!/usr/bin/env bash
#SBATCH --job-name=recovar-${name}
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${mem}
#SBATCH --time=${walltime}
#SBATCH --output=${SLURMO_DIR}/recovar-${name}-%j.out
${EXCLUSIVE_LINE}

set -euo pipefail

cd ${WORKDIR}

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/slurm_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR"

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

pixi run python -m pytest ${pytest_args} --junitxml=${XML_OUT} || true
EOF

    chmod +x "$SCRIPT"
    JID=$(sbatch --parsable "$SCRIPT")
    JOB_IDS+=("$JID")
    echo "  ${name} → job ${JID}  (${walltime}, ${mem})"
done

# ---------------------------------------------------------------------------
# Submit summary job that waits for all groups
# ---------------------------------------------------------------------------
DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
SUMMARY_SCRIPT="${SLURMO_DIR}/job_${TAG}_summary.sh"

cat > "$SUMMARY_SCRIPT" << EOF
#!/usr/bin/env bash
#SBATCH --job-name=recovar-summary
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=00:05:00
#SBATCH --output=${SLURMO_DIR}/recovar-summary-%j.out

set -euo pipefail
cd ${WORKDIR}

PIXI_PY="\$(pixi run which python)"
"\$PIXI_PY" scripts/summarize_test_results.py ${RESULTS_DIR}/${TAG}_*.xml
EOF

chmod +x "$SUMMARY_SCRIPT"
SUM_JID=$(sbatch --parsable --dependency=afterany:"$DEPS" "$SUMMARY_SCRIPT")

echo ""
echo "===== PARALLEL TEST SUBMISSION SUMMARY ====="
echo "  Mode:     ${MODE}"
echo "  Groups:   ${#G_NAMES[@]}"
echo "  Job IDs:  ${JOB_IDS[*]}"
echo "  Summary:  job ${SUM_JID} (runs after all groups finish)"
echo "  Results:  ${RESULTS_DIR}/${TAG}_*.xml"
echo ""
echo "Monitor: squeue -u \$USER -j ${DEPS},${SUM_JID}"
echo "Summary log: ${SLURMO_DIR}/recovar-summary-${SUM_JID}.out"
