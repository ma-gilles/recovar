#!/usr/bin/env bash
set -euo pipefail

# Submit test groups as parallel Slurm jobs, then a summary job that waits
# for all of them.  Each group gets its own sbatch script so WORKDIR is
# baked in at generation time (no variable sharing at runtime).
#
# Usage:
#   ./scripts/run_tests_parallel.sh full       # unit + gpu + integration
#   ./scripts/run_tests_parallel.sh long-test   # all of the above + long-test

MODE="${1:-full}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
SLURMO_DIR="/scratch/gpfs/GILLES/mg6942/slurmo"
RESULTS_DIR="${WORKDIR}/.test_results"
mkdir -p "$SLURMO_DIR" "$RESULTS_DIR"

TAG="parallel_$(date +%Y%m%d_%H%M%S)_${RANDOM}"

# ---------------------------------------------------------------------------
# Define test groups.  Format: NAME GPU MEM TIME PYTEST_ARGS
# ---------------------------------------------------------------------------
declare -a GROUPS=()

add_group() {  # name gpu mem time pytest_args...
    local name=$1 gpu=$2 mem=$3 time=$4
    shift 4
    GROUPS+=("${name}|${gpu}|${mem}|${time}|$*")
}

add_group unit   no  16G  00:30:00 tests/ -q --ignore=tests/unit/test_gui_app.py
add_group gpu    yes 300G 02:00:00 tests/ -q --ignore=tests/unit/test_gui_app.py --run-gpu
add_group integration yes 300G 02:00:00 tests/ -q --ignore=tests/unit/test_gui_app.py --run-integration --run-gpu --run-slow --run-tiny-metrics

if [[ "$MODE" == "long-test" ]]; then
    add_group long-test yes 500G 12:00:00 tests/ -q --ignore=tests/unit/test_gui_app.py --long-test
fi

# ---------------------------------------------------------------------------
# Submit each group as a separate Slurm job
# ---------------------------------------------------------------------------
JOB_IDS=()
for entry in "${GROUPS[@]}"; do
    IFS='|' read -r name gpu mem walltime pytest_args <<< "$entry"

    GRES_LINE=""
    EXCLUSIVE_LINE=""
    if [[ "$gpu" == "yes" ]]; then
        GRES_LINE="#SBATCH --gres=gpu:1"
        if [[ "$mem" == "500G" ]]; then
            EXCLUSIVE_LINE="#SBATCH --exclusive"
        fi
    fi

    SCRIPT="${SLURMO_DIR}/job_${TAG}_${name}.sh"
    XML_OUT="${RESULTS_DIR}/${TAG}_${name}.xml"

    cat > "$SCRIPT" << EOF
#!/usr/bin/env bash
#SBATCH --job-name=recovar-${name}
#SBATCH --account=amits
#SBATCH --partition=cryoem
${GRES_LINE}
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
print('ENV_OK')
"

pixi run python -m pytest ${pytest_args} --junitxml=${XML_OUT} || true
EOF

    chmod +x "$SCRIPT"
    JID=$(sbatch --parsable "$SCRIPT")
    JOB_IDS+=("$JID")
    echo "  ${name} → job ${JID}  (log: ${SLURMO_DIR}/recovar-${name}-${JID}.out)"
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
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
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
echo "  Groups:   ${#GROUPS[@]}"
echo "  Job IDs:  ${JOB_IDS[*]}"
echo "  Summary:  job ${SUM_JID} (runs after all groups finish)"
echo "  Results:  ${RESULTS_DIR}/${TAG}_*.xml"
echo ""
echo "Monitor: squeue -u \$USER -j ${DEPS},${SUM_JID}"
echo "Summary log: ${SLURMO_DIR}/recovar-summary-${SUM_JID}.out"
