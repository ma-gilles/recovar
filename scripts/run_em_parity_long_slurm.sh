#!/usr/bin/env bash
# EM-scoped long parity Slurm launcher.
#
# Runs ONLY the EM-long parity regression tests (K=1 256² 50k + K=4 256² 50k).
# Disjoint from ./scripts/run_tests_parallel.sh long-test by design — that one
# runs the cross-cutting SPA/ET pipeline regression suite, which is forbidden
# for EM-only PRs (see recovar/em/CLAUDE.md "Testing" section).
#
# Submits two parallel Slurm jobs (one per test) and a summary job that
# waits for both. Each job is given its own GPU and ~5 hr wall budget.
#
# Usage:
#   ./scripts/run_em_parity_long_slurm.sh          # submit and exit
#   ./scripts/run_em_parity_long_slurm.sh --watch  # submit and tail logs
#
# Outputs:
#   tests/baselines/em_parity_quality_long_ledger_*.json (per-test ledgers)
#   /scratch/gpfs/GILLES/mg6942/_agent_scratch/em_parity_long_<timestamp>/  (Slurm logs)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="em_parity_long_${TIMESTAMP}_${RANDOM}"
SCRATCH_DIR="/scratch/gpfs/GILLES/mg6942/_agent_scratch/${RUN_ID}"
mkdir -p "${SCRATCH_DIR}"
touch "${SCRATCH_DIR}/SAFE_TO_DELETE"

WATCH=0
for arg in "$@"; do
  case "$arg" in
    --watch) WATCH=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

# Common Slurm preamble for one test
make_test_script() {
  local job_name="$1"
  local test_path="$2"
  local script_path="${SCRATCH_DIR}/${job_name}.sh"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${SCRATCH_DIR}/${job_name}.out
#SBATCH --error=${SCRATCH_DIR}/${job_name}.err
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${RUN_ID}_\${SLURM_JOB_ID}"
mkdir -p "\${TMPDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=== EM-long parity Slurm job ${job_name} ==="
echo "Repo: ${REPO_ROOT}"
echo "Test: ${test_path}"
echo "Slurm job: \${SLURM_JOB_ID}"
echo "Hostname: \$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

# Provenance gate runs inside the test, but print a short banner for the log.
git -C "${REPO_ROOT}" rev-parse HEAD
git -C "${REPO_ROOT}" symbolic-ref --short HEAD || echo '<detached>'

pixi run python -m pytest --em-parity-long -v -s "${test_path}"
EOF
  chmod +x "${script_path}"
  echo "${script_path}"
}

K1_SCRIPT="$(make_test_script em_parity_long_k1 "${REPO_ROOT}/tests/long_test/test_em_parity_long.py::test_em_parity_long_k1_full")"
K4_SCRIPT="$(make_test_script em_parity_long_k4 "${REPO_ROOT}/tests/long_test/test_em_parity_long.py::test_em_parity_long_kclass_full")"

K1_JOB=$(sbatch --parsable "${K1_SCRIPT}")
K4_JOB=$(sbatch --parsable "${K4_SCRIPT}")
echo "Submitted K=1 long parity job: ${K1_JOB}"
echo "Submitted K=4 long parity job: ${K4_JOB}"

# Summary job waits for both, then writes a combined report.
SUMMARY_SCRIPT="${SCRATCH_DIR}/em_parity_long_summary.sh"
cat > "${SUMMARY_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_parity_long_summary
#SBATCH --output=${SCRATCH_DIR}/summary.out
#SBATCH --error=${SCRATCH_DIR}/summary.err
#SBATCH --partition=cryoem
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --dependency=afterany:${K1_JOB}:${K4_JOB}

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1

echo "=== EM-long parity summary ==="
echo "K=1 job: ${K1_JOB}"
echo "K=4 job: ${K4_JOB}"
echo

for job_name in em_parity_long_k1 em_parity_long_k4; do
  echo "--- \${job_name} stdout tail ---"
  tail -40 "${SCRATCH_DIR}/\${job_name}.out" 2>/dev/null || echo "(no stdout)"
  echo
  echo "--- \${job_name} stderr tail ---"
  tail -20 "${SCRATCH_DIR}/\${job_name}.err" 2>/dev/null || echo "(no stderr)"
  echo
done

echo "=== EM-parity ledgers ==="
ls -la tests/baselines/em_parity_quality_long_ledger_*.json 2>/dev/null || echo "(no ledgers)"
echo
pixi run python scripts/extract_em_parity_tables.py --tier long || true
EOF
chmod +x "${SUMMARY_SCRIPT}"
SUMMARY_JOB=$(sbatch --parsable "${SUMMARY_SCRIPT}")
echo "Submitted summary job: ${SUMMARY_JOB}"
echo
echo "Logs land in: ${SCRATCH_DIR}/"
echo "Summary will run after both test jobs complete."

if [[ "${WATCH}" -eq 1 ]]; then
  echo
  echo "Waiting for jobs (Ctrl-C to stop watching; jobs keep running):"
  while squeue -j "${K1_JOB},${K4_JOB},${SUMMARY_JOB}" -h 2>/dev/null | grep -q .; do
    sleep 60
    squeue -j "${K1_JOB},${K4_JOB},${SUMMARY_JOB}" 2>/dev/null || true
  done
  echo
  echo "All jobs done. Summary at: ${SCRATCH_DIR}/summary.out"
  cat "${SCRATCH_DIR}/summary.out" 2>/dev/null || true
fi
