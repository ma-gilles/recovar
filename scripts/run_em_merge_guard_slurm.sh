#!/usr/bin/env bash
# EM merge guard Slurm launcher.
#
# Use after merging EM / VDAM / PPCA refinement branches that may touch the
# dense EM iteration loop. This is the fast, high-signal guard for the K-class
# RELION-parity improvements on this branch. It saves provenance, pytest logs,
# ledger JSON, table output, and refinement_results.npz paths under scratch.
#
# This is intentionally EM-scoped. It does not run the broad SPA/ET long-test.
#
# Usage:
#   ./scripts/run_em_merge_guard_slurm.sh
#   ./scripts/run_em_merge_guard_slurm.sh --watch
#
# Outputs:
#   /scratch/gpfs/GILLES/mg6942/_agent_scratch/em_merge_guard_<timestamp>/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="em_merge_guard_${TIMESTAMP}_${RANDOM}"
SCRATCH_DIR="/scratch/gpfs/GILLES/mg6942/_agent_scratch/${RUN_ID}"
ACCOUNT="${SBATCH_ACCOUNT:-gilles}"
mkdir -p "${SCRATCH_DIR}"
touch "${SCRATCH_DIR}/SAFE_TO_DELETE"

WATCH=0
for arg in "$@"; do
  case "$arg" in
    --watch) WATCH=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

write_common_header() {
  local job_name="$1"
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${SCRATCH_DIR}/${job_name}.out
#SBATCH --error=${SCRATCH_DIR}/${job_name}.err
#SBATCH --partition=cryoem
#SBATCH --account=${ACCOUNT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${SCRATCH_DIR}/tmp/${job_name}_\${SLURM_JOB_ID}"
export PIXI_HOME="${SCRATCH_DIR}/pixi_home/${job_name}_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${SCRATCH_DIR}/rattler_cache/${job_name}_\${SLURM_JOB_ID}"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}"

echo "=== ${job_name} ==="
echo "Repo: ${REPO_ROOT}"
echo "Scratch: ${SCRATCH_DIR}"
echo "Slurm job: \${SLURM_JOB_ID}"
echo "Hostname: \$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo
git rev-parse HEAD
git symbolic-ref --short HEAD || echo '<detached>'
git status --porcelain
flock "${REPO_ROOT}/.pixi/install-recovar.lock" pixi run install-recovar
flock "${SCRATCH_DIR}/relion_bind_build.lock" bash -lc 'rm -rf recovar/relion_bind/build && pixi run python recovar/relion_bind/build.py'
pixi run python - <<'PY'
import pathlib
import jax
import recovar
from recovar.relion_bind import _relion_bind_core as relion_bind

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
jax_file = pathlib.Path(jax.__file__).resolve()
relion_bind_file = pathlib.Path(relion_bind.__file__).resolve()
print(recovar_file)
print(jax_file)
print(jax.devices())
print(relion_bind_file)
assert str(recovar_file).startswith(str(repo) + "/"), recovar_file
assert ".pixi/envs/default/" in str(jax_file), jax_file
assert str(relion_bind_file).startswith(str(repo) + "/"), relion_bind_file
PY
EOF
}

UNIT_SCRIPT="${SCRATCH_DIR}/em_merge_guard_units.sh"
{
  write_common_header "em_merge_guard_units"
  cat <<'EOF'

pixi run test-em-fast-guard

JAX_PLATFORMS=cpu pixi run python -m pytest -v -s \
  tests/unit/test_refine_relion_mode.py::TestRelionModeSmokeTest::test_relion_sigma_offset_prior_center_matches_store_weighted_sums_units \
  tests/unit/test_refine_relion_mode.py::test_relion_mode_writes_absolute_translations_from_previous_offset \
  tests/unit/test_refine_relion_mode.py::test_relion_mode_dense_k_class_writes_absolute_translations_from_previous_offset \
  tests/unit/test_k_class_joint_semantics.py
EOF
} > "${UNIT_SCRIPT}"
chmod +x "${UNIT_SCRIPT}"

KCLASS_SCRIPT="${SCRATCH_DIR}/em_merge_guard_kclass_fast.sh"
{
  write_common_header "em_merge_guard_kclass_fast"
  cat <<EOF

pixi run python -m pytest -v -s --run-slow --run-integration --run-gpu \\
  tests/integration/test_em_parity_fast.py::test_em_parity_fast_kclass_replay \\
  tests/integration/test_em_parity_fast.py::test_em_parity_fast_kclass_coldstart \\
  tests/integration/test_em_parity_fast.py::test_em_parity_fast_kclass_strict_coldstart \\
  tests/integration/test_em_parity_fast.py::test_em_parity_fast_kclass_strict_oversample_coldstart

find "\${TMPDIR}" -name refinement_results.npz -print | tee "${SCRATCH_DIR}/kclass_refinement_npz_paths.txt"
mkdir -p "${SCRATCH_DIR}/ledgers"
cp tests/baselines/em_parity_quality_fast_ledger_kclass*.json "${SCRATCH_DIR}/ledgers/" 2>/dev/null || true
pixi run python scripts/extract_em_parity_tables.py --tier fast | tee "${SCRATCH_DIR}/fast_tables.md"
EOF
} > "${KCLASS_SCRIPT}"
chmod +x "${KCLASS_SCRIPT}"

UNIT_JOB=$(sbatch --parsable "${UNIT_SCRIPT}")
KCLASS_JOB=$(sbatch --parsable "${KCLASS_SCRIPT}")

SUMMARY_SCRIPT="${SCRATCH_DIR}/em_merge_guard_summary.sh"
cat > "${SUMMARY_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_merge_guard_summary
#SBATCH --output=${SCRATCH_DIR}/summary.out
#SBATCH --error=${SCRATCH_DIR}/summary.err
#SBATCH --partition=cryoem
#SBATCH --account=${ACCOUNT}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --dependency=afterany:${UNIT_JOB}:${KCLASS_JOB}

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export TMPDIR="${SCRATCH_DIR}/tmp/em_merge_guard_summary_\${SLURM_JOB_ID}"
export PIXI_HOME="${SCRATCH_DIR}/pixi_home/em_merge_guard_summary_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${SCRATCH_DIR}/rattler_cache/em_merge_guard_summary_\${SLURM_JOB_ID}"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}"

echo "=== EM merge guard summary ==="
echo "Repo: ${REPO_ROOT}"
echo "Scratch: ${SCRATCH_DIR}"
echo "Unit job: ${UNIT_JOB}"
echo "K-class fast job: ${KCLASS_JOB}"
echo

failed=0
for job_id in ${UNIT_JOB} ${KCLASS_JOB}; do
  state=\$(sacct -j "\${job_id}" -n -X -o State 2>/dev/null | awk 'NR==1{print \$1}')
  echo "Job \${job_id} state: \${state:-UNKNOWN}"
  if [[ "\${state:-UNKNOWN}" != COMPLETED ]]; then
    failed=1
  fi
done
echo

for job_name in em_merge_guard_units em_merge_guard_kclass_fast; do
  echo "--- \${job_name} stdout tail ---"
  tail -80 "${SCRATCH_DIR}/\${job_name}.out" 2>/dev/null || echo "(no stdout)"
  echo
  echo "--- \${job_name} stderr tail ---"
  tail -40 "${SCRATCH_DIR}/\${job_name}.err" 2>/dev/null || echo "(no stderr)"
  echo
done

echo "=== Saved artifacts ==="
find "${SCRATCH_DIR}" -maxdepth 3 -type f | sort
echo

echo "=== EM fast parity tables ==="
pixi run python scripts/extract_em_parity_tables.py --tier fast || true

exit "\${failed}"
EOF
chmod +x "${SUMMARY_SCRIPT}"
SUMMARY_JOB=$(sbatch --parsable "${SUMMARY_SCRIPT}")

echo "Submitted EM merge guard unit job: ${UNIT_JOB}"
echo "Submitted EM merge guard K-class fast job: ${KCLASS_JOB}"
echo "Submitted EM merge guard summary job: ${SUMMARY_JOB}"
echo
echo "Logs and artifacts land in: ${SCRATCH_DIR}/"

if [[ "${WATCH}" -eq 1 ]]; then
  echo
  echo "Waiting for jobs (Ctrl-C stops watching; jobs keep running):"
  while squeue -j "${UNIT_JOB},${KCLASS_JOB},${SUMMARY_JOB}" -h 2>/dev/null | grep -q .; do
    sleep 60
    squeue -j "${UNIT_JOB},${KCLASS_JOB},${SUMMARY_JOB}" 2>/dev/null || true
  done
  echo
  echo "All jobs left the queue. Summary at: ${SCRATCH_DIR}/summary.out"
  cat "${SCRATCH_DIR}/summary.out" 2>/dev/null || true
fi
