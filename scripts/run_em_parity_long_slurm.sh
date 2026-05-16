#!/usr/bin/env bash
# EM-scoped long parity Slurm launcher.
#
# Runs ONLY the EM-long parity regression tests:
#   - K=1 256² 50k run_full_refinement parity against RELION auto-refine
#   - K=1 256² 50k native InitialModel/run_ab_initio quality against a
#     RELION --grad --denovo_3dref reference
#   - K=4 256² 50k K-class parity
# Disjoint from ./scripts/run_tests_parallel.sh long-test by design — that one
# runs the cross-cutting SPA/ET pipeline regression suite, which is forbidden
# for EM-only PRs (see recovar/em/CLAUDE.md "Testing" section).
#
# Submits parallel Slurm jobs plus a summary job. The native InitialModel
# quality job depends on a reference-preparation job that creates or validates
# the RELION --grad --denovo_3dref fixture before the test starts.
# Each GPU job is given its own GPU and ~12 hr wall budget.
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
ACCOUNT="${SBATCH_ACCOUNT:-gilles}"
K1_FIXTURE_DIR="${K1_FIXTURE_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_50k_256_normalized}"
K1_NATIVE_RELION_DIR="${K1_NATIVE_RELION_DIR:-${K1_FIXTURE_DIR}/relion_initialmodel_k1_it008}"
RELION_REFINE="${RELION_REFINE:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}"
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
#SBATCH --account=${ACCOUNT}
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=12:00:00

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export TMPDIR="${SCRATCH_DIR}/tmp/${job_name}_\${SLURM_JOB_ID}"
export PIXI_HOME="${SCRATCH_DIR}/pixi_home/${job_name}_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${SCRATCH_DIR}/rattler_cache/${job_name}_\${SLURM_JOB_ID}"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}"
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

RELION_REF_SCRIPT="${SCRATCH_DIR}/em_parity_long_k1_native_relion_ref.sh"
cat > "${RELION_REF_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_parity_long_k1_native_ref
#SBATCH --output=${SCRATCH_DIR}/em_parity_long_k1_native_ref.out
#SBATCH --error=${SCRATCH_DIR}/em_parity_long_k1_native_ref.err
#SBATCH --partition=cryoem
#SBATCH --account=${ACCOUNT}
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=12:00:00

set -euo pipefail
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export TMPDIR="${SCRATCH_DIR}/tmp/em_parity_long_k1_native_ref_\${SLURM_JOB_ID}"
mkdir -p "\${TMPDIR}" "${K1_NATIVE_RELION_DIR}"

echo "=== RELION InitialModel K=1 50k/256 reference ==="
echo "Data: ${K1_FIXTURE_DIR}"
echo "Out:  ${K1_NATIVE_RELION_DIR}"
echo "RELION_REFINE=${RELION_REFINE}"
echo "Slurm job: \${SLURM_JOB_ID}"
echo "Hostname: \$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

OPTIMISER="${K1_NATIVE_RELION_DIR}/run_it008_optimiser.star"
if [[ -s "\${OPTIMISER}" ]]; then
  if grep -q -- "--grad" "\${OPTIMISER}" \
    && grep -q -- "--denovo_3dref" "\${OPTIMISER}" \
    && grep -q "_rlnDoGradientRefine[[:space:]]*1" "\${OPTIMISER}" \
    && grep -q "_rlnDoAutoRefine[[:space:]]*0" "\${OPTIMISER}"; then
    echo "Reusing existing RELION InitialModel fixture: ${K1_NATIVE_RELION_DIR}"
    exit 0
  fi
  echo "Existing ${K1_NATIVE_RELION_DIR} is not a RELION InitialModel --grad fixture." >&2
  echo "Move it aside or set K1_NATIVE_RELION_DIR in this script to a clean path." >&2
  exit 2
fi

if compgen -G "${K1_NATIVE_RELION_DIR}/run_*" >/dev/null; then
  echo "Partial RELION InitialModel fixture exists at ${K1_NATIVE_RELION_DIR}; refusing to overwrite." >&2
  echo "Move it aside before rerunning this launcher." >&2
  exit 2
fi

cd "${K1_FIXTURE_DIR}"
srun -n 1 "${RELION_REFINE}" \
  --o "${K1_NATIVE_RELION_DIR}/run" \
  --iter 8 \
  --grad \
  --denovo_3dref \
  --i particles.star \
  --ctf \
  --K 1 \
  --sym C1 \
  --flatten_solvent \
  --zero_mask \
  --dont_combine_weights_via_disc \
  --pool 3 \
  --pad 1 \
  --particle_diameter 200 \
  --oversampling 1 \
  --healpix_order 1 \
  --offset_range 6 \
  --offset_step 2 \
  --auto_sampling \
  --tau2_fudge 4 \
  --j 4 \
  --gpu 0 \
  --random_seed 0 \
  2>&1 | tee "${K1_NATIVE_RELION_DIR}/relion_initialmodel.log"
EOF
chmod +x "${RELION_REF_SCRIPT}"

K1_SCRIPT="$(make_test_script em_parity_long_k1 "${REPO_ROOT}/tests/long_test/test_em_parity_long.py::test_em_parity_long_k1_full")"
K1_NATIVE_SCRIPT="$(make_test_script em_parity_long_k1_native "${REPO_ROOT}/tests/long_test/test_em_parity_long.py::test_em_parity_long_k1_native_initialmodel_quality")"
K4_SCRIPT="$(make_test_script em_parity_long_k4 "${REPO_ROOT}/tests/long_test/test_em_parity_long.py::test_em_parity_long_kclass_full")"

K1_NATIVE_REF_JOB=$(sbatch --parsable "${RELION_REF_SCRIPT}")
K1_JOB=$(sbatch --parsable "${K1_SCRIPT}")
K1_NATIVE_JOB=$(sbatch --parsable --dependency=afterok:${K1_NATIVE_REF_JOB} "${K1_NATIVE_SCRIPT}")
K4_JOB=$(sbatch --parsable "${K4_SCRIPT}")
echo "Submitted K=1 native RELION InitialModel reference job: ${K1_NATIVE_REF_JOB}"
echo "Submitted K=1 long parity job: ${K1_JOB}"
echo "Submitted K=1 native InitialModel quality job: ${K1_NATIVE_JOB} (afterok:${K1_NATIVE_REF_JOB})"
echo "Submitted K=4 long parity job: ${K4_JOB}"

# Summary job waits for all EM-long jobs, then writes a combined report.
SUMMARY_SCRIPT="${SCRATCH_DIR}/em_parity_long_summary.sh"
cat > "${SUMMARY_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=em_parity_long_summary
#SBATCH --output=${SCRATCH_DIR}/summary.out
#SBATCH --error=${SCRATCH_DIR}/summary.err
#SBATCH --partition=cryoem
#SBATCH --account=${ACCOUNT}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --dependency=afterany:${K1_NATIVE_REF_JOB}:${K1_JOB}:${K1_NATIVE_JOB}:${K4_JOB}

set -euo pipefail
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export TMPDIR="${SCRATCH_DIR}/tmp/em_parity_long_summary_\${SLURM_JOB_ID}"
export PIXI_HOME="${SCRATCH_DIR}/pixi_home/em_parity_long_summary_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${SCRATCH_DIR}/rattler_cache/em_parity_long_summary_\${SLURM_JOB_ID}"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}"

echo "=== EM-long parity summary ==="
echo "K=1 native RELION InitialModel reference job: ${K1_NATIVE_REF_JOB}"
echo "K=1 job: ${K1_JOB}"
echo "K=1 native InitialModel job: ${K1_NATIVE_JOB}"
echo "K=4 job: ${K4_JOB}"
echo

failed=0
for job_id in ${K1_NATIVE_REF_JOB} ${K1_JOB} ${K1_NATIVE_JOB} ${K4_JOB}; do
  state=\$(sacct -j "\${job_id}" -n -X -o State 2>/dev/null | awk 'NR==1{print \$1}')
  echo "Job \${job_id} state: \${state:-UNKNOWN}"
  if [[ "\${state:-UNKNOWN}" != COMPLETED ]]; then
    failed=1
  fi
done
echo

for job_name in em_parity_long_k1_native_ref em_parity_long_k1 em_parity_long_k1_native em_parity_long_k4; do
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

for ledger in \
  tests/baselines/em_parity_quality_long_ledger_k1_long.json \
  tests/baselines/em_parity_quality_long_ledger_k1_native_initialmodel.json \
  tests/baselines/em_parity_quality_long_ledger_kclass_long.json
do
  if [[ ! -s "\${ledger}" ]]; then
    echo "Missing expected EM-long ledger: \${ledger}" >&2
    failed=1
  fi
done

exit "\${failed}"
EOF
chmod +x "${SUMMARY_SCRIPT}"
SUMMARY_JOB=$(sbatch --parsable "${SUMMARY_SCRIPT}")
echo "Submitted summary job: ${SUMMARY_JOB}"
echo
echo "Logs land in: ${SCRATCH_DIR}/"
echo "Summary will run after all EM-long jobs complete."

if [[ "${WATCH}" -eq 1 ]]; then
  echo
  echo "Waiting for jobs (Ctrl-C to stop watching; jobs keep running):"
  while squeue -j "${K1_NATIVE_REF_JOB},${K1_JOB},${K1_NATIVE_JOB},${K4_JOB},${SUMMARY_JOB}" -h 2>/dev/null | grep -q .; do
    sleep 60
    squeue -j "${K1_NATIVE_REF_JOB},${K1_JOB},${K1_NATIVE_JOB},${K4_JOB},${SUMMARY_JOB}" 2>/dev/null || true
  done
  echo
  echo "All jobs done. Summary at: ${SCRATCH_DIR}/summary.out"
  cat "${SCRATCH_DIR}/summary.out" 2>/dev/null || true
fi
