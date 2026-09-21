#!/usr/bin/env bash
# EM-scoped long parity Slurm launcher.
#
# Runs ONLY the EM-long parity regression tests:
#   - K=1 256² 50k run_full_refinement parity against RELION auto-refine
#   - K=1 256² 50k native InitialModel quality against a
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
#   <run-root>/results/<job-name>/**/em_parity_quality_long_ledger_*.json
#   /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_parity_long_<timestamp>/  (Slurm logs)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="em_parity_long_${TIMESTAMP}_${RANDOM}"
SCRATCH_DIR="${EM_PARITY_LONG_SCRATCH_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/${RUN_ID}}"
ACCOUNT="${SBATCH_ACCOUNT:-gilles}"
K1_FIXTURE_DIR="${K1_FIXTURE_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_50k_256_normalized}"
K1_NATIVE_RELION_DIR="${K1_NATIVE_RELION_DIR:-${K1_FIXTURE_DIR}/relion_initialmodel_k1_it008}"
RELION_REFINE="${RELION_REFINE:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}"
mkdir -p "${SCRATCH_DIR}"
# pytest --basetemp creates only its own leaf directory, so the results root has to
# exist first; without it every test job dies in setup with FileNotFoundError before
# running anything.
mkdir -p "${SCRATCH_DIR}/results"
touch "${SCRATCH_DIR}/SAFE_TO_DELETE"

# Build the custom CUDA library once, here, into the cache root the jobs will use.
# The builder serializes on a lock file, so without this the four GPU jobs would each
# hold an allocation while waiting for whichever of them won the build. Building up
# front also means every job in this tier loads one binary whose digest is printed in
# each job log.
export RECOVAR_CUDA_CACHE_DIR="${SCRATCH_DIR}/cuda_cache"
mkdir -p "${RECOVAR_CUDA_CACHE_DIR}"
RELION_SRC_DIR="${RELION_SRC_DIR:-/scratch/gpfs/GILLES/mg6942/relion/src}"
if [[ ! -f "${RELION_SRC_DIR}/projector.h" ]]; then
  echo "RELION_SRC_DIR must name a RELION src directory containing projector.h" >&2
  echo "  got: ${RELION_SRC_DIR}" >&2
  exit 2
fi
export RELION_SRC_DIR

# run_full_refinement.py imports recovar.relion_bind._relion_bind_core for the RELION
# half-set ordering, and a fresh checkout has no built extension, so every K=1 rung
# dies in seconds with an ImportError. Build it once here, into this tier's own scratch
# directory, and let the jobs load it through RECOVAR_RELION_BIND_BUILD_DIR -- the same
# arrangement the robustness matrix uses, and for the same reason the CUDA library is
# built here: a login-node build keeps a foreign toolchain and a per-job rebuild out of
# the GPU allocations.
export RECOVAR_RELION_BIND_BUILD_DIR="${SCRATCH_DIR}/relion_bind_build"
mkdir -p "${RECOVAR_RELION_BIND_BUILD_DIR}"
echo "Building the RELION binding into ${RECOVAR_RELION_BIND_BUILD_DIR} ..."
"${REPO_ROOT}/.pixi/envs/default/bin/python" "${REPO_ROOT}/recovar/relion_bind/build.py"
ls -1 "${RECOVAR_RELION_BIND_BUILD_DIR}"/_relion_bind_core*.so \
  || { echo "RELION binding build produced no extension" >&2; exit 2; }

echo "Building the custom CUDA library into ${RECOVAR_CUDA_CACHE_DIR} ..."
"${REPO_ROOT}/.pixi/envs/default/bin/python" -m recovar.commands.build_custom_cuda \
  --output "${RECOVAR_CUDA_CACHE_DIR}/libcuda_backproject.so"
sha256sum "${RECOVAR_CUDA_CACHE_DIR}/libcuda_backproject.so"

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
# PATH is a contaminating variable too. A run of this tier picked up the conda
# toolchain of an unrelated checkout's pixi environment, whose linker could not find
# the CUDA driver stub, and the custom CUDA build failed with "ld: cannot find -lcuda"
# inside the GPU allocation. Drop every pixi environment that is not this checkout's
# so the toolchain comes from here or from the system, never from someone else's tree.
PATH="\$(printf '%s' "\${PATH}" | tr ':' '\\n' \\
  | grep -v '/\\.pixi/envs/' \\
  | paste -sd: -)"
export PATH="${REPO_ROOT}/.pixi/envs/default/bin:\${PATH}"
export PYTHONNOUSERSITE=1
export TMPDIR="${SCRATCH_DIR}/tmp/${job_name}_\${SLURM_JOB_ID}"
export PIXI_HOME="${SCRATCH_DIR}/pixi_home/${job_name}_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${SCRATCH_DIR}/rattler_cache/${job_name}_\${SLURM_JOB_ID}"
# The custom CUDA library cache is keyed on the home directory alone, not on the
# CUDA sources, so every checkout on this machine shares one libcuda_backproject.so.
# The staleness test compares source mtimes against that one file, which means a
# rebuild triggered here would rewrite the binary underneath any other job already
# running against it, and a checkout whose sources are older than someone else's
# build silently loads someone else's binary. Giving the tier its own cache root
# makes the library this tier builds and loads private to this tier.
export RECOVAR_CUDA_CACHE_DIR="${SCRATCH_DIR}/cuda_cache"
# tests/conftest.py builds its own library at <repo>/.tmp/pytest_custom_cuda/ and
# overrides RECOVAR_CUDA_CACHE_DIR while doing so, so that path -- not the cache root
# -- is what these jobs would load. It is shared by every test job running from this
# checkout and is resolved by a plain exists() check taken before the build lock, so a
# second concurrent job can pick up a partially written .so. conftest honors
# RECOVAR_CUDA_LIB read-only, resolving it and never rebuilding, so pointing every job
# at the one library the launcher already built removes the race, avoids an nvcc build
# inside each GPU allocation, and gives the tier a single binary identity.
export RECOVAR_CUDA_LIB="${SCRATCH_DIR}/cuda_cache/libcuda_backproject.so"
export RECOVAR_RELION_BIND_BUILD_DIR="${SCRATCH_DIR}/relion_bind_build"
export RELION_SRC_DIR="${RELION_SRC_DIR}"
mkdir -p "\${TMPDIR}" "\${PIXI_HOME}" "\${RATTLER_CACHE_DIR}" "\${RECOVAR_CUDA_CACHE_DIR}"
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
# A configured path is not a loaded-binary identity; print the digest of the file
# that will actually be loaded so a result can be tied to one binary after the fact.
sha256sum "\${RECOVAR_CUDA_LIB}" 2>/dev/null \
  || { echo "custom CUDA library missing at \${RECOVAR_CUDA_LIB}" >&2; exit 1; }

pixi run python -m pytest --em-parity-long -v -s \
  --basetemp "${SCRATCH_DIR}/results/${job_name}_\${SLURM_JOB_ID}" "${test_path}"
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

# A Slurm state of COMPLETED only says the job exited 0. pytest exits 0 when it
# skips, so a rung whose fixture is missing looks exactly like a rung that passed.
# Classify each rung from its own pytest summary line instead, and never let a
# skipped rung contribute to an overall pass: a missing measurement is not agreement.
skipped=0
measured=0
echo "=== rung outcomes ==="
for job_name in em_parity_long_k1 em_parity_long_k1_native em_parity_long_k4; do
  out="${SCRATCH_DIR}/\${job_name}.out"
  if [[ ! -f "\${out}" ]]; then
    echo "\${job_name}: NO OUTPUT"
    failed=1
    continue
  fi
  line=\$(grep -ohE '[0-9]+ (passed|failed|error|skipped)[^=]*' "\${out}" | tail -1)
  if grep -qE '^SKIPPED|[0-9]+ skipped' "\${out}" && ! grep -qE '[0-9]+ passed' "\${out}"; then
    echo "\${job_name}: SKIPPED (not measured) -- \${line:-no pytest summary}"
    grep -A4 'short test summary' "\${out}" 2>/dev/null | tail -4
    skipped=1
  elif grep -qE '[0-9]+ (failed|error)' "\${out}"; then
    echo "\${job_name}: FAILED -- \${line:-no pytest summary}"
    failed=1
  elif grep -qE '[0-9]+ passed' "\${out}"; then
    echo "\${job_name}: passed -- \${line}"
    measured=\$((measured + 1))
  else
    echo "\${job_name}: INDETERMINATE -- no pytest summary line"
    failed=1
  fi
done
echo "rungs actually measured: \${measured}"
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
find "${SCRATCH_DIR}/results" -name 'em_parity_quality_long_ledger_*.json' -print 2>/dev/null || true
echo
pixi run python scripts/extract_em_parity_tables.py --tier long \
  --ledger-root "${SCRATCH_DIR}/results" \
  --require-case k1_long k1_native_initialmodel kclass_long || failed=1

for ledger in \
  em_parity_quality_long_ledger_k1_long.json \
  em_parity_quality_long_ledger_k1_native_initialmodel.json \
  em_parity_quality_long_ledger_kclass_long.json
do
  count=\$(find "${SCRATCH_DIR}/results" -name "\${ledger}" -type f -size +0c 2>/dev/null | wc -l)
  if [[ "\${count}" -ne 1 ]]; then
    echo "Expected exactly one nonempty EM-long ledger: \${ledger}" >&2
    failed=1
  fi
done

if [[ "\${skipped}" -ne 0 ]]; then
  echo "EM-long tier did NOT validate: at least one rung was skipped for a missing" >&2
  echo "fixture and therefore measured nothing. A skipped rung is not a passing rung." >&2
  failed=1
fi

if [[ "\${failed}" -eq 0 ]]; then
  echo "EM-long tier: all \${measured} rungs measured and passed."
fi
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
