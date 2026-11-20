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
#   ./scripts/run_tests_parallel.sh long-test   # all of the above + long regression groups

MODE="${1:-full}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
SLURMO_DIR="${SLURMO_DIR:-${WORKDIR}/logs}"
RESULTS_DIR="${RECOVAR_TEST_RESULTS_DIR:-${WORKDIR}/.test_results}"
RUNTIME_ROOT="${RECOVAR_TEST_RUNTIME_ROOT:-${WORKDIR}/.tmp}"
GPU_REQUEST="${RECOVAR_TEST_GPU_REQUEST:-gpu:1}"
PIXI_PY="${WORKDIR}/.pixi/envs/default/bin/python"
if [[ "$MODE" != full && "$MODE" != long-test ]]; then
    echo "Usage: $0 [full|long-test]" >&2
    exit 2
fi
if [[ ! -x "$PIXI_PY" ]]; then
    echo "Install the frozen pixi environment before submitting tests." >&2
    exit 2
fi
mkdir -p "$SLURMO_DIR" "$RESULTS_DIR" "$RUNTIME_ROOT"

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
G_MEM+=(64G)
G_TIME+=(01:00:00)
G_ARGS+=("tests/ --ignore=tests/integration --run-gpu --run-slow --run-integration -v")

# Smoke + tiny-metrics integration tests
G_NAMES+=(smoke-tiny)
G_MEM+=(64G)
G_TIME+=(05:00:00)
G_ARGS+=("tests/integration/test_pipeline_smoke.py tests/integration/test_run_test_all_metrics_tiny_integration.py tests/integration/test_run_test_all_metrics_tiny_regression_baseline.py tests/integration/test_run_test_outliers_pipeline_tiny_integration.py tests/integration/test_run_test_outliers_pipeline_tiny_regression_baseline.py tests/integration/test_run_test_dataset_e2e.py --run-integration --run-gpu --run-slow --run-tiny-metrics -v")

# Downstream commands (module-scoped fixtures — must stay together)
G_NAMES+=(downstream)
G_MEM+=(64G)
G_TIME+=(02:00:00)
G_ARGS+=("tests/integration/test_downstream_commands_gpu.py tests/integration/test_downstream_commands_regression.py tests/integration/test_uninvert_contrast.py --run-integration --run-gpu --run-slow -v")

# PPCA quality and fast GPU memory budget coverage.
G_NAMES+=(ppca-metrics)
G_MEM+=(64G)
G_TIME+=(00:45:00)
G_ARGS+=("tests/integration/test_ppca_pipeline_metrics.py::test_ppca_pipeline_metrics --run-integration --run-gpu --run-slow -v")

G_NAMES+=(memory-matrix-fast)
G_MEM+=(64G)
G_TIME+=(01:00:00)
G_ARGS+=("tests/integration/test_gpu_memory_matrix.py --run-integration --run-gpu -k test_memory_matrix_fast -v")

# -- Long-test groups (only with long-test mode) ----------------------------

if [[ "$MODE" == "long-test" ]]; then

    # Full metrics regression — SPA (50k images, 128³)
    G_NAMES+=(metrics-spa)
    G_MEM+=(140G)
    G_TIME+=(01:30:00)
    G_ARGS+=("tests/integration/test_run_test_all_metrics_regression_long.py::test_run_test_all_metrics_regression_against_baseline --long-test -v")

    # PPCA and covariance on the same full-size SPA fixture.
    G_NAMES+=(ppca-spa-long)
    G_MEM+=(128G)
    G_TIME+=(01:30:00)
    G_ARGS+=("tests/integration/test_ppca_pipeline_metrics.py::test_ppca_pipeline_full_resolution --long-test -v")

    # Full metrics regression — cryo-ET
    G_NAMES+=(metrics-et)
    G_MEM+=(140G)
    G_TIME+=(01:30:00)
    G_ARGS+=("tests/integration/test_run_test_all_metrics_regression_long.py::test_run_test_all_metrics_cryo_et_subsampling_regression_against_baseline --long-test -v")

    # Outliers pipeline regression (SPA + ET long tests)
    G_NAMES+=(outliers-long)
    G_MEM+=(140G)
    G_TIME+=(02:00:00)
    G_ARGS+=("tests/integration/test_run_test_outliers_pipeline_regression.py --long-test --run-tiny-metrics -v")

    # PDB trajectory regression
    G_NAMES+=(pdb-traj)
    G_MEM+=(140G)
    G_TIME+=(01:30:00)
    G_ARGS+=("tests/integration/test_pdb_trajectory_regression_long.py --long-test -v")

    # Pipeline with user-supplied indices (SPA + ET)
    G_NAMES+=(indices)
    G_MEM+=(140G)
    G_TIME+=(02:00:00)
    G_ARGS+=("tests/integration/test_pipeline_with_indices_long.py tests/integration/test_cryo_et_outliers_with_indices.py --long-test -v")

    # GPU memory stress tests (~30 min)
    # Job 15784267 on H100 reached 248 GiB RSS on the 256^3 stress workloads.
    G_NAMES+=(stress)
    G_MEM+=(320G)
    G_TIME+=(02:00:00)
    G_ARGS+=("tests/integration/test_gpu_memory_stress.py tests/integration/test_compute_state_gpu_stress.py --long-test -v")

    G_NAMES+=(compute-state)
    G_MEM+=(140G)
    G_TIME+=(03:00:00)
    G_ARGS+=("tests/integration/test_compute_state_regression.py --long-test -v")

    G_NAMES+=(memory-matrix-long)
    G_MEM+=(64G)
    G_TIME+=(03:00:00)
    G_ARGS+=("tests/integration/test_gpu_memory_matrix.py -m gpu_memory_matrix --long-test -v")

    # Isolated pipeline function tests (~1.5h)
    G_NAMES+=(isolated-funcs)
    G_MEM+=(140G)
    G_TIME+=(03:00:00)
    G_ARGS+=("tests/integration/test_pipeline_functions_isolated.py --long-test -v")

fi

# ---------------------------------------------------------------------------
# Submit each group as a separate Slurm job
# ---------------------------------------------------------------------------
JOB_IDS=()
XML_FILES=()
echo "===== SUBMITTING ${#G_NAMES[@]} TEST GROUPS (mode=${MODE}) ====="
for i in "${!G_NAMES[@]}"; do
    name="${G_NAMES[$i]}"
    mem="${G_MEM[$i]}"
    walltime="${G_TIME[$i]}"
    pytest_args="${G_ARGS[$i]}"

    SCRIPT="${SLURMO_DIR}/job_${TAG}_${name}.sh"
    XML_OUT="${RESULTS_DIR}/${TAG}_${name}.xml"
    XML_FILES+=("$XML_OUT")

    cat > "$SCRIPT" << EOF
#!/usr/bin/env bash
#SBATCH --job-name=recovar-${name}
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=${GPU_REQUEST}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${walltime}
#SBATCH --output=${SLURMO_DIR}/recovar-${name}-%j.out

set -euo pipefail

cd ${WORKDIR}

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export TMPDIR="${RUNTIME_ROOT}/slurm_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR"
export RECOVAR_JAX_CACHE_DIR="\$TMPDIR/recovar_cache"
export JAX_COMPILATION_CACHE_DIR="\$TMPDIR/jax_cache"
export MPLCONFIGDIR="\$TMPDIR/matplotlib" XDG_CACHE_HOME="\$TMPDIR/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORM_NAME
unset PERF_OVERWRITE_BASELINE CS_REGR_REGENERATE_BASELINE LONG_METRICS_WRITE_BASELINE

PIXI_PY="${PIXI_PY}"
export PATH="\$(dirname "\$PIXI_PY"):\$PATH"
test -n "\${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "\${RECOVAR_CUDA_LIB:-}" ]]; then
    sha256sum "\$RECOVAR_CUDA_LIB"
fi
git rev-parse HEAD
git status --short --branch
git diff HEAD --stat
git diff HEAD | sha256sum

# Provenance gate
"\$PIXI_PY" -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
assert jax.default_backend() == 'gpu', jax.devices()
print('ENV_OK — devices:', jax.devices())
"

"\$PIXI_PY" -m pytest ${pytest_args} -ra --basetemp="\$TMPDIR/pytest" --junitxml="${XML_OUT}"
if [[ -n "\${RECOVAR_CUDA_LIB:-}" ]]; then
    sha256sum "\$RECOVAR_CUDA_LIB"
fi
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
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=${SLURMO_DIR}/recovar-summary-%j.out

set -euo pipefail
cd ${WORKDIR}

export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

"${PIXI_PY}" scripts/summarize_test_results.py ${XML_FILES[*]}
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
