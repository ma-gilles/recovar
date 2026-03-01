#!/bin/bash
#SBATCH --job-name=recovar-pytest
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/heterogeneity_dev-4/pytest_gpu_%j.out

set -uo pipefail
# Note: do NOT set PYTHONNOUSERSITE=1 as pytest lives in ~/.local

cd /scratch/gpfs/GILLES/mg6942/heterogeneity_dev-4

echo "============================================================"
echo "  RECOVAR GPU Tests"
echo "============================================================"
echo "Start:  $(date)"
echo "Host:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================================"

# pytest/pytest-cov must already be installed in the pixi env
# (compute nodes have no internet; install deps on login node first)
echo ">>> Checking deps..."
pixi run python -c "import pytest; print(f'pytest {pytest.__version__}')" 2>&1
pixi run python -c "import recovar; print(f'recovar loaded from {recovar.__file__}')" 2>&1
echo ">>> Deps OK: $(date)"

# Run unit + gpu tests (primary focus)
echo ">>> Running unit + GPU tests..."
pixi run python -m pytest --run-gpu -m "unit or gpu" --tb=short -q 2>&1
unit_rc=$?
echo ">>> Unit + GPU tests done (exit=$unit_rc): $(date)"
echo

# Run smoke/integration tests (may fail if optional deps like grain missing)
echo ">>> Running smoke tests (best-effort)..."
pixi run python -m pytest --run-integration tests/integration/test_pipeline_smoke.py -v 2>&1 || true
echo ">>> Smoke tests done: $(date)"

echo "============================================================"
echo "  All GPU tests completed (unit exit=$unit_rc): $(date)"
echo "============================================================"
exit $unit_rc
