#!/bin/bash
# prepare_test_environments.sh
# Run this on the HEAD NODE (has internet access)
# Creates two separate conda environments for testing old vs new code

set -e

echo "=========================================="
echo "Preparing Test Environments"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Initialize conda
eval "$(conda shell.bash hook)"

BASE_DIR="/home/mg6942/recovar/"
RESULTS_DIR="/home/mg6942/mytigress/recovar_test_envs"

mkdir -p "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "STEP 1: Setup OLD version repository"
echo "=========================================="

cd "$RESULTS_DIR"
if [ -d "recovar_old" ]; then
    echo "Removing existing recovar_old directory..."
    rm -rf recovar_old
fi

# Clone repository locally (much faster than copying, no network needed)
echo "Cloning repository from local path..."
git clone "$BASE_DIR" recovar_old
cd recovar_old

# Checkout commit before rewrites (using commit before e410164)
BEFORE_REWRITE_COMMIT="33dba5f"
echo "Checking out old version at commit: $BEFORE_REWRITE_COMMIT"
git checkout $BEFORE_REWRITE_COMMIT 2>&1

echo "Old version checked out successfully"

echo ""
echo "=========================================="
echo "STEP 2: Create OLD version environment"
echo "=========================================="

OLD_ENV="recovar_old_test"

# Remove if exists
conda env remove --name "$OLD_ENV" -y 2>/dev/null || true

echo "Creating conda environment: $OLD_ENV"
conda create --name "$OLD_ENV" python=3.11 -y

echo "Activating conda environment: $OLD_ENV"
conda activate "$OLD_ENV"

if [[ "$CONDA_DEFAULT_ENV" != "$OLD_ENV" ]]; then
    echo "Error: Failed to activate $OLD_ENV"
    exit 1
fi

echo "Installing dependencies for OLD version..."
pip install "git+https://github.com/scikit-fmm/scikit-fmm.git"
pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install "jax[cuda12]==0.5.0"

echo "Installing OLD version of recovar..."
# Install in non-editable mode so it uses the code at this commit
pip install .

echo "OLD version environment created successfully!"
conda deactivate

echo ""
echo "=========================================="
echo "STEP 3: Setup NEW version repository"
echo "=========================================="

cd "$RESULTS_DIR"
if [ -d "recovar_new" ]; then
    echo "Removing existing recovar_new directory..."
    rm -rf recovar_new
fi

# Clone repository locally for new version too
echo "Cloning repository from local path..."
git clone "$BASE_DIR" recovar_new
cd recovar_new

# Stay on current HEAD (latest commit)
echo "Using current HEAD commit"
git log --oneline -1

echo ""
echo "=========================================="
echo "STEP 4: Create NEW version environment"
echo "=========================================="

NEW_ENV="recovar_new_test"

# Remove if exists
conda env remove --name "$NEW_ENV" -y 2>/dev/null || true

echo "Creating conda environment: $NEW_ENV"
conda create --name "$NEW_ENV" python=3.11 -y

echo "Activating conda environment: $NEW_ENV"
conda activate "$NEW_ENV"

if [[ "$CONDA_DEFAULT_ENV" != "$NEW_ENV" ]]; then
    echo "Error: Failed to activate $NEW_ENV"
    exit 1
fi

echo "Installing dependencies for NEW version..."
pip install "git+https://github.com/scikit-fmm/scikit-fmm.git"
pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install "jax[cuda12]==0.5.0"

echo "Installing NEW version of recovar..."
echo "  Installing from: $(pwd)"
# Install in non-editable mode to ensure isolation
pip install .

echo "Verifying NEW version installation..."
python -c "import recovar; import os; print(f'  Installed from: {os.path.dirname(recovar.__file__)}'); import subprocess; result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd='$(pwd)', capture_output=True, text=True); print(f'  Git commit: {result.stdout.strip()}')"

echo "NEW version environment created successfully!"
conda deactivate

echo ""
echo "=========================================="
echo "PREPARATION COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Environments created:"
echo "  OLD version: $OLD_ENV (commit 33dba5f)"
echo "  NEW version: $NEW_ENV (current HEAD)"
echo ""
echo "Repositories saved at:"
echo "  OLD version: $RESULTS_DIR/recovar_old"
echo "  NEW version: $RESULTS_DIR/recovar_new"
echo ""
echo "Now you can run the comparison test:"
echo "  ./run_comparison_test.sh quick      # Quick cryo-EM test"
echo "  ./run_comparison_test.sh tomo_quick # Quick tomography test"
echo "  sbatch run_comparison_test.sh quick # Submit to compute node"
