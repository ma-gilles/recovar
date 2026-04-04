# Sketched Normal-Operator Products — Student Guide

## What this is

This code implements **sketched products** of the normal-operator gradient
$G(X) = A^*(A(X) - b)$ for cryo-EM heterogeneity.  Given a low-rank iterate
$X = U \text{diag}(\sigma) V^T$, it computes $S_L G(X)$ (left sketch) and
$G(X) Q_R$ (right sketch) without ever forming the dense matrix $G(X)$.

These are the building blocks for randomized algorithms that recover principal
components from cryo-EM data.  Your task is to build such an algorithm on top
of these primitives.

**Key files:**

| File | What |
|---|---|
| `recovar/ppca/sketched_normal.py` | The implementation (~130 lines) |
| `examples/sketched_normal_demo.ipynb` | Notebook with all results (viewable on GitHub) |
| `docs/math/sketched_normal_operator.md` | Math derivations |
| `tests/unit/test_sketched_normal_operator.py` | Unit tests |

**Math reference:** see `docs/math/sketched_normal_operator.md`

---

## How to set up on Della

### 1. Clone the repo

```bash
ssh della-mol
cd /scratch/gpfs/GILLES/<your_netid>
git clone git@github.com:ma-gilles/recovar.git
cd recovar
git checkout claude/sketched-normal-op-v2
```

### 2. Install the environment

The repo uses [pixi](https://pixi.sh) to manage Python, JAX, CUDA, and all
dependencies.  It's already installed on Della.

```bash
pixi install
pixi run install-recovar
pixi run smoke-import-recovar     # should print "recovar_import_ok"
```

This takes ~2 minutes the first time (downloads cached packages).

### 3. Verify GPU access

The Della login node has 4 GPUs available for quick tests:

```bash
nvidia-smi
pixi run python -c "import jax; print(jax.devices())"
```

You should see CUDA devices.  If not, something is wrong — ask Marc.

### 4. Run the notebook

#### Option A: VS Code / Cursor (recommended)

1. Connect to `della-mol` via VS Code Remote SSH
2. Open the repo folder
3. Open `examples/sketched_normal_demo.ipynb`
4. **Select kernel**: click the kernel selector in the top right, choose
   "Select Another Kernel" → "Python Environments" → pick the pixi env at:
   ```
   /scratch/gpfs/GILLES/<your_netid>/recovar/.pixi/envs/default/bin/python
   ```
5. Set these environment variables **before running any cell** (add a first cell):
   ```python
   import os
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # pick a free GPU (check nvidia-smi)
   ```
6. Run all cells

#### Option B: Command line

```bash
cd /path/to/recovar
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=1    # pick a free GPU
pixi run jupyter lab
# Then open examples/sketched_normal_demo.ipynb in the browser
```

### 5. For large runs: submit via Slurm

The login node GPUs are shared — don't run heavy jobs there.
For full-scale runs (128^3, 100k images), submit to Slurm:

```bash
cat > my_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=sketch-test
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB
#SBATCH --time=04:00:00
#SBATCH --output=sketch-%j.out
#SBATCH --exclusive

cd /scratch/gpfs/GILLES/<your_netid>/recovar
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pixi run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    examples/sketched_normal_demo.ipynb \
    --output sketched_normal_demo_executed.ipynb
EOF

sbatch my_job.sh
```

Check status with `squeue -u $USER`.

---

## Using the sketch primitives

```python
from recovar.ppca.sketched_normal import compute_normal_residual_sketches

result = compute_normal_residual_sketches(
    cryo,              # CryoEMDataset (loaded with noise model)
    U_X_half,          # (half_vol, rank) — basis columns of X in half-volume layout
    sigma_X,           # (rank,) — singular values
    V_X,               # (n_images, rank) — right factor
    mean_half,         # (half_vol,) — mean volume in half-volume layout
    batch_size=500,    # images per GPU batch
    left_sketch_half=S_left,   # (s, half_vol) or None
    right_sketch=Q_right,       # (n_images, t) or None
)

# result["left"]  = S_L @ G(X),  shape (s, n_images)
# result["right"] = G(X) @ Q_R,  shape (half_vol, t)
```

**Half-volume layout:** volumes are stored in rfft3 half-spectrum format
`(N, N, N//2+1)` flattened to `(N * N * (N//2+1),)`.  Convert with:
```python
import recovar.core.fourier_transform_utils as ftu
half = ftu.full_volume_to_half_volume(full_vol.reshape(vs), vs).reshape(-1)
full = ftu.half_volume_to_full_volume(half, vs).reshape(-1)
```

---

## Performance at 128^3, 100k images

| sketch_rank | time (compiled) |
|---|---|
| 10 | 7.7s |
| 20 | 8.1s |
| 50 | 10.4s |
| 100 | 13.9s |
| 200 | 21.1s |

Base cost (~7.5s) is the residual computation (forward projecting 10 PCs).
The marginal cost per sketch dimension is a matmul.

For reference, PPCA EM (10 iterations, 10 PCs) takes ~90–115s.
