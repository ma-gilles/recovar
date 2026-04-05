# Sketched Normal-Operator Products — Student Guide

## What this is

Given a low-rank iterate $X = U \operatorname{diag}(\sigma) V^T$ and the
cryo-EM forward model, this code computes $S\, G(X)$ (left sketch) and
$G(X)\, Q$ (right sketch) without forming the dense gradient matrix
$G(X) = A^*(A(X) - b)$.

These are the building blocks for randomized algorithms that recover
principal components from cryo-EM data.

**Key files:**

| File | What |
|---|---|
| `recovar/ppca/sketched_normal.py` | Implementation (~200 lines) |
| `examples/sketched_normal_demo.ipynb` | Notebook with all results (viewable on GitHub) |
| `docs/math/sketched_normal_operator.md` | Math derivations |
| `tests/unit/test_sketched_normal_operator.py` | Unit tests |

---

## Setup on Della

### 1. Clone and checkout

```bash
ssh della-mol
cd /scratch/gpfs/GILLES/<your_netid>
git clone git@github.com:ma-gilles/recovar.git
cd recovar
git checkout claude/sketched-normal-op-v2
```

### 2. Install

```bash
pixi install
pixi run install-recovar
pixi run smoke-import-recovar   # should print "recovar_import_ok"
```

### 3. GPU check

```bash
nvidia-smi
pixi run python -c "import jax; print(jax.devices())"
```

4 login-node GPUs for quick tests.  For real runs → Slurm (see below).

### 4. Run the notebook

**VS Code / Cursor (recommended):**

1. Connect to `della-mol` via Remote SSH
2. Open `examples/sketched_normal_demo.ipynb`
3. Select kernel → Python Environments → pick:
   `/scratch/gpfs/GILLES/<netid>/recovar/.pixi/envs/default/bin/python`
4. Add as first cell:
   ```python
   import os
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # pick a free GPU
   ```
5. Run all cells

**Command line:**
```bash
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=1
pixi run jupyter lab
```

### 5. Slurm for large runs

```bash
cat > sketch_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=sketch
#SBATCH --account=amits --partition=cryoem
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=500GB
#SBATCH --time=04:00:00 --exclusive
#SBATCH --output=sketch-%j.out

cd /scratch/gpfs/GILLES/<netid>/recovar
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pixi run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    examples/sketched_normal_demo.ipynb \
    --output sketched_normal_demo_executed.ipynb
EOF

sbatch sketch_job.sh
```

---

## API

```python
from recovar.ppca.sketched_normal import SketchedNormalOperator

# mean is Fourier (complex, from pipeline).  U, S, Q are real-space.
op = SketchedNormalOperator(cryo, mean_fourier, batch_size=500)

left  = op.left_matvec(U, s, V, S)         # S @ G(X),  (s, n_images)
right = op.right_matvec(U, s, V, Q)         # G(X) @ Q,  (vol_size, t)
left, right = op.both_matvecs(U, s, V, S, Q)

# For Fourier-domain U (e.g. from gt.get_vol_svd()):
right = op.right_matvec_fourier(U_fourier, s, V, Q)
```

**Shapes:**
- `U`: `(vol_size, rank)` real
- `s`: `(rank,)` float
- `V`: `(n_images, rank)` float
- `S`: `(sketch_rank, vol_size)` real
- `Q`: `(n_images, t)` float

---

## Performance (128^3, 100k images, A100)

| sketch_rank | time (compiled) |
|---|---|
| 10 | ~8s |
| 50 | ~8s |
| 100 | ~9s |
| 200 | ~13s |

Base cost is residual computation (forward-projecting the rank PCs).
Marginal cost per sketch dimension is a matmul.

PPCA EM baseline: 10 iterations in ~120s, relvar@10 = 0.65.
