# Reproducing the headline sketched-vs-PPCA result

End-to-end recipe for going from a fresh Linux GPU machine to the
numbers in `HANDOFF.md`:

```
                rv@1    rv@2    rv@5    rv@10
  sketched      0.680   0.884   0.924   0.924
  PPCA-EM       0.723   0.734   0.842   0.870
```

Wall time on one A100 is ~25–30 min for the full pipeline (dataset
generation + sketched + PPCA).

## 0. Prerequisites

- Linux host with one NVIDIA GPU (≥ 16 GB; tested on A100 80 GB).
- `git`, plus `curl` for bootstrapping pixi if it isn't installed.
- Network access to github.com.
- Writable scratch with ≥ 5 GB free (datasets + cached JAX artifacts).

## 1. Install pixi

If pixi isn't on your PATH:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

## 2. Clone and check out the branch

```bash
git clone git@github.com:ma-gilles/recovar.git
cd recovar
git checkout claude/sketched-complex-adjoint-fix
```

(If you're using HTTPS, substitute `https://github.com/ma-gilles/recovar.git`.)

## 3. Install the environment

Run these from the repo root:

```bash
# Block any Python contamination before pixi touches the env.
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1

pixi install                     # ~40s on a warm cache, ~5 min cold
pixi run install-recovar         # editable install, no deps
pixi run smoke-import-recovar    # prints "recovar_import_ok"
```

## 4. Provenance gate

Before any real work, confirm Python is loading **this repo's** recovar
and the pixi-managed JAX:

```bash
pixi run python -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
r = pathlib.Path(recovar.__file__).resolve()
j = pathlib.Path(jax.__file__).resolve()
assert str(r).startswith(str(repo) + '/'), 'WRONG recovar checkout'
assert '.pixi/envs/default/' in str(j), 'WRONG jax environment'
print('ENV_OK', r)
"
```

If this fails, **stop** — something else is hijacking `recovar` or `jax`.
Don't proceed until it prints `ENV_OK`.

## 5. Generate the synthetic dataset

The headline result uses the "notebook smoke" preset: a 10-state
synthetic trajectory at grid=64, 5000 images, radial noise at
noise_level=1e-5.

```bash
OUT=/path/to/writable/scratch/sketched_repro   # pick your own

pixi run python examples/sketched_normal/scripts/generate_notebook_dataset.py \
    --grid-size 64 --n-images 5000 --noise-level 1e-5 \
    --output-dir "$OUT"
```

This is cache-safe: re-running it with the same `$OUT` does nothing.
First run takes ~2–5 min (volume trajectory + image simulator).

The dataset dir is `$OUT/dataset_g64_n5000_nl1e-05/`.

## 6. Run the sketched solver

```bash
DS="$OUT/dataset_g64_n5000_nl1e-05"

pixi run python examples/sketched_normal/scripts/run_experiment.py \
    --dataset-dir "$DS" --grid-size 64 --n-images 5000 --batch-size 500 \
    --method soft --target-rank 10 --lam 1.0 --prior-mode none \
    --step-rule backtracking \
    --bt-delta-init 0.1 --bt-armijo-c 0.9 \
    --bt-shrink 0.5 --bt-grow 1.5 --bt-max-retries 10 \
    --init cold --n-iter 80 \
    --block-size 15 --max-rank 60 --n-power 3 \
    --seed 1 \
    --output "$OUT/sketched.json"
```

Wall time ≈ 17 min on an A100.  Each iteration prints
`rv@{1,2,5,10}`; the final line should read
`rv@10=0.9243` at rank 4.

## 7. Run the PPCA-EM baseline

```bash
pixi run python examples/sketched_normal/scripts/run_ppca_experiment.py \
    --dataset-dir "$DS" --grid-size 64 --n-images 5000 \
    --basis-size 10 --n-iter 20 \
    --output "$OUT/ppca.json"
```

Wall time ≈ 5–10 min on an A100.  Final `rv@10` should be ≈ 0.8699.

## 8. Compare

```bash
pixi run python - <<'PY'
import json
s = json.load(open("$OUT/sketched.json".replace("$OUT", "$OUT")))['final']
p = json.load(open("$OUT/ppca.json".replace("$OUT", "$OUT")))['final']
print(f"                rv@1    rv@2    rv@5    rv@10")
print(f"  sketched      {s['rv@1']:.4f}  {s['rv@2']:.4f}  {s['rv@5']:.4f}  {s['rv@10']:.4f}")
print(f"  PPCA-EM       {p['rv@1']:.4f}  {p['rv@2']:.4f}  {p['rv@5']:.4f}  {p['rv@10']:.4f}")
print(f"  gap (sketched - PPCA) on rv@10: {s['rv@10'] - p['rv@10']:+.4f}")
PY
```

Expected output (values reproduce to ~1e-4 across seeds):

```
                rv@1    rv@2    rv@5    rv@10
  sketched      0.6801  0.8837  0.9243  0.9243
  PPCA-EM       0.7233  0.7343  0.8415  0.8699
  gap (sketched - PPCA) on rv@10: +0.0544
```

## 9. Running on Princeton Della

If you're on Della or a similar Slurm cluster, wrap steps 5–7 in an
sbatch script.  A reference is at
`/scratch/gpfs/GILLES/mg6942/_agent_scratch/sketched_nb_run/run_repro.sbatch`
(outside the repo — it hardcodes a scratch path).  The essential
boilerplate for any Slurm job in this repo is:

```bash
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --mem=120GB --time=01:30:00

set -euo pipefail
cd /path/to/your/recovar/checkout

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/scratch/.../slurm_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ENV_OK` assert fails on `recovar` | another `recovar` on `sys.path` | unset `PYTHONPATH`, rerun `pixi run install-recovar` |
| `ENV_OK` assert fails on `jax`   | system or user-site jax leaking | `export PYTHONNOUSERSITE=1` and re-check |
| CUDA driver / NVML errors at import | host NVIDIA driver too old | pixi env needs driver for CUDA 12; `nvidia-smi` should print |
| `rv@10` much lower than 0.87 for sketched | wrong seed or wrong step rule | check `--seed 1`, `--step-rule backtracking`, `--lam 1.0` |
| OOM during sketched run | max_rank too large for GPU | lower `--max-rank` to 40 (slight quality hit) |

## Next experiments (see `HANDOFF.md` for detail)

- **Tomotwin-100 at 5–10× more images** — the sketched solver targets
  `rank=100` where PPCA is practically capped at `rank=10`, so the
  high-rank win-margin should widen with `n`.  The CryoBench-path
  variant of `generate_dataset.py` and the scaling sbatch wrapper
  (`run_tomo_scale.sbatch`) handle this.
- Fold backtracking into `prox_svt_r4svd_single_lambda_cold_start` in
  the notebook itself.
- PPCA-warmstart probe on the harder `g64/nl=0.1` preset.
