# Contributing to RECOVAR

## Development Setup

```bash
git clone git@github.com:ma-gilles/recovar.git
cd recovar
git checkout dev

# Option A: pixi (recommended)
pixi install
pixi run install-recovar
pixi run smoke-import-recovar

# Option B: conda
conda create --name recovar_dev python=3.11 -y
conda activate recovar_dev
pip install -e ".[gpu,dev]"
```

`.[gpu]` installs CUDA-enabled JAX wheels, and `.[cuda]` remains as a
compatibility alias. RECOVAR prefers its custom CUDA extension by default on
GPU, so make sure `nvcc` is available locally and either let RECOVAR auto-build
the shared library on first GPU use or prebuild it with `recovar build_custom_cuda`.
If you need to debug without it, set `RECOVAR_DISABLE_CUDA=1`, but that is slower
and not the preferred path. The fast-marching C++ extension builds automatically
when a compiler is present for source installs and otherwise falls back to the
pure-Python implementation.

## Workflow

1. **Branch off `dev`**: `git checkout -b your-name/feature-name dev`
2. **Make changes** with small, targeted commits
3. **Run tests**: `pixi run test-fast` (quick, no GPU) or submit `pixi run test-full` via Slurm for GPU tests
4. **Push and open a PR** against `dev`

## Testing

See `tests/README.md` for the full test guide. Quick reference:

| Command | Time | GPU | What |
|---------|------|-----|------|
| `pixi run test-fast` | ~30s | No | Unit tests |
| `pixi run test-full` | ~2h | Yes | All tests (use Slurm) |
| `pytest --long-test` | 6-12h | Yes | Full regression (Slurm) |

## Code Style

- **Formatting**: [ruff](https://docs.astral.sh/ruff/) with 120-char line length (see `pyproject.toml`)
- **Pre-commit**: Install with `pip install pre-commit && pre-commit install`
- Formatting is enforced on changed files only — you don't need to reformat the whole codebase

## Rules

- **Never modify `tests/baselines/`** — these are ground truth from the published code
- **Never widen test tolerances** to make failing tests pass — fix the code instead
- Small targeted diffs. No drive-by formatting of files you didn't change.
- See `CLAUDE.md` for architecture overview and `tests/CLAUDE.md` for detailed testing rules.

## AI-Assisted Development

This repo supports Claude Code. `CLAUDE.md` files in the repo root, `recovar/`, `tests/`, and `recovar/cuda/` provide context for AI assistants. You can also use `@claude` on PRs and issues.
