# RECOVAR: Tools for cryo-EM heterogeneity analysis

RECOVAR analyzes conformational heterogeneity in cryo-EM and cryo-ET datasets. It reconstructs high-resolution volumes, estimates conformational density and low free-energy motions, and automatically identifies image subsets associated with specific volume features.

**[Full Documentation](https://ma-gilles.github.io/recovar)** | **[Paper](https://www.pnas.org/doi/abs/10.1073/pnas.2419140122)** | **[Talk](https://www.youtube.com/watch?v=cQBQlCCRp8Q&t=740s)**

## Key features

- **High resolution** — top performer on [CryoBench](https://cryobench.cs.princeton.edu)
- **Direct input** — accepts RELION `.star` and cryoSPARC `.cs` files (no preprocessing needed)
- **Image-to-volume attribution** — extract images that produced a specific volume feature
- **Conformational density** — estimate free energy landscapes
- **Focus masks** — targeted heterogeneity analysis
- **Cryo-ET support** — tilt-series data with focus masks

## Installation (Only Supported Workflow)

This repository supports exactly one setup flow: isolated clone + isolated pixi env + editable install bound to that clone.

Direct `conda`/`pip` installs are not supported for `heterogeneity_dev` development and will cause cross-repo import drift.

### Strict setup (copy/paste)

```bash
set -euo pipefail

# 1) Unique clone per run/agent (prevents one checkout overwriting another)
AGENT_ID="agent_$(date +%Y%m%d_%H%M%S)_$RANDOM"
WORKDIR="$HOME/myscratch/heterogeneity_dev_${AGENT_ID}"
git clone git@github.com:ma-gilles/heterogeneity_dev.git "$WORKDIR"
cd "$WORKDIR"
test -z "$(git status --porcelain)"

# 2) Block external python contamination
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1

# 3) Unique temp/cache roots per run (prevents cross-agent lock/cache collisions)
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

# 4) Build env, then bind recovar import to THIS checkout
pixi install
PIXI_PY="$(pixi run which python)"
export PATH="$(dirname "$PIXI_PY"):$PATH"
"$PIXI_PY" -m pip uninstall -y recovar || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed

# 5) Build CUDA extension with the same python used above
PYTHON="$PIXI_PY" make -C recovar/cuda clean all

# 6) Provenance gate (must pass before any test/run)
"$PIXI_PY" - <<'PY'
import pathlib, recovar, jax, sys
repo = pathlib.Path.cwd().resolve()
r = pathlib.Path(recovar.__file__).resolve()
j = pathlib.Path(jax.__file__).resolve()
print("python:", sys.executable)
print("repo:", repo)
print("recovar:", r)
print("jax:", j)
assert str(r).startswith(str(repo) + "/"), "WRONG recovar checkout"
assert ".pixi/envs/default/" in str(j), "WRONG jax environment"
print("ENV_OK")
PY
```

Run commands only via `"$PIXI_PY" -m ...` or `pixi run ...` in this checkout.

### Fast marching backend

The editable install above also builds RECOVAR's optional native fast marching extension. Trajectory computations use this in-tree implementation, while a pure-Python fallback remains available for editable installs and unsupported platforms.

- `RECOVAR_FORCE_PYTHON_FMM=1` forces the fallback implementation.
- `RECOVAR_REQUIRE_NATIVE_FMM=1` makes import fail fast if the native extension is unavailable.
- Released wheels can bundle the native extension for `pip install recovar`, but `heterogeneity_dev` development remains pixi-only as described above.

## Quick start

```bash
# Run the pipeline
"$PIXI_PY" -m recovar.command_line pipeline particles.star -o output --mask mask.mrc

# With downsampling (auto pre-downsamples to disk)
"$PIXI_PY" -m recovar.command_line pipeline particles.star -o output --mask mask.mrc --downsample 128

# Analyze results
"$PIXI_PY" -m recovar.command_line analyze output --zdim=10
```

Or use the interactive wizard: `recovar quickstart`

See the [quick start guide](https://ma-gilles.github.io/recovar/getting-started/quickstart/) for more examples.

## Documentation

Full documentation is available at **[ma-gilles.github.io/recovar](https://ma-gilles.github.io/recovar)**:

- [Input Data](https://ma-gilles.github.io/recovar/guide/input-data/) — supported formats, path fixing
- [Downsampling](https://ma-gilles.github.io/recovar/guide/downsampling/) — when and how to downsample
- [Running the Pipeline](https://ma-gilles.github.io/recovar/guide/pipeline/) — all options explained
- [Analyzing Results](https://ma-gilles.github.io/recovar/guide/analysis/) — volumes, trajectories, UMAP
- [CLI Reference](https://ma-gilles.github.io/recovar/reference/cli/) — all commands and flags
- [Troubleshooting](https://ma-gilles.github.io/recovar/troubleshooting/) — common issues and fixes

## SLURM Usage (Strict)

Inside every sbatch script, use the same isolation and provenance checks:

```bash
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}_${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}_${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

cd "$WORKDIR"
pixi install
PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip uninstall -y recovar || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed
PYTHON="$PIXI_PY" make -C recovar/cuda clean all

"$PIXI_PY" - <<'PY'
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
print("ENV_OK")
PY
```

## Using the source code

If you'd like to use RECOVAR functions directly in Python (e.g., for custom analysis or integration with other tools), the key modules are:

- `recovar.data_io.cryoem_dataset` — dataset loading (`load_dataset`, `CryoEMDataset`, `CryoEMHalfsets`)
- `recovar.data_io.metadata_readers` — extract poses/CTF from `.star`/`.cs` files
- `recovar.heterogeneity.heterogeneity_volume` — volume generation via kernel regression
- `recovar.heterogeneity.embedding` — latent space embedding
- `recovar.heterogeneity.covariance_estimation` — covariance estimation

See the [Python API reference](https://ma-gilles.github.io/recovar/reference/api/heterogeneity/) for function-by-function documentation.

## Citation

If you use RECOVAR in your research, please cite:

> Gilles, M.A. and Singer, A. (2025). Cryo-EM heterogeneity analysis using regularized covariance estimation and kernel regression. *Proceedings of the National Academy of Sciences*, 122(9), e2419140122. [doi:10.1073/pnas.2419140122](https://doi.org/10.1073/pnas.2419140122)

## Contact

Marc Aurèle Gilles — [gilles@princeton.edu](mailto:gilles@princeton.edu)

Issues and feature requests: [GitHub Issues](https://github.com/ma-gilles/recovar/issues)
