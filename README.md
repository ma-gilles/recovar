# RECOVAR: Tools for cryo-EM heterogeneity analysis

RECOVAR analyzes conformational heterogeneity in cryo-EM and cryo-ET datasets. It reconstructs high-resolution volumes, estimates conformational density and low free-energy motions, and automatically identifies image subsets associated with specific volume features.

**[Full Documentation](https://ma-gilles.github.io/recovar)** | **[Paper](https://www.pnas.org/doi/abs/10.1073/pnas.2419140122)** | **[Talk](https://www.youtube.com/watch?v=cQBQlCCRp8Q&t=740s)**

> **Looking for the older release?** This is the current main branch. If you want the previous release (`0.4.5`, possibly more stable but missing recent features like `.cs`/`.star` auto-extraction), install with `pip install recovar==0.4.5` or check out the [`legacy-0.4.5`](https://github.com/ma-gilles/recovar/tree/legacy-0.4.5) branch.

**License**: the code has been modified and is now under the PU-RL v2.0 license, and the code imports libraries that are under non-PU-RL v2.0 (including GPL) licenses. See [LICENSE](LICENSE).

## Key features

- **High resolution** — top performer on [CryoBench](https://cryobench.cs.princeton.edu)
- **Direct input** — accepts RELION `.star` and cryoSPARC `.cs` files (no preprocessing needed)
- **Image-to-volume attribution** — extract images that produced a specific volume feature
- **Conformational density** — estimate free energy landscapes
- **Focus masks** — targeted heterogeneity analysis
- **Cryo-ET support** — tilt-series data with focus masks
- **Transparent volume generation** — kernel regression produces no hallucinations
- **Web GUI** — browser-based interface for launching jobs, exploring latent spaces, and viewing 3D volumes

## Installation

RECOVAR requires Python 3.11. A CUDA GPU is strongly recommended for real workloads, and a CPU-only path is available for testing.

### Quick install (pip)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install "recovar[gpu]"
```

Verify:
```bash
python -c "import jax; print(jax.devices())"
recovar run_test_dataset
```

`recovar[cuda]` remains available as a compatibility alias.

### Development install

For the latest version or contributing:

```bash
git clone https://github.com/ma-gilles/recovar.git
cd recovar

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install -e ".[gpu,dev]"

# Verify
python -c "import jax; print(jax.devices())"
recovar run_test_dataset
```

### CPU-only install

For testing without a GPU (not practical for real datasets):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install recovar
recovar run_test_dataset --cpu
```

### Pixi (alternative)

If you use [pixi](https://prefix.dev/), a `pixi.toml` is provided:

```bash
git clone https://github.com/ma-gilles/recovar.git
cd recovar
pixi install
pixi run install-recovar
pixi run smoke-import-recovar
```

### Native extensions

RECOVAR ships two compiled extensions:

- The fast-marching C++ extension is bundled in published Linux and macOS wheels for supported builds. Source and editable installs build it locally when a C++ compiler is available. If that build fails, RECOVAR falls back to the pure-Python implementation.
- Installing `recovar[gpu]` gives you the CUDA-enabled JAX wheels. On GPU, RECOVAR also tries to build and use its faster custom CUDA backproject/project extension by default. That requires a local CUDA toolkit/compiler reachable through `NVCC`, `CUDACXX`, `PATH`, `LOCAL_CUDA_PATH`, `CUDA_HOME`, or `CUDA_PATH`. You can prebuild it with `recovar build_custom_cuda`. If that custom CUDA build/load fails, RECOVAR stops with fix instructions. `RECOVAR_DISABLE_CUDA=1` forces the slower JAX GPU path as a temporary workaround, but that is not the preferred configuration.

### Docker

See the [Docker & Containers guide](https://ma-gilles.github.io/recovar/getting-started/docker/) for Docker and Apptainer/Singularity instructions.

## Quick start

```bash
# Run the pipeline
recovar pipeline particles.star -o output --mask mask.mrc

# With downsampling (auto pre-downsamples to disk)
recovar pipeline particles.star -o output --mask mask.mrc --downsample 128

# Analyze results
recovar analyze output --zdim=10
```

Or use the **project system** for organized, auto-numbered job directories:

```bash
recovar init_project my_project
cd my_project
recovar pipeline particles.star --mask mask.mrc --project .
recovar analyze Pipeline/job_0001 --zdim=10 --project .
recovar project_status
```

Or use the interactive wizard: `recovar quickstart`

See the [quick start guide](https://ma-gilles.github.io/recovar/getting-started/quickstart/) for more examples.

## Documentation

Full documentation is available at **[ma-gilles.github.io/recovar](https://ma-gilles.github.io/recovar)**:

- [Installation](https://ma-gilles.github.io/recovar/getting-started/installation/) — pip, conda, pixi, Docker
- [Input Data](https://ma-gilles.github.io/recovar/guide/input-data/) — supported formats, path fixing
- [Running the Pipeline](https://ma-gilles.github.io/recovar/guide/pipeline/) — all options explained
- [Analyzing Results](https://ma-gilles.github.io/recovar/guide/analysis/) — volumes, trajectories, UMAP
- [Web GUI](https://ma-gilles.github.io/recovar/guide/gui/) — browser-based interface
- [CLI Reference](https://ma-gilles.github.io/recovar/reference/cli/) — all commands and flags
- [Troubleshooting](https://ma-gilles.github.io/recovar/troubleshooting/) — common issues and fixes

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

Marc Aurele Gilles — [gilles@princeton.edu](mailto:gilles@princeton.edu)

Issues and feature requests: [GitHub Issues](https://github.com/ma-gilles/recovar/issues)
