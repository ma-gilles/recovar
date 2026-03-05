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

## Installation

```bash
conda create --name recovar python=3.11 -y
conda activate recovar
pip install git+https://github.com/scikit-fmm/scikit-fmm.git "jax[cuda12]"==0.9.0.1 recovar
```

Verify: `recovar run_test_dataset`

See the [installation guide](https://ma-gilles.github.io/recovar/getting-started/installation/) for development setup and alternative methods.

**Docker/HPC:** For containerized GPU environments, see the [Docker & Containers guide](https://ma-gilles.github.io/recovar/getting-started/docker/).

## Quick start

```bash
# Run the pipeline
recovar pipeline particles.star -o output --mask mask.mrc

# With downsampling (auto pre-downsamples to disk)
recovar pipeline particles.star -o output --mask mask.mrc --downsample 128

# Analyze results
recovar analyze output --zdim=10
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

## Development setup (pixi)

For development and running tests, use [pixi](https://pixi.sh) (not the conda/pip install above):

```bash
git clone git@github.com:ma-gilles/heterogeneity_dev.git && cd heterogeneity_dev
pixi install                      # creates .pixi/envs/default with all deps
pixi run install-recovar          # editable install of recovar into the env
pixi run smoke-import-recovar     # quick check
```

Run tests (requires a GPU node on HPC):

```bash
# Unit tests
.pixi/envs/default/bin/python -m pytest tests/unit/ -v --ignore=tests/unit/test_gui_app.py

# Or via pixi tasks
pixi run test-fast                # unit + smoke
pixi run test-full                # all tests including GPU and integration
```

The CUDA kernels (`recovar/cuda/libcuda_backproject.so`) are auto-compiled on first use via `make`. The Makefile uses the running Python to locate JAX FFI headers, so always run tests through the pixi environment.

**HPC/SLURM notes:** Set `PYTHONNOUSERSITE=1` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` in SBATCH scripts. Use `--exclusive` or verify the GPU is free to avoid OOM from shared GPU memory.

### Clean clone + GPU queue run (recommended)

Use this flow when you want a fresh checkout and a reproducible GPU test job:

```bash
git clone git@github.com:ma-gilles/heterogeneity_dev.git ~/myscratch/heterogeneity_dev_codex
cd ~/myscratch/heterogeneity_dev_codex
pixi install --locked || pixi install
pixi run install-recovar

# Submit GPU tests against the exact current commit
sbatch --export=ALL,RECOVAR_REPO_ROOT="$(pwd)",RECOVAR_EXPECT_COMMIT="$(git rev-parse HEAD)",RECOVAR_FORCE_CUDA_REBUILD=1,RECOVAR_REQUIRE_CUDA=1 scripts/run_gpu_tests.slurm
```

Alternative helper (clone/refresh + setup + submit in one step):

```bash
GIT_REF=main bash scripts/download_recovar_dev_and_run_test.sh
```

This avoids common CUDA mismatch issues by:
- pinning the commit used by the queued job (`RECOVAR_EXPECT_COMMIT`)
- forcing a clean CUDA extension rebuild on the node (`RECOVAR_FORCE_CUDA_REBUILD=1`)
- failing fast if CUDA kernels are unavailable (`RECOVAR_REQUIRE_CUDA=1`)

## Using the source code

If you'd like to use RECOVAR functions directly in Python (e.g., for custom analysis or integration with other tools), the key modules are:

- `recovar.data_io.dataset` — dataset loading (`load_dataset`, `CryoEMDataset`, `CryoEMHalfsets`)
- `recovar.data_io.metadata_parsing` — extract poses/CTF from `.star`/`.cs` files
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
