# RECOVAR: Tools for cryo-EM heterogeneity analysis

RECOVAR analyzes conformational heterogeneity in cryo-EM and cryo-ET datasets. It reconstructs high-resolution volumes, estimates conformational density and low free-energy motions, and automatically identifies image subsets associated with specific volume features.

**[Full Documentation](https://ma-gilles.github.io/recovar)** | **[Paper](https://www.pnas.org/doi/abs/10.1073/pnas.2419140122)** | **[Talk](https://www.youtube.com/watch?v=cQBQlCCRp8Q&t=740s)**

> **Looking for the older release?** Active development happens on the `dev` branch. If you want the previous stable release (`0.4.5`, possibly more stable but missing recent features like `.cs`/`.star` auto-extraction), install with `pip install recovar==0.4.5` or check out the [`legacy-0.4.5`](https://github.com/ma-gilles/recovar/tree/legacy-0.4.5) branch.

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
python3.11 -m venv recovar_env
source recovar_env/bin/activate
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

python3.11 -m venv recovar_env
source recovar_env/bin/activate
pip install -U pip

pip install -e ".[gpu,dev]"

# Verify
python -c "import jax; print(jax.devices())"
recovar run_test_dataset
```

### CPU-only install

For testing without a GPU (not practical for real datasets):

```bash
python3.11 -m venv recovar_env
source recovar_env/bin/activate
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

**Minimum GPU compute capability: 7.0** (NVIDIA Volta or newer). The custom CUDA kernel ships precompiled targets for sm_70, sm_75, sm_80, sm_86, sm_89, sm_90 plus a compute_75 PTX fallback. For Pascal (sm_60/61) or other archs not in the default set, rebuild locally:

```bash
cd recovar/cuda
make clean
make CUDA_ARCH="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60"
```

As a temporary alternative, set `RECOVAR_DISABLE_CUDA=1` to use the slower JAX-native path (≈2x slower; matches recovar 0.4.5 behavior). For one-off runs on small datasets that's fine; for production, rebuild.

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

## GPU memory

Every heavy-GPU command (`pipeline`, `analyze`, `compute_state`, `compute_trajectory`, `pipeline_with_outliers`, `reconstruct_from_external_embedding`, `junk_particle_detection`, `outlier_detection`, `run_test_dataset`) accepts the same memory-planning flags. They control RECOVAR's batch-size and PC choices — they do **not** cap JAX's allocation. JAX-level memory behavior is controlled separately via `XLA_PYTHON_CLIENT_MEM_FRACTION` and `XLA_PYTHON_CLIENT_PREALLOCATE`.

```bash
# Tell RECOVAR to size batches as if the GPU has only 40 GB available.
# Useful when the GPU is shared, or you want to leave headroom for
# another process. (Soft hint to RECOVAR; not a JAX cap.)
recovar pipeline ... --gpu-budget-gb 40

# Adapt n_pcs to the largest value that fits the budget (reproducible:
# same flags + same dataset = same n_pcs).
recovar pipeline ... --gpu-budget-gb 24 --adaptive-n-pcs

# Tighten batch sizes further for tight budgets.
recovar pipeline ... --gpu-budget-gb 12 --low-memory-option
recovar pipeline ... --gpu-budget-gb 8  --very-low-memory-option

# memory_plan.json is always written to <outdir>/_diagnostics/.
# For per-phase memory_trace.jsonl, args.json, allocator_env.json,
# and heavyweight JAX-profiler captures, add --memory-profile.
recovar pipeline ... --gpu-budget-gb 40 --memory-profile
```

The planner never refuses to launch. If it predicts the run will exceed the budget (based on a calibrated peak-memory table when present, or the heuristic in `covariance_estimation` when absent), it logs a loud WARNING and launches anyway. If the run actually OOMs, the error message is followed by an actionable hint suggesting `--gpu-budget-gb`, `--adaptive-n-pcs`, `--low-memory-option`, etc. — the hint is the **last** thing on stderr so it doesn't get lost above the JAX traceback.

The peak-memory table at `recovar/utils/memory_calibration_data.json` is **optional** — when present, the planner uses it to predict per-phase peaks (so the warning above is more accurate) and to drive `--adaptive-n-pcs`. When **absent**, `--adaptive-n-pcs` falls back to the same heuristic in `covariance_estimation.get_default_covariance_computation_options` that walks `n_pcs` down from 200 until predicted memory fits the budget. To populate the table on your hardware, run `scripts/submit_calibrate_memory_planner.sh` (Slurm) and then `pixi run python scripts/aggregate_memory_calibration.py`.

`run_test_dataset` always splices `--adaptive-n-pcs` into its inner pipeline calls so the install-sanity test always finishes. Pass `--full-memory-test` if you specifically want the fixed 200-PC, non-adaptive configuration.

### Workstation / shared-GPU OOM

If you OOM on a workstation or shared GPU even after passing `--gpu-budget-gb`, the underlying cause is usually JAX's default *preallocation* behavior — JAX grabs ~90 % of physical VRAM on first allocation, regardless of what RECOVAR plans. This is orthogonal to RECOVAR's batch-size budget. The fix is a JAX env var:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
recovar pipeline ...
```

That makes JAX allocate on demand, so the run can succeed if the *actual* peak is smaller than `MEM_FRACTION × physical`. Recommended for any non-Slurm-exclusive GPU; for dedicated cluster allocations, leave preallocation on for the small startup-perf win.

### CUDA-fallback env var

The canonical CUDA-fallback env var is `RECOVAR_DISABLE_CUDA=1`. The common typo `RECOVAR_CUDA_DISABLE` is treated as an alias for the duration of the run, with a one-time warning telling you to rename it in your shell init.

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
