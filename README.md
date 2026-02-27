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

## Using the source code

If you'd like to use RECOVAR functions directly in Python (e.g., for custom analysis or integration with other tools), the key modules are:

- `recovar.dataset` — dataset loading (`load_dataset`, `CryoEMDataset`)
- `recovar.metadata_parsing` — extract poses/CTF from `.star`/`.cs` files
- `recovar.heterogeneity_volume` — volume generation via kernel regression
- `recovar.embedding` — latent space embedding
- `recovar.covariance_estimation` — covariance estimation

## Citation

If you use RECOVAR in your research, please cite:

> Gilles, M.A. and Singer, A. (2024). Heterogeneity analysis of cryo-EM datasets using 3D covariance estimation and kernel regression. *Proceedings of the National Academy of Sciences*, 122(3), e2419140122.

## Contact

Marc Aurele Gilles — [mg6942@princeton.edu](mailto:mg6942@princeton.edu)

Issues and feature requests: [GitHub Issues](https://github.com/ma-gilles/recovar/issues)
