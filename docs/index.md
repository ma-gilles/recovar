# RECOVAR

**Tools for cryo-EM heterogeneity analysis**

RECOVAR is a software tool for analyzing conformational heterogeneity in cryo-EM and cryo-ET datasets. It reconstructs high-resolution volumes, estimates conformational density and low free-energy motions, and automatically identifies subsets of images with a particular volume feature.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get started in 5 minutes**

    ---

    Install RECOVAR and run your first analysis

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-file-document:{ .lg .middle } **Input any format**

    ---

    Accepts RELION `.star` and cryoSPARC `.cs` files directly

    [:octicons-arrow-right-24: Input Data](guide/input-data.md)

-   :material-magnify:{ .lg .middle } **Analyze results**

    ---

    K-means clustering, trajectories, UMAP, and volume generation

    [:octicons-arrow-right-24: Analysis](guide/analysis.md)

-   :material-cube-outline:{ .lg .middle } **Cryo-ET support**

    ---

    Tilt-series data with focus masks

    [:octicons-arrow-right-24: Cryo-ET](guide/cryo-et.md)

-   :material-monitor-dashboard:{ .lg .middle } **Web GUI**

    ---

    Browser-based interface for job management and interactive analysis

    [:octicons-arrow-right-24: Web GUI](guide/gui.md)

</div>

## Key features

- **High resolution** — achieves higher resolution than other methods in most cases according to [CryoBench](https://cryobench.cs.princeton.edu)
- **Image-to-volume attribution** — automatically extract the set of images that produced a particular volume feature
- **Conformational density estimation** — accurate estimation of probability density in latent space
- **Free-energy motions** — estimation of low free-energy conformational motions
- **Focus masks** — supports focus masks for targeted heterogeneity analysis
- **Cryo-ET support** — tilt-series data with focus masks (same format as cryoDRGN-ET)
- **Transparent volume generation** — kernel regression method produces no hallucinations, useful for validating other methods
- **Web GUI** — browser-based interface for launching jobs, exploring latent spaces, and viewing 3D volumes interactively

## How it works

RECOVAR uses principal component analysis of the 3D covariance to find a low-dimensional latent space describing the heterogeneity. It then uses kernel regression to generate volumes at any point in this space. See the [paper](https://www.pnas.org/doi/abs/10.1073/pnas.2419140122) and [recorded talk](https://www.youtube.com/watch?v=7ycfzGcWOVI) for details.

## Typical workflow

```bash
# 1. Run the pipeline
recovar pipeline particles.star -o output --mask mask.mrc

# 2. Analyze results (k-means, trajectories, UMAP)
recovar analyze output --zdim=10

# 3. Explore results interactively
recovar gui --scan-dir output
# Or view volumes in ChimeraX:
# chimerax output/output/analysis_10/centers/all_volumes/vol0000.mrc
```
