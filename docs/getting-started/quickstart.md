# Quick Start

## Interactive wizard

The easiest way to set up your first run:

```bash
recovar quickstart
```

The wizard walks you through selecting your input files, mask, downsampling, and other options, then generates and optionally executes the pipeline command. Works over SSH with no extra dependencies.

## Manual quick start

### From RELION

```bash
recovar pipeline particles.star -o output --mask mask.mrc
```

### From cryoSPARC

```bash
recovar pipeline particles.cs -o output --mask mask.mrc --datadir /path/to/cryosparc/project
```

### With downsampling

If your images are larger than ~256 pixels, downsample for faster processing:

```bash
recovar pipeline particles.star -o output --mask mask.mrc --downsample 128
```

This automatically pre-downsamples images to disk the first time and caches them for re-runs.

### Analyze the results

```bash
recovar analyze output --zdim=10
```

This runs k-means clustering, generates representative volumes, computes UMAP embeddings, and optionally creates trajectory movies.

### View volumes

Open the generated `.mrc` files in ChimeraX, Chimera, or any MRC viewer:

```
output/output/analysis_10/centers/all_volumes/vol0000.mrc
output/output/analysis_10/centers/all_volumes/vol0001.mrc
...
```

## What's next

- [Tutorial](../guide/tutorial.md) — full worked example with plots on EMPIAR-10076
- [Input Data](../guide/input-data.md) — understand supported formats and data preparation
- [Downsampling](../guide/downsampling.md) — when and how to downsample
- [Running the Pipeline](../guide/pipeline.md) — all pipeline options explained
- [Analyzing Results](../guide/analysis.md) — interpreting and visualizing output
