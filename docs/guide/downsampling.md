# Downsampling

Downsampling shrinks the image box size before processing. It significantly speeds up the pipeline without meaningfully affecting results at the resolutions relevant for heterogeneity analysis.

!!! info "The pipeline downsamples to 256 by default"
    `recovar pipeline` runs with `--downsample 256`, so images larger than 256x256 are automatically downsampled to a box size of 256. If your images are already at or below 256, the step is skipped.

## Choosing a different box size

By default the pipeline downsamples to 256. To use a different target, pass `--downsample D`; to keep the original box size, pass `--no-downsample`:

```bash
recovar init_project my_project
cd my_project

# Default: downsamples to a box size of 256
recovar pipeline particles.star --mask mask.mrc --project .

# Smaller box for faster runs (e.g. 128)
recovar pipeline particles.star --mask mask.mrc --downsample 128 --project .

# Keep the original box size
recovar pipeline particles.star --mask mask.mrc --no-downsample --project .
```

The pipeline caches the downsampled images in the shared project cache and reuses them across matching runs, so you only pay the downsampling cost once.

## Running the pipeline and volumes at different box sizes

If the full-resolution pipeline is too slow, run the pipeline at a smaller box size and then reconstruct your final volumes at a higher resolution. The latent space comes from the fast low-resolution run, but `compute_state` can regenerate volumes at full resolution by pointing `--particles` at the original (un-downsampled) stack:

```bash
# Fast pipeline + analysis at box size 128
recovar pipeline particles.star --mask mask.mrc --downsample 128 --project .
recovar analyze output --zdim=10

# Recompute selected volumes at full resolution
recovar compute_state output -o volumes_highres \
    --latent-points output/analysis_10/kmeans/centers.txt \
    --particles particles.star --Bfactor=50
```

## Using the GUI

In the web GUI (`recovar gui`), the **Pipeline** form downsamples to 256 by default — expand **Advanced** and change the **Downsample** field to pick a different box size (or disable it). To reconstruct higher-resolution volumes from a low-resolution run, submit a **Compute State** job and point it at the original particle stack.

See the [GUI Guide](gui.md) for project setup and job submission.
