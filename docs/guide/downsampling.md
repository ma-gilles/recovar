# Downsampling

If your images are larger than 256x256 pixels, downsampling significantly speeds up the pipeline without meaningfully affecting results at the resolutions relevant for heterogeneity analysis.

## Automatic downsampling (recommended)

Use `--downsample D` with the pipeline inside a project:

```bash
recovar init_project my_project
cd my_project
recovar pipeline particles.star --mask mask.mrc --downsample 128 --project .
```

The pipeline automatically:

1. Pre-downsamples all images to disk in the shared project cache (`Cache/downsample/...`)
2. Runs the pipeline on the downsampled images in the numbered Pipeline job directory
3. Reuses the cached downsampled images across matching project runs, instead of tying the cache to one output directory

This is the simplest and recommended approach — one command handles everything.

## Manual pre-downsampling

For maximum control or to materialize a reusable downsampled dataset explicitly:

```bash
# Project-style downsample job
recovar downsample particles.star -D 128 --project . --output-name particles_d128

# Or an explicit standalone output directory
recovar downsample particles.star -D 128 -o downsampled/

# Run pipeline on the materialized downsampled data
recovar pipeline downsampled/particles.128.star -o output --mask mask.mrc
```

### `recovar downsample` options

| Flag | Description |
|------|-------------|
| `-D`, `--target-D` | Target box size (must be even) |
| `-o`, `--outdir` | Output directory (optional in project mode; shared project cache is still used when available) |
| `--datadir` | Base directory for image paths |
| `--strip-prefix` | Strip prefix from paths |
| `--batch-size` | Images per batch (default: 1000) |

### Output files

```
downsampled/
  particles.128.mrcs    # New MRC stack at target resolution
  particles.128.star    # Updated STAR with correct pixel size + image paths
```

When input is `.star`, the output `.star` preserves all metadata (poses, CTF, optics) with updated pixel size and image size. When input is `.cs` or `.mrcs`, a minimal `.star` is created.

## Performance comparison

For 100,000 images at 512x512 downsampled to 128x128:

| Approach | Time |
|----------|------|
| Pre-downsample (one-time) | ~18 minutes |
| Pipeline without downsampling | Baseline |
| Pipeline with `--downsample 128` | Baseline + ~18 min (first run only) |
| On-the-fly (no caching) | Baseline + **~9 hours** |

The pipeline makes ~30 passes over the data. Pre-downsampling avoids repeating the Fourier-crop FFT on every pass.

## How it works

Downsampling uses 2D Fourier cropping:

1. FFT the image to Fourier space
2. Crop the central `D x D` region (keeping low frequencies)
3. Inverse FFT back to real space
4. Normalize to preserve pixel values

This is mathematically equivalent to low-pass filtering and resampling, and is the standard approach used by RELION, cryoSPARC, and cryoDRGN.

## Choosing a box size

- **128** — good default for most datasets, fast
- **256** — higher resolution, needed for very fine structural differences
- **64** — very fast, useful for quick exploration

The box size must be even and no larger than the original image size.

!!! tip
    Start with `--downsample 128` for initial exploration, then re-run at higher resolution if needed for publication figures.

## Using the GUI

In the web GUI (`recovar gui`), you can downsample in two ways:

- **As part of a Pipeline job**: expand **Advanced** options and set the **Downsample** field to your target box size (e.g., 128). The pipeline handles caching automatically.
- **As a standalone job**: click **+ New Job**, select **Downsample** from the Job Type dropdown, browse to your particle file, and set the target box size.

See the [GUI Guide](gui.md) for project setup and job submission.
