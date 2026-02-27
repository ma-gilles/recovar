# Running the Pipeline

The RECOVAR pipeline takes particle images and a mask, then computes the mean reconstruction, covariance, principal components, and embeddings.

## Basic usage

```bash
# RELION star file
recovar pipeline particles.star -o output --mask mask.mrc

# cryoSPARC cs file
recovar pipeline particles.cs -o output --mask mask.mrc --datadir /project/

# With downsampling
recovar pipeline particles.star -o output --mask mask.mrc --downsample 128

# Legacy pickle files
recovar pipeline particles.128.mrcs -o output \
    --poses poses.pkl --ctf ctf.pkl --mask mask.mrc
```

## Required arguments

| Argument | Description |
|----------|-------------|
| `particles` | Input particles (`.star`, `.cs`, `.mrcs`, or `.txt`) |
| `-o`, `--outdir` | Output directory |
| `--mask` | Solvent mask (`.mrc`), or `from_halfmaps`, `sphere`, `none` |

## Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--downsample D` | None | Downsample images to box size D (pre-downsamples to disk) |
| `--poses` | Auto | Poses file (`.pkl`). Auto-extracted from `.star`/`.cs` |
| `--ctf` | Auto | CTF file (`.pkl`). Auto-extracted from `.star`/`.cs` |
| `--focus-mask` | None | Focus mask for targeted heterogeneity |
| `--mask-dilate-iter` | 0 | Dilate the mask by this many iterations |
| `--zdim` | 1,2,4,10,20 | PCA dimensions for embedding |
| `--only-mean` | False | Only compute the mean (fast, for verifying setup) |
| `--correct-contrast` | False | Estimate and correct amplitude scaling |
| `--lazy` | False | Lazy loading for large datasets |
| `--multi-gpu` | False | Enable multi-GPU parallelization |

## Dataset loading options

| Flag | Default | Description |
|------|---------|-------------|
| `--datadir` | None | Path prefix for resolving relative image paths |
| `--strip-prefix` | None | Strip prefix from image paths |
| `--ind` | None | Filter to specific image indices (`.pkl`) |
| `--n-images` | All | Number of images to use |
| `--halfsets` | None | Pre-computed half-set split (`.pkl`) |
| `--padding` | 0 | Real-space padding |

## Advanced options

| Flag | Default | Description |
|------|---------|-------------|
| `--noise-model` | radial | Noise model: `radial` or `white` |
| `--mean-fn` | triangular | Mean function: `triangular`, `old`, `triangular_reg` |
| `--gpu-gb` | All | GPU memory limit in GB |
| `--n-gpus` | All | Number of GPUs to use |
| `--keep-intermediate` | False | Save intermediate results |
| `--accept-cpu` | False | Allow running without GPU |
| `--ignore-zero-frequency` | False | Useful if images are normalized to zero mean |
| `--low-memory-option` | False | Lower memory for covariance estimation |
| `--very-low-memory-option` | False | Lowest memory for covariance estimation |

## Cryo-ET options

| Flag | Default | Description |
|------|---------|-------------|
| `--tilt-series` | False | Use tilt-series data |
| `--tilt-series-ctf` | Auto | CTF model: `cryoem`, `relion5`, `warp` |
| `--dose-per-tilt` | From file | Dose per tilt |
| `--angle-per-tilt` | From file | Angle per tilt |
| `--ntilts` | All | Number of tilts per series |

See [Cryo-ET](../advanced/cryo-et.md) for details.

## Output structure

```
output/
  command.txt              # Command that was run
  run.log                  # Full log
  downsampled/             # Pre-downsampled images (if --downsample used)
    particles.128.mrcs
    particles.128.star
  output/
    volumes/
      mean.mrc             # Mean reconstruction
      mean_half1_unfil.mrc # Unfiltered half-map 1
      mean_half2_unfil.mrc # Unfiltered half-map 2
      mask.mrc             # Solvent mask used
    analysis_*/            # Results per zdim (after running analyze)
```

## Tips

!!! tip "Quick setup check"
    Use `--only-mean` for a fast run that only computes the mean reconstruction. This verifies your data, mask, and CTF are correct before committing to a full run.

!!! tip "Large datasets"
    For datasets > 500k particles, use `--lazy` for lazy loading and `--downsample 128` for speed. Consider `--n-images 100000` for initial exploration.

!!! tip "Memory"
    If you run out of GPU memory, try `--low-memory-option` or `--very-low-memory-option`. You can also limit memory with `--gpu-gb 8`.
