# CLI Commands

All RECOVAR commands follow the pattern `recovar <command> [arguments]`.

## Main workflow

| Command | Description |
|---------|-------------|
| `pipeline` | Run the full heterogeneity analysis pipeline |
| `analyze` | Post-pipeline analysis (k-means, volumes, UMAP) |
| `quickstart` | Interactive wizard for pipeline setup |
| `downsample` | Pre-downsample images to disk |

## Volume generation

| Command | Description |
|---------|-------------|
| `compute_state` | Generate volumes at specific latent coordinates |
| `compute_trajectory` | Compute trajectories and generate volume series |

## Particle selection

| Command | Description |
|---------|-------------|
| `extract_image_subset` | Extract particles based on volume features |
| `extract_image_subset_from_kmeans` | Extract particles from k-means clusters |

## Density estimation

| Command | Description |
|---------|-------------|
| `estimate_conformational_density` | Estimate probability density in latent space |
| `estimate_stable_states` | Identify stable conformational states |

## Quality control

| Command | Description |
|---------|-------------|
| `outlier_detection` | Detect outlier particles |
| `junk_particle_detection` | Detect junk particles |
| `pipeline_with_outliers` | Combined pipeline + outlier detection |

## Advanced

| Command | Description |
|---------|-------------|
| `reconstruct_from_external_embedding` | Volume generation from external latent spaces |
| `compute_embedding` | Compute latent space embeddings |
| `postprocess` | Post-processing and output refinement |

## Testing

| Command | Description |
|---------|-------------|
| `run_test_dataset` | Quick pipeline test on synthetic data |
| `run_test_all_metrics` | Comprehensive metric testing |
| `make_test_dataset` | Generate synthetic test datasets |

## Getting help

For any command:

```bash
recovar <command> -h
```

---

## `pipeline`

Run the full heterogeneity analysis pipeline.

```bash
recovar pipeline particles.star -o output --mask mask.mrc [options]
```

See [Running the Pipeline](../guide/pipeline.md) for full documentation.

---

## `analyze`

Post-pipeline analysis: k-means clustering, volume generation, UMAP, trajectories.

```bash
recovar analyze result_dir --zdim=10 [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--zdim` | Required | Latent dimension (single integer) |
| `-o` | Auto | Output directory |
| `--n-clusters` | 40 | Number of k-means clusters |
| `--n-trajectories` | 0 | Trajectories between cluster pairs |
| `--n-vols-along-path` | 6 | Volumes per trajectory |
| `--Bfactor` | 0 | B-factor sharpening |
| `--n-bins` | 50 | Bins for kernel regression |
| `--skip-umap` | False | Skip UMAP |
| `--skip-centers` | False | Skip cluster center volumes |
| `--normalize-kmeans` | False | Normalize z before k-means |
| `--no-z-regularization` | False | Use unregularized z |
| `--lazy` | False | Lazy loading |

---

## `downsample`

Pre-downsample particle images to disk via Fourier cropping.

```bash
recovar downsample particles.star -D 128 -o downsampled/ [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-D`, `--target-D` | Required | Target box size (even integer) |
| `-o`, `--outdir` | Required | Output directory |
| `--datadir` | None | Base directory for image paths |
| `--strip-prefix` | None | Strip prefix from paths |
| `--batch-size` | 1000 | Images per batch |

See [Downsampling](../guide/downsampling.md) for details.

---

## `compute_state`

Generate volumes at specific latent space coordinates.

```bash
recovar compute_state result_dir -o volumes --latent-points coords.txt [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--latent-points` | Required | Coordinates file (`.txt`) |
| `-o` | Required | Output directory |
| `--Bfactor` | 0 | B-factor sharpening |
| `--n-bins` | 50 | Bins for kernel regression |
| `--maskrad-fraction` | 20 | Kernel radius = `grid_size / value` |
| `--n-min-particles` | 100 | Minimum particles for regression |
| `--particles` | Same | Higher-resolution particle stack |
| `--datadir` | Same | Path prefix for particles |
| `--zdim1` | False | Enable for 1D latent space |

---

## `compute_trajectory`

Compute high-density trajectories through latent space with volume series.

```bash
recovar compute_trajectory result_dir -o trajectory --zdim=10 \
    --endpts centers.txt --ind 0,1 [options]
```

### Endpoint specification (one required)

| Flags | Description |
|-------|-------------|
| `--endpts FILE --ind 0,1` | Lines from coordinate file |
| `--endpts FILE` | First two lines of file |
| `--z_st FILE --z_end FILE` | Separate start/end files |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--zdim` | Required | Latent dimension |
| `-o` | Required | Output directory |
| `--density` | None | Density file for high-density path |
| `--n-vols-along-path` | 6 | Volumes along the path |
| `--Bfactor` | 0 | B-factor sharpening |
| `--n-bins` | 50 | Bins for kernel regression |
| `--maskrad-fraction` | 20 | Kernel radius parameter |

---

## `estimate_conformational_density`

Estimate probability density in latent space using deconvolution.

```bash
recovar estimate_conformational_density result_dir [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pca_dim` | 4 | PCA dimensions for density estimation |
| `--z_dim_used` | 4 | Latent dimension to use |
| `--output_dir` | Auto | Output directory |
| `--percentile_reject` | 10 | Reject % of data with large covariance |
| `--num_disc_points` | 50 | Grid points per dimension |
| `--alphas` | Auto | Regularization values (space-separated) |

!!! note
    Runtime scales exponentially with `--pca_dim`. Keep it at 4 or below.

---

## `extract_image_subset`

Extract particles that produced a specific volume feature.

```bash
recovar extract_image_subset --input-dir vol_dir --output indices.pkl [selection]
```

Selection (one required):

| Flag | Description |
|------|-------------|
| `--mask mask.mrc` | Select by mask center of mass |
| `--coordinate x,y,z` | Select by pixel coordinates |
| `--subvol-idx N` | Select by subvolume index |

---

## `extract_image_subset_from_kmeans`

Extract particles from k-means clusters.

```bash
recovar extract_image_subset_from_kmeans centers.pkl output.pkl indices [-i]
```

| Argument | Description |
|----------|-------------|
| `centers.pkl` | Path to centers.pkl from analyze |
| `output.pkl` | Output indices file |
| `indices` | Comma-separated cluster indices |
| `-i` | Invert selection |
