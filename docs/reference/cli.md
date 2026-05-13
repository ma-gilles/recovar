# CLI Commands

All RECOVAR commands follow the pattern `recovar <command> [arguments]`.

Project mode is the standard workflow: pass `--project <dir>` (or run from inside a project), let RECOVAR place outputs in stable numbered directories such as `Pipeline/job_0001/`, and use project metadata for readable names in the CLI and GUI. Standalone explicit paths remain supported.

## Main workflow

| Command | Description |
|---------|-------------|
| `pipeline` | Run the full heterogeneity analysis pipeline |
| `analyze` | Post-pipeline analysis (k-means, volumes, UMAP) |
| `gui` | Launch the web GUI (default: `http://localhost:8080`) |
| `quickstart` | Interactive wizard for pipeline setup |
| `init_project` | Initialize a project directory with auto-numbered jobs |
| `project_status` | Show status of all jobs in a project |
| `downsample` | Pre-downsample images to disk |
| `parse_relion5_tomo` | Convert RELION5 tilt-series data to 2D tilt format |

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
| `postprocess` | Post-processing and output refinement |

## Diagnostics

| Command | Description |
|---------|-------------|
| `check_paths` | Preview how image paths resolve without running the pipeline |
| `build_custom_cuda` | Pre-build the custom CUDA backproject/project extension |

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
recovar pipeline particles.star --mask mask.mrc --project .
# or: recovar pipeline particles.star -o output --mask mask.mrc [options]
```

See [Running the Pipeline](../guide/pipeline.md) for full documentation.

---

## `analyze`

Post-pipeline analysis: k-means clustering, volume generation, UMAP, trajectories.

```bash
recovar analyze --zdim=10 --project .
# or: recovar analyze result_dir --zdim=10 [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--zdim` | Required | Latent dimension (single integer) |
| `-o` | Auto | Output directory |
| `--n-clusters` | 20 | Number of k-means clusters |
| `--n-trajectories` | 0 | Trajectories between cluster pairs |
| `--n-vols-along-path` | 6 | Volumes per trajectory |
| `--density` | None | Density `.pkl` file for trajectory guidance |
| `--Bfactor` | 0 | B-factor sharpening |
| `--n-bins` | 50 | Bins for kernel regression |
| `--maskrad-fraction` | 20 | Kernel radius = `grid_size / value` |
| `--skip-umap` | False | Skip UMAP (recommended for >200k particles) |
| `--skip-centers` | False | Skip cluster center volumes |
| `--normalize-kmeans` | False | Normalize z before k-means |
| `--no-z-regularization` | False | Use unregularized z |
| `--lazy` | False | Lazy loading |
| `--particles` | Same | Higher-resolution particle stack (overrides pipeline stack) |
| `--datadir` | Same | Path prefix for particle paths |
| `--strip-prefix` | None | Strip prefix from paths in star file |
| `--apply-global-filtering` | False | Apply global FSC filtering to half-maps |

---

## `downsample`

Pre-downsample particle images to disk via Fourier cropping.

```bash
recovar downsample particles.star -D 128 --project . --output-name particles_d128
# or: recovar downsample particles.star -D 128 -o downsampled/ [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-D`, `--target-D` | Required | Target box size (even integer) |
| `-o`, `--outdir` | Auto in project mode | Output directory |
| `--datadir` | None | Base directory for image paths |
| `--strip-prefix` | None | Strip prefix from paths |
| `--batch-size` | 1000 | Images per batch |
| `--chunk-size` | None | Split output into chunks of this many images |

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
| `--kernel-regression-mode` | `standard` | `standard`, `deconvolved`, or experimental `local_poly` |
| `--deconv-lambda-grid` | Built-in grid | Comma-separated lambda grid for `deconvolved` |
| `--local-poly-degree` | 3 | Polynomial degree for `local_poly` |
| `--local-poly-bandwidth-multipliers` | Built-in grid | Comma-separated bandwidth multipliers for `local_poly` |

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
| `--z_dim_used` | Auto | Latent dimension to use (smallest zdim >= pca_dim) |
| `--output_dir` | Auto | Output directory |
| `--percentile_reject` | 10 | Reject % of data with large covariance |
| `--num_disc_points` | Auto | Grid points per dimension (50 for dim>3, 100 for dim=3, 200 for dim=2) |
| `--alphas` | Auto | Regularization values (space-separated) |

!!! note
    Runtime scales exponentially with `--pca_dim`. Keep it at 4 or below.

---

## `extract_image_subset`

Extract particles that produced a specific volume feature.

```bash
recovar extract_image_subset vol_dir --output indices.pkl [selection]
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
recovar extract_image_subset_from_kmeans kmeans_result.pkl output_dir indices [-i]
```

Output is written to `output_dir/indices.pkl`.

| Argument | Description |
|----------|-------------|
| `kmeans_result.pkl` | Path to `data/kmeans_result.pkl` from analyze |
| `output_dir` | Output directory (indices saved as `indices.pkl` inside) |
| `indices` | Comma-separated cluster indices |
| `-i` | Invert selection |

---

## `parse_relion5_tomo`

Convert RELION5 tilt-series data to RECOVAR's 2D tilt format.

```bash
recovar parse_relion5_tomo \
    -t Polish/job249/tomograms.star \
    -p Extract/job260/particles.star \
    -o particles_2d.star
```

| Flag | Default | Description |
|------|---------|-------------|
| `-t`, `--tomograms` | Required | RELION5 `tomograms.star` (from Polish or Tomograms job) |
| `-p`, `--particles` | Required | RELION5 `particles.star` (from Extract or Refine job) |
| `-o`, `--output` | `particles_2d.star` | Output 2D STAR file |
| `--tilt-dim` | Auto | Tilt image dimensions in pixels (auto-detected from MRC headers) |
| `-v`, `--verbose` | False | Enable verbose logging |

See [Cryo-ET](../guide/cryo-et.md#importing-from-relion5) for usage details.

---

## `gui`

Launch the web GUI for interactive job management and result exploration.

```bash
recovar gui [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | Port to bind to |
| `--host` | 127.0.0.1 | Bind address (`0.0.0.0` for remote access) |
| `--reload` | False | Auto-reload for development |

See the [GUI Guide](../guide/gui.md) for full documentation.

---

## `init_project`

Initialize a new project directory with auto-numbered job tracking.

```bash
recovar init_project [directory] [--name "Project Name"]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `directory` | `.` | Directory to initialize (created if needed) |
| `--name` | Directory name | Human-readable project name |

Creates a `recovar_project.db` file in the directory. Subsequent commands using `--project` will auto-generate numbered job directories (e.g. `Pipeline/job_0001/`, `Analyze/job_0001/`).

---

## `project_status`

Show status of all jobs in a project.

```bash
recovar project_status [directory] [--tree]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `directory` | `.` | Project directory |
| `--tree` | False | Show job dependency tree instead of table |

---

## Common flag: `--project`

All commands that produce output accept `--project <dir>` to enable project mode. This is the recommended way to run RECOVAR. When active, output directories are auto-generated, downstream commands may omit `result_dir` to use the latest completed Pipeline job, and RECOVAR stores human-readable job names alongside the numbered directories. If you run from within a project directory (containing `recovar_project.db`), it is auto-detected without needing the flag.

Each job creates `job.json`, `command.txt`, `run.log`, and `README.txt` metadata files.

### Project directory naming

| CLI command | Directory name | Example |
|-------------|----------------|---------|
| `pipeline` | `Pipeline/` | `Pipeline/job_0001/` (alias shown separately in CLI/GUI) |
| `analyze` | `Analyze/` | `Analyze/job_0001/` |
| `compute_state` | `ReconstructState/` | `ReconstructState/job_0001/` |
| `compute_trajectory` | `ReconstructTrajectory/` | `ReconstructTrajectory/job_0001/` |
| `estimate_conformational_density` | `Density/` | `Density/job_0001/` |
| `junk_particle_detection` | `JunkDetection/` | `JunkDetection/job_0001/` |
| `outlier_detection` | `OutlierDetection/` | `OutlierDetection/job_0001/` |
