# Web GUI

RECOVAR includes a browser-based GUI for launching jobs, exploring latent spaces interactively, and viewing 3D volumes — all without writing commands.

## Launching the GUI

```bash
recovar gui
```

This starts a local web server (default: `http://localhost:5000`). Open the URL in your browser.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 5000 | Port to bind to |
| `--host` | 127.0.0.1 | Host to bind to (`0.0.0.0` for remote access) |
| `--scan-dir` | None | Auto-discover existing pipeline outputs in a directory (repeatable) |
| `--debug` | False | Debug mode with auto-reload |
| `--python-path` | Current | Python interpreter for job execution |

### Remote access via SSH

When running on a remote cluster, forward the port through SSH:

```bash
# On your local machine:
ssh -L 5000:localhost:5000 user@cluster

# Then on the cluster:
recovar gui --scan-dir /path/to/results

# Open http://localhost:5000 in your local browser
```

### Auto-discovering existing results

Use `--scan-dir` to import existing pipeline outputs into the GUI:

```bash
# Discover all results under a directory
recovar gui --scan-dir /path/to/results

# Multiple directories
recovar gui --scan-dir /path/to/project1 --scan-dir /path/to/project2
```

The GUI scans for directories containing RECOVAR `model/` folders and imports them automatically.

## Dashboard

The dashboard shows all jobs with status indicators:

- **Running** — currently executing (with SLURM job tracking if on a cluster)
- **Queued** — submitted to SLURM, waiting for resources
- **Completed** — finished successfully
- **Failed** — exited with an error (click to view error details)

Sort jobs by newest, oldest, name, or status. Click any job to open its detail view.

## Creating jobs

Click **New Job** to create a pipeline or analysis run.

### Pipeline job

Configure all pipeline parameters through the form:

- **Particles** — browse to select your `.star`, `.cs`, or `.mrcs` file. The GUI validates the file and shows particle count, pixel size, and available columns.
- **Mask** — select an `.mrc` mask file, or choose `from_halfmaps` / `sphere` for automatic masking.
- **Downsampling** — set the target box size (e.g., 128, 256).
- **Advanced options** — focus mask, contrast correction, lazy loading, tilt-series mode, and more.

### Analyze job

After a pipeline completes, run analysis:

- **Z-dimension** — which latent dimension to use.
- **K-means clusters** — number of clusters.
- **Trajectories** — number of trajectories between most-distant cluster pairs.

### Execution modes

Jobs can run in two modes:

| Mode | Description |
|------|-------------|
| **Local** | Run directly on the current machine |
| **SLURM** | Submit to a SLURM cluster with configurable partition, GPUs, memory, and time |

For SLURM jobs, configure:

- Partition and account
- Number of GPUs, CPUs, and memory
- Time limit
- Extra SBATCH flags

## Exploring results

The job detail view has several tabs for inspecting outputs.

### Latent space explorer

The interactive scatter plot shows all particles in the latent space, with options to:

- **Switch between PCA and UMAP** representations
- **Color by k-means cluster** assignment or use uniform coloring
- **Click a point** to generate a volume at that latent coordinate (computed on demand)
- **Select two points** to compute a trajectory between them

This is the GUI equivalent of manually running `recovar compute_state` or `recovar compute_trajectory` — but interactive.

### 3D volume viewer

View isosurface renderings of any volume directly in the browser:

- Adjustable **sigma threshold** for isosurface level
- **Opacity** controls for transparency
- **Ctrl+click** on multiple volumes in the volume browser to overlay them
- Automatic rendering when you click a point in the latent space explorer

### Slice viewer

Orthogonal slices through any volume:

- Switch between **X** (sagittal), **Y** (coronal), and **Z** (axial) axes
- Navigate through slices with a slider
- Useful for quick inspection of internal density features

### Volume browser

Organized listing of all output volumes, categorized by type:

| Category | Contents |
|----------|----------|
| Reconstruction | Mean volume, filtered mean |
| Half-maps | For FSC calculation |
| Eigenvectors | Principal component volumes |
| Variance | Variance maps |
| Masks | Solvent, dilated, focus, complement |
| K-means centers | Cluster center volumes |
| Trajectories | Volumes along computed paths |

Click any volume to view it in the 3D viewer or slice viewer.

### Result images

Pipeline diagnostic plots (eigenvalue spectrum, mean/eigenvolume slices, FSC, contrast histogram) are displayed as a preview grid.

## GUI vs CLI workflow

The GUI mirrors the CLI workflow, adding interactivity:

| CLI step | GUI equivalent |
|----------|----------------|
| `recovar pipeline ...` | New Job :material-arrow-right: Pipeline form |
| `recovar analyze ...` | New Job :material-arrow-right: Analyze form |
| `recovar compute_state ...` | Click a point in the latent space explorer |
| `recovar compute_trajectory ...` | Select two points in the latent space explorer |
| View `.mrc` in ChimeraX | Built-in 3D viewer and slice viewer |
| Inspect `.png` plots | Result images tab |

!!! tip "Use both"
    The GUI and CLI work on the same output directories. You can run `pipeline` and `analyze` from the command line (or a SLURM script), then launch `recovar gui --scan-dir output` to explore the results interactively. Or do everything through the GUI.

## Requirements

The GUI requires Flask, which is included in the default RECOVAR installation. If Flask is not installed:

```bash
pip install flask
# or
pip install recovar[gui]
```

No other dependencies are required — the GUI uses vendored JavaScript libraries (Plotly, Alpine.js, HTMX, Tailwind CSS) that work offline.
