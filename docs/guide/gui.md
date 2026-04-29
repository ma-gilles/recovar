# Web GUI

RECOVAR includes a browser-based GUI for launching jobs, exploring latent spaces interactively, and viewing 3D volumes — all without writing commands.

## Launching the GUI

```bash
recovar gui
```

This starts a local web server (default: `http://localhost:8080`). Open the URL in your browser.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | Port to bind to |
| `--host` | 127.0.0.1 | Host to bind to (`0.0.0.0` for remote access) |
| `--reload` | False | Auto-reload for development |

### Remote access via SSH

When your data lives on a remote cluster and you want to view the GUI in a browser on your laptop, set up an SSH tunnel:

```bash
# Step 1: On your LOCAL machine, open an SSH tunnel:
ssh -L 8080:localhost:8080 user@cluster

# Step 2: On the CLUSTER (inside the SSH session), launch the GUI:
recovar gui

# Step 3: Open http://localhost:8080 in your local browser
```

The tunnel forwards traffic so the remote server appears local. If port 8080 is taken, use a different port in both the SSH command and `recovar gui --port <port>`.

## Projects

The GUI organizes work into **projects** — directories that contain your pipeline outputs, analyses, and computed volumes. On first launch, create a project by pointing to an existing directory or choosing a new one.

The GUI stores a lightweight SQLite index in each project directory (`recovar_project.db`). This database is fully rebuildable — if you delete it, use **Scan for Existing Jobs** to re-import everything from the filesystem.

### Importing existing results

Click **Scan for Existing Jobs** on the dashboard and point to the directory *containing* your pipeline outputs. For example, if your output is at `/scratch/my_project/Pipeline/job_0001/`, scan `/scratch/my_project/`.

## Dashboard

The dashboard shows all jobs with status indicators:

- **Running** — currently executing (with SLURM job tracking if on a cluster)
- **Queued** — submitted to SLURM, waiting for resources
- **Completed** — finished successfully
- **Failed** — exited with an error (click to view error details)

The sidebar groups jobs by type (Pipeline, Analyze, Compute State, Trajectory, Density, etc.) for quick navigation.

## Creating jobs

Click **New Job** to create a pipeline or analysis run.

### Pipeline job

Configure all pipeline parameters through the form:

- **Particles** — browse to select your `.star`, `.cs`, or `.mrcs` file.
- **Mask** — select an `.mrc` mask file.
- **Downsampling** — set the target box size (e.g., 128, 256).
- **Advanced options** — focus mask, contrast correction, lazy loading, B-factor, and more.

### Analyze job

After a pipeline completes, the **Suggested Next Steps** section offers a one-click link to run analysis with the result directory pre-filled:

- **Z-dimension** — which latent dimension to use (default: 4).
- **K-means clusters** — number of clusters (default: 40).
- **Trajectories** — number of trajectories between most-distant cluster pairs.

### Other job types

- **Compute State** — generate a volume at a specific latent coordinate
- **Compute Trajectory** — generate volumes along a path between two latent points (optionally density-guided)
- **Density Estimation** — estimate conformational density in latent space
- **Stable States** — find local maxima of the conformational density
- **Postprocess** — sharpen and filter volumes
- **Downsample** — pre-downsample particle images

### Executor selection (SLURM or Local GPU)

Each job can be submitted to either **SLURM** (for cluster batch scheduling) or **Local GPU** (direct subprocess on the current machine). When `sbatch` is detected on the system, every job form shows an **executor toggle** so you can choose per job.

#### SLURM execution

On clusters with SLURM, the SLURM settings panel appears when the SLURM executor is selected:

- Partition and account
- Number of GPUs, CPUs, and memory
- Time limit
- Extra SBATCH directives (for advanced users)

#### Local GPU execution

When the Local GPU executor is selected, the local settings panel lets you configure:

- **GPU picker** — select which GPU(s) to use (sets `CUDA_VISIBLE_DEVICES`)
- **Setup command** — a shell command run before the pipeline (e.g., `module load cudatoolkit/12.8`)
- **Environment variables** — extra env vars for the job

#### Jobs survive GUI restarts

Jobs submitted through the GUI (both SLURM and local) are tracked in the project database. If the GUI server restarts, it reconnects to all in-flight jobs automatically — no work is lost.

### Settings page

The **Settings page** (gear icon in the sidebar) lets you configure SLURM and local execution defaults at two levels:

- **User-global** — stored in `~/.config/recovar/config.toml`, applies to all projects
- **Per-project** — stored in `<project_dir>/recovar.toml`, overrides user-global for that project

The page shows an **effective defaults** summary with provenance badges (built-in / user / project) so you can see where each value comes from. Defaults are layered: built-in < user-global < per-project < per-job form override.

## Exploring results

### Latent space explorer

The **Explore** page shows interactive scatter plots of the full particle latent space (50K+ points at 60 fps), with:

- **PCA and UMAP** projections side by side
- **Color by** k-means cluster, point density, or deconvolved conformational density
- **Selectable zdim** — switch between latent dimensions (1, 2, 4, 10, 20, etc.)
- Axis selectors for PCA components

#### Point clicks

Click a particle or k-means center to see its full latent coordinate vector and density value. From there:

- **Compute State** — submit a job to generate a volume at that point
- **Compute Trajectory** — select two points and submit a trajectory job (with optional density-guided path)

#### Lasso / rectangle / polygon selection

Use the selection tools to draw a region on the scatter plot:

- See the number of selected particles
- **Export .star** — one-click export of selected particles as a RELION .star file (if the original .star is available)
- **Export .ind** — export particle indices as an index file
- A link to **rerun pipeline** with the exported subset appears immediately

### 3D volume viewer

View isosurface renderings of any volume directly in the browser:

- Adjustable **sigma threshold** for isosurface level
- **Slice view** with axis selection and slider navigation
- Click volumes in the **Volumes** tab to load them

### Volume browser

Organized listing of all output volumes, categorized by type (K-means centers, trajectories, eigenvectors, half-maps, etc.). Toggle display of half-map and unfiltered volumes.

### Result images

Pipeline diagnostic plots (eigenvalue spectrum, mean FSC, contrast histogram) are displayed as a preview grid on the job detail page.

## GUI vs CLI workflow

The GUI mirrors the CLI workflow, adding interactivity:

| CLI step | GUI equivalent |
|----------|----------------|
| `recovar pipeline ...` | New Job → Pipeline form |
| `recovar analyze ...` | New Job → Analyze form (or click "Suggested Next Step") |
| `recovar compute_state ...` | Click a point in the latent space explorer |
| `recovar compute_trajectory ...` | Select two points in the latent space explorer |
| View `.mrc` in ChimeraX | Built-in 3D viewer and slice viewer |
| Inspect `.png` plots | Result images tab |
| `recovar estimate_conformational_density ...` | New Job → Density Estimation form |

!!! tip "Use both"
    The GUI and CLI work on the same output directories. You can run `pipeline` and `analyze` from the command line, then launch `recovar gui` and scan the directory to explore the results interactively. Or do everything through the GUI.

## Requirements

The GUI requires FastAPI, uvicorn, SQLAlchemy, and a few other packages, which are all included when you install with:

```bash
pip install "recovar[gui]"
```

This installs the `gui` extra, which includes `fastapi`, `uvicorn`, `python-multipart`, `sqlalchemy`, `aiofiles`, and `tomli_w` (for writing TOML settings files).

The frontend is pre-built and bundled — no Node.js required at runtime.
