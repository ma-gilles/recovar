# RECOVAR GUI v2 — Specification

## Overview

A professional, browser-based GUI for the RECOVAR heterogeneity analysis toolbox. Replaces the existing Flask/Alpine.js GUI with a modern React + FastAPI application designed for HPC cluster deployment, with local-GPU fallback.

**Design philosophy:** Professional, commercial-grade polish and usability. Drag/click over type. Accessible to users with no coding knowledge. Progressive disclosure — simple by default, powerful when needed.

**Target audience:** The global cryo-EM community, from day one. Grad students, postdocs, and PIs who understand cryo-EM but may not be comfortable with command lines.

---

## Architecture

### Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Frontend** | React 18+ / TypeScript / Vite | Largest ecosystem for 3D (vtk.js), interactive charts, and complex UIs. Used by Dagster, Prefect, OHIF — the most polished scientific web tools. |
| **UI Components** | Shadcn/ui (Radix + Tailwind CSS) | Accessibility-first primitives with full styling control. Used by OHIF. Modern, clean aesthetic. |
| **API** | FastAPI (Python, async) | Auto-generates OpenAPI spec. Faster than Flask. Async support for WebSocket streaming. |
| **API Client** | OpenAPI codegen → TypeScript | Type-safe frontend-backend contract. Types can't drift. Like Prefect's approach. |
| **3D Rendering** | vtk.js | Built for scientific volume data. Isosurface (marching cubes), slice views, interactive widgets (sphere, box, plane). Used by OHIF, Trame, ParaView Web. |
| **Scatter Plots** | regl-scatterplot | WebGL, smooth at 1M+ points. Built-in lasso selection and linked views. Purpose-built for large scatter plots (genomics cell atlases). |
| **Other Charts** | Plotly.js | Interactive FSC curves, histograms, eigenvalue spectra. Hover, zoom, click. |
| **Saved Outputs** | Matplotlib (unchanged) | Pipeline code continues generating static PNGs for job output directories. |
| **Real-time** | WebSocket (native) | Server pushes log lines, status changes, intermediate results. True real-time, not polling. |
| **Database** | SQLite | Single file, zero setup, safe concurrent access, queryable. HPC-friendly. |
| **State Management** | TanStack Query (React Query) | Server state caching, automatic refetching, optimistic updates. |
| **Routing** | TanStack Router | Type-safe, file-based routing with data prefetching. |

### Directory Structure

```
recovar/gui_v2/
├── SPEC.md                    # This file
├── CLAUDE.md                  # AI assistant instructions for this codebase
├── backend/                   # FastAPI Python backend
│   ├── main.py               # App factory, startup, middleware
│   ├── api/                  # API route modules
│   │   ├── jobs.py           # Job CRUD, submission, status
│   │   ├── volumes.py        # Volume serving, slicing, mask export
│   │   ├── embeddings.py     # Latent coordinates, UMAP, k-means
│   │   ├── subsets.py        # Particle subset management
│   │   ├── project.py        # Project CRUD, import, scan
│   │   ├── files.py          # File browser, validation
│   │   ├── system.py         # System info, SLURM detection, disk usage
│   │   └── ws.py             # WebSocket endpoints (logs, status stream)
│   ├── models/               # SQLAlchemy/Pydantic models
│   │   ├── job.py
│   │   ├── project.py
│   │   ├── subset.py
│   │   └── mask.py
│   ├── services/             # Business logic
│   │   ├── slurm.py          # SLURM submission, status polling, template rendering
│   │   ├── local_runner.py   # Local subprocess execution (no-SLURM mode)
│   │   ├── scanner.py        # Scan directories for existing pipeline outputs
│   │   └── notifier.py       # Browser notification triggers
│   ├── db.py                 # SQLite connection, migrations
│   └── config.py             # Settings (global defaults, SLURM template, paths)
├── frontend/                  # React TypeScript frontend
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── src/
│   │   ├── main.tsx          # Entry point
│   │   ├── routes/           # File-based routing (TanStack Router)
│   │   │   ├── __root.tsx    # Root layout (sidebar + main)
│   │   │   ├── index.tsx     # Dashboard / home
│   │   │   ├── jobs/
│   │   │   │   ├── new.tsx   # New job form
│   │   │   │   └── $jobId.tsx # Job detail view
│   │   │   ├── explore/      # Latent space explorer
│   │   │   ├── masks/        # Mask editor
│   │   │   ├── subsets/      # Particle subset browser
│   │   │   ├── compare/      # Job comparison view
│   │   │   ├── settings/     # Global settings
│   │   │   └── external/     # External tools (RELION plugin)
│   │   ├── components/       # Reusable React components
│   │   │   ├── ui/           # Shadcn/ui primitives
│   │   │   ├── volume-viewer/ # vtk.js 3D viewer
│   │   │   ├── latent-explorer/ # regl-scatterplot wrapper
│   │   │   ├── mask-editor/  # Mask creation wizard + tools
│   │   │   ├── job-form/     # Dynamic job parameter forms
│   │   │   ├── file-browser/ # Server filesystem browser
│   │   │   ├── log-viewer/   # WebSocket log streamer
│   │   │   └── sidebar/      # Project tree sidebar
│   │   ├── hooks/            # Custom React hooks
│   │   ├── lib/              # Utilities, API client (generated)
│   │   └── styles/           # Global styles, Tailwind config
│   └── tests/                # Vitest + Testing Library
├── plugins/                   # External tool plugins
│   └── relion/               # RELION integration
│       ├── __init__.py       # Plugin registration
│       ├── jobs.py           # RELION job type definitions
│       ├── detect.py         # Auto-detect RELION installation
│       └── frontend/         # React components for RELION job forms
├── tests/                     # Backend tests (pytest) + E2E (Playwright)
│   ├── backend/
│   ├── e2e/
│   └── fixtures/
└── scripts/                   # Build, dev server, deployment helpers
```

### Communication Model

```
Browser (React SPA)
  |-- HTTP REST --> FastAPI (JSON responses)
  |-- WebSocket --> FastAPI (real-time logs, status, notifications)
  '-- vtk.js / regl-scatterplot (client-side rendering, no server round-trip)

FastAPI Backend
  |-- SQLite (project state, job metadata, provenance)
  |-- Filesystem (MRC volumes, .star files, job outputs)
  |-- SLURM (sbatch submit, sacct status) OR local subprocess
  '-- recovar Python library (import directly for metadata parsing)
```

---

## Remote Access

Three supported modes, covering the most common HPC deployment scenarios:

1. **SSH tunnel (default):** `ssh -L 8080:localhost:8080 user@cluster`. Simplest, most secure. The GUI prints the correct SSH command on startup.
2. **Direct network access with lightweight auth:** Email/password or token-based authentication for trusted lab networks. Not hardened for internet exposure.
3. **Reverse proxy (nginx + HTTPS):** Documented nginx configuration with WebSocket upgrade support. For institutional deployments at `https://recovar.institution.edu`.

On startup, the server auto-detects the environment and prints appropriate connection instructions.

---

## Project Model

### Concept

A project is a **directory on the filesystem** that contains all job outputs, the SQLite database, masks, subsets, and configuration. Input data (particle .star files, tomograms) can live anywhere and are referenced by path.

```
/scratch/user/my_project/
├── recovar_project.db          # SQLite database (jobs, subsets, provenance, config)
├── Pipeline/                   # Pipeline job outputs
│   ├── P001/                  # Job P001
│   └── P002/
├── Analyze/                   # Analyze job outputs
│   └── A001/
├── ComputeState/
├── ComputeTrajectory/
├── Density/
├── Masks/                     # Exported masks
├── Subsets/                   # Exported particle subsets (.ind, .star)
├── External/                  # External tool outputs (RELION, etc.)
│   └── Relion/
│       └── R001/
└── .recovar/                  # Internal config
    ├── settings.toml          # Project settings (SLURM defaults, paths)
    └── slurm_template.sh      # Editable sbatch template
```

### Import / Scan

The GUI can scan a directory (or the project directory itself) for existing recovar pipeline outputs and import them as completed jobs. This enables adopting the GUI for existing work done via CLI.

### Provenance

Every entity (job, subset, mask) has a full provenance chain stored in SQLite:
- Jobs record: parent job(s), parameters, input files, output files, SLURM job ID, timestamps
- Subsets record: source job, zdim, selection method (lasso coordinates, cluster IDs, threshold), parent subset
- Masks record: source volume, creation method, parameters (threshold, dilation, smoothing)

A "Show provenance" button on any entity displays its full history chain.

---

## Layout

Sidebar + main panel (standard pattern for project-based scientific tools):

```
+-----------------------------------------------------------+
| = RECOVAR          [project name]            bell  gear   |
+-----------+-----------------------------------------------+
| PROJECT   |                                               |
|           |  [Context-dependent main panel]                |
| + New Job |                                               |
|           |  When viewing a completed job:                 |
| > Pipeline|  +------------------+--------------------+    |
|   P001 ok |  |  3D Volume       |  Latent Space      |    |
|   P002 .. |  |  Viewer          |  Explorer           |    |
|           |  |  (vtk.js)        |  (regl-scatter)     |    |
| > Analyze |  +------------------+--------------------+    |
|   A001 ok |                                               |
|           |  Suggested: [Run Analyze >]                    |
| > Subsets |                                               |
|   sel_01  |  +-------------------------------------------+|
|           |  | Logs | Params | Images | Volumes          ||
| > Masks   |  | [tab content]                             ||
|   mask_01 |  +-------------------------------------------+|
|           |                                               |
| > External|  Disk: 45.2 GB / 500 GB                       |
|  (RELION) |                                               |
+-----------+                                               |
| Compare   |                                               |
| Cmd+K     |                                               |
+-----------+-----------------------------------------------+
```

- Sidebar is collapsible
- Status indicators: ok=completed, ..=running, X=failed, ??=queued
- Sidebar shows disk usage at bottom
- Cmd+K spotlight search for quick navigation

### Dark Mode

Dark theme by default. Light mode toggle available in settings. Clean, modern aesthetic with Shadcn/ui components.

---

## Core Features

### 1. Job Management

Every recovar CLI command is available as a job type in the GUI:

| Job Type | CLI Command | Priority |
|----------|------------|----------|
| Pipeline | `recovar pipeline` | Core |
| Analyze | `recovar analyze` | Core |
| Compute State | `recovar compute_state` | Core |
| Compute Trajectory | `recovar compute_trajectory` | Core |
| Density Estimation | `recovar estimate_conformational_density` | Core |
| Stable States | `recovar estimate_stable_states` | Core |
| Postprocess | `recovar postprocess` | Core |
| Downsample | `recovar downsample` | Secondary |
| Extract Subset | `recovar extract_image_subset` | Secondary |
| Extract by K-means | `recovar extract_image_subset_from_kmeans` | Secondary |
| Outlier Detection | `recovar outlier_detection` | Secondary |
| Junk Detection | `recovar junk_particle_detection` | Secondary |
| External Embedding | `recovar reconstruct_from_external_embedding` | Secondary |

**Job forms** use progressive disclosure:
- **Default view:** Only essential parameters (particles, mask, output name, zdim)
- **Advanced section** (collapsed): All other parameters with sensible defaults
- **Raw command preview:** Shows the exact CLI command that will be run
- **Contextual help:** Every parameter has a tooltip (hover or ? icon) explaining what it does, typical range, and when to change it. E.g. "zdim: Number of latent dimensions. Start with 4-10. Higher values capture more heterogeneity modes but require more particles and compute time."

**Input validation before submission:**
- Verify .star/.cs file exists and is parseable; display particle count and box size
- Check mask dimensions match particle box size
- Warn if zdim seems high relative to particle count
- Estimate disk usage and memory requirements; warn if they exceed available resources
- Validate that referenced pipeline outputs (for analyze, trajectory, etc.) exist and are complete
- All validation happens client-side or via quick API calls — no GPU time wasted on bad inputs

**Job lifecycle:** Draft -> Queued -> Running -> Completed / Failed
- Jobs can be cloned (copy parameters to new job)
- Jobs can be cancelled (SLURM `scancel` or subprocess kill)
- Failed jobs show error message prominently with log tail

**Parameter sweeps:**
- On any job form, mark one or more parameters as "sweep" and provide a list of values (e.g. zdim: 2, 4, 6, 8, 10)
- Submits one job per value combination, all in parallel via SLURM
- When all jobs complete, auto-opens the comparison view with results side by side
- Common sweeps: zdim, downsampling factor, mask choice

**Suggested next steps:** After a job completes, the UI shows contextual suggestions:
- Pipeline -> "Run Analyze" (pre-filled with pipeline output path)
- Analyze -> "Compute Trajectory" or "Estimate Density"
- Density -> "Compute Trajectory between stable states"

### 2. SLURM / Local Execution

**Auto-detect mode:** On startup, check for `sbatch`. If available: cluster mode. If not: local mode (subprocess). Same UI either way.

**Global defaults** stored in project settings:
```toml
[slurm]
partition = "cryoem"
account = "amits"
gpus = 1
cpus = 4
memory = "300G"
time = "12:00:00"
extra_flags = "--exclusive"

[slurm.environment]
PYTHONNOUSERSITE = "1"
XLA_PYTHON_CLIENT_PREALLOCATE = "false"

[staging]
# RECOVAR_CACHE_DIR: local staging directory for MRC particle stacks.
# Copies particle data to fast local storage on first access; all subsequent
# reads (across all pipeline passes) hit the local copy instead of the
# parallel filesystem. Typically gives 3-6x speedup.
#
# SLURM mode: defaults to /dev/shm (RAM-backed, always available, fastest).
#   Alternatives: $TMPDIR (node-local NVMe, if cluster supports --tmp),
#   or /local/scratch/$(whoami).
# Local mode: defaults to "" (disabled, no staging).
recovar_cache_dir = "/dev/shm"
```

**Editable sbatch template:** Power users can edit the raw template in Settings. Template uses Jinja2 variables (`{{ partition }}`, `{{ gpus }}`, etc.) filled from the form.

**Per-job overrides:** Each job form shows SLURM settings pre-filled from global defaults, editable before submission.

**Live resource monitoring (running jobs):**
- Poll `nvidia-smi` on the compute node during execution (via SSH or SLURM job wrapper)
- Display in the job detail view: GPU utilization %, GPU memory usage, temperature
- Show estimated time remaining based on log progress parsing (e.g. "Pass 3/5, ~12 min left")
- CPU and system memory usage from `/proc/stat` and `/proc/meminfo`

### 3. Volume Viewer (vtk.js)

**Primary view: 3D isosurface rendering**
- Marching cubes isosurface at adjustable sigma threshold
- Real-time threshold slider (surface updates live)
- Professional lighting/shading (ambient, diffuse, specular)
- Opacity control per volume
- Zoom, rotate, pan (standard 3D controls)
- Camera reset button

**Multi-volume display:**
- By default, clicking a new volume **replaces** the current one
- A "pin" toggle (checkbox or lock icon) on each loaded volume keeps it visible when loading the next one
- Each pinned volume gets a distinct color from a palette and its own independent contour level slider
- Volumes panel lists all currently displayed volumes with per-volume controls (color, opacity, contour level, visibility toggle, remove)
- Primary use case: compare two reconstructions side by side, or overlay a mask on top of a density map

**Trajectory movie playback:**
- When viewing a trajectory job (series of volumes), a playback bar appears: play / pause / scrub / loop
- Animates through conformational states as a smooth movie in the 3D viewer
- Adjustable playback speed
- "Export movie" button: render as MP4 or GIF for presentations and publications

**Volume difference maps:**
- Select two volumes -> compute A-B difference -> render as colored isosurface (blue=negative, red=positive)
- Shows exactly what changes between two conformational states
- Accessible from the comparison view or from multi-volume display

**Local resolution coloring:**
- Color an isosurface by local resolution values (from FSC or variance maps)
- Smooth color gradient (e.g. blue=high-res -> red=low-res) mapped onto the surface
- Helps assess reconstruction quality spatially

**Secondary view: Orthogonal slices**
- X/Y/Z axis slice navigation with slider
- Overlay mask contours on slices

**Performance:**
- Volumes loaded as binary MRC from server
- Marching cubes runs in browser (WebGL)
- Subsampling for large volumes during interaction, full resolution on release

### 4. Latent Space Explorer (regl-scatterplot)

**Interactive scatter plot** of particle latent coordinates:
- **Axes:** Selectable PC dimensions (PC1 vs PC2, PC1 vs PC3, etc.)
- **Representations:** PCA, UMAP (both available simultaneously in linked views)
- **Color by:** None, K-means cluster, density, custom continuous value
- **Scale:** Smooth at 1M+ particles via WebGL rendering

**Point interactions (click):**
- **Single click** on any plot places a colored marker (e.g. red dot) at that location
  - The corresponding point appears as a **matching colored marker in ALL other plots** (all UMAP and PCA views), so the user can see where one point in one coordinate system maps to in all others
  - Clicking triggers a **Compute State** job at that latent coordinate, generating a volume
  - The volume viewer loads the result when ready
- **Click a second point** (different color, e.g. blue dot) to define a pair
  - Both points are cross-highlighted in all views
  - A **Compute Trajectory** button appears, pre-filled with the two latent coordinates as start/end
- **K-means centers** can be overlaid as large markers on all plots
  - Clicking a k-means center selects it as a point (same behavior as above)
  - Useful for density estimation workflows: overlay centers, then pick start/end for trajectories

**Region selection tools:**
- Freehand lasso: draw arbitrary region
- Polygon: click to define vertices
- Rectangle: drag to select box region

**Linked views:** Everything is cross-linked across all visible plots:
- A lasso selection in one plot (e.g. UMAP) highlights the same particles in all other plots (PCA views, other UMAP projections)
- A point marker in one plot shows the corresponding marker in all others
- Volume viewer updates to show the mean of selected particles (for region selections)

**Actions on region selection:**
- "Export as subset" -> creates a named subset in the project with full provenance
- "Export all except selection" -> inverse selection
- "Compute volume at centroid" -> launch compute_state job at the centroid of selected particles
- "Rerun pipeline on subset" -> launch a new pipeline job with `--ind` set to this subset

### 5. Particle Subset Management

**Subsets are first-class entities** in the project:
- Named, stored in SQLite with full provenance
- Visible in sidebar under "Subsets"
- Can be used as input to any job (via `--ind` parameter, auto-filled)

**Provenance tracking:**
- Source: which job's analysis generated the latent space
- Method: lasso region coordinates, cluster IDs, threshold values
- Parent: if refined from another subset
- "Show provenance" button traces the full chain

**Export formats:**
- `.ind` file (recovar native) -- particle index list
- RELION `.star` file -- full metadata for selected particles
- Extensible handler pattern for future formats (.cs, etc.)

### 6. Mask Editor

**Guided workflow (primary UX):**

1. **Select source volume** -- mean reconstruction, variance map, or any MRC in the project
2. **Adjust isosurface threshold** -- slider with live 3D preview
3. **Generate mask** -- binarize at threshold, apply dilation and Gaussian smoothing
   - Dilation (voxels): slider, default based on resolution
   - Soft edge width (voxels): slider, default = 5 x resolution / pixel_size
4. **Live preview** -- mask overlaid on original volume in different color/opacity
5. **Refine** (optional) -- use editing tools on the preview:
   - Adjust threshold/dilation/smoothing sliders (preview updates live)
   - Use sphere eraser, box clip, or slice brush for manual editing
6. **Accept & export** -- save as MRC, auto-linked to project

**Editing tools (available at step 5):**

| Tool | Description |
|------|------------|
| **3D Sphere eraser** | Translucent sphere positioned on isosurface. Keep inside or outside. Resize with scroll wheel. |
| **Box clipping** | Define 3D box, keep contents. For slabs and clean cuts. |
| **Slice brush** | Switch to 2D slice view, paint/erase with adjustable brush size. For precision. |
| **Boolean ops** | Union, intersection, subtraction of two masks. |
| **Auto-segment** | Watershed segmentation (like ChimeraX Segger). Volume splits into regions. Click to keep/discard. |
| **Undo/redo** | Full history stack. Experiment freely. |

**Post-processing (always applied):**
- Dilation by N voxels
- Gaussian smoothing for soft edges
- Preview before final export

### 7. Job Comparison View

Select 2+ jobs to compare:
- **Volume overlay:** Isosurfaces from different jobs rendered together with distinct colors
- **FSC curves:** On the same axes, different colors per job
- **Metrics table:** Resolution, eigenvalues, particle count, timing side by side
- **Parameter diff:** Highlight which parameters differ between jobs

Accessible from sidebar ("Compare" button) or by multi-selecting jobs.

### 8. External Tools -- RELION Plugin

**Architecture:** Plugin system under `plugins/`. RELION is the first built-in plugin. Pattern is extensible for future tools (cryoSPARC, other processing packages, etc.).

**Visual distinction:** When working with RELION jobs, the UI accent color shifts subtly. Sidebar section clearly labeled "External > RELION". Makes it obvious you're running RELION, not recovar.

**Auto-detection:** On startup, check `which relion_refine_mpi`. If not found, section is hidden or shows configuration instructions.

**Built-in RELION job types (v1):**

| Job Type | Purpose | Key Integration |
|----------|---------|-----------------|
| **Refine3D** | High-res refinement of a recovar subset | Auto-fills .star from subset, reference from k-means center |
| **PostProcess** | Auto-sharpen + gold-standard FSC | Takes Refine3D output |
| **CtfRefine** | Per-particle CTF refinement | Improves resolution after subset selection |
| **Class3D** | Further classification within a subset | Uses recovar subset as input |
| **Polish** | Bayesian polishing (particle trajectories) | Common refinement step |

**Smart defaults:** When launching a RELION job from a recovar subset:
- Particle .star file: auto-filled from subset export
- Reference volume: auto-filled from nearest k-means center or trajectory state
- Mask: auto-filled from project masks
- Symmetry: inherited from pipeline job parameters

**SLURM submission:** Uses the same sbatch template with MPI-specific additions (`--ntasks`, `mpirun` prefix).

### 9. Cryo-ET Support

**v1 (implemented):**
- Tilt-series metadata import: parse RELION5 tomo .star files via `parse_relion5_tomo`
- Validate and configure `--tilt-series` mode for pipeline jobs
- GUI file browser recognizes tomo-specific file types

**Planned (TODO):**
- RELION tomo re-extract subtomograms (from recovar-selected subsets)
- RELION tomo refine subtomograms
- Per-tilt weighting visualization
- Integration with Warp/M metadata formats

---

## UX Features

### Onboarding

**First-run wizard:**
1. Welcome screen with recovar overview
2. Create or open a project (pick a directory on scratch)
3. Configure SLURM defaults (auto-detected where possible):
   - Partition, account, GPU count, memory, time limit
   - **Staging directory (RECOVAR_CACHE_DIR):** Explained in tooltip: "Copies particle image stacks to fast local storage on first access. All subsequent reads use the local copy instead of the parallel filesystem, giving 3-6x speedup. Set to /dev/shm (RAM-backed) or node-local NVMe."
     - **SLURM mode default:** `/dev/shm` (RAM-backed, always available, fastest). User can change to `$TMPDIR` (node-local NVMe) or a custom path.
     - **Local mode default:** Disabled (empty string). User can opt in by setting a path to a fast local SSD.
   - Environment variables (PYTHONNOUSERSITE, XLA_PYTHON_CLIENT_PREALLOCATE, etc.)
4. Select particle .star file (file browser, defaults to project dir)
5. Launch first pipeline job with guided parameter selection
6. "Try with example data" -- generates a small synthetic test dataset and runs a demo pipeline

**Tutorial dataset:** One-click **generation** (not download) of a synthetic dataset using recovar's built-in `make_test_dataset` command (128^3 box, 100K images). Same generator used by the test suite. Demonstrates the full workflow (pipeline -> analyze -> density -> trajectory -> subset selection -> mask creation).

### Settings -- Progressive Disclosure

**Three layers, each hidden behind the previous:**

1. **Job forms:** Only essential parameters visible by default. "Advanced" expandable section for everything else.
2. **Settings page** (gear icon): Global SLURM defaults, file browser bookmarks, theme toggle, notification preferences.
3. **Config file on disk** (`project/.recovar/settings.toml`): Full configuration for power users. Editable in any text editor. The GUI reads this file.

### File Browser

- Built-in filesystem browser (navigates server filesystem)
- **Defaults to project directory** (not root)
- Quick-access bookmarks: project dir, scratch dir, recent paths
- Shows file metadata (size, date, type)
- Validates .star files inline (particle count, columns present)
- Remembers last-used directory per file type

### Keyboard Shortcuts

- **Cmd+K / Ctrl+K:** Spotlight search -- navigate to any job, subset, mask, or action
- **N:** New job
- **Esc:** Back / close modal
- **Enter:** Submit form / confirm action
- Standard: Cmd+Z undo, Cmd+S save (in mask editor)

### Browser Notifications

When a job completes or fails, send a browser push notification. User gets alerted even if they're in another tab. Configurable in settings.

### Disk Usage Tracking

- Project total size displayed in sidebar footer
- Per-job disk usage shown in job detail
- Warning banner when scratch space falls below threshold (configurable)

### Publication Figure Export

"Export" button on any visualization (volume viewer, scatter plot, FSC curve, comparison view):
- Formats: PNG (high-res), SVG, PDF
- Customizable: labels, scale bars, colormaps, font sizes, background color
- Presets: "Journal" (300 DPI, white background), "Presentation" (dark background, larger fonts)

---

## Testing Strategy

**Every user action has a test.** Full-stack testing:

| Layer | Tool | What's Tested |
|-------|------|---------------|
| **Backend API** | pytest + httpx | Every endpoint: CRUD, validation, error cases, SLURM mock |
| **Database** | pytest | Schema, migrations, provenance queries, concurrent access |
| **React Components** | Vitest + Testing Library | Rendering, user interaction, state changes |
| **3D Viewer** | Vitest + canvas mock | Volume loading, isosurface threshold changes, tool interactions |
| **E2E Workflows** | Playwright | Full user flows: create project -> submit job -> view results -> select particles -> export subset -> create mask |
| **RELION Plugin** | pytest + Playwright | Job form rendering, parameter validation, SLURM submission with MPI |

**CI:** All tests run on every PR. Backend and frontend tests in parallel. E2E tests run against a test server with mock SLURM.

**Test dataset:** Synthetic dataset (small, fast) included in the repo for E2E tests. Same dataset used by the tutorial wizard.

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Initial page load | < 2s |
| Volume viewer: load + render 128^3 MRC | < 1s |
| Volume viewer: load + render 256^3 MRC | < 3s |
| Isosurface threshold slider response | < 100ms |
| Scatter plot: render 100K points | < 500ms |
| Scatter plot: render 1M points | < 2s |
| Lasso selection response | < 200ms |
| Job status update latency (WebSocket) | < 500ms |
| File browser directory listing | < 1s |

---

## Implementation Phases

### Phase 1: Foundation (MVP)
- Project creation/opening
- FastAPI backend with SQLite
- React shell with sidebar layout, dark theme
- Pipeline + Analyze job forms with SLURM submission
- Input validation and contextual help tooltips on all parameters
- Job status monitoring (WebSocket logs)
- Basic volume viewer (vtk.js isosurface, threshold slider, multi-volume pin/overlay)
- File browser (defaults to project dir)
- Auto-detect SLURM vs local mode
- SSH tunnel access

### Phase 2: Interactive Analysis
- Latent space explorer (regl-scatterplot, lasso selection, linked PCA+UMAP)
- Point interactions: click to compute state, two-point trajectory, cross-plot markers
- K-means center overlay on latent plots
- Particle subset management (create, export .ind/.star, provenance)
- All remaining job types (density, trajectory, compute_state, etc.)
- Suggested next steps after job completion
- Job cloning
- Diagnostic plots/images tab (Plotly interactive charts)
- Trajectory movie playback in volume viewer (play/pause/scrub, MP4/GIF export)

### Phase 3: Mask Editor
- Guided threshold workflow with live preview
- 3D sphere eraser
- Box clipping + boolean operations
- Slice brush painting
- Auto-segmentation (watershed)
- Undo/redo
- Mask export (MRC)

### Phase 4: Polish & Extras
- RELION plugin (Refine3D, PostProcess, CtfRefine, Class3D, Polish)
- Job comparison view + volume difference maps + local resolution coloring
- Parameter sweeps (multi-value jobs with auto-comparison)
- Live GPU/resource monitoring for running jobs
- Onboarding wizard + tutorial dataset generation (128^3, 100K images)
- Cmd+K spotlight search
- Browser notifications
- Disk usage tracking
- Publication figure export
- Lightweight auth (email/password)
- Reverse proxy documentation (nginx)
- Cryo-ET tilt-series import

### Phase 5: Future
- Multi-user support (user accounts, permissions)
- Cryo-ET tomo re-extract/re-refine via RELION plugin
- Additional external tool plugins (cryoSPARC, other processing packages)
- Workflow templates (predefined multi-job chains)
- Visual DAG builder for power users
- Additional export formats (.cs, etc.)
- Warp/M metadata import

---

## Key Design Decisions Log

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Frontend framework | React + TypeScript + Vite | Largest ecosystem for 3D/interactive UIs, used by best-in-class tools |
| 2 | UI components | Shadcn/ui (Radix + Tailwind) | Accessibility-first, fully customizable, modern aesthetic |
| 3 | Backend | FastAPI (Python, async) | OpenAPI codegen, WebSocket support, async, faster than Flask |
| 4 | API design | REST + OpenAPI codegen | Type-safe, simple, well-understood |
| 5 | 3D rendering | vtk.js | Built for scientific volumes, has interactive widgets for mask editing |
| 6 | Scatter plots | regl-scatterplot | Smooth at 1M+ points, built-in lasso + linked views |
| 7 | Other charts | Plotly.js | Interactive, rich chart types, good for small-data charts |
| 8 | Database | SQLite | Zero-setup, HPC-friendly, queryable, safe concurrent access |
| 9 | Real-time | WebSocket | True real-time, no polling, server push |
| 10 | Remote access | SSH tunnel + auth + reverse proxy | Three modes covering all HPC deployment scenarios |
| 11 | Project model | Directory-based | Self-contained, portable, builds on existing recovar concept |
| 12 | Job chaining | Suggested next steps | Simple, discoverable, no upfront planning needed |
| 13 | Settings | Progressive disclosure (3 layers) | Simple for beginners, powerful for experts |
| 14 | Theme | Dark mode default | Industry standard for scientific tools, light mode toggle available |
| 15 | Multi-user | Single-user now, multi-user ready schema | Avoids complexity, schema supports future upgrade |
| 16 | Execution | Auto-detect SLURM vs local | Works on clusters and local GPU machines transparently |
| 17 | Testing | Full stack (pytest + vitest + Playwright) | Every action tested, CI on every PR |
| 18 | RELION integration | Plugin system, 5 job types built-in | Extensible pattern, covers key post-heterogeneity workflow |
| 19 | Export formats | .ind + RELION .star, extensible handler | Covers native + universal cryo-EM exchange format |
| 20 | Layout | Sidebar + main panel | Proven pattern for project-based tools, collapsible, context-dependent main area |
