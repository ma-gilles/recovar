# Phase 1 Spec — MVP

The minimum GUI that is worth using over the CLI. Everything not listed here is a **non-goal for Phase 1**.

See `VISION.md` for the full long-term vision.

---

## Acceptance Criteria

Phase 1 is done when a user can:

1. Open the GUI in a browser (via SSH tunnel), create a project, select a .star file, configure a pipeline job, and submit it (to SLURM or local GPU) — without touching a terminal.
2. Watch real-time logs while the job runs.
3. Open the completed pipeline output: view the mean volume as a 3D isosurface, browse eigenvolumes, view diagnostic plots.
4. Submit an analyze job on that pipeline output (suggested as next step).
5. Open the analyze results: explore the latent space as a scatter plot, click a k-means center to compute a volume at its coordinates, click two k-means centers to compute a trajectory between them.
6. Draw a lasso on the scatter plot, export the selection as a .ind file.
7. Do all of the above on a machine without SLURM (local mode), transparently.

If any of these steps is broken or unreasonably slow, Phase 1 is not done.

---

## Non-Goals for Phase 1

These are explicitly deferred. Do not implement, even partially:

- **Authentication / direct network access / reverse proxy.** SSH tunnel only.
- **RELION plugin or any external tool integration.**
- **Mask editor.** Users use ChimeraX for masks and point the GUI at the resulting .mrc file.
- **Parameter sweeps.**
- **Cryo-ET tilt-series specific UI.** (The pipeline's `--tilt-series` flag works from the CLI; the GUI just passes it through as a text field.)
- **Publication figure export.**
- **Onboarding wizard / tutorial dataset generation.**
- **Job comparison view.**
- **Browser push notifications.** (In-app status badges are sufficient.)
- **GPU/resource monitoring during jobs.**
- **Trajectory movie playback.**
- **Volume difference maps / local resolution coloring.**
- **Auto-segmentation (watershed).**
- **.star file export for subsets.** (.ind only for Phase 1.)
- **Cmd+K spotlight search.**
- **Multi-user or per-user isolation.**

---

## Scope

### Project Model
- Create new project (pick a directory). Creates `recovar_project.db` (SQLite GUI index) alongside the existing `project.json` (CLI canonical metadata). Uses `{JobType}/job_{NNNN}` directory naming to match the existing CLI convention.
- Open existing project (detects `project.json` or `recovar_project.db`).
- Scan a directory for existing pipeline outputs and import them as completed jobs.
- SQLite database with WAL mode. Schema defined in SQLAlchemy with Alembic migrations.
- **SQLite is a rebuildable index.** If `recovar_project.db` is deleted, scanning reconstructs job state from the filesystem. CLI outputs (`project.json`, `job.json`, `command.txt`, `run.log`) remain the canonical source of truth. See `CLAUDE.md` compatibility contract.

### Job Types (Phase 1)

Only these four job types have GUI forms:

| Job Type | Required Fields | Advanced Fields (collapsed) |
|----------|----------------|----------------------------|
| **Pipeline** | particles (.star/.cs/.mrcs), mask (.mrc or special value), output name | zdim, downsample, lazy, correct-contrast, focus-mask, datadir, strip-prefix, n-images, halfsets, poses, ctf, tilt-series, and all other CLI flags |
| **Analyze** | result_dir (auto-filled from pipeline), zdim (dropdown from available zdims) | n-clusters, n-trajectories, output name |
| **Compute State** | result_dir, zdim, latent coordinate (auto-filled from scatter plot click; see access paths below) | output name |
| **Compute Trajectory** | result_dir, zdim, z_start, z_end (auto-filled from two scatter plot clicks; see access paths below) | n-vols-along-path, output name |

All other recovar commands (density, stable_states, postprocess, downsample, extract_subset, etc.) are not in the GUI for Phase 1. Users run them from the CLI.

**Compute State / Compute Trajectory: dual access paths.**

These forms are reachable in two ways:

1. **From the latent explorer** (primary path): User clicks a k-means center or particle in the scatter plot. The explorer pre-fills `result_dir`, `zdim`, and the latent coordinate(s), then shows a compact confirmation dialog: "Compute volume at [z1, z2, z3, z4]? [Submit]". This is the expected flow for most users.

2. **From "+ New Job" in the sidebar** (secondary path): User selects "Compute State" or "Compute Trajectory" from the job type dropdown. The form shows `result_dir` as a file browser (defaulting to existing pipeline outputs in the project) and `zdim` as a dropdown populated after selecting a result_dir. The latent coordinate field is a text input accepting comma-separated values. This path exists for power users who know the coordinates they want (e.g., copied from a script).

### Input Validation

Before submission, the GUI validates (blocking submission on failure):

| Check | Rule | Error Message |
|-------|------|---------------|
| Particles file exists | `os.path.isfile(path)` | "File not found: {path}" |
| Particles file readable | Parse header of .star/.cs/.mrcs | "Cannot parse {path}: {error}" |
| Mask file exists | File exists or is a special value (`from_halfmaps`, `sphere`, `none`) | "Mask file not found: {path}" |
| Mask dimensions match | If .mrc mask: `mask.shape == (D, D, D)` where D = particle box size | "Mask box size {mask_D} does not match particle box size {particle_D}" |
| Output directory writable | `os.access(parent_dir, os.W_OK)` | "Cannot write to {parent_dir}" |
| Result dir valid (analyze/state/trajectory) | `ResultPaths(dir).metadata_path` exists and is parseable | "Not a valid pipeline output: {dir}" |
| zdim valid (analyze) | Requested zdim is in the set computed by the pipeline | "zdim={z} not found in pipeline output. Available: {available}" |

Warnings (shown but don't block submission):

| Check | Rule | Warning |
|-------|------|---------|
| Particle count low | `n_particles < 10_000` | "Only {n} particles. Results may be unreliable below ~10K." |
| Disk space low | `shutil.disk_usage(project_dir).free < 50 * 1e9` | "Less than 50 GB free on {filesystem}." |

**Validation timeout:** All file validation checks (star parsing, mask dimension reading) have a 10-second server-side timeout. If the parallel filesystem is under heavy load, the validation returns a timeout warning rather than blocking indefinitely. The frontend shows a spinner during validation and the timeout message if it fires. See `docs/API.md` for endpoint details.

### Contextual Help

Every parameter in every job form has a tooltip (? icon next to the label). Tooltip text is sourced from a single `tooltips.json` file so it can be reviewed and updated in one place. Example entries:

```json
{
  "pipeline.particles": "Input particle images. Accepts .star (RELION), .cs (cryoSPARC), .mrcs (MRC stack), or .txt (list of .mrcs paths).",
  "pipeline.mask": "Solvent mask defining the molecular envelope. Use 'from_halfmaps' to auto-generate from consensus reconstruction, 'sphere' for a spherical mask, or provide a .mrc file.",
  "pipeline.zdim": "Latent space dimensions to compute (comma-separated). Each value runs an independent embedding. Start with the default (1,2,4,10,20). Higher values capture more heterogeneity modes but are slower.",
  "pipeline.downsample": "Downsample particle images to this box size before processing. Default 256. Lower values are faster but lose high-resolution information. Set to original box size or use --no-downsample to disable.",
  "pipeline.lazy": "Enable lazy loading. Required when particle images are too large to fit in memory. Slower due to repeated disk reads (mitigated by RECOVAR_CACHE_DIR staging).",
  "pipeline.correct_contrast": "Correct per-particle contrast variation (amplitude scaling). Recommended for datasets with heterogeneous ice thickness or variable defocus.",
  "analyze.zdim": "Which latent dimension to analyze. Must be one of the zdim values computed by the pipeline.",
  "analyze.n_clusters": "Number of k-means clusters for partitioning the latent space. Default 20. More clusters = finer partitioning but smaller per-cluster particle counts.",
  "analyze.n_trajectories": "Number of linear trajectories to compute through the latent space. Default 0 (skip). Each trajectory generates a series of volumes."
}
```

### SLURM / Local Execution

See `ADR-001-executor-security.md` for the full executor design. Summary:

- **Auto-detect:** On startup, check `shutil.which("sbatch")`. When found, `executor_mode` is `"both"` — both SLURM and local executors are available simultaneously.
- **Per-job selection:** When both executors are available, each job form shows an ExecutorSelector toggle (SLURM Cluster / Local GPU). The user picks at submit time.
- **SLURM mode:** Jobs submitted via `sbatch`. Status polled via `squeue`/`sacct`. Logs tailed from SLURM output files.
- **Local mode:** Jobs run as subprocesses. Status tracked via PID. Logs captured from stdout/stderr. Supports GPU selection (`CUDA_VISIBLE_DEVICES`), setup commands (e.g. `module load cudatoolkit/12.8`), and extra env vars via `local_opts`.
- **Settings page:** Available at `/settings` for configuring SLURM and local execution defaults at user-global and per-project levels. Replaces editing TOML files by hand.
- **Defaults layering:** Built-in defaults -> user-global `~/.config/recovar/config.toml` -> project `recovar.toml` -> per-job form override. Both `[slurm]` and `[local]` sections are supported in TOML config.
- **Staging:** `RECOVAR_CACHE_DIR` configured in settings. Default: `/dev/shm` (SLURM), disabled (local).

### Volume Viewer

**Capabilities for Phase 1:**
- vtk.js isosurface rendering of MRC volumes
- Sigma threshold slider (continuous, updates isosurface in real time)
- Opacity slider
- Zoom, rotate, pan (mouse controls)
- Camera reset
- Multi-volume: default replaces; "pin" checkbox keeps current volume visible when loading next. Each pinned volume has independent contour slider and color.
- Maximum 4 pinned volumes simultaneously (hard limit to bound browser memory)
- Orthogonal slice view (X/Y/Z toggle, slice index slider)

**Not in Phase 1:** trajectory playback, difference maps, local resolution coloring, mask editing tools.

**Performance targets** (tested on Chrome 120+, 16 GB RAM laptop):

| Metric | Target | Test Volume |
|--------|--------|-------------|
| Load + first render | < 1.5s | 128^3 MRC |
| Load + first render | < 4s | 256^3 MRC |
| Threshold slider response | < 150ms | 128^3 MRC |
| Threshold slider response | < 500ms | 256^3 MRC |

Volumes with any dimension > 256 are downsampled server-side to 256^3 before serving (`MAX_SERVE_DIM` in `backend/config.py`; see `docs/API.md` for the canonical definition). The original is available via an explicit "Load full resolution" button.

### Latent Space Explorer

**Capabilities for Phase 1:**
- regl-scatterplot rendering of particle latent coordinates
- Axis selector (PC1 vs PC2, etc.)
- PCA and UMAP views, displayed simultaneously as two panels
- Color by: none, k-means cluster, density
- K-means centers overlaid as large labeled markers
- **Single click on k-means center or particle:** selects the point, places a colored marker; marker appears at corresponding position in all other plots; confirmation dialog to launch Compute State with that point's full z-vector
- **Second click on a different k-means center or particle:** places a second marker (different color); both markers cross-linked in all plots; "Compute Trajectory" button appears pre-filled with both points' full z-vectors
- **Clicking empty space:** visual selection only (no compute action). UMAP is not invertible and PCA projections with zdim > 2 do not specify the remaining dimensions, so arbitrary empty-space clicks cannot produce valid z-coordinates.
- **Trajectory type:** Phase 1 trajectories are straight-line interpolations in z-space. Density-guided paths require density estimation, which is not in Phase 1 scope.
- **Lasso selection:** freehand draw; selected particles highlighted in all linked plots
- **Actions on lasso:** "Export as subset" (saves .ind + records provenance in SQLite), "Export all except selection"

**Performance targets** (tested on Chrome 120+, 16 GB RAM laptop, integrated GPU):

| Metric | Target | Dataset |
|--------|--------|---------|
| Initial render | < 1s | 100K points |
| Initial render | < 3s | 1M points |
| Lasso completion → highlight | < 300ms | 100K points |
| Lasso completion → highlight | < 1s | 1M points |
| Cross-plot marker sync | < 100ms | Any size |

If particle count exceeds 2M, the scatter plot shows a random subsample of 1M with a banner: "Showing 1M of {N} particles. Full dataset used for selection export."

### Sidebar

- Project tree: Pipeline jobs, Analyze jobs, Subsets (expandable sections)
- Status icons per job: spinner (running), checkmark (completed), X (failed), clock (queued)
- Click job → main panel shows job detail
- "+ New Job" button at top
- Collapsible (hamburger icon)
- Disk usage at bottom: "{used} GB / {total} GB" with color (green < 80%, yellow 80-95%, red > 95%)

### Job Detail View

Tabs:
- **Overview:** Status badge, job type, creation time, duration, SLURM job ID (if applicable), error message (if failed). "Suggested next steps" buttons below.
- **Logs:** WebSocket-streamed log output. Auto-scroll to bottom. "Download full log" link.
- **Parameters:** Table of all parameters used. "Clone job" button. "Show CLI command" collapsible.
- **Volumes:** Grid of MRC files in the job output. Click to load in volume viewer.
- **Plots:** Grid of diagnostic PNG images from the job output directory. Click to enlarge.

### File Browser

- Navigates server filesystem via API
- Opens at project directory by default
- Shows: filename, size, modified date, type icon (.star, .mrc, .mrcs, .cs)
- Breadcrumb navigation
- "Go to parent" button
- Bookmarks: project dir (always), plus user-added bookmarks stored in settings
- Double-click directory to enter, single-click file to select
- .star files: on select, show inline preview (particle count, box size, columns)
- **Path allowlist:** Only serves directories under the project dir and paths listed in `settings.toml` `[file_browser.allowed_roots]`. Returns 403 for anything else. See `ADR-001-executor-security.md`.

### Real-time Updates

- **WebSocket connection** opened per active job
- Server pushes: new log lines, status transitions (queued→running→completed/failed), progress events
- On status transition: sidebar icon updates, tab badge updates
- On connection drop: automatic reconnect with exponential backoff (1s, 2s, 4s, max 30s). Banner: "Reconnecting..."
- On backend restart: see `FAILURE-STATES.md` for reconnect behavior

### Dark Theme

- Dark theme only for Phase 1 (light mode toggle is a Phase 2 feature)
- Shadcn/ui defaults with Tailwind dark color palette
- See `DESIGN-SYSTEM.md` for exact colors and patterns

---

## What Phase 1 Does NOT Need to Handle

These scenarios are explicitly out of scope. The behavior for each is "don't crash, show a sensible message":

- Concurrent GUI sessions on the same project → show warning "Another session may be active" (detect via SQLite advisory lock), but allow proceeding
- Volumes > 512^3 → downsample server-side, show banner
- Particle counts > 5M → subsample scatter plot, show banner
- SLURM not responding → show error after 30s timeout, suggest checking cluster status
- Pipeline output from old recovar version → best-effort import; if metadata.json is missing, mark as "imported (legacy)" and skip validation

---

## Technology Checklist

| Component | Version / Tool | Pinned? |
|-----------|---------------|---------|
| Node.js | 20 LTS | Yes, in `.nvmrc` |
| Package manager | npm with `package-lock.json` committed | `npm ci` in CI |
| React | 18.x | Yes |
| TypeScript | 5.x strict mode | Yes |
| Vite | 5.x | Yes |
| Tailwind CSS | 3.x | Yes |
| Shadcn/ui | Latest (copy-paste, not package) | Components committed |
| TanStack Router | 1.x | Yes |
| TanStack Query | 5.x | Yes |
| vtk.js | Latest | Yes |
| regl-scatterplot | Latest | Yes |
| Plotly.js | Latest | Yes (for diagnostic charts only) |
| Python | 3.11+ (via pixi, matches `requires-python = ">=3.11"`) | Yes |
| FastAPI | 0.110+ | Yes |
| SQLAlchemy | 2.x | Yes |
| Alembic | Latest | Yes |
| Pydantic | 2.x | Yes |
| uvicorn | Latest (with WebSocket support) | Yes |
