# API Contract & Data Flow

The bridge between frontend and backend. Endpoints, request/response shapes, WebSocket messages, and how recovar outputs get served to the browser.

---

## REST Endpoints

### Projects

| Method | Path | Request | Response | Notes |
|--------|------|---------|----------|-------|
| `POST` | `/api/projects` | `{ path: string, name: string }` | `{ id: string, path: string, name: string, created: string }` | Creates project dir + SQLite DB |
| `GET` | `/api/projects/:id` | — | `{ id, path, name, created, jobs: Job[], disk_usage_bytes: int }` | |
| `POST` | `/api/projects/:id/scan` | `{ scan_path: string }` | `{ imported: ImportedJob[] }` | Scans directory for existing pipeline outputs |

### Jobs

| Method | Path | Request | Response | Notes |
|--------|------|---------|----------|-------|
| `POST` | `/api/jobs` | `{ project_id, type, params: {...} }` | `{ id, type, status, created, handle }` | Validates params, submits to executor |
| `GET` | `/api/jobs/:id` | — | `{ id, type, status, params, created, completed, handle, slurm_id, error, parent_jobs, output_dir }` | |
| `POST` | `/api/jobs/:id/cancel` | — | `{ status: "cancelled" }` | Calls executor.cancel |
| `DELETE` | `/api/jobs/:id` | — | `204` | Removes DB record + optionally output files |
| `GET` | `/api/jobs/:id/volumes` | — | `[{ name, path, category, size_bytes }]` | Lists MRC files in job output, categorized |
| `GET` | `/api/jobs/:id/plots` | — | `[{ name, path }]` | Lists diagnostic PNGs |
| `GET` | `/api/jobs/:id/suggested-next` | — | `[{ type, label, prefilled_params }]` | Context-dependent suggestions |

### Volumes

| Method | Path | Request | Response | Notes |
|--------|------|---------|----------|-------|
| `GET` | `/api/volumes/raw` | `?path=<abs_path>` | Binary MRC (application/octet-stream) | Path must pass allowlist. Downsampled if > 256^3. Header `X-Original-Shape` if downsampled. |
| `GET` | `/api/volumes/raw` | `?path=<abs_path>&full=true` | Binary MRC | Full resolution, no downsampling. Client must request explicitly. |
| `GET` | `/api/volumes/slice` | `?path=<abs_path>&axis=<0|1|2>&idx=<int>` | PNG image | Server-rendered orthogonal slice |
| `GET` | `/api/volumes/info` | `?path=<abs_path>` | `{ shape: [D,D,D], voxel_size: float, min: float, max: float, mean: float }` | Metadata only, no heavy I/O |

### Embeddings (latent coordinates)

| Method | Path | Request | Response | Notes |
|--------|------|---------|----------|-------|
| `GET` | `/api/jobs/:id/embeddings` | `?zdim=<int>` | See below | Main data endpoint for latent explorer |
| `GET` | `/api/jobs/:id/embeddings/available` | — | `{ zdims: [1,2,4,10,20], has_umap: {4: true, 10: true} }` | Which zdims exist, which have UMAP |

**Embeddings response format:**

The embedding data is potentially large (1M particles x 20 floats = 80 MB as JSON). Served as a binary ArrayBuffer with a JSON header:

```
Response headers:
  Content-Type: application/octet-stream
  X-Embedding-Meta: {"n_particles": 300000, "zdim": 4, "has_umap": true, "has_kmeans": true, "n_clusters": 20}

Response body (binary, float32, row-major):
  [pca_coords: n x zdim] [umap_coords: n x 2 | empty] [kmeans_labels: n x 1 int32 | empty] [kmeans_centers: k x zdim | empty]
```

Frontend deserializes with `Float32Array` / `Int32Array` views on the ArrayBuffer. The JSON header tells the frontend the layout. This keeps transfer size ~5x smaller than JSON and avoids parsing overhead.

Fallback: if `?format=json` is specified, return JSON (for debugging only, not for production use with large datasets).

**Data source mapping:**

| GUI concept | recovar file | Backend access |
|-------------|-------------|----------------|
| PCA coordinates | `model/zdim_{N}/latent_coords.npy` (or legacy `model/embeddings.pkl`) | `PipelineOutput.get_embedding_component('latent_coords', zdim)` |
| UMAP coordinates | `analysis_{N}/plots/umap/umap_embedding.pkl` | Load pickle, extract 2D array |
| K-means labels | `analysis_{N}/data/kmeans_result.pkl` → `['labels']` | Load pickle, extract int array |
| K-means centers | `analysis_{N}/data/kmeans_result.pkl` → `['centers']` | Load pickle, extract float array |
| Density values | Not in Phase 1 | — |

**Important:** The frontend never sees pickle filenames or internal paths. It requests "embeddings for job X at zdim Y" and the backend resolves the file locations via `ResultPaths` and `AnalysisPaths`. If the storage format changes (e.g., issue #34: pkl → npz), only the backend adapter changes.

### Subsets

| Method | Path | Request | Response | Notes |
|--------|------|---------|----------|-------|
| `POST` | `/api/subsets` | `{ project_id, name, source_job_id, zdim, method, indices: int[] }` | `{ id, name, path, n_particles }` | Creates .ind file + DB record |
| `GET` | `/api/subsets` | `?project_id=<id>` | `[{ id, name, n_particles, source_job_id, method, created }]` | List subsets |
| `GET` | `/api/subsets/:id` | — | Full subset detail including provenance chain | |
| `DELETE` | `/api/subsets/:id` | — | `204` | |

The `method` field records how the subset was created:
```typescript
type SubsetMethod =
  | { type: "lasso", plot: "umap" | "pca", axes: [number, number], vertices: [number, number][] }
  | { type: "rectangle", plot: "umap" | "pca", axes: [number, number], bounds: { x: [number, number], y: [number, number] } }
  | { type: "kmeans_cluster", cluster_ids: number[] }
  | { type: "manual_indices" }
```

### Files

| Method | Path | Request | Response | Notes |
|--------|------|---------|----------|-------|
| `GET` | `/api/files/browse` | `?path=<dir>` | `[{ name, path, is_dir, size, modified, type }]` | Allowlist enforced |
| `POST` | `/api/files/validate-star` | `{ path: string }` | `{ valid, n_particles, box_size, columns, error }` | Timeout: 10s. Loading indicator on frontend. |
| `POST` | `/api/files/validate-mrc` | `{ path: string }` | `{ valid, shape, voxel_size, error }` | Timeout: 10s. |

**Validation timeout:** All file validation endpoints have a 10-second server-side timeout. If the filesystem is slow (loaded GPFS), the endpoint returns `{ valid: null, error: "Validation timed out. The filesystem may be under heavy load. Try again." }` The frontend shows a spinner during validation and the timeout message if it fires.

### System

| Method | Path | Request | Response | Notes |
|--------|------|---------|----------|-------|
| `GET` | `/api/system/info` | — | `{ slurm_available, executor_mode, recovar_version, gpu_count, hostname, disk: { path, total, used, free } }` | |

---

## WebSocket Messages

### Connection

```
WS /api/jobs/:id/stream
```

One WebSocket per active job view. The server maintains a byte offset per connection so reconnects resume from the correct log position.

### Message Vocabulary

All messages are JSON: `{ "type": "<type>", "data": <payload>, "ts": <unix_ms> }`

| Type | Direction | Payload | When |
|------|-----------|---------|------|
| `log_line` | server→client | `{ line: string, offset: int }` | New line appended to job log |
| `status_change` | server→client | `{ old: string, new: string, error?: string }` | Job transitions state |
| `progress` | server→client | `{ step: int, total: int, label: string }` | Parsed from log (e.g., "Pass 3/5") |
| `reconnect_sync` | server→client | `{ status: string, log_offset: int, log_tail: string[] }` | Sent immediately on (re)connect; last 50 log lines + current status |
| `ping` | server→client | `{}` | Every 30s keepalive |
| `pong` | client→server | `{}` | Response to ping |

### Reconnect Protocol

1. Client opens WebSocket with query param `?last_offset=<N>` (0 on first connect).
2. Server sends `reconnect_sync` with current status and log lines since offset N.
3. Server then streams new `log_line` / `status_change` / `progress` as they occur.
4. If the job is in a terminal state, server sends `reconnect_sync` with final status and closes.

---

## Data Flow: recovar Outputs → Browser

### Pipeline Job → Volume Viewer

```
Filesystem                          Backend API                     Browser
─────────────────────────────────────────────────────────────────────────────
output/volumes/mean.mrc      →  GET /api/jobs/:id/volumes    →  Volume list
                             →  GET /api/volumes/raw?path=   →  ArrayBuffer
                                                              →  vtk.js MRC parser
                                                              →  Marching cubes
                                                              →  WebGL isosurface
```

### Pipeline Job → Diagnostic Plots

```
output/plots/*.png           →  GET /api/jobs/:id/plots      →  Plot list
                             →  GET /api/files/browse         →  <img src="...">
                                (served as static images)
```

### Pipeline + Analyze → Latent Explorer

```
model/zdim_4/latent_coords.npy  ─┐
analysis_4/plots/umap/           │  GET /api/jobs/:id/embeddings?zdim=4
  umap_embedding.pkl             ├→ Backend loads, concatenates into binary buffer
analysis_4/data/                 │  with JSON header describing layout
  kmeans_result.pkl              ─┘
                                    → ArrayBuffer response
                                    → Frontend: Float32Array views
                                    → regl-scatterplot: PCA panel + UMAP panel
                                    → K-means centers as overlay markers
```

### Latent Explorer → Compute State

```
User clicks k-means center #3       → Frontend reads center coords from
                                       kmeans_centers array (already in memory)
                                     → Frontend writes coords to temp .txt
                                       via POST /api/jobs (type=ComputeState,
                                       params.latent_points=[x1,x2,x3,x4])
                                     → Backend writes coords to .txt file
                                     → Backend builds command:
                                       recovar compute_state <result_dir>
                                         --latent-points <coords.txt>
                                     → Executor submits
                                     → On completion: volume appears in job output
                                     → Frontend loads volume in viewer
```

### Latent Explorer → Compute Trajectory

```
User clicks two k-means centers     → Frontend has both coord vectors
                                     → POST /api/jobs (type=ComputeTrajectory,
                                       params.z_start=[...], params.z_end=[...])
                                     → Backend writes start/end to .txt files
                                     → Backend builds command:
                                       recovar compute_trajectory <result_dir>
                                         --z_st <start.txt> --z_end <end.txt>
                                         --zdim <N> --n-vols-along-path 6
                                     → Executor submits
```

### Lasso Selection → Subset Export

```
User draws lasso on UMAP plot       → Frontend: regl-scatterplot lasso
                                       returns indices of selected points
                                     → POST /api/subsets { indices: [...],
                                       method: { type: "lasso", ... } }
                                     → Backend: writes .ind file (pickle of
                                       numpy int array, matching recovar format)
                                     → Backend: records provenance in SQLite
                                     → Subset appears in sidebar
```

### Project Scan → Import

```
User points at /scratch/.../old_run  → POST /api/projects/:id/scan
                                      → Backend walks directory, checks for
                                        metadata.json or model/params.pkl
                                      → For each match: creates job record
                                        in SQLite with status="completed",
                                        type inferred from directory structure
                                      → Returns list of imported jobs
                                      → Sidebar updates with new jobs
```

---

## Volume Downsampling Threshold

**Single source of truth:** Volumes with any dimension > 256 are downsampled server-side to 256^3 before serving via `/api/volumes/raw`. The response includes `X-Original-Shape: 384,384,384` header so the frontend can show the banner and offer "Load full resolution." This threshold is defined as `MAX_SERVE_DIM = 256` in `backend/config.py` and referenced by all other documents.
