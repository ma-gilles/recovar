# RECOVAR GUI v2 — Development Instructions

## Document Precedence

1. **`docs/PHASE1.md`** is binding for implementation scope. If it says "non-goal," do not implement.
2. **Accepted ADRs** (`docs/ADR-*.md`) are binding for architecture and security.
3. **`docs/DESIGN-SYSTEM.md`** is binding for UI behavior and accessibility.
4. **`docs/API.md`** is binding for endpoint contracts and data flow.
5. **`VISION.md`** is roadmap only and must not justify Phase 1 code.

If documents conflict, follow the higher-precedence document and record the conflict in a comment rather than implementing both sides.

## Phase 1 Discipline

- Do not create routes, nav items, DB tables, services, or placeholder components for features listed as non-goals in `docs/PHASE1.md`.
- **No:** RELION, plugin infrastructure, auth, reverse proxy, mask editor, comparison view, notifications, light mode, figure export, resource monitoring, trajectory playback, parameter sweeps, .star export.
- Diagnostic PNGs from pipeline/analyze are displayed as `<img>` tags. Plotly.js is not needed in Phase 1 unless an interactive chart is required by an acceptance criterion.
- The Phase 1 directory tree is in `docs/PHASE1.md`. Do not scaffold directories from `VISION.md`.

## Documentation Map

| Document | Purpose |
|----------|---------|
| `VISION.md` | Long-term product vision (all phases). Roadmap only. |
| `docs/PHASE1.md` | **Current build target.** Strict scope, acceptance criteria, non-goals. |
| `docs/API.md` | REST endpoints, WebSocket messages, data flow from recovar outputs to browser. |
| `docs/ADR-001-executor-security.md` | Executor abstraction (SLURM/local), security model, job wrapper. |
| `docs/FAILURE-STATES.md` | Every failure mode and its recovery behavior. |
| `docs/DESIGN-SYSTEM.md` | Colors, typography, spacing, component patterns, accessibility. |

## Build Order

Implement in this sequence. Each step builds on the previous:

1. **DB schema + project API.** SQLAlchemy models, Alembic initial migration, `POST/GET /api/projects`, scan/import. Test: create project, scan existing pipeline output.
2. **Executor abstraction.** `Executor` ABC, `SlurmExecutor`, `LocalExecutor`, job wrapper script, reconnect-on-restart procedure. Test: mock submit/status/cancel/reconnect.
3. **Pipeline job form + submission.** Backend `POST /api/jobs`, command builder for pipeline, SLURM submission. Frontend: job form with validation, file browser. Test: submit pipeline job, watch it appear in job list.
4. **Log streaming.** WebSocket endpoint, log tailing, reconnect protocol. Frontend: log viewer component. Test: stream logs, reconnect after drop.
5. **Job detail + volume viewer.** `GET /api/jobs/:id/volumes`, volume raw endpoint, vtk.js isosurface renderer. Frontend: job detail tabs, volume grid, 3D viewer. Test: load mean.mrc, adjust threshold.
6. **Analyze job + latent explorer.** Analyze job form (suggested after pipeline), embedding endpoint (binary format), regl-scatterplot with PCA + UMAP panels, k-means overlay. Test: submit analyze, view scatter plot.
7. **Point click + compute + lasso export.** Click k-means center → compute state, two clicks → compute trajectory, lasso → export .ind subset. Test: full acceptance criteria 5 + 6.

## Architecture

```
backend/         FastAPI (Python 3.11+, async, uvicorn)
frontend/        React 18 / TypeScript 5 / Vite 5 (Node 20 LTS)
```

- **Database:** SQLite (WAL mode), SQLAlchemy 2.x ORM, Alembic migrations
- **3D Rendering:** vtk.js (isosurface via marching cubes, client-side)
- **Scatter Plots:** regl-scatterplot (WebGL, 1M+ points)
- **Real-time:** WebSocket (log streaming, status push)
- **UI Components:** Shadcn/ui (Radix + Tailwind CSS)
- **Routing:** TanStack Router (file-based, type-safe)
- **State:** TanStack Query (server state), React useState/useReducer (UI state)

### Why these choices (non-obvious)
- **vtk.js over Three.js:** Built-in marching cubes, isosurface widgets (sphere, box, plane), and slice views for scientific volume data. Three.js would require implementing all of these from scratch.
- **regl-scatterplot over Plotly/deck.gl:** Purpose-built for 1M+ point scatter plots with built-in lasso and linked views. Plotly chokes above ~500K; deck.gl requires custom lasso code.
- **SQLite over MongoDB/Postgres:** Zero-setup, single-file, HPC-friendly (no database server to run). WAL mode handles concurrent reads. Phase 1 is single-user on one host, which is SQLite's sweet spot. If multi-user or network-filesystem deployment is needed later, define a storage interface and swap. SQLite's own docs state WAL does not work over network filesystems.

## Compatibility Contract

- **CLI outputs are canonical.** The GUI does not replace `project.json`, `job.json`, `command.txt`, `run.log`, or `README.txt` files that the CLI creates. It reads them.
- **Existing directory names are preserved.** Jobs use `{JobType}/job_{NNNN}` naming (e.g., `Pipeline/job_0001`), matching the existing `recovar init_project` convention. Not `P001`.
- **SQLite is a rebuildable index.** If `recovar_project.db` is deleted, `POST /api/projects/:id/scan` reconstructs the job list from the filesystem. The DB adds provenance and subset tracking on top, but the pipeline outputs remain the source of truth.
- **`recovar gui` remains the user-facing entry point.** Node.js is a dev dependency only. The shipped package bundles prebuilt frontend assets into the Python wheel (or a committed `static/` directory). End users do `pip install recovar[gui]` or `pixi install` and run `recovar gui`. They never run `npm`.
- **Frontend must not depend on pickle filenames.** The backend resolves `latent_coords.npy`, `kmeans_result.pkl`, `umap_embedding.pkl`, etc. via `ResultPaths`/`AnalysisPaths`. If storage formats change (issue #34), only backend adapters change. See `docs/API.md` for the data source mapping.

## Latent Explorer Semantics

- **Compute actions (state, trajectory) require full z-space coordinates.** UMAP is not invertible; a 2D click in UMAP space has no unique z-space mapping. Clicking empty space in a PCA plot with zdim > 2 does not specify the remaining dimensions.
- **Therefore: compute actions are restricted to existing data points.** Clicks select the nearest particle or k-means center (which have known full z-vectors). The selected entity's full latent coordinate vector is used for compute_state / compute_trajectory.
- **Arbitrary empty-space clicks** are selection-only (for visual inspection), not compute triggers.
- **Trajectories are linear interpolation in z-space** for Phase 1. The two-click trajectory computes `recovar compute_trajectory --z_st <start.txt> --z_end <end.txt>`, which is a straight line in latent space. Density-guided paths require `--density <file>`, which is a Phase 2+ feature when density estimation is in the GUI.
- **When zdim > 2 and PCA axes show only 2 dims:** The scatter plot displays a 2D projection. A click selects the nearest particle in the displayed 2D space, but the full z-vector of that particle (all zdim dimensions) is used for compute actions. The UI shows the full z-vector in a tooltip on hover.

## Build & Run

### Toolchain (pinned)
- **Python 3.11+** (matches recovar's `requires-python = ">=3.11"`)
- Node.js 20 LTS (enforced via `.nvmrc`)
- `npm ci` (not `npm install`) — uses lockfile exactly. `package-lock.json` is committed.
- Python via pixi (same as the rest of recovar)

### Commands
```bash
# Backend
pixi install && pixi run install-recovar
pixi run pip install -e ".[gui]"
pixi run python -m recovar.gui_v2.backend.main --port 8080

# Frontend (dev)
cd recovar/gui_v2/frontend
npm ci
npm run dev             # Vite dev server, proxies /api to backend

# Frontend (prod build)
npm run build           # Output to backend/static/

# Regenerate API client after backend changes
npm run generate-api    # openapi-typescript-codegen from /openapi.json
```

## Code Rules

### TypeScript
- Strict mode (`"strict": true` in tsconfig). No `any` except in generated code.
- Explicit return types on exported functions.
- Functional React components only. No class components.
- One component per file. Filename = component name (PascalCase).

### Python
- Follow existing recovar style (see `recovar/CLAUDE.md`).
- Type hints on all public functions. Pydantic v2 models for all API schemas.
- Async endpoints where I/O is involved (file reads, SLURM calls, DB queries).

### CSS
- Tailwind utility classes only. No custom CSS files.
- Follow `docs/DESIGN-SYSTEM.md` for all colors, spacing, typography.

### API Contract
- FastAPI generates OpenAPI spec at `/openapi.json`.
- After any backend API change: run `npm run generate-api` to regenerate the TypeScript client.
- **Use the generated client for all REST calls.** Exceptions: WebSocket connections (`new WebSocket(url)`) and binary volume/embedding streaming (`fetch` with `arrayBuffer()` response). These are hand-written in `frontend/src/lib/api/` with explicit types.
- WebSocket message types are defined in `docs/API.md` and implemented in shared type files: `backend/ws_types.py` and `frontend/src/lib/ws_types.ts`. Keep them in sync manually (they are too small for codegen overhead).

### Database
- SQLite with WAL mode (set at DB creation).
- Alembic migration for every schema change. Test migrations against existing project databases.
- Write retries: 3 attempts with 100ms/500ms/2000ms delays on `OperationalError: database is locked`.
- `created_by` field on all entities (default value for now, multi-user later).
- Never hold transactions open during long operations.

## Testing

### Test Pyramid (risk-based)

**Tier 1 — Must have (blocks merge):**
- Backend: pytest tests for every API endpoint (success + at least one error case)
- Backend: executor tests with mock SLURM (submit, status, cancel, reconnect-after-restart)
- E2E: Playwright tests for the 7 acceptance criteria in `docs/PHASE1.md`
- E2E: Playwright tests for server restart reconnect and WebSocket reconnect (these are core reliability, not optional on HPC)

**E2E tests must be interaction tests, not existence tests.** Do not just check that a button/element exists on the page. Every E2E test must:
- **Click** the interactive element
- **Verify the result** of the click (page navigation, data loading, form pre-filling, API response)
- Example BAD test: `expect(page.locator('button:has-text("Explore")')).toBeVisible()`
- Example GOOD test: `await page.click('button:has-text("Explore")'); expect(page.url()).toContain('/explore/'); expect(page.locator('.scatter-plot canvas')).toBeVisible()`
- For volume clicks: verify a slice image actually loads after clicking
- For form submissions: verify the API returns 200, not just that the button exists
- For suggested next steps: verify the target form has pre-filled values, not just that the button exists

**Tier 2 — Should have (best effort):**
- Frontend: Vitest tests for components with non-trivial logic (job form validation, latent space selection state, volume viewer controls)
- Backend: SQLite migration tests (upgrade from previous schema)

**Tier 3 — Nice to have:**
- Frontend: Vitest tests for simple presentational components
- Visual regression tests (Playwright screenshots)

**Test fixtures** (`tests/fixtures/`):
- Generated by `make_test_dataset` (32^3 box, ~800 images for speed)
- Includes pre-computed pipeline output (so E2E tests for acceptance criteria 3-6 don't need to run a real pipeline)
- Includes pre-computed analyze output (UMAP, k-means, PCA coords)
- Fixture generation script committed; re-run to regenerate after output format changes

**All tests:**
- Runnable without SLURM (mock executor)
- Runnable without GPU (mock vtk.js rendering context via `createCanvas`)
- No network access required

```bash
pixi run pytest recovar/gui_v2/tests/backend/ -v   # Backend
cd recovar/gui_v2/frontend && npm test              # Frontend unit
cd recovar/gui_v2/frontend && npm run test:e2e      # E2E
pixi run test-gui                                   # All
```

### Visual Verification (mandatory after any frontend change)

After modifying any frontend code, you MUST verify the UI works in a real browser before pushing:

1. **Build the frontend:** `cd recovar/gui_v2/frontend && npm run build`
2. **Start the server:** `pixi run python -m recovar.gui_v2.backend.main --port 8090 &`
3. **Take screenshots of key pages:**
   ```bash
   npx playwright install firefox  # first time only
   npx playwright screenshot --browser=firefox http://localhost:8090 /tmp/gui_home.png
   # After creating a project and importing data:
   npx playwright screenshot --browser=firefox http://localhost:8090/jobs/<id> /tmp/gui_job.png
   ```
4. **Read the screenshots** to verify:
   - All expected buttons, forms, and navigation elements are visible
   - Layout is correct (sidebar, main panel, tabs)
   - Empty states show helpful messages with actionable buttons
   - Status indicators render correctly
5. **Run Playwright E2E tests** against the running server:
   ```bash
   cd recovar/gui_v2/frontend && npx playwright test --browser=firefox
   ```
6. **Kill the server** when done: `kill %1`

If screenshots show UI problems or E2E tests fail, fix the code before pushing. Do not push frontend changes that have not been visually verified.

**Environment note:** This cluster has Firefox and Xvfb available. Use `--browser=firefox` with Playwright. Chromium may not be installed.

## Key Patterns

### Job Execution
The full flow is documented in `docs/ADR-001-executor-security.md`. Summary:
1. Frontend `POST /api/jobs` with validated parameters
2. Backend creates job record (status: QUEUED), renders sbatch/command
3. Executor submits (sbatch or subprocess), stores handle
4. Background task polls status, tails logs, pushes to WebSocket
5. On terminal state: update SQLite, close WebSocket stream
6. On server restart: reconnect procedure syncs all in-flight jobs

### Volume Serving
See `docs/API.md` for endpoint details.
- `MAX_SERVE_DIM = 256` in `backend/config.py` — single source of truth for the downsampling threshold. All documents reference this constant.
- Max 4 pinned volumes in viewer simultaneously (hard limit, `MAX_PINNED_VOLUMES = 4` in `frontend/src/lib/constants.ts`).

### Filesystem Security
- All file-serving endpoints resolve paths via `Path.resolve()` and check against allowlist.
- Allowlist: project dir (always) + `settings.toml` `[file_browser.allowed_roots]`.
- Reject any path outside allowlist with HTTP 403.
- No `shell=True` in subprocess calls. No user strings interpolated into commands.
- See `docs/ADR-001-executor-security.md` for full security model.

## Dependency Boundaries

- **`import recovar` must work without GUI dependencies.** Never import GUI packages at the recovar top level. The GUI is an optional extra (`pip install recovar[gui]`).
- **GUI backend may import recovar library** (ResultPaths, PipelineOutput, MRC reading, metadata parsing).
- **recovar library must never import GUI code.**
- **Matplotlib PNGs** are generated by the pipeline, not the GUI. The GUI displays them as `<img>` tags.
- **`recovar/gui/`** (old Flask GUI) is untouched. Do not modify it.

## Common Tasks

### Add a new recovar job type
1. Pydantic model in `backend/models/job.py` (parameter schema)
2. Command builder in `backend/services/` (params → CLI args list)
3. React form component in `frontend/src/components/job-form/`
4. Tooltip entries in `tooltips.json`
5. "Suggested next step" linkage in job completion handler
6. Endpoint definition in `docs/API.md`
7. Tier 1 tests: API endpoint test + E2E test
8. `npm run generate-api`

### Modify the database schema
1. Update SQLAlchemy model
2. `alembic revision --autogenerate -m "description"`
3. Test migration on an existing project database
4. `npm run generate-api` if API types changed

## Phase 1 Directory Tree

Only these directories/files exist in Phase 1. Do not scaffold anything else.

```
recovar/gui_v2/
├── CLAUDE.md
├── VISION.md
├── docs/
│   ├── PHASE1.md
│   ├── API.md
│   ├── ADR-001-executor-security.md
│   ├── FAILURE-STATES.md
│   └── DESIGN-SYSTEM.md
├── backend/
│   ├── __init__.py
│   ├── main.py                # FastAPI app factory, static file serving
│   ├── config.py              # MAX_SERVE_DIM, settings loader
│   ├── db.py                  # SQLite connection, WAL setup
│   ├── api/
│   │   ├── jobs.py            # Job CRUD, submission, suggested-next
│   │   ├── volumes.py         # Raw MRC, slice, info
│   │   ├── embeddings.py      # Latent coords, UMAP, k-means (binary format)
│   │   ├── subsets.py         # Subset CRUD, .ind export
│   │   ├── project.py         # Project CRUD, scan/import
│   │   ├── files.py           # File browser, .star/.mrc validation
│   │   ├── system.py          # SLURM detection, disk usage
│   │   └── ws.py              # WebSocket log/status stream
│   ├── models/
│   │   ├── job.py
│   │   ├── project.py
│   │   └── subset.py
│   ├── services/
│   │   ├── executor.py        # ABC + SlurmExecutor + LocalExecutor
│   │   └── scanner.py         # Scan directories for existing outputs
│   ├── ws_types.py            # WebSocket message type definitions
│   └── static/                # Prebuilt frontend assets (generated by npm run build)
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── .nvmrc
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── src/
│   │   ├── main.tsx
│   │   ├── routes/
│   │   │   ├── __root.tsx     # Sidebar + main layout
│   │   │   ├── index.tsx      # Dashboard
│   │   │   ├── jobs/
│   │   │   │   ├── new.tsx    # New job form
│   │   │   │   └── $jobId.tsx # Job detail (tabs: overview, logs, params, volumes, plots)
│   │   │   └── explore/
│   │   │       └── $jobId.tsx # Latent space explorer for a specific analyze job
│   │   ├── components/
│   │   │   ├── ui/            # Shadcn/ui primitives
│   │   │   ├── volume-viewer/ # vtk.js 3D viewer + multi-volume controls
│   │   │   ├── latent-explorer/ # regl-scatterplot + linked views + click/lasso
│   │   │   ├── job-form/      # Pipeline, Analyze, ComputeState, ComputeTrajectory forms
│   │   │   ├── file-browser/  # Server filesystem browser
│   │   │   ├── log-viewer/    # WebSocket log streamer
│   │   │   └── sidebar/       # Project tree
│   │   ├── lib/
│   │   │   ├── api/           # Generated REST client + hand-written WS/binary helpers
│   │   │   ├── ws_types.ts    # WebSocket message types (matches backend/ws_types.py)
│   │   │   └── constants.ts   # MAX_PINNED_VOLUMES, SUBSAMPLE_THRESHOLD, etc.
│   │   └── styles/
│   └── tests/
├── tests/
│   ├── backend/               # pytest
│   ├── e2e/                   # Playwright
│   └── fixtures/              # Pre-computed pipeline + analyze outputs
└── tooltips.json              # Contextual help text for all parameters
```
