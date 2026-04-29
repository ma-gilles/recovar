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

### Mandatory QA (after any frontend or backend change)

After modifying any code, you MUST run the acceptance criteria test suite before pushing:

```bash
./scripts/gui_qa.sh
```

This script is **self-contained** — it handles its own server lifecycle, builds the frontend, scans real data (both pipeline-only AND pipeline-with-analyze-results), runs API tests, runs Playwright interaction tests for all 7 acceptance criteria, takes screenshots, and reports per-AC pass/fail. It exits with code 1 if any AC fails.

**What it tests:**
- AC-1: Dashboard, job forms
- AC-3: Volume click → slice viewer loads, plots render
- AC-4: Suggested next → pre-fills Analyze form with result_dir
- AC-5: Scatter plot renders with REAL particle data (50K+ points)
- AC-6: Lasso selection → export button
- AC-7: System info, sidebar categories

**After `gui_qa.sh` runs:**
1. If exit code is non-zero, read the FAILURES and fix before pushing.
2. Read screenshots in `/tmp/gui_qa/screenshots/` for visual issues the automated tests can't catch.
3. Machine-readable results: `/tmp/gui_qa/results.json`

**Do not push code that fails `gui_qa.sh`.** The git pre-push hook runs TypeScript + pytest; `gui_qa.sh` covers the browser interaction layer on top.

**After pushing, spawn the QA agent for independent review:**
```
Spawn the gui-qa agent to review my changes. It will run functional tests AND a UX review with fresh eyes.
```
The QA agent (`.claude/agents/gui-qa.md`) runs `gui_qa.sh`, reads all screenshots, walks through the full user journey with Playwright, and reports both bugs and UX issues. It does NOT fix code — it only reports. You fix issues based on its report. This is mandatory, not optional. The QA agent catches problems you can't see in your own work.

**Environment:** Firefox + Xvfb on Della. Playwright uses Firefox headless.

## Key Patterns

### Job Execution
The full flow is documented in `docs/ADR-001-executor-security.md`. Summary:
1. Frontend `POST /api/jobs` with validated parameters and `executor` field (`"slurm"` or `"local"`)
2. Backend creates job record (status: QUEUED), picks executor from pool based on `executor` field
3. For SLURM: renders sbatch script, submits via `sbatch`. For local: applies `local_opts` (GPU selection, setup command, env vars), starts subprocess
4. Background task polls status, tails logs, pushes to WebSocket
5. On terminal state: update SQLite, close WebSocket stream
6. On server restart: reconnect procedure syncs all in-flight jobs (uses `slurm_id` presence to pick the right executor)

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
- **`recovar/gui/`** was the old Flask GUI; it has been removed. `gui_v2` is now the only GUI.

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
│   │   ├── system.py          # SLURM detection, GPU enumeration, disk usage
│   │   ├── settings.py        # SLURM + local defaults CRUD (layered)
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
│   │   │   ├── settings.tsx   # Settings page (SLURM + local defaults)
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
│   │   │   │   ├── ExecutorSelector.tsx  # Per-job SLURM/Local toggle
│   │   │   │   ├── LocalSettings.tsx     # GPU picker, setup cmd, env vars
│   │   │   │   └── SlurmSettings.tsx     # SLURM resource settings
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

## Site-portability rules (sbatch rendering)

The GUI is shipped to users on arbitrary HPC clusters. The sbatch renderer
(`backend/services/executor.py::_render_sbatch_script`) must produce a
script that works on any reasonable SLURM site, with no implicit Princeton
assumptions. Treat the rules below as load-bearing — they exist because
v1.0.0 was almost shipped with `partition=cryoem`, `account=amits`, and
`TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp` baked into the default render.

**Hardcoding bans.**
- No site-specific identifier (partition name, account, group, host, path)
  may appear as a literal string anywhere in the renderer source. This
  includes function-default arguments and the `_SBATCH_TEMPLATE` body.
  Defaults belong in `backend/config.py::DEFAULT_SLURM` and must be empty
  strings, not site values.
- The frontend (`frontend/src/components/job-form/SlurmSettings.tsx`) must
  not have its own fallbacks. All defaults flow from the server's
  `/api/system/slurm-defaults`.
- The regression test
  `tests/backend/test_executor.py::TestSbatchTemplate::test_no_site_specific_strings_in_default_render`
  enforces this — keep it passing.

**Empty values mean "omit the directive."**
- `partition=""` / `account=""` → the renderer must NOT emit
  `#SBATCH --partition=` (a parse error on some SLURM versions). It must
  omit the directive entirely so the cluster's default applies.
- Same rule for any future optional directive.

**Shell-bearing fragments stay out of the format template.**
- `_SBATCH_TEMPLATE` is processed by `str.format`. Anything containing
  shell `${VAR}` or `${VAR:-default}` will be misparsed by `.format()` as
  a format spec and crash.
- Build such fragments as plain Python strings (see `_TMPDIR_BLOCK`,
  `_build_cleanup_block`) and substitute them in via a single named slot.
- When a future Phase 2 introduces user-editable templates, switch the
  engine away from `str.format` (Jinja2 with `StrictUndefined` is the
  intended target — surfaces typos as `UndefinedError` instead of
  silently submitting a broken script).

**Quote everything user-controlled.**
- The submitted command must be `shlex.join(argv)` at the call site —
  the renderer does not re-quote `command` (paths with spaces would
  otherwise break).
- Env-var values are `shlex.quote`d inside the renderer.
- The interpreter path (`sys.executable`) is `shlex.quote`d before going
  into `PATH=...`.

**Scheduler-provided scratch wins over our own.**
- The TMPDIR block prefers `$SLURM_TMPDIR` first, then an inherited
  `$TMPDIR`, and only falls back to `mktemp` if neither exists. The
  cleanup trap removes the tmpdir only if the renderer created it
  (`RECOVAR_CREATED_TMPDIR=1`); never delete `$SLURM_TMPDIR` or an
  inherited `$TMPDIR` — many clusters reap those automatically.

**One trap, not two.**
- Cache-staging cleanup and tmpdir cleanup share a single
  `_recovar_cleanup` function and one `trap _recovar_cleanup EXIT TERM
  INT`. Stacking traps clobbers the previous one.

## Executor selection

Executor selection is **per-job**, not server-wide.  When `sbatch` is on
PATH, `/api/system/info` returns `executor_mode: "both"` and every job
form shows an `ExecutorSelector` toggle (SLURM Cluster / Local GPU).
The user picks at submit time; each job can go to a different executor.

`backend/services/executor.py::slurm_available()` reads the
`RECOVAR_EXECUTOR` env var (power-user override, not exposed as a CLI
flag). Values:

- `auto` (default) — probe `sbatch` on PATH; when found, mode is
  `"both"` (both executors available, user picks per job). When absent,
  only the local executor is available.
- `local` — force local-subprocess mode even on a SLURM login node.
- `slurm` — force SLURM only.

In the common case, both executors are available simultaneously and the
user picks per job via the toggle in the job form.

**Frontend components for executor selection:**
- `ExecutorSelector.tsx` — per-job toggle shown when `executor_mode === "both"`.
- `LocalSettings.tsx` — collapsible panel for GPU picker, setup command,
  and env-var editor (shown when local executor is selected).
- `SlurmSettings.tsx` — SLURM resource settings panel (shown when SLURM
  executor is selected).

## Defaults are layered (not flat)

Both SLURM and local-execution defaults use the same layering system.

### SLURM defaults

`backend/services/project_config.py::resolve_slurm_defaults` merges, in
order of increasing precedence:

1. Built-in `DEFAULT_SLURM` in `backend/config.py` (intentionally
   minimal: empty partition/account, sane CPU/mem/time).
2. User-global `~/.config/recovar/config.toml` (or
   `$XDG_CONFIG_HOME/recovar/config.toml`) `[slurm]` section.
3. Project-local `<project_dir>/recovar.toml` `[slurm]` section.
4. Per-job form override sent in the submit body (the frontend's SLURM
   Settings panel — handled at the API layer, not in the loader).

### Local-execution defaults

`backend/services/project_config.py::resolve_local_defaults` merges:

1. Built-in `DEFAULT_LOCAL` in `backend/config.py` (`gpus: "all"`,
   empty `setup_command`, empty `env_vars`).
2. User-global `~/.config/recovar/config.toml` `[local]` section.
3. Project-local `<project_dir>/recovar.toml` `[local]` section.
4. Per-job form override via `local_opts` in the submit body.

`LocalExecutor.submit()` accepts `local_opts` with fields:
- `gpus` — `"all"`, or comma-separated GPU indices (`"0"`, `"0,1"`).
  Sets `CUDA_VISIBLE_DEVICES`.
- `setup_command` — Shell command run before the pipeline (e.g.
  `module load cudatoolkit/12.8`). When set, the command is wrapped in
  `bash -c "setup_command && pipeline_command"`.
- `env_vars` — Extra environment variables as `{key: value}`.

### Settings page

The Settings page at `/settings` (gear icon in sidebar) lets users view
and edit both SLURM and local defaults at the user-global and per-project
levels.  It shows an "effective defaults" summary with provenance badges
(built-in / user / project) and provides save buttons per layer.

API endpoints for settings:
- `GET /api/settings/slurm-defaults` — layered view (built-in + user + project + effective)
- `PUT /api/settings/slurm-defaults/user` — update user-global SLURM defaults
- `PUT /api/settings/slurm-defaults/project` — update per-project SLURM defaults
- `GET /api/settings/local-defaults` — layered view
- `PUT /api/settings/local-defaults/user` — update user-global local defaults
- `PUT /api/settings/local-defaults/project` — update per-project local defaults

The legacy `/api/system/slurm-defaults` endpoint still exists for
backward compatibility but the Settings page uses the layered endpoints.

When adding a new field to either executor, plumb it through all four
layers, not just one.

### TOML writing

Settings are persisted to TOML files using `tomli_w` (added as a
dependency in `pyproject.toml`). The `project_config.py` module handles
reading with `tomllib` (stdlib) and writing with `tomli_w`.

## User-supplied templates (Jinja2)

`backend/sbatch_templates/` ships preset `.sh` templates
(`generic_slurm.sh`, `generic_pbs.sh`, `local.sh`,
`princeton_della.sh`). Users opt in via `template_path = "..."` in their
`recovar.toml` `[slurm]` section.

Variables available to a template (passed positionally by the renderer):

- Reserved: `job_name`, `command`, `output_path`, `partition`, `account`,
  `gpus`, `cpus`, `memory`, `time`, `raw_directives`, `gpu_resource_spec`,
  `cache_dir`, `slurm_directives` (pre-built `#SBATCH` block),
  `tmpdir_block`, `cleanup_block`, `extra_exports`, `pixi_bin_dir`,
  `env_vars`. These names cannot be shadowed by `template_vars`.
- Custom: anything in `[slurm.template_vars]` in the TOML.

Templates use Jinja2 with `StrictUndefined`. A typo (`{{ commnad }}`)
raises `ValueError` at render time, surfaced to the GUI as a 400 from
`POST /api/jobs/preview-sbatch` — never silently submits a broken script.

When extending the renderer with new reserved names, update
`_RESERVED_TEMPLATE_VARS` in `executor.py` AND add an entry to the list
above.
