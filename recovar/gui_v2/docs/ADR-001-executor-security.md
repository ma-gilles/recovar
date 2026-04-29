# ADR-001: Executor Model and Security

## Status: Accepted (Phase 1)

## Context

The GUI runs on an HPC login node (or local machine) and submits compute jobs. We need an executor abstraction that handles SLURM and local subprocess, plus a security model that doesn't expose the cluster.

---

## Decision: Executor Abstraction

### Interface

```python
class Executor(ABC):
    """Abstraction over job execution backends."""

    @abstractmethod
    async def submit(self, job: Job, command: list[str], env: dict[str, str]) -> str:
        """Submit a job. Returns an executor-specific job handle (SLURM ID or PID)."""

    @abstractmethod
    async def status(self, handle: str) -> JobStatus:
        """Poll current status. Returns QUEUED, RUNNING, COMPLETED, FAILED, UNKNOWN."""

    @abstractmethod
    async def cancel(self, handle: str) -> None:
        """Cancel a running or queued job."""

    @abstractmethod
    async def log_path(self, handle: str) -> Path | None:
        """Return the path to the job's log file, or None if not yet available."""

    @abstractmethod
    async def cleanup(self, handle: str) -> None:
        """Clean up resources after job completion (staging files, temp dirs)."""
```

### SLURM Executor

```python
class SlurmExecutor(Executor):
    """Submits jobs via sbatch, polls via squeue/sacct."""
```

**Submit:**
1. Render sbatch script from Jinja2 template + job parameters + global settings
2. Write script to `{job_output_dir}/submit.sh`
3. Run `sbatch --parsable submit.sh`, capture SLURM job ID
4. Store `(job.id, slurm_id)` mapping in SQLite

**Status polling (squeue first, sacct fallback):**

`sacct` alone is not reliable for live status — it reflects accounting data, not the scheduling queue. Use `squeue` for live jobs, `sacct` for terminal-state reconciliation:

1. Try `squeue -j {slurm_id} --noheader --format=%T` first.
   - If it returns a state: use it (most timely for PENDING/RUNNING).
   - If the job is not in the queue (empty output): fall back to `sacct`.
2. Fallback: `sacct -j {slurm_id} --format=State --noheader --parsable2`
3. Map SLURM states to internal `JobStatus` enum:
   - `PENDING`, `CONFIGURING` → `QUEUED`
   - `RUNNING`, `COMPLETING` → `RUNNING`
   - `COMPLETED` → `COMPLETED`
   - `CANCELLED`, `CANCELLED+` → `CANCELLED` (distinct from FAILED — see note below)
   - `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, `PREEMPTED` → `FAILED`
   - Command fails or empty output from both squeue and sacct → `UNKNOWN`
4. Poll interval: 5 seconds while `RUNNING`, 15 seconds while `QUEUED`, stop on terminal state

**Note:** `CANCELLED` is a distinct terminal state, not a `FAILED` variant. The user intentionally cancelled the job (or an admin preempted it). The UI shows it differently: gray icon, no error banner, "Cancelled" status badge. The `JobStatus` enum is: `QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED, UNKNOWN`.

**Log tailing:**
- Primary: SLURM output file (`--output` path from sbatch template)
- Fallback: `{job_output_dir}/run.log` (if job writes its own log)
- Tail via `aiofiles` async read, push new lines to WebSocket

**Cancel:**
- `scancel {slurm_id}`

**Cleanup:**
- SLURM auto-cleans `$TMPDIR` on job exit
- Staging uses **per-job directories** keyed by SLURM job ID: `$RECOVAR_CACHE_DIR/recovar_cache_${SLURM_JOB_ID}/`. This prevents concurrent jobs from interfering with each other's caches. The cleanup trap removes only the job's own staging directory.

### Job Wrapper Script

Every SLURM job runs through a wrapper that handles environment setup, staging cleanup, and metrics collection:

```bash
#!/bin/bash
#SBATCH --job-name={{ job_name }}
#SBATCH --partition={{ partition }}
#SBATCH --account={{ account }}
#SBATCH --gres=gpu:{{ gpus }}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{ cpus }}
#SBATCH --mem={{ memory }}
#SBATCH --time={{ time }}
#SBATCH --output={{ slurm_output_path }}

# ── Environment ──
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
{% if recovar_cache_dir %}
export RECOVAR_CACHE_DIR={{ recovar_cache_dir }}/recovar_cache_${SLURM_JOB_ID}
mkdir -p "$RECOVAR_CACHE_DIR"
{% endif %}

# ── Staging cleanup trap (per-job directory, safe for concurrent jobs) ──
cleanup() {
    if [ -n "$RECOVAR_CACHE_DIR" ] && [ -d "$RECOVAR_CACHE_DIR" ]; then
        rm -rf "$RECOVAR_CACHE_DIR"
    fi
}
trap cleanup EXIT TERM INT

# ── Run the actual command ──
{{ command }}
EXIT_CODE=$?

exit $EXIT_CODE
```

**Note:** GPU metrics monitoring is a Phase 4 non-goal. When added later, the wrapper can be extended with an nvidia-smi sidecar that writes metrics to a JSONL file in the job output directory. The GUI server reads the file via filesystem access — no SSH to compute nodes needed.

### Local Executor

```python
class LocalExecutor(Executor):
    """Runs jobs as local subprocesses."""
```

**Submit:**
1. Build command list: `[sys.executable, "-m", "recovar.commands.pipeline", ...args]`
2. Set environment: inherit current env + `PYTHONNOUSERSITE=1` + `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `RECOVAR_CACHE_DIR` if configured (per-job: `{base}/recovar_cache_{job_id}/`)
3. Apply `local_opts` if provided:
   - `gpus`: set `CUDA_VISIBLE_DEVICES` (unless `"all"`)
   - `env_vars`: merge extra env vars into the process environment
   - `setup_command`: wrap command in `bash -c "setup_command && pipeline_command"`
4. Start via `asyncio.create_subprocess_exec` with `start_new_session=True` (creates a new process group so cancel kills child processes too — CUDA workers, data loaders, etc.)
5. Store `(job.id, pid, pgid)` in SQLite
6. Redirect stdout/stderr to `{job_output_dir}/run.log`

**Status:**
- Check if PID is alive via `os.kill(pid, 0)`
- On process exit: check return code (0 = COMPLETED, else FAILED)

**Log tailing:**
- Tail `{job_output_dir}/run.log` via async file read

**Cancel:**
- `os.killpg(pgid, signal.SIGTERM)` to kill the entire process group, then `os.killpg(pgid, signal.SIGKILL)` after 10s grace period

**Cleanup:**
- Remove per-job staging directory `RECOVAR_CACHE_DIR/recovar_cache_{job_id}/` if configured

### Reconnect After GUI Restart

When the GUI server starts:
1. Query SQLite for all jobs with status `RUNNING` or `QUEUED`
2. For each, call `executor.status(handle)` to get current state
3. If SLURM says `COMPLETED` or `FAILED` but SQLite says `RUNNING` → update SQLite, check for output files
4. If SLURM says `RUNNING` → resume log tailing and WebSocket streaming
5. If executor returns `UNKNOWN` (SLURM job ID not found, PID dead with no exit code) → mark as `FAILED` with message "Job status unknown after server restart. Check SLURM output: {log_path}"
6. If local executor and PID is dead → check exit code from process table or mark as `FAILED`

This ensures the GUI never shows stale "Running" status after a restart.

---

## Decision: Security Model

### Phase 1: SSH Tunnel Only

- The GUI server binds to `127.0.0.1:8080` (localhost only). It is **not** accessible from the network.
- Users access it via `ssh -L 8080:localhost:8080 user@cluster`
- No authentication layer. If you can SSH to the machine, you can use the GUI.
- This matches the security model of Jupyter on HPC clusters.

### Filesystem Sandboxing

The file browser API and all file-serving endpoints enforce an allowlist:

```toml
# project/.recovar/settings.toml
[file_browser]
allowed_roots = [
    "/scratch/gpfs/GILLES/mg6942",    # user's scratch
    "/projects/cryoem/shared_data",    # shared datasets
]
```

Rules:
1. The project directory is always allowed (implicitly).
2. `allowed_roots` lists additional directories the file browser can navigate.
3. All paths are resolved via `Path.resolve()` to prevent symlink escapes.
4. Paths outside allowed roots return HTTP 403 with message "Access denied: path is outside allowed directories."
5. No path traversal: reject any path containing `..` after resolution.
6. The settings file itself is not modifiable from the GUI API — it's edited on disk by the user.

### Job Command Safety

- Job commands are built by the backend from validated parameters, not from user-supplied shell strings.
- Parameters are passed to recovar CLI via `subprocess.run(command_list, ...)` (not `shell=True`).
- The sbatch template uses Jinja2 with `autoescape=True` and a restricted sandbox (no access to `os`, `subprocess`, etc.).
- Users can edit the sbatch template on disk (`project/.recovar/slurm_template.sh`), but the GUI only reads it — no template editing API.

### Future: Direct Network Access (Phase 4+)

When/if we add direct network access, the following are required:
- Token-based auth: random 256-bit token generated at startup, appended to URL
- Session cookies with `SameSite=Strict`, `HttpOnly`, `Secure` (if HTTPS)
- CORS: allow only the server's own origin
- CSRF: double-submit cookie pattern on all state-changing requests
- Rate limiting on auth endpoints
- Audit logging: all job submissions and file accesses logged with timestamp and source IP

This is explicitly deferred — do not implement any of it in Phase 1.

---

## Update: Per-Job Executor Selection

The original design assumed a server-wide executor mode. The implementation
now supports **per-job executor selection**:

- When `sbatch` is on PATH, `/api/system/info` returns `executor_mode: "both"`.
- Each job form shows an `ExecutorSelector` toggle (SLURM Cluster / Local GPU).
- The `POST /api/jobs` body accepts an `executor` field (`"slurm"` or `"local"`).
- Both `SlurmExecutor` and `LocalExecutor` instances are kept in a pool
  (`backend/api/jobs.py`), created lazily on first use.
- For existing jobs, `get_executor_for_job()` picks the right executor
  based on whether the job has a `slurm_id`.

`LocalExecutor.submit()` accepts `local_opts` with:
- `gpus` — `"all"` or comma-separated indices. Sets `CUDA_VISIBLE_DEVICES`.
- `setup_command` — Shell command run before the pipeline.
- `env_vars` — Extra environment variables as `{key: value}`.

Local-execution defaults follow the same layering as SLURM defaults
(built-in, user-global `[local]`, project `[local]`, per-job override).
See `backend/services/project_config.py` for the merge logic.

The `--executor` CLI flag still works (sets `RECOVAR_EXECUTOR` env var)
but is no longer the primary way to select an executor. In the common case
(`auto`), both executors are available and the user picks per job.

---

## Consequences

- Two executor implementations to maintain (SLURM + Local), but they share the same interface and the same test harness (mock executor).
- Both executors are instantiated simultaneously when `sbatch` is on PATH.
  Each job picks its executor at submit time.
- No GPU metrics via SSH — the wrapper script approach is simpler and doesn't require network access to compute nodes.
- Filesystem sandboxing adds a validation step to every file API call. This is cheap (one `Path.resolve()` + set membership check) but must never be bypassed.
- SSH-tunnel-only means no "share a link with your PI" in Phase 1. This is an acceptable trade-off for zero security complexity.
