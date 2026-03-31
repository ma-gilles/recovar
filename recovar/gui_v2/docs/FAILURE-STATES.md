# Failure-State Matrix

Every failure mode the GUI must handle on HPC. For each: what triggers it, how we detect it, what the user sees, and what recovery looks like.

---

## Job Execution Failures

### SLURM job killed by walltime
- **Trigger:** Job exceeds `--time` limit.
- **Detection:** `sacct` returns state `TIMEOUT`.
- **User sees:** Job status → "Failed (walltime)". Log tab shows last lines before kill. Banner: "Job exceeded the {time} time limit. Increase the time limit in job settings and resubmit."
- **Recovery:** Clone job with longer `--time`, resubmit.

### SLURM job killed by OOM
- **Trigger:** Job exceeds `--mem` limit.
- **Detection:** `sacct` returns state `OUT_OF_MEMORY`.
- **User sees:** Job status → "Failed (out of memory)". Banner: "Job exceeded {mem} memory limit. Try increasing memory or enabling --lazy mode."
- **Recovery:** Clone job with more memory or `--lazy` flag.

### Compute node crash during job
- **Trigger:** Hardware failure, kernel panic, network partition.
- **Detection:** `sacct` returns state `NODE_FAIL`. Or: status was `RUNNING`, now `sacct` returns empty/error.
- **User sees:** Job status → "Failed (node failure)". Banner: "The compute node failed during execution. This is not a bug in your job. Resubmit to try again."
- **Recovery:** Resubmit (clone + submit). No parameter changes needed.

### SLURM job cancelled by user or admin
- **Trigger:** `scancel` from terminal, admin preemption, maintenance.
- **Detection:** `sacct` returns state `CANCELLED`.
- **User sees:** Job status → "Cancelled". No error banner — this was intentional.
- **Recovery:** Resubmit if desired.

### SLURM queue full / job stuck in PENDING
- **Trigger:** Cluster is busy, partition limits reached.
- **Detection:** `sacct` returns `PENDING` for extended time.
- **User sees:** Job status → "Queued" with elapsed wait time. After 1 hour: yellow warning "Job has been queued for {N} hours. The cluster may be busy." No automatic action.
- **Recovery:** Wait, or cancel and resubmit to a different partition.

### Local subprocess crashes
- **Trigger:** Segfault, CUDA error, Python exception.
- **Detection:** Process exit code != 0.
- **User sees:** Job status → "Failed". Last 50 lines of stderr shown in the error section. Full log available in Logs tab.
- **Recovery:** Check error, fix parameters, resubmit.

---

## Filesystem Failures

### Disk full during job
- **Trigger:** Scratch filesystem reaches quota or capacity.
- **Detection:** Job fails with IOError/OSError in logs. GUI's periodic disk check (every 60s) detects < 5 GB free.
- **User sees:** If detected during job: "Failed (disk full)". If detected by GUI poll: red banner at top "Less than 5 GB free on {filesystem}. Jobs may fail. Free space before submitting."
- **Recovery:** Delete old job outputs, then resubmit. GUI's disk usage display helps identify large jobs.

### Output directory not writable
- **Trigger:** Permissions, quota, nonexistent parent.
- **Detection:** Pre-submission validation (`os.access` check).
- **User sees:** Submission blocked with error: "Cannot write to {path}: {reason}".
- **Recovery:** Fix permissions or choose different output directory.

### Pipeline output corrupted / incomplete
- **Trigger:** Job killed mid-write, filesystem error.
- **Detection:** When opening job results, check for `metadata.json` and key output files. Missing files → "incomplete".
- **User sees:** Job status → "Completed (incomplete)". Warning: "Some output files are missing. The job may have been interrupted. Re-run to generate complete output." Available files are still viewable.
- **Recovery:** Resubmit the job.

### Imported pipeline output has unexpected structure
- **Trigger:** Scanning/importing output from old recovar version or manual directory.
- **Detection:** `ResultPaths` validation fails (missing metadata.json, different directory layout).
- **User sees:** Import status → "Imported (legacy)". Warning: "Imported from an older format. Some features may be unavailable." Available data is displayed on a best-effort basis.
- **Recovery:** None needed — this is informational. User can rerun pipeline if they want full GUI features.

---

## GUI Server Failures

### Backend restarts while jobs are running
- **Trigger:** Server crash, user restarts, machine reboot.
- **Detection:** On startup, the server runs the reconnect procedure (see ADR-001): query SQLite for RUNNING/QUEUED jobs, check actual status via executor.
- **User sees:** After browser refresh: jobs show correct current status (not stale). If a job completed while server was down, it shows as "Completed" with full results. Toast: "Server was restarted. Job statuses have been synchronized."
- **Recovery:** Automatic. No user action needed.

### Browser refresh during job
- **Trigger:** User hits F5 or navigates away and back.
- **Detection:** React app remounts, reconnects WebSocket.
- **User sees:** Brief loading state, then current job status and log position restored. No data loss. WebSocket reconnects and resumes streaming from current log position (server tracks last-sent byte offset per client).
- **Recovery:** Automatic.

### Browser tab closed, reopened later
- **Trigger:** User closes tab, comes back hours later.
- **Detection:** New WebSocket connection. Server sends current status for all jobs.
- **User sees:** Dashboard shows current state of all jobs. Any jobs that completed or failed while tab was closed show their final status with full logs available.
- **Recovery:** Automatic.

### WebSocket connection drops (network blip)
- **Trigger:** SSH tunnel hiccup, network timeout.
- **Detection:** WebSocket `onclose` event.
- **User sees:** Yellow banner: "Connection lost. Reconnecting..." Automatic reconnect with exponential backoff (1s, 2s, 4s, 8s, max 30s). On reconnect: banner disappears, log stream resumes from where it left off.
- **Recovery:** Automatic. If reconnect fails after 5 minutes: red banner "Cannot reach server. Check your SSH tunnel." with manual "Retry" button.

---

## Database Failures

### SQLite lock contention
- **Trigger:** Multiple concurrent writes (e.g., two SLURM jobs completing simultaneously update the same project DB).
- **Detection:** SQLite raises `OperationalError: database is locked`.
- **Mitigation:** WAL mode (configured at DB creation) allows concurrent reads + one writer. Write operations use a retry loop: 3 retries with 100ms, 500ms, 2000ms delays.
- **User sees:** Nothing (retries are invisible). If all retries fail (extremely rare): error toast "Database busy, please try again."
- **Recovery:** Automatic retry. If persistent: likely another process holding a lock — user checks for stale processes.

### SQLite database corrupted
- **Trigger:** Filesystem crash during write, NFS bugs (rare with WAL on local GPFS).
- **Detection:** SQLite raises `DatabaseError` on open or query.
- **User sees:** "Project database is corrupted. A backup may be available." The server checks for `recovar_project.db-wal` and `recovar_project.db-shm` files and attempts WAL recovery. If recovery fails, it checks for the most recent auto-backup (written before each migration).
- **Recovery:** Automatic WAL recovery in most cases. If that fails: restore from backup. Job output files on disk are unaffected — worst case, user recreates the project and re-scans.

### Database migration needed (version upgrade)
- **Trigger:** User upgrades recovar and opens a project created with an older version.
- **Detection:** Alembic version check on project open.
- **User sees:** "This project was created with an older version of recovar. Upgrading database..." Progress bar. Backup created automatically before migration.
- **Recovery:** Automatic. If migration fails: error with instructions to report a bug. Backup is preserved.

---

## Volume Viewer Failures

### Volume too large for browser memory
- **Trigger:** Loading a 512^3+ volume on a machine with limited RAM/GPU.
- **Detection:** Server-side: if volume dimensions exceed 256^3, downsample before serving. Client-side: if vtk.js allocation fails, catch the error.
- **User sees:** For auto-downsampled: banner "Volume downsampled from {original} to 256^3 for display. [Load full resolution]" button. For allocation failure: "Volume too large for your browser. Try closing other tabs or using a machine with more memory."
- **Recovery:** Downsample is automatic. Full resolution is opt-in.

### WebGL context lost
- **Trigger:** GPU driver reset, too many WebGL contexts, browser resource pressure.
- **Detection:** vtk.js `webglcontextlost` event.
- **User sees:** Volume viewer shows "3D rendering lost. [Restore]" button.
- **Recovery:** Click restore button → re-creates WebGL context and reloads current volume.

### vtk.js fails to load (no WebGL)
- **Trigger:** Older browser, software rendering, remote desktop without GPU passthrough.
- **Detection:** Check `WebGLRenderingContext` on app load.
- **User sees:** Volume viewer area shows: "3D rendering requires WebGL. Your browser does not support it. Try Chrome or Firefox on a machine with a GPU. Slice views are still available." Falls back to server-rendered PNG slice viewer.
- **Recovery:** Use a different browser or machine. Slice viewer works everywhere.

---

## Scatter Plot Failures

### Too many particles for browser
- **Trigger:** Dataset with > 2M particles.
- **Detection:** Server-side: if particle count > 2M, return a random 1M subsample + total count.
- **User sees:** Banner: "Showing 1M of {N} particles (random subsample). Selections and exports use all {N} particles."
- **Recovery:** Automatic. Subsampling is for display only — all operations (lasso export, subset creation) use the full dataset server-side.

---

## Network / Access Failures

### SSH tunnel not set up
- **Trigger:** User tries to access the GUI without an SSH tunnel.
- **Detection:** Connection refused in browser.
- **User sees:** Browser shows "Connection refused." They need to set up the tunnel.
- **Recovery:** The GUI prints the correct SSH command on startup in the terminal: `Access the GUI at: http://localhost:8080  |  SSH tunnel: ssh -L 8080:localhost:8080 user@cluster`

### Port already in use
- **Trigger:** Another process (or another GUI instance) is on port 8080.
- **Detection:** `uvicorn` fails to bind.
- **User sees:** Terminal error: "Port 8080 is in use. Use --port to specify a different port, or kill the existing process: lsof -i :8080"
- **Recovery:** `recovar gui --port 8081` or kill the other process.
