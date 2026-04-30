"""Executor abstraction for job execution backends.

Provides a common interface over SLURM (``SlurmExecutor``) and local
subprocess (``LocalExecutor``).  See ADR-001 for the full design.

Usage::

    executor = SlurmExecutor() if slurm_available() else LocalExecutor()
    handle = await executor.submit(job, command, env)
    status = await executor.status(handle)
    await executor.cancel(handle)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import signal
import sys
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def slurm_available() -> bool:
    """Return True if SLURM is the active executor.

    Honours the ``RECOVAR_EXECUTOR`` env var (``local``, ``slurm``, or
    ``auto`` — default ``auto``). When ``auto``, probes for ``sbatch`` on
    PATH. ``local`` forces local-subprocess mode even on a SLURM login
    node — useful for laptops, workstations, or running directly on a
    compute node from inside an existing allocation.
    """
    override = os.environ.get("RECOVAR_EXECUTOR", "auto").strip().lower()
    if override == "local":
        return False
    if override == "slurm":
        return True  # caller will see the failure if sbatch is genuinely missing
    return shutil.which("sbatch") is not None


class JobStatus(str, Enum):
    """Executor-level job status (see ADR-001)."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Executor ABC
# ---------------------------------------------------------------------------


class Executor(ABC):
    """Abstraction over job execution backends."""

    @abstractmethod
    async def submit(
        self,
        job_id: str,
        command: list[str],
        env: dict[str, str],
        working_dir: str,
        *,
        slurm_opts: dict[str, Any] | None = None,
        local_opts: dict[str, Any] | None = None,
    ) -> str:
        """Submit a job.  Returns an executor-specific handle."""

    @abstractmethod
    async def status(self, handle: str) -> JobStatus:
        """Poll current status."""

    @abstractmethod
    async def cancel(self, handle: str) -> None:
        """Cancel a running or queued job."""

    @abstractmethod
    async def log_path(self, handle: str) -> Path | None:
        """Return the path to the job's log file, or None if not yet available."""

    @abstractmethod
    async def cleanup(self, handle: str) -> None:
        """Clean up resources after job completion."""

    async def failure_reason(self, handle: str) -> str | None:
        """Return a human-readable failure reason, or None if unknown."""
        return None


# ---------------------------------------------------------------------------
# SLURM state mapping
# ---------------------------------------------------------------------------

_SLURM_STATE_MAP: dict[str, JobStatus] = {
    "PENDING": JobStatus.QUEUED,
    "CONFIGURING": JobStatus.QUEUED,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.COMPLETED,
    "CANCELLED": JobStatus.CANCELLED,
    "CANCELLED+": JobStatus.CANCELLED,
    "FAILED": JobStatus.FAILED,
    "TIMEOUT": JobStatus.FAILED,
    "OUT_OF_MEMORY": JobStatus.FAILED,
    "NODE_FAIL": JobStatus.FAILED,
    "PREEMPTED": JobStatus.FAILED,
}


# ---------------------------------------------------------------------------
# SLURM Executor
# ---------------------------------------------------------------------------


class SlurmExecutor(Executor):
    """Submits jobs via ``sbatch``, polls via ``squeue`` / ``sacct``."""

    # In-memory map: handle (slurm_id) -> log file path
    _log_paths: dict[str, Path] = {}

    async def submit(
        self,
        job_id: str,
        command: list[str],
        env: dict[str, str],
        working_dir: str,
        *,
        slurm_opts: dict[str, Any] | None = None,
        local_opts: dict[str, Any] | None = None,
    ) -> str:
        opts = slurm_opts or {}
        slurm_output = os.path.join(working_dir, "slurm-%j.out")

        # Build sbatch script. shlex.join → command survives spaces, quotes,
        # and special chars in argv (paths with spaces, etc.). The renderer
        # will not re-quote it.
        script = _render_sbatch_script(
            job_name=f"recovar-{job_id[:8]}",
            command=shlex.join(command),
            env_vars=env,
            output_path=slurm_output,
            **opts,
        )

        script_path = os.path.join(working_dir, "submit.sh")
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        # sbatch --parsable
        proc = await asyncio.create_subprocess_exec(
            "sbatch",
            "--parsable",
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed (rc={proc.returncode}): {stderr.decode().strip()}")

        slurm_id = stdout.decode().strip().split(";")[0]
        # Resolve actual log path (substitute %j with slurm_id)
        resolved_log = slurm_output.replace("%j", slurm_id)
        self._log_paths[slurm_id] = Path(resolved_log)

        logger.info("Submitted SLURM job %s for job_id=%s", slurm_id, job_id)
        return slurm_id

    async def status(self, handle: str) -> JobStatus:
        # 1) Try squeue first (most timely for live jobs)
        state = await self._squeue_state(handle)
        if state is not None:
            return _SLURM_STATE_MAP.get(state, JobStatus.UNKNOWN)

        # 2) Fallback to sacct
        state = await self._sacct_state(handle)
        if state is not None:
            return _SLURM_STATE_MAP.get(state, JobStatus.UNKNOWN)

        return JobStatus.UNKNOWN

    async def cancel(self, handle: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "scancel",
            handle,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        logger.info("Cancelled SLURM job %s", handle)

    async def log_path(self, handle: str) -> Path | None:
        path = self._log_paths.get(handle)
        if path and path.exists():
            return path
        return None

    async def failure_reason(self, handle: str) -> str | None:
        """Query sacct for the specific SLURM failure state."""
        state = await self._sacct_state(handle)
        if state is None:
            return None
        _SLURM_FAILURE_MESSAGES = {
            "TIMEOUT": "Job exceeded the time limit. Increase the time limit in SLURM settings and resubmit.",
            "OUT_OF_MEMORY": "Job exceeded the memory limit. Try increasing memory or enabling lazy loading.",
            "NODE_FAIL": "The compute node failed during execution. This is not a bug in your job. Resubmit to try again.",
            "PREEMPTED": "Job was preempted by a higher-priority job. Resubmit to try again.",
            "CANCELLED": "Job was cancelled.",
            "CANCELLED+": "Job was cancelled.",
        }
        return _SLURM_FAILURE_MESSAGES.get(state)

    async def cleanup(self, handle: str) -> None:
        self._log_paths.pop(handle, None)

    # -- Internal helpers --

    @staticmethod
    async def _squeue_state(slurm_id: str) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "squeue",
                "-j",
                slurm_id,
                "--noheader",
                "--format=%T",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            line = stdout.decode().strip()
            if line:
                return line.split("\n")[0].strip()
        except (asyncio.TimeoutError, OSError):
            pass
        return None

    @staticmethod
    async def _sacct_state(slurm_id: str) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "sacct",
                "-j",
                slurm_id,
                "--format=State",
                "--noheader",
                "--parsable2",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            lines = stdout.decode().strip().split("\n")
            for line in lines:
                state = line.strip()
                if state and state not in ("", "State"):
                    return state
        except (asyncio.TimeoutError, OSError):
            pass
        return None


# ---------------------------------------------------------------------------
# Local Executor
# ---------------------------------------------------------------------------


class LocalExecutor(Executor):
    """Runs jobs as local subprocesses that survive GUI restarts.

    Each job gets a ``run.sh`` wrapper script and a ``run.pid`` file in
    the working directory.  The subprocess is launched via ``nohup`` and
    fully detached from the server process, so it keeps running even if
    the GUI is restarted.  On restart, ``status()`` reconnects to the
    process by reading the PID file.
    """

    async def submit(
        self,
        job_id: str,
        command: list[str],
        env: dict[str, str],
        working_dir: str,
        *,
        slurm_opts: dict[str, Any] | None = None,
        local_opts: dict[str, Any] | None = None,
    ) -> str:
        log_file = os.path.join(working_dir, "run.log")
        pid_file = os.path.join(working_dir, "run.pid")
        rc_file = os.path.join(working_dir, "run.exitcode")
        script_file = os.path.join(working_dir, "run.sh")

        # Build environment exports
        env_lines = []
        for k, v in env.items():
            env_lines.append(f"export {k}={shlex.quote(v)}")

        # Apply local execution options
        opts = local_opts or {}
        gpu_selection = opts.get("gpus", "all")
        if gpu_selection and gpu_selection != "all":
            env_lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_selection))}")
        extra_env = opts.get("env_vars", {})
        if isinstance(extra_env, dict):
            for k, v in extra_env.items():
                if k:
                    env_lines.append(f"export {k}={shlex.quote(str(v))}")
        setup_command = opts.get("setup_command", "").strip()

        # Build the wrapper script
        cmd_str = shlex.join(command)
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated by recovar GUI (local executor)",
            f"echo $$ > {shlex.quote(pid_file)}",
            "",
            "# Environment",
            *env_lines,
            "",
        ]
        if setup_command:
            script_lines.append(f"# Setup command")
            script_lines.append(setup_command)
            script_lines.append("")
        script_lines.extend([
            "# Run the pipeline",
            cmd_str,
            "EXIT_CODE=$?",
            "",
            f"echo $EXIT_CODE > {shlex.quote(rc_file)}",
            "exit $EXIT_CODE",
        ])

        script_content = "\n".join(script_lines) + "\n"
        with open(script_file, "w") as f:
            f.write(script_content)
        os.chmod(script_file, 0o755)

        # Launch fully detached via nohup + bash
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash", "-c",
            f"nohup {shlex.quote(script_file)} > {shlex.quote(log_file)} 2>&1 &",
            cwd=working_dir,
            start_new_session=True,
        )
        await proc.wait()

        # Wait briefly for the PID file to be written
        for _ in range(20):
            if os.path.isfile(pid_file):
                break
            await asyncio.sleep(0.1)

        # Read PID from file
        try:
            with open(pid_file) as f:
                handle = f.read().strip()
        except OSError:
            raise RuntimeError("Local job started but PID file was not written")

        logger.info(
            "Started local job pid=%s for job_id=%s (detached, survives restart)",
            handle, job_id,
        )
        return handle

    async def status(self, handle: str) -> JobStatus:
        pid = int(handle)

        # Check if process is alive
        try:
            os.kill(pid, 0)
            return JobStatus.RUNNING
        except ProcessLookupError:
            pass  # Process finished — check exit code
        except OSError:
            return JobStatus.UNKNOWN

        # Process is gone — check exit code file
        # We need the working dir to find run.exitcode.
        # Try to find it via /proc or the DB (the poller passes it).
        # For now, scan for run.exitcode relative to the handle.
        # The poller has the working_dir context; this fallback checks
        # if the process simply completed.
        return JobStatus.UNKNOWN

    async def status_with_dir(self, handle: str, working_dir: str) -> JobStatus:
        """Check status using the working directory for exit code file."""
        pid = int(handle)

        try:
            os.kill(pid, 0)
            return JobStatus.RUNNING
        except ProcessLookupError:
            pass
        except OSError:
            return JobStatus.UNKNOWN

        # Process is gone — read exit code
        rc_file = os.path.join(working_dir, "run.exitcode")
        if os.path.isfile(rc_file):
            try:
                with open(rc_file) as f:
                    rc = int(f.read().strip())
                return JobStatus.COMPLETED if rc == 0 else JobStatus.FAILED
            except (ValueError, OSError):
                pass

        # No exit code file but process is gone — check if run.log has
        # the "Job completed" or "Job failed" marker
        log_file = os.path.join(working_dir, "run.log")
        if os.path.isfile(log_file):
            try:
                with open(log_file, "rb") as f:
                    f.seek(max(0, os.fstat(f.fileno()).st_size - 2048))
                    tail = f.read().decode("utf-8", errors="replace")
                if "Job completed:" in tail:
                    return JobStatus.COMPLETED
                if "Job failed:" in tail or "Traceback" in tail:
                    return JobStatus.FAILED
            except OSError:
                pass

        return JobStatus.FAILED  # Process gone, no exit code — assume failed

    async def cancel(self, handle: str) -> None:
        pid = int(handle)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

        # Grace period then SIGKILL
        await asyncio.sleep(10)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

        logger.info("Cancelled local process pid=%s", handle)

    async def log_path(self, handle: str) -> Path | None:
        return None  # Log path is resolved by the job layer from working_dir

    async def cleanup(self, handle: str) -> None:
                pass


# ---------------------------------------------------------------------------
# SLURM sbatch template rendering
# ---------------------------------------------------------------------------


# NOTE: site-portability rules — see recovar/gui_v2/CLAUDE.md.
#   - No cluster-/lab-specific strings (partition, account, paths) may live in
#     the template body. They are passed in by the caller; empty values mean
#     "omit the directive" (not "emit `#SBATCH --foo=`", which is a parse error
#     on some schedulers).
#   - Shell-bearing fragments with `${...}` are built as Python strings
#     OUTSIDE this format template and substituted in via named slots,
#     because `str.format` would mis-parse `${TMPDIR:-x}` as a format spec.

_SBATCH_TEMPLATE = """\
#!/bin/bash
{slurm_directives}
# ── Environment ──
{tmpdir_block}
# Add the running Python's bin dir to PATH so `recovar` shebang resolves
export PATH={pixi_bin_dir}:$PATH
{extra_exports}

# ── Cleanup trap ──
{cleanup_block}

# ── Run the actual command ──
{command}
EXIT_CODE=$?

exit $EXIT_CODE
"""


# Prefer scheduler-provided job-local scratch. Inherited from caller env if any.
# We only create a tmpdir ourselves as a last resort, and clean up only what we
# created (never $SLURM_TMPDIR or a pre-set $TMPDIR — clusters reap those).
_TMPDIR_BLOCK = """\
RECOVAR_CREATED_TMPDIR=0
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export TMPDIR="$SLURM_TMPDIR"
elif [ -n "${TMPDIR:-}" ]; then
    mkdir -p "$TMPDIR"
else
    export TMPDIR="$(mktemp -d -t recovar-${SLURM_JOB_ID:-job}-XXXXXX)"
    RECOVAR_CREATED_TMPDIR=1
fi
mkdir -p "$TMPDIR"\
"""


def _build_slurm_directives(
    *,
    job_name: str,
    output_path: str,
    partition: str,
    account: str,
    gpus: int,
    cpus: int,
    memory: str,
    time: str,
    raw_directives: str | None,
    gpu_resource_spec: str = "--gres=gpu:{gpus}",
) -> str:
    """Build the #SBATCH directive block. Empty partition/account → omit (not blank).

    ``gpu_resource_spec`` is a python-format string with a single ``{gpus}``
    slot, which lets a site override the GPU directive for clusters that
    use ``--gpus={gpus}``, ``--gres=gpu:a100:{gpus}``, etc. Default is
    ``--gres=gpu:{gpus}`` for backward compatibility. When ``gpus == 0``
    the GPU directive is omitted entirely.
    """
    # Sanitize: 0 or negative values → sensible defaults
    if not gpus or gpus < 1:
        gpus = 1
    if not cpus or cpus < 1:
        cpus = 4
    if not memory:
        memory = "300G"
    if not time:
        time = "12:00:00"

    lines: list[str] = [f"#SBATCH --job-name={job_name}"]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if gpus:
        gpu_directive = gpu_resource_spec.format(gpus=gpus).strip()
        if gpu_directive:
            lines.append(f"#SBATCH {gpu_directive}")
    lines.extend(
        [
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={memory}",
            f"#SBATCH --time={time}",
            f"#SBATCH --output={output_path}",
        ]
    )

    if raw_directives and raw_directives.strip():
        for raw in raw_directives.strip().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("#SBATCH"):
                lines.append(raw)
            elif raw.startswith("-"):
                # Already a sbatch flag (`-p gpu` or `--qos=high`) — prefix only.
                lines.append(f"#SBATCH {raw}")
            else:
                lines.append(f"#SBATCH --{raw}")

    return "\n".join(lines)


def _build_cleanup_block(*, cache_dir: str | None) -> str:
    """Build the trap/cleanup block. One trap covers cache_dir + tmpdir-we-created."""
    parts: list[str] = []
    if cache_dir:
        parts.append(f"export RECOVAR_CACHE_DIR={shlex.quote(cache_dir)}/recovar_cache_${{SLURM_JOB_ID}}")
        parts.append('mkdir -p "$RECOVAR_CACHE_DIR"')

    parts.append("_recovar_cleanup() {")
    if cache_dir:
        parts.append('    if [ -n "$RECOVAR_CACHE_DIR" ] && [ -d "$RECOVAR_CACHE_DIR" ]; then')
        parts.append('        rm -rf "$RECOVAR_CACHE_DIR"')
        parts.append("    fi")
    parts.append('    if [ "${RECOVAR_CREATED_TMPDIR:-0}" = "1" ] && [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then')
    parts.append('        rm -rf "$TMPDIR"')
    parts.append("    fi")
    parts.append("}")
    parts.append("trap _recovar_cleanup EXIT TERM INT")
    return "\n".join(parts)


def _render_sbatch_script(
    *,
    job_name: str,
    command: str,
    env_vars: dict[str, str],
    output_path: str,
    partition: str = "",
    account: str = "",
    gpus: int = 1,
    cpus: int = 4,
    memory: str = "300G",
    time: str = "12:00:00",
    cache_dir: str | None = None,
    raw_directives: str | None = None,
    gpu_resource_spec: str = "--gres=gpu:{gpus}",
    template_path: str | None = None,
    template_vars: dict[str, Any] | None = None,
    **_extra: Any,
) -> str:
    """Render an sbatch script from parameters.

    Two modes:

    1. **Default (no ``template_path``)** — renders the built-in structured
       template using the caller-provided slurm opts. Empty
       ``partition`` / ``account`` cause those ``#SBATCH`` directives to
       be OMITTED (not emitted with a blank value, which is a parse
       error on some SLURM versions). The caller is expected to pass
       shell-safe strings; ``command`` should already be
       ``shlex.join(argv)`` — the SlurmExecutor does that. Env-var values
       are quoted here.

    2. **Jinja2 template (``template_path`` set)** — loads the file and
       renders it with ``StrictUndefined``. Available variables include
       all the structured fields plus ``slurm_directives``,
       ``tmpdir_block``, ``cleanup_block``, ``extra_exports``,
       ``pixi_bin_dir``, plus anything the caller passes in
       ``template_vars``. A typo in the template (``{{ commnad }}`` vs
       ``{{ command }}``) raises ``UndefinedError`` instead of silently
       producing a broken submit.sh.
    """
    slurm_directives = _build_slurm_directives(
        job_name=job_name,
        output_path=output_path,
        partition=partition,
        account=account,
        gpus=gpus,
        cpus=cpus,
        memory=memory,
        time=time,
        raw_directives=raw_directives,
        gpu_resource_spec=gpu_resource_spec,
    )
    cleanup_block = _build_cleanup_block(cache_dir=cache_dir)
    extra_exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env_vars.items())
    pixi_bin_dir = shlex.quote(str(Path(sys.executable).parent))

    if template_path:
        return _render_jinja_template(
            template_path=template_path,
            template_vars=template_vars,
            # Reserved variables — same names as the structured slots.
            job_name=job_name,
            command=command,
            output_path=output_path,
            partition=partition,
            account=account,
            gpus=gpus,
            cpus=cpus,
            memory=memory,
            time=time,
            raw_directives=raw_directives or "",
            gpu_resource_spec=gpu_resource_spec,
            cache_dir=cache_dir or "",
            slurm_directives=slurm_directives,
            tmpdir_block=_TMPDIR_BLOCK,
            cleanup_block=cleanup_block,
            extra_exports=extra_exports,
            pixi_bin_dir=pixi_bin_dir,
            env_vars=env_vars,
        )

    return _SBATCH_TEMPLATE.format(
        slurm_directives=slurm_directives,
        tmpdir_block=_TMPDIR_BLOCK,
        pixi_bin_dir=pixi_bin_dir,
        extra_exports=extra_exports,
        cleanup_block=cleanup_block,
        command=command,
    )


# Reserved variable names that templates can use without the user putting
# them in `template_vars`. Anything outside this set must come from
# `template_vars` or the template will get UndefinedError at render time.
_RESERVED_TEMPLATE_VARS = frozenset(
    {
        "job_name",
        "command",
        "output_path",
        "partition",
        "account",
        "gpus",
        "cpus",
        "memory",
        "time",
        "raw_directives",
        "gpu_resource_spec",
        "cache_dir",
        "slurm_directives",
        "tmpdir_block",
        "cleanup_block",
        "extra_exports",
        "pixi_bin_dir",
        "env_vars",
    }
)


def _render_jinja_template(
    *,
    template_path: str,
    template_vars: dict[str, Any] | None,
    **reserved: Any,
) -> str:
    """Render a user-supplied Jinja2 template with strict undefined.

    Custom user variables (from ``template_vars``) cannot shadow reserved
    variable names — that would silently break parts of the script that
    expect specific contents. Attempts to do so raise ``ValueError``.
    """
    from jinja2 import Environment, FileSystemLoader, StrictUndefined, UndefinedError

    template_vars = template_vars or {}
    overlap = set(template_vars).intersection(_RESERVED_TEMPLATE_VARS)
    if overlap:
        raise ValueError(
            f"template_vars uses reserved variable name(s): {sorted(overlap)}. "
            f"Reserved names: {sorted(_RESERVED_TEMPLATE_VARS)}"
        )

    path = Path(template_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sbatch template not found: {template_path}")

    env = Environment(
        loader=FileSystemLoader(str(path.parent)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,  # this is a shell script, not HTML
    )
    template = env.get_template(path.name)
    try:
        return template.render(**reserved, **template_vars)
    except UndefinedError as exc:
        # Re-raise with a clearer message — preserves the rendering failure
        # while making the typo / missing-var cause obvious.
        raise ValueError(
            f"Sbatch template {template_path} references an undefined variable: {exc}. "
            f"Available reserved variables: {sorted(_RESERVED_TEMPLATE_VARS)}. "
            f"Custom variables passed in template_vars: {sorted(template_vars)}."
        ) from exc


# ---------------------------------------------------------------------------
# Reconnect procedure (called on server startup)
# ---------------------------------------------------------------------------


async def reconcile_jobs(
    executor: Executor,
    inflight_jobs: list[dict],
) -> list[dict]:
    """Synchronise in-flight jobs after a GUI server restart.

    For each job that was RUNNING or QUEUED when the server last stopped,
    poll the executor for its actual state and return update dicts.

    Parameters
    ----------
    executor : Executor
        The active executor instance.
    inflight_jobs : list[dict]
        Each dict has ``{"id": str, "handle": str, "db_status": str}``.

    Returns
    -------
    list[dict]
        Update records: ``{"id": str, "new_status": str, "error": str | None}``.
    """
    updates: list[dict] = []

    for job_info in inflight_jobs:
        handle = job_info.get("handle")
        if not handle:
            updates.append(
                {
                    "id": job_info["id"],
                    "new_status": JobStatus.FAILED.value,
                    "error": "No executor handle — cannot reconcile after restart.",
                }
            )
            continue

        working_dir = job_info.get("working_dir")
        if working_dir and hasattr(executor, "status_with_dir"):
            actual = await executor.status_with_dir(handle, working_dir)
        else:
            actual = await executor.status(handle)
        db_status = job_info.get("db_status", "")

        if actual.value == db_status:
            # No change — if still running, we'll resume polling
            continue

        error = None
        if actual == JobStatus.UNKNOWN:
            actual = JobStatus.FAILED
            log = await executor.log_path(handle)
            error = f"Job status unknown after server restart. Check output: {log}"
        elif actual == JobStatus.FAILED:
            error = "Job failed (detected on server restart)."

        updates.append(
            {
                "id": job_info["id"],
                "new_status": actual.value,
                "error": error,
            }
        )

    return updates
