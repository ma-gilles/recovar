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
import json
import logging
import os
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
    """Return True if ``sbatch`` is on PATH."""
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
    ) -> str:
        opts = slurm_opts or {}
        slurm_output = os.path.join(working_dir, "slurm-%j.out")

        # Build sbatch script
        script = _render_sbatch_script(
            job_name=f"recovar-{job_id[:8]}",
            command=" ".join(command),
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
            "sbatch", "--parsable", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (rc={proc.returncode}): {stderr.decode().strip()}"
            )

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
            "scancel", handle,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        logger.info("Cancelled SLURM job %s", handle)

    async def log_path(self, handle: str) -> Path | None:
        path = self._log_paths.get(handle)
        if path and path.exists():
            return path
        # Try to find it by glob in the working dir
        return None

    async def cleanup(self, handle: str) -> None:
        self._log_paths.pop(handle, None)

    # -- Internal helpers --

    @staticmethod
    async def _squeue_state(slurm_id: str) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "squeue", "-j", slurm_id, "--noheader", "--format=%T",
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
                "sacct", "-j", slurm_id,
                "--format=State", "--noheader", "--parsable2",
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
    """Runs jobs as local subprocesses with process-group isolation."""

    # handle -> (process, log_path, pgid)
    _processes: dict[str, tuple[asyncio.subprocess.Process, Path, int]] = {}

    async def submit(
        self,
        job_id: str,
        command: list[str],
        env: dict[str, str],
        working_dir: str,
        *,
        slurm_opts: dict[str, Any] | None = None,
    ) -> str:
        log_file = os.path.join(working_dir, "run.log")
        full_env = {**os.environ, **env}

        # Open log file for stdout/stderr
        log_fh = open(log_file, "w")

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=log_fh,
            stderr=asyncio.subprocess.STDOUT,
            env=full_env,
            cwd=working_dir,
            start_new_session=True,  # new process group for clean cancel
        )

        handle = str(proc.pid)
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = proc.pid

        self._processes[handle] = (proc, Path(log_file), pgid)
        logger.info(
            "Started local process pid=%s pgid=%d for job_id=%s",
            handle, pgid, job_id,
        )
        return handle

    async def status(self, handle: str) -> JobStatus:
        entry = self._processes.get(handle)
        if entry is None:
            return JobStatus.UNKNOWN

        proc = entry[0]
        if proc.returncode is None:
            # Still running — check
            try:
                os.kill(proc.pid, 0)
                return JobStatus.RUNNING
            except ProcessLookupError:
                # Reap the process
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
            except OSError:
                return JobStatus.UNKNOWN

        rc = proc.returncode
        if rc is None:
            return JobStatus.RUNNING
        if rc == 0:
            return JobStatus.COMPLETED
        if rc == -signal.SIGTERM or rc == -signal.SIGKILL:
            return JobStatus.CANCELLED
        return JobStatus.FAILED

    async def cancel(self, handle: str) -> None:
        entry = self._processes.get(handle)
        if entry is None:
            return

        _proc, _log, pgid = entry
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass

        # Grace period then SIGKILL
        await asyncio.sleep(10)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

        logger.info("Cancelled local process group pgid=%d", pgid)

    async def log_path(self, handle: str) -> Path | None:
        entry = self._processes.get(handle)
        if entry is None:
            return None
        return entry[1]

    async def cleanup(self, handle: str) -> None:
        entry = self._processes.pop(handle, None)
        if entry is None:
            return
        proc = entry[0]
        if proc.returncode is None:
            try:
                proc.kill()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# SLURM sbatch template rendering
# ---------------------------------------------------------------------------


_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --gres=gpu:{gpus}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output={output_path}

# ── Environment ──
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp
mkdir -p "$TMPDIR"
{extra_exports}

# ── Staging cleanup trap ──
{cache_setup}

# ── Run the actual command ──
{command}
EXIT_CODE=$?

exit $EXIT_CODE
"""


def _render_sbatch_script(
    *,
    job_name: str,
    command: str,
    env_vars: dict[str, str],
    output_path: str,
    partition: str = "cryoem",
    account: str = "amits",
    gpus: int = 1,
    cpus: int = 4,
    memory: str = "300G",
    time: str = "12:00:00",
    cache_dir: str | None = None,
    **_extra: Any,
) -> str:
    """Render an sbatch script from parameters."""
    extra_exports = "\n".join(
        f"export {k}={v}" for k, v in env_vars.items()
    )

    if cache_dir:
        cache_setup = (
            f'export RECOVAR_CACHE_DIR={cache_dir}/recovar_cache_${{SLURM_JOB_ID}}\n'
            f'mkdir -p "$RECOVAR_CACHE_DIR"\n'
            f'cleanup() {{\n'
            f'    if [ -n "$RECOVAR_CACHE_DIR" ] && [ -d "$RECOVAR_CACHE_DIR" ]; then\n'
            f'        rm -rf "$RECOVAR_CACHE_DIR"\n'
            f'    fi\n'
            f'}}\n'
            f'trap cleanup EXIT TERM INT'
        )
    else:
        cache_setup = "# No cache staging configured"

    return _SBATCH_TEMPLATE.format(
        job_name=job_name,
        partition=partition,
        account=account,
        gpus=gpus,
        cpus=cpus,
        memory=memory,
        time=time,
        output_path=output_path,
        extra_exports=extra_exports,
        cache_setup=cache_setup,
        command=command,
    )


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
            updates.append({
                "id": job_info["id"],
                "new_status": JobStatus.FAILED.value,
                "error": "No executor handle — cannot reconcile after restart.",
            })
            continue

        actual = await executor.status(handle)
        db_status = job_info.get("db_status", "")

        if actual.value == db_status:
            # No change — if still running, we'll resume polling
            continue

        error = None
        if actual == JobStatus.UNKNOWN:
            actual = JobStatus.FAILED
            log = await executor.log_path(handle)
            error = (
                f"Job status unknown after server restart. "
                f"Check output: {log}"
            )
        elif actual == JobStatus.FAILED:
            error = "Job failed (detected on server restart)."

        updates.append({
            "id": job_info["id"],
            "new_status": actual.value,
            "error": error,
        })

    return updates
