"""WebSocket log and status streaming.

Endpoint:
    WS /api/jobs/:id/stream?last_offset=0

Sends: log_line, status_change, progress, reconnect_sync, ping
Receives: pong
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select

from recovar.gui_v2.backend.api.jobs import _get_job, get_executor_for_job
from recovar.gui_v2.backend.api.project import get_project_path
from recovar.gui_v2.backend.config import get_db_path
from recovar.gui_v2.backend.db import init_db
from recovar.gui_v2.backend.models.job import Job, JobStatus
from recovar.gui_v2.backend.services.executor import JobStatus as ExecJobStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Progress pattern: "Pass 3/5" or "Step 3 of 10"
_PROGRESS_RE = re.compile(
    r"(?:Pass|Step|Iteration)\s+(\d+)\s*(?:/|of)\s*(\d+)", re.IGNORECASE
)


def _msg(type: str, data: dict) -> str:
    """Format a WebSocket message."""
    return json.dumps({"type": type, "data": data, "ts": int(time.time() * 1000)})


async def _tail_log(
    log_path: Path,
    offset: int,
    ws: WebSocket,
    stop_event: asyncio.Event,
) -> int:
    """Tail a log file from *offset*, sending lines over WebSocket.

    Returns the new byte offset.
    """
    if not log_path.exists():
        return offset

    try:
        with open(log_path, "r") as f:
            f.seek(offset)
            while not stop_event.is_set():
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue

                offset = f.tell()
                line = line.rstrip("\n")

                await ws.send_text(_msg("log_line", {
                    "line": line,
                    "offset": offset,
                }))

                # Check for progress patterns
                m = _PROGRESS_RE.search(line)
                if m:
                    await ws.send_text(_msg("progress", {
                        "step": int(m.group(1)),
                        "total": int(m.group(2)),
                        "label": line.strip(),
                    }))
    except (OSError, asyncio.CancelledError):
        pass

    return offset


@router.websocket("/api/jobs/{job_id}/stream")
async def job_stream(
    ws: WebSocket,
    job_id: str,
    last_offset: int = Query(0),
):
    """Stream log lines and status updates for a job."""
    await ws.accept()

    # Look up the job
    try:
        job, session = await _get_job(job_id)
    except Exception:
        await ws.send_text(_msg("status_change", {
            "old": "unknown",
            "new": "unknown",
            "error": "Job not found",
        }))
        await ws.close()
        return
    finally:
        try:
            await session.close()
        except Exception:
            pass

    # Find log file. Use the job's OWN executor (PID-based local vs SLURM id);
    # the server default would query the wrong backend for a mixed-mode host.
    log_path: Path | None = None
    executor = get_executor_for_job(job)
    if job.executor_handle:
        log_path = await executor.log_path(job.executor_handle)
    if log_path is None and job.executor_handle:
        # After server restart, executor loses in-memory log paths.
        # Try the standard SLURM output pattern.
        candidate = Path(job.output_dir) / f"slurm-{job.executor_handle}.out"
        if candidate.exists():
            log_path = candidate
    if log_path is None:
        log_path = Path(job.output_dir) / "run.log"

    # Send reconnect_sync with current state and recent log lines
    tail_lines: list[str] = []
    if log_path and log_path.exists():
        try:
            text = log_path.read_text()
            all_lines = text.split("\n")
            tail_lines = all_lines[-50:]
        except OSError:
            pass

    await ws.send_text(_msg("reconnect_sync", {
        "status": job.status,
        "log_offset": last_offset,
        "log_tail": tail_lines,
    }))

    # If terminal state, close immediately
    if job.status in (
        JobStatus.COMPLETED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
    ):
        await ws.close()
        return

    # Start tailing and status polling
    stop_event = asyncio.Event()
    current_offset = last_offset
    current_status = job.status

    async def tail_task():
        nonlocal current_offset
        if log_path:
            current_offset = await _tail_log(log_path, current_offset, ws, stop_event)

    async def status_poll_task():
        nonlocal current_status
        while not stop_event.is_set():
            await asyncio.sleep(5)

            if not job.executor_handle:
                continue

            try:
                # Local jobs need the working dir to read run.exitcode on exit;
                # otherwise a finished local PID reports UNKNOWN.
                if job.output_dir and hasattr(executor, "status_with_dir"):
                    new_status = await executor.status_with_dir(job.executor_handle, job.output_dir)
                else:
                    new_status = await executor.status(job.executor_handle)
            except Exception:
                continue

            if new_status.value != current_status:
                old = current_status
                current_status = new_status.value
                await ws.send_text(_msg("status_change", {
                    "old": old,
                    "new": current_status,
                }))

                if new_status in (
                    ExecJobStatus.COMPLETED,
                    ExecJobStatus.FAILED,
                    ExecJobStatus.CANCELLED,
                ):
                    stop_event.set()

    async def ping_task():
        while not stop_event.is_set():
            try:
                await ws.send_text(_msg("ping", {}))
            except Exception:
                stop_event.set()
                return
            await asyncio.sleep(30)

    async def receive_task():
        """Listen for client messages (pong) or disconnects."""
        try:
            while not stop_event.is_set():
                await ws.receive_text()
        except WebSocketDisconnect:
            stop_event.set()

    # Run all tasks concurrently
    tasks = [
        asyncio.create_task(tail_task()),
        asyncio.create_task(status_poll_task()),
        asyncio.create_task(ping_task()),
        asyncio.create_task(receive_task()),
    ]

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        stop_event.set()
        for t in pending:
            t.cancel()
    except Exception:
        stop_event.set()
        for t in tasks:
            t.cancel()

    try:
        await ws.close()
    except Exception:
        pass
