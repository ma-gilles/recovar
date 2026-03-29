"""Jobs REST API.

Endpoints:
    POST   /api/jobs                    — Submit a new job
    GET    /api/jobs/:id                — Get job detail
    POST   /api/jobs/:id/cancel         — Cancel a job
    DELETE /api/jobs/:id                — Delete job record
    GET    /api/jobs/:id/volumes        — List MRC volumes in job output
    GET    /api/jobs/:id/plots          — List diagnostic PNGs
    GET    /api/jobs/:id/suggested-next — Suggest next steps
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from recovar.gui_v2.backend.api.project import _load_project_by_id, get_project_path
from recovar.gui_v2.backend.config import get_db_path
from recovar.gui_v2.backend.db import init_db
from recovar.gui_v2.backend.models.job import Job, JobStatus
from recovar.gui_v2.backend.services.command_builder import (
    build_analyze_command,
    build_compute_state_command,
    build_compute_trajectory_command,
    build_pipeline_command,
)
from recovar.gui_v2.backend.services.executor import (
    Executor,
    LocalExecutor,
    SlurmExecutor,
    slurm_available,
)
from recovar.gui_v2.backend.services.executor import (
    JobStatus as ExecJobStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# ---------------------------------------------------------------------------
# Module-level executor (initialized on first use)
# ---------------------------------------------------------------------------

_executor: Executor | None = None


def get_executor() -> Executor:
    global _executor
    if _executor is None:
        _executor = SlurmExecutor() if slurm_available() else LocalExecutor()
        mode = "SLURM" if isinstance(_executor, SlurmExecutor) else "local"
        logger.info("Executor initialized: %s", mode)
    return _executor


# ---------------------------------------------------------------------------
# Background status poller
# ---------------------------------------------------------------------------

_poll_tasks: dict[str, asyncio.Task] = {}


async def _poll_job_status(job_id: str, handle: str, project_path: str) -> None:
    """Background task that polls executor status until terminal."""
    executor = get_executor()
    while True:
        try:
            status = await executor.status(handle)
        except Exception:
            logger.exception("Error polling job %s", job_id)
            await asyncio.sleep(15)
            continue

        is_terminal = status in (
            ExecJobStatus.COMPLETED,
            ExecJobStatus.FAILED,
            ExecJobStatus.CANCELLED,
        )

        if is_terminal:
            # Update DB
            db_path = get_db_path(project_path)
            session_factory = await init_db(db_path)
            async with session_factory() as session:
                stmt = select(Job).where(Job.id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                if job:
                    job.status = status.value
                    job.completed_at = datetime.datetime.utcnow()
                    if status == ExecJobStatus.FAILED:
                        job.error = "Job failed (detected by status poller)."
                    await session.commit()
            _poll_tasks.pop(job_id, None)
            logger.info("Job %s reached terminal state: %s", job_id, status.value)
            return

        # Update running status if changed
        db_path = get_db_path(project_path)
        session_factory = await init_db(db_path)
        async with session_factory() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job and job.status != status.value:
                job.status = status.value
                await session.commit()

        interval = 5 if status == ExecJobStatus.RUNNING else 15
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SubmitJobRequest(BaseModel):
    project_id: str
    type: str  # Pipeline, Analyze, ComputeState, ComputeTrajectory
    params: dict[str, Any]


class SubmitJobResponse(BaseModel):
    id: str
    type: str
    status: str
    created: str
    handle: str | None = None


class JobDetailResponse(BaseModel):
    id: str
    type: str
    status: str
    params: dict | None = None
    created: str
    completed: str | None = None
    handle: str | None = None
    slurm_id: str | None = None
    error: str | None = None
    parent_jobs: list[str] | None = None
    output_dir: str


class VolumeEntry(BaseModel):
    name: str
    path: str
    category: str  # mean, eigen, variance, halfmap, mask, other
    size_bytes: int


class PlotEntry(BaseModel):
    name: str
    path: str


class SuggestedNext(BaseModel):
    type: str
    label: str
    prefilled_params: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_job(job_id: str) -> tuple[Job, Any]:
    """Find a job by ID across all known projects.  Returns (job, session)."""
    from recovar.gui_v2.backend.api.project import _project_registry

    for pid, ppath in _project_registry.items():
        db_path = get_db_path(ppath)
        session_factory = await init_db(db_path)
        session = session_factory()
        stmt = select(Job).where(Job.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        if job:
            return job, session
        await session.close()

    raise HTTPException(status_code=404, detail="Job not found")


def _categorize_volume(name: str) -> str:
    """Assign a display category to an MRC filename."""
    lower = name.lower()
    if "mean" in lower:
        return "mean"
    if "eigen" in lower:
        return "eigen"
    if "variance" in lower:
        return "variance"
    if "half" in lower and "unfil" in lower:
        return "halfmap"
    if "mask" in lower:
        return "mask"
    if "center" in lower or "state" in lower:
        return "reconstruction"
    return "other"


def _write_coords_file(path: str, coords: list[float]) -> None:
    """Write a latent coordinate vector to a text file."""
    with open(path, "w") as f:
        f.write(" ".join(f"{c:.8f}" for c in coords) + "\n")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=SubmitJobResponse, status_code=201)
async def submit_job(req: SubmitJobRequest) -> SubmitJobResponse:
    """Validate parameters, create job record, submit to executor."""
    project_path = get_project_path(req.project_id)
    if project_path is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Build command
    job_type = req.type
    params = req.params

    # Allocate output directory
    type_dir_map = {
        "Pipeline": "Pipeline",
        "Analyze": "Analyze",
        "ComputeState": "ReconstructState",
        "ComputeTrajectory": "ReconstructTrajectory",
    }
    dir_name = type_dir_map.get(job_type)
    if dir_name is None:
        raise HTTPException(status_code=400, detail=f"Unknown job type: {job_type}")

    # Find next job number
    type_dir = os.path.join(project_path, dir_name)
    os.makedirs(type_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(type_dir)
        if os.path.isdir(os.path.join(type_dir, d)) and d.startswith("job_")
    ]
    next_num = max((int(d.split("_")[1]) for d in existing), default=0) + 1
    job_dir = os.path.join(type_dir, f"job_{next_num:04d}")
    os.makedirs(job_dir, exist_ok=True)

    # Set outdir in params
    params["outdir"] = job_dir

    # Build command based on type
    if job_type == "Pipeline":
        command = build_pipeline_command(params)
    elif job_type == "Analyze":
        command = build_analyze_command(params)
    elif job_type == "ComputeState":
        coords_file = os.path.join(job_dir, "latent_points.txt")
        _write_coords_file(coords_file, params.get("latent_points", []))
        command = build_compute_state_command(params, coords_file)
    elif job_type == "ComputeTrajectory":
        z_start_file = os.path.join(job_dir, "z_start.txt")
        z_end_file = os.path.join(job_dir, "z_end.txt")
        _write_coords_file(z_start_file, params.get("z_start", []))
        _write_coords_file(z_end_file, params.get("z_end", []))
        command = build_compute_trajectory_command(params, z_start_file, z_end_file)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown job type: {job_type}")

    # Create job record in DB
    db_path = get_db_path(project_path)
    session_factory = await init_db(db_path)

    job = Job(
        project_id=req.project_id,
        type=job_type,
        status=JobStatus.QUEUED.value,
        params=params,
        output_dir=job_dir,
    )

    async with session_factory() as session:
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = job.id

    # Submit to executor
    executor = get_executor()
    env = {
        "PYTHONNOUSERSITE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    try:
        handle = await executor.submit(
            job_id=job_id,
            command=command,
            env=env,
            working_dir=job_dir,
            slurm_opts=params.get("slurm_opts"),
        )
    except Exception as exc:
        # Update job as failed
        async with session_factory() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = await session.execute(stmt)
            j = result.scalar_one()
            j.status = JobStatus.FAILED.value
            j.error = f"Submission failed: {exc}"
            await session.commit()
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}")

    # Store handle
    async with session_factory() as session:
        stmt = select(Job).where(Job.id == job_id)
        result = await session.execute(stmt)
        j = result.scalar_one()
        j.executor_handle = handle
        if isinstance(executor, SlurmExecutor):
            j.slurm_id = handle
        j.status = JobStatus.RUNNING.value
        await session.commit()

    # Start background status poller
    task = asyncio.create_task(_poll_job_status(job_id, handle, project_path))
    _poll_tasks[job_id] = task

    return SubmitJobResponse(
        id=job_id,
        type=job_type,
        status=JobStatus.RUNNING.value,
        created=job.created_at.isoformat(),
        handle=handle,
    )


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str) -> JobDetailResponse:
    job, session = await _get_job(job_id)
    try:
        return JobDetailResponse(
            id=job.id,
            type=job.type,
            status=job.status,
            params=job.params,
            created=job.created_at.isoformat(),
            completed=job.completed_at.isoformat() if job.completed_at else None,
            handle=job.executor_handle,
            slurm_id=job.slurm_id,
            error=job.error,
            parent_jobs=job.parent_job_ids,
            output_dir=job.output_dir,
        )
    finally:
        await session.close()


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job, session = await _get_job(job_id)
    try:
        if job.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value,
                          JobStatus.CANCELLED.value):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job in {job.status} state",
            )

        if job.executor_handle:
            executor = get_executor()
            await executor.cancel(job.executor_handle)

        job.status = JobStatus.CANCELLED.value
        job.completed_at = datetime.datetime.utcnow()
        await session.commit()

        # Cancel the poller
        poll_task = _poll_tasks.pop(job_id, None)
        if poll_task:
            poll_task.cancel()

        return {"status": "cancelled"}
    finally:
        await session.close()


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    job, session = await _get_job(job_id)
    try:
        await session.delete(job)
        await session.commit()
    finally:
        await session.close()


@router.get("/{job_id}/volumes", response_model=list[VolumeEntry])
async def list_volumes(job_id: str) -> list[VolumeEntry]:
    job, session = await _get_job(job_id)
    await session.close()

    volumes: list[VolumeEntry] = []
    output_dir = job.output_dir

    if not os.path.isdir(output_dir):
        return volumes

    for dirpath, _, filenames in os.walk(output_dir):
        for fname in sorted(filenames):
            if not fname.endswith(".mrc"):
                continue
            full = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            volumes.append(VolumeEntry(
                name=fname,
                path=full,
                category=_categorize_volume(fname),
                size_bytes=size,
            ))

    return volumes


@router.get("/{job_id}/plots", response_model=list[PlotEntry])
async def list_plots(job_id: str) -> list[PlotEntry]:
    job, session = await _get_job(job_id)
    await session.close()

    plots: list[PlotEntry] = []
    output_dir = job.output_dir

    if not os.path.isdir(output_dir):
        return plots

    for dirpath, _, filenames in os.walk(output_dir):
        for fname in sorted(filenames):
            if fname.endswith((".png", ".pdf")):
                full = os.path.join(dirpath, fname)
                plots.append(PlotEntry(name=fname, path=full))

    return plots


@router.get("/{job_id}/suggested-next", response_model=list[SuggestedNext])
async def suggested_next(job_id: str) -> list[SuggestedNext]:
    job, session = await _get_job(job_id)
    await session.close()

    suggestions: list[SuggestedNext] = []

    if job.status != JobStatus.COMPLETED.value:
        return suggestions

    if job.type == "Pipeline":
        suggestions.append(SuggestedNext(
            type="Analyze",
            label="Analyze this pipeline output",
            prefilled_params={"result_dir": job.output_dir},
        ))
    elif job.type == "Analyze":
        suggestions.append(SuggestedNext(
            type="ComputeState",
            label="Compute volume at a latent point",
            prefilled_params={"result_dir": job.params.get("result_dir", "")},
        ))
        suggestions.append(SuggestedNext(
            type="ComputeTrajectory",
            label="Compute trajectory between two points",
            prefilled_params={"result_dir": job.params.get("result_dir", "")},
        ))

    return suggestions
