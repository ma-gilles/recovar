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
    build_density_command,
    build_downsample_command,
    build_pipeline_command,
    build_postprocess_command,
    build_stable_states_command,
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
    execution_mode: str  # "slurm" or "local"
    execution_summary: str  # Human-readable, e.g. "SLURM (job 123, partition: cryoem)"


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
    params = req.params

    # Normalize job type: accept lowercase/underscore forms from the frontend
    _type_aliases = {
        "pipeline": "Pipeline",
        "analyze": "Analyze",
        "compute_state": "ComputeState",
        "computestate": "ComputeState",
        "compute_trajectory": "ComputeTrajectory",
        "computetrajectory": "ComputeTrajectory",
        "density": "Density",
        "estimate_conformational_density": "Density",
        "stable_states": "StableStates",
        "stablestates": "StableStates",
        "estimate_stable_states": "StableStates",
        "postprocess": "Postprocess",
        "downsample": "Downsample",
    }
    job_type = _type_aliases.get(req.type.lower().replace("-", "_"), req.type)

    # Allocate output directory
    type_dir_map = {
        "Pipeline": "Pipeline",
        "Analyze": "Analyze",
        "ComputeState": "ReconstructState",
        "ComputeTrajectory": "ReconstructTrajectory",
        "Density": "Density",
        "StableStates": "StableStates",
        "Postprocess": "Postprocess",
        "Downsample": "Downsample",
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
    elif job_type == "Density":
        command = build_density_command(params)
    elif job_type == "StableStates":
        command = build_stable_states_command(params)
    elif job_type == "Postprocess":
        command = build_postprocess_command(params)
    elif job_type == "Downsample":
        command = build_downsample_command(params)
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
        # Build human-readable execution summary
        if job.slurm_id:
            exec_mode = "slurm"
            # Try to read partition/account from the submit.sh if available
            slurm_partition = None
            slurm_account = None
            submit_sh = os.path.join(job.output_dir, "submit.sh")
            if os.path.isfile(submit_sh):
                try:
                    with open(submit_sh) as f:
                        for line in f:
                            if line.startswith("#SBATCH --partition="):
                                slurm_partition = line.strip().split("=", 1)[1]
                            elif line.startswith("#SBATCH --account="):
                                slurm_account = line.strip().split("=", 1)[1]
                except OSError:
                    pass
            parts = [f"job {job.slurm_id}"]
            if slurm_partition:
                parts.append(f"partition: {slurm_partition}")
            if slurm_account:
                parts.append(f"account: {slurm_account}")
            exec_summary = f"SLURM ({', '.join(parts)})"
        else:
            exec_mode = "local"
            pid_str = job.executor_handle or "unknown"
            exec_summary = f"Local (PID {pid_str})"

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
            execution_mode=exec_mode,
            execution_summary=exec_summary,
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
        suggestions.append(SuggestedNext(
            type="Density",
            label="Estimate conformational density",
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
    elif job.type == "Density":
        density_pkl = os.path.join(
            job.output_dir, "data", "deconv_density_knee.pkl"
        )
        suggestions.append(SuggestedNext(
            type="StableStates",
            label="Find stable states from density",
            prefilled_params={"density": density_pkl},
        ))

    return suggestions


class ReconcileResponse(BaseModel):
    id: str
    previous_status: str
    new_status: str
    changed: bool
    error: str | None = None


@router.post("/{job_id}/reconcile", response_model=ReconcileResponse)
async def reconcile_job(job_id: str) -> ReconcileResponse:
    """Force a status check for a specific job against the executor.

    Useful when a job appears stuck (e.g. SLURM says COMPLETED but the
    GUI still shows running).  Queries the actual executor status, updates
    the DB, and cancels/restarts the background poller as needed.
    """
    job, session = await _get_job(job_id)
    try:
        previous_status = job.status

        # Already terminal — nothing to do
        if previous_status in (
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ):
            return ReconcileResponse(
                id=job_id,
                previous_status=previous_status,
                new_status=previous_status,
                changed=False,
            )

        # No executor handle — cannot query SLURM
        if not job.executor_handle:
            job.status = JobStatus.FAILED.value
            job.error = "No executor handle — cannot check status."
            job.completed_at = datetime.datetime.utcnow()
            await session.commit()
            return ReconcileResponse(
                id=job_id,
                previous_status=previous_status,
                new_status=job.status,
                changed=True,
                error=job.error,
            )

        # Query the real executor status
        executor = get_executor()
        actual = await executor.status(job.executor_handle)

        is_terminal = actual in (
            ExecJobStatus.COMPLETED,
            ExecJobStatus.FAILED,
            ExecJobStatus.CANCELLED,
        )

        error_msg: str | None = None
        if actual == ExecJobStatus.UNKNOWN:
            actual = ExecJobStatus.FAILED
            error_msg = "Job status unknown — SLURM may have purged the record."
        elif actual == ExecJobStatus.FAILED:
            error_msg = "Job failed (detected on manual reconcile)."

        new_status = actual.value
        changed = new_status != previous_status

        if changed:
            job.status = new_status
            if error_msg:
                job.error = error_msg
            if is_terminal or actual == ExecJobStatus.FAILED:
                job.completed_at = datetime.datetime.utcnow()
            await session.commit()

            # If the job reached a terminal state, cancel any running poller
            if is_terminal:
                poll_task = _poll_tasks.pop(job_id, None)
                if poll_task:
                    poll_task.cancel()
            else:
                # Still active — ensure a poller is running
                if job_id not in _poll_tasks:
                    # Find the project path for this job
                    from recovar.gui_v2.backend.api.project import (
                        _project_registry,
                    )
                    project_path = _project_registry.get(job.project_id)
                    if project_path:
                        task = asyncio.create_task(
                            _poll_job_status(
                                job_id, job.executor_handle, project_path
                            )
                        )
                        _poll_tasks[job_id] = task

        return ReconcileResponse(
            id=job_id,
            previous_status=previous_status,
            new_status=new_status,
            changed=changed,
            error=error_msg,
        )
    finally:
        await session.close()


class SbatchScriptResponse(BaseModel):
    script: str
    source: str  # "file" (read from submit.sh) or "preview" (rendered)


@router.get("/{job_id}/sbatch-script", response_model=SbatchScriptResponse)
async def get_sbatch_script(job_id: str) -> SbatchScriptResponse:
    """Return the sbatch script for a job.

    For completed/running SLURM jobs, reads the actual ``submit.sh`` file.
    For local jobs, returns the CLI command with a note.
    """
    job, session = await _get_job(job_id)
    await session.close()

    submit_sh = os.path.join(job.output_dir, "submit.sh")

    if os.path.isfile(submit_sh):
        try:
            with open(submit_sh) as f:
                content = f.read()
            return SbatchScriptResponse(script=content, source="file")
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Cannot read submit.sh: {exc}",
            )

    # No submit.sh — this was a local job
    if job.slurm_id is None:
        cmd_line = _format_cli_command(job)
        return SbatchScriptResponse(
            script=f"# This job ran locally (not via SLURM).\n"
                   f"# PID: {job.executor_handle or 'unknown'}\n\n"
                   f"{cmd_line}\n",
            source="preview",
        )

    # SLURM job but submit.sh was deleted
    raise HTTPException(
        status_code=404,
        detail="submit.sh not found in job output directory.",
    )


def _format_cli_command(job: Job) -> str:
    """Format a reconstructed CLI command from job params for display."""
    if not job.params:
        return f"# No parameters recorded for {job.type} job."
    parts = [f"recovar {job.type.lower()}"]
    for key, val in job.params.items():
        if key in ("outdir", "slurm_opts"):
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                parts.append(flag)
        else:
            parts.append(f"{flag} {val}")
    return " ".join(parts)
