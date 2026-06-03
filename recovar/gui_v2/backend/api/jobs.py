"""Jobs REST API.

Endpoints:
    POST   /api/jobs                    — Submit a new job
    GET    /api/jobs/:id                — Get job detail
    POST   /api/jobs/:id/cancel         — Cancel a job
    DELETE /api/jobs/:id                — Delete job record
    GET    /api/jobs/:id/volumes        — List MRC volumes in job output
    GET    /api/jobs/:id/plots          — List diagnostic PNGs
    GET    /api/jobs/:id/chart-data     — Get interactive chart data for a plot
    POST   /api/jobs/validate           — Validate job params without submitting
    GET    /api/jobs/:id/suggested-next — Suggest next steps
"""

from __future__ import annotations

import asyncio
import datetime
import glob
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from recovar.gui_v2.backend.api.project import get_project_path
from recovar.gui_v2.backend.config import get_db_path, iso_utc
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
# Executor pool — both executors are available; jobs pick which to use.
# ---------------------------------------------------------------------------

_slurm_executor: SlurmExecutor | None = None
_local_executor: LocalExecutor | None = None


def _get_slurm_executor() -> SlurmExecutor:
    global _slurm_executor
    if _slurm_executor is None:
        _slurm_executor = SlurmExecutor()
        logger.info("SLURM executor initialized")
    return _slurm_executor


def _get_local_executor() -> LocalExecutor:
    global _local_executor
    if _local_executor is None:
        _local_executor = LocalExecutor()
        logger.info("Local executor initialized")
    return _local_executor


def get_executor(mode: str | None = None) -> Executor:
    """Return the executor for the given mode.

    mode can be "slurm", "local", or None (use server default from
    RECOVAR_EXECUTOR env var / auto-detection).
    """
    if mode == "slurm":
        return _get_slurm_executor()
    if mode == "local":
        return _get_local_executor()
    # Default: use server-wide setting
    if slurm_available():
        return _get_slurm_executor()
    return _get_local_executor()


def get_executor_for_job(job) -> Executor:
    """Return the correct executor for an existing job (for polling/cancel)."""
    if job.slurm_id:
        return _get_slurm_executor()
    return _get_local_executor()


# ---------------------------------------------------------------------------
# Background status poller
# ---------------------------------------------------------------------------

_poll_tasks: dict[str, asyncio.Task] = {}


async def _poll_job_status(job_id: str, handle: str, project_path: str, executor_mode: str | None = None, working_dir: str | None = None) -> None:
    """Background task that polls executor status until terminal."""
    executor = get_executor(executor_mode)
    while True:
        try:
            # Local executor needs working_dir to read exit code file
            if working_dir and hasattr(executor, "status_with_dir"):
                status = await executor.status_with_dir(handle, working_dir)
            else:
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
                        reason = await executor.failure_reason(handle)
                        job.error = reason or "Job failed (detected by status poller)."
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
    executor: str | None = None  # "slurm", "local", or None (server default)


class SubmitJobResponse(BaseModel):
    id: str
    type: str
    status: str
    created: str
    handle: str | None = None
    slurm_id: str | None = None
    warnings: list[str] = []


class JobDetailResponse(BaseModel):
    id: str
    project_id: str
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


class ChartDataResponse(BaseModel):
    chart_type: str
    traces: list[dict[str, Any]]  # Each trace is a Plotly-compatible dict
    layout: dict[str, Any] | None = None  # Optional layout overrides


class ValidateJobRequest(BaseModel):
    project_id: str
    type: str
    params: dict[str, Any]


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}


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


def _categorize_volume(name: str, rel_path: str = "") -> str:
    """Assign a display category to an MRC filename.

    *rel_path* is the path relative to the job output directory,
    used to detect subfolder structure (e.g. kmeans/, trajectories/).
    """
    lower = name.lower()
    rel_lower = rel_path.lower()

    # Local resolution files are shading overlays, not standalone volumes
    if "locres" in lower or "local_res" in lower or "local_resolution" in lower:
        return "locres"

    # Sampling maps are diagnostic, not standalone viewable volumes
    if lower == "sampling.mrc" or "sampling" in lower:
        return "sampling"

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

    # Analyze job subfolder detection
    if "kmeans" in rel_lower or "center" in lower:
        return "kmeans_center"
    if "trajectory" in rel_lower or "traj" in rel_lower or "traj" in lower:
        return "trajectory"

    if "state" in lower:
        return "reconstruction"
    if "density" in lower or "deconv" in lower:
        return "density"

    return "other"


def _write_coords_file(path: str, coords: list[float]) -> None:
    """Write a latent coordinate vector to a text file."""
    with open(path, "w") as f:
        f.write(" ".join(f"{c:.8f}" for c in coords) + "\n")


# ---------------------------------------------------------------------------
# Chart data helpers (Plotly interactive charts)
# ---------------------------------------------------------------------------


def _find_data_files(output_dir: str, chart_name: str) -> list[str]:
    """Find .pkl or .npy files matching a chart name pattern in the output dir."""
    matches: list[str] = []
    for ext in ("pkl", "npy"):
        pattern = os.path.join(output_dir, "**", f"*{chart_name}*.{ext}")
        matches.extend(glob.glob(pattern, recursive=True))
    return sorted(matches)


def _load_data_file(path: str) -> Any:
    """Load a .pkl or .npy file and return its contents."""
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=True)
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def _numpy_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to Python lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_list(v) for v in obj]
    return obj


def _build_fsc_response(data: Any) -> ChartDataResponse:
    """Build a ChartDataResponse for FSC data."""
    traces: list[dict[str, Any]] = []
    if isinstance(data, dict):
        fsc_vals = _numpy_to_list(data.get("fsc", data.get("FSC", [])))
        resolution = _numpy_to_list(data.get("resolution", data.get("Resolution", [])))
        if isinstance(fsc_vals, list) and len(fsc_vals) > 0:
            if isinstance(fsc_vals[0], list):
                for i, curve in enumerate(fsc_vals):
                    trace: dict[str, Any] = {"y": curve, "type": "scatter", "mode": "lines", "name": f"FSC {i}"}
                    if resolution:
                        trace["x"] = resolution
                    traces.append(trace)
            else:
                trace = {"y": fsc_vals, "type": "scatter", "mode": "lines", "name": "FSC"}
                if resolution:
                    trace["x"] = resolution
                traces.append(trace)
    elif isinstance(data, np.ndarray):
        arr = _numpy_to_list(data)
        if isinstance(arr, list):
            if isinstance(arr[0], list):
                for i, curve in enumerate(arr):
                    traces.append({"y": curve, "type": "scatter", "mode": "lines", "name": f"FSC {i}"})
            else:
                traces.append({"y": arr, "type": "scatter", "mode": "lines", "name": "FSC"})

    layout: dict[str, Any] = {"xaxis": {"title": "Spatial Frequency"}, "yaxis": {"title": "FSC", "range": [0, 1]}}
    return ChartDataResponse(chart_type="fsc", traces=traces, layout=layout)


def _build_eigenvalue_response(data: Any) -> ChartDataResponse:
    """Build a ChartDataResponse for eigenvalue spectrum data."""
    values = _numpy_to_list(data)
    if isinstance(values, dict):
        values = _numpy_to_list(values.get("eigenvalues", values.get("values", [])))
    if isinstance(values, list) and len(values) > 0:
        indices = list(range(1, len(values) + 1))
        traces: list[dict[str, Any]] = [{"x": indices, "y": values, "type": "bar", "name": "Eigenvalues"}]
    else:
        traces = []
    layout: dict[str, Any] = {"xaxis": {"title": "Component"}, "yaxis": {"title": "Eigenvalue"}}
    return ChartDataResponse(chart_type="eigenvalues", traces=traces, layout=layout)


def _build_histogram_response(data: Any) -> ChartDataResponse:
    """Build a ChartDataResponse for histogram data."""
    values = _numpy_to_list(data)
    if isinstance(values, dict):
        bins = values.get("bins", [])
        counts = values.get("counts", values.get("values", []))
        traces: list[dict[str, Any]] = [{"x": bins, "y": counts, "type": "bar", "name": "Histogram"}]
    elif isinstance(values, list):
        traces = [{"x": values, "type": "histogram", "name": "Histogram"}]
    else:
        traces = []
    return ChartDataResponse(chart_type="histogram", traces=traces)


_CHART_BUILDERS = {
    "fsc": _build_fsc_response,
    "eigenvalue": _build_eigenvalue_response,
    "eigenvalues": _build_eigenvalue_response,
    "histogram": _build_histogram_response,
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _read_available_zdims(result_dir: str) -> list[int] | None:
    """Read available zdims from pipeline output."""
    # Primary: read embeddings.pkl which stores zs as {zdim: array, ...}
    embeddings_path = os.path.join(result_dir, "model", "embeddings.pkl")
    if os.path.isfile(embeddings_path):
        try:
            with open(embeddings_path, "rb") as f:
                emb = pickle.load(f)
            zs = emb.get("zs", {})
            if isinstance(zs, dict):
                zdims = [int(k) for k in zs.keys() if isinstance(k, (int, float))]
                if zdims:
                    return sorted(zdims)
        except Exception:
            pass

    # Secondary: check params.pkl for zdims_computed
    params_path = os.path.join(result_dir, "model", "params.pkl")
    if os.path.isfile(params_path):
        try:
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            zdims = params.get("zdims_computed") or params.get("zdim")
            if zdims is not None:
                if isinstance(zdims, (list, tuple)):
                    return sorted(int(z) for z in zdims)
                return [int(zdims)]
        except Exception:
            pass

    # Fallback: scan for analysis directories
    output_dir = os.path.join(result_dir, "output")
    if os.path.isdir(output_dir):
        zdims = []
        for name in os.listdir(output_dir):
            if name.startswith("analysis_") and os.path.isdir(os.path.join(output_dir, name)):
                try:
                    z = int(name.split("_")[1])
                    zdims.append(z)
                except (IndexError, ValueError):
                    pass
        if zdims:
            return sorted(zdims)
    return None


def _check_mask_dims(mask_path: str, expected_box_size: int) -> str | None:
    """Check mask .mrc dimensions match particle box size. Returns error or None."""
    try:
        import mrcfile

        with mrcfile.open(mask_path, mode="r", header_only=True) as mrc:
            nx, ny, nz = (
                int(mrc.header.nx),
                int(mrc.header.ny),
                int(mrc.header.nz),
            )
            if nx != expected_box_size or ny != expected_box_size or nz != expected_box_size:
                return f"Mask box size ({nx}x{ny}x{nz}) does not match particle box size {expected_box_size}"
    except Exception as e:
        return f"Cannot read mask file: {e}"
    return None


_VALIDATION_TIMEOUT = 10  # seconds for filesystem operations


async def _validate_job_params(
    project_path: str,
    job_type: str,
    params: dict[str, Any],
) -> ValidationResult:
    """Run all validation checks for a job without submitting it."""
    from recovar.gui_v2.backend.api.files import _parse_star_sync

    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}

    if job_type == "Pipeline":
        particles = params.get("particles", "")
        if particles and not os.path.isfile(particles):
            errors.append(f"Particles file not found: {particles}")
        mask = params.get("mask", "from_halfmaps")
        if mask not in ("from_halfmaps", "sphere", "none", "") and not os.path.isfile(mask):
            errors.append(f"Mask file not found: {mask}")

        # Mask dimension validation
        if (
            particles
            and os.path.isfile(particles)
            and mask not in ("from_halfmaps", "sphere", "none", "")
            and os.path.isfile(mask)
        ):
            try:
                star_result = await asyncio.wait_for(
                    asyncio.to_thread(_parse_star_sync, particles),
                    timeout=_VALIDATION_TIMEOUT,
                )
                box_size = star_result.box_size
                if box_size is not None:
                    info["box_size"] = box_size
                    mask_err = await asyncio.wait_for(
                        asyncio.to_thread(_check_mask_dims, mask, box_size),
                        timeout=_VALIDATION_TIMEOUT,
                    )
                    if mask_err:
                        errors.append(mask_err)
                if star_result.n_particles is not None:
                    info["particle_count"] = star_result.n_particles
            except asyncio.TimeoutError:
                warnings.append("Mask validation timed out (filesystem may be slow).")
            except Exception as exc:
                warnings.append(f"Could not validate mask dimensions: {exc}")

    elif job_type in (
        "Analyze",
        "ComputeState",
        "ComputeTrajectory",
        "Density",
        "StableStates",
    ):
        result_dir = params.get("result_dir", "")
        if result_dir and not os.path.isdir(result_dir):
            errors.append(f"Result directory not found: {result_dir}")
        elif result_dir:
            metadata = os.path.join(result_dir, "model", "params.pkl")
            metadata_json = os.path.join(result_dir, "model", "metadata.json")
            if not os.path.isfile(metadata) and not os.path.isfile(metadata_json):
                errors.append(f"Not a valid pipeline output: {result_dir}")

        # zdim validation for Analyze jobs
        if job_type == "Analyze" and result_dir and os.path.isdir(result_dir):
            requested_zdim = params.get("zdim")
            if requested_zdim is not None:
                try:
                    available = await asyncio.wait_for(
                        asyncio.to_thread(_read_available_zdims, result_dir),
                        timeout=_VALIDATION_TIMEOUT,
                    )
                    if available is not None:
                        info["available_zdims"] = available
                        if int(requested_zdim) not in available:
                            errors.append(f"zdim={requested_zdim} not found in pipeline output. Available: {available}")
                except asyncio.TimeoutError:
                    warnings.append("zdim validation timed out (filesystem may be slow).")
                except Exception as exc:
                    warnings.append(f"Could not validate zdim: {exc}")

    # Check disk space
    try:
        usage = os.statvfs(project_path)
        free_gb = (usage.f_frsize * usage.f_bavail) / (1024**3)
        if free_gb < 50:
            warnings.append(f"Less than {free_gb:.0f} GB free on disk. Jobs may fail if space runs out.")
    except OSError:
        pass

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/validate", response_model=ValidationResult)
async def validate_job(req: ValidateJobRequest) -> ValidationResult:
    """Validate job parameters without submitting. Returns errors/warnings."""
    project_path = get_project_path(req.project_id)
    if project_path is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Normalize job type
    _type_aliases_local = {
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
    job_type = _type_aliases_local.get(req.type.lower().replace("-", "_"), req.type)

    return await _validate_job_params(project_path, job_type, req.params)


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
    existing = [d for d in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, d)) and d.startswith("job_")]
    next_num = max((int(d.split("_")[1]) for d in existing), default=0) + 1
    job_dir = os.path.join(type_dir, f"job_{next_num:04d}")
    os.makedirs(job_dir, exist_ok=True)

    # Set outdir in params
    params["outdir"] = job_dir

    # Validate key inputs before submission
    warnings: list[str] = []
    if job_type == "Pipeline":
        particles = params.get("particles", "")
        if particles and not os.path.isfile(particles):
            raise HTTPException(status_code=400, detail=f"Particles file not found: {particles}")
        mask = params.get("mask", "from_halfmaps")
        if mask not in ("from_halfmaps", "sphere", "none", "") and not os.path.isfile(mask):
            raise HTTPException(status_code=400, detail=f"Mask file not found: {mask}")
    elif job_type in ("Analyze", "ComputeState", "ComputeTrajectory", "Density", "StableStates"):
        result_dir = params.get("result_dir", "")
        if result_dir and not os.path.isdir(result_dir):
            raise HTTPException(status_code=400, detail=f"Result directory not found: {result_dir}")
        metadata = os.path.join(result_dir, "model", "params.pkl") if result_dir else ""
        if (
            result_dir
            and not os.path.isfile(metadata)
            and not os.path.isfile(os.path.join(result_dir, "model", "metadata.json"))
        ):
            raise HTTPException(status_code=400, detail=f"Not a valid pipeline output: {result_dir}")

    # Check disk space
    try:
        usage = os.statvfs(project_path)
        free_gb = (usage.f_frsize * usage.f_bavail) / (1024**3)
        if free_gb < 50:
            warnings.append(f"Less than {free_gb:.0f} GB free on disk. Jobs may fail if space runs out.")
    except OSError:
        pass

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

    # Submit to executor — per-job choice overrides server default
    chosen_mode = req.executor or params.pop("executor", None)
    executor = get_executor(chosen_mode)
    env = {
        "PYTHONNOUSERSITE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    # Resolve effective slurm opts: built-in defaults ← user-global toml ←
    # project-local recovar.toml ← per-job form override (form wins).
    from recovar.gui_v2.backend.services.project_config import (
        resolve_local_defaults,
        resolve_slurm_defaults,
    )

    merged_slurm_opts = resolve_slurm_defaults(project_dir=project_path)
    form_slurm_opts = params.get("slurm_opts") or {}
    if isinstance(form_slurm_opts, dict):
        merged_slurm_opts.update(form_slurm_opts)

    # Resolve local execution opts
    merged_local_opts = resolve_local_defaults(project_dir=project_path)
    form_local_opts = params.get("local_opts") or {}
    if isinstance(form_local_opts, dict):
        merged_local_opts.update(form_local_opts)

    try:
        handle = await executor.submit(
            job_id=job_id,
            command=command,
            env=env,
            working_dir=job_dir,
            slurm_opts=merged_slurm_opts,
            local_opts=merged_local_opts,
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
    poll_mode = "slurm" if isinstance(executor, SlurmExecutor) else "local"
    task = asyncio.create_task(_poll_job_status(job_id, handle, project_path, executor_mode=poll_mode, working_dir=job_dir))
    _poll_tasks[job_id] = task

    return SubmitJobResponse(
        id=job_id,
        type=job_type,
        status=JobStatus.RUNNING.value,
        created=iso_utc(job.created_at),
        handle=handle,
        slurm_id=handle if isinstance(executor, SlurmExecutor) else None,
        warnings=warnings,
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
            project_id=job.project_id,
            type=job.type,
            status=job.status,
            params=job.params,
            created=iso_utc(job.created_at),
            completed=iso_utc(job.completed_at),
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
        if job.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
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
        rel_dir = os.path.relpath(dirpath, output_dir)
        for fname in sorted(filenames):
            if not fname.endswith(".mrc"):
                continue
            full = os.path.join(dirpath, fname)
            rel_path = os.path.join(rel_dir, fname)
            category = _categorize_volume(fname, rel_path)
            # Skip diagnostic volumes (not standalone viewable)
            if category in ("locres", "sampling"):
                continue
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            # Use subfolder in display name for clarity
            display_name = fname if rel_dir == "." else os.path.join(rel_dir, fname)
            volumes.append(
                VolumeEntry(
                    name=display_name,
                    path=full,
                    category=category,
                    size_bytes=size,
                )
            )

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
        suggestions.append(
            SuggestedNext(
                type="Analyze",
                label="Analyze this pipeline output",
                prefilled_params={"result_dir": job.output_dir},
            )
        )
        suggestions.append(
            SuggestedNext(
                type="Density",
                label="Estimate conformational density",
                prefilled_params={"result_dir": job.output_dir},
            )
        )
    elif job.type == "Analyze":
        suggestions.append(
            SuggestedNext(
                type="ComputeState",
                label="Compute volume at a latent point",
                prefilled_params={"result_dir": (job.params or {}).get("result_dir", "")},
            )
        )
        suggestions.append(
            SuggestedNext(
                type="ComputeTrajectory",
                label="Compute trajectory between two points",
                prefilled_params={"result_dir": (job.params or {}).get("result_dir", "")},
            )
        )
    elif job.type == "Density":
        density_pkl = os.path.join(job.output_dir, "data", "deconv_density_knee.pkl")
        suggestions.append(
            SuggestedNext(
                type="StableStates",
                label="Find stable states from density",
                prefilled_params={"density": density_pkl},
            )
        )

    return suggestions


class LogResponse(BaseModel):
    log: str
    path: str | None = None
    error: str | None = None


@router.get("/{job_id}/logs")
async def get_job_logs(job_id: str) -> LogResponse:
    """Get log file content for a completed/failed job."""
    job, session = await _get_job(job_id)
    await session.close()

    # Try executor log path first
    executor = get_executor()
    log_path = None
    if job.executor_handle:
        log_path = await executor.log_path(job.executor_handle)
    # SLURM fallback: slurm-{id}.out
    if log_path is None and job.executor_handle and job.output_dir:
        candidate = Path(job.output_dir) / f"slurm-{job.executor_handle}.out"
        if candidate.exists():
            log_path = candidate
    # Generic fallback
    if log_path is None and job.output_dir:
        candidate = Path(job.output_dir) / "run.log"
        if candidate.exists():
            log_path = candidate

    if log_path and log_path.exists():
        try:
            text = log_path.read_text(errors="replace")
            return LogResponse(log=text, path=str(log_path))
        except OSError as e:
            return LogResponse(log="", error=f"Failed to read log: {e}")

    return LogResponse(log="", error="Log file not found")


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
                        task = asyncio.create_task(_poll_job_status(job_id, job.executor_handle, project_path))
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


class SbatchPreviewRequest(BaseModel):
    command: list[str]
    env_vars: dict[str, str] = {}
    output_path: str = "/tmp/slurm-%j.out"
    job_name: str = "recovar-preview"
    slurm_opts: dict[str, Any] = {}


class SbatchPreviewResponse(BaseModel):
    script: str
    warnings: list[str] = []


@router.post("/preview-sbatch", response_model=SbatchPreviewResponse)
async def preview_sbatch_script(req: SbatchPreviewRequest) -> SbatchPreviewResponse:
    """Render the sbatch script that *would* be submitted, without writing
    or submitting anything. Used by the job-form preview pane so users can
    see — and debug — their template/SLURM-opts before clicking Submit."""
    import shlex

    from recovar.gui_v2.backend.services.executor import _render_sbatch_script

    warnings: list[str] = []
    opts = dict(req.slurm_opts or {})

    # Surface helpful warnings the renderer can't (it just omits silently).
    if not opts.get("partition"):
        warnings.append(
            "Partition is blank — `#SBATCH --partition` will be omitted; the cluster's default partition will be used."
        )
    if not opts.get("account"):
        warnings.append(
            "Account is blank — `#SBATCH --account` will be omitted; the cluster's default account will be used."
        )
    if opts.get("gpus", 1) == 0:
        warnings.append("gpus=0 — `#SBATCH --gres=gpu:N` will be omitted (CPU-only job).")

    try:
        script = _render_sbatch_script(
            job_name=req.job_name,
            command=shlex.join(req.command) if req.command else "",
            env_vars=req.env_vars,
            output_path=req.output_path,
            **opts,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to render sbatch script: {exc}",
        )

    return SbatchPreviewResponse(script=script, warnings=warnings)


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
        # Omit unset optionals so the command is runnable as shown. Skip only
        # None and empty/blank strings — NOT meaningful falsy values: False
        # booleans are emitted as bare flags below, and numeric 0 / -1 (e.g.
        # `--n-images -1`) must be preserved.
        if val is None or (isinstance(val, str) and val.strip() == ""):
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                parts.append(flag)
        else:
            parts.append(f"{flag} {val}")
    return " ".join(parts)


@router.get("/{job_id}/chart-data", response_model=ChartDataResponse)
async def get_chart_data(job_id: str, name: str) -> ChartDataResponse:
    """Return structured chart data suitable for Plotly rendering.

    Query parameter *name* should be one of: fsc, eigenvalue(s), histogram.
    The endpoint searches the job output directory for matching data files.
    """
    job, session = await _get_job(job_id)
    await session.close()

    output_dir = job.output_dir
    if not output_dir or not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output directory not found")

    # Normalize the chart name for matching
    chart_key = name.lower().rstrip("s")  # eigenvalues -> eigenvalue
    if chart_key not in ("fsc", "eigenvalue", "histogram"):
        raise HTTPException(status_code=400, detail=f"Unknown chart name: {name}")

    # Search for data files
    files = await asyncio.to_thread(_find_data_files, output_dir, chart_key)
    # Also try the plural form for eigenvalues
    if not files and chart_key == "eigenvalue":
        files = await asyncio.to_thread(_find_data_files, output_dir, "eigenvalues")
    if not files:
        raise HTTPException(status_code=404, detail=f"No data file found for chart: {name}")

    # Load the first matching file
    try:
        data = await asyncio.to_thread(_load_data_file, files[0])
    except Exception as exc:
        logger.warning("Failed to load chart data from %s: %s", files[0], exc)
        raise HTTPException(status_code=500, detail=f"Failed to load data: {exc}")

    builder = _CHART_BUILDERS.get(chart_key, _CHART_BUILDERS.get(name.lower()))
    if builder is None:
        raise HTTPException(status_code=400, detail=f"Unknown chart type: {name}")

    return builder(data)
