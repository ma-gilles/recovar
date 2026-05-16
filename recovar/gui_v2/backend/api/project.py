"""Project REST API.

Endpoints:
    GET    /api/projects            — List known (registered) projects
    POST   /api/projects            — Create a new project
    GET    /api/projects/:id        — Get project with jobs
    POST   /api/projects/:id/scan   — Scan/import existing pipeline outputs
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from recovar.gui_v2.backend.config import get_db_path, iso_utc
from recovar.gui_v2.backend.db import get_session, init_db, with_write_retry
from recovar.gui_v2.backend.models.job import Job, JobStatus
from recovar.gui_v2.backend.models.project import Project
from recovar.gui_v2.backend.services.scanner import (
    scan_arbitrary_directory,
    scan_project_directory,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class CreateProjectRequest(BaseModel):
    path: str
    name: str

    @field_validator("path")
    @classmethod
    def path_must_be_absolute(cls, v: str) -> str:
        # Expand "~" so users can type e.g. ~/projects/foo.
        v = os.path.expanduser(v.strip())
        if not os.path.isabs(v):
            raise ValueError("path must be absolute (or start with ~)")
        return v


class ProjectResponse(BaseModel):
    id: str
    path: str
    name: str
    created: str  # ISO 8601


class ProjectListItem(BaseModel):
    id: str
    path: str


class JobSummary(BaseModel):
    id: str
    type: str
    status: str
    output_dir: str
    created: str
    completed: str | None = None
    slurm_id: str | None = None
    error: str | None = None


class ProjectDetailResponse(BaseModel):
    id: str
    path: str
    name: str
    created: str
    jobs: list[JobSummary]
    disk_usage_bytes: int
    disk_usage_total: int


class ScanRequest(BaseModel):
    scan_path: str

    @field_validator("scan_path")
    @classmethod
    def scan_path_must_be_absolute(cls, v: str) -> str:
        # Expand "~" so users can type e.g. ~/mytigress/. After expansion it
        # must be absolute (a bare relative path is still rejected).
        v = os.path.expanduser(v.strip())
        if not os.path.isabs(v):
            raise ValueError("scan_path must be absolute (or start with ~)")
        return v


class ImportedJobResponse(BaseModel):
    id: str
    type: str
    status: str
    output_dir: str
    legacy: bool = False


class ScanResponse(BaseModel):
    imported: list[ImportedJobResponse]
    hint: str | None = None


# ---------------------------------------------------------------------------
# Dependency: async session from the project's DB
# ---------------------------------------------------------------------------


async def _get_project_session(project_path: str):
    """Yield an async session for a project's database."""
    db_path = get_db_path(project_path)
    async for session in get_session(db_path):
        yield session


# ---------------------------------------------------------------------------
# Module-level project registry (in-memory, persisted to disk)
# ---------------------------------------------------------------------------

# Maps project_id -> project_path for quick lookups.  Populated by
# create/open operations and rehydrated from a persisted index on startup so
# that job/subset lookups and the startup reconcile survive a server restart.
_project_registry: dict[str, str] = {}


def _index_path() -> Path:
    """Location of the persisted known-projects index.

    Honors ``XDG_CONFIG_HOME``; falls back to ``~/.config/recovar``.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return Path(base) / "recovar" / "projects.json"


def _load_persisted_index() -> dict[str, str]:
    """Read the persisted project_id -> path map, or {} on any failure."""
    path = _index_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Keep only well-formed string entries.
    return {
        str(k): str(v)
        for k, v in data.items()
        if isinstance(k, str) and isinstance(v, str)
    }


def _save_persisted_index() -> None:
    """Atomically write the in-memory registry to the persisted index.

    Failures are logged but never raised — persistence is best-effort and must
    not break project create/open.
    """
    path = _index_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(_project_registry, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except OSError:
        logger.warning("Could not persist project index to %s", path, exc_info=True)


def load_persisted_projects() -> int:
    """Rehydrate ``_project_registry`` from the persisted index.

    Mutates the existing dict in place (consumers import it by reference).
    Entries whose path or project DB no longer exists are dropped.  Returns
    the number of projects loaded.  Safe to call multiple times.
    """
    persisted = _load_persisted_index()
    loaded = 0
    changed = False
    for project_id, project_path in persisted.items():
        if not get_db_path(project_path).exists():
            changed = True  # stale entry — prune it on next save
            continue
        if _project_registry.get(project_id) != project_path:
            _project_registry[project_id] = project_path
            changed = True
        loaded += 1
    if changed:
        _save_persisted_index()
    return loaded


def _register_project(project_id: str, project_path: str) -> None:
    if _project_registry.get(project_id) != project_path:
        _project_registry[project_id] = project_path
        _save_persisted_index()


def get_project_path(project_id: str) -> str | None:
    path = _project_registry.get(project_id)
    if path is not None:
        return path
    # Not in memory (e.g. registered by another process, or registry not yet
    # hydrated this run): consult the persisted index lazily.
    persisted = _load_persisted_index()
    path = persisted.get(project_id)
    if path is not None:
        _project_registry[project_id] = path
    return path


# Hydrate the registry from disk at import so registry-iterating consumers
# (job/subset lookups) and the startup reconcile work after a server restart,
# even before any project is explicitly opened this run.
try:
    load_persisted_projects()
except Exception:  # pragma: no cover - never block import on a bad index
    logger.warning("Failed to load persisted project index", exc_info=True)


async def _load_project_by_id(project_id: str) -> tuple[Project, AsyncSession]:
    """Look up a project by ID and return (project, session).

    Raises HTTPException(404) if not found.
    """
    project_path = get_project_path(project_id)
    if project_path is None:
        raise HTTPException(status_code=404, detail="Project not found")

    db_path = get_db_path(project_path)
    session_factory = await init_db(db_path)
    session = session_factory()

    stmt = select(Project).where(Project.id == project_id)
    result = await session.execute(stmt)
    project = result.scalar_one_or_none()
    if project is None:
        await session.close()
        raise HTTPException(status_code=404, detail="Project not found in database")
    return project, session


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ProjectListItem])
async def list_projects() -> list[ProjectListItem]:
    """List known projects (id + path) from the persisted registry.

    Lets the frontend rediscover/re-register projects after a server restart.
    Entries whose project DB no longer exists are pruned.
    """
    load_persisted_projects()
    return [
        ProjectListItem(id=pid, path=ppath)
        for pid, ppath in sorted(_project_registry.items(), key=lambda kv: kv[1])
    ]


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(req: CreateProjectRequest) -> ProjectResponse:
    """Create a new project at the given path.

    Creates the directory (if needed), initializes ``recovar_project.db``,
    and registers the project.  If a ``project.json`` already exists (CLI
    project), the GUI index is created alongside it.
    """
    project_path = os.path.abspath(req.path)

    # Create directory if it doesn't exist
    os.makedirs(project_path, exist_ok=True)

    # Initialize the DB
    db_path = get_db_path(project_path)
    session_factory = await init_db(db_path)

    async with session_factory() as session:
        # Check if a project already exists for this path
        stmt = select(Project).where(Project.path == project_path)
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            _register_project(existing.id, project_path)
            return ProjectResponse(
                id=existing.id,
                path=existing.path,
                name=existing.name,
                created=iso_utc(existing.created_at),
            )

        # Create new project record
        project = Project(name=req.name, path=project_path)
        session.add(project)
        await session.commit()
        await session.refresh(project)

        _register_project(project.id, project_path)

        return ProjectResponse(
            id=project.id,
            path=project.path,
            name=project.name,
            created=iso_utc(project.created_at),
        )


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(project_id: str) -> ProjectDetailResponse:
    """Get a project with its jobs and disk usage."""
    project, session = await _load_project_by_id(project_id)

    try:
        # Load jobs eagerly
        stmt = (
            select(Project)
            .where(Project.id == project_id)
            .options(selectinload(Project.jobs))
        )
        result = await session.execute(stmt)
        project = result.scalar_one()

        # Compute disk usage
        disk_usage_bytes = 0
        disk_usage_total = 0
        try:
            usage = shutil.disk_usage(project.path)
            disk_usage_bytes = usage.used
            disk_usage_total = usage.total
        except OSError:
            pass

        jobs = [
            JobSummary(
                id=j.id,
                type=j.type,
                status=j.status,
                output_dir=j.output_dir,
                created=iso_utc(j.created_at),
                completed=iso_utc(j.completed_at),
                slurm_id=j.slurm_id,
                error=j.error,
            )
            for j in project.jobs
        ]

        return ProjectDetailResponse(
            id=project.id,
            path=project.path,
            name=project.name,
            created=iso_utc(project.created_at),
            jobs=jobs,
            disk_usage_bytes=disk_usage_bytes,
            disk_usage_total=disk_usage_total,
        )
    finally:
        await session.close()


@router.post("/{project_id}/scan", response_model=ScanResponse)
async def scan_project(project_id: str, req: ScanRequest) -> ScanResponse:
    """Scan a directory for existing pipeline outputs and import them.

    If ``scan_path`` matches the project directory, uses project-structure
    scanning.  Otherwise, scans the path as an arbitrary directory.
    """
    project, session = await _load_project_by_id(project_id)

    try:
        scan_path = os.path.abspath(os.path.expanduser(req.scan_path))
        if not os.path.isdir(scan_path):
            raise HTTPException(
                status_code=400,
                detail=f"Scan path does not exist: {scan_path}",
            )

        # Choose scanning strategy
        if os.path.samefile(scan_path, project.path):
            scanned_jobs = scan_project_directory(scan_path)
        else:
            scanned_jobs = scan_arbitrary_directory(scan_path)

        if not scanned_jobs:
            # If the scan found nothing, check if the scanned directory
            # itself looks like a pipeline output.  Users commonly point
            # to the pipeline_output directory rather than its parent.
            hint: str | None = None
            looks_like_output = (
                os.path.isfile(os.path.join(scan_path, "model", "params.pkl"))
                or os.path.isfile(
                    os.path.join(scan_path, "model", "metadata.json")
                )
            )
            if looks_like_output:
                parent = os.path.dirname(scan_path)
                hint = (
                    f"This directory itself looks like a pipeline output. "
                    f"Try scanning its parent directory instead: {parent}"
                )
            return ScanResponse(imported=[], hint=hint)

        # Check which output_dirs are already in the DB
        existing_dirs_stmt = select(Job.output_dir).where(
            Job.project_id == project_id
        )
        result = await session.execute(existing_dirs_stmt)
        existing_dirs = {row[0] for row in result.all()}

        imported: list[ImportedJobResponse] = []

        for scanned in scanned_jobs:
            if scanned.output_dir in existing_dirs:
                continue  # Already imported

            job = Job(
                project_id=project_id,
                type=scanned.type,
                status=scanned.status,
                params=scanned.params,
                output_dir=scanned.output_dir,
                created_at=scanned.created_at or datetime.datetime.utcnow(),
                completed_at=scanned.completed_at,
                created_by="scan",
            )
            if scanned.legacy:
                job.params = {**(job.params or {}), "legacy_import": True}

            session.add(job)
            await session.flush()  # Get the ID

            imported.append(
                ImportedJobResponse(
                    id=job.id,
                    type=job.type,
                    status=job.status,
                    output_dir=job.output_dir,
                    legacy=scanned.legacy,
                )
            )

        await session.commit()

        logger.info(
            "Imported %d jobs into project %s from %s",
            len(imported),
            project_id,
            scan_path,
        )
        return ScanResponse(imported=imported)
    finally:
        await session.close()
