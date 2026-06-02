"""FastAPI application factory for the recovar GUI v2 backend.

Usage::

    pixi run python -m recovar.gui_v2.backend.main --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from recovar.gui_v2.backend.api.embeddings import router as embeddings_router
from recovar.gui_v2.backend.api.files import configure_allowed_roots
from recovar.gui_v2.backend.api.files import router as files_router
from recovar.gui_v2.backend.api.jobs import router as jobs_router
from recovar.gui_v2.backend.api.project import router as project_router
from recovar.gui_v2.backend.api.settings import router as settings_router
from recovar.gui_v2.backend.api.subsets import router as subsets_router
from recovar.gui_v2.backend.api.system import router as system_router
from recovar.gui_v2.backend.api.volumes import router as volumes_router
from recovar.gui_v2.backend.api.ws import router as ws_router
from recovar.gui_v2.backend.config import DEFAULT_HOST, DEFAULT_PORT
from recovar.gui_v2.backend.db import close_all, init_db
from recovar.gui_v2.backend.models.job import Job, JobStatus
from recovar.gui_v2.backend.services.executor import (
    reconcile_jobs,
)

logger = logging.getLogger(__name__)


async def _reconcile_on_startup() -> None:
    """Scan all known project databases for in-flight jobs and reconcile.

    This runs once at server startup to catch jobs that completed (or failed)
    while the GUI server was down.  For each project DB found, it queries
    jobs with status ``running`` or ``queued``, checks their actual SLURM/
    local status, updates the DB, and restarts background pollers for any
    that are still active.
    """
    from recovar.gui_v2.backend.api.jobs import (
        _poll_job_status,
        _poll_tasks,
    )
    from recovar.gui_v2.backend.api.project import _project_registry
    from recovar.gui_v2.backend.config import get_db_path

    # Collect all project paths from the registry.
    # The registry may be empty on first start; in that case,
    # nothing to reconcile (projects are loaded on Open/Create).
    project_paths = list(_project_registry.values())
    if not project_paths:
        logger.info("Startup reconcile: no projects registered yet, skipping")
        return

    total_reconciled = 0
    total_restarted = 0

    for project_path in project_paths:
        db_path = get_db_path(project_path)
        if not db_path.exists():
            continue

        try:
            session_factory = await init_db(db_path)
        except Exception:
            logger.exception("Startup reconcile: failed to open DB at %s", db_path)
            continue

        async with session_factory() as session:
            # Find all non-terminal jobs
            stmt = select(Job).where(Job.status.in_([JobStatus.RUNNING.value, JobStatus.QUEUED.value]))
            result = await session.execute(stmt)
            inflight = result.scalars().all()

            if not inflight:
                continue

            logger.info(
                "Startup reconcile: %d in-flight jobs in %s",
                len(inflight),
                project_path,
            )

            # Build the list for reconcile_jobs
            inflight_dicts = [
                {
                    "id": j.id,
                    "handle": j.executor_handle,
                    "db_status": j.status,
                    "working_dir": j.output_dir,
                }
                for j in inflight
            ]

            # Reconcile jobs grouped by executor type
            slurm_jobs = [d for d, j in zip(inflight_dicts, inflight) if j.slurm_id]
            local_jobs = [d for d, j in zip(inflight_dicts, inflight) if not j.slurm_id]
            updates = []
            if slurm_jobs:
                from recovar.gui_v2.backend.api.jobs import _get_slurm_executor

                updates.extend(await reconcile_jobs(_get_slurm_executor(), slurm_jobs))
            if local_jobs:
                from recovar.gui_v2.backend.api.jobs import _get_local_executor

                updates.extend(await reconcile_jobs(_get_local_executor(), local_jobs))

            # Apply updates to DB
            for upd in updates:
                stmt2 = select(Job).where(Job.id == upd["id"])
                result2 = await session.execute(stmt2)
                job = result2.scalar_one_or_none()
                if not job:
                    continue
                job.status = upd["new_status"]
                if upd.get("error"):
                    job.error = upd["error"]
                if upd["new_status"] in ("completed", "failed", "cancelled"):
                    job.completed_at = datetime.datetime.utcnow()
                total_reconciled += 1

            await session.commit()

            # Restart pollers for jobs that are still running/queued
            for j in inflight:
                if j.executor_handle and j.status in (
                    JobStatus.RUNNING.value,
                    JobStatus.QUEUED.value,
                ):
                    # Check if this job was updated to terminal
                    was_updated = any(u["id"] == j.id for u in updates)
                    if not was_updated and j.id not in _poll_tasks:
                        poll_mode = "slurm" if j.slurm_id else "local"
                        task = asyncio.create_task(
                            _poll_job_status(
                                j.id,
                                j.executor_handle,
                                project_path,
                                executor_mode=poll_mode,
                                working_dir=j.output_dir,
                            )
                        )
                        _poll_tasks[j.id] = task
                        total_restarted += 1

    logger.info(
        "Startup reconcile complete: %d jobs updated, %d pollers restarted",
        total_reconciled,
        total_restarted,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("recovar GUI v2 backend starting")

    # Configure file browser allowed roots.
    # Default: user home, /scratch, /tmp, and common HPC paths.
    import os

    default_roots = [
        os.path.expanduser("~"),
        "/tmp",
    ]
    # Add common HPC scratch paths if they exist
    for candidate in ["/scratch", "/gpfs", "/projects", "/data"]:
        if os.path.isdir(candidate):
            default_roots.append(candidate)
    configure_allowed_roots(default_roots)
    logger.info("File browser allowed roots: %s", default_roots)

    # Reconcile any in-flight jobs that completed while the server was down.
    try:
        await _reconcile_on_startup()
    except Exception:
        logger.exception("Startup reconcile failed (non-fatal)")

    yield
    logger.info("Shutting down — closing database connections")
    await close_all()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="recovar GUI v2",
        description="Web interface for recovar cryo-EM heterogeneity analysis",
        lifespan=lifespan,
    )

    # CORS for frontend dev server (Vite on port 5173)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Any unhandled exception becomes a structured error instead of an opaque
    # 500 with a bare traceback: the full traceback is logged server-side and
    # the client gets a legible {error, detail, hint} it can show the user.
    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": str(exc),
                "hint": "See the GUI server log for the full traceback.",
            },
        )

    # Mount API routers
    app.include_router(project_router)
    app.include_router(jobs_router)
    app.include_router(files_router)
    app.include_router(volumes_router)
    app.include_router(embeddings_router)
    app.include_router(subsets_router)
    app.include_router(settings_router)
    app.include_router(system_router)
    app.include_router(ws_router)

    # Serve prebuilt frontend assets if the static directory exists.
    # Static assets first, then a catch-all for SPA client-side routing.
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

        from fastapi.responses import FileResponse

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            """Serve index.html for all non-API paths (SPA catch-all)."""
            # API paths that weren't matched by any router should 404,
            # not serve the SPA.
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            # Try to serve an actual static file first
            file_path = static_dir / full_path
            if full_path and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(static_dir / "index.html")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="recovar GUI v2 server")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    print(
        f"\n  recovar GUI v2\n"
        f"  Access the GUI at: http://localhost:{args.port}\n"
        f"  SSH tunnel: ssh -L {args.port}:localhost:{args.port} user@cluster\n",
        file=sys.stderr,
    )

    uvicorn.run(
        "recovar.gui_v2.backend.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
