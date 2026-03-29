"""FastAPI application factory for the recovar GUI v2 backend.

Usage::

    pixi run python -m recovar.gui_v2.backend.main --port 8080
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from recovar.gui_v2.backend.api.embeddings import router as embeddings_router
from recovar.gui_v2.backend.api.files import router as files_router
from recovar.gui_v2.backend.api.jobs import router as jobs_router
from recovar.gui_v2.backend.api.project import router as project_router
from recovar.gui_v2.backend.api.subsets import router as subsets_router
from recovar.gui_v2.backend.api.system import router as system_router
from recovar.gui_v2.backend.api.volumes import router as volumes_router
from recovar.gui_v2.backend.api.ws import router as ws_router
from recovar.gui_v2.backend.config import DEFAULT_HOST, DEFAULT_PORT
from recovar.gui_v2.backend.db import close_all

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("recovar GUI v2 backend starting")
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

    # Mount API routers
    app.include_router(project_router)
    app.include_router(jobs_router)
    app.include_router(files_router)
    app.include_router(volumes_router)
    app.include_router(embeddings_router)
    app.include_router(subsets_router)
    app.include_router(system_router)
    app.include_router(ws_router)

    # Serve prebuilt frontend assets if the static directory exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True))

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
