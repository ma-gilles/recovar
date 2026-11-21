"""File browser and validation API.

Endpoints:
    GET    /api/files/roots           — Allowed root directories
    GET    /api/files/browse          — List directory contents
    POST   /api/files/validate-star   — Validate a .star file
    POST   /api/files/validate-mrc    — Validate a .mrc file
"""

from __future__ import annotations

import asyncio
import logging
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])

# Validation timeout (seconds) for filesystem operations on loaded GPFS.
_VALIDATION_TIMEOUT = 10

# Allowlist of directory roots the file browser may navigate. Built at startup
# by ``default_allowed_roots`` (home, cwd, temp, every shared filesystem mounted
# on this machine, plus anything the user configured); project directories are
# added as they are opened.
_allowed_roots: list[str] = []

ALLOWED_ROOTS_ENV = "RECOVAR_GUI_ALLOWED_ROOTS"

_ADD_ROOT_HINT = (
    "To allow another directory, restart with `recovar gui --allow-root /path`, "
    f"set {ALLOWED_ROOTS_ENV}=/path1{os.pathsep}/path2, or add "
    "`allowed_roots = [\"/path\"]` under `[gui]` in ~/.config/recovar/config.toml."
)


def _canonical(path: str) -> str:
    # Resolve symlinks so roots compare canonically with the resolved request
    # paths in _check_path_allowed (e.g. HPC homes that symlink into project
    # storage).
    return str(Path(os.path.expanduser(path)).resolve())


def default_allowed_roots() -> list[str]:
    """Roots allowed without any configuration, measured from this machine."""
    import tempfile

    from recovar.gui_v2.backend.services.project_config import configured_allowed_roots
    from recovar.gui_v2.backend.services.site import shared_filesystem_roots

    roots = [os.path.expanduser("~"), os.getcwd(), "/tmp", tempfile.gettempdir()]
    roots += shared_filesystem_roots()
    roots += configured_allowed_roots()
    roots += [r for r in os.environ.get(ALLOWED_ROOTS_ENV, "").split(os.pathsep) if r]
    return roots


def configure_allowed_roots(roots: list[str]) -> None:
    """Set the allowed root directories for the file browser."""
    global _allowed_roots
    _allowed_roots = []
    for root in roots:
        add_allowed_root(root)


def add_allowed_root(root: str) -> None:
    """Add an allowed root directory (no-op if already covered)."""
    resolved = _canonical(root)
    if not _is_under_allowed_root(resolved):
        _allowed_roots.append(resolved)


def allowed_roots() -> list[str]:
    return list(_allowed_roots)


def _is_under_allowed_root(resolved: str) -> bool:
    # Compare on path boundaries, not raw string prefixes, so that a sibling
    # directory ("/tmpsecret" when root is "/tmp") cannot escape.
    return any(resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep) for root in _allowed_roots)


def _check_path_allowed(path: str) -> None:
    """Raise 403 if *path* is outside all allowed roots."""
    resolved = _canonical(path)
    if _is_under_allowed_root(resolved):
        return
    raise HTTPException(
        status_code=403,
        detail=(
            f"Access denied: {resolved} is outside the allowed directories "
            f"({', '.join(_allowed_roots)}). {_ADD_ROOT_HINT}"
        ),
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class FileEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int
    modified: str  # ISO 8601
    type: str  # "directory", "star", "mrc", "mrcs", "cs", "other"


class ValidateStarResponse(BaseModel):
    valid: bool | None = None
    n_particles: int | None = None
    box_size: int | None = None
    columns: list[str] | None = None
    error: str | None = None


class ValidateMrcResponse(BaseModel):
    valid: bool | None = None
    shape: list[int] | None = None
    voxel_size: float | None = None
    error: str | None = None


class ValidateRequest(BaseModel):
    path: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_type(name: str, is_dir: bool) -> str:
    if is_dir:
        return "directory"
    ext = os.path.splitext(name)[1].lower()
    return {
        ".star": "star",
        ".mrc": "mrc",
        ".mrcs": "mrcs",
        ".cs": "cs",
        ".pkl": "pkl",
        ".npy": "npy",
        ".png": "png",
        ".pdf": "pdf",
        ".txt": "txt",
    }.get(ext, "other")


def _gateway_entries(path: str) -> list[FileEntry]:
    """Entries for a directory that is *above* the allowed roots.

    Such a directory is not listed for real. It shows only the sub-directories
    that lead to an allowed root, so the path breadcrumb ("/" > "scratch" >
    "gpfs") can be walked down to where the data is instead of dead-ending in
    a 403. Empty when *path* is itself allowed or leads nowhere allowed.
    """
    resolved = _canonical(path)
    if _is_under_allowed_root(resolved):
        return []
    prefix = resolved.rstrip(os.sep) + os.sep
    names = sorted({root[len(prefix) :].split(os.sep)[0] for root in _allowed_roots if root.startswith(prefix)})
    return [
        FileEntry(name=name, path=prefix + name, is_dir=True, size=0, modified="", type="directory") for name in names
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/roots", response_model=list[str])
async def list_roots() -> list[str]:
    """Directories the file browser may navigate (for quick-access links)."""
    return allowed_roots()


@router.get("/serve")
async def serve_file(path: str) -> FileResponse:
    """Serve a single file with the correct MIME type."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    _check_path_allowed(abs_path)

    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found: {abs_path}")

    return FileResponse(abs_path)


@router.get("/browse", response_model=list[FileEntry])
async def browse(path: str) -> list[FileEntry]:
    """List contents of a directory."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    gateway = _gateway_entries(abs_path)
    if gateway:
        return gateway
    _check_path_allowed(abs_path)

    if not os.path.isdir(abs_path):
        raise HTTPException(status_code=400, detail=f"Not a directory: {abs_path}")

    entries: list[FileEntry] = []
    try:
        for name in sorted(os.listdir(abs_path)):
            full = os.path.join(abs_path, name)
            try:
                st = os.stat(full)
                is_dir = stat.S_ISDIR(st.st_mode)
                entries.append(FileEntry(
                    name=name,
                    path=full,
                    is_dir=is_dir,
                    size=st.st_size if not is_dir else 0,
                    modified=datetime.fromtimestamp(st.st_mtime).isoformat(),
                    type=_file_type(name, is_dir),
                ))
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"You do not have permission to read {abs_path}")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot list directory: {exc}")

    return entries


@router.post("/validate-star", response_model=ValidateStarResponse)
async def validate_star(req: ValidateRequest) -> ValidateStarResponse:
    """Validate a .star file: parse header, count particles, get box size."""
    _check_path_allowed(req.path)

    if not os.path.isfile(req.path):
        return ValidateStarResponse(valid=False, error=f"File not found: {req.path}")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_parse_star_sync, req.path),
            timeout=_VALIDATION_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        return ValidateStarResponse(
            valid=None,
            error="Validation timed out. The filesystem may be under heavy load. Try again.",
        )
    except Exception as exc:
        return ValidateStarResponse(valid=False, error=f"Cannot parse: {exc}")


@router.post("/validate-mrc", response_model=ValidateMrcResponse)
async def validate_mrc(req: ValidateRequest) -> ValidateMrcResponse:
    """Validate a .mrc file: read header for shape and voxel size."""
    _check_path_allowed(req.path)

    if not os.path.isfile(req.path):
        return ValidateMrcResponse(valid=False, error=f"File not found: {req.path}")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_parse_mrc_sync, req.path),
            timeout=_VALIDATION_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        return ValidateMrcResponse(
            valid=None,
            error="Validation timed out. The filesystem may be under heavy load. Try again.",
        )
    except Exception as exc:
        return ValidateMrcResponse(valid=False, error=f"Cannot parse: {exc}")


# ---------------------------------------------------------------------------
# Sync parsing helpers (run in thread pool)
# ---------------------------------------------------------------------------


def _parse_star_sync(path: str) -> ValidateStarResponse:
    """Parse a .star file and return validation results.

    Box size is derived exactly the way the pipeline derives it (via the
    canonical ``StarFile`` reader): the optics-table ``_rlnImageSize`` for
    RELION 3.1 files, otherwise the MRC header of the first referenced
    particle stack. This makes the preview match what RECOVAR will actually
    load. ``_rlnCoordinateX`` is the micrograph pick coordinate, NOT the box
    size, and must never be used for this.
    """
    try:
        from recovar.data_io.starfile import StarFile

        sf = StarFile(path)
        n_particles = len(sf)

        box_size = None
        resolution = sf.resolution
        if resolution is not None and len(resolution) > 0:
            box_size = int(resolution[0])

        # Strip the single leading "_" so column names follow the RELION
        # convention API consumers expect (e.g. "rlnImageName").
        columns = [c[1:] if c.startswith("_") else c for c in map(str, sf.df.columns)]

        return ValidateStarResponse(
            valid=True,
            n_particles=n_particles,
            box_size=box_size,
            columns=columns[:50],  # Cap at 50 columns
        )
    except Exception as exc:
        return ValidateStarResponse(valid=False, error=str(exc))


def _parse_mrc_sync(path: str) -> ValidateMrcResponse:
    """Parse an MRC file header."""
    try:
        import mrcfile
        with mrcfile.open(path, mode="r", header_only=True) as mrc:
            shape = [int(mrc.header.nz), int(mrc.header.ny), int(mrc.header.nx)]
            voxel_size = float(mrc.voxel_size.x)
            return ValidateMrcResponse(valid=True, shape=shape, voxel_size=voxel_size)
    except Exception as exc:
        return ValidateMrcResponse(valid=False, error=str(exc))
