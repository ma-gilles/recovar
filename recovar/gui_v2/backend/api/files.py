"""File browser and validation API.

Endpoints:
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

# Allowlist of directory roots the file browser may navigate.
# The project directory is always implicitly allowed.
# Additional roots can be configured in settings.toml.
_allowed_roots: list[str] = []


def configure_allowed_roots(roots: list[str]) -> None:
    """Set the allowed root directories for the file browser."""
    global _allowed_roots
    _allowed_roots = [os.path.abspath(r) for r in roots]


def add_allowed_root(root: str) -> None:
    """Add an allowed root directory."""
    _allowed_roots.append(os.path.abspath(root))


def _check_path_allowed(path: str) -> None:
    """Raise 403 if *path* is outside all allowed roots."""
    resolved = str(Path(os.path.expanduser(path)).resolve())
    for root in _allowed_roots:
        if resolved.startswith(root):
            return
    raise HTTPException(
        status_code=403,
        detail="Access denied: path is outside allowed directories.",
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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
    """Parse a .star file and return validation results."""
    try:
        import starfile
        data = starfile.read(path)

        # starfile returns a DataFrame or dict of DataFrames
        if isinstance(data, dict):
            # Multi-block: look for particles block
            for key in ("particles", "data_particles", ""):
                if key in data:
                    data = data[key]
                    break
            else:
                # Use the largest DataFrame
                data = max(data.values(), key=len)

        n_particles = len(data)
        columns = list(data.columns)

        # Try to get box size from image dimensions
        box_size = None
        for col in ("rlnImageSize", "rlnCoordinateX"):
            if col in data.columns:
                try:
                    box_size = int(data[col].iloc[0])
                except (ValueError, TypeError, IndexError):
                    pass
                break

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
