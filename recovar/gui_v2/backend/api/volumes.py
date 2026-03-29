"""Volume serving API.

Endpoints:
    GET /api/volumes/raw    — Serve MRC binary (with optional downsampling)
    GET /api/volumes/slice  — Serve orthogonal slice as PNG
    GET /api/volumes/info   — Volume metadata (shape, voxel_size, stats)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from recovar.gui_v2.backend.api.files import _check_path_allowed
from recovar.gui_v2.backend.config import MAX_SERVE_DIM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/volumes", tags=["volumes"])


# ---------------------------------------------------------------------------
# Sync helpers (run in thread pool to avoid blocking the event loop)
# ---------------------------------------------------------------------------


def _read_mrc_sync(path: str):
    """Read an MRC file, return (data_3d_array, voxel_size)."""
    import mrcfile
    import numpy as np
    with mrcfile.open(path, mode="r") as mrc:
        data = mrc.data.copy()
        voxel_size = float(mrc.voxel_size.x) if mrc.voxel_size.x > 0 else 1.0
    return data, voxel_size


def _downsample_volume(data, target_dim: int):
    """Downsample a 3D volume to target_dim^3 using scipy zoom."""
    import numpy as np
    from scipy.ndimage import zoom
    shape = data.shape
    factors = tuple(target_dim / s for s in shape)
    return zoom(data, factors, order=1).astype(np.float32)


def _volume_needs_downsample(path: str) -> bool:
    """Check if a volume exceeds MAX_SERVE_DIM without loading full data."""
    import mrcfile
    with mrcfile.open(path, mode="r") as mrc:
        shape = mrc.data.shape
    return any(d > MAX_SERVE_DIM for d in shape)


def _volume_to_mrc_bytes(data, voxel_size: float = 1.0) -> bytes:
    """Serialize a numpy array to MRC format via a temporary file.

    mrcfile 1.5.x does not reliably write to BytesIO objects, so we
    write to a NamedTemporaryFile and read the bytes back.
    """
    import mrcfile
    import numpy as np
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".mrc")
        os.close(fd)
        fd = None
        with mrcfile.new(tmp_path, overwrite=True) as mrc:
            mrc.set_data(data.astype(np.float32))
            mrc.voxel_size = voxel_size
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _render_slice_png(data, axis: int, idx: int) -> bytes:
    """Render an orthogonal slice as a PNG image."""
    import numpy as np

    if axis == 0:
        slc = data[idx, :, :]
    elif axis == 1:
        slc = data[:, idx, :]
    else:
        slc = data[:, :, idx]

    # Normalize to 0-255
    mn, mx = float(slc.min()), float(slc.max())
    if mx - mn > 0:
        slc = ((slc - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        slc = np.zeros_like(slc, dtype=np.uint8)

    # Encode as PNG
    from PIL import Image
    img = Image.fromarray(slc, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/raw")
async def volume_raw(
    path: str = Query(..., description="Absolute path to MRC file"),
    full: bool = Query(False, description="Serve full resolution (skip downsampling)"),
) -> Response:
    """Serve an MRC volume as binary.

    Volumes exceeding MAX_SERVE_DIM in any dimension are downsampled
    to MAX_SERVE_DIM^3 unless ``full=true``.

    When no downsampling is needed, the original file is served directly
    via FileResponse (zero-copy, no serialization overhead).
    """
    _check_path_allowed(path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    # Fast path: serve the original file directly when no downsampling needed.
    needs_downsample = await asyncio.to_thread(_volume_needs_downsample, path)
    if full or not needs_downsample:
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=os.path.basename(path),
        )

    # Slow path: read, downsample, serialize via temp file.
    data, voxel_size = await asyncio.to_thread(_read_mrc_sync, path)
    original_shape = ",".join(str(d) for d in data.shape)
    data = await asyncio.to_thread(_downsample_volume, data, MAX_SERVE_DIM)
    mrc_bytes = await asyncio.to_thread(_volume_to_mrc_bytes, data, voxel_size)
    return Response(
        content=mrc_bytes,
        media_type="application/octet-stream",
        headers={"X-Original-Shape": original_shape},
    )


@router.get("/slice")
async def volume_slice(
    path: str = Query(...),
    axis: int = Query(0, ge=0, le=2),
    idx: int = Query(0, ge=0),
) -> Response:
    """Serve an orthogonal slice as PNG."""
    _check_path_allowed(path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    data, _ = await asyncio.to_thread(_read_mrc_sync, path)

    if idx >= data.shape[axis]:
        raise HTTPException(
            status_code=400,
            detail=f"Slice index {idx} out of range for axis {axis} (max {data.shape[axis] - 1})",
        )

    png_bytes = await asyncio.to_thread(_render_slice_png, data, axis, idx)
    return Response(content=png_bytes, media_type="image/png")


@router.get("/info")
async def volume_info(path: str = Query(...)) -> dict:
    """Return volume metadata without loading the full data."""
    _check_path_allowed(path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    import mrcfile
    import numpy as np

    def _info():
        with mrcfile.open(path, mode="r") as mrc:
            data = mrc.data
            return {
                "shape": list(data.shape),
                "voxel_size": float(mrc.voxel_size.x) if mrc.voxel_size.x > 0 else 1.0,
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
            }

    return await asyncio.to_thread(_info)
