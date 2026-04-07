"""Mask creation API.

Endpoints:
    POST /api/masks/preview         — Generate a mask in memory and return
                                       a slice PNG with the source overlaid
                                       in green, for live wizard preview.
    POST /api/masks/save            — Generate a mask at full resolution and
                                       write it to ``<project>/Masks/<name>.mrc``.
    GET  /api/projects/{id}/masks   — List masks saved under a project.

The mask generation logic delegates to :func:`recovar.core.mask.make_mask`,
which is the same function used by the recovar pipeline ``--mask`` flag.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from recovar.gui_v2.backend.api.files import _check_path_allowed
from recovar.gui_v2.backend.api.project import _load_project_by_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/masks", tags=["masks"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EraseSphere(BaseModel):
    """A sphere of voxels to subtract from the generated mask."""

    x: float
    y: float
    z: float
    r: float = Field(..., gt=0, description="Radius in voxels")


class MaskParams(BaseModel):
    """Parameters forwarded to ``recovar.core.mask.make_mask``."""

    source_path: str = Field(..., description="Absolute path to source MRC volume")
    threshold: float | None = Field(
        None,
        description="Density threshold; if None, Otsu auto-threshold is used",
    )
    lowpass_sigma: float | None = Field(
        None,
        description="Gaussian sigma in voxels for low-pass smoothing; None=auto, 0=disable",
    )
    extend: int | None = Field(
        None,
        description="Dilation in voxels; None=auto",
    )
    soft_edge: float = Field(6.0, description="Width of cosine soft edge in voxels")
    cleanup: bool = Field(
        True,
        description="Fill holes and keep only the largest connected component",
    )
    erase_spheres: list[EraseSphere] = Field(
        default_factory=list,
        description="Spheres in voxel coordinates whose contents are zeroed in the final mask",
    )


class PreviewRequest(MaskParams):
    axis: int = Field(2, ge=0, le=2)
    idx: int | None = Field(None, description="Slice index; defaults to middle")


class PreviewVolumeRequest(MaskParams):
    project_id: str = Field(..., description="Used to locate the project Masks/ dir")


class SaveRequest(MaskParams):
    project_id: str
    output_name: str = Field(..., min_length=1)


class MaskInfo(BaseModel):
    name: str
    path: str
    size_bytes: int
    modified: str  # ISO 8601


# ---------------------------------------------------------------------------
# Helpers (run in thread pool to avoid blocking the event loop)
# ---------------------------------------------------------------------------


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _sanitize_output_name(name: str) -> str:
    """Validate a user-supplied mask filename. Returns the bare basename.

    Rejects path separators and unusual characters; appends ``.mrc`` if
    missing. Raises HTTPException(400) on invalid input.
    """
    base = os.path.basename(name).strip()
    if not base:
        raise HTTPException(status_code=400, detail="Output name is empty")
    if not base.lower().endswith(".mrc"):
        base = base + ".mrc"
    stem = base[:-4]
    if not _SAFE_NAME_RE.match(stem):
        raise HTTPException(
            status_code=400,
            detail="Output name must contain only letters, digits, '.', '_' or '-'",
        )
    return base


def _read_mrc(path: str) -> tuple[Any, float]:
    """Synchronously read an MRC file and return (data, voxel_size)."""
    import mrcfile
    with mrcfile.open(path, mode="r") as mrc:
        data = mrc.data.copy()
        vs = float(mrc.voxel_size.x) if mrc.voxel_size.x > 0 else 1.0
    return data, vs


def _write_mrc(path: str, data: Any, voxel_size: float) -> None:
    """Synchronously write a numpy array to an MRC file."""
    import mrcfile
    import numpy as np
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = voxel_size


def _generate_mask(volume: Any, params: MaskParams) -> Any:
    """Run :func:`recovar.core.mask.make_mask` then apply any sphere erases."""
    from recovar.core.mask import make_mask
    import numpy as np

    threshold: Any = "auto" if params.threshold is None else float(params.threshold)
    mask = make_mask(
        volume,
        threshold=threshold,
        lowpass_sigma=params.lowpass_sigma,
        extend=params.extend,
        soft_edge=params.soft_edge,
        cleanup=params.cleanup,
    )

    if params.erase_spheres:
        mask = np.asarray(mask, dtype=np.float32).copy()
        nz, ny, nx = mask.shape
        # Build coordinate grids once per call (the volumes here are small).
        zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
        for s in params.erase_spheres:
            # ``EraseSphere`` uses (x, y, z) which we map to MRC axes
            # (column-major: data[z, y, x]).
            dx = xx - s.x
            dy = yy - s.y
            dz = zz - s.z
            inside = (dx * dx + dy * dy + dz * dz) <= (s.r * s.r)
            mask[inside] = 0.0
    return mask


def _render_overlay_png(source: Any, mask: Any, axis: int, idx: int) -> bytes:
    """Render a slice of *source* with *mask* overlaid in semi-transparent green."""
    import numpy as np
    from PIL import Image

    if axis == 0:
        src_slc = source[idx, :, :]
        mask_slc = mask[idx, :, :]
    elif axis == 1:
        src_slc = source[:, idx, :]
        mask_slc = mask[:, idx, :]
    else:
        src_slc = source[:, :, idx]
        mask_slc = mask[:, :, idx]

    src_slc = np.asarray(src_slc, dtype=np.float32)
    mask_slc = np.clip(np.asarray(mask_slc, dtype=np.float32), 0.0, 1.0)

    mn, mx = float(src_slc.min()), float(src_slc.max())
    if mx - mn > 0:
        src_norm = (src_slc - mn) / (mx - mn)
    else:
        src_norm = np.zeros_like(src_slc)
    src_u8 = (src_norm * 255).astype(np.uint8)

    rgb = np.stack([src_u8, src_u8, src_u8], axis=-1).astype(np.float32)
    overlay_strength = 0.55
    overlay_g = mask_slc * 255.0 * overlay_strength
    rgb[..., 1] = np.clip(rgb[..., 1] + overlay_g, 0, 255)
    rgb[..., 0] = np.clip(rgb[..., 0] - overlay_g * 0.4, 0, 255)
    rgb[..., 2] = np.clip(rgb[..., 2] - overlay_g * 0.4, 0, 255)
    rgb_u8 = rgb.astype(np.uint8)

    img = Image.fromarray(rgb_u8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/preview")
async def preview_mask(req: PreviewRequest) -> Response:
    """Generate a mask in memory and return a single PNG slice with overlay."""
    _check_path_allowed(req.source_path)
    if not os.path.isfile(req.source_path):
        raise HTTPException(status_code=404, detail=f"Source not found: {req.source_path}")

    try:
        data, _voxel = await asyncio.to_thread(_read_mrc, req.source_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read MRC: {exc}") from exc

    if data.ndim != 3 or any(d < 4 for d in data.shape):
        raise HTTPException(status_code=400, detail=f"Source is not a 3D volume: shape={list(data.shape)}")

    try:
        mask = await asyncio.to_thread(_generate_mask, data, req)
    except Exception as exc:
        logger.exception("Mask generation failed")
        raise HTTPException(status_code=400, detail=f"Mask generation failed: {exc}") from exc

    idx = req.idx if req.idx is not None else data.shape[req.axis] // 2
    if idx < 0 or idx >= data.shape[req.axis]:
        raise HTTPException(
            status_code=400,
            detail=f"Slice index {idx} out of range for axis {req.axis}",
        )

    png = await asyncio.to_thread(_render_overlay_png, data, mask, req.axis, idx)
    coverage = float((mask > 0.5).mean())
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Mask-Coverage": f"{coverage:.4f}",
            "X-Volume-Shape": ",".join(str(d) for d in data.shape),
        },
    )


@router.post("/save")
async def save_mask(req: SaveRequest) -> dict:
    """Generate the mask at full resolution and save it under ``<project>/Masks/``."""
    _check_path_allowed(req.source_path)
    if not os.path.isfile(req.source_path):
        raise HTTPException(status_code=404, detail=f"Source not found: {req.source_path}")

    project, session = await _load_project_by_id(req.project_id)
    try:
        out_basename = _sanitize_output_name(req.output_name)
        masks_dir = os.path.join(project.path, "Masks")
        os.makedirs(masks_dir, exist_ok=True)
        out_path = os.path.join(masks_dir, out_basename)
    finally:
        await session.close()

    try:
        data, voxel = await asyncio.to_thread(_read_mrc, req.source_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read MRC: {exc}") from exc

    if data.ndim != 3:
        raise HTTPException(status_code=400, detail=f"Source is not a 3D volume: shape={list(data.shape)}")

    try:
        mask = await asyncio.to_thread(_generate_mask, data, req)
    except Exception as exc:
        logger.exception("Mask generation failed")
        raise HTTPException(status_code=400, detail=f"Mask generation failed: {exc}") from exc

    try:
        await asyncio.to_thread(_write_mrc, out_path, mask, voxel)
    except Exception as exc:
        logger.exception("Failed to write mask")
        raise HTTPException(status_code=500, detail=f"Failed to write mask: {exc}") from exc

    import datetime as _dt
    return {
        "name": out_basename,
        "path": out_path,
        "size_bytes": os.path.getsize(out_path),
        "modified": _dt.datetime.fromtimestamp(os.path.getmtime(out_path)).isoformat(),
    }


@router.post("/preview-volume")
async def preview_mask_volume(req: PreviewVolumeRequest) -> dict:
    """Generate a mask and write it to a temporary MRC file the client can
    fetch via ``/api/volumes/raw`` for 3D rendering.

    Returns ``{path, voxel_size, shape}``. The temp file lives under
    ``<project>/Masks/.preview_<uuid>.mrc`` (hidden from list_project_masks
    by the leading dot). Callers should ``DELETE`` it via
    ``/api/masks/preview-volume?path=...`` when done.
    """
    _check_path_allowed(req.source_path)
    if not os.path.isfile(req.source_path):
        raise HTTPException(status_code=404, detail=f"Source not found: {req.source_path}")

    project, session = await _load_project_by_id(req.project_id)
    try:
        masks_dir = os.path.join(project.path, "Masks")
        os.makedirs(masks_dir, exist_ok=True)
    finally:
        await session.close()

    try:
        data, voxel = await asyncio.to_thread(_read_mrc, req.source_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read MRC: {exc}") from exc
    if data.ndim != 3:
        raise HTTPException(status_code=400, detail=f"Source is not a 3D volume: shape={list(data.shape)}")

    try:
        mask = await asyncio.to_thread(_generate_mask, data, req)
    except Exception as exc:
        logger.exception("Mask generation failed")
        raise HTTPException(status_code=400, detail=f"Mask generation failed: {exc}") from exc

    import uuid as _uuid
    out_path = os.path.join(masks_dir, f".preview_{_uuid.uuid4().hex}.mrc")
    try:
        await asyncio.to_thread(_write_mrc, out_path, mask, voxel)
    except Exception as exc:
        logger.exception("Failed to write preview")
        raise HTTPException(status_code=500, detail=f"Failed to write preview: {exc}") from exc

    return {
        "path": out_path,
        "voxel_size": voxel,
        "shape": list(mask.shape),
    }


@router.delete("/preview-volume")
async def delete_preview_volume(path: str) -> dict:
    """Delete a preview MRC produced by /api/masks/preview-volume.

    Only files whose basename starts with ``.preview_`` and end with
    ``.mrc`` are eligible — prevents the endpoint from being used as
    a generic delete API.
    """
    _check_path_allowed(path)
    base = os.path.basename(path)
    if not base.startswith(".preview_") or not base.endswith(".mrc"):
        raise HTTPException(status_code=400, detail="Not a preview file")
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError as exc:
        logger.warning("Could not delete preview %s: %s", path, exc)
    return {"deleted": True, "path": path}


@router.get("/by-project/{project_id}", response_model=list[MaskInfo])
async def list_project_masks(project_id: str) -> list[MaskInfo]:
    """List masks under ``<project>/Masks/``."""
    project, session = await _load_project_by_id(project_id)
    try:
        masks_dir = os.path.join(project.path, "Masks")
    finally:
        await session.close()

    if not os.path.isdir(masks_dir):
        return []

    import datetime as _dt
    out: list[MaskInfo] = []
    for entry in sorted(os.listdir(masks_dir)):
        full = os.path.join(masks_dir, entry)
        if not os.path.isfile(full) or not entry.lower().endswith(".mrc"):
            continue
        if entry.startswith("."):
            continue  # hidden / preview files
        try:
            stat = os.stat(full)
        except OSError:
            continue
        out.append(
            MaskInfo(
                name=entry,
                path=full,
                size_bytes=stat.st_size,
                modified=_dt.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            )
        )
    return out
