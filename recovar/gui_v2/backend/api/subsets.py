"""Subset CRUD API.

Endpoints:
    POST   /api/subsets           — Create a subset (.ind file + DB record)
    GET    /api/subsets            — List subsets for a project
    GET    /api/subsets/:id        — Get subset detail
    DELETE /api/subsets/:id        — Delete subset
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from recovar.gui_v2.backend.api.project import get_project_path
from recovar.gui_v2.backend.config import get_db_path, iso_utc
from recovar.gui_v2.backend.db import init_db
from recovar.gui_v2.backend.models.subset import Subset

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/subsets", tags=["subsets"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CreateSubsetRequest(BaseModel):
    project_id: str
    name: str
    source_job_id: str | None = None
    zdim: int | None = None
    method: dict[str, Any] | None = None
    indices: list[int]


class SubsetResponse(BaseModel):
    id: str
    name: str
    path: str
    n_particles: int


class SubsetDetailResponse(BaseModel):
    id: str
    name: str
    n_particles: int
    source_job_id: str | None = None
    zdim: int | None = None
    method: dict | None = None
    created: str
    ind_path: str


class SubsetListEntry(BaseModel):
    id: str
    name: str
    n_particles: int
    source_job_id: str | None = None
    zdim: int | None = None
    method: dict | None = None
    created: str


class SubsetProvenanceResponse(BaseModel):
    id: str
    name: str
    n_particles: int
    source_job_id: str | None = None
    source_job_name: str | None = None
    zdim: int | None = None
    method: dict | None = None
    created: str
    ind_path: str
    star_exports: list[str]


class ExportStarRequest(BaseModel):
    particles_star: str


class ExportStarResponse(BaseModel):
    path: str
    n_particles: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=SubsetResponse, status_code=201)
async def create_subset(req: CreateSubsetRequest) -> SubsetResponse:
    """Create a .ind file and DB record from particle indices."""
    project_path = get_project_path(req.project_id)
    if project_path is None:
        raise HTTPException(status_code=404, detail="Project not found")

    if not req.indices:
        raise HTTPException(status_code=400, detail="indices must not be empty")

    # Create subsets directory
    subsets_dir = os.path.join(project_path, "subsets")
    os.makedirs(subsets_dir, exist_ok=True)

    # Write .ind file (pickle of numpy int array, matching recovar format)
    ind_array = np.array(req.indices, dtype=np.int64)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in req.name)
    ind_filename = f"{safe_name}.ind"
    ind_path = os.path.join(subsets_dir, ind_filename)

    # Avoid overwriting
    counter = 1
    while os.path.exists(ind_path):
        ind_path = os.path.join(subsets_dir, f"{safe_name}_{counter}.ind")
        counter += 1

    with open(ind_path, "wb") as f:
        pickle.dump(ind_array, f)

    # Create DB record
    db_path = get_db_path(project_path)
    session_factory = await init_db(db_path)

    subset = Subset(
        project_id=req.project_id,
        name=req.name,
        source_job_id=req.source_job_id,
        zdim=req.zdim,
        method=req.method,
        n_particles=len(req.indices),
        ind_path=ind_path,
    )

    async with session_factory() as session:
        session.add(subset)
        await session.commit()
        await session.refresh(subset)

        return SubsetResponse(
            id=subset.id,
            name=subset.name,
            path=ind_path,
            n_particles=subset.n_particles,
        )


@router.get("", response_model=list[SubsetListEntry])
async def list_subsets(project_id: str = Query(...)) -> list[SubsetListEntry]:
    project_path = get_project_path(project_id)
    if project_path is None:
        raise HTTPException(status_code=404, detail="Project not found")

    db_path = get_db_path(project_path)
    session_factory = await init_db(db_path)

    async with session_factory() as session:
        stmt = select(Subset).where(Subset.project_id == project_id)
        result = await session.execute(stmt)
        subsets = result.scalars().all()

        return [
            SubsetListEntry(
                id=s.id,
                name=s.name,
                n_particles=s.n_particles,
                source_job_id=s.source_job_id,
                zdim=s.zdim,
                method=s.method,
                created=iso_utc(s.created_at),
            )
            for s in subsets
        ]


@router.get("/{subset_id}", response_model=SubsetDetailResponse)
async def get_subset(subset_id: str) -> SubsetDetailResponse:
    from recovar.gui_v2.backend.api.project import _project_registry

    for pid, ppath in _project_registry.items():
        db_path = get_db_path(ppath)
        session_factory = await init_db(db_path)
        async with session_factory() as session:
            stmt = select(Subset).where(Subset.id == subset_id)
            result = await session.execute(stmt)
            subset = result.scalar_one_or_none()
            if subset:
                return SubsetDetailResponse(
                    id=subset.id,
                    name=subset.name,
                    n_particles=subset.n_particles,
                    source_job_id=subset.source_job_id,
                    zdim=subset.zdim,
                    method=subset.method,
                    created=iso_utc(subset.created_at),
                    ind_path=subset.ind_path,
                )

    raise HTTPException(status_code=404, detail="Subset not found")


@router.delete("/{subset_id}", status_code=204)
async def delete_subset(subset_id: str) -> None:
    from recovar.gui_v2.backend.api.project import _project_registry

    for pid, ppath in _project_registry.items():
        db_path = get_db_path(ppath)
        session_factory = await init_db(db_path)
        async with session_factory() as session:
            stmt = select(Subset).where(Subset.id == subset_id)
            result = await session.execute(stmt)
            subset = result.scalar_one_or_none()
            if subset:
                await session.delete(subset)
                await session.commit()
                return

    raise HTTPException(status_code=404, detail="Subset not found")


@router.get("/{subset_id}/provenance", response_model=SubsetProvenanceResponse)
async def get_subset_provenance(subset_id: str) -> SubsetProvenanceResponse:
    """Return full creation provenance for a subset."""
    from recovar.gui_v2.backend.api.project import _project_registry
    from recovar.gui_v2.backend.models.job import Job

    for _pid, ppath in _project_registry.items():
        db_path = get_db_path(ppath)
        session_factory = await init_db(db_path)
        async with session_factory() as session:
            stmt = select(Subset).where(Subset.id == subset_id)
            result = await session.execute(stmt)
            subset = result.scalar_one_or_none()
            if subset:
                # Look up source job name if available
                source_job_name: str | None = None
                if subset.source_job_id:
                    job_stmt = select(Job).where(Job.id == subset.source_job_id)
                    job_result = await session.execute(job_stmt)
                    job = job_result.scalar_one_or_none()
                    if job:
                        source_job_name = f"{job.type} ({job.output_dir})"

                # Find .star exports (files with same base name in subsets dir)
                star_exports: list[str] = []
                subsets_dir = os.path.dirname(subset.ind_path)
                if os.path.isdir(subsets_dir):
                    safe_name = os.path.splitext(os.path.basename(subset.ind_path))[0]
                    for fname in os.listdir(subsets_dir):
                        if fname.endswith(".star") and fname.startswith(safe_name):
                            star_exports.append(os.path.join(subsets_dir, fname))

                return SubsetProvenanceResponse(
                    id=subset.id,
                    name=subset.name,
                    n_particles=subset.n_particles,
                    source_job_id=subset.source_job_id,
                    source_job_name=source_job_name,
                    zdim=subset.zdim,
                    method=subset.method,
                    created=iso_utc(subset.created_at),
                    ind_path=subset.ind_path,
                    star_exports=star_exports,
                )

    raise HTTPException(status_code=404, detail="Subset not found")


@router.post("/{subset_id}/export-star", response_model=ExportStarResponse)
async def export_star(
    subset_id: str, req: ExportStarRequest,
) -> ExportStarResponse:
    """Export a subset as a filtered .star file."""
    from recovar.gui_v2.backend.api.project import _project_registry

    subset_obj = None
    project_path = None
    for pid, ppath in _project_registry.items():
        db_path = get_db_path(ppath)
        session_factory = await init_db(db_path)
        async with session_factory() as session:
            stmt = select(Subset).where(Subset.id == subset_id)
            result = await session.execute(stmt)
            subset_obj = result.scalar_one_or_none()
            if subset_obj:
                project_path = ppath
                break

    if subset_obj is None:
        raise HTTPException(status_code=404, detail="Subset not found")

    if not os.path.isfile(req.particles_star):
        raise HTTPException(
            status_code=400,
            detail=f"Particles star file not found: {req.particles_star}",
        )
    if not req.particles_star.endswith(".star"):
        raise HTTPException(
            status_code=400, detail="particles_star must be a .star file",
        )

    if not os.path.isfile(subset_obj.ind_path):
        raise HTTPException(
            status_code=400,
            detail=f"Subset .ind file not found: {subset_obj.ind_path}",
        )
    with open(subset_obj.ind_path, "rb") as f:
        indices = pickle.load(f)
    indices = np.asarray(indices, dtype=np.int64)

    try:
        from recovar.data_io.starfile import read_star, write_star
    except ImportError:
        raise HTTPException(
            status_code=500, detail="recovar.data_io.starfile not available",
        )

    try:
        particles_df, optics_df = read_star(req.particles_star)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse star file: {exc}",
        )

    if indices.max() >= len(particles_df):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Subset index {indices.max()} out of range "
                f"(star file has {len(particles_df)} particles)"
            ),
        )
    filtered_df = particles_df.iloc[indices].reset_index(drop=True)

    subsets_dir = os.path.join(project_path, "subsets")
    os.makedirs(subsets_dir, exist_ok=True)
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in subset_obj.name
    )
    star_path = os.path.join(subsets_dir, f"{safe_name}.star")
    counter = 1
    while os.path.exists(star_path):
        star_path = os.path.join(subsets_dir, f"{safe_name}_{counter}.star")
        counter += 1

    write_star(star_path, filtered_df, optics_df)
    logger.info(
        "Exported subset %s as .star: %s (%d particles)",
        subset_id, star_path, len(filtered_df),
    )
    return ExportStarResponse(path=star_path, n_particles=len(filtered_df))
