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
from recovar.gui_v2.backend.config import get_db_path
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
    method: dict | None = None
    created: str
    ind_path: str


class SubsetListEntry(BaseModel):
    id: str
    name: str
    n_particles: int
    source_job_id: str | None = None
    method: dict | None = None
    created: str


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
                method=s.method,
                created=s.created_at.isoformat(),
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
                    method=subset.method,
                    created=subset.created_at.isoformat(),
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
