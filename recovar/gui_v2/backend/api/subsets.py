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
    # Optional: when omitted/empty, the source dataset is resolved from the
    # subset's source job (and a star is constructed from .mrcs+poses+ctf if
    # the dataset has no native .star file).
    particles_star: str | None = None


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


def _build_full_star_from_dataset(
    particles_path: str, poses_path: str, ctf_path: str, out_path: str,
) -> str:
    """Construct a full .star file for a .mrcs dataset from poses/ctf pkls.

    Uses recovar's own cryoDRGN-format writer so the result round-trips through
    ``read_star``.  Writes every particle in original order; callers filter to a
    subset afterwards by positional index.  Runs in a worker thread (blocking).
    """
    from recovar.utils.helpers import write_starfile_from_cryodrgn_format

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Write to a temp path first so a concurrent/interrupted build never leaves a
    # truncated star that later reads as valid-but-short.
    tmp_path = out_path + ".tmp"
    write_starfile_from_cryodrgn_format(
        ctf_path, poses_path, particles_path, tmp_path,
    )
    os.replace(tmp_path, out_path)
    return out_path


async def _resolve_source_star(
    subset_obj, project_path: str, session_factory, given_star: str | None,
):
    """Resolve a full .star to filter a subset from, plus an optional index map.

    Returns ``(star_path, index_map)``.  ``index_map`` (when not ``None``) maps
    analyzed-dataset positions — the index space of the stored subset — to rows
    of the returned star.  It is needed when the star is *constructed* from a
    ``.mrcs`` dataset's full poses/ctf but the pipeline analyzed only an
    ``--ind`` subset: row ``index_map[i]`` of the full star is analyzed particle
    ``i``.  ``None`` means the subset indices address the star rows directly.

    Resolution order:
      1. An explicit, valid ``given_star`` (the dataset's native star file).
      2. The ``particles`` recorded on the source job (or its parent Pipeline)
         when that is itself a ``.star``.
      3. A star constructed from a ``.mrcs`` dataset's particles + poses + ctf
         pkls (cached under ``<project>/subsets/.star_cache``), with an index
         map derived from the pipeline's ``--ind`` when present.

    Returns ``(None, None)`` if no source can be determined.
    """
    if given_star and os.path.isfile(given_star) and given_star.endswith(".star"):
        return given_star, None

    from recovar.gui_v2.backend.models.job import Job

    particles = poses = ctf = ind = None
    pipeline_dir = None
    async with session_factory() as session:
        src = None
        if subset_obj.source_job_id:
            src = (
                await session.execute(
                    select(Job).where(Job.id == subset_obj.source_job_id)
                )
            ).scalar_one_or_none()
        if src is not None:
            p = src.params or {}
            if src.type == "Pipeline":
                particles = p.get("particles")
                poses = p.get("poses")
                ctf = p.get("ctf")
                ind = p.get("ind")
                pipeline_dir = src.output_dir
            else:
                # Analyze (and similar) jobs record the pipeline they ran on.
                pipeline_dir = p.get("result_dir") or src.output_dir
        # Fall back to the Pipeline job that produced this output directory.
        if not particles and pipeline_dir:
            pj = (
                await session.execute(
                    select(Job)
                    .where(Job.project_id == subset_obj.project_id)
                    .where(Job.type == "Pipeline")
                    .where(Job.output_dir == pipeline_dir)
                )
            ).scalar_one_or_none()
            if pj is not None:
                pp = pj.params or {}
                particles = pp.get("particles")
                poses = pp.get("poses")
                ctf = pp.get("ctf")
                ind = pp.get("ind")

    if particles and str(particles).endswith(".star") and os.path.isfile(particles):
        return particles, None

    if (
        particles and poses and ctf
        and os.path.isfile(particles)
        and os.path.isfile(poses)
        and os.path.isfile(ctf)
    ):
        import asyncio
        import hashlib

        cache_dir = os.path.join(project_path, "subsets", ".star_cache")
        base = "".join(
            c if c.isalnum() or c in "-_." else "_"
            for c in os.path.basename(str(particles))
        )
        # Disambiguate by the full absolute path so same-named particle files in
        # different pipelines never collide on the same cached star.
        digest = hashlib.sha1(
            os.path.abspath(str(particles)).encode("utf-8")
        ).hexdigest()[:12]
        out_path = os.path.join(cache_dir, f"{base}.{digest}.star")
        if not os.path.isfile(out_path):
            await asyncio.to_thread(
                _build_full_star_from_dataset, particles, poses, ctf, out_path,
            )

        # The constructed star holds the FULL stack in original order, but the
        # subset indices address the analyzed dataset.  When the pipeline ran on
        # an --ind subset, map analyzed position i -> original star row ind[i].
        index_map = None
        if ind and os.path.isfile(str(ind)):
            try:
                with open(str(ind), "rb") as f:
                    index_map = np.asarray(pickle.load(f), dtype=np.int64)
            except Exception as exc:  # noqa: BLE001 — bad ind shouldn't 500
                logger.warning("Ignoring unreadable --ind file %s: %s", ind, exc)
                index_map = None
        return out_path, index_map

    return None, None


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

    source_star, index_map = await _resolve_source_star(
        subset_obj, project_path, session_factory, req.particles_star,
    )
    if source_star is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not resolve a particle source for this subset. Provide a "
                ".star dataset, or ensure the source job records its "
                "particles/poses/ctf so a star file can be built."
            ),
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
        particles_df, optics_df = read_star(source_star)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse star file: {exc}",
        )

    if indices.size == 0:
        raise HTTPException(
            status_code=400, detail="Subset has no particle indices to export",
        )

    # When the star was built from a full .mrcs stack but the pipeline analyzed
    # an --ind subset, the stored indices address the analyzed dataset; map them
    # through ind to the corresponding rows of the full star.
    if index_map is not None:
        if indices.min() < 0 or indices.max() >= len(index_map):
            bad = int(indices.max()) if indices.max() >= len(index_map) else int(indices.min())
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Subset index {bad} out of range "
                    f"(analyzed dataset has {len(index_map)} particles)"
                ),
            )
        row_indices = index_map[indices]
    else:
        row_indices = indices

    # Validate against the FULL star file.  Subset indices are original particle
    # indices (the Explore view maps any subsampled selection back through the
    # original-index array before creating the subset), so a valid index must
    # fall within [0, len(particles_df)).
    if row_indices.min() < 0 or row_indices.max() >= len(particles_df):
        bad = int(row_indices.max()) if row_indices.max() >= len(particles_df) else int(row_indices.min())
        raise HTTPException(
            status_code=400,
            detail=(
                f"Subset index {bad} out of range "
                f"(star file has {len(particles_df)} particles)"
            ),
        )
    filtered_df = particles_df.iloc[row_indices].reset_index(drop=True)

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
