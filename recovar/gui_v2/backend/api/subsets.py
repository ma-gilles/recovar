"""Subset CRUD API.

Endpoints:
    POST   /api/subsets           — Create a subset (.ind file + DB record)
    GET    /api/subsets/{id}/pipeline-params — Pipeline form values to rerun on the subset
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


class SubsetPipelineParams(BaseModel):
    params: dict
    n_particles: int
    source_pipeline_dir: str


# Per-run settings of the source pipeline that must not carry over to a rerun.
_NOT_INHERITED = frozenset({"outdir", "output_name", "slurm_opts", "local_opts", "executor", "n_images"})


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

    session_factory = await init_db(get_db_path(project_path))
    if req.source_job_id:
        from recovar.gui_v2.backend.models.job import Job

        async with session_factory() as session:
            source = (await session.execute(select(Job).where(Job.id == req.source_job_id))).scalar_one_or_none()
        if source is None:
            raise HTTPException(
                status_code=400,
                detail=f"Job {req.source_job_id} does not belong to project {project_path}; "
                "the selection was made in a different project.",
            )

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
    """Resolve the full .star a subset is filtered from, or ``None``.

    Subset indices are original-file particle indices: the pipeline stores
    embeddings in original-file order (NaN rows for particles left out by
    ``--ind``), so they address the rows of the full star directly.

    Resolution order:
      1. An explicit, valid ``given_star`` (the dataset's native star file).
      2. The ``particles`` recorded on the source job (or its parent Pipeline)
         when that is itself a ``.star``.
      3. A star constructed from a ``.mrcs`` dataset's particles + poses + ctf
         pkls (cached under ``<project>/subsets/.star_cache``).
    """
    if given_star and os.path.isfile(given_star) and given_star.endswith(".star"):
        return given_star

    from recovar.gui_v2.backend.models.job import Job

    particles = poses = ctf = None
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

    if particles and str(particles).endswith(".star") and os.path.isfile(particles):
        return particles

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

        return out_path

    return None


async def _get_subset(subset_id: str):
    """Find a subset in any open project. Returns ``(subset, session_factory, project_path)``."""
    from recovar.gui_v2.backend.api.project import _project_registry

    for project_path in list(_project_registry.values()):
        session_factory = await init_db(get_db_path(project_path))
        async with session_factory() as session:
            subset_obj = (await session.execute(select(Subset).where(Subset.id == subset_id))).scalar_one_or_none()
        if subset_obj is not None:
            return subset_obj, session_factory, project_path
    raise HTTPException(status_code=404, detail="Subset not found")


async def _source_pipeline_job(subset_obj, session):
    """The Pipeline job whose dataset a subset's indices address."""
    from recovar.gui_v2.backend.models.job import Job

    if not subset_obj.source_job_id:
        raise HTTPException(status_code=400, detail="Subset has no source job; cannot tell which dataset it selects from.")
    src = (await session.execute(select(Job).where(Job.id == subset_obj.source_job_id))).scalar_one_or_none()
    if src is None:
        raise HTTPException(status_code=400, detail="The job this subset was selected from no longer exists.")
    if src.type == "Pipeline":
        return src
    pipeline_dir = (src.params or {}).get("result_dir")
    candidates = (
        (
            await session.execute(
                select(Job).where(Job.project_id == subset_obj.project_id).where(Job.type == "Pipeline")
            )
        )
        .scalars()
        .all()
    )
    for job in candidates:
        if pipeline_dir and os.path.realpath(job.output_dir) == os.path.realpath(pipeline_dir):
            return job
    raise HTTPException(
        status_code=400,
        detail=(
            f"No Pipeline job in this project produced {pipeline_dir}, so its inputs are unknown. "
            "Import that pipeline run with Scan, or export a .star and choose the inputs by hand."
        ),
    )


@router.get("/{subset_id}/pipeline-params", response_model=SubsetPipelineParams)
async def subset_pipeline_params(subset_id: str) -> SubsetPipelineParams:
    """Pipeline form values that rerun the source pipeline on just this subset.

    The rerun reads the same particles/poses/ctf as the source pipeline and
    restricts them with ``--ind``. Subset indices are original-file particle
    indices (the pipeline stores embeddings in original-file order), so the
    subset's index file is used as is, also when the source pipeline was itself
    restricted with ``--ind``.
    """
    subset_obj, session_factory, project_path = await _get_subset(subset_id)
    async with session_factory() as session:
        pipeline = await _source_pipeline_job(subset_obj, session)

    source = dict(pipeline.params or {})
    if not source.get("particles"):
        raise HTTPException(status_code=400, detail="The source pipeline job has no recorded particles input.")
    if not os.path.isfile(subset_obj.ind_path):
        raise HTTPException(status_code=404, detail=f"Subset .ind file not found: {subset_obj.ind_path}")

    source_ind = source.get("ind")
    if source_ind:
        from recovar.data_io._index_utils import load_index_like

        with open(subset_obj.ind_path, "rb") as f:
            selection = np.asarray(pickle.load(f), dtype=np.int64)
        stray = np.setdiff1d(selection, np.asarray(load_index_like(str(source_ind)), dtype=np.int64))
        if stray.size:
            raise HTTPException(
                status_code=400,
                detail=f"{stray.size} selected particles (e.g. {int(stray[0])}) were not analyzed by the source "
                f"pipeline (restricted by {source_ind}).",
            )

    params = {k: v for k, v in source.items() if k not in _NOT_INHERITED and v not in (None, "")}
    params["ind"] = subset_obj.ind_path
    return SubsetPipelineParams(
        params=params,
        n_particles=subset_obj.n_particles,
        source_pipeline_dir=pipeline.output_dir,
    )


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

    source_star = await _resolve_source_star(
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
