"""Embeddings (latent coordinates) API.

Endpoints:
    GET /api/jobs/:id/embeddings            — Binary embedding data
    GET /api/jobs/:id/embeddings/available   — Which zdims exist
    GET /api/jobs/:id/related-density       — Find completed Density jobs for same pipeline
    GET /api/jobs/:id/embeddings/density     — Evaluate density grid at particle coords
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import struct
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import select

from recovar.gui_v2.backend.api.jobs import _get_job
from recovar.gui_v2.backend.config import get_db_path
from recovar.gui_v2.backend.db import init_db
from recovar.gui_v2.backend.models.job import Job, JobStatus

try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    RegularGridInterpolator = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["embeddings"])

# Maximum number of particles to serve in a single response.  If the dataset
# exceeds this, we subsample uniformly and indicate it in the response header.
MAX_EMBEDDING_PARTICLES: int = 200_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_analysis_dir(job_output_dir: str, zdim: int) -> str | None:
    """Find the analysis directory for a given zdim."""
    analysis_dir = os.path.join(job_output_dir, f"analysis_{zdim}")
    if os.path.isdir(analysis_dir):
        return analysis_dir
    return None


def _subsample_indices(n_total: int, max_n: int) -> np.ndarray:
    """Return sorted, deterministic indices for uniform subsampling."""
    rng = np.random.RandomState(42)  # deterministic for reproducibility
    indices = np.sort(rng.choice(n_total, size=max_n, replace=False))
    return indices


def _load_embeddings_sync(
    pipeline_dir: str,
    zdim: int,
    analyze_dir: str | None = None,
    max_particles: int = MAX_EMBEDDING_PARTICLES,
) -> dict:
    """Load embedding data for a specific zdim (sync, for thread pool).

    Parameters
    ----------
    pipeline_dir : str
        Path to the pipeline output (contains ``model/``).
    zdim : int
        Latent dimensionality to load.
    analyze_dir : str | None
        Path to the analyze job output (contains ``data/``, ``plots/``).
        If None, falls back to ``{pipeline_dir}/analysis_{zdim}/``.
    max_particles : int
        Maximum particles to return.  If the dataset exceeds this, a
        uniform subsample is returned and ``subsampled`` is set to True.
    """
    result: dict[str, Any] = {
        "pca_coords": None,
        "umap_coords": None,
        "kmeans_labels": None,
        "kmeans_centers": None,
        "n_particles": 0,
        "n_particles_total": 0,
        "subsampled": False,
        "zdim": zdim,
    }

    # PCA coords: model/zdim_N/latent_coords.npy (new) or embeddings.pkl (legacy)
    zdim_dir = os.path.join(pipeline_dir, "model", f"zdim_{zdim}")
    latent_path = os.path.join(zdim_dir, "latent_coords.npy")
    try:
        if os.path.isfile(latent_path):
            result["pca_coords"] = np.load(latent_path).astype(np.float32)
        else:
            # Legacy: embeddings.pkl — may use 'zs' key (old format) or integer keys
            emb_path = os.path.join(pipeline_dir, "model", "embeddings.pkl")
            if os.path.isfile(emb_path):
                with open(emb_path, "rb") as f:
                    emb = pickle.load(f)
                if isinstance(emb, dict):
                    # New format: latent_coords[zdim]
                    if "latent_coords" in emb and zdim in emb["latent_coords"]:
                        result["pca_coords"] = np.array(
                            emb["latent_coords"][zdim], dtype=np.float32
                        )
                    # Old format: zs[zdim]
                    elif "zs" in emb and zdim in emb["zs"]:
                        result["pca_coords"] = np.array(
                            emb["zs"][zdim], dtype=np.float32
                        )
                    # Direct dict: {zdim: array}
                    elif zdim in emb:
                        result["pca_coords"] = np.array(emb[zdim], dtype=np.float32)
                elif isinstance(emb, np.ndarray):
                    result["pca_coords"] = emb.astype(np.float32)
    except (OSError, pickle.UnpicklingError, ValueError, EOFError) as exc:
        logger.error("Failed to load embedding coords from %s: %s", pipeline_dir, exc)
        raise _EmbeddingLoadError(
            f"Corrupted or unreadable embedding file: {exc}"
        ) from exc

    if result["pca_coords"] is not None:
        n_total = result["pca_coords"].shape[0]
        result["n_particles_total"] = n_total
        result["n_particles"] = n_total

    # Resolve analysis directory.  Priority:
    # 1. Explicit analyze_dir (from an Analyze job's output_dir)
    # 2. {pipeline_dir}/analysis_{zdim}/  (inline analysis, legacy)
    candidates: list[str] = []
    if analyze_dir and os.path.isdir(analyze_dir):
        candidates.append(analyze_dir)
    inline = os.path.join(pipeline_dir, f"analysis_{zdim}")
    if os.path.isdir(inline):
        candidates.append(inline)

    for a_dir in candidates:
        # UMAP coords
        if result["umap_coords"] is None:
            umap_path = os.path.join(a_dir, "plots", "umap", "umap_embedding.pkl")
            try:
                if os.path.isfile(umap_path):
                    with open(umap_path, "rb") as f:
                        umap_data = pickle.load(f)
                    if isinstance(umap_data, np.ndarray):
                        result["umap_coords"] = umap_data.astype(np.float32)
            except (OSError, pickle.UnpicklingError, ValueError, EOFError) as exc:
                logger.warning("Failed to load UMAP from %s: %s", umap_path, exc)
                # Non-fatal: continue without UMAP

        # K-means
        if result["kmeans_labels"] is None:
            kmeans_path = os.path.join(a_dir, "data", "kmeans_result.pkl")
            try:
                if os.path.isfile(kmeans_path):
                    with open(kmeans_path, "rb") as f:
                        km = pickle.load(f)
                    if isinstance(km, dict):
                        if "labels" in km:
                            result["kmeans_labels"] = np.array(
                                km["labels"], dtype=np.int32
                            )
                        if "centers" in km:
                            result["kmeans_centers"] = np.array(
                                km["centers"], dtype=np.float32
                            )
            except (OSError, pickle.UnpicklingError, ValueError, EOFError) as exc:
                logger.warning("Failed to load k-means from %s: %s", kmeans_path, exc)
                # Non-fatal: continue without k-means

    # ----- Pad or trim k-means centers to match requested zdim -----
    # The analyze job may have been run at a different zdim than the requested
    # one, so the k-means centers may have a different dimensionality.
    if result["kmeans_centers"] is not None and result["pca_coords"] is not None:
        center_zdim = result["kmeans_centers"].shape[1]
        if center_zdim < zdim:
            # Pad with zeros for extra dimensions
            result["kmeans_centers"] = np.pad(
                result["kmeans_centers"],
                ((0, 0), (0, zdim - center_zdim)),
                mode="constant",
                constant_values=0.0,
            )
        elif center_zdim > zdim:
            # Trim to requested zdim
            result["kmeans_centers"] = result["kmeans_centers"][:, :zdim]

    # ----- Subsample if dataset exceeds max_particles -----
    if (
        result["pca_coords"] is not None
        and result["n_particles_total"] > max_particles
    ):
        n_total = result["n_particles_total"]
        logger.info(
            "Subsampling embeddings: %d -> %d particles (zdim=%d)",
            n_total, max_particles, zdim,
        )
        indices = _subsample_indices(n_total, max_particles)
        result["pca_coords"] = result["pca_coords"][indices]
        if result["umap_coords"] is not None:
            result["umap_coords"] = result["umap_coords"][indices]
        if result["kmeans_labels"] is not None:
            result["kmeans_labels"] = result["kmeans_labels"][indices]
        # kmeans_centers are NOT subsampled (they are cluster centers, not per-particle)
        result["n_particles"] = max_particles
        result["subsampled"] = True

    return result


class _EmbeddingLoadError(Exception):
    """Raised when an embedding file cannot be loaded (corrupt, unreadable)."""


def _discover_zdims(
    pipeline_dir: str, analyze_dir: str | None = None
) -> dict:
    """Discover which zdim values and analyses are available.

    Parameters
    ----------
    pipeline_dir : str
        Path to the pipeline output (contains ``model/``).
    analyze_dir : str | None
        Path to the analyze job output (contains ``data/``, ``plots/``).
    """
    zdims: list[int] = []
    has_umap: dict[int, bool] = {}

    # Check model/zdim_* directories
    model_dir = os.path.join(pipeline_dir, "model")
    if os.path.isdir(model_dir):
        for name in sorted(os.listdir(model_dir)):
            if name.startswith("zdim_"):
                try:
                    z = int(name.split("_")[1])
                    latent = os.path.join(model_dir, name, "latent_coords.npy")
                    if os.path.isfile(latent):
                        zdims.append(z)
                except (ValueError, IndexError):
                    pass

    # Check for legacy embeddings.pkl
    if not zdims:
        emb_path = os.path.join(model_dir, "embeddings.pkl")
        if os.path.isfile(emb_path):
            try:
                with open(emb_path, "rb") as f:
                    emb = pickle.load(f)
                if isinstance(emb, dict):
                    # New format: latent_coords dict
                    if "latent_coords" in emb:
                        zdims = sorted(
                            int(k)
                            for k in emb["latent_coords"].keys()
                            if isinstance(k, int)
                        )
                    # Old format: zs dict
                    elif "zs" in emb:
                        zdims = sorted(
                            int(k)
                            for k in emb["zs"].keys()
                            if isinstance(k, int)
                        )
                    else:
                        zdims = sorted(
                            int(k)
                            for k in emb.keys()
                            if isinstance(k, (int, str)) and str(k).isdigit()
                        )
            except Exception:
                pass

    # Check which zdims have UMAP
    for z in zdims:
        found = False
        # Check analyze dir first (Analyze job output)
        if analyze_dir:
            umap_file = os.path.join(
                analyze_dir, "plots", "umap", "umap_embedding.pkl"
            )
            if os.path.isfile(umap_file):
                found = True
        # Also check inline analysis dir (legacy)
        if not found:
            analysis = os.path.join(pipeline_dir, f"analysis_{z}")
            umap_file = os.path.join(
                analysis, "plots", "umap", "umap_embedding.pkl"
            )
            if os.path.isfile(umap_file):
                found = True
        has_umap[z] = found

    return {"zdims": zdims, "has_umap": has_umap}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/{job_id}/embeddings")
async def get_embeddings(
    job_id: str,
    zdim: int = Query(..., description="Latent dimension"),
    format: str = Query("binary", description="'binary' or 'json'"),
) -> Response:
    """Serve embedding data as a binary ArrayBuffer with JSON header.

    Binary format (float32/int32, row-major):
        [pca_coords: n x zdim] [umap_coords: n x 2 | empty]
        [kmeans_labels: n x 1 int32 | empty] [kmeans_centers: k x zdim | empty]

    If the particle count exceeds ``MAX_EMBEDDING_PARTICLES``, the data is
    uniformly subsampled and the response includes
    ``X-Embedding-Subsampled: true`` with the original count in
    ``X-Embedding-Total-Particles``.
    """
    job, session = await _get_job(job_id)
    await session.close()

    # Resolve the pipeline output directory and optional analyze dir.
    # For Analyze jobs:
    #   - pipeline_dir = params["result_dir"] (where model/ lives)
    #   - analyze_dir  = job.output_dir (where data/, plots/ live)
    # For Pipeline jobs:
    #   - pipeline_dir = job.output_dir
    #   - analyze_dir  = None (falls back to analysis_N/ inside pipeline_dir)
    pipeline_dir = job.output_dir
    analyze_dir: str | None = None
    if job.type == "Analyze" and job.params and job.params.get("result_dir"):
        pipeline_dir = job.params["result_dir"]
        analyze_dir = job.output_dir

    try:
        data = await asyncio.to_thread(
            _load_embeddings_sync, pipeline_dir, zdim, analyze_dir
        )
    except _EmbeddingLoadError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    if data["pca_coords"] is None:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding data found for zdim={zdim}",
        )

    # Common metadata included in every response format.
    meta = {
        "n_particles": data["n_particles"],
        "n_particles_total": data["n_particles_total"],
        "subsampled": data["subsampled"],
        "zdim": zdim,
        "has_umap": data["umap_coords"] is not None,
        "has_kmeans": data["kmeans_labels"] is not None,
        "n_clusters": (
            int(data["kmeans_centers"].shape[0])
            if data["kmeans_centers"] is not None
            else 0
        ),
    }

    if format == "json":
        # Debug-only JSON fallback.  Still respects subsampling.
        json_data = {
            **meta,
            "pca_coords": (
                data["pca_coords"].tolist()
                if data["pca_coords"] is not None
                else None
            ),
            "umap_coords": (
                data["umap_coords"].tolist()
                if data["umap_coords"] is not None
                else None
            ),
            "kmeans_labels": (
                data["kmeans_labels"].tolist()
                if data["kmeans_labels"] is not None
                else None
            ),
            "kmeans_centers": (
                data["kmeans_centers"].tolist()
                if data["kmeans_centers"] is not None
                else None
            ),
        }
        return Response(
            content=json.dumps(json_data),
            media_type="application/json",
        )

    # Build binary response
    parts: list[bytes] = []
    parts.append(data["pca_coords"].tobytes())

    if data["umap_coords"] is not None:
        parts.append(data["umap_coords"].tobytes())

    if data["kmeans_labels"] is not None:
        parts.append(data["kmeans_labels"].tobytes())

    if data["kmeans_centers"] is not None:
        parts.append(data["kmeans_centers"].tobytes())

    body = b"".join(parts)

    headers = {"X-Embedding-Meta": json.dumps(meta)}
    if data["subsampled"]:
        headers["X-Embedding-Subsampled"] = "true"
        headers["X-Embedding-Total-Particles"] = str(data["n_particles_total"])

    return Response(
        content=body,
        media_type="application/octet-stream",
        headers=headers,
    )


@router.get("/{job_id}/embeddings/available")
async def embeddings_available(job_id: str) -> dict:
    job, session = await _get_job(job_id)
    await session.close()

    pipeline_dir = job.output_dir
    analyze_dir: str | None = None
    if job.type == "Analyze" and job.params and job.params.get("result_dir"):
        pipeline_dir = job.params["result_dir"]
        analyze_dir = job.output_dir

    return await asyncio.to_thread(_discover_zdims, pipeline_dir, analyze_dir)


# ---------------------------------------------------------------------------
# Density-related endpoints
# ---------------------------------------------------------------------------


@router.get("/{job_id}/related-density")
async def get_related_density(job_id: str) -> list[dict]:
    """Find completed Density jobs that reference the same pipeline output.

    Works for both Pipeline and Analyze jobs: resolves the pipeline
    ``result_dir`` and matches it against Density jobs in the same project.
    """
    job, session = await _get_job(job_id)
    await session.close()

    # Determine the pipeline result_dir this job is associated with.
    if job.type == "Analyze" and job.params and job.params.get("result_dir"):
        pipeline_dir = job.params["result_dir"]
    else:
        pipeline_dir = job.output_dir

    # Normalize for comparison.
    pipeline_dir_resolved = str(Path(pipeline_dir).resolve())

    # Query all completed Density jobs in the same project.
    from recovar.gui_v2.backend.api.project import get_project_path

    project_path = get_project_path(job.project_id)
    if not project_path:
        return []

    db_path = get_db_path(project_path)
    session_factory = await init_db(db_path)
    async with session_factory() as db_session:
        stmt = (
            select(Job)
            .where(Job.project_id == job.project_id)
            .where(Job.type == "Density")
            .where(Job.status == JobStatus.COMPLETED.value)
        )
        result = await db_session.execute(stmt)
        density_jobs = result.scalars().all()

    # Filter: only density jobs whose result_dir matches the pipeline dir.
    related: list[dict] = []
    for dj in density_jobs:
        dj_params = dj.params or {}
        dj_result_dir = (
            dj_params.get("result_dir")
            or dj_params.get("recovar_result_dir")
            or ""
        )
        if str(Path(dj_result_dir).resolve()) != pipeline_dir_resolved:
            continue

        density_pkl_path = os.path.join(
            dj.output_dir, "data", "deconv_density_knee.pkl"
        )
        entry: dict[str, Any] = {
            "id": dj.id,
            "output_dir": dj.output_dir,
            "pca_dim": (dj.params or {}).get("pca_dim"),
            "created": (
                dj.created_at.isoformat() if dj.created_at else None
            ),
        }
        if os.path.isfile(density_pkl_path):
            entry["density_pkl_path"] = density_pkl_path
        related.append(entry)

    return related


def _evaluate_density_sync(
    pipeline_dir: str,
    density_output_dir: str,
    zdim: int,
    pca_dim: int,
    max_particles: int = MAX_EMBEDDING_PARTICLES,
) -> dict:
    """Evaluate the deconvolved density grid at each particle's PCA coords.

    Returns a dict with binary buffers and metadata.  Runs synchronously
    (called via ``asyncio.to_thread``).
    """
    if RegularGridInterpolator is None:
        raise _EmbeddingLoadError(
            "scipy is required for density evaluation but is not installed"
        )

    # Load density pkl.
    density_pkl_path = os.path.join(
        density_output_dir, "data", "deconv_density_knee.pkl"
    )
    if not os.path.isfile(density_pkl_path):
        raise _EmbeddingLoadError(
            f"Density file not found: {density_pkl_path}"
        )

    try:
        with open(density_pkl_path, "rb") as f:
            density_data = pickle.load(f)
    except (OSError, pickle.UnpicklingError, ValueError, EOFError) as exc:
        raise _EmbeddingLoadError(
            f"Failed to load density file: {exc}"
        ) from exc

    density_grid = density_data["density"]  # ndarray, shape varies by pca_dim
    bounds = density_data["latent_space_bounds"]  # (pca_dim, 2)
    alpha = float(density_data.get("alpha", 0.0))

    # Load PCA coords using the same logic as _load_embeddings_sync.
    embeddings = _load_embeddings_sync(
        pipeline_dir, zdim, analyze_dir=None, max_particles=max_particles
    )
    pca_coords = embeddings["pca_coords"]
    if pca_coords is None:
        raise _EmbeddingLoadError(
            f"No PCA coordinates found for zdim={zdim}"
        )

    n_total = embeddings["n_particles_total"]
    n_particles = embeddings["n_particles"]
    subsampled = embeddings["subsampled"]
    kmeans_centers = embeddings["kmeans_centers"]

    # Build the interpolator.
    # Axis arrays: for each dim i, linspace(bounds[i,0], bounds[i,1], density.shape[i])
    axes = []
    for i in range(pca_dim):
        axes.append(
            np.linspace(bounds[i, 0], bounds[i, 1], density_grid.shape[i])
        )

    interpolator = RegularGridInterpolator(
        tuple(axes), density_grid, bounds_error=False, fill_value=0.0
    )

    # Evaluate density at particle positions (first pca_dim columns).
    # If zdim < pca_dim, pad with zeros for the missing dimensions.
    if pca_coords.shape[1] < pca_dim:
        pad_width = pca_dim - pca_coords.shape[1]
        particle_points = np.pad(
            pca_coords.astype(np.float64),
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        particle_points = pca_coords[:, :pca_dim].astype(np.float64)
    particle_density = interpolator(particle_points).astype(np.float64)

    # Evaluate at k-means centers if available.
    center_density = np.array([], dtype=np.float64)
    n_clusters = 0
    if kmeans_centers is not None and kmeans_centers.shape[0] > 0:
        if kmeans_centers.shape[1] < pca_dim:
            center_points = np.pad(
                kmeans_centers.astype(np.float64),
                ((0, 0), (0, pca_dim - kmeans_centers.shape[1])),
                mode="constant",
                constant_values=0.0,
            )
        else:
            center_points = kmeans_centers[:, :pca_dim].astype(np.float64)
        center_density = interpolator(center_points).astype(np.float64)
        n_clusters = kmeans_centers.shape[0]

    # Normalize all values to [0, 1].
    all_values = np.concatenate([particle_density, center_density])
    max_val = np.max(all_values) if len(all_values) > 0 else 1.0
    eps = 1e-12
    max_val = max(max_val, eps)

    particle_density = (particle_density / max_val).astype(np.float32)
    if len(center_density) > 0:
        center_density = (center_density / max_val).astype(np.float32)
    else:
        center_density = center_density.astype(np.float32)

    return {
        "particle_density": particle_density,
        "center_density": center_density,
        "n_particles": n_particles,
        "n_clusters": n_clusters,
        "pca_dim": pca_dim,
        "alpha": alpha,
        "subsampled": subsampled,
    }


@router.get("/{job_id}/embeddings/density")
async def get_embeddings_density(
    job_id: str,
    zdim: int = Query(..., description="Latent dimension for PCA coords"),
    density_job_id: str = Query(
        ..., description="ID of the completed Density job"
    ),
) -> Response:
    """Evaluate the deconvolved density grid at each particle's PCA coords.

    Binary format (float32, row-major):
        [particle_density: n x 1] [center_density: k x 1]

    Metadata returned in ``X-Density-Meta`` header as JSON.
    """
    # Load the target job to resolve the pipeline dir.
    job, session = await _get_job(job_id)
    await session.close()

    if job.type == "Analyze" and job.params and job.params.get("result_dir"):
        pipeline_dir = job.params["result_dir"]
    else:
        pipeline_dir = job.output_dir

    # Load the density job to get its output_dir and pca_dim.
    density_job, density_session = await _get_job(density_job_id)
    await density_session.close()

    if density_job.type != "Density":
        raise HTTPException(
            status_code=400,
            detail=f"Job {density_job_id} is not a Density job (type={density_job.type})",
        )
    if density_job.status != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Density job {density_job_id} is not completed (status={density_job.status})",
        )

    pca_dim = (density_job.params or {}).get("pca_dim")
    if pca_dim is None:
        raise HTTPException(
            status_code=400,
            detail="Density job is missing pca_dim in params",
        )
    pca_dim = int(pca_dim)

    try:
        data = await asyncio.to_thread(
            _evaluate_density_sync,
            pipeline_dir,
            density_job.output_dir,
            zdim,
            pca_dim,
        )
    except _EmbeddingLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Build binary response.
    parts: list[bytes] = []
    parts.append(data["particle_density"].tobytes())
    if len(data["center_density"]) > 0:
        parts.append(data["center_density"].tobytes())
    body = b"".join(parts)

    meta = {
        "n_particles": data["n_particles"],
        "n_clusters": data["n_clusters"],
        "pca_dim": data["pca_dim"],
        "alpha": data["alpha"],
        "density_job_id": density_job_id,
        "subsampled": data["subsampled"],
    }

    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={"X-Density-Meta": json.dumps(meta)},
    )
