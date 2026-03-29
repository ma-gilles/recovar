"""Embeddings (latent coordinates) API.

Endpoints:
    GET /api/jobs/:id/embeddings            — Binary embedding data
    GET /api/jobs/:id/embeddings/available   — Which zdims exist
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

from recovar.gui_v2.backend.api.jobs import _get_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["embeddings"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_analysis_dir(job_output_dir: str, zdim: int) -> str | None:
    """Find the analysis directory for a given zdim."""
    analysis_dir = os.path.join(job_output_dir, f"analysis_{zdim}")
    if os.path.isdir(analysis_dir):
        return analysis_dir
    return None


def _load_embeddings_sync(
    pipeline_dir: str,
    zdim: int,
    analyze_dir: str | None = None,
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
    """
    result: dict[str, Any] = {
        "pca_coords": None,
        "umap_coords": None,
        "kmeans_labels": None,
        "kmeans_centers": None,
        "n_particles": 0,
        "zdim": zdim,
    }

    # PCA coords: model/zdim_N/latent_coords.npy (new) or embeddings.pkl (legacy)
    zdim_dir = os.path.join(pipeline_dir, "model", f"zdim_{zdim}")
    latent_path = os.path.join(zdim_dir, "latent_coords.npy")
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

    if result["pca_coords"] is not None:
        result["n_particles"] = result["pca_coords"].shape[0]

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
            if os.path.isfile(umap_path):
                with open(umap_path, "rb") as f:
                    umap_data = pickle.load(f)
                if isinstance(umap_data, np.ndarray):
                    result["umap_coords"] = umap_data.astype(np.float32)

        # K-means
        if result["kmeans_labels"] is None:
            kmeans_path = os.path.join(a_dir, "data", "kmeans_result.pkl")
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

    return result


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

    data = await asyncio.to_thread(
        _load_embeddings_sync, pipeline_dir, zdim, analyze_dir
    )

    if data["pca_coords"] is None:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding data found for zdim={zdim}",
        )

    if format == "json":
        # Debug-only JSON fallback
        json_data = {
            "n_particles": data["n_particles"],
            "zdim": zdim,
            "pca_coords": data["pca_coords"].tolist() if data["pca_coords"] is not None else None,
            "umap_coords": data["umap_coords"].tolist() if data["umap_coords"] is not None else None,
            "kmeans_labels": data["kmeans_labels"].tolist() if data["kmeans_labels"] is not None else None,
            "kmeans_centers": data["kmeans_centers"].tolist() if data["kmeans_centers"] is not None else None,
        }
        return Response(
            content=json.dumps(json_data),
            media_type="application/json",
        )

    # Build binary response
    meta = {
        "n_particles": data["n_particles"],
        "zdim": zdim,
        "has_umap": data["umap_coords"] is not None,
        "has_kmeans": data["kmeans_labels"] is not None,
        "n_clusters": int(data["kmeans_centers"].shape[0]) if data["kmeans_centers"] is not None else 0,
    }

    parts: list[bytes] = []
    parts.append(data["pca_coords"].tobytes())

    if data["umap_coords"] is not None:
        parts.append(data["umap_coords"].tobytes())

    if data["kmeans_labels"] is not None:
        parts.append(data["kmeans_labels"].tobytes())

    if data["kmeans_centers"] is not None:
        parts.append(data["kmeans_centers"].tobytes())

    body = b"".join(parts)

    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={"X-Embedding-Meta": json.dumps(meta)},
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
