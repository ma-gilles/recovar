"""Settings API for editing SLURM defaults from the GUI.

Endpoints:
    GET  /api/settings/slurm-defaults       — Layered view of SLURM defaults
    PUT  /api/settings/slurm-defaults/user   — Update user-global defaults
    PUT  /api/settings/slurm-defaults/project — Update per-project defaults
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SlurmDefaultsLayered(BaseModel):
    builtin: dict[str, Any]
    user: dict[str, Any]
    project: dict[str, Any]
    effective: dict[str, Any]
    user_config_path: str
    project_config_path: str | None = None


class SlurmDefaultsUpdate(BaseModel):
    partition: str | None = None
    account: str | None = None
    gpus: int | None = None
    cpus: int | None = None
    memory: str | None = None
    time: str | None = None


class ProjectSlurmDefaultsUpdate(SlurmDefaultsUpdate):
    project_dir: str


@router.get("/slurm-defaults", response_model=SlurmDefaultsLayered)
async def get_slurm_defaults_layered(
    project_dir: str | None = None,
) -> SlurmDefaultsLayered:
    """Return SLURM defaults broken out by layer (built-in, user, project)."""
    from recovar.gui_v2.backend.services.project_config import (
        resolve_slurm_defaults_layered,
    )

    result = resolve_slurm_defaults_layered(project_dir=project_dir)
    return SlurmDefaultsLayered(**result)


@router.put("/slurm-defaults/user", response_model=SlurmDefaultsLayered)
async def update_user_slurm_defaults(
    req: SlurmDefaultsUpdate,
    project_dir: str | None = None,
) -> SlurmDefaultsLayered:
    """Update the user-global SLURM defaults (~/.config/recovar/config.toml)."""
    from recovar.gui_v2.backend.services.project_config import (
        resolve_slurm_defaults_layered,
        save_user_slurm_defaults,
    )

    values = {k: v for k, v in req.model_dump().items() if v is not None}
    save_user_slurm_defaults(values)
    return SlurmDefaultsLayered(**resolve_slurm_defaults_layered(project_dir=project_dir))


@router.put("/slurm-defaults/project", response_model=SlurmDefaultsLayered)
async def update_project_slurm_defaults(
    req: ProjectSlurmDefaultsUpdate,
) -> SlurmDefaultsLayered:
    """Update SLURM defaults for a specific project (recovar.toml)."""
    import os

    from recovar.gui_v2.backend.services.project_config import (
        resolve_slurm_defaults_layered,
        save_project_slurm_defaults,
    )

    if not os.path.isdir(req.project_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.project_dir}")

    values = {k: v for k, v in req.model_dump(exclude={"project_dir"}).items() if v is not None}
    save_project_slurm_defaults(req.project_dir, values)
    return SlurmDefaultsLayered(**resolve_slurm_defaults_layered(project_dir=req.project_dir))
