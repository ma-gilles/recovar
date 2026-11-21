"""Settings API for editing SLURM and local execution defaults from the GUI.

Endpoints:
    GET  /api/settings/slurm-defaults       — Layered view of SLURM defaults
    PUT  /api/settings/slurm-defaults/user   — Update user-global SLURM defaults
    PUT  /api/settings/slurm-defaults/project — Update per-project SLURM defaults
    GET  /api/settings/local-defaults        — Layered view of local exec defaults
    PUT  /api/settings/local-defaults/user   — Update user-global local defaults
    PUT  /api/settings/local-defaults/project — Update per-project local defaults
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SlurmDefaultsLayered(BaseModel):
    builtin: dict[str, Any]
    detected: dict[str, Any] = {}
    user: dict[str, Any]
    project: dict[str, Any]
    effective: dict[str, Any]
    user_config_path: str
    project_config_path: str | None = None


# What sbatch accepts; a blank value clears the saved setting.
_SLURM_NAME = re.compile(r"[\w.+-]+(,[\w.+-]+)*")  # partition or account (comma list allowed)
_SLURM_MEMORY = re.compile(r"\d+[KMGT]?", re.IGNORECASE)  # 64G, 4000M, 2T
_SLURM_TIME = re.compile(r"(\d+-)?\d+(:\d{1,2}){0,2}")  # 30, 08:00:00, 2-12, 1-00:00:00


class SlurmDefaultsUpdate(BaseModel):
    partition: str | None = None
    account: str | None = None
    gpus: int | None = Field(default=None, ge=0)
    cpus: int | None = Field(default=None, ge=1)
    memory: str | None = None
    time: str | None = None

    @field_validator("partition", "account", "memory", "time")
    @classmethod
    def _what_sbatch_accepts(cls, value: str | None, info) -> str | None:
        if not value:
            return value
        value = value.strip()
        pattern, example = {
            "partition": (_SLURM_NAME, "gpu"),
            "account": (_SLURM_NAME, "mylab"),
            "memory": (_SLURM_MEMORY, "64G"),
            "time": (_SLURM_TIME, "08:00:00 or 2-00:00:00"),
        }[info.field_name]
        if not pattern.fullmatch(value):
            raise ValueError(f"{value!r} is not a valid SLURM {info.field_name} (for example {example})")
        return value


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

    from recovar.gui_v2.backend.services import preflight

    values = {k: v for k, v in req.model_dump().items() if v is not None}
    save_user_slurm_defaults(values)
    preflight.invalidate()  # the scheduler check tests these settings
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


# ---------------------------------------------------------------------------
# Local execution defaults
# ---------------------------------------------------------------------------


class LocalDefaultsLayered(BaseModel):
    builtin: dict[str, Any]
    user: dict[str, Any]
    project: dict[str, Any]
    effective: dict[str, Any]
    user_config_path: str
    project_config_path: str | None = None


class LocalDefaultsUpdate(BaseModel):
    gpus: str | None = None  # "all", "0", "0,1", etc.
    setup_command: str | None = None
    env_vars: dict[str, str] | None = None
    preallocate_gpu: bool | None = None


class ProjectLocalDefaultsUpdate(LocalDefaultsUpdate):
    project_dir: str


@router.get("/local-defaults", response_model=LocalDefaultsLayered)
async def get_local_defaults_layered(
    project_dir: str | None = None,
) -> LocalDefaultsLayered:
    from recovar.gui_v2.backend.services.project_config import (
        resolve_local_defaults_layered,
    )

    return LocalDefaultsLayered(**resolve_local_defaults_layered(project_dir=project_dir))


@router.put("/local-defaults/user", response_model=LocalDefaultsLayered)
async def update_user_local_defaults(
    req: LocalDefaultsUpdate,
    project_dir: str | None = None,
) -> LocalDefaultsLayered:
    from recovar.gui_v2.backend.services.project_config import (
        resolve_local_defaults_layered,
        save_user_local_defaults,
    )

    values = {k: v for k, v in req.model_dump().items() if v is not None}
    save_user_local_defaults(values)
    return LocalDefaultsLayered(**resolve_local_defaults_layered(project_dir=project_dir))


@router.put("/local-defaults/project", response_model=LocalDefaultsLayered)
async def update_project_local_defaults(
    req: ProjectLocalDefaultsUpdate,
) -> LocalDefaultsLayered:
    import os

    from recovar.gui_v2.backend.services.project_config import (
        resolve_local_defaults_layered,
        save_project_local_defaults,
    )

    if not os.path.isdir(req.project_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.project_dir}")

    values = {k: v for k, v in req.model_dump(exclude={"project_dir"}).items() if v is not None}
    save_project_local_defaults(req.project_dir, values)
    return LocalDefaultsLayered(**resolve_local_defaults_layered(project_dir=req.project_dir))
