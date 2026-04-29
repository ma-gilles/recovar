"""System info API.

Endpoints:
    GET /api/system/info            — Server environment details
    GET /api/system/slurm-defaults  — Default SLURM settings for job forms
"""

from __future__ import annotations

import os
import platform
import shutil

from fastapi import APIRouter
from pydantic import BaseModel

from recovar.gui_v2.backend.services.executor import slurm_available

router = APIRouter(prefix="/api/system", tags=["system"])


class DiskInfo(BaseModel):
    path: str
    total: int
    used: int
    free: int


class SystemInfoResponse(BaseModel):
    slurm_available: bool
    executor_mode: str  # "slurm" or "local"
    recovar_version: str
    gpu_count: int
    hostname: str
    disk: DiskInfo | None = None


def _recovar_version() -> str:
    try:
        from recovar import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _gpu_count() -> int:
    """Count GPUs via CUDA_VISIBLE_DEVICES or nvidia-smi."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len(visible.split(","))
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().split("\n") if l.strip()])
    except Exception:
        pass
    return 0


@router.get("/info", response_model=SystemInfoResponse)
async def system_info() -> SystemInfoResponse:
    has_slurm = slurm_available()

    # Disk usage for the current working directory
    disk = None
    try:
        cwd = os.getcwd()
        usage = shutil.disk_usage(cwd)
        disk = DiskInfo(
            path=cwd,
            total=usage.total,
            used=usage.used,
            free=usage.free,
        )
    except OSError:
        pass

    return SystemInfoResponse(
        slurm_available=has_slurm,
        executor_mode="slurm" if has_slurm else "local",
        recovar_version=_recovar_version(),
        gpu_count=_gpu_count(),
        hostname=platform.node(),
        disk=disk,
    )


class SlurmDefaultsResponse(BaseModel):
    partition: str
    account: str
    gpus: int
    cpus: int
    memory: str
    time: str
    gpu_resource_spec: str = "--gres=gpu:{gpus}"
    template_path: str | None = None


@router.get("/slurm-defaults", response_model=SlurmDefaultsResponse)
async def slurm_defaults(project_dir: str | None = None) -> SlurmDefaultsResponse:
    """Return effective SLURM defaults for pre-filling the job-submission form.

    Layers, lowest to highest precedence:
      1. Built-in DEFAULT_SLURM.
      2. ``~/.config/recovar/config.toml`` ``[slurm]`` section.
      3. ``<project_dir>/recovar.toml`` ``[slurm]`` section (if
         ``project_dir`` query param is provided).

    The frontend should pass ``project_dir`` whenever it knows which
    project the form is being filled for, so per-project settings flow
    through. When ``project_dir`` is omitted only layers 1+2 apply.
    """
    from recovar.gui_v2.backend.services.project_config import resolve_slurm_defaults

    merged = resolve_slurm_defaults(project_dir=project_dir)
    return SlurmDefaultsResponse(
        partition=merged.get("partition", ""),
        account=merged.get("account", ""),
        gpus=int(merged.get("gpus", 1)),
        cpus=int(merged.get("cpus", 4)),
        memory=str(merged.get("memory", "300G")),
        time=str(merged.get("time", "12:00:00")),
        gpu_resource_spec=str(merged.get("gpu_resource_spec", "--gres=gpu:{gpus}")),
        template_path=merged.get("template_path") or None,
    )
