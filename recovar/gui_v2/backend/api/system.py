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

from recovar.gui_v2.backend.config import DEFAULT_SLURM
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
            capture_output=True, text=True, timeout=5,
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


@router.get("/slurm-defaults", response_model=SlurmDefaultsResponse)
async def slurm_defaults() -> SlurmDefaultsResponse:
    """Return default SLURM settings for pre-filling job submission forms."""
    return SlurmDefaultsResponse(
        partition=DEFAULT_SLURM["partition"],
        account=DEFAULT_SLURM["account"],
        gpus=DEFAULT_SLURM["gpus"],
        cpus=DEFAULT_SLURM["cpus"],
        memory=DEFAULT_SLURM["memory"],
        time=DEFAULT_SLURM["time"],
    )
