"""System info API.

Endpoints:
    GET  /api/system/info                  — Server environment details
    GET  /api/system/slurm-defaults        — Default SLURM settings for job forms
    POST /api/system/generate-test-dataset — Run recovar make_test_dataset to create
                                              a small synthetic dataset for tutorials
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from recovar.gui_v2.backend.api.files import _check_path_allowed
from recovar.gui_v2.backend.config import DEFAULT_SLURM
from recovar.gui_v2.backend.services.executor import slurm_available

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Test dataset generation (tutorial)
# ---------------------------------------------------------------------------


class TestDatasetRequest(BaseModel):
    output_dir: str = Field(
        ...,
        description="Absolute directory under an allowed root. Will be created if missing.",
    )
    image_size: int = Field(64, ge=32, le=256)
    n_images: int = Field(2000, ge=100, le=200000)
    seed: int | None = Field(0, description="Random seed for reproducibility")


class TestDatasetResponse(BaseModel):
    output_dir: str
    files_created: list[str]
    duration_seconds: float


def _find_recovar_binary() -> str | None:
    """Find the recovar CLI binary that the running Python provides."""
    candidate = shutil.which("recovar")
    if candidate:
        return candidate
    # Fall back to the binary next to the running Python (works when the
    # server is launched via `pixi run python -m recovar.gui_v2.backend.main`).
    import sys
    py_dir = os.path.dirname(sys.executable)
    candidate = os.path.join(py_dir, "recovar")
    if os.path.isfile(candidate):
        return candidate
    return None


def _run_make_test_dataset_sync(req: TestDatasetRequest) -> TestDatasetResponse:
    import time

    binary = _find_recovar_binary()
    if not binary:
        raise HTTPException(
            status_code=500,
            detail="Could not locate the 'recovar' CLI binary alongside the GUI server.",
        )

    os.makedirs(req.output_dir, exist_ok=True)

    cmd = [
        binary,
        "make_test_dataset",
        req.output_dir,
        "--image-size",
        str(req.image_size),
        "--n-images",
        str(req.n_images),
    ]
    if req.seed is not None:
        cmd += ["--seed", str(req.seed)]

    logger.info("Running %s", " ".join(cmd))
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 min cap; 64^3 x 2000 finishes in well under a minute
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail=f"make_test_dataset timed out after {exc.timeout}s",
        ) from exc
    duration = time.time() - start

    if result.returncode != 0:
        logger.warning("make_test_dataset failed: %s", result.stderr[:2000])
        raise HTTPException(
            status_code=500,
            detail=f"make_test_dataset exited with {result.returncode}: {result.stderr.strip()[-500:]}",
        )

    # List the files the command actually created so the client can verify.
    try:
        created = sorted(
            os.path.relpath(os.path.join(d, f), req.output_dir)
            for d, _, fs in os.walk(req.output_dir)
            for f in fs
        )
    except OSError:
        created = []

    return TestDatasetResponse(
        output_dir=req.output_dir,
        files_created=created,
        duration_seconds=round(duration, 2),
    )


@router.post("/generate-test-dataset", response_model=TestDatasetResponse)
async def generate_test_dataset(req: TestDatasetRequest) -> TestDatasetResponse:
    """Run ``recovar make_test_dataset`` to create a small synthetic dataset.

    Defaults to a 64^3 box × 2000 images that finishes in seconds and is
    enough to demo the full pipeline / analyze / density / trajectory
    workflow without downloading anything.
    """
    out = os.path.abspath(req.output_dir)
    _check_path_allowed(out)
    req.output_dir = out
    return await asyncio.to_thread(_run_make_test_dataset_sync, req)
