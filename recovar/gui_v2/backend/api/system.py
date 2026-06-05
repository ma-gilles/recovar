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


class GpuInfo(BaseModel):
    index: int
    name: str


class SystemInfoResponse(BaseModel):
    slurm_available: bool
    executor_mode: str  # "slurm", "local", or "both"
    recovar_version: str
    gpu_count: int
    gpu_list: list[GpuInfo] = []
    hostname: str
    disk: DiskInfo | None = None


def _recovar_version() -> str:
    try:
        from recovar import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _gpu_info() -> tuple[int, list[dict]]:
    """Return (count, list of {index, name}) for available GPUs."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(", ", 1)
                if len(parts) == 2:
                    gpus.append({"index": int(parts[0]), "name": parts[1]})
            return len(gpus), gpus
    except Exception:
        pass
    return 0, []


@router.get("/info", response_model=SystemInfoResponse)
async def system_info() -> SystemInfoResponse:
    has_slurm = slurm_available()
    # sbatch on PATH means both executors are available (user can pick per job)
    sbatch_on_path = shutil.which("sbatch") is not None

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

    if sbatch_on_path:
        mode = "both"
    elif has_slurm:
        mode = "slurm"
    else:
        mode = "local"

    gpu_count, gpu_list = _gpu_info()

    return SystemInfoResponse(
        slurm_available=has_slurm or sbatch_on_path,
        executor_mode=mode,
        recovar_version=_recovar_version(),
        gpu_count=gpu_count,
        gpu_list=[GpuInfo(**g) for g in gpu_list],
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
        memory=str(merged.get("memory", "400G")),
        time=str(merged.get("time", "08:00:00")),
        gpu_resource_spec=str(merged.get("gpu_resource_spec", "--gres=gpu:{gpus}")),
        template_path=merged.get("template_path") or None,
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
