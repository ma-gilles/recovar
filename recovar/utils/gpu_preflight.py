"""Pre-jax-init GPU memory probing via NVML or nvidia-smi.

Used in two ways:

1. Bootstrap parser in ``command_line.py`` — query physical GPU total
   to convert ``--gpu-gb`` into ``XLA_PYTHON_CLIENT_MEM_FRACTION`` for
   the hard-limit path. Must work BEFORE jax is imported.

2. Memory planner — sample ``physical_free_gb`` to detect conflicting
   processes and to pick a sensible effective budget.

Both NVML and the ``nvidia-smi`` subprocess are best-effort: the helper
returns ``None`` rather than raising when neither is available so the
caller can degrade gracefully (planner runs without conflict detection,
hard limit downgrades to soft).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_NVIDIA_SMI_TIMEOUT_S = 2.0


@dataclass
class ProcessInfo:
    pid: int
    name: str
    used_mb: int


@dataclass
class PhysicalGpuMemoryInfo:
    device_idx: int
    total_gb: float
    used_gb: float
    free_gb: float
    processes: list[ProcessInfo] = field(default_factory=list)
    source: str = ""  # "pynvml" or "nvidia-smi"


def _try_pynvml(device_idx: int) -> PhysicalGpuMemoryInfo | None:
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            procs: list[ProcessInfo] = []
            try:
                raw = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in raw:
                    name = ""
                    try:
                        name = pynvml.nvmlSystemGetProcessName(p.pid).decode("utf-8", "replace")
                    except Exception:
                        pass
                    used_mb = int((p.usedGpuMemory or 0) // (1024 * 1024))
                    procs.append(ProcessInfo(pid=int(p.pid), name=name, used_mb=used_mb))
            except Exception:
                pass
            return PhysicalGpuMemoryInfo(
                device_idx=device_idx,
                total_gb=mem.total / 1e9,
                used_gb=mem.used / 1e9,
                free_gb=mem.free / 1e9,
                processes=procs,
                source="pynvml",
            )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("pynvml probe failed: %s", exc)
        return None


def _try_nvidia_smi(device_idx: int) -> PhysicalGpuMemoryInfo | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_idx}",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
        if gpu_query.returncode != 0:
            return None
        line = gpu_query.stdout.strip().splitlines()[0]
        total_mib, used_mib, free_mib = (float(x.strip()) for x in line.split(","))
    except Exception as exc:
        logger.debug("nvidia-smi --query-gpu failed: %s", exc)
        return None

    procs: list[ProcessInfo] = []
    try:
        proc_query = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_idx}",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
        if proc_query.returncode == 0:
            for raw in proc_query.stdout.splitlines():
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) != 3:
                    continue
                try:
                    pid = int(parts[0])
                    used_mb = int(float(parts[2])) if parts[2] not in ("", "[N/A]") else 0
                except ValueError:
                    continue
                procs.append(ProcessInfo(pid=pid, name=parts[1], used_mb=used_mb))
    except Exception as exc:
        logger.debug("nvidia-smi --query-compute-apps failed: %s", exc)

    return PhysicalGpuMemoryInfo(
        device_idx=device_idx,
        total_gb=total_mib / 1024.0,
        used_gb=used_mib / 1024.0,
        free_gb=free_mib / 1024.0,
        processes=procs,
        source="nvidia-smi",
    )


def get_physical_gpu_memory_info(device_idx: int = 0) -> PhysicalGpuMemoryInfo | None:
    """Return total/used/free + compute-process list, or ``None`` on failure."""
    info = _try_pynvml(device_idx)
    if info is not None:
        return info
    return _try_nvidia_smi(device_idx)


def get_compute_processes(device_idx: int = 0) -> list[ProcessInfo]:
    """Convenience wrapper: just the process list."""
    info = get_physical_gpu_memory_info(device_idx)
    return info.processes if info else []
