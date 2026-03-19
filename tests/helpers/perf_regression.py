"""Performance baseline tracking for long/isolated tests.

Saves timing, GPU memory, hardware info, and code version to separate
perf baseline files.  Tests WARN (not fail) on performance regressions.

Quality baselines (all_scores*.json) are from OLD code and never auto-updated.
Perf baselines (perf_baseline*.json) are from NEW code and can be regenerated.
"""

import json
import logging
import os
import platform
import subprocess
import time
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# Regression thresholds — warn if exceeded (same hardware assumed)
WALL_TIME_TOL = 0.10      # 10% slower
GPU_MEMORY_TOL = 0.10     # 10% more GPU memory
CPU_MEMORY_TOL = 0.10     # 10% more CPU memory


def get_code_version():
    """Return git commit hash and dirty flag."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL
        ) != 0
        return f"{commit}{'*' if dirty else ''}"
    except Exception:
        return "unknown"


def get_package_versions():
    """Return versions of key packages."""
    versions = {}
    for pkg in ("jax", "jaxlib", "numpy", "scipy", "sklearn"):
        try:
            mod = __import__(pkg)
            versions[pkg] = mod.__version__
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def get_gpu_name():
    """Return GPU device name."""
    try:
        import jax
        devs = jax.devices("gpu")
        if devs:
            return str(devs[0].device_kind)
    except Exception:
        pass
    return "cpu"


def get_hardware_info():
    """Return hardware description."""
    return {
        "gpu_name": get_gpu_name(),
        "cpu": platform.processor() or platform.machine(),
        "hostname": platform.node(),
    }


def perf_snapshot():
    """Capture wall-clock + memory snapshot.

    Includes child process memory (important when tests run pipelines
    as subprocesses via subprocess.run).
    """
    import psutil
    proc = psutil.Process(os.getpid())
    # Include children (subprocess.run spawns child processes)
    try:
        children_rss = sum(c.memory_info().rss for c in proc.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children_rss = 0
    snap = {
        "wall_time": time.monotonic(),
        "cpu_rss_bytes": proc.memory_info().rss + children_rss,
        "gpu_bytes_in_use": 0,
        "gpu_peak_bytes": 0,
    }
    try:
        import jax
        stats = jax.local_devices()[0].memory_stats()
        if stats:
            snap["gpu_bytes_in_use"] = stats.get("bytes_in_use", 0)
            snap["gpu_peak_bytes"] = stats.get("peak_bytes_in_use", 0)
    except Exception:
        pass
    # Also try nvidia-smi for GPU memory (works across processes)
    if snap["gpu_bytes_in_use"] == 0:
        try:
            import subprocess as sp
            result = sp.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                # Take max across GPUs (MiB)
                vals = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
                if vals:
                    gpu_mib = max(vals)
                    snap["gpu_bytes_in_use"] = gpu_mib * 1024 * 1024
                    snap["gpu_peak_bytes"] = max(snap["gpu_peak_bytes"], snap["gpu_bytes_in_use"])
        except Exception:
            pass
    return snap


def stage_perf(before, after, name=""):
    """Compute per-stage performance from two snapshots."""
    wall = after["wall_time"] - before["wall_time"]
    cpu_peak = max(before["cpu_rss_bytes"], after["cpu_rss_bytes"])
    if after["gpu_peak_bytes"] > before["gpu_peak_bytes"]:
        gpu_peak = after["gpu_peak_bytes"]
    else:
        gpu_peak = max(before["gpu_bytes_in_use"], after["gpu_bytes_in_use"])
    return {
        "wall_seconds": round(wall, 2),
        "peak_cpu_memory_gb": round(cpu_peak / 1e9, 3),
        "peak_gpu_memory_gb": round(gpu_peak / 1e9, 3),
    }


def build_perf_record(stages: dict) -> dict:
    """Build a complete perf record with metadata.

    Parameters
    ----------
    stages : dict
        Mapping of stage_name → stage_perf dict (from ``stage_perf()``).

    Returns
    -------
    dict with keys: hardware, code_version, package_versions, stages.
    """
    return {
        "hardware": get_hardware_info(),
        "code_version": get_code_version(),
        "package_versions": get_package_versions(),
        "stages": stages,
    }


def save_perf_baseline(perf_record: dict, path: str):
    """Save perf record to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(perf_record, f, indent=2, default=str)
    logger.info("Perf baseline saved to %s", path)


def load_perf_baseline(path: str) -> dict | None:
    """Load perf baseline, return None if not found."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compare_perf(
    current: dict,
    baseline: dict,
    wall_tol: float = WALL_TIME_TOL,
    gpu_tol: float = GPU_MEMORY_TOL,
    cpu_tol: float = CPU_MEMORY_TOL,
) -> list[str]:
    """Compare current perf against baseline, return list of warnings.

    Only compares if hardware matches (same GPU).  Returns empty list if
    no regressions detected.
    """
    warns = []

    # Check hardware match
    cur_hw = current.get("hardware", {})
    bl_hw = baseline.get("hardware", {})
    if cur_hw.get("gpu_name") != bl_hw.get("gpu_name"):
        warns.append(
            f"Hardware mismatch: {cur_hw.get('gpu_name')} vs baseline {bl_hw.get('gpu_name')} — skipping perf comparison"
        )
        return warns

    cur_stages = current.get("stages", {})
    bl_stages = baseline.get("stages", {})

    for stage_name in sorted(set(cur_stages) & set(bl_stages)):
        cur_s = cur_stages[stage_name]
        bl_s = bl_stages[stage_name]

        # Skip non-dict entries (e.g. "gpu_name": "NVIDIA A100...")
        if not isinstance(cur_s, dict) or not isinstance(bl_s, dict):
            continue

        # Wall time
        cur_wall = cur_s.get("wall_seconds", 0)
        bl_wall = bl_s.get("wall_seconds", 0)
        if bl_wall > 0:
            wall_delta = (cur_wall - bl_wall) / bl_wall
            if wall_delta > wall_tol:
                warns.append(
                    f"{stage_name}: wall_time {cur_wall:.1f}s vs {bl_wall:.1f}s "
                    f"(+{wall_delta*100:.0f}%, threshold {wall_tol*100:.0f}%)"
                )

        # GPU memory
        cur_gpu = cur_s.get("peak_gpu_memory_gb", 0)
        bl_gpu = bl_s.get("peak_gpu_memory_gb", 0)
        if bl_gpu > 0.1:  # only compare if baseline used meaningful GPU
            gpu_delta = (cur_gpu - bl_gpu) / bl_gpu
            if gpu_delta > gpu_tol:
                warns.append(
                    f"{stage_name}: GPU memory {cur_gpu:.1f}GB vs {bl_gpu:.1f}GB "
                    f"(+{gpu_delta*100:.0f}%, threshold {gpu_tol*100:.0f}%)"
                )

        # CPU memory
        cur_cpu = cur_s.get("peak_cpu_memory_gb", 0)
        bl_cpu = bl_s.get("peak_cpu_memory_gb", 0)
        if bl_cpu > 0.5:  # only compare if baseline used meaningful CPU
            cpu_delta = (cur_cpu - bl_cpu) / bl_cpu
            if cpu_delta > cpu_tol:
                warns.append(
                    f"{stage_name}: CPU memory {cur_cpu:.1f}GB vs {bl_cpu:.1f}GB "
                    f"(+{cpu_delta*100:.0f}%, threshold {cpu_tol*100:.0f}%)"
                )

    return warns


def check_perf_regression(current: dict, baseline_path: str, test_name: str = ""):
    """Compare perf against committed baseline and warn on regression.

    Baselines are stored in the repo and never auto-updated.
    If no baseline exists, the test skips the perf check (does not create one).
    To generate baselines, run with PERF_OVERWRITE_BASELINE=1.
    """
    overwrite = os.environ.get("PERF_OVERWRITE_BASELINE")
    if overwrite:
        save_perf_baseline(current, baseline_path)
        logger.info("Wrote perf baseline: %s", baseline_path)
        return

    baseline = load_perf_baseline(baseline_path)
    if baseline is None:
        logger.info("No perf baseline at %s — skipping perf check", baseline_path)
        return

    warns = compare_perf(current, baseline)
    if warns:
        msg = f"Performance regressions in {test_name}:\n" + "\n".join(f"  - {w}" for w in warns)
        warnings.warn(msg, stacklevel=2)
        logger.warning(msg)
    else:
        logger.info("Perf check passed for %s", test_name)
