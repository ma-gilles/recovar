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
    """Save perf record to multi-hardware JSON file.

    The file is a dict keyed by gpu_name.  Existing entries for other
    hardware are preserved; only the current hardware's entry is
    written/overwritten.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gpu_name = perf_record.get("hardware", {}).get("gpu_name", "unknown")

    # Load existing multi-hardware file (or migrate old single-record format)
    existing = _load_multi_hw(path)
    existing[gpu_name] = perf_record

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    logger.info("Perf baseline saved for %s to %s", gpu_name, path)


def _load_multi_hw(path: str) -> dict:
    """Load a multi-hardware baseline file.

    Handles three cases:
      1. File does not exist → empty dict.
      2. Old single-record format (has "hardware" key at top level) →
         migrate to ``{gpu_name: record}`` on the fly.
      3. Already multi-hardware → return as-is.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    # Old format: top-level dict has a "hardware" key
    if "hardware" in data:
        gpu_name = data["hardware"].get("gpu_name", "unknown")
        return {gpu_name: data}
    return data


def load_perf_baseline(path: str, gpu_name: str | None = None) -> dict | None:
    """Load perf baseline for a specific GPU, return None if not found.

    Parameters
    ----------
    path : str
        Path to the baseline JSON file.
    gpu_name : str or None
        GPU to look up.  If None, uses the current GPU.
    """
    if gpu_name is None:
        gpu_name = get_gpu_name()
    multi = _load_multi_hw(path)
    return multi.get(gpu_name)


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

    If no baseline exists for the current hardware, the current perf is
    saved alongside any existing entries for other hardware (which are
    never deleted).  To force-overwrite all entries, set
    PERF_OVERWRITE_BASELINE=1.
    """
    cur_gpu = current.get("hardware", {}).get("gpu_name", "unknown")

    overwrite = os.environ.get("PERF_OVERWRITE_BASELINE")
    if overwrite:
        save_perf_baseline(current, baseline_path)
        logger.info("Wrote perf baseline: %s", baseline_path)
        return

    baseline = load_perf_baseline(baseline_path, gpu_name=cur_gpu)

    # Always print current perf for visibility
    cur_stages = current.get("stages", {})
    lines = [f"\n  Perf: {test_name} ({cur_gpu})"]
    for k, v in cur_stages.items():
        if isinstance(v, dict):
            lines.append(f"    {k:30s} wall={v.get('wall_seconds',0):7.0f}s  "
                         f"gpu={v.get('peak_gpu_memory_gb',0):5.1f}GB  "
                         f"cpu={v.get('peak_cpu_memory_gb',0):5.1f}GB")
    print("\n".join(lines))

    if baseline is None:
        # No baseline for this hardware — add it (preserves other entries)
        save_perf_baseline(current, baseline_path)
        print(f"  No baseline for {cur_gpu} — saved as new baseline entry")
        return

    warns = compare_perf(current, baseline)
    if warns:
        msg = f"PERF REGRESSION in {test_name}:\n" + "\n".join(f"  - {w}" for w in warns)
        print(f"\n  *** {msg}")
        warnings.warn(msg, stacklevel=2)
    else:
        print(f"  Perf OK (within {WALL_TIME_TOL*100:.0f}% of baseline)")
