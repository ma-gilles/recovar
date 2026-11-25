"""Readiness checks for the machine the GUI server runs on.

One list of checks feeds three places: ``recovar gui --check``, the server log
at startup, and ``GET /api/system/preflight`` (shown as a banner in the UI).
Each check states what is wrong *and the command that fixes it*, so a user on
a cluster we have never seen is not left with a job that fails after queueing.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import threading
import time
from dataclasses import dataclass

from recovar.gui_v2.backend.config import DEFAULT_SLURM
from recovar.gui_v2.backend.services import site

logger = logging.getLogger(__name__)

OK, WARN, ERROR = "ok", "warn", "error"

# Set by ``recovar gui`` when it restricted itself to a few cores.
CORE_CAP_ENV = "RECOVAR_GUI_CORE_CAP"


@dataclass
class Check:
    id: str
    level: str  # ok | warn | error
    title: str
    detail: str
    fix: str = ""


def _check_process_limit() -> Check:
    limit = site.process_limit()
    n_cpus = os.cpu_count() or 1
    cap = os.environ.get(CORE_CAP_ENV)
    if limit is None or site.thread_budget(limit, n_cpus) is None:
        detail = "no restrictive limit" if limit is None else f"limit {limit} is ample for {n_cpus} cores"
        return Check("process_limit", OK, "Process limit", detail)
    if cap:
        return Check(
            "process_limit",
            OK,
            "Process limit",
            f"this node allows {limit} processes/threads per user; the GUI restricted itself to {cap} cores. "
            "Jobs run with the Local executor on this node share that restriction — prefer SLURM here.",
        )
    return Check(
        "process_limit",
        ERROR,
        "Process limit",
        f"this node allows only {limit} processes/threads per user but has {n_cpus} cores; "
        "numeric libraries start one thread per core and the server can be killed without an error.",
        "Start the server with `recovar gui` (it caps its own threads), not `python -m recovar.gui_v2.backend.main`.",
    )


def _check_scheduler() -> Check:
    if shutil.which("sbatch") is None:
        other = next((c for c in ("qsub", "bsub") if shutil.which(c)), None)
        if other:
            return Check(
                "scheduler",
                WARN,
                "Scheduler",
                f"found `{other}` but not `sbatch`: only SLURM submission is built in, so jobs would run "
                "on this node with the Local executor.",
                "Run the GUI inside an interactive GPU allocation, or submit the commands shown in each "
                "job's Parameters tab through your scheduler.",
            )
        return Check("scheduler", OK, "Scheduler", "no SLURM: jobs run on this machine (Local executor)")

    partitions = site.slurm_partitions()
    if not partitions:
        return Check(
            "scheduler",
            WARN,
            "Scheduler",
            "`sbatch` is available but `sinfo` returned no partitions, so GPU partitions could not be detected.",
            "Set the partition (and account, if required) in Settings.",
        )
    target = site.recommend_partition(partitions)
    if target is None:
        names = ", ".join(p.name for p in partitions)
        return Check(
            "scheduler",
            WARN,
            "Scheduler",
            f"no partition advertises GPUs (partitions: {names}).",
            "Set the GPU partition and, if your site uses a different GPU flag, `gpu_resource_spec` in Settings.",
        )

    from recovar.gui_v2.backend.services.project_config import resolve_slurm_defaults

    effective = resolve_slurm_defaults()
    accounts = site.slurm_accounts()
    partition, account, memory = (str(effective.get(k) or "") for k in ("partition", "account", "memory"))
    accepted, message = site.sbatch_test(partition, memory, account)
    if accepted is False:
        return Check(
            "scheduler",
            ERROR,
            "Scheduler",
            f"the scheduler refuses a 1-GPU job with the current settings (partition `{partition or 'default'}`, "
            f"account `{account or 'default'}`, memory {memory}): {message}",
            "Correct the SLURM settings in Settings. GPU partitions here: "
            + ", ".join(p.name for p in partitions if p.gpus)
            + (f"; your accounts: {', '.join(accounts)}" if accounts else "")
            + ".",
        )
    chosen = effective.get("partition") or next((p.name for p in partitions if p.is_default), "")
    known = {p.name: p for p in partitions}
    # A partition setting may be a comma-separated list.
    chosen_parts = [known.get(name.strip()) for name in str(chosen).split(",") if name.strip()]
    gpu_parts = ", ".join(p.name for p in partitions if p.gpus)
    # Only when the scheduler could not be asked: judge the partition from sinfo.
    if accepted is None and chosen_parts and not any(p is not None and p.gpus for p in chosen_parts):
        return Check(
            "scheduler",
            ERROR,
            "Scheduler",
            f"jobs would go to partition `{chosen}`, which has no GPUs. GPU partitions here: {gpu_parts}.",
            f"Set the partition to `{target.name}` in Settings.",
        )
    summary = (
        f"SLURM; jobs go to partition `{chosen}`, account `{effective.get('account') or 'scheduler default'}`, "
        f"memory {effective.get('memory')} (GPU partitions: {gpu_parts})"
    )
    detected_account = site.detected_slurm_defaults(DEFAULT_SLURM["memory"]).get("account")
    if detected_account and effective.get("account") == detected_account and len(accounts) > 1:
        return Check(
            "scheduler",
            WARN,
            "Scheduler",
            f"this site requires an account and you have several ({', '.join(accounts)}); "
            f"jobs will be charged to `{detected_account}`, the first one the scheduler accepted.",
            "Choose the account to charge in Settings.",
        )
    return Check("scheduler", OK, "Scheduler", summary)


def _check_cuda_kernel() -> Check:
    kernel = site.cuda_kernel()
    on_cluster = shutil.which("sbatch") is not None
    title = "CUDA kernel"
    if kernel.probe_error:
        return Check("cuda_kernel", WARN, title, f"could not check the custom CUDA kernel: {kernel.probe_error}")
    if kernel.disabled:
        return Check(
            "cuda_kernel",
            WARN,
            title,
            "RECOVAR_DISABLE_CUDA is set: jobs use the much slower JAX path.",
            "Unset RECOVAR_DISABLE_CUDA and run `recovar build_custom_cuda`.",
        )
    if kernel.path:
        if on_cluster and site.is_shared_path(kernel.path) is False:
            return Check(
                "cuda_kernel",
                ERROR,
                title,
                f"built at {kernel.path}, which is node-local storage ({site.fs_type(kernel.path)}): "
                "compute nodes cannot read it and jobs will abort.",
                f"recovar build_custom_cuda --output {kernel.package_path}",
            )
        return Check("cuda_kernel", OK, title, f"{kernel.path} (exported to every job as RECOVAR_CUDA_LIB)")

    # Build next to the package when the default cache is not visible cluster-wide.
    cache_path = next((p for p in kernel.searched if p != kernel.package_path), None)
    build = "recovar build_custom_cuda"
    if on_cluster and cache_path and site.is_shared_path(os.path.dirname(os.path.dirname(cache_path))) is False:
        build += f" --output {kernel.package_path}"
    if kernel.nvcc:
        return Check(
            "cuda_kernel",
            ERROR,
            title,
            "the custom CUDA kernel is not built; GPU jobs abort without it.",
            f"{build}   (then restart the GUI)",
        )
    return Check(
        "cuda_kernel",
        ERROR,
        title,
        "the custom CUDA kernel is not built and no CUDA compiler (`nvcc`) is on PATH; GPU jobs abort without it.",
        f"Load your site's CUDA toolkit (try `module avail cuda`, then `module load <name>`), run `{build}`, "
        "then restart the GUI.",
    )


def _check_gpu() -> Check:
    if shutil.which("sbatch") is not None:
        return Check("gpu", OK, "Local GPU", "not required: jobs can be submitted to SLURM")
    if shutil.which("nvidia-smi") is None:
        return Check(
            "gpu",
            ERROR,
            "Local GPU",
            "no NVIDIA GPU detected (nvidia-smi not found) and no SLURM to submit to.",
            "Run the GUI on a machine with an NVIDIA GPU, or on a SLURM login node.",
        )
    return Check("gpu", OK, "Local GPU", "nvidia-smi found")


def _check_disk() -> Check:
    home = os.path.expanduser("~")
    try:
        free_gb = shutil.disk_usage(home).free / 1024**3
    except OSError as exc:
        return Check("disk", WARN, "Home directory", f"cannot stat {home}: {exc}")
    if free_gb < 2:
        return Check(
            "disk",
            WARN,
            "Home directory",
            f"only {free_gb:.1f} GB free in {home}; settings and the kernel cache are written there.",
            "Free space, or set XDG_CONFIG_HOME / RECOVAR_CUDA_CACHE_DIR to a larger filesystem.",
        )
    return Check("disk", OK, "Home directory", f"{free_gb:.0f} GB free in {home}")


# The checks ask the scheduler and start a child interpreter: seconds of work
# that every page load would otherwise repeat.
_CACHE_SECONDS = 300.0
_cache: tuple[float, list[Check]] | None = None
_cache_lock = threading.Lock()


def cached_checks(refresh: bool = False) -> list[Check]:
    """:func:`run_checks`, at most once per ``_CACHE_SECONDS`` unless ``refresh``."""
    global _cache
    with _cache_lock:
        if refresh or _cache is None or time.monotonic() - _cache[0] > _CACHE_SECONDS:
            _cache = (time.monotonic(), run_checks())
        return _cache[1]


def invalidate() -> None:
    """Forget the cached checks (settings they depend on changed)."""
    global _cache
    with _cache_lock:
        _cache = None


def run_checks() -> list[Check]:
    return [
        _check_process_limit(),
        _check_scheduler(),
        _check_gpu(),
        _check_cuda_kernel(),
        _check_disk(),
    ]


def log_report(checks: list[Check]) -> None:
    for check in checks:
        if check.level == OK:
            logger.info("Preflight %s: %s", check.title, check.detail)
            continue
        log = logger.error if check.level == ERROR else logger.warning
        log("Preflight %s: %s%s", check.title, check.detail, f" Fix: {check.fix}" if check.fix else "")


def tunnel_hint(port: int) -> str:
    """The ``ssh -L`` command for reaching this server from a laptop.

    Names *this* node: cluster login addresses are often load-balanced across
    several nodes, and a tunnel to a different one reaches nothing.
    """
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "<user>"
    return f"ssh -L {port}:localhost:{port} {user}@{socket.getfqdn()}"


TUNNEL_NOTE = (
    "If that host name does not resolve from your computer, use the address you normally ssh to, "
    "and check that it lands on this same node (`hostname`): login addresses can map to several nodes."
)
