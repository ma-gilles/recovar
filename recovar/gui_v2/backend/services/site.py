"""Probe the machine the GUI server runs on.

Everything site-specific that the GUI needs is *measured* here rather than
assumed: which filesystems are shared, which SLURM partitions have GPUs, how
many threads the login node lets one user create, and whether the custom CUDA
kernel is built somewhere compute nodes can read. No cluster names, partition
names or mount points are hard-coded.

Probes are cheap, cached for the life of the server, and never raise: a probe
that cannot run returns ``None`` / an empty result and the caller reports it.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import resource
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Filesystem types that are visible from every node of a cluster.
_SHARED_FS_TYPES = frozenset(
    {
        "nfs",
        "nfs4",
        "lustre",
        "gpfs",
        "ceph",
        "fuse.ceph",
        "beegfs",
        "wekafs",
        "panfs",
        "pvfs2",
        "orangefs",
        "glusterfs",
        "fuse.glusterfs",
        "cifs",
        "smb3",
        "afs",
        "vastnfs",
        "fhgfs",
    }
)
# FUSE mounts that are per-session plumbing, not storage. Any other FUSE
# filesystem is taken to be cluster storage (CephFS, for one, is commonly
# mounted through FUSE and then reports the bare type ``fuse``).
_LOCAL_FUSE_TYPES = frozenset(
    {"fuse.portal", "fuse.gvfsd-fuse", "fuse.lxcfs", "fuse.snapfuse", "fuse.vmware-vmblock", "fusectl"}
)


def _is_shared_fs(kind: str) -> bool:
    if kind in _SHARED_FS_TYPES:
        return True
    return (kind == "fuse" or kind.startswith("fuse.")) and kind not in _LOCAL_FUSE_TYPES


# Filesystem types that live in RAM or on one node only.
_NODE_LOCAL_FS_TYPES = frozenset({"tmpfs", "ramfs", "devtmpfs", "overlay", "squashfs"})


# ---------------------------------------------------------------------------
# Process / thread limits
# ---------------------------------------------------------------------------


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None  # "max" → no limit


def _cgroup_pids_limit() -> int | None:
    """Smallest ``pids.max`` on the path from this process's cgroup to the root."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    limits: list[int] = []
    for line in lines:
        _, controllers, rel = line.split(":", 2)
        if controllers == "":  # cgroup v2
            base = Path("/sys/fs/cgroup")
        elif "pids" in controllers.split(","):  # cgroup v1
            base = Path("/sys/fs/cgroup/pids")
        else:
            continue
        node = base / rel.lstrip("/")
        while True:
            limit = _read_int(node / "pids.max")
            if limit is not None:
                limits.append(limit)
            if node == base or base not in node.parents:
                break
            node = node.parent
    return min(limits) if limits else None


def process_limit() -> int | None:
    """Max number of processes+threads this user may run here, or ``None`` if unlimited.

    Login nodes commonly cap this (systemd ``TasksMax`` or ``ulimit -u``). Numeric
    libraries start one thread per core, so on a many-core login node a low cap
    kills the server with no error message.
    """
    limits = []
    cgroup = _cgroup_pids_limit()
    if cgroup is not None:
        limits.append(cgroup)
    soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY:
        limits.append(soft)
    return min(limits) if limits else None


def thread_budget(limit: int | None, n_cpus: int) -> int | None:
    """Cores the server should restrict itself to, or ``None`` if no cap is needed.

    Each visible core costs several threads (BLAS, XLA, the event loop's pool),
    and the limit is shared with the user's shells and editors.
    """
    if limit is None or limit >= 6 * n_cpus + 128:
        return None
    return max(1, min(8, limit // 32))


# ---------------------------------------------------------------------------
# Filesystems
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _mounts() -> tuple[tuple[str, str], ...]:
    """``(mount_point, fs_type)`` pairs, longest mount point first."""
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return ()
    mounts = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            # /proc/mounts escapes spaces as \040.
            mounts.append((parts[1].replace("\\040", " "), parts[2]))
    return tuple(sorted(mounts, key=lambda m: len(m[0]), reverse=True))


def fs_type(path: str | os.PathLike[str]) -> str | None:
    """Filesystem type holding *path* (resolved), or ``None`` if unknown."""
    resolved = str(Path(path).expanduser().resolve())
    for mount_point, kind in _mounts():
        if resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/"):
            return kind
    return None


def is_shared_path(path: str | os.PathLike[str]) -> bool | None:
    """True if *path* is on a cluster-wide filesystem, False if node-local, None if unknown."""
    kind = fs_type(path)
    if kind is None:
        return None
    if _is_shared_fs(kind):
        return True
    if kind in _NODE_LOCAL_FS_TYPES:
        return False
    return None  # a local disk: shared on a workstation, not on a cluster


def shared_filesystem_roots() -> list[str]:
    """Top-level directories under which this machine's shared storage lives.

    Normally the mount point of each network/parallel filesystem. Where storage
    is automounted on demand (``/hpc/projects/<group>`` appearing only once
    someone touches it), the automount root is returned instead: enumerating
    what happens to be mounted right now would miss the directory the user
    wants next.
    """
    mounts = set(_mounts())
    automount_roots = sorted((mp for mp, kind in mounts if kind == "autofs"), key=len)

    def _root_for(mount_point: str) -> str:
        for auto in automount_roots:  # shortest first → outermost automount
            if mount_point.startswith(auto.rstrip("/") + "/"):
                return auto
        return mount_point

    points = sorted({_root_for(mp) for mp, kind in mounts if _is_shared_fs(kind) and mp != "/"})
    roots: list[str] = []
    for mp in points:
        if not any(mp.startswith(r.rstrip("/") + "/") for r in roots):
            roots.append(mp)
    return roots


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------


@dataclass
class Partition:
    name: str
    is_default: bool = False
    gpus: bool = False
    up: bool = False
    max_node_memory_mb: int = 0


def parse_sinfo(output: str) -> list[Partition]:
    """Parse ``sinfo -h -o '%P|%G|%m|%a'`` (one line per partition/node group)."""
    by_name: dict[str, Partition] = {}
    for line in output.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 4:
            continue
        raw_name, gres, memory, avail = fields
        name = raw_name.rstrip("*")
        part = by_name.setdefault(name, Partition(name=name))
        part.is_default |= raw_name.endswith("*")
        part.up |= avail == "up"
        if "gpu" in gres.lower():
            part.gpus = True
            mem = memory.rstrip("+")
            if mem.isdigit():
                part.max_node_memory_mb = max(part.max_node_memory_mb, int(mem))
    return list(by_name.values())


@functools.lru_cache(maxsize=1)
def slurm_partitions() -> tuple[Partition, ...]:
    if shutil.which("sinfo") is None:
        return ()
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-o", "%P|%G|%m|%a"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("sinfo failed: %s", exc)
        return ()
    if result.returncode != 0:
        logger.warning("sinfo failed: %s", result.stderr.strip())
        return ()
    return tuple(parse_sinfo(result.stdout))


def _candidate_order(partitions: list[Partition] | tuple[Partition, ...]) -> list[Partition]:
    """GPU partitions that are up, most likely choice first: the scheduler's
    default, then one literally named ``gpu``, then the rest as listed."""
    gpu_parts = [p for p in partitions if p.gpus and p.up]
    return sorted(gpu_parts, key=lambda p: (not p.is_default, p.name != "gpu"))


def recommend_partition(partitions: list[Partition] | tuple[Partition, ...]) -> Partition | None:
    """Best guess from ``sinfo`` alone at where a GPU job should go, or ``None``."""
    candidates = _candidate_order(partitions)
    return candidates[0] if candidates else None


def _fitting_memory(part: Partition, builtin_memory: str) -> str:
    """*builtin_memory*, lowered if the partition's largest GPU node has less."""
    node_gb = part.max_node_memory_mb // 1024
    if node_gb and node_gb < _memory_gb(builtin_memory):
        # Leave headroom: nodes advertise slightly less than nominal RAM and
        # sites often reserve some for the OS.
        return f"{max(1, int(node_gb * 0.9))}G"
    return builtin_memory


def _sacctmgr(*args: str) -> list[str]:
    try:
        result = subprocess.run(["sacctmgr", "-nP", "show", *args], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else []


@functools.lru_cache(maxsize=1)
def slurm_accounts() -> tuple[str, ...]:
    """Accounts this user may charge jobs to, default account first."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user or shutil.which("sacctmgr") is None:
        return ()
    names = _sacctmgr("user", user, "format=defaultaccount")
    names += [line.split("|")[0] for line in _sacctmgr("assoc", f"user={user}", "format=account")]
    return tuple(dict.fromkeys(n for n in names if n))


def sbatch_test(partition: str, memory: str, account: str = "") -> tuple[bool | None, str]:
    """Ask the scheduler whether it would take a 1-GPU job with these settings.

    Uses ``sbatch --test-only``, which validates the request (partition access,
    account, node configuration) without queueing anything. Returns
    ``(accepted, scheduler_message)``; ``accepted`` is ``None`` when the
    question could not be asked.
    """
    cmd = ["sbatch", "--test-only", "--gres=gpu:1", "--ntasks=1", f"--mem={memory}", "--time=00:10:00"]
    if partition:
        cmd.append(f"--partition={partition}")
    if account:
        cmd.append(f"--account={account}")
    cmd += ["--wrap", "true"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("sbatch --test-only failed to run: %s", exc)
        return None, str(exc)
    message = " ".join(result.stderr.split())
    if result.returncode != 0:
        logger.info(
            "sbatch --test-only rejected partition=%s account=%s mem=%s: %s",
            partition or "<default>",
            account or "<default>",
            memory,
            message,
        )
    return result.returncode == 0, message


def sbatch_accepts(partition: str, memory: str, account: str = "") -> bool | None:
    return sbatch_test(partition, memory, account)[0]


@functools.lru_cache(maxsize=4)
def detected_slurm_defaults(builtin_memory: str) -> dict[str, str]:
    """SLURM settings measured from this site, for keys the user has not set.

    The scheduler is asked whether it accepts a GPU job, trying in order: no
    account, then each of the user's accounts (sites where a user has several
    require one); and for each, no partition (many sites route GPU jobs
    themselves), then each GPU partition. The first accepted combination wins;
    only values that differ from the built-in defaults are returned:

    * ``account`` — when the site refuses jobs without one.
    * ``partition`` — when a blank partition is rejected (typically because the
      default partition is CPU-only).
    * ``memory`` — when the built-in request exceeds the largest GPU node.

    If the scheduler cannot be asked, falls back to the ``sinfo`` listing.
    """
    partitions = slurm_partitions()
    candidates = _candidate_order(partitions)
    if not candidates:
        return {}

    default = next((p for p in partitions if p.is_default), None)
    default_memory = _fitting_memory(default, builtin_memory) if default and default.gpus else builtin_memory
    trials = [("", default_memory)] + [(p.name, _fitting_memory(p, builtin_memory)) for p in candidates]
    chosen: tuple[str, str, str] | None = None
    for account in ("", *slurm_accounts()):
        for partition, memory in trials:
            verdict = sbatch_accepts(partition, memory, account)
            if verdict is None:  # cannot ask: guess from sinfo
                best = candidates[0]
                chosen = ("", "" if best.is_default else best.name, _fitting_memory(best, builtin_memory))
                break
            if verdict:
                chosen = (account, partition, memory)
                break
        if chosen is not None:
            break
    if chosen is None:
        logger.warning(
            "The scheduler rejected a 1-GPU test job for every account and GPU partition; set SLURM options in Settings."
        )
        return {}

    detected: dict[str, str] = {}
    if chosen[0]:
        detected["account"] = chosen[0]
    if chosen[1]:
        detected["partition"] = chosen[1]
    if chosen[2] != builtin_memory:
        detected["memory"] = chosen[2]
    return detected


def _memory_gb(spec: str) -> float:
    """Convert a SLURM memory string (``400G``, ``64000M``, ``1T``, ``4000``) to GB."""
    spec = spec.strip().upper().removesuffix("B")
    units = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
    if spec and spec[-1] in units:
        return float(spec[:-1]) * units[spec[-1]]
    return float(spec) / 1024  # bare number = MB


# ---------------------------------------------------------------------------
# Custom CUDA kernel
# ---------------------------------------------------------------------------


@dataclass
class CudaKernel:
    disabled: bool = False  # RECOVAR_DISABLE_CUDA opt-in
    path: str | None = None  # the built library, if one exists
    searched: list[str] = field(default_factory=list)
    package_path: str | None = None  # next to the installed package
    nvcc: str | None = None
    probe_error: str | None = None


_KERNEL_PROBE = """
import json, shutil
from recovar import cuda_backproject as cb
existing = cb._existing_lib_path()
print(json.dumps({
    "disabled": not cb.custom_cuda_requested(),
    "path": str(existing) if existing else None,
    "searched": [str(p) for p in cb._candidate_lib_paths()],
    "package_path": str(cb._PACKAGE_LIB_PATH),
    "nvcc": shutil.which("nvcc"),
}))
"""


@functools.lru_cache(maxsize=1)
def cuda_kernel() -> CudaKernel:
    """Locate the custom CUDA kernel exactly the way a job would.

    Runs in a child interpreter so the lookup uses the pipeline's own code
    without loading JAX into the server process.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _KERNEL_PROBE],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            # The last line of a traceback is the error itself.
            return CudaKernel(probe_error=(result.stderr.strip().splitlines() or ["probe failed"])[-1])
        return CudaKernel(**json.loads(result.stdout.strip().splitlines()[-1]))
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as exc:
        return CudaKernel(probe_error=str(exc))


def reset_caches() -> None:
    """Forget cached probes (after the user fixes something, and in tests)."""
    _mounts.cache_clear()
    slurm_partitions.cache_clear()
    slurm_accounts.cache_clear()
    detected_slurm_defaults.cache_clear()
    cuda_kernel.cache_clear()


def partitions_as_dicts() -> list[dict]:
    return [asdict(p) for p in slurm_partitions()]
