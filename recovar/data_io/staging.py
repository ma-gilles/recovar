"""Transparent local staging of MRC particle stacks.

Motivation
----------
On HPC clusters with shared parallel filesystems (GPFS, Lustre, ...), reading
a 300K-image particle stack 20 times over a pipeline run takes ~78 min of
wall time at typical ~335 MB/s bandwidth.  The same reads from RAM or a fast
local SSD take ~4 min.

When ``RECOVAR_CACHE_DIR`` is set, every :class:`~recovar.data_io.image_loader.MRCLoader`
silently copies its MRC file to that directory on first access.  Every
subsequent read -- across all pipeline passes, for the lifetime of the job --
goes to the fast local copy.  No code changes are required outside of setting
one environment variable.

Quick start (SLURM)
-------------------
Option A -- RAM-backed (``/dev/shm``, always available, fast as RAM)::

    export RECOVAR_CACHE_DIR=/dev/shm

Option B -- Node-local NVMe (if your cluster supports per-job local scratch)::

    #SBATCH --tmp=50G          # request 50 GB of local scratch
    export RECOVAR_CACHE_DIR=$TMPDIR

Option C -- Any fast local path::

    export RECOVAR_CACHE_DIR=/local/scratch/$(whoami)

Add the ``export`` line to your SLURM job script before calling ``recovar``.
SLURM deletes ``$TMPDIR`` automatically on job exit; for ``/dev/shm`` or a
fixed path, delete ``$RECOVAR_CACHE_DIR/recovar_cache/`` manually when done.

To disable staging even when $TMPDIR is set::

    export RECOVAR_CACHE_DIR=        # empty string = disabled

Measured speedup (D=256, 300K images, 39 GB, A100-SXM4-80GB)
-------------------------------------------------------------
+---------------------------+--------+--------+--------+-----------+
| Condition                 | Load   | Noise  | Mean   | Total     |
+===========================+========+========+========+===========+
| GPFS baseline             | 1.5 s  | 268 s  | 438 s  | 708 s     |
+---------------------------+--------+--------+--------+-----------+
| /dev/shm cold (1st pass)  | 97 s   | 3.3 s  | 107 s  | 207 s     |
+---------------------------+--------+--------+--------+-----------+
| /dev/shm warm (2nd+ pass) | 1.5 s  | 3.3 s  | 107 s  | 112 s     |
+---------------------------+--------+--------+--------+-----------+

**6.3x total speedup** (warm vs baseline); **4.1x mean reconstruction**.
Cold pass pays ~97 s copy cost once; every subsequent pass reads from RAM.
The residual ~107 s mean reconstruction time is the GPU compute floor.

For a 20-pass pipeline the copy cost is amortised: 97 + 19x112 = 2225 s
vs 20x708 = 14160 s baseline -- **6.4x end-to-end speedup**.

Multi-file datasets
-------------------
For datasets stored as many small MRC files (one per micrograph), each unique
file is staged independently the first time it is opened.  Only files that are
actually read get staged.

Cache validity
--------------
A staged file is reused as long as the source absolute path, mtime (ns), and
size all match.  If the source is updated on disk, a new cache entry is
written automatically.  Stale entries accumulate in ``recovar_cache/``; delete
the directory to reclaim space.

Implementation
--------------
The hook is in :meth:`~recovar.data_io.image_loader.MRCLoader.__init__`
which calls ``get_cache_dir()`` and ``stage_mrc()`` to redirect
``self._filepath`` before any I/O.  All read paths (memory-mapped
sequential, seek+fromfile random access, and ``load_all`` eager load) follow
the redirect transparently.

Default behavior
----------------
By default (no env vars set), ``get_cache_dir()`` falls back to ``$TMPDIR``.
On Slurm, ``$TMPDIR`` is typically set to a per-job scratch directory, so
**staging is enabled automatically on Slurm** with no user configuration.
Outside Slurm (e.g. interactive login node), ``$TMPDIR`` is usually
``/tmp``, which may or may not have enough space -- set
``RECOVAR_CACHE_DIR=/dev/shm`` explicitly for RAM-backed staging, or
``RECOVAR_CACHE_DIR=`` (empty) to disable staging entirely.

When a job ends and a new job starts (possibly on a different node), the
old ``$TMPDIR`` is gone, so staging happens again from scratch. This is
by design -- each Slurm job gets fresh local storage.

What gets staged
----------------
Only MRC/MRCS particle stacks are staged, because they are the only
large files read repeatedly (10-20 passes over the data during the
pipeline).  STAR files, pkl files, and other metadata are small
(typically < 1 GB) and are read into memory once, so staging them
would add complexity for negligible benefit.

For one-shot reads (e.g. downsampling full-res images that are only
read once), staging is wasteful.  Pass ``skip_staging=True`` to
``load_images()`` or ``MRCLoader`` to bypass it.  The
``downsample_to_disk`` function does this automatically.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_cache_dir() -> Optional[str]:
    """Return the configured staging directory, or None if disabled.

    Resolution order:
    1. ``RECOVAR_CACHE_DIR`` env var (empty string -> disabled)
    2. ``TMPDIR`` env var (standard SLURM/HPC per-job local scratch)
    3. None (no staging)
    """
    val = os.environ.get("RECOVAR_CACHE_DIR")
    if val is not None:
        return val or None  # '' -> explicitly disabled
    return os.environ.get("TMPDIR")


def stage_mrc(src_path: str, cache_dir: str) -> str:
    """Copy *src_path* to *cache_dir* if not already staged; return staged path.

    The file is identified by its absolute path, mtime (nanoseconds), and
    size -- a fast staleness check that requires no content hashing.

    Two concurrent callers racing on the same file copy to independent
    temp files then ``os.replace`` atomically, so the final staged file
    is always complete.

    Parameters
    ----------
    src_path : str
        Absolute or relative path to the source MRC/MRCS file.
    cache_dir : str
        Root directory for staged files.  ``recovar_cache/`` is created
        inside it.

    Returns
    -------
    str
        Path to the staged file, or *src_path* unchanged if staging
        fails for any reason (permission error, disk full, ...).
    """
    # Skip if source is already inside the cache directory.
    abs_src = os.path.abspath(src_path)
    abs_cache = os.path.abspath(cache_dir)
    if abs_src.startswith(abs_cache + os.sep) or abs_src == abs_cache:
        logger.debug("Source already under cache_dir, skipping: %s", src_path)
        return src_path

    try:
        stat = os.stat(src_path)
    except OSError:
        return src_path  # source missing -- let the caller raise

    key = _cache_key(abs_src, stat)
    suffix = Path(src_path).suffix or ".mrcs"
    stage_dir = Path(abs_cache) / "recovar_cache"
    dest = stage_dir / f"{key}{suffix}"
    sentinel = stage_dir / f"{key}.ok"

    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create cache dir %s (%s); reading from source.", stage_dir, exc)
        return src_path

    if dest.exists() and sentinel.exists():
        logger.debug("Cache hit: %s -> %s", os.path.basename(src_path), dest)
        return str(dest)

    size_gb = stat.st_size / 1e9
    logger.info(
        "Staging %.2f GB to local storage: %s",
        size_gb,
        os.path.basename(src_path),
    )
    t0 = time.monotonic()

    tmp_path: Optional[str] = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=stage_dir, suffix=".tmp")
        os.close(tmp_fd)
        shutil.copy2(src_path, tmp_path)
        os.replace(tmp_path, str(dest))  # atomic
        sentinel.write_text(str(stat.st_mtime_ns))
        tmp_path = None  # ownership transferred
    except Exception as exc:
        logger.warning("Staging failed (%s); falling back to source: %s", exc, src_path)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return src_path

    elapsed = time.monotonic() - t0
    mb_s = (stat.st_size / 1e6) / elapsed if elapsed > 0 else 0
    logger.info("Staged %s in %.1fs (%.0f MB/s)", os.path.basename(src_path), elapsed, mb_s)
    return str(dest)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_key(abs_path: str, stat: os.stat_result) -> str:
    """Short, collision-resistant key derived from path, mtime, and size."""
    raw = f"{abs_path}:{stat.st_mtime_ns}:{stat.st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]
