"""Transparent local staging of MRC particle stacks.

When RECOVAR_CACHE_DIR (or $TMPDIR as fallback) is set, MRC files are
automatically copied to fast local storage on first access.  All subsequent
reads — across every pipeline pass — bypass the network filesystem entirely.

On SLURM clusters, add to your job script::

    #SBATCH --tmp=50G
    export RECOVAR_CACHE_DIR=$TMPDIR

No other changes are needed.  SLURM deletes $TMPDIR on job exit, so there is
no manual cleanup.  For a persistent cache (reused across jobs), set
RECOVAR_CACHE_DIR to a fixed path on a fast local disk.

To disable staging even when $TMPDIR is set::

    export RECOVAR_CACHE_DIR=        # empty string = disabled

Cache validity
--------------
A staged file is reused as long as the source mtime and size match.  If the
source is updated (re-extracted particles, etc.), the old cache entry is left
in place and a new one is written beside it.  Use ``recovar clear_cache`` or
just delete RECOVAR_CACHE_DIR manually to reclaim space.
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
    1. ``RECOVAR_CACHE_DIR`` env var (empty string → disabled)
    2. ``TMPDIR`` env var (standard SLURM/HPC per-job local scratch)
    3. None (no staging)
    """
    val = os.environ.get("RECOVAR_CACHE_DIR")
    if val is not None:
        return val or None   # '' → explicitly disabled
    return os.environ.get("TMPDIR")


def stage_mrc(src_path: str, cache_dir: str) -> str:
    """Copy *src_path* to *cache_dir* if not already staged; return staged path.

    The file is identified by its absolute path, mtime (nanoseconds), and
    size — a fast staleness check that requires no content hashing.

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
        fails for any reason (permission error, disk full, …).
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
        return src_path   # source missing — let the caller raise

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
        logger.debug("Cache hit: %s → %s", os.path.basename(src_path), dest)
        return str(dest)

    size_gb = stat.st_size / 1e9
    logger.info(
        "Staging %.2f GB to local storage: %s",
        size_gb, os.path.basename(src_path),
    )
    t0 = time.monotonic()

    tmp_path: Optional[str] = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=stage_dir, suffix=".tmp")
        os.close(tmp_fd)
        shutil.copy2(src_path, tmp_path)
        os.replace(tmp_path, str(dest))     # atomic
        sentinel.write_text(str(stat.st_mtime_ns))
        tmp_path = None                     # ownership transferred
    except Exception as exc:
        logger.warning(
            "Staging failed (%s); falling back to source: %s", exc, src_path
        )
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return src_path

    elapsed = time.monotonic() - t0
    mb_s = (stat.st_size / 1e6) / elapsed if elapsed > 0 else 0
    logger.info(
        "Staged %s in %.1fs (%.0f MB/s)", os.path.basename(src_path), elapsed, mb_s
    )
    return str(dest)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_key(abs_path: str, stat: os.stat_result) -> str:
    """Short, collision-resistant key derived from path, mtime, and size."""
    raw = f"{abs_path}:{stat.st_mtime_ns}:{stat.st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]
