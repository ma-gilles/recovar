"""SQLite database connection and session management.

Each project directory contains its own ``recovar_project.db``.  The GUI
server keeps one engine per open project.  WAL mode is enabled at connection
time for concurrent-read safety.

Write operations use an automatic retry loop to handle transient
``OperationalError: database is locked`` errors (see CLAUDE.md § Database).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from recovar.gui_v2.backend.config import WRITE_RETRY_DELAYS_MS
from recovar.gui_v2.backend.models import Base

logger = logging.getLogger(__name__)

# Module-level registry: project_path -> (engine, sessionmaker)
_engines: dict[str, tuple] = {}


# Filesystem types where SQLite's WAL shared-memory file is unsafe — its
# memory-mapped -wal/-shm files corrupt or fail to lock over the network. On
# these we fall back to a network-safe rollback journal so the project DB just
# works wherever the user put their project (e.g. /scratch/gpfs), with no
# action on their part.
_NETWORK_FS_PREFIXES = (
    "nfs",
    "gpfs",
    "lustre",
    "cifs",
    "smb",
    "fuse.sshfs",
    "fuse.glusterfs",
    "beegfs",
    "ceph",
    "9p",
)


def _is_network_fs(path: Path) -> bool:
    """Best-effort: is *path* on a network filesystem? Reads ``/proc/mounts``
    and matches the longest mount point covering the path. Returns ``False``
    when it can't tell (e.g. no ``/proc/mounts`` on non-Linux)."""
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return False
    target = str(path.resolve())
    best_mount = ""
    best_fstype = ""
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mountpoint, fstype = parts[1], parts[2]
        if target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/"):
            if len(mountpoint) >= len(best_mount):
                best_mount, best_fstype = mountpoint, fstype.lower()
    return any(best_fstype == p or best_fstype.startswith(p) for p in _NETWORK_FS_PREFIXES)


def _choose_journal_mode(db_path: Path) -> str:
    """WAL on local disk (fast, concurrent reads); a network-safe rollback
    journal on NFS/GPFS/Lustre/etc."""
    try:
        return "TRUNCATE" if _is_network_fs(db_path) else "WAL"
    except Exception:  # pragma: no cover - defensive; never block DB open
        return "WAL"


def _apply_pragmas(dbapi_conn, journal_mode: str) -> None:
    """Set journal mode, a lock wait, and foreign keys on a new connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute(f"PRAGMA journal_mode={journal_mode}")
    cursor.execute("PRAGMA busy_timeout=10000")  # wait up to 10s on a lock instead of erroring
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


async def init_db(db_path: Path) -> async_sessionmaker[AsyncSession]:
    """Create (or reuse) an async engine + sessionmaker for *db_path*.

    Creates all tables if they don't exist.  Returns the sessionmaker.
    """
    key = str(db_path.resolve())
    if key in _engines:
        return _engines[key][1]

    url = f"sqlite+aiosqlite:///{key}"
    engine = create_async_engine(url, echo=False)

    journal_mode = _choose_journal_mode(db_path)
    if journal_mode != "WAL":
        logger.warning(
            "Project database is on a network filesystem; using %s journal "
            "instead of WAL (WAL is unsafe over NFS/GPFS/Lustre): %s",
            journal_mode,
            key,
        )

    # Pragmas applied via synchronous event (aiosqlite proxies to a thread).
    @event.listens_for(engine.sync_engine, "connect")
    def on_connect(dbapi_conn, connection_record):
        _apply_pragmas(dbapi_conn, journal_mode)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    _engines[key] = (engine, session_factory)
    logger.info("Database initialized: %s", key)
    return session_factory


async def get_session(db_path: Path) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session for *db_path*, creating the DB if needed."""
    session_factory = await init_db(db_path)
    async with session_factory() as session:
        yield session


async def close_all() -> None:
    """Dispose of all engines (call on shutdown)."""
    for key, (engine, _) in list(_engines.items()):
        await engine.dispose()
    _engines.clear()


def with_write_retry(func):
    """Decorator that retries on ``OperationalError`` (database locked).

    Uses the delays defined in ``config.WRITE_RETRY_DELAYS_MS``.
    Works on both sync and async callables.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        last_exc: OperationalError | None = None
        for attempt, delay_ms in enumerate(WRITE_RETRY_DELAYS_MS):
            try:
                return await func(*args, **kwargs)
            except OperationalError as exc:
                if "database is locked" not in str(exc):
                    raise
                last_exc = exc
                logger.warning(
                    "DB locked (attempt %d/%d), retrying in %dms",
                    attempt + 1,
                    len(WRITE_RETRY_DELAYS_MS),
                    delay_ms,
                )
                await asyncio.sleep(delay_ms / 1000.0)
        # Final attempt — let it raise if it fails
        try:
            return await func(*args, **kwargs)
        except OperationalError as exc:
            if "database is locked" not in str(exc):
                raise
            logger.error("DB locked after all retries")
            raise last_exc  # type: ignore[misc]

    return wrapper
