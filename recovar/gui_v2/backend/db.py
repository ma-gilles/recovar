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
import time
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import event, text
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


def _set_wal_mode(dbapi_conn, connection_record):
    """Enable WAL journal mode on every new SQLite connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
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

    # WAL mode via synchronous event (aiosqlite proxies to a background thread)
    @event.listens_for(engine.sync_engine, "connect")
    def on_connect(dbapi_conn, connection_record):
        _set_wal_mode(dbapi_conn, connection_record)

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
