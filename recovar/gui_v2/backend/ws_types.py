"""WebSocket message type definitions.

Keep in sync with frontend/src/lib/ws_types.ts.
These are intentionally simple — no codegen overhead needed.
"""

from __future__ import annotations

from typing import Any


# Message types sent server → client
LOG_LINE = "log_line"
STATUS_CHANGE = "status_change"
PROGRESS = "progress"
RECONNECT_SYNC = "reconnect_sync"
PING = "ping"

# Message types sent client → server
PONG = "pong"

# All valid message types
ALL_TYPES = {LOG_LINE, STATUS_CHANGE, PROGRESS, RECONNECT_SYNC, PING, PONG}
