"""Unit tests for the ``recovar gui`` launcher helpers.

These cover the "just works" launch behaviour (auto-select a free port, only
auto-open a browser for a genuine local session). They import only the launcher
module, which uses the standard library at import time — no GUI extras needed,
so they run in ``test-fast``.
"""

from __future__ import annotations

import socket

from recovar.commands import gui


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pick_port_returns_requested_when_free():
    port = _free_port()
    assert gui._pick_port("127.0.0.1", port) == port


def test_pick_port_rolls_to_next_free_when_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        busy = occupied.getsockname()[1]
        occupied.listen()
        picked = gui._pick_port("127.0.0.1", busy)
    assert picked != busy


def test_browser_not_opened_when_disabled_or_remote(monkeypatch):
    import webbrowser

    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))

    # Disabled explicitly, and bound to a non-loopback host: both must no-op
    # (and must not even schedule a deferred open).
    gui._maybe_open_browser("http://x", "127.0.0.1", no_browser=True)
    gui._maybe_open_browser("http://x", "0.0.0.0", no_browser=False)
    assert opened == []
