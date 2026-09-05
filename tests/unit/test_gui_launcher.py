"""Unit tests for the ``recovar gui`` launcher helpers.

These cover the "just works" launch behaviour (auto-select a free port, only
auto-open a browser for a genuine local session). They import only the launcher
module, which uses the standard library at import time — no GUI extras needed,
so they run in ``test-fast``.
"""

from __future__ import annotations

import pytest
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


def test_port_probe_sets_reuseaddr(monkeypatch):
    """The probe must not read a TIME_WAIT socket as busy.

    uvicorn binds with SO_REUSEADDR, so a probe without it is stricter than
    the real bind: restarting the server would report the port busy and the
    launch would drift to another one, silently breaking an SSH tunnel.
    """
    seen = {}

    class Recording(socket.socket):
        def setsockopt(self, level, optname, value):  # noqa: D102
            if level == socket.SOL_SOCKET and optname == socket.SO_REUSEADDR:
                seen["reuseaddr"] = value
            return super().setsockopt(level, optname, value)

    monkeypatch.setattr(gui.socket, "socket", Recording)
    gui._port_is_free("127.0.0.1", _free_port())

    assert seen.get("reuseaddr") == 1


def test_port_is_free_reports_a_listening_port_as_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy = occupied.getsockname()[1]
        assert gui._port_is_free("127.0.0.1", busy) is False
    assert gui._port_is_free("127.0.0.1", busy) is True


def test_default_port_drifts_when_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy = occupied.getsockname()[1]
        assert gui._resolve_port("127.0.0.1", busy, explicit=False) != busy


def test_explicit_port_is_refused_rather_than_moved():
    """--port names the port being forwarded: moving it looks like a dead GUI."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy = occupied.getsockname()[1]
        with pytest.raises(gui.PortUnavailableError):
            gui._resolve_port("127.0.0.1", busy, explicit=True)


def test_explicit_free_port_is_used_as_is():
    port = _free_port()
    assert gui._resolve_port("127.0.0.1", port, explicit=True) == port
