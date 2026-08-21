"""Regression test for the GUI global exception handler.

The handler returns a ``JSONResponse``; its import was once silently stripped
by the auto-formatter (it was briefly unused between two edits). Registration /
app-build checks do NOT catch that, because the handler body only runs when an
exception actually flows through it. This test fires the handler and asserts the
structured payload, so a missing import (or a malformed envelope) fails loudly.

Needs the GUI deps (fastapi/starlette), which the default pixi env now installs;
skips cleanly otherwise. Does not need httpx.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from starlette.requests import Request  # noqa: E402

from recovar.gui_v2.backend.main import create_app  # noqa: E402


def test_unhandled_exception_returns_structured_json():
    app = create_app()
    handler = app.exception_handlers[Exception]
    req = Request({"type": "http", "method": "GET", "path": "/api/x", "headers": []})

    resp = asyncio.run(handler(req, ValueError("boom")))

    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["error"] == "internal_error"
    assert "boom" in body["detail"]
    assert "hint" in body
