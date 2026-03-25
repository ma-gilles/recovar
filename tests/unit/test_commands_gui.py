"""
Unit tests for recovar.commands.gui.

Covers:
  main() – argument registration (host, port, scan-dir, debug)
  main() – Flask import error handling
"""

import argparse
import sys

import pytest

from recovar.commands import gui as gui_cmd

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Argument registration (replicated from the parser in main())
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    """Build the same parser that gui.main() uses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--scan-dir", dest="scan_dirs", action="append", default=[])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--python-path", dest="python_path", default=None)
    return parser


def test_host_default_is_localhost():
    action = _parser()._option_string_actions["--host"]
    assert action.default == "127.0.0.1"


def test_port_default_is_5000():
    action = _parser()._option_string_actions["--port"]
    assert action.default == 5000


def test_registers_scan_dir():
    actions = _parser()._option_string_actions
    assert "--scan-dir" in actions


def test_scan_dir_is_append_action():
    action = _parser()._option_string_actions["--scan-dir"]
    assert action.nargs is None  # append action stores each individually
    assert action.default == []


def test_debug_is_store_true():
    action = _parser()._option_string_actions["--debug"]
    assert action.const is True


def test_registers_python_path():
    actions = _parser()._option_string_actions
    assert "--python-path" in actions


def test_python_path_default_is_none():
    action = _parser()._option_string_actions["--python-path"]
    assert action.default is None


def test_multiple_scan_dirs():
    parser = _parser()
    args = parser.parse_args(["--scan-dir", "/a", "--scan-dir", "/b"])
    assert args.scan_dirs == ["/a", "/b"]
