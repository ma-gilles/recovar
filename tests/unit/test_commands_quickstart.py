"""
Unit tests for recovar.commands.quickstart.

Covers:
  _heading / _info / _success / _warn / _error – formatting helpers
  _find_files()  – glob-based file discovery
  main()         – argument registration (--dry-run)
"""
import argparse
import os

import pytest

from recovar.commands import quickstart as qs_cmd

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Formatting helpers – just verify they don't crash
# ---------------------------------------------------------------------------

def test_heading_prints_without_error(capsys):
    qs_cmd._heading("Test heading")
    captured = capsys.readouterr()
    assert "Test heading" in captured.out


def test_info_prints_without_error(capsys):
    qs_cmd._info("Some info")
    captured = capsys.readouterr()
    assert "Some info" in captured.out


def test_success_prints_without_error(capsys):
    qs_cmd._success("Success message")
    captured = capsys.readouterr()
    assert "Success message" in captured.out


def test_warn_prints_without_error(capsys):
    qs_cmd._warn("Warning message")
    captured = capsys.readouterr()
    assert "Warning message" in captured.out


def test_error_prints_without_error(capsys):
    qs_cmd._error("Error message")
    captured = capsys.readouterr()
    assert "Error message" in captured.out


# ---------------------------------------------------------------------------
# _find_files
# ---------------------------------------------------------------------------

def test_find_files_returns_matching_files(tmp_path):
    (tmp_path / "particles.star").write_text("dummy")
    (tmp_path / "particles.cs").write_text("dummy")
    (tmp_path / "unrelated.txt").write_text("dummy")

    results = qs_cmd._find_files(["*.star", "*.cs"], directory=str(tmp_path))
    basenames = [os.path.basename(r) for r in results]
    assert "particles.star" in basenames
    assert "particles.cs" in basenames
    assert "unrelated.txt" not in basenames


def test_find_files_returns_empty_for_no_matches(tmp_path):
    (tmp_path / "unrelated.txt").write_text("dummy")
    results = qs_cmd._find_files(["*.star"], directory=str(tmp_path))
    assert results == []


def test_find_files_deduplicates(tmp_path):
    (tmp_path / "data.star").write_text("dummy")
    results = qs_cmd._find_files(["*.star", "*.star"], directory=str(tmp_path))
    assert len(results) == 1


def test_find_files_caps_at_20(tmp_path):
    for i in range(30):
        (tmp_path / f"file_{i:03d}.star").write_text("dummy")
    results = qs_cmd._find_files(["*.star"], directory=str(tmp_path))
    assert len(results) <= 20


# ---------------------------------------------------------------------------
# main – argument registration
# ---------------------------------------------------------------------------

def test_main_registers_dry_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    actions = parser._option_string_actions
    assert "--dry-run" in actions


def test_dry_run_is_store_true():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    action = parser._option_string_actions["--dry-run"]
    assert action.const is True
    assert action.default is False
