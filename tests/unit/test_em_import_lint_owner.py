"""EM sources carry no import-order or unused-import findings."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_em_imports_are_sorted_and_used():
    ruff = Path(sys.executable).parent / "ruff"
    if not ruff.exists():
        pytest.skip("ruff is not installed next to the interpreter")
    result = subprocess.run([str(ruff), "check", "recovar/em", "--select", "I001,F401", "--exclude", "*.ipynb"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:]


def test_em_scripts_and_tests_bind_every_name():
    """A moved definition must be re-imported wherever it is used: no undefined names or syntax errors in scripts, unit tests or EM sources."""
    ruff = Path(sys.executable).parent / "ruff"
    if not ruff.exists():
        pytest.skip("ruff is not installed next to the interpreter")
    result = subprocess.run([str(ruff), "check", "scripts", "tests/unit", "recovar/em", "--select", "F821", "--exclude", "*.ipynb"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:]
