"""Import boundary between recovar and the EM code that moves to relax (relax split, invariants I1/I2).

Non-EM recovar modules must not import ``recovar.em`` or ``recovar.relion_bind`` -- at top level,
lazily inside functions, through ``importlib`` or as ``mock.patch``/``monkeypatch`` target strings --
so that recovar runs with the EM code deleted (PLAN.md section 1.1).
"""

import ast
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
EM_PREFIXES = ("recovar.em", "recovar.relion_bind")
EM_PATHS = ("recovar/em/", "recovar/relion_bind/")
# The InitialModel CLI moves to relax/commands/initial_model.py in P3 (PLAN.md section 1.1).
PENDING_P3 = {"recovar/commands/initial_model.py"}


def _is_em_module(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in EM_PREFIXES)


def _em_references(path):
    tree = ast.parse((ROOT / path).read_text())
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        found += [(node.lineno, name) for name in names if _is_em_module(name)]
    return found


def _non_em_sources():
    files = subprocess.check_output(["git", "ls-files", "recovar"], cwd=ROOT, text=True).split()
    return [f for f in files if f.endswith(".py") and not f.startswith(EM_PATHS)]


def test_non_em_recovar_never_references_em_modules():
    violations = {}
    for path in _non_em_sources():
        if path in PENDING_P3:
            continue
        refs = _em_references(path)
        if refs:
            violations[path] = refs
    assert violations == {}


def test_pending_p3_allowlist_is_still_needed():
    for path in PENDING_P3:
        assert (ROOT / path).is_file() and _em_references(path), path
