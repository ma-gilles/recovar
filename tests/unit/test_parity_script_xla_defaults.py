"""The RELION parity replay scripts run with the EM XLA defaults.

``scripts/run_k_class_parity.py`` and ``scripts/run_multi_iter_parity.py`` are
EM entry points: the fast and long parity tiers launch them as child
processes. Like ``scripts/run_full_refinement.py`` they must set
``RECOVAR_EM_XLA_DEFAULTS`` before the first import that reaches jax, because
``recovar.jax_config`` reads it when it is imported. Without it their children
ran with XLA autotuning (and its fusion autotuner) on, which is not the
production EM configuration (D2 10345 diagnosis).

The order is checked on the module AST, and the effect in child interpreters
that import each script module and report ``XLA_FLAGS``. The scripts are never
imported into the test process itself.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = (
    REPO / "scripts" / "run_k_class_parity.py",
    REPO / "scripts" / "run_multi_iter_parity.py",
)
MARKER = "RECOVAR_EM_XLA_DEFAULTS"
FLAG = "--xla_gpu_autotune_level=0"


def _is_marker_setdefault(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "setdefault"
        and ast.unparse(func.value) == "os.environ"
        and [ast.unparse(a) for a in call.args] == [repr(MARKER), repr("1")]
    )


def _imported_modules(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    return []


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_marker_is_set_at_module_level_before_any_jax_or_recovar_import(script):
    body = ast.parse(script.read_text()).body
    marker_at = [i for i, node in enumerate(body) if _is_marker_setdefault(node)]
    assert marker_at, f'{script.name} does not call os.environ.setdefault("{MARKER}", "1") at module level'
    for i, node in enumerate(body[: marker_at[0]]):
        reaching = [m for m in _imported_modules(node) if m.split(".")[0] in {"jax", "jaxlib", "recovar"}]
        assert not reaching, (
            f"{script.name} imports {reaching} at statement {i} (line {node.lineno}), before the "
            f"marker at line {body[marker_at[0]].lineno}; XLA_FLAGS would already have been read"
        )


def _xla_flags_after_loading(script: Path, tmp_path: Path, marker: str | None) -> str:
    code = (
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('parity_entry', {str(script)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        "import recovar.jax_config\n"
        "print('XLA_FLAGS_AFTER_IMPORT=' + os.environ.get('XLA_FLAGS', ''), flush=True)\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO),
        "JAX_PLATFORMS": "cpu",
        "RECOVAR_DISABLE_CUDA": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": str(tmp_path),
    }
    if marker is not None:
        env[MARKER] = marker
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("XLA_FLAGS_AFTER_IMPORT=")]
    assert lines, proc.stdout[-2000:]
    return lines[-1].split("=", 1)[1]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_end_to_end_the_script_reaches_xla_with_the_em_default(script, tmp_path):
    assert FLAG in _xla_flags_after_loading(script, tmp_path, marker=None).split()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_end_to_end_an_explicit_zero_still_wins(script, tmp_path):
    assert FLAG not in _xla_flags_after_loading(script, tmp_path, marker="0").split()
