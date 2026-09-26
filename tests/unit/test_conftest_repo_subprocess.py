"""The pinned child launcher must run the invocation unchanged and refuse a foreign ``recovar``."""

import json
import subprocess
from pathlib import Path

import pytest
from conftest import REPO_IMPORT_ROOT_FAILURE_STATUS, ROOT, repo_python_command, repo_subprocess_env

pytestmark = pytest.mark.unit

_PROBE = "import json, sys, recovar; print(json.dumps([__name__, sys.argv, sys.path[0], recovar.__file__]))"


def _run(*args, env=None, cwd=None):
    return subprocess.run(
        repo_python_command(*args),
        env=repo_subprocess_env() if env is None else env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_code_child_imports_this_checkout_and_keeps_its_exit_status():
    proc = _run("-c", _PROBE + "; sys.exit(3)", "a", "b")
    assert proc.returncode == 3, proc.stderr[-2000:]
    name, argv, _, origin = json.loads(proc.stdout.splitlines()[-1])
    assert (name, argv) == ("__main__", ["-c", "a", "b"])
    assert Path(origin).resolve().is_relative_to(ROOT)


def test_script_and_module_children_run_as_python_would(tmp_path):
    script = tmp_path / "probe_script.py"
    script.write_text(_PROBE + "\n")
    proc = _run(str(script), "x")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.splitlines()[-1])[:3] == ["__main__", [str(script), "x"], str(tmp_path)]

    proc = _run("-m", "probe_script", "y", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    name, argv, _, _ = json.loads(proc.stdout.splitlines()[-1])
    assert (name, argv) == ("__main__", [str(script), "y"])


def test_child_fails_on_a_recovar_from_another_checkout(tmp_path):
    foreign = tmp_path / "other_checkout"
    (foreign / "recovar").mkdir(parents=True)
    (foreign / "recovar" / "__init__.py").write_text("")
    # What a shared environment's editable finder amounts to when the pin is missing.
    env = repo_subprocess_env()
    env["PYTHONPATH"] = str(foreign)
    # Away from the repo root, so the working directory cannot supply the import either.
    proc = _run("-c", "import recovar; raise SystemExit(3)", env=env, cwd=tmp_path)
    assert proc.returncode == REPO_IMPORT_ROOT_FAILURE_STATUS, proc.stderr[-2000:]
    assert f"child imported recovar from {foreign / 'recovar' / '__init__.py'}" in proc.stderr


def test_child_fails_when_it_never_imports_recovar():
    proc = _run("-c", "pass")
    assert proc.returncode == REPO_IMPORT_ROOT_FAILURE_STATUS, proc.stderr[-2000:]
    assert "child imported recovar from None" in proc.stderr
