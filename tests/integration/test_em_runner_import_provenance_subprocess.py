"""Subprocess coverage for the import-bound full-refinement entry point."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _subprocess_env(expected_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["JAX_PLATFORMS"] = "cpu"
    env["RECOVAR_EXPECTED_REPO_ROOT"] = str(expected_root)
    return env


@pytest.mark.integration
def test_module_entry_point_checks_concrete_imports(tmp_path):
    command = [sys.executable, "-m", "scripts.run_full_refinement", "--help"]

    accepted = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_env(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "recovar.em.dense_single_volume.iteration_loop=" in accepted.stderr
    assert "recovar.em.dense_single_volume.k_class=" in accepted.stderr
    assert "recovar.em.dense_single_volume.helpers.significance=" in accepted.stderr

    rejected = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "RECOVAR import provenance failure" in rejected.stderr
