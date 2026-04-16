import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_psutil_declared_for_runtime_installs():
    pyproject = _load_toml(REPO_ROOT / "pyproject.toml")
    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("psutil") for dep in dependencies)


def test_psutil_declared_for_pixi_dev_env():
    pixi = _load_toml(REPO_ROOT / "pixi.toml")
    dependencies = pixi["dependencies"]

    assert "psutil" in dependencies
