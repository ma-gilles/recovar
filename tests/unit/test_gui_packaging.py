"""Packaging guards for the web GUI extra.

These protect against the issue #142 class of bug: the GUI backend imports a
package at *runtime* (often lazily — e.g. the ``aiosqlite`` driver only loads
when the first project database is created) that the ``[gui]`` extra forgets
to declare. The package then installs cleanly via ``pip install recovar[gui]``
and only crashes mid-session.

This test checks the *declaration* in ``pyproject.toml``, so it catches a
missing dependency even in a developer environment that happens to have it
installed. It needs no GUI dependencies and runs in ``test-fast``. The
complementary clean-room install check is ``scripts/test_gui_clean_install.sh``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - project requires Python >= 3.11
    tomllib = None


# Runtime dependencies the GUI backend imports. Keyed by PEP 503-normalized
# distribution name. Update this set when the backend grows a new third-party
# runtime import — that is the whole point of the guard.
REQUIRED_GUI_DISTS = {
    "fastapi",  # api framework
    "uvicorn",  # asgi server
    "python-multipart",  # form/file uploads (fastapi dependency at runtime)
    "sqlalchemy",  # orm
    "aiosqlite",  # async sqlite driver for sqlite+aiosqlite (db.py) — issue #142
    "aiofiles",  # async file streaming
    "tomli-w",  # writes user/project config TOML (project_config.py)
    "jinja2",  # user-supplied sbatch templates (executor.py)
}


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate pyproject.toml above this test")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _gui_extra_dists() -> set[str]:
    data = tomllib.loads((_repo_root() / "pyproject.toml").read_text())
    specs = data["project"]["optional-dependencies"]["gui"]
    names = set()
    for spec in specs:
        # "uvicorn[standard]>=0.30" -> "uvicorn"; "tomli_w>=1.0" -> "tomli-w"
        bare = re.split(r"[<>=!~;\[\s]", spec, maxsplit=1)[0]
        names.add(_normalize(bare))
    return names


def _pixi_gui_dists() -> set[str]:
    """Normalized package names from pixi.toml [feature.gui.pypi-dependencies]."""
    data = tomllib.loads((_repo_root() / "pixi.toml").read_text())
    deps = data.get("feature", {}).get("gui", {}).get("pypi-dependencies", {})
    return {_normalize(name) for name in deps}


@pytest.mark.skipif(tomllib is None, reason="needs Python >= 3.11 tomllib")
def test_gui_extra_declares_all_runtime_deps():
    declared = _gui_extra_dists()
    missing = {d for d in REQUIRED_GUI_DISTS if d not in declared}
    assert not missing, (
        "pyproject.toml [project.optional-dependencies].gui is missing runtime "
        f"deps: {sorted(missing)}. The GUI backend imports these (some lazily), "
        "so 'pip install recovar[gui]' must declare them or users hit a "
        "mid-session crash (issue #142). Declared: " + repr(sorted(declared))
    )


@pytest.mark.skipif(tomllib is None, reason="needs Python >= 3.11 tomllib")
def test_pixi_gui_feature_matches_pyproject_extra():
    """The pixi `gui` feature and the pyproject `gui` extra must list the same
    packages. If they drift, one environment installs a dependency the other
    lacks — which is exactly how the canonical pixi env masked issue #142."""
    pyproject = _gui_extra_dists()
    pixi = _pixi_gui_dists()
    assert pyproject == pixi, (
        "pixi.toml [feature.gui.pypi-dependencies] and pyproject.toml "
        "[project.optional-dependencies].gui have drifted.\n"
        f"  only in pyproject: {sorted(pyproject - pixi)}\n"
        f"  only in pixi:      {sorted(pixi - pyproject)}\n"
        "Keep the two lists identical so `pixi run gui` and `pip install "
        "recovar[gui]` install the same packages."
    )
