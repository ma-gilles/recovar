"""Per-project ``recovar.toml`` loader.

Resolution order for SLURM defaults (lowest precedence first):

    1. Built-in :data:`recovar.gui_v2.backend.config.DEFAULT_SLURM`.
    2. User-global ``~/.config/recovar/config.toml`` (or
       ``$XDG_CONFIG_HOME/recovar/config.toml``).
    3. Project-local ``<project_dir>/recovar.toml``.
    4. Per-job form override (sent in the submit body — handled by callers,
       not here).

Each layer can specify the same keys; later layers override earlier ones.

TOML structure (all keys optional)::

    [slurm]
    partition = "gpu"
    account   = "myacct"
    gpus      = 1
    cpus      = 4
    memory    = "300G"
    time      = "12:00:00"
    gpu_resource_spec = "--gres=gpu:{gpus}"
    template_path     = "/abs/path/to/my_template.sh"

    # Custom variables available to the Jinja2 template.
    [slurm.template_vars]
    qos = "high"
    constraint = "intel"
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from recovar.gui_v2.backend.config import DEFAULT_SLURM

logger = logging.getLogger(__name__)

PROJECT_CONFIG_FILENAME = "recovar.toml"
USER_CONFIG_FILENAME = "config.toml"


def _user_config_path() -> Path:
    """Return ``$XDG_CONFIG_HOME/recovar/config.toml`` or
    ``~/.config/recovar/config.toml``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "recovar" / USER_CONFIG_FILENAME


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}


def resolve_slurm_defaults(project_dir: str | Path | None = None) -> dict[str, Any]:
    """Resolve SLURM defaults for a project.

    Layers (later overrides earlier):

    1. ``DEFAULT_SLURM`` (built-in).
    2. User-global ``~/.config/recovar/config.toml`` ``[slurm]`` section.
    3. Project ``<project_dir>/recovar.toml`` ``[slurm]`` section, if
       ``project_dir`` is provided.

    Returns a dict suitable for ``**``-splatting into ``_render_sbatch_script``.
    Unknown keys in any TOML are passed through; the renderer ignores
    unknown ``**_extra``.
    """
    merged: dict[str, Any] = dict(DEFAULT_SLURM)

    user_toml = _load_toml(_user_config_path())
    user_slurm = user_toml.get("slurm")
    if isinstance(user_slurm, dict):
        merged.update(user_slurm)

    if project_dir is not None:
        project_toml = _load_toml(Path(project_dir) / PROJECT_CONFIG_FILENAME)
        project_slurm = project_toml.get("slurm")
        if isinstance(project_slurm, dict):
            merged.update(project_slurm)

    return merged


def project_config_exists(project_dir: str | Path) -> bool:
    return (Path(project_dir) / PROJECT_CONFIG_FILENAME).is_file()
