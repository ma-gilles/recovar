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
    memory    = "400G"
    time      = "08:00:00"
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

from recovar.gui_v2.backend.config import DEFAULT_LOCAL, DEFAULT_SLURM

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


def resolve_slurm_defaults_layered(
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return SLURM defaults broken out by layer for the Settings UI."""
    builtin = dict(DEFAULT_SLURM)

    user_toml = _load_toml(_user_config_path())
    user_slurm = user_toml.get("slurm", {})
    if not isinstance(user_slurm, dict):
        user_slurm = {}

    project_slurm: dict[str, Any] = {}
    project_path: str | None = None
    if project_dir is not None:
        p = Path(project_dir) / PROJECT_CONFIG_FILENAME
        project_path = str(p)
        project_toml = _load_toml(p)
        ps = project_toml.get("slurm", {})
        if isinstance(ps, dict):
            project_slurm = ps

    effective = dict(builtin)
    effective.update(user_slurm)
    effective.update(project_slurm)

    return {
        "builtin": builtin,
        "user": user_slurm,
        "project": project_slurm,
        "effective": effective,
        "user_config_path": str(_user_config_path()),
        "project_config_path": project_path,
    }


def _save_toml_slurm(path: Path, values: dict[str, Any]) -> None:
    """Update the [slurm] table of a TOML file.

    Preserves both sibling top-level tables (e.g. ``[local]``) and any
    keys within ``[slurm]`` that the Settings form does not send
    (``gpu_resource_spec``, ``template_path``, ``[slurm.template_vars]``,
    etc.): the cleaned form values are merged into the existing table
    rather than replacing it wholesale.
    """
    import tomli_w

    existing: dict[str, Any] = {}
    if path.is_file():
        existing = _load_toml(path)

    clean = {k: v for k, v in values.items() if v != "" and v is not None}
    section = existing.get("slurm")
    if not isinstance(section, dict):
        section = {}
    section.update(clean)
    existing["slurm"] = section

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(existing, f)
    logger.info("Wrote SLURM defaults to %s", path)


def save_user_slurm_defaults(values: dict[str, Any]) -> None:
    """Write SLURM defaults to the user-global config file."""
    _save_toml_slurm(_user_config_path(), values)


def save_project_slurm_defaults(
    project_dir: str | Path, values: dict[str, Any]
) -> None:
    """Write SLURM defaults to the per-project recovar.toml."""
    _save_toml_slurm(Path(project_dir) / PROJECT_CONFIG_FILENAME, values)


def resolve_local_defaults(project_dir: str | Path | None = None) -> dict[str, Any]:
    """Resolve local-execution defaults, same layering as SLURM defaults."""
    merged: dict[str, Any] = dict(DEFAULT_LOCAL)
    # Deep-copy env_vars so we don't mutate the default
    merged["env_vars"] = dict(DEFAULT_LOCAL.get("env_vars", {}))

    user_toml = _load_toml(_user_config_path())
    user_local = user_toml.get("local")
    if isinstance(user_local, dict):
        if "env_vars" in user_local and isinstance(user_local["env_vars"], dict):
            merged["env_vars"].update(user_local["env_vars"])
        merged.update({k: v for k, v in user_local.items() if k != "env_vars"})

    if project_dir is not None:
        project_toml = _load_toml(Path(project_dir) / PROJECT_CONFIG_FILENAME)
        project_local = project_toml.get("local")
        if isinstance(project_local, dict):
            if "env_vars" in project_local and isinstance(project_local["env_vars"], dict):
                merged["env_vars"].update(project_local["env_vars"])
            merged.update({k: v for k, v in project_local.items() if k != "env_vars"})

    return merged


def resolve_local_defaults_layered(
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return local defaults broken out by layer for the Settings UI."""
    builtin = dict(DEFAULT_LOCAL)

    user_toml = _load_toml(_user_config_path())
    user_local = user_toml.get("local", {})
    if not isinstance(user_local, dict):
        user_local = {}

    project_local: dict[str, Any] = {}
    project_path: str | None = None
    if project_dir is not None:
        p = Path(project_dir) / PROJECT_CONFIG_FILENAME
        project_path = str(p)
        project_toml = _load_toml(p)
        pl = project_toml.get("local", {})
        if isinstance(pl, dict):
            project_local = pl

    effective = dict(builtin)
    effective.update(user_local)
    effective.update(project_local)

    return {
        "builtin": builtin,
        "user": user_local,
        "project": project_local,
        "effective": effective,
        "user_config_path": str(_user_config_path()),
        "project_config_path": project_path,
    }


def save_user_local_defaults(values: dict[str, Any]) -> None:
    """Write local-execution defaults to the user-global config file.

    Merges the cleaned form values into the existing ``[local]`` table so
    non-form keys (e.g. ``preallocate_gpu``) are preserved.
    """
    path = _user_config_path()
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = _load_toml(path)
    clean = {k: v for k, v in values.items() if v != "" and v is not None}
    section = existing.get("local")
    if not isinstance(section, dict):
        section = {}
    section.update(clean)
    existing["local"] = section
    import tomli_w
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(existing, f)


def save_project_local_defaults(
    project_dir: str | Path, values: dict[str, Any]
) -> None:
    """Write local-execution defaults to the per-project recovar.toml.

    Merges the cleaned form values into the existing ``[local]`` table so
    non-form keys (e.g. ``preallocate_gpu``) are preserved.
    """
    path = Path(project_dir) / PROJECT_CONFIG_FILENAME
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = _load_toml(path)
    clean = {k: v for k, v in values.items() if v != "" and v is not None}
    section = existing.get("local")
    if not isinstance(section, dict):
        section = {}
    section.update(clean)
    existing["local"] = section
    import tomli_w
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(existing, f)


def project_config_exists(project_dir: str | Path) -> bool:
    return (Path(project_dir) / PROJECT_CONFIG_FILENAME).is_file()
