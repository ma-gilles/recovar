"""Environment flag parsing shared by dense and local EM engines."""

from __future__ import annotations

import os

_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_AUTO_VALUES = {"", "auto"}


def parse_env_bool(name: str, default: bool) -> bool:
    """Parse a strict boolean environment flag."""
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw in _FALSE_VALUES:
        return False
    if raw in _TRUE_VALUES:
        return True
    raise ValueError(f"{name} must be one of 1/0/true/false")


def parse_env_auto_bool(name: str, default: str = "auto") -> bool | None:
    """Parse an auto/boolean environment flag.

    Returns ``None`` for ``auto`` so callers can keep their existing
    size- or path-dependent default logic explicit.
    """
    raw = os.environ.get(name, default).strip().lower()
    if raw in _AUTO_VALUES:
        return None
    if raw in _FALSE_VALUES:
        return False
    if raw in _TRUE_VALUES:
        return True
    raise ValueError(f"{name} must be one of auto/1/0/true/false")


def parse_env_auto_mode(name: str, default: str = "auto") -> str:
    """Parse an auto/boolean flag into ``auto``, ``off``, or ``on``."""
    parsed = parse_env_auto_bool(name, default)
    if parsed is None:
        return "auto"
    return "on" if parsed else "off"


def parse_int_set(value: str | None) -> set[int] | None:
    """Parse comma/semicolon/whitespace separated integer sets."""

    if not value:
        return None
    parsed = {int(token) for token in value.replace(",", " ").replace(";", " ").split()}
    return parsed or None


def parse_env_int_set(name: str) -> set[int] | None:
    """Parse an integer-set environment variable."""

    return parse_int_set(os.environ.get(name))
