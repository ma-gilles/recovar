"""Environment flag parsing shared by dense and local EM engines."""

from __future__ import annotations

import logging
import os


def parse_env_float_or_default(name: str, default: float, *, logger: logging.Logger) -> float:
    """Read a float override, warning through the caller's logger if invalid."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %.3f", name, value, default)
        return default


def parse_env_int_or_default(name: str, default: int, *, logger: logging.Logger) -> int:
    """Read an integer override, warning through the caller's logger if invalid."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default


def parse_int_set(value: str | None) -> set[int] | None:
    """Parse comma/semicolon/whitespace separated integer sets."""

    if not value:
        return None
    parsed = {int(token) for token in value.replace(",", " ").replace(";", " ").split()}
    return parsed or None


def parse_env_int_set(name: str) -> set[int] | None:
    """Parse an integer-set environment variable."""

    return parse_int_set(os.environ.get(name))


def parse_env_nonnegative_int(name: str) -> int | None:
    """Read a non-negative integer; unset or empty values have no override."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


def parse_env_flag(name: str, *, default: bool = False) -> bool:
    """Read a boolean override; unset or blank values use the caller default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}
