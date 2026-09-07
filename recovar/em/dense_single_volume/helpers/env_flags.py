"""Environment flag parsing shared by dense and local EM engines."""

from __future__ import annotations

import os


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
