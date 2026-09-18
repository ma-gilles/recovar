"""File and selection fingerprints for standalone diagnostics.

Keep this owner independent of the scientific runtime: importing
``recovar.utils.file_hash`` initializes RECOVAR and JAX first. Runtime callers
retain that packaged helper; these script callers also work outside a RECOVAR
installation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv1a64(text: str) -> int:
    value = 14695981039346656037
    for byte in text.encode():
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value
