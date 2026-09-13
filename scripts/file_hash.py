"""File fingerprints for standalone diagnostics.

Keep this owner independent of the scientific runtime: importing
``recovar.utils.file_hash`` initializes RECOVAR and JAX first. Runtime callers
retain that packaged helper; these script callers also work outside a RECOVAR
installation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
