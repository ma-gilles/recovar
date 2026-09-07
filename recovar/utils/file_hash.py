"""Streaming file identities for benchmark and diagnostic artifacts."""

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash the file contents in 8 MiB blocks, propagating file access errors.

    The digest identifies the bytes read. Callers own manifest validation and
    must protect or recheck files if they need an immutable input snapshot.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
