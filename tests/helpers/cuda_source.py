"""Read CUDA source and local includes, retaining the include directives."""

import re
from pathlib import Path

from recovar.cuda_build import include_dir

# Located through the package, not the checkout layout.
PUBLIC_INCLUDE_DIR = include_dir()
CUDA_DIR = PUBLIC_INCLUDE_DIR.parent


def read_cuda_source(filename="cuda_backproject.cu", base_dir=CUDA_DIR, skip=()):
    """Return ``filename`` with every resolvable local ``#include`` inlined after its directive.

    Includes resolve relative to the including file, as the compiler does, then in recovar's public
    header directory (relax's EM Makefile passes it with ``-I``).
    """
    path = (Path(base_dir) / filename).resolve()
    source = path.read_text()

    def inline(match):
        if Path(match[1]).name in skip:
            return match[0]
        for directory in (path.parent, PUBLIC_INCLUDE_DIR):
            if (directory / match[1]).is_file():
                return match[0] + "\n" + read_cuda_source(match[1], directory, skip)
        return match[0]

    return re.sub(r'(?m)^#include "([^"]+)"$', inline, source)

