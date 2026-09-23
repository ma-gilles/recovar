"""Read CUDA source and local includes, retaining the include directives."""

import re
from pathlib import Path

CUDA_DIR = Path(__file__).resolve().parents[2] / "recovar" / "cuda"


def read_cuda_source(filename="cuda_backproject.cu", base_dir=CUDA_DIR):
    """Return ``filename`` with every resolvable local ``#include`` inlined after its directive.

    Includes resolve relative to the including file, as the compiler does (the EM headers live in
    ``recovar/em/cuda`` and are included as ``../em/cuda/<name>``).
    """
    path = (Path(base_dir) / filename).resolve()
    source = path.read_text()
    return re.sub(
        r'(?m)^#include "([^"]+)"$',
        lambda match: (
            match[0] + "\n" + read_cuda_source(match[1], path.parent)
            if (path.parent / match[1]).is_file()
            else match[0]
        ),
        source,
    )
