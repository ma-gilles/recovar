"""Read CUDA source and local includes, retaining the include directives."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA_DIR = ROOT / "recovar" / "cuda"
EM_CUDA_DIR = ROOT / "recovar" / "em" / "cuda"
PUBLIC_INCLUDE_DIR = CUDA_DIR / "include"


def read_cuda_source(filename="cuda_backproject.cu", base_dir=CUDA_DIR, skip=()):
    """Return ``filename`` with every resolvable local ``#include`` inlined after its directive.

    Includes resolve relative to the including file, as the compiler does, then in recovar's public
    header directory (the EM Makefile passes it with ``-I``).
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


def read_em_cuda_source():
    """The EM library's translation unit (``librelax_cuda.so``) with its includes inlined."""
    return read_cuda_source("relax_kernels.cu", EM_CUDA_DIR)


def read_all_cuda_source():
    """Both libraries' units (pipeline first, then EM) with the shared public headers inlined once.

    Equals the former single translation unit's text up to item order; for source pins that span the
    pipeline and EM kernels.
    """
    return read_cuda_source() + "\n" + read_cuda_source(
        "relax_kernels.cu", EM_CUDA_DIR, skip=("recovar_cuda_common.cuh", "device_scratch.cuh")
    )
