"""Guard the CUDA_ARCH default — these compute caps must always be covered."""
import re
from pathlib import Path


def test_makefile_covers_v1_release_floor():
    mk = Path("recovar/cuda/Makefile").read_text()
    m = re.search(r"CUDA_ARCH\s*\?=([^\n]+(?:\\\n[^\n]+)*)", mk)
    assert m, "CUDA_ARCH default not found in Makefile"
    block = m.group(1)
    for sm in ("70", "75", "80", "86", "89", "90"):
        assert f"code=sm_{sm}" in block, (
            f"CUDA_ARCH default missing sm_{sm} cubin — "
            f"this breaks {sm} GPUs (e.g. V100/T4/A100/etc.). "
            f"Block was: {block}"
        )
    assert "code=compute_" in block, (
        "CUDA_ARCH default has no PTX (code=compute_XX) target — "
        "future archs will fail with no JIT fallback."
    )
