"""Read CUDA source and local includes, retaining the include directives."""

import re
from pathlib import Path


def read_cuda_source(filename="cuda_backproject.cu"):
    cuda_dir = Path(__file__).resolve().parents[2] / "recovar" / "cuda"
    source = (cuda_dir / filename).read_text()
    return re.sub(
        r'(?m)^#include "([^"]+)"$',
        lambda match: match[0] + "\n" + read_cuda_source(match[1]) if (cuda_dir / match[1]).is_file() else match[0],
        source,
    )
