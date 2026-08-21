"""RELION pybind11 bindings for exact parity testing.

Provides Python wrappers around RELION's C++ subfunctions so that
recovar and RELION can be called on identical inputs and their outputs
diffed numerically.

Submodules
----------
conversions : Layout conversion functions (pure Python, no C++ dependency)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from recovar.relion_bind.conversions import (
    compute_relion_pad_size,
    fftw_half_to_recovar_centered,
    fftw_half_to_relion_projector,
    recovar_centered_to_fftw_half,
    recovar_centered_to_relion_projector,
    recovar_real_to_relion_real,
    relion_projector_to_fftw_half,
    relion_projector_to_recovar_centered,
    relion_real_to_recovar_real,
)


def _load_external_core_if_available() -> None:
    """Load `_relion_bind_core` from an external Slurm build directory.

    Long EM jobs build the RELION binding under scratch via
    `RECOVAR_RELION_BIND_BUILD_DIR`; copying the shared object back into the
    checkout can fail when the source filesystem is quota-full. Preloading the
    extension here lets both `from recovar.relion_bind import _relion_bind_core`
    and `from recovar.relion_bind._relion_bind_core import ...` work without a
    package-local copy.
    """

    module_name = f"{__name__}._relion_bind_core"
    if module_name in sys.modules:
        return

    package_dir = Path(__file__).resolve().parent
    candidate_dirs = []
    build_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")
    if build_dir:
        candidate_dirs.append(Path(build_dir).expanduser())
    candidate_dirs.append(package_dir / "build")

    for candidate_dir in candidate_dirs:
        try:
            so_files = sorted(candidate_dir.glob("_relion_bind_core*.so"))
        except OSError:
            continue
        for so_path in so_files:
            try:
                if so_path.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            spec = importlib.util.spec_from_file_location(module_name, so_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            globals()["_relion_bind_core"] = module
            return


_load_external_core_if_available()

__all__ = [
    "compute_relion_pad_size",
    "fftw_half_to_recovar_centered",
    "fftw_half_to_relion_projector",
    "recovar_centered_to_fftw_half",
    "recovar_centered_to_relion_projector",
    "recovar_real_to_relion_real",
    "relion_projector_to_fftw_half",
    "relion_projector_to_recovar_centered",
    "relion_real_to_recovar_real",
]
