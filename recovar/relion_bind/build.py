#!/usr/bin/env python
"""Build the RELION pybind11 bindings.

Usage from repo root:
    pixi run python recovar/relion_bind/build.py

Or to rebuild after C++ changes:
    cd recovar/relion_bind/build && make -j8
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

BIND_DIR = Path(__file__).parent
BUILD_DIR = Path(os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR", BIND_DIR / "build")).expanduser()


def get_relion_src() -> Path:
    """Return the explicitly configured RELION ``src`` directory."""

    configured = os.environ.get("RELION_SRC_DIR")
    if not configured:
        raise RuntimeError(
            "RELION_SRC_DIR is not set; point it at the RELION 'src' directory "
            "before building recovar.relion_bind"
        )
    relion_src = Path(configured).expanduser().resolve()
    if not (relion_src / "projector.h").is_file():
        raise FileNotFoundError(f"RELION_SRC_DIR={relion_src} does not contain projector.h")
    return relion_src


def get_pybind11_cmake_dir():
    import pybind11

    return pybind11.get_cmake_dir()


def build():
    relion_src = get_relion_src()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmake_cmd = [
        "cmake",
        str(BIND_DIR),
        f"-Dpybind11_DIR={get_pybind11_cmake_dir()}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DRELION_SRC_DIR={relion_src}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    print(f"Configuring: {' '.join(cmake_cmd)}")
    subprocess.check_call(cmake_cmd, cwd=BUILD_DIR)

    ncpu = os.cpu_count() or 4
    make_cmd = ["make", f"-j{ncpu}"]
    print(f"Building: {' '.join(make_cmd)}")
    subprocess.check_call(make_cmd, cwd=BUILD_DIR)

    # Copy .so to package directory for local developer builds. Slurm jobs set
    # RECOVAR_RELION_BIND_BUILD_DIR to a scratch build directory; keep the
    # artifact there so quota-full source checkouts do not break setup.
    so_files = list(BUILD_DIR.glob("_relion_bind_core*.so"))
    if not so_files:
        print("ERROR: No .so file produced", file=sys.stderr)
        sys.exit(1)

    if os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR") and os.environ.get(
        "RECOVAR_RELION_BIND_COPY_TO_PACKAGE", "0"
    ).strip().lower() not in {"1", "true", "yes", "on"}:
        print(f"Built external RELION binding: {so_files[0]}")
        return

    dest = BIND_DIR / so_files[0].name
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(so_files[0], tmp_dest)
    tmp_dest.replace(dest)
    print(f"Installed: {dest}")


if __name__ == "__main__":
    build()
