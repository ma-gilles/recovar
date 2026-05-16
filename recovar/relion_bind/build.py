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
BUILD_DIR = BIND_DIR / "build"
RELION_SRC = Path("/scratch/gpfs/GILLES/mg6942/relion/src")


def get_pybind11_cmake_dir():
    import pybind11

    return pybind11.get_cmake_dir()


def build():
    if not RELION_SRC.exists():
        print(f"ERROR: RELION source not found at {RELION_SRC}", file=sys.stderr)
        sys.exit(1)

    BUILD_DIR.mkdir(exist_ok=True)

    cmake_cmd = [
        "cmake",
        str(BIND_DIR),
        f"-Dpybind11_DIR={get_pybind11_cmake_dir()}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DRELION_SRC_DIR={RELION_SRC}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    print(f"Configuring: {' '.join(cmake_cmd)}")
    subprocess.check_call(cmake_cmd, cwd=BUILD_DIR)

    ncpu = os.cpu_count() or 4
    make_cmd = ["make", f"-j{ncpu}"]
    print(f"Building: {' '.join(make_cmd)}")
    subprocess.check_call(make_cmd, cwd=BUILD_DIR)

    # Copy .so to package directory
    so_files = list(BUILD_DIR.glob("_relion_bind_core*.so"))
    if not so_files:
        print("ERROR: No .so file produced", file=sys.stderr)
        sys.exit(1)

    dest = BIND_DIR / so_files[0].name
    shutil.copy2(so_files[0], dest)
    print(f"Installed: {dest}")


if __name__ == "__main__":
    build()
