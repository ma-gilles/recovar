from __future__ import annotations

import os
import sys
from pathlib import Path

from distutils import log
import numpy
from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext


REPO_ROOT = Path(__file__).resolve().parent
NATIVE_FAST_MARCHING_DIR = REPO_ROOT / "recovar" / "heterogeneity" / "native_fast_marching"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _native_extension() -> Extension:
    sources = ["recovar/_fast_marching_native.cpp"]
    sources.extend(
        str(path.relative_to(REPO_ROOT))
        for path in sorted(NATIVE_FAST_MARCHING_DIR.glob("*.cpp"))
    )
    extension = Extension(
        "recovar._fast_marching_native",
        sources,
        include_dirs=[numpy.get_include(), "recovar/heterogeneity/native_fast_marching"],
        extra_compile_args=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"],
        language="c++",
    )
    extension.optional = True
    return extension


class OptionalBuildExt(build_ext):
    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:  # pragma: no cover - exercised in packaging failures
            if _env_flag("RECOVAR_REQUIRE_NATIVE_FMM"):
                raise
            log.warn("building optional fast marching extension failed: %s", exc)

    def build_extension(self, ext) -> None:
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - exercised in packaging failures
            if _env_flag("RECOVAR_REQUIRE_NATIVE_FMM") or not getattr(ext, "optional", False):
                raise
            log.warn("skipping optional extension %s: %s", ext.name, exc)


setup(
    ext_modules=[_native_extension()],
    cmdclass={"build_ext": OptionalBuildExt},
)
