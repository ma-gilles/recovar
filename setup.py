from __future__ import annotations

import importlib.util
import os
import sys
from distutils import log
from pathlib import Path

import numpy
from setuptools import Extension, setup
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext

REPO_ROOT = Path(__file__).resolve().parent
_HELPERS_SPEC = importlib.util.spec_from_file_location("recovar_setup_helpers", REPO_ROOT / "setup_helpers.py")
if _HELPERS_SPEC is None or _HELPERS_SPEC.loader is None:  # pragma: no cover - packaging bootstrap failure
    raise RuntimeError("Could not load setup_helpers.py")
_HELPERS_MODULE = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS_MODULE)
remove_stale_fast_marching_build_artifacts = _HELPERS_MODULE.remove_stale_fast_marching_build_artifacts


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _native_extension() -> Extension:
    extension = Extension(
        "recovar.trajectory._fast_marching_native",
        ["recovar/trajectory/_fast_marching_native.cpp"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"],
        language="c++",
    )
    extension.optional = True
    return extension


class OptionalBuildExt(build_ext):
    def run(self) -> None:
        remove_stale_fast_marching_build_artifacts(self.build_lib, self.build_temp)
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


class RecovarBuild(build):
    def run(self) -> None:
        remove_stale_fast_marching_build_artifacts(self.build_lib, self.build_temp)
        super().run()


setup(
    ext_modules=[_native_extension()],
    cmdclass={"build": RecovarBuild, "build_ext": OptionalBuildExt},
)
