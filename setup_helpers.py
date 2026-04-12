from __future__ import annotations

from pathlib import Path


def _matching_paths(root: str | Path | None, *patterns: str) -> list[Path]:
    if root is None:
        return []

    root_path = Path(root)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root_path.glob(pattern) if path.is_file())
    return matches


def remove_stale_fast_marching_build_artifacts(
    build_lib: str | Path | None = None,
    build_temp: str | Path | None = None,
) -> list[Path]:
    """Remove legacy root-level fast-marching build outputs from old layouts."""

    candidates = _matching_paths(
        build_lib,
        "recovar/_fast_marching_native.cpp",
        "recovar/_fast_marching_native*.so",
        "recovar/_fast_marching_native*.pyd",
        "recovar/_fast_marching_native*.dylib",
    )
    candidates.extend(
        _matching_paths(
            build_temp,
            "recovar/_fast_marching_native*.o",
            "recovar/_fast_marching_native*.obj",
        )
    )

    removed: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        path.unlink(missing_ok=True)
        removed.append(path)

    return removed
