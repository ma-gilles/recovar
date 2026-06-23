#!/usr/bin/env python3
"""Extract reusable camera/view settings from a ChimeraX .cxs session.

The cluster ChimeraX module may not be able to restore .cxs files saved with
newer/local bundles, but the camera state is still available in the compressed
session stream.  This writes a small JSON file that can be passed to render
scripts or converted to ChimeraX commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lz4.frame
import msgpack
import numpy as np
from msgpack import ExtType


def _decode_ext(ext: ExtType) -> Any:
    return msgpack.unpackb(ext.data, raw=False, strict_map_key=False)


def _array_from_ext(ext: ExtType) -> np.ndarray | None:
    try:
        data = _decode_ext(ext)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    def get(key: str) -> Any:
        return data.get(key, data.get(key.encode()))

    dtype = get("dtype")
    shape = get("shape")
    raw = get("data")
    if dtype is None or shape is None or raw is None:
        return None
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)


def _load_objects(path: Path) -> list[Any]:
    payload = lz4.frame.decompress(path.read_bytes()).split(b"\n", 1)[1]
    unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
    unpacker.feed(payload)
    return list(unpacker)


def _first_geometry_matrix_before_camera(objects: list[Any]) -> np.ndarray:
    last_matrix: np.ndarray | None = None
    for obj in objects:
        if isinstance(obj, dict) and "matrix" in obj:
            matrix = _array_from_ext(obj["matrix"])
            if matrix is not None and matrix.shape == (3, 4):
                last_matrix = matrix
        if isinstance(obj, dict) and obj.get("name") == "mono" and "field_of_view" in obj:
            if last_matrix is None:
                raise RuntimeError("Found camera state but no preceding 3x4 GeometryPlace matrix")
            return last_matrix
    raise RuntimeError("Could not find ChimeraX mono camera state")


def _find_camera_state(objects: list[Any]) -> dict[str, Any]:
    for obj in objects:
        if isinstance(obj, dict) and obj.get("name") == "mono" and "field_of_view" in obj:
            return obj
    raise RuntimeError("Could not find ChimeraX mono camera state")


def _find_main_view(objects: list[Any]) -> dict[str, Any]:
    for obj in objects:
        if isinstance(obj, dict) and "window_size" in obj and "center_of_rotation" in obj:
            return obj
    raise RuntimeError("Could not find ChimeraX main view state")


def _decode_window_size(value: ExtType) -> list[int] | None:
    try:
        data = _decode_ext(value)
    except Exception:
        return None
    if isinstance(data, (list, tuple)) and len(data) == 2:
        return [int(data[0]), int(data[1])]
    return None


def _matrix_arg(matrix: np.ndarray) -> str:
    return ",".join(f"{x:.9g}" for x in matrix.reshape(-1))


def _jsonify(value: Any) -> Any:
    if isinstance(value, ExtType):
        try:
            return _jsonify(_decode_ext(value))
        except Exception:
            return {"ext_code": value.code, "n_bytes": len(value.data)}
    if isinstance(value, bytes):
        try:
            return value.decode()
        except UnicodeDecodeError:
            return {"n_bytes": len(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(_jsonify(k)): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cxs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    objects = _load_objects(args.cxs)
    camera = _find_camera_state(objects)
    view = _find_main_view(objects)
    matrix = _first_geometry_matrix_before_camera(objects)
    window_size = _decode_window_size(view["window_size"])

    out = {
        "source_cxs": str(args.cxs),
        "camera_matrix": _matrix_arg(matrix),
        "camera_matrix_rows": matrix.tolist(),
        "field_of_view": float(camera["field_of_view"]),
        "window_size": window_size,
        "background_color": None,
        "silhouettes": _jsonify(view.get("silhouettes")),
        "center_of_rotation_method": _jsonify(view.get("center_of_rotation_method")),
    }
    background = _array_from_ext(view.get("background_color"))
    if background is not None:
        out["background_color"] = [float(x) for x in background.tolist()]

    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
