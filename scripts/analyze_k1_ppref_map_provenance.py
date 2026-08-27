#!/usr/bin/env python3
"""Compare live RELION PPref with PPref rebuilt from numbered input maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar.em.initial_model.dense_adapter import (  # noqa: E402
    reference_to_relion_projector_half_maps,
)
from recovar.utils import helpers  # noqa: E402
from scripts.analyze_k1_exact_ppref_fine_boundary import _load_ppref  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_counted(path: Path, dtype: np.dtype) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 4, f"truncated counted array: {path}")
    count = int(np.frombuffer(payload, dtype="<i4", count=1)[0])
    values = np.frombuffer(payload, dtype=dtype, count=count, offset=4).copy()
    _require(4 + values.nbytes == len(payload), f"counted-array size mismatch: {path}")
    return values


def _load_verbose_ppref(directory: Path) -> tuple[np.ndarray, dict[str, object]]:
    dims = _load_counted(directory / "pass1_class0_ppref_dims.bin", np.dtype("<i4"))
    _require(dims.shape == (7,), "verbose PPref dimensions must contain seven values")
    shape = (int(dims[2]), int(dims[1]), int(dims[0]))
    real = _load_counted(directory / "pass1_class0_ppref_real.bin", np.dtype("<f8"))
    imag = _load_counted(directory / "pass1_class0_ppref_imag.bin", np.dtype("<f8"))
    _require(real.size == int(np.prod(shape)) and imag.shape == real.shape, "verbose PPref payload shape differs")
    ppref = (real.astype(np.float32) + 1j * imag.astype(np.float32)).astype(
        np.complex64
    ).reshape(shape)
    padding_factor = float(
        np.fromfile(
            directory / "pass1_class0_ppref_padding_factor.bin",
            dtype="<f8",
            count=1,
        )[0]
    )
    image_current_size = int(
        round(
            float(
                np.fromfile(
                    directory / "pass1_img0_exp_current_image_size.bin",
                    dtype="<f8",
                    count=1,
                )[0]
            )
        )
    )
    return ppref, {
        "version": "verbose-flat",
        "iteration": None,
        "rank": 3,
        "model": 0,
        # The optics-remapped particle image can be two pixels larger than
        # the model sphere. PPref construction uses 2*r_max, while the verbose
        # image field records the scoring-image current size.
        "current_size": 2 * int(dims[6]),
        "image_current_size": image_current_size,
        "shape_zyx": list(shape),
        "origin_xyz": [int(dims[3]), int(dims[4]), int(dims[5])],
        "r_max": int(dims[6]),
        "padding_factor": padding_factor,
        "complex_count": int(ppref.size),
    }


def _load_setup_ppref(directory: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load the contiguous complex-double PPref emitted by expectation setup."""

    data_path = directory / "ppref_c0_data_post_setup.bin"
    meta_path = directory / "ppref_c0_meta.txt"
    metadata_text = {}
    for line in meta_path.read_text().splitlines():
        key, separator, value = line.partition("=")
        _require(bool(separator) and bool(key) and bool(value), f"invalid PPref metadata line: {line!r}")
        metadata_text[key] = value
    required = {"iter", "r_max", "ori_size", "padding_factor", "z", "y", "x"}
    _require(required <= metadata_text.keys(), f"setup PPref metadata lacks {sorted(required - metadata_text.keys())}")

    payload = data_path.read_bytes()
    _require(len(payload) >= 12, f"truncated setup PPref dump: {data_path}")
    shape = struct.unpack_from("<iii", payload)
    expected_shape = tuple(int(metadata_text[axis]) for axis in ("z", "y", "x"))
    _require(shape == expected_shape and min(shape) > 0, "setup PPref shape differs from metadata")
    count = int(np.prod(shape))
    _require(len(payload) == 12 + count * 16, f"setup PPref payload size mismatch: {data_path}")
    ppref = np.frombuffer(payload, dtype="<c16", count=count, offset=12).copy()
    ppref = ppref.astype(np.complex64).reshape(shape)
    r_max = int(metadata_text["r_max"])
    return ppref, {
        "version": "expectation-setup-contiguous",
        "iteration": int(metadata_text["iter"]),
        "rank": None,
        "model": 0,
        "current_size": 2 * r_max,
        "image_current_size": int(shape[2]),
        "original_image_size": int(metadata_text["ori_size"]),
        "shape_zyx": list(shape),
        "origin_xyz": [0, -(shape[1] // 2), -(shape[0] // 2)],
        "r_max": r_max,
        "padding_factor": float(metadata_text["padding_factor"]),
        "complex_count": int(ppref.size),
    }


def _load_native_ppref(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    if not path.is_dir():
        return _load_ppref(path)
    if (path / "pass1_class0_ppref_real.bin").is_file():
        return _load_verbose_ppref(path)
    if (path / "ppref_c0_data_post_setup.bin").is_file():
        return _load_setup_ppref(path)
    raise FileNotFoundError(f"no supported PPref dump found in {path}")


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(candidate, dtype=np.complex64)
    right = np.asarray(reference, dtype=np.complex64)
    _require(left.shape == right.shape and left.size > 0, "PPref topology mismatch")
    residual = left.astype(np.complex128) - right.astype(np.complex128)
    absolute = np.abs(residual)
    left_bits = np.ascontiguousarray(left).view(np.uint32).reshape(-1, 2)
    right_bits = np.ascontiguousarray(right).view(np.uint32).reshape(-1, 2)
    denominator = max(float(np.linalg.norm(right.astype(np.complex128))), np.finfo(float).tiny)
    return {
        "complex_count": int(left.size),
        "bitwise_equal_complex_count": int(np.count_nonzero(np.all(left_bits == right_bits, axis=-1))),
        "bitwise_equal_float32_component_count": int(np.count_nonzero(left_bits == right_bits)),
        "float32_component_count": int(left_bits.size),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
    }


def _shell_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    origin_xyz: list[int],
    r_max: int,
) -> dict[str, dict[str, float | int]]:
    left = np.asarray(candidate, dtype=np.complex64)
    right = np.asarray(reference, dtype=np.complex64)
    _require(left.shape == right.shape and left.ndim == 3, "PPref shell topology mismatch")
    xinit, yinit, zinit = (int(value) for value in origin_xyz)
    z = np.arange(left.shape[0], dtype=np.int64) + zinit
    y = np.arange(left.shape[1], dtype=np.int64) + yinit
    x = np.arange(left.shape[2], dtype=np.int64) + xinit
    radius = np.rint(
        np.sqrt(
            z[:, None, None].astype(np.float64) ** 2
            + y[None, :, None].astype(np.float64) ** 2
            + x[None, None, :].astype(np.float64) ** 2
        )
    ).astype(np.int64)
    active = (radius <= int(r_max)) & ((left != 0) | (right != 0))
    return {
        str(shell): _metric(left[active & (radius == shell)], right[active & (radius == shell)])
        for shell in range(int(r_max) + 1)
        if np.any(active & (radius == shell))
    }


def _parse_map_spec(value: str) -> tuple[str, Path, str]:
    fields = value.split(":", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("map must be LABEL:PATH:CONVENTION")
    label, path_text, convention = fields
    if not label or convention not in {"recovar", "relion"}:
        raise argparse.ArgumentTypeError("map convention must be recovar or relion")
    return label, Path(path_text), convention


def _parse_iref_spec(value: str) -> tuple[str, Path]:
    fields = value.split(":", 1)
    if len(fields) != 2 or not fields[0] or not fields[1]:
        raise argparse.ArgumentTypeError("Iref must be LABEL:PATH")
    return fields[0], Path(fields[1])


def _load_relion_iref(path: Path) -> np.ndarray:
    """Load ``int32 z,y,x`` plus float64 RELION in-memory Iref data."""

    payload = path.read_bytes()
    _require(len(payload) >= 12, f"truncated RELION Iref dump: {path}")
    zdim, ydim, xdim = struct.unpack_from("<iii", payload)
    _require(min(zdim, ydim, xdim) > 0, f"invalid RELION Iref shape: {path}")
    count = zdim * ydim * xdim
    _require(len(payload) == 12 + count * 8, f"RELION Iref payload size mismatch: {path}")
    relion_frame = np.frombuffer(payload, dtype="<f8", count=count, offset=12).copy()
    relion_frame = relion_frame.reshape(zdim, ydim, xdim)
    return np.asarray(helpers.relion_volume_to_recovar(relion_frame), dtype=np.float64)


def _load_map(path: Path, convention: str) -> np.ndarray:
    if convention == "recovar":
        return np.asarray(helpers.load_mrc(str(path)), dtype=np.float64)
    if convention == "relion":
        return np.asarray(helpers.load_relion_volume(str(path)), dtype=np.float64)
    raise ValueError(f"unsupported map convention: {convention}")


def _real_metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    _require(left.shape == right.shape and left.size > 0, "real-map topology mismatch")
    residual = left - right
    denominator = max(float(np.linalg.norm(right)), np.finfo(float).tiny)
    return {
        "voxel_count": int(left.size),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "max_abs": float(np.max(np.abs(residual))),
        "candidate_float32_equals_reference_float32": bool(
            np.array_equal(left.astype(np.float32), right.astype(np.float32))
        ),
    }


def analyze(
    ppref_path: Path,
    map_specs: list[tuple[str, Path, str]],
    iref_specs: list[tuple[str, Path]] | None = None,
) -> dict[str, object]:
    native_ppref, metadata = _load_native_ppref(ppref_path)
    _require(len(map_specs) > 0, "at least one map is required")
    iref_specs = [] if iref_specs is None else iref_specs
    labels = [label for label, _path, _convention in map_specs] + [
        label for label, _path in iref_specs
    ]
    _require(len(set(labels)) == len(labels), "map labels must be unique")

    candidates: dict[str, np.ndarray] = {}
    real_sources: dict[str, np.ndarray] = {}
    source_kinds: dict[str, str] = {}
    artifacts: dict[str, object] = {
        "native_ppref": str(ppref_path.resolve()),
        "native_ppref_sha256": (
            None if ppref_path.is_dir() else _sha256(ppref_path)
        ),
        "native_ppref_verbose_files": (
            {
                path.name: _sha256(path)
                for path in sorted(ppref_path.glob("pass1_class0_ppref_*.bin"))
            }
            if ppref_path.is_dir()
            else None
        ),
        "maps": {},
        "irefs": {},
    }
    for label, path, convention in map_specs:
        reference = _load_map(path, convention)
        _require(reference.ndim == 3 and len(set(reference.shape)) == 1, f"{label}: map must be cubic")
        rebuilt, r_max = reference_to_relion_projector_half_maps(
            reference[None],
            current_size=int(metadata["current_size"]),
            padding_factor=int(metadata["padding_factor"]),
        )
        _require(int(r_max) == int(metadata["r_max"]), f"{label}: r_max mismatch")
        candidate = np.asarray(rebuilt[0], dtype=np.complex64)
        _require(candidate.shape == native_ppref.shape, f"{label}: PPref shape mismatch")
        candidates[label] = candidate
        real_sources[label] = reference
        source_kinds[label] = "numbered_map"
        artifacts["maps"][label] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "convention": convention,
            "shape": list(reference.shape),
            "dtype_after_load": str(reference.dtype),
        }

    for label, path in iref_specs:
        reference = _load_relion_iref(path)
        _require(reference.ndim == 3 and len(set(reference.shape)) == 1, f"{label}: Iref must be cubic")
        rebuilt, r_max = reference_to_relion_projector_half_maps(
            reference[None],
            current_size=int(metadata["current_size"]),
            padding_factor=int(metadata["padding_factor"]),
        )
        _require(int(r_max) == int(metadata["r_max"]), f"{label}: r_max mismatch")
        candidate = np.asarray(rebuilt[0], dtype=np.complex64)
        _require(candidate.shape == native_ppref.shape, f"{label}: PPref shape mismatch")
        candidates[label] = candidate
        real_sources[label] = reference
        source_kinds[label] = "in_memory_iref"
        artifacts["irefs"][label] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "convention": "relion_in_memory_converted_to_recovar",
            "shape": list(reference.shape),
            "dtype_after_load": str(reference.dtype),
        }

    versus_native = {
        label: {
            **_metric(candidate, native_ppref),
            "shells": _shell_metrics(
                candidate,
                native_ppref,
                origin_xyz=list(metadata["origin_xyz"]),
                r_max=int(metadata["r_max"]),
            ),
        }
        for label, candidate in candidates.items()
    }
    pairwise = {
        f"{left}_vs_{right}": _metric(candidates[left], candidates[right])
        for left_index, left in enumerate(labels)
        for right in labels[left_index + 1 :]
    }
    best_label = min(labels, key=lambda label: float(versus_native[label]["relative_l2"]))
    best_exact = int(versus_native[best_label]["bitwise_equal_float32_component_count"]) == int(
        versus_native[best_label]["float32_component_count"]
    )
    if best_exact and source_kinds[best_label] == "in_memory_iref":
        classification = "captured in-memory Iref rebuild is float32-bitwise exact with live native PPref"
    elif best_exact:
        classification = "numbered map rebuild is float32-bitwise exact with live native PPref"
    elif source_kinds[best_label] == "in_memory_iref":
        classification = "captured in-memory Iref is closest but does not exactly reproduce live native PPref"
    else:
        classification = "live native PPref retains state not reproduced by the numbered map rebuild"
    return {
        "schema": "recovar.em.k1_ppref_map_provenance.v1",
        "status": "complete",
        "native_ppref": metadata,
        "rebuilt_ppref_vs_live_native": versus_native,
        "rebuilt_ppref_pairwise": pairwise,
        "in_memory_iref_vs_numbered_maps_real": {
            f"{iref_label}_vs_{map_label}": _real_metric(
                real_sources[iref_label], real_sources[map_label]
            )
            for iref_label, _path in iref_specs
            for map_label, _map_path, _convention in map_specs
        },
        "best_rebuilt_source": best_label,
        "best_rebuilt_source_kind": source_kinds[best_label],
        "classification": classification,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppref",
        type=Path,
        required=True,
        help="PPref capture file or verbose dump directory",
    )
    parser.add_argument("--map", dest="maps", action="append", type=_parse_map_spec, required=True)
    parser.add_argument("--iref", dest="irefs", action="append", type=_parse_iref_spec, default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(args.ppref, args.maps, args.irefs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
