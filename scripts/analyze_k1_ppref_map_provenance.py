#!/usr/bin/env python3
"""Compare live RELION PPref with PPref rebuilt from numbered input maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar.em.initial_model.dense_adapter import reference_to_relion_projector_half_maps
from recovar.utils import helpers
from scripts.analyze_k1_exact_ppref_fine_boundary import _load_ppref


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_map(path: Path, convention: str) -> np.ndarray:
    if convention == "recovar":
        return np.asarray(helpers.load_mrc(str(path)), dtype=np.float64)
    if convention == "relion":
        return np.asarray(helpers.load_relion_volume(str(path)), dtype=np.float64)
    raise ValueError(f"unsupported map convention: {convention}")


def analyze(ppref_path: Path, map_specs: list[tuple[str, Path, str]]) -> dict[str, object]:
    native_ppref, metadata = _load_ppref(ppref_path)
    _require(len(map_specs) > 0, "at least one map is required")
    labels = [label for label, _path, _convention in map_specs]
    _require(len(set(labels)) == len(labels), "map labels must be unique")

    candidates: dict[str, np.ndarray] = {}
    artifacts: dict[str, object] = {
        "native_ppref": str(ppref_path.resolve()),
        "native_ppref_sha256": _sha256(ppref_path),
        "maps": {},
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
        artifacts["maps"][label] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "convention": convention,
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
    return {
        "schema": "recovar.em.k1_ppref_map_provenance.v1",
        "status": "complete",
        "native_ppref": metadata,
        "rebuilt_ppref_vs_live_native": versus_native,
        "rebuilt_ppref_pairwise": pairwise,
        "best_rebuilt_source": best_label,
        "classification": (
            "numbered map rebuild is float32-bitwise exact with live native PPref"
            if int(versus_native[best_label]["bitwise_equal_float32_component_count"])
            == int(versus_native[best_label]["float32_component_count"])
            else "live native PPref retains state not reproduced by the numbered map rebuild"
        ),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppref", type=Path, required=True)
    parser.add_argument("--map", dest="maps", action="append", type=_parse_map_spec, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(args.ppref, args.maps)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
