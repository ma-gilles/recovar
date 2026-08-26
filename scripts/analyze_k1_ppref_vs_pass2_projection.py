#!/usr/bin/env python3
"""Compare a rebuilt K=1 PPref with the projections used by a pass-2 dump."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_diff2_sum,
    _relion_cuda_fine_full_to_compact_lookup,
)
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


def _float32_bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values).view(np.float32).reshape(-1).view(np.uint32)


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    left = np.asarray(candidate)
    right = np.asarray(reference)
    _require(left.shape == right.shape and left.size > 0, "comparison topology mismatch")
    if np.iscomplexobj(left) or np.iscomplexobj(right):
        left = np.asarray(left, dtype=np.complex64)
        right = np.asarray(right, dtype=np.complex64)
    else:
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
    residual = left.astype(np.complex128 if np.iscomplexobj(left) else np.float64) - right.astype(
        np.complex128 if np.iscomplexobj(right) else np.float64
    )
    absolute = np.abs(residual).reshape(-1)
    left_bits = _float32_bits(left)
    right_bits = _float32_bits(right)
    unequal = np.flatnonzero(left_bits != right_bits)
    denominator = max(float(np.linalg.norm(right.reshape(-1))), np.finfo(float).tiny)
    return {
        "shape": list(left.shape),
        "float32_component_count": int(left_bits.size),
        "bitwise_equal_float32_component_count": int(np.count_nonzero(left_bits == right_bits)),
        "first_unequal_float32_component": None if unequal.size == 0 else int(unequal[0]),
        "relative_l2": float(np.linalg.norm(residual.reshape(-1)) / denominator),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def _align_native_ppref_to_live_score_gauge(values: np.ndarray) -> np.ndarray:
    """Convert native PPref texture output to the strict RECOVAR score gauge."""

    return np.negative(np.asarray(values, dtype=np.complex64), dtype=np.complex64)


def analyze(
    *,
    ppref_path: Path,
    pass2_path: Path,
    physical_image_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    _require(jax.default_backend() == "gpu", "PPref projection comparison requires a GPU")
    ppref, metadata = _load_ppref(ppref_path)
    with np.load(pass2_path, allow_pickle=False) as archive:
        capture = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "current_size",
        "rotations",
        "proj_half",
        "shifted_corrected",
        "ctf2_over_nv_score",
        "half_weights",
        "window_indices",
        "relion_highres_xi2_half",
        "relion_raw_diff2",
    }
    _require(required <= set(capture), f"pass-2 dump misses {sorted(required - set(capture))}")
    image_current_size = int(np.asarray(capture["current_size"]).item())
    model_current_size = int(metadata["current_size"])
    _require(float(metadata["padding_factor"]) == 2.0, "padding factor differs")
    _require(int(metadata["r_max"]) == model_current_size // 2, "projector radius differs")

    rotations = np.asarray(capture["rotations"], dtype=np.float32)
    live_projection = np.asarray(capture["proj_half"], dtype=np.complex64)
    _require(live_projection.shape[0] == rotations.shape[0], "projection rotation count differs")
    window_indices = np.asarray(capture["window_indices"], dtype=np.int32)
    rebuilt_blocks: list[np.ndarray] = []
    for start in range(0, rotations.shape[0], chunk_size):
        stop = min(start + chunk_size, rotations.shape[0])
        rebuilt, _ = compute_relion_projector_projections_block(
            jnp.asarray(ppref),
            jnp.asarray(rotations[start:stop]),
            (physical_image_size, physical_image_size),
            r_max=int(metadata["r_max"]),
            padding_factor=2,
            return_abs2=False,
            centered_rows=True,
            # Current strict-parity pass-2 captures store projections in
            # RELION score units.  The captured PPref is already in the same
            # convention, so applying RECOVAR's historical dense -N^2 scale
            # here would introduce a spurious factor of 65,536 at box 256.
            dense_scale=False,
            projector_output_size=image_current_size,
            pixel_indices=jnp.asarray(window_indices),
            relion_texture_interp=True,
        )
        # RECOVAR's strict score capture uses the common CTF gauge opposite to
        # native RELION.  Both image and reference are negated in that gauge;
        # align the standalone native PPref projection before replaying it
        # against RECOVAR's captured shifted image.
        rebuilt_blocks.append(
            _align_native_ppref_to_live_score_gauge(jax.block_until_ready(rebuilt))
        )
    rebuilt_projection = np.concatenate(rebuilt_blocks, axis=0)
    _require(rebuilt_projection.shape == live_projection.shape, "rebuilt projection shape differs")

    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        (physical_image_size, physical_image_size), image_current_size, window_indices
    )
    shifted = jnp.asarray(capture["shifted_corrected"], dtype=jnp.complex64)
    pixel_weight = jnp.asarray(
        np.multiply(
            np.asarray(capture["ctf2_over_nv_score"], dtype=np.float32),
            np.asarray(capture["half_weights"], dtype=np.float32),
            dtype=np.float32,
        )
    )
    highres = jnp.asarray(capture["relion_highres_xi2_half"], dtype=jnp.float32)

    @jax.jit
    def score(projection: jax.Array) -> jax.Array:
        raw = _relion_cuda_fine_diff2_sum(
            projection[:, None, :],
            shifted[None, :, :],
            pixel_weight[None, None, :],
            jnp.asarray(full_to_compact, dtype=jnp.int32),
        )
        return raw + highres

    rebuilt_score = np.asarray(jax.block_until_ready(score(jnp.asarray(rebuilt_projection))), dtype=np.float32)
    live_replayed_score = np.asarray(
        jax.block_until_ready(score(jnp.asarray(live_projection))), dtype=np.float32
    )
    live_score = np.asarray(capture["relion_raw_diff2"], dtype=np.float32)
    _require(rebuilt_score.shape == live_replayed_score.shape == live_score.shape, "score shape differs")

    projection_metric = _metric(rebuilt_projection, live_projection)
    return {
        "schema": "recovar.em.k1_ppref_vs_pass2_projection.v1",
        "status": "complete",
        "classification": (
            "standalone_ppref_reproduces_live_pass2_projection_bitwise"
            if projection_metric["bitwise_equal_float32_component_count"]
            == projection_metric["float32_component_count"]
            else "standalone_ppref_and_live_pass2_projection_differ"
        ),
        "identity": {
            "physical_iteration": int(np.asarray(capture.get("iteration", -1)).item()),
            "half": int(np.asarray(capture.get("half", -1)).item()),
            "original_index": int(np.asarray(capture.get("original_index", -1)).item()),
            "rotation_count": int(rotations.shape[0]),
            "translation_count": int(shifted.shape[0]),
            "model_current_size": model_current_size,
            "image_current_size": image_current_size,
        },
        "standalone_ppref_vs_live_pass2_projection": projection_metric,
        "score_from_standalone_ppref_vs_live_score": _metric(rebuilt_score, live_score),
        "score_replayed_from_live_projection_vs_live_score": _metric(live_replayed_score, live_score),
        "score_from_standalone_ppref_vs_live_projection_replay": _metric(
            rebuilt_score, live_replayed_score
        ),
        "artifacts": {
            "ppref": str(ppref_path.resolve()),
            "ppref_sha256": _sha256(ppref_path),
            "pass2": str(pass2_path.resolve()),
            "pass2_sha256": _sha256(pass2_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppref", type=Path, required=True)
    parser.add_argument("--pass2", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        ppref_path=args.ppref,
        pass2_path=args.pass2,
        physical_image_size=args.physical_image_size,
        chunk_size=args.chunk_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
