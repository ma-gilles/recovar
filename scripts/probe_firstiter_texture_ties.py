#!/usr/bin/env python
"""Re-score firstiter-CC coarse ties with RELION-style CUDA textures.

This diagnostic converts cached RELION ``Projector::data`` into the centered
full-volume layout consumed by RECOVAR's CUDA texture projector, then compares
the RECOVAR and RELION coarse winners saved by the tie-adjudication harness.
It is intentionally separate from the production E-step until the texture
scores have been validated against the patched RELION dumps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.cuda_backproject import project
from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block,
    relion_projector_half_to_texture_full,
)


DEFAULT_ORIGINAL_INDICES = (1087, 1280, 1794, 431, 2693)


def _tree_sum_float32(values: np.ndarray, block_size: int = 128) -> np.float32:
    """Mirror RELION CUDA coarse-CC per-thread accumulation and tree reduce."""

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    shared = np.zeros(block_size, dtype=np.float32)
    for start in range(0, values.size, block_size):
        part = values[start : start + block_size]
        shared[: part.size] = np.asarray(shared[: part.size] + part, dtype=np.float32)
    stride = block_size // 2
    while stride:
        shared[:stride] = np.asarray(shared[:stride] + shared[stride : 2 * stride], dtype=np.float32)
        stride //= 2
    return shared[0]


def _window_to_texture_rows(window_indices: np.ndarray, full_size: int, current_size: int) -> np.ndarray:
    """Map full centered-row indices to the texture projector's centered crop."""

    full_x_half = full_size // 2 + 1
    rows = window_indices // full_x_half
    cols = window_indices % full_x_half
    ky = rows - full_size // 2
    crop_rows = np.where(ky == current_size // 2, 0, ky + current_size // 2)
    if np.any(crop_rows < 0) or np.any(crop_rows >= current_size):
        raise ValueError("Window contains rows outside the current-size texture projection")
    return crop_rows * (current_size // 2 + 1) + cols


def run_probe(root: Path, original_indices: tuple[int, ...]) -> dict[str, object]:
    cache_paths = sorted((root / "projector_cache").glob("projector_*.npz"))
    if len(cache_paths) != 1:
        raise ValueError(f"Expected one projector cache in {root}, found {len(cache_paths)}")
    cache = np.load(cache_paths[0])
    projector_half = np.asarray(cache["projector_half"])
    if projector_half.shape[0] != 1:
        raise ValueError(f"Expected K=1 projector cache, got {projector_half.shape}")
    current_size = int(cache["current_size"])
    padding_factor = int(cache["padding_factor"])
    r_max = int(cache["projector_r_max"])
    full_projector = relion_projector_half_to_texture_full(jnp.asarray(projector_half[0]))
    projector_size = int(full_projector.shape[0])

    results: dict[str, object] = {
        "root": str(root),
        "device": str(jax.devices()[0]),
        "projector_cache": str(cache_paths[0]),
        "current_size": current_size,
        "padding_factor": padding_factor,
        "projector_r_max": r_max,
        "particles": {},
    }
    for original_index in original_indices:
        score_path = root / "coarse" / "significance" / f"significance_orig{original_index:06d}_cs{current_size:03d}.npz"
        comparison_path = root / "comparisons" / f"orig{original_index}_coarse.json"
        scores = np.load(score_path)
        comparison = json.loads(comparison_path.read_text())
        keys = [tuple(comparison["recovar_top_key"]), tuple(comparison["relion_top_key"])]
        keys = list(dict.fromkeys(keys))
        rotations = jnp.asarray(np.stack([scores["rotations"][rotation] for rotation, _ in keys]))
        direct_projections = np.asarray(
            project(
                full_projector.reshape(-1),
                rotations,
                image_shape=(current_size, current_size),
                volume_shape=(projector_size,) * 3,
                order=1,
                half_volume=False,
                half_image=True,
                max_r=float(current_size // 2),
                relion_texture_interp=True,
            )
        ).astype(np.complex64)
        direct_projections *= np.float32(-(128**2))
        texture_indices = _window_to_texture_rows(scores["window_indices"], 128, current_size)
        direct_projections = direct_projections[:, texture_indices]
        production_full, _ = compute_relion_projector_projections_block(
            jnp.asarray(projector_half[0]),
            rotations,
            (128, 128),
            r_max=r_max,
            padding_factor=padding_factor,
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=current_size,
        )
        projections = np.asarray(production_full)[:, scores["window_indices"]].astype(np.complex64)
        production_projection_max_abs_diff = float(np.max(np.abs(projections - direct_projections)))

        candidate_results = []
        for candidate_index, (rotation, translation) in enumerate(keys):
            shifted = np.asarray(scores["shifted_data"][translation], dtype=np.complex64)
            ctf2 = np.asarray(scores["ctf2_data"][0], dtype=np.float32)
            projection = projections[candidate_index]
            cross = np.asarray((np.conj(shifted) * projection).real, dtype=np.float32)
            norm = np.asarray(ctf2 * np.asarray(np.abs(projection) ** 2, dtype=np.float32), dtype=np.float32)
            score = np.float32(_tree_sum_float32(cross) / np.sqrt(_tree_sum_float32(norm), dtype=np.float32))
            candidate_results.append(
                {
                    "key": [int(rotation), int(translation)],
                    "texture_tree_score": float(score),
                    "gemm_score": float(scores["scores_pre_prior_per_class"][0, rotation, translation]),
                }
            )
        texture_winner = candidate_results[int(np.argmax([row["texture_tree_score"] for row in candidate_results]))]["key"]
        relion_winner = [int(value) for value in comparison["relion_top_key"]]
        results["particles"][str(original_index)] = {
            "candidates": candidate_results,
            "texture_winner": texture_winner,
            "relion_winner": relion_winner,
            "texture_matches_relion": texture_winner == relion_winner,
            "production_projection_max_abs_diff": production_projection_max_abs_diff,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-indices", default=",".join(map(str, DEFAULT_ORIGINAL_INDICES)))
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    original_indices = tuple(int(value) for value in args.original_indices.split(",") if value)
    result = run_probe(args.root.resolve(), original_indices)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
