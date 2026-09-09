#!/usr/bin/env python3
"""Replay one live VDAM top-pair decision after swapping only its input map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError("projection comparison requires aligned nonempty arrays")
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        raise ValueError("captured projection has zero norm")
    return float(np.linalg.norm(candidate - reference) / denominator)


def summarize_pair_spacing(
    *,
    first_diff2: float,
    second_diff2: float,
    first_prior: float,
    second_prior: float,
    first_translation: int,
    second_translation: int,
) -> dict[str, object]:
    """Return likelihood/prior/total spacing in RECOVAR score convention."""

    likelihood_spacing = -float(first_diff2) + float(second_diff2)
    prior_spacing = float(first_prior) - float(second_prior)
    total_spacing = likelihood_spacing + prior_spacing
    return {
        "first_translation_index": int(first_translation),
        "second_translation_index": int(second_translation),
        "likelihood_spacing_first_minus_second": likelihood_spacing,
        "prior_spacing_first_minus_second": prior_spacing,
        "total_spacing_first_minus_second": total_spacing,
        "winner_translation_index": int(
            first_translation if total_spacing >= 0.0 else second_translation
        ),
    }


def _parse_map(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("maps must use LABEL=PATH")
    return label, Path(raw_path)


def analyze(
    *,
    live_score_path: Path,
    maps: list[tuple[str, Path]],
    rotation_row: int,
    first_translation: int,
    second_translation: int,
    full_image_size: int,
    padding_factor: int,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_indices_np,
    )
    from recovar.em.dense_single_volume.helpers.half_spectrum import (
        make_scoring_half_image_weights,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_cuda_fine_full_to_compact_lookup,
    )
    from recovar.em.initial_model.dense_adapter import (
        reference_to_relion_projector_half_maps,
    )
    from recovar.utils.helpers import load_relion_volume

    with np.load(live_score_path, allow_pickle=False) as archive:
        live = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "current_size",
        "local_rotation_matrices",
        "debug_proj_weighted",
        "debug_shifted_score",
        "debug_ctf2_over_nv",
        "pass2_scores_raw",
        "pass2_scores_total",
        "rotation_log_prior",
        "translation_log_prior",
        "translations",
    }
    missing = required - set(live)
    if missing:
        raise ValueError(f"live score dump is missing fields: {sorted(missing)}")

    current_size = int(np.asarray(live["current_size"]).reshape(-1)[0])
    rotations = np.asarray(live["local_rotation_matrices"], dtype=np.float32)
    translation_grid = np.asarray(live["translations"], dtype=np.float32)
    if rotation_row < 0 or rotation_row >= rotations.shape[0]:
        raise ValueError("rotation row is outside the live support")
    for translation_index in (first_translation, second_translation):
        if translation_index < 0 or translation_index >= translation_grid.shape[0]:
            raise ValueError("translation row is outside the live support")

    score_indices, _ = make_fourier_window_indices_np(
        (full_image_size, full_image_size),
        current_size,
        square=False,
        include_dc=False,
    )
    captured_projection = np.asarray(live["debug_proj_weighted"], dtype=np.complex64)
    if captured_projection.shape != (rotations.shape[0], score_indices.size):
        raise ValueError("captured projection does not match the score window")
    score_half_weights = np.asarray(
        make_scoring_half_image_weights(
            (full_image_size, full_image_size),
            relion_half_sum=True,
        ),
        dtype=np.float32,
    )[score_indices]
    live_weight = np.asarray(live["debug_ctf2_over_nv"], dtype=np.float32)
    shifted_weighted = np.asarray(live["debug_shifted_score"], dtype=np.complex64)
    if live_weight.shape != (score_indices.size,) or shifted_weighted.shape != (
        translation_grid.shape[0],
        score_indices.size,
    ):
        raise ValueError("captured image/weight operands do not match the score window")
    shifted_unweighted = np.zeros_like(shifted_weighted)
    np.divide(
        shifted_weighted,
        live_weight[None],
        out=shifted_unweighted,
        where=live_weight[None] != 0.0,
    )
    direct_weight = np.multiply(live_weight, score_half_weights, dtype=np.float32)
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        (full_image_size, full_image_size),
        current_size,
        score_indices,
    )

    prior = np.asarray(live["rotation_log_prior"], dtype=np.float64)[0, rotation_row] + np.asarray(
        live["translation_log_prior"], dtype=np.float64
    )[0]
    live_raw = np.asarray(live["pass2_scores_raw"], dtype=np.float64)[0, rotation_row]
    live_total = np.asarray(live["pass2_scores_total"], dtype=np.float64)[0, rotation_row]
    captured = {
        "likelihood_spacing_first_minus_second": float(
            live_raw[first_translation] - live_raw[second_translation]
        ),
        "prior_spacing_first_minus_second": float(
            prior[first_translation] - prior[second_translation]
        ),
        "total_spacing_first_minus_second": float(
            live_total[first_translation] - live_total[second_translation]
        ),
        "winner_translation_index": int(
            first_translation
            if live_total[first_translation] >= live_total[second_translation]
            else second_translation
        ),
    }

    map_reports: dict[str, object] = {}
    for label, map_path in maps:
        volume = np.asarray(load_relion_volume(map_path), dtype=np.float32)
        expected_shape = (full_image_size,) * 3
        if volume.shape != expected_shape:
            raise ValueError(f"map {label!r} has shape {volume.shape}, expected {expected_shape}")
        projector, r_max = reference_to_relion_projector_half_maps(
            volume[None],
            current_size=current_size,
            padding_factor=padding_factor,
        )
        projected, _ = compute_relion_projector_projections_block(
            jnp.asarray(projector[0], dtype=jnp.complex64),
            jnp.asarray(rotations, dtype=jnp.float32),
            (full_image_size, full_image_size),
            r_max=int(r_max),
            padding_factor=padding_factor,
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=current_size,
            pixel_indices=jnp.asarray(score_indices, dtype=jnp.int32),
            relion_texture_interp=True,
            mask_current_image_disk=False,
        )
        projected = np.asarray(jax.block_until_ready(projected), dtype=np.complex64)
        projected_weighted = np.multiply(
            projected,
            score_half_weights[None],
            dtype=np.complex64,
        )
        pair_references = projected_weighted[[rotation_row, rotation_row]]
        pair_shifted = shifted_unweighted[[first_translation, second_translation]]
        diff2 = cuda_backproject.relion_fine_diff2_pairs_f32(
            jnp.asarray(pair_references[None], dtype=jnp.complex64),
            jnp.asarray(pair_shifted[None], dtype=jnp.complex64),
            jnp.asarray(direct_weight[None], dtype=jnp.float32),
            jnp.asarray(full_to_compact, dtype=jnp.int32),
        )[0]
        diff2 = np.asarray(jax.block_until_ready(diff2), dtype=np.float32)
        spacing = summarize_pair_spacing(
            first_diff2=float(diff2[0]),
            second_diff2=float(diff2[1]),
            first_prior=float(prior[first_translation]),
            second_prior=float(prior[second_translation]),
            first_translation=first_translation,
            second_translation=second_translation,
        )
        map_reports[label] = {
            "map_path": str(map_path.resolve()),
            "projector_r_max": int(r_max),
            "projection_relative_l2_vs_live_capture": _relative_l2(
                captured_projection,
                projected_weighted,
            ),
            "pair_diff2": diff2.astype(float).tolist(),
            **spacing,
        }

    return {
        "schema": "recovar.vdam_live_map_top_pair_counterfactual.v1",
        "status": "complete",
        "device": str(jax.devices()[0]),
        "identity": {
            "live_score": str(live_score_path.resolve()),
            "current_size": current_size,
            "full_image_size": int(full_image_size),
            "rotation_row": int(rotation_row),
            "first_translation_index": int(first_translation),
            "second_translation_index": int(second_translation),
            "first_translation": translation_grid[first_translation].astype(float).tolist(),
            "second_translation": translation_grid[second_translation].astype(float).tolist(),
        },
        "captured_live_spacing": captured,
        "map_counterfactuals": map_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-score", type=Path, required=True)
    parser.add_argument("--map", type=_parse_map, action="append", required=True)
    parser.add_argument("--rotation-row", type=int, required=True)
    parser.add_argument("--first-translation", type=int, required=True)
    parser.add_argument("--second-translation", type=int, required=True)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--padding-factor", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        live_score_path=args.live_score,
        maps=args.map,
        rotation_row=args.rotation_row,
        first_translation=args.first_translation,
        second_translation=args.second_translation,
        full_image_size=args.full_image_size,
        padding_factor=args.padding_factor,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
