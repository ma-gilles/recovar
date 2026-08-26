#!/usr/bin/env python3
"""Split a K=1 coarse raw-score margin into image, weight, and power terms."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.significance import (
    _compact_projection_window_positions,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_translation_angles_f32,
)

FACTORS = ("image", "weight", "initial_diff2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _shapley(values: dict[frozenset[str], float]) -> dict[str, float]:
    """Return exact Shapley attribution for a complete factor powerset."""

    factor_set = frozenset(FACTORS)
    expected = {frozenset(items) for size in range(4) for items in itertools.combinations(FACTORS, size)}
    if set(values) != expected:
        raise ValueError("Shapley input must contain the complete three-factor powerset")
    count = len(FACTORS)
    result = {}
    for factor in FACTORS:
        others = factor_set - {factor}
        contribution = 0.0
        for size in range(count):
            coefficient = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for subset_tuple in itertools.combinations(others, size):
                subset = frozenset(subset_tuple)
                contribution += coefficient * (values[subset | {factor}] - values[subset])
        result[factor] = float(contribution)
    return result


def _load(path: Path) -> dict[str, np.ndarray]:
    required = {
        "current_size",
        "n_trans",
        "translations",
        "translation_phase_source",
        "coarse_gaussian_score_indices",
        "coarse_gaussian_unshifted_corrected",
        "coarse_gaussian_pixel_weight",
        "coarse_gaussian_initial_diff2",
        "scores_pre_prior_per_class",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"coarse capture misses fields: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _restore_square_references(
    references: np.ndarray,
    *,
    image_shape: tuple[int, int],
    current_size: int,
    score_indices: np.ndarray,
) -> np.ndarray:
    """Restore compact circular projections to the square score layout."""

    values = np.asarray(references, dtype=np.complex64)
    score_indices = np.asarray(score_indices, dtype=np.int32)
    if values.shape[-1] == score_indices.size:
        return values
    active_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        square=False,
        include_dc=False,
    )
    active_positions = _compact_projection_window_positions(
        score_indices,
        active_indices,
    )
    if values.shape[-1] != active_positions.size:
        raise ValueError(
            "captured projection cannot be restored to the square score layout: "
            f"projection={values.shape}, active={active_positions.shape}, "
            f"score_indices={score_indices.shape}"
        )
    square = np.zeros(values.shape[:-1] + (score_indices.size,), dtype=np.complex64)
    square[..., active_positions] = values
    return square


def analyze(
    *,
    exact_path: Path,
    live_path: Path,
    physical_image_size: int,
    winner: tuple[int, int],
    cutoff: tuple[int, int],
    crossing: tuple[int, int],
) -> dict[str, object]:
    if jax.default_backend() != "gpu" or not cuda_backproject.cuda_available():
        raise RuntimeError("coarse operand swap requires the custom CUDA GPU backend")
    exact = _load(exact_path)
    live = _load(live_path)
    required_live = {"projected_reference_rotation_ids", "projected_reference_per_class"}
    missing_live = required_live - set(live)
    if missing_live:
        raise ValueError(f"live capture misses fields: {sorted(missing_live)}")
    for field in ("current_size", "n_trans", "translations", "translation_phase_source", "coarse_gaussian_score_indices"):
        if not np.array_equal(exact[field], live[field]):
            raise ValueError(f"coarse topology field {field} differs")

    rotation_ids = np.asarray(live["projected_reference_rotation_ids"], dtype=np.int64)
    references = np.asarray(live["projected_reference_per_class"], dtype=np.complex64)[0]
    rotation_to_local = {int(rotation): local for local, rotation in enumerate(rotation_ids)}
    for name, coordinate in {"winner": winner, "cutoff": cutoff, "crossing": crossing}.items():
        if coordinate[0] not in rotation_to_local:
            raise ValueError(f"{name} rotation {coordinate[0]} is absent from captured projections")

    current_size = int(np.asarray(live["current_size"]).item())
    image_shape = (int(physical_image_size), int(physical_image_size))
    score_indices = np.asarray(live["coarse_gaussian_score_indices"], dtype=np.int32)
    references = _restore_square_references(
        references,
        image_shape=image_shape,
        current_size=current_size,
        score_indices=score_indices,
    )
    lookup = _relion_cuda_fine_full_to_compact_lookup(
        image_shape, current_size, score_indices
    ).astype(np.int32, copy=False)
    angles = _relion_translation_angles_f32(
        np.asarray(live["translation_phase_source"], dtype=np.float64), image_shape
    ).astype(np.float32, copy=False)

    def operand(payload: dict[str, np.ndarray], field: str, dtype) -> np.ndarray:
        value = np.asarray(payload[field], dtype=dtype)
        return value[None, :] if value.ndim == 1 else value

    source = {
        "exact": {
            "image": operand(exact, "coarse_gaussian_unshifted_corrected", np.complex64),
            "weight": operand(exact, "coarse_gaussian_pixel_weight", np.float32),
            "initial_diff2": np.asarray(exact["coarse_gaussian_initial_diff2"], dtype=np.float32).reshape(-1),
        },
        "live": {
            "image": operand(live, "coarse_gaussian_unshifted_corrected", np.complex64),
            "weight": operand(live, "coarse_gaussian_pixel_weight", np.float32),
            "initial_diff2": np.asarray(live["coarse_gaussian_initial_diff2"], dtype=np.float32).reshape(-1),
        },
    }

    score_tables: dict[frozenset[str], np.ndarray] = {}
    for size in range(4):
        for exact_factors_tuple in itertools.combinations(FACTORS, size):
            exact_factors = frozenset(exact_factors_tuple)
            selected = {
                factor: source["exact" if factor in exact_factors else "live"][factor]
                for factor in FACTORS
            }
            diff2 = cuda_backproject.relion_coarse_diff2_fused_translate_rectangular_f32(
                jnp.asarray(references),
                jnp.asarray(selected["image"]),
                jnp.asarray(angles),
                jnp.asarray(selected["weight"]),
                jnp.asarray(selected["initial_diff2"]),
                jnp.asarray(lookup),
                current_size=current_size,
            )
            score_tables[exact_factors] = -np.asarray(
                jax.block_until_ready(diff2), dtype=np.float32
            )[0]

    def value(table: np.ndarray, coordinate: tuple[int, int]) -> float:
        return float(table[rotation_to_local[coordinate[0]], coordinate[1]])

    coordinate_map = {"winner": winner, "cutoff": cutoff, "crossing": crossing}
    scores = {
        "+".join(sorted(exact_factors)) if exact_factors else "live_all": {
            name: value(table, coordinate) for name, coordinate in coordinate_map.items()
        }
        for exact_factors, table in score_tables.items()
    }
    margin_values = {
        comparison: {
            exact_factors: value(table, target) - value(table, winner)
            for exact_factors, table in score_tables.items()
        }
        for comparison, target in {"crossing_vs_winner": crossing, "cutoff_vs_winner": cutoff}.items()
    }
    shapley = {
        comparison: {
            "exact_minus_live_by_operand": _shapley(values),
            "live_margin": float(values[frozenset()]),
            "exact_operand_margin": float(values[frozenset(FACTORS)]),
            "exact_minus_live_total": float(values[frozenset(FACTORS)] - values[frozenset()]),
        }
        for comparison, values in margin_values.items()
    }

    live_production = np.asarray(live["scores_pre_prior_per_class"], dtype=np.float32)[0]
    live_recomputed = score_tables[frozenset()]
    production_panel = live_production[rotation_ids]
    delta = live_recomputed.astype(np.float64) - production_panel.astype(np.float64)
    return {
        "schema": "recovar.em.k1_coarse_operand_swap.v1",
        "status": "complete",
        "device": str(jax.devices()[0]),
        "kernel": "relion_coarse_diff2_fused_translate_rectangular_f32",
        "coordinates": {name: list(coordinate) for name, coordinate in coordinate_map.items()},
        "rotation_ids": rotation_ids.tolist(),
        "live_reproduction": {
            "exact_count": int(np.count_nonzero(live_recomputed == production_panel)),
            "candidate_count": int(live_recomputed.size),
            "max_abs_delta": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(production_panel.astype(np.float64))),
        },
        "scores": scores,
        "margin_attribution": shapley,
        "artifacts": {
            "exact": {"path": str(exact_path.resolve()), "sha256": _sha256(exact_path)},
            "live": {"path": str(live_path.resolve()), "sha256": _sha256(live_path)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--winner", type=int, nargs=2, required=True)
    parser.add_argument("--cutoff", type=int, nargs=2, required=True)
    parser.add_argument("--crossing", type=int, nargs=2, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        exact_path=args.exact,
        live_path=args.live,
        physical_image_size=args.physical_image_size,
        winner=tuple(args.winner),
        cutoff=tuple(args.cutoff),
        crossing=tuple(args.crossing),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
