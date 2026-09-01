#!/usr/bin/env python3
"""Replay a complete K=1 coarse table from native in-memory RELION PPref."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.projection import (
    relion_projector_half_to_texture_full,
)
from recovar.em.dense_single_volume.helpers.significance import _dense_projection_scale
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_translation_angles_f32,
)
from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _translation_permutation,
)
from scripts.analyze_k1_coarse_map_score_counterfactual import (
    _posterior_summary,
    _scores_with_priors,
)
from scripts.analyze_k1_exact_ppref_fine_boundary import _load_ppref
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
    load_artifact,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _centered_stats(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float32)[mask]
    right = np.asarray(reference, dtype=np.float32)[mask]
    _require(left.shape == right.shape and left.size > 0, "score comparison is empty")
    residual = (
        left.astype(np.float64)
        - float(np.max(left))
        - right.astype(np.float64)
        + float(np.max(right))
    )
    absolute = np.abs(residual)
    return {
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
        "rms": float(np.sqrt(np.mean(residual * residual))),
    }


def _posterior_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float64).reshape(-1).copy()
    right = np.asarray(reference, dtype=np.float64).reshape(-1).copy()
    _require(left.shape == right.shape and left.size > 0, "posterior topology mismatch")
    left /= np.sum(left)
    right /= np.sum(right)
    residual = left - right
    return {
        "total_variation": float(0.5 * np.sum(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _top_mass(weights: np.ndarray, count: int) -> float:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    values /= np.sum(values)
    order = np.argsort(-values, kind="stable")
    return float(np.sum(values[order[:count]], dtype=np.float64))


def analyze(
    *,
    ppref_path: Path,
    recovar_path: Path,
    native_components_path: Path,
    physical_image_size: int,
) -> dict[str, object]:
    _require(jax.default_backend() == "gpu", "exact PPref coarse replay requires a GPU")
    _require(cuda_backproject.cuda_available(), "custom CUDA backend is unavailable")
    ppref, ppref_metadata = _load_ppref(ppref_path)
    components = load_artifact(native_components_path)
    with np.load(recovar_path, allow_pickle=False) as archive:
        required = {
            "current_size", "rotations", "translations", "translation_phase_source",
            "coarse_gaussian_unshifted_corrected", "coarse_gaussian_pixel_weight",
            "coarse_gaussian_initial_diff2", "coarse_gaussian_score_indices",
            "scores_pre_prior_per_class", "scores_with_prior_per_class",
            "weights_per_class", "significant_mask", "class_log_priors",
            "rotation_log_prior", "translation_log_prior", "adaptive_fraction",
            "max_significants", "original_index",
        }
        _require(required <= set(archive.files), f"RECOVAR dump misses {sorted(required - set(archive.files))}")
        recovar = {name: np.asarray(archive[name]) for name in archive.files}

    current_size = int(np.asarray(recovar["current_size"]).item())
    _require(int(ppref_metadata["iteration"]) == int(components.header[5]), "iteration mismatch")
    _require(int(ppref_metadata["rank"]) == int(components.header[3]), "MPI-rank/half mismatch")
    _require(int(ppref_metadata["current_size"]) == current_size, "current-size mismatch")
    _require(float(ppref_metadata["padding_factor"]) == 2.0, "PPref padding mismatch")
    _require(int(ppref_metadata["r_max"]) == current_size // 2, "PPref radius mismatch")
    _require(int(components.stack_index) - 1 == int(recovar["original_index"]), "particle identity mismatch")

    rotations = np.asarray(recovar["rotations"], dtype=np.float32)
    translations = np.asarray(recovar["translations"], dtype=np.float64)
    unshifted = np.asarray(recovar["coarse_gaussian_unshifted_corrected"], dtype=np.complex64)
    pixel_weight = np.asarray(recovar["coarse_gaussian_pixel_weight"], dtype=np.float32)
    initial_diff2 = np.asarray(recovar["coarse_gaussian_initial_diff2"], dtype=np.float32).reshape(1)
    score_indices = np.asarray(recovar["coarse_gaussian_score_indices"], dtype=np.int32)
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        (physical_image_size, physical_image_size), current_size, score_indices
    )
    translation_angles = _relion_translation_angles_f32(
        np.asarray(recovar["translation_phase_source"], dtype=np.float64),
        (physical_image_size, physical_image_size),
    )
    projector_full = np.asarray(
        relion_projector_half_to_texture_full(jnp.asarray(ppref))
        * jnp.asarray(_dense_projection_scale((physical_image_size, physical_image_size)), dtype=jnp.float32),
        dtype=np.complex64,
    )
    diff2 = cuda_backproject.relion_coarse_diff2_native_texture_rectangular_f32(
        jnp.asarray(projector_full),
        jnp.asarray(rotations),
        jnp.asarray(unshifted[None]),
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(pixel_weight[None]),
        jnp.asarray(initial_diff2),
        jnp.asarray(full_to_compact, dtype=jnp.int32),
        current_size,
        2,
        int(ppref_metadata["r_max"]),
    )[0]
    candidate_diff2 = np.asarray(jax.block_until_ready(diff2), dtype=np.float32)
    candidate_scores = -candidate_diff2
    candidate_scores_with_priors = _scores_with_priors(
        candidate_diff2,
        class_log_prior=float(np.asarray(recovar["class_log_priors"], dtype=np.float32)[0]),
        rotation_log_prior=np.asarray(recovar["rotation_log_prior"], dtype=np.float32)[0],
        translation_log_prior=np.asarray(recovar["translation_log_prior"], dtype=np.float32),
    )
    candidate_summary, candidate_weights, candidate_mask = _posterior_summary(
        candidate_scores_with_priors,
        adaptive_fraction=float(np.asarray(recovar["adaptive_fraction"]).item()),
        max_significants=int(np.asarray(recovar["max_significants"]).item()),
    )

    recorded_scores = np.asarray(recovar["scores_pre_prior_per_class"], dtype=np.float32)[0]
    recorded_scores_with_priors = np.asarray(recovar["scores_with_prior_per_class"], dtype=np.float32)[0]
    recorded_summary, recorded_replay_weights, recorded_replay_mask = _posterior_summary(
        recorded_scores_with_priors,
        adaptive_fraction=float(np.asarray(recovar["adaptive_fraction"]).item()),
        max_significants=int(np.asarray(recovar["max_significants"]).item()),
    )
    recorded_weights = np.asarray(recovar["weights_per_class"], dtype=np.float64)[0]
    recorded_mask = np.asarray(recovar["significant_mask"], dtype=bool).reshape(recorded_scores.shape)

    n_directions, n_psi = (int(components.header[10]), int(components.header[11]))
    translation_permutation, translation_mapping = _translation_permutation(
        components.translations, translations
    )
    native_diff2 = _map_relion_table(
        components.raw_diff2,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(np.float32)
    native_weights = _map_relion_table(
        components.weights,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(np.float64)
    native_mask = _map_relion_table(
        components.significant_mask,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(bool)
    active = native_diff2 != RELION_INVALID_DIFF2
    native_scores = -native_diff2

    native_count = int(np.count_nonzero(native_mask))
    adaptive_fraction = float(np.asarray(recovar["adaptive_fraction"]).item())
    report = {
        "schema": "recovar.em.k1_exact_ppref_coarse_boundary.v1",
        "status": "complete",
        "identity": {
            "stack_index_one_based": int(components.stack_index),
            "original_index_zero_based": int(recovar["original_index"]),
            "physical_iteration": int(components.header[5]),
            "mpi_rank": int(components.header[3]),
            "rotation_count": int(rotations.shape[0]),
            "translation_count": int(translations.shape[0]),
        },
        "ppref": ppref_metadata,
        "raw_score_centered_vs_native_relion": {
            "recorded_recovar": _centered_stats(recorded_scores, native_scores, active),
            "native_ppref_with_recovar_other_operands": _centered_stats(candidate_scores, native_scores, active),
        },
        "posterior_vs_native_relion": {
            "recorded_recovar": _posterior_metrics(recorded_weights, native_weights),
            "native_ppref_with_recovar_other_operands": _posterior_metrics(candidate_weights, native_weights),
        },
        "significant_support": {
            "adaptive_fraction": adaptive_fraction,
            "native_count": native_count,
            "recorded_recovar_count": int(np.count_nonzero(recorded_mask)),
            "recorded_replay_count": int(np.count_nonzero(recorded_replay_mask)),
            "native_ppref_count": int(np.count_nonzero(candidate_mask)),
            "recorded_recovar_mismatch_vs_native": int(np.count_nonzero(recorded_mask != native_mask)),
            "recorded_replay_mismatch_vs_recorded": int(np.count_nonzero(recorded_replay_mask != recorded_mask)),
            "native_ppref_mismatch_vs_native": int(np.count_nonzero(candidate_mask != native_mask)),
            "native_top_mass_at_native_count": _top_mass(native_weights, native_count),
            "recorded_top_mass_at_native_count": _top_mass(recorded_weights, native_count),
            "native_ppref_top_mass_at_native_count": _top_mass(candidate_weights, native_count),
            "native_margin_at_native_count": _top_mass(native_weights, native_count) - adaptive_fraction,
            "recorded_margin_at_native_count": _top_mass(recorded_weights, native_count) - adaptive_fraction,
            "native_ppref_margin_at_native_count": _top_mass(candidate_weights, native_count) - adaptive_fraction,
        },
        "posterior_summaries": {
            "recorded_replay": recorded_summary,
            "native_ppref": candidate_summary,
        },
        "translation_mapping": translation_mapping,
        "artifacts": {
            "ppref": str(ppref_path.resolve()),
            "ppref_sha256": _sha256(ppref_path),
            "recovar": str(recovar_path.resolve()),
            "recovar_sha256": _sha256(recovar_path),
            "native_components": str(native_components_path.resolve()),
            "native_components_sha256": _sha256(native_components_path),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppref", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--native-components", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        ppref_path=args.ppref,
        recovar_path=args.recovar,
        native_components_path=args.native_components,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
