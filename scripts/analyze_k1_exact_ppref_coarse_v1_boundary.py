#!/usr/bin/env python3
"""Replay a RELION coarse-v1 table from its exact in-memory PPref."""

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
    compute_relion_projector_projections_block,
    relion_projector_half_to_texture_full,
)
from recovar.em.dense_single_volume.helpers.significance import _dense_projection_scale
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_translation_angles_f32,
)
from scripts.analyze_em_k1_coarse_pass1_boundary import _map_relion_table
from scripts.analyze_k1_coarse_map_score_counterfactual import (
    _posterior_summary,
    _scores_with_priors,
)
from scripts.analyze_k1_exact_ppref_coarse_boundary import (
    _centered_stats,
    _load_ppref,
    _posterior_metrics,
    _top_mass,
)
from scripts.analyze_k1_native_coarse_boundary import (
    _float32_from_bits,
    load_native_coarse_capture,
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


def _candidate_record(
    coordinate: tuple[int, int],
    *,
    native_scores: np.ndarray,
    recorded_scores: np.ndarray,
    native_texture_scores: np.ndarray,
    model_window_texture_scores: np.ndarray,
    preprojected_scores: np.ndarray,
    active: np.ndarray,
    native_mask: np.ndarray,
    recorded_mask: np.ndarray,
    native_texture_mask: np.ndarray,
    model_window_texture_mask: np.ndarray,
    preprojected_mask: np.ndarray,
) -> dict[str, object]:
    rotation, translation = coordinate
    native_best = float(np.max(native_scores[active]))
    recorded_best = float(np.max(recorded_scores[active]))
    native_texture_best = float(np.max(native_texture_scores[active]))
    model_window_texture_best = float(np.max(model_window_texture_scores[active]))
    preprojected_best = float(np.max(preprojected_scores[active]))
    return {
        "rotation": rotation,
        "translation": translation,
        "native_centered_raw_score": float(native_scores[coordinate] - native_best),
        "recorded_centered_raw_score": float(recorded_scores[coordinate] - recorded_best),
        "exact_ppref_native_texture_centered_raw_score": float(
            native_texture_scores[coordinate] - native_texture_best
        ),
        "exact_ppref_model_window_texture_centered_raw_score": float(
            model_window_texture_scores[coordinate] - model_window_texture_best
        ),
        "exact_ppref_preprojected_centered_raw_score": float(
            preprojected_scores[coordinate] - preprojected_best
        ),
        "recorded_minus_native": float(
            recorded_scores[coordinate] - recorded_best
            - native_scores[coordinate] + native_best
        ),
        "exact_ppref_native_texture_minus_native": float(
            native_texture_scores[coordinate] - native_texture_best
            - native_scores[coordinate] + native_best
        ),
        "exact_ppref_model_window_texture_minus_native": float(
            model_window_texture_scores[coordinate] - model_window_texture_best
            - native_scores[coordinate] + native_best
        ),
        "exact_ppref_preprojected_minus_native": float(
            preprojected_scores[coordinate] - preprojected_best
            - native_scores[coordinate] + native_best
        ),
        "native_selected": bool(native_mask[coordinate]),
        "recorded_selected": bool(recorded_mask[coordinate]),
        "exact_ppref_native_texture_selected": bool(native_texture_mask[coordinate]),
        "exact_ppref_model_window_texture_selected": bool(
            model_window_texture_mask[coordinate]
        ),
        "exact_ppref_preprojected_selected": bool(preprojected_mask[coordinate]),
    }


def _square_score_subset(
    score_indices: np.ndarray,
    *,
    physical_image_size: int,
    score_size: int,
) -> np.ndarray:
    """Select a centered RELION rectangular FFTW score box."""

    indices = np.asarray(score_indices, dtype=np.int64).reshape(-1)
    half_width = int(physical_image_size) // 2 + 1
    rows = indices // half_width
    columns = indices % half_width
    ky = rows - int(physical_image_size) // 2
    inside = (
        (ky > -(int(score_size) // 2))
        & (ky <= int(score_size) // 2)
        & (columns <= int(score_size) // 2)
    )
    expected = int(score_size) * (int(score_size) // 2 + 1)
    _require(int(np.count_nonzero(inside)) == expected, "score subset topology mismatch")
    return inside


def analyze(
    *,
    ppref_path: Path,
    recovar_path: Path,
    native_coarse_path: Path,
    physical_image_size: int,
    native_directions: int,
    native_psi: int,
) -> dict[str, object]:
    _require(jax.default_backend() == "gpu", "exact PPref coarse replay requires a GPU")
    _require(cuda_backproject.cuda_available(), "custom CUDA backend is unavailable")
    ppref, ppref_metadata = _load_ppref(ppref_path)
    native = load_native_coarse_capture(native_coarse_path)
    with np.load(recovar_path, allow_pickle=False) as archive:
        required = {
            "current_size",
            "rotations",
            "translations",
            "translation_phase_source",
            "coarse_gaussian_unshifted_corrected",
            "coarse_gaussian_pixel_weight",
            "coarse_gaussian_initial_diff2",
            "coarse_gaussian_score_indices",
            "scores_pre_prior_per_class",
            "class_log_priors",
            "rotation_log_prior",
            "translation_log_prior",
            "adaptive_fraction",
            "max_significants",
            "significant_mask",
        }
        _require(required <= set(archive.files), f"RECOVAR dump misses {sorted(required - set(archive.files))}")
        recovar = {name: np.asarray(archive[name]) for name in archive.files}

    current_size = int(np.asarray(recovar["current_size"]).item())
    rotations = np.asarray(recovar["rotations"], dtype=np.float32)
    translations = np.asarray(recovar["translations"], dtype=np.float64)
    n_rot, n_trans = rotations.shape[0], translations.shape[0]
    _require(native_directions * native_psi == n_rot, "native rotation topology mismatch")
    _require(int(native.header[4]) == n_rot and int(native.header[5]) == n_trans, "coarse topology mismatch")
    _require(int(ppref_metadata["iteration"]) == int(native.header[1]), "iteration mismatch")
    projector_size = int(ppref_metadata["current_size"])
    _require(projector_size <= current_size, "projector size exceeds particle score size")
    _require(int(ppref_metadata["r_max"]) == projector_size // 2, "PPref radius mismatch")
    _require(int(ppref.shape[-1]) >= current_size, "PPref x extent cannot serve particle score box")
    _require(float(ppref_metadata["padding_factor"]) == 2.0, "PPref padding mismatch")

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
        * jnp.asarray(
            _dense_projection_scale((physical_image_size, physical_image_size)),
            dtype=jnp.float32,
        ),
        dtype=np.complex64,
    )
    native_texture_diff2 = np.asarray(
        jax.block_until_ready(
            cuda_backproject.relion_coarse_diff2_native_texture_rectangular_f32(
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
        ),
        dtype=np.float32,
    )
    native_texture_scores = -native_texture_diff2
    native_texture_scores_with_priors = _scores_with_priors(
        native_texture_diff2,
        class_log_prior=float(np.asarray(recovar["class_log_priors"], dtype=np.float32)[0]),
        rotation_log_prior=np.asarray(recovar["rotation_log_prior"], dtype=np.float32)[0],
        translation_log_prior=np.asarray(recovar["translation_log_prior"], dtype=np.float32),
    )
    native_texture_summary, native_texture_weights, native_texture_mask = _posterior_summary(
        native_texture_scores_with_priors,
        adaptive_fraction=float(np.asarray(recovar["adaptive_fraction"]).item()),
        max_significants=int(np.asarray(recovar["max_significants"]).item()),
    )

    # RELION remaps model current size with the serialized model-STAR pixel
    # size.  RECOVAR historically reused higher-precision MRC sampling here,
    # making this particle score box two pixels larger.  This arm keeps PPref,
    # particle operands, candidates, priors, and reduction fixed and changes
    # only that rectangular score box.  The omitted image-only power is an
    # additive candidate-independent term, so the recorded initial_diff2 is
    # sufficient for centered scores and normalized posterior comparisons.
    model_window_subset = _square_score_subset(
        score_indices,
        physical_image_size=physical_image_size,
        score_size=projector_size,
    )
    model_window_score_indices = score_indices[model_window_subset]
    model_window_full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        (physical_image_size, physical_image_size),
        projector_size,
        model_window_score_indices,
    )
    model_window_texture_diff2 = np.asarray(
        jax.block_until_ready(
            cuda_backproject.relion_coarse_diff2_native_texture_rectangular_f32(
                jnp.asarray(projector_full),
                jnp.asarray(rotations),
                jnp.asarray(unshifted[model_window_subset][None]),
                jnp.asarray(translation_angles, dtype=jnp.float32),
                jnp.asarray(pixel_weight[model_window_subset][None]),
                jnp.asarray(initial_diff2),
                jnp.asarray(model_window_full_to_compact, dtype=jnp.int32),
                projector_size,
                2,
                int(ppref_metadata["r_max"]),
            )[0]
        ),
        dtype=np.float32,
    )
    model_window_texture_scores = -model_window_texture_diff2
    model_window_texture_scores_with_priors = _scores_with_priors(
        model_window_texture_diff2,
        class_log_prior=float(np.asarray(recovar["class_log_priors"], dtype=np.float32)[0]),
        rotation_log_prior=np.asarray(recovar["rotation_log_prior"], dtype=np.float32)[0],
        translation_log_prior=np.asarray(recovar["translation_log_prior"], dtype=np.float32),
    )
    (
        model_window_texture_summary,
        model_window_texture_weights,
        model_window_texture_mask,
    ) = _posterior_summary(
        model_window_texture_scores_with_priors,
        adaptive_fraction=float(np.asarray(recovar["adaptive_fraction"]).item()),
        max_significants=int(np.asarray(recovar["max_significants"]).item()),
    )

    projected, _ = compute_relion_projector_projections_block(
        jnp.asarray(ppref),
        jnp.asarray(rotations, dtype=jnp.float32),
        (physical_image_size, physical_image_size),
        r_max=int(ppref_metadata["r_max"]),
        padding_factor=2,
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=current_size,
        pixel_indices=jnp.asarray(score_indices, dtype=jnp.int32),
        relion_texture_interp=True,
    )
    preprojected_diff2 = np.asarray(
        jax.block_until_ready(
            cuda_backproject.relion_coarse_diff2_rectangular_f32(
                projected,
                jnp.asarray(recovar["coarse_gaussian_shifted_corrected"], dtype=jnp.complex64)[None],
                jnp.asarray(pixel_weight[None]),
                jnp.asarray(initial_diff2),
                jnp.asarray(full_to_compact, dtype=jnp.int32),
            )[0]
        ),
        dtype=np.float32,
    )
    preprojected_scores = -preprojected_diff2
    preprojected_scores_with_priors = _scores_with_priors(
        preprojected_diff2,
        class_log_prior=float(np.asarray(recovar["class_log_priors"], dtype=np.float32)[0]),
        rotation_log_prior=np.asarray(recovar["rotation_log_prior"], dtype=np.float32)[0],
        translation_log_prior=np.asarray(recovar["translation_log_prior"], dtype=np.float32),
    )
    preprojected_summary, preprojected_weights, preprojected_mask = _posterior_summary(
        preprojected_scores_with_priors,
        adaptive_fraction=float(np.asarray(recovar["adaptive_fraction"]).item()),
        max_significants=int(np.asarray(recovar["max_significants"]).item()),
    )

    recorded_scores = np.asarray(recovar["scores_pre_prior_per_class"], dtype=np.float32)[0]
    recorded_mask = np.asarray(recovar["significant_mask"], dtype=bool).reshape(n_rot, n_trans)
    native_diff2 = _map_relion_table(
        native.raw_diff2.reshape(n_rot, n_trans),
        n_directions=native_directions,
        n_psi=native_psi,
        relion_to_recovar_translation=np.arange(n_trans, dtype=np.int64),
    ).astype(np.float32)
    native_scores = -native_diff2
    native_postexponent = _map_relion_table(
        native.postexponent.reshape(n_rot, n_trans),
        n_directions=native_directions,
        n_psi=native_psi,
        relion_to_recovar_translation=np.arange(n_trans, dtype=np.int64),
    ).astype(np.float32)
    native_weights = native_postexponent.astype(np.float64)
    native_weights /= np.sum(native_weights, dtype=np.float64)
    native_threshold = np.float32(_float32_from_bits(int(native.header[13])))
    native_strict_mask = native_postexponent > native_threshold
    native_tie_mask = native_postexponent == native_threshold
    native_mask = native_strict_mask | native_tie_mask
    active = native_diff2 != -np.finfo(np.float32).max

    native_count = int(native.header[9])
    native_strict_count = int(np.count_nonzero(native_strict_mask))
    native_tie_count = int(np.count_nonzero(native_tie_mask))
    _require(native_strict_count <= native_count <= native_strict_count + native_tie_count, "native cutoff is outside its threshold tie group")
    recorded_mismatches = np.argwhere(recorded_mask != native_mask)
    native_texture_mismatches = np.argwhere(native_texture_mask != native_mask)
    model_window_texture_mismatches = np.argwhere(
        model_window_texture_mask != native_mask
    )
    preprojected_mismatches = np.argwhere(preprojected_mask != native_mask)
    mismatch_coordinates = sorted(
        {
            tuple(int(value) for value in row)
            for row in np.concatenate(
                (
                    recorded_mismatches,
                    native_texture_mismatches,
                    model_window_texture_mismatches,
                    preprojected_mismatches,
                    np.argwhere(native_tie_mask),
                ),
                axis=0,
            )
        }
    )
    return {
        "schema": "recovar.em.k1_exact_ppref_coarse_v1_boundary.v2",
        "status": "complete",
        "device": str(jax.devices()[0]),
        "identity": {
            "stack_index_one_based": int(native.header[2]),
            "physical_iteration": int(native.header[1]),
            "ppref_rank": int(ppref_metadata["rank"]),
            "projector_current_size": projector_size,
            "particle_score_current_size": current_size,
            "rotation_count": n_rot,
            "translation_count": n_trans,
        },
        "ppref": ppref_metadata,
        "raw_score_centered_vs_native_relion": {
            "recorded_recovar": _centered_stats(recorded_scores, native_scores, active),
            "exact_native_ppref_native_texture": _centered_stats(
                native_texture_scores, native_scores, active
            ),
            "exact_native_ppref_model_window_texture": _centered_stats(
                model_window_texture_scores, native_scores, active
            ),
            "exact_native_ppref_preprojected": _centered_stats(
                preprojected_scores, native_scores, active
            ),
            "exact_native_ppref_native_texture_vs_recorded_recovar": _centered_stats(
                native_texture_scores, recorded_scores, active
            ),
            "exact_native_ppref_preprojected_vs_recorded_recovar": _centered_stats(
                preprojected_scores, recorded_scores, active
            ),
        },
        "posterior_vs_native_relion": {
            "exact_native_ppref_native_texture": _posterior_metrics(
                native_texture_weights, native_weights
            ),
            "exact_native_ppref_model_window_texture": _posterior_metrics(
                model_window_texture_weights, native_weights
            ),
            "exact_native_ppref_preprojected": _posterior_metrics(
                preprojected_weights, native_weights
            ),
        },
        "significant_support": {
            "native_count": native_count,
            "native_strictly_above_threshold_count": native_strict_count,
            "native_equal_to_threshold_count": native_tie_count,
            "native_threshold_inclusive_count": int(np.count_nonzero(native_mask)),
            "native_mask_semantics": (
                "threshold-inclusive reconstruction; coarse-v1 does not serialize "
                "which equal-weight CUB key filled the final cutoff slot"
            ),
            "native_threshold_tie_coordinates": np.argwhere(native_tie_mask).astype(int).tolist(),
            "recorded_recovar_count": int(np.count_nonzero(recorded_mask)),
            "exact_native_ppref_native_texture_count": int(
                np.count_nonzero(native_texture_mask)
            ),
            "exact_native_ppref_model_window_texture_count": int(
                np.count_nonzero(model_window_texture_mask)
            ),
            "exact_native_ppref_preprojected_count": int(
                np.count_nonzero(preprojected_mask)
            ),
            "recorded_recovar_mismatch_vs_native": int(recorded_mismatches.shape[0]),
            "exact_native_ppref_native_texture_mismatch_vs_native": int(
                native_texture_mismatches.shape[0]
            ),
            "exact_native_ppref_model_window_texture_mismatch_vs_native": int(
                model_window_texture_mismatches.shape[0]
            ),
            "exact_native_ppref_preprojected_mismatch_vs_native": int(
                preprojected_mismatches.shape[0]
            ),
            "native_top_mass_at_native_count": _top_mass(native_weights, native_count),
            "exact_native_ppref_native_texture_top_mass_at_native_count": _top_mass(
                native_texture_weights, native_count
            ),
            "exact_native_ppref_model_window_texture_top_mass_at_native_count": _top_mass(
                model_window_texture_weights, native_count
            ),
            "exact_native_ppref_preprojected_top_mass_at_native_count": _top_mass(
                preprojected_weights, native_count
            ),
        },
        "exact_native_ppref_native_texture_posterior": native_texture_summary,
        "exact_native_ppref_model_window_texture_posterior": model_window_texture_summary,
        "exact_native_ppref_preprojected_posterior": preprojected_summary,
        "boundary_candidates": [
            _candidate_record(
                coordinate,
                native_scores=native_scores,
                recorded_scores=recorded_scores,
                native_texture_scores=native_texture_scores,
                model_window_texture_scores=model_window_texture_scores,
                preprojected_scores=preprojected_scores,
                active=active,
                native_mask=native_mask,
                recorded_mask=recorded_mask,
                native_texture_mask=native_texture_mask,
                model_window_texture_mask=model_window_texture_mask,
                preprojected_mask=preprojected_mask,
            )
            for coordinate in mismatch_coordinates[:64]
        ],
        "artifacts": {
            "ppref": str(ppref_path.resolve()),
            "ppref_sha256": _sha256(ppref_path),
            "recovar": str(recovar_path.resolve()),
            "recovar_sha256": _sha256(recovar_path),
            "native_coarse": str(native_coarse_path.resolve()),
            "native_coarse_sha256": _sha256(native_coarse_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppref", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--native-coarse", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--native-directions", type=int, required=True)
    parser.add_argument("--native-psi", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        ppref_path=args.ppref,
        recovar_path=args.recovar,
        native_coarse_path=args.native_coarse,
        physical_image_size=args.physical_image_size,
        native_directions=args.native_directions,
        native_psi=args.native_psi,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
