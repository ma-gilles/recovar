#!/usr/bin/env python3
"""Test whether an incoming native map closes one K=1 coarse score margin."""

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
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_translation_angles_f32,
)
from recovar.em.dense_single_volume.helpers.significance import _dense_projection_scale
from recovar.em.initial_model.dense_adapter import (
    reference_to_relion_projector_half_maps,
)
from recovar.utils import helpers
from scripts.analyze_em_k1_coarse_pass1_boundary import _map_relion_table
from scripts.analyze_k1_native_coarse_boundary import load_native_coarse_capture


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_margin(diff2: np.ndarray, *, winner: tuple[int, int], target: tuple[int, int]) -> float:
    """Return target-minus-winner raw log score from a rotation/translation table."""

    values = np.asarray(diff2, dtype=np.float32)
    return float(-(values[target] - values[winner]))


def _project(
    reference: np.ndarray,
    rotations: np.ndarray,
    score_indices: np.ndarray,
    *,
    current_size: int,
) -> np.ndarray:
    physical_size = int(reference.shape[0])
    ppref, r_max = reference_to_relion_projector_half_maps(
        np.asarray(reference, dtype=np.float32)[None],
        current_size=current_size,
        padding_factor=2,
    )
    projected, _ = compute_relion_projector_projections_block(
        jnp.asarray(ppref[0]),
        jnp.asarray(rotations, dtype=jnp.float32),
        (physical_size, physical_size),
        r_max=int(r_max),
        padding_factor=2,
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=current_size,
        pixel_indices=jnp.asarray(score_indices, dtype=jnp.int32),
        relion_texture_interp=True,
    )
    return np.asarray(jax.block_until_ready(projected), dtype=np.complex64)


def analyze(
    *,
    native_capture_path: Path,
    recovar_capture_path: Path,
    native_map_path: Path,
    recovar_map_path: Path,
    winner: tuple[int, int],
    target: tuple[int, int],
    native_directions: int,
    native_psi: int,
) -> dict[str, object]:
    if jax.default_backend() != "gpu" or not cuda_backproject.cuda_available():
        raise RuntimeError("map-score counterfactual requires the custom CUDA GPU backend")

    native = load_native_coarse_capture(native_capture_path)
    with np.load(recovar_capture_path, allow_pickle=False) as payload:
        rotations = np.asarray(payload["rotations"], dtype=np.float32)
        current_size = int(payload["current_size"])
        score_indices = np.asarray(payload["coarse_gaussian_score_indices"], dtype=np.int32)
        shifted = np.asarray(payload["coarse_gaussian_shifted_corrected"], dtype=np.complex64)
        unshifted = np.asarray(payload["coarse_gaussian_unshifted_corrected"], dtype=np.complex64)
        pixel_weight = np.asarray(payload["coarse_gaussian_pixel_weight"], dtype=np.float32)
        initial_diff2 = np.asarray(payload["coarse_gaussian_initial_diff2"], dtype=np.float32).reshape(1)
        translation_source = np.asarray(payload["translation_phase_source"], dtype=np.float64)
        captured_scores = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float32)[0]
        n_trans = int(payload["n_trans"])
    if n_trans != shifted.shape[0]:
        raise ValueError("RECOVAR translation topology changed")

    native_raw = _map_relion_table(
        native.raw_diff2.reshape(int(native.header[16]), int(native.header[17])),
        n_directions=native_directions,
        n_psi=native_psi,
        relion_to_recovar_translation=np.arange(n_trans, dtype=np.int64),
    ).astype(np.float32)
    selected_rotation_ids = np.asarray([winner[0], target[0]], dtype=np.int64)
    selected_rotations = rotations[selected_rotation_ids]

    native_map = np.asarray(helpers.load_relion_volume(str(native_map_path)), dtype=np.float32)
    recovar_map = np.asarray(helpers.load_mrc(str(recovar_map_path)), dtype=np.float32)
    if native_map.shape != recovar_map.shape:
        raise ValueError("incoming native and RECOVAR map shapes differ")

    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        native_map.shape[:2],
        current_size,
        score_indices,
    )

    def score(reference: np.ndarray) -> np.ndarray:
        projected = _project(
            reference,
            selected_rotations,
            score_indices,
            current_size=current_size,
        )
        values = cuda_backproject.relion_coarse_diff2_rectangular_f32(
            jnp.asarray(projected),
            jnp.asarray(shifted[None]),
            jnp.asarray(pixel_weight[None]),
            jnp.asarray(initial_diff2),
            jnp.asarray(full_to_compact, dtype=jnp.int32),
        )[0]
        return np.asarray(jax.block_until_ready(values), dtype=np.float32)

    native_map_diff2 = score(native_map)
    recovar_map_diff2 = score(recovar_map)

    def native_texture_score(reference: np.ndarray) -> np.ndarray:
        ppref, r_max = reference_to_relion_projector_half_maps(
            np.asarray(reference, dtype=np.float32)[None],
            current_size=current_size,
            padding_factor=2,
        )
        projector_full = np.asarray(
            relion_projector_half_to_texture_full(jnp.asarray(ppref[0]))
            * jnp.asarray(_dense_projection_scale(native_map.shape[:2]), dtype=jnp.float32),
            dtype=np.complex64,
        )
        translation_angles = _relion_translation_angles_f32(
            translation_source,
            native_map.shape[:2],
        )
        # Exercise the complete production orientation grid.  The exact
        # RELION coarse specialization has 16 Euler matrices per block; a
        # two-rotation panel changes code generation and atomic scheduling.
        values = cuda_backproject.relion_coarse_diff2_native_texture_rectangular_f32(
            jnp.asarray(projector_full),
            jnp.asarray(rotations),
            jnp.asarray(unshifted[None]),
            jnp.asarray(translation_angles, dtype=jnp.float32),
            jnp.asarray(pixel_weight[None]),
            jnp.asarray(initial_diff2),
            jnp.asarray(full_to_compact, dtype=jnp.int32),
            current_size,
            2,
            int(r_max),
        )[0]
        return np.asarray(jax.block_until_ready(values), dtype=np.float32)

    native_texture_recovar_map_diff2 = native_texture_score(recovar_map)
    native_texture_native_map_diff2 = native_texture_score(native_map)
    local_winner = (0, winner[1])
    local_target = (1, target[1])
    native_margin = _score_margin(native_raw, winner=winner, target=target)
    captured_recovar_margin = float(
        captured_scores[target] - captured_scores[winner]
    )
    regenerated_recovar_margin = _score_margin(
        recovar_map_diff2,
        winner=local_winner,
        target=local_target,
    )
    regenerated_native_margin = _score_margin(
        native_map_diff2,
        winner=local_winner,
        target=local_target,
    )
    native_texture_recovar_map_margin = _score_margin(
        native_texture_recovar_map_diff2,
        winner=winner,
        target=target,
    )
    native_texture_native_map_margin = _score_margin(
        native_texture_native_map_diff2,
        winner=winner,
        target=target,
    )
    map_delta = regenerated_native_margin - regenerated_recovar_margin
    counterfactual_margin = captured_recovar_margin + map_delta
    baseline_error = captured_recovar_margin - native_margin
    counterfactual_error = counterfactual_margin - native_margin
    removal_fraction = (
        1.0
        if baseline_error == 0.0 and counterfactual_error == 0.0
        else 0.0
        if baseline_error == 0.0
        else float(1.0 - abs(counterfactual_error) / abs(baseline_error))
    )
    return {
        "schema": "recovar.em.k1_coarse_map_score_counterfactual.v1",
        "status": "complete",
        "device": str(jax.devices()[0]),
        "coordinates": {"winner": list(winner), "target": list(target)},
        "raw_log_score_margin_target_minus_winner": {
            "native": native_margin,
            "recovar_captured": captured_recovar_margin,
            "recovar_regenerated": regenerated_recovar_margin,
            "native_map_regenerated": regenerated_native_margin,
            "native_texture_recovar_map": native_texture_recovar_map_margin,
            "native_texture_native_map": native_texture_native_map_margin,
            "native_texture_recovar_map_minus_native": (
                native_texture_recovar_map_margin - native_margin
            ),
            "native_map_minus_recovar_map_regenerated": map_delta,
            "recovar_plus_map_delta_counterfactual": counterfactual_margin,
            "baseline_recovar_minus_native": baseline_error,
            "counterfactual_minus_native": counterfactual_error,
            "absolute_error_removal_fraction": removal_fraction,
        },
        "classification": (
            "incoming map closes the raw score margin"
            if abs(counterfactual_error) <= np.finfo(np.float32).eps * max(1.0, abs(native_margin))
            else "incoming map is dominant for the raw score margin"
            if removal_fraction >= 0.8
            else "incoming map is not dominant for the raw score margin"
            if removal_fraction <= 0.2
            else "incoming map partially contributes to the raw score margin"
        ),
        "artifacts": {
            "native_capture": str(native_capture_path.resolve()),
            "native_capture_sha256": _sha256(native_capture_path),
            "recovar_capture": str(recovar_capture_path.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture_path),
            "native_map": str(native_map_path.resolve()),
            "native_map_sha256": _sha256(native_map_path),
            "recovar_map": str(recovar_map_path.resolve()),
            "recovar_map_sha256": _sha256(recovar_map_path),
        },
    }


def _pair(value: str) -> tuple[int, int]:
    fields = value.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("coordinate must be ROTATION,TRANSLATION")
    return int(fields[0]), int(fields[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-capture", type=Path, required=True)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--native-map", type=Path, required=True)
    parser.add_argument("--recovar-map", type=Path, required=True)
    parser.add_argument("--winner", type=_pair, required=True)
    parser.add_argument("--target", type=_pair, required=True)
    parser.add_argument("--native-directions", type=int, required=True)
    parser.add_argument("--native-psi", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        native_capture_path=args.native_capture,
        recovar_capture_path=args.recovar_capture,
        native_map_path=args.native_map,
        recovar_map_path=args.recovar_map,
        winner=args.winner,
        target=args.target,
        native_directions=args.native_directions,
        native_psi=args.native_psi,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
